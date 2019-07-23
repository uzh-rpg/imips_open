# Copyright (C) 2019 Titus Cieslewski, RPG, University of Zurich, Switzerland
#   You can contact the author at <titus at ifi dot uzh dot ch>
# Copyright (C) 2019 Michael Bloesch,
#   Dept. of Computing, Imperial College London, United Kingdom
# Copyright (C) 2019 Davide Scaramuzza, RPG, University of Zurich, Switzerland
#
# This file is part of imips_open.
#
# imips_open is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# imips_open is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with imips_open. If not, see <http:#www.gnu.org/licenses/>.

import absl.flags
import numpy as np
import tensorflow as tf

import losses

FLAGS = absl.flags.FLAGS


def draft2Net(x, reuse=False):
    """ x is [None, None], height and width of image. """

    assert (FLAGS.depth % 2) == 0

    x = x - 127

    with tf.variable_scope('imips', reuse=reuse):
        if FLAGS.chan == 64:
            w = 64
        else:
            w = FLAGS.chan / 2
        k = [3, 3]
        s = [1, 1]
        h = x
        activ = tf.nn.leaky_relu

        for _ in range(int(FLAGS.depth / 2)):
            h = tf.layers.conv2d(
                h, w, k, s, padding='valid', activation=activ)

        if FLAGS.chan == 64:
            w = 128
        else:
            w = FLAGS.chan

        for _ in range(int(FLAGS.depth / 2) - 1):
            h = tf.layers.conv2d(
                h, w, k, s, padding='valid', activation=activ)
        h = tf.layers.conv2d(h, FLAGS.chan, k, s, padding='valid')
        return tf.nn.sigmoid(h - 1.5)


def linAmax(x):
    """ x is [h x w x c], return [c] linear argmaxes of each channel """
    #if FLAGS.parallel_forward:
    #    c = x.shape[3]
    #    lin = tf.reshape(x, [2, -1, c])
    #    linamax = tf.argmax(lin, axis=1)
    #else:
    c = x.shape[2]
    lin = tf.reshape(x, [-1, c])
    linamax = tf.argmax(lin, axis=0)
    return linamax


class Deploy2Net(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, None])
        expanded = tf.expand_dims(tf.expand_dims(self.input, 0), 3)
        scores = draft2Net(expanded, reuse=tf.AUTO_REUSE)
        scores = tf.squeeze(scores, axis=0)
        d = FLAGS.depth
        paddings = tf.constant([[d, d], [d, d], [0, 0]])
        padded_scores = tf.pad(scores, paddings, "CONSTANT")
        self.output = padded_scores
        self.lin_amax = linAmax(self.output)


def inlierLoss(x):
    return -tf.log(x)


def outlierLoss(x):
    return -tf.log(1 - x)


class CorrespTrainNet(object):
    def __init__(self):
        patch_sz = FLAGS.depth * 2 + 1
        batch_sz = FLAGS.chan
        
        # Patches and outputs
        for attr in ['ip', 'corr']:
            setattr(self, attr + '_patches', tf.placeholder(
                    tf.float32, [batch_sz, patch_sz, patch_sz, 1]))
            setattr(self, attr + '_outs', draft2Net(
                    getattr(self, attr + '_patches'), reuse=tf.AUTO_REUSE))
        
        # Quad_Nets: Two patches and the activation diff in the other image.
        # The latter is a bit of a hack but should be equivalent if I recall
        # correctly how backprop works...
        if FLAGS.use_qn:
            self.qn_patches = tf.placeholder(
                    tf.float32, [2, patch_sz, patch_sz, 1])
            self.qn_outs = draft2Net(self.qn_patches, reuse=tf.AUTO_REUSE)
            self.qn_other_diff = tf.placeholder(tf.float32, [FLAGS.chan])
        
        # Labels in 1D
        # Note that outlier is not simply 'not inlier', as we might leave it
        # open for some patches not to be trained either way.
        for attr in ['inlier', 'outlier']:
            setattr(self, 'is_' + attr + '_label', tf.placeholder(
                    tf.bool, [batch_sz]))
        
        # Cast labels into proper shape: Things that happen on the
        # batch-channel diagonal
        for attr in ['inlier', 'outlier']:
            setattr(self, 'is_' + attr + '_diag', tf.expand_dims(tf.expand_dims(
                    tf.matrix_diag(
                            getattr(self, 'is_' + attr + '_label')), 1), 1))
        
        # LOSSES        
        # Pull up correspondences (outlier)
        self.own_corr_outs = tf.boolean_mask(
            self.corr_outs, self.is_outlier_diag)
        self.corresp_loss = tf.reduce_sum(inlierLoss(self.own_corr_outs))
        # Pull up inliers
        self.inlier_outs = tf.boolean_mask(
            self.ip_outs, self.is_inlier_diag)
        self.inlier_loss = tf.reduce_sum(inlierLoss(self.inlier_outs))
        # Push down outliers
        self.outlier_outs = tf.boolean_mask(
            self.ip_outs, self.is_outlier_diag)
        self.outlier_loss = tf.reduce_sum(outlierLoss(self.outlier_outs))
        # Suppress other channels at inliers:
        self.is_noninlier_activation = tf.logical_xor(
                self.is_inlier_diag, tf.expand_dims(tf.expand_dims(
                        tf.expand_dims(self.is_inlier_label, -1), -1), -1))
        if FLAGS.dsup:
            upper_right = tf.constant(np.expand_dims(np.expand_dims(
                np.triu(np.ones([batch_sz, FLAGS.chan], dtype=bool)), 1), 1))
            self.is_noninlier_activation = tf.logical_and(
                self.is_noninlier_activation, upper_right)
        self.noninlier_activations = tf.boolean_mask(
                self.ip_outs, self.is_noninlier_activation)
        # TODO nonlinear robust loss?
        self.suppression_loss = tf.reduce_sum(self.noninlier_activations)

        self.loss = FLAGS.corr_w * self.corresp_loss + self.outlier_loss + \
                    self.suppression_loss + self.inlier_loss
        if FLAGS.use_qn:
            self.qn_loss = tf.reduce_mean(tf.maximum(
                    0., 1 - self.qn_other_diff * \
                    (self.qn_outs[1] - self.qn_outs[0])))
            self.loss = self.loss + self.qn_loss
            
        self.train_step = tf.train.AdamOptimizer(
                10 ** -FLAGS.lr).minimize(self.loss)