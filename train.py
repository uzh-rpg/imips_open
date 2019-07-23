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

import absl.app
import cProfile
import cv2
from IPython.core import ultratb
import numpy as np
import os
import tensorflow as tf
import sys

from rpg_common_py.trainer import Trainer

import imips.augment as augment
import imips.evaluate as evaluate
import imips.flags as flags
import imips.hyperparams as hyperparams
import imips.plot_utils as plot_utils
import imips.nets as nets
import imips.system as system

FLAGS = flags.FLAGS


def run():
    print('\n\n%s\n\n' % hyperparams.shortString())

    tf.reset_default_graph()
    deploy_net = nets.Deploy2Net()
    train_net = nets.CorrespTrainNet()
    
    sess = tf.Session()

    train_set = hyperparams.getTrainSet()
    val_set = hyperparams.getEvalSet(train=True)
    tval_set = hyperparams.getTrainValidationSubset()
    fpasser = hyperparams.getForwardPasser(deploy_net, sess)

    def step_fun(sess):
        dp = train_set.getRandomDataPoint()
        if FLAGS.tds == 'hp' and (FLAGS.aug_ang > 0. or FLAGS.aug_sc > 0.):
            dp = augment.augmentHpatchPair(
                dp, ang_range=FLAGS.aug_ang, sc_range=FLAGS.aug_sc)

        fps = [fpasser(i) for i in dp.im]
        flat = system.unifyForwardPasses(fps)
        corr_rc, corr_valid_masks = system.getCorrespondences(dp, flat)

        for i in [0, 1]:
            if FLAGS.use_qn:
                raise NotImplementedError
                sess.run(train_net.train_step, feed_dict={
                        train_net.ip_patches: ip_patches[i],
                        train_net.corr_patches: corr_patches[i],
                        train_net.is_inlier_label: inl[i],
                        train_net.is_outlier_label: outl[i],
                        train_net.qn_patches: qns[i],
                        train_net.qn_other_diff: qnd[i]})
            else:
                if np.min(fps[i].ips_rc) == 0:
                    viz = plot_utils.tile(
                        fps[i].image_scores, 16, 8, 4, fps[0].ips_rc)
                    cv2.imwrite('debug_out.png', viz)
                    raise Exception(
                        'Something went very wrong! See debug_out.png. '
                        'Consider restarting the training.')

                ip_patches, corr_patches, inl, outl = \
                    system.getTrainingBatchesImage(
                        dp.im[i], fps[i], corr_rc[i], corr_valid_masks[i])

                if FLAGS.render_train_samples:
                    evaluate.renderTrainSample(dp, fps, corr_rc, inl)

                sess.run(train_net.train_step, feed_dict={
                        train_net.ip_patches: ip_patches,
                        train_net.corr_patches: corr_patches,
                        train_net.is_inlier_label: inl,
                        train_net.is_outlier_label: outl})
    
    def val_fun(_):
        apparent, true, _, _, _ = evaluate.evaluatePairs(fpasser, val_set)
        app_tr, true_tr, _, _, _ = evaluate.evaluatePairs(fpasser, tval_set)
        return [apparent.mean(), true.mean(), app_tr.mean(), true_tr.mean()]
    
    trainer = Trainer(step_fun, val_fun, hyperparams.checkpointRoot(),
                      hyperparams.shortString(), check_every=FLAGS.val_every,
                      best_index=1, best_is_max=True)
    trainer.trainUpToNumIts(sess, FLAGS.its)


def main(_):
    sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=1)
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats('%s.profile' %  os.path.basename(__file__))


if __name__ == '__main__':
    absl.app.run(main)
