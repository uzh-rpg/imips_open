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
import cv2
import numpy as np
import skimage.measure
import time

import hyperparams

FLAGS = absl.flags.FLAGS


def tile(net_outs, rows, cols, downscale, ips_rc=None):
    assert net_outs.shape[2] == 128
    xdim = net_outs.shape[1]
    ydim = net_outs.shape[0]

    im = np.zeros([rows * ydim, cols * xdim, 3])
    for r in range(rows):
        for c in range(cols):
            im_i = cv2.applyColorMap(
                (net_outs[:, :, r * cols + c] * 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            if ips_rc is not None:
                cv2.circle(im_i, tuple(ips_rc[[1, 0], r * cols + c]),
                           downscale * 5, (0, 0, 0), downscale * 3,
                           cv2.LINE_AA)
            im[r * ydim:(r + 1) * ydim, c * xdim:(c + 1) * xdim, :] = im_i

    return skimage.measure.block_reduce(im, (downscale, downscale, 1), np.max)


def saveChannelOuts(data_point, deploy_net, tf_sess):
    if not FLAGS.parallel_forward:
        net_outs = tf_sess.run(deploy_net.output, 
                               feed_dict={deploy_net.input: data_point.im[0]})
    else:
        net_outs = tf_sess.run(
                deploy_net.output, 
                feed_dict={deploy_net.input: 
                    np.concatenate((
                            np.expand_dims(data_point.im[0], 0),
                            np.expand_dims(data_point.im[1], 0)), axis=0)})
        net_outs = net_outs[0, :, :, :]

    cv2.imwrite('chouts/%s_%d.png' % (hyperparams.shortString(), time.time()),
                tile(net_outs, 16, 8, 4))
