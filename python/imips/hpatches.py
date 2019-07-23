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

from absl import flags
import numpy as np

import losses
import patch_batch

FLAGS = flags.FLAGS


def evaluateHpatchPair(pair, fps, pr_dists=False,
                       get_apparent=False):
    assert all([i.ips_rc.shape[0] == 2 for i in fps])
    ips_xy = [np.flipud(fps[i].ips_rc) for i in [0, 1]]
    warped = pair.correspondences(ips_xy[0])
    dists = np.linalg.norm(warped - ips_xy[1], axis=0)
    mult = np.maximum(fps[0].scales, fps[1].scales)
    in_tolerance = dists < (mult * 3)
    if pr_dists:
        print(dists)
    apparent = np.count_nonzero(in_tolerance)
    inliers = ips_xy[0][:, in_tolerance].T
    unique_inliers = []
    for inlier in inliers:
        if len(unique_inliers) == 0 or np.all(np.linalg.norm(
                np.array(unique_inliers) - inlier, axis=1) > 3):
            unique_inliers.append(inlier)
    true = len(unique_inliers)
    print('Apparent, true inliers: %d %d' % (apparent, true))
    if get_apparent:
        return apparent, true, in_tolerance
    else:
        return apparent, true


def getOtherImageCorresps(pair, ips_rc):
    ips_xy = [np.flipud(ips_rc[i]) for i in [0,1]]
    in1 = pair.correspondences(ips_xy[0])
    in0 = pair.correspondences(ips_xy[1], inv=True)
    return [np.flipud(in0).astype(int), np.flipud(in1).astype(int)]


def getBatchesForImage(image, ips_rc, corresps_rc, patchable_filter):
    is_inlier_label = np.linalg.norm(ips_rc - corresps_rc, axis=0) < 3
    patch_sz = FLAGS.depth * 2 + 1
    
    patchable_ips = ips_rc[:, patchable_filter]
    for i in range(patchable_ips.shape[1]):
        assert patch_batch.patchable(patchable_ips[:, i], image)
    
    ips_patches = np.zeros((FLAGS.chan, patch_sz, patch_sz, 1))
    ips_patches[patchable_filter, :, :, :] = losses.imageToBatchInput(
            image, patchable_ips, FLAGS.depth)
    
    patchable_corr = corresps_rc[:, patchable_filter]
    for i in range(patchable_corr.shape[1]):
        assert patch_batch.patchable(patchable_corr[:, i], image)
    
    corr_patches = np.zeros((FLAGS.chan, patch_sz, patch_sz, 1))
    corr_patches[patchable_filter, :, :, :] = losses.imageToBatchInput(
            image, patchable_corr, FLAGS.depth)
    return ips_patches, corr_patches, np.logical_and(
            is_inlier_label, patchable_filter), np.logical_and(
            np.logical_not(is_inlier_label), patchable_filter)


def getRandomPointWithCorresp(pair):
    while True:
        pt0_xy = np.array([np.random.randint(0, pair.im[0].shape[i]) \
                           for i in [1, 0]])
        pt0 = np.array([pt0_xy[1], pt0_xy[0]])
        pt1_xy = pair.correspondences(pt0_xy.reshape(2, 1))
        pt1 = np.flipud(pt1_xy).ravel().astype(int)
        if patch_batch.patchable(pt0, pair.im[0]) and \
            patch_batch.patchable(pt1, pair.im[1]):
            return pt0, pt1


def getRandomPointsWithCorresp(pair, n_pts):
    im0_pts = []
    im1_pts = []
    for i in range(n_pts):
        im0_pt, im1_pt = getRandomPointWithCorresp(pair)
        im0_pts.append(im0_pt)
        im1_pts.append(im1_pt)
    return [np.array(im0_pts).T, np.array(im1_pts).T]


def activationDiff(net_out, qn_pts):
    in_activ = qn_pts - FLAGS.depth # remember activation is smaller
    return net_out[in_activ[0, 1], in_activ[1, 1], :] - \
        net_out[in_activ[0, 0], in_activ[1, 0], :]


def getValidCorrespondences(pair, ips_rc):
    corr_rc = getOtherImageCorresps(pair, ips_rc)
    patchable_masks = [np.array([patch_batch.patchable(
        corr_rc[im_i][:, ip_i], pair.im[im_i])
        for ip_i in range(corr_rc[im_i].shape[1])]) for im_i in [0, 1]]
    return corr_rc, patchable_masks


def getBatchGivenFf(pair, ips_rc, net_outs, do_plot=None):
    for im_i in [0, 1]:
        for ip_i in range(ips_rc[im_i].shape[1]):
            assert patch_batch.patchable(ips_rc[im_i][:, ip_i], pair.im[im_i])

    corresps_rc, patchable_masks = getValidCorrespondences(pair, ips_rc)

    ips0, corr0, inl0, outl0 = getBatchesForImage(
            pair.im[0], ips_rc[0], corresps_rc[0], patchable_masks[0])
    ips1, corr1, inl1, outl1 = getBatchesForImage(
            pair.im[1], ips_rc[1], corresps_rc[1], patchable_masks[0])
    
    # [colwise two pts in im0, corresps in im1]:
    qn_pts = getRandomPointsWithCorresp(pair, 2)
    qn_patches = [losses.imageToBatchInput(
            pair.im[i], qn_pts[i], FLAGS.depth) for i in [0, 1]]
    qn_diffs = [activationDiff(net_outs[i], qn_pts[i]) for i in [1, 0]]
    
    return [ips0, ips1], [corr0, corr1], [inl0, inl1], [outl0, outl1], \
        qn_patches, qn_diffs
