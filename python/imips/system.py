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

import rpg_datasets_py.hpatches

import baselines
import datasets
import hpatches
import interest_points
import losses
import multiscale
import patch_batch
import sequences
import sips_system

FLAGS = absl.flags.FLAGS


def renderHueMultiplier():
    assert FLAGS.chan == 128
    return 180. / 128.


def renderHues():
    assert FLAGS.chan == 128
    return renderHueMultiplier() * np.arange(128, dtype=float)


def renderColors():
    h = renderHues()
    hsv = np.expand_dims(np.vstack(
        (h, 255 * np.ones_like(h), 255 * np.ones_like(h))).astype(
        np.uint8).T, 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).squeeze().astype(int)


class ForwardPass(object):
    def __init__(self, ips_rc, ip_scores, scales=None, image_scores=None):
        self.ips_rc = ips_rc
        assert ips_rc.dtype == int
        self.ip_scores = ip_scores
        if scales is None:
            self.scales = np.ones_like(ip_scores)
        else:
            self.scales = scales
        self.image_scores = image_scores

    def render(self, with_points=True, fallback_im=None):
        if self.image_scores is not None:
            assert len(self.image_scores.shape) == 3
            assert self.image_scores.shape[2] == 128
            dom_channel = np.argmax(self.image_scores, axis=2)
            max_score = np.max(self.image_scores, axis=2)
            hsv = np.stack((dom_channel * renderHueMultiplier(),
                            255 * np.ones_like(dom_channel),
                            (np.clip(max_score, 0, 0.5)) * 2 * 255),
                           axis=2).astype(np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            bgr = cv2.cvtColor(fallback_im, cv2.COLOR_GRAY2BGR)

        if with_points:
            rc = renderColors()
            for i in range(128):
                if i == len(self.ip_scores):
                    assert FLAGS.baseline != ''
                    break
                thickness_relevant_score = \
                    np.clip(self.ip_scores[i], 0.4, 0.6) - 0.4
                thickness = int(thickness_relevant_score * 40)
                radius = int(self.scales[i] * 10)
                cv2.circle(bgr, tuple(self.ips_rc[[1, 0], i]),
                           radius, tuple(rc[i]), thickness, cv2.LINE_AA)

        return bgr


class ForwardPasser(object):
    def __init__(self, deploy_net, sess):
        self._deploy_net = deploy_net
        self._sess = sess

    def __call__(self, image):
        image_scores, lin_amax = self._sess.run(
            [self._deploy_net.output, self._deploy_net.lin_amax],
            feed_dict={self._deploy_net.input: image})
        ips_rc, ip_scores = interest_points.getFromNetOut(
            image_scores, lin_amax)
        return ForwardPass(ips_rc, ip_scores, image_scores=image_scores)


def unifyForwardPasses(fps):
    if type(fps[0]) == multiscale.ForwardPass:
        fps = [fp.toFlatForwardPass() for fp in fps]
    elif type(fps[0]) == sips_system.ForwardPass:
        matched_indices = sips_system.match(fps)
        fps = [ForwardPass(
            i[0].ips_rc[:, i[1]], i[0].ip_scores[i[1]],
            None if type(i[0].scales) is float else i[0].scales)
            for i in zip(fps, matched_indices)]
    elif type(fps[0]) == baselines.ScorelessForwardPass:
        matched_indices = sips_system.match(fps)
        fps = [ForwardPass(
            i[0].ips_rc[:, i[1]], None,
            scales=(None if type(i[0].scales) is float else i[0].scales))
            for i in zip(fps, matched_indices)]
    elif not type(fps[0]) == ForwardPass:
        raise NotImplementedError

    return fps


def getCorrespondences(pair, forward_passes):
    ips_rc = [fp.ips_rc for fp in forward_passes]
    if type(pair) == rpg_datasets_py.hpatches.HPatchPair:
        corr_rc, valid_masks = hpatches.getValidCorrespondences(pair, ips_rc)
    elif type(pair) == sequences.PairWithIntermediates:
        v_corr_rc = [None, None]
        patchable_indices = [None, None]
        v_corr_rc[0], patchable_indices[0] = pair.trackThrough(
            ips_rc[1], reverse=True)
        v_corr_rc[1], patchable_indices[1] = pair.trackThrough(
            ips_rc[0], reverse=False)
        valid_masks = [np.zeros(ips_rc[i].shape[1], dtype=bool) for i in [0, 1]]
        corr_rc = [np.zeros_like(i) for i in ips_rc]
        for i in [0, 1]:
            valid_masks[i][patchable_indices[i]] = True
            corr_rc[i][:, patchable_indices[i]] = v_corr_rc[i]
    elif type(pair) == datasets.DTUPair:
        n = ips_rc[0].shape[1]
        assert ips_rc[1].shape[1] == n
        corr_rc = [np.zeros_like(i) for i in ips_rc]
        valid_masks = [np.zeros(n, dtype=bool) for i in [0, 1]]
        for i in [0, 1]:
            v_corr_rc, corr_indices = pair.getCorrespondence(
                ips_rc[(i + 1) % 2], reverse=(i == 0))
            corr_indices = np.array(corr_indices)
            patchable = np.array(
                [patch_batch.patchable(pt, pair.im[i]) for pt in v_corr_rc.T])
            assert len(patchable) == v_corr_rc.shape[1]
            patchable_indices = corr_indices[patchable]
            v_corr_rc = v_corr_rc[:, patchable]

            valid_masks[i][patchable_indices] = True
            corr_rc[i][:, patchable_indices] = v_corr_rc
    else:
        raise NotImplementedError
    assert all([corr_rc[i].shape[1] == ips_rc[i].shape[1] for i in [0, 1]])
    return corr_rc, valid_masks


def getInliersOutliersImage(fpass, corr_rc, corr_valid_mask):
    assert type(fpass) == ForwardPass
    inlier_mask = np.linalg.norm(
        fpass.ips_rc - corr_rc, axis=0) < (fpass.scales * 3)
    return np.logical_and(inlier_mask, corr_valid_mask), \
        np.logical_and(np.logical_not(inlier_mask), corr_valid_mask)


def getInliersUsingCorrespondences(pair, forward_passes):
    """ Returns a binary mask indicating inliers. """
    corr_rc, valid_masks = getCorrespondences(pair, forward_passes)
    inl_outl = [getInliersOutliersImage(
        forward_passes[i], corr_rc[i], valid_masks[i]) for i in [0, 1]]
    return np.logical_and(inl_outl[0][0], inl_outl[1][0])


def getTrainingBatchesImage(
        image, fpass, corr_rc, corr_valid_mask):
    if type(fpass) == ForwardPass:
        inl, outl = getInliersOutliersImage(
            fpass, corr_rc, corr_valid_mask)

        patch_sz = FLAGS.depth * 2 + 1
        assert FLAGS.chan == corr_rc.shape[1]
        ips_patches = losses.imageToBatchInput(image, fpass.ips_rc, FLAGS.depth)
        corr_patches = np.zeros((FLAGS.chan, patch_sz, patch_sz, 1))
        corr_patches[corr_valid_mask, :, :, :] = losses.imageToBatchInput(
            image, corr_rc[:, corr_valid_mask], FLAGS.depth)
        return ips_patches, corr_patches, inl, outl
    elif type(fpass) == multiscale.ForwardPass:
        return multiscale.getTrainingBatch(
            image, fpass, corr_rc, corr_valid_mask)
    else:
        raise NotImplementedError
