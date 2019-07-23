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

import losses
import points
import sips_multiscale
import system

FLAGS = absl.flags.FLAGS


class ForwardPass(sips_multiscale.ForwardPass):
    def selectTop(self, num_ips, nms_size):
        raise Exception('Meaningless in IMIPS')

    def reducedTo(self, num_ips):
        raise Exception('Meaningless in IMIPS')

    def toFlatForwardPass(self):
        chosen_scale = self.scaleArgMax()
        ips_rc = [
            self.forward_passes[chosen_scale[i]].ips_rc[:, i].astype(float) /
            (self.factor ** chosen_scale[i])
            for i in range(self.forward_passes[0].ips_rc.shape[1])]
        scales = np.array(
            [1. / (self.factor ** chosen_scale[i])
             for i in range(self.forward_passes[0].ips_rc.shape[1])])
        ips_rc = np.array(ips_rc).T.astype(int)
        ip_scores = [
            self.forward_passes[chosen_scale[i]].ip_scores[i]
            for i in range(self.forward_passes[0].ips_rc.shape[1])]
        return system.ForwardPass(ips_rc, ip_scores, scales=scales)

    def scaleArgMax(self):
        score_stack = np.vstack(tuple(
            [fp.ip_scores for fp in self.forward_passes]))
        return np.argmax(score_stack, axis=0)


class ForwardPasser(sips_multiscale.ForwardPasser):
    def __call__(self, image):
        pyramid = sips_multiscale.Pyramid(image, self._factor, self._size)
        forward_passes = [self._forward_passer(im) for im in pyramid.images]
        ret = ForwardPass(self._factor, forward_passes, pyramid.tiling)
        return ret


def getStrongestCorrespondence(corr_rc_at_1, fpass):
    """ Returns location at scale with strongest response, and that scale. """
    corr_at_best = []
    scale_i = []
    n_scales = len(fpass.forward_passes)
    scale_is = np.arange(n_scales)
    for corr_i in range(corr_rc_at_1.shape[1]):
        corr = corr_rc_at_1[:, corr_i]
        corr_at_scales = [(corr * (fpass.factor ** si)).astype(int)
                          for si in scale_is]
        responses_at_scales = \
            [fpass.forward_passes[si].image_scores[
                 corr_at_scales[si][0], corr_at_scales[si][1], :]
             for si in scale_is]
        own_response_at_scales = np.diagonal(np.array(responses_at_scales))
        best = np.argsort(-own_response_at_scales)
        ordered_scale_is = scale_is[best]
        best_scale_i = None
        for cand_scale_i in ordered_scale_is:
            if points.hasMarginToBorder(
                    corr_at_scales[cand_scale_i],
                    fpass.forward_passes[cand_scale_i].image_scores.shape,
                    FLAGS.depth):
                best_scale_i = cand_scale_i
                break
        assert best_scale_i is not None
        corr_at_best.append(corr_at_scales[best_scale_i])
        scale_i.append(best_scale_i)
    return np.array(corr_at_best).T.reshape((2, -1)), scale_i


def patchesAt(pyramid_images, scale_is, points_rc_at_scale):
    ips_patches = []
    n = points_rc_at_scale.shape[1]
    for i in range(n):
        scale_i = scale_is[i]
        ips_patches.append(
            losses.imageToBatchInput(
                pyramid_images[scale_i],
                points_rc_at_scale[:, i].reshape((2, 1)), FLAGS.depth))
    return np.concatenate(ips_patches, axis=0)


def getTrainingBatch(image, fpass, corr_rc, corr_valid_mask):
    inl, outl = system.getInliersOutliersImage(
        fpass.toFlatForwardPass(), corr_rc, corr_valid_mask)

    patch_sz = FLAGS.depth * 2 + 1
    assert FLAGS.chan == corr_rc.shape[1]
    pyramid = sips_multiscale.Pyramid(
        image, fpass.factor, len(fpass.forward_passes))
    sam = fpass.scaleArgMax()

    ips_at_scales = np.array([fpass.forward_passes[sam[i]].ips_rc[:, i]
                              for i in range(corr_rc.shape[1])]).T
    ips_patches = patchesAt(pyramid.images, sam, ips_at_scales)

    if not pyramid.images[1].shape == \
            fpass.forward_passes[1].image_scores.shape[:2]:
        print(pyramid.images[1].shape)
        print(fpass.forward_passes[1].image_scores.shape)
        raise Exception

    corr_at_best, corr_scale_is = getStrongestCorrespondence(
        corr_rc[:, corr_valid_mask], fpass)

    corr_patches = np.zeros((FLAGS.chan, patch_sz, patch_sz, 1))
    if np.count_nonzero(corr_valid_mask) > 0:
        corr_patches[corr_valid_mask, :, :, :] = patchesAt(
            pyramid.images, corr_scale_is, corr_at_best)
    else:
        print('Warning: no valid correspondences!!!')
    return ips_patches, corr_patches, inl, outl
