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

import cv2
import IPython
import math
import numpy as np
import os

import rpg_common_py.geometry
import rpg_datasets_py.hpatches

import datasets
import hyperparams
import p3p
import sequences
import system


def poseError(pair, T_1_0):
    if T_1_0 is None:
        return None, None

    T_1_0_true = pair.T_0_1.inverse()

    R_err = math.degrees(rpg_common_py.geometry.geodesicDistanceSO3(
        T_1_0.R, T_1_0_true.R))
    t_err = np.linalg.norm(T_1_0.t - T_1_0_true.t)

    return R_err, t_err


def p3pMaskAndError(pair, matched_fps):
    mask, T_1_0 = p3p.ransac(pair, matched_fps, get_pose=True)
    R_err, t_err = poseError(pair, T_1_0)
    return mask, R_err, t_err


def evaluatePair(pair, forward_passes):
    if type(pair) in [rpg_datasets_py.hpatches.HPatchPair,
                      sequences.PairWithIntermediates]:
        inl_mask = system.getInliersUsingCorrespondences(pair, forward_passes)
        R_err, t_err = None, None
    elif type(pair) == sequences.PairWithStereo:
        inl_mask, R_err, t_err = p3pMaskAndError(pair, forward_passes)
    elif type(pair) == datasets.DTUPair:
        inl_mask = system.getInliersUsingCorrespondences(pair, forward_passes)
        R_err, t_err = None, None
    else:
        raise NotImplementedError
    apparent = np.count_nonzero(inl_mask)
    inliers = forward_passes[0].ips_rc[:, inl_mask].T
    unique_inliers = []
    for inlier in inliers:
        if len(unique_inliers) == 0 or np.all(np.linalg.norm(
                np.array(unique_inliers) - inlier, axis=1) > 3):
            unique_inliers.append(inlier)
    true = len(unique_inliers)
    print('Apparent, true inliers: %d %d' % (apparent, true))
    return apparent, true, inl_mask, R_err, t_err


def evaluatePairs(forward_passer, pairs):
    apparent_inliers = []
    true_inliers = []
    inlierness_stats = []
    R_errs = []
    t_errs = []
    for pair in pairs:
        fps = [forward_passer(i) for i in pair.im]

        fps = system.unifyForwardPasses(fps)

        ips_rc = [fp.ips_rc for fp in fps]
        ips_scores = [fp.ip_scores for fp in fps]
        print(pair.name())
        apparent, true, mask, R_err, t_err = evaluatePair(pair, fps)
        apparent_inliers.append(apparent)
        true_inliers.append(true)
        for i in [0, 1]:
            try:
                inlierness_stats.append(np.vstack((
                    np.arange(ips_rc[i].shape[1]), ips_scores[i],
                    mask.astype(int))).T)
            except ValueError as e:
                print(type(fps[0]))
                print(ips_rc[i].shape)
                print(ips_scores[i].shape)
                print(mask.shape)
                raise e
        R_errs.append(R_err)
        t_errs.append(t_err)
    return np.array(apparent_inliers), np.array(true_inliers), \
           np.vstack(tuple(inlierness_stats)), R_errs, t_errs


def renderMatching(pair, fps):
    """ returns true inlier count, R_err and t_err """
    _, true_count, mask, R_err, t_err = evaluatePair(pair, fps)
    renderings = [i.render(fallback_im=im) for i, im in zip(fps, pair.im)]
    ims = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in pair.im]
    roffsets = [i * ims[0].shape[0] for i in [1, 2]]

    full_im = np.concatenate(
        [ims[0], renderings[0], renderings[1], ims[1]], axis=0)

    inl_ips = [fp.ips_rc[:, mask] for fp in fps]
    endpoints = [inl_ips[i] + np.array([[roffsets[i], 0]]).T for i in [0, 1]]

    rc = system.renderColors()
    inl_colors = rc[np.nonzero(mask)[0]]

    for i in range(endpoints[0].shape[1]):
        cv2.line(full_im, tuple(endpoints[0][[1, 0], i]),
                 tuple(endpoints[1][[1, 0], i]), tuple(inl_colors[i]), 2,
                 cv2.LINE_AA)

    outdir = os.path.join(
        'results', 'match_render', hyperparams.methodEvalString())
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, pair.name() + '.png')
    cv2.imwrite(outfile, full_im)

    return true_count, R_err, t_err


def renderTrainSample(pair, fps, corr_rc, inl):
    ims = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in pair.im]

    rc = system.renderColors()
    for fp, corr, im in zip(fps, corr_rc, ims):
        for i in range(128):
            cv2.circle(im, tuple(fp.ips_rc[[1, 0], i]), 6, tuple(rc[i]), 1,
                       cv2.LINE_AA)
            if inl[i]:
                thck = -1
            else:
                thck = 1
            cv2.circle(im, tuple(corr[[1, 0], i]), 2, tuple(rc[i]), thck,
                       cv2.LINE_AA)

    renderings = [i.render(with_points=False) for i in fps]
    gray_ims = np.concatenate(ims, axis=0)
    rend = np.concatenate(renderings, axis=0)
    full_im = np.concatenate([gray_ims, rend], axis=1)
    outdir = os.path.join(
        'results', 'train_samples', hyperparams.methodEvalString())
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, pair.name() + '.png')
    cv2.imwrite(outfile, full_im)
