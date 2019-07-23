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

import cachetools
import cv2
import numpy as np

import rpg_common_py.pose

import sequences
import stereo


class StereoCache(cachetools.Cache):
    def __missing__(self, key):
        raise Exception('Need to pre-allocate cache contents!')


def pointsFromStereo(images, points_rc, K, baseline):
    try:
        disps = stereo.getDisparity(images[0], images[1], points=points_rc,
                                    max_disp=100)
    except TypeError as e:
        print(points_rc)
        raise e
    points_3d = stereo.disparityToPointCloud(disps, K, baseline, images[0],
                                             points=points_rc)
    return points_3d, disps > 0


def ransac(stereo_pair, matched_fps, stereo_cache=None, get_pose=False,
           refine=False):
    assert type(stereo_pair) == sequences.PairWithStereo

    if stereo_cache is not None:
        raise NotImplementedError
        P_C0, full_has_depth = stereo_cache[stereo_pair.imname(0)]
    else:
        P_C0, has_depth = pointsFromStereo(
            [stereo_pair.im[0], stereo_pair.rim_0], matched_fps[0].ips_rc,
            stereo_pair.K, stereo_pair.baseline)

    valid_ips1 = matched_fps[1].ips_rc[:, has_depth]

    if np.count_nonzero(has_depth) < 5:
        print('Warning: Not enough depths!')
        if not get_pose:
            return np.zeros_like(has_depth).astype(bool)
        else:
            return np.zeros_like(has_depth).astype(bool), None

    assert P_C0 is not None
    assert valid_ips1 is not None
    try:
        if np.count_nonzero(has_depth) == 4:
            # For some reason, this does not want to work...
            _, rvec, tvec = cv2.solvePnP(
                objectPoints=np.float32(P_C0.T),
                imagePoints=np.fliplr(np.float32(valid_ips1.T)),
                cameraMatrix=stereo_pair.K, distCoeffs=None,
                flags=cv2.SOLVEPNP_P3P)
            inliers = np.ones(4, dtype=bool)
        else:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.float32(P_C0.T), np.fliplr(np.float32(valid_ips1.T)),
                stereo_pair.K, distCoeffs=None, reprojectionError=3,
                flags=cv2.SOLVEPNP_P3P, iterationsCount=3000)
    except cv2.error as e:
        print(P_C0)
        print(valid_ips1)
        raise e

    if inliers is None:
        print('Warning: Ransac returned None inliers!')
        if not get_pose:
            return np.zeros_like(has_depth).astype(bool)
        else:
            return np.zeros_like(has_depth).astype(bool), None

    inliers = np.ravel(inliers)
    if refine:
        _, rvec, tvec = cv2.solvePnP(
            objectPoints=np.float32(P_C0.T)[inliers],
            imagePoints=np.fliplr(np.float32(valid_ips1.T)[inliers]),
            cameraMatrix=stereo_pair.K, distCoeffs=None, rvec=rvec, tvec=tvec,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    inl_indices = np.nonzero(has_depth)[0][inliers]
    inlier_mask = np.bincount(
        inl_indices, minlength=has_depth.size).astype(bool)

    if not get_pose:
        return inlier_mask
    else:
        T_1_0 = rpg_common_py.pose.Pose(cv2.Rodrigues(rvec)[0], tvec)
        return inlier_mask, T_1_0
