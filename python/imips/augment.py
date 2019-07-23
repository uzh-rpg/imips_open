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

import copy
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def generateHomography(debug_plot=False, ang_range=1., sc_range=1.):
    angle = ang_range * (np.random.random() - 0.5) * 2 * math.pi
    scale = 2 ** ((np.random.random() - 0.5) * sc_range)
    if debug_plot:
        print('Angle %f scale %f' % (np.rad2deg(angle), scale))
    R = scale * np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    result = np.eye(3)
    result[:2, :2] = R
    return result


def shiftHomography(shift_xy):
    result = np.eye(3)
    result [:2, 2] = shift_xy
    return result


def applyHomographyAtCenter(image, origin_homography):
    center_rc = np.array(image.shape) / 2
    center_xy = np.flip(center_rc, axis=0)
    shift = shiftHomography(-center_xy)
    unshift = shiftHomography(center_xy)
    H = np.dot(unshift, np.dot(origin_homography, shift))
    return cv2.warpPerspective(
        image, H, dsize=tuple([image.shape[1], image.shape[0]])), H


def augmentHpatchPair(pair, ang_range=1., sc_range=1.):
    result = copy.deepcopy(pair)
    origin_H = generateHomography(ang_range=ang_range, sc_range=sc_range)
    result.im[1], H = applyHomographyAtCenter(result.im[1], origin_H)
    result.H = np.dot(H, result.H)
    return result


def testHpatchPair(pair):
    plt.figure()
    plt.imshow(pair.im[1])
    plt.figure()
    plt.imshow(cv2.warpPerspective(
        pair.im[0], pair.H, dsize=tuple(
            [pair.im[1].shape[1], pair.im[1].shape[0]])))