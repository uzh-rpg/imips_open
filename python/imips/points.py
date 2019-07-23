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

import matplotlib.pyplot as plt
import numpy as np


def hasMarginToBorder(point_rc, image_shape, margin):
    rc_ok = [margin <= point_rc[i] < image_shape[i] - margin for i in [0, 1]]
    return all(rc_ok)


def haveMarginToBorder(points_rc, image_shape, margin):
    assert points_rc.shape[0] == 2
    rc_bigger = [margin <= points_rc[i, :] for i in [0, 1]]
    bigger = np.logical_and(rc_bigger[0], rc_bigger[1])
    rc_smaller = [points_rc[i, :] < image_shape[i] - margin for i in [0, 1]]
    smaller = np.logical_and(rc_smaller[0], rc_smaller[1])
    return np.logical_and(bigger, smaller)


def plotOnImage(points_rc, image, color='r', imcmap='gray'):
    plt.imshow(image, cmap=imcmap)
    plt.plot(points_rc[1, :], points_rc[0, :], ls='None', marker='x', ms=5,
             c=color)


def fromCvKpts(cv_kps, response=False):
    result = np.array([[i.pt[1], i.pt[0]] for i in cv_kps]).T
    if response:
        responses = np.array([i.response for i in cv_kps])
        return result, responses
    else:
        return result
