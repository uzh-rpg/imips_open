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
import numpy as np
import os
import sklearn.cluster
import sklearn.decomposition
import sklearn.neighbors

import flags
import klt
import sequences

FLAGS = flags.FLAGS


class PCA(object):
    def __init__(self, mean, components, eva):
        self._mean = mean
        self._components = components
        self._eva = eva

    def transform(self, X):
        out = X - self._mean
        out = np.dot(out, self._components.T)
        out /= np.sqrt(self._eva)
        return out


class ProductQuantization(object):
    def __init__(self, centers):
        # centers: list of arrays of centers (one per subdiv)
        self._dims = [center.shape[1] for center in centers]
        self._nn = [sklearn.neighbors.NearestNeighbors(
            n_neighbors=1).fit(center) for center in centers]

    def transform(self, X):
        part = copy.copy(X)
        coeffs = []
        for i in range(len(self._dims)):
            coeffs.append(np.squeeze(self._nn[i].kneighbors(
                part[:, :self._dims[i]])[1]))
            part = part[:, self._dims[i]:]
        return np.array(coeffs).T


def getKeyFrames(sequence, olap):
    hickle = sequences.trackingHickleLocation(sequence.name())
    assert os.path.exists(hickle)
    tracked_indices = klt.TrackedIndices(hickle=hickle)
    indices = [0]
    while True:
        selection = tracked_indices.laterFramesWithMinOverlap(indices[-1], olap)
        next_kf = selection[-1] + 1
        if next_kf >= len(sequence):
            break
        indices.append(next_kf)
    return indices
