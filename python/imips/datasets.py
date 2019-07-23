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
import numpy as np

import sequences


class DTUPair(sequences.Pair):
    def __init__(self, seq, indices):
        sequences.Pair.__init__(
            self, [cv2.imread(seq[i], cv2.IMREAD_GRAYSCALE) for i in indices],
            seq.K, seq.getT_A_B(indices), seq.name(), indices)
        self.seq = seq

    def getCorrespondence(self, ips_rc, reverse=False):
        corr_rc = []
        success_indices = []
        for i in range(ips_rc.shape[1]):
            indices = list(reversed(self.indices)) if reverse else self.indices
            corr = self.seq.getCorrespondence(
                ips_rc[:, i], indices[0], indices[1])
            if corr is not None:
                corr_rc.append(corr)
                success_indices.append(i)
        return np.array(corr_rc).T, success_indices


class DTUPicker(object):
    def __init__(self, sequences):
        self._sequences = sequences
        for seq in sequences:
            print(seq.name())

    def getRandomDataPoint(self):
        # np.random.choice(sequences) does not work.
        sequence = self._sequences[np.random.randint(0, len(self._sequences))]
        indices = np.random.choice(np.arange(119), 2, replace=False)
        return DTUPair(sequence, indices)
