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
import os
import time

import klt

FLAGS = absl.flags.FLAGS


class Pair(object):
    def __init__(self, images, K, T_0_1, seqname, indices):
        self.im = images
        self.K = K
        self.T_0_1 = T_0_1
        self.seqname = seqname
        assert len(indices) == 2
        self.indices = indices

    def name(self):
        return '%s %d %d' % (self.seqname, self.indices[0], self.indices[1])

    def imname(self, i):
        return '%s %d' % (self.seqname, self.indices[i])


class PairWithStereo(Pair):
    def __init__(self, left_images, left_K, T_0_1, seqname, indices,
                 right_image_0, right_K, baseline):
        Pair.__init__(self, left_images, left_K, T_0_1, seqname, indices)
        self.rim_0 = right_image_0
        self.rK = right_K
        self.baseline = baseline


class PairWithIntermediates(Pair):
    def __init__(self, image_sequence, K, T_0_1, seqname, firstlast_indices):
        Pair.__init__(self, [image_sequence[0], image_sequence[-1]], K, T_0_1,
                      seqname, firstlast_indices)
        self._sequence = image_sequence

    def trackThrough(self, ips_rc, reverse=False, border_margin=15):
        if reverse:
            order = range(len(self._sequence) -1, -1, -1)
        else:
            order = range(len(self._sequence))

        prev_im = self._sequence[order[0]]
        tracked_ips_rc = ips_rc
        tracked_indices = np.arange(ips_rc.shape[1])
        ordered_ims = [self._sequence[i] for i in order]
        for im in ordered_ims[1:]:
            status, tracked_ips_rc = klt.track(prev_im, tracked_ips_rc, im,
                                               border_margin=border_margin)
            tracked_indices = tracked_indices[status == klt.TRACK_SUCCESS]
            prev_im = im
        return tracked_ips_rc, tracked_indices

    def getInliers(self, ips_rc):
        corr_1_rc, forward_indices = self.trackThrough(ips_rc[0])
        corr_0_rc, backward_indices = self.trackThrough(ips_rc[1], reverse=True)

        def maskedInliers(ips, corrs, corr_indices):
            dist = np.linalg.norm(ips[:, corr_indices] - corrs, axis=0)
            corr_inlier_mask = dist < 3
            inlier_mask = np.zeros(ips.shape[1], dtype=bool)
            inlier_mask[corr_indices[corr_inlier_mask]] = True
            return inlier_mask

        return maskedInliers(ips_rc[0], corr_0_rc, backward_indices) & \
            maskedInliers(ips_rc[1], corr_1_rc, forward_indices)


class Wrapper(object):
    def __init__(self, sequence, pair_class):
        self.sequence = sequence
        self._pair_class = pair_class

    def makePair(self, indices):
        seq = self.sequence
        image_paths = [seq.images[i] for i in indices]
        images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in image_paths]
        if hasattr(seq, 'T_W_C'):
            T_W_C = [seq.T_W_C[i] for i in indices]
            T_0_1 = T_W_C[0].inverse() * T_W_C[1]
        else:
            assert self._pair_class == PairWithIntermediates
            T_0_1 = None
        if self._pair_class == PairWithStereo:
            assert hasattr(seq, 'right_images')
            right_image = cv2.imread(seq.right_images[indices[0]],
                                     cv2.IMREAD_GRAYSCALE)
            return PairWithStereo(
                images, seq.K, T_0_1, seq.name(), indices, right_image,
                seq.right_K, seq.baseline)
        elif self._pair_class == PairWithIntermediates:
            if indices[0] < indices[1]:
                images = [seq.images[i]
                          for i in range(indices[0], indices[1] + 1)]
                firstlast = indices
            else:
                images = [seq.images[i]
                          for i in range(indices[1], indices[0] + 1)]
                if T_0_1 is not None:
                    T_0_1 = T_0_1.inverse()
                firstlast = list(reversed(indices))
            ims = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in images]
            if hasattr(seq, 'K'):
                K = seq.K
            else:
                K = None
            return PairWithIntermediates(ims, K, T_0_1, seq.name(), firstlast)
        else:
            raise NotImplementedError(
                'Not implemented for %s' % self._pair_class)

    def __len__(self):
        return len(self.sequence)


def trackingHickleLocation(shortname):
    return os.path.join(os.path.dirname(__file__),  'tracked_indices',
                        shortname + '.hkl')


class TrackingPairPicker(object):
    def __init__(self, sequence, min_overlap, pair_class):
        self._wrapped = Wrapper(sequence, pair_class)
        self._min_overlap = min_overlap
        hickle = trackingHickleLocation(sequence.name())
        if os.path.exists(hickle):
            self._tracked_indices = klt.TrackedIndices(hickle=hickle)
        else:
            print('Need to compute tracked indices for %s...' %
                  sequence.name())
            if FLAGS.tds == 'kt':
                self._tracked_indices = klt.trackedIndicesFromSequence(
                    sequence.images, 1000, 5)
            elif FLAGS.tds == 'rc':
                self._tracked_indices = klt.trackedIndicesFromSequence(
                    sequence.images, 1000, 5)
            elif FLAGS.tds in ['tm', 'tmb']:
                self._tracked_indices = klt.trackedIndicesFromSequence(
                    sequence.images, 500, 5)
            else:
                assert False
            self._tracked_indices.dump(hickle)

    def getRandomDataPoint(self):
        im1_selection = np.zeros(0)
        while im1_selection.size == 0:
            im0_index = np.random.randint(0, len(self._wrapped))
            im1_selection = self._tracked_indices.laterFramesWithMinOverlap(
                im0_index, self._min_overlap)
        choice = np.random.randint(0, im1_selection.size)
        im1_index = im1_selection[choice]
        assert im1_index > im0_index

        return self._wrapped.makePair([im0_index, im1_index])


class FixPairs(list):
    def __init__(self, data_gen, n, seed=0):
        np.random.seed(seed)
        for _ in range(n):
            self.append(data_gen.getRandomDataPoint())
        np.random.seed(int(time.time()))


class MultiSeqTrackingPairPicker(object):
    def __init__(self, sequences, min_overlap, pair_classes):
        if type(pair_classes) is not list:
            pair_classes = [pair_classes] * len(sequences)
        self._sub_pickers = [
            TrackingPairPicker(seq_cl[0], min_overlap, seq_cl[1])
            for seq_cl in zip(sequences, pair_classes)]
        counts = np.array(
            [len(sequence) for sequence in sequences], dtype=float)
        self._pick_weights = counts / np.sum(counts)

    def getRandomDataPoint(self):
        this_time = np.random.choice(self._sub_pickers, p=self._pick_weights)
        return this_time.getRandomDataPoint()
