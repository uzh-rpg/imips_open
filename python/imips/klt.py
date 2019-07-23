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
import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as scidist
import sklearn.neighbors

import points

TRACK_SUCCESS = 1
BORDER_LOST = 2
POOR_TRACK = 3


def track(img_0, ips_rc_0, img_1, border_margin=15):
    """ Returns keypoint tracking status and the set
        of tracked keypoints. Keypoints given colwise and in (row, col).
        Status is TRACK_SUCCESS, BORDER_LOST or POOR_TRACK
        """
    status = np.ones(ips_rc_0.shape[1], dtype=int) * TRACK_SUCCESS
    if ips_rc_0.size == 0:
        return status, np.zeros_like(ips_rc_0)

    ips_xy_0 = np.reshape(np.fliplr(ips_rc_0.T), (-1, 1, 2)).astype(np.float32)
    ips_xy_1, _, _ = cv2.calcOpticalFlowPyrLK(
        img_0, img_1, ips_xy_0, None)

    # Point lost at border.
    ips_rc_1_for_bordlost = np.fliplr(np.reshape(ips_xy_1, (-1, 2))).T
    not_bordlost = points.haveMarginToBorder(
        ips_rc_1_for_bordlost, img_1.shape, border_margin)

    # Symmetry: Reject all that don't come back to within 1 px:
    re_ips_xy_0, _, _ = cv2.calcOpticalFlowPyrLK(
        img_1, img_0, ips_xy_1, None)
    err = np.linalg.norm(re_ips_xy_0 - ips_xy_0, axis=2)
    tracked = (err < 1).ravel()
    status[np.logical_not(tracked)] = POOR_TRACK
    status[np.logical_not(not_bordlost)] = BORDER_LOST

    successful = np.logical_and(not_bordlost, tracked)

    ips_xy_1 = ips_xy_1[successful, :, :]
    ips_rc_1 = np.fliplr(np.reshape(ips_xy_1, (-1, 2))).T

    assert (np.count_nonzero(status == TRACK_SUCCESS) == ips_rc_1.shape[1])

    return status, ips_rc_1


class TrackedIndices(object):
    def __init__(self, first_n_kps=None, existing_track=None, hickle=None):
        if existing_track is not None:
            self.tracked_ids = existing_track
            self.nextid = existing_track[-1][-1] + 1
        elif first_n_kps is not None:
            self.tracked_ids = [np.array(range(first_n_kps))]
            self.nextid = first_n_kps
        else:
            assert hickle is not None
            self.tracked_ids, self.nextid = hkl.load(open(hickle, 'r'))

    def dump(self, filename):
        hkl.dump([self.tracked_ids, self.nextid], open(filename, 'w'))

    def addFrame(self, tracked_mask, n_new_kps):
        self.tracked_ids.append(np.hstack((
            self.tracked_ids[-1][tracked_mask],
            np.array(range(n_new_kps)) + self.nextid)))
        self.nextid = self.nextid + n_new_kps

    def plot(self):
        frame_ids = [i * np.ones(self.tracked_ids[i].size)
                     for i in range(len(self.tracked_ids))]
        frame_ids = np.hstack(frame_ids)
        kp_ids = np.hstack(self.tracked_ids)
        plt.plot(kp_ids, frame_ids, ls='None', marker='.', ms=1)

    def plotFrameOverlap(self):
        grid = np.zeros((len(self.tracked_ids), len(self.tracked_ids)))
        for i in range(len(self.tracked_ids)):
            grid[i, i] = 1.
            for j in range(i + 1, len(self.tracked_ids)):
                overlap = np.intersect1d(
                    self.tracked_ids[i], self.tracked_ids[j], True)
                if overlap.size == 0:
                    break
                grid[i, j] = float(overlap.size) / self.tracked_ids[i].size
        plt.imshow(grid)
        plt.colorbar()

    def laterFramesWithMinOverlap(self, frame_id, min_overlap):
        frames = []
        if self.tracked_ids[frame_id].size != 0:
            for j in range(frame_id + 1, len(self.tracked_ids)):
                overlap = np.intersect1d(
                    self.tracked_ids[frame_id], self.tracked_ids[j], True)
                ratio = float(overlap.size) / self.tracked_ids[frame_id].size
                if ratio < min_overlap:
                    break
                frames.append(j)
        return np.array(frames)


def squf(i, n):
    return i * n - (i * (i + 1)) / 2


class FASTAdder(object):
    def __init__(self, target_n, nonmax_radius):
        self._scorer = cv2.FastFeatureDetector_create()
        self._target_n = target_n
        self._nonmax_radius = nonmax_radius

    def __call__(self, image, existing_ips_rc):
        assert existing_ips_rc.shape[0] == 2

        cv_kps = self._scorer.detect(image)
        fast_ips_rc, responses = points.fromCvKpts(cv_kps, response=True)

        # Suppress close to existing
        if existing_ips_rc.shape[1] > 0:
            nn = sklearn.neighbors.NearestNeighbors(
                n_neighbors=1, metric='chebyshev').fit(existing_ips_rc.T)
            dists, _ = nn.kneighbors(fast_ips_rc.T)
            filt = (dists > self._nonmax_radius).ravel()
            fast_ips_rc = fast_ips_rc[:, filt]
            responses = responses[filt]

        if fast_ips_rc.size == 0:
            print("Warning: No interest points found!")
            return np.zeros([2, 0], dtype=int)

        # Needs to be sorted low to high for following code to work:
        try:
            sorted_cand_rc = fast_ips_rc[:, np.argsort(responses)]
        except IndexError as e:
            print(responses.shape)
            print(fast_ips_rc.shape)
            print(existing_ips_rc.shape)
            raise e

        # Suppress close to self
        dists = scidist.pdist(sorted_cand_rc.T, metric='chebyshev')
        n = sorted_cand_rc.shape[1]
        # Might seem concocted in this form, but significant performance boost
        # VS applying scidist.squareform().
        filt = np.array([not np.any(
            dists[squf(i, n):squf(i+1, n)] < self._nonmax_radius)
                         for i in xrange(n)])
        sorted_cand_rc = sorted_cand_rc[:, filt]
        # Sort high to low for rest:
        sorted_cand_rc = np.fliplr(sorted_cand_rc)

        n_existing = existing_ips_rc.shape[1]
        n_new = self._target_n - n_existing
        return sorted_cand_rc[:, :n_new]


def trackedIndicesFromSequence(image_sequence, target_n, nms_radius,
                               do_plot=False):
    prev_img = cv2.imread(image_sequence[0])
    fast_adder = FASTAdder(target_n, nms_radius)
    ips_rc = fast_adder(prev_img, np.zeros([2, 0]))
    tracked_indices = TrackedIndices(first_n_kps=ips_rc.shape[1])

    if do_plot:
        plt.clf()
        points.plotOnImage(ips_rc, prev_img)
        plt.show()
        plt.pause(0.001)
    print('%d / %d' % (0, len(image_sequence)))

    i = 1
    for image_file in image_sequence[1:]:
        print(image_file)
        curr_img = cv2.imread(image_file)
        status, ips_rc = track(prev_img, ips_rc, curr_img)
        success = status == TRACK_SUCCESS
        new_ips_rc = fast_adder(curr_img, ips_rc)
        tracked_indices.addFrame(success, new_ips_rc.shape[1])
        ips_rc = np.hstack((ips_rc, new_ips_rc))

        if do_plot:
            plt.clf()
            points.plotOnImage(ips_rc, curr_img)
            plt.show()
            plt.pause(0.001)
        print('%d / %d' % (i, len(image_sequence)))

        prev_img = curr_img
        i = i + 1

    return tracked_indices
