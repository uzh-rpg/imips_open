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

from absl import flags
import cv2
import numpy as np

FLAGS = flags.FLAGS


def withinRadiusOfBorder(point_rc, image_shape, radius):
    below = np.any(point_rc < radius)
    above = np.any(point_rc >= np.array(image_shape) - radius)
    return below or above


class ForwardPass(object):
    def __init__(self, ips_rc, ip_scores, descriptors, scales=1.,
                 image_scores=None):
        assert descriptors.shape[0] == ips_rc.shape[1]
        self.ips_rc = ips_rc
        assert ips_rc.dtype == int
        self.ip_scores = ip_scores
        self.descriptors = descriptors
        self.scales = scales
        if type(self.scales) == np.ndarray:
            assert len(scales) == ips_rc.shape[1]
        self.image_scores = image_scores

    def __getitem__(self, slc):
        # Only applies for subselection; matched forward passes are ok to have
        # non-ordered score.
        assert np.all(self.ip_scores[:-1] >= self.ip_scores[1:])
        if type(self.scales) == np.ndarray:
            scales = self.scales[slc]
        else:
            scales = self.scales
        return ForwardPass(self.ips_rc[:, slc], self.ip_scores[slc],
                           self.descriptors[slc, :], scales, self.image_scores)

    def __len__(self):
        return len(self.ip_scores)

    def reducedTo(self, num_ips):
        return self[:num_ips]

    def hicklable(self):
        return [self.ips_rc, self.ip_scores, self.descriptors, self.scales]

    def render(self, n_max=0, fallback_im=None):
        if self.image_scores is not None:
            im = cv2.applyColorMap((self.image_scores * 255).astype(np.uint8),
                                   cv2.COLORMAP_JET)
        else:
            assert fallback_im is not None
            im = cv2.cvtColor(fallback_im, cv2.COLOR_GRAY2BGR)

        if n_max == 0:
            n_max = self.ips_rc.shape[1]
        for i in range(n_max):
            thickness_relevant_score = \
                np.clip(self.ip_scores[i], 0.2, 0.6) - 0.2
            thickness = int(thickness_relevant_score * 20)
            if type(self.scales) == np.ndarray:
                radius = int(self.scales[i] * 10)
            else:
                radius = 10
            cv2.circle(im, tuple(self.ips_rc[[1, 0], i]),
                       radius, (0, 255, 0), thickness, cv2.LINE_AA)
        return im


def match(forward_passes):
    if FLAGS.baseline == 'orb':
        print('Using Hamming matching for ORB...')
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(forward_passes[0].descriptors,
                            forward_passes[1].descriptors)
    return [np.array([x.queryIdx for x in matches]),
            np.array([x.trainIdx for x in matches])]
