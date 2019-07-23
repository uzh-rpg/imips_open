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
import time

import imips.baselines as baselines
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS

eval_set = hyperparams.getEvalSet()


def evalSift():
    # SIFT
    print('SIFT')
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher()
    times = []
    for pair in eval_set:
        t = time.time()
        kpts = [sift.detect(im) for im in pair.im]
        srt = [sorted(kpt, key=lambda x: x.response, reverse=True) for kpt in kpts]
        srt128 = [s[:128] for s in srt]
        d = [sift.compute(im, s)[1] for im, s in zip(pair.im, srt128)]
        matches = matcher.match(d[0], d[1])
        times.append(time.time() - t)
        print(times[-1])
    return times


def evalSurf():
    # SIFT
    print('SURF')
    sift = cv2.xfeatures2d.SURF_create()
    matcher = cv2.BFMatcher()
    times = []
    for pair in eval_set:
        t = time.time()
        kpts = [sift.detect(im) for im in pair.im]
        srt = [sorted(kpt, key=lambda x: x.response, reverse=True) for kpt in kpts]
        srt128 = [s[:128] for s in srt]
        d = [sift.compute(im, s)[1] for im, s in zip(pair.im, srt128)]
        matches = matcher.match(d[0], d[1])
        times.append(time.time() - t)
        print(times[-1])
    return times


def evalImips():
    net, sess = hyperparams.bootstrapFromCheckpoint()
    fpasser = hyperparams.getForwardPasser(net, sess)
    times = []
    for pair in eval_set:
        t = time.time()
        fps = [fpasser(i) for i in pair.im]
        times.append(time.time() - t)
        print(times[-1])
    return times

def evalLfNet():
    # lf_net took 2:49 for 2761 images according to its own progress bar timer
    # 169/2761*2 -> 0.122 for two images
    # Add to that the following time for matching (0.48ms -> negligible):
    seq_fps = baselines.parseLFNetOuts(
        eval_set, FLAGS.baseline_num_ips)
    matcher = cv2.BFMatcher()
    times = []
    for pair in eval_set:
        folder = pair.seqname
        [a, b] = pair.indices
        forward_passes = [seq_fps['%s%s' % (folder, i)] for i in [a, b]]
        t = time.time()
        matches = matcher.match(
            forward_passes[0].descriptors, forward_passes[1].descriptors)
        times.append(time.time() - t)
        print(times[-1])
    return times


if FLAGS.baseline == 'sift':
    times = evalSift()
if FLAGS.baseline == 'surf':
    times = evalSurf()
if FLAGS.baseline == 'lfnet':
    times = evalLfNet()
if FLAGS.baseline == '':
    times = evalImips()
# Superpoint timed 4:52 / 292s for 2761 images -> 212 for two images
print('Mean is')
print(np.mean(np.array(times[1:])))
