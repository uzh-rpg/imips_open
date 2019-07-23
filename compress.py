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
import numpy as np
import os
import sklearn.neighbors

import rpg_datasets_py.tum_mono

import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


if __name__ == '__main__':
    assert FLAGS.tds == 'tm'
    assert FLAGS.baseline == 'sift'

    # Collect all (keyframe) descriptors
    dump_file = os.path.join('compress', 'tm_%s.hkl' % FLAGS.baseline)
    if not os.path.exists(dump_file):
        tms = [rpg_datasets_py.tum_mono.Sequence(i)
               for i in ['01', '02', '03', '48', '49', '50']]
        assert FLAGS.baseline_num_ips == 500
        fpasser = hyperparams.getForwardPasser(None, None)

        all_descs = []
        for tm in tms:
            kf_indices = getKeyFrames(tm, 0.5)
            print(tm.name())
            print('Frames reduced from %d to %d' % (len(tm), len(kf_indices)))
            for index in kf_indices:
                print('%d %d' % (index, kf_indices[-1]))
                image = cv2.imread(tm[index], cv2.IMREAD_GRAYSCALE)
                fp = fpasser(image)
                all_descs = all_descs + fp.descriptors.tolist()

        descs_array = np.array(all_descs)
        hkl.dump(descs_array, open(dump_file, 'w'))
    else:
        all_descs = hkl.load(open(dump_file, 'r'))

    # Calculate PCA
    dump_file = os.path.join('compress', 'tm_%s_pca.hkl' % FLAGS.baseline)
    if not os.path.exists(dump_file):
        pca = sklearn.decomposition.PCA(whiten=True)
        pca.fit(all_descs)
        hkl.dump([pca.mean_, pca.components_, pca.explained_variance_],
                 open(dump_file, 'w'))

    # Calculate kmeans
    dump_file = os.path.join(
        'compress', 'tm_%s_pq_%d_%d_%d.hkl' % (
            FLAGS.baseline, FLAGS.pq_m, FLAGS.pq_k, FLAGS.pq_n_init))
    if not os.path.exists(dump_file):
        centers = []
        assert all_descs.shape[1] % FLAGS.pq_m == 0
        dims_per = all_descs.shape[1] / FLAGS.pq_m
        for i in range(FLAGS.pq_m):
            # Somehow, n_jobs does not seem to work...
            km = sklearn.cluster.KMeans(n_clusters=FLAGS.pq_k, verbose=1,
                                        n_init=FLAGS.pq_n_init)
            print('\n\nFit %d\n\n' % i)
            km.fit(all_descs[:, (i * dims_per):((i + 1) * dims_per)])
            centers.append(km.cluster_centers_)
        hkl.dump(centers, open(dump_file, 'w'))
