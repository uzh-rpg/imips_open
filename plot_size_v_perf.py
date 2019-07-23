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

import imips.cache as cache
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


def accuracy(R_errs, t_errs):
    if FLAGS.eds == 'eu':
        R_thresh = 3.
    elif FLAGS.ds == 'kt':
        R_thresh = 1.
    else:
        raise NotImplementedError
    R_filt = np.array(R_errs) < R_thresh
    if FLAGS.eds == 'eu':
        t_thresh = 0.1
    elif FLAGS.ds == 'kt':
        t_thresh = 0.3
    else:
        raise NotImplementedError
    t_filt = np.array(t_errs) < t_thresh
    return float(np.count_nonzero(R_filt & t_filt)) / R_filt.size * 100


def repsizeAcc():
    print(hyperparams.methodEvalString())
    _, _, _, R_errs, t_errs = cache.getOrEval()
    acc = accuracy(R_errs, t_errs)
    if FLAGS.baseline == '':
        repsize = FLAGS.chan * 3
    elif FLAGS.baseline == 'sift_pq':
        if FLAGS.pq_m == 4 and FLAGS.pq_k == 16:
            repsize = FLAGS.baseline_num_ips * 5
        elif FLAGS.pq_m == 4 and FLAGS.pq_k == 256:
            repsize = FLAGS.baseline_num_ips * 7
        elif FLAGS.pq_m == 1 and FLAGS.pq_k == 256:
            repsize = FLAGS.baseline_num_ips * 4
        elif FLAGS.pq_m == 2 and FLAGS.pq_k == 16:
            repsize = FLAGS.baseline_num_ips * 4
        elif FLAGS.pq_m == 4 and FLAGS.pq_k == 4:
            repsize = FLAGS.baseline_num_ips * 4
        elif FLAGS.pq_m == 2 and FLAGS.pq_k == 256:
            repsize = FLAGS.baseline_num_ips * 5
        elif FLAGS.pq_m == 8 and FLAGS.pq_k == 4:
            repsize = FLAGS.baseline_num_ips * 5
        else:
            raise NotImplementedError
    elif FLAGS.baseline == 'sift_pca':
        repsize = FLAGS.baseline_num_ips * (
                3 + FLAGS.pca_dim * 4)
    else:
        desc_coeffs = {'sift': 128, 'surf': 64, 'super': 256, 'lfnet': 256,
                       'orb': 8}
        repsize = FLAGS.baseline_num_ips * (
                3 + desc_coeffs[FLAGS.baseline] * 4)

    return repsize, acc


def label():
    if FLAGS.baseline == '':
        return 'ours'
    else:
        return FLAGS.baseline


def plot2():
    x = []
    y = []

    if FLAGS.baseline == '':
        ochan = FLAGS.chan
        for chan in [64, 128, 256]:
            FLAGS.chan = chan
            repsize, acc = repsizeAcc()
            x.append(repsize)
            y.append(acc)
        FLAGS.chan = ochan
    elif FLAGS.baseline == 'sift_pca':
        for pca_dim in [3, 5, 10]:
            FLAGS.pca_dim = pca_dim
            for num in [64, 128, 256, 500, 1000]:
                FLAGS.baseline_num_ips = num
                repsize, acc = repsizeAcc()
                x.append(repsize)
                y.append(acc)
            x.append(None)
            y.append(None)
    elif FLAGS.baseline == 'sift_pq':
        for m, k, i in [(4, 16, 10), (2, 256, 10), (2, 16, 10), (8, 4, 10),
                        (4, 4, 10), (1, 256, 10)]:
            FLAGS.pq_m = m
            FLAGS.pq_k = k
            FLAGS.pq_n_init = i
            for num in [64, 128, 256, 500, 1000]:
                FLAGS.baseline_num_ips = num
                repsize, acc = repsizeAcc()
                x.append(repsize)
                y.append(acc)
            x.append(None)
            y.append(None)
    else:
        for num in [64, 128, 256, 500, 1000]:
            FLAGS.baseline_num_ips = num
            repsize, acc = repsizeAcc()
            x.append(repsize)
            y.append(acc)

    plt.semilogx(x, y, '--o',
                 label='%s' % (label()),
                 color=hyperparams.methodColor())


if __name__ == '__main__':
    plt.figure(figsize=[5, 2.5])
    assert FLAGS.baseline == 'all'

    for bl in ['', 'sift', 'surf', 'super', 'lfnet', 'sift_pca', 'sift_pq',
               'orb']:
        FLAGS.baseline = bl
        ods = FLAGS.ds
        if bl != '':
            if FLAGS.eds == 'eu':
                FLAGS.ds = 'eu'
        plot2()
        FLAGS.ds = ods
    FLAGS.baseline = 'all'

    plt.grid()
    plt.legend(ncol=3)
    plt.xlabel('Representation size [bytes]')
    plt.ylabel('Accuracy [%]')
    plt.ylim([50, 100])
    plt.xlim([100, 1000000])
    plt.tight_layout()
    plt.show()
