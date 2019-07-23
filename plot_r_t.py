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

import imips.cache as cache
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


def plot():
    print(hyperparams.methodEvalString())

    _, true_inl, _, R_errs, t_errs = cache.getOrEval()
    assert R_errs[0] is not None

    plt.semilogy(
         true_inl, R_errs, 'o', label='R')
    plt.semilogy(
         true_inl, t_errs, 'v', label='t')


if __name__ == '__main__':
    print('\n\n%s\n\n' % hyperparams.shortString())
    plt.figure(figsize=(4, 4))

    if FLAGS.baseline == 'all':
        for bl in ['', 'sift', 'surf']:
            FLAGS.baseline = bl
            plot()
        FLAGS.baseline = 'all'
    else:
        plot()

    plt.xlabel('Inlier count')
    plt.ylabel('Error (degree, meters)')
    plt.ylim([0.001, 1000])
    plt.grid()
    plt.legend()

    plt.gcf().tight_layout()

    plt.show()
