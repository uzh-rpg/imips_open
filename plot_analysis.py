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

import absl.app
import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import os

import imips.cache as cache
import imips.flags as flags
import imips.hpatches as hpatches
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


def main(_):

    print('\n\n%s\n\n' % hyperparams.shortString())

    _, _, stats, _, _ = cache.getOrEval()

    frequencies = np.zeros(128)
    for i in range(128):
        frequencies[i] = np.mean(stats[stats[:, 0] == i, 2])

    # Inlierness over channel
    plt.figure(0, figsize=[5, 2.5])
    plt.stem(frequencies)
    plt.xlabel('Channel #')
    plt.ylim(ymin=0)
    plt.ylabel('Inlier frequency')
    plt.tight_layout()
    ax = plt.axes()
    ax.yaxis.grid()
    plt.show()

    # Inlierness VS prediction
    num_bins = 20
    bins = [[] for _ in range(num_bins)]
    for row in range(stats.shape[0]):
        index = int(stats[row, 1] * num_bins)
        bins[index].append(stats[row, 2])

    bin_frequencies = np.array([np.mean(bin_) for bin_ in bins])

    counts = np.array([len(bin_) for bin_ in bins])

    bin_width = float(1) / num_bins

    fig, ax1 = plt.subplots(figsize=[5, 2.5])
    ax1.hlines(bin_frequencies, bin_width * np.arange(num_bins),
               bin_width * (np.arange(num_bins) + 1), colors='red', linewidth=3)
    ax1.set_xlabel('Response')
    ax1.set_ylabel('Inlierness frequency', color='r')
    ax1.grid()
    ax1.set_xlim([0, 1])

    ax2 = ax1.twinx()
    ax2.set_ylabel('# points with given prediction', color='b')
    ax2.stem(bin_width * (np.arange(num_bins) + 0.5), counts, color='b',
             basefmt='')
    print(counts)
    ax2.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    absl.app.run(main)
