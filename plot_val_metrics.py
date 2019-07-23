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

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import os

import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


def doPlot(_):
    plt.clf()
    path = hyperparams.trainStatsPath()
    if os.path.exists(path):

        stats = np.loadtxt(path)
        if len(stats.shape) < 2:
            stats = stats.reshape([1, stats.size])

        plt.plot(stats[:, 0], stats[:, 1], label='Apparent inliers')
        plt.plot(stats[:, 0], stats[:, 2], label='True inliers')
        plt.plot(stats[:, 0], stats[:, 3], '--',
                 label='Apparent inliers (train)')
        plt.plot(stats[:, 0], stats[:, 4], '--', label='True inliers (train)')
        plt.grid()
        plt.title(hyperparams.shortString())
        plt.ylabel('counts')
        plt.ylim(bottom=0)
        plt.xlabel('training iterations')
        plt.legend()
    else:
        plt.title('Checkpoint not yet around.')


if __name__ == '__main__':
    fig = plt.figure()

    doPlot(None)

    ani = matplotlib.animation.FuncAnimation(
        fig, doPlot, repeat=True, interval=1000)

    plt.show()