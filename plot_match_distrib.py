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

from IPython.core import ultratb
import matplotlib.pyplot as plt
import numpy as np
import sys

import imips.cache as cache
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


def plot(dashed=False):
    print(hyperparams.methodEvalString())

    _, true_inl, _, _, _ = cache.getOrEval()

    true_inl = sorted(true_inl)
    mscores = np.array(true_inl, dtype=float) / float(hyperparams.methodNPts())

    yax = np.arange(len(mscores)).astype(float) / len(mscores)

    if dashed:
        style = '--'
    else:
        style = '-'

    plt.step(mscores, np.flip(yax), linestyle=style,
             label='%s: %.02f' % (hyperparams.label(), np.mean(mscores)),
             color=hyperparams.methodColor())


if __name__ == '__main__':
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux', call_pdb=1)
    print('\n\n%s\n\n' % hyperparams.shortString())
    plt.figure(figsize=(5, 5))

    if FLAGS.baseline == 'all':
        for bl in ['', 'sift', 'surf', 'super', 'lfnet', 'orb']:
            FLAGS.baseline = bl
            if bl != '':
                if FLAGS.eds == 'eu':
                    FLAGS.ds = 'eu'
                for nps in [128, 500]:
                    FLAGS.baseline_num_ips = nps
                    plot(dashed=nps == 500)
            else:
                plot()
        FLAGS.baseline = 'all'
    else:
        if FLAGS.baseline != '' and FLAGS.eds == 'eu':
            FLAGS.ds = 'eu'
        plot()

    if hyperparams.evalDs() in ['kt', 'eu']:
        plt.axvline(x=10./128., label='10 inliers at 128', color='black')

    plt.grid()
    plt.legend()
    plt.xlabel('Matching score')
    plt.ylabel('Fraction of pairs with higher matching score')
    plt.ylim([0, 1])
    plt.show()
