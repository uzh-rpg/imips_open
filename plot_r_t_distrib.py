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

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os

import rpg_common_py.geometry

import imips.cache as cache
import imips.evaluate as evaluate
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


if __name__ == '__main__':
    print(hyperparams.methodEvalString())
    eval_pairs = hyperparams.getEvalSet()

    _, true_counts, _, eR, et = cache.getOrEval()
    dR, dt = [], []
    for pair in eval_pairs:
        T = pair.T_0_1
        dR.append(math.degrees(rpg_common_py.geometry.getRotationAngle(T.R)))
        dt.append(np.linalg.norm(T.t))
    dR = np.array(dR)
    dt = np.array(dt)

    # Save csv
    csv_dir = os.path.join('results', 'csv',
                           hyperparams.methodEvalString())
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, '%s.csv' % hyperparams.methodEvalString())
    assert FLAGS.chan == 128
    csv_vals = zip([i.name() for i in eval_pairs], dR, dt,
                   true_counts.astype(float) / 128., eR, et)
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['name', 'dR', 'dt', 'matching score', 'eR', 'et'])
        for val in csv_vals:
            writer.writerow(val)

    mt10filt = true_counts >= 10
    lt10filt = true_counts < 10
    plt.figure(figsize=(5, 3))
    plt.plot(dR[mt10filt], dt[mt10filt], '^', label='>= 10 inliers')
    plt.plot(dR[lt10filt], dt[lt10filt], 'v', label='< 10 inliers')
    plt.grid()
    plt.legend()
    plt.xlabel('angle difference [degrees]')
    plt.ylabel('distance [meters]')
    plt.title('Dataset characterization %s' % hyperparams.evalString())
    plt.tight_layout()
    plt.show()
