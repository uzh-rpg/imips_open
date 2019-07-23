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
import scipy.spatial.distance as scidist

import flags
import hpatches
import hyperparams


if __name__ == '__main__':
    deploy_net, sess = hyperparams.bootstrapFromCheckpoint()
    eval_set = hyperparams.getEvalSet()
    fpasser = hyperparams.getForwardPasser(deploy_net, sess)

    _, true_inl = hpatches.evaluate(fpasser, eval_set)

    n = len(eval_set.folder_names)
    assert len(true_inl) == 15 * n

    w = 3
    h = int(np.ceil(float(n) / w))
    im = np.ones([8 * h, 8 * w]) * np.max(true_inl)
    title = hyperparams.methodEvalString() + ':'
    for i in range(n):
        row = i / w
        col = i % w
        squ = scidist.squareform(true_inl[i * 15:(i + 1) * 15])
        im[(row * 8 + 1):(row * 8 + 7), (col * 8 + 1):(col * 8 + 7)] = squ
        if col == 0:
            title = title + '\n'
        title = title + eval_set.folder_names[i] + ', '

    plt.imshow(im)
    plt.colorbar()
    plt.title(title)
    plt.show()
