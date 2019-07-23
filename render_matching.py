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

import imips.evaluate as evaluate
import imips.flags as flags
import imips.hyperparams as hyperparams
import imips.system as system

FLAGS = flags.FLAGS


if __name__ == '__main__':
    print(hyperparams.methodEvalString())
    eval_pairs = hyperparams.getEvalSet()
    graph, sess = hyperparams.bootstrapFromCheckpoint()
    forward_passer = hyperparams.getForwardPasser(graph, sess)

    for pair in eval_pairs:
        fps = [forward_passer(im) for im in pair.im]
        fps = system.unifyForwardPasses(fps)
        evaluate.renderMatching(pair, fps)
