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

import absl.flags
import hickle as hkl
import os

import baselines
import evaluate
import hyperparams

FLAGS = absl.flags.FLAGS


def path():
    cache_root = os.path.join('results', 'cache')
    if not os.path.exists(cache_root):
        os.makedirs(cache_root)
    return os.path.join(cache_root, hyperparams.methodEvalString() + '.hkl')


def cache(n_apparent, n_true, inl_stats, R_errs, t_errs):
    hkl.dump([n_apparent, n_true, inl_stats, R_errs, t_errs], open(path(), 'w'))


def hasCache():
    return os.path.exists(path())


def getFromCache():
    return hkl.load(open(path(), 'r'))


def getOrEval():
    if hasCache():
        return getFromCache()
    else:
        val_set = hyperparams.getEvalSet()
        if FLAGS.baseline in ['super', 'lfnet']:
            n_true, R_errs, t_errs = baselines.evaluate(val_set)
            n_apparent, inl_stats = None, None
        else:
            deploy_net, sess = hyperparams.bootstrapFromCheckpoint()
            fpasser = hyperparams.getForwardPasser(deploy_net, sess)
            n_apparent, n_true, inl_stats, R_errs, t_errs = \
                evaluate.evaluatePairs(fpasser, val_set)
        cache(n_apparent, n_true, inl_stats, R_errs, t_errs)
        return n_apparent, n_true, inl_stats, R_errs, t_errs
