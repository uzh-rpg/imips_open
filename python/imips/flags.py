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

from absl import flags
import pprint
import sys

# Training control.
flags.DEFINE_integer('its', 100000, 'Total iterations to do.')
flags.DEFINE_integer('val_every', 250, 'Validate every n iterations.')
flags.DEFINE_bool('dsup', False, 'Discriminative suppression?')

# Net config (general)
flags.DEFINE_integer('lr', 6, 'Negative log learning rate.')
flags.DEFINE_integer('depth', 14, 'NN depth.')
flags.DEFINE_integer('chan', 128, 'Output channel count.')
flags.DEFINE_integer('rr', 0, 'Random roll id (does not affect execution).')
flags.DEFINE_bool('use_qn', False, 'Use QuadNets loss?')
flags.DEFINE_float('corr_w', 1., 'Correspondence loss weight.')
# Train config.
flags.DEFINE_string('tds', 'tm', 'Training dataset (hp, kt, rc, tm, en)')
flags.DEFINE_string('pck', 'tr', 'Pair pick method (gt, tr, tt)')
flags.DEFINE_float('ol', 0.3, 'KLT overlap between picked frames')
flags.DEFINE_float('aug_ang', 0.,
                   'Angle augmentation range (n*2pi) for HPatches')
flags.DEFINE_float('aug_sc', 0., 'Scale augmentation range (2^n) for HPatches')

# Evaluation parameters
flags.DEFINE_string('ds', 'kt', 'Validation dataset (kt, hp)')
flags.DEFINE_string('eds', '', 'Evaluation dataset (eu), empty to use ds')
flags.DEFINE_integer('k', 10, 'Amount of inliers to consider success')
flags.DEFINE_string('baseline', '', 'Baseline to be used instead of NN')
flags.DEFINE_integer('baseline_num_ips', 128, 
                     'Amount of interest points extracted by baseline')
flags.DEFINE_bool('testing', False, 'Evaluate using testing data?')
flags.DEFINE_bool('val_best', False, 'use best net for validation?')

# Multiscale
flags.DEFINE_integer('num_scales', 1, 'How many scales to extract.')
flags.DEFINE_float('scale_factor', 0.71, 'Factor between scales.')

# Product_quantization
flags.DEFINE_integer('pq_m', 4, 'Amount of dimension divisions.')
flags.DEFINE_integer('pq_k', 16, 'Amount of cluster centers.')
flags.DEFINE_integer('pq_n_init', 1, 'kmean trial count.')
# PCA
flags.DEFINE_integer('pca_dim', 5, 'PCA dimensions.')

flags.DEFINE_bool('render_train_samples', False, '')

# Workaround for SIPS2 compatibility
flags.DEFINE_string('ransac', 'vanilla', 'RANSAC: vanilla, untilInl or prisac')

FLAGS = flags.FLAGS

sys.argv = FLAGS(sys.argv)


def printVals():
    pp = pprint.PrettyPrinter()
    print_flags_dict = {}
    for key in FLAGS.__flags.keys():
        print_flags_dict[key] = getattr(FLAGS, key)
    pp.pprint(print_flags_dict)