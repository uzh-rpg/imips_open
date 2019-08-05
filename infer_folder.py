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
import cv2
import numpy as np
import os

absl.flags.DEFINE_string('in_dir', '', 'Input directory')
absl.flags.DEFINE_string('out_dir', '', 'Output directory ($HOME/imips_out/in_dir if empty)')
absl.flags.DEFINE_string('ext', '', 'Image extension, if not .jpg or .png (with dot)')
import imips.flags as flags
import imips.hyperparams as hyperparams

FLAGS = flags.FLAGS


class FolderInference(object):
    def __init__(self):
        assert FLAGS.in_dir != ''
        assert os.path.exists(FLAGS.in_dir)
        self._in = os.path.abspath(FLAGS.in_dir)
        if FLAGS.out_dir == '':
            self._out = os.path.join(os.environ['HOME'], 'imips_out',
                                     self._in[1:])
        else:
            self._out = os.path.abspath(FLAGS.out_dir)
        if not os.path.exists(self._out):
            os.makedirs(self._out)

        print('Reading from %s' % self._in)
        print('Writing to %s' % self._out)

        exts = ['.jpg', '.png']
        if FLAGS.ext != '':
            exts.append(FLAGS.ext)
        images = sorted(os.listdir(self._in))
        self._images = [i for i in images if os.path.splitext(i)[1] in exts]

    def __getitem__(self, item):
        return os.path.join(self._in, self._images[item]), \
               os.path.join(self._out, os.path.splitext(self._images[item])[0])

    def __len__(self):
        return len(self._images)


if __name__ == '__main__':
    FLAGS.val_best = True
    print(hyperparams.methodEvalString())
    graph, sess = hyperparams.bootstrapFromCheckpoint()
    forward_passer = hyperparams.getForwardPasser(graph, sess)
    folder_inference = FolderInference()

    for i, in_path_out_root in enumerate(folder_inference):
        print('%d/%d' % (i, len(folder_inference)))
        in_path, out_root = in_path_out_root
        fp = forward_passer(cv2.imread(in_path, cv2.IMREAD_GRAYSCALE))
        np.savetxt(out_root + '.txt', np.vstack((fp.ips_rc, fp.ip_scores)).T,
                   header='row column score', fmt='%5.0f %5.0f %.3f')
        cv2.imwrite(out_root + '_render.png', fp.render())
