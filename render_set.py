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

import cv2
import os

import imips.flags as flags
import imips.hyperparams as hyperparams


if __name__ == '__main__':
    seq = hyperparams.getEvalSequence()
    fpasser = hyperparams.getForwardPasser()

    save_path = os.path.join('results', 'render', hyperparams.methodString(),
                             seq.set_name, seq.seq_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    for image in seq.images:
        fp = fpasser(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
        bgr = fp.render()
        imname = os.path.basename(image)
        print('%d / %d' % (i, len(seq)))
        cv2.imwrite(os.path.join(save_path, '%05d.png' % i), bgr)
        i = i + 1
