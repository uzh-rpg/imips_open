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
import numpy as np

FLAGS = absl.flags.FLAGS


def imageToBatchInput(image, ips_rc, num_layers):
    nc = ips_rc.shape[1]
    
    diam = num_layers * 2 + 1
    batch_in = np.zeros([nc, diam, diam, 1])
    
    for c in range(nc):
        if ips_rc[0, c]-num_layers < 0 or \
                ips_rc[0, c]+num_layers+1 > image.shape[0]:
            print(ips_rc[:, c])
            print(image.shape)
            assert False
        if ips_rc[1, c]-num_layers < 0 or \
                ips_rc[1, c]+num_layers+1 > image.shape[1]:
            print(ips_rc[:, c])
            print(image.shape)
            assert False
        batch_in[c, :, :, 0] = image[
                ips_rc[0, c]-num_layers:ips_rc[0, c]+num_layers+1, 
                ips_rc[1, c]-num_layers:ips_rc[1, c]+num_layers+1]
        
    return batch_in
