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

import numpy as np


def getFromNetOut(net_out, lin_amax):
    """ Returns indices in (row,col), colwise, and corresponding scores. """
    # TODO(tcies) vectorize?
    nc = net_out.shape[-1]
    assert lin_amax.shape[0] == nc
    ips_rc = np.zeros((2, nc), dtype=int)
    scores = np.zeros(nc, dtype=float)
    for c in range(nc):
        ips_rc[:, c] = np.unravel_index(lin_amax[c], net_out.shape[:2])
        scores[c] = net_out[ips_rc[0, c], ips_rc[1, c], c]
        
    return ips_rc, scores
