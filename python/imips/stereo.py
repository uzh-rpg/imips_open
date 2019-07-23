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

import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def getSSDs(left_im, right_im, row, col, r, max_disp, min_disp, patch_size,
            debug):
    # Patch and strip matched against each other
    left_patch = left_im[row-r:row+r+1, col-r:col+r+1]
    right_strip = right_im[
            row-r:row+r+1, col-r-max_disp:col+r+1-min_disp].T
    
    # Patches into vectors for cdist
    lpvec = np.array(np.ravel(left_patch.T), ndmin=2)
    rsvecs = np.zeros((max_disp - min_disp + 1, patch_size**2))
    for i in range(patch_size):
        rsvecs[:, i*patch_size:(i+1)*patch_size] = \
                right_strip[i:(max_disp - min_disp + i + 1), :]
    
    ssds = np.ravel(cdist(lpvec, rsvecs, 'sqeuclidean'))
    
    if debug:
        plt.imshow(left_im, cmap='gray')
        print(col)
        print(row)
        plt.plot([col], [row], c='red')
        plt.show()                    
        plt.imshow(left_patch)
        plt.show()
        plt.imshow(right_strip)
        plt.show()
        plt.plot(ssds)
        plt.show()

    return ssds
    

# If points are provided, returns a 1D array with disparity for each point (in
# the left image), otherwise a 2D array with disparity for the full image.
# Points given as 2xN numpy array, of (row, col) coordinates.
def getDisparity(left_im, right_im, points=None, patch_radius=5, min_disp=5, 
                 max_disp=50):
    # Adapted from the cvcourse matlab solution.
    
    debug = False
    
    r = patch_radius
    patch_size = 2 * patch_radius + 1
    
    disp_im = np.zeros_like(left_im, dtype=np.double)
    
    assert left_im.shape == right_im.shape
    
    rows = left_im.shape[0]
    cols = left_im.shape[1]
    
    if points is None:
        R, C = np.meshgrid(
                range(patch_radius, rows - patch_radius),
                range(max_disp + patch_radius, cols - patch_radius))
        qpts = np.vstack((R.ravel(), C.ravel()))
    else:
        if not np.all(points[0, :] < rows):
            print(points)
            raise Exception('Input points out of bounds!')
        assert np.all(points[1, :] < cols)
        # Get rid of points too close to the border
        valid = points[0, :] >= patch_radius
        valid &= points[0, :] < rows - patch_radius
        valid &= points[1, :] >= max_disp + patch_radius
        valid &= points[1, :] < cols - patch_radius
        
        qpts = points[:, valid]
    
    for i in range(qpts.shape[1]):
        row = qpts[0, i]
        col = qpts[1, i]
        
        ssds = getSSDs(left_im, right_im, row, col, r, max_disp, min_disp,
                       patch_size, debug)
        
        neg_disp = np.argmin(ssds)
        min_ssd = ssds[neg_disp]
        
        if debug:
            print(neg_disp)
        
        num_low = np.count_nonzero(ssds <= 1.5 * min_ssd)
        if debug:
            print(num_low)
        not_border = neg_disp != 0 and neg_disp != (ssds.size-1)
        if num_low < 3 and not_border:
            x = [neg_disp - 1, neg_disp, neg_disp + 1]
            # p = np.polyfit(x, ssds[x], 2)
            # disp_im[row, col] = max_disp + p[1]/(2 * p[0])
            y = ssds[x]
            a = (y[0] + y[2]) / 2 - y[1]
            b = (y[2] - y[0]) / 2
            disp_im[row, col] = max_disp - neg_disp + b / (2 * a)
            if debug:
                print(disp_im[row, col])
        
        if debug:
            raw_input()
    
    if points is None:
        return disp_im
    else:
        return disp_im[points[0, :], points[1, :]]


# If points are provided, 3d points only of the provided points (2xN)
def disparityToPointCloud(disp_im, K, baseline, left_im, points=None):
    # Adapted from the cvcourse matlab solution.
    if points is None:
        X, Y = np.meshgrid(range(disp_im.shape[1]), range(disp_im.shape[0]))
        X = np.ravel(X).astype(float)
        Y = np.ravel(Y).astype(float)
    else:
        X = points[1, :].astype(float)
        Y = points[0, :].astype(float)
    
    # row, col, 1
    px_left = np.array([Y, X, np.ones_like(X)])
    px_right = copy.copy(px_left)
    px_right[1, :] = px_right[1, :] - np.ravel(disp_im)
    
    # Keep pixels with non-zero disparity
    px_left = px_left[:, np.ravel(disp_im) > 0]
    px_right = px_right[:, np.ravel(disp_im) > 0]
    
    # u, v, 1
    px_left[:2, :] = np.flipud(px_left[:2, :])
    px_right[:2, :] = np.flipud(px_right[:2, :])
    
    # Reproject
    bv_left = np.matrix(K)**-1 * np.matrix(px_left)
    bv_right = np.matrix(K)**-1 * np.matrix(px_right)
    
    points = np.array(np.zeros_like(bv_left))
    b = np.matrix([[baseline], [0], [0]])
    for i in range(px_left.shape[1]):
        A = np.hstack((bv_left[:, i], -bv_right[:, i]))
        x = (A.T * A) ** -1 * (A.T * b)
        points[:, i:i+1] = bv_left[:, i] * x[0]
    
    return points


def plot3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[0, :], points[1, :], points[2, :], c='r', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
