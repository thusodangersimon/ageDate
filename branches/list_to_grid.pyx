#!/usr/bin/python2.7
#
# Name:  girds
#
# Author: Thuso S Simon
#
# Date: 11 Aug 2011
# TODO: 
#
#    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#    Copyright (C) 2011  Thuso S Simon
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    For the GNU General Public License, see <http://www.gnu.org/licenses/>.
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#History (version,date, change author)
#  Taken from matplotlib example and sped up with cython
#
#

import numpy as np
cimport numpy as np
cimport cython
#from cpython cimport bool
from libcpp cimport bool
@cython.boundscheck(False)


def griddata(np.ndarray[double, ndim=1] x,np.ndarray[double, ndim=1]  y
             ,np.ndarray[double, ndim=1]  z,float binsize=0.1, reduice='median'):
    #ctypedef np.float_t DTYPE_t
    """
     6     Place unevenly spaced 2D data on a grid by 2D binning (nearest
   7     neighbor interpolation).
   8     
   9     Parameters
  10     ----------
  11     x : ndarray (1D)
  12         The idependent data x-axis of the grid.
  13     y : ndarray (1D)
  14         The idependent data y-axis of the grid.
  15     z : ndarray (1D)
  16         The dependent data in the form z = f(x,y).
  17     binsize : scalar, optional
  18         The full width and height of each bin on the grid.  If each
  19         bin is a cube, then this is the x and y dimension.  This is
  20         the step in both directions, x and y. Defaults to 0.01.
  27    
  28     Returns
  29     -------
  30     grid : ndarray (2D)
  31         The evenly gridded data.  The value of each cell is the median
  42     Revisions
  43     ---------
  44     2010-07-11  ccampo  Initial version
  """

    cdef double xmin, xmax,ymin, ymax,xc,yc
    cdef unsigned int nrow, ncol, row,col,N,i
    cdef np.ndarray[double, ndim=2] xi,yi,grid
    cdef np.ndarray[double, ndim=1] posx,posy,xii,yii
    cdef list bin
# get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    xii= np.arange(xmin, xmax+binsize, binsize)
    yii= np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xii,yii)

    grid = np.zeros([xi.shape[0],xi.shape[1]])
    nrow, ncol = grid.shape[0],grid.shape[1]
    #decide reduice type
    if reduice=='median':
        func=np.median
    elif reduice=='mean':
        func=np.mean
    elif reduice=='max':
        func=max
    elif reduice=='min':
        func=min
    else:
        print 'wrong input, going with median'
        
    N=len(y)
    for row in xrange(nrow):
        yc = yi[row, col]  # y coordinate.
        posy = abs(y - yc)
        for col in xrange(ncol):
            bin=[]
            xc = xi[row, col]    # x coordinate.
            posx = np.abs(x - xc)
            #for i in xrange(N):
            #    if posx[i]<binsize/2. and posy[i] < binsize/2.:
            #        break
            bin = z[np.logical_and(posx < binsize/2., posy < binsize/2.)]
            if bin.size != 0:
                print row, col
                
                grid[row, col] = func(bin)
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.
    return grid
