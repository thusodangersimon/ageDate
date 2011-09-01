#!/usr/bin/env python
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
from time import time
def griddata(x, y, z, binsize=0.01,reduice='median'):

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
  21     retbin : boolean, optional
  22         Function returns `bins` variable (see below for description)
  23         if set to True.  Defaults to True.
  24     retloc : boolean, optional
  25         Function returns `wherebins` variable (see below for description)
  26         if set to True.  Defaults to True.
  27    
  28     Returns
  29     -------
  30     grid : ndarray (2D)
  31         The evenly gridded data.  The value of each cell is the median
  32         value of the contents of the bin.
  33     bins : ndarray (2D)
  34         A grid the same shape as `grid`, except the value of each cell
  35         is the number of points in that bin.  Returns only if
  36         `retbin` is set to True.
  37     wherebin : list (2D)
  38         A 2D list the same shape as `grid` and `bins` where each cell
  39         contains the indicies of `z` which contain the values stored
  40         in the particular bin.
  41 
  42     Revisions
  43     ---------
  44     2010-07-11  ccampo  Initial version
  """


# get extrema values.
    t=time()
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    xi= np.arange(xmin, xmax+binsize, binsize)
    yi= np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xi,yi)

    grid = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
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
        print 'wrong input, goint with median'
        
    print time()-t
    for row in xrange(nrow):
        for col in xrange(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.
# find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)

            bin = z[np.logical_and(posx < binsize/2., posy < binsize/2.)]
            if bin.size != 0:
                grid[row, col] = func(bin)
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.
    return grid

if __name__=='__main__':
    import cProfile as pro
    import cPickle as pik
    param,chi=pik.load(open('/home/thuso/Desktop/data.pik'))
    pro.runctx('griddata(param1,param3,chi,val,reduice)'
               , globals(),{'param1':param[:,1],'param3':param[:,3],'chi':chi,'val':.1,'reduice':'min'}
               ,filename='agedata.Profile')
