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
from multiprocessing import Pool
from time import time


def griddata(x, y, z,xi,yi,reduice='median'):

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
  surface at the points specified by (*xi*, *yi*) to produce
    *zi*. *xi* and *yi* must describe a regular grid, can be either 1D
    or 2D, but must be monotonically increasing.


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
    binsizex,binsizey=abs(np.diff(xi)[0]),abs(np.diff(yi)[0])
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
    col =0
   
    for row in xrange(nrow):
        print 'on row %i of %i' %(row,nrow)
        yc = yi[row, col] 
        posy = abs(y - yc)
        grid=grid+multi_grid(row,ncol,posy,x,xi,z,func,np.zeros(xi.shape, dtype=x.dtype)
                             ,binsizex,binsizey)

    #interpolate values
    

    return np.ma.masked_invalid(grid)

def multi_grid(row,ncol,posy,x,xi,z,func,grid,binsizex,binsizey):
    for col in xrange(ncol):
        xc = xi[row, col]    # x coordinate.
        posx = abs(x - xc)
        bin = z[np.logical_and(posx < binsizex/2., posy < binsizey/2.)]
        if bin.size != 0:
            grid[row, col] = func(bin)
        else:
            grid[row, col] = np.nan   # fill empty bins with nans.
    return grid
               
def cb(r):
    pass

def find_box(x_eval,y_eval,x,y,z):
    #finds 4 closest points for interpolation
    #checks to see if on edge

def bilinear_interpolation(x,y,z,x_eval,y_eval):
    #takes a 2 d array and interpolates between known points
    
    F=np.zeros(x_eval.shape, dtype=x.dtype)
    for i in range(len(x)):
        for j in range(len(y)):
            F[i,j]=(z*(x-x_eval[i])*(y-y_eval[j])+z*(x_eval[i]-x)*(y-y_eval[j])
               +z*(x-x_eval[i])*(y_eval[j]-y)+z*(x_eval[i]-x)*(y_eval[j]-y)
               )/((x-x)*(y-y))
    return F


#def det(M):
#    return linalg.det(M)

#po = Pool()
#for i in xrange(1,300):
    #j = random.normal(1,1,(100,100))
    #po.apply_async(det,(j,),callback=cb)
if __name__=='__main__':
    npts=100
    x = np.random.uniform(-2,2,npts)
    y =np.random.uniform(-2,2,npts)
    z = x*np.exp(-x**2-y**2)
  
    xi = np.linspace(-2.1,2.1,100)
    yi = np.linspace(-2.1,2.1,100)
  
