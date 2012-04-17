#!/usr/bin/env python
#
# Name:  Interpolation libraies
#
# Author: Thuso S Simon
#
# Date: 28th of June, 2011
#TODO: make sure it works, change to Cython
#
#    vvvvvvvvvvMake more moduals, add a check to see if cpu's left are greater t
# than itterations needed
# add age bins for complex galaxy modelsvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
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
#
''' Linear and bilinear interpolation library can hadel interp of
    arrays
'''


import numpy as nu
cimport numpy as nu
cimport cython

@cython.boundscheck(False)
def bilinear_interpolation(x,y,z_temp,x_eval,y_eval):
    #takes in x,y as a len(x)=4, z is a len(z)>4 array and x_eval and y_eval
    #are floats
   #cdef float x,y
   x,y=nu.unique(x),nu.unique(y)
   return (z_temp[0]*(x[1]-x_eval)*(y[1]-y_eval)+z_temp[1]*(x_eval-x[0])*(y[1]-y_eval)+z_temp[2]*(x[1]-x_eval)*(y_eval-y[0])+z_temp[3]*(x_eval-x[0])*(y_eval-y[0]))/((x[1]-x[0])*(y[1]-y[0]))

@cython.boundscheck(False)
def linear_interpolation(x,y_temp,x_eval):
    #y=nu.zeros([len(y_temp[0]),2])  
   return y_temp[0]+(x_eval-x[0])*(y_temp[1]-y_temp[0])/(x[1]-x[0])

def spectra_lin_interp(x,y,x_eval):
    #interpolates spectra with x to x_eval
    return nu.interp(x_eval, x, y)
