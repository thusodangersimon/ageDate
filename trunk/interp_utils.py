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
''' Interpolation library uses numpy and numexpr to speed up interpolation of arrays
    
'''


import numpy as nu
#import numexpr as ne
from scipy.interpolate import griddata
import numpy as np
import scipy.spatial.qhull as qhull
import warnings

def bilinear_interpolation(x,y,z_temp,x_eval,y_eval):
    '''takes in x,y as a len(x)=4, z is a len(z)>4 array and x_eval and y_eval
    are floats'''
    x,y=nu.unique(x),nu.unique(y)
    return ((z_temp[0] * (x[1] - x_eval) * (y[1]-y_eval) +
            z_temp[1] * (x_eval - x[0]) * (y[1]-y_eval) + 
            z_temp[2] * (x[1] - x_eval) * (y_eval - y[0]) + 
            z_temp[3] * (x_eval - x[0]) * (y_eval - y[0])) / 
           ((x[1]-x[0])*(y[1]-y[0])))
    #z_temp0, x1, y1, z_temp1, x0 = z_temp[0], x[1], y[1], z_temp[1], x[0]
    #z_temp2, y0, z_temp3 = z_temp[2], y[0], z_temp[3]
    #return ne.evaluate('''((z_temp0 * (x1 - x_eval) * (y1-y_eval) +
    #        z_temp1 * (x_eval - x0) * (y1-y_eval) + 
    #        z_temp2 * (x1 - x_eval) * (y_eval - y0) + 
    #        z_temp3 * (x_eval - x0) * (y_eval - y0)) / 
    #       ((x1-x0)*(y1-y0)))''')
  
def linear_interpolation(x,y_temp,x_eval):
    #y_temp0, x0, y_temp1, x1 = y_temp[0],x[0],y_temp[1],x[1]
    #return ne.evaluate('y_temp0+(x_eval - x0) * (y_temp1 - y_temp0)/(x1 - x0)')
    return y_temp[0] + (x_eval - x[0]) * (y_temp[1] - y_temp[0])/(x[1] - x[0])

def spectra_lin_interp(x,y,x_eval):
    #interpolates spectra with x to x_eval
    return nu.interp(x_eval, x, y)

def n_dim_interp(points, eval_points,spec):
    '''Does n-dimensional interpolation. Needs at least 2*n points to run'''
    assert nu.any(nu.asarray(points.shape) == len(eval_points)*2),'Needs more points to run'
    #check indexing
    if len(points.shape) == 3:
        points = points[0]
    if len(eval_points.shape) == 1:
        eval_points = nu.asarray([eval_points])
    try:
        out = griddata(points,spec,eval_points)[0]
        return out
    except: #qhullerror
        return spec[0] * nu.nan

if __name__ == '__main__':
    import likelihood_class as lik
    import numpy as nu
    from database_utils import NN,get_param_from_hdf5

    fun = lik.CV_fit(None)
    param = fun.initalize(1)[0]
    
    Nei_clas = NN(len(param)*2).fit(fun.all_param)
    index = Nei_clas.kneighbors(param)[1]
    #get spec
    spec = []
    for i in index[0]:
        spec.append(fun.tab.cols.spec[i][:,1])
    #try interpolation
    points, eval_points = all_param[index],param
    
    a = LinearNDInterpolator(points[0],spec)
     
