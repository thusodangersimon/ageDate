#!/usr/bin/env python
#
# Name:  
#
# Author: Thuso S Simon
#
# Date: Sep. 1, 2011
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
#
#
#
"""
programs to visualize the Age date program
"""
import Age_date as ag ##temp import##
import numpy as nu
from multiprocessing import Pool,pool

def make_chi_grid(data,dust=None,points=500):
    'makes a 3d pic of metal,age,chi with input spectra 2-D only'
    fun=ag.MC_func(data)
    ag.spect = fun.spect
    #create grid
    metal,age=nu.meshgrid(nu.linspace(fun._metal_unq.min(),
                                      fun._metal_unq.max(),points),
                          nu.linspace(fun._age_unq.min(),
                                      fun._age_unq.max(),points))

    param = nu.array(zip(metal.ravel(),age.ravel(),nu.ones_like(age.ravel())))
    #probably need to handel dust in a better way
    dust_param = nu.zeros([len(param),2])
    if nu.any(dust):
        dust_param[:,0] = dust[0]
        dust_param[:,1] = dust[1]
        
    
    #start making calculations for chi squared value
    po,out=Pool(),[]
    for i in xrange(len(param)):
        out.append(po.apply_async(func,args = (fun.data,param[i],dust_param[i],
                            fun._lib_vals ,fun._age_unq,fun._metal_unq)))
    b=nu.array(map(get, out))
    po.close()
    po.join()
    
    return metal,age,b[:,-1].reshape(points,points)

def get(f):
    return f.get()

def func(data,param,param_dust,lib_vals,age_unq,metal_unq):

    bins = param.shape[0] / 3
    model = ag.get_model_fit_opt(param, lib_vals, age_unq,metal_unq, bins)
    model = ag.dust(nu.hstack((param, param_dust)), model)
    N,chi = ag.N_normalize(data, model, bins)
    return nu.hstack((param, param_dust,chi))
