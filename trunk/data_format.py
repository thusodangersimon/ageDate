#!/usr/bin/env python
#
# Name:  data format 
#
# Author: Thuso S Simon
#
# Date: 20/1/12
# TODO: add citations
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
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    For the GNU General Public License, see <http://www.gnu.org/licenses/>.
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#History (version,date, change author)
#
#
#
'''
Creates models that will work with EZGAL. Makes CSP (composite spectra
) using multiproccessing or MPI. Adds meta data for models that don't have it
'''

import numpy as nu
import pyfits as fits
import os
import ezgal as gal
from scipy.stats import signaltonoise as stn
from glob import glob

def add_meta():
    '''adds meta data to exsisting ezgal libraires'''
    pass

def make_mpi_burst(lengths, spec_lib='p2',imf='salp',spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
    '''(list, int, str,str,str) -> EZGAL.wrapper
    makes burst using mpi
    '''
    #load models and make wrapper
    cur_lib = ['basti', 'bc03', 'cb07','m05','c09','p2']
    assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + str(cur_lib))
    if not spec_lib_path.endswith('/') :
        spec_lib_path += '/'
    models = glob(spec_lib_path+spec_lib+'*'+imf+'*')
    if len(models) == 0:
        models = glob(spec_lib_path+spec_lib.lower()+'*'+imf+'*')
    assert len(models) > 0, "Did not find any models"
        #crate ezgal class of models
    SSP = gal.wrapper(models)
   
    #genrate models from list
    out = []
    for i in lengths:
        for j in SSP.models:
            out.append(j.make_burst(round(i,19)))
    #return models
    return out

if __name__ == '__main__':

    '''do MPI stuff here'''
    import mpi4py.MPI as mpi
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    min_sfh, max_sfh = 1, 16
    lin_space = True
    outdir = '/home/thuso/Phd/stellar_models/ezgal/lin_burst/'
    age = nu.array([  1.00000000e+06,   2.00000000e+06,   3.00000000e+06,   4.00000000e+06,
   5.00000000e+06,   6.00000000e+06,   7.00000000e+06,   8.00000000e+06,
   9.00000000e+06,   1.00000000e+07,   1.20000000e+07 ,  1.40000000e+07,
   1.60000000e+07,   1.80000000e+07,   2.00000000e+07 ,  2.50000000e+07,
   3.00000000e+07,   3.50000000e+07,   4.00000000e+07   ,4.50000000e+07,
   5.00000000e+07 ,  6.00000000e+07   ,7.00000000e+07 ,  8.00000000e+07,
   9.00000000e+07  , 1.00000000e+08,   1.20000000e+08 ,  1.40000000e+08,           1.60000000e+08 ,  1.80000000e+08 ,  2.00000000e+08 ,  2.50000000e+08,
   3.00000000e+08,   3.50000000e+08,   4.00000000e+08,   4.50000000e+08,
   5.00000000e+08,   6.00000000e+08,   7.00000000e+08,   8.00000000e+08,
   9.00000000e+08  , 1.00000000e+09 ,  1.20000000e+09 ,  1.40000000e+09,
   1.60000000e+09 ,  1.80000000e+09  , 2.00000000e+09  , 2.50000000e+09,
   3.00000000e+09 ,  3.50000000e+09   ,4.00000000e+09,   4.49999974e+09,
   5.00000000e+09,   6.00000000e+09,   7.00000000e+09 ,  8.00000000e+09,
   8.99999949e+09 ,  1.00000000e+10  , 1.10000005e+10 ,  1.20000000e+10,
   1.29999995e+10 ,  1.40000000e+10   ,1.50000005e+10 ,  1.60000000e+10,
    1.69999995e+10 ,  1.79999990e+10  , 1.90000005e+10,   2.00000000e+10])
    ####stuff for make_burst
    if rank == 0:
        needed = []
        for i in range(min_sfh, max_sfh+1):
            if not lin_space:
                #log spacelengths = nu.unique(needed[:,1])
                space_age = nu.logspace(nu.log10(age).min(),nu.log10(age).max()
                                        ,i+1)
            else:
                #linear spacing
                space_age = nu.linspace(age.min(), age.max(), i+1)
            for j in range(i):
                #[mean_age_log, length (gyrs)]
                needed.append([space_age[j:j+2].mean(),space_age[j:j+2].ptp()/10**9])
        needed = nu.asarray(needed)
        #sort and combine lengths lelengths = nu.unique(needed[:,1])ngths = nu.unique(needed[:,1])
        lengths = nu.unique(needed[:,1])
        #split up for scatter
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(lengths):
            chunks[i % size].append(chunk)
    else:
        lengths = None
        chunks = None
    lengths = comm.scatter(chunks, root=0)
    
    #start progam up
    to_save = make_mpi_burst(lengths)
    #save
    for i in to_save:
        if i.meta_data['imf'] == 'Salpeter':
            imf = 'salp'
        name = '{0}_{1}_z_{2}_{3}_{4}.fits'.format(i.meta_data['model'],i.meta_data['sfh'],i.meta_data['met'],imf,i.meta_data['length'])
    
        i.save_model(outdir+name)
