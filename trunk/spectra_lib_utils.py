#!/usr/bin/env python
#
# Name:  Age Dating Spectra Fitting Program
#
# Author: Thuso S Simon
#
# Date: 28th of June, 2011
#TODO: clean up and add dust to them
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
""" Basic IO functions for Age_date.py. Also creates synthetic galaxy spectra using sfh files
"""

import cPickle as pik
import numpy as nu
import os
import sys
from multiprocessing import Pool
import boundary as bound
import ezgal as gal
from glob import glob

def read_spec(name, lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #reads in spec and turns into numpy array [lambda,flux]
    return nu.loadtxt(lib_path+name)

def get_fitting_info(lib_path='/home/thuso/Phd/Code/Spectra_lib/',
                     spec_lib='BC03.splib'):
    #gets list of ages,metalicity and asocated file names
    lib = os.listdir(lib_path)
    find_fun = lambda i: i if i.endswith('spec') else None
    standard_file = map(find_fun, lib)
    try:
        while True:
            standard_file.remove(None)
    except ValueError:
        pass
    if len(standard_file) == 0:
        #import Age_date as ag
        #lib = ag.info
        spect,lib = load_spec_lib(lib_path,spec_lib)
        out = nu.zeros([len(lib), 2])
    else:
        out = nu.zeros([len(lib), 2])
    for j, i in enumerate(lib):
        out[j, :] = [float(i[4:10]), float(i[11:-5])]
    #not sure why this is ok 
    #out[standard_file,:],nu.array(lib)[standard_file]   

    return out,lib

def load_spec_lib(lib_path='/home/thuso/Phd/Spectra_lib/', lib='BC03.splib'):
    '''loads all spectra into libary first load of lib may be slow.
    .splib files take precendence over .spec files'''
    #check if exsiting .splib file 
    files = os.listdir(lib_path)
    splib_files = []
    splib_checker = lambda i: True if i.endswith('splib') else False
    if sum(map(splib_checker,files)) > 0: #quick check to se
        for i in files:
             if i.endswith('splib'):
                 splib_files.append(i)
        if lib in splib_files:
            return pik.load(open(lib_path + lib))
        else:
            raise IOError('SSP Libraiy does not exsist.')
    else:
        print 'First Load, May take some time'
        outname = raw_input('Please Give name of input Lib ')
        
        lib = get_fitting_info(lib_path)[1]
        temp = read_spec(lib[0], lib_path)
        #create array
        out = nu.zeros([temp.shape[0], len(lib) + 1])#out[:,0]=wavelength range
        out[:,0] = temp[:,0]
        tempout =map(read_spec,lib,[lib_path] * len(lib))
        for i, j in enumerate(tempout):
            out[:, i + 1] = j[:, 1]
        pik.dump((out, lib), open(lib_path + outname + '.splib', 'w'), 2)
        return out, lib

    
def edit_spec_range(spect, lam_min, lam_max):
    index = nu.nonzero(nu.logical_and(
            spect[:,0] >= lam_min, spect[:, 0] <= lam_max))[0]
    return spect[index,:]


def load_ezgal_lib(spec_lib, spec_lib_path, imf):
    '''Loads all models in a path with a certan imf and model as an
    ezgal wrapper object
    '''
    cur_lib = ['basti', 'bc03', 'cb07','m05','c09','p2']
    assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + str(cur_lib))
    search_query = os.path.join(spec_lib_path,spec_lib) + '*%s*'%imf

    models = glob(search_query)
    assert len(models) > 0, "Did not find any models, check path"
    #crate ezgal class of models
    SSP = gal.wrapper(models)
    
    return SSP

def info_for_lib(lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #calculates information content of spectral library
    spec_lib, info = load_spec_lib(lib_path)
    #create prob dist for each wavelength
    prob_dist = {}
    for i in xrange(len(spec_lib[:, 0])):
        temp = nu.zeros([nu.unique(spec_lib[i, 1:]).shape[0] - 1, 2])
        temp[:, 0] = nu.unique(spec_lib[i, 1:])[:-1]
        temp[:,1] = nu.histogram(spec_lib[i, 1:], nu.unique(spec_lib[i, 1:]))[0]
        temp[:,1] = temp[:,1] / nu.sum(temp[:, 1])
        prob_dist[str(spec_lib[i, 0])] = nu.copy(temp)

    return prob_dist

def info_for_spec(data, prob_dist=None,
                  lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #calculates information content of single spectra with respect to
    #current spectral library #only works is spectra have same wavelenths
    if not prob_dist:
        prob_dist = info_for_lib(lib_path)
    H = 0
    for i in range(data.shape[0]):
        if not any(nu.array(prob_dist.keys()) == str(data[i, 0])):
            print 'wavelenthd to not match'
            raise

        temp = prob_dist[str(data[i, 0])]
        H = (H - temp[nu.argsort((temp[:, 0] - 
                                  data[i, 1])**2), 1][0] * 
             nu.log10(temp[nu.argsort(temp[:, 0] - data[i, 1]), 1][0]))

    return H


if __name__=='__main__':

    import cPickle as pik
    po=Pool()
    prob=info_for_lib()
    spec_lib,info=load_spec_lib()
    results=[po.apply_async(info_for_spec,(nu.vstack((spec_lib[:,0],spec_lib[:,i])).T,prob,)) for i in range(1,1195)]
    H=[]
    for i,j in enumerate(results):
        print '%i out of 1195' %i
        H.append([i,j.get()])
    po.close()
    po.join()
    pik.dump(nu.array(H),open('H.pik','w'),2)
