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

def read_spec(name, lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #reads in spec and turns into numpy array [lambda,flux]
    return nu.loadtxt(lib_path+name)

def get_fitting_info(lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
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
        import Age_date as ag
        lib = ag.info
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

def iterp_spec(bins, func='flat', has_dust=True, age_bins=False, lam_min=0,
                   lam_max=nu.inf, slope=None,
               lib_path='/home/thuso/Phd/Spectra_lib/'):
    '''does everything from create_spectra but values not only from libary
    #also adds dust absorption returns specrtra,info in ssp lib fmt, 
    #weight, dust(tau_ism,tau_BC) age_bins=[False, "linear","log"]'''
    from Age_date import get_model_fit_opt, dust
    lib_vals = get_fitting_info(lib_path)
    lib_vals[0][:,0] = 10**nu.log10(lib_vals[0][:, 0])
    metal_unq = nu.log10(nu.unique(lib_vals[0][:, 0]))
    age_unq = nu.unique(lib_vals[0][:,1])
    dust_bound = nu.array([0,4])
    if age_bins:
        #does different types of age binning
        if age_bins.lower() == 'linear':
            age_bins = nu.log10(nu.linspace(10**age_unq.min(),
                                            10**age_unq.max(), 1+bins))
        elif age_bins.lower() == 'log':
            age_bins = nu.linspace(age_unq.min(),age_unq.max(), 1+bins)
        else:
            raise ValueError('"%s" is not a valid spacing opion' %age_bins)
    else:
        age_bins = nu.array([age_unq.min(), age_unq.max()])
    #generate random parameters
    while True:
        if len(age_bins) > 2:
            age = nu.zeros(bins)
            for i in xrange(bins):
                age[i] = (nu.random.rand() * (age_bins[i+1] - age_bins[i]) + 
                          age_bins[i])
        else:
            age = nu.random.rand(bins) * age_unq.ptp() +age_unq.min()
        metal = nu.random.rand(bins) * metal_unq.ptp() + metal_unq.min()
        age.sort()
        norm = SFR_func(func, bins, age)
        #turn params into standard format
        param = nu.zeros(len(age) * 3)
        index = 0
        for i in xrange(0, len(age) * 3, 3):
            param[i:i + 3] = [metal[index], age[index], norm[index]]
            index += 1
        if has_dust:
            #add dust
            param = nu.hstack((param, nu.random.rand(2) * dust_bound.ptp() +
                       dust_bound.min()))
        else:
            param = nu.hstack((param, nu.zeros(2)))
        #make sure params in range
        if check(param,age_bins,dust_bound,lib_vals,age_unq,metal_unq):
            break
        
    #get spectra
    lib_vals[0][:,0] = 10**lib_vals[0][:,0]
    model = get_model_fit_opt(param[:-2], lib_vals, age_unq, metal_unq, bins)
    model = dust(param, model)
    #get specified wavelenght range
    index = nu.nonzero(nu.logical_and(model['wave'] >= 
                                      lam_min, model['wave'] <= lam_max))[0]
    #combine and apply wavelength range
    out = nu.zeros([len(index), 2])
    for i in model.keys():
        if i == 'wave':
            out[:,0] = model[i][index]
            continue
        out[:, 1] += (model[i][index] * 
                      param.take(xrange(2, len(param), 3))[int(i)])
    #turn age and metal into spect format
    info_out = []
    for i in xrange(len(age)):
        info_out.append('ssp_%1.4f_%1.6f.spec' %(10**metal[i], age[i]))
    #specrtra,info in ssp lib fmt, weight, dust(tau_ism,tau_BC)
    return out, info_out, norm, param[-2:]

def check(param, age_bins, dust_bound, lib_vals, age_unq, metal_unq):
    #checks to see if points are in param space
    bins = (len(param) - 2) / 3
    out = True
    if len(age_bins) <= 2:
        #all types of binning should look same
        for i in xrange(1, 3 * bins, 3):
            #age
            if param[i] > age_bins.max() or param[i] < age_bins.min(): 
                return False
                
    else: #some type of binning
        for i in xrange(bins):
            if (param[i * 3 + 1] > age_bins[i + 1] or 
                param[i * 3 + 1] < age_bins[i]):
                return False
                
    #make sure in boundary of points
    index = nu.sort(nu.hstack((range(0,bins*3,3),range(1,bins*3,3))))
    index = index.reshape(bins,2)
    if nu.all(lib_vals[0][:,0] < 0): #metalicity is in log form
        hull = bound.find_boundary(lib_vals[0])
    else:
        lib_vals[0][:,0] = nu.log10(lib_vals[0][:,0])
        hull = bound.find_boundary(lib_vals[0])

    for i in xrange(bins):
        if not bound.pinp_wbounds(param[index][i], hull):
            return False

    #check dust
    if (nu.any(param[-2:] > dust_bound.max()) or 
        nu.any(param[-2:] < dust_bound.min())):
        return False
    return True


#create spectra with different SFR
####add a weighting function
def SFR_func(func, bins, age):
    if func == 'flat':
        norm = nu.ones(bins)
    elif func == 'slope':
        norm = slope * age
    elif func == 'norm':
        norm = 5 * nu.exp(-(age - 9.98) ** 2 / (2))
    elif func == 'expo':
        norm = expo(age)
    elif func == 'sinc':
        norm = sinc(age)
    elif func == 'stnt':
            #double peak
        norm = (5 * (1 + ((age - 8.5) / 0.05)**2 / 0.4)**(-(.4 + 
                                                             1) / 2) +
                (1 + ((age - 5.5) / 0.05)**2 / 0.4)**(-(.4 + 1) / 2))
    if bins == 1: 
        norm[0] = 1.
    return norm

def sinc(age,x=7.5):
    #sinc^2 functions centerd at x
    return 50 * nu.sinc(age - x)**2.

def expo(age, decay=2.):
    #exponential decay function
    return 0.1 * nu.exp(age / decay)

def flat(bins, age_lower, age_upper, 
         lib,lib_path):
    #makes a sfr that is constant over time
    t = nu.log10(nu.linspace(10**age_lower, 10**age_upper, num=bins + 1))
    specra_names = []
    for i in range(len(t) - 1):
        specra_names.append(search(lib, t[i], t[i + 1]))
        try:
            outspec[:,1 ] = (outspec[:, 1] + 
                             read_spec(specra_names[-1], lib_path)[:, 1])
        except NameError: #for first itteration when outspec not defined
            outspec = read_spec(specra_names[-1], lib_path)
    return outspec, specra_names, nu.ones(bins)

def normal(bins, age_lower, age_upper, lib, lib_path):
    bin = nu.log10(nu.linspace(10**age_lower, 10**age_upper, num=bins + 1))
    norm = []
    specra_names = []
    for i in range(len(bin)-1):
        specra_names.append(search(lib, bin[i], bin[i + 1]))
        norm.append(5 * nu.exp(-(float(specra_names[-1][11:-5]) 
                                 - (7.))**2 / (2 * (age_upper - 
                                                    age_lower) * 0.1)))
        try:
            outspec[:,1] = (outspec[:,1] + norm[i] * 
                            read_spec(specra_names[-1], lib_path)[:, 1])
        except NameError: #for first itteration when outspec not defined
            outspec = read_spec(specra_names[-1], lib_path)
            outspec[:,1] = norm[i] * outspec[:, 1]
    return outspec,specra_names,nu.array(norm)

def line(slope, bins, age_lower, age_upper, lib, lib_path):
    #makes a spectra with a line
    bin = nu.log10(nu.linspace(10**age_lower, 10**age_upper, num=bins+1))
    norm = []
    specra_names = []
    for i in range(len(bin) - 1):
        specra_names.append(search(lib, bin[i], bin[i+1]))
        if slope >= 0:        
            norm.append(float(specra_names[-1][11:-5]) * slope)
        else:
            norm.append(float(specra_names[-1][11:-5]) *
                        slope + (1. - slope * age_upper))
        try:
            outspec[:,1] = (outspec[:,1] + norm[i] *
                            read_spec(specra_names[-1], lib_path)[:,1])
        except NameError: #for first itteration when outspec not defined
            outspec = read_spec(specra_names[-1], lib_path)
            outspec[:, 1] = norm[i] * outspec[:, 1]

    if nu.any(nu.array(norm) < 0):
        print 'Value less than zero'
        raise
    return outspec, specra_names, nu.array(norm)

def student_t(bins,max_SFR,std_SFR):
    pass

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
