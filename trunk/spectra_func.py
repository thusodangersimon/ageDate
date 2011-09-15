#!/usr/bin/env python
#
# Name:  Age Dating Spectra Fitting Program
#
# Author: Thuso S Simon
#
# Date: 28th of June, 2011
#TODO:
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

import numpy as nu
import os, sys
from multiprocessing import Pool

def read_spec(name,lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #reads in spec and turns into numpy array [lambda,flux]
    return nu.loadtxt(lib_path+name)

def get_fitting_info(lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #gets list of ages,metalicity and asocated file names
    lib=os.listdir(lib_path)
    out=nu.zeros([len(lib),2])
    standard_file=[]
    for j,i in enumerate(lib):
        if i[-4:]!='spec':
            
            continue
        standard_file.append(j)
        out[j,:]=[float(i[4:10]),float(i[11:-5])]
     
    return out[standard_file,:],nu.array(lib)[standard_file]   

def load_spec_lib(lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #loads all spectra into libary
    lib=get_fitting_info(lib_path)[1]
    temp=read_spec(lib[0],lib_path)
    #create array
    out=nu.zeros([temp.shape[0],len(lib)+1])#out[:,0]=wavelength range
    out[:,0]=temp[:,0]
    #pool=Pool()
    #m=pool.map
    m=map
    tempout=m(read_spec,lib,[lib_path]*len(lib))
    #pool.close()
    for i,j in enumerate(tempout):
        out[:,i+1]=j[:,1]
    return out, lib

def edit_spec_range(spect,lam_min,lam_max):
    index=nu.nonzero(nu.logical_and(spect[:,0]>=lam_min,spect[:,0]<=lam_max))[0]
    return spect[index,:]

def create_spectra(bins,func='flat',lam_min=0,
                   lam_max=nu.inf,lib_path='/home/thuso/Phd/Spectra_lib/'):
    #creates a SFH function and matches SSP's to it with 
    #inputted bins based on sfr/t*delta*
    lib=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib[0][:,0])
    age_unq=nu.unique(lib[0][:,1])

    #####old########
    if bins==1:
        names=lib[1][nu.random.randint(0,len(lib[0]))]
        spect= nu.loadtxt(lib_path+names)
        weights=1
    else:
    #initalize out spectra
    #make sfh and bin areas
        if func=='normal':
            SFR=normal(t,gal_mass)
        elif func=='expo':
            SFR=expo(t,gal_mass)
        elif func=='sinc':
            SFR=sinc(t,gal_mass)
        else:
            spect,names,weights=flat(bins,age_unq.min(),age_unq.max(),lib) 

    #fix wavelenth range
    index=nu.nonzero(nu.logical_and(spect[:,0]>=lam_min,spect[:,0]<=lam_max))[0]
    return spect[index,:],names,weights

#create spectra with different SFR
####add a weighting function
def flat(bins,age_lower,age_upper,lib):
    #makes a sfr that is constant over time
    t=nu.linspace(age_lower,age_upper,num=bins+1)
    specra_names=[]
    for i in range(len(t)-1):
        specra_names.append(search(lib,t[i],t[i+1]))
        try:
            outspec[:,1]=outspec[:,1]+read_spec(specra_names[-1])[:,1]
        except NameError: #for first itteration when outspec not defined
            outspec=read_spec(specra_names[-1])
    return outspec,specra_names,nu.ones(bins)

def normal(bins,max_SFR,std_SFR):
    pass

def expo(t,gal_mass,lam_min,lam_max):
    pass

def age_covert(age):
    #turns age from lister values into Gyr
    return 10** (age -8)

def metal_convert(metal):
    #converts metalicity to bishops calc 
    ####need covert to better way
    return 10** (metal -5)

def search(lib,point_min,point_max):  
    #finds closest age spectra and returns file name
    index=nu.nonzero(nu.logical_and(lib[0][:,1]>=point_min,
                                    lib[0][:,1]<=point_max))[0]
    #lib[0][nu.argsort(dist)]
    return lib[1][index[
        nu.random.randint(len(index))]] 

def info_for_lib(lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #calculates information content of spectral library
    spec_lib,info=load_spec_lib(lib_path)
    #create prob dist for each wavelength
    prob_dist={}
    for i in range(len(spec_lib[:,0])):
        temp=nu.zeros([nu.unique(spec_lib[i,1:]).shape[0]-1,2])
        temp[:,0]=nu.unique(spec_lib[i,1:])[:-1]
        temp[:,1]=nu.histogram(spec_lib[i,1:],nu.unique(spec_lib[i,1:]))[0]
        temp[:,1]=temp[:,1]/sum( temp[:,1])
        prob_dist[str(spec_lib[i,0])]=nu.copy(temp)

    return prob_dist

def info_for_spec(data,prob_dist=None,lib_path='/home/thuso/Phd/Code/Spectra_lib/'):
    #calculates information content of single spectra with respect to
    #current spectral library #only works is spectra have same wavelenths
    if not prob_dist:
        prob_dist=info_for_lib(lib_path)
    H=0
    for i in range(data.shape[0]):
        if not any(nu.array(prob_dist.keys())==str(data[i,0])):
            print 'wavelenthd to not match'
            raise

        temp=prob_dist[str(data[i,0])]
        H=H-temp[nu.argsort((temp[:,0]-data[i,1])**2),1][0]*nu.log10(
            temp[nu.argsort(temp[:,0]-data[i,1]),1][0])

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
