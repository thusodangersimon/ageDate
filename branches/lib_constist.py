#!/usr/bin/env python
#
# Name:  lib consist
#
# Author: Thuso S Simon
#
# Date: 15 Aug, 2011
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
'''
checks the consistancy of spectral libraries, and reports trends between them
'''

lib_path='/home/thuso/Phd/Code/Spectra_lib/'
import numpy as nu
import pylab as lab
import multiprocessing as multi
import os
 
def chi_check(lib_path):
    spect,info= load_spec_lib(lib_path)  
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    split_spec=split(spect[:,1:],lib_vals,age_unq,metal_unq)
    for j in range(len(metal_unq)):
        age=[]
        for i in range(1,len(age_unq)):
            age.append(sum((split_spec[j][:,i-1]-normalize(split_spec[j][:,i-1],split_spec[j][:,i])*split_spec[j][:,i])**2))
        lab.semilogy(age_unq[1:],age,label=str(metal_unq[j]))

    lab.legend()
    lab.xlabel('log(AGE [Gyr])')
    lab.ylabel('log(chi^2)')
    lab.show()

def split(spect,lib_vals,age_unq,metal_unq):
    #splits spectra into seperate list by metals
    out=[]
    for i in metal_unq:
        index=nu.nonzero(lib_vals[0][:,0]==i)[0]
        temp=lib_vals[0][index,1]
        out.append(spect[:,index])
        out[-1]=out[-1][:,temp.argsort()]

    return out

def slope_check(lib_path):
    spect,info= load_spec_lib(lib_path)  
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    split_spec=split(spect[:,1:],lib_vals,age_unq,metal_unq)
    for j in range(len(metal_unq)):
        age=[]
        for i in range(1,len(age_unq)):


def ratio_check(lib_path):
    spect,info= load_spec_lib(lib_path)  
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    split_spec=split(spect[:,1:],lib_vals,age_unq,metal_unq)
    for j in range(len(metal_unq)):
        age=[]
        for i in range(1,len(age_unq)):
            #temp=abs(sum((1+split_spec[j][:,i-1])/(1+split_spec[j][:,i]))-len(split_spec[j][:,i-1]))
            temp=sum((1+split_spec[j][:,i-1])/(1+split_spec[j][:,i]))
            age.append(temp)
        lab.semilogy(age_unq[1:],age,label=str(metal_unq[j]))

    lab.legend()
    lab.xlabel('AGE Gyr')
    lab.ylabel('chi^2')
    lab.show()


def normalize(data,model):
    #normalizes the model spectra so it is closest to the data
    return sum(data*model)/sum(model**2)


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
def search(lib,point_min,point_max):  
    #finds closest age spectra and returns file name
    index=nu.nonzero(nu.logical_and(lib[0][:,1]>=point_min,
                                    lib[0][:,1]<=point_max))[0]
    #lib[0][nu.argsort(dist)]
    return lib[1][index[
        nu.random.randint(len(index))]] 
