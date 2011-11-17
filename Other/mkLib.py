#!/usr/bin/env python
#
# Name: Make spectral library
#
# Author: Thuso S Simon
#
# Date: 4th of Aug. 2011
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
funtions to make a spectral library, from raw files, wrong format etc...

Gives standard names to files
"""


import os
import numpy as nu
import csv

def from_file(inpath,outpath):
    #makes exsisting spectra files into stadard using a file with the refeences
    
    if inpath[-1]!='/':
        inpath=inpath+'/'
    if outpath[-1]!='/':
        outpath=outpath+'/'

    #load file names
    files=nu.array(os.listdir(inpath))
    #search for file without *.spec name
    for i in files: 
        if i[-4:]!='spec':
            break
    
    #open and start reading in file
    con_table=open(inpath+i,'r')
    files_match,params=[],[]
    for ii in con_table:
        if ii[0][0]=='#':
            continue
        params.append([float(ii[40:50]),float(ii[56:63])])
        files_match.append(ii[:32])
    params=nu.array(params)
    params[:,0]=age_convrt(params[:,0])
    #check for inf and remove
    index=nu.nonzero(~nu.isinf(params[:,0]))[0]
    params=params[index,:]
    files_match=nu.array(files_match)[index]
    #write files in new format
    for i,j in enumerate( files_match):
        temp_spec=open(inpath+j,'r')
        #make outname
        outname='ssp_%1.4f_%1.6f.spec' %(params[i,1],params[i,0])
        out_file=open(outpath+outname,'w')
        out_file.write(temp_spec.read())

        temp_spec.close()
        out_file.close()

def age_convrt(age):
    #goes from Gyrs to Enrinco format
    return nu.log10(age)
def metal_convrt(Z):
    #goes from Z_solar to Enrico format
    return nu.log10(Z+1.1111)+5
def metal_non_inf_conver(Z):
    #if their are 0's in metalicty so no nan crop up
    return 

def Mar05(inpath,outpath):
    #formats the Mar05 ssp into standard format
    
    if inpath[-1]!='/':
        inpath=inpath+'/'
    if outpath[-1]!='/':
        outpath=outpath+'/'

    #load file names
    files=nu.array(os.listdir(inpath))
    for i in files:
        #load file
        temp=nu.loadtxt(inpath+i)
        #split
        split=temp.shape[0]/nu.unique(temp[:,2]).shape[0]
        split_files=nu.split(temp,split)
        for j in split_files:
            if j[0,1]==.67:
                metal=.007
            elif j[0,1]==.35:
                metal=.004
            elif j[0,1]==0:
                metal=.002
            elif j[0,1]==-.33:
                metal=.001
            elif j[0,1]==-1.35:
                metal=.0001
            elif j[0,1]==-2.25:
                metal=10**-4
            outname='ssp_%1.4f_%1.6f.spec' %(metal,nu.log10(j[0,0]*10**9))
            nu.savetxt(outpath+outname,j[:,2:4])

def speed(inpath,outpath):
    pass

def miles(inpath,outpath):
    if inpath[-1]!='/':
        inpath=inpath+'/'
    if outpath[-1]!='/':
        outpath=outpath+'/'

    #load file names
    files=nu.array(os.listdir(inpath))
    for i in files:
        #load file
        temp=nu.loadtxt(inpath+i)
        outname='ssp_%1.4f_%1.6f.spec' %(metal_non_inf_conver(float(i[9:13])),nu.log10(float(i[14:])*10**9))
        nu.savetxt(outpath+outname,temp)

def FSPS(inpath,outpath,lambda_file):
    #should have a location to wavelength file as well as in and out path
    pass
