#!/usr/bin/env python
#
# Name:  check mc output
#
# Author: Thuso S Simon
#
# Date: 
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
'''tools to check the output of MCMC and RJMCMC'''

import numpy as nu
import pylab as lab
import cPickle as pik
import os, sys
import Age_date as mc
from scipy.stats import sigmaclip


def RJmcmc_LRG_check(indir,bins=1):

    'looks at LRG output from fit_all.py Looks at all .pik files in dir and plots them'
    if not indir[-1]=='/':
        indir+='/'
    #load file names
    files=os.listdir(indir)
    print 'Plotting from %i bin' %bins
    age,norm,metal,Z=[],[],[],[]
    for i in files:
        try:
            temp=pik.load(open(indir+i))
            lab.figure()
            mc.plot_model(temp[0][str(bins)][temp[1][str(bins)].min()==
                                         temp[1][str(bins)]][0],     
                          temp[3][i[:-4]],bins)
            lab.title(i)
        #get median,lower, upper bound of parameters
            age.append(nu.percentile(sigmaclip(10**temp[0][str(bins)][:,1])[0],[50,15.9,84.1]))
            metal.append(nu.percentile(sigmaclip(temp[0][str(bins)][:,0])[0],[50,15.9,84.1]))
            norm.append(nu.percentile(sigmaclip(temp[0][str(bins)][:,2])[0],[50,15.9,84.1]))
            Z.append(float(i[:-4]))
        except:
            continue
    age,metal,norm,Z=nu.array(age),nu.array(metal),nu.array(norm),nu.array(Z)
    #make uncertanties relitive
    age[:,1],age[:,2]=nu.abs(age[:,0]-age[:,1]),nu.abs(age[:,0]-age[:,2])
    age=age/10**9.

    lab.figure()
    lab.errorbar(Z,age[:,0],yerr=age[:,1:].T,fmt='.')
    lab.xlabel('Redshift')
    lab.ylabel('Age (Gyr)')
    
    lab.show()
