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
from scipy.cluster import vq 

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


def plot_SFR_age(param,chi,plot=True):
    '''plots age vs SFR for all models with above 30k itterations in them
    uses k means clustering to seperate chains incase the went through
    some "label switiching."'''

    results = {}
    for i in param.keys():
        #make sure has 30K chains
        if len(param[i]) >= 3 * 10**4:
            results[i] = {'age':[], 'SFR': [], 'metal':[]}
            #get age and norm chains
            age = 10**param[i][:,range(1,3*int(i),3)].ravel()/10**9.
            norm = param[i][:,range(2,3*int(i),3)].ravel()
            metal = param[i][:,range(0,3*int(i),3)].ravel()
            #do kmeans clustering
            means, p = vq.kmeans2(age,int(i))
            for j in nu.unique(p):
                results[i]['age'].append(nu.percentile(age[p == j],
                                                       [50,16,84]))
                results[i]['SFR'].append(nu.percentile(norm[p == j],
                                                       [50,16,84]))
                results[i]['metal'].append(nu.percentile(metal[p == j],
                                                         [50,16,84]))
            
            results[i]['age'] = nu.array(results[i]['age'])
            results[i]['SFR'] = nu.array(results[i]['SFR'])
            results[i]['metal'] = nu.array(results[i]['metal'])
            #make quitiles relitive
            results[i]['age'][:,1] = (results[i]['age'][:,0] - 
                                      results[i]['age'][:,1])
            results[i]['age'][:,2] = (results[i]['age'][:,2] - 
                                      results[i]['age'][:,0])
            
            results[i]['SFR'][:,1] = (results[i]['SFR'][:,0] - 
                                      results[i]['SFR'][:,1])
            results[i]['SFR'][:,2] = (results[i]['SFR'][:,2] - 
                                      results[i]['SFR'][:,0])
            
            results[i]['metal'][:,1] = (results[i]['metal'][:,0] - 
                                        results[i]['metal'][:,1])
            results[i]['metal'][:,2] = (results[i]['metal'][:,2] - 
                                        results[i]['metal'][:,0])
            
            #plot
            if plot:
                lab.figure()
                lab.title('%s params' %i)
                lab.xlabel('Age (Gyrs)')
                lab.ylabel('SFR')
                lab.errorbar(results[i]['age'][:,0],results[i]['SFR'][:,0],
                             xerr=results[i]['age'][:,1:].T,
                             yerr=results[i]['SFR'][:,1:].T,
                             fmt='.')
    return results
