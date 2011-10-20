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
does test on age date program
"""


from Age_date import *

def nn_least(data):
    #uses a non-linear least squares to fit data
    lam=spect[:,0]
    index=nu.nonzero(nu.logical_and(lam>=min(data[:,0]),
                                    lam<=max(data[:,0])))[0]
    spect=spect[index,1:]
    N=T.nnls(spect,data[:,1]*1000)[0]
    current=info[N>0.01]
    metal,age=[],[]
    for i in current:
        metal.append(float(i[4:10]))
        age.append(float(i[11:-5]))
    #check if any left
    if len(current)<2:
        return float(current[4:10]),float(current[11:-5]),N[N>10**-4]

    return metal[nu.argsort(age)],age[nu.argsort(age)],N[N>10**-4][nu.argsort(age)]/1000.


'''
age1=[]
for i in info1:
    age1.append(float(i[11:-5]))
out=nu.zeros(len(    lam=spect[:,0]
    index=nu.nonzero(nu.logical_and(lam>=min(data[:,0]),
                                    lam<=max(data[:,0])))[0]
    spect=spect[index,1:]
    N=T.nnls(spect,data[:,1]*1000)[0]
    current=info[N>0.01]
    metal,age=[],[]
    for i in current:
        metal.append(float(i[4:10]))
        age.append(float(i[11:-5]))
    #check if any left
    if len(current)<2:
        return float(current[4:10]),float(current[11:-5]),N[N>10**-4]

    return metal[nu.argsort(age)],age[nu.argsort(age)],N[N>10**-4][nu.argsort(age)]/1000.



age1=[]
for i in info1:
    age1.append(float(i[11:-5]))
out=nu.zeros(len(spect[:,0]))
for i in range(len(N)):
   out+=N[0][i]*spect[:,i]


'''
