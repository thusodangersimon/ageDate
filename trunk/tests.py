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
scap develoment programs
"""
import likelihood_class as lik
import numpy as nu
import pylab as lab


def make_chi(flux,spec,t,z,del_index):
    chi = nu.zeros_like(t)
    d = nu.vstack((t.ravel(),z.ravel())).T
    for i in range(d.shape[0]):
        index = nu.unravel_index(i,t.shape)
        chi[index] = nu.sum((spec[i,del_index]-flux[del_index])**2)
    return chi
#entropy calculation with plots
fun = lik.VESPA_fit(nu.ones((2,2)),spec_lib='bc03')

SSP = fun.SSP
ages = fun._age_unq
metal = fun._metal_unq

t,z = nu.meshgrid(ages,metal)
spec = []
d = nu.vstack((t.ravel(),z.ravel())).T
for i in d:
    try:
        spec.append(SSP.get_sed(10**(i[0]-9),10**i[1]))
    except:
        spec.append(SSP.get_sed(round(10**(i[0]-9)),10**i[1]))

#make array
spec = nu.asarray(spec)
norm = nu.ones_like(spec[:,0])
#find normalization
for i in range(1,len(spec)):
    norm[i] = nu.sum(spec[0,:]*spec[i,:])/nu.sum(spec[i,:]**2)

#make probabity mass function matrix
pmf = nu.zeros_like(spec)
for i in range(spec.shape[1]):
    p = nu.copy(spec[:,i]*norm)
    #histogram
    h=nu.histogram(p,bins=nu.sort(p))
    H = []
    for j in h[0]:
        H.append(j)
    H[-1] /= 2.
    H.append(H[-1])
    unsorted = H/nu.float64(sum(H))
    pmf[:,i] = unsorted[nu.argsort(nu.argsort(p))]

#set minimun prob
pm[pmf == 0] = 10**-99
lab.plot(d[:,0],  -nu.sum(nu.log10(pmf)*pmf,axis=1),'.') 
for i in range(pmf.shape[0]):
    print -nu.sum(nu.log10(pmf)*pmf,axis=1)
#make animation of how information changes likelihood
#get spectra
flux = list(spec[nu.argmax(-nu.sum(nu.log10(pmf)*pmf,axis=1))])
H = pmf[nu.argmax(-nu.sum(nu.log10(pmf)*pmf,axis=1))]
wave = list(SSP.sed_ls)
del_index = flux == flux
for i in xrange(wave.shape[0]):
    chi.append( make_chi(flux,spec,t,z,del_index))
    del_index[nu.argmin(H[del_index])]=False 
