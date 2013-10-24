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
from matplotlib.animation import FuncAnimation
import cPickle as pik

def make_chi(flux,spec,t,z,del_index):
    chi = nu.zeros_like(t)
    d = nu.vstack((t.ravel(),z.ravel())).T
    for i in range(d.shape[0]):
        index = nu.unravel_index(i,t.shape)
        chi[index] = nu.sum((spec[i,del_index]-flux)**2)/float(len(flux)-3)
    return chi

class anim(object):

    def __init__(self, t, z, chi, wave, flux):
        #set all things needed for making animation
        self.t = t
        self.z = z
        self.wave = wave
        self.flux = flux
        self.chi = chi
        self.fig = lab.figure()
        self.plt_spec = self.fig.add_subplot(211)
        self.plt_chi = self.fig.add_subplot(212)
        self.plt_chi.set_xlabel('$log_{10}(age)$')
        self.plt_chi.set_ylabel('$log_{10}(Z)$')
        self.plt_spec.set_xlabel('$\lambda$ (A)')
        self.plt_spec.set_ylabel('Normalized flus')
        self.plt_spec.set_xlim((2000,10000))
        self.fig.canvas.draw()

    def make_im(self,j):        
        #generator that does plotting
        i = self.chi[j]
        self.plt_chi.clear()
        self.plt_chi.set_xlabel('$log_{10}(age)$')
        self.plt_chi.set_ylabel('$log_{10}(Z)$')
        self.plt_chi.pcolormesh(self.t,self.z,i[0])
        self.plt_spec.set_title('Total Information content is %2.2f'%i[1])
        #plot spectrum
        if len(self.plt_spec.lines) > 0:
            self.plt_spec.lines.pop(0)
        self.plt_spec.plot(self.wave[i[2]],self.flux[i[2]],'b.')
    
        #fig.canvas.draw()
def get_information(data,spec):
    #calculates information content of data from spec
    ###make information 
    norm = nu.ones_like(spec[:,0])
    #find normalization
    for i in range(1,len(spec)):
        norm[i] = nu.sum(data[:,1]*spec[i,:])/nu.sum(spec[i,:]**2)
    #replace nans with 0
    norm[nu.isnan(norm)] = 0.
    #get normalization for data
    #make probabity mass function matrix
    pmf = nu.zeros_like(spec)
    for i in xrange(spec.shape[1]):
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
    pmf[pmf == 0] = 10**-99
    #find infomation content of data
    H = []
    for i in xrange(spec.shape[1]):
        sorted_spec = nu.sort(spec[:,i]*norm)
        arg_sort = nu.argsort(spec[:,i]*norm)
        j = nu.searchsorted(sorted_spec,data[:,1][i])
        if j == sorted_spec.shape[0]:
            H.append(pmf[arg_sort,i][j-1])
        else:
            H.append(pmf[arg_sort,i][j])
        
    return nu.asarray(H)

def shannon(p):
    return -nu.log10(p)*p

def mod_shannon(p):
    return -nu.log10(p)*(1-p)
        
def main(data,outmov):
    #entropy calculation with plots
    fun = lik.VESPA_fit(data,spec_lib='bc03')
    
    SSP = fun.SSP
    ages = fun._age_unq
    metal = nu.linspace(fun._metal_unq.min(),fun._metal_unq.max(),10)

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
    #match wavelenth with data
    if not nu.all(nu.sort(SSP.sed_ls) == SSP.sed_ls):
        #check if sorted
        wave = SSP.sed_ls[::-1]
        spec = spec[:,::-1]
    else:
        wave = SSP.sed_ls
    new_spec = nu.zeros((len(d),len(data)))
    for i in xrange(len(d)):
        new_spec[i,:] = nu.interp(data[:,0],wave,spec[i,:])
    spec = new_spec
    H = get_information(data,spec)
    
    #make animation of how information changes likelihood
    #get spectra
    chi = []
    flux = data[:,1]
    #how i think the enropy should look -sum((1-p)*log(p))
    #tot_infor = nu.sum(mod_shannon(H))
    #H = mod_shannon(H)
    H = shannon(H)
    wave =data[:,0]
    del_index = flux == flux
    print 'Making images'
    for i in xrange(len(wave)):
        index = nu.nanargmin(H)
        H[index] = nu.nan
        del_index[index] = False
        chi.append([make_chi(flux[del_index],spec,t,z,del_index),nu.nansum(H),nu.copy(del_index)])
    pik.dump((chi,z,t,wave,flux),open('temp.pik','w'),2)
    print 'Saving animations as movie'
    #make animation
    an = anim(t, z, chi, wave, flux)
    ani = FuncAnimation(an.fig,an.make_im,frames = len(chi))
    ani.save(outmov+'.mp4')
    #lab.show()
