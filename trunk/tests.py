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

def make_chi(flux,SSP,t,z,del_index):
    chi = nu.zeros_like(t)
    d = nu.vstack((t.ravel(),z.ravel())).T
    for i in range(d.shape[0]):
        index = nu.unravel_index(i,t.shape)
        try:
            spec = SSP.get_sed(10**(d[i][0]-9),10**d[i][1])
        except ValueError:
            if nu.allclose(10**(d[i]-9)[0] ,20):
                spec = SSP.get_sed(20.,10**d[i][1])
            else:
                chi = nu.inf
                continue
        chi[index] = nu.sum((spec[del_index]-flux)**2)
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
if __name__ == '__main__':
    #entropy calculation with plots
    fun = lik.VESPA_fit(nu.ones((2,2)),spec_lib='bc03')
    
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
    pmf[pmf == 0] = 10**-99
    #lab.plot(d[:,0],  -nu.sum(nu.log10(pmf)*pmf,axis=1),'.') 
    
    #make animation of how information changes likelihood
    #get spectra
    chi = []
    flux = spec[nu.argmax(-nu.sum(nu.log10(pmf)*pmf,axis=1))]
    H = pmf[nu.argmax(-nu.sum(nu.log10(pmf)*pmf,axis=1))]
    wave = SSP.sed_ls
    del_index = flux == flux
    print 'Making images'
    for i in xrange(len(wave)):
        chi.append([make_chi(flux[del_index],SSP,t,z,del_index),nu.nansum(H),nu.copy(del_index)])
        index = nu.nanargmin(H)
        H[index] = nu.nan
        del_index[index] = False
    pik.dump((chi,z,t,wave,flux),open('temp.pik','w'),2)
    print 'Saving animations as movie'
    #make animation
    an = anim(t, z, chi, wave, flux)
    ani = FuncAnimation(an.fig,an.make_im,frames = len(chi))
    ani.save('test.mp4')
    
