#!/usr/bin/env python
#
# Name:  Age Dating Spectra Fitting Program
#
# Author: Thuso S Simon
#
# Date: 7th of June, 2011
#TODO:  
#    
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
""" Utilites used for spectral fitting with ssps"""

import numpy as nu
import os
import sys
from interp_utils import *
from spectra_lib_utils import *
from scipy.optimize import nnls
from scipy.optimize.minpack import leastsq
from scipy.optimize import fmin_l_bfgs_b as fmin_bound
from scipy.special import exp1 
from scipy.integrate import simps
from pysynphot import observation,spectrum
import time as Time
import boundary as bound
from MC_utils import issorted
#sfr2energy = 1.0/7.9D-42    ; (erg/s) / (M_sun/yr) [Kennicutt 1998]

#123456789012345678901234567890123456789012345678901234567890123456789
###decorators
class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
      self.func = func
      self.__doc__ = func.__doc__
      self.cache = {}
      self.cache['gauss1'] = {}
      self.cache['dust'] = {}
    def __call__(self, *args):
      #works different for different functions
      if self.func.__name__ == 'gauss_kernel':
        #remove cache if get's too big
        if nu.random.rand() < .01:
          l = len(self.cache['gauss1'])
          if l > 10**5:
            self.cache['gauss1'].clear()
        arg = ''
        for i in args:
          arg+=str(i)+ ' '
        if arg in self.cache['gauss1']:
          return self.cache['gauss1'][arg]
        else:
          value = self.func(*args)
          self.cache['gauss1'][arg] = value
          return value
      
      elif self.func.__name__ == 'f_dust':
        if nu.random.rand() < .01:
          l = len(self.cache['dust'])
          if l > 10**5:
            self.cache['dust'].clear()

        arg = str(args)
        if not arg in self.cache['dust']:
          self.cache['dust'][arg] = self.func(*args)
        return self.cache['dust'][arg]
      else:
          if args in self.cache:
            return self.cache[args]
          else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
                  

###spectral lib stuff####
'''global lib_path,spect
lib_path = '/home/thuso/Phd/Spectra_lib/'
spect,info = None,None'''
'''try:
    #load in spectral library, if default paths isn't correct ask
    #for correct path to lib
    spect,info = load_spec_lib(lib_path)  
except OSError :
    lib_path=raw_input('location to spectra libray? eg.'
                       + '/home/thuso/Phd/Spectra_lib/')
    if not lib_path[-1] == '/':
        lib_path += '/'
    spect,info = load_spec_lib(lib_path)
    '''              
def find_az_box(param, age_unq, metal_unq):
    '''Find 4 cloest values from param in age and meta_unq'''

    #find closest metal
    line = None
    #check metal
    metal = []
    if nu.any(metal_unq == param[0]):
        #on metal line 
        line = 'metal'
        metal.append(metal_unq[metal_unq == param[0]][0])
    else:
        index = nu.searchsorted(metal_unq, param[0])
        if index <= 0 or index >= len(metal_unq): #out of bounds
            print 'metalicity param is out of bounds'
            raise
        metal.append(metal_unq[index])
        metal.append(metal_unq[index - 1])
    metal *= 2
    age = []
    if nu.any(age_unq == param[1]): #check if on age line
        if line == 'metal': #on metal line and age line
            line = 'both'
        else: #only on age line
            line = 'age'
        age.append(age_unq[age_unq == param[1]][0])
    else: #not on age line
        index = nu.searchsorted(age_unq,param[1])
        if index <= 0 or index >= len(age_unq):
            #out of bounds
            raise ValueError('age param is out of bounds')
        age.append(age_unq[index])
        age.append(age_unq[index - 1])
    age *= 2
    return metal, age,line


def get_model_fit_opt(param, lib_vals, age_unq, metal_unq, bins, 
						spect):
    '''does dirty work to make spectra models
    search age_unq and metal_unq to find closet box spectra and interps
    does multi componets spectra'''
    out = {}
    for ii in xrange(bins):
        temp_param = param[ii]
        metal, age, line = find_az_box(temp_param, age_unq, metal_unq)
        closest = []
        #check to see if on a lib spectra or on a line
        if line == 'age': #run 1 d interp along metal only
            metal = nu.array([metal[0], metal[-1]])
            metal.sort()
            age = age[0]
            #find spectra
            for i in  metal:
                index = nu.nonzero(nu.logical_and(lib_vals[0][:,0] == i
                                                  ,lib_vals[0][:,1] ==age))[0]
                if len(index)<1: #if not in lib_vals return dummy array
                    out[str(ii)] = spect[:,0] + nu.inf
                    interp = False
                    break
                else:
                    closest.append(spect[:,index[0] + 1])
                    interp = True
            if interp:
                out[str(ii)] = linear_interpolation(metal,
                                                    closest,  
                                                    temp_param[0])
        elif line == 'metal': #run 1 d interp along age only
            age = nu.array([age[0], age[-1]])
            age.sort()
            metal = metal[0]
            #find spectra
            
            for i in age:
                index = nu.nonzero(nu.logical_and(lib_vals[0][:,1] ==
                                                  i, lib_vals[0][:,0]
                                                  ==  metal))[0]
                
                if len(index)<1: #if not in lib_vals return dummy array
                    out[str(ii)] = spect[:,0] + nu.inf
                    interp = False
                    break
                else:
                    closest.append(spect[:,index[0] + 1])
                    interp = True
            if interp:
                out[str(ii)] = linear_interpolation(age, closest,
                                                    temp_param[1])

        elif line == 'both': #on a lib spectra
            index=nu.nonzero(nu.logical_and(lib_vals[0][:,0] == 
                                            temp_param[0], 
                                            lib_vals[0][:,1] == 
                                            temp_param[1]))[0]
            if len(index) < 1: #if not in lib_vals return dummy array
                out[str(ii)] = spect[:,0] + nu.inf
            else:
                out[str(ii)] = nu.copy(spect[:,index[0] + 1])
        #run 2 d interp
        else:
            metal.sort()
            metal = nu.array(metal)[nu.array([0, 3, 1, 2], dtype =
                                             'int32')]
            age.sort()
            
            for i in xrange(4):
                index = nu.nonzero(nu.logical_and(lib_vals[0][:, 1] ==
                                                  age[i],
                                                  lib_vals[0][:, 0] ==
                                                   metal[i]))[0]
                if len(index)<1: #if not in lib_vals return dummy array
                    out[str(ii)] = spect[:, 0] + nu.inf
                    interp = False
                    break
                else:
                    closest.append(spect[:, index[0] + 1])
                    interp = True
            if interp:
                out[str(ii)] = bilinear_interpolation( metal, age
                                                      , closest,
                                                      temp_param[0],
                                                      temp_param[1])
    #give wavelength axis
    out['wave'] = nu.copy(spect[:, 0])

   #exit program
    return out

def ez_to_rj(SSP):
    '''Get ssp's from EZGAL and make usable for this program'''
    spect, info = [SSP.sed_ls], []
    for i in SSP:
        metal = float(i.meta_data['met'])
        ages = nu.float64(i.ages)
        for j in ages:
            if j == 0:
                continue
            spect.append(i.get_sed(j,age_units='yrs'))
            info.append([metal+.0,j])
    info, spect = [nu.log10(info),None],nu.asarray(spect).T
    #test if sorted
    if not issorted(spect[:,0]):
        spect = spect[::-1,:]

    return info, spect

#@profile
def make_burst(length, T, metal, lib_vals, spect):
    '''def make_burst(length, t, metal, lib_vals,spect)
    (float, float,float, list(ndarray,ndarry),ndarray) -> ndarray(float)
Turns SSP into busrt of constant stellar formation and of length dt at
age t for a const metalicity 10**(t-9) - length/2 to 10**(t-9) + length/2.
All terms are logrythmic.
'''
    #set up boundaries
    age_unq = nu.unique(lib_vals[0][:,1])
    t = 10**T
    #metal = 10**Metal
    metal_unq = nu.unique(lib_vals[0][:,0])
    if T < age_unq.min() or T > age_unq.max():
		#Age not in range
		return spect[:,0] +  nu.inf
    if metal < metal_unq.min() or metal > metal_unq.max():
		#Metalicity not in range
		return spect[:,0] +  nu.inf
	#get all ssp's with correct age range and metalicity
	#min age range
    if (T - length/2.) < age_unq.min():
		t_min = age_unq.min() + 0.
    else:
		t_min = T - length/2.
	#max age range
    if 	T + length/2. > age_unq.max():
		t_max = age_unq.max() + 0.
    else:
		t_max = T + length/2.
    index = nu.searchsorted(age_unq, [t_min,t_max])
    ages = age_unq[index[0]:index[1]]
	#get SSP's
	
	#handel situation where stepsize is small
    if len(ages) < 10:
        ages = nu.linspace(t_min,t_max,10)
    temp_param = []
    #get ssp's
    try:
        #param [[metal,age,norm]]
        for i in ages:
            temp_param.append([metal,i,0]) 
        ssps = get_model_fit_opt(temp_param, lib_vals, age_unq, metal_unq,
                                ages.shape[0],spect)
        ssps.pop('wave')
        #sort for simps
        ssp = nu.zeros((ages.shape[0],ssps['0'].shape[0]))
        for i in xrange(ages.shape[0]):
            ssp[i] = ssps[str(i)]
        ages = 10**ages
    except ValueError as e:
        #interp failed
        print e
        return spect[:,0] +  nu.inf
    #integrate and normalize
    #del ssps,ages, age_unq,metal_unq
    return simps(ssp, ages, axis=0)/(ages.ptp())
    


def make_numeric( age, sfh, max_bins, metals=None, return_param=False):
        '''(ndarray,ndarray,ndarray,int,ndarray or None) -> ndarray
        Like EZGAL's function but with a vespa like framework.
        Bins function into flat chunks of average of SFH of function in that range.
        Can also put a function for metalicity to, otherwise will be random.
        '''
        age, sfh = nu.asarray(age), nu.asarray(sfh)
        assert age.shape[0] == sfh.shape[0], 'Age and SFH must have same shape'
        if max_bins > len(age):
            print 'Warning, more bins that ages not supported yet'
            
        #split up age range and give lengths
        param = []
        length = self._age_unq.ptp()/max_bins
        bins = nu.histogram(age,bins=max_bins)[1]
        #need to get length, t, norm and metals if avalible
        length, t, norm, metal = [], [] ,[] ,[]
        for i in range(0,len(bins)-1):
            index = nu.searchsorted(age,[bins[i],bins[i+1]])
            length.append(age[index[0]:index[1]].ptp())
            t.append(age[index[0]:index[1]].mean())
            norm.append(sfh[index[0]:index[1]].mean())
            if metals is None:
                #no input metal choose random
                metal.append(nu.random.choice(nu.linspace(self._metal_unq.min(),self._metal_unq.max())))
            else:
                pass
        #normalize norm to 1
        norm = nu.asarray(norm)/nu.sum(norm)
        #get bursts
        out = nu.zeros_like(self._spect[:,:2])
        out[:,0] = self._spect[:,0].copy()
        param = []
        for i in range(len(t)):
            if return_param:
                param.append([length[i],t[i],metal[i],norm[i]])
            out[:,1] += norm[i] * ag.make_burst(length[i],t[i],metal[i]
                          ,self._metal_unq, self._age_unq, self._spect
                          , self._lib_vals)
        if return_param:
            return out, nu.asarray(param)
        else:
            return out
        
def random_permute(seed):
    '''(seed (int)) -> int
    does random sequences to produice a random seed for parallel programs'''
    ##middle squared method
    seed = str(seed**2)
    while len(seed) < 7:
        seed=str(int(seed)**2)
    #do more randomization
    ##multiply with carry
    a,b,c = int(seed[-1]), 2**32, int(seed[-3])
    i = 0
    while a == 0:
        a = int(seed[-i])
        i+=1
    j = nu.random.random_integers(4, len(seed))
    for i in range(int(seed[-j:-j+3])):
        seed = (a*int(seed) + c) % b
        c = (a * seed + c) / b
        seed = str(seed)
    return int(seed)

def data_match_all(data, spect):
    '''makes sure data and model have same 
    wavelength range and points for library. Removes uncertanties = 0
    from data and lib spec'''
    model = {}
    #remove sigma = 0
    if data.shape[1] == 3:
        if nu.sum(data[:,2] == 0) > 0:
            data = data[data[:,2] !=0]
 
    out_data = nu.copy(data)
    #make spect_lib correct shape for changing
    for i in xrange(spect[0, :].shape[0]):
        if i == 0:
            model['wave'] = nu.copy(spect[:, i])
        else:
            model[str(i - 1)] = nu.copy(spect[:, i])
   #check to see if data wavelength is outside of spectra lib
    min_index = nu.searchsorted(model['wave'], out_data[0, 0])
    max_index = nu.searchsorted(model['wave'], out_data[-1, 0])
    if min_index == 0: #data is bellow spectra lib
        print ('Spectra Library has does not go down to the same'
               + ' wavelength as input spectrum.')
        out_data = data[nu.searchsorted(model['wave'], out_data[:, 0])
                        > 0, :]
    #data goes above spectra lib
    if max_index == model['wave'].shape[0]: 
        print ('Spectra Library has does not go up to the same '
               +  'wavelength as input spectrum.')
        out_data = out_data[nu.searchsorted(model['wave'], 
                                            out_data[:,0]) <
                            model['wave'].shape[0], :]
    #interpolate spectra lib so it has same wavelength axis as data 
    model = data_match(out_data, model,spect[0, :].shape[0] - 1)
    out = nu.zeros([model['0'].shape[0], len(model.keys()) + 1])
    out[:, 0] = nu.copy(out_data[:, 0])
    for i in model.keys():
        out[:, int(i) + 1] = model[i]
    spect = nu.copy(out)
    return spect, out_data

def data_match(data, model, keep_wave=False, rebin=True):
   '''(ndarray,dict(ndarray),bool) -> dict(ndarray)
   Makes sure data and model have same wavelength range 
    and points but with a dictionary
    assumes model wavelength range is longer than data.
    Keeps intergrated flux the same'''
   
   out = {}
    #if they have the same x-axis
   if nu.all(model['wave'] == data[:, 0]):
      for i in model.keys():
          if i == 'wave':
              continue
          out[i] = model[i]
   else:
     #not same shape, interp or rebin flux (correct way)
      for i in model.keys():
        if i == 'wave':
          continue
        if rebin:
            out[i] = rebin_spec(model['wave'], model[i], data[:,0])
        else:
          # interpolate
          out[i] = linear_interpolation(model['wave'], model[i], data[:,0])
          
   if keep_wave:
      out['wave'] = nu.copy(data[:,0])
   return out

def redshift(wave, redshift):
   #changes redshift of models
   return wave * (1. + nu.asarray(redshift))

def normalize(data, model):
    #normalizes the model spectra so it is closest to the data
    if data.shape[1] == 2:
        return nu.sum(data[:, 1] * model) / nu.sum(model ** 2)
    elif data.shape[1] == 3:
        return (nu.sum(data[:, 1] * model / data[:, 2] ** 2) / 
                nu.sum((model / data[:, 2]) ** 2))
    else:
        print 'wrong data shape'
        raise(KeyError)
    
def N_normalize(data, model, bins):
    #takes the norm for combined data and does a minimization 
    #for best fits value
    
    #match data axis with model already done by program
    #need to remove 'wave' key since no data_match_new
    if model.has_key('wave'):
        model.pop('wave')
    #do non-negitave least squares fit
    if bins == 1:
        N = [normalize(data, model['0'])]
        return N, nu.sum((data[:, 1] - N[0] * model['0']) ** 2)
    try:
        #if data has only wavelength and flux axis
        if data.shape[1] == 2:
            N, chi = nnls(nu.array(model.values()).T[:, 
                    nu.argsort(nu.int64(nu.array(model.keys())))],
                          data[:, 1])
        #if data has wave,flux and uncertanty axis
        elif data.shape[1] == 3:
            #from P Dosesquelles1, T M H Ha1, A Korichi1,
            #F Le Blanc2 and C M Petrache 2009
            N , chi = nnls(nu.array(model.values()).T[:, 
                    nu.argsort(nu.int64(nu.array(model.keys())))] /
                           nu.tile(data[:, 2], (bins, 1)).T,data[:,1]
                           / data[:,2]) 
        else:
            print 'wrong data shape'
            raise(KeyError)

    except RuntimeError:
        print "nnls error"
        N = nu.zeros([len(model.keys())])
        chi = nu.inf
    except ValueError:
        N = nu.zeros([len(model.keys())])
        chi = nu.inf
    #N[N==0] += 10 ** -6 #may cause problems in data[:,1]<10**-6 and not == 0
    #for numpy version 1.6
    try:
        return N, chi ** 2
    except OverflowError:
        return N, chi
    
def multivariate_student(mu,sigma,n):
    #samples from a multivariate student t distriburtion
    #with mean mu,sigma as covarence matrix, and n is degrees of freedom
    #as n->inf this goes to gaussian
    return mu + (nu.random.multivariate_normal([0] * len(mu), sigma) *
                 (n / nu.random.chisquare(n)) ** 0.5)

def nn_ls_fit(data,spect, max_bins=16, min_norm=10**-4):
    #not used?
    #uses non-negitive least squares to fit data
    #spect is libaray array
    #match wavelength of spectra to data change in to appropeate format
    model = {}
    for i in xrange(spect[0, :].shape[0]):
        if i == 0:
            model['wave'] = nu.copy(spect[:, i])
        else:
            model[str(i-1)] = nu.copy(spect[:, i])

    model = data_match_new(data, model, spect[0, :].shape[0] - 1)
    index = nu.int64(model.keys())
    
    #nnls fit handles uncertanty now 
    if data.shape[1] == 2:
        N,chi = nnls(nu.array(model.values()).T[:,
                nu.argsort(nu.int64(nu.array(model.keys())))],
                     data[:, 1])
    elif data.shape[1] == 3:
        N, chi = nnls(nu.array(model.values()).T[:, 
                nu.argsort(nu.int64(nu.array(model.keys())))] / nu.tile(data[:, 2], (bins, 1)).T, data[:, 1] / data[:, 2]) 
    N = N[index.argsort()]
    
    #check if above max number of binns
    if len(N[N > min_norm]) > max_bins:
        #remove the lowest normilization
        N_max_arg = nu.nonzero(N > min_norm)[0]
        N_max_arg = N_max_arg[N[N_max_arg].argsort()]
        #sort by norm value
        current = []
        for i in xrange(N_max_arg.shape[0] - 1, -1, -1):
            current.append(info[N_max_arg[i]])
            if len(current) == max_bins:
                break
        current = nu.array(current)
    else:
        current = info[N > min_norm]
    metal, age=[], []
    for i in current:
        metal.append(float(i[4: 10]))
        age.append(float(i[11: -5]))
    metal, age=nu.array(metal), nu.array(age)
    #check if any left
    return (metal[nu.argsort(age)], age[nu.argsort(age)], 
            N[N > min_norm][nu.argsort(age)])

def dict_size(dic):
    #returns total number of elements in dict
    size = 0
    for i in dic.keys():
        size += len(dic[i])

    return size

def dust(param, model):
    '''(ndarray, dict(ndarray)) -> ndarray
    
    Applies 2 componet dust model following charlot and fall 2000 on model
    returns same shape as model
    
    Keys of the model dict should be age of ssp.
    
    param is dust parameters [tau_bc, tau_ism]
    '''
    t_bc = 7.4771212547196626 #log10(.03*10**9)
    if nu.any(param[-2:] <= 0):
        return model
    #set all itterated varibles for speed
    bins = (param.shape[0] - 2) / 3
    tau_lam = (model['wave'] / 5500.) ** (-.7)
    T_ism = f_dust(param[1] * tau_lam)
    T_bc = f_dust(param[0] * tau_lam)
    for i in model.keys():
        if i == 'wave':
            continue 
        #choose which combo of dust models to use
        if float(i) <= t_bc: 
            #fdust*fbc
            model[i] *= T_ism * T_bc
        else:
            #fdust
            model[str(i)] *= T_ism
    return model

#@memoized
def f_dust(tau): 
    '''(ndarray) -> ndarray
    Dust extinction functin'''
    out = nu.zeros_like(tau)
    if nu.all(out == tau): #if all zeros
        return nu.ones_like(tau)
    if nu.any(tau < 0): #if has negitive tau
        out.fill(nu.inf)
        return out
    temp = tau[tau <= 1]
    out[tau <= 1] = (1 / (2. * temp) * (1 + (temp - 1) * nu.exp(-temp)
                                        - temp ** 2 * exp1(temp)))
    out[tau > 1] = nu.exp(-tau[tau > 1])
    return out

#@memoized
def gauss_kernel(resol,sigma,vel=0,h3=0,h4=0):
    '''sigma:     Dispersion velocity in km/s. Defaulted to 1.
    h3, h4:       Gauss-Hermite coefficients. Defaulted to 0.
    resol:        Size (in km/s) of each pixel of the output array
    vel:          velocity (in km/s) related to redshift (not used yet)
    '''
    if sigma < 1:
        sigma = 1.
    if sigma > 10**4:
        sigma = 10**4
   #make sure redshift is positve    
    c = 299792.458
    
    logl_shift = nu.log(1. + vel/c) / resol * c     #; shift in pixels
    logl_sigma = nu.log(1. + sigma/c) / resol * c  #; sigma in pixels
    N = nu.ceil( 5.*logl_sigma + abs(logl_shift))
    #N = nu.ceil(shift + 6. * sigma)
    x = nu.arange(2*N+1) - N
    y = (x-logl_shift)/logl_sigma
    
    #normal terms
    slitout = nu.exp(-y**2/2.) / logl_sigma / nu.sqrt(2.*nu.pi) 
    #hemite terms
    slitout = slitout*( 1.+ h3 * 2**.5 / (6.**.5) * (2 * y**3 - 3 * y) +
                        h4 / (24**.5) * (4 * y**4 - 12 * y**2 + 3))
    #normalize
    Sum = slitout.sum()
    if not Sum == 0:
       slitout /= Sum
    return slitout

def LOSVD(model, param, wave_range,resolution):
    '''(dict(ndarray), ndarray, ndarray) -> dict(ndarray)

    Convolves data with a gausian with dispersion of sigma, velocity, and hermite poly
    of h3 and h4

    model is dict of ssp with ages being the keys
    param is a 4x1 flat array with params of losvd [log10(sigma), v , h3, h4]
    wave_range is gives edge support for convolution
    resolution is the resolution of spectra in km/s
    '''
    #find wavelength range to do convolution for
    wave_range = nu.array(wave_range) - nu.asarray([100, -100])
    index = nu.searchsorted(model['wave'], [wave_range[0], wave_range[1]])
    wave = nu.copy(model['wave'][index[0]:index[1]])
    #convolve each spectra
    sum_spec = {'wave':wave}
    tparam = param.copy()
    tparam[0] = 10**tparam[0]
    for i in model.keys():
        if i == 'wave':
            continue
        sum_spec[i] = fft_conv(wave,model[i][index[0]:index[1]],resolution,tparam)
    return sum_spec

def fft_conv(x, y, vs, losvd_param):
    '''
    (wavelength,flux,velocity scale,losvd parameters) -> convolved flux
    Does LOSVD convolution by rebinning the wavelength into log scale and
    taking the fft to convolve it.
    '''
    #make logspaced
    lx,ly = logify(x,y)
    #lx,ly = x,y
    #make kernelresol,sigma,vel=0,h3=0,h4=0
    kernel = gauss_kernel(vs,losvd_param[0],losvd_param[2],losvd_param[3])
    #convolve fast
    ys = nu.convolve(ly,kernel,'same')
    #rebin to match input
    #return rebin_spec(lx,ys,x)
    return linearize(lx,ys)[1]
    
##not mine
def rebin_spec(wave, specin, wavnew):
    '''(inwave (ndarray),influx (ndarray),outwave(ndarray)-> outflux
    Correctly rebins spectra
    '''
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = nu.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
 
    return obs.binflux

#https://github.com/eteq/astropysics/blob/master/astropysics/spec.py

def getDx(x,mean=True):
    """
    get the spacing of the x-axis, which is always 1 element shorter than x
        
    if mean, returns the mean of the spacing
    """
    dx = nu.convolve(x,(1,-1),mode='valid')
    if mean:
        return dx.mean()
    else:
        return dx
        
def getDlogx(X,mean=True,logbase=10):
    """
    get the logarithmic spacing of the x-axis, which is always 1 element
    shorter than x
        
    if mean, returns the mean of the spacing
    """
    x = nu.log(X)/nu.log(logbase)
    dlogx = nu.convolve(x,(1,-1),mode='valid')
    if mean:
        return dlogx.mean()
    else:
        return dlogx
        
def isXMatched(x,other,tol=1e-10):
    """
    tests if the x-axis of this Spectrum matches that of another Spectrum
    or equal length array, with an average deviation less than tol
    """
    from operator import isSequenceType
    if isSequenceType(other):
        ox = other
    else:
        ox = other.x
            
    try:
        return nu.std(x - ox) < tol
    except (TypeError,ValueError):
        return False
    
def isLinear(x,eps=1e-10):
    """
    Determines if the x-spacing is linear (e.g. evenly spaced so that 
    dx ~ constant with a standard deviation of eps)
    """
    return nu.std(getDx(x,False)) < eps
        
def isLogarithmic(x,eps=1e-10):
    """
    Determines if the x-spacing is logarithmic (e.g. evenly spaced in 
    logarithmic bins so that dx ~ x to tolerance of eps)
    """
    return nu.std(getDlogx(x,False)) < eps
    
    #<----------------------Operations---------------------------->
    
def smooth(width=1,filtertype='gaussian',replace=True):
        """
        smooths the flux in this object by a filter of the given `filtertype`
        (can be either 'gaussian' or 'boxcar'/'uniform'). Note that `filtertype`
        can also be None, in which case a gaussian filter will be used if 
        width>0, or boxcar if width<0.
        
        if replace is True, the flux in this object is replaced by the smoothed
        flux and the error is smoothed in the same fashion
        
        width is in pixels, either as sigma for gaussian or half-width 
        for boxcar
        
        returns smoothedflux,smoothederr
        """
        import scipy.ndimage as ndi 
        
        if filtertype is None:
            if width > 0:
                filtertype = 'gaussian'
            else:
                filtertype = 'boxcar'
                width = -1*width
        
        if filtertype == 'gaussian':
            filter = ndi.gaussian_filter1d
            err = self._err
        elif filtertype == 'boxcar' or type == 'uniform':
            filter = ndi.uniform_filter1d
            width = 2*width
            err = self._err.copy()
            err[~np.isfinite(err)] = 0

        else:
            raise ValueError('unrecognized filter type %s'%filtertype)
        
        smoothedflux = filter(self._flux,width)
        smoothederr = filter(err,width)
        
        if replace:
            self.flux = smoothedflux
            self.err = smoothederr
        
        return smoothedflux,smoothederr

def linearize(wave,flux):
        """
        (wavelength array, flux array) -> resampled (wave,flux)
        Convinience function for resampling to an equally-spaced linear x-axis
            
        """
        newx = nu.linspace(wave.min(),wave.max(),len(wave))
        newflux = rebin_spec(wave, flux, newx)
        return newx,newflux
    
def logify(wave,flux):
        """
        (wavelength array, flux array) -> resampled (wave,flux)
        convinience function for resampling to an x-axis that is evenly spaced
        in logarithmic bins.  Note that lower and upper are the x-axis values 
        themselves, NOT log(xvalue)
        """
        newx = nu.logspace(nu.log10(wave.min()),nu.log10(wave.max()),len(wave))
        newflux = rebin_spec(wave, flux, newx)
        return newx,newflux

def fitContinuum(self,model=None,weighted=False,evaluate=False,
                          interactive=False,**kwargs):
        """
        this method computes a continuum fit to the spectrum using a model
        from astropysics.models (list_models will give all options) or
        an callable with a fitData(x,y) function
        
        if model is None, the existing model will be used if present, 
        or if there is None, the default is 'uniformknotspline'.  Otherwise,
        it may be any acceptable model (see :func:`models.get_model_class`)
        
        kwargs are passed into the constructor for the model
        
        if weighted, the inverse variance will be used as weights to the 
        continuum fit
        
        if interactive is True, the fitgui interface will be displayed to 
        tune the continuum fit
        
        the fitted model is assigned to self.continuum or evaluated at the
        spectrum points if evaluate is True and the results are set to 
        self.continuum
        """
        
        if model is None and self.continuum is None:
            model = 'uniformknotspline'
        
        #for the default, choose a reasonable number of knots
        if model == 'uniformknotspline' and 'nknots' not in kwargs:
            kwargs['nknots'] = 4
        
        if model is None and self.continuum is not None:
            model = self.continuum
            if not interactive:
                model.fitData(self.x,self.flux,weights=(self.ivar if weighted else None))
        else:
            if isinstance(model,basestring):
                from .models import get_model_class
                model = get_model_class(model)(**kwargs)
        
            if not (callable(model) and hasattr(model,'fitData')):
                raise ValueError('provided model object cannot fit data')
            
            model.fitData(self.x,self.flux,weights=(self.ivar if weighted else None))
        
        if interactive:
            from pymodelfit.fitgui import FitGui
            
            if interactive == 'reuse' and hasattr(self,'_contfg'):
                fg = self._contfg
            else:
                fg = FitGui(self.x,self.flux,model=model,weights=(self.ivar if weighted else None))
                fg.plot.plots['data'][0].marker = 'dot'
                fg.plot.plots['data'][0].marker_size = 2
                fg.plot.plots['model'][0].line_style = 'solid'
                
            if fg.configure_traits(kind='livemodal'):
                model = fg.tmodel.model
            else:
                model = None
                
            if interactive == 'reuse':
                self._contfg = fg
            elif hasattr(self,'_contfg'):
                del self._contfg
                
        if model is not None:    
            self.continuum = model(self.x) if evaluate else model
        
def subtractContinuum(self):
        """
        Subtract the continuum from the flux
        """
        if hasattr(self,'_contop'):
            raise ValueError('%s already performed on continuum'%self._contop)
        
        if self.continuum is None:
            raise ValueError('no continuum defined')
        elif callable(self.continuum):
            cont = self.continuum(self.x)
        else:
            cont = self.continuum
            
        self.flux = self.flux - cont
        self._contop = 'subtraction'
            
def normalizeByContinuum(self):
        """
        Divide by the flux by the continuum
        """
        if hasattr(self,'_contop'):
            raise ValueError('%s already performed on continuum'%self._contop)
        
        if self.continuum is None:
            raise ValueError('no continuum defined')
        elif callable(self.continuum):
            cont = self.continuum(self.x)
        else:
            cont = self.continuum
            
        self.flux = self.flux/cont
        self._contop = 'normalize'
        
def rejectOutliersFromContinuum(self,sig=3,iters=1,center='median',savecont=False):
        """
        rejects outliers and returns the resulting continuum. see 
        `utils.sigma_clip` for arguments
        
        returns a pair of maksed arrays xmasked,contmasked
        
        if savecont is True, the outlier-rejected value will be saved as
        the new continuum
        """
        from .utils import sigma_clip
        
        if self.continuum is None:
            raise ValueError('no continuum defined')
        elif callable(self.continuum):
            cont = self.continuum(self.x)
        else:
            cont = self.continuum
            
        contma = sigma_clip(cont,sig=sig,iters=iters,center=center,maout='copy')
        xma = np.ma.MaskedArray(self.x,contma.mask,copy=True)
        
        if savecont:
            self.continuum = contma
        
        return xma,contma
            
def revertContinuum(self):
        """
        Revert to flux before continuum subtraction
        """
        if self.continuum is None:
            raise ValueError('no continuum defined')
        elif callable(self.continuum):
            cont = self.continuum(self.x)
        else:
            cont = self.continuum
            
        if hasattr(self,'_contop'):
            if self._contop == 'subtraction':
                self.flux = self.flux+cont
            elif self._contop == 'normalize':
                self.flux = self.flux*cont
            else:
                raise RuntimeError('invalid continuum operation')
            del self._contop
        else:
            raise ValueError('no continuum action performed')

def plot(self,fmt=None,ploterrs=.1,plotcontinuum=True,smoothing=None,
                  step=True,clf=True,colors=('b','g','r','k'),restframe=True,
                  xrng=None,**kwargs):
        """
        Use :mod:`matplotlib` to plot the :class:`Spectrum` object. The
        resulting plot shows the flux, error (if present), and continuum (if
        present).
        
        If `step` is True, the plot will be a step plot instead of a line plot.
        
        `smoothing` is passed into the :meth:`Spectrum.smooth` method - see that
        method for details.
        
        `colors` should be a 3-tuple that applies to
        (spectrum,error,invaliderror,continuum) and kwargs go into spectrum and
        error plots.
        
        If `restframe` is True, the x-axis is offset to the rest frame.
        
        If `ploterrs` or `plotcontinuum` is a number, the plot will be scaled so
        that the mean value matches the mean of the spectrum times the numeric
        value. If either are True, the scaling will match the actual value. If
        False, the plots will not be shown.
        
        `xrng` can specify the range of x-values to plot (lowerx,upperx), or can
        be None to plot the whole spectrum.
        
        kwargs are passed into either the :func:`matplotlib.pyplot.plot` or
        :func:`matplotlib.pyplot.step` function.
        """
        
        import matplotlib.pyplot as plt
        
        if step:
            kwargs.setdefault('where','mid')
        
        if smoothing:
            x,(y,e) = self.x0 if restframe else self.x,self.smooth(smoothing,filtertype=None,replace=False)
        else:
            x,y,e = self.x0 if restframe else self.x,self.flux,self.err
            
        if len(x)==3:
            dx1 = x[1]-x[0]
            dx2 = x[2]-x[1]
            x = np.array((x[0]-dx1/2,x[0]+dx1/2,x[1]+dx2/2,x[2]+dx2/2))
            y = np.array((y[0],y[0],y[1],y[2]))
            e = np.array((e[0],e[0],e[1],e[2]))
        elif len(x)==2:
            dx = x[1]-x[0]
            x = np.array((x[0]-dx/2,x[0]+dx/2,x[1]+dx/2))
            y = np.array((y[0],y[0],y[1]))
            e = np.array((e[0],e[0],e[1]))
        elif len(x)==1:
            x = np.array((0,2*x[0]))
            y = np.array((y[0],y[0]))
            e = np.array((e[0],e[0]))
            
        if xrng is not None:
            xl,xu = xrng
            if xl>xu:
                xl,xu = xu,xl
                
            msk = (xl<x)&(x<xu)
            x = x[msk]
            y = y[msk]
            e = e[msk]
            
        if clf:
            plt.clf()
            
        kwargs['c'] = colors[0]
        if fmt is None:
            if step:
                res = [plt.step(x,y,**kwargs)]
            else:
                res = [plt.plot(x,y,**kwargs)]
        else:
            if step:
                res = [plt.step(x,y,fmt,**kwargs)]
            else:
                res = [plt.plot(x,y,fmt,**kwargs)]
            
        if ploterrs and np.any(e):
            from operator import isMappingType
            if isMappingType(ploterrs):
                ploterrs = ploterrs.copy()
            
            m = (e < np.max(y)*2) & np.isfinite(e)
            
            if isMappingType(ploterrs) and 'scale' in ploterrs:
                scale = ploterrs.pop('scale')
            elif np.isscalar(ploterrs):
                scale = float(ploterrs)*np.mean(y)/np.mean(e[m])
            elif ploterrs is True:
                scale = 1
            else:
                scale = .1*np.mean(y)/np.mean(e[m])
            
            if not isMappingType(ploterrs):
                ploterrs = {}
            kwargs.update(ploterrs)
            kwargs.setdefault('ls','-')
            
            
            if np.sum(m) > 0:
                kwargs['c'] = colors[1]
                if step:
                    res.append(plt.step(x[m],scale*e[m],**kwargs))
                else:
                    res.append(plt.plot(x[m],scale*e[m],**kwargs))
            if np.sum(~m) > 0:
                if step:
                    res.append(plt.step(x[~m],scale*np.mean(e[m] if np.sum(m)>0 else y)*np.ones(sum(~m)),'*',lw=0,mew=0,color=colors[2]))
                else:
                    res.append(plt.plot(x[~m],scale*np.mean(e[m] if np.sum(m)>0 else y)*np.ones(sum(~m)),'*',lw=0,mew=0,color=colors[2]))
                
        if plotcontinuum and self.continuum is not None:
            if callable(self.continuum):
                cont = self.continuum(self.x)
            else:
                cont = self.continuum
            
            if plotcontinuum is True:
                scale = 1
            elif np.isscalar(plotcontinuum):
                scale = float(plotcontinuum)*np.mean(y)/np.mean(cont)
                
            kwargs['c'] = colors[3]
            kwargs['ls'] =  '--'
            if step:
                res.append(plt.step(self.x,scale*cont,**kwargs))
            else:
                res.append(plt.plot(self.x,scale*cont,**kwargs))
                
                
        plt.xlim(np.min(x),np.max(x))
        
        xl=self.unit
        xl=xl.replace('wavelength','\\lambda')
        xl=xl.replace('frequency','\\nu')
        xl=xl.replace('energy','E')
        xl=xl.replace('angstrom','\\AA')
        xl=xl.replace('micron','\\mu m')
        xl=tuple(xl.split('-'))
        plt.xlabel('$%s/{\\rm %s}$'%xl)
        
        plt.ylabel('$ {\\rm Flux}/({\\rm erg}\\, {\\rm s}^{-1}\\, {\\rm cm}^{-2} {\\rm %s}^{-1})$'%xl[1])
            
        return res


def weight_wavelegth(data, SSP, nlambda, age_to_weight=None):
    '''Does PCA to find the wavelengths that contribute the most to an age
    data -spectrum to fit
    SSP -EZgal wrapper of spectra lib used in fitting
    nlambda - number of wavelength user wants to use
    age_to_weigth - age user thinks spectra should fall in (like the prior)
    [lower,upper] yrs
    returns:
    ndarray (n,) of [0:1] weight for each wavelength point
    '''
    if age_to_weight is None:
        #get ages and metals from SSP
        age,Z,lam = nu.meshgrid(SSP[0].ages,nu.sort(nu.float64(SSP.meta_data['met'])),data[:,0])
    else:
        t_age = nu.linspace(age_to_weight[0],age_to_weight[1],69)
        age,Z,lam = nu.meshgrid(t_age,nu.sort(nu.float64(SSP.meta_data['met'])),data[:,0])
    ssps = nu.zeros_like(age)
    #get ssp in same order as above
    k = 0
    #norm close to 4020 A
    index_4020 = nu.searchsorted(data[:,0],[4020])
    for i,j in zip(age[:,:,0].ravel(),Z[:,:,0].ravel()):
        index = nu.unravel_index(k,ssps.shape[:2])
        try:
            temp_ssp = SSP.get_sed(i+1,j,'yrs')
        except ValueError:
             temp_ssp = SSP.get_sed(i-1,j,'yrs')
        #make wavelengths match data
        ssps[index] = rebin_spec(SSP.sed_ls, temp_ssp, data[:,0])
        n = data[index_4020,1]/ssps[index][index_4020]
        ssps[index] = n * ssps[index]
        k+=1

        
    #get pca
    pca = PCA(10)
    #age_vs wavelength
    temp = nu.sum(ssps,axis=0)
    s=pca.fit_transform(temp.T)
    #metals vs wavelentght
    #plot
    lab.scatter(s[:,0],s[:,1],c=data[:,0])
    k = 45
    cen, i = None,0
    while True:
        if cen is None:
            cen, lables = kmeans2(X, k)
            i += 1
            continue
        else:
            t_cen, lables = kmeans2(X, cen,iter=30, minit='matrix')
        #check is centroids have changed
        if (nu.allclose(cen,t_cen) and i > 4) or i > 100:
            break
        else:
            i += 1
            cen = t_cen.copy()

    #make pdist array with metals vs wavelength
    met_vs_wave = squareform(pdist(ssps.sum(axis=1)))
