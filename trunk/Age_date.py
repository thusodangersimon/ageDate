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
""" Utilites used for spectral fitting"""

import numpy as nu
import os
import sys
from multiprocessing import *
from interp_func import *
from spectra_func import *
from scipy.optimize import nnls
from scipy.optimize.minpack import leastsq
from scipy.optimize import fmin_l_bfgs_b as fmin_bound
from scipy.special import exp1 
from scipy.integrate import simps
#from scipy import weave
#from scipy.signal import fftconvolve
import time as Time
import boundary as bound
#import Age_MCMC as mc
#import Age_RJMCMC as rjmc
from losvd_convolve import convolve
#import Age_hybrid as hy

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

   def __call__(self, *args):
      #works different for different functions
      if self.func.__name__ == 'gauss1':
         arg = str(args[1])
         if arg in self.cache:
            return self.cache[arg]
         else:
            value = self.func(*args)
            self.cache[arg] = value
            return value
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
global lib_path,spect
lib_path = '/home/thuso/Phd/Spectra_lib/'
spect,info = None,None
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
        if index <= 0 or index >= len(age_unq): #out of bounds
            print 'age param is out of bounds'
            raise
        age.append(age_unq[index])
        age.append(age_unq[index - 1])
    age *= 2
    return metal, age,line

def get_model_fit_opt(param, lib_vals, age_unq, metal_unq, bins, 
						spect):
    #does dirty work to make spectra models
    #search age_unq and metal_unq to find closet box spectra and interps
    #does multi componets spectra and fits optimal normalization
    out = {}
    for ii in xrange(bins):
        temp_param = param[ii * 3: ii * 3 + 2]
        metal, age, line = find_az_box(temp_param, age_unq, metal_unq)
        closest = []
        #check to see if on a lib spectra or on a line
        if line == 'age': #run 1 d interp along metal only
            metal = nu.array([metal[0], metal[-1]])
            metal.sort()
            age = age[0]
            #find spectra
            for i in 10 ** metal:
                index = nu.nonzero(nu.logical_and(lib_vals[0][:,0] ==
                                                  i, lib_vals[0][:,1]
                                                  == age))[0]
                if len(index)<1: #if not in lib_vals return dummy array
                    out[str(ii)] = spect[:,0] + nu.inf
                    interp = False
                    break
                else:
                    closest.append(spect[:,index[0] + 1])
                    interp = True
            if interp:
                out[str(ii)] = linear_interpolation(10 ** metal,
                                                    closest, 10 ** 
                                                    temp_param[0])
        elif line == 'metal': #run 1 d interp along age only
            age = nu.array([age[0], age[-1]])
            age.sort()
            metal = metal[0]
            #find spectra
            
            for i in age:
                index = nu.nonzero(nu.logical_and(lib_vals[0][:,1] ==
                                                  i, lib_vals[0][:,0]
                                                  == 10 ** metal))[0]
                
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
            index=nu.nonzero(nu.logical_and(lib_vals[0][:,0] == 10 **
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
                                                  10 ** metal[i]))[0]
                if len(index)<1: #if not in lib_vals return dummy array
                    out[str(ii)] = spect[:, 0] + nu.inf
                    interp = False
                    break
                else:
                    closest.append(spect[:, index[0] + 1])
                    interp = True
            if interp:
                out[str(ii)] = bilinear_interpolation(10 ** metal, age
                                                      , closest,
                                                      10 ** 
                                                      temp_param[0],
                                                      temp_param[1])
    #give wavelength axis
    out['wave'] = nu.copy(spect[:, 0])

   #exit program
    return out

def make_burst(length, t, metal, metal_unq, age_unq, spect, lib_vals):
	'''(float, float,float, ndarray(float),ndarray(float)
		ndarray(float) tuple(ndarray(floats),ndarray(str))) -> ndarray(float)
		Turns SSP into busrt of constant stellar formation and of length dt at
		age t for a const metalicity 10**(t-9) - length/2 to 10**(t-9) + length/2.
		All terms are logrythmic.
	'''
	#lib_vals[0][:,0] = 10**nu.log10(lib_vals[0][:,0])
	assert t >= age_unq.min() and t <= age_unq.max(), 'Age not in range'
	assert metal >= metal_unq.min() and metal <= metal_unq.max(), 'Metalicity not in range'
	#get all ssp's with correct age range and metalicity
	#min age range
	if t - length/2. < age_unq.min():
		t_min = age_unq.min() + 0.
	else:
		t_min = t- length/2.
	#max age range
	if 	t + length/2. > age_unq.max():
		t_max = age_unq.max() + 0.
	else:
		t_max = t + length/2.
	index = nu.searchsorted(age_unq, [t_min,t_max])
	ages = age_unq[index[0]:index[1]]
	#get SSP's
	
	#handel situation where stepsize is small
	if len(ages) < 2:
		pass
	temp_param = []
	for i in ages:
		temp_param.append([metal, i ,1.])
	ssp = get_model_fit_opt(nu.ravel(temp_param), lib_vals, age_unq, metal_unq, 
							len(ages), spect)
	#sort for intergration
	inters = []
	#integrate ssp's
	ssp.pop('wave')
	for i in nu.sort(nu.int64(ssp.keys())):
		if not nu.any(nu.isfinite(ssp[str(i)])):
			continue
		inters.append(ssp[str(i)])
	return simps(inters, ages, axis=0)/float(len(ages))
	#return new ssp
	
		
def random_permute(seed):
    #does random sequences to produice a random seed for parallel programs
    ##middle squared method
    seed = str(seed**2)
    while len(seed) < 7:
        seed=str(int(seed)**2)
    #do more randomization
    ##multiply with carry
    a,b,c = int(seed[-1]), 2**32, int(seed[-3])
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

def data_match(data, model, bins, keep_wave=False):
   #makes sure data and model have same wavelength range 
   #and points but with a dictionary
   #assumes spect_lib is longer than data
   ####add resolution downgrading!!!
   out = {}
    #if they have the same x-axis
   if nu.all(model['wave'] == data[:, 0]):
      for i in xrange(bins):
         out[str(i)] = model[str(i)]
   else: #not same shape, interp 
      for i in xrange(bins):
         out[str(i)] = spectra_lin_interp(model['wave'],
                                          model[str(i)], data[:,0])
   if keep_wave:
      out['wave'] = nu.copy(data[:,0])
   return out

def redshift(wave, redshift):
   #changes redshift of models
   return wave * (1. + redshift)

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

def nn_ls_fit(data, max_bins=16, min_norm=10**-4, spect=spect):
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
   '''(ndarray or list, dict(ndarray)) -> ndarray

   Applies 2 componet dust model following charlot and fall 2000 on model
   returns same shape as model
   '''
   t_bc = 7.4771212547196626 #log10(.03*10**9)
   if nu.any(param[-2:] <= 0):
      return model
    #set all itterated varibles for speed
   bins = (param.shape[0] - 2) / 3
   tau_lam = (model['wave'] / 5500.) ** (-.7)
   T_ism = f_dust(param[-2] * tau_lam)
   T_bc = f_dust(param[-1] * tau_lam)
   for i in xrange(bins): 
      #choose which combo of dust models to use
      if param[3 * i + 1] <= t_bc: 
         #fdust*fbc
         model[str(i)] *= T_ism * T_bc
      else:
         #fdust
         model[str(i)] *= T_ism
   return model

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


def gauss_kernel(velscale,sigma=1,h3=0,h4=0):
    '''sigma:        Dispersion velocity in km/s. Defaulted to 1.
    h3, h4:       Gauss-Hermite coefficients. Defaulted to 0.
    resol:        Size (in km/s) of each pixel of the output array
    '''
    if sigma < 1:
        sigma = 1.
    if sigma > 10**4:
        sigma = 10**4
   #make sure redshift is positve    
    c = 299792.458
    #v_red = (c *Z * (Z + 2.)) / (Z**2. + 2 * Z + 2)
    #logl_shift = nu.log(1. + Z) / velscale * c     #; shift in pixels
    logl_sigma = nu.log(1. + sigma/c) / velscale * c  #; sigma in pixels
    #shift = v_red / float(velscale)
    #sigma = sigma / float(velscale)
    N = nu.ceil( 5.*logl_sigma)
    #N = nu.ceil(shift + 6. * sigma)
    x = nu.arange(2*N+1) - N
    y = (x)/logl_sigma
    #y = (x - shift) / sigma
    #normal terms
    slitout = nu.exp(-y**2/2.) / logl_sigma / nu.sqrt(2.*nu.pi) 
    #hemite terms
    slitout = slitout*( 1.+ h3 * 2**.5 / (6.**.5) * (2 * y**3 - 3 * y) +
                        h4 / (24**.5) * (4 * y**4 - 12 * y**2 + 3))
    #normalize
    if not slitout.sum() == 0:
       slitout /= nu.sum(slitout)
    return slitout

def LOSVD(model,param,wave_range):
    '''convolves data with a gausian with dispersion of sigma, and hermite poly
    of h3 and h4'''
    #unlog sigma
    #import pylab as lab
    tparam = param.copy()
    tparam[0] = 10**tparam[0]
    tmodel = model.copy()
    #resample specturm so resolution is same at all places
    wave_range = nu.array(wave_range)/(1. + tparam[1]) - [100, -100]
    index = nu.searchsorted(model['wave'], [wave_range[0], wave_range[1]])
    try:
       wave_diff = nu.diff(model['wave'][index[0]:index[1]]).min()
    except ValueError:
       #model and data range are off
       return tmodel
    wave = nu.arange(model['wave'][index[0]], model['wave'][index[1]], wave_diff)
    #print wave.min(),wave.max(),redshift(wave[[0,-1]],tparam[1])
    #convolve individual spectra
    for i in model.keys():
        if i == 'wave':
            continue
        #python convlve
        model[i] = spectra_lin_interp(model['wave'], model[i], wave)
        tmodel[i] = convolve_python_fast(wave, nu.ascontiguousarray(model[i]), tparam)
        #c++ convolve
        #tmodel[i] = spectra_lin_interp(tmodel['wave'], tmodel[i], wave)
        #convolve(wave, tmodel[i], tparam ,tmodel[i])
    #apply redshift
    tmodel['wave'] = redshift(wave,tparam[1])
    #uncertanty convolve
    #if data.shape[1] == 3:
    #    out[:,2] = nu.sqrt(nu.convolve(kernel**2, data[:,2]**2,'same'))

    return tmodel

def convolve_python_fast(x, y, losvd_param, option='rebin'):
   '''convolve(array, kernel)
   does convoluton useing input kernel. in a faster way '''
   diff_wave = nu.mean(nu.diff(x))
   Len_data = len(x)
   #ys = nu.repeat(y,Len_data).reshape(Len_data,Len_data).T
   #make kernels
   Kernals = map(gauss1,[diff_wave]*Len_data, x,[losvd_param[0]]*Len_data,[losvd_param[2]]*Len_data, [losvd_param[3]]*Len_data)
   #make an array with same size as y 
   Kernals = map(make_kernel_array, Kernals,[Len_data]*Len_data,range(Len_data))
   #convovle
   ys = nu.array(map(lambda x,y: nu.sum(x*y), Kernals, [y]*Len_data))
   return ys
   
def make_kernel_array(Kernel, Length, i):
   '''Puts arrays from Kernels into matrix array. If kernel is
   longer than array with return error'''
   kern_len = len(Kernel)
   middle = (kern_len - 1)/2
   out_array = nu.zeros(Length)
   index = nu.arange(kern_len) - middle + i
   nu.put(out_array,index,Kernel,mode='clip')
   return out_array

def convolve_python(x, y, losvd_param, option='rebin'):
   '''convolve(array, kernel)
	 does convoluton useing input kernel '''
   #alocate parameters
   #start covloution
   #x, y =  data[:,0], data[:,1]
   xs, ys = nu.zeros_like(x), nu.zeros_like(y)
   diff_wave = nu.mean(nu.diff(x))
   Len_data = len(x)
   #t=[]
   for  i in xrange(Len_data):   
      kernel = gauss1(diff_wave, x[i], losvd_param[0], losvd_param[2], losvd_param[3])
      kernel[kernel < 0] = 0
      Len = len(kernel)  
      #check if kernel is longer than input array
      if Len > Len_data:
         ys.fill(nu.nan)
         return ys
      m2 = i + (Len - 1)/2 + 1
      m1 = i - (Len - 1)/2 
      if m1 < 0:
         m1 = 0
         i1 = (Len - 1)/2 - i
         i2 = Len 
         k = kernel[i1:i2] / kernel[i1:i2].sum()
      elif m2 > Len_data - 1:
         m2 = Len_data - 1
         i1 = 0
         i2 = Len - ((Len - 1)/2 - m2 + i) - 1
         k = kernel[i1:i2] / kernel[i1:i2].sum()
      else:
         #i1, i2 = 0, Len - 1
         k = kernel
      u = x[m1:m2]
      g = y[m1:m2] * k
      ys[i] = nu.trapz(g, u, dx=diff_wave)
      xs[i] = x[i]
      #t.append((min(k),x[i]))
   #t=nu.array(t)
      #t.append(len(k))
   return  ys

@memoized
def gauss1(diff_wave,  wave_current,  sigma,  h3,  h4):
   '''inline gauss(nu.ndarray diff_wave, float wave_current float sigma, float h3, float h4)
	Returns value of gaussian-hermite function normalized to area = 1'''
   c = 299792.458
   vel_scale = diff_wave / wave_current * c
   logl_sigma = nu.log(1. + sigma/c) / vel_scale * c
   N = nu.ceil( 5.*logl_sigma)
   x = nu.arange(2*N+1) - N
   y = x / logl_sigma
   slitout = nu.exp(-y**2/2.) / logl_sigma #/ nu.sqrt(2.*nu.pi)
   slitout *= ( 1.+ h3 * 2**.5 / (6.**.5) * (2 * y**3 - 3 * y) + 
                h4 / (24**.5) * (4 * y**4 - 12 * y**2 + 3))
   if not slitout.sum() == 0:
      slitout /= nu.sum(slitout)
   return slitout      

#####classes############# 
class MC_func:
    '''Does everything the is required for and Age_date mcmc sampler'''
    def __init__(self, data,option='rjmc', burnin=10**4, itter=5*10**5,
                 cpus=cpu_count(), use_dust=True, use_lovsd=True,spec_lib='BC03.splib',bins=None):
        #match spectra for use in class /Turned off due to redshift fitting
        global spect,info
       #load spectra or use default
        if not nu.any(spect): 
          spect,info = load_spec_lib(lib_path,spec_lib)
        self.spect = nu.copy(spect)
        self.data = nu.copy(data)
        #normalized so area under curve is 1 to keep chi 
        #values resonalble
        #need to properly handel uncertanty
        self.norms = self.area_under_curve(data) * 10 ** -5 #need to turn off
        self.data[:,1] = self.data[:, 1] / self.norms

        #initalize bound varables
        lib_vals = get_fitting_info(lib_path)
        #to keep roundoff error constistant
        lib_vals[0][:,0] = nu.log10(lib_vals[0][:, 0]) 
        metal_unq = nu.unique(lib_vals[0][:, 0])
        #get boundary of parameters
        self.hull = bound.find_boundary(lib_vals[0])
        lib_vals[0][:,0] = 10**lib_vals[0][:,0]
        age_unq = nu.unique(lib_vals[0][:, 1])
        self._lib_vals = lib_vals
        self._age_unq = age_unq
        self._metal_unq = metal_unq
        self._option = option
        self._cpus = cpus
        self.bins = bins
        self._burnin = burnin
        self._iter = itter
        #set prior info for age, and metalicity
        self.metal_bound = nu.array([metal_unq.min(),metal_unq.max()])
        #dust options
        self._dust = use_dust
        self.dust_bound = nu.array([0., 4.])
        #line of sight velocity despersion stuff
        self._losvd = use_lovsd
        self._velocityscale = (nu.diff(self.data[:,0]).mean()
                               / self.data[:,0].mean() * 299792.458)

    def run(self,verbose=True):
        'starts run of Age_date using configuation files'
        option = Value('b',True)
        option.cpu_tot = self._cpus
        option.iter = Value('i',True)
        option.chibest = Value('d',nu.inf)
        option.parambest = Array('d',nu.ones(self._k_max * 3 + 2 + 4) + nu.nan)
        #start multiprocess need to make work with mpi
        work=[]
        q_talk,q_final=Queue(),Queue()

        for ii in range(self._cpus):
            work.append(Process(target=self.samp,
                                args=(self.send_class, self._burnin, self._k_max,
                                      option, ii, q_talk, q_final,)))
            work[-1].start()
        while (option.iter.value <= self._iter + self._burnin * self._cpus 
               and option.value):  
            Time.sleep(5)
            sys.stdout.flush()
            print '%2.2f percent done' %((float(option.iter.value)/
                                          (self._iter + self._burnin
                                           * self._cpus))*100.)

        #put in convergence diagnosis
        option.value=False
        #wait for proceses to finish
        count=0
        temp=[]
        while count<self._cpus:
            count+=1
            try:
                temp.append(q_final.get(timeout=5))
            except:
                print 'having trouble recieving data from queue please wait'
                
        if not temp:
            print 'Recived no data from processes, exiting'
            return False,False,False
        for i in work:
            i.terminate()
        #figure out how to handel output
        if type(temp[0][0]) is dict:
            #probably rjmcmc output
            self.param, self.chi, self.bayes = rjmc.dic_data(temp, self._burnin)
            return self.param, self.chi, self.bayes
        elif type(temp[0][0]) is nu.ndarray:
            #probably regurlar mcmc output
            self.param, self.chi = mc.np_data(temp, self.send_class._bins)
            return self.param, self.chi 

    def autosetup(self,bins=None):
        'auto sets up class so ready for running'
        #choose sampler
        self.sampler(self._option)
        '''#choose prior/boundary(not 100% ready yet)
        self.prior = self.uniform_prior
        #choose proposal function (only multivariate norm now)
        self.proposal = nu.random.multivariate_normal
        #choose -2log liklihood function (normal dist aka chi squared)
        self.log_lik =  self.func_N_norm
        '''
        #make into send class
        if self._option == 'mcmc':
            if not bins:
                bins = raw_input('How many bins do you want to use? ')
            
            self.send_class = self.send_functions(self.samp,self.uniform_prior,
                                                  nu.random.multivariate_normal,
                                                  self.func_N_norm)
            self.send_class._bins = int(bins)
            #fuction to tell mcmc how to bin chains
            self.send_class.bin = self.age_bound(self._age_unq,int(bins))
            #dummy varible
            self._k_max = int(bins)
            
        elif self._option == 'rjmc':
            self.send_class = self.send_functions(self.samp,self.uniform_prior,
                                                  nu.random.multivariate_normal,
                                                  self.func_N_norm)  
        else:
            print 'Option not set up yet, set up manually.'

        #dust and losvd
        self.send_class._dust = self._dust
        self.send_class._losvd = self._losvd
        self.send_class._velocityscale = self._velocityscale
        self.send_class.norms = self.norms

    def sampler(self,option):
        '''puts samplers for use'''
        if option == 'rjmc':
            self.samp = rjmc.rjmcmc
            self._k_max = 16
            self.age_bound = lambda age_unq, bins: nu.array(
                [age_unq.min(), age_unq.max()])
        elif option == 'mcmc':
            self.samp = mc.MCMC_SA
            #if want age to be binned only works
            if self.bins == 'linear':
                self.age_bound = lambda age_unq, bins: (
                    nu.log10(nu.linspace(10 ** age_unq.min(), 
                                         10 ** age_unq.max(), bins + 1)))
            else : #log
                self.age_bound = lambda age_unq, bins: nu.linspace(
                    age_unq.min(), age_unq.max(), bins + 1)

        elif option == 'rjmpi':
            raise FutureWarning('Not yet working')
            self.samp = rjmc.rjmpi
            self.age_bound = lambda age_unq, bins: nu.array(
                [age_unq.min(), age_unq.max()])

        elif option == 'mcmpi':
            raise FutureWarning('Not yet working')
            self.samp = mc.mcmpi
        else:
            print 'Wrong option, use "rjmc, mcmc, rjmpi, mcmpi"'
    
    def use_old_run(self,param=None,chi=None):
        '''interactve way of using old run for constraining bins, and/or
        making prior'''
        import pylab as lab
        lab.ion()
        #check if has input
        if not (nu.any(param) and nu.any(chi)):
            if not (nu.any(self.param) and nu.any(self.chi)):
                raise ValueError('No prior run avalible, please do' + 
                                 'run or input one')
        else:
            self.param, self.chi = param, chi
        #check to see if ndarray
        if type(param) is nu.ndarray:
            #turn into dict
            #temp_param = {}
            #temp_chi = {}
            bins = str((param.shape[1] - 2) / 3)
            self.param = {bins:nu.copy(self.param)}
            self.chi = {bins:nu.copy(self.chi)}

        #do hard part of deciding where to put bins and stuff
        #assume gaussian parts
        import check_mc_output as chk
        guess = chk.plot_SFR_age(self.param, self.chi, False)
        #guess best fit
        best_fit = ['0', 1., nu.inf]
        for i in self.param.keys():
            if len(self.param[i]) == 0:
                continue
            if best_fit[2] >= self.chi[i].min(): #chi must be less
                if len(self.param[i])/best_fit[1] > 1/3.:
                    if int(i) > int(best_fit[0]): #if has more params
                        #penalize it and see if still better
                        if (self.chi[i].min() * 
                            2.**((3*(int(i) -int(best_fit[0])))) < 
                            best_fit[2]): 
                            best_fit = [i,len(self.param[i]),self.chi[i].min()]
                    else: 
                        best_fit = [i,len(self.param[i]),self.chi[i].min()]
        print 'Using best fit param with %s bins and min chi %f' %(best_fit[0], 
                                                                   best_fit[2])

        if best_fit[0] in set(guess.keys()): 
            #use points from guess
            temp_age = self.param[best_fit[0]]
            temp_age = temp_age[:,range(1,int(best_fit[0])*3,3)].ravel()
            age_bound = [temp_age.min()]
            temp = guess[best_fit[0]]['age']
            temp = temp[temp[:,0].argsort()]
            for i in range(1,temp.shape[0]):
                k = 1
                while (nu.sum(temp[i-1:i+1,[0,2]][0] * nu.array([1,k])) < 
                       -nu.diff(temp[i-1:i+1,[0,2]][1] * nu.array([1,k]))):
                    k += 1
                age_bound.append(nu.mean([nu.sum(temp[i-1:i+1,[0,2]][0] * 
                                                 nu.array([1,k])),
                                          -nu.diff(temp[i-1:i+1,[0,2]][1] * 
                                                   nu.array([1,k]))]))  
            age_bound.append(temp_age.max())
            fig = lab.figure()
            plt1 = fig.add_subplot(211) #histogram with bins
            plt2 = fig.add_subplot(212) #residuals from best fit
            #histogram plot
            plt1.hist(temp_age, 200, normed=True)
            for i in age_bound:
                plt1.plot(nu.tile(i,2),[0,plt1.get_ylim()[1]],'r')
            #make sure no spaces between bins
            
                      
        else: #have user put them in or make them automatically
            print 'No guess fit, run with more iterations'
            raise FutureWarning('not ready yet')

        #turn new bins in to self.age_bound
        self.age_bound = lambda age_unq, bins : nu.array(age_bound)
        #change prior to output?

        #use output as sampler?

        #set up stuff for next run
        self._option = 'mcmc'
        self.samp = mc.MCMC_SA
        self.autosetup(int(best_fit[0]))
        print 'Use run() method to start new run'
            
    
    def uniform_prior(self, param, dust=None, losvd=None, bol=True):
        #calculates prior probablity for a value 
        #(sort of, if not in range returns 0 else 1
        
        #need to add non bool output, doesn't catch holes in param space
        #sees if ages are too close together
        self.bins = (len(param)) / 3
        age = param[range(1,self.bins*3,3)]
        age.sort()
        if nu.any(nu.diff(age) < 0.01):
            return 0
        #check losvd
        if nu.any(losvd):
           #[sigma,Z,h3,h4]
           #sigma is in log scale
           if losvd[0] <= 10**-3 or losvd[0] > 4:
              #sigma can't be negitive or too large
              return 0
           if nu.any(nu.abs(losvd[1:]) > 2.):
              #redshift can't be too high
              #skewness and kurtosis can't be too high
              return 0
        #check to see if no bins or some type of binning
        temp = self.age_bound(self._age_unq, self.bins)
        out = 1./(self._age_unq.ptp()*self._metal_unq.ptp())**self.bins
        if len(temp) <= 2:
            #all types of binning should look same
            for i in xrange(1, 3 * self.bins, 3):
                #age
                if param[i] > temp.max() or param[i] < temp.min(): 
                    out = 0
                    break
                #metal
                if (param[i - 1] > self.metal_bound.max() or 
                    param[i - 1] < self.metal_bound.min()): 
                    out = 0
                    break
        else: #some type of binning
            for i in xrange(self.bins):
                if (param[i * 3 + 1] > temp[i + 1] or 
                    param[i * 3 + 1] < temp[i]):
                    out = 0
                    break
                #metal
                if (param[i * 3] > self.metal_bound.max() or 
                    param[i * 3] < self.metal_bound.min()): 
                    out=0
                    break
        #make sure in boundary of points make into own funtion
        index = nu.sort(nu.hstack((range(0, self.bins*3,3),
                                   range(1,self.bins*3,3))))
        index = index.reshape(self.bins,2)
        for i in xrange(self.bins):
            if not bound.pinp_wbounds(param[index[i]], self.hull):
                return 0
        #check dust
        if nu.any(dust):
           if (nu.any(dust > self.dust_bound.max()) or 
               nu.any(dust < self.dust_bound.min())):
              return 0
        
        #check losvd
        if nu.any(losvd):
           if nu.any(losvd < 0) or nu.any(losvd > 10**4) or losvd[1] > 13:
              return 0
	if bol:
	    return 1
	else:
            return out
        
    def area_under_curve(self, data):
        #gives area under curve of the data spectra
        return nu.trapz(data[:, 1], data[:, 0])

    def func(self,param, dust_param=None, losvd_param=None, N=None):
        '''returns y axis of ssp from parameters
        if no N value, then does least squares for normallization params'''
        bins = param.shape[0] / 3
        if len(param) != bins * 3:
            return nu.nan
        
       #check if params are in correct range
        if self.uniform_prior(param, dust_param, losvd_param) < 1:
            return nu.inf, nu.zeros(bins)
        
        model = get_model_fit_opt(param, self._lib_vals, self._age_unq,
                                  self._metal_unq, bins)
        #don't bother with analysis if model isn't finite
        temp = nu.copy(model.values())
        if not nu.any(nu.isfinite(temp)):
            pik.dump((param,bins),open('error.pik','w'),2)
            return nu.inf, nu.zeros(bins)

        
       #line of sight velocity despersion
        if nu.any(losvd_param):
            model = LOSVD(model,losvd_param,[self.data[:,0].min(), self.data[:,0].max()])
        #dust
        if nu.any(dust_param):
            model = dust(nu.hstack((param, dust_param)), model)
 
        #make sure wavelength match up and have same range
        if not len(model['wave']) == len(self.data[:,0]):
           try:
              model = data_match(self.data, model, bins,True)
           except ValueError:
              import cPickle as pik
              pik.dump((self.data, model,temp,losvd_param, bins),open('error2.pik','w'),2)
              raise

        elif nu.allclose(model['wave'], self.data[:,0]):
           #make model wavelength match data
           try:
              model = data_match(self.data, model, bins,True)
           except ValueError:
              print 'here'

              import cPickle as pik
              pik.dump((self.data, model, bins),open('error2.pik','w'),2)
              raise
        wave = model['wave']
        #Find normalization and chi squared value
        if not nu.any(N):
           try:
              N,chi = N_normalize(self.data, model, bins)
           except ValueError:
              import cPickle as pik
              pik.dump((temp,model['0'],losvd_param),open('error1.pik','w'),2)
              print temp.shape,model['0'].shape,self.data.shape
              raise
        else:
           #model.pop('wave')
           N = param[range(2, bins * 3, 3)]
           #check if N is greater than 0
        if nu.any(N < 0):
           return nu.inf, N
        out = nu.zeros_like(model['0'])
        for i in model.keys():
           if i == 'wave':
              continue
           out += N[int(i)] * model[i]
        if self.data.shape[1] == 2:
           chi = ((self.data[:,1] - out)**2).sum()
        else: 
              #if data has uncertanty
           chi = nu.sum((self.data[:,1] - out)**2 / self.data[:,2])
            #otherwise use from param array
        return nu.vstack((wave,out)).T
        
    def func_N_norm(self, param, dust_param=None, losvd_param=None, N=None):
        '''returns chi and N norm best fit params if N==None'''
        bins = param.shape[0] / 3
        if len(param) != bins * 3:
            return nu.nan
        
       #check if params are in correct range
        if self.uniform_prior(param, dust_param, losvd_param) < 1:
            return nu.inf, nu.zeros(bins)
        
        model = get_model_fit_opt(param, self._lib_vals, self._age_unq,
                                  self._metal_unq, bins)
        #don't bother with analysis if model isn't finite
        temp = nu.copy(model.values())
        if not nu.all(nu.isfinite(temp)):
           import cPickle as pik
           pik.dump((param,bins),open('error.pik','w'),2)
           return nu.inf, nu.zeros(bins)
        #dust
        if nu.any(dust_param):
            model = dust(nu.hstack((param, dust_param)), model)
        #line of sight velocity despersion
        if nu.any(losvd_param):
            model = LOSVD(model,losvd_param, [self.data[:,0].min(), self.data[:,0].max()])
        #make sure wavelength match up and have same range
        if not len(model['wave']) == len(self.data[:,0]):
           try:
              model = data_match(self.data, model, bins,True)
           except ValueError:
              import cPickle as pik
              pik.dump((self.data, model,temp,losvd_param, bins),open('error2.pik','w'),2)
              raise

        elif nu.allclose(model['wave'], self.data[:,0]):
           #make model wavelength match data
           try:
              model = data_match(self.data, model, bins,True)
           except ValueError:
              print 'here'

              import cPickle as pik
              pik.dump((self.data, model, bins),open('error2.pik','w'),2)
              raise
        #Find normalization and chi squared value
        if not nu.any(N):
           try:
              N,chi = N_normalize(self.data, model, bins)
           except ValueError:
              import cPickle as pik
              pik.dump((temp,model['0'],losvd_param),open('error1.pik','w'),2)
              print temp.shape,model['0'].shape,self.data.shape
              raise
        else:
           #model.pop('wave')
           N = param[range(2, bins * 3, 3)]
           #check if N is greater than 0
           if nu.any(N < 0):
              return nu.inf, N
           out = nu.zeros_like(model['0'])
           for i in model.keys():
              if i == 'wave':
                 continue
              out += N[int(i)] * model[i]
           if self.data.shape[1] == 2:
              chi = ((self.data[:,1] - out)**2).sum()
           else: 
              #if data has uncertanty
              chi = nu.sum((self.data[:,1] - out)**2 / self.data[:,2])
        return chi, N

    def min_bound(self, bins):
        #outputs an array of minimum values for parameters
        out = nu.zeros(bins * 3)
        bin = nu.log10(nu.linspace(10 ** self.age_unq.min(),
                                   10 ** self.age_unq.max(), 
                                   bins + 1))
        bin_index = 0
        for k in range(bins * 3):
            if any(nu.array(range(0, bins * 3, 3)) == k): #metal
                out[k] = self.metal_unq[0]
            elif any(nu.array(range(1, bins * 3, 3)) == k): #age
                out[k] = bin[bin_index]
                bin_index += 1
            elif any(nu.array(range(2, bins * 3, 3)) == k): #norm
                out[k] = 0.0
        return out

    def max_bound(self, bins):
        #outputs an array of maximum values for parameters
        out = nu.zeros(bins * 3)
        bin = nu.log10(nu.linspace(10 ** self.age_unq.min(), 
                                   10 ** self.age_unq.max(),
                                   bins + 1))
        bin_index = 1
        for k in range(bins * 3):
            if any(nu.array(range(0, bins * 3, 3)) == k): #metal
                out[k] = self.metal_unq[-1]
            elif any(nu.array(range(1, bins * 3, 3)) == k): #age
                out[k] = bin[bin_index]
                bin_index += 1
            elif any(nu.array(range(2, bins * 3, 3)) == k): #norm
                out[k] = nu.inf
        return out

    def bounds(self, bins):
        Min = self.min_bound(bins)
        Max = self.max_bound(bins)
        out = []
        for i in range(len(Min)):
            out.append((Min[i], Max[i]))
        self.bounds = nu.copy(out)
        return out

    def n_neg_lest(self, param):
        #does bounded non linear fit
        bins = len(param) / 3
        try:
            out = fmin_bound(self.func, param, 
                             bounds = self.bounds(bins),
                             approx_grad = True)[0]
        except IndexError:
            out = param
        return out
    
    class send_functions(object):
        'groups functions needed for MCMC or RJMCMC sampling'
        def __init__(self,sampler,prior,proposal,log_lik):
            self.sampler = sampler
            self.prior = prior
            self.proposal = proposal
            self.lik = log_lik
            #things needed for functions to work in both samplers
            lib_vals=get_fitting_info(lib_path)
            #to keep roundoff error constistant
            lib_vals[0][:,0]= nu.log10(lib_vals[0][:, 0]) 
            metal_unq = nu.unique(lib_vals[0][:, 0])
            lib_vals[0][:,0] = 10**lib_vals[0][:,0]
            age_unq = nu.unique(lib_vals[0][:, 1])
            self._lib_vals = lib_vals
            self._age_unq = age_unq
            self._metal_unq = metal_unq


############plotting
def plot_model(param, data, bins, plot=True):
    import pylab as lab
    #takes parameters and returns spectra associated with it
    lib_vals = get_fitting_info(lib_path)
    lib_vals[0][:, 0] = 10 ** nu.log10(lib_vals[0][:, 0])
    metal_unq = nu.log10(nu.unique(lib_vals[0][:, 0]))
    #check to see if metalicity is in log range (sort of)
    if (any(param[range(0, bins * 3, 3)] > metal_unq.max()) or 
        any(param[range(0, bins * 3, 3)] < metal_unq.min())):
        print 'taking log of metalicity'
        param[range(0, bins * 3, 3)] = nu.log10(param[range(0,
                                                            bins * 3, 
                                                            3)])

    fun = MC_func(data)
    fun.autosetup()
    out = fun.func(param[range(3*bins)], param[-6:-4],param[-4:], False)
    #out = fun.func(param[range(3*bins)], param[-2:],N=False)
    if plot:
        lab.plot(fun.data[:, 0], fun.data[:, 1] * fun.norms, label='Data')
        lab.plot(fun.data[:, 0], out, label='Model')
        lab.legend()
    return nu.vstack((fun.data[:, 0], out)).T


def chi_squared_plot(param,chi,bins,data,points=100):
    import visualizations as vis
    import pylab as plt
    lab = plt
    #define axis
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    #grid data
    metal = nu.linspace(param[str(bins)][:,range(0,bins*3,3)].ravel().min(),
                          param[str(bins)][:,range(0,bins*3,3)].ravel().max(),points)
    age = nu.linspace(param[str(bins)][:,range(1,bins*3,3)].ravel().min(),
                          param[str(bins)][:,range(1,bins*3,3)].ravel().max(),points)
    chi_plot = lab.griddata(param[str(bins)][:,range(0,bins*3,3)].ravel(),
                            param[str(bins)][:,range(1,bins*3,3)].ravel(),
                            nu.tile(chi[str(bins)],[bins,1]).T.ravel(),
                            metal,age)
                            
    #get min
    vmin = nu.min(chi[str(bins)].min())
    chi_plot[nu.isnan(chi_plot)] = 1
    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.hist(param[str(bins)][:,range(0,bins*3,3)].ravel(),200)
    axHisty.hist(param[str(bins)][:,range(1,bins*3,3)].ravel(),
                 bins=200,orientation='horizontal')
    axScatter.pcolor(metal,age,lab.log10(chi_plot),vmin=lab.log10(vmin),
                     cmap='gray')

    #set limits
    xlim,ylim = axScatter.get_xlim(),axScatter.get_ylim()
    axHistx.set_xlim(xlim)
    axHisty.set_ylim(ylim)
    lab.show()

if __name__=='__main__':
    '''    import cProfile as pro
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    bins=16
    active_param=nu.zeros(bins*3)
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0
    #start in random place
    for k in xrange(3*bins):
        if any(nu.array(range(0,bins*3,3))==k):#metalicity
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,bins*3,3))==k): #age
                #active_param[k]=nu.random.random()
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
            else: #norm
                #active_param[k]=nu.random.random()
                pass

    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)
    data,info1,weight=create_spectra(bins,lam_min=2000,lam_max=10**4)
    pro.runctx('N_normalize(data,model,bins)'
               , globals(),{'data':data,'model':model,'bins':bins}
               ,filename='agedata.Profile')
               '''

    #LOSVD test
    sigma = [50.]
    chi = [nu.sum((fun.data[:,1] - ag.LOSVD(out,sigma[-1])[:,1])**2)]
    for i in range(10**5):
        sigma_new = sigma[-1] + nu.random.randn()*10.
        if nu.exp(chi[-1] - nu.sum((fun.data[:,1] - ag.LOSVD(out,sigma_new)[:,1])**2))>nu.random.rand():
            sigma.append( nu.copy(sigma_new))
            chi.append(nu.sum((fun.data[:,1] - ag.LOSVD(out,sigma_new)[:,1])**2))
