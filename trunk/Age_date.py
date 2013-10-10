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
#from losvd_convolve import convolve


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
        arg = ''
        for i in args:
          arg+=str(i)
        #arg = str(args[1])
        if arg in self.cache:
          return self.cache[arg]
        else:
          value = self.func(*args)
          self.cache[arg] = value
          return value
      elif self.func.__name__ == 'make_burst':
          arg = str((args[0],args[1],args[2]))
          if not arg in self.cache:
            self.cache[arg] = self.func(*args)
          return self.cache[arg]
      elif self.func.__name__ == 'f_dust':
        arg = str(args)
        if not arg in self.cache:
          self.cache[arg] = self.func(*args)
        return self.cache[arg]
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
        if index <= 0 or index >= len(age_unq):
            #out of bounds
            raise ValueError('age param is out of bounds')
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

#@memoized
def make_burst(length, t, metal, metal_unq, age_unq, spect, lib_vals):
    '''def make_burst(length, t, metal, metal_unq, age_unq, spect, lib_vals)
    (float, float,float, ndarray(float),ndarray(float)
ndarray(float) tuple(ndarray(floats),ndarray(str))) -> ndarray(float)
Turns SSP into busrt of constant stellar formation and of length dt at
age t for a const metalicity 10**(t-9) - length/2 to 10**(t-9) + length/2.
All terms are logrythmic.
    '''
	#lib_vals[0][:,0] = 10**nu.log10(lib_vals[0][:,0])
    if t < age_unq.min() or t > age_unq.max():
		#Age not in range
		return spect[:,0] + nu.inf
    if metal < metal_unq.min() or metal > metal_unq.max():
		#Metalicity not in range
		return spect[:,0] + nu.inf
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
    if len(ages) < 10:
        ages = nu.linspace(t_min,t_max,10)
    temp_param = []
    #get ssp's
    for i in ages:
		temp_param.append([metal, i ,1.])
    try:
        ssp = get_model_fit_opt(nu.ravel(temp_param), lib_vals, age_unq, metal_unq, 
        len(ages), spect)
    except ValueError:
        #interp failed
        return spect[:,0] + nu.inf
	#sort for intergration
    inters = []
	#integrate ssp's
    ssp.pop('wave')
    for i in nu.sort(nu.int64(ssp.keys())):
        if not nu.any(nu.isfinite(ssp[str(i)])):
            continue
        inters.append(ssp[str(i)])
    #check if any arrays are bad
    if len(inters) == 0 or len(inters) != len(ages):
        return spect[:,0] + nu.inf
    #normalize to 1 solar mass
    
    return simps(inters, ages, axis=0)/(ages.ptp())
	#return new ssp
	
		
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

def data_match(data, model, bins, keep_wave=False):
   '''(ndarray,dict(ndarray),int,bool) -> dict(ndarray)
   Makes sure data and model have same wavelength range 
    and points but with a dictionary
    assumes model wavelength range is longer than data.
    Uses linear interpolation to match wavelengths'''
   ####add resolution downgrading!!!
   out = {}
    #if they have the same x-axis
   if nu.all(model['wave'] == data[:, 0]):
      for i in model.keys():
          if i == 'wave':
              continue
          out[i] = model[i]
   else: #not same shape, interp 
      for i in model.keys():
        if i == 'wave':
          continue
        out[i] = spectra_lin_interp(model['wave'],
                                    model[i], data[:,0])
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

def LOSVD(model, param, wave_range, convlve='python'):
    '''(dict(ndarray), ndarray, ndarray) -> dict(ndarray)

    Convolves data with a gausian with dispersion of sigma, and hermite poly
    of h3 and h4

    model is dict of ssp with ages being the keys
    param is a 4x1 flat array with params of losvd [log10(sigma), v (redshift), h3, h4]
    wave_range is gives edge support for convolution

    Has option for c++ wrapped convolution (50x faster than python) doesn't work with parallel processing
    or python convolutin'''
    #unlog sigma
    tparam = param.copy()
    tparam[0] = 10**tparam[0]
    tmodel = model.copy()
    #resample specturm so resolution is same at all places
    wave_range = nu.array(wave_range)/(1. + tparam[1]) - [100, -100]
    index = nu.searchsorted(model['wave'], [wave_range[0], wave_range[1]])
    #if not in range
    if index[1] == len(model['wave']):
        index[1] -= 1
    try:
        #calculate resolution
       wave_diff = nu.diff(model['wave'][index[0]:index[1]]).min()
    except ValueError:
       #model and data range are off
       return tmodel
    wave = nu.arange(model['wave'][index[0]], model['wave'][index[1]], wave_diff)
    #sum spectra and convole all
    sum_spec = {}
    sum_spec['wave'] = model['wave']
    sum_spec['0'] = nu.zeros_like(model['wave'])
    for i in model.keys():
        if i == 'wave':
            continue
        sum_spec['0'] += model[i]
    sum_spec['0'] = spectra_lin_interp(sum_spec['wave'], sum_spec['0'], wave)
    sum_spec['0'] = convolve_python_fast(wave, nu.ascontiguousarray(sum_spec['0']), tparam)
    sum_spec['wave'] = redshift(wave,tparam[1])
    ''' #convolve individual spectra
    for i in model.keys():
        if i == 'wave':
            continue
        if 'python' == convlve:
            model[i] = spectra_lin_interp(model['wave'], model[i], wave)
            tmodel[i] = convolve_python_fast(wave, nu.ascontiguousarray(model[i]), tparam)
        else:
            #c++ convolve
            tmodel[i] = spectra_lin_interp(tmodel['wave'], tmodel[i], wave)
            convolve(wave, tmodel[i], tparam ,tmodel[i])
    #apply redshift
    tmodel['wave'] = redshift(wave,tparam[1])
    #uncertanty convolve
    #if data.shape[1] == 3:
    #    out[:,2] = nu.sqrt(nu.convolve(kernel**2, data[:,2]**2,'same'))

    return tmodel'''
    return sum_spec

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

#@memoized
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

