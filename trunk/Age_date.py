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
""" A python version of the age dating spectral fitting code done by Mongwane 2010"""

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
import time as Time
#123456789012345678901234567890123456789012345678901234567890123456789

###spectral lib stuff####
global lib_path,spect
lib_path = '/home/thuso/Phd/Spectra_lib/'
try:
    #load in spectral library, if default paths isn't correct ask
    #for correct path to lib
    spect,info = load_spec_lib(lib_path)  
except OSError :
    lib_path=raw_input('location to spectra libray? eg.'
                       + '/home/thuso/Phd/Spectra_lib/')
    if not lib_path[-1] == '/':
        lib_path += '/'
    spect,info = load_spec_lib(lib_path)


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

def get_model_fit_opt(param, lib_vals, age_unq, metal_unq, bins):
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
                if not index: #if not in lib_vals return dummy array
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
                
                if not index: #if not in lib_vals return dummy array
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
            if not index: #if not in lib_vals return dummy array
                out[str(ii)] = spect[:,0] + nu.inf
            else:
                out[str(ii)] = nu.copy(spect[:,index[0] + 1])
        #run 2 d interp
        else:
            metal.sort()
            metal = nu.array(metal)[nu.array([0, 3, 1, 2], dtype =
                                             'int32')]
            age.sort()
            
            for i in range(4):
                index = nu.nonzero(nu.logical_and(lib_vals[0][:, 1] ==
                                                  age[i],
                                                  lib_vals[0][:, 0] ==
                                                  10 ** metal[i]))[0]
                if not index: #if not in lib_vals return dummy array
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

def data_match_all(data):
    #makes sure data and model have same wavelength range and points for library
    model = {}
    out_data = nu.copy(data)
    global spect
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
    model = data_match_new(out_data, model,spect[0, :].shape[0] - 1)
    out = nu.zeros([model['0'].shape[0], len(model.keys()) + 1])
    out[:, 0] = nu.copy(out_data[:, 0])
    for i in model.keys():
        out[:, int(i) + 1] = model[i]
    spect = nu.copy(out)
    return spect, out_data

def data_match_new(data, model, bins):
    #makes sure data and model have same wavelength range 
    #and points but with a dictionary
    #assumes spect_lib is longer than data
    out = {}
    #if they have the same x-axis
    if nu.all(model['wave'] == data[:, 0]):
        for i in xrange(bins):
            out[str(i)] = model[str(i)]
    else: #not same shape, interp 
        for i in xrange(bins):
            out[str(i)] = spectra_lin_interp(model['wave'],
                                             model[str(i)], data[:,0])
    return out

def check(param, metal_unq, age_unq, bins): 
    #checks if params are in bounds
    #linear spacing
    #age=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))
    #log spacing
    age = nu.linspace(age_unq.min(), age_unq.max(), bins + 1) 
    for j in xrange(bins): #check age and metalicity
        if (any([metal_unq[-1], age[j + 1]] < param[j * 3: j * 3 + 2])
            or any([metal_unq[0], age[j]] > param[j * 3 :j * 3 + 2])):
            
            '''if any([metal_unq[-1],age_unq[-1]]<param[j*3:j*3+2]) or any([metal_unq[0],age_unq[0]]>param[j*3:j*3+2]):'''
            return True
        #if any(nu.abs(nu.diff(param.take(range(1,bins*3,3))))<.3):
        #    return True
        if not (0 < param[j * 3 + 2]): #and param[j*3+2]<1): #check normalizations
            return True
    return False

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
    
def chain_gen_all(means, metal_unq, age_unq, bins, sigma):
    #creates new chain for MCMC, does log spacing for metalicity
    #lin spacing for everything else, runs check on values also
    out=nu.random.multivariate_normal(means,sigma)
    return out

def chain_gen_dust(means,sigma): #redundent
    #creates new chain for MCMC, for dust componet
    out=nu.random.multivariate_normal(means,sigma)
    return out

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

def rand_choice(x, prob):
    #chooses value from x with probabity of prob**-1 
    #so lower prob values are more likeliy
    #x must be monotonically increasing
    if not nu.sum(prob) == 1: #make sure prob equals 1
        prob = prob / nu.sum(prob)
    if nu.any(prob == 0): #get weird behavor when 0
        #smallest value for float32 and 1/value!=inf
        prob[prob == 0] = 6.4e-309 
    #check is increasing
    u = nu.random.rand()
    if nu.all(x == nu.sort(x)): #if sorted easy
        N = nu.cumsum(prob ** -1 / nu.sum(prob ** -1))
        index = nu.array(range(len(x)))
    else:
        index = nu.argsort(x)
        temp_x = nu.sort(x)
        N = nu.cumsum(prob[index] ** -1 / nu.sum(prob[index] ** - 1))
    try:
        return index[nu.min(nu.abs(N - u)) == nu.abs(N - u)][0]
    except IndexError:
        print x,prob
        raise

def dict_size(dic):
    #returns total number of elements in dict
    size = 0
    for i in dic.keys():
        size += len(dic[i])

    return size

def dust(param, model):
    #does 2 componet dust calibration model following charlot and fall 2000
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
    #dust extinction functin
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

#####classes############# 
class MC_func:
    #makes more like function, so input params and the chi is outputted
    def __init__(self, data, bins=None):
        #match spectra for use in class
        self.spect, self.data=data_match_all(data)
        #normalized so area under curve is 1 to keep chi 
        #values resonalble
        #need to properly handel uncertanty
        self.norms=self.area_under_curve(data) * 10 ** -5 #need to turn off
        self.data[:,1] = self.data[:, 1] / self.norms

        #initalize bound varables
        lib_vals=get_fitting_info(lib_path)
        #to keep roundoff error constistant
        lib_vals[0][:,0]=10 ** nu.log10(lib_vals[0][:, 0]) 
        metal_unq = nu.log10(nu.unique(lib_vals[0][:, 0]))
        age_unq = nu.unique(lib_vals[0][:, 1])
        self.lib_vals = lib_vals
        self.age_unq = age_unq
        self.metal_unq = metal_unq

        #set prior info for age, and metalicity
        self.metal_bound = nu.array([metal_unq.min(),metal_unq.max()])
        self.dust_bound = nu.array([0., 4.])
        #if want age to be binned
        if bins == 'linear':
            self.age_bound = lambda age_unq, bins: (
                nu.log10(nu.linspace(10 ** age_unq.min(), 
                                     10 ** age_unq.max(), bins + 1)))
        elif bins == 'log':
             self.age_bound = lambda age_unq, bins: nu.linspace(
                 age_unq.min(), age_unq.max(), bins + 1)
        else: #no binning
            self.age_bound=lambda age_unq, bins: nu.array(
                [age_unq.min(), age_unq.max()])
       
    def uniform_prior(self, param):
        #calculates prior probablity for a value 
        #(sort of, if not in range returns 0 else 1
        
        #check to see if no bins or some type of binning
        self.bins = (len(param) - 2) / 3
        temp = self.age_bound(self.age_unq, self.bins)
        out = 1.
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
        #check dust
        if (nu.any(param[-2:] > self.dust_bound.max()) or 
            nu.any(param[-2:] < self.dust_bound.min())):
            return 0
        return out
        
    def area_under_curve(self, data):
        #gives area under curve of the data spectra
        return nu.trapz(data[:, 1], data[:, 0])

    def func(self,param, dust_param, N=None):
        '''returns y axis of ssp from parameters
        if no N value, then does least squares for normallization params'''
        bins = param.shape[0] / 3
        if len(param) != bins * 3:
            return nu.nan
        #check if params are in correct range
        if self.uniform_prior(nu.hstack((param,dust_param))) < 1:
            return nu.zeros_like(self.spect[:, 0])
        model = get_model_fit_opt(param, self.lib_vals, self.age_unq,
                                  self.metal_unq, bins)  
        model = dust(nu.hstack((param, dust_param)), model) #dust
        #if no N use N_norm to find nnls best fit
        if not N: 
            N, chi = N_normalize(self.data, model, bins)
        else:
            #otherwise use from param array
            model.pop('wave')
            N = param[range(2, bins * 3, 3)]
        
        return nu.sum(nu.array(model.values()).T[:,
                nu.argsort(nu.int64(model.keys()))] * 
                      N[nu.int64(model.keys())], 1)
 
    def func_N_norm(self, param, dust_param):
        '''returns chi and N norm best fit params'''
        bins = param.shape[0] / 3
        if len(param) != bins * 3:
            return nu.nan
        #check if params are in correct range
        if self.uniform_prior(nu.hstack((param,dust_param))) < 1:
            return nu.inf, nu.zeros(bins)

        model = get_model_fit_opt(param, self.lib_vals, self.age_unq,
                                  self.metal_unq, bins)  
        model = dust(nu.hstack((param, dust_param)), model) #dust
        N,chi = N_normalize(self.data, model, bins)
    
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

def plot_model(param, data, bins):
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
    out = fun.func(param[:-2], param[-2:], True)
    lab.plot(fun.data[:, 0], fun.data[:, 1] * fun.norms, label='Data')
    lab.plot(fun.data[:, 0], out, label='Model')
    lab.legend()
    return nu.vstack((fun.data[:, 0], out)).T




if __name__=='__main__':
    import cProfile as pro
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
