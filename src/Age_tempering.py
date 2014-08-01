#!/usr/bin/env python2.7
#
# Name:  Age parallel tempering MCMC
#
# Author: Thuso S Simon
#
# Date: 31th of July, 2014
# TODO:  
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
# History (version,date, change author)
# More general version of RJMCMC, allows multiple objects to be fitted 
# independetly or hierarically. Also fits single objects and splits into 
# multiple independent componets via coverance matrix

import numpy as nu
import sys
import os
import time as Time
import cPickle as pik
import MC_utils as MC
import pandas as pd
from Age_mltry import Param_MCMC, MCMCError
import ipdb
# import acor
# from memory_profiler import profile
from glob import glob
a = nu.seterr(all='ignore')


def tempering_main(fun, option, burnin=5*10**3,  max_iter=10**5,
            seed=None, fail_recover=False):
    '''Main multi RJMCMC program. Like gibbs sampler but for RJMCMC'''
    # see if to use specific seed
    if seed is not None:
        nu.random.seed(seed)
    # initalize paramerts/class for use by program
    Param = Param_temp(fun, burnin)
    if fail_recover:
        # fail recovery
        fun, option, burnin = Param.fail_recover(fail_recover,fun, option)
        Param.burnin =  burnin
    else:
        # initalize and check if param are in range
        timeInit = 0
        while Param.initalize(fun):
            timeInit += 0
            if  timeInit > 10:
                raise MCMCError('Bad Starting position, check params')
    # Start RJMCMC
    while option.iter_stop:
        bins = Param.bins
        if option.current % 1 == 0 and option.current > 0:
            acpt = nu.min([i[-1] for i in Param.acept_rate[bins].values()])
            chi = nu.sum([i[-1] for i in Param.chi[bins].values()])
            try:
                show = ('acpt = %.2f,log lik = %e, model = %s, steps = %i,Temp = %2.0f'
                    %(acpt, chi, bins, option.current, nu.min(Param.sa.values())))
                print show
            except:
                pass
            sys.stdout.flush()
        #do  M-H
        mcmc(Param, fun)
        # Change Step size
        if option.current < burnin * 2:
            Param.step(fun, option.current, 500)
       # Tune delta T
        if option.current > burnin * 2 and  option.current < burnin * 3:
            Param.tune_sa()
        # stop SA and run chains in parallel
        # Convergence assement
        if option.current % 5000 == 0 and option.current > 1:
            pass
            #Param.eff = MC.effectiveSampleSize(Param.param[bins])
        # Save currnent Chain state
        Param.save_state(option.current)
        option.current += 1
        if option.current >= max_iter:
            option.iter_stop = False
    # Finish and return
    fun.exit_signal()
    return Param


def mcmc(Param, fun):
    '''Does stay step for MCMC'''
    bins = Param.bins
    # sample from distiburtion
    Param.active_param[bins] = fun.proposal(Param.active_param[bins], Param.sigma[bins])
    # calculate new model and chi
    prior = fun.prior(Param.active_param, bins)
    lik = fun.lik(Param.active_param, bins)
    new_chi = {}
    #calc posterior for each object
    for Prior,index in prior:
        new_chi[index] = Prior
    for Lik,index in lik:
        if nu.isfinite(new_chi[index]):
            new_chi[index] += Lik
    #MH critera
    for key in new_chi.keys():
        if mh_critera(Param.active_chi[bins][key], new_chi[key],
                      Param.sa[key] ):
            #accept
            #Param.active_chi[bins][key] = new_chi[key] + 0.
            Param.accept(key, new_chi[key])
        else:
            #reject
            #Param.active_param[bins][key] = Param.param[bins][key].copy()
            Param.reject(key)
    #Param.save_chain()
    Param.cal_accept()
        
def mh_critera(chi_old, chi_new, sa=1.):
    '''Does Metropolis-Hastings criera, with simulated anneling'''
    a = float((chi_new - chi_old)/(2. * nu.ravel(sa)))
    if not nu.isfinite(a):
        return False
    if nu.exp(a) > nu.random.rand():
        # acepted
        return True
    else:
        # rejected
        return False
    
class Param_temp(Param_MCMC):

    def initalize(self, lik_fun):
        '''initalize parallel tempering'''
        self.bins = lik_fun.models.keys()[0]
        for bins in lik_fun.models:
            # model level
            for gal in lik_fun.models[bins]:
                self.active_param[bins][gal], self.sigma[bins][gal] = lik_fun.initalize_param(gal)
                #self.active_chi[bins][gal] = {}
                self.out_sigma[bins][gal]  =  [self.sigma[bins][gal][:]]
            
                self.acept_rate[bins][gal] = []
            
        # check if params are in range
        lik, prior = (lik_fun.lik(self.active_param, bins),
                               lik_fun.prior(self.active_param, bins))
        #self.chi[bins] = {}
        #get intal params lik and priors
        for Prior, gal in prior:
            if not nu.isfinite(Prior):
                return True
            self.chi[bins][gal] = [Prior]
            self.active_chi[bins][gal] = Prior
        for Lik, gal in lik:
            if not nu.isfinite(Lik):
                return True
            self.chi[bins][gal][-1] += Lik
            self.active_chi[bins][gal] = Lik
            self.param[bins][gal] = [self.active_param[bins][gal].copy()]
            self.T_start[gal] = abs(nu.max(self.chi[bins].values()))
            
        self.SA(0)
        self.save_state(0, lik_fun)
        return not nu.all(nu.isfinite(self.chi[bins].values()))

    def tune_sa(self):
        '''Turns T and delta T so interchange rate is 20-60%'''

    def stop_tempering(self):
        '''stops tempering and starts all chains at T=1 state in
        parallel to save time'''
    
