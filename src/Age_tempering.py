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
from mpi4py import MPI as mpi
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
    # set mpi comm
    comm = mpi.COMM_WORLD
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
            Param.SA(option.current)
       # Tune delta T
        if option.current > burnin and  option.current < burnin * 3:
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

    def initalize(self, lik_fun, comm):
        '''initalize parallel tempering'''
        self.bins = lik_fun.models.keys()[0]
        self.comm = comm
        self.temp_exchange = {'accept':1,'reject':1}
        lik_fun.initalize_temp(comm.size)
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
        # initalize tempering
        self.SA(0)
        self.save_state(0, lik_fun)
        return not nu.all(nu.isfinite(self.chi[bins].values()))

    def tune_sa(self, min_rate=.2, max_rate=.6):
        '''Turns T and delta T so interchange rate is 20-60%'''

    def _change_state(self, src, dest, swap=True):
        '''Changes state of mcmc. Everything except aneeling param'''
        if swap:
            #make temps
            temp_chi = nu.copy(self.chi[bins][src][-1])
            temp_active_chi = nu.copy(self.active_chi[bins][src])
            temp_param = self.param[bins][src][-1].copy()
            temp_active_param = self.active_param[bins][src].copy()
            temp_Sigma = nu.copy(self.Sigma[bins][src])
            # save to src
            self.chi[bins][src][-1] = nu.copy(self.chi[bins][dest][-1])
            self.active_chi[bins][src] = nu.copy(self.active_chi[bins][dest])
            self.param[bins][src][-1] = self.param[bins][dest][-1].copy()
            self.active_param[bins][src] = self.active_param[bins][dest].copy()
            self.Sigma[bins][src] = nu.copy(self.Sigma[bins][dest])
            # save to dest
            self.chi[bins][dest][-1] = temp_chi
            self.active_chi[bins][dest] = temp_active_chi
            self.param[bins][dest][-1] = temp_param
            self.active_param[bins][dest] = temp_active_param
            self.Sigma[bins][dest] = temp_Sigma      
        else:
            key = src
            gal = dest
            self.chi[bins][gal][-1] = nu.copy(self.chi[bins][key][-1])
            self.active_chi[bins][gal] = nu.copy(self.active_chi[bins][key])
            self.param[bins][gal][-1] = self.param[bins][key][-1].copy()
            self.active_param[bins][gal] = self.active_param[bins][key].copy()
            self.Sigma[bins][gal] = nu.copy(self.Sigma[bins][key])

                
    def stop_tempering(self):
        '''stops tempering and starts all chains at T=1 state in
        parallel to save time'''
        bins = self.active_param.keys()[0]
        # find gal key for min temperature
        for key in self.active_param[bins]:
            if int(key.split('_')[-1]) == 0:
                break
        for gal in self.active_param[bins]:
            self._change_state(key, gal, False)
            self.sa[gal] = 1.

            
    def SA(self, chain_number, fail_recover=False):
        '''Calculates and initalizes anneeling params'''
        bins = self.active_param.keys()[0]
        max_chi = nu.max(self.active_chi[bins].values())
        min_chi = nu.min(self.active_chi[bins].values())
        if chain_number == 0 or fail_recover:
            # initalize
            self.sa = {}
            self.T_start = {}
            self.T_stop = {}
            self.delta_T = (max_chi - min_chi) / float(self.comm.size)
            # set range of temperatures
            for gal,sa in enumerate(nu.arange(min_chi, max_chi, self.delta_T)):
                for key in self.active_chi[bins]:
                    if int(key.split('_')[-1]) == gal:
                        # stop at temp before current
                        if gal == 0:
                            self.T_stop[key] = 1.
                        else:
                            self.T_stop[key] = nu.max(self.sa.values())
                        self.sa[key] = abs(sa)
                        self.T_start[gal] = abs(sa)
            if fail_recover:
                # Check if before or after burnin
                if chain_number > self.burnin * 2:
                    for key in self.sa:
                        self.sa[key] = 1.
                elif chain_number > self.burnin and chain_number < self.burnin * 2:
                    # initalize at bottom levels
                    for key in self.sa:
                        self.sa[key] = self.T_stop[key] + 0.
                        
        else:
            if chain_number < self.burnin:
                # reduce anneeling incase likely places
                for gal in self.active_chi[bins]:
                    self.sa[gal] = MC.SA(chain_number, self.burnin,
                                         self.T_start[gal], self.T_stop[gal])
                    
                
