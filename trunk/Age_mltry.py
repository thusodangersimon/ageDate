#!/usr/bin/env python
#
# Name:  Age multiy-try metropolis with RJMCMC
#
# Author: Thuso S Simon
#
# Date: 29th of June, 2012
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
# More general version of RJMCMC, allows multiple objects to be fitted 
# independetly or hierarically. Also fits single objects and splits into 
# multiple independent componets via coverance matrix

import numpy as nu
import sys,os
import time as Time
import cPickle as pik
import MC_utils as MC
# import acor
# from memory_profiler import profile
from glob import glob

a=nu.seterr(all='ignore')



def multi_main(fun, option, burnin=5*10**3, birth_rate=0.5,max_iter=10**5, seed=None, fail_recover=False):
    '''Main multi RJMCMC program'''
    #see if to use specific seed
    if seed is not None:
        nu.random.seed(seed)
    #initalize paramerts/class for use by program
    Param = param(fun)
    if fail_recover:
        # fail recovery
        Param.fail_recover(fail_recover)
    else:
        # initalize and check if param are in range
        timeInit = Time.time()
        while Param.initalize(fun) or Time.time()- timeInit < 60.:
            pass
        if Time.time()- timeInit < 60.:
            raise MCMCError('Bad Starting position, check params')
    
    # Start RJMCMC
    while option.iter_stop:
        bins = Param.bins
        if Param.T_cuurent[bins] % 501 == 0:
            show = ('acpt = %.2f,log lik = %e, bins = %s, steps = %i,ESS = %2.0f'
                    %(acept_rate[bins][-1],chi[bins][-1],bins, option.current,eff))
            print show
            sys.stdout.flush()
        #stay, try or jump
        doStayTryJump =  nu.random.rand()
        if doStayTryJump <= .3:
            # Stay
            pass
        elif doStayTryJump > .3 and doStayTryJump < .6:
            # Try
            pass
        else:
            # Jump
            pass
        # Change Step size
        # Change parameter grouping
        # Change temperature
        # Convergence assement
        # Save currnent Chain state
        # Finish and return

class param(object):
    def __doc__(self):
        '''stores params for use in multi_main'''

    def __init__(self, lik_class):
        self.active_param, self.sigma = {}, {}
        self.param, self.chi = {}, {}
        self.Nacept, self.Nreject = {},{}
        self.acept_rate, self.out_sigma = {},{}
        self.bayes_fact = {} #to calculate bayes factor
        #simulated anneling param
        self.T_cuurent = {}
        self.bins = nu.random.choice(fun.models.keys())
        self.Nacept[self.bins] , self.Nreject[self.bins] = 1.,1.
        self.acept_rate[self.bins], = [1.], [sigma[bins][0][:]]
        self.Nexchange_ratio = 1.0
        self.size, self.a = 0,0
        self.time, self.timeleft = 1, nu.random.exponential(100)
        self.T_stop =  1.
        self.trans_moves = 0
        #bayes_fact[bins] = #something
        self.T_cuurent[self.bins] = 0
        #set storage functions

    def initalize(self, lik_fun):
        '''Initalize certan parms'''
        for i in lik_fun.models.keys():
            self.active_param[i], self.sigma[i] = fun.initalize_param(i)
        # check if params are in range
        self.chi[self.bins] = [lik_fun.lik(self.active_param, self.bins) +
                               lik_fun.prior(self.active_param, self.bins)]
        self.param[self.bins] = [self.active_param[self.bins].copy()]
        self.self.T_start = self.chi[self.bins][-1] + 0
        return not nu.isfinite(self.chi[self.bins][-1])

    def fail_recover(self, path):
        '''Loads params from old run'''
        raise NotImplementedError

    def save_state(self, path=None):
        '''Saves current state of chain incase run crashes'''
        raise NotImplementedError
    
    def singleObjSplit(self):
        '''Checks correlation between params to see if should split'''
        raise NotImplementedError


class MCMCError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
