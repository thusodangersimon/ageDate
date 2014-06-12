#!/usr/bin/env python2.7
#
# Name:  Age multiy-try metropolis with RJMCMC
#
# Author: Thuso S Simon
#
# Date: 29th of June, 2012
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
from  database_utils import numpy_sql
from sqlalchemy import create_engine
import ipdb
from glob import glob
import acor
# from memory_profiler import profile

a = nu.seterr(all='ignore')


def multi_main(fun, option, burnin=5*10**3,  max_iter=10**5,
            seed=None, fail_recover=False):
    '''Main multi RJMCMC program. Like gibbs sampler but for RJMCMC'''
    # see if to use specific seed
    if seed is not None:
        nu.random.seed(seed)
    # initalize paramerts/class for use by program
    Param = Param_MCMC(fun, burnin)
    if fail_recover:
        # fail recovery
        if isinstance(fail_recover, str):
            Param.fail_recover(fail_recover)
        else:
            raise TypeError('Put in str of path to temporary database')
    else:
        # initalize and check if param are in range
        timeInit = Time.time()
        while Param.initalize(fun):
            if Time.time() - timeInit > 60.:
                raise MCMCError('Bad Starting position, check params')
    # Start RJMCMC
    while option.iter_stop:
        bins = Param.bins
        if option.current % 50 == 0: 
            try:
                show = ('acpt = %.2f,log lik = %e, model = %s, steps = %i,ESS = %2.0f'
                        %(nu.min(Param.acept_rate[bins]),float(Param.chi[bins].tail(1).sum(1)),bins,
                        option.current, nu.max(Param.sa)))
                print show
            except TypeError:
                pass
            
            sys.stdout.flush()
        # stay, try or jump
        doStayTryJump =  nu.random.rand()
        stay(Param, fun)
        '''if doStayTryJump <= .3:
            # Stay
            stay(Param, fun)
        elif doStayTryJump > .3 and doStayTryJump < .6:
            # Try
            pass
        else:
            # Jump
            pass
            # jump(Param, fun, birth_rate)'''
        # Change Step size
        Param.step(fun, option.current, 500)
        # Change parameter grouping
        # reconfigure(Param)
        # Change temperature
        Param.SA(option.current)
        # Convergence assement
        if option.current % 5000 == 0 and option.current > 1:
            pass
            #Param.eff = MC.effectiveSampleSize(Param.param[bins])
        # Save currnent Chain state
        Param.save_state(option)
        option.current += 1
        if option.current >= max_iter:
            option.iter_stop = False
    # Finish and return
    fun.exit_signal()
    Param.delete_temp()
    return Param


def stay(Param, fun):
    '''Does stay step for RJMCMC'''
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
                      Param.sa[key]):
            #accept
            #Param.active_chi[bins][key] = new_chi[key] + 0.
            Param.accept(key, new_chi[key])
        else:
            #reject
            #Param.active_param[bins][key] = Param.param[bins][key].copy()
            Param.reject(key)
    #Param.save_chain()
    Param.cal_accept()
    
        
def jump(Param, fun, birth_rate):
    '''Does cross model jump for RJMCM'''
    bins = Param.bins
    Param.active_param, temp_bins, attempt, critera = fun.birth_death(
        birth_rate, bins, Param.active_param)
    # if attempt:
    # check if accept move
    tchi = fun.prior(Param.active_param, temp_bins)
    if nu.isfinite(tchi):
        tchi += fun.lik(Param.active_param, temp_bins)
        # likelihoods
        rj_a = (tchi - Param.chi[bins][-1])
        # model prior
        rj_a += (fun.model_prior(temp_bins) - fun.model_prior(Param.bins))
        Param.trans_moves += 1
        # simulated aneeling 
        rj_a /= MC.SA(Param.trans_moves, 50, abs(Param.chi[bins][-1]),
                      Param.T_stop)
        # RJ-MH critera
    else:
        rj_a, critera = -nu.inf, 1.  
        
    if nu.exp(rj_a) * critera > nu.random.rand():
        # accept move
        Param.accept(temp_bins)
        bins = Param.bins
        # bins = temp_bins
        Param.chi[bins].append(tchi + 0)
        # allow for quick tuneing of sigma
        if Param.T_cuurent[bins] > Param.burnin + 5000:
            Param.T_cuurent[bins] = Param.burnin + 4800
            Param.Nacept[bins] , Param.Nreject[bins] = 1., 1.
    else:
        # rejected
        Param.reject()

        
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
    
class Param_MCMC(object):
    def __doc__(self):
        '''stores params for use in multi_main'''

    def __init__(self, lik_fun, burnin):
        models = lik_fun.models
        # global params (Vals)
        self.eff = -9999999.
        self.burnin = burnin
        self.T_stop =  1.
        # Local but simple (Model->objects as columns)
        self.chi = pd.Panel({model:pd.DataFrame(columns=models[model]
                                                , dtype=float) for model in models})
        self.acept_rate = pd.Panel({model:pd.DataFrame(columns=models[model]
                                                       , dtype=float) for model in models})
        self.active_chi = pd.Panel({model:pd.DataFrame(columns=models[model]
                                                       , dtype=float) for model in models})
        self.mixing_rates = pd.Panel({model:pd.DataFrame(columns=models[model]) for model in models})
        self.T_start = pd.Panel({model:pd.DataFrame(columns=models[model]
                                                      , dtype=float) for model in models})
        # Local but complex (Model->object->multiple colums
        #self.active_param
        #self.out_sigma 
        #self.sigma 
        #self.param        


    def initalize(self, lik_fun):
        '''Initalize certan parms'''
        self.bins = lik_fun.models.keys()[0]
        panel4d_param, panel4d_sigma= {},{}
        for bins in lik_fun.models:
            panel_param, panel_sigma = {},{}
            for gal in lik_fun.models[bins]:
                panel_param[gal], panel_sigma[gal] = lik_fun.initalize_param(gal)
            panel4d_param[bins] = pd.Panel(panel_param)
            panel4d_sigma[bins] = pd.Panel(panel_sigma)
        
        self.param = pd.Panel4D(panel4d_param)
        self.active_param = pd.Panel4D(panel4d_param)
        self.sigma = pd.Panel4D(panel4d_sigma)
        self.out_sigma  =  [self.sigma.copy()]
        for bins in lik_fun.models:
            # check if params are in range
            lik, prior = (lik_fun.lik(self.active_param, bins),
                               lik_fun.prior(self.active_param, bins))
            #get intal params lik and priors
            for Prior, gal in prior:
                if not nu.isfinite(Prior):
                    # Need to try again with param here
                    #return True
                    continue
                self.active_chi[bins][gal] = Prior
            for Lik, gal in lik:
                if not nu.isfinite(Lik):
                    #return True
                    continue
                self.active_chi[bins][gal] += Lik
                self.chi[bins][gal] = self.active_chi[bins][gal].copy()
                self.T_start[bins][gal] = self.active_chi[bins][gal].abs().copy()
                self.acept_rate[bins][gal] = [1.]
        # Set storage params
            self.param[bins] = self.active_param[bins].copy()
            self.SA(0)
        return not nu.isfinite(float(self.chi[bins].sum(1)))

    def fail_recover(self, path):
        '''Loads params from old run'''
        raise NotImplementedError

    def save_chain(self):
        '''Records current chain state'''
        raise NotImplementedError
    
    def save_state(self, option, save_num=500):
        '''Saves current state of chain incase run crashes'''
        # Make db if none created
        if option.current == 0 or not 'con' in vars(self):
            #ipdb.set_trace()
            # check if database exsists in local dir
            path = os.path.realpath('.')
            self._temp_db_path =  os.path.join(path,'Age_mltry_save.db')
            if os.path.exists(self._temp_db_path):
                raise OSError('Recovery data base exsist. Turn on recovery or delete')
            self.con = create_engine('sqlite:///%s'%self._temp_db_path)
            self._vars = vars(self).keys()
            # create tables for different parameters
        if option.current % save_num == 0:
            self._save_db(option)
        
    def delete_temp(self):
        '''Deletes temporary database and other clean ups needed'''
        os.remove(self._temp_db_path)
            
    def _save_db(self, option):
        '''Creates db to save chain state'''
        con = self.con.connect()
        for var in self._vars:
            temp = eval('self.%s'%var)
            if isinstance(temp, pd.DataFrame):
                # dicts with DataFrames mostly for chi
                # does not work with multiple models
                for model in temp:
                    temp[model].to_sql(var, self.con, if_exists='append')
                    
            elif isinstance(temp, pd.Panel) and not isinstance(temp,
                                                               pd.Panel4D) :
                for frame in temp:
                    temp[frame].to_sql(var+'_%s'%frame,
                        self.con, if_exists='append')
                
            elif isinstance(temp, pd.Panel4D):
                    for panel in temp:
                        temp[panel].to_frame().to_sql(var+'_%s'%panel, self.con,
                                                      if_exists='append')
            elif isinstance(temp, (float,int)):
                # save floats as data frames
                temp_DF = pd.DataFrame([temp],columns=['val'])
                temp_DF.to_sql(var, self.con, if_exists='replace')
        
    def singleObjSplit(self):
        '''Checks correlation between params to see if should split'''
        raise NotImplementedError
        
    def accept(self, gal, new_chi):
        '''Accepts current state of chain, active_param get saved in param
        if bin is different then model is changed'''
        if self.mixing_rates[self.bins].empty:
            self.mixing_rates[self.bins][gal] = [1.]
        else:
            self.mixing_rates[self.bins][gal] += 1
        index = self.param[self.bins][gal].shape[0]
        self.param[self.bins][gal].loc[index] = self.active_param[
                    self.bins][gal].loc[0]
        
        if index == self.chi[self.bins].shape[0]:
            #create new row
            self.chi[self.bins].loc[index] = None
        self.chi[self.bins][gal][index] = float(new_chi)+0
        
    def reject(self, gal):
        '''Rejects current state and gets data from memory'''
        index = self.param[self.bins][gal].shape[0]
        self.active_param[self.bins][gal] = self.param[self.bins][gal].tail(1).copy()
        if index == self.chi[self.bins].shape[0]:
            #create new row
            self.chi[self.bins].loc[index] = None
        self.chi[self.bins][gal][index] = self.chi[self.bins][gal][index-1] + 0
        self.active_chi[self.bins][gal] = self.chi[self.bins][gal][index-1] + 0
        self.param[self.bins][gal].loc[index] =self.param[self.bins][gal].loc[index-1].copy()
        
        
    def step(self, fun, num_iter,step_freq=500.):
        '''check if time to change step size'''
        bins = self.bins
        self.sigma[bins] = fun.step_func(self.acept_rate,
                                            self.param, self.sigma, bins)
        
    def cal_accept(self):
        '''Calculates accepance rate'''
        bins = self.bins
        for gal in self.acept_rate[bins]:
            index = self.acept_rate[bins][gal].shape[0]
            self.acept_rate[bins][gal].loc[index] =self.mixing_rates[bins][gal]/float(index)
            

    def reconfigure(self, param_max):
        '''Changes grouping or something'''
        # each param is different
        self.on[self.bins] = self.active_param[self.bins].keys()[0]
        self.on_dict[self.bins] = range(self.bins)
        # make new step size
        self.sigma[self.bins] = {}
        for i in self.on_dict[self.bins]:
            self.sigma[self.bins][i] = nu.eye(len(self.active_param[self.bins][
                self.on[self.bins]]))
        # group on correlation

    def SA(self, chain_number):
        '''Calculates anneeling parameter'''
        bins = self.bins
        if chain_number < self.burnin:
            #check acceptance rate if to high lower temp
            '''if self.acept_rate[bins][-1] > .6 and self.T_start > self.T_stop :
                self.T_start /= 1.05
            if  self.acept_rate[bins][-1] < .06:
                self.T_start *= 2.'''
            # make temp close to chi
            chi_max = self.chi[bins].tail(1).max().abs()
            if nu.any(self.T_start[bins] > chi_max):
                self.T_start[bins][self.T_start[bins] > chi_max] = [chi_max[self.T_start[bins] > chi_max]]
            #calculate anneeling
            self.sa = MC.SA(chain_number, self.burnin, self.T_start[bins], self.T_stop)
            
    def plot_param(self):
        '''Plots chains'''
        import pylab as lab
        data = {}
        for i in self.param.keys():
            # M, E, b, m_i
            out = []
            for j in self.param[i]:
                t = []
                for k in j:
                    t.append(j[k][-2:])
                out.append(nu.ravel(t))
            lab.plot(nu.asarray(out))
            lab.show()
    
            
class MCMCError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
