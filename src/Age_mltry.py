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
import pylab as lab
import ipdb
# import acor
# from memory_profiler import profile
from glob import glob
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
        Param.fail_recover(fail_recover)
    else:
        # initalize and check if param are in range
        timeInit = Time.time()
        while Param.initalize(fun):
            if Time.time() - timeInit > 60.:
                raise MCMCError('Bad Starting position, check params')
    # Start RJMCMC
    while option.iter_stop:
        bins = Param.bins
        if option.current % 50 == 0 and option.current > 0:
            acpt = nu.min([i[-1] for i in Param.acept_rate[bins].values()])
            chi = nu.sum([i[-1] for i in Param.chi[bins].values()])
            
            show = ('acpt = %.2f,log lik = %e, model = %s, steps = %i,ESS = %2.0f'
                    %(acpt, chi, bins, option.current, Param.sa))
            print show
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
        Param.save_state(option.current)
        option.current += 1
        if option.current >= max_iter:
            option.iter_stop = False
    # Finish and return
    fun.exit_signal()
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
        if mh_critera(Param.active_chi[bins][key], new_chi[key], Param.sa ):
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

    def __init__(self, lik_class, burnin):
        self.eff = -9999999.
        self.burnin = burnin
        self._look_back = 500
        self.on_dict, self.on = {}, {}
        self.active_param, self.sigma = {} ,{}
        self.active_chi = {}
        self.acept_rate, self.out_sigma = {},{}
        self.param, self.chi = {}, {}
        self.Nacept, self.Nreject = {},{}
        for bins in lik_class.models:
            self.active_param[bins], self.sigma[bins] = {}, {}
            self.active_chi[bins] = {}
            self.acept_rate[bins], self.out_sigma[bins] = {},{}
            self.param[bins], self.chi[bins] = {}, {}
            self.Nacept[bins], self.Nreject[bins] = {},{}
        
        # to calculate bayes factor
        self.bayes_fact = {}
        # simulated anneling param
        self.T_cuurent = {}
        self.Nexchange_ratio = 1.0
        self.time, self.timeleft = 1, nu.random.exponential(100)
        self.T_stop =  1.
        self.trans_moves = 0
        # bayes_fact[bins] = #something
        # set storage functions

    def initalize(self, lik_fun):
        '''Initalize certan parms'''
        self.bins = lik_fun.models.keys()[0]
        for bins in lik_fun.models:
            # model level
            self.T_cuurent[bins] = 0
            for gal in lik_fun.models[bins]:
                self.active_param[bins][gal], self.sigma[bins][gal] = lik_fun.initalize_param(gal)
                #self.active_chi[bins][gal] = {}
                self.out_sigma[bins][gal]  =  [self.sigma[bins][gal][:]]
            #self.reconfigure(i)
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
        self.T_start = abs(nu.max(self.chi[bins].values()))
        self.SA(0)
        return not nu.all(nu.isfinite(self.chi[bins].values()))

    def fail_recover(self, path):
        '''Loads params from old run'''
        raise NotImplementedError
        
    def save_state(self, path=None):
        '''Saves current state of chain incase run crashes'''
        raise NotImplementedError

    def _create_dir_sturct(self, path):
        '''Create dir structure for failure recovery.
        Each model -> gal or object is giving a dir and
        each varible is given own file. Global vars like sigma will be under
        appropeate places'''
        cur_parent = path
        self.save_path = {}
        # Top is model
        models = self.chi.keys()
        for model in models:
            if not os.path.exists(os.path.join(cur_parent, model)):
                os.mkdir(os.path.join(cur_parent, model))
            cur_parent = os.path.join(cur_parent, model)
            self.save_path[model] = {}
            # Gal or obj
            for gal in self.chi[model]:
                if not os.path.exists(os.path.join(cur_parent, gal)):
                    os.mkdir(os.path.join(cur_parent, gal))
                cur_parent = os.path.join(cur_parent, gal)
                self.save_path[model][gal] = {}
                # Params in each Gal
                for param in vars(self):
                    if param == 'eff' or param == 'save_path':
                        continue
                    # [save_obj, path]
                    cur_parent = os.path.join(cur_parent, param +'.csv')
                    self.save_path[model][gal][param] = [open(cur_parent,'w'),
                                                         cur_parent]
                    cur_parent = os.path.split(cur_parent)[0]
                cur_parent = os.path.split(cur_parent)[0]
            cur_parent = os.path.split(cur_parent)[0]
 
    def save_state(self, itter):
        '''Saves current state of chain incase run crashes'''
        # Make state folder if none created
        save_num = self._look_back
        if itter == 0:
            # Make directory for saving
            if not os.path.exists('save_files'):
                os.mkdir('save_files')
            else:
                raise OSError('Fail recovery exsits. Please delete before running agai')
            self._create_dir_sturct('save_files')
        if itter % save_num == 0 and itter > 0:
            self._save_csv(itter, save_num)
            print 'done'
            
    def singleObjSplit(self):
        '''Checks correlation between params to see if should split'''
        raise NotImplementedError
        
    def accept(self, gal, new_chi):
        '''Accepts current state of chain, active_param get saved in param
        if bin is different then model is changed'''
        if not gal in self.Nacept[self.bins]:
            self.Nacept[self.bins][gal] = 1
        else:
            self.Nacept[self.bins][gal] += 1
        
        self.param[self.bins][gal].append(self.active_param[self.bins][gal].copy())
        self.chi[self.bins][gal].append((new_chi)+0)
        
    def reject(self, gal):
        '''Rejects current state and gets data from memory'''
        #ipdb.set_trace()
        if gal in self.Nreject[self.bins]:
            self.Nreject[self.bins][gal] += 1
        else:
            self.Nreject[self.bins][gal] = 1
        
        self.active_param[self.bins][gal] = self.param[self.bins][gal][-1].copy()
        self.param[self.bins][gal].append(self.param[self.bins][gal][-1].copy())
        self.chi[self.bins][gal].append(self.chi[self.bins][gal][-1].copy())
        self.active_chi[self.bins][gal] = self.chi[self.bins][gal][-1].copy()
        
    def step(self, fun, num_iter,step_freq=500.):
        '''check if time to change step size'''
        bins = self.bins
        if num_iter % step_freq == 0 and num_iter > 0:
            for gal in self.sigma[bins]:
                self.sigma[bins][gal] = fun.step_func(self.acept_rate[bins][gal][-1],
                                            self.param[bins][gal],
                                            self.sigma[bins][gal],self._look_back)
        
    def cal_accept(self):
        '''Calculates accepance rate'''
        bins = self.bins
        for gal in self.chi[bins]:
            if not gal in self.acept_rate[bins]:
                self.acept_rate[bins][gal] = []
            # No Nacept acept_rate = 0
            if not gal in self.Nacept[bins]:
                self.acept_rate[bins][gal].append(0.)
            # No Nreject acept_rate = 1
            elif not gal in self.Nreject[bins]:
                self.acept_rate[bins][gal].append(1.)
            else:
                self.acept_rate[bins][gal].append(self.Nacept[bins][gal] /
                                                float(self.Nacept[bins][gal] +
                                                self.Nreject[bins][gal]))

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
            # make temp close to chi
            chi_max = abs(nu.max([self.chi[bins][gal][-1] for gal in self.chi[bins]]))
            if self.T_start > chi_max:
                self.T_start = chi_max
            #calculate anneeling
            self.sa = MC.SA(chain_number, self.burnin, self.T_start, self.T_stop)
            
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
