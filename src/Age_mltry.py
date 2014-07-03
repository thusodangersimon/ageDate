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
        fun, option, burnin = Param.fail_recover(fail_recover,fun, option)
    else:
        # initalize and check if param are in range
        timeInit = Time.time()
        while Param.initalize(fun):
            if Time.time() - timeInit > 60.:
                raise MCMCError('Bad Starting position, check params')
    # Start RJMCMC
    while option.iter_stop:
        bins = Param.bins
        if option.current % 1 == 0 and option.current > 0:
            acpt = nu.min([i[-1] for i in Param.acept_rate[bins].values()])
            chi = nu.sum([i[-1] for i in Param.chi[bins].values()])
            try:
                show = ('acpt = %.2f,log lik = %e, model = %s, steps = %i,ESS = %2.0f'
                    %(acpt, chi, bins, option.current, nu.min(Param.sa.values())))
                print show
            except:
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
        # simulated anneling param
        self.T_start = {}
        self.sa = {}
        self.T_stop =  1.
        
    def initalize(self, lik_fun):
        '''Initalize certan parms'''
        self.bins = lik_fun.models.keys()[0]
        for bins in lik_fun.models:
            # model level
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
            self.T_start[gal] = abs(nu.max(self.chi[bins].values()))
        self.SA(0)
        self.save_state(0, lik_fun)
        return not nu.all(nu.isfinite(self.chi[bins].values()))

    def fail_recover(self, path, lik_fun, option):
        '''Loads params from old run'''
        out_size = 0
        if isinstance(path, str):
            # Path is a path
            if not os.path.exists(path):
                # Print warning and make save recovery dir
                pass
        else:
            # standard path
            if not os.path.exists('save_files'):
                raise MCMCError('No fail recovery. Turn off in options')
            else:
                path = 'save_files'
        self.save_path = {}
        lik_fun.data = {}
        lik_fun.norm_prior = {}
        # should be 3 layers of files [models,gal,local_parameters]
        walker = os.walk(path)
        dir, models,files = walker.next()
        for model in models:
            self.save_path[model] = {}
        # Get gal
        dir, gals, files = walker.next()
        dir = path
        for model in models:
            self.bins = model
            dir = os.path.join(dir, model)
            # global params
            for globes in files:
                g = globes.split('.')[0]
                if not g in ['burnin']:
                    exec('self.%s = nu.loadtxt(open(os.path.join(dir,globes)))'%g)
                    
                else:
                    burnin = nu.loadtxt(open(os.path.join(dir,globes)))
            for gal in gals:
                dir = os.path.join(dir, gal)
                self.save_path[model][gal] = {}
                files = os.listdir(dir)
                for param in files:
                    p = param.split('.')[0]
                    if p in gals:
                        lik_fun.data[p] = nu.loadtxt(os.path.join(dir,param))
                        lik_fun.norm_prior[p] = 1.
                        # Get header?
                        continue
                    self.save_path[model][gal][p] = []
                    # Open file
                    if p in ['sigma', 'T_start', 'Nacept', 'Nreject']:
                        self.save_path[model][gal][p].append(os.path.join(dir,param))
                    else:
                        self.save_path[model][gal][p].append(open(os.path.join(dir,param),
                                                              'a'))
                    # Path
                    self.save_path[model][gal][p].append(os.path.join(dir,param))
                    # Load param
                    try:
                        if p == 'param':
                            self.param[model][gal] = []
                            temp = pd.DataFrame.from_csv(open(os.path.join(dir,param)),
                                                              sep=' ',index_col=None
                                                              ,header=0)
                            # append into param
                            for i in range(temp.index.max()-self._look_back,
                                           temp.index.max()):
                                self.param[model][gal].append(temp.irow([i]))
                        else:
                            exec('self.%s["%s"]["%s"]'%(p, model, gal)
                            +' = nu.loadtxt(open(os.path.join(dir,param)))')
                            #print 'self.%s["%s"]["%s"]'%(p, model, gal)
                            # Check shape
                            size = eval('self.%s["%s"]["%s"].size'%(p, model, gal))
                            if size > 1 and not p in ['sigma']:
                                out_size = max(out_size,
                                    eval('self.%s["%s"]["%s"].shape[0]'%(p, model, gal)))
                                exec('self.%s["%s"]["%s"] = '%(p, model, gal) +
                                 'self.%s["%s"]["%s"][-self._look_back:]'%(p, model, gal))
                            elif not p in ['sigma']:
                                # change from ndarray(0-d) to float or int
                                #print 'self.%s["%s"]["%s"].shape[0]'%(p, model, gal)
                                exec('self.%s["%s"]["%s"] = '%(p, model, gal) +
                                 'float(self.%s["%s"]["%s"])'%(p, model, gal))
                    except KeyError:
                        exec('self.%s["%s"]'%(p, gal)
                            +' = float(nu.loadtxt(open(os.path.join(dir,param))))')
                        #print 'self.%s["%s"]'%(p, gal)
                    
                    # Remove all but curent few
                dir = os.path.split(dir)[0]
                # set activie param
                self.active_chi[model][gal] = nu.copy(self.chi[model][gal][-1])
                self.chi[model][gal] = [i for i in self.chi[model][gal]]
                self.acept_rate[model][gal] = [i for i in self.acept_rate[model][gal]]
                self.active_param[model][gal] = self.param[model][gal][-1].copy()
            self.active_param[model] = pd.Panel(self.active_param[model])
        option.current = nu.array([[out_size]])
        # if multiprocessing send data
        lik_fun.send_fitting_data()
        self.SA(out_size, True)
        return lik_fun, option, burnin
    
    def _save_csv(self, indici, dump_number):
        '''Saves current state of chain incase run crashes'''
        #ipdb.set_trace()
        for model in self.save_path:
            for gal in self.save_path[model]:
                for param in self.save_path[model][gal]:
                    if param in ['T_start']:
                        save_param = eval('self.%s["%s"]'%(param,gal))
                    else:
                        try:
                        
                            save_param = eval('self.%s["%s"]["%s"]'%(param,
                                                                    model, gal))
                        except KeyError:
                            print 'self.%s["%s"]["%s"] Does not exsist.'%(param
                                                                ,model,gal)
                        except:
                            ipdb.set_trace()
                    if isinstance(save_param, (nu.ndarray,list)):
                        #check contents
                        
                        if isinstance(save_param[0], (float, nu.ndarray, list)):
                            #check if
                            nu.savetxt(self.save_path[model][gal][param][0],
                                save_param[:-1])
                        elif isinstance(save_param[0], pd.DataFrame):
                            nu.savetxt(self.save_path[model][gal][param][0],
                                    [i.values[0] for i in save_param[:-1]])
                        # Save only current val
                        if param in ['sigma']:
                            nu.savetxt(self.save_path[model][gal][param][0],save_param)
                            continue
                        exec('self.%s["%s"]["%s"]=[save_param[-1]]'%(param, model, gal))
                    elif isinstance(save_param, (float,int)):
                        nu.savetxt(self.save_path[model][gal][param][0],
                                   [save_param])
                    else:
                        ipdb.set_trace()
                    if isinstance(self.save_path[model][gal][param][0],str):
                        continue
                    self.save_path[model][gal][param][0].flush()

    def _create_dir_sturct(self, path, lik):
        '''Create dir structure for failure recovery.
        Each model -> gal or object is giving a dir and
        each varible is given own file. Global vars like sigma will be under
        appropeate places'''
        cur_parent = path
        save_list = ['acept_rate', 'chi', 'sigma' ,'param'
                     ,'T_start', 'Nacept', 'Nreject'] 
        self.save_path = {}
        # Top is model
        models = self.chi.keys()
        for model in models:
            if not os.path.exists(os.path.join(cur_parent, model)):
                os.mkdir(os.path.join(cur_parent, model))
            cur_parent = os.path.join(cur_parent, model)
            self.save_path[model] = {}
            # save model param
            for glob_modle in ['T_stop', 'burnin']:
                nu.savetxt(os.path.join(cur_parent,glob_modle + '.csv'),
                           [eval('self.%s'%glob_modle)])
                
            # Gal or obj
            for gal in self.chi[model]:
                if not os.path.exists(os.path.join(cur_parent, gal)):
                    os.mkdir(os.path.join(cur_parent, gal))
                cur_parent = os.path.join(cur_parent, gal)
                self.save_path[model][gal] = {}
                # save fitting data
                nu.savetxt(os.path.join(cur_parent, gal+ '.csv'), lik.data[gal]
                           ,header='wavelength, flux*%2.2f'%lik.norm_prior[gal])
                # Params in each Gal
                for param in vars(self):
                    if not param in save_list :
                        continue
                    # [save_obj, path]
                    cur_parent = os.path.join(cur_parent, param +'.csv')
                    # whether to append
                    self.save_path[model][gal][param] =[]
                    appender = self.save_path[model][gal][param].append
                    if param in ['sigma','T_start', 'Nacept', 'Nreject']:
                        # Overwrite
                        appender(cur_parent)
                    else:
                        # Append
                        appender(open(cur_parent,'a'))
                    appender(cur_parent)
                    if param == 'param':
                        # Save header
                        writer = self.save_path[model][gal][param][0].write
                        writer(' '.join(self.param[model][gal][0].columns)+'\n')
                        self.save_path[model][gal][param][0].flush()
                    cur_parent = os.path.split(cur_parent)[0]
                cur_parent = os.path.split(cur_parent)[0]
            cur_parent = os.path.split(cur_parent)[0]
 
    def save_state(self, itter, lik=None):
        '''Saves current state of chain incase run crashes'''
        # Make state folder if none created
        save_num = self._look_back
        if itter == 0 and not lik is None:
            # Make directory for saving
            if not os.path.exists('save_files'):
                os.mkdir('save_files')
            else:
                raise OSError('Fail recovery exsits. Please delete before running again')
            self._create_dir_sturct('save_files', lik)
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
        #if num_iter % step_freq == 0 and num_iter > 0:
        if num_iter > 10:
            for gal in self.sigma[bins]:
                
                self.sigma[bins][gal] = fun.step_func(self.acept_rate[bins][gal][-1],
                                            self.param[bins][gal],
                                            self.sigma[bins][gal],num_iter)
        
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

    def SA(self, chain_number, fail_recover=False):
        '''Calculates anneeling parameter'''
        bins = self.bins
        if chain_number < self.burnin or fail_recover:
            for gal in self.T_start:
                # make temp close to chi
                chi_max = abs(nu.max(self.chi[bins][gal]))
                if self.T_start[gal] > chi_max:
                    self.T_start[gal] = chi_max
                #calculate anneeling
                self.sa[gal] = MC.SA(chain_number, self.burnin, self.T_start[gal],
                                     self.T_stop)
            
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
