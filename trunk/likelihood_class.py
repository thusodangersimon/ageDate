#!/usr/bin/env python
#
# Name:  likelihood_class
#
# Author: Thuso S Simon
#
# Date: 25 of April, 2013
#TODO: 
#
#    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#    Copyright (C) 2013 Thuso S Simon
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
""" Likelihood classes for running of MCMC, RJMCMC and other fitting methods. 
First class is and example and has all required methods needed to run MCMC or
RJMCMC. Also has specific classes for use of spectral fitting"""

import numpy as nu
from glob import glob
import Age_date as ag
import ezgal as gal
import scipy.stats as stats_dist
import multiprocessing as multi
from itertools import izip

class Example_lik_class(object):

    '''exmaple class for use with RJCMCM or MCMC program, all methods are
    required and inputs are required till the comma, and outputs are also
    not mutable. The body of the class can be filled in to users delight'''

    def __init__(self,):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, can do whatever you want. User to define functions'''
        #return #whatever you want or optional
        pass


    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param 
        pass

    def lik(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        
        #return loglik
        pass

    def prior(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        pass


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        pass

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

        Used to initalize all starting points for run of RJMCMC and MCMC.
        outputs starting point and starting step size'''
        #return init_param, init_step
        pass

        
    def step_func(self,step_crit,param,step_size,model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        #return new_step
        pass

    def birth_death(self,birth_rate, model, param):
        '''(Example_lik_class, float, any type, dict(ndarray)) -> 
           dict(ndarray), any type, bool, float

        For RJMCMC only. Does between model move. Birth rate is probablity to
        move from one model to another, models is current model and param is 
        dict of all localtions in param space. 
        Returns new param array with move updated, key for new model moving to,
        whether of not to attempt model jump (False to make run as MCMC) and the
        Jocobian for move.
        '''
        #for RJCMC
        #return new_param, try_model, attemp_jump, Jocobian
        #for MCMC
        #return None, None, False, None
        pass

#=============================================
#spectral fitting with RJCMC Class
class VESPA_fit(object):
    '''Finds the age, metalicity, star formation history, 
    dust obsorption and line of sight velocity distribution
    to fit a Spectrum.

    Uses vespa methodology splitting const sfh into multiple componets
    '''
    def __init__(self,data, min_sfh=1,max_sfh=16,lin_space=False,use_dust=True, use_losvd=True, spec_lib='p2',imf='salp',spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
        '''(VESPA_fitclass, ndarray,int,int) -> NoneType
        data - spectrum to fit
        *_sfh - range number of burst to allow
        lin_space - make age bins linearly space or log spaced
        use_* allow useage of dust and line of sigt velocity dispersion
        spec_lib - spectral lib to use
        imf - inital mass function to use
        spec_lib_path - path to ssps
        sets up vespa like fits
        '''
        self.data = data
        #load models
        cur_lib = ['basti', 'bc03', 'cb07','m05','c09','p2']
        assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + str(cur_lib))
        if not spec_lib_path.endswith('/') :
            spec_lib_path += '/'
        models = glob(spec_lib_path+spec_lib+'*'+imf+'*')
        if len(models) == 0:
            models = glob(spec_lib_path+spec_lib.lower()+'*'+imf+'*')
        assert len(models) > 0, "Did not find any models"
        #crate ezgal class of models
        SSP = gal.wrapper(models)
        #make all burst modes that can be used
        age = SSP.sed_ages
        self.models = {}
        print 'Initalizing models. Please wait'
        for i in range(min_sfh, max_sfh+1):
            models = []
            if lin_space:
                #lin space
                lenght = age.ptp()/float(i)/10**9
            else:
                lenght = nu.exp(nu.log(age).ptp()/float(i))
            for j in SSP:
                models.append(j.make_burst(lenght))
            models = class_map(
            self.models['burst'+str(i)] = [['metal', 'sfr']*i,gal.wrapper(models)]
            print 'Finished %i out of %i models' %(i,max_sfh)
        #check to see if properties are the same
        self._metal_unq = nu.float64(SSP['met'])
        self._age_unq = nu.copy(SSP.sed_ages)/10.**9
 

    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param 
        pass

    def lik(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        
        #return loglik
        pass

    def prior(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        pass


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        pass

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

        Used to initalize all starting points for run of RJMCMC and MCMC.
        outputs starting point and starting step size'''
        #return init_param, init_step
        pass

        
    def step_func(self,step_crit,param,step_size,model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        #return new_step
        pass

    def birth_death(self,birth_rate, model, param):
        '''(Example_lik_class, float, any type, dict(ndarray)) -> 
           dict(ndarray), any type, bool, float

        For RJMCMC only. Does between model move. Birth rate is probablity to
        move from one model to another, models is current model and param is 
        dict of all localtions in param space. 
        Returns new param array with move updated, key for new model moving to,
        whether of not to attempt model jump (False to make run as MCMC) and the
        Jocobian for move.
        '''
        #for RJCMC
        #return new_param, try_model, attemp_jump, Jocobian
        #for MCMC
        #return None, None, False, None
        pass

class Spectral_fit(object):
    '''Finds the age, metalicity, star formation history, 
    dust obsorption and line of sight velocity distribution
    to fit a Spectrum. 
    '''

    def __init__(self,data, use_dust=True, use_losvd=True, spec_lib='p2',imf='salp',spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, initalize spectal func, put nx2 or nx3 specta
        ndarray (wave,flux,uncert (optional)).

        use_ tells if you want to fit for dust and/or line of sight
        velocity dispersion.
        
        spec_lib is the spectral lib to use. models avalible for use:
        BaSTI - Percival et al. 2009 (ApJ, 690, 472)
        BC03 - Bruzual and Charlot 2003 (MNRAS, 344, 1000)
        CB07 - Currently unpublished. Please reference as an updated BC03 model.
        M05 - Maraston et al. 2005 (MNRAS, 362, 799)
        C09 - Conroy, Gunn, and White 2009 (ApJ, 699, 486C) and Conroy and Gunn 2010 (ApJ, 712, 833C (Please cite both)
        PEGASE2 (p2) - Fioc and Rocca-Volmerange 1997 (A&A, 326, 950)
        More to come!'''
        
        #initalize data and make ezgal class for uses
        self.data = nu.copy(data)
        #check data, reduice wavelenght range, match wavelengths to lib
        #get all ssp libs with spec_lib name
        cur_lib = ['basti', 'bc03', 'cb07','m05','c09','p2']
        assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + str(cur_lib))
        if not spec_lib_path.endswith('/') :
            spec_lib_path += '/'
        models = glob(spec_lib_path+spec_lib+'*'+imf+'*')
        if len(models) == 0:
            models = glob(spec_lib_path+spec_lib.lower()+'*'+imf+'*')
        assert len(models) > 0, "Did not find any models"
        #crate ezgal class of models
        self.SSP = gal.wrapper(models)
        #check to see if properties are the same
        self._metal_unq = nu.float64(self.SSP['met'])
        self._age_unq = nu.copy(self.SSP.sed_ages)/10.**9
        #make keys for models (all caps=required, lower not required
        #+ means additive modesl, * is multiplicative or convolution
        self.get_sed = lambda x: x[2] * self.SSP.get_sed(x[0],x[1])
        self.models = {'SSP+':[['age','metal','norm'],self.get_sed],
			'dust*':[['tbc','tsm'],ag.dust]}
        #set values for priors
        
    def _model_handeler(self,models):
        '''(Example_lik_class, str) -> str
        
        Not called by RJMMCMC or MCMC, but handels how models interact
        '''
        pass
		
    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        out = []
        for i in xrange(len(mu)):
            out.append(nu.random.multivariate_normal(mu[i],sigma[i]))

        return out

    def lik(self,param,model):
        '''(Example_lik_class, ndarray, str) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        #get model
        imodel = []
		#get additive models
        for i,j in enumerate(model.split(',')):
            if j.endswith('+'):
                try:
                    imodel.append(self.models[j][1](param[model][i]))
                except ValueError:
                    return -nu.inf

		#combine data with
		imodel = nu.sum(imodel,0)
        #apply multipliticave or convolution models
        for i,j in enumerate(model.split(',')):
            if j.endswith('*'):
				imodel = self.models[j][1](imodel,param[model][i])

        #make model and data have same wavelength
        
        #get likelyhood
        out = stats_dist.norm.logpdf(self.data[:,1],nu.sum(imodel,0))
        #return loglik
        return out.sum()

    def prior(self,param, model):
        '''(Example_lik_class, ndarray, str) -> float
        Calculates log-probablity for prior'''
        #return logprior
        #uniform
        out = 0
        for i,j in enumerate(model.split(',')):
            if j == 'SSP':
            #'age':
                loc = self._age_unq.min()
                scale = self._age_unq.ptp()
                out += stats_dist.uniform.logpdf(param[model][i][0],loc,scale).sum()
            #'metal':
                loc = self._metal_unq.min()
                scale = self._metal_unq.ptp()
                out += stats_dist.uniform.logpdf(param[model][i][1], loc, scale).sum()
            #'norm':
                out += stats_dist.uniform.logpdf(param[model][i][2],0,10**4).sum()
        return out
        #conj of uniform
        #stats_dist.pareto.logpdf
        #normal
        #stats_dist.norm.logpdf
        #multinormal conjuigates
        #stats_dist.gamma.logpdf
        #stats_dist.invgamma.logpdf
        #exponetal
        #stats_dist.expon
        #half normal (never negitive)
        #stats_dist.halfnorm
        


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        return 0.

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

        Used to initalize all starting points for run of RJMCMC and MCMC.
        outputs starting point and starting step size'''
        if model == 'SSP':
            out_ar, outsig = nu.zeros(3), nu.identity(3)
            loc = self._age_unq.min()
            scale = self._age_unq.ptp()
            out_ar[0] =  stats_dist.uniform.rvs(loc,scale)
            #metal
            loc = self._metal_unq.min()
            scale = self._metal_unq.ptp()
            out_ar[1] = stats_dist.uniform.rvs(loc, scale)
            #normalization
            out_ar[2] =  stats_dist.uniform.rvs(0,10**4)
            return out_ar, outsig
        elif model == 'dust':
            pass
        else:
            raise KeyError("Key dosen't exsist")

        
    def step_func(self, step_crit, param, step_size, model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        #return new_step
        if step_crit > .60:
            for i in range(len(model.split(','))):
                step_size[model][i] *= 1.05
        elif step_crit < .2:
            for i in range(len(model.split(','))):
                step_size[model][i] /= 1.05
        #cov matrix
        if len(param) % 2000 == 0:
            step_size[model] = [nu.cov(nu.asarray(param[-2000:])[:,0,:].T)]
        return step_size[model]


    def birth_death(self,birth_rate, model, param):
        '''(Example_lik_class, float, any type, rj_dict(ndarray)) -> 
           dict(ndarray), any type, bool, float

        For RJMCMC only. Does between model move. Birth rate is probablity to
        move from one model to another, models is current model and param is 
        dict of all localtions in param space. 
        Returns new param array with move updated, key for new model moving to,
        whether of not to attempt model jump (False to make run as MCMC) and the
        Jocobian for move.
        '''
        #for RJCMC
        if birth_rate > nu.random.rand():
            #birth
            #choose random model to add
            new_model = self.models.keys()[1]
            out_param = param + {new_model:[self.initalize_param(new_model)[0]]}
            new_model = out_param.keys()[0]
            
        else:
            #death
            if len(param[param.keys()[0]]) > 1:
                out_param = param - 'SSP'
                new_model = out_param.keys()[0]
            else:
                return param, model, False, 1.

        return out_param, new_model, True, 1.
        #return new_param, try_model, attemp_jump, Jocobian
        #for MCMC
        #return None, None, False, None
        
#######other functions

#used for class_map
def spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i == None:
                break
            q_out.put((i,f(x)))
    return fun

def parmap(f, X, nprocs = multi.cpu_count()):
    q_in   = multi.Queue(1)
    q_out  = multi.Queue()

    proc = [multi.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

def test(fun,*iters):

    for i in iters:
        print fun, i
