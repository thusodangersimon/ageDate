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
class Spectral_fit(object):
    '''Finds the age, metalicity, star formation history, 
    dust obsorption and line of sight velocity distribution
    to fit a Spectrum. 
    '''

    def __init__(self,data, use_dust=True, use_losvd=True, spec_lib='Bc03',imf='chab',spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
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
        PEGASE2 - Fioc and Rocca-Volmerange 1997 (A&A, 326, 950)
        More to come!'''
        
        #initalize data and make ezgal class for uses
        self.data = nu.copy(data)
        #get all ssp libs with spec_lib name
        cur_lib = ['basti', 'bc03', 'cb07','m05','c09','pegase2']
        assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + cur_lib)
        if not spec_lib_path.endswith('/') :
            spec_lib_path += '/'
        models = glob(spec_lib_path+spec_lib+'*'+imf+'*')
        if len(models) == 0:
            models = glob(spec_lib_path+spec_lib.lower()+'*'+imf+'*')
        assert len(models) > 0, "Did not find any models"
        #crate ezgal class of models
        self.SSP = gal.wrapper(models)
        #check to see if properties are the same
        

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
