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
    def __init__(self,data, min_sfh=1,max_sfh=16,lin_space=False,use_dust=True, 
		use_losvd=True, spec_lib='p2',imf='salp',
			spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
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
        self.data = nu.copy(data)
		#make mean value of data= 100
        self._norm = 1./(self.data[:,1].mean()/100.)
        self.data[:,1] *= self._norm
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
        #extract seds from ezgal wrapper
        spect, info = [SSP.sed_ls], []
        for i in SSP:
			metal = float(i.meta_data['met'])
			ages = nu.float64(i.ages)
			for j in ages:
				if j == 0:
					continue
				spect.append(i.get_sed(j,age_units='yrs'))
				info.append([metal+0,j])
        info,self._spect = [nu.log10(info),None],nu.asarray(spect).T
        #test if sorted
        self._spect = self._spect[::-1,:]
        #make spect match wavelengths of data
        self._spect = ag.data_match_all(data,self._spect)[0]
		#set hidden varibles
        self._lib_vals = info
        self._age_unq = nu.unique(info[0][:,1])
        self._metal_unq = nu.unique(info[0][:,0])
        self._lib_vals[0][:,0] = 10**self._lib_vals[0][:,0]
        self._min_sfh, self._max_sfh = min_sfh,max_sfh +1
		#params
        self.curent_param = nu.empty(2)
        self.models = {}
        for i in xrange(min_sfh,max_sfh+1):
            self.models[str(i)]= ['burst_length','mean_age', 'metal','norm'] * i
        #multiple_block for bad performance
        self._multi_block = False

		
    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
		Proposal distribution, draws steps for chain. Should use a symetric
		distribution
        '''
		#save length and mean age they don't change  self._multi_block
        try:
            t_out = nu.random.multivariate_normal(nu.ravel(mu),sigma)
        except nu.linalg.LinAlgError:
            print sigma
            raise
        self._sigma = nu.copy(sigma)
        bins = sigma.shape[0]/4
        t_out = nu.reshape(t_out, (bins, 4))
        #t_out[:,:2] = nu.abs(t_out[:,:2])
        #set length and age back to original and make norm positive
        for i,j in enumerate(mu):
            t_out[i][:2] = j[:2]
            #t_out[i][-1] = t_out[i][-1])
        bins = str(bins)
        t_out = self._make_square({bins:t_out},bins)
        return t_out
        
    def multi_try_all(self,param,bins,N=15):
        '''(VESPA_class,ndarray,str,int)-> float
        Does all things for multi try (proposal,lik, and selects best param'''
        temp_param = map(self.proposal,[param]*N, [self._sigma]*N)
    
        
    def lik(self,param, bins,return_all=False):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        if not nu.any(self._make_square(param,bins) == param[bins]):
            return -nu.inf
        burst_model = {}
        for i in param[bins]:
            burst_model[str(i[1])] =  10**i[3]*ag.make_burst(i[0],i[1],i[2],
            self._metal_unq, self._age_unq, self._spect, self._lib_vals)
		#do dust

		#do losvd

		#get loglik
        model = nu.sum(burst_model.values(),0)
		#return loglik
        if self.data.shape[1] == 3:
            #uncertanty calc
            pass
        else:
            prob = stats_dist.norm.logpdf(model,self.data[:,1]).sum()
            #prob = -nu.sum((model -	self.data[:,1])**2)
        #return
        if 	return_all:
            return prob, model
        else:
            return prob

    def prior(self,param,bins):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        out = 0.
        #uniform priors
        for i in param[bins]:
            #length
            out += stats_dist.uniform.logpdf(i[0],0.,self._age_unq.ptp())
            #age
            out += stats_dist.uniform.logpdf(i[1],self._age_unq.min(),self._age_unq.ptp())
            #metals 
            out += stats_dist.uniform.logpdf(i[2],self._metal_unq.min(),self._metal_unq.ptp())
            #weight
            out += stats_dist.uniform.logpdf(i[3], -300, 500)
        
        return out


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        return 0.

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

		Used to initalize all starting points for run of RJMCMC and MCMC.
		outputs starting point and starting step size
        '''
        #any amount of splitting
        out = []
        age, metal, norm =  self._age_unq,self._metal_unq, self._norm
        lengths = self._age_unq.ptp()/float(model)
        for i in range(int(model)):
            out.append([lengths, (i+.5)*lengths + age.min(),0,nu.log10(self._norm*nu.random.rand())])
            #metals
            out[-1][2] = nu.random.rand()*metal.ptp()+metal.min()
        out = nu.asarray(out)
        #make step size
        sigma = nu.identity(out.size)

        #make coorrect shape
        out = self._make_square({model:out},model)
        return out, sigma

        
    def step_func(self,step_crit,param,step_size,model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        if step_crit > .60:
            step_size[model] *= 1.05
        elif step_crit < .2 and nu.any(step_size[model].diagonal() > 10**-6):
            step_size[model] /= 1.05
        #cov matrix
        if len(param) % 200 == 0 and len(param) > 0.:
            new_shape = nu.prod(nu.asarray(param[0]).shape)
            temp = nu.cov(nu.asarray(param[-200:]).reshape(200,new_shape).T)
            #make sure not stuck
            if nu.any(temp.diagonal() > 10**-6):
                step_size[model] = temp
        if step_crit < .18:
            #print step_size[model][0].diagonal()
            #self._multi_blocpass
            step_size[model][step_size[model] < 10**-3] = 10**-3
            
        
        return step_size[model]


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
        return param, None, False, None
        step = nu.random.choice(['birth','split','merge','death'],p=birth_rate)
        if step == 'birth' and int(model) + 1 < self._max_sfh:
            new_param, jacob, temp_model = self._birth(param,model)
        elif step == 'death' and int(model) - 1 > self._min_sfh :
            new_param, jacob, temp_model = self._death(param,model)
        elif step == 'merge' and int(model) - 2 > self._min_sfh:
            new_param, jacob, temp_model = self._merge(param,model)
        elif step == 'split' and int(model) + 1 < self._max_sfh:
            new_param, jacob, temp_model = self._split(param,model)
        else:
            #failed, return nulls
            return param, None, False, None
        param[temp_model] = new_param
        param[temp_model] = self._make_square(param,temp_model)
        return param, temp_model, True, jacob #need to sort out jacobian

    def _merge(self,param,model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Combines nearby bins together.
        '''
        temp_model = str(int(model) - 1)
        #get lowest weight and give random amount to neighbor
        #prob = nu.abs(1/(param[model][:,-1] - param[model][:,-1].min()+1))
        #inverse prob
        #prob /= prob.sum()
        index = nu.where(param[model][:,-1] ==
                        nu.random.choice(param[model][:,-1]))[0]
        index = int(index)
        new_param = []
        split = param[model][index]
        u = nu.random.rand()
        num_comb = nu.random.randint(1,3)
        comb = 0
        if index - 1 > -1 :
            #combine with younger
            comb += 1                 
            temp = param[model][index-1]
            new_metal = [self._metal_unq.min()+nu.random.rand()*self._metal_unq.ptp(), temp[2] + split[2] * u, temp[2]+0.,split[2]+0.]
            new_param.append([temp[0] + split[0] * u, 0., 
                                new_metal[nu.random.randint(4)],
                                temp[3] + split[3] * u])
            new_param[-1][1] = temp[1] + (split[0] * u)/2.
            #make sure metalicity is in bounds
            if new_param[-1][2] < self._metal_unq.min():
                new_param[-1][2] = self._metal_unq.min() + 0.
            if new_param[-1][2] > self._metal_unq.max():
                new_param[-1][2] = self._metal_unq.max() + 0.

        if index + 1 < int(model) and comb != num_comb:
                #combine with older
                temp = param[model][index+1]
                new_metal = [self._metal_unq.min()+nu.random.rand()*self._metal_unq.ptp(), temp[2] + split[2] * (1-u), temp[2]+0.,split[2]+0.]
                new_param.append([temp[0] + split[0] * (1-u), 0.,
                                new_metal[nu.random.randint(4)],
                                temp[3] + split[3] * u])
                new_param[-1][1] =  temp[1] - (split[0]*(1-u))/2.
                #make sure metalicity is in bounds
                if new_param[-1][2] < self._metal_unq.min():
                    new_param[-1][2] = self._metal_unq.min() + 0.
                if new_param[-1][2] > self._metal_unq.max():
                    new_param[-1][2] = self._metal_unq.max() + 0.
        for i in range(param[model].shape[0]):
            if i in range(index-1,index+2) or i == index:
                continue
            new_param.append(nu.copy(param[model][i]))
        #set for output
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = 2.**(-int(model))
            
        return nu.asarray(new_param), jacob, temp_model
        
    def _birth(self,param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Creates a new bins, with parameters randomized.
        '''
        new_param = []
        age, metal, norm =  self._age_unq,self._metal_unq, self._norm
        lengths = self._age_unq.ptp()/float(model) * nu.random.rand()
        new_param.append([lengths, 0.,0,nu.log10(self._norm*nu.random.rand())])
        #metals
        new_param[-1][2] = nu.random.rand()*metal.ptp()+metal.min()
        #age
        new_param[-1][1] = age.min()+nu.random.rand()*age.ptp()
        for i in param[model]:
            new_param.append(nu.copy(i))
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = 2**(int(temp_model))
        
        return nu.asarray(new_param), jacob, temp_model
    
    def _split(self, param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Splits a bin into 2 with a random weight.
        '''
        #split component with prob proprotional to weights
        temp_param = nu.reshape(param[model], (int(model), 4))
        #prob = nu.abs(param[model][:,-1] - param[model][:,-1].min())
        #prob /= prob.sum()
        #if prob.sum() != 1. or nu.any(prob > 0):
        #        prob[:] = 1.
        #        prob /= prob.sum()
        index = nu.where(param[model][:,-1] ==
                             nu.random.choice(param[model][:,-1]))[0]
        index = int(index)
        u = nu.random.rand()
        #[len,t,metal,norm]
        new_metal = [self._metal_unq.min()+nu.random.rand()*self._metal_unq.ptp(), temp_param[index,2] + 0]

        new_param = ([[temp_param[index,0]/2., 
                           (2*temp_param[index,1] - temp_param[index,0]/2.)/2.,
                           new_metal[nu.random.randint(2)], temp_param[index,3] * u]])

        new_param.append([temp_param[index,0]/2.,
                              (2*temp_param[index,1] + temp_param[index,0]/2.)/2.,
                              new_metal[nu.random.randint(2)], temp_param[index,3] * (1 -u)])
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = 2.**int(model)
        return nu.asarray(new_param), jacob, temp_model
    
    def _death(self,param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Removes a bin from array.
        '''
        index = nu.random.randint(int(model))
        new_param = []
        for i in range(param[model].shape[0]):
            if i == index: 
                continue
            new_param.append(nu.copy(param[model][i]))
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = 2.**(-int(temp_model))

        return nu.asarray(new_param), jacob, temp_model
    
    def _make_block(self,param,bins):
        '''VESPA_Class,dict(list),str) ->NoneType
        Groups parameters into correlated blocks to help with mixing rate.
        Uses correlation to group parameters
        '''
        pass
    #tests

    
    def _make_square(self,param,key):
        '''(dict(ndarray),str)-> ndarray
        Makes lengths and ages line up,sorts and makes sure length covers all length
        '''
        #sort params by age
        out = nu.copy(param[key][nu.argsort(param[key][:,1]),:])
        #check that lengths are correct 
        if not out[:,0].sum() == self._age_unq.ptp():
            out[:,0] =out[:,0]/out[:,0].sum()
            out[:,0] *= self._age_unq.ptp()

        #and cover full range of age space
        for i in range(int(key)):
            if i == 0:
                out[i,1] = self._age_unq.min()+ out[i,0]/2.

            else:
                out[i,1] = self._age_unq.min()+out[:i,0].sum()
                out[i,1] += out[i,0]/2.
                
        return out
    
    def make_sfh_plot(self,param, model=None):
        '''(dict(ndarray)-> Nonetype
        Make plot of sfh vs log age of all models in param 
        '''
        import pylab as lab
        if not model is None:
            x,y = [], []
            for i in param[model]:
                x.append(i[1]-i[0]/2.)
                x.append(i[1]+i[0]/2.)
                y.append(i[3])
                y.append(i[3])
            lab.plot(x,y,label=model)
            lab.legend()
            lab.show()
        else:
            for i in param.keys():
                pass

    def make_numeric(self, age, sfh, max_bins, metals=None, return_param=False):
        '''(ndarray,ndarray,ndarray,int,ndarray or None) -> ndarray
        Like EZGAL's function but with a vespa like framework.
        Bins function into flat chunks of average of SFH of function in that range.
        Can also put a function for metalicity to, otherwise will be random.
        '''
        age, sfh = nu.asarray(age), nu.asarray(sfh)
        assert age.shape[0] == sfh.shape[0], 'Age and SFH must have same shape'
        if max_bins > len(age):
            print 'Warning, more bins that ages not supported yet'
            
        #split up age range and give lengths
        param = []
        length = self._age_unq.ptp()/max_bins
        bins = nu.histogram(age,bins=max_bins)[1]
        #need to get length, t, norm and metals if avalible
        length, t, norm, metal = [], [] ,[] ,[]
        for i in range(0,len(bins)-1):
            index = nu.searchsorted(age,[bins[i],bins[i+1]])
            length.append(age[index[0]:index[1]].ptp())
            t.append(age[index[0]:index[1]].mean())
            norm.append(sfh[index[0]:index[1]].mean())
            if metals is None:
                #no input metal choose random
                metal.append(nu.random.choice(nu.linspace(self._metal_unq.min(),self._metal_unq.max())))
            else:
                pass
        #normalize norm to 1
        norm = nu.asarray(norm)/nu.sum(norm)
        #get bursts
        out = nu.zeros_like(self._spect[:,:2])
        out[:,0] = self._spect[:,0].copy()
        param = []
        for i in range(len(t)):
            if return_param:
                param.append([length[i],t[i],metal[i],norm[i]])
            out[:,1] += norm[i] * ag.make_burst(length[i],t[i],metal[i]
                          ,self._metal_unq, self._age_unq, self._spect
                          , self._lib_vals)
        if return_param:
            return out, nu.asarray(param)
        else:
            return out
        
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

def parmap(f, *X):
    nprocs = multi.cpu_count()
    q_in   = multi.Queue(1)
    q_out  = multi.Queue()

    proc = [multi.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(zip(*X))]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]
