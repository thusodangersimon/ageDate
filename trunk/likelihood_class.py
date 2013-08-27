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
from scipy.cluster.hierarchy import fcluster,linkage
import os, sys
#meqtrees stuff
try:
    import pyrap.tables
    import scipy.constants as sc
    from Timba import dmi
    from Timba.Meq import meq
    from Timba.Apps import meqserver
    from Timba.TDL import Compile
    from Timba.TDL import TDLOptions
except ImportError:
    pass

np=nu

class Example_lik_class(object):

    '''exmaple class for use with RJCMCM or MCMC program, all methods are
    required and inputs are required till the comma, and outputs are also
    not mutable. The body of the class can be filled in to users delight'''

    def __init__(self,):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, can do whatever you want. User to define functions'''
        #return #whatever you want or optional
        #needs to have the following as the right types
        
        #self.models = {'name of model':[param names or other junk],'name2':nu.asarray([junk])} #initalizes models
        pass


    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param 
        pass

    def lik(self,param,bins):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        
        #return loglik
        pass

    def prior(self,param,bins):
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

#===========================================    
#catacysmic varible fitting
class CV_Fit(object):

    '''Fits cv spectrum using fortran codes to generate model spectra.
    Runs with only MCMC'''

    def __init__(self,data,model_name='thuso',spec_path='/home/thuso/Phd/other_codes/Valerio'):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, can do whatever you want. User to define functions'''
        self.data = data
        #move to working dir
        os.chdir(spec_path)
        #load in conf files and store
        self.conf_file_name = '%i.5'%os.getpid()
        temp = open(model_name + '.5')
        #make temp dir
        if not os.path.exists('temp/'):
            os.mkdir('temp/')
        batch_file = open('temp/'+self.conf_file_name,'wr+')
        self.org_file = []
        for i in temp:
            batch_file.write(i)
            self.org_file.append(i)
        batch_file.flush()
        batch_file.seek(0)
        self.temp_model = 'temp/'+self.conf_file_name
        #find number of abn are used
        while batch_file.next() != '* mode abn modpf\n':
            pass
        self._no_abn = 0
        for i in batch_file :
            if i.lstrip().split(' ' )[0] == '2':
                #count
                self._no_abn += 1
            elif i.startswith('*'):
                #if finished with section
                break
        batch_file.close()
        self.models = {'mcmc':[2+self._no_abn]}
            
    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param 
        pass

    def lik(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        #param = [T,g,abn...]
        #overwrite new param to temp file
        batch_file = open(self.temp_model,'wr+')
        #set temp, g
        batch_file.write(' %2.1f   %2.1f\n' %(param[0],param[1]))
        #write to abn
        i = 1
        while self.org_file[i] != '* mode abn modpf\n':
            batch_file.write(self.org_file[i])
            i += 1
        batch_file.write(self.org_file[i])
        #set abn if mode == 2
        j = 2
        i += 1
        for ii,k in enumerate(self.org_file[i:]):
            #print i.split(' ' )
            if k.lstrip().split(' ' )[0] == '2':
                #find what to add at end
                adn = k.rstrip().split(' ')[-1]
                if not adn.isalpha():
                    #make sure it's just a letter
                    adn = adn.split()[-1]
                    if not adn.isalpha():
                        #still not a letter?
                        adn = adn.split()[-1].replace('!','')
                #write
                batch_file.write('2 %1.1f 0\t! %s\n'%(param[j],adn))
                j += 1
            elif k.startswith('*'):
                #if finished with section
                break
            else:
                #write non param
                batch_file.write(k)
        #write remaining file
        for k in range(i+ii,len(self.org_file)):
            batch_file.write(self.org_file[k])
        batch_file.close()
        #call Tl on Tl path
        #os.popen()
        #return loglik
        

    def prior(self,param):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        out = 0.
        #param = [T,g,abn...]
        #uniform priors
        #T prior
        out += stats_dist.uniform.logpdf(param[0],2*10**4,4*10**4)
        #g
        out += stats_dist.uniform.logpdf(i[1],4,8)
        #abns
        out += stats_dist.uniform.logpdf(i[2],self._metal_unq.min(),self._metal_unq.ptp())
  


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
        self.SSP = SSP
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
        #self._spect = ag.data_match_all(data,self._spect)[0]
        #extra models to use
        self._has_dust = use_dust
        self._has_losvd = use_losvd
        #key order
        self._key_order = ['gal']
        if use_dust:
            self._key_order.append('dust')
        if use_losvd:
            self._key_order.append('losvd')
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
        self._multi_block_param = {}
        self._multi_block_index = {}
        self._multi_block_i = 0
        #max total length of bins constraints
        self._max_age = self._age_unq.ptp()
		
    def proposal(self,Mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
		Proposal distribution, draws steps for chain. Should use a symetric
		distribution
        '''
		#save length and mean age they don't change  self._multi_block
        self._sigma = nu.copy(sigma)
        self._mu = Mu.copy()
        #extract out of dict
        
        mu = nu.hstack([i for j in self._key_order for i in Mu[j] ])
        try:
            t_out = nu.random.multivariate_normal(mu,sigma)
        except nu.linalg.LinAlgError:
            print sigma
        bins = Mu['gal'].shape[0]
        #save params for multi-block
        if str(bins) not in self._multi_block_param.keys():
            self._multi_block_param[str(bins)] = []
        self._multi_block_param[str(bins)].append(nu.copy(mu))
        #if rjmcmc see that performance is bad will turn multi block on
        #finds correlated parameters and changes them together
        bins = str(bins)
        if self._multi_block:
            #see if need initalization
            if bins not in self._multi_block_index.keys():
                self._multi_block_index[bins] = self.cov_sort(
                    self._multi_block_param[bins], int(bins))
                #self._hist[bins] = []
            #update params to change correlated params
            if len(self._multi_block_param[bins]) % 200 == 0:
                
                if int(bins) > 3:
                    self._multi_block_index[bins] = self.cov_sort(
                        self._multi_block_param[bins], int(bins))
                else:
                    self._multi_block_index[bins] = self.cov_sort(
                        self._multi_block_param[bins], 3)
                #if multiblock not working make random block
                if nu.unique(self._multi_block_index[bins]).size == 1:
                    self._multi_block_index[bins] = nu.random.choice(range(3),self._multi_block_index[bins].size)
            #set all non-changing params to original
            if self._multi_block_i > self._multi_block_index[bins].max():
                self._multi_block_i = 1
            index = self._multi_block_index[bins] == self._multi_block_i
            mu[index] = t_out[index]
            t_out = nu.copy(mu)
            #check iteratior
            self._multi_block_i += 1

        #extract out of mu into correct dict shape
        out = {}
        i,bins = 0,int(bins)
        for j in self._key_order:
            #gal
            if j == 'gal':
                out[j] = nu.reshape(t_out[i:i+bins*4], (bins, 4))
                i+= bins*4
            #dust
            if j == 'dust':
                out[j] = t_out[i:i+2]
                i += 2
            #losvd
            if j == 'losvd':
                out[j] = t_out[i:i+4]
                out[j][1:] = 0
                i+=4
        #gal lengths must be positve
        out['gal'][:,0] = nu.abs(out['gal'][:,0])
        #chech if only 1 metalicity
        if len(self._metal_unq) == 1:
            #set all metalicites to only value
            out['gal'][:,2] = nu.copy(self._metal_unq)

        return out
        
    def multi_try_all(self,param,bins,N=15):
        '''(VESPA_class,ndarray,str,int)-> float
        Does all things for multi try (proposal,lik, and selects best param'''
        temp_param = map(self.proposal,[param]*N, [self._sigma]*N)
    
        
    def lik(self,param, bins,return_all=False):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        if not self._check_len(param[bins]['gal'],bins):
            return -nu.inf
        burst_model = {}
        for i in param[bins]['gal']:
            burst_model[str(i[1])] =  10**i[3]*ag.make_burst(i[0],i[1],i[2],
            self._metal_unq, self._age_unq, self._spect, self._lib_vals)
        burst_model['wave'] = nu.copy(self._spect[:,0])
		#do dust
        if self._has_dust:
            #dust requires wavelengths
            
            #not correct
            burst_model = ag.dust(param[bins]['dust'],burst_model)
		#do losvd
        if self._has_losvd:
            #check if wavelength exsist
            if 'wave' not in burst_model.keys():
                burst_model['wave'] = nu.copy(self._spect[:,0])
            #make buffer for edge effects
            wave_range = [self.data[:,0].min(),self.data[:,0].max()]
            burst_model = ag.LOSVD(burst_model, param[bins]['losvd'], wave_range)
        #need to match data wavelength ranges and wavelengths
		#get loglik
        
        burst_model = ag.data_match(self.data,burst_model,bins)
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
        #make sure shape is ok
        if not self._check_len(param[bins]['gal'],bins):
            return -nu.inf
        #uniform priors
        #gal priors
        for i in param[bins]['gal']:
            #length
            out += stats_dist.uniform.logpdf(i[0],0.,self._age_unq.ptp())
            #age
            out += stats_dist.uniform.logpdf(i[1],self._age_unq.min(),self._age_unq.ptp())
            #metals
            if len(self._metal_unq) > 1:
                #if has metal range
                out += stats_dist.uniform.logpdf(i[2],self._metal_unq.min(),self._metal_unq.ptp())
            #weight
            out += stats_dist.uniform.logpdf(i[3], -300, 500)
        #dust
        if self._has_dust:
            #uniform priors
            out += stats_dist.uniform.logpdf(param[bins]['dust'],0,4).sum()
        #losvd
        if self._has_losvd:
            #sigma
            out += stats_dist.uniform.logpdf(param[bins]['losvd'][0],0,5)
            #z
            out += stats_dist.uniform.logpdf(param[bins]['losvd'][1],0,2)
            #h3 and h4
            out += stats_dist.uniform.logpdf(param[bins]['losvd'][2:],0,8).sum()
        return out


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #peak around 5 bins with heavy tail
        #can't allow for -inf
        out = stats_dist.maxwell.logpdf(int(model),2,3)+1
        if nu.isfinite(out):
            return out
        else:
            return -7.78061839
        #return stats_dist.maxwell.logpdf(int(model),2,3)+1

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

		Used to initalize all starting points for run of RJMCMC and MCMC.
		outputs starting point and starting step size
        '''
        #any amount of splitting
        out = {'gal':[], 'losvd':[],'dust':[]}
        #gal param
        age, metal, norm =  self._age_unq,self._metal_unq, self._norm
        lengths = self._age_unq.ptp()/float(model)
        for i in range(int(model)):
            out['gal'].append([lengths*nu.random.rand(), (i+.5)*lengths + age.min(),0,nu.log10(self._norm*nu.random.rand())])
            #metals
            out['gal'][-1][2] = nu.random.rand()*metal.ptp()+metal.min()
        out['gal'] = nu.asarray(out['gal'])
        #losvd param
        if self._has_dust:
            out['dust'] = nu.random.rand(2)*4
        #dust param
        if self._has_losvd:
            #[log10(sigma), v (redshift), h3, h4]
            out['losvd'] = nu.asarray([nu.random.rand()*4,0.,0.,0.])
        #make step size
        sigma = nu.identity(len([j for i in out.values() for j in nu.ravel(i)]))

        #make coorrect shape
        out['gal'] = self._make_square({model:out['gal']},model)
        #check if only 1 metalicity
        if len(self._metal_unq) == 1:
            #set all metalicites to only value
            out['gal'][:,2] = nu.copy(self._metal_unq)
            
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
            temp = nu.cov(self.list_dict_to(param[-2000:]).T)
            #make sure not stuck
            if nu.any(temp.diagonal() > 10**-6):
                step_size[model] = temp
        
        return step_size[model]


    def birth_death(self,birth_rate, model, Param):
        '''(Example_lik_class, float, any type, dict(ndarray)) -> 
        dict(ndarray), any type, bool, float
        
        For RJMCMC only. Does between model move. Birth rate is probablity to
        move from one model to another, models is current model and param is 
        dict of all localtions in param space.
        
        Returns new param array with move updated, key for new model moving to,
        whether of not to attempt model jump (False to make run as MCMC) and the
        Jocobian for move.

        Brith_rate is ['birth','split','merge','death']
        '''
        
        step = nu.random.choice(['birth','split','merge','death'],p=birth_rate)
        param = {model:Param[model]['gal'].copy()}
        if step == 'birth' and int(model) + 1 < self._max_sfh:
            new_param, jacob, temp_model = self._birth(param,model)
        elif step == 'death' and int(model) - 1 >= self._min_sfh :
            new_param, jacob, temp_model = self._death(param,model)
        elif step == 'merge' and int(model) - 1 >= self._min_sfh:
            new_param, jacob, temp_model = self._merge(param,model)
        elif step == 'split' and int(model) + 1 < self._max_sfh:
            new_param, jacob, temp_model = self._split(param,model)
        elif step == 'len_chng' and int(model) > 1 :
            new_param, jacob, temp_model = self._len_chng(param,model)
        else:
            #failed, return nulls
            return Param, None, False, None
        if new_param is None:
            #if change didn't work return nulls
            return param, None, False, None
        if not Param.has_key(temp_model):
            Param[temp_model] = {}
        Param[temp_model]['gal'] = new_param[nu.argsort(new_param[:,1])]
        #add dust and losvd to output
        if self._has_dust:
            Param[temp_model]['dust'] = Param[model]['dust'].copy()
        if self._has_losvd:
            Param[temp_model]['losvd'] = Param[model]['losvd'].copy()
        #param[temp_model] = self._make_square(param,temp_model)
        return Param, temp_model, True, abs(jacob) 
        
    def _len_chng(self, param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Stays at same dimesnsion just changes length parameters
        '''
        #randoms
        U1,U2,u = nu.random.rand(3)
        index = nu.random.randint(int(model))
        if U1 > .5:
            #increase size
            if index > 0 and index < int(model) - 1 :
                if U2 > .5:
                    #take from higher index
                    param[model][index,0] += u*param[model][index+1,0]
                    param[model][index+1,0] = param[model][index+1,0] * (1-u)
                else:
                    #take from lower index
                    param[model][index,0] += u*param[model][index-1,0]
                    param[model][index+1,0] = param[model][index-1,0] * (1-u)
            elif index == 0:
                #can only take from higher
                param[model][index,0] += u*param[model][index+1,0]
                param[model][index+1,0] = param[model][index+1,0] * (1-u)
            elif index == int(model)-1:
                #can only take from lower
                param[model][index,0] += u*param[model][index-1,0]
                param[model][index-1,0] = param[model][index-1,0] * (1-u)
        else:
            #decrease size
            if index > 0 and index < int(model) -1 :
                if U2 > .5:
                    #take from higher index
                    param[model][index+1,0] += param[model][index,0]*u
                    param[model][index,0] =(1- u)*param[model][index,0]
                else:
                    #take from lower index
                    param[model][index-1,0] += param[model][index,0]*u
                    param[model][index,0] =(1- u)*param[model][index,0]

            elif index == 0:
                #can only take from higher
                param[model][index+1,0] += param[model][index,0]*u
                param[model][index,0] =(1- u)*param[model][index,0]

            elif index == int(model) -1 :
                #can only take from lower
                param[model][index-1,0] += param[model][index,0]*u
                param[model][index,0] =(1- u)*param[model][index,0]

        return param[model], 1., model

    def _merge(self,param,model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Combines nearby bins together.
        '''
        #temp_model = str(int(model) - 1)
        #get lowest weight and give random amount to neighbor
        index = nu.where(param[model][:,-1] ==
                        nu.random.choice(param[model][:,-1]))[0]
        index = int(index)
        new_param = []
        split = param[model][index]
        u,U = nu.random.rand(2)
        if  U > .5:
            if index - 1 > -1 :
                #combine with younger
                temp = param[model][index-1]
            else:
                try:
                    index += 1
                    temp = param[model][index-1]
                except:
                    return None,None,None          
                
        elif U < .5:
            if index + 1 < int(model):
                #combine with older
                temp = param[model][index+1]
            else:
                try:
                    index -= 1
                    temp = param[model][index-1]
                except:
                    return None,None,None  
        else:
            #return Null
            return None,None,None
            
        new_param.append([temp[0] + split[0], 0., 0., 0.])
        #age
        new_param[-1][1] = ((1-u)*temp[1] + split[1] * u)
        #metal
        new_param[-1][2] = ((1-u)*temp[2] + split[2] * u)
        #norm * correction factior(assumes logs)
        new_param[-1][3] =nu.log10(10**split[3]+10**temp[3]*5*u)
        #make sure metalicity is in bounds
        if new_param[-1][2] < self._metal_unq.min():
                new_param[-1][2] = self._metal_unq.min() + 0.
        if new_param[-1][2] > self._metal_unq.max():
                new_param[-1][2] = self._metal_unq.max() + 0.
        #copy the rest of the params
        for i in param[model]:
            if nu.all(temp == i) or nu.all(split == i):
                continue
            new_param.append(nu.copy(i))
        #set for output
        temp_model = str(nu.asarray(new_param).shape[0])
        #inverse of split
        jacob = ((u-1)*u*nu.log(10))/((2*u-1)*new_param[0][0])
            
        return nu.asarray(new_param), jacob, temp_model
        
    def _birth(self,param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Creates a new bins, with parameters randomized.
        '''
        new_param = []
        age, metal, norm =  self._age_unq,self._metal_unq, self._norm
        lengths = age.ptp()/float(model) * nu.random.rand()
        new_param.append([lengths, 0.,0,nu.log10(self._norm*nu.random.rand())])
        #metals
        new_param[-1][2] = nu.random.rand()*metal.ptp()+metal.min()
        #age
        new_param[-1][1] = age.min()+nu.random.rand()*age.ptp()
        for i in param[model]:
            new_param.append(nu.copy(i))
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = metal.ptp()*age.ptp()**2/float(model)
        new_param = nu.asarray(new_param)
        
        return new_param, jacob, temp_model
    
    def _split(self, param, model):
        '''(VESPA_Class,dict(ndarray),str)-> ndarray,float,str
        Splits a bin into 2 with a random weight.
        '''
        #split component with prob proprotional to weights
        temp_param = nu.reshape(param[model], (int(model), 4))
        index = nu.where(param[model][:,-1] ==
                             nu.random.choice(param[model][:,-1]))[0]
        index = int(index)
        u = nu.random.rand()
        #[len,t,metal,norm]
        new_param = [[0,0,0,0]]
        new_param.append([0.,0.,0.,0.])
        #metal
        new_param[0][2] = temp_param[index,2] + 0.
        new_param[1][2] = temp_param[index,2] + 0.
        #length
        new_param[0][0] = temp_param[index,0] * u
        new_param[1][0] = temp_param[index,0] * (1 - u)
        #age
        new_param[0][1] = temp_param[index,1] - new_param[0][0]/2. 
        new_param[1][1] = temp_param[index,1] + new_param[1][0]/2
        #norm
        new_param[0][3] = nu.log10(10**temp_param[index,3] * u)
        new_param[1][3] = nu.log10(10**temp_param[index,3] * (1 - u))
        #copy the rest of the params
        for i in param[model]:
            if nu.all(temp_param[index] == i):
                continue
            new_param.append(nu.copy(i))

        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = ((2*u-1)*temp_param[index,0])/((u-1)*u*nu.log(10))
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
        jacob = float(model)/(self._metal_unq.ptp()*self._age_unq.ptp()**2)

        return nu.asarray(new_param), jacob, temp_model
    
    
    def _make_square(self,param,key):
        '''(dict(ndarray),str)-> ndarray
        DEPRICATED!
        Makes lengths and ages line up,sorts and makes sure length
        covers all length
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

    def _check_len(self, tparam, key):
        '''(VESPA_class, dict(ndarray) or ndarray,str)-> ndarray
        Make sure parameters ahere to criterion listed below:
        1. bins cannot overlap
        2. total length of bins must be less than length of age_unq
        3. ages must be in increseing order
        4. make sure bins are with in age range
        5. No bin is larger than the age range
        '''
        
        #check input type
        if type(tparam) == dict:
            param = tparam.copy()
        elif  type(tparam) == nu.ndarray:
            param = {key:tparam}
        else:
            raise TypeError('input must be dict or ndarray')
        #make sure age is sorted
        if not self.issorted(param[key][:,1]):
            return False
        #make sure bins do not overlap fix if they are
        for i in xrange(param[key].shape[0]-1):
            #assume orderd by age
            max_age = param[key][i,0]/2. + param[key][i,1]
            min_age_i = param[key][i+1,1] - param[key][i+1,0]/2.
            if max_age > min_age_i:
                #overlap
                #make sure overlap is significant
                if not nu.allclose(max_age, min_age_i):
                    return False
            #check if in age bounds
            if i == 0:
                if self._age_unq.min() > param[key][i,1] - param[key][i,0]/2.:
                    return False
        else:
            if self._age_unq.max() < param[key][-1,1] + param[key][-1,0]/2.:
                if not nu.allclose(param[key][-1,1] + param[key][-1,0]/2,self._age_unq.max()):
                    return False
        #check if length is less than age_unq.ptp()
        if param[key][:,0].sum() > self._age_unq.ptp():
            if not nu.allclose( param[key][:,0].sum(),self._age_unq.ptp()):
                return False
        #make sure age is in bounds
        
        #passed all tests
        return True

    def issorted(self,l):
        '''(list or ndarray) -> bool
        Returns True is array is sorted
        '''
        for i in xrange(len(l)-1):
            if not l[i] <= l[i+1]:
                return False
        return True

    def list_dict_to(self, s,outtype='ndarray'):
        '''(VESPA class list(dict(ndarray)),str) -> outtype
        
        Turns a list of dictoraies into a ndarray or type specified
        '''
        size = sum([nu.size(j) for j in s[0].values()])
        out = nu.hstack([i for Mu in s for j in self._key_order for i in Mu[j] ])
        return out.reshape((len(s),size))
        
    def cov_sort(self, param, k):
        '''(VESPA class, ndarray, int) -> ndarray
        groups params by their correlation to each other.
        Cov array and number of groups to have
        '''
        #make correlation matrix
        p = nu.asarray(param)
        #l,w,h = p.shape
        #p = nu.reshape(p,(l,w*h))
        Sigma = nu.corrcoef(p.T)
        #find nans and stuff and replace with 0's
        Sigma[~nu.isfinite(Sigma)] = 0
        #if all values are now 0
        #clusters by their correlation
        z = linkage(Sigma,'single','correlation')
        #returns which cluster each param belongs to
        try:
            clusters = fcluster(z,k,'maxclust')
        except ValueError:
            #failed to run successfully
            return nu.zeros(Sigma.shape[0])
        '''#sort clusters into indexs
        loc = []
        for i in range(k):
            loc.append(nu.where(clusters == i+1))

        self._loc = loc
        return loc'''
        return clusters
   
    def make_sfh_plot(self,param, model=None):
        '''(dict(ndarray)-> Nonetype
        Make plot of sfh vs log age of all models in param 
        '''
        import pylab as lab
        if not model is None:
            x,y = [], []
            for i in param[model]:
                x,y = [] , []
                #make square bracket
                x.append(i[1]-i[0]/2.)
                x.append(i[1]-i[0]/2.)
                x.append(i[1]+i[0]/2.)
                x.append(i[1]+i[0]/2.)
                y.append(i[3]-50)
                y.append(i[3])
                y.append(i[3])
                y.append(i[3]-50)
                lab.plot(x,y,'b',label=model)
            lab.legend()
            lab.show()
        else:
            for i in param.keys():
                pass
        return x,y
            
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

#=======UV source finder
class UV_SOURCE(object):

    '''exmaple class for use with RJCMCM or MCMC program, all methods are
    required and inputs are required till the comma, and outputs are also
    not mutable. The body of the class can be filled in to users delight'''
    def __init__(self,script):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, can do whatever you want. User to define functions'''
        
        #initalize
        # first time we're invoked, do startup and get data
        # This starts a meqserver. Note how we pass the "-mt 2" option to run two threads.
        # A proper pipeline script may want to get the value of "-mt" from its own arguments (sys.argv).
        print "Starting meqserver"
        self._mqs = meqserver.default_mqs(wait_init=10,extra=["-mt","16"]);
        print "Loading config";
        TDLOptions.config.read("tdlconf.profiles");
        print "Compiling TDL script";
        #script = "mcmcsim.py";
        mod,ns,msg = Compile.compile_file(self._mqs,script);

        self._mqs.execute('VisDataMux',mod.mssel.create_io_request(),wait=True)
        self._request = self._mqs.getnodestate("DT").request
        self._data = self._mqs.execute("DT",self._request,wait=True);
        #models avalible
        self.models = {'3':None}
        #multiblock for poor performance
        self._multi_block = False
        

    def call_meqtrees(self, params, hypothesis):
        #global request,ndomain,data,mqs;

        B = None;
        lmn = None;
        shape = None;
        hypothesis = int(hypothesis)
      
        # Specify l,m,n values in radians
        # l = np.cos(dec) * np.sin(ra-ra0);
        # m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra-ra0);

        # Harcoded values for the phase centre - bad practice!!!
        deg2rad = sc.pi / 180.0;
        ra0 = 0.0; dec0 = 60.0 * deg2rad;

        if hypothesis == 0:
            B = np.array([[0.+0j,0],[0,0.]]);
            lmn = np.array([0.,0,0]);
            shape = np.array([0.,0,0]);

        elif hypothesis == 1:
            B = np.array([[[params[0],0.+0j],[0.+0j,params[0]]],[[params[3],0.+0j],[0.+0j,params[3]]]]);

            l1 = np.cos(dec0+params[2]) * np.sin(params[1]);
            m1 = np.sin(dec0+params[2]) * np.cos(dec0) - np.cos(dec0+params[2]) * np.sin(dec0) * np.cos(params[1]);
            n1 = 0.0;
            l2 = np.cos(dec0+params[5]) * np.sin(params[4]);
            m2 = np.sin(dec0+params[5]) * np.cos(dec0) - np.cos(dec0+params[5]) * np.sin(dec0) * np.cos(params[4]);
            n2 = 0.0;

            lmn = np.array([[l1,m1,n1],[l2,m2,n2]]);

            shape = np.array([[0.,0,0],[0.,0,0]]);
    
        elif hypothesis == 2:
            B = np.array([[[params[0],0.+0j],[0.+0j,params[0]]]]);

            l1 = np.cos(dec0+params[2]) * np.sin(params[1]);
            m1 = np.sin(dec0+params[2]) * np.cos(dec0) - np.cos(dec0+params[2]) * np.sin(dec0) * np.cos(params[1]);
            n1 = 0.0;
            lmn = np.array([[l1,m1,n1]]);
            shape = np.array([[params[4]*np.sin(params[3]),params[4]*np.cos(params[3]),float(params[5])/params[4]]]);

        elif hypothesis == 3:
            B = np.array([[[params[0],0.+0j],[0.+0j,params[0]]]]);

            #l1 = np.cos(dec0+params[2]) * np.sin(params[1]);
            #m1 = np.sin(dec0+params[2]) * np.cos(dec0) - np.cos(dec0+params[2]) * np.sin(dec0) * np.cos(params[1]);
            #l1=m1=n1 = 0.0;
            #lmn = np.array([[l1,m1,n1]]);

            shape = np.array([[0.,0,0]]);

        """print "B:\n",B
        print "lmn:\n",lmn
        print "shape:\n",shape
        print "B:\n",type(B),len(B)
        print "lmn:\n",type(lmn),len(lmn)
        print "shape:\n",type(lmn),len(shape)"""
          
        self._mqs.setnodestate("BT0",dmi.record(value=B),sync=True);
        #self._mqs.setnodestate("lmnT0",dmi.record(value=lmn),sync=True);
        #self._mqs.setnodestate("shapeT0",dmi.record(value=shape),sync=True);

        #t0 = time.time();
        self._mqs.clearcache("MT");
        model = self._mqs.execute("MT",self._request,wait=True);
        
        return model

    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param 
        out = nu.random.multivariate_normal(mu,sigma)
        return out

    def lik(self,cube, hypothesis):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        
        #return loglik
        #def myloglike(cube, ndim, nparams):
        """
        Simple chisq likelihood for straight-line fit (m=1,c=1)
        
        cube is the unit hypercube containing the current values of parameters
        ndim is the number of dimensions of cube
        nparams (>= ndim) allows extra derived parameters to be carried along
        """
        
        model = self.call_meqtrees(cube[hypothesis], hypothesis)
        
        sigma = 0.01
        #chi2 = 0.
        ndata = 0

        # loop over arrays in data and model to form up chisq
        for vd,vm in zip(self._data.result.vellsets,model.result.vellsets):
            delta = vd.value - vm.value
        
        chi2 = (delta.real**2/sigma**2).sum() + (delta.imag**2/sigma**2).sum()
                
         
        return -chi2

    def prior(self,cube1, hypothesis):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        """
        This function just transforms parameters to the unit hypercube

        cube is the unit hypercube containing the current values of parameters
        ndim is the number of dimensions of cube
        nparams (>= ndim) allows extra derived parameters to be carried along

        You can use Priors from priors.py for convenience functions:

        from priors import Priors
        pri=Priors()
        cube[0]=pri.UniformPrior(cube[0],x1,x2)
        cube[1]=pri.GaussianPrior(cube[1],mu,sigma)
        cube[2]=pri.DeltaFunctionPrior(cube[2],x1,anything_ignored)
        """
        
        logprior = 0.
        hypothesis = int(hypothesis);
        deg2rad = sc.pi / 180.0;
        arcsec2rad = sc.pi / 180.0 / 3600.0;
        
        dxmin=-4.0; dxmax=+4.0; dymin=-4.0; dymax=+4.0 # arcsec
        dxmin *= arcsec2rad;
        dxmax *= arcsec2rad;
        dymin *= arcsec2rad;
        dymax *= arcsec2rad;

        Smin=0.0; Smax=2.0 # Jy
        cube = cube1[str(hypothesis)]
        # Need to convert RA, Dec to dra, ddec
        #ra0 = 0.0; dec0 = 60.0; # user-specified (in degrees)
        #ra0 = ra0 * deg2rad;
        #dec0 = dec0 * deg2rad;
        #ra = ra0 - cube[1]; dec = dec0 + cube[2];
        
        # Model 0 (noise only) -- 3 params (all = 0.0)
        if hypothesis == 0:
            #not correct need to test if just noise
            cube[0] = cube[0] * 0.0  # S0
            cube[1] = cube[1] * 0.0  # dx0
            cube[2] = cube[2] * 0.0  # dy0

        # Model 1 (noise + source 1 + source 2) -- distinct position priors
        elif hypothesis == 1:
            #S
            logprior += stats_dist.uniform.pdf([cube[0],cube[3]],Smin,(Smax-Smin)).sum()
            #dx
            logprior += stats_dist.uniform.pdf([cube[1],cube[4]],dxmin,(dxmax-dxmin)).sum()
            #dy
            logprior += stats_dist.uniform.pdf([cube[2],cube[5]],0,dymax).sum()
       

        # Model 2 (noise + source 3 [gaussian]) - Flux in Jy; Pos in ra/dec; PA
        elif hypothesis == 2:
            thetamin = 0.0 * deg2rad; thetamax = 180.0 * deg2rad;
            e1min = 0.0; e1max = 10.0 * arcsec2rad;
            e2min = 0.0; e2max = 10.0 * arcsec2rad;
            """Smin = Smax = 0.993808;
            thetamin = thetamax = 92.0 * deg2rad;
            e1min = e1max = 7.0 * arcsec2rad;
            e2min = e2max = 4.0 * arcsec2rad;"""

            # Flux in Jy, angles in rad.
            #S
            logprior += stats_dist.uniform.pdf(cube[0],Smin,(Smax-Smin)).sum()
            #dx
            logprior += stats_dist.uniform.pdf(cube[1],dxmin,(dxmax-dxmin)).sum()
            #dy
            logprior += stats_dist.uniform.pdf(cube[2],dymin,(dymax-dymin)).sum()
            #posn angle
            logprior += stats_dist.uniform.pdf(cube[3],thetamin,(thetamax-thetamin)).sum()
            #emaj
            logprior += stats_dist.uniform.pdf(cube[4],e1min,(e1max-e1min)).sum()
            #emin
            logprior += stats_dist.uniform.pdf(cube[5],e2min,(e2max-e2min)).sum()
          

        # Model 3 (noise + source 1 [single atom] )
        elif hypothesis == 3:
            #Smax=Smin=5.0
            dxmin=dxmax=dymin=dymax=0.0
            #S
            logprior += stats_dist.uniform.pdf(cube[0],Smin,(Smax-Smin)).sum()
            #dx
            logprior += stats_dist.uniform.pdf( cube[1],dxmin,(dxmax-dxmin)).sum()
            #dy
            logprior += stats_dist.uniform.pdf(cube[2],dymin,(dymax-dymin)).sum()
                 
            
        else:
            #print '*** WARNING: Illegal hypothesis'
            return -nu.inf

        return nu.sum(logprior)


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        return 0.

    def initalize_param(self, hypothesis):
        '''(Example_lik_class, any type) -> ndarray, ndarray

        Used to initalize all starting points for run of RJMCMC and MCMC.
        outputs starting point and starting step size'''

        hypothesis = int(hypothesis);
        sigma=0.01 #error on each visibility
        deg2rad = sc.pi / 180.0;
        arcsec2rad = sc.pi / 180.0 / 3600.0;

        dxmin=-4.0; dxmax=+4.0; dymin=-4.0; dymax=+4.0 # arcsec
        dxmin *= arcsec2rad;
        dxmax *= arcsec2rad;
        dymin *= arcsec2rad;
        dymax *= arcsec2rad;

        Smin=0.0; Smax=2.0 # Jy

        #return init_param, init_step
        if hypothesis == 1:
            cube = nu.zeros(6)
            step = nu.identity(6)
            #S
            cube[0],cube[3] = stats_dist.uniform.rvs(Smin,(Smax-Smin),2)
            #dx
            cube[1],cube[4] = stats_dist.uniform.rvs(dxmin,(dxmax-dxmin),2)
            #dy
            cube[2],cube[5] = stats_dist.uniform.rvs(0,dymax,2)
            

        # Model 2 (noise + source 3 [gaussian]) - Flux in Jy; Pos in ra/dec; PA
        elif hypothesis == 2:
            thetamin = 0.0 * deg2rad; thetamax = 180.0 * deg2rad;
            e1min = 0.0; e1max = 10.0 * arcsec2rad;
            e2min = 0.0; e2max = 10.0 * arcsec2rad;
            """Smin = Smax = 0.993808;
            thetamin = thetamax = 92.0 * deg2rad;
            e1min = e1max = 7.0 * arcsec2rad;
            e2min = e2max = 4.0 * arcsec2rad;"""
            #make arrays
            cube = nu.zeros(6)
            step = nu.identity(6)
            # Flux in Jy, angles in rad.
            #S
            cube[0] = stats_dist.uniform.rvs(Smin,(Smax-Smin))
            #dx
            cube[1] = stats_dist.uniform.rvs(dxmin,(dxmax-dxmin))
            #dy
            cube[2] = stats_dist.uniform.rvs(dymin,(dymax-dymin))
            #posn angle
            cube[3] = stats_dist.uniform.rvs(thetamin,(thetamax-thetamin))
            #emaj
            cube[4] = stats_dist.uniform.rvs(e1min,(e1max-e1min))
            #emin
            cube[5] = stats_dist.uniform.rvs(e2min,(e2max-e2min))

        # Model 3 (noise + source 1 [single atom] )
        elif hypothesis == 3:
            #Smax=Smin=5.0
            cube = nu.zeros(3)
            step = nu.identity(3)
            dxmin=dxmax=dymin=dymax=0.0
            #S
            cube[0] = stats_dist.uniform.rvs(Smin,(Smax-Smin))
            #dx
            cube[1] = stats_dist.uniform.rvs(dxmin,(dxmax-dxmin))
            #dy
            cube[2] = stats_dist.uniform.rvs(dymin,(dymax-dymin))

        return cube,step
        
    def step_func(self,step_crit,param,step_size,model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        #return new_step
        if step_crit > .60:
            step_size[model] *= 1.05
        elif step_crit < .2 and nu.any(step_size[model].diagonal() > 10**-6):
            step_size[model] /= 1.05
        #cov matrix
        '''if len(param) % 200 == 0 and len(param) > 0.:
            temp = nu.cov(self.list_dict_to(param[-2000:]).T)
            #make sure not stuck
            if nu.any(temp.diagonal() > 10**-6):
                step_size[model] = temp'''
        
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
        #for RJCMC
        #return new_param, try_model, attemp_jump, Jocobian
        #for MCMC
        return param, None, False, None

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
