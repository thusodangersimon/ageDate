#!/usr/bin/env python
#
# Name:  Markov Chain Utilities
#
# Author: Thuso S Simon
#
# Date: 4th of Feb, 2014
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
'''
All the functions needed to build a likelihood class for different type of
Markov Chain methods.

Also convergence diagonistics.

'''
from anderson_darling import anderson_darling_k as ad_k
import numpy as nu
from scipy.stats import levene, f_oneway,kruskal
from glob import glob
import ezgal as gal
import scipy.stats as stats_dist
from scipy.special import erfinv
import multiprocessing as multi
from itertools import izip
from scipy.cluster.hierarchy import fcluster,linkage
import os, sys, subprocess
from time import time
import signal
import acor
import ipdb
###ALL##########

##########Proposals######
def normal(mu,sigma):
    pass

def student_t(mu,sigma,dof):
    pass

def multi_block():
    pass

def cov_sort(self, param, k):
        '''(ndarray, int) -> ndarray
        Helper function for muli_block.
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

    
#######Samplelers########
def swarm_func(param,dust_param,chi,parambest,chibest,bins):
    #pushes current chain towards global best fit with streangth
    #porportional to the difference in chi squared values * prior volume
    #if not in same bin number, changes birth/death rate
    #dust is always affected?
    best_param = nu.array(parambest)
    best_param = best_param[~nu.isnan(best_param)]
    best_bins = (len(best_param)-2)/3
    if not bins == best_bins:
        if bins > best_bins:
            #raise prob of death
            return nu.zeros(bins*3),nu.zeros(2),0.2
        else:
            #raise prob of birth
            return nu.zeros(bins*3),nu.zeros(2),0.8
    else:
        vect = best_param[:-2] - param
        vect_dust = best_param[-2:] - dust_param
        #add vector in porportion to chi values
        weight = nu.tanh(0.000346 * 
                         abs(chi - chibest.value ))
        weight_dust = weight * nu.sum(vect_dust**2)**0.5/4.
        weight *=  nu.sum(vect**2)**0.5/4.
    return (param + weight * vect, dust_param + weight_dust * vect_dust,
            0.5)

def delayed_rejection(xi, pxi,xnext,pxnext,sigma,bins, fun):
    """(currnent_state, current_posterior, lik_object) ->
    Generates a second proposal based on rejected proposal xi
    """
    #make step
    for i in xrange(50):
        #generate new point
        zdr = {bins:fun.proposal(xi[bins],sigma[bins])}
        #check if in prior
        if not nu.isfinite(fun.prior(zdr,bins)):
            break
    else:
        #after 50 trials give up if not in priors
        return 
    #get next proposal
    propphi_zdr = self._prop_phi([zdr])
    #calc lik
    zdrprob,  zdrlik = self._get_post_prob([zdr],propphi_zdr)
    #acceptance prob for new param a(zdrprob[0],zprob)
    alpha2 = min(zdrprob[0]*(1-self._alpha1(self,zdrprob[0],zprob))/(pxi*(1-self._alpha1(self, pxi, zprob))), 1)
    acc = 0; lik = 0; pr = 0; prop = 0
    if random()< alpha2:
        xi = zdr
        acc = 1
        liks = zdrlik
        pr = zdrprob[0]
        prop = propphi_zdr
    return xi, acc, lik, pr, prop


########Step Calulators#############
def param_cov(param,samples):
    pass

def smallchange(step,accept_rate):
    pass

def acceptchange(step,accept_rate):
    pass
######PMC##########

#######RJMC#############
def _len_chng(param, model,s):
        '''(dict(ndarray),str)-> ndarray,float,str
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

def _merge(param,model,s):
        '''(dict(ndarray),str)-> ndarray,float,str
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
        new_param[-1][3] = nu.log10(10**split[3]+10**temp[3]*5*u)
        #make sure metalicity is in bounds
        if new_param[-1][2] < s._metal_unq.min():
                new_param[-1][2] = s._metal_unq.min() + 0.
        if new_param[-1][2] > s._metal_unq.max():
                new_param[-1][2] = s._metal_unq.max() + 0.
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
        
def _birth(param, model,s):
        '''(dict(ndarray),str)-> ndarray,float,str
        Creates a new bins, with parameters randomized.
        '''
        new_param = []
        age, metal, norm =  s._age_unq,s._metal_unq, s._norm
        lengths = age.ptp()/float(model) * nu.random.rand()
        new_param.append([lengths, 0.,0,nu.log10(s._norm*nu.random.rand())])
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
    
def _split(param, model,s):
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
    
def _death(param, model,s):
        '''(dict(ndarray),str,vespa class)-> ndarray,float,str
        Removes a bin from array.
        '''
        index = nu.random.randint(int(model))
        new_param = []
        for i in xrange(param[model].shape[0]):
            if i == index: 
                continue
            new_param.append(nu.copy(param[model][i]))
        temp_model = str(nu.asarray(new_param).shape[0])
        jacob = float(model)/(s._metal_unq.ptp()*s._age_unq.ptp()**2)

        return nu.asarray(new_param), jacob, temp_model

def _check_len(tparam, key, age_unq):
        '''( dict(ndarray) or ndarray,str)-> ndarray
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
        if not issorted(param[key][:,1]):
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
                if age_unq.min() > param[key][i,1] - param[key][i,0]/2.:
                    return False
        else:
            if age_unq.max() < param[key][-1,1] + param[key][-1,0]/2.:
                if not nu.allclose(param[key][-1,1] + param[key][-1,0]/2,age_unq.max()):
                    return False
        #check if length is less than age_unq.ptp()
        if param[key][:,0].sum() > age_unq.ptp():
            if not nu.allclose( param[key][:,0].sum(),age_unq.ptp()):
                return False
        #make sure age is in bounds
        
        #passed all tests
        return True

######Diagnostics
def gr_convergence(relevantHistoryEnd, relevantHistoryStart):
    """
    Gelman-Rubin Convergence
    Converged when sum(R <= 1.2) == nparam
    """
    start = relevantHistoryStart
    end = relevantHistoryEnd
    N = end - start
    if N==0:
        return  np.inf*np.ones(self.nchains)
    N = min(min([len(self.seqhist[c]) for c in range(self.nchains)]), N)
    seq = [self.seqhist[c][-N:] for c in range(self.nchains)]
    sequences = array(seq) #this becomes an array (nchains,samples,dimensions)
    variances  = var(sequences,axis = 1)#array(nchains,dim)
    means = mean(sequences, axis = 1)#array(nchains,dim)
    withinChainVariances = mean(variances, axis = 0)
    betweenChainVariances = var(means, axis = 0) * N
    varEstimate = (1 - 1.0/N) * withinChainVariances + (1.0/N) * betweenChainVariances
    R = sqrt(varEstimate/ withinChainVariances)
    return R

def Convergence_tests(param,keys,n=1000):
    #uses Levene's test to see if var between chains are the same if that is True
    #uses f_oneway (ANOVA) to see if means are from same distrubution 
    #if both are true then tells program to exit
    #uses last n chains only
    for i in param:
        for j in keys:
            i[j]=nu.array(i[j])
    ad_k
    D_result={}
    for i in keys:
        for k in range(param[0][i].shape[1]):
            samples='' 
            for j in range(len(param)):
                samples+='param['+str(j)+']["'+i+'"][-'+str(int(n))+':,'+str(k)+'],'
        D_result[i]=eval('ad_k('+samples[:-1]+')')[-1]>.05
    if any(D_result.values()):
        print 'A-D test says they are the same'
        return True
    #try kustkal test
    A_result={}
    out=False
    for i in keys:
        A_result[i]=nu.zeros(param[0][i].shape[1])
        for k in range(param[0][i].shape[1]):
            samples='' 
            for j in range(len(param)):
                samples+='param['+str(j)+']["'+i+'"][-'+str(int(n))+':,'+str(k)+'],'
            A_result[i][k]=eval('kruskal('+samples[:-1]+')')[1]
        if nu.all(A_result[i]>.05): #if kruskal test is true
            out=True
            print "ANOVA says chains have converged. Ending program"
            return True            
        else:
            print '%i out of %i parameters have same means' %(sum( A_result[i]>.05),
                                                              param[0][i].shape[1])
    #do ANOVA to see if means are same
    if out:
        L_result={}
        out=False
        #turn into an array
        for i in keys:
            param[0][i]=nu.array(param[0][i])
            L_result[i]=nu.zeros(param[0][i].shape[1])
            for k in range(param[0][i].shape[1]):
                samples='' 
                for j in range(len(param)):
                    samples+='param['+str(j)+']["'+i+'"][-'+str(int(n))+':,'+str(k)+'],'
                L_result[i][k]=eval('levene('+samples[:-1]+')')[1]
            if nu.all(L_result[i]>.05): #if leven test is true
                print "Levene's test is true for %s bins" %i
                return True
            else:
                print '%i out of %i parameters have same varance' %(sum( L_result[i]>.05),
                                                                param[0][i].shape[1])
    return False

def ess(t):
    '''returns the average sample size of a N,M ndarray (doesn't actually take a N,M ndarray'''
    #extract data from dictoranry
    data = []
    for i in t:
        dics = []
        for j in i.keys():
            dics.append(nu.ravel(i[j]))
        data.append(nu.hstack(dics))
    data = nu.asarray(data)
    temp_ess,Len = [],float(len(data[:,0]))
    #calculate ess for each parameter
    for i in range(data.shape[1]):
        try:
            temp_ess.append(Len/acor.acor(data[:,i])[0])
        except ZeroDivisionError:
            pass
        except RuntimeError:
            pass
    if len(temp_ess) == 0:
        return 0.
    #return max
    return nu.nanmax(temp_ess)


def effectiveSampleSize(data, stepSize = 1) :
  """ Effective sample size, as computed by BEAST Tracer."""
  samples = len(data)

  assert len(data) > 1,"no stats for short sequences"
  
  maxLag = min(samples // 3, 1000)

  gammaStat = [0,] * maxLag
  #varGammaStat = [0,]*maxLag

  varStat = 0.0

  if isinstance(data[0], dict):
    data = list_dict_to(data,data[0].keys())
    data = nu.array(data)[:,-1]

  normalizedData = data - data.mean()
  
  for lag in xrange(maxLag) :
    v1 = normalizedData[:samples-lag]
    v2 = normalizedData[lag:]
    v = v1 * v2
    gammaStat[lag] = nu.sum(v) / len(v)
    #varGammaStat[lag] = nu.sum(v*v) / len(v)
    #varGammaStat[lag] -= gammaStat[0] ** 2

    #print lag, gammaStat[lag], varGammaStat[lag]
    
    if lag == 0 :
      varStat = gammaStat[0]
    elif lag % 2 == 0 :
      s = gammaStat[lag-1] + gammaStat[lag]
      if s > 0:
        varStat += 2.0*s
      else:
        break
      '''
      varStat[s > 0] += 2.0 * s[s > 0]
      if not nu.any(s[s > 0]):
        break'''
      
  # standard error of mean
  #stdErrorOfMean = nu.sqrt(varStat/samples);

  # auto correlation time
  act = stepSize * varStat / gammaStat[0]

  # effective sample size
  ess = (stepSize * samples) / act

  return ess

#####SIMULATED ANNEELING#####
def SA(i,i_fin,T_start,T_stop):
    '''temperature parameter for Simulated anneling (SA). 
    reduices false acceptance rate if a<60% as a function on acceptance rate'''
    if i > i_fin:
        return 1.0
    else:
        return ((T_stop - T_start) / float(i_fin)) * float(i) + T_start
#####MISC####################
def issorted(l):
    '''(list or ndarray) -> bool
    Returns True is array is sorted
    '''
    for i in xrange(len(l)-1):
        if not l[i] <= l[i+1]:
            return False
    return True

def list_dict_to(s, key_order, outtype='ndarray'):
        '''(list(dict(ndarray)),str) -> outtype
        Turns a list of dictoraies into a ndarray or type specified
        '''
        size = sum([nu.size(j) for j in s[0].values()])
        out = nu.hstack([i for Mu in s for j in key_order for i in Mu[j] ])
        return out.reshape((len(s),size))
  
def make_square(param, key, age_unq):
        '''(dict(ndarray),str)-> ndarray
       
        Makes lengths and ages line up,sorts and makes sure length
        covers all length
        '''
        #sort params by age
        out = nu.copy(param[key][nu.argsort(param[key][:,1]),:])
        #check that lengths are correct 
        if not out[:,0].sum() == age_unq.ptp():
            out[:,0] = out[:,0] / out[:,0].sum()
            out[:,0] *= age_unq.ptp()

        #and cover full range of age space
        for i in xrange(int(key)):
            if i == 0:
                out[i,1] = age_unq.min()+ out[i,0]/2.

            else:
                out[i,1] = age_unq.min()+out[:i,0].sum()
                out[i,1] += out[i,0]/2.
                
        return out
