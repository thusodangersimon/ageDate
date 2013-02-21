#!/usr/bin/env python
#
# Name:  Gauss_landscape
#
# Author: Thuso S Simon
#
# Date: 17 of April, 2011
#TODO: 
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
#
#
#
""" creates Likelihood classes for testing of MCMC,RJMCMC and other fitting methods. Also servers as an example on how to write functions needed to run programs"""

_module_name='gauss_param_space'

import numpy as nu
import pylab as lab
#import mpl_toolkits.mplot3d.axes3d as lab3
#from mpl_toolkits.mplot3d import Axes3D
import sympy as sy
import scipy.stats as stat_dist
from scipy.special import expn
import rpy2.robjects as ro


#template class things needed for programs to run
class template_class(object):
    #things needed for mcmc or rjmcmc to run
    def __init__(self):
        #initalize and put input data
        pass

    def sampler(self):
        #sampler or proposial distribution for drawing numbers
        pass
    
    def prior(self):
        #prior, should be a distribution, return values between [0,1] and also returns 0's for out of bounds parameters
        pass
    def lik(self):
        #likelihood calculation, recomended return -2log(lik) or chi squared where smaller values are higher likelihoods
        #if not using log likelihoods, use higher percicion numbers so values wont go to zero
        #should not return NaNs if miss calculations should return inf
        pass

    #stuff for rjmcmc. Not needed for MCMC
    def birth_function(self):
        #how to change between models
        pass
    
    #functions to speed up convergence (optional)
    def partical_swarm(self):
        #partical swarm, has mpi, multiprocessing or single chain methods
        pass

    def convergece_test(self):
        pass
class Gauss_lik(object):
    #does N dimensional gaussians with x points
    
    def __init__(self):
        #initalize and put input data
        self._true_sigma = nu.random.rand() * 50
        self._true_mu = 0. #nu.random.rand() * 50

    def sampler(self,mu,sigma):
        #sampler or proposial distribution for drawing numbers
        return nu.random.multivariate_normal(mu,sigma)
    
    def prior(self,x,a=0,b=1):
        #prior, should be a distribution, return values between [0,1] and also returns 0's for out of bounds parameters
        #uniform prior with 1 axis only
        prior_bool = nu.logical_and(x >=a,x<=b)
        #if true set equal to prior value else make zero
        out = nu.zeros_like(prior_bool,dtype=float)
        out[prior_bool] = 1 / float(b - a)
        return out
        

    def lik(self,x):
        #likelihood calculation, recomended return -2log(lik) or chi squared where smaller values are higher likelihoods
        #if not using log likelihoods, use higher percicion numbers so values wont go to zero
        #should not return NaNs if miss calculations should return inf
        return stat_dist.norm.pdf(x,loc = self._true_mu, scale=self._true_sigma)


def toyI(p=None,x=115,n=200):
    '''samples from a binomial distbution with like is binomal and prior is 
    uniform. n=200, x=115 p(p)=u(0,1) the marginal likelihood should be
    0.0049753.
    When called will sample from prior and calc liklihood. If p is given will 
    cal like of p
    '''
    #if no p value sample
    if not nu.any(p):
        p = nu.ranom.rand()
    #calc likelhood
    
    return stat_dist.binom.pmf(x,n,p)

def toyII(p,mu=0.,std=1):
    '''Test weather a point is drawn from gaussian with unknown mean, and 
    known std. Has a gausian prior on mean, with mu and std as mean and stadard
    devation'''
    return stat_dist.norm.pdf(p,mu,std)

def toyIII(x,y,theta,sigma=1,answer=False):
    '''linear regression model x and y data points. length of theta give 
    polynomial order. If answer is true, and len(x)<5 will give analytical 
    answer with theta=[-inf,int] with a normal prior for all theta'''
    if answer:
        #calculate analyitical solution
        return False
    if len(theta.shape) == 1: #1-d array
        mu = y - nu.polyval(theta,x)
        return stat_dist.norm.logpdf(y, nu.polyval(theta,x),sigma)
    else:
        mu = nu.zeros((theta.shape[0],len(x)))
        for i in range(theta.shape[1]):
            #mu += nu.tile(y,(len(mu),1)) - (nu.tile(theta[:,i],(len(x),1)).T*
                                            #x**(theta.shape[1] -1-i))
            mu += nu.tile(theta[:,i],(len(x),1)).T*x**(theta.shape[1] -1-i)
        return stat_dist.norm.logpdf(nu.tile(y,(len(mu),1)), mu,sigma)
    #return stat_dist.norm.logpdf(mu,0,sigma)


def sample(x, size, replace=False, prob=None):
    '''like R package sample. Samples from x with prob of prob, can be any size
    and with or without replacement. Not complete yet, only does sampling with replacement with prob vector'''
    #set up for R
    r_x = ro.Vector(list(x))
    r_size = ro.Vector(size)
    r_replace = ro.Vector(replace)
    r_prob = ro.Vector(list(prob))
    r_sample = ro.r.sample(r_x,r_size,r_replace,r_prob)
    out =[]
    for i in r_sample:
        out.append(i.__getitem__(0))
    return nu.array(out)

if __name__ == '__main__':
   #pmc test of calc evidence
    #with binomial
    N,K = 542,115 #param for toy
    n =10**5
    x = nu.random.rand(n)
    q = toyI(x,K,N) / stat_dist.uniform.pdf(x)
    print 'Binomial is estimated %f, real %f'%(nu.mean(q),1/(1.+N))
   
    #with norm and norm prior (unknow mean, knowns std=prior_std)
    std = 5.
    prior_mu = -4
    x = nu.random.randn(n)
    q = toyII(x,prior_mu,std)**2 /stat_dist.norm.pdf(x)
    print 'Normal is estimated %f, real %f'%(nu.mean(q),stat_dist.norm.pdf(prior_mu,0,std))

    #linear regssion model
    order = 2 #polynomial order
    N = 3 #number of point to generate
    x = nu.linspace(1,5,3)
    real_m = nu.random.randn(order)
    y = [  6.33623224,  -4.97248028, -16.18410389]
    #y = nu.polyval(real_m,x) + nu.random.randn(len(x))
    #pm
    X = nu.random.multivariate_normal([0,0],nu.identity(2)*9,n)
    #cal prior
    #prior = nu.zeros(n)
    for i in range(order):
        if i == 0:
          prior =  stat_dist.norm.pdf(X[:,i],-3,1)
        else:
            prior *= stat_dist.norm.pdf(X[:,i],-3,1)
    logq = nu.sum(toyIII(x,y,X,2),1) + nu.log(prior) - nu.sum(stat_dist.norm.logpdf(X,0,9),1)
    print 'Evidence is %.3g' %(nu.exp(logq,dtype=nu.float128).mean())

    #liklihood plot x=m,y=b,z loglik
    real_m = nu.array([-3,-3.]) #nu.random.randn(order)*9
    y = nu.polyval(real_m,x) + nu.random.randn(len(x))
    M,B = nu.meshgrid(nu.linspace(-50,50,10**2),nu.linspace(-50,50,10**2))
    X = nu.array(zip(M.ravel(),B.ravel()))
    for i in range(order):
        if i == 0:
          prior =  stat_dist.norm.pdf(X[:,i],-3,1)
        else:
            prior *= stat_dist.norm.pdf(X[:,i],-3,1)

    chi = nu.sum(toyIII(x,y,X),1) + nu.log(prior)
    print 'Highest likelihood place is at (%1.2f,%1.2f)'%(X[chi.argmax()][0],X[chi.argmax()][1])
    chi[nu.isinf(chi)] = chi[~nu.isinf(chi)].min()
    chi = chi.reshape(M.shape)
    lab.contour(M,B,chi)
    lab.colorbar()
