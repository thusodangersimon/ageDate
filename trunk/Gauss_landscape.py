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
#    Copyright (C) 2012  Thuso S Simon
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
#import sympy as sy
import scipy.stats as stat_dist
from scipy.special import expn
import rpy2.robjects as ro
#import Age_hybrid as hy

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

class Mixture_model(object):
    '''Uses a mixture model of linear regression to test RJMCMC infrenece'''
    def __init__(self,n_points=100,real_theta=None):
        '''Creates data and stores real infomation for model
        (n_points number of points to generate for model 
        y_i = \sum_i theta_j*x_i^(i-1)'''
        if not nu.all(real_theta):
            #if no input params make random
            self._order = nu.random.randint(1,10)
            self._real_theta = nu.random.randn(self._order)
        else:    
            self._order = len(real_theta)
            self._real_theta = real_theta
        #maximum amount of varibles to look at
        self._min_order = 1
        self._max_order = self._order + 5
        self.data = nu.zeros((n_points,2))
        self.data[:,0] = nu.random.rand(n_points)*10-5
        self.data[:,0].sort()
        self.data[:,1] = (nu.polyval(self._real_theta,self.data[:,0]) +
                          nu.random.rand(n_points))
        #set prior values
        self.mu,self.var = -3,1

    def proposal(self,mu,sigma):
        '''drawing function for RJMC'''
        return nu.random.multivariate_normal(mu,sigma)

    def lik(self,param,**args):
        '''log liklyhood calculation'''
        out =  lik_toyIII(self.data[:,0],self.data[:,1],param,
                          self.var,answer=False)
        return nu.sum(out) + self.prior(param)

    def prior(self,param):
        '''log prior and boundary calculation'''
        return stat_dist.norm.logpdf(param,self.mu,self.var).sum()
    
    def initalize_param(self,order):
        '''initalizes parameters for uses in MCMC'''
        params = nu.random.multivariate_normal([0]*order,nu.identity(order)*9)
        sigma = nu.identity(order)*nu.random.rand()*9
        return params,sigma
    
    def step_func(self,accept_rate,param,sigma,order):
        '''changes step size'''
        if accept_rate > .25  and nu.all(sigma < 100):
            sigma *= 1.05
        elif accept_rate <.23  and nu.any(sigma > 10**-5):
            sigma /= 1.05
        if len(param[-1000:]) % 100:
            try:
                sigma = nu.cov(param[-1000:].T)
            except AttributeError:
                sigma = nu.var(param[-1000:])
        return sigma

    def birth_death(self,birth_rate, bins, j, j_timeleft, active_param):
        if birth_rate > nu.random.rand() and bins+1 != self._max_order :
            #birth
            attempt = True #so program knows to attempt a new model
            #create step
            temp_bins = bins + 1 #nu.random.randint(bins+1,self._max_order)
            #criteria for this step
            critera = 2**(bins-temp_bins) * birth_rate 
            #new param step
            #active_param[str(temp_bins)][:bins] = active_param[str(bins)].copy()
            #draw new points randomly from prior (bad)
            #active_param[str(temp_bins)][bins:] = self.initalize_param(temp_bins)[0][bins:]
            #split move params W1 = w *u, W2=w*(1-u) 
            for j in range(len(active_param[str(bins)])):
                if j * 2 + 1 >= len(active_param[str(temp_bins)]):
                    break
                u = nu.random.rand()
                active_param[str(temp_bins)][2*j] = active_param[str(bins)][j] *u
                active_param[str(temp_bins)][2*j+1] = active_param[str(bins)][j] *(1-u)
                
            #print active_param[str(temp_bins)],active_param[str(bins)]
            #check to see if in bounds
            #other methods of creating params
        elif bins - 1 > self._min_order:
            #death
            attempt = True #so program knows to attempt a new model
            #choose param randomly delete
            temp_bins = bins - 1 #nu.random.randint(self._min_order,bins)
            critera = 2**(bins - temp_bins) * (1. - birth_rate )
            #merge 2 params together
            for i in range(len(active_param[str(temp_bins)])):
                index = nu.random.randint(len(active_param[str(bins)]),size=2)
                u = nu.random.rand()
                active_param[str(temp_bins)][i] = active_param[str(bins)][index[0]] * u +  active_param[str(bins)][index[1]] * (1-u)
        else:
            #do nothing
            attempt = False
        if not attempt:
            temp_bins, critera = None,None
        else:
            j, j_timeleft = 0, nu.random.exponential(200)
        return active_param, temp_bins, attempt, critera, j, j_timeleft


class Gaussian_Mixture_model(object):
    '''Uses a mixture model of gausians to test RJMCMC infrenece'''
    def __init__(self,n_points=100,real_k=None):
        '''Creates data and stores real infomation for model
        y = sum_i^k w_i*N(mu_i,sigma_i)
        where sum_i^k w_i =1'''
        #initalize how many mixture compents
        if not real_k:
            self._k = nu.random.randint(1,10)
        else:
            self._k = real_k
        self._max_order = 10
        self._min_order = 1
        self._order = self._k
        #generate real parametes from normal [mu,sigma] sigma>0
        self._param = nu.random.randn(self._k,2)*9 
        self._param[:,1] = nu.abs(self._param[:,1])
        self._param = self._param[self._param[:,0].argsort()]
        #generate points
        '''temp_points = []
        for i in range(self._k):
            temp_points.append(nu.random.randn(n_points)*self._param[i,1] +
                               self._param[i,0])
        y,x = nu.histogram(nu.ravel(temp_points),bins=n_points)#,density=True)
        x = x[1:]'''
        x,y = nu.random.randn(n_points)*10, nu.zeros(n_points)
        x.sort()
        for i in range(0,self._k,2):
            y += stat_dist.norm.pdf(x,self._param[i,0],self._param[i,1])
        y *= 1000
        self.data = nu.vstack((x,y)).T
        #set prior values
        self.mu,self.var = 0.,9.
        
    

    def proposal(self,mu,sigma):
        '''drawing function for RJMC'''
        param = nu.random.multivariate_normal(mu,sigma)
        param = param.reshape((param.shape[0]/2,2))
        param = param[param[:,0].argsort()]
        param[:,1] = nu.abs(param[:,1])
        return nu.ravel(param)

    def lik(self,param,**args):
        '''log liklyhood calculation'''
        #create random points for histograming
        k = len(param)
        #n_points = self.data.shape[0]
        #temp_points = []
        y = nu.zeros_like(self.data[:,1])
        x = self.data[:,0]
        for i in range(0,k,2):
            y += stat_dist.norm.pdf(x,param[i],nu.abs(param[i+1]))
        y *= 1000
        out = stat_dist.norm.logpdf(self.data[:,1],y)
        #out = -nu.sum((self.data[:,1] - y)**2)
        return nu.sum(out) + self.prior(param)

    def disp_model(self,param):
        '''makes x,y axis for plotting of params'''
        k = len(param)
        #n_points = self.data.shape[0]
        #temp_points = []
        y = nu.zeros_like(self.data[:,1])
        x = self.data[:,0]
        for i in range(0,k,2):
            y += stat_dist.norm.pdf(x,param[i],nu.abs(param[i+1]))
        y *= 1000
        return x,y

    def prior(self,param):
        '''log prior and boundary calculation'''
        return stat_dist.norm.logpdf(param,self.mu,self.var).sum()
    
    def initalize_param(self,order):
        '''initalizes parameters for uses in MCMC'''
        order *= 2
        params = nu.random.multivariate_normal([0]*order,nu.identity(order)*9)
        sigma = nu.identity(order)*nu.random.rand()*9
        return params,sigma
    
    def step_func(self,accept_rate,param,sigma,order):
        '''changes step size'''
        if accept_rate > .25  and nu.all(sigma < 10):
            sigma *= 1.05
        elif accept_rate <.23  and nu.any(sigma > 10**-5):
            sigma /= 1.05
        if len(param[-1000:]) % 100 and len(param)>500:
            try:
                sigma = nu.cov(param[-1000:].T)
            except AttributeError:
                sigma = nu.cov(nu.array(param[-1000:]).T)
        return sigma

    def birth_death(self,birth_rate, bins, j, j_timeleft, active_param):
        if birth_rate > nu.random.rand() and bins+1 != self._max_order :
            #birth
            attempt = True #so program knows to attempt a new model
            #create step
            temp_bins = bins + 1 #nu.random.randint(bins+1,self._max_order)
            #criteria for this step
            critera = 10**(-temp_bins) * birth_rate
            #print temp_bins
            #new param step
            #active_param[str(temp_bins)][:bins] = active_param[str(bins)].copy()
            #draw new points randomly from prior (bad)
            #active_param[str(temp_bins)][bins:] = self.initalize_param(temp_bins)[0][bins:]
            #split move params W1 = w *u, W2=w*(1-u) 
            u,index = nu.random.rand(),[nu.random.randint(0,bins)]
            #add split to last 2 indicic's
            for j in range(2):
                active_param[str(temp_bins)][j*2] = (
                    active_param[str(bins)][index[0]*2] * abs(j-u))
                active_param[str(temp_bins)][j*2+1] = abs(
                    active_param[str(bins)][index[0]*2+1] *abs(j-u))
            #copy other vars
            j = 2
            while j< temp_bins:
                i = nu.random.randint(0,bins)
                while i  in index:
                    i = nu.random.randint(0,bins)
                index.append(i)
                active_param[str(temp_bins)][j*2] = active_param[str(bins)][i*2] 
                active_param[str(temp_bins)][j*2+1] = abs(active_param[str(bins)][i*2+1] )
                j+=1
            
            #check to see if in bounds
            #other methods of creating params
        elif bins - 1 >= self._min_order:
            #death
            attempt = True #so program knows to attempt a new model
            #choose param randomly delete
            temp_bins = bins - 1 #nu.random.randint(self._min_order,bins)
            critera = 10**(temp_bins) * (1. - birth_rate )
            if .5>nu.random.rand():
                #merge 2 params together
                #W1*u+W2*(1-u) = w 
                u,index = nu.random.rand(),nu.random.randint(0,bins,2)
                active_param[str(temp_bins)][0] = (
                    active_param[str(bins)][index[0]*2] *(1-u) +
                    active_param[str(bins)][index[1]*2]*u)
                active_param[str(temp_bins)][1] = (
                    active_param[str(bins)][index[0]*2+1] *(1-u) +
                    active_param[str(bins)][index[1]*2+1]*u)
                j = 1
                while j< temp_bins:
                    i = nu.random.randint(0,bins)
                    while i  in index:
                        i = nu.random.randint(0,bins)
                    index = nu.append(index,i)
                    active_param[str(temp_bins)][j*2] = active_param[str(bins)][i*2] 
                    active_param[str(temp_bins)][j*2+1] = abs(active_param[str(bins)][i*2+1] )
                    j+=1
            else:
                #drop param with smallest prior
                prob =[]
                for i in range(bins):
                    prob.append(self.prior(active_param[str(bins)][2*i:2*i+2]))
                index = nu.where(min(prob) != prob)[0]
                j = 0
                for i in index:
                    active_param[str(temp_bins)][j:j+2] = (
                        active_param[str(bins)][i:i+2].copy())
                    j += 1
        else:
            #do nothing
            attempt = False
        if not attempt:
            temp_bins, critera = None,None
        else:
            j, j_timeleft = 0, nu.random.exponential(200)
        return active_param, temp_bins, attempt, critera, j, j_timeleft


def lik_toyI(p=None,x=115,n=200):
    '''likelihood for toy I'''
    #if no p value sample
    if not nu.any(p):
        p = nu.ranom.rand()
    #calc likelhood
    
    return stat_dist.binom.pmf(x,n,p)

def lik_toyII(p,mu=0.,std=1):
    '''Test weather a point is drawn from gaussian with unknown mean, and 
    known std. Has a gausian prior on mean, with mu and std as mean and stadard
    devation'''
    return stat_dist.norm.pdf(p,mu,std)

def lik_toyIII(x,y,theta,sigma=1,answer=False):
    if answer:
        #calculate analyitical solution
        return False
    if len(theta.shape) == 1: #1-d array
        #mu = y - nu.polyval(theta,x)
        return stat_dist.norm.logpdf(y, nu.polyval(theta,x),sigma)
    else:
        mu = nu.zeros((theta.shape[0],len(x)))
        for i in range(theta.shape[1]):
            #mu += nu.tile(y,(len(mu),1)) - (nu.tile(theta[:,i],(len(x),1)).T*
                                            #x**(theta.shape[1] -1-i))
            mu += nu.tile(theta[:,i],(len(x),1)).T*x**(theta.shape[1] -1-i)
        return stat_dist.norm.logpdf(nu.tile(y,(len(mu),1)), mu,sigma)
    #return stat_dist.norm.logpdf(mu,0,sigma)

######################toy models############
def toyI(N,K,n=10**4):
    '''samples from a binomial distbution with like is binomal 
    and prior is uniform. Evidence is 1/(1+N). N is number of draws
    k number of succeses  and n is number of points to sample
    '''
    x = nu.random.rand(n)
    q = lik_toyI(x,K,N) / stat_dist.uniform.pdf(x)
    print 'Binomial is estimated %f, real %f'%(nu.mean(q),1/(1.+N))
   
def toyII(mu,std,n=10**4):
    '''With norm and norm prior (unknow mean, knowns std=prior_std)
    '''
    prior_mu = mu
    x = nu.random.randn(n)
    q = lik_toyII(x,prior_mu,std)**2 /stat_dist.norm.pdf(x)
    print 'Normal is estimated %f, real %f'%(nu.mean(q),stat_dist.norm.pdf(prior_mu,0,std))

def toyIII(order,n_points,n=10**4):
    '''linear regression model x and y data points. length of theta give 
    polynomial order. If answer is true, and len(x)<5 will give analytical 
    answer with theta=[-inf,int] with a normal prior for all theta'''
    order = 2 #polynomial order
    N = n_points #number of point to generate
    x = nu.linspace(1,5,N)
    real_m = nu.random.randn(order)-3
    #y = [  6.33623224,  -4.97248028, -16.18410389]
    y = nu.polyval(real_m,x) + nu.random.randn(len(x))
    #pm
    X = nu.random.multivariate_normal([0,0],nu.identity(2)*9,n)
    #cal prior
    #prior = nu.zeros(n)
    for i in range(order):
        if i == 0:
          prior =  stat_dist.norm.pdf(X[:,i],-3,1)
        else:
            prior *= stat_dist.norm.pdf(X[:,i],-3,1)
    logq = nu.sum(lik_toyIII(x,y,X,2),1) + nu.log(prior) - nu.sum(stat_dist.norm.logpdf(X,0,9),1)
    print 'Evidence is %.3g' %(nu.exp(logq,dtype=nu.float128).mean())

    #plot
    '''real_m = nu.array([-3,-3.]) #nu.random.randn(order)*9
    y = nu.polyval(real_m,x) + nu.random.randn(len(x))
    M,B = nu.meshgrid(nu.linspace(-50,50,10**2),nu.linspace(-50,50,10**2))
    X = nu.array(zip(M.ravel(),B.ravel()))
    for i in range(order):
        if i == 0:
          prior =  stat_dist.norm.pdf(X[:,i],-3,1)
        else:
            prior *= stat_dist.norm.pdf(X[:,i],-3,1)

    chi = nu.sum(lik_toyIII(x,y,X),1) + nu.log(prior)
    print 'Highest likelihood place is at (%1.2f,%1.2f)'%(X[chi.argmax()][0],X[chi.argmax()][1])
    chi[nu.isinf(chi)] = chi[~nu.isinf(chi)].min()
    chi = chi.reshape(M.shape)
    lab.contour(M,B,chi)
    lab.colorbar()
    '''
def toyIV(n_points=100,noise_sigma=1,n=10**4):
    '''draws data from a polynomial with noise and determineds which
    order of polynomial data came from, using evidence!
    '''
    #initalize param and input data
    x = nu.random.rand(n_points)*10 - 5
    x.sort()
    real_order = nu.random.randint(1,n_points-1)
    real_theta = nu.random.randn(real_order)*9
    y = (nu.polyval(real_theta,x) + 
         nu.random.randn(n_points)*noise_sigma)
    #start inference
    logq,offset,range_rev =[],[0.,True],range(1,11)
    range_rev.reverse()
    for i in range_rev:#n_points + 3+1):
        #sampling points
        X = nu.random.multivariate_normal(nu.zeros(i),
                                          nu.identity(i)*9,n)
        
        #calc prior
        for j in range(i):
            if j == 0:
                prior =  stat_dist.norm.pdf(X[:,j],0,9)
            else:
                prior *= stat_dist.norm.pdf(X[:,j],0,9)
        
        q = nu.sum(lik_toyIII(x,y,X,1),1) + nu.log(prior) - nu.sum(stat_dist.norm.logpdf(X,0,9),1)
        if offset[0] > q.min() and offset[1]:
            offset[0] = q.min()
            offset[1] = False
        #logq.append(nu.mean(nu.exp(q/offset[0],dtype=nu.float128)))
        logq.append(nu.mean(1/nu.abs(q)))
    print 'Real order is %i'%real_order
    lab.semilogy(range(1,n_points + 3+1),nu.mean(nu.exp(logq,dtype=nu.float128),1))
    print nu.mean(nu.exp(logq,dtype=nu.float128),1)[:5]


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
    #mpi version of Gaussian_model
    import Age_hybrid as hy
    import Gauss_landscape as G
    
    #initalize lik funcions and communication topologies
    fun = G.Gaussian_Mixture_model(1000,5)
    top = hy.Topologies('ring')
    #send root data
    if top.comm_world.rank == 0:
        top.comm_world.bcast(fun.data,root=0)
    else:
       fun.data =  top.comm_world.bcast(fun.data,root=0)
    
    param,chi,t = hy.mpi_general(fun, top, swarm_func=hy.vanilla, burnin=5000, itter=10**5)
