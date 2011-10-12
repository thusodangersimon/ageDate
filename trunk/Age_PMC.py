#!/usr/bin/env python
#
# Name:  population monte carlo
#
# Author: Thuso S Simon
#
# Date: Oct. 10 2011
# TODO: parallize with mpi 
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
""" uses PMC to find prosterior of stellar spectra age, metalicity and SFR"""

from Age_date import *
from mpi4py import MPI
from x_means import xmean
#import time as Time

a=nu.seterr(all='ignore')

def test():
    comm = MPI.COMM_WORLD
#size = comm.Get_size()
    rank = comm.Get_rank()
    print 'i am %i' %rank

def PMC_mixture(data,bins,n_dist=5,pop_num=10**4):
    #uses population monte carlo to find best fits and prosteror
    #data[:,1]=data[:,1]*1000.      
   #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
    alpha=nu.array([n_dist**-1.]*n_dist) #[U,N]
    '''#for multivarate dist params
    mu=nu.zeros([n_dist,bins*3])
    age_bins=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    for jj in range(n_dist):
        bin_index=0
        for k in xrange(mu.shape[1]):
            if any(nu.array(range(0,mu.shape[1],3))==k):#metalicity
                mu[jj,k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
            else:#age and normilization
                if any(nu.array(range(1,mu.shape[1],3))==k): #age
                #mu[k]=nu.random.random()
                    mu[jj,k]=nu.random.rand()*age_unq.ptp()/float(bins)+age_bins[bin_index]
               # mu[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                    bin_index+=1
                else: #norm
                    mu[jj,k]=nu.random.random()*10**4
    sigma=nu.array([nu.identity(bins*3)]*n_dist)
    for i in range(n_dist):
        sigma[i]=sigma[i]*(1000*nu.random.random()/bins)'''
    points=nu.zeros([pop_num,bins*3])
    bin_index=0
    age_bins=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    for k in xrange(bins*3):
        if any(nu.array(range(0,bins*3,3))==k):#metalicity
            points[:,k]=(nu.random.random(pop_num)*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,bins*3,3))==k): #age
                #mu[k]=nu.random.random()
                points[:,k]=nu.random.rand(pop_num)*age_unq.ptp()/float(bins)+age_bins[bin_index]
               # mu[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                bin_index+=1
            else: #norm
                points[:,k]=nu.random.random(pop_num)*10**4
 
    #build population parameters
    print 'initalizing mixture'
    #points=pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins)
    
    #get likelihoods
    lik=[]
    pool=Pool()
    for ii in points:
        pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
    pool.close()
    pool.join()
    lik=nu.array(lik,dtype=nu.float128)
    #calculate weights
    #keep in log space if are smaller than percision
    #alpha,mu,sigma=resample_first(lik)
    for i in range(10): #start refinement loop
        if i==0:
            if sum(lik[:,-1]<=11399)<30:
                lik[:,-1] =(1/lik[:,-1])
            else:
                lik[:,-1] =nu.exp(-lik[:,-1])
        else:
            if sum(lik[:,-1]<=11399)<30:
                for j in lik:
                    j[-1] =(1/j[-1])/(nu.nansum(alpha*norm_func(j[:-1],mu,sigma)))
            else:
                for j in lik:
                    j[-1] =nu.exp(-j[-1])/(nu.nansum(alpha*norm_func(j[:-1],mu,sigma)))
        #create best chi sample
        parambest=nu.zeros(bins*3)
        for j in range(bins*3):
            parambest[j]=sum(lik[:,j]*lik[:,-1]/sum(lik[:,-1]))
        chibest=like_gen(data,parambest,lib_vals,age_unq,metal_unq,bins)[-1]
        print 'best estimate chi squared values is %f, num of dist %i' %(chibest,len(alpha))
        #resample and get new alpha
        if i==0:
            alpha,mu,sigma=resample_first(lik)
        else:
            alpha,mu,sigma=resample(lik,nu.copy(alpha),nu.copy(mu),nu.copy(sigma))
        #gen new points
        points=pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins)
    #get likelihoods
        lik=[]
        pool=Pool()
        for ii in points:
            pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
        pool.close()
        pool.join()
        lik=nu.array(lik,dtype=nu.float128)
        #for j in lik:
         #   j[-1] =(1/j[-1])/(nu.nansum(alpha*norm_func(j[:-1],mu,sigma)))

    return lik,mu,sigma

def resample_first(lik):
    #uses xmeans clustering to adaptivly find starting  mixture densities
    weight=lik[:,-1]/sum(lik[:,-1])
    for j in range(lik.shape[1]-1): #gen cdf for each param
        sort_index=lik[:,j].argsort()
        x,y=nu.array(lik[sort_index,j],dtype=nu.float64),nu.array(nu.cumsum(weight[sort_index])/sum(weight),dtype=nu.float64)
            #gen rand numbers
        lik[:,j]=nu.interp(nu.random.rand(lik.shape[0]),y,x)
    #xmeans may fail so keeps trying till successfull run
    while True:
        try:
            clusters=xmean(lik[:,:-1],100)
            break
        except:
            pass
    #create distributions
    mu=nu.zeros([len(clusters.keys()),lik.shape[1]-1])
    sigma=nu.array([nu.identity(lik.shape[1]-1)]*len(clusters.keys()))
    alpha=nu.zeros([len(clusters.keys())])
    for i in range(len(clusters.keys())):
        mu[i,:]=nu.mean(clusters[clusters.keys()[i]],0)
        sigma[i]=nu.cov(clusters[clusters.keys()[i]].T)
        alpha[i]=float(clusters[clusters.keys()[i]].shape[0])/lik.shape[0]

    return alpha,mu,sigma

def resample(lik,alpha,mu,sigma):
    #resamples points according to weights and makes new 
    weight_norm=lik[:,-1]/sum(lik[:,-1])
    rho=nu.zeros([lik.shape[0],len(alpha)])
    for i in xrange(lik.shape[0]):#probablity of point beloging to distribution
        rho[i,:]= norm_func(lik[i,:-1],mu,sigma)/(sum(norm_func(lik[i,:-1],mu,sigma)))
    for i in xrange(len(alpha)): #calculate alpha
        alpha[i]=nu.sum( weight_norm*rho[:,i])
    for i in xrange(len(alpha)):#calc mu and sigma
        for j in xrange(mu.shape[1]):
            mu[i,j]=nu.sum(weight_norm*lik[:,j]*rho[:,i])/alpha[i]
    for k in xrange(mu.shape[0]):
        for i in xrange(mu.shape[1]):
            for j in xrange(mu.shape[1]):
                sigma[k][i,j]=nu.sum(weight_norm*rho[:,k]*(lik[:,i]-mu[k,i])*(lik[:,j]-mu[k,j]).T)/alpha[k]
    '''    
    for i in range(len(alpha)):
        if i==0:#calc indexes for sample
            start=0
        else:
            start=stop
        stop=start+int(lik.shape[0]*alpha[i])
        
        temp_lik=lik[start:stop,:]
        temp_weight=weight_norm[start:stop]
        for j in range(temp_lik.shape[1]-1): #gen cdf for each param
            sort_index=temp_lik[:,j].argsort()
            x,y=nu.array(temp_lik[sort_index,j],dtype=nu.float64),nu.array(nu.cumsum(temp_weight[sort_index])/sum(temp_weight),dtype=nu.float64)
            #gen rand numbers
            lik[start:stop,j]=nu.interp(nu.random.rand(stop-start),y,x)

        mu[i]=nu.mean(lik[start:stop,:-1],0)
        sigma[i]=nu.cov(lik[start:stop,:-1].T)
        alpha[i]=sum(temp_weight) #calculate new alpha
        '''
    #alpha=alpha/sum(alpha)
    #remove samples with not enough values
    while any(alpha*lik.shape[0]<100):
        index=nu.nonzero(alpha*lik.shape[0]<100)[0]
        alpha=nu.delete(alpha,index[0])
        mu=nu.delete(mu,index[0],0)
        sigma=nu.delete(sigma,index[0],0)

    #    alpha=alpha+100./lik.shape[0]
    alpha=alpha/sum(alpha)
    return alpha,mu,sigma


def student_t(x,mu,sigma,v=4):
    #calculates the proablility density of uniform multi dimensonal student
    #t dist with v degrees of freedom
    pass
    
def norm_func(x,mu,sigma):
    #calculates values of normal dist for set of points
    out=nu.zeros(mu.shape[0])
    for i in range(mu.shape[0]):
        out[i]=(2*nu.pi)**(-len(mu[i])/2.)*nu.linalg.det(sigma[i])**(-.5)
        try:
            out[i]=out[i]*nu.exp(-.5*(nu.dot((x-mu[i]),nu.dot(nu.linalg.inv(sigma[i]),(x-mu[i]).T))))
        except nu.linalg.LinAlgError:
            #sigma[i][sigma[i]==0]=10**-6
            out[i]=out[i]*nu.exp(-.5*(nu.dot((x-mu[i]),nu.dot(sigma[i]**-1,(x-mu[i]).T))))
            
    return out

def like_gen(data,active_param,lib_vals,age_unq,metal_unq,bins):
   #calcs chi squared values
    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    model=data_match_new(data,model,bins)
    N=[]
    for k in range(2,len(active_param),3):
        N.append(active_param[k])
    N=nu.array(N)
    model=nu.sum(nu.array(model.values()).T*N,1)
    #make weight paramer start closer to where ave data value
    return nu.hstack((active_param,sum((data[:,1]-model)**2)))
 
def pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins):
    #creates pop_num of points for evaluation
    #only uses a multivarate norm and unifor dist for now

    #check if alpha sums to 1
    if sum(alpha)!=1:
        alpha=alpha/sum(alpha)
    #initalize params
    points=nu.zeros([pop_num,bins*3])
    age_bins=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    #multivariate norm
    for j in range(mu.shape[0]):
        #start and stop points
        if j==0:
            start=0
        else:
            start=stop
        try:
            stop=start+int(pop_num*alpha[j])
        except ValueError:
            print alpha
            raise
        points[start:stop,:]=nu.random.multivariate_normal(mu[j],sigma[j],(stop-start))
        #check for values outside range
        bin_index=0
        for i in range(bins*3):
            if i==0 or i%3==0: #metalicity
                index=nu.nonzero(nu.logical_or(points[start:stop,i]< metal_unq[0],points[start:stop,i]> metal_unq[-1]))[0]
                index+=start
                for jj in index:
                    points[jj,:]=nu.random.multivariate_normal(mu[j],sigma[j])
                    while check(points[jj,:],metal_unq, age_unq, bins):
                        points[jj,:]=nu.random.multivariate_normal(mu[j],sigma[j])
        
            elif (i-1)%3==0 or i-1==0:#age
                index=nu.nonzero(nu.logical_or(points[start:stop,i]< age_bins[bin_index],points[start:stop,i]>age_bins[bin_index+1]))[0]
                bin_index+=1
                index+=start
                for jj in index:
                    points[jj,:]=nu.random.multivariate_normal(mu[j],sigma[j])
                    while check(points[jj,:],metal_unq, age_unq, bins):
                        points[jj,:]=nu.random.multivariate_normal(mu[j],sigma[j])
            elif (i-2)%3==0 or i==2: #norm
               #gen rand numbs to keep cov from being singular
                #index=nu.nonzero(nu.logical_or(points[start:stop,i]< 0,points[start:stop,i]>1))[0]
                #index+=start
                #points[index,i]=nu.random.rand(len(index))
                points[:,i]=nu.abs(points[:,i])
    #check for any missed values in array
    if nu.sum(points[:,0]==0)>0:
        index=nu.nonzero(points[:,0]==0)[0]
        alpha_index = nu.nonzero(alpha==alpha.max())[0][0] #gen points from most probablity place
        points[index,:]=nu.random.multivariate_normal(mu[alpha_index],sigma[alpha_index],len(index))
        for j in index:
            while check(points[j,:],metal_unq, age_unq, bins):
                points[j,:]=nu.random.multivariate_normal(mu[alpha_index],sigma[alpha_index])
        for j in range(2,bins*3,3):
            points[:,j]=nu.abs(points[:,j])

    #normalize normalization parameter
    '''if bins>1:
        temp=nu.zeros(pop_num)
        for i in range(2,bins*3,3):
            temp+=points[:,i]
        for i in range(2,bins*3,3):
            points[:,i]=points[:,i]/temp
            '''
    return points


def toy():

    pass
