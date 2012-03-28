#!/usr/bin/env python
#
# Name:  Age Dating Spectra Fitting Program
#
# Author: Thuso S Simon
#
# Date: 7th of June, 2011
#TODO:  
#     make solutions of N_normalize>0 always
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
""" A python version of the age dating spectral fitting code done by Mongwane 2010"""

import numpy as nu
import os,sys
from multiprocessing import *
from interp_func import *
from spectra_func import *
from scipy.optimize import nnls
from scipy.optimize import fmin_l_bfgs_b as fmin_bound
from scipy.special import expi
import time as Time

###spectral lib stuff####
global lib_path,spect
lib_path='/home/thuso/Phd/Spectra_lib/'
spect,info= load_spec_lib(lib_path)  


def find_az_box(param,age_unq,metal_unq):
    #find closest metal
    line=None
    dist=(param[0]-metal_unq)**2
    #test to see if smaller or larger
    metal=[metal_unq[nu.argsort(dist)][0]]*2
    try:
        if metal_unq[nu.argsort(dist)][0]>param[0]:
            for i in xrange(2):
                metal.append(metal_unq[metal_unq<metal[0]][-1])
        elif any(dist==0):
            line='metal'
        else:
            for i in xrange(2):
                metal.append(metal_unq[metal_unq>metal[0]][0])
    except IndexError: #if on a line
        line='metal'
    #find closest age 
        
    try:
        dist=(param[1]-age_unq)**2
        age=[age_unq[nu.argsort(dist)][0]]*2
        if age_unq[nu.argsort(dist)][0]>param[1]:
            for i in xrange(2):
                age.append(age_unq[age_unq<age[0]][-1])
        elif any(dist==0):
            if not line:
                line='age'
            else:
                line='both'
            return metal,age,line
        else:
            for i in xrange(2):
                age.append(age_unq[age_unq>age[0]][0])
    except IndexError:
        if not line:
            line='age'
        else:
            line='both'
            
    return metal,age,line

def binary_search(array,points):
    #does binary search for closest points returns 2 closest points
    #array.sort()
    start=int(len(array)/2.)
    if array[start]>points: #go down
        i=-nu.array(range(start))
        option='greater'
    else:
        i=xrange(start)
        option='less'
    for j in i:
        if option=='greater' and array[start+j]<points:
            out= (array[start+j],array[start+j+1])
            break
        elif option=='less' and array[start+j]>points:
            out= (array[start+j],array[start+j-1])
            break 

    return out

def get_model_fit_opt(param,lib_vals,age_unq,metal_unq,bins):
    #does dirty work to make spectra models
    #search age_unq and metal_unq to find closet box spectra and interps
    #does multi componets spectra and fits optimal normalization
    out={}
    for ii in xrange(bins):
        temp_param=param[ii*3:ii*3+2]
        metal,age,line=find_az_box(temp_param,age_unq,metal_unq)
        closest=[]
        #check to see if on a lib spectra or on a line
        if line=='age': #run 1 d interp along metal only
            metal=nu.array([metal[0],metal[-1]])
            metal.sort()
            age=age[0]
            #find spectra
            for i in 10**metal:
                index=nu.nonzero(nu.logical_and(lib_vals[0][:,0]==i,
                                            lib_vals[0][:,1]==age))[0]
                #closest.append(read_spec(lib_vals[1][index][0]))
                closest.append(spect[:,index[0]+1])

            out[str(ii)]=linear_interpolation(10**metal,closest,10**temp_param[0])
        elif line=='metal': #run 1 d interp along age only
            age=nu.array([age[0],age[-1]])
            age.sort()
            metal=metal[0]
            #find spectra
            
            for i in age:
                index=nu.nonzero(nu.logical_and(lib_vals[0][:,1]==i,
                                            lib_vals[0][:,0]==10**metal))[0]
                #closest.append(read_spec(lib_vals[1][index][0]))
                closest.append(spect[:,index[0]+1])
            
            out[str(ii)]=linear_interpolation(age,closest,temp_param[1])

        elif line=='both': #on a lib spectra
            index=nu.nonzero(nu.logical_and(lib_vals[0][:,0]==10**temp_param[0],
                                            lib_vals[0][:,1]==temp_param[1]))[0]
            out[str(ii)]=nu.copy(spect[:,index[0]+1])
        #run 2 d interp
        else:
            metal.sort()
            metal=nu.array(metal)[nu.array([0,3,1,2],dtype='int32')]
            age.sort()
            
            for i in range(4):
                #print metal[i],age[i]
                index=nu.nonzero(nu.logical_and(lib_vals[0][:,1]==age[i],
                                            lib_vals[0][:,0]==10**metal[i]))[0]
                #closest.append(read_spec(lib_vals[1][index][0]))
                closest.append(spect[:,index[0]+1])

        #interp
            out[str(ii)]=bilinear_interpolation(10**metal,age,closest,
                                                      10**temp_param[0],temp_param[1])
    #give wavelength axis
    out['wave']=nu.copy(spect[:,0])

   #exit program
    return out

def data_match_all(data):
    #makes sure data and model have same wavelength range and points for library
    model={}
    global spect
    for i in xrange(spect[0,:].shape[0]):
        if i==0:
            model['wave']=nu.copy(spect[:,i])
        else:
            model[str(i-1)]=nu.copy(spect[:,i])
    #see if spectral lib is square ie has all combo's of age and metals
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    if not lib_vals[0].shape[0]==metal_unq.shape[0]*age_unq[0]:
        print 'Warrning: Not every combination of age and metalicty avalible in library'
        print 'May cause problems with fitting'

    model=data_match_new(data,model,spect[0,:].shape[0]-1)
    out=nu.zeros([model['0'].shape[0],len(model.keys())+1])
    out[:,0]=nu.copy(data[:,0])
    for i in model.keys():
        out[:,int(i)+1]=model[i]
    spect=nu.copy(out)

def data_match_new(data,model,bins):
    #makes sure data and model have same wavelength range and points but with a dictionary
    out={}
    if model['wave'].shape[0]==data.shape[0]: #if same number of points
        if all(model['wave']==data[:,0]):#if they have the same x-axis
            for i in xrange(bins):
                out[str(i)]=model[str(i)]
        else: #if same number of points but different at points
            print 'data match not ready yet'
            raise
    else: #not same shape, interp and or cut
        index=nu.nonzero(nu.logical_and(model['wave']>=min(data[:,0]),
                                        model['wave']<=max(data[:,0])))[0]
        if index.shape[0]==data.shape[0]: #see if need to cut only
            if sum(model['wave'][index]==data[:,0])==index.shape[0]:
                for i in xrange(bins):
                    out[str(i)]=model[str(i)][index]
                #out['wave']=model['wave'][index]
            else: #need to interpolate but keep same size
                for i in xrange(bins):
                    out[str(i)]=spectra_lin_interp(model['wave'],model[str(i)],data[:,0])
                #model['wave']=data[:,0]
        else:
            for i in range(bins):
                out[str(i)]=spectra_lin_interp(model['wave'],model[str(i)],data[:,0])
            #model['wave']=data[:,0]
    return out
    

def check(param,metal_unq, age_unq,bins): #checks if params are in bounds
    age=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))#linear spacing
    for j in xrange(bins):#check age and metalicity
        if any([metal_unq[-1],age[j+1]]<param[j*3:j*3+2]) or any([metal_unq[0],age[j]]>
                                                                 param[j*3:j*3+2]):
            
            '''if any([metal_unq[-1],age_unq[-1]]<param[j*3:j*3+2]) or any([metal_unq[0],age_unq[0]]>param[j*3:j*3+2]):'''
            return True
        #if any(nu.abs(nu.diff(param.take(range(1,bins*3,3))))<.3):
        #    return True
        if not (0<param[j*3+2]): #and param[j*3+2]<1): #check normalizations
            return True
    return False

def normalize(data,model):
    #normalizes the model spectra so it is closest to the data
    return sum(data[:,1]*model)/sum(model**2)

def N_normalize(data, model,bins):
    #takes the norm for combined data and does a minimization for best fits value
    
    #match data axis with model
    model=data_match_new(data,model,bins)
    #do non-negitave least squares fit
    if bins==1:
        N=[normalize(data,model['0'])]
        return N, N[0]*model['0'],sum((data[:,1]-N[0]*model['0'])**2)
    try:
        N,chi=nnls(nu.array(model.values()).T[:,nu.argsort(nu.int64(nu.array(model.keys())))],data[:,1])
    except RuntimeError:
        print "nnls error"
        N=nu.zeros([len(model.keys())])
        chi=nu.inf
    index=nu.nonzero(N==0)[0]
    N[index]+=10**-6
    index=nu.int64(model.keys())
    return N,nu.sum(nu.array(model.values()).T*N[index],1),chi**2

def chain_gen_all(means,metal_unq, age_unq,bins,sigma):
    #creates new chain for MCMC, does log spacing for metalicity
    #lin spacing for everything else, runs check on values also
    out=nu.random.multivariate_normal(means,sigma)
    t=Time.time()
    while check(out,metal_unq, age_unq,bins):
        out=nu.random.multivariate_normal(means,sigma)
        if Time.time()-t>.5:
            sigma=sigma/1.05
            if Time.time()-t>2.:
                print "I'm %i and I'm stuck" %current_process().ident
                print means,sigma
                raise
    #N=sum(means.take(range(2,bins*3,3)))
    #for i in range(2,bins*3,3):#normalize normalization to 1
    #    means[i]=means[i]/N

    return out

def multivariate_student(mu,sigma,n):
    #samples from a multivariate student t distriburtion
    #with mean mu,sigma as covarence matrix, and n is degrees of freedom
    #as n->inf this goes to gaussian
    return mu+nu.random.multivariate_normal([0]*len(mu),sigma)*(n/nu.random.chisquare(n))**0.5

def nn_ls_fit(data,max_bins=16,min_norm=10**-4,spect=spect):
    #uses non-negitive least squares to fit data
    #spect is libaray array
    #match wavelength of spectra to data change in to appropeate format
    model={}
    for i in xrange(spect[0,:].shape[0]):
        if i==0:
            model['wave']=nu.copy(spect[:,i])
        else:
            model[str(i-1)]=nu.copy(spect[:,i])

    model=data_match_new(data,model,spect[0,:].shape[0]-1)
    index=nu.int64(model.keys())
    
    #nnls fit handles uncertanty now 
    if data.shape[1]==2:
        N,chi=nnls(nu.array(model.values()).T[:,nu.argsort(nu.int64(nu.array(model.keys())))],data[:,1])
    elif data.shape[1]==3:
        N,chi=nnls(nu.array(model.values()).T[:,nu.argsort(nu.int64(nu.array(model.keys())))]/nu.tile(data[:,2],(bins,1)).T,data[:,1]/data[:,2])
    #N,chi=nnls(nu.array(model.values()).T,data[:,1])
    N=N[index.argsort()]
    
    #check if above max number of binns
    if len(N[N>min_norm])>max_bins:
        #remove the lowest normilization
        N_max_arg=nu.nonzero(N>min_norm)[0]
        N_max_arg=N_max_arg[N[N_max_arg].argsort()]
        #sort by norm value
        current=[]
        for i in xrange(N_max_arg.shape[0]-1,-1,-1):
            current.append(info[N_max_arg[i]])
            if len(current)==max_bins:
                break
        current=nu.array(current)
    else:
        current=info[N>min_norm]
    metal,age=[],[]
    for i in current:
        metal.append(float(i[4:10]))
        age.append(float(i[11:-5]))
    metal,age=nu.array(metal),nu.array(age)
    #check if any left
    return metal[nu.argsort(age)],age[nu.argsort(age)],N[N>min_norm][nu.argsort(age)]

def rand_choice(x,prob):
    #chooses value from x with probabity of prob**-1 so lower prob values are more likeliy
    #x must be monotonically increasing
    if not nu.sum(prob)==1: #make sure prob equals 1
        prob=prob/nu.sum(prob)
    #check is increasing
    u=nu.random.rand()
    if nu.all(x==nu.sort(x)): #if sorted easy
        N=nu.cumsum(prob**-1/nu.sum(prob**-1))
        index=nu.array(range(len(x)))
        return index[nu.min(nu.abs(N-u))==nu.abs(N-u)][0]
    else:
        index=nu.argsort(x)
        temp_x=nu.sort(x)
        N=nu.cumsum(prob[index]**-1/nu.sum(prob[index]**-1))
        return index[nu.min(nu.abs(N-u))==nu.abs(N-u)][0]

def dict_size(dic):
    #returns total number of elements in dict
    size=0
    for i in dic.keys():
        size+=len(dic[i])

    return size

def dust(param,model,tau_ism,tau_bc):
    #does 2 componet dust calibration model following charlot and fall 2000
    t_bc=7.4771212547196626 #log10(.03*10**9)
    bins=int(param.shape[0]/3.)
    if model['wave'].mean()>9000 or model['wave'].mean()<3000:
        print 'Warring: dust calculations may be wrong'
    for i in range(bins):
        if param[i+1]<=t_bc: #choose which combo of dust models to use
            #fdust*fbc
            model[str(i)]=f_dust(tau_ism)*f_dust(tau_bc)*model[str(i)]
        else:
            #fdust
            model[str(i)]=f_dust(tau_ism)*model[str(i)]
    return model

def f_dust(tau):
    #dust extinction functin
    if tau<.03: #this is only valid when mean of obs wavelength is close to 5500 angstroms
        return 1/(2.*tau)*(1+(tau-1)*nu.exp(-tau)-tau**2*expi(tau))
    else:
        return nu.exp(-tau)

#####classes############# 
class PMC_func:
    #makes more like function, so input params and the chi is outputted
    def __init__(self,data,bins,spect=spect):
        data_match_all(data)
        self.data=data
        lib_vals=get_fitting_info(lib_path)
        lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
        metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
        age_unq=nu.unique(lib_vals[0][:,1])
        self.lib_vals=lib_vals
        self.age_unq= age_unq
        self.metal_unq,self.bins,self.spect=metal_unq,bins,spect
        self.bounds()

    def func(self,param):
        if len(param)!=self.bins*3:
            return nu.nan
        if check(param,self.metal_unq, self.age_unq,self.bins): #make sure params are in correct range
            for i in xrange(len(self.bounds)): #find which is out and fix
                if self.bounds[i][0]>param[i]: #if below bounds
                    param[i]=nu.copy(self.bounds[i][0])
                if self.bounds[i][1]<param[i]: #if above bounds
                    param[i]=nu.copy(self.bounds[i][1])

        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,self.bins)  
    #model=data_match_new(data,model,bins)
        index=xrange(2,self.bins*3,3)
        model['wave']= model['wave']*.0
        for ii in model.keys():
            if ii!='wave':
                model['wave']+=model[ii]*param[index[int(ii)]]
        return nu.sum((self.data[:,1]-model['wave'])**2)

    def func_N_norm(self,param):
        #returns chi and N norm best fit params
        if len(param)!=self.bins*3:
            return nu.nan
        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,self.bins)  
        N,model,chi=N_normalize(self.data, model,self.bins)
    
        return chi,N

 
    def min_bound(self):
        #outputs an array of minimum values for parameters
        out=nu.zeros(self.bins*3)
        bin=nu.log10(nu.linspace(10**self.age_unq.min(),10**self.age_unq.max(),self.bins+1))
        bin_index=0
        for k in range(self.bins*3):
            if any(nu.array(range(0,self.bins*3,3))==k): #metal
                out[k]=self.metal_unq[0]
            elif any(nu.array(range(1,self.bins*3,3))==k): #age
                out[k]=bin[bin_index]
                bin_index+=1
            elif any(nu.array(range(2,self.bins*3,3))==k): #norm
                out[k]=0.0
        return out

    def max_bound(self):
        #outputs an array of maximum values for parameters
        out=nu.zeros(self.bins*3)
        bin=nu.log10(nu.linspace(10**self.age_unq.min(),10**self.age_unq.max(),self.bins+1))
        bin_index=1
        for k in range(self.bins*3):
            if any(nu.array(range(0,self.bins*3,3))==k): #metal
                out[k]=self.metal_unq[-1]
            elif any(nu.array(range(1,self.bins*3,3))==k): #age
                out[k]=bin[bin_index]
                bin_index+=1
            elif any(nu.array(range(2,self.bins*3,3))==k): #norm
                out[k]=nu.inf
        return out

    def bounds(self):
        Min=self.min_bound()
        Max=self.max_bound()
        out=[]
        for i in range(len(Min)):
            out.append((Min[i],Max[i]))
        self.bounds=nu.copy(out)
        return out

    def n_neg_lest(self,param):
        #does bounded non linear fit
        try:
            out=fmin_bound(self.func,param, bounds = self.bounds,approx_grad=True)[0]
        except IndexError:
            out=param
        return out




if __name__=='__main__':
    import cProfile as pro
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    bins=16
    active_param=nu.zeros(bins*3)
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0
    #start in random place
    for k in xrange(3*bins):
        if any(nu.array(range(0,bins*3,3))==k):#metalicity
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,bins*3,3))==k): #age
                #active_param[k]=nu.random.random()
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
            else: #norm
                #active_param[k]=nu.random.random()
                pass

    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)
    data,info1,weight=create_spectra(bins,lam_min=2000,lam_max=10**4)
    pro.runctx('N_normalize(data,model,bins)'
               , globals(),{'data':data,'model':model,'bins':bins}
               ,filename='agedata.Profile')
