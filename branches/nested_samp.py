#!/usr/bin/env python
#
# Name:  Nested Sampling Posterior Evalulation
#
# Author: Thuso S Simon
#
# Date: 17 of April, 2011
#TODO: Add multiprocessing,put in stop evidence condition later
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
""" A program to sample the Baysien evidence (Z) in hopes of determing the Postererior. Method comes from Skilling and Feroz & Hobson (2008) 

Method:

Calculates the evidence by sampling the prior and evaulaing the likeiyhood at N possiotons e.g. 

                    Z=int(L(theta)*Pi(theta) d theta)

changes the Pi(thedta)d theta into dX 1-D integral and evalueates it by trapizode law. Each itterations composes of from Mukherjoe, Parkinson and Liddle (2008)

        1. sample N points randomly in prior
        2. Find point with lowest liklihood (L_i) and the X_i can be estimated probablistically where X_i/X_i-1=t
        3. increment the evidence by E_i=L_i*(X_i-1-X_i+1)/2
        4. discard L_i and sample new point in new prior volume
        5.repeat till evidence comes to desired accuracy

"""



#import Gauss_landscape as gl
from x_means import xmean
from Age_date import *
a=nu.seterr(all='ignore')
def nested_sampling(data,N,bins,elogf=1.1,stop=10**-3):
#main nested sampling program
    #sample N points evenly through out boundaries
    #param location [points in M dimensions,like] 
    points=nu.zeros([N,3*bins+2])
    data[:,1]=data[:,1]*1000.
    #get boundaries
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0
    #start in random place
    for k in xrange(points.shape[1]):
        if any(nu.array(range(0,points.shape[1],3))==k):#metalicity
           points[:,k]=(nu.random.random(points.shape[0])*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,points.shape[1],3))==k): #age
                points[:,k]=nu.random.random(points.shape[0])*age_unq.ptp()/float(bins)+bin[bin_index]
                #points[:,k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                bin_index+=1
            else: #norm
                points[:,k]=nu.random.random(points.shape[0])
                       
    #calculate likeihood for each point and sort
    po=Pool()
    temp=[]
    for j in points:
        #multicore
        po.apply_async(multi_samp,(j,data,lib_vals,metal_unq,age_unq,bins,),callback=temp.append)
        #j=multi_samp(j,data,lib_vals,metal_unq,age_unq,bins)
    po.close()
    po.join()
    points=nu.copy(nu.array(temp))

    index=nu.argsort(points[:,-1])
    points=points[index,:]
    #store discared points for later inference
    old_points=[] 
    #old_points_txt='likelihood[-1]*weight[-1]'
    #for i in range(len(points[0,1:])):
    #    old_points_txt=old_points_txt+',points[0,1:]['+str(i)+']'
    #initalize prior volume and evidence calculation
    prior_vol,likelihood,weight=[1],[],[]
    #evid=[0]
    i,n_fail=1,0
    #start nested sampling
    while (points[-1,-1]-points[0,-1])*prior_vol[-1]>2*10**-3 or i<500: ####put in stop condition later
        #step 2 and 3
        print (points[-1,-1]-points[0,-1])*prior_vol[-1], i
        likelihood.append(points[-1,-1])
        weight.append((nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        #evid.append(points[0,0]*(nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-i/(N+.0)))
        #recode old point for later inference
        old_points.append(points[-1,:])
        #step 4 get new liklihood value
        #find new sampling by L_new>L_old ###slow
        
        #simple ellipse sampling
        '''temp_points=ellipse_Samp(points[:N-2,:].max(0),points[:N-2,:].min(0),age_unq,metal_unq,
                                 elogf,bins)
        temp_points=multi_samp(temp_points,data,lib_vals,metal_unq,age_unq,bins)
        while points[-1,-1]<=temp_points[-1]:
            temp_points=ellipse_Samp(points[:N-2,:].max(0),points[:N-2,:].min(0),age_unq,metal_unq,
                                 elogf,bins)
            temp_points=multi_samp(temp_points,data,lib_vals,metal_unq,age_unq,bins)
            n_fail+=1
            '''
        #MCMC samp
        '''temp_points=MCMC_samp(data,points,age_unq,metal_unq,lib_vals,bins)
        while points[-2,-1]<=temp_points[-1]:
            temp_points=MCMC_samp(data,points,age_unq,metal_unq,lib_vals,bins)
            '''
        #better ellipse samp
        temp_points= cov_ellipse_samp(points,age_unq,metal_unq,elogf,bins)
        temp_points=multi_samp(temp_points,data,lib_vals,metal_unq,age_unq,bins)
        while points[-5,-1]<=temp_points[-1]:
            temp_points=ellipse_Samp(points[:N-2,:].max(0),points[:N-2,:].min(0),age_unq,metal_unq,
                                 elogf,bins)
            temp_points=multi_samp(temp_points,data,lib_vals,metal_unq,age_unq,bins)
            n_fail+=1
 
         #print abs(i-n_fail)/float(i)
        #insert new point, sort
        points[-1,:]=nu.copy(temp_points)
        index=nu.argsort(points[:,-1])
        points=points[index,:]
        i+=1

    #change old_point_txt to match new stuff
    #old_points_txt='likelihood[-1]*weight[-1]'
    #for j in range(len(points[0,1:])):
    #    old_points_txt=old_points_txt+',points[j-i,1:]['+str(j)+']'
 
#calculate evidence for remaing points
    for j in range(len(points[:,0])):
        #evid.append(points[j-i,0]*(nu.exp(-(j-1.0)/(N+.0))-nu.exp(-(j+1.0)/(N+.0)))/2.0)
        likelihood.append(points[j,-1])
        weight.append((nu.exp(-(j-1.0+i)/(N+.0))-nu.exp(-(j+1.0+i)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-(j+j)/(N+.0)))
        old_points.append(points[-j,:])
        #old_points.append([eval(old_points_txt)])
    #turn list into for manipulation
    likelihood,weight,prior_vol,old_points=nu.array(likelihood),nu.array(weight),nu.array(prior_vol),nu.array(old_points)
    #calculate the uncertany in evidence
    #calculate entropy
    H=nu.nansum(likelihood*weight/sum(likelihood*weight)*nu.log(likelihood/sum(likelihood*weight)))
    #print H
    #calculate uncertanty in evidemce
    evid=likelihood*weight
    evid_error=nu.sqrt(H/float(N))
    data[:,1]=data[:,1]/1000.
    return evid,nu.array(prior_vol[1:]),evid_error,old_points

def simple_ellipse_Samp(point_max,point_min,age_unq,metal_unq,elogf,bins):
    #samples points inside a N-d ellipse given input points and expanded by elogf
    #makes sure in boundaries of problem for first few itterations
    temp_out=(nu.random.rand(point_max.shape[0])*(point_max-point_min)+point_min)*elogf
    #make sure in boundaries of paramerter space
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0
    #start in random place
    for k in xrange(len(temp_out)-2):
        if any(nu.array(range(0,len(temp_out)-2,3))==k):#metalicity
           temp_out[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,len(temp_out)-2,3))==k): #age
                temp_out[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                #points[:,k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                bin_index+=1
            else: #norm
                temp_out[k]=nu.random.random()
 
    return temp_out

def cov_ellipse_samp(points,age_unq,metal_unq,elogf,bins):
    #uses covarence matrix to constrain new points
    cov=nu.mat(nu.cov(points[:,:2].T))
    mean=points[:,:2].mean(0)
    #x=nu.mat(nu.ones(2))
    temp_out=nu.mat(nu.random.multivariate_normal(mean,cov))
    while (temp_out-mean)*cov**-1*(temp_out-mean).T>7 or check(nu.hstack((nu.array(temp_out)[0],0.5)),metal_unq, age_unq,bins):
        temp_out=nu.mat(nu.random.multivariate_normal(mean,cov))
    #make work for multiple binsfor i in range(0,points.shape[1],3):
    out=nu.zeros(points.shape[1])
    out[0]=temp_out[0,0]
    out[1]=temp_out[0,1]
    out[2]=nu.random.rand()
    return out
    
def MCMC_samp(data,points,age_unq,metal_unq,lib_vals,bins,itter=20):
    #runs quick MCMC on the point with the highest liklihood to generate new points
    
    active_param=nu.copy(points[-1,:])
    #figure out current limits
    age=nu.array([points[:,1].min(),points[:,1].max()])
    metal=nu.array([points[:,0].min(),points[:,0].max()])
    #step
    sigma=nu.cov(points[:,:3].T)
    for i in xrange(itter):
        active_param[:-2]= chain_gen_all(active_param[:-2],metal, age,bins,sigma)
        model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins) 
        model=data_match_new(data,model,bins)
        N=[]
        for k in range(2,len(active_param),3):
            N.append(active_param[k])
        N=nu.array(N)
        model=nu.sum(nu.array(model.values()).T*N,1)
        active_param[-2]=normalize(data,model)
        active_param[-1]=sum((data[:,1]-active_param[-2]*model)**2)
        if min([1,nu.exp(((points[-1,-1]-active_param[-1])/2))])>nu.random.rand():
            #accepted
            points[-1,:]=nu.copy(active_param)
        
    return active_param
               


def multi_samp(j,data,lib_vals,metal_unq,age_unq,bins):
    #callable function for generating first iteration of points
    model=get_model_fit_opt(j[:-2],lib_vals,age_unq,metal_unq,bins) 
    model=data_match_new(data,model,bins)
    N=[]
    for k in range(2,len(j),3):
        N.append(j[k])
    N=nu.array(N)
    model=nu.sum(nu.array(model.values()).T*N,1)
    j[-2]=normalize(data,model)
    j[-1]=sum((data[:,1]-j[-2]*model)**2)
    return j

########################################################

def nest_elips(like_obj,N=500,elogf=1.06):
    #Does Ellipsoidal Nested Sampling as discribed in Feroz and Hobson (2008)
    #the elipse is s'pose to map the iso-likelhoood contours of the likelihood

    points=nu.zeros([N,like_obj._bounds.shape[0]+1])
    for i in range(1,1+points[:,1:].shape[1]): #assing points in bounds
        points[:,i]=nu.random.rand(len(points[:,i]))*nu.diff(like_obj._bounds[i-1])+nu.mean(like_obj._bounds[i-1,0])
        
    #calculate likeihood for each point and sort
    points[:,0]=like_obj.likeihood_value(points[:,1], points[:,2]) #2-d
    index=nu.argsort(points[:,0])
    points=points[index,:]
    #store discared points for later inference
    old_points=[] #[prior weight,points]
    old_points_txt='likelihood[-1]*weight[-1]'
    for i in range(len(points[0,1:])):
        old_points_txt=old_points_txt+',points[0,1:]['+str(i)+']'
    #initalize prior volume and evidence calculation
    prior_vol,likelihood,weight=[1],[],[]
    #evid=[0]
    i,n_fail=1,0
    #start nested sampling
    while (points[-1,0]-points[0,0])*prior_vol[-1]>10**-4: ####put in stop condition later
        #step 2 and 3
        likelihood.append(points[0,0])
        weight.append((nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        #evid.append(points[0,0]*(nu.exp(-(i-1.0)/(N+.0))-nu.exp(-(i+1.0)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-i/(N+.0)))
        #recode old point for later inference
        old_points.append([eval(old_points_txt)])
        #step 4 get new liklihood value
           #transform axis using Shaw's method
        
        #find new sampling by L_new>L_old ###slow
        temp_points=nu.zeros([1,points[:,1:].shape[1]+1])
        temp_points[0,1:]=elogf*(coord_trans(points[:,1:])*unit_cir_dis(points[:,1:].shape[1])).ravel()+nu.mean(points[1:,1:],axis=0) #new point
        temp_points[0,0]=like_obj.likeihood_value(temp_points[0,1],temp_points[0,2])
        while points[0,0]>=temp_points[0,0]:
            temp_points[0,1:]=elogf*(coord_trans(points[:,1:])*unit_cir_dis(points[:,1:].shape[1])).ravel()+nu.mean(points[1:,1:],axis=0) #new point
            temp_points[0,0]=like_obj.likeihood_value(temp_points[0,1],temp_points[0,2])
            n_fail+=1
        #print abs(i-n_fail)/float(i)
        #insert new point, sort
        points[0,:]=temp_points+0.0
        index=nu.argsort(points[:,0])
        points=points[index,:]
        i+=1

    #change old_point_txt to match new stuff
    old_points_txt='likelihood[-1]*weight[-1]'
    for j in range(len(points[0,1:])):
        old_points_txt=old_points_txt+',points[j-i,1:]['+str(j)+']'
 
#calculate evidence for remaing points
    for j in range(i+1,i+len(points[:,0])):
        #evid.append(points[j-i,0]*(nu.exp(-(j-1.0)/(N+.0))-nu.exp(-(j+1.0)/(N+.0)))/2.0)
        likelihood.append(points[j-i-1,0])
        weight.append((nu.exp(-(j-1.0)/(N+.0))-nu.exp(-(j+1.0)/(N+.0)))/2.0)
        prior_vol.append(nu.exp(-(j)/(N+.0)))
        old_points.append([eval(old_points_txt)])
    #turn list into for manipulation
    likelihood,weight,prior_vol,old_points=nu.array(likelihood),nu.array(weight),nu.array(prior_vol),nu.array(old_points)[:,0]
    #calculate the uncertany in evidence
    #calculate entropy
    H=nu.nansum(likelihood*weight/sum(likelihood*weight)*nu.log(likelihood/sum(likelihood*weight)))
    #print H
    #calculate uncertanty in evidemce
    evid=likelihood*weight
    evid_error=nu.sqrt(H/float(N))

    return evid,prior_vol[1:],evid_error,old_points

    
def unit_cir_dis(dimens=1):
    #samples points from a unit circle distribution so the r**2<1
    x=nu.random.rand(dimens)*2-1
    return nu.matrix(nu.random.rand()**(1/2.)*x/(nu.sqrt(sum(x**2)))).T

def coord_trans(active_points):
    #transforms axis using correlation matrix
    #T=k*X.T*D, where X=eigvector of C=cov(X) C_diag=X.T*C*C
    #D=C_diag**.5 and k=max(x.T*C**-1*x) x=active points
    C=nu.matrix(nu.cov(active_points.T))
    [x,X]=nu.linalg.eig(C)
    #k=nu.max(x*C**-1*nu.matrix(x).T)

    return X.T*nu.sqrt(nu.abs(X.T*C*X))

