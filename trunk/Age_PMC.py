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
import pypar as par
from x_means import xmean
from scipy.special import gamma
#import time as Time

#a=nu.seterr(all='ignore')

def test():
    myID = par.rank()
    bins=2
    n_dist=5
    pop_num=10**4
    if myID==0: #master process
        data,info1,weight=create_spectra(bins,'line',2000,10**4,slope=1.2)
        lib_vals=get_fitting_info(lib_path)
        lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
        metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
        age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
        alpha=nu.array([n_dist**-1.]*n_dist) #[U,N]
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
                else: #norm stuff
                    points[:,k]=nu.random.random(pop_num)*10**4
        #send message to calculate liklihoods
        par.send()

        #like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.appen


    else:
        lib_vals=get_fitting_info(lib_path)
        lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
        metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
        age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
        alpha=nu.array([n_dist**-1.]*n_dist) #[U,N]
        print '%i is ready!' %myID
        while True:
            todo=par.receive(0)
            if todo=='lik': #calculate likelihood
                pass
            if todo=='q': #do norm stuff
                pass

    par.finalize()

def PMC_mixture_old(data,bins,n_dist=1,pop_num=10**4):
    #uses population monte carlo to find best fits and prosteror
    #data[:,1]=data[:,1]*1000.      
   #initalize parmeters and chi squared
    data_match_all(data)
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
    alpha=nu.array([n_dist**-1.]*n_dist) #[U,N]
    out={}
    
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
                points[:,k]=nu.random.random(pop_num)*10**3
 
    #build population parameters
    print 'initalizing mixture'
    i=0
    while True:
    #get likelihoods
        if not i==0:
            points=pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins)
        lik=[]
        pool=Pool()
        for ii in points:
            pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
        pool.close()
        pool.join()
        lik=nu.array(lik,dtype=nu.float128)
        lik=lik[lik[:,-1].argsort(),:]
        #find min and bootstrap (sort of)
        mu=nu.array([lik[:100,:-1].mean(0)])
        sigma=nu.array([nu.cov(lik[:100,:-1].T)])*2.
        i+=1
        print "number of trys is %i" %i
        if sum(lik[:,-1]<=11399)>3000:
            break
        #initalize mcmc lik params
    chibest=nu.ones(2)+nu.inf
    for i in range(50): #start refinement loop
        if i==0:
            if sum(lik[:,-1]<=11399)<30:
                lik[:,-1] =(1/lik[:,-1])
            else:
                lik[:,-1] =nu.exp(-lik[:,-1]/2.)
        else:
            if sum(lik[:,-1]<=11399)<30:
                 lik[:,-1] =(1/lik[:,-1])                  
            else:
                lik[:,-1] =nu.exp(-lik[:,-1]/2.)
            #q_sum=nu.sum(map(norm_func,lik[:,:-1],[[mu]]*len(lik),[[sigma]]*len(lik)),1)
            q_sum=nu.sum(map(student_t,lik[:,:-1],[[mu]]*len(lik),[[sigma]]*len(lik)),1)
            lik[:,-1]=lik[:,-1]/q_sum
        out[str(i)]=nu.copy(lik)
        #create best chi sample
        parambest=nu.zeros(bins*3)
        for j in range(bins*3):
            parambest[j]=nu.sum(lik[:,j]*lik[:,-1]/nu.sum(lik[:,-1]))
        chibest[1]=like_gen(data,parambest,lib_vals,age_unq,metal_unq,bins)[-1]
        weight_norm=lik[:,-1]/nu.sum(lik[:,-1])
        #print 'best estimate chi squared values is %f, num of dist %i' %(chibest[0],len(alpha))
        print 'Entropy is %f, and ESS is %f' %(nu.exp(-nu.nansum(weight_norm*nu.log10(weight_norm)))/pop_num,
                                               (nu.sum(weight_norm**2))**-1/pop_num)
        #resample and get new alpha
        if chibest[0]<chibest[1]: #if old was better fit than new look for clustering
            alpha,mu,sigma=resample_first(lik)
            #expand sigma so may find better fits
        else: #else keep iterating
            chibest[0]=nu.copy(chibest[1])
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

    return nu.vstack(out.values())

def PMC(data,bins,pop_num=10**4):
    #does PMC but minimizes Kullback-Leibler divergence to find best fit
    data_match_all(data)
    data[:,1]=data[:,1]*1000.
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
    #alpha=nu.array([5**-1.]*5)
    alpha=nu.array([1.])
    func_type=[student_t]
    '''for i in xrange(5):
        if nu.random.rand()>.5:
            func_type.append(norm_func)
        else:
           func_type.append(student_t) '''
    start_stuff=nn_ls_fit(data,max_bins=bins)
    mu,sigma=[],[]
    #sort out first guess
    temp=nu.zeros(bins*3)
    for i in range(bins):
        temp[i*bins*3]=nu.log10(start_stuff[0][i])
        temp[1+i*bins*3]=start_stuff[1][i]
        temp[2+i*bins*3]=start_stuff[2][i]
    mu.append(nu.copy(temp))
    #sigma.append(nu.identity(bins*3))
    #other guesses
    
    out={}
    for i in range(len(alpha)):
        sigma.append(nu.identity(bins*3))
        for k in xrange(bins*3):
            if any(nu.array(range(0,bins*3,3))==k):#metalicity
                temp[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
                sigma[-1][k,k]=nu.random.rand()*metal_unq.ptp()*.5
            else:#age and normilization
                if any(nu.array(range(1,bins*3,3))==k): #age
                    temp[k]=nu.random.rand()*age_unq.ptp()+age_unq[0]
                    sigma[-1][k,k]=nu.random.rand()*age_unq.ptp()/bins
                else: #norm
                    temp[k]=nu.random.random()*10**3
                    sigma[-1][k,k]=nu.random.rand()*10**3/4
        #mu.append(nu.copy(temp))
    mu,sigma=nu.array(mu),nu.array(sigma)
    #find best inital guess of mean and cov of prosterior
    print 'initalizing mixture'
    j,grad,lik,Sig=0,[],[],[]
    while j<8:
    #get likelihoods
        points=pop_builder(pop_num,alpha,mu,nu.copy(sigma),age_unq,metal_unq,bins)
        #lik=[]
        pool=Pool()
        for ii in points:
            pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
        pool.close()
        pool.join()
        new_lik=nu.copy(nu.array(lik,dtype=nu.float128))
        if sum(new_lik[:,-1]<=11399)<30:
            mu= nu.float64(new_lik[new_lik[:,-1].min()==new_lik[:,-1],:-1])
            continue
        else:
            new_lik[:,-1] =nu.exp(-new_lik[:,-1]/2.)
        
        #lik[:,-1]=lik[:,-1]/nu.sum(IFunc,1)
        grad.append(nu.float64(nu.hstack((mu[0],min_func(new_lik[:,-1])/new_lik.shape[0]))))
        Sig.append(nu.copy(sigma[0]))
        
        #find gradient using "chi squared disttance"
        '''if j>1:
            v=(grad[-1][:-1]-grad[-2][:-1])/(grad[-1][-1]-grad[-2][-1])
            if nu.sum(v)==0: #perteb it
                v=nu.array([nu.random.rand(len(mu[0]))-.5])
            mu=nu.array([grad[-1][:-1]-grad[-1][-1]*v])
            alpha,mu,sigma=resample(lik,alpha,mu,sigma,func_type,False)
            
            
        else:'''
        alpha,mu,sigma=resample(new_lik,alpha,mu,sigma,func_type)
        j+=1
        print "number of trys is %i" %j
        #if sum(lik[:,-1]<=11399)>3000:
       #     break
 

def min_func(Pi):
    #calculates chi squared function for minization
    Pi_norm=Pi/nu.sum(Pi)
    return 1/(nu.sum(Pi_norm**2))

def resample_first(lik):
    #uses xmeans clustering to adaptivly find starting  mixture densities
    weight=lik[:,-1]/nu.sum(lik[:,-1])
    for j in xrange(lik.shape[1]-1): #gen cdf for each param
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
        sigma[i]=nu.cov(clusters[clusters.keys()[i]].T)*2. #expansion factor
        alpha[i]=float(clusters[clusters.keys()[i]].shape[0])/lik.shape[0]

    return alpha,mu,sigma

def resample(lik,alpha,mu,sigma,func_type,option=True):
    #resamples points according to weights and makes new 
    #try this for pool.map map(norm_func,lik[:,:-1],[[mu]]*len(lik),[[sigma]]*len(lik))
    weight_norm=lik[:,-1]/nu.sum(lik[:,-1])
    rho=nu.array(map(func_type[0],lik[:,:-1],[[mu]]*len(lik),[[sigma]]*len(lik)))
    #rho=nu.array(map(func,lik[:,:-1],[[mu]]*len(lik),[[sigma]]*len(lik)))
    for i in xrange(rho.shape[1]):
        rho[:,i]=rho[:,i]/nu.sum(rho,1)
     #calculate alpha
        alpha[i]=nu.sum( weight_norm*rho[:,i])
    #calc mu and sigma
        if option:
            for j in xrange(mu.shape[1]):
                mu[i,j]=nu.sum(weight_norm*lik[:,j]*rho[:,i])/alpha[i]
    for k in xrange(mu.shape[0]):
        for i in xrange(mu.shape[1]):
            for j in xrange(mu.shape[1]):
                sigma[k][i,j]=nu.sum(weight_norm*rho[:,k]*(lik[:,i]-mu[k,i])*(lik[:,j]-mu[k,j]).T)/alpha[k]
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


def student_t(x,mu,sigma,n=1.,**kwargs):
    #calculates the proablility density of uniform multi dimensonal student
    #t dist with n degrees of freedom
    out=nu.zeros(mu[0].shape[0])
    for i in xrange(mu[0].shape[0]):
        #print sigma[0][i]
        try:
            out[i]=gamma(n+sigma[0][i].shape[0])/2./(gamma(n/2.)*(n*nu.pi)**(sigma[0][i].shape[0]/2.)*
                                                     nu.linalg.det(sigma[0][i])**(-.5)*
                                                     (1+1/float(n)*(nu.dot((x-mu[0][i]),nu.dot(nu.linalg.inv(sigma[0][i]),(x-mu[0][i]).T))))**((n+sigma[0][i].shape[0])/2.))
        except nu.linalg.LinAlgError:
            out[i]=gamma(n+sigma[0][i].shape[0])/2./(gamma(n/2.)*(n*nu.pi)**(sigma[0][i].shape[0]/2.)*
                                                     nu.linalg.det(sigma[0][i])**(-.5)*
                                                     (1+1/float(n)*(nu.dot((x-mu[0][i]),nu.dot((sigma[0][i])**-1,(x-mu[0][i]).T))))**((n+sigma[0][i].shape[0])/2.))
    return out
    
def norm_func(x,mu,sigma,**kwargs):
    #calculates values of normal dist for set of points
    out=nu.zeros(mu[0].shape[0])
    for i in xrange(mu[0].shape[0]):
        out[i]=(2*nu.pi)**(-len(mu[0][i])/2.)*nu.linalg.det(sigma[0][i])**(-.5)
        try:
            out[i]=out[i]*nu.exp(-.5*(nu.dot((x-mu[0][i]),nu.dot(nu.linalg.inv(sigma[0][i]),(x-mu[0][i]).T))))
        except nu.linalg.LinAlgError:
            #sigma[i][sigma[i]==0]=10**-6
            out[i]=out[i]*nu.exp(-.5*(nu.dot((x-mu[0][i]),nu.dot(sigma[0][i]**-1,(x-mu[0][i]).T))))
            
    return out

def like_gen(data,active_param,lib_vals,age_unq,metal_unq,bins):
   #calcs chi squared values
    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    #model=data_match_new(data,model,bins)
    index=xrange(2,bins*3,3)
    model['wave']= model['wave']*.0
    for ii in model.keys():
        if ii!='wave':
            model['wave']+=model[ii]*active_param[index[int(ii)]]

    #make weight paramer start closer to where ave data value
    return nu.hstack((active_param,nu.sum((data[:,1]-model['wave'])**2)))
 
def pop_builder(pop_num,alpha,mu,sig,age_unq,metal_unq,bins):
    #creates pop_num of points for evaluation
    #only uses a multivarate norm and unifor dist for now
                     
    #check if alpha sums to 1
    if nu.sum(alpha)!=1:
        alpha=alpha/nu.sum(alpha)
    #initalize params
    points=nu.zeros([pop_num,bins*3])
    #age_bins=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    #multivariate norm
    for j in xrange(mu.shape[0]):
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
        points[start:stop,:]=nu.random.multivariate_normal(mu[j],sig[j],(stop-start))
        if nu.sum(points[:,0]==0)>0:
            index=nu.nonzero(points[:,0]==0)[0]
            points[index,:]=nu.random.multivariate_normal(mu[j],sig[j],len(index))
        #check for values outside range
        #bin_index=0
        for i in range(bins*3):
            if any(i==nu.array(range(0,bins*3,3))): #metalicity
                index=nu.nonzero(nu.logical_or(points[start:stop,i]< metal_unq[0],points[start:stop,i]> metal_unq[-1]))[0]
                while index.shape[0]>0:
                    index+=start
                    points[index,:]=nu.random.multivariate_normal(mu[j],sig[j],len(index))
                    index=nu.nonzero(nu.logical_or(points[start:stop,i]< metal_unq[0],points[start:stop,i]> metal_unq[-1]))[0]  
                    sig[j][i,i]=sig[j][i,i]/2.
            elif (i-1)%3==0 or i-1==0:#age
                index=nu.nonzero(nu.logical_or(points[start:stop,i]< age_unq[0],points[start:stop,i]>age_unq[-1]))[0]
                while index.shape[0]>0:
                    index+=start
                    points[index,:]=nu.random.multivariate_normal(mu[j],sig[j],len(index))
                    index=nu.nonzero(nu.logical_or(points[start:stop,i]< age_unq[0],points[start:stop,i]> age_unq[-1]))[0] 
                    sig[j][i,i]=sig[j][i,i]/2.
            elif (i-2)%3==0 or i==2: #norm
                if nu.any(points[:,i]<0):
                    points[:,i]=nu.abs(points[:,i])
        #one last check to see if all points are in param range
    for i in range(bins*3):
        if any(i==nu.array(range(0,bins*3,3))): #metal
            index=nu.nonzero(points[:,i]<metal_unq[0])[0] #lower range
            points[index,i]=nu.copy(metal_unq[0])
            index=nu.nonzero(points[:,i]>metal_unq[-1])[0] #upper range
            points[index,i]=nu.copy(metal_unq[-1])
        elif any(i==nu.array(range(1,bins*3,3))): #age
            index=nu.nonzero(points[:,i]<age_unq[0])[0] #lower range
            points[index,i]=nu.copy(age_unq[0])
            index=nu.nonzero(points[:,i]>age_unq[-1])[0] #upper range
            points[index,i]=nu.copy(age_unq[-1])

    return points


def toy():

    pass

if __name__=='__main__':
    test()
