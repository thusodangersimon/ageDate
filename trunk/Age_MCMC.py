#!/usr/bin/env python

import numpy as nu
from Age_date import *
from mpi4py import MPI
from multiprocessing import *
import time as Time

a=nu.seterr(all='ignore')

def test():
    comm = MPI.COMM_WORLD
#size = comm.Get_size()
    rank = comm.Get_rank()
    print 'i am %i' %rank

def MCMC_multi(data,itter,bins,cpus=cpu_count()):
    #more effecent version of multi core MCMC
    #uses cominication methods instead of creating and distroying processes

    #shared arrays (chibest, parambest,i)
    chibest=Value('f', nu.inf)

    i=Value('i', 0)

    parambest=Array('d',nu.zeros([3*bins]))

    option=Value('b',True)
    option.burnin=10**3
    option.itter=int(itter+option.burnin)


    #sig_share=Array('d',nu.zeros([3*bins]))
    work=[]
    q=Queue()
    #start multiprocess mcmc
    for ii in range(cpus):
        work.append(Process(target=MCMC_vanila,args=(data,bins,i,chibest
                                                     ,parambest,option,q)))
        work[-1].start()
    while i.value<itter:
        print '%2.2f percent done' %(i.value/float(itter)*100)
        sys.stdout.flush()
        #print i.value
        Time.sleep(5)
    option.value=False
    #wait for proceses to finish
    count=0
    temp=[]
    while count<cpus:
        if q.qsize()>0:
           temp.append(q.get())
           count+=1
    #post processing
    count=0
    outsigma={}
    #outrate,outparam,outchi={},{},{}

    outparam,outchi=nu.zeros([2,3*bins]),nu.array([nu.inf])
    for ii in temp:
        outparam=nu.concatenate((outparam,ii[0][~nu.isinf(ii[1]),:]
                                 ),axis=0)
        outchi=nu.concatenate((outchi,ii[1][~nu.isinf(ii[1])]))
        #outsigma[str(count)]=ii[2]
        '''outparam[str(count)]=ii[0][nu.nonzero(ii[0][:,0]>0)[0],:]
        outchi[str(count)]=ii[1][~nu.isinf(ii[1])]
        outsigma[str(count)]=nu.array(ii[2])
        outrate[str(count)]=nu.array(ii[3])
        '''
        count+=1

    return outparam[2:,:],outchi[1:]
    #return outparam,outchi,outsigma,outrate
    
def MCMC_vanila(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC parameter estimation with a floating step size till 10k iterations
    #itter needs to be a array of normaly distrbuted numbers
    #so there are no problems with multiprocessing

    #part on every modual wanting to fit the spectra
    #controls input and expot of files for fitt
      
    
    #change random seed for random numbers for multiprocessing
    nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    param=nu.zeros([option.itter,len(parambest)])
    active_param=nu.zeros(len(parambest))
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0

    #start in random place
    for k in xrange(len(parambest)):
        if k %2==0 and len(parambest)-bins-1>k:#metalicity
            active_param[k]=10**(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if len(parambest)-bins-1<k: #normilization
                active_param[k]=10*nu.random.random()
            else: #age
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
    #active_param[0]=0.020
    #active_param[2]=1

    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter)+nu.inf
    sigma=nu.identity(len(active_param))*nu.concatenate((nu.tile([.9,1.0],bins),
                          nu.array([nu.sqrt(bins)]*bins)))

    model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
    model=data_match(model,data)
    #make weight paramer start closer to where ave data value
    for j in range(bins):
        active_param[2+j]=normalize(data,model)*nu.random.random()
    chi[0]=sum((data[:,1]-model)**2)
    
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject=1.0,1.0
    acept_rate,out_sigma=[],[]
    j=1
    while option.value and i.value<option.itter:
        #for k in xrange(len(active_param)):
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
      #calculate new model and chi
        model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
        model=data_match(model,data)
        #active_param[2]=normalize(data,model)
        chi[j]=sum((data[:,1]-model)**2)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2.0)
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if chi[j]< chibest.value:
                print 'best fit value %f' %chi[j]
                sys.stdout.flush()
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
        else:
            if a>nu.random.rand():#false accept
                param[j,:]=nu.copy(active_param)
                Nacept+=1
            else:
                param[j,:]=nu.copy( param[j-1,:])
                active_param=nu.copy( param[j-1,:])
                chi[j]=nu.copy(chi[j-1])
                Nreject+=1
 
        if j<100: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if Nacept/Nreject<.23 and all(sigma.diagonal()>=10**-6): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif Nacept/Nreject>.25 and all(sigma.diagonal()<10): #not enough
                sigma=sigma*1.05
        else: #use covarnence matrix
            sigma=Covarence_mat(param,j)

        j+=1
        i.value=i.value+1
        acept_rate.append(nu.copy([Nacept,Nreject]))
        out_sigma.append(nu.copy(sigma))
    #return once finished 
    param=outprep(param)
    q.put((param[option.burnin:,:],chi[option.burnin:] ))
    #q.put((param,chi,out_sigma,acept_rate))


def MCMC_SA(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC and reduices the false acceptance rate over a threshold
      
    
    #change random seed for random numbers for multiprocessing
    nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    param=nu.zeros([option.itter,len(parambest)])
    active_param=nu.zeros(len(parambest))
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0

    #start in random place
    for k in xrange(len(parambest)):
        if k %2==0 and len(parambest)-bins-1>k:#metalicity
            active_param[k]=10**(nu.random.random()*nu.log10(metal_unq).ptp()+nu.log10(metal_unq)[0])
        else:#age and normilization
            if len(parambest)-bins-1<k: #normilization
                active_param[k]=10*nu.random.random()
            else: #age
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
    #active_param[0]=0.020
    #active_param[2]=1

    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter)+nu.inf
    sigma=nu.concatenate((nu.tile([.9,1.0],bins),
                          nu.array([nu.sqrt(bins)]*bins)))

    model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
    model=data_match(model,data)
    #active_param[2]=normalize(data,model)
    chi[0]=sum((data[:,1]-model)**2)
    
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject=1.0,1.0
    acept_rate,out_sigma=[],[]
    j=1
    while option.value and i.value<option.itter:
       #for k in xrange(len(active_param)):
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
        #active_param[2]=1
       #calculate new model and chi
        model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
        model=data_match(model,data)
        #active_param[2]=normalize(data,model)
        chi[j]=sum((data[:,1]-model)**2)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/SA(i.value,acept_rate))
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if chi[j]< chibest.value:
                print 'best fit value %f' %chi[j]
                sys.stdout.flush()
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
        else:
            if a>nu.random.rand():#false accept
                acept_rate.append([a,j])
                param[j,:]=nu.copy(active_param)
                Nacept+=1
            else:
                param[j,:]=nu.copy( param[j-1,:])
                active_param=nu.copy( param[j-1,:])
                chi[j]=nu.copy(chi[j-1])
                Nreject+=1
 
        if j<100: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if Nacept/Nreject<.23 and all(sigma.diagonal()>=10**-6): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif Nacept/Nreject>.25 and all(sigma.diagonal()<10): #not enough
                sigma=sigma*1.05
        else: #use covarnence matrix
            sigma=Covarence_mat(param,j)
        j+=1
        i.value=i.value+1
        acept_rate.append(Nacept/Nreject)
        #out_sigma.append(nu.copy(sigma))
    #return once finished 
    param=outprep(param)
    q.put((param[option.burnin:,:],chi[option.burnin:]))
    #q.put((param,chi,out_sigma,acept_rate))


def SA(i,rate):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<50% as a function on acceptance rate
    lamdbaa=1.
    if nu.mean(rate>.5):
        N=1.5
    else:
        N=1.
    return (1/(1+lamdbaa*(i+1)))**N

def Covarence_mat(param,j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j-1000<0:
        return nu.cov(param[:j,:].T)
    else:
        return nu.cov(param[j-1000:j,:].T)

def outprep(param):
    #changes metals from log to normal
    for i in range(0,param.shape[1],3):
        param[:,i]=10**param[:,i]
    return param



if __name__=='__main__':

    def Stuff(data_og,i,age,lib_vals,age_unq,metal_unq):
        data=data_og*1.
        chi=age*0
        data[:,1]=data_og[:,1]*i
        print i
        for j in age:
            model=get_model_fit([metal_unq[0],j,1],lib_vals,age_unq,metal_unq,1)
            chi[j]=sum((data[:,1]- normalize(data,model[:,1])*model[:,1])**2)
        return i,chi.ptp()
 
    po=Pool()
    lib_vals=get_fitting_info()
    metal_unq=nu.unique(lib_vals[0][:,0])
    age_unq=nu.unique(lib_vals[0][:,1])
    age=nu.linspace(age_unq[0],age_unq[-1],1000)
    
    data_og=get_model_fit([metal_unq[0],age_unq[5],1],lib_vals,age_unq,metal_unq,1)
    data=data_og*1.
    chi=age*0
    max_range=[]
    [po.apply_async(Stuff,(data_og,i,age,lib_vals,age_unq,metal_unq,),callback=max_range.append) for i in xrange(500)]
    po.close()
    po.join()
    import cPickle as pik
    pik.dump(max_range,open('max_range.pik','w'),2)

     #for i in xrange(500*500):
      #  max_range.append(Stuff(data_og,i,age,lib_vals,age_unq,metal_unq))
