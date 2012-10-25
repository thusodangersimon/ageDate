#!/usr/bin/env python
#
# Name:  Age multiy-try metropolis with RJMCMC
#
# Author: Thuso S Simon
#
# Date: 29th of June, 2012
#TODO:  
#    
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
# test
#
# 

import numpy as nu
import mpi4py.MPI as mpi
import Age_date as ag
import os
import pylab as lab
import cPickle as pik

#123456789012345678901234567890123456789012345678901234567890123456789
def worker_chi_cal(fun, N_try, root=0):
    #does chi squared calc for root
    comm = mpi.COMM_WORLD
    rank =  comm.Get_rank()
    size = comm.Get_size()
    #get unique seed
    seed =os.getpid()
    if seed % 10 == 0:
        seed += nu.random.randint(1,10)
    seed = ag.random_permute(os.getpid())
    nu.random.seed(seed)
    Quit = nu.array([0])
    print 'Starting node on %s with rank %i, with seed %i, pid %i' %(mpi.Get_processor_name(),rank,seed, os.getpid())
    #calculate incoming size of data
    recv_size = int(nu.ceil(N_try/size))
    bins = nu.array([1])
    none = nu.array([None])
    #i = 0
    #init params that don't change shape
    if fun._dust:
        active_dust = nu.zeros(2)
        sigma_dust = nu.zeros((2,2))
    else:
        active_dust = None
    if fun._losvd:
        active_losvd = nu.zeros(4)
        sigma_dust = nu.zeros((4,4))
    else:
        active_losvd = None
    while True:
        #get number of bins
        
        #if i == 0:
        comm.Bcast(bins, root)
            #get current position
        active_param = nu.zeros(bins * 3)
        sigma = nu.zeros((bins*3, bins*3))
        #recive param
        temp = comm.bcast(none, root =root)[0]
        for ii in temp.keys():
            exec(ii + '= temp["%s"]'%ii)
        #generate trial params
        send_param = nu.array(map(fun.proposal,[active_param]*N_try, [sigma]*N_try))
        if fun._dust:
            send_dust = nu.array(map(fun.proposal,[active_dust]*N_try, [sigma_dust]*N_try))
        else:
            send_dust = [None] * N_try
        if fun._losvd:
            send_losvd = nu.array(map(fun.proposal,[active_losvd]*N_try, [sigma_losvd]*N_try))
        else:
            send_losvd = [None] * N_try
        #chi calc
        recv= map(fun.lik, send_param,send_dust,send_losvd)
        recv_chi,recv_N = zip(*recv)        
        recv_chi = nu.array(recv_chi)
        send_param[:, range(2,bins*3,3)] = nu.array(recv_N)
        
        #remove non usefull chi values
        index = nu.isfinite(recv_chi)
        ratio = nu.array(index.sum() / float(N_try)/(size - 1))
        recv_chi = recv_chi[index]
        send_param = send_param[index]
        if fun._dust:
            send_dust = send_dust[index]
        if fun._losvd:
            send_losvd = send_losvd[index]
        
        #return chi
        start2 = mpi.Wtime()
        comm.Reduce(ratio, none, op=mpi.SUM, root=root)
        temp_len = nu.array([index.sum(),rank])
        comm.Gather(temp_len, none, root=root)
        comm.Gatherv(recv_chi, none, root = root)
        comm.Gatherv(send_param, none, root = root)
        if fun._dust:
            comm.Gatherv(send_dust, none, root = root)
        if fun._losvd:
            comm.Gatherv(send_losvd, none, root = root)
        #print rank,mpi.Wtime() - start2
        #check if need to quit
        comm.Barrier()
        #i += 1
        #if i == 2:
        comm.Bcast(Quit, root = root)
            #i == 0
        if Quit:
            break
    

def root_mc(fun, N_try, burnin=500 ,itter=500, k_max=2):
    #handels root stuff
    print 'Starting head node on %s with rank %i' %(mpi.Get_processor_name(),mpi.COMM_WORLD.Get_rank())
    ####initalize vars and workers
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    #make sure N_try is evenly devideable by all processors
    N_try = int(nu.ceil(N_try/(size+.0)) * size)
    recv_size = int(nu.ceil(N_try/size))
    lib_vals = fun._lib_vals
    metal_unq = fun._metal_unq
    age_unq = fun._age_unq
    none = nu.array([None])
    #create fun for all number of bins
    attempt=False
    param,active_param,chi,sigma={},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    #set up dust if in use
    if fun._dust:
        active_dust = nu.random.rand(2)*4.
        sigma_dust = nu.identity(2)*nu.random.rand()*2
    else:
        active_dust = nu.zeros(2)
        sigma_dust = nu.zeros([2,2])
    #set up LOSVD
    if fun._losvd:
        '''#only gaussian dispersion with no shift for now
        active_losvd = nu.array([nu.random.rand()*150,0,0,0])
        sigma_losvd = nu.zeros([4,4])
        sigma_losvd[0,0] = nu.random.rand() * 10'''
        active_losvd = nu.random.rand(4)*150
        sigma_losvd = nu.random.rand(4,4)
    else:
        active_losvd = nu.zeros(4)
        sigma_losvd = nu.zeros([4,4])
 
    bayes_fact={} #to calculate bayes factor
    #fun=MC_func(data)
    for i in range(1,k_max+1):
        param[str(i)]=[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,age_unq.ptp()*nu.random.rand(),1.],i)
        #active_dust[str(i)]=nu.random.rand(2)*5.
        #sigma_dust[str(i)]=nu.identity(2)*nu.random.rand()*2
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
        bayes_fact[str(i)]=[]
    #bins to start with
    bins = nu.random.randint(1,k_max)
    while True:
    #create starting active params
        bin=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))
        bin_index=0
    #start in random place
        for k in xrange(3*bins):
            if any(nu.array(range(0,bins*3,3))==k):#metalicity
                active_param[str(bins)][k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
            else:#age and normilization
                if any(nu.array(range(1,bins*3,3))==k): #age
                    active_param[str(bins)][k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
                    bin_index+=1
                else: #norm
                    active_param[str(bins)][k]=nu.random.random()*10000
    #try leastquares fit
        if len(chi[str(bins)])==1:
            chi[str(bins)].append(0.)
    #active_param[str(bins)]=fun[str(bins)].n_neg_lest(active_param[str(bins)])
        (chi[str(bins)][-1],
         active_param[str(bins)][range(2,bins*3,3)]) = fun.lik(
            active_param[str(bins)], active_dust, active_losvd)
    #check if starting off in bad place ie chi=inf or nan
        if not nu.isfinite(chi[str(bins)][-1]):
            continue
        else:
            break

    param[str(bins)].append(nu.copy(nu.hstack((
                    active_param[str(bins)], active_dust, active_losvd))))
    #set best chi and param
    chibest = chi[str(bins)][-1]+.0
    parambest = nu.hstack((active_param[str(bins)],
                           active_dust,active_losvd))
    #start multi-try
    #simulated anneling vars
    #T_cuurent,Nexchange_ratio=1.0,1.0
    T_start,T_stop = chibest,0.
    #save dust and losvd sigmas
    out_dust_sig, out_losvd_sig = [sigma_dust], [sigma_losvd]
    t = [ag.Time.time()]
    recv_chi, recv_N = nu.zeros(N_try), nu.zeros((N_try,bins))
    step_ratio = 0.
    fig = lab.figure()
    plt = fig.add_subplot(111)
    for i in xrange(itter):
        #if i % 50 == 0:
        print '%1.2f precent done, at %1.2f acceptance' %(i/float(itter)*100, step_ratio*100)
        #send number of bins
        comm.Bcast(nu.array([bins]), root = rank)
        #send active param
        send_param ={'active_param':active_param[str(bins)], 'sigma': sigma[str(bins)]}
        if fun._dust:
            send_param['active_dust'] = active_dust
            send_param['sigma_dust'] = sigma_dust
        if fun._losvd:
            send_param['active_losvd'] = active_losvd
            send_param['sigma_losvd'] = sigma_losvd
        #calculate chi
        #mpi cal chi
        comm.bcast((send_param,mpi.DOUBLE), root = rank)
        #recv from nodes
        ratio = nu.zeros(1)
        temp_len = nu.zeros((size,2),dtype=int)
        index = nu.array([0,rank])
        comm.Reduce(nu.array(ratio/(size-1)), ratio, op=mpi.SUM, root=rank)
        comm.Gather(index, temp_len, root=rank)
        #make correct length arrays for recving
        recv_chi, send_param = nu.zeros(temp_len[:,0].sum()), nu.zeros((temp_len[:,0].sum(),bins*3))
        temp_len[2:,1] =temp_len[1:-1,0]
        temp_len[1,1]=0
        comm.Gatherv(temp_len, (recv_chi, tuple(temp_len[:,0]), tuple(temp_len[:,1])
                                , mpi.DOUBLE), root = rank)
        comm.Gatherv(send_param, (send_param, tuple(temp_len[:,0]*3),  
                                  tuple(temp_len[:,1]*3), mpi.DOUBLE), root = rank)
        if fun._dust:
            send_dust = nu.zeros((temp_len[:,0].sum(), 2))
            comm.Gatherv( send_dust , (send_dust,tuple(temp_len[:,0]*2), 
                                       tuple(temp_len[:,1]*2), mpi.DOUBLE), root = rank)
        if fun._losvd:
            send_losvd =  nu.zeros((temp_len[:,0].sum(), 4))
            comm.Gatherv( send_dust , (send_losvd,tuple(temp_len[:,0]*4), 
                                       tuple(temp_len[:,1]*4), mpi.DOUBLE), root = rank)  
        comm.Barrier()
        #print 'recived files from workers'
        ##select param with best chi square with weighted prob            
        weight = nu.exp(-recv_chi/recv_chi.min())
        index = weight.argsort()
        recv_chi, weight, send_param = recv_chi[index], weight[index], send_param[index]
        index = nu.searchsorted(nu.cumsum(weight)/weight.sum(),nu.random.rand())
        #index = recv_chi.argmin()
        active_param[str(bins)]  = nu.copy(send_param[index])
        #print '%2.2f chosen, best is %2.2f' %(recv_chi[index],recv_chi.min())
        chi[str(bins)].append(nu.copy(recv_chi[index]))
        if fun._dust:
            active_dust = nu.copy(send_dust[index])
        if fun._losvd:
            active_losvd = nu.copy(send_losvd[index])
        #generate new params for new and old test
        #print ratio
        plt.plot(send_param[:,0],send_param[:,1],'b.')
        plt.plot(active_param[str(bins)][0], active_param[str(bins)][1],'r+')
        plt.set_xlim((metal_unq[0],metal_unq[-1]))
        plt.set_ylim((age_unq[0],age_unq[-1]))
        fig.savefig('mlt%i.png'%i)
        plt.clear()
        pik.dump((send_param,recv_chi),open('mlt%i.pik'%i,'w'),2)
        #acceptance criteria
        '''prob_old = nu.exp(-recv_chi[:]/(recv_chi[nu.isfinite(recv_chi)]).max())
        prob_old = prob_old[0]/prob_old.sum()
        prob_new = weight[index]/weight.sum()'''
        a = (nu.exp(-(chi[str(bins)][-1] - chi[str(bins)][-2])/
                     SA(i,burnin,T_start,T_stop)) ) #* prob_old/prob_new)
        if a > nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(nu.hstack((active_param[str(bins)]
                                                       , active_dust,
                                                       active_losvd))))
            Nacept[str(bins)] += 1
            #see if global best fit
            if chibest > chi[str(bins)][-1]:
                chibest = nu.copy(chi[str(bins)][-1])
                parambest = nu.copy(param[str(bins)][-1])
                T_start = nu.round(min(chi[str(bins)]))+1.
                print 'New best fit. With chi=%f and %i bins from %s' %(chibest,bins,mpi.Get_processor_name())
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)] = nu.copy(param[str(bins)][-1][range(3*bins)])
            if fun._dust:
                active_dust = nu.copy(param[str(bins)][-1][-6:-4])
            if fun._losvd:
                active_losvd = nu.copy(param[str(bins)][-1][-4:])
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1
        #tune step size
        step_ratio = Nacept[str(bins)]/float(Nacept[str(bins)] + Nreject[str(bins)])
        if step_ratio > .5:
            if not nu.any(sigma[str(bins)][range(0,bins*3,3) +range(1,bins*3,3)].diagonal() >
                          [metal_unq.ptp(),age_unq.ptp()] * bins):
            #too few aceptnce decrease sigma
                sigma[str(bins)] *= 1.05
                if fun._dust:
                    sigma_dust *=1.05
                if fun._losvd: 
                    sigma_losvd *= 1.05
        elif step_ratio < .23:
                sigma[str(bins)] /= 1.05
                #dust step
                if fun._dust:
                    sigma_dust /= 1.05
                #losvd step
                if fun._losvd:
                    sigma_losvd /= 1.05
                
        #compute cov as step size
        if i%100==0 and i != 0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
            t_param = nu.array(param[str(bins)][-2000:])
            try:
                sigma[str(bins)]=Covarence_mat(t_param[:,range(3*bins)],
                                           t_param.shape[0]-1)
                if fun._dust:
                    sigma_dust = Covarence_mat(t_param[:,-6:-4],t_param.shape[0]-1)
                if fun._losvd:
                    sigma_losvd = Covarence_mat(t_param[:,-4:],t_param.shape[0]-1)
            except IndexError:
                print t_param.shape
            #error handeling some time cov is nan
            if not (nu.any(nu.isfinite(sigma[str(bins)])) or 
                    nu.any(nu.isfinite(sigma_dust)) or 
                    nu.any(nu.isfinite(sigma_losvd))):
                #set equal to last cov matirx
                sigma[str(bins)] = nu.copy(out_sigma[str(bins)][-1])
                if fun._dust:
                    sigma_dust = nu.copy(out_sig_dust[-1])
                if fun._losvd:
                    sigma_losvd = nu.copy(out_losvd_sig[-1])
        
        #store params and chi values
        t[-1] -= ag.Time.time()
        t.append(ag.Time.time())
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))
        comm.Bcast(nu.array([0]), root=rank)


    


def worker_burnin(fun):
    comm = mpi.COMM_WORLD
    worker_size = mpi.COMM_WORLD
    recv_param = None
    dummy = None
    while True:
        comm.Scatterv(dummy,recv_param,0)

def worker_other(fun):
    pass


def mult_single(fun, N_try, itter=10**5, burnin=5*10**3,  k_max=16):
    '''test of how multi-try mcmc will work on a single processor'''
    lib_vals = fun._lib_vals
    metal_unq = fun._metal_unq
    age_unq = fun._age_unq
    #create fun for all number of bins
    attempt=False
    param,active_param,chi,sigma={},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    #set up dust if in use
    if fun._dust:
        active_dust = nu.random.rand(2)*4.
        sigma_dust = nu.identity(2)*nu.random.rand()*2
    else:
        active_dust = nu.zeros(2)
        sigma_dust = nu.zeros([2,2])
    #set up LOSVD
    if fun._losvd:
        '''#only gaussian dispersion with no shift for now
        active_losvd = nu.array([nu.random.rand()*150,0,0,0])
        sigma_losvd = nu.zeros([4,4])
        sigma_losvd[0,0] = nu.random.rand() * 10'''
        active_losvd = nu.random.rand(4)*150
        sigma_losvd = nu.random.rand(4,4)
    else:
        active_losvd = nu.zeros(4)
        sigma_losvd = nu.zeros([4,4])
 
    bayes_fact={} #to calculate bayes factor
    #fun=MC_func(data)
    for i in range(1,k_max+1):
        param[str(i)]=[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,age_unq.ptp()*nu.random.rand(),1.],i)
        #active_dust[str(i)]=nu.random.rand(2)*5.
        #sigma_dust[str(i)]=nu.identity(2)*nu.random.rand()*2
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
        bayes_fact[str(i)]=[]
    #bins to start with
    bins=nu.random.randint(1,k_max)
    while True:
    #create starting active params
        bin=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))
        bin_index=0
    #start in random place
        for k in xrange(3*bins):
            if any(nu.array(range(0,bins*3,3))==k):#metalicity
                active_param[str(bins)][k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
            else:#age and normilization
                if any(nu.array(range(1,bins*3,3))==k): #age
                    active_param[str(bins)][k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
                    bin_index+=1
                else: #norm
                    active_param[str(bins)][k]=nu.random.random()*10000
    #try leastquares fit
        if len(chi[str(bins)])==1:
            chi[str(bins)].append(0.)
    #active_param[str(bins)]=fun[str(bins)].n_neg_lest(active_param[str(bins)])
        (chi[str(bins)][-1],
         active_param[str(bins)][range(2,bins*3,3)]) = fun.lik(
            active_param[str(bins)], active_dust, active_losvd)
    #check if starting off in bad place ie chi=inf or nan
        if not nu.isfinite(chi[str(bins)][-1]):
            continue
        else:
            break

    param[str(bins)].append(nu.copy(nu.hstack((
                    active_param[str(bins)], active_dust, active_losvd))))
    #set best chi and param
    chibest = chi[str(bins)][-1]+.0
    parambest = nu.hstack((active_param[str(bins)],
                           active_dust,active_losvd))
    #start multi-try
    #number of multi-trys
    
    #simulated anneling vars
    #T_cuurent,Nexchange_ratio=1.0,1.0
    T_start,T_stop = chibest,0.
    #save dust and losvd sigmas
    out_dust_sig, out_losvd_sig = [sigma_dust], [sigma_losvd]
    t = [ag.Time.time()]
    for i in xrange(itter):
        if i % 50 == 0:
            print '%1.2f precent done, at %1.2f seconds per iteration' %(i/float(itter)*100, nu.median(t))
        #generate params
        send_param = nu.zeros((N_try, bins*3))
        send_dust = nu.zeros((N_try, 2))
        send_losvd = nu.zeros((N_try, 4))
        for ii in xrange(N_try):
            send_param[ii, :] = fun.proposal(active_param[str(bins)],
                                             sigma[str(bins)])
            if fun._dust:
                send_dust[ii,:]= fun.proposal(active_dust,sigma_dust)
            if fun._losvd:
                send_losvd[ii,:]  = fun.proposal(active_losvd, sigma_losvd)

        #calculate chi
        recv= map(fun.lik, send_param,send_dust,send_losvd)
        recv_chi, recv_N = nu.zeros(N_try), nu.zeros((N_try,bins))
        for ii in range(N_try):
            recv_chi[ii], recv_N[ii,:] = recv[ii]
        ##select param with best chi square with weighted prob
        index = nu.isfinite(recv_chi)
        send_param[:, range(2, bins*3,3)] = recv_N
        recv_chi = recv_chi[index]
        send_param = send_param[index]
        if fun._dust:
            send_dust = send_dust[index]
        if fun._losvd:
            send_losvd = send_losvd[index]
        weight = nu.exp(-recv_chi/recv_chi.min())
        index = weight.argsort()
        recv_chi, weight, send_param = recv_chi[index], weight[index], send_param[index]
        index = nu.searchsorted(nu.cumsum(weight)/weight.sum(),nu.random.rand())
        #index = recv_chi.argmin()
        active_param[str(bins)]  = nu.copy(send_param[index])
        #print '%2.2f chosen, best is %2.2f' %(recv_chi[index],recv_chi.min())
        chi[str(bins)].append(nu.copy(recv_chi[index]))
        if fun._dust:
            active_dust = nu.copy(send_dust[index])
        if fun._losvd:
            active_losvd = nu.copy(send_losvd[index])
        #generate new params for new and old test
        send_param = nu.zeros((N_try, bins*3))
        if fun._dust:
            send_dust = nu.zeros((N_try, 2))
        if fun._losvd:
            send_losvd = nu.zeros((N_try, 4))
        for ii in xrange(N_try):
            #old param
            #if ii < N_try:
            if ii == 0:
                send_param[ii, :] = nu.copy(param[str(bins)][-1][range(bins*3)])
                if fun._dust:
                    send_dust[ii,:] = nu.copy(param[str(bins)][-1][-6:-4])
                if fun._losvd:
                    send_losvd[ii,:] = nu.copy(param[str(bins)][-1][-4:])
                  
            else:
                send_param[ii, :] = fun.proposal(active_param[str(bins)],
                                                 sigma[str(bins)])
                if fun._dust:
                    send_dust[ii,:]= fun.proposal(active_dust, sigma_dust)
                if fun._losvd:
                    send_losvd[ii,:]  = fun.proposal(active_losvd, 
                                                         sigma_losvd)
            '''#new parms
            else:
                send_param[ii, :] = fun.proposal(param[str(bins)][-1][range(3*bins)],
                                                 sigma[str(bins)])
                if fun._dust:
                    send_dust[ii,:]= fun.proposal(param[str(bins)][-1][-6:-4],sigma_dust)
                if fun._losvd:
                        send_losvd[ii,:]  = fun.proposal(param[str(bins)][-1][-4:], sigma_losvd)'''
 
        #calc chi
        recv= map(fun.lik, send_param,send_dust,send_losvd)
        recv_chi, recv_N = nu.zeros(N_try), nu.zeros((N_try,bins))
        for ii in range(N_try):
            recv_chi[ii], recv_N[ii,:] = recv[ii]
        
        #acceptance criteria
        '''prob_old = nu.exp(-recv_chi[:]/(recv_chi[nu.isfinite(recv_chi)]).max())
        prob_old = prob_old[0]/prob_old.sum()
        prob_new = weight[index]/weight.sum()'''
        a = (nu.exp(-(chi[str(bins)][-1] - chi[str(bins)][-2])/
                     SA(i,burnin,T_start,T_stop)) ) #* prob_old/prob_new)
        if a > nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(nu.hstack((active_param[str(bins)]
                                                       , active_dust,
                                                       active_losvd))))
            Nacept[str(bins)] += 1
            #see if global best fit
            if chibest > chi[str(bins)][-1]:
                chibest = nu.copy(chi[str(bins)][-1])
                parambest = nu.copy(param[str(bins)][-1])
                T_start = nu.round(min(chi[str(bins)]))+1.
                print 'New best fit. With chi=%f and %i bins' %(chibest,bins)
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)] = nu.copy(param[str(bins)][-1][range(3*bins)])
            if fun._dust:
                active_dust = nu.copy(param[str(bins)][-1][-6:-4])
            if fun._losvd:
                active_losvd = nu.copy(param[str(bins)][-1][-4:])
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1
        #tune step size
        step_ratio = Nacept[str(bins)]/float(Nacept[str(bins)] + Nreject[str(bins)])
        if step_ratio > .5:
            if not nu.any(sigma[str(bins)][range(0,bins*3,3) +range(1,bins*3,3)].diagonal() >
                          [metal_unq.ptp(),age_unq.ptp()] * bins):
            #too few aceptnce decrease sigma
                sigma[str(bins)] *= 1.05
                if fun._dust:
                    sigma_dust *=1.05
                if fun._losvd: 
                    sigma_losvd *= 1.05
        elif step_ratio < .23:
                sigma[str(bins)] /= 1.05
                #dust step
                if fun._dust:
                    sigma_dust /= 1.05
                #losvd step
                if fun._losvd:
                    sigma_losvd /= 1.05
                
        #compute cov as step size
        if i%100==0 and i != 0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
            t_param = nu.array(param[str(bins)][-2000:])
            try:
                sigma[str(bins)]=Covarence_mat(t_param[:,range(3*bins)],
                                           t_param.shape[0]-1)
                if fun._dust:
                    sigma_dust = Covarence_mat(t_param[:,-6:-4],t_param.shape[0]-1)
                if fun._losvd:
                    sigma_losvd = Covarence_mat(t_param[:,-4:],t_param.shape[0]-1)
            except IndexError:
                print t_param.shape
            #error handeling some time cov is nan
            if not (nu.any(nu.isfinite(sigma[str(bins)])) or 
                    nu.any(nu.isfinite(sigma_dust)) or 
                    nu.any(nu.isfinite(sigma_losvd))):
                #set equal to last cov matirx
                sigma[str(bins)] = nu.copy(out_sigma[str(bins)][-1])
                if fun._dust:
                    sigma_dust = nu.copy(out_sig_dust[-1])
                if fun._losvd:
                    sigma_losvd = nu.copy(out_losvd_sig[-1])
        
        #store params and chi values
        t[-1] -= ag.Time.time()
        t.append(ag.Time.time())
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))

    return param,chi,out_sigma,Nacept,Nreject

def Covarence_mat(param,j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j-2000<0:
        cov = nu.cov(param[:j,:].T)
        if nu.any(nu.isnan(cov)):
            return False
        else:
            return cov
    else:
        cov = nu.cov(param[j-5000:j,:].T)
        if nu.any(nu.isnan(cov)):
            return False
        else:
            return cov

def SA(i,i_fin,T_start,T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    if i>i_fin:
        return 1.0
    else:
        return (T_stop-T_start)/float(i_fin)*i+T_start


if __name__ == '__main__':
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #send data to other processors
    N_try = 1000
    if rank == 0:
        data,info1,dust,weight = ag.iterp_spec(1,lam_min=2000,lam_max=10000)
        print info1
        dat_len = nu.array(data.shape[0])
        comm.Bcast((dat_len, mpi.INT), root=0)
        comm.Bcast(data, root=0)
        
    else:
        dat_len = nu.array([0])
        comm.Bcast( (dat_len, mpi.INT), root=0)
        data = nu.zeros((dat_len, 2))
        comm.Bcast(data, root=0)


    '''if rank == 0:
        test1 = nu.random.rand(3,3)
        #print test1.reshape(5,2)
        test2 = nu.zeros((9,3))
        comm.Gatherv(test1,(test2, (9,9,9), (0,9,18),mpi.DOUBLE),0)
        print test2,test1.ravel().shape #test2.reshape(15,2)
    else:
        test1 = nu.random.rand(3,3)
        comm.Gatherv(test1,test1,0)
    start = mpi.Wtime()'''
    
    #comm.Gather(nu.zeros((4000/size,4)),test1,root=0)
    
   
     #create fun classes for everyone
    fun_temp = ag.MC_func(data)
    fun_temp.autosetup()
    fun = fun_temp.send_class
    fun._losvd = False
    #start multi try mcmc
    if rank == 0:
        param,chi,out_sigma,Nacept,Nreject = root_mc(fun, N_try)
    else:
        worker_chi_cal(fun, N_try)
