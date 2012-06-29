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
#
#
#

import numpy as nu
import mpi4py.MPI as mpi

#123456789012345678901234567890123456789012345678901234567890123456789
def run_multi(data,burnin=5*10**3,k_max=16,option=True,rank=0,q_talk=None,q_final=None,fun=None):
    comm = mpi.COMM_WORLD
    if comm.Get_rank() == 0:
        param,chi,bayes = root_mc(fun,burnin,option.iter.value,k_max)
        return param,chi,bayes
    else:
        worker_burnin(fun)
        worker_other(fun)

    

def root_mc(fun,burnin,itter,k_max):
    #handels root stuff

    ####initalize vars and workers
    comm = mpi.COMM_WORLD
    worker_size = mpi.COMM_WORLD
    #param, model stuff
    bins = 1 #nu.random.randint(k_max) + 1
    param,active_param,chi,sigma={},{},{},{}
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
    
    #sigma stuff
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    for i in range(1,k_max + 1):
        param[str(i)]=[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,fun._age_unq.ptp()*nu.random.rand(),1.],i)
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
        bayes_fact[str(i)]=[]
    while True:
        #create starting active params
        bin = nu.log10(nu.linspace(10**fun._age_unq.min(),
                                 10**fun._age_unq.max(), bins + 1))
        bin_index = 0
        #start in random place
        for k in xrange(3*bins):
            if any(nu.array(range(0, bins*3, 3)) == k):#metalicity
                active_param[str(bins)][k]=(nu.random.random() *
                                            fun._metal_unq.ptp() + 
                                            fun._metal_unq[0])
            else:#age and normilization
                if any(nu.array(range(1, bins*3, 3)) == k): #age
                    #random place anywhere
                    active_param[str(bins)][k] = (nu.random.random()*
                                                fun._age_unq.ptp()+
                                                fun._age_unq[0]) 
                    bin_index+=1
                else: #norm
                    active_param[str(bins)][k] = nu.random.random()*10000
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

    #RJ part
    step,step_life = 0, nu.random.exponential(100)
    step_type = 'stay'
    
    #burnin stuff

    #starts burnin
    mult_N = 100
    for i in xrange(burnin):
        #choose if should stay,die,birth for step
        if step >= step_life: 
            #if past step life
            if (bins != k_max or bins != 1):
                step_type = ['stay','die','birth'][nu.random.randint(3)]
    
        if step_type == 'stay':
            param_send = map(fun.proposal,[active_param[str(bins)]]*mult_N,
                             [sigma[str(bins)]]*mult_N)
            dust_send = map(fun.proposal,[active_dust]*mult_N,
                            [sigma_dust]*mult_N)
            losvd_send = map(fun.proposal,[active_losvd]*mult_N,
                             [sigma_losvd]*mult_N)
            #send test params to workers
            comm.Scatterv([zip(param_send,dust_send,losvd_send),mpi.DOUBLE],
                          all_param,0)

    #starts main loops

    #post-processing 


def worker_burnin(fun):
    comm = mpi.COMM_WORLD
    worker_size = mpi.COMM_WORLD
    recv_param = None
    dummy = None
    while True:
        comm.Scatterv(dummy,recv_param,0)

def worker_other(fun):
    pass


if __name__ == '__main__':

    import Age_date as ag
    comm = mpi.COMM_WORLD
    if comm.Get_rank() == 0:
        data,info1,dust,weight = ag.iterp_spec(1,lam_min=2000,lam_max=10000)
        for i in range(1,comm.Get_size()):
            shape = nu.array([data.shape[0],data.shape[1]])
            comm.Send([shape,mpi.INT],dest = i)
            comm.Send([data,mpi.DOUBLE],dest = i)
    else:
        shape = nu.zeros(2,dtype='int')
        comm.Recv(shape,source = 0)
        data = nu.zeros(shape)
        comm.Recv(data,source = 0)
    assert nu.all(nu.isfinite(data[:,1]))
    fun1 = ag.MC_func(data)
    fun1.autosetup()
    fun = fun1.send_class
    if comm.Get_rank() == 0:
        root_mc(fun,fun1._burnin,fun1._itter,16)
    else:
           worker_burnin(fun1)                                
