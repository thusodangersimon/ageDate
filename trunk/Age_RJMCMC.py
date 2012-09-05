#!/usr/bin/env python
#
# Name: reverse jump monte carlo
#
# Author: Thuso S Simon
#
# Date: 25/1/12
# TODO: 
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
''' all moduals asociated with reverse jump mcmc'''

import numpy as nu
import sys
from scipy.cluster import vq as sci
from scipy.stats import levene, f_oneway,kruskal
from anderson_darling import anderson_darling_k as ad_k
from multiprocessing import *
a=nu.seterr(all='ignore')

def rjmcmc(fun, burnin=5*10**3,k_max=16,option=True,rank=0,q_talk=None,q_final=None):
    #parallel worker program reverse jump mcmc program
    nu.random.seed(random_permute(current_process().pid))
    #initalize boundaries
    lib_vals = fun._lib_vals
    metal_unq = fun._metal_unq
    age_unq = fun._age_unq
    #create fun for all number of bins
    #attempt=False
    fun._k_max = k_max
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
    if option.chibest.value>chi[str(bins)][-1]:
        option.chibest.value=chi[str(bins)][-1]+.0
        for kk in range(len(option.parambest)):
            if kk<bins*3+2+4:
                option.parambest[kk]=nu.hstack((active_param[str(bins)],
                                               active_dust,active_losvd))[kk]
            else:
                    option.parambest[kk]=nu.nan
        print ('%i has best fit with chi of %2.2f and %i bins' 
               %(rank,option.chibest.value,bins))
        sys.stdout.flush()
    #start rjMCMC
    T_cuurent,Nexchange_ratio=1.0,1.0
    size=0
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    out_dust_sig, out_losvd_sig = [sigma_dust], [sigma_losvd]

    while option.value:
        if option.iter.value%5000==0:
            print "hi, I'm at itter %i, chi %f from %s bins and for accept %2.2f" %(len(param[str(bins)]),chi[str(bins)][-1],bins,acept_rate[str(bins)][-1]*100)
            sys.stdout.flush()
            #print sigma[str(bins)].diagonal()
            #print 'Acceptance %i reject %i' %(Nacept,Nreject)
            #print active_param[str(bins)][range(2,bins*3,3)]
        #sample from distiburtion
        active_param[str(bins)] = fun.proposal(active_param[str(bins)],
                                               sigma[str(bins)])
        if fun._dust:
            active_dust = fun.proposal(active_dust,sigma_dust)
        if fun._losvd:
            active_losvd  = fun.proposal(active_losvd, sigma_losvd)
        #calculate new model and chi
        chi[str(bins)].append(0.)
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.lik(
            active_param[str(bins)], active_dust, active_losvd)
        #sort by age
        if not nu.all(active_param[str(bins)][range(1,bins*3,3)]==
                      nu.sort(active_param[str(bins)][range(1,bins*3,3)])):
            index=nu.argsort(active_param[str(bins)][range(1,bins*3,3)])
            temp_index=[] #create sorting indcci
            for k in index:
                for kk in range(3):
                    temp_index.append(3*k + kk)
            active_param[str(bins)] = active_param[str(bins)][temp_index]
         
        #decide to accept or not
        a = nu.exp((chi[str(bins)][-2] - chi[str(bins)][-1])/
                 SA(T_cuurent,burnin,T_start,T_stop))
        #metropolis hastings
        if a > nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(nu.hstack((active_param[str(bins)]
                                                       , active_dust,
                                                       active_losvd))))
            Nacept[str(bins)] += 1
            if not nu.isinf(min(chi[str(bins)])): #put temperature on order of chi calue
                T_start = nu.round(min(chi[str(bins)]))+1.
            #see if global best fit
            if option.chibest.value>chi[str(bins)][-1]:
                #set global in sharred arrays
                #option.chibest.acquire();option.parambest.acquire()
                option.chibest.value=chi[str(bins)][-1]+.0
                for kk in xrange(k_max*3):
                    if kk<bins*3+2+4:
                        option.parambest[kk]=nu.hstack((active_param[str(bins)],
                                               active_dust, active_losvd))[kk]
                    else:
                        option.parambest[kk] = nu.nan
                #option.chibest.release();option.parambest.release()
                print('%i has best fit with chi of %2.2f and %i bins, %i steps left' %(rank,option.chibest.value,bins,j_timeleft-j))
                sys.stdout.flush()
                #break
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)] = nu.copy(param[str(bins)][-1][range(3*bins)])
            if fun._dust:
                active_dust = nu.copy(param[str(bins)][-1][-6:-4])
            if fun._losvd:
                active_losvd = nu.copy(param[str(bins)][-1][-4:])
            if len(active_dust) != 2:
                print 'best'
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1

        ###########################step stuff
        sigma[str(bins)],sigma_dust,sigma_losvd = Step_func(acept_rate[str(bins)][-1],param[str(bins)][-2000:],
                                                            sigma[str(bins)],sigma_dust,sigma_losvd, bins, j,fun._dust, fun._losvd)


        #############################decide if birth or death
        active_param, temp_bins, attempt, critera = death_birth(fun, birth_rate, bins, j, j_timeleft, active_param)
        #calc chi of new model
        if attempt:
            attempt = False
            tchi, active_param[str(temp_bins)][range(2,temp_bins*3,3)] = fun.lik(
                active_param[str(temp_bins)], active_dust, active_losvd)
            bayes_fact[str(bins)].append(nu.exp((chi[str(bins)][-1]-tchi)/2.)*critera) #save acceptance critera for later
            #rjmcmc acceptance critera ##############
            if bayes_fact[str(bins)][-1]  > nu.random.rand():
                #print '%i has changed from %i to %i' %(rank,bins,temp_bins)
                #accept model change
                bins = temp_bins + 0
                chi[str(bins)].append(nu.copy(tchi))
                #sort by age so active_param[bins*i+1]<active_param[bins*(i+1)+1]
                if not nu.all(active_param[str(bins)][range(1,bins*3,3)] ==
                          nu.sort(active_param[str(bins)][range(1,bins*3,3)])):
                    index = nu.argsort(active_param[str(bins)][range(1,bins*3,3)])
                    temp_index = [] #create sorting indcci
                    for k in index:
                        for kk in range(3):
                            temp_index.append(3*k+kk)
                    active_param[str(bins)] = active_param[str(bins)][temp_index]
                param[str(bins)].append(nu.copy((nu.hstack((active_param[str(bins)],active_dust,active_losvd)))))
                j, j_timeleft = 0, nu.random.exponential(200)
                #continue
            if T_cuurent >= burnin:
                j, j_timeleft = 0, nu.random.exponential(200)
        else: #reset j and time till check for attempt jump
            j, j_timeleft = 0, nu.random.exponential(200)

        #########################################change temperature
        if nu.min([1,nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/(2.*SA(T_cuurent+1,burnin,T_start,T_stop))-(chi[str(bins)][-2]+chi[str(bins)][-1])/(2.*SA(T_cuurent,burnin,T_start,T_stop)))/T])>nu.random.rand():
            if T_cuurent<burnin:
                T_cuurent+=1
                #print T_cuurent,burnin,rank
            elif T_cuurent==round(burnin):
                print 'done with cooling'
                T_cuurent+=1
            Nexchange_ratio+=1   
        #make sure the change temp rate is aroudn 20%
        if Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))>.25:
            T=T*1.05
        elif Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))<.20:
            T=T/1.05
        #change current temperature with size of param[bin]
        if len(param[str(bins)])<burnin:
            T_cuurent=len(param[str(bins)])
        #keep on order with chi squared
        '''if j%20==0:
            if acept_rate[str(bins)][-1]>.5 and T_start<10**-5:
                T_start/=2.
                #T_stop+=.1
            elif acept_rate[str(bins)][-1]<.25 and T_start<3*10**5:
                T_start*=2.
                #T_stop-=.1'''
     ##############################convergece assment
        '''size=dict_size(param)
        if size%999==0 and size>30000:
            q_talk.put((rank,size,param))
           ''' 
        ##############################house keeping
        j+=1
        option.iter.value+=1
        acept_rate[str(bins)].append(nu.copy(Nacept[str(bins)]/(Nacept[str(bins)]+Nreject[str(bins)])))
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))
        if fun._dust:
            out_dust_sig.append(nu.copy(sigma_dust))
        if fun._losvd:
            out_losvd_sig.append(nu.copy(sigma_losvd))
    #####################################return once finished 
    for i in param.keys():
        chi[i]=nu.array(chi[i])
        param[i]=nu.array(param[i])
        ###correct metalicity and norm 
        try:
            param[i][:,range(0,3*int(i),3)]=10**param[i][:,range(0,3*int(i),3)] #metalicity conversion
            param[i][:,range(2,3*int(i),3)]=param[i][:,range(2,3*int(i),3)] #*fun.norms #norm conversion
        except ValueError:
            pass
        #acept_rate[i]=nu.array(acept_rate[i])
        #out_sigma[i]=nu.array(out_sigma[i])
        bayes_fact[i]=nu.array(bayes_fact[i])
    q_final.put((param,chi,bayes_fact,out_sigma))
    #return param,chi,sigma,acept_rate,out_sigma

def death_birth(fun, birth_rate, bins, j,j_timeleft, active_param):
    #does birth or death moved
        attempt = False
        if ((birth_rate > nu.random.rand() and bins < fun._k_max and 
             j > j_timeleft ) or (j > j_timeleft and bins == 1)):
            #birth
            attempt = True #so program knows to attempt a new model
            rand_step = nu.random.rand(3)*[fun._metal_unq.ptp(), fun._age_unq.ptp(),1.]
            rand_index = nu.random.randint(bins)
            temp_bins = 1 + bins
            #criteria for this step
            critera = 1/4.**3 * birth_rate #(1/3.)**temp_bins
            #new param step
            for k in range(len(active_param[str(bins)])):
                active_param[str(temp_bins)][k]=active_param[str(bins)][k]
            #set last 3 and rand_index 3 to new
            if .5 > nu.random.rand(): #x'=x+-u
                active_param[str(temp_bins)][-3:] = (active_param[str(bins)][rand_index*3:rand_index*3+3] + 
                                                     rand_step)
                active_param[str(temp_bins)][rand_index*3:rand_index*3+3] = (
                    active_param[str(bins)][rand_index*3:rand_index*3+3] - rand_step)
                k = 0
                #check to see if in bounds
                while fun.prior(nu.hstack((active_param[str(temp_bins)],
                                           nu.zeros(2)))): 
                    k += 1
                    if k < 100:
                        rand_step = nu.random.rand(3) * [fun._metal_unq.ptp(), fun._age_unq.ptp(),1.]
                    else:
                        rand_step /= 2.
                    active_param[str(temp_bins)][-3:] = (
                        active_param[str(bins)][rand_index*3:rand_index*3+3] + rand_step)
                    active_param[str(temp_bins)][rand_index*3:rand_index*3+3]=(
                        active_param[str(bins)][rand_index*3:rand_index*3+3]-rand_step)
            else: #draw new values randomly from param space
                active_param[str(temp_bins)][-3:] = (nu.random.rand(3) * 
                                                     nu.array([fun._metal_unq.ptp(), fun._age_unq.ptp(),5.]) + 
                                                     nu.array([fun._metal_unq.min(), fun._age_unq.min(), 0]))
        elif j > j_timeleft and bins > 1 and  0.01 < nu.random.rand():
            #death
            attempt = True #so program knows to attempt a new model
            temp_bins = bins - 1
            #criteria for this step
            critera = 4.**3 * (1 - birth_rate) #3.**temp_bins
            if .5 > nu.random.rand():
                #remove bins with 1-N/Ntot probablitiy
                Ntot = nu.sum(active_param[str(bins)][range(2,bins*3,3)])
                rand_index = (rand_choice(active_param[str(bins)][range(2,bins*3,3)],
                                          active_param[str(bins)][range(2,bins*3,3)]/Ntot))
                k = 0
                for ii in xrange(bins): #copy to lower dimestion
                    if not ii == rand_index:
                        active_param[str(temp_bins)][3*k:3*k+3] = nu.copy(active_param[str(bins)]
                                                                        [3*ii:3*ii+3])
                        k += 1
            else: #average 2 componets together for new values
                rand_index = nu.random.permutation(bins)[:2] #2 random indeci
                k = 0
                for ii in xrange(bins):
                    if not any(ii == rand_index):
                        active_param[str(temp_bins)][3*k:3*k+3] = nu.copy(active_param[str(bins)]
                                                                        [3*ii:3*ii+3])
                        k += 1
                active_param[str(temp_bins)][3*k:3*k+3] = (active_param[str(bins)][3*rand_index[0]:3*rand_index[0]+3]+active_param[str(bins)] [3*rand_index[1]:3*rand_index[1]+3])/2.
        '''elif j > j_timeleft: #move to global bestfit
            attempt = True
            #extract best fit from global array
            best_param = nu.array(option.parambest)
            best_param = best_param[~nu.isnan(best_param)]
            temp_bins = (len(best_param)-2-4)/3
            #calculate occam factor * model select prob
            critera = 4. **(bins - temp_bins) * 0.01
            active_param[str(temp_bins)] = nu.copy(best_param[range(3*temp_bins)])
            active_dust = nu.copy(best_param[-6:-4])
            active_losvd = nu.copy(best_param[-4:])'''
        if attempt:
            return active_param, temp_bins, attempt, critera
        else:
            return active_param, None, attempt, None

def is_send(N1,N2,N_prev): 
    #counds the number of values in a list inside of a dict
    val_N=0
    for i in N1.keys():
        val_N+=N1[i]+N2[i]-N_prev['accept'][i]-N_prev['reject'][i]
    return val_N

def SA(i,i_fin,T_start,T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    if i>i_fin:
        return 1.0
    else:
        return (T_stop-T_start)/float(i_fin)*i+T_start

def random_permute(seed):
    #does random sequences to produice a random seed for parallel programs
    ##middle squared method
    seed = str(seed**2)
    while len(seed) < 7:
        seed=str(int(seed)**2)
    #do more randomization
    ##multiply with carry
    a,b,c = int(seed[-1]), 2**32, int(seed[-3])
    j = nu.random.random_integers(4, len(seed))
    for i in range(int(seed[-j:-j+3])):
        seed = (a*int(seed) + c) % b
        c = (a * seed + c) / b
        seed = str(seed)
    return int(seed)

def Step_func(acept_rate, param, sigma, sigma_dust, sigma_losvd, bins, j, isdust, islosvd):
    #changes step size if needed
    if  (acept_rate < 0.234 and all(sigma.diagonal() >= 10**-5)):
               #too few aceptnce decrease sigma
        sigma  /= 1.05
        if isdust:
            sigma_dust /= 1.05
        if islosvd: 
            sigma_losvd /= 1.05

    elif (acept_rate > .040 and 
          all(sigma.diagonal()[nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()] < 5.19)): #not enough
        sigma *= 1.05
            #dust step
        if isdust:
            sigma_dust *= 1.05
            #losvd step
        if islosvd:
            sigma_losvd *= 1.05
    #use covarnence matrix
    if j %100 == 0 and j != 0: 
        t_param = nu.array(param)
        try:
            tsigma = Covarence_mat(t_param[:,range(3*bins)],
                                  t_param.shape[0]-1)
            if isdust:
                tsigma_dust = Covarence_mat(t_param[:,-6:-4],t_param.shape[0]-1)
            if islosvd:
                tsigma_losvd = Covarence_mat(t_param[:,-4:],t_param.shape[0]-1)
        except IndexError:
            print t_param.shape
            #error handeling some time cov is nan
        if  (nu.all(nu.isfinite(tsigma)) or 
             nu.all(nu.isfinite(tsigma_dust)) or 
             nu.all(nu.isfinite(tsigma_losvd))):
                #set equal to last cov matirx
            sigma = tsigma
            if isdust:
                sigma_dust = tsigma_dust
            if islosvd:
                sigma_losvd = tsigma_losvd

    return sigma, sigma_dust, sigma_losvd

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


def rand_choice(x, prob):
    #chooses value from x with probabity of prob**-1 
    #so lower prob values are more likeliy
    #x must be monotonically increasing
    if not nu.sum(prob) == 1: #make sure prob equals 1
        prob = prob / nu.sum(prob)
    if nu.any(prob == 0): #get weird behavor when 0
        #smallest value for float32 and 1/value!=inf
        prob[prob == 0] = 6.4e-309 
    #check is increasing
    u = nu.random.rand()
    if nu.all(x == nu.sort(x)): #if sorted easy
        N = nu.cumsum(prob ** -1 / nu.sum(prob ** -1))
        index = nu.array(range(len(x)))
    else:
        index = nu.argsort(x)
        temp_x = nu.sort(x)
        N = nu.cumsum(prob[index] ** -1 / nu.sum(prob[index] ** - 1))
    try:
        return index[nu.min(nu.abs(N - u)) == nu.abs(N - u)][0]
    except IndexError:
        print x,prob
        raise

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

def dic_data(temp,burnin):
    '''processes data from rjmcmc, should be a list of tuples,
    containin dicts'''
    bayes_fac={}
    for i in temp:
        for j in i[2].keys():
            try:
                bayes_fac[j]=nu.concatenate((bayes_fac[j],i[2][j]))
            except KeyError:
                bayes_fac[j]=i[2][j]

    fac=[]
    for i in bayes_fac.keys():
        if bayes_fac[i].shape[0]>0:
            bayes_fac[i][bayes_fac[i]>1]=1. #accept critera is min(1,alpha)
            fac.append([int(i),nu.mean(nu.nan_to_num(bayes_fac[i])),len(bayes_fac[i])])
            #remove 1st bin for now#############
    fac=nu.array(fac)
    #grab chains with best fit and chech to see if mixed properly
    outparam,outchi={},{}
    for i in fac[:,0]:
        outparam[str(int(i))],outchi[str(int(i))]=nu.zeros([2,3*i+2+4]),nu.array([nu.inf])
    for i in temp:
        for j in fac[:,0]:
            try:
                outparam[str(int(j))]=nu.concatenate((outparam[str(int(j))],i[0][str(int(j))][~nu.isinf(i[1][str(int(j))][1:]),:]),axis=0)
                outchi[str(int(j))]=nu.concatenate((outchi[str(int(j))],i[1][str(int(j))][~nu.isinf(i[1][str(int(j))])]))
            except ValueError: #if empty skip
                pass
    for j in nu.int64(fac[:,0]): #post processing
        outparam[str(int(j))],outchi[str(int(j))]=outparam[str(int(j))][2+burnin:,:],outchi[str(int(j))][1+burnin:]
    #remove empty bins
    for i in outparam.keys():
        if not nu.any(outparam[i]):
            outparam.pop(i)
            outchi.pop(i)
            fac= fac[fac[:,0]!=int(i)]
    
    return outparam,outchi,fac[fac[:,0].argsort(),:]

if __name__=='__main__':

    #profiling
    import cProfile as pro
    import cPickle as pik
    from Age_date import *
    #temp=pik.load(open('0.3836114.pik'))
    data,info1,weight,dust=iterp_spec(1)
    #j='0.865598733333'
    #data=temp[3][j]
    burnin,k_max,cpus=5000,16,1
    option=Value('b',True)
    option.cpu_tot=cpus
    option.iter=Value('i',True)
    option.chibest=Value('d',nu.inf)
    option.parambest=Array('d',nu.ones(k_max*3+2)+nu.nan)
    fun=MC_func(data)
    fun.autosetup()
    #interpolate spectra so it matches the data
    #global spect
    #spect=data_match_all(data)
    assert fun.send_class.__dict__.has_key('_lib_vals')
    rank=1
    q_talk,q_final=Queue(),Queue()
    pro.runctx('rjmcmc(fun,burnin,k_max,option,rank,q_talk,q_final)'
               , globals(),{'fun': fun.send_class, 'burnin':burnin,'k_max':k_max,
                            'rank':1,'q_talk':q_talk,'q_final':q_final
                            ,'option':option}
               ,filename='agedata1.Profile')
