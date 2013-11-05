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
import sys,os
from scipy.cluster import vq as sci
from scipy.stats import levene, f_oneway,kruskal
from anderson_darling import anderson_darling_k as ad_k
from multiprocessing import *
import time as Time
import signal
import cPickle as pik
from glob import glob
import acor
a=nu.seterr(all='ignore')


def RJMC_main(fun, option, burnin=5*10**3, birth_rate=0.5,max_iter=10**5, seed=None, fail_recover=True):
    '''(likelihood object, running object, int, int, bool) ->
    dict(ndarray), dict(ndarray)

    Runs Reversible jump markov chain monte carlo using to maximize
    the likelihood class.

    fun is the likelyhod class.
    option is running class and controls multiprocessing and stopping 
    of the algorthom.
    burnin is the number of itterations to do burn-in.
    seed is random seed (optional), usefull when runing in multiprocess
    mode.
    tot_iter is number of iterations to stop at.
    fail recover tell program to search local dir for file called "failed_gid.pik" incase
    tring to revcover from crash. Can be file path to recovery file or bool wheather to check

    outputs:
    dictonary of params, the different keys use different modesl.
    dictornary of log of the likelihood*log-priors
    '''
    #set global for graceful exit
    #stop = option.iter_stop
    #global stop
    #signal.signal(signal.SIGINT, signal_save_rjmcmc)
    #see if to use specific seed
    if seed is not None:
        nu.random.seed(seed)
    #see if should recover from file
    eff = -999999
    if type(fail_recover) is bool:
            #look for previous file
            if fail_recover:
                initalize = False
                reover_file = glob('failed*.pik')
                if len(reover_file) == 0:
                    print 'No recovery files found. Starting new run'
                    initalize = True
                    
                else:
                    initalize = False
            else:
                initalize = True
    elif type(fail_recover) is str:
            #load from path
            initalize = False
            reover_file = glob(fail_recover)
    else:
            #do nothing and initalize normaliy
            initalize = True
    #initalize parameters from class
    if initalize:
        active_param, sigma = rj_dict(), rj_dict()
        param,chi = rj_dict(), rj_dict()
        Nacept, Nreject = rj_dict(), rj_dict()
        acept_rate, out_sigma = rj_dict(), rj_dict()
        bayes_fact = rj_dict() #to calculate bayes factor
        #simulated anneling param
        T_cuurent = rj_dict()
        for i in fun.models.keys(): ####todo add random combination of models
                active_param[i], sigma[i] = fun.initalize_param(i)
        #start with random model
        bins = nu.random.choice(fun.models.keys())

        #set other RJ params
        Nacept[bins] , Nreject[bins] = 1.,1.
        acept_rate[bins], out_sigma[bins] = [1.], [sigma[bins][0][:]]
        #bayes_fact[bins] = #something
        T_cuurent[bins] = 0
        #set storage functions
        param[bins] = [active_param[bins].copy()]
        #first lik calc
        #print active_param[bins]
        chi[bins] = [fun.lik(active_param,bins) + fun.prior(active_param,bins)]
        #check if starting off in bad place ie chi=inf or nan
        if not nu.isfinite(chi[bins][-1]):
                t = Time.time()
                print 'Inital possition failed retrying for 1 min'
                #try different chi for 1 min and then give up
                while Time.time() - t < 60:
                        active_param[bins], sigma[bins] = fun.initalize_param(bins)
                        temp = fun.lik(active_param,bins) + fun.prior(active_param,bins)
                        if nu.isfinite(temp):
                                chi[bins][-1] = nu.copy(temp)
                                param[bins][-1] = active_param[bins].copy()
                                print 'Good starting point found. Starting RJMCMC'
                                break
                else:
                        #didn't find good place exit program
                        raise ValueError('No good starting place found, check initalization')
	
        #start rjMCMC
        Nexchange_ratio = 1.0
        size,a = 0,0
        j, j_timeleft = 1, nu.random.exponential(100)
        T_start,T_stop = chi[bins][-1]+0, 1.
        trans_moves = 0
    else:
        #load failed params
        (fun.data,active_param,sigma,param,chi,bins,Nacept,Nreject,acept_rate,out_sigma,
                    option.current,T_cuurent,j,j_timeleft,fun._multi_block,T_start,T_stop,
                    trans_moves) = pik.load(open(reover_file[0]))
    #profiling
    t_pro,t_swarm,t_lik,t_accept,t_step,t_unsitc,t_birth,t_house,t_comm = [],[],[],[],[],[],[],[],[] 
    while option.iter_stop:
        #show status of running code
        if T_cuurent[bins] % 501 == 0:
            show = ('acpt = %.2f,log lik = %e, bins = %s, steps = %i,ESS = %2.0f'
                    %(acept_rate[bins][-1],chi[bins][-1],bins, option.current,eff))
            print show
            sys.stdout.flush()
			
        #sample from distiburtion
        t_pro.append(Time.time())
        
        active_param[bins] = fun.proposal(active_param[bins], sigma[bins])
        t_pro[-1] -= Time.time()
        #swarm stuff
        t_swarm.append(Time.time())
        '''active_param[str(bins)], active_dust, active_losvd, birth_rate = swarm_function(active_param[str(bins)]'''
        #if option.rank == 1:
            #print 'after',active_param[str(bins)] 

        t_swarm[-1] -= Time.time()
        #calculate new model and chi
        t_lik.append(Time.time())
        chi[bins].append(0.)
        chi[bins][-1] = fun.lik(active_param,bins) + fun.prior(active_param,bins)
        #just lik part
        a = (chi[bins][-1] - chi[bins][-2])
        #simulated anneling
        a /= SA(T_cuurent[bins],burnin,abs(T_start),T_stop)
        
        t_lik[-1]-=Time.time()
        t_accept.append(Time.time())
        #put temperature on order of chi calue
        if T_start < chi[str(bins)][-1]:
            T_start = chi[str(bins)][-1]+0
        #metropolis hastings
        #print a
        if nu.exp(a) > nu.random.rand():
            #acepted
            param[bins].append(active_param[bins].copy())
            Nacept[bins] += 1
        else:
            #rejected
            param[bins].append(param[bins][-1].copy())
            active_param[bins] = param[bins][-1].copy()
            chi[bins][-1] = nu.copy(chi[bins][-2])
            Nreject[bins]+=1
        t_accept[-1]-=Time.time()
        ###########################step stuff
        t_step.append(Time.time())
        if T_cuurent[bins] < burnin + 5000 or acept_rate[bins][-1]<.11:
            #only tune step if in burn-in
            sigma[bins] =  fun.step_func(acept_rate[bins][-1] ,param[bins], sigma, bins)
        t_step[-1]-=Time.time()
        #############################decide if birth or death
        t_birth.append(Time.time())
        
        if j >= j_timeleft:
            #print len(active_param[bins].keys())
            active_param, temp_bins, attempt, critera = fun.birth_death(birth_rate, bins, active_param)
            #print len(active_param[bins].keys())
            #print bins, active_param[bins].shape, sigma[bins][0].shape
            if attempt:
                #check if accept move
                tchi = fun.lik(active_param, temp_bins)+fun.prior(active_param, temp_bins)
                #likelihoods
                rj_a = (tchi - chi[bins][-1])
                #model priors
                rj_a += (fun.model_prior(temp_bins) - fun.model_prior(bins))
                trans_moves += 1
                #simulated aneeling 
                rj_a /= SA(trans_moves,50,abs(chi[bins][0]),T_stop)
                #RJ-MH critera
                #print active_param[temp_bins]
                if nu.exp(rj_a) * critera > nu.random.rand():
                    #accept move
                    bins = temp_bins + ''
                    #see if model has be created before
                    if not chi.has_key(bins):
                        chi[bins] ,param[bins] = [] ,[] 
                        T_cuurent[bins] = 0
                        acept_rate[bins] = [.5]
                        Nacept[bins] , Nreject[bins] = 1., 1.
                        out_sigma[bins] = []
                    chi[bins].append(tchi + 0)
                    param[bins].append(active_param[bins].copy())
                    
                    #allow for quick tuneing of sigma
                    if T_cuurent[bins] > burnin + 5000:
                        T_cuurent[bins] = burnin + 4800
                        Nacept[bins] , Nreject[bins] = 1., 1.
                else:
                    pass
                j, j_timeleft = 0 , 1#nu.random.exponential(100)
                #print T_cuurent[str(bins)],burnin,T_start,T_stop
                attempt = False
        t_birth[-1]-=Time.time()
        
        #########################################change temperature
        T_cuurent[bins] += 1
        ###################bad performance notifier
        if  (acept_rate[bins][-1] < .15
                                 and T_cuurent[bins] > burnin):
            if not fun._multi_block:
                print "acceptance is bad starting multi-block"
                fun._multi_block = True
                Nacept[bins] , Nreject[bins] = 1., 1.
                #T_cuurent[bins] = burnin + 4000 
        ################turn off burnin after 40% percent of total iterations
        if option.current == int(max_iter*.4):
            for i in T_cuurent.keys():
                if not T_cuurent[i] > burnin:
                    T_cuurent[i] = burnin + 1
        ##############################convergece assment
       
        ##############################house keeping
        t_house.append(Time.time())
        j+=1
        option.current += 1
        acept_rate[bins].append(nu.copy(Nacept[bins]/(Nacept[bins]+Nreject[bins])))
        #out_sigma[bins].append(sigma[bins][:])
        #save current state incase of crash
        if option.current % 500 == 0:
            try:
                os.popen('mv failed_%i.pik failed_%i.pik.bak'%(os.getpid(),os.getpid()))
                pik.dump((fun.data,active_param,sigma,param,chi,bins,Nacept,Nreject,acept_rate,out_sigma,option.current,T_cuurent,j,j_timeleft,fun._multi_block,T_start,T_stop,trans_moves),open('failed_%i.pik'%(os.getpid()),'w'),2)
                os.popen('rm failed_%i.pik.bak'%os.getpid())
            except OSError:
                print 'Warning: Running out of memory. May crash soon'
                pik.dump((fun.data,active_param,sigma,param,chi,bins,Nacept,Nreject,acept_rate,out_sigma,option.current,T_cuurent,j,j_timeleft,fun._multi_block,T_start,T_stop,trans_moves),open('failed_%i.pik'%(os.getpid()),'w'),2)
        t_house[-1]-=Time.time()
        #t_comm.append(Time.time())
        #t_comm[-1]-=Time.time()
        #if mpi isn't on allow exit
        if option.comm_world.size == 1:
            #exit if reached max iterations
            if option.current > max_iter:
                option.iter_stop = False
            #exit if reached target effective sample size
            if option.current % 501 == 0:
                eff = ess(param[bins])
                if eff > 10**5:
                    option.iter_stop = False
        #pik.dump((t_pro,t_swarm,t_lik,t_accept,t_step,t_unsitc,t_birth,t_house,t_comm),open('time_%i.pik'%option.rank_world,'w'),2)
    #####################################return once finished 
    #remove incase of crash file
    os.popen('rm failed_%i.pik'%(os.getpid()))
		#pik.dump((t_pro,t_swarm,t_lik,t_accept,t_step,t_unsitc,t_birth,t_house,t_comm,param,chi),open('time_%i.pik'%option.rank_world,'w'),2)
    return param, chi, acept_rate #, out_sigma, param.keys()

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

def ess(t):
    '''returns the average sample size of a N,M ndarray (doesn't actually take a N,M ndarray'''
    #extract data from dictoranry
    data = []
    for i in t:
        dics = []
        for j in i.keys():
            dics.append(nu.ravel(i[j]))
        data.append(nu.hstack(dics))
    data = nu.asarray(data)
    temp_ess,Len = [],float(len(data[:,0]))
    #calculate ess for each parameter
    for i in range(data.shape[1]):
        try:
            temp_ess.append(Len/acor.acor(data[:,i])[0])
        except ZeroDivisionError:
            pass
        except RuntimeError:
            pass
    if len(temp_ess) == 0:
        return 0.
    #return max
    return nu.nanmax(temp_ess)

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

class rj_dict(dict):
    '''like built in dict, but has methods __add_ and __sub__ to add more key
    words and subrtract them'''
    def __add__(self,x):
        '''Combines models together, keywords become almagination of the 2
        '''
        newkey = ''
        new_vals = []
        for i in x.keys():
            newkey += i + ','
            new_vals.append(x[i])
        for i in self.keys():
            newkey += i + ','
            new_vals.append(self[i])
        newkey = newkey[:-1]
        out = rj_dict()
        out[newkey] = new_vals
        return out
        #remove old key
        
    def __iadd__(self,x):
        return __add__(x)

    def __isub__(self,x):
        return __sub__(x)

    def __sub__(self,x):
        '''removes keyword from dict does last one first
        >>>a = rj_dict()
        >>>a['1,2,1']=[[1],[2],[3]]
        >>>a -'2'
        {'1,1':[[1],[3]]}
        >>>a - '1'
        {'1,2':[[1],[2]]}'''
        assert type(x) is str, "can only remove proper key value"
        keys = self.keys()[0]
        keys = keys.split(',')
        vals = self.copy().values()[0][:]
        max = -1
        for i,j in enumerate(keys):
            if j == x:
                max = i
        vals.pop(max)
        keys.pop(max)
        newkeys = ''
        for i in keys:
            newkeys += i + ','
        newkeys = newkeys[:-1]
        out = rj_dict()
        out[newkeys] = vals
        return out

#gracefull exit
#####signal catcher so can exit anytime and save results
def signal_save_rjmcmc(signal, frame):
    print 'Stopping run'
    global stop
    print stop
    stop = False
    Time.sleep(5)
    raise
    
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
