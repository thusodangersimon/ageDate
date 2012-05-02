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

from Age_date import *
from scipy.cluster import vq as sci
from scipy.stats import levene, f_oneway,kruskal
from anderson_darling import anderson_darling_k as ad_k
a=nu.seterr(all='ignore')



def RJ_multi(data,burnin,max_itter=5*10**5,k_max=16,cpus=cpu_count()):
    #does multiple chains for RJMCMC
    #shares info about chain to other chains
    option=Value('b',True)
    option.cpu_tot=cpus
    option.iter=Value('i',True)
    option.chibest=Value('d',nu.inf)
    option.parambest=Array('d',nu.ones(k_max*3+2)+nu.nan)

    #interpolate spectra so it matches the data
    global spect
    spect=data_match_all(data)
    work=[]
    q_talk,q_final=Queue(),Queue()
    #start multiprocess mcmc
    for ii in range(cpus):
        work.append(Process(target=rjmcmc,args=(data,burnin,k_max,option,ii,q_talk,q_final)))
        work[-1].start()

    rank,size,conver_test=-nu.ones(cpus),nu.zeros(cpus),[]
    while option.iter.value<=max_itter+burnin*cpus and option.value: 
        if q_talk.qsize()>=cpus:
            conver_test=[]
            for i in range(cpus):
                try:
                    rank[i],size[i],temp=q_talk.get(timeout=1)
                except:
                    break
                conver_test.append(temp)
            #make sure recived chains are from different processes
            if not all(nu.sort(rank)==range(cpus)):
                pass#continue
            else:
                print 'Starting convergence test'
            key_to_use=conver_test[0].keys()
            for i in conver_test:
                j=0
                while j<len( key_to_use):
                    if len(i[key_to_use[j]])<1000:
                        key_to_use.remove(key_to_use[j])
                    else:
                        j+=1
            if key_to_use and not len(nu.unique(rank))<=cpus/2.:
                #do convergence calc
                if Convergence_tests(conver_test,key_to_use):
                    #make sure not all from same chain
                    print 'Convergence! wrapping up'
                    #tidy up and end
                    sys.stdout.flush()
                    option.value=False
                else:
                    print 'Convergence test failed'
            else:
                print 'Convergence test failed'
        else:
            Time.sleep(5)
            print '%2.2f percent done' %((float(option.iter.value)/max_itter)*100.)
    option.value=False
    #wait for proceses to finish
    count=0
    temp=[]
    while count<cpus:
        option.value=False
        try:
            temp.append(q_final.get(timeout=1))
            count+=1
        except:
            print len(temp)
            break
    #post processing
    #decides which model is the best and which one has best chi sqared
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
        outparam[str(int(i))],outchi[str(int(i))]=nu.zeros([2,3*i+2]),nu.array([nu.inf])
    for i in temp:
        for j in fac[:,0]:
            try:
                outparam[str(int(j))]=nu.concatenate((outparam[str(int(j))],i[0][str(int(j))][~nu.isinf(i[1][str(int(j))][1:]),:]),axis=0)
                outchi[str(int(j))]=nu.concatenate((outchi[str(int(j))],i[1][str(int(j))][~nu.isinf(i[1][str(int(j))])]))
            except ValueError: #if empty skip
                pass
    for j in nu.int64(fac[:,0]): #post processing
        outparam[str(int(j))],outchi[str(int(j))]=outparam[str(int(j))][2+burnin:,:],outchi[str(int(j))][1+burnin:]
        #outparam[str(int(j))][:,range(0,3*j,3)]=10**outparam[str(int(j))][:,range(0,3*j,3)]
        #outparam[str(int(j))][:,range(2,3*j,3)]=outparam[str(int(j))][:,range(2,3*j,3)]
    
    return outparam,outchi,fac[fac[:,0].argsort(),:]


def rjmcmc(data,burnin=5*10**3,k_max=16,option=True,rank=0,q_talk=None,q_final=None):
    #parallel worker program reverse jump mcmc program
    nu.random.seed(random_permute(current_process().pid))
    #initalize boundaries
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #create fun for all number of bins
    attempt=False
    fun,param,active_param,chi,sigma={},{},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    active_dust,sigma_dust=nu.random.rand(2)*4.,nu.identity(2)*nu.random.rand()*2
    bayes_fact={} #to calculate bayes factor
    fun=MC_func(data)
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
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.func_N_norm(active_param[str(bins)],active_dust)
    #check if starting off in bad place ie chi=inf
        if nu.isinf(chi[str(bins)][-1]):
            continue
        else:
            break
    param[str(bins)].append(nu.copy(nu.hstack((active_param[str(bins)],active_dust))))
    #set best chi and param
    if option.chibest.value>chi[str(bins)][-1]:
        option.chibest.value=chi[str(bins)][-1]+.0
        for kk in range(k_max):
            if kk<bins*3+2:
                option.parambest[kk]=nu.hstack((active_param[str(bins)],
                                               active_dust))[kk]
            else:
                option.parambest[kk]=nu.nan
    #start rjMCMC
    T_cuurent,Nexchange_ratio=1.0,1.0
    size=0
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    
    while option.value:
        if size%5000==0:
            print "hi, I'm at itter %i, chi %f from %s bins and for cpu %i" %(len(param[str(bins)]),chi[str(bins)][-1],bins,rank)
            sys.stdout.flush()
            #print sigma[str(bins)].diagonal()
            #print 'Acceptance %i reject %i' %(Nacept,Nreject)
            #print active_param[str(bins)][range(2,bins*3,3)]
        #sample from distiburtion
        active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
        active_dust=chain_gen_dust(active_dust,sigma_dust)
        #calculate new model and chi
        chi[str(bins)].append(0.)
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.func_N_norm(active_param[str(bins)],active_dust)
        #sort by age
        if not nu.all(active_param[str(bins)][range(1,bins*3,3)]==
                      nu.sort(active_param[str(bins)][range(1,bins*3,3)])):
            index=nu.argsort(active_param[str(bins)][range(1,bins*3,3)])
            temp_index=[] #create sorting indcci
            for k in index:
                for kk in range(3):
                    temp_index.append(3*k+kk)
            active_param[str(bins)]=active_param[str(bins)][temp_index]
         
        #decide to accept or not
        a=nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/
                 SA(T_cuurent,burnin,T_start,T_stop))
        #metropolis hastings
        if a>nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(nu.hstack((active_param[str(bins)],active_dust))))
            Nacept[str(bins)]+=1
            if not nu.isinf(min(chi[str(bins)])): #put temperature on order of chi calue
                T_start=nu.round(min(chi[str(bins)]))+1.
            #see if global best fit
            if option.chibest.value>chi[str(bins)][-1]:
                #set global in sharred arrays
                option.chibest.value=chi[str(bins)][-1]+.0
                for kk in xrange(k_max*3):
                    if kk<bins*3+2:
                        option.parambest[kk]=nu.hstack((active_param[str(bins)],
                                               active_dust))[kk]
                    else:
                        option.parambest[kk]=nu.nan
                print '%i has best fit with chi of %2.2f and %i bins, %i steps left' %(rank,option.chibest.value,bins,j_timeleft-j)
                #break
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)]=nu.copy(param[str(bins)][-1][:-2])
            active_dust=nu.copy(param[str(bins)][-1][-2:])
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1

        ###########################step stuff
        #if len(param[str(bins)])<6000: #change sigma with acceptance rate
        if  (acept_rate[str(bins)][-1]<.234 and 
             all(sigma[str(bins)].diagonal()[nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()]<5.19)):
               #too few aceptnce decrease sigma
            sigma[str(bins)]=sigma[str(bins)]/1.05
            sigma_dust/=1.05
        elif acept_rate[str(bins)][-1]>.40 and all(sigma[str(bins)].diagonal()>=10**-5): #not enough
            sigma[str(bins)]=sigma[str(bins)]*1.05
            sigma_dust*=1.05
            # else: #use covarnence matrix
        if j%100==0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
            sigma[str(bins)]=Covarence_mat(nu.array(param[str(bins)])[:,:-2],len(param[str(bins)]))
                
            sigma_dust=Covarence_mat(nu.array(param[str(bins)])[:,-2:],len(param[str(bins)]))
                #active_param=fun.n_neg_lest(active_param)
        #if nu.all(sigma[str(bins)].diagonal()==0): #if step is 0 make small
        #    for k in xrange(sigma[str(bins)].shape[0]):
        #        sigma[str(bins)][k,k]=10.**-5

        #############################decide if birth or death
        if (birth_rate>nu.random.rand() and bins<k_max and j>j_timeleft ) or (j>j_timeleft and bins==1):
            #birth
            attempt=True #so program knows to attempt a new model
            rand_step,rand_index=nu.random.rand(3)*[metal_unq.ptp(), age_unq.ptp(),1.],nu.random.randint(bins)
            temp_bins=1+bins
            #criteria for this step
            critera=(1/2.)**temp_bins
            #new param step
            for k in range(len(active_param[str(bins)])):
                active_param[str(temp_bins)][k]=active_param[str(bins)][k]
            #set last 3 and rand_index 3 to new
            if .5>nu.random.rand(): #x'=x+-u
                active_param[str(temp_bins)][-3:]=active_param[str(bins)][rand_index*3:rand_index*3+3]+rand_step
                active_param[str(temp_bins)][rand_index*3:rand_index*3+3]=active_param[str(bins)][rand_index*3:rand_index*3+3]-rand_step
                k=0
                while Check(active_param[str(temp_bins)],metal_unq, age_unq, temp_bins): #check to see if in bounds
                    k+=1
                    if k<100:
                        rand_step=nu.random.rand(3)*[metal_unq.ptp(), age_unq.ptp(),1.]
                    else:
                        rand_step=rand_step/2.
                    active_param[str(temp_bins)][-3:]=active_param[str(bins)][rand_index*3:rand_index*3+3]+rand_step
                    active_param[str(temp_bins)][rand_index*3:rand_index*3+3]=active_param[str(bins)][rand_index*3:rand_index*3+3]-rand_step
            else: #draw new values randomly from param space
                active_param[str(temp_bins)][-3:]=nu.random.rand(3)*nu.array([metal_unq.ptp(),age_unq.ptp(),5.])+nu.array([metal_unq.min(),age_unq.min(),0])
        elif j>j_timeleft and bins>1:
            #death
            attempt=True #so program knows to attempt a new model
            temp_bins=bins-1
            #criteria for this step
            critera=2.**temp_bins
            if .5>nu.random.rand():
                #remove bins with 1-N/Ntot probablitiy
                Ntot=nu.sum(active_param[str(bins)][range(2,bins*3,3)])
                rand_index=rand_choice(active_param[str(bins)][range(2,bins*3,3)],active_param[str(bins)][range(2,bins*3,3)]/Ntot)
                k=0
                for ii in xrange(bins): #copy to lower dimestion
                    if not ii==rand_index:
                        active_param[str(temp_bins)][3*k:3*k+3]=nu.copy(active_param[str(bins)]
                                                                        [3*ii:3*ii+3])
                        k+=1
            else: #average 2 componets together for new values
                rand_index=nu.random.permutation(bins)[:2] #2 random indeci
                k=0
                for ii in xrange(bins):
                    if not any(ii==rand_index):
                        active_param[str(temp_bins)][3*k:3*k+3]=nu.copy(active_param[str(bins)]
                                                                        [3*ii:3*ii+3])
                        k+=1
                active_param[str(temp_bins)][3*k:3*k+3]=(active_param[str(bins)][3*rand_index[0]:3*rand_index[0]+3]+active_param[str(bins)] [3*rand_index[1]:3*rand_index[1]+3])/2.
        #calc chi of new model
        if attempt:
            attempt=False
            tchi,active_param[str(temp_bins)][range(2,temp_bins*3,3)]=fun.func_N_norm(active_param[str(temp_bins)],active_dust)
            bayes_fact[str(bins)].append(nu.exp((chi[str(bins)][-1]-tchi)/2.)*birth_rate*critera) #save acceptance critera for later
            #rjmcmc acceptance critera ##############
            if bayes_fact[str(bins)][-1]>nu.random.rand():
                #print '%i has changed from %i to %i' %(rank,bins,temp_bins)
                #accept model change
                bins=temp_bins+0
                chi[str(bins)].append(nu.copy(tchi))
                #sort by age so active_param[bins*i+1]<active_param[bins*(i+1)+1]
                if not nu.all(active_param[str(bins)][range(1,bins*3,3)]==
                          nu.sort(active_param[str(bins)][range(1,bins*3,3)])):
                    index=nu.argsort(active_param[str(bins)][range(1,bins*3,3)])
                    temp_index=[] #create sorting indcci
                    for k in index:
                        for kk in range(3):
                            temp_index.append(3*k+kk)
                    active_param[str(bins)]=active_param[str(bins)][temp_index]
                param[str(bins)].append(nu.copy((nu.hstack((active_param[str(bins)],active_dust)))))
                j,j_timeleft=0,nu.random.exponential(200)
            if T_cuurent>=burnin:
                j,j_timeleft=0,nu.random.exponential(200)

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
        ################go to best fit place if after burnin
        if (T_cuurent>burnin-500 and 
            bins!=(nu.sum(~nu.isnan(option.parambest))-2)/3 and 
            nu.random.rand()<.01):
            #get correct RJ factor
            temp_bins=(nu.sum(~nu.isnan(option.parambest))-2)/3
            if bins>temp_bins:
                critera=(1/2.)**temp_bins
            else:
                critera=(2.)**temp_bins
            if critera*nu.exp((chi[str(bins)][-1]-option.chibest.value)/2.)>nu.random.rand():
                #accept change
                #sigma[str(temp_bins)]=nu.i
                bins=temp_bins+0
                for kk in xrange(3*bins):
                    active_param[str(bins)][kk]=nu.copy(option.parambest[kk])
                for kk in xrange(3*bins,3*bins+2):
                    active_dust[kk-3*bins]=nu.copy(option.parambest[kk])
                chi[str(bins)].append(nu.copy(option.chibest.value))
                param[str(bins)].append(nu.copy(option.parambest)[:3*bins+2])
                print '%i is switching to best fit' %rank
                #print active_param[str(bins)].shape,sigma[str(bins)].shape,bins
            
        ##############################convergece assment
        size=dict_size(param)
        if size%999==0 and size>30000:
            q_talk.put((rank,size,param))
                
        ##############################house keeping
        j+=1
        option.iter.value+=1
        acept_rate[str(bins)].append(nu.copy(Nacept[str(bins)]/(Nacept[str(bins)]+Nreject[str(bins)])))
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))
    #####################################return once finished 
    for i in param.keys():
        chi[i]=nu.array(chi[i])
        param[i]=nu.array(param[i])
        ###correct metalicity and norm 
        try:
            param[i][:,range(0,3*int(i),3)]=10**param[i][:,range(0,3*int(i),3)] #metalicity conversion
            param[i][:,range(2,3*int(i),3)]=param[i][:,range(2,3*int(i),3)]*fun.norms #norm conversion
        except ValueError:
            pass
        acept_rate[i]=nu.array(acept_rate[i])
        out_sigma[i]=nu.array(out_sigma[i])
        bayes_fact[i]=nu.array(bayes_fact[i])
    q_final.put((param,chi,bayes_fact,out_sigma))
    #return param,chi,sigma,acept_rate,out_sigma

def is_send(N1,N2,N_prev): 
    #counds the number of values in a list inside of a dict
    val_N=0
    for i in N1.keys():
        val_N+=N1[i]+N2[i]-N_prev['accept'][i]-N_prev['reject'][i]
    return val_N

def Check(param,metal_unq, age_unq,bins,lib_vals=get_fitting_info(lib_path)): #checks if params are in bounds no bins!!!
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0])
    for j in xrange(bins):#check age and metalicity
        if any([metal_unq[-1],age_unq[-1]]<param[j*3:j*3+2]) or any([metal_unq[0],age_unq[0]]>param[j*3:j*3+2]):
            return True
        if not (0<param[j*3+2]):  #check normalizations
            #print 'here',j
            return True
    return False

def Chain_gen_all(means,metal_unq, age_unq,bins,sigma):
    #creates new chain for MCMC, does log spacing for metalicity
    #lin spacing for everything else, runs check on values also
    out=nu.random.multivariate_normal(means,sigma)
    '''t=Time.time()
    while Check(out,metal_unq, age_unq,bins):
        out=nu.random.multivariate_normal(means,sigma)
        if Time.time()-t>.1:
            sigma/=1.05'''
    return out

def SA(i,i_fin,T_start,T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    if i>i_fin:
        return 1.0
    else:
        return (T_stop-T_start)/float(i_fin)*i+T_start

def Covarence_mat(param,j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j-2000<0:
        return nu.cov(param[:j,:].T)
    else:
        return nu.cov(param[j-5000:j,:].T)

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
 
if __name__=='__main__':

    #profiling
    import cProfile as pro
    import cPickle as pik
    temp=pik.load(open('0.3836114.pik'))
    #data,info1,weight,dust=dust_iterp_spec(3,'norm',2000,10000)
    j='0.865598733333'
    data=temp[3][j]
    burnin,k_max,cpus=5000,16,1
    option=Value('b',True)
    option.cpu_tot=cpus
    option.iter=Value('i',True)
    option.chibest=Value('d',nu.inf)
    option.parambest=Array('d',nu.ones(k_max*3+2)+nu.nan)

    #interpolate spectra so it matches the data
    global spect
    spect=data_match_all(data)

    rank=1
    q_talk,q_final=Queue(),Queue()
    pro.runctx('rjmcmc(data,burnin,k_max,option,rank,q_talk,q_final)'
               , globals(),{'data':data,'burnin':burnin,'k_max':k_max,
                            'rank':1,'q_talk':q_talk,'q_final':q_final
                            ,'option':option}
               ,filename='agedata.Profile')
