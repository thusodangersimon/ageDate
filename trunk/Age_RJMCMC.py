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
a=nu.seterr(all='ignore')



def RJ_multi(data,burnin,k_max=16,cpus=cpu_count()):
    #does multiple chains for RJMCMC
    #shares info about chain every 300 itterations to other chains
    #import cPickle as pik
    option=Value('b',True)
    #option.burnin=burnin
    #option.itter=int(itter+option.burnin)
    #option.send_num=Value('d',0) #tells who is sending and who is reciving new data
    option.cpu_tot=cpus

    work=[]
    q_talk,q_final=Queue(),Queue()
    #start multiprocess mcmc
    for ii in range(cpus):
        work.append(Process(target=rjmcmc,args=(data,burnin,k_max,option,ii,q_talk,q_final)))
        work[-1].start()

    rank,size,conver_test=-nu.ones(cpus),nu.zeros(cpus),[]
    while True: 
        if q_final.qsize()>=cpus:
            conver_test=[]
            print 'Starting convergence test'
            for i in range(cpus):
                rank[i],size[i],temp=q_final.get()
                conver_test.append(temp)
            #calculate W and B for convergence for all bins with len>1000
            key_to_use=conver_test[0].keys()
            for i in conver_test:
                for j in key_to_use:
                    if len(i[j])<1000:
                        key_to_use.remove(j)
            if key_to_use:
                #do convergence calc
                if Convergence_tests(conver_test,key_to_use):
                    #convergence!
                    #tidy up and end
                    sys.stdout.flush()
                    while q_final.qsize()>0:
                        q_final.get()
                    option.value=False
                    break         
                else: #doesn't seem like convergence works use other methods to stop chain
                    continue
                    for j in key_to_use:
                        temp=0
                        for i in conver_test:
                            temp+=len(i[j])
                        if temp>10**5:
                            sys.stdout.flush()
                            while q_final.qsize()>0:
                                q_final.get()
                            #option.value=False
                            break   
                    if temp>10**5:
                        break

        else:
            Time.sleep(5)
    '''t=Time.time()
    while t+600>Time.time():
        Time.sleep(5)
        print '%i seconds left' %(round(t+600-Time.time()))
        sys.stdout.flush()
    #print 'Starting 5 min wait'
    #Time.sleep(300)'''
    while q_final.qsize()>0: #clear final queue
        try:
            a= q_final.get(timeout=1)
        except:
            break
    option.value=False
    #wait for proceses to finish
    count=0
    temp=[]
    while count<cpus:
        if q_final.qsize()>0:
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
            #bayes_fac[i][bayes_fac[i]>1]=1. #accept critera is min(1,alpha)
            fac.append([int(i),nu.mean(nu.nan_to_num(bayes_fac[i])),len(bayes_fac[i])])
            #remove 1st bin for now#############
    fac=nu.array(fac)
    '''
    fac=fac[nu.nonzero(fac[:,0]!=1)[0],:]
    if all(fac[fac[:,1].argmax(),1]/fac[nu.arange(4)!=2,1]>3.):
        print 'substantal best fit is with %i bins' %fac[fac[:,1].argmax(),0]
        bins=str(int(ffac[fac[:,1].argmax(),0]))
    elif all(fac[fac[:,1].argmax(),1]/fac[nu.arange(4)!=2,1]>1.):
        print 'sort of best fit is with %i bins' %fac[fac[:,1].argmax(),0]
        bins=str(int(fac[fac[:,1].argmax(),0]))
    else:
        print 'No model is the best choosing longest chain'
        bins=''
        '''
    #grab chains with best fit and chech to see if mixed properly
    outparam,outchi={},{}
    for i in fac[:,0]:
        outparam[str(int(i))],outchi[str(int(i))]=nu.zeros([2,3*i]),nu.array([nu.inf])
    for i in temp:
        for j in fac[:,0]:
            try:
                outparam[str(int(j))]=nu.concatenate((outparam[str(int(j))],i[0][str(int(j))][~nu.isinf(i[1][str(int(j))][1:]),:]),axis=0)
                outchi[str(int(j))]=nu.concatenate((outchi[str(int(j))],i[1][str(int(j))][~nu.isinf(i[1][str(int(j))])]))
            except ValueError: #if empty skip
                pass
    for j in nu.int64(fac[:,0]): #post processing
        outparam[str(int(j))],outchi[str(int(j))]=outparam[str(int(j))][2:,:],outchi[str(int(j))][1:]
        outparam[str(int(j))][:,range(0,3*j,3)]=10**outparam[str(int(j))][:,range(0,3*j,3)]
        outparam[str(int(j))][:,range(2,3*j,3)]=outparam[str(int(j))][:,range(2,3*j,3)]
    
    return outparam,outchi,fac[fac[:,0].argsort(),:]


def rjmcmc(data,burnin=5*10**3,k_max=16,option=True,rank=0,q_talk=None,q_final=None):
    #parallel worker program reverse jump mcmc program
    nu.random.seed(current_process().pid)
    #initalize boundaries
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    #data[:,1]=data[:,1]*1000  
    #create fun for all number of bins
    attempt=False
    fun,param,active_param,chi,sigma={},{},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    bayes_fact={} #to calculate bayes factor
    fun=RJMC_func(data)
    for i in range(1,k_max+1):
        param[str(i)]=[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,age_unq.ptp()*nu.random.rand(),1.],i)
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
        bayes_fact[str(i)]=[]
    #for parallel to know how many iterations have gone by
    #N_all={'accept':dict(Nacept),'reject':dict(Nreject)}
    #bins to start with
    bins=nu.random.randint(1,k_max)
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
    chi[str(bins)].append(0.)
    #active_param[str(bins)]=fun[str(bins)].n_neg_lest(active_param[str(bins)])
    chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.func_N_norm(active_param[str(bins)],bins)
    param[str(bins)].append(nu.copy(active_param[str(bins)]))
    #parambest=nu.copy(active_param)

    mybest=nu.copy([chi[str(bins)][0],bins])
    parambest=nu.copy(active_param[str(bins)])

    #start rjMCMC
    T_cuurent,Nexchange_ratio=1.0,1.0
    size=0
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    while option.value:
        if size%500==0:
            print "hi, I'm at itter %i, chi %f from %s bins and for cpu %i" %(len(param[str(bins)]),chi[str(bins)][-1],bins,rank)
            sys.stdout.flush()
            #print sigma[str(bins)].diagonal()
            #print 'Acceptance %i reject %i' %(Nacept,Nreject)
            #print active_param[str(bins)][range(2,bins*3,3)]
        if q_talk.qsize()==0:
            active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
        else:
            try:
                temp=q_talk.get(timeout=1)
                if mybest[0]>temp[1] and bins==temp[0]:
                    mybest=[temp[1]+0,temp[0]+0]
                    active_param[str(bins)]=temp[2]+0
                    q_talk.put(temp)
                elif  bins!=temp[0] and mybest[0]>temp[1]: #only accept best move is bins are the same
                    q_talk.put(temp)
                    mybest=[temp[1]+0,temp[0]+0]
                    active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
                else:
                    active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
            except:
                active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
        #bin_index=0
        #calculate new model and chi
        chi[str(bins)].append(0.)
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.func_N_norm(active_param[str(bins)],bins)
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
        a=nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/SA(T_cuurent,burnin,T_start,T_stop))
        #metropolis hastings
        if a>nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(active_param[str(bins)]))
            Nacept[str(bins)]+=1
            if not nu.isinf(min(chi[str(bins)])): #put temperature on order of chi calue
                T_start=nu.round(min(chi[str(bins)]))+1.
            #see if global best fit
            if chi[str(bins)][-1]< mybest[0]:
                mybest=nu.copy([chi[str(bins)][-1],bins])
                parambest=nu.copy(active_param[str(bins)]) 
                print '%i has best fit with chi of %2.2f and %i bins' %(rank,mybest[0],mybest[1])
                #send to other params
                q_talk.put((bins,chi[str(bins)][-1],active_param[str(bins)]))
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)]=nu.copy(param[str(bins)][-1])
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1

        ###########################step stuff
        if len(param[str(bins)])<6000: #change sigma with acceptance rate
            if  (acept_rate[str(bins)][-1]<.234 and 
                                        all(sigma[str(bins)].diagonal()[nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()]<5.19)):
               #too few aceptnce decrease sigma
                sigma[str(bins)]=sigma[str(bins)]/1.05
            elif acept_rate[str(bins)][-1]>.40 and all(sigma[str(bins)].diagonal()>=10**-5): #not enough
                sigma[str(bins)]=sigma[str(bins)]*1.05
            # else: #use covarnence matrix
            if j%100==0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
                sigma[str(bins)]=Covarence_mat(nu.array(param[str(bins)]),len(param[str(bins)]))
                #active_param=fun.n_neg_lest(active_param)
        if nu.all(sigma['2'].diagonal()==0): #if step is 0 make small
            for k in xrange(sigma[str(bins)].shape[0]):
                sigma[str(bins)][k,k]=10.**-5

        #############################decide if birth or death
        if (birth_rate>nu.random.rand() and bins<k_max and j>j_timeleft ) or bins==1:
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
        else:
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
            tchi,active_param[str(temp_bins)][range(2,temp_bins*3,3)]=fun.func_N_norm(active_param[str(temp_bins)],temp_bins)
            bayes_fact[str(bins)].append(nu.exp((chi[str(bins)][-1]-tchi)/2.)*birth_rate*critera) #save acceptance critera for later
            #rjmcmc acceptance critera ##############
            if bayes_fact[str(bins)][-1]>nu.random.rand():
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
                param[str(bins)].append(nu.copy(active_param[str(bins)]))
                j,j_timeleft=0,nu.random.exponential(200)

        #########################################change temperature
        if nu.min([1,nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/(2.*SA(T_cuurent+1,burnin,T_start,T_stop))-(chi[str(bins)][-2]+chi[str(bins)][-1])/(2.*SA(T_cuurent,burnin,T_start,T_stop)))/T])>nu.random.rand():
            if T_cuurent<burnin:
                T_cuurent+=1
                #print T_cuurent,rank
            elif T_cuurent==round(burnin):
                print 'done with cooling'
                T_cuurent+=1
            Nexchange_ratio+=1   
        #make sure the change temp rate is aroudn 2%
        if Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))>.02:
            T=T*1.05
        elif Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))<.005:
            T=T/1.05
        #change temperature schedual
        #keep on order with chi squared
        '''if j%20==0:
            if acept_rate[str(bins)][-1]>.5 and T_start<10**-5:
                T_start/=2.
                #T_stop+=.1
            elif acept_rate[str(bins)][-1]<.25 and T_start<3*10**5:
                T_start*=2.
                #T_stop-=.1'''
        ##############################convergece assment
        size=dict_size(param)
        if size%999==0 and size>6000:
            q_final.put((rank,size,param))
                
        ##############################house keeping
        j+=1
        acept_rate[str(bins)].append(nu.copy(Nacept[str(bins)]/(Nacept[str(bins)]+Nreject[str(bins)])))
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))
    #####################################return once finished 
    for i in param.keys():
        chi[i]=nu.array(chi[i])
        param[i]=nu.array(param[i])
        acept_rate[i]=nu.array(acept_rate[i])
        out_sigma[i]=nu.array(out_sigma[i])
        bayes_fact[i]=nu.array(bayes_fact[i])
    #data[:,1]=data[:,1]/1000.
    q_final.put((param,chi,bayes_fact,out_sigma))
    #return param,chi,sigma,acept_rate,out_sigma

def is_send(N1,N2,N_prev): 
    #counds the number of values in a list inside of a dict
    val_N=0
    for i in N1.keys():
        val_N+=N1[i]+N2[i]-N_prev['accept'][i]-N_prev['reject'][i]
    return val_N

def Check(param,metal_unq, age_unq,bins): #checks if params are in bounds no bins!!!
    #age=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))#linear spacing
    for j in xrange(bins):#check age and metalicity
        if any([metal_unq[-1],age_unq[-1]]<param[j*3:j*3+2]) or any([metal_unq[0],age_unq[0]]>param[j*3:j*3+2]):
            return True
        #if any(nu.abs(nu.diff(param.take(range(1,bins*3,3))))<.3):
        #    return True
        if not (0<param[j*3+2]): #and param[j*3+2]<1): #check normalizations
            return True
    return False

def Chain_gen_all(means,metal_unq, age_unq,bins,sigma):
    #creates new chain for MCMC, does log spacing for metalicity
    #lin spacing for everything else, runs check on values also
    out=nu.random.multivariate_normal(means,sigma)
    t=Time.time()
    while Check(out,metal_unq, age_unq,bins):
        out=nu.random.multivariate_normal(means,sigma)
        if Time.time()-t>.5:
            sigma=sigma/1.05
            if Time.time()-t>2.:
                print "I'm %i and I'm stuck" %current_process().ident
                print means,sigma
                #raise

    return out

def SA(i,i_fin,T_start,T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    if i>.02*i_fin:
        return 1.0
    else:
        return (T_stop-T_start)/float(.02*i_fin)*i+T_start

def Covarence_mat(param,j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j-2000<0:
        return nu.cov(param[:j,:].T)
    else:
        return nu.cov(param[j-2000:j,:].T)

def Convergence_tests(param,keys,n=1000):
    #uses Levene's test to see if var between chains are the same if that is True
    #uses f_oneway (ANOVA) to see if means are from same distrubution 
    #if both are true then tells program to exit
    #uses last n chains only
    for i in param:
        for j in keys:
            i[j]=nu.array(i[j])
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


def plot_model(param,data,bins):
    import pylab as lab
    #takes parameters and returns spectra associated with it
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0])
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #check to see if metalicity is in log range (sort of)
    if any(param[range(0,bins*3,3)]>metal_unq.max() or param[range(0,bins*3,3)]<metal_unq.min()):
        print 'taking log of metalicity'
        param[range(0,bins*3,3)]=nu.log10(param[range(0,bins*3,3)])
    model=get_model_fit_opt(param,lib_vals,age_unq,metal_unq,bins)  
    #out= model['0']*.0
    model=data_match_new(data,model,bins)
    index=xrange(2,bins*3,3)
    for ii in model.keys():
        model[ii]=model[ii]*param[index[int(ii)]]
    index=nu.int64(model.keys())
    out=nu.sum(nu.array(model.values()).T,1)
    lab.plot(data[:,0],out)
    return nu.vstack((data[:,0],out)).T
#return nu.vstack((model['wave'],out)).T



#####classes############# 
class RJMC_func:
    #makes more like function, so input params and the chi is outputted
    def __init__(self,data,spect=spect):
        data_match_all(data)
        self.data=data
        lib_vals=get_fitting_info(lib_path)
        lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
        metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
        age_unq=nu.unique(lib_vals[0][:,1])
        self.lib_vals=lib_vals
        self.age_unq= age_unq
        self.metal_unq,self.spect=metal_unq,spect
        #self.bins=bins

    def func(self,param,bins):
        if len(param)!=bins*3:
            return nu.nan
        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,bins)  
        
        model=data_match_new(data,model,bins)
        out= model['0']*.0
        index=xrange(2,bins*3,3)
        
        for ii in model.keys():
            if ii!='wave':
                out+=model[ii]*param[index[int(ii)]]
        return nu.sum((self.data[:,1]-out)**2)

    def func_N_norm(self,param,bins):
        #returns chi and N norm best fit params
        if len(param)!=bins*3:
            return nu.nan
        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,bins)  
        N,model,chi=self.N_normalize(model,bins)
    
        return chi,N

    def normalize(self,model):
        #normalizes the model spectra so it is closest to the data
        if self.data.shape[1]==2:
            return nu.sum(self.data[:,1]*model)/nu.sum(model**2)
        elif self.data.shape[1]==3:
            return nu.sum(self.data[:,1]*model/data[:,2]**2)/nu.sum((model/self.data[:,2])**2)
        else:
            print 'wrong data shape'
            raise(KeyError)

    def N_normalize(self,model,bins):
        #takes the norm for combined data and does a minimization for best fits value
    
        #match data axis with model
        model=data_match_new(self.data,model,bins)
    #do non-negitave least squares fit
        if bins==1:
            N=[normalize(self.data,model['0'])]
            return N, N[0]*model['0'],nu.sum((self.data[:,1]-N[0]*model['0'])**2)
        try:
            if self.data.shape[1]==2:
                N,chi=nnls(nu.array(model.values()).T[:,nu.argsort(nu.int64(nu.array(model.keys())))],self.data[:,1])
            elif self.data.shape[1]==3:
                N,chi=nnls(nu.array(model.values()).T[:,nu.argsort(nu.int64(nu.array(model.keys())))]/nu.tile(self.data[:,2],(bins,1)).T,self.data[:,1]/self.data[:,2]) #from P Dosesquelles1, T M H Ha1, A Korichi1, F Le Blanc2 and C M Petrache 2009
            else:
                print 'wrong data shape'
                raise(KeyError)

        except RuntimeError:
            print "nnls error"
            N=nu.zeros([len(model.keys())])
            chi=nu.inf

        index=nu.nonzero(N==0)[0]
        N[index]+=10**-6
        index=nu.int64(model.keys())
        return N,nu.sum(nu.array(model.values()).T*N[index],1),chi**2
