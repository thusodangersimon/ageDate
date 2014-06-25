#!/usr/bin/env python


#from Age_date import *
#from mpi4py import MPI
import numpy as nu
from multiprocessing import *
from scipy.cluster import vq as sci
from scipy.stats import levene, f_oneway
import sys
a=nu.seterr(all='ignore')


def MCMC_multi(data, bins, burnin=5000, itter=10**5, cpus=cpu_count()):
    '''more effecent version of multi core MCMC, bins can either be a number
    or a n+1 array of lower and upper limits of each bin (n)'''
    chibest = Value('f', nu.inf)
    i = Value('i', 0)
    parambest = Array('d',nu.zeros([3*bins+2]))
    option = Value('b',True)
    option.burnin = burnin
    option.itter = int(itter + option.burnin * cpus)

    #sig_share=Array('d',nu.zeros([3*bins]))
    work = []
    q = Queue()
    #start multiprocess mcmc
    for ii in range(cpus):
        work.append(Process(target=MCMC_SA,args=(data, bins, i, chibest
                                                     , parambest, option, q)))
        work[-1].start()
    while i.value < option.itter:
        print '%2.2f percent done' %(i.value / float(option.itter) * 100)
        sys.stdout.flush()
        #print i.value
        Time.sleep(5)
    option.value = False
    #wait for proceses to finish
    count = 0
    temp = []
    while count < cpus:
        if q.qsize() > 0:
           temp.append(q.get())
           count += 1
    #post processing
    count = 0
    #outsigma={}
    #outrate,outparam,outchi={},{},{}

    outparam,outchi = nu.zeros([2,3 * bins + 2]), nu.array([nu.inf])
    for ii in temp:
        outparam = nu.concatenate((outparam,ii[0][~nu.isinf(ii[1]),:]
                                 ),axis=0)
        outchi = nu.concatenate((outchi,ii[1][~nu.isinf(ii[1])]))
        #debuging output options
        '''outsigma[str(count)]=ii[2]
        outparam[str(count)]=ii[0][~nu.isinf(ii[1]),:]
        outchi[str(count)]=ii[1][~nu.isinf(ii[1])]
        outsigma[str(count)]=nu.array(ii[2])
        outrate[str(count)]=nu.array(ii[3])'''
        
        count += 1

    return outparam[2:,:],outchi[1:]
    #return outparam,outchi,outsigma,outrate
 
def MCMC_comunicate(data, bins, itter):
    comm = MPI.COMM_WORLD
    size = comm.size                            
    myid = comm.rank   
    #comm tags
    sig_to_trans_tag,trans_tag = 0,1
    #comm notices
    prerecieve = False
    #acts a 1 chain but uses multiple feelers
    fun = MC_func(data,bins,itter)
    #change random seed for random numbers for multiprocessing
    #nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals = get_fitting_info(lib_path)
    lib_vals[0][:,0] = 10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq = nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq = nu.unique(lib_vals[0][:,1])

    #param=nu.zeros([itter+1,bins*3])
    param = []
    active_param = nu.zeros(bins * 3)
    
    bin = nu.log10(nu.linspace(10**age_unq.min(),
                               10**age_unq.max(), bins+1))
    bin_index = 0
    #start in random place
    for k in xrange(bins*3):
        if any(nu.array(range(0,bins*3,3)) == k):#metalicity
            active_param[k] = (nu.random.random() * metal_unq.ptp()
                               + metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,bins*3,3)) == k): #age
                #active_param[k]=nu.random.random() #random
                #active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index] #random in bin
                active_param[k] = nu.mean([bin[bin_index],bin[1+bin_index]]) #mean position in bin
                bin_index += 1
                #active_param[k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
            else: #norm
                active_param[k] = nu.random.random() * 1000
    #dust
    active_param[-2:] = nu.random(2) * 5.
    chi = []
    #chiappend=chi.append
    sigma = nu.identity(bins*3) * nu.tile(
        [0.5, age_unq.ptp() * nu.random.rand() * 1, 100.], bins)
    #try leastquares fit
    #active_param=fun.n_neg_lest(active_param)
    chi.append(0)
    chi[-1],active_param[range(2,bins*3,3)] = fun.func_N_norm(active_param)
    param.append(nu.copy(active_param))
    #set up shared varibles
    (current_iter, acpt_rate, chi_best, 
     param_best, turn_iter) = 0, 0.5, nu.copy(chi[-1]), nu.copy(param[-1]), 500
    if myid == 0:
        myturn = [True,1,0] #[if turn, rank of next person,number times i control stuff]
        
    else:
        myturn = [False,myid+1,0]
        if myturn[1] > size - 1: #make circular
            myturn[1] = 0
    #start MCMC
    #Naccept,Nrecjet=0,0
    while current_iter < itter:
        #print "hi, I'm %i at itter %i and chi %f" %(current_process().ident,j,chi[j-1])
        active_param = fun.New_chain(active_param,sigma,'norm')
        chi.append(0)
        if current_iter < 1000:
            chi[-1],active_param = fun.SA(chi[-2], active_param,param[-1])
        else:
            chi[-1],active_param = fun.Mh_criteria(chi[-2],
                                                   active_param, param[-1])
        #check for best fit
        if chi[-1] < chi_best:
            
            chi_best = nu.copy(chi[-1])
            param_best = nu.copy(active_param)
        param.append(nu.copy(active_param))
        current_iter += 1
        #if my turn then control sigma
        if myturn[0]:
            if current_iter > 900 and current_iter % 100 == 0:
                print nu.min(chi), chi[-1]
                sigma = fun.Step(sigma,param,'cov')
            elif current_iter < 900:
                print nu.min(chi), chi[-1]
                sigma = fun.Step(sigma, param, 'adapt')

            #decide if should send to next processor
            if myturn[2] > turn_iter: #send
                comm.isend([True,MPI.LOGICAL],dest=myturn[1],
                           tag=sig_to_trans_tag)
                #send last iterations, current sigma, accept rate, best fit and param
                #acpt_rate=fun.Naccept/(fun.Naccept+fun.Nreject)
                #comm.send((param[-turn_iter:],sigma,rate,
                break
            else:
                myturn[2] += 1
                comm.Irecv([prerecieve,MPI.LOGICAL], 
                           source=MPI.ANY_SOURCE, tag=sig_to_trans_tag)
                if prerecieve: #about te get update
                    pass

    return outprep(nu.array(param)), nu.array(chi)
   
def MCMC_SA(data, burnin, k_max, option, rank, q_talk, q_final, fun):
    #does MCMC and reduices the false acceptance rate over a threshold
    #itter needs to be a array of normaly distrbuted numbers
    #so there are no problems with multiprocessing
    print "Starting processor %i" %rank
    #part on every modual wanting to fit the spectra
    #controls input and expot of files for fitt
    nu.random.seed(random_permute(current_process().ident))
    #fun = MC_func(data,bins)
    #cpu = float(cpu_count())
    #initalize parmeters and chi squared
    lib_vals = fun._lib_vals
    metal_unq = fun._metal_unq
    age_unq = fun._age_unq
    bins = fun._bins
    #param set up
    non_N_index=nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()
    param = []
    active_param = nu.zeros(3*fun._bins)
    active_dust = nu.random.rand(2) * 4.

    #if no binning is defined
    if not nu.any(fun.bin):
       fun.bin = nu.linspace(age_unq.min(),age_unq.max(),bins+1) #log space
    bin_index = 0
    #start in random place
    for k in xrange(len(active_param)):
        if any(nu.array(range(0,len(active_param),3)) == k):#metalicity
            active_param[k] = (nu.random.random() * metal_unq.ptp() + 
                               metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,len(active_param),3)) == k): #age
                #random in bin
                active_param[k] = (nu.random.random() * age_unq.ptp()
                                   / float(bins) + fun.bin[bin_index])
                bin_index += 1

    chi = [nu.inf]
    sigma = nu.identity(len(active_param)) * nu.tile(
        [0.5, age_unq.ptp()*.1 * nu.random.rand(), 1.], bins)
    sigma_dust = nu.identity(2) * nu.random.rand() * .4
    #try leastquares fit
    #active_param=fun.n_neg_lest(active_param)
    chi[0],active_param[range(2,bins*3,3)] = fun.lik(active_param,
                                                             active_dust)
    param.append(nu.copy(nu.hstack((active_param, active_dust))))
    #set global best fit
    if option.chibest.value > chi[0]:
        option.chibest.value = nu.copy(chi[0])
        print (('best fit value %f in iteration %i,' +
                       'from processor %i') %(chi[0],0,rank))
        for k in range(len(param[-1])):
            option.parambest[k] = nu.copy(param[-1][k])
    #start MCMC
    Nacept,Nreject,Nexchange_ratio,T_cuurent = 1.0, 1.0, 1.0, 0.
    acept_rate, out_sigma = [], []
    j, T = 1, 9.
    T_start, T_stop = nu.copy(chi[0]), 0.9
    while option.value:
        if j % 1000 == 0:
            print "hi, I'm %i at itter %i and chi %f" %(rank, j, chi[j-1])    
            sys.stdout.flush()
        #draw new param from proposal
        #has low acceptance, this will speed it up
        chi.append(nu.inf)
        for ii in xrange(100):
            temp = (fun.proposal(active_param, sigma),
                    fun.proposal(active_dust, sigma_dust))
            chi[j], N = fun.lik(temp[0],temp[1])  
            if nu.isfinite(chi[j]):
                active_param, active_dust = temp
                active_param[range(2,bins*3,3)] = N
                break
        #active_param = fun.proposal(active_param, sigma)
        #active_dust = fun.proposal(active_dust, sigma_dust)
        #get -2log likeliyhood
        #chi.append(nu.inf)
        #chi[j], active_param[range(2,bins*3,3)] = fun.lik(active_param,
        #                                                  active_dust)
        #decide to accept or not
        a = nu.exp((chi[j-1] - chi[j]) / 
                   SA(T_cuurent, burnin, T_start, T_stop))
        #metropolis hastings
        if a >= nu.random.rand(): #acepted and false accept
            param.append( nu.copy(nu.hstack((active_param, active_dust))))
            Nacept += 1
            if T_start > T_stop:
                T_start = nu.min(chi)
            else:
                T_start = T_stop + .0
            if option.chibest.value > chi[-1]:
                option.chibest.value = nu.copy(chi[-1])
                print (('best fit value %f in iteration %i,' +
                        'from processor %i') %(chi[-1],j,rank))
                for k in range(len(param[-1])):
                    option.parambest[k] = nu.copy(param[-1][k])
                
        else: #reject
            param.append(nu.copy(param[-1]))
            active_param = nu.copy(param[-1][:-2])
            active_dust = nu.copy(param[-1][-2:])
            chi[-1] = nu.copy(chi[-2])
            Nreject += 1
 
        #if j<1000: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
        if Nacept / (Nacept + Nreject) > .50: 
                #too few aceptnce decrease sigma
            sigma *= 1.05
            sigma_dust *= 1.05
        elif (Nacept / (Nacept + Nreject)<.25):
            sigma /= 1.05
            sigma_dust /= 1.05
        #use cov matrix
        if j % 500 == 0: 
            temp = nu.array(param[-1000:])
            sigma_dust = Covarence_mat(temp[:,-2:], j)
            #incase chain it stuck for 1000 iterations
            if sigma_dust.size == 0 or nu.any(nu.isnan(sigma_dust)):
                sigma_dust = nu.copy(out_sigma[-1])
                j += 1
                continue
            sigma = Covarence_mat(temp[:,:-2], j)
            
        #change temperature
        if nu.min([1,nu.exp(-(chi[-2] - chi[-1]) / (2. *
                            SA(T_cuurent + 1,burnin,T_start,T_stop))
                             - (chi[j-1]+chi[j]) / 
                             (2.*SA(T_cuurent,burnin,T_start,T_stop)))
                   / T])>nu.random.rand():
            if j > burnin and T_cuurent !=burnin:
                T_cuurent = burnin
                
                active_param = nu.array(option.parambest)[:-2]
                active_dust = nu.array(option.parambest)[-2:]
            else:
                T_cuurent += 1
            Nexchange_ratio += 1   
        #make sure the change temp rate is aroudn 2%
        if Nexchange_ratio / (Nacept+Nreject) > .02:
            T = T * 1.05
        elif Nexchange_ratio / (Nacept+Nreject) < .005:
            T = T / 1.05
        #change temperature schedual                  
        j += 1
        option.iter.value += 1
        acept_rate.append(nu.copy(Nacept / (Nacept + Nreject)))
        out_sigma.append(nu.copy(sigma_dust))
    #return once finished 
    param = outprep(nu.array(param[burnin:]),fun)
    q_final.put((param, nu.array(chi[burnin:])))
    #q.put((param,chi))
    #q.put((param,chi,out_sigma,acept_rate))
    #return param,chi,out_sigma,acept_rate


def SA(i, i_fin, T_start, T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    if i > i_fin:
        return 1.0
    else:
        return (T_stop - T_start) / float(i_fin) * i + T_start

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


def Covarence_mat(param, j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j - 2000 < 0:
        return nu.cov(param[:j,:].T)
    else:
        return nu.cov(param[j - 2000:j,:].T)

def outprep(param, func):
    #changes metals from log to normal
    bins = (param.shape[1] - 2) / 3
    for i in range(0,param.shape[1] - 2, 3):
        param[:,i] = 10**param[:,i]
        #param[:,i+2] *= func.norms
        
    return param

def autocorr(Y, k=1):
    """
    Computes the sample autocorrelation function coeffficient rho
    for given lag k
def xcorr(x):
  FFT based autocorrelation function, which is faster than numpy.correlate
  # x is supposed to be an array of sequences, of shape (totalelements, length)
  """
    fftx = nu.fft.fft(Y, n=(len(Y)*2-1), axis=0)
    ret = nu.fft.ifft(fftx * nu.conjugate(fftx), axis=0)
    ret = nu.fft.fftshift(ret, axes=0)
    return nu.real(ret)

def np_data(temp,bins):
    count = 0
    outparam,outchi = nu.zeros([2,3 * bins + 2]), nu.array([nu.inf])
    for ii in temp:
        outparam = nu.concatenate((outparam,ii[0][~nu.isinf(ii[1]),:]
                                 ),axis=0)
        outchi = nu.concatenate((outchi,ii[1][~nu.isinf(ii[1])]))        
        count += 1
    return outparam[2:,:],outchi[1:]


if __name__=='__main__':

    '''import cProfile as pro
    data,info,weight=create_spectra(2)
    bins=2
    chibest_global=Value('f', nu.inf)
    i=Value('i', 0)
    parambest=Array('d',nu.zeros([3*bins]))
    option=Value('b',True)
    option.itter=5000
    pro.runctx('MCMC_SA(data,bins,i,chibest,parambest,option)'
               , globals(),{'data':data,'bins':bins,'i':i,
                            'chibest':chibest_global,'parambest':parambest
                            ,'option':option}
               ,filename='agedata.Profile')
               '''
    
