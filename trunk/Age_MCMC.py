#!/usr/bin/env python


from Age_date import *
from mpi4py import MPI
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
        work.append(Process(target=MCMC_SA,args=(data,bins,i,chibest
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
    #outsigma={}
    #outrate,outparam,outchi={},{},{}

    outparam,outchi=nu.zeros([2,3*bins]),nu.array([nu.inf])
    for ii in temp:
        outparam=nu.concatenate((outparam,ii[0][~nu.isinf(ii[1]),:]
                                 ),axis=0)
        outchi=nu.concatenate((outchi,ii[1][~nu.isinf(ii[1])]))
        '''outsigma[str(count)]=ii[2]
        outparam[str(count)]=ii[0][~nu.isinf(ii[1]),:]
        outchi[str(count)]=ii[1][~nu.isinf(ii[1])]
        outsigma[str(count)]=nu.array(ii[2])
        outrate[str(count)]=nu.array(ii[3])'''
        
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
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if len(parambest)-bins-1<k: #normilization
                active_param[k]=10*nu.random.random()
            else: #age
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
    
    #active_param[2]=1

    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter)+nu.inf
    sigma=nu.identity(len(active_param))*nu.concatenate((nu.tile(
                [metal_unq.ptp()*.1,age_unq.ptp()/bins*.1],bins),
                          nu.array([nu.sqrt(bins)]*bins)))


    model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
    model=data_match(model,data)
    #make weight paramer start closer to where ave data value
    #for j in range(bins):
    #    active_param[2+j]=normalize(data,model)*nu.random.random()
    chi[0]=sum((data[:,1]-model)**2)+nu.abs(data[:,1]-model).max()
    
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject,Nfalseaccept=1.0,1.0,1.
    acept_rate,out_sigma=[],[]
    j=1
    while option.value and i.value<option.itter:
        #for k in xrange(len(active_param)):
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
      #calculate new model and chi
        model=get_model_fit(active_param,lib_vals,age_unq,metal_unq,bins)
        model=data_match(model,data)
        #active_param[2]=normalize(data,model)
        chi[j]=sum((data[:,1]-model)**2)+nu.abs(data[:,1]-model).max()
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2.0)
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if j>5 and chi[j]< chi[:j-1].min():
                print 'best fit value %f in iteration %i' %(chi[j],j)
                sys.stdout.flush()
            if .01>nu.random.rand():
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
               
        else:
            if a>nu.random.rand():#false accept
                param[j,:]=nu.copy(active_param)
                Nfalseaccept+=1
            else:
                param[j,:]=nu.copy( param[j-1,:])
                active_param=nu.copy( param[j-1,:])
                chi[j]=nu.copy(chi[j-1])
                Nreject+=1
 
        if j<1000: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if Nacept/Nreject<.50: #and all(sigma.diagonal()>=10**-6): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif Nacept/Nreject>.25:# and all(sigma.diagonal()[:2]<10): #not enough
                sigma=sigma*1.05
        else: #use covarnence matrix
            if j%500==0:
                sigma=Covarence_mat(param,j)
            
        #print Nacept/Nreject
        j+=1
        i.value=i.value+1
        acept_rate.append(nu.copy(Nacept/Nfalseaccept))
        out_sigma.append(nu.copy(sigma.diagonal()))
        #change positions to best param
        if .001>nu.random.rand(): #every hundred itterations
            a=nu.exp((chi[j]-chibest.value)/2.0)
            if a>1: #accept change in param
                #print 'here'
                for k in range(len(active_param)): 
                    param[j,k]=nu.copy(parambest[k])
    #return once finished 
    param=outprep(param)
    q.put((param[option.burnin:,:],chi[option.burnin:] ))
    #q.put((param,chi,out_sigma,acept_rate))


def MCMC_SA(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC and reduices the false acceptance rate over a threshold
    #itter needs to be a array of normaly distrbuted numbers
    #so there are no problems with multiprocessing

    #part on every modual wanting to fit the spectra
    #controls input and expot of files for fitt
    data[:,1]=data[:,1]*1000.  
    cpu=float(cpu_count())
    #change random seed for random numbers for multiprocessing
    nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    param=nu.zeros([option.itter+1,len(parambest)])
    active_param=nu.zeros(len(parambest))
    
    bin=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    bin_index=0
    #start in random place
    for k in xrange(len(parambest)):
        if any(nu.array(range(0,len(parambest),3))==k):#metalicity
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,len(parambest),3))==k): #age
                #active_param[k]=nu.random.random()
                active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                bin_index+=1
            else: #norm
                #active_param[k]=nu.random.random()
                pass

    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter+1)+nu.inf
    sigma=nu.identity(len(active_param))*nu.concatenate((nu.tile(
                [metal_unq.ptp()*nu.random.rand(),age_unq.ptp()/bins*nu.random.rand()],bins),
                          nu.array([nu.sqrt(bins)]*bins)))


    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    N,model=N_normalize(data,model,bins)
    ii=0
    for k in range(2,len(parambest),3):
        active_param[k]=nu.log10(N[ii])
        ii+=1
    #make weight paramer start closer to where ave data value
    chi[0]=sum((data[:,1]-model)**2)
    chibest.value=chi[0]
    for k in range(len(active_param)):
        parambest[k]=nu.copy(active_param[k])
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject,Nexchange_ratio,T_cuurent=1.0,1.0,1.0,0.
    acept_rate,out_sigma=[],[]
    j,T=1,37450725549.
    while option.value and i.value<option.itter:
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
       #calculate new model and chi
        model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
        N,model=N_normalize(data,model,bins)
        ii=0
        for k in range(2,len(parambest),3):
            active_param[k]=nu.log10(N[ii])
            ii+=1
        chi[j]=sum((data[:,1]-model)**2)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2)
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if j>5 and chi[j]< chi[:j-1].min():
                print 'best fit value %f in iteration %i' %(chi[j],j)
                sys.stdout.flush()
            if .01>nu.random.rand():
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
        else:
            if nu.exp(nu.log(a)/SA(T_cuurent,option.itter/(cpu),1.1,.11))>nu.random.rand():#false accept
                param[j,:]=nu.copy(active_param)
                Nacept+=1
            else:
                param[j,:]=nu.copy( param[j-1,:])
                active_param=nu.copy( param[j-1,:])
                chi[j]=nu.copy(chi[j-1])
                Nreject+=1
 
        if j<1000: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if Nacept/Nreject<.50 and all(sigma.diagonal()[:2]>=10**-6): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif Nacept/Nreject>.25 and all(sigma.diagonal()[:2]<10): #not enough
                sigma=sigma*1.05
        else: #use covarnence matrix
            if j%500==0:
                sigma=Covarence_mat(param,j)
        #change temperature
        if nu.min([1,nu.exp(-(chi[j-1]-chi[j])/(2.*SA(T_cuurent+1,option.itter/(cpu),1.1,.11))-(chi[j-1]+chi[j])/(2.*SA(T_cuurent,option.itter/(cpu),1.1,.11)))/T])>nu.random.rand():
            T_cuurent+=1
            Nexchange_ratio+=1   
        #make sure the change temp rate is aroudn 2%
        if Nexchange_ratio/(Nacept+Nreject)>.02:
            T=T*1.05
        elif Nexchange_ratio/(Nacept+Nreject)<.005:
            T=T/1.05

        '''if .01>nu.random.rand(): #every hundred itterations
            a=nu.exp((chi[j]-chibest.value)/2.0)
            if a>1: #accept change in param
                #print j
                chi[j]=nu.copy(chibest.value)
                for k in range(len(active_param)): 
                    param[j,k]=nu.copy(parambest[k])
                    active_param[k]=nu.copy(parambest[k])
                    '''
        j+=1
        i.value=i.value+1
        acept_rate.append(nu.copy(Nacept/(Nacept+Nreject)))
        out_sigma.append(nu.copy(sigma))
    #return once finished 
    param=outprep(param)
    for k in range(2,len(parambest),3):
        param[:,k]=param[:,k]/1000.
    data[:,1]=data[:,1]/1000.
    q.put((param[option.burnin:,:],chi[option.burnin:]))
    #q.put((param,chi,out_sigma,acept_rate))
    #return param,chi,out_sigma,acept_rate

def SA(i,i_fin,T_start,T_stop):
    #temperature parameter for Simulated anneling (SA). 
    #reduices false acceptance rate if a<60% as a function on acceptance rate
    m=(T_start-T_stop)/(0.98*i_fin)
    b=T_start-m
    return m*i+b



def Covarence_mat(param,j):
    #creates a covarence matrix for the step size 
    #only takes cov of last 1000 itterations
    if j-2000<0:
        return nu.cov(param[:j,:].T)
    else:
        return nu.cov(param[j-2000:j,:].T)

def outprep(param):
    #changes metals from log to normal
    for i in range(0,param.shape[1],3):
        param[:,i]=10**param[:,i]
        param[:,i+2]=10**param[:,i+2]
        
    return param



if __name__=='__main__':
    import cProfile as pro
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
