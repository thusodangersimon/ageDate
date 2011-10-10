#!/usr/bin/env python


from Age_date import *
from mpi4py import MPI
#import time as Time

a=nu.seterr(all='ignore')

def test():
    comm = MPI.COMM_WORLD
#size = comm.Get_size()
    rank = comm.Get_rank()
    print 'i am %i' %rank

def PMC(data,bins,n_dist=2,pop_num=10**4):
    #uses population monte carlo to find best fits and prosteror
    data[:,1]=data[:,1]*1000.  
    cpu=float(cpu_count())
    
   #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])
    #initalize importance functions
    alpha=nu.array([n_dist**-1.]*n_dist) #[U,N]
    #for multivarate dist params
    mu=nu.zeros([n_dist,bins*3])
    age_bins=nu.linspace(age_unq.min(),age_unq.max(),bins+1)
    for jj in range(n_dist):
        bin_index=0
        for k in xrange(mu.shape[1]):
            if any(nu.array(range(0,mu.shape[1],3))==k):#metalicity
                mu[jj,k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
            else:#age and normilization
                if any(nu.array(range(1,mu.shape[1],3))==k): #age
                #mu[k]=nu.random.random()
                    mu[jj,k]=nu.random.rand()*age_unq.ptp()/float(bins)+age_bins[bin_index]
               # mu[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                    bin_index+=1
                else: #norm
                    mu[jj,k]=nu.random.random()
    sigma=nu.array([nu.identity(bins*3)*nu.random.rand()/bins]*n_dist)
    #build population parameters
    points=pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins)
    #get likelihoods
    lik=[]
    pool=Pool()
    for ii in points:
        pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
    pool.close()
    pool.join()
    lik=nu.array(lik)
    #calculate weights
    for j in lik:
        j[-1]=nu.exp(-j[-1])/(alpha[0]*uni_func(age_unq,metal_unq,bins)+alpha[1]*norm_func(j[:-1],mu,sigma))
    weight_norm=lik[:,-1]/sum(lik[:,-1])
    #resample
    
    #start loop
    norm_start=int(pop_num*alpha[0])
    for i in range(10):
    #calculate new alpha, mu and sigma values
        temp_phi_norm=nu.zeros(pop_num-norm_start)
        for j in xrange(pop_num-norm_start):
            temp_phi_norm[j]=norm_func(lik[norm_start+j,:-1],mu,sigma)
        temp_alpha=nu.zeros(2)
        for k in xrange(2):
            if k==1: #norm dist
                temp_alpha[k]=sum(weight_norm[norm_start+1:]*alpha[k]*temp_phi_norm/sum(alpha[k]*temp_phi_norm))
            else: #uniform dist
                temp_alpha[k]=sum(weight_norm[:norm_start]*alpha[k]*uni_func(age_unq,metal_unq,bins)/(pop_num*alpha[k]**2*uni_func(age_unq,metal_unq,bins)))
        #make alpha sum to 1
        temp_alpha=temp_alpha/sum(temp_alpha)
        mu=nu.sum(nu.tile(weight_norm,(bins*3,1)).T*lik[:,:-1],0)
        sigma=nu.cov((nu.tile(weight_norm,(bins*3,1)).T*lik[:,:-1]).T)
        alpha=nu.copy(temp_alpha)
        #make new points
        points=pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins)
    #get likelihoods
        lik=[]
        pool=Pool()
        for ii in points:
            pool.apply_async(like_gen,(data,ii,lib_vals,age_unq,metal_unq,bins,),callback=lik.append)
        pool.close()
        pool.join()
        lik=nu.array(lik)



def uni_func(age_unq,metal_unq,bins):
    #calculates the proablility density of uniform function
    return 1/metal_unq.ptp()**bins*1/(age_unq.ptp()/bins)**bins
        

def norm_func(x,mu,sigma):
    #calculates values of normal dist for set of points
    out=(2*nu.pi)**(-len(mu)/2.)*nu.linalg.det(sigma)**(-.5)
    out=out*nu.exp(-.5*(nu.dot((x-mu),nu.dot(nu.linalg.inv(sigma),(x-mu).T))))
    return out

def like_gen(data,active_param,lib_vals,age_unq,metal_unq,bins):
   #calcs chi squared values
    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    model=data_match_new(data,model,bins)
    N=[]
    for k in range(2,len(active_param),3):
        N.append(active_param[k])
    N=nu.array(N)
    model=nu.sum(nu.array(model.values()).T*N,1)
    #make weight paramer start closer to where ave data value
    return nu.hstack((active_param,sum((data[:,1]-normalize(data,model)*model)**2)))
 

def pop_builder(pop_num,alpha,mu,sigma,age_unq,metal_unq,bins):
    #creates pop_num of points for evaluation
    #only uses a multivarate norm and unifor dist for now

    #check if alpha sums to 1
    if sum(alpha)!=1:
        alpha=alpha/sum(alpha)
    #initalize params
    points=nu.zeros([pop_num,bins*3])
    #multivariate norm
    for j in range(mu.shape[0]):
        #start and stop points
        if j==0:
            start=0
        else:
            start=stop+1
        stop=start+int(pop_num*alpha[j])-1
        points[start:stop,:]=nu.random.multivariate_normal(mu[j],sigma[j],(stop-start)))
    #check for values outside range
    bin_index=0
    for i in range(bins*3):
        if i==0 or i%3==0: #metalicity
            index=nu.nonzero(nu.logical_or(points[:,i]< metal_unq[0],points[:,i]> metal_unq[-1]))[0]
            for j in index:
                points[j,:]=nu.random.multivariate_normal(mu,sigma)
                while check(points[j,:],metal_unq, age_unq, bins):
                    points[j,:]=nu.random.multivariate_normal(mu,sigma)
        
        elif (i-1)%3==0 or i-1==0:#age
             index=nu.nonzero(nu.logical_or(points[:,i]< age_bins[bin_index],points[:,i]>age_bins[bin_index+1]))[0]
             for j in index:
                points[j,:]=nu.random.multivariate_normal(mu,sigma)
                while check(points[j,:],metal_unq, age_unq, bins):
                    points[j,:]=nu.random.multivariate_normal(mu,sigma)
        elif (i-2)%3==0 or i==2: #norm
            index=nu.nonzero(nu.logical_or(points[:,i]< 0,points[:,i]>1))[0]
            for j in index:
                points[j,i]=nu.random.rand()

    #normalize normalization parameter
    if bins>1:
        temp=nu.zeros(pop_num)
        for i in range(2,bins*3,3):
            temp+=points[:,i]
        for i in range(2,bins*3,3):
            points[:,i]=points[:,i]/temp

    return points

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
                #active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index]
                active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
                bin_index+=1
            else: #norm
                active_param[k]=10*nu.random.random()
                

    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter+1)+nu.inf
    sigma=nu.identity(len(active_param))*nu.concatenate((nu.tile(
                [metal_unq.ptp()*nu.random.rand(),age_unq.ptp()/bins*nu.random.rand()],bins),
                          nu.array([nu.sqrt(bins)]*bins)))

    #for k in range(0,len(parambest),3):
    #    active_param[k]=nu.log10(0.0080)
    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    '''N,model=N_normalize(data,model,bins)
    ii=0
    for k in range(2,len(parambest),3):
        active_param[k]=nu.log10(N[ii])
        ii+=1'''
    model=data_match_new(data,model,bins)
    N=[]
    for k in range(2,len(parambest),3):
        N.append(active_param[k])
    N=nu.array(N)
    model=nu.sum(nu.array(model.values()).T*N,1)
    #make weight paramer start closer to where ave data value
    chi[0]=sum((data[:,1]-normalize(data,model)*model)**2)
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
        bin_index=0
        for k in range(1,len(parambest),3):
            active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
            bin_index+=1
        #for k in range(0,len(parambest),3):
        #    active_param[k]=nu.log10(0.0080)
       #calculate new model and chi
        model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
        '''N,model=N_normalize(data,model,bins)
        ii=0
        for k in range(2,len(parambest),3):
            active_param[k]=nu.log10(N[ii])
            ii+=1'''
        model=data_match_new(data,model,bins)
        N=[]
        for k in range(2,len(parambest),3):
            N.append(active_param[k])
        N=nu.array(N)
        model=nu.sum(nu.array(model.values()).T*N,1)

        chi[j]=sum((data[:,1]-normalize(data,model)*model)**2)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2)
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if chi[j]< chibest.value:
                print 'best fit value %f in iteration %i' %(chi[j],j)
                sys.stdout.flush()
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
        else:
            if nu.exp(nu.log(a)/SA(T_cuurent,option.itter/(cpu),1.5,.11))>nu.random.rand():#false accept
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
        if nu.min([1,nu.exp(-(chi[j-1]-chi[j])/(2.*SA(T_cuurent+1,option.itter/(cpu),1.5,.11))-(chi[j-1]+chi[j])/(2.*SA(T_cuurent,option.itter/(cpu),1.5,.11)))/T])>nu.random.rand():
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
    #for k in range(2,len(parambest),3):
    #    param[:,k]=param[:,k]/1000.
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
        param[:,i+2]=param[:,i+2]/10.
        
    return param

def MCMC_vanila(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC parameter estimation with a floating step size till 10k iterations
    #itter needs to be a array of normaly distrbuted numbers
    #so there are no problems with multiprocessing

    #part on every modual wanting to fit the spectra
    #controls input and expot of files for fitt
      
    
    #change random seed for random numbers for multiprocessing
    data[:,1]=data[:,1]*1000.  
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
    Nacept,Nreject=1.0,1.0
    acept_rate,out_sigma=[],[]
    j=1
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
            if chi[j]< chibest.value:
                print 'best fit value %f in iteration %i' %(chi[j],j)
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
