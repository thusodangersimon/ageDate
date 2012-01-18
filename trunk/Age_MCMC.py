#!/usr/bin/env python


from Age_date import *
from mpi4py import MPI
from scipy.cluster import vq as sci
from scipy.stats import levene, f_oneway
#import time as Time
a=nu.seterr(all='ignore')

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
 
def MCMC_comunicate(data,bins,itter):
    comm=MPI.COMM_WORLD
    size=comm.size                            
    myid = comm.rank   
    #comm tags
    sig_to_trans_tag,trans_tag=0,1
    #comm notices
    prerecieve=False
    #acts a 1 chain but uses multiple feelers
    fun=MC_func(data,bins,itter)
    #change random seed for random numbers for multiprocessing
    #nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    #param=nu.zeros([itter+1,bins*3])
    param=[]
    active_param=nu.zeros(bins*3)
    
    bin=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))
    bin_index=0
    #start in random place
    for k in xrange(bins*3):
        if any(nu.array(range(0,bins*3,3))==k):#metalicity
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,bins*3,3))==k): #age
                #active_param[k]=nu.random.random() #random
                #active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index] #random in bin
                active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]]) #mean position in bin
                bin_index+=1
                #active_param[k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
            else: #norm
                active_param[k]=nu.random.random()*1000

    chi=[]
    #chiappend=chi.append
    sigma=nu.identity(bins*3)*nu.tile(
                [0.5,age_unq.ptp()*nu.random.rand()*1,100.],bins)
    #try leastquares fit
    #active_param=fun.n_neg_lest(active_param)
    chi.append(0)
    chi[-1],active_param[range(2,bins*3,3)]=fun.func_N_norm(active_param)
    param.append(nu.copy(active_param))
    #set up shared varibles
    current_iter,acpt_rate,chi_best,param_best,turn_iter=0,.5,nu.copy(chi[-1]),nu.copy(param[-1]),500
    if myid==0:
        myturn=[True,1,0] #[if turn, rank of next person,number times i control stuff]
        
    else:
        myturn=[False,myid+1,0]
        if myturn[1]>size-1: #make circular
            myturn[1]=0
    #start MCMC
    #Naccept,Nrecjet=0,0
    while current_iter<itter:
        #print "hi, I'm %i at itter %i and chi %f" %(current_process().ident,j,chi[j-1])
        active_param=fun.New_chain(active_param,sigma,'norm')
        chi.append(0)
        if current_iter<1000:
            chi[-1],active_param=fun.SA(chi[-2],active_param,param[-1])
        else:
            chi[-1],active_param=fun.Mh_criteria(chi[-2],active_param,param[-1])
        #check for best fit
        if chi[-1]<chi_best:
            
            chi_best=nu.copy(chi[-1])
            param_best=nu.copy(active_param)
        param.append(nu.copy(active_param))
        current_iter+=1
        #if my turn then control sigma
        if myturn[0]:
            if current_iter>900 and current_iter%100==0:
                print nu.min(chi),chi[-1]
                sigma=fun.Step(sigma,param,'cov')
            elif current_iter<900:
                print nu.min(chi),chi[-1]
                sigma=fun.Step(sigma,param,'adapt')

            #decide if should send to next processor
            if myturn[2]>turn_iter: #send
                comm.isend([True,MPI.LOGICAL],dest=myturn[1],tag=sig_to_trans_tag)
                #send last iterations, current sigma, accept rate, best fit and param
                #acpt_rate=fun.Naccept/(fun.Naccept+fun.Nreject)
                #comm.send((param[-turn_iter:],sigma,rate,
                break
            else:
                myturn[2]+=1
                comm.Irecv([prerecieve,MPI.LOGICAL],source=MPI.ANY_SOURCE,tag=sig_to_trans_tag)
                if prerecieve: #about te get update
                    pass

    return outprep(nu.array(param)),nu.array(chi)

class MC_func:
    #compact MCMC function, can add new parts by calling in program
    def __init__(self,data,bins,iter_stop,spect=spect):
        #initalize bounds
        data_match_all(data)
        self.data=nu.copy(data)
        self.data[:,1]=self.data[:,1]*1000
        lib_vals=get_fitting_info(lib_path)
        lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
        metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
        age_unq=nu.unique(lib_vals[0][:,1])
        self.lib_vals=lib_vals
        self.age_unq= age_unq
        self.metal_unq,self.bins,self.spect=metal_unq,bins,spect
        self.bounds()
        #create random seed
        seed = open("/dev/random")
        rand_int = 0
        for i in seed.read(4):
            rand_int <<= 8
            rand_int += ord(i)
        print rand_int
        self.seed=rand_int
        nu.random.seed(self.seed)
        #accept and reject
        self.Nreject=1.
        self.Naccept=1.
        self.iteration=1.
        self.iter_stop=iter_stop*.1
        #sigma for step
        self.non_N_index=nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()
        #temperature stuff
        self.T,self.T_start,self.T_stop=9.905971092132212,1.8,.9
        self.Nexchange_ratio,self.T_cuurent=0.,0.

    def func(self,param): #changes param vlaues if not in range
        if len(param)!=self.bins*3:
            return nu.nan
        if check(param,self.metal_unq, self.age_unq,self.bins): #make sure params are in correct range
            for i in xrange(len(self.bounds)): #find which is out and fix
                if self.bounds[i][0]>param[i]: #if below bounds
                    param[i]=nu.copy(self.bounds[i][0])
                if self.bounds[i][1]<param[i]: #if above bounds
                    param[i]=nu.copy(self.bounds[i][1])

        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,self.bins)  
    #model=data_match_new(data,model,bins)
        index=xrange(2,self.bins*3,3)
        model['wave']= model['wave']*.0
        for ii in model.keys():
            if ii!='wave':
                model['wave']+=model[ii]*param[index[int(ii)]]
        return nu.sum((self.data[:,1]-model['wave'])**2)

    def func_N_norm(self,param):
        #returns chi and N norm best fit params
        if len(param)!=self.bins*3:
            return nu.nan
        model=get_model_fit_opt(param,self.lib_vals,self.age_unq,self.metal_unq,self.bins)  
        N,model,chi=N_normalize(self.data, model,self.bins)
    
        return chi,N
 
    def min_bound(self):
        #outputs an array of minimum values for parameters
        out=nu.zeros(self.bins*3)
        bin=nu.log10(nu.linspace(10**self.age_unq.min(),10**self.age_unq.max(),self.bins+1))
        bin_index=0
        for k in range(self.bins*3):
            if any(nu.array(range(0,self.bins*3,3))==k): #metal
                out[k]=self.metal_unq[0]
            elif any(nu.array(range(1,self.bins*3,3))==k): #age
                out[k]=bin[bin_index]
                bin_index+=1
            elif any(nu.array(range(2,self.bins*3,3))==k): #norm
                out[k]=0.0
        return out

    def max_bound(self):
        #outputs an array of maximum values for parameters
        out=nu.zeros(self.bins*3)
        bin=nu.log10(nu.linspace(10**self.age_unq.min(),10**self.age_unq.max(),self.bins+1))
        bin_index=1
        for k in range(self.bins*3):
            if any(nu.array(range(0,self.bins*3,3))==k): #metal
                out[k]=self.metal_unq[-1]
            elif any(nu.array(range(1,self.bins*3,3))==k): #age
                out[k]=bin[bin_index]
                bin_index+=1
            elif any(nu.array(range(2,self.bins*3,3))==k): #norm
                out[k]=nu.inf
        return out

    def bounds(self):
        #puts bounds into a easy reconizible format
        Min=self.min_bound()
        Max=self.max_bound()
        out=[]
        for i in range(len(Min)):
            out.append((Min[i],Max[i]))
        self.bounds=nu.copy(out)
        return out

    def n_neg_lest(self,param):
        #does bounded non linear fit
        try:
            out=fmin_bound(self.func,param, bounds = self.bounds,approx_grad=True)[0]
        except IndexError:
            out=param
        return out
    
    def Mh_criteria(self,chiold,param_new,param_old):
        #does metropolis hastings critera works with PMC class
        chinew,param_new[range(2,self.bins*3,3)]=self.func_N_norm(param_new)
        a=nu.exp((chiold-chinew)*100)
        if min([1,a])>nu.random.rand(): #accepted
            #out_param[-1]=nu.copy(out_param[-2])
            #chiold[-1]=nu.copy(chiold[-2])
            self.Naccept+=1
        else:
            self.Nreject+=1
            chinew=nu.copy(chiold)
            param_new=nu.copy(param_old)
        self.iteration+=1

        return chinew,param_new

    def SA(self,chiold,param_new,param_old):
        #does simulated anniling
        self.iteration+=1.
        chinew,param_new[range(2,self.bins*3,3)]=self.func_N_norm(param_new)
        a=nu.exp((chiold-chinew)/2)
        if a>=1: #acepted
            #param[j,:]=nu.copy(active_param)
            self.Naccept+=1.
        else:
            if nu.exp(nu.log(a)/SA(self.T_cuurent,self.iter_stop,self.T_start,self.T_stop))>nu.random.rand():#false accept
                #param[j,:]=nu.copy(active_param)
               self.Naccept+=1.
               print 'false accept',chiold,chinew,SA(self.T_cuurent,self.iter_stop,self.T_start,self.T_stop)
            else:
                chinew=nu.copy(chiold)
                param_new=nu.copy(param_old)
                self.Nreject+=1
         #change temperature
        if nu.min([1,nu.exp(nu.log(a)*(1./SA(self.T_cuurent+1,self.iter_stop,self.T_start,self.T_stop))-1/SA(self.T_cuurent,self.iter_stop,self.T_start,self.T_stop))/self.T])>nu.random.rand():
            self.T_cuurent+=1
            self.Nexchange_ratio+=1.  
        #if self.Naccept/(self.Naccept+self.Nreject)>.6:
        #    self.T_cuurent+=1
        #make sure the change temp rate is aroudn 2%
        if self.Nexchange_ratio/(self.Naccept+self.Nreject)>.02:
            self.T=self.T*1.05
        elif self.Nexchange_ratio/(self.Naccept+self.Nreject)<.005:
            self.T=self.T/1.05
        #change temperature schedual
        '''if self.iteration%50==0:
            if self.Naccept/(self.Naccept+self.Nreject)>.50:
                if self.T_start>0:
                    self.T_start-=.1
                #self.T_stop+=.1
            elif self.Naccept/(self.Naccept+self.Nreject)<.25:
                self.T_start+=.1
                #self.T_stop-=.1
                '''       
        
        #if .001>nu.random.rand() and j>500: #every hundred itterations
        #    a=nu.exp((mybest-chibest.value)/2.0)
        #    if a>1: #accept change in param
        #        #print j
        #        chi[j]=nu.copy(chibest.value)
        #        mybest=nu.copy(chibest.value)
        #        print "swiched places. I'm %i" %current_process().ident
        #        for k in range(len(active_param)): 
        #            param[j,k]=nu.copy(parambest[k])
        #            active_param[k]=nu.copy(parambest[k])
        return chinew,param_new

    def New_chain(self,old_chain,sigma,Type='norm'):
        #selecs new chain
        if Type=='norm':
            out=chain_gen_all(old_chain,self.metal_unq,self.age_unq,self.bins,sigma)

        elif Type=='stnt':
            out=multivariate_student(old_chain,sigma,2.3)
            t=Time.time()
            while check(out,self.metal_unq,self.age_unq,self.bins):
                out=multivariate_student(old_chain,sigma,2.3)
                if Time.time()-t>2.:
                    print 'taking to long'
                    sigma=sigma/1.05
        else: 
            print 'that distribution is not ready yet'
            raise
        return out

    def Step(self,sigma,param,Type='scale'):
        acc_rate=self.Naccept/(self.Naccept+self.Nreject+.0)
        #print acc_rate
        if Type=='adapt': #change sigma with acceptance rate
            if acc_rate>.50 and all(sigma.diagonal()[self.non_N_index]>=10**-5): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif acc_rate<.25 and all(sigma.diagonal()[self.non_N_index]<2.): #not enough
                sigma=sigma*1.05
        elif Type=='cov': #use covarnence matrix
                #take points that have changed only
            temp=nu.array(param)
            tsigma=Covarence_mat(temp[nu.abs(nu.diff(temp[:,0]))>0,:],temp.shape[0])
            if nu.all(tsigma==0):
                print 'chain not mixing'
                sigma=sigma/1.05
            else:
                sigma=tsigma
        elif Type=='scale':
            if acc_rate<0.001:
            # reduce by 90 percent
                sigma= sigma*0.1
            elif acc_rate<0.05:
                # reduce by 50 percent
                sigma=sigma* 0.5
            elif acc_rate<0.2:
                # reduce by ten percent
                sigma= sigma*0.9
            elif acc_rate>0.95:
                # increase by factor of ten
                sigma=sigma* 10.0
            elif acc_rate>0.75:
                # increase by double
                sigma= sigma*2.0
            elif acc_rate>0.5:
                # increase by ten percent
                sigma= sigma*1.1
            else:
                pass
  
        #self.Naccept,self.Nreject=1.,1.
        return sigma

   
def MCMC_SA(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC and reduices the false acceptance rate over a threshold
    #itter needs to be a array of normaly distrbuted numbers
    #so there are no problems with multiprocessing
    print "Starting processor %i" %current_process().ident
    #part on every modual wanting to fit the spectra
    #controls input and expot of files for fitt
    data[:,1]=data[:,1]*1000  
    fun=PMC_func(data,bins)
    cpu=float(cpu_count())
    non_N_index=nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()
    #change random seed for random numbers for multiprocessing
    nu.random.seed(current_process().ident)
    #initalize parmeters and chi squared
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    param=nu.zeros([option.itter+1,len(parambest)])
    active_param=nu.zeros(len(parambest))
    
    bin=nu.log10(nu.linspace(10**age_unq.min(),10**age_unq.max(),bins+1))
    bin_index=0
    #start in random place
    for k in xrange(len(parambest)):
        if any(nu.array(range(0,len(parambest),3))==k):#metalicity
            active_param[k]=(nu.random.random()*metal_unq.ptp()+metal_unq[0])
        else:#age and normilization
            if any(nu.array(range(1,len(parambest),3))==k): #age
                #active_param[k]=nu.random.random() #random
                #active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index] #random in bin
                active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]]) #mean position in bin
                bin_index+=1
                #active_param[k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
            else: #norm
                active_param[k]=nu.random.random()*10000

    chi=nu.zeros(option.itter+1)+nu.inf
    sigma=nu.identity(len(active_param))*nu.tile(
                [0.5,age_unq.ptp()*nu.random.rand(),1.],bins)
    #try leastquares fit
    active_param=fun.n_neg_lest(active_param)
    chi[0],active_param[range(2,bins*3,3)]=fun.func_N_norm(active_param)
    param[0,:]=nu.copy(active_param)
    #parambest=nu.copy(active_param)

    mybest=nu.copy(chi[0])
    for k in range(len(active_param)):
        parambest[k]=nu.copy(active_param[k])
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject,Nexchange_ratio,T_cuurent=1.0,1.0,1.0,0.
    acept_rate,out_sigma=[],[]
    j,T=1,9.
    T_start,T_stop=1.8,0.9
    while option.value and i.value<option.itter:
        if j%1000==0:
            #print "hi, I'm %i at itter %i and chi %f" %(current_process().ident,j,chi[j-1])
            #print sigma.diagonal()
            print SA(T_cuurent,option.itter/(cpu),T_start,T_stop),T_cuurent,chi[j-1]
            sys.stdout.flush()
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
        #bin_index=0
      #calculate new model and chi
        chi[j],active_param[range(2,bins*3,3)]=fun.func_N_norm(active_param)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2)
        #metropolis hastings
        if a>=1: #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if chi[j]< mybest:
                mybest=nu.copy(chi[j])
            if chi[j]<chibest.value:
                print 'best fit value %f in iteration %i, from processor %i' %(chi[j],j,
                                                                               current_process().ident)
                sys.stdout.flush()
                
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
        else:
            if nu.exp(nu.log(a)/SA(T_cuurent,option.itter/(cpu),T_start,T_stop))>nu.random.rand():#false accept
                param[j,:]=nu.copy(active_param)
                Nacept+=1
            else:
                param[j,:]=nu.copy( param[j-1,:])
                active_param=nu.copy( param[j-1,:])
                chi[j]=nu.copy(chi[j-1])
                Nreject+=1
 
        if j<1000: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if Nacept/(Nacept+Nreject)>.50 and all(sigma.diagonal()>=10**-5): 
               #too few aceptnce decrease sigma
                sigma=sigma/1.05
            elif Nacept/(Nacept+Nreject)<.25 and all(sigma.diagonal()[non_N_index]<10): #not enough
                sigma=sigma*1.05
        else: #use covarnence matrix
            if j%1000==0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
                sigma=Covarence_mat(param,j)
                active_param=fun.n_neg_lest(active_param)
        #change temperature
        if nu.min([1,nu.exp(-(chi[j-1]-chi[j])/(2.*SA(T_cuurent+1,option.itter/(cpu),T_start,T_stop))-(chi[j-1]+chi[j])/(2.*SA(T_cuurent,option.itter/(cpu),T_start,T_stop)))/T])>nu.random.rand():
            T_cuurent+=1
            Nexchange_ratio+=1   
        #make sure the change temp rate is aroudn 2%
        if Nexchange_ratio/(Nacept+Nreject)>.02:
            T=T*1.05
        elif Nexchange_ratio/(Nacept+Nreject)<.005:
            T=T/1.05
        #change temperature schedual
        if j%50==0:
            if Nacept/(Nacept+Nreject)>.50:
                T_start+=.1
                T_stop+=.1
            elif Nacept/(Nacept+Nreject)<.25:
                T_start-=.1
                T_stop-=.1
                
        
        #if .001>nu.random.rand() and j>500: #every hundred itterations
        #    a=nu.exp((mybest-chibest.value)/2.0)
        #    if a>1: #accept change in param
        #        #print j
        #        chi[j]=nu.copy(chibest.value)
        #        mybest=nu.copy(chibest.value)
        #        print "swiched places. I'm %i" %current_process().ident
        #        for k in range(len(active_param)): 
        #            param[j,k]=nu.copy(parambest[k])
        #            active_param[k]=nu.copy(parambest[k])
                    
                       
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
   # q.put((param,chi))
    #q.put((param,chi,out_sigma,acept_rate))
    #return param,chi,out_sigma,acept_rate

def Convergence_tests(param,keys,n=1000):
    #uses Levene's test to see if var between chains are the same if that is True
    #uses f_oneway (ANOVA) to see if means are from same distrubution 
    #if both are true then tells program to exit
    #uses last n chains only
    L_result={}
    out=False
    #turn into an array
    for i in param:
        for j in keys:
            i[j]=nu.array(i[j])
    for i in keys:
        param[0][i]=nu.array(param[0][i])
        L_result[i]=nu.zeros(param[0][i].shape[1])
        for k in range(param[0][i].shape[1]):
            samples='' 
            for j in range(len(param)):
                samples+='param['+str(j)+']["'+i+'"][-'+str(int(n))+':,'+str(k)+'],'
            L_result[i][k]=eval('levene('+samples[:-1]+')')[1]
        if nu.all(L_result[i]>.05): #if leven test is true
            out=True
            print "Levene's test is true for %s bins" %i
        else:
            print '%i out of %i parameters have same varance' %(sum( L_result[i]>.05),
                                                                param[0][i].shape[1])

    #do ANOVA to see if means are same
    if out:
        #try f_oneway test
        A_result={}
        out=False
        for i in keys:
            A_result[i]=nu.zeros(param[0][i].shape[1])
            for k in range(param[0][i].shape[1]):
                samples='' 
                for j in range(len(param)):
                    samples+='param['+str(j)+']["'+i+'"][-'+str(int(n))+':,'+str(k)+'],'
                A_result[i][k]=eval('f_oneway('+samples[:-1]+')')[1]
            if nu.all(A_result[i]>.05): #if leven test is true
                print "ANOVA says chains have converged. Ending program"
                return True
            else:
                print '%i out of %i parameters have same means' %(sum( A_result[i]>.05),
                                                                    param[0][i].shape[1])
    return False

def RJ_multi(data,itter,k_max=16,cpus=cpu_count()):
    #does multiple chains for RJMCMC
    #shares info about chain every 300 itterations to other chains
    option=Value('b',True)
    option.burnin=10**3
    option.itter=int(itter+option.burnin)
    option.send_num=Value('d',0) #tells who is sending and who is reciving new data
    option.cpu_tot=cpus

    work=[]
    q_talk,q_final=Queue(),Queue()
    #start multiprocess mcmc
    for ii in range(cpus):
        work.append(Process(target=rjmcmc,args=(data,itter,k_max,option,ii,q_talk,q_final)))
        work[-1].start()

    rank,size,conver_test=-nu.ones(cpus),nu.zeros(cpus),[]
    while True: #wait 5 min after all complete cooling
        if q_final.qsize()>=cpus:
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
       a= q_final.get()
    option.value=False
    #wait for proceses to finish
    count=0
    temp=[]
    while count<cpus:
        if q_final.qsize()>0:
           temp.append(q_final.get())
           count+=1
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
            fac.append([int(i),nu.mean(bayes_fac[i]),len(bayes_fac[i])])
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
        outparam[str(int(j))][:,range(2,3*j,3)]=outparam[str(int(j))][:,range(2,3*j,3)]/1000.
    
    return outparam,outchi,fac[fac[:,0].argsort(),:]

def rjmcmc(data,itter=5*10**3,k_max=16,option=True,rank=0,q_talk=None,q_final=None):
    #parallel worker program reverse jump mcmc program
    nu.random.seed(current_process().pid)
    #initalize boundaries
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    data[:,1]=data[:,1]*1000  
    #create fun for all number of bins
    attempt=False
    fun,param,active_param,chi,sigma={},{},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    bayes_fact={} #to calculate bayes factor
    for i in range(1,k_max+1):
        fun[str(i)],param[str(i)]=PMC_func(data,i),[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,age_unq.ptp()*nu.random.rand(),1.],i)
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
        bayes_fact[str(i)]=[]
    #for parallel to know how many iterations have gone by
    N_all={'accept':dict(Nacept),'reject':dict(Nreject)}
    #bins to start with
    bins=1
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
    active_param[str(bins)]=fun[str(bins)].n_neg_lest(active_param[str(bins)])
    chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun[str(bins)].func_N_norm(active_param[str(bins)])
    param[str(bins)].append(nu.copy(active_param[str(bins)]))
    #parambest=nu.copy(active_param)

    mybest=nu.copy([chi[str(bins)][0],bins])
    parambest=nu.copy(active_param[str(bins)])

    #start rjMCMC
    T_cuurent,Nexchange_ratio=1.0,1.0
    #acept_rate,out_sigma=[.35],[]
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    while option.value:
        if j%500==0:
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
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun[str(bins)].func_N_norm(active_param[str(bins)])
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
        a=nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/SA(T_cuurent,itter,T_start,T_stop))
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
        if len(param[str(bins)])<50000: #change sigma with acceptance rate
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
            tchi,active_param[str(temp_bins)][range(2,temp_bins*3,3)]=fun[str(temp_bins)].func_N_norm(active_param[str(temp_bins)])
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
            #Nexchange_ratio=1.
            #print "Changed number of bins to %i for better fit!" %bins

        #########################################change temperature
        if nu.min([1,nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/(2.*SA(T_cuurent+1,itter,T_start,T_stop))-(chi[str(bins)][-2]+chi[str(bins)][-1])/(2.*SA(T_cuurent,itter,T_start,T_stop)))/T])>nu.random.rand():
            if T_cuurent<.02*itter:
                T_cuurent+=1
                #print T_cuurent,T
            elif T_cuurent==round(.02*itter):
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
    data[:,1]=data[:,1]/1000.
    #q.put((param[option.burnin:,:],chi[option.burnin:]))
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


def dict_size(dic):
    #returns total number of elements in dict
    size=0
    for i in dic.keys():
        size+=len(dic[i])

    return size

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
                raise
    #N=sum(means.take(range(2,bins*3,3)))
    #for i in range(2,bins*3,3):#normalize normalization to 1
    #    means[i]=means[i]/N

    return out


def rand_choice(x,prob):
    #chooses value from x with probabity of prob**-1 so lower prob values are more likeliy
    #x must be monotonically increasing
    if not nu.sum(prob)==1: #make sure prob equals 1
        prob=prob/nu.sum(prob)
    #check is increasing
    u=nu.random.rand()
    if nu.all(x==nu.sort(x)): #if sorted easy
        N=nu.cumsum(prob**-1/nu.sum(prob**-1))
        index=nu.array(range(len(x)))
        return index[nu.min(nu.abs(N-u))==nu.abs(N-u)][0]
    else:
        index=nu.argsort(x)
        temp_x=nu.sort(x)
        N=nu.cumsum(prob[index]**-1/nu.sum(prob[index]**-1))
        return index[nu.min(nu.abs(N-u))==nu.abs(N-u)][0]


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

def outprep(param):
    #changes metals from log to normal
    for i in range(0,param.shape[1],3):
        param[:,i]=10**param[:,i]
        param[:,i+2]=param[:,i+2]/1000.
        
    return param

def MCMC_vanila(data,bins,i,chibest,parambest,option,q=None):
    #does MCMC parameter estimation with a floating step size till 10k iterations
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
                #active_param[k]=nu.random.random() #random
                #active_param[k]=nu.random.random()*age_unq.ptp()/float(bins)+bin[bin_index] #random in bin
                #active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]]) #mean position in bin
                #bin_index+=1
                active_param[k]=nu.random.random()*age_unq.ptp()+age_unq[0] #random place anywhere
            else: #norm
                active_param[k]=nu.random.random()*10000
    #N=sum(active_param.take(range(2,bins*3,3)))
    #for j in range(2,bins*3,3):            
    #    active_param[j]=active_param[j]/N
    param[0,:]=nu.copy(active_param)
    parambest=nu.copy(active_param)
    chi=nu.zeros(option.itter+1)+nu.inf
    sigma=nu.identity(len(active_param))*nu.concatenate((nu.tile(
                [metal_unq.ptp()*nu.random.rand(),age_unq.ptp()/bins*nu.random.rand()],bins),
                          nu.array([nu.sqrt(bins)]*bins)))

    model=get_model_fit_opt(active_param,lib_vals,age_unq,metal_unq,bins)  
    model=data_match_new(data,model,bins)
    model=nu.sum(nu.array(model.values()).T*active_param.take(range(2,bins*3,3)),1)
    #make weight paramer start closer to where ave data value
    #chi[0]=sum((data[:,1]-normalize(data,model)*model)**2)
    chi[0]=sum((data[:,1]-model)**2)
    chibest.value=chi[0]
    for k in range(len(active_param)):
        parambest[k]=nu.copy(active_param[k])
    #stuff just for age_date
    #start MCMC
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    Nacept,Nreject,Nexchange_ratio,T_cuurent=1.0,1.0,1.0,0.
    acept_rate,out_sigma=[],[]
    j=1
    while option.value and i.value<option.itter:
        active_param= chain_gen_all(active_param,metal_unq, age_unq,bins,sigma)
        bin_index=0
        '''for k in range(1,len(parambest),3):
            active_param[k]=nu.mean([bin[bin_index],bin[1+bin_index]])
            bin_index+=1'''
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
        model=nu.sum(nu.array(model.values()).T*active_param.take(range(2,bins*3,3)),1)
        chi[j]=sum((data[:,1]-normalize(data,model)*model)**2)
        #decide to accept or not
        a=nu.exp((chi[j-1]-chi[j])/2)
        #metropolis hastings
        if nu.min([1,a])>nu.random.rand(): #acepted
            param[j,:]=nu.copy(active_param)
            Nacept+=1
            if chi[j]< chibest.value:
                print 'best fit value %f in iteration %i' %(chi[j],j)
                sys.stdout.flush()
                chibest.value=nu.copy(chi[j])
                for k in range(len(active_param)):
                    parambest[k]=nu.copy(active_param[k])
                
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

        if .01>nu.random.rand(): #every hundred itterations
            a=nu.exp((chi[j]-chibest.value)/2.0)
            if a>1: #accept change in param
                #print j
                chi[j]=nu.copy(chibest.value)
                for k in range(len(active_param)): 
                    param[j,k]=nu.copy(parambest[k])
                    active_param[k]=nu.copy(parambest[k])
                
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
    
    """
    ybar = nu.mean(Y)
    if k!=0:
        N = nu.sum((Y[:-k]-ybar)* (Y[k:] -ybar))
    else:
        N = nu.sum((Y-ybar)**2)
    return N/nu.var(Y)
    """

if __name__=='__main__':
    import cPickle as pik
    age=nu.array([5.78,8.27,10.23])
    metal=nu.array([.0131]*3)

    Norm=nu.array([10,5,2.5])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('10525.pik','w'),2)

    Norm=nu.array([5,10,2.5])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('51025.pik','w'),2)

    Norm=nu.array([5,2.5,10])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('52510.pik','w'),2)

    Norm=nu.array([2.5,5,10])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('25510.pik','w'),2)

    Norm=nu.array([2.5,10,5])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('25105.pik','w'),2)

    Norm=nu.array([10,2.5,5])
    data,info1,weight=own_array_spect(age,nu.log10(metal),Norm,lam_min=2000,lam_max=10**4)
    param,chi,fac=RJ_multi(data,5*10**3)
    pik.dump((param,chi,fac,data),open('10255.pik','w'),2)

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
    
