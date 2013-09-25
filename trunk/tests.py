#!/usr/bin/env python
#
# Name:  
#
# Author: Thuso S Simon
#
# Date: Sep. 1, 2011
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
"""
scap develoment programs
"""


from Age_date import *
from Age_MCMC import SA,Covarence_mat
import Age_date as ag
import ezgal as gal
#from mpi4py import MPI
import numpy as nu
from glob import glob
import scipy.stats as stats_dist

def rjmcmc(data,itter=10**5,k_max=16):
    #test reverse jump mcmc program
    #initalize boundaries
    lib_vals=get_fitting_info(lib_path)
    lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
    metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
    age_unq=nu.unique(lib_vals[0][:,1])

    data[:,1]=data[:,1]*1000  
    #create fun for all number of bins
    fun,param,active_param,chi,sigma={},{},{},{},{}
    Nacept,Nreject,acept_rate,out_sigma={},{},{},{}
    for i in range(1,k_max+1):
        fun[str(i)],param[str(i)]=PMC_func(data,i),[]
        active_param[str(i)],chi[str(i)]=nu.zeros(3*i),[nu.inf]
        sigma[str(i)]=nu.identity(3*i)*nu.tile(
            [0.5,age_unq.ptp()*nu.random.rand(),1.],i)
        Nacept[str(i)],Nreject[str(i)]=1.,0.
        acept_rate[str(i)],out_sigma[str(i)]=[.35],[]
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
    #Nacept,Nreject=nu.ones(len(active_param)),nu.ones(len(active_param))
    #Nacept,Nreject,Nexchange_ratio,T_cuurent=1.0,1.0,1.0,0.
    T_cuurent,Nexchange_ratio=1.0,1.0
    #acept_rate,out_sigma=[.35],[]
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    while True:
        if j%100==0:
            print "hi, I'm at itter %i, chi %f, acceptance %0.2f, SA %f" %(len(param[str(bins)]),chi[str(bins)][-1],acept_rate[str(bins)][-1],SA(T_cuurent,itter,T_start,T_stop))
            #print sigma[str(bins)].diagonal()
            #print 'Acceptance %i reject %i' %(Nacept,Nreject)
            #print active_param[str(bins)][range(2,bins*3,3)]
        active_param[str(bins)]= Chain_gen_all(active_param[str(bins)],metal_unq, age_unq,bins,sigma[str(bins)])
        #bin_index=0
        #calculate new model and chi
        chi[str(bins)].append(0.)
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun[str(bins)].func_N_norm(active_param[str(bins)])
        #decide to accept or not
        a=nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/SA(T_cuurent,itter,T_start,T_stop))
        #print chi[str(bins)][-2]-chi[str(bins)][-1]
        #metropolis hastings
        if a>nu.random.rand(): #acepted
            param[str(bins)].append(nu.copy(active_param[str(bins)]))
            Nacept[str(bins)]+=1
            if not nu.isinf(min(chi[str(bins)])): #put temperature on order of chi calue
                T_start=nu.round(min(chi[str(bins)]))

            #see if global best fit
            if chi[str(bins)][-1]< mybest[0]:
                mybest=nu.copy([chi[str(bins)][-1],bins])
                parambest=nu.copy(active_param[str(bins)]) 
                print 'New global best fit with chi of %2.2f and %i bins' %(mybest[0],mybest[1])
        else:
            param[str(bins)].append(nu.copy(param[str(bins)][-1]))
            active_param[str(bins)]=nu.copy(param[str(bins)][-1])
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1

        #step stuff
        if len(param[str(bins)])<800: #change sigma with acceptance rate
            #k=random.randint(0,len(sigma)-1)
            if  (acept_rate[str(bins)][-1]<.25 and 
                                        all(sigma[str(bins)].diagonal()[nu.array([range(1,bins*3,3),range(0,bins*3,3)]).ravel()]<5.19)):
               #too few aceptnce decrease sigma
                sigma[str(bins)]=sigma[str(bins)]/1.05
            elif acept_rate[str(bins)][-1]>.50 and all(sigma[str(bins)].diagonal()>=10**-5): #not enough
                sigma[str(bins)]=sigma[str(bins)]*1.05
        else: #use covarnence matrix
            if j%100==0: #and (Nacept/Nreject>.50 or Nacept/Nreject<.25):
                print 'Cov time'
                sigma[str(bins)]=Covarence_mat(nu.array(param[str(bins)]),j)
                #active_param=fun.n_neg_lest(active_param)
        if nu.all(sigma['2'].diagonal()==0): #if step is 0 make small
            for k in xrange(sigma[str(bins)].shape[0]):
                sigma[str(bins)][k,k]=10.**-5

        #inter model step
        #decide if birth or death
        if (birth_rate>nu.random.rand() and bins<k_max and j>j_timeleft ) or bins==1:
            #birth
            rand_step,rand_index=nu.random.rand(3)*[metal_unq.ptp(), age_unq.ptp(),1.],nu.random.randint(bins)
            temp_bins=1+bins
            #criteria for this step
            critera=(1/2.)**temp_bins
            #new param step
            for k in range(len(active_param[str(bins)])):
                active_param[str(temp_bins)][k]=active_param[str(bins)][k]
            #set last 3 and rand_index 3 to new
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
            
        else:
            #death
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
        tchi,active_param[str(temp_bins)][range(2,temp_bins*3,3)]=fun[str(temp_bins)].func_N_norm(active_param[str(temp_bins)])
        #rjmcmc acceptance critera 
        #print tchi,chi[str(bins)][-1]
        if nu.exp((chi[str(bins)][-1]-tchi)/2.)*birth_rate*critera>nu.random.rand() and j>j_timeleft:
            #accept model change
            bins=temp_bins+0
            chi[str(bins)].append(nu.copy(tchi))
            param[str(bins)].append(nu.copy(active_param[str(bins)]))
            j,j_timeleft=0,nu.random.exponential(200)
            #Nexchange_ratio=1.
            print "Changed number of bins to %i for better fit!" %bins

        #change temperature
        if nu.min([1,nu.exp((chi[str(bins)][-2]-chi[str(bins)][-1])/(2.*SA(T_cuurent+1,itter,T_start,T_stop))-(chi[str(bins)][-2]+chi[str(bins)][-1])/(2.*SA(T_cuurent,itter,T_start,T_stop)))/T])>nu.random.rand():
            if T_cuurent<.02*itter:
                T_cuurent+=1
                #print T_cuurent,T
            elif T_cuurent==.02*itter:
                print 'done with cooling'
            Nexchange_ratio+=1   
        #make sure the change temp rate is aroudn 2%
        if Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))>.02:
            T=T*1.05
        elif Nexchange_ratio/(nu.sum(Nacept.values())+nu.sum(Nreject.values()))<.005:
            T=T/1.05
        #change temperature schedual
        #keep on order with chi squared
        if j%20==0:
            if acept_rate[str(bins)][-1]>.5 and T_start<10**-5:
                T_start/=2.
                #T_stop+=.1
            elif acept_rate[str(bins)][-1]<.25 and T_start<3*10**5:
                T_start*=2.
                #T_stop-=.1
                                         
        j+=1
        acept_rate[str(bins)].append(nu.copy(Nacept[str(bins)]/(Nacept[str(bins)]+Nreject[str(bins)])))
        if len(param[str(bins)])>50000:
            break
        out_sigma[str(bins)].append(nu.copy(sigma[str(bins)].diagonal()))
    #return once finished 
    for i in param.keys():
        chi[i]=nu.array(chi[i])
        param[i]=nu.array(param[i])
        acept_rate[i]=nu.array(acept_rate[i])
        out_sigma[i]=nu.array(out_sigma[i])
    #for k in range(2,len(parambest),3):
    #    param[:,k]=param[:,k]/1000.
    data[:,1]=data[:,1]/1000.
    #q.put((param[option.burnin:,:],chi[option.burnin:]))
   # q.put((param,chi))
    #q.put((param,chi,out_sigma,acept_rate))
    return param,chi,sigma,acept_rate,out_sigma

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

def quick_cov_MCMC(x,y,params,func=[],constrants=[],sigma=0.8,itter=10**5,quiet=False):
    pypar=MPI.COMM_WORLD
    proc = pypar.size                                
    myid = pypar.Get_rank()
    param_info,sigma_tag,end_tag=0,1,2 #tags for sending and reciving
     #master process handels cov calculations and storring of out data
    out_param=[]
    out_chi=[]
    i,j,i_out=0,0,0
    sigma=nu.identity(len(params))*sigma
    #print myid
    if myid!=0:
        param=nu.array([params,params])#set input parameters [0]=old [1]=new
        if not func: #see if inputed a function
            raise
        #first fit
        y_fit=func(x,param[1])
        #start up chi
        chi=[nu.sum((y_fit-y)**2),nu.inf]
        if nu.isnan(chi[0]):
            chi[0]=nu.inf
        chibest=nu.inf
        parambest=nu.copy(param[0])
        out_param.append(nu.copy(param[0]))
        out_chi.append(nu.copy(chi[0]))

       #start mcmc
        keep_iter=True
        while keep_iter:
            i+=1
            i_out+=1
            #print i_out
            if i%1000==0 and not quiet:
                print 'current accptence rate %2.2f and chi2 is %2.2f' %(j/(i+1.0)*100.0,chi[1])
                #print "my id is %i" %myid
            # print param[1] 
             #select new param
            param[1]=nu.random.multivariate_normal(param[0],sigma)
            for ii in range(len(params)):
                i2=0
                while param[1][ii]<=constrants[ii][0] or param[1][ii]>=constrants[ii][1]: 
                    param[1][ii]=param[0][ii]+nu.random.randn()*sigma[ii,ii]
                    i2+=1
                    if i2>50:#sigma may be too big
                        sigma[ii,ii]=sigma[ii,ii]/1.05
            #sample new distribution
            y_fit=func(x,param[1])
            chi[1]=nu.sum((y_fit-y)**2)
            if nu.isnan(chi[1]):
                chi[1]=nu.inf
        
            #decide to accept or not
            a=nu.exp((chi[0]-chi[1])/2.0)
            #metropolis hastings
            if a>=1: #acepted
                chi[0]=chi[1]+0.0
                param[0]=param[1]+0.0
                out_param.append(nu.copy(param[0]))
                out_chi.append(nu.copy(chi[0]))
                j+=1
                if chi[0]<chibest:
                    chibest=chi[0]+0.0
                    parambest=param[0]+0.0
                    #if not quiet:
                    print 'best fit value for %3.2f,%3.2f with chi2=%4.2f' %(parambest[0],parambest[1],chibest)
                    #print i

            else:
                if a>nu.random.rand():#false accept
                    chi[0]=chi[1]+0.0
                    param[0]=param[1]+0.0
                    j+=1
                    out_param.append(nu.copy(param[0]))
                    out_chi.append(nu.copy(chi[0]))

                else:
                    out_param.append(nu.copy(param[0]))
                    out_chi.append(nu.copy(chi[0]))
            
            if i_out==50:
                pypar.isend([out_param,out_chi,j,i],dest=0,tag=param_info)
                #print 'here 1'
                sigma=pypar.recv(source=0, tag=sigma_tag)
                keep_iter=pypar.recv(source=0, tag=end_tag)
                #print sigma
                i_out=0
                out_param,out_chi=[],[]
        pypar.Barrier()
    else:
        while True:
            status=MPI.Status()
            temp=pypar.recv(source=MPI.ANY_SOURCE,tag=param_info,status=status)
            temp_param,temp_chi,temp_j,temp_i=temp
            for ii in temp_param:
                out_param.append(nu.copy(ii))
            for ii in temp_chi:
                out_chi.append(nu.copy(ii))
            j+=temp_j
            i+=temp_i
            #print float(j/(i+j))
       #change sigma with acceptance rate
            if float(j/(i+j))>.24 and any(sigma.diagonal()<3): #too many aceptnce increase sigma
                sigma=sigma*5.0
            elif float(j/(i+j))<.34 and any(sigma.diagonal()>10**-5): #not enough
                sigma=sigma/5.0
        #change sigma with cov matrix  
            if i>1000 and i%500==0:
                sigma=nu.cov(nu.array(out_param)[i-1000:i,:].T)
            pypar.isend(sigma,status.Get_source(),tag=sigma_tag)
            #print 'on %i out of %i' %(i,itter)
            if i<itter:
                pypar.isend(True,status.Get_source(),tag=end_tag)
            else:
                keep_iter=False
                for i in range(1,proc):
                    pypar.send(sigma,i,tag=sigma_tag)
                    pypar.send(keep_iter,i,tag=end_tag)
                out_param,out_chi=nu.array(out_param),nu.array(out_chi)
                chibest=out_chi.min()
                parambest=out_param[chibest==out_chi,:][0]
                pypar.Barrier()
                print 'ending'
                #pypar.finalize()
                return out_chi,out_param,parambest,chibest

def scatter():
    comm=MPI.COMM_WORLD
    size=comm.size                            
    myid = comm.rank   
    if myid==0:
        m=nu.random.rand()
        v=comm.Scatter(m)

    print v, myid
    

#pymulti nest
class multinest_fit(object):
    '''Finds the age, metalicity, star formation history, 
    dust obsorption and line of sight velocity distribution
    to fit a Spectrum.

    Uses vespa methodology splitting const sfh into multiple componets
    '''
    def __init__(self,data, min_sfh=1,max_sfh=16,lin_space=False,use_dust=True, 
		use_losvd=True, spec_lib='p2',imf='salp',
			spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):
        '''(VESPA_fitclass, ndarray,int,int) -> NoneType
        data - spectrum to fit
        *_sfh - range number of burst to allow
        lin_space - make age bins linearly space or log spaced
        use_* allow useage of dust and line of sigt velocity dispersion
        spec_lib - spectral lib to use
        imf - inital mass function to use
        spec_lib_path - path to ssps
        sets up vespa like fits
        '''
        self.data = nu.copy(data)
		#make mean value of data= 100
        self._norm = 1./(self.data[:,1].mean()/100.)
        self.data[:,1] *= self._norm
        #load models
        cur_lib = ['basti', 'bc03', 'cb07','m05','c09','p2']
        assert spec_lib.lower() in cur_lib, ('%s is not in ' %spec_lib.lower() + str(cur_lib))
        if not spec_lib_path.endswith('/') :
            spec_lib_path += '/'
        models = glob(spec_lib_path+spec_lib+'*'+imf+'*')
        if len(models) == 0:
			models = glob(spec_lib_path+spec_lib.lower()+'*'+imf+'*')
        assert len(models) > 0, "Did not find any models"
        #crate ezgal class of models
        SSP = gal.wrapper(models)
        self.SSP = SSP
        #extract seds from ezgal wrapper
        spect, info = [SSP.sed_ls], []
        for i in SSP:
			metal = float(i.meta_data['met'])
			ages = nu.float64(i.ages)
			for j in ages:
				if j == 0:
					continue
				spect.append(i.get_sed(j,age_units='yrs'))
				info.append([metal+0,j])
        info,self._spect = [nu.log10(info),None],nu.asarray(spect).T
        #test if sorted
        self._spect = self._spect[::-1,:]
        #make spect match wavelengths of data
        #self._spect = ag.data_match_all(data,self._spect)[0]
        #extra models to use
        self._has_dust = use_dust
        self._has_losvd = use_losvd
        #key order
        self._key_order = ['gal']
        if use_dust:
            self._key_order.append('dust')
        if use_losvd:
            self._key_order.append('losvd')
		#set hidden varibles
        self._lib_vals = info
        self._age_unq = nu.unique(info[0][:,1])
        self._metal_unq = nu.unique(info[0][:,0])
        self._lib_vals[0][:,0] = 10**self._lib_vals[0][:,0]
        self._min_sfh, self._max_sfh = min_sfh,max_sfh +1
        #max total length of bins constraints
        self._max_age = self._age_unq.ptp()    
        #save
        self.param = []
        self.param_prior = []

    def lik(self,p, bins,nbins):
        '''(Example_lik_class, ndarray) -> float
        Calculates likelihood for input parameters. Outuputs log-likelyhood'''
        #change to correct format
        self.param.append(p)
        param = {'gal':nu.asarray(p[:4]),
             'dust':nu.asarray(p[4:6]),
             'losvd':nu.asarray(p[6:10])}
        if not self._check_len(param['gal'],'1'):
            return -nu.inf
        burst_model = {}
        for i in [param['gal']]:
            burst_model[str(i[1])] =  10**i[3]*ag.make_burst(i[0],i[1],i[2],
            self._metal_unq, self._age_unq, self._spect, self._lib_vals)
        burst_model['wave'] = nu.copy(self._spect[:,0])
		#do dust
        if self._has_dust:
            #dust requires wavelengths
            
            #not correct
            burst_model = ag.dust(param['dust'],burst_model)
		#do losvd
        if self._has_losvd:
            #check if wavelength exsist
            if 'wave' not in burst_model.keys():
                burst_model['wave'] = nu.copy(self._spect[:,0])
            #make buffer for edge effects
            wave_range = [self.data[:,0].min(),self.data[:,0].max()]
            burst_model = ag.LOSVD(burst_model, param['losvd'], wave_range)
        #need to match data wavelength ranges and wavelengths
		#get loglik
        
        burst_model = ag.data_match(self.data,burst_model,bins)
        model = nu.sum(burst_model.values(),0)
        
		#return loglik
        if self.data.shape[1] == 3:
            #uncertanty calc
            pass
        else:
            prob = stats_dist.norm.logpdf(model,self.data[:,1]).sum()
            #prob = -nu.sum((model -	self.data[:,1])**2)
        #return
        if nu.isnan(prob):
            return -nu.inf
        print prob
        return prob        

    def prior(self,param, bins, nbins):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #[len,age,metals,weight,tbc,tism,sigma,z,h3,h4]
        param[0] = param[0] * self._age_unq.ptp()
        param[1] = param[1] * self._age_unq.ptp() + self._age_unq.min()
        param[2] = param[2]*self._metal_unq.ptp()+self._metal_unq.min()
        param[3] = param[3]*700-300
        #dust
        param[4] = param[4]*4.
        param[5] = param[5]*4.
        #losvd
        param[6] *= 3.
        param[7] *= .05
        param[8] *= 1.
        param[9] *= 1.
        #temp save
        self.param_prior.append(param)
        return param
    
    def _check_len(self, tparam, key):
        '''(VESPA_class, dict(ndarray) or ndarray,str)-> ndarray
        Make sure parameters ahere to criterion listed below:
        1. bins cannot overlap
        2. total length of bins must be less than length of age_unq
        3. ages must be in increseing order
        4. make sure bins are with in age range
        5. No bin is larger than the age range
        '''
        
        #check input type
        if type(tparam) == dict:
            param = tparam.copy()
        elif  type(tparam) == nu.ndarray:
            param = {key:tparam}
        else:
            raise TypeError('input must be dict or ndarray')
        #make sure age is sorted
        '''if not self.issorted(param[key][:,1]):
            return False'''
        #make sure bins do not overlap fix if they are
        '''for i in xrange(param[key].shape[0]-1):
            #assume orderd by age
            max_age = param[key][i,0]/2. + param[key][i,1]
            min_age_i = param[key][i+1,1] - param[key][i+1,0]/2.
            if max_age > min_age_i:
                #overlap
                #make sure overlap is significant
                if not nu.allclose(max_age, min_age_i):
                    return False
            #check if in age bounds
            if i == 0:
                if self._age_unq.min() > param[key][i,1] - param[key][i,0]/2.:
                    return False
        else:
            if self._age_unq.max() < param[key][-1,1] + param[key][-1,0]/2.:
                if not nu.allclose(param[key][-1,1] + param[key][-1,0]/2,self._age_unq.max()):
                    return False
        #check if length is less than age_unq.ptp()
        if param[key][:,0].sum() > self._age_unq.ptp():
            if not nu.allclose( param[key][:,0].sum(),self._age_unq.ptp()):
                return False
        #make sure age is in bounds
        
        #passed all tests'''
        return True


        
if __name__=='__main__':
    #must be in GC dir
    import pymultinest as nest
    if not os.path.exists("chains"): os.mkdir("chains")
    import runit
    data = runit.make_usable('List/NGC104_a_1.fits')
    
    fun = multinest_fit(data,spec_lib='bc03')
    nest.run(fun.lik,fun.prior,10,init_MPI = True)
