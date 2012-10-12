#!/usr/bin/env python
#
# Name:  RJMCMC and partical swarm
#
# Author: Thuso S Simon
#
# Date: Oct. 20 2011
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
''' Does RJMCMC with partical swarm. Trys different topologies of comunication from Mendes et al 2004 and different weighting types for ps'''

from Age_date import MC_func
from Age_RJMCMC import *
from mpi4py import MPI as mpi

def root_run(fun, topology, burnin=5000, itter=10**6, k_max=16):
    '''From MPI start, starts workers doing RJMCMC and coordinates comunication 
    topologies'''
    #start multiprocess need to make work with mpi
    work=[]
    q_talk,q_final=Queue(),Queue()
    
    for ii in range(topology._cpus):
        work.append(Process(target=vanila_swarm,
                            args=(fun,burnin , k_max,
                                 topology , ii, q_talk, q_final,)))
        work[-1].start()
    while (option.iter.value <= self._iter + self._burnin * self._cpus 
           and option.value):  
        Time.sleep(5)
        sys.stdout.flush()
        print '%2.2f percent done' %((float(option.iter.value)/
                                      (self._iter + self._burnin
                                       * self._cpus))*100.)

    #put in convergence diagnosis
    option.value=False
    #wait for proceses to finish
    count=0
    temp=[]
    while count<self._cpus:
        count+=1
        try:
            temp.append(q_final.get(timeout=5))
        except:
            print 'having trouble recieving data from queue please wait'
                
    if not temp:
        print 'Recived no data from processes, exiting'
        return False,False,False
    for i in work:
        i.terminate()
 


def vanila_swarm(fun, option, burnin=5*10**3, k_max=16, rank=0, q_talk=None, q_final=None):
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
        #[tau_ism, tau_BC ]
        active_dust = nu.random.rand(2)*4.
        sigma_dust = nu.identity(2)*nu.random.rand()*2
    else:
        active_dust = nu.zeros(2)
        sigma_dust = nu.zeros([2,2])
    #set up LOSVD
    if fun._losvd:
        #[sigma, redshift, h3, h4]
        active_losvd = nu.random.rand(4)*10
        active_losvd[1] = 0.
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
                option.parambest[kk] = nu.hstack((active_param[str(bins)],
                                               active_dust,active_losvd))[kk]
            else:
                    option.parambest[kk] = nu.nan
        print ('%i has best fit with chi of %2.2f and %i bins' 
               %(rank,option.chibest.value,bins))
        sys.stdout.flush()
        #set current swarm value
    for kk in range(len(option.swarm[rank])):
        if kk<bins*3+2+4:
            option.swarm[rank][kk] = nu.hstack((active_param[str(bins)],
                                                active_dust,active_losvd))[kk]
        else:
            option.swarm[rank][kk] = nu.nan
    option.swarmChi[rank].value = chi[str(bins)][-1]
    #start rjMCMC
    T_cuurent,Nexchange_ratio=1.0,1.0
    size=0
    j,T,j_timeleft=1,9.,nu.random.exponential(100)
    T_start,T_stop=3*10**5.,0.9
    birth_rate=.5
    out_dust_sig, out_losvd_sig = [sigma_dust], [sigma_losvd]

    while option.value:
        if option.iter.value%5000==0:
            print "hi, I'm at itter %i, chi %f from %s bins and from %i redshift of %1.2f" %(len(param[str(bins)]),chi[str(bins)][-1],bins, rank, active_losvd[1])
            sys.stdout.flush()

        #sample from distiburtion
        active_param[str(bins)] = fun.proposal(active_param[str(bins)],
                                               sigma[str(bins)])
        if fun._dust:
            active_dust = fun.proposal(active_dust,sigma_dust)
        if fun._losvd:
            active_losvd  = fun.proposal(active_losvd, sigma_losvd)
            active_losvd[1:] = 0.
        active_param[str(bins)], active_dust, active_losvd = swarm_vect(active_param[str(bins)], active_dust, active_losvd, option)
        #calculate new model and chi
        chi[str(bins)].append(0.)
        chi[str(bins)][-1],active_param[str(bins)][range(2,bins*3,3)]=fun.lik(
            active_param[str(bins)], active_dust, active_losvd)
        #sort by age
        if not nu.all(active_param[str(bins)][range(1,bins*3,3)]==
                      nu.sort(active_param[str(bins)][range(1,bins*3,3)])):
            index = nu.argsort(active_param[str(bins)][range(1,bins*3,3)])
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
                
            chi[str(bins)][-1]=nu.copy(chi[str(bins)][-2])
            Nreject[str(bins)]+=1

        ###########################step stuff
        sigma[str(bins)],sigma_dust,sigma_losvd = Step_func(acept_rate[str(bins)][-1]
                                                            ,param[str(bins)][-2000:]
                                                            ,sigma[str(bins)],
                                                            sigma_dust,
                                                            sigma_losvd,
                                                            bins, j,fun._dust, 
                                                            fun._losvd)


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

def tuning_swarm():
    '''uses aceptance rate to tune how strong to allow
    swarm param'''
    pass

def hybrid_swarm():
    '''chooses between swarm+proposal vector and only proposal verctor using
    standard rjmcmc routine'''
    pass



#########swarm functions only in this program######
def swarm_vect(active_param, active_dust, active_losvd, option):
    '''does swarm vector calculations and returns swarm*c+active'''
    tot_chi = 0
    for i in range(len(option.swarmChi)):
        tot_chi += option.swarmChi[i].value



    return out_param,out_dust,out_losvd


class Topologies(object):
    """Defines different topologies used in communication. Will probably affect
    performance if using a highly communicative topology.
    Topologies include :
    all, ring, cliques and square.
    all - every worker is allowed to communicate with each other, no buffer
    ring -  the workers are only in direct contact with 2 other workers
    cliques - has 1 worked connected to other head workers, which talks to all the other sub workers
    square - every worker is connect to 4 other workers"""


    def __init__(self,mpicomm,top = 'cliques',k_max=16):
        self.current = Value('b',True)
        self.cpu_tot = cpu_count() - 1
        self.iter_stop = Value('i',True)
        self.chibest = Value('d',nu.inf)
        self.parambest = Array('d',nu.ones(k_max * 3 + 2 + 4) + nu.nan)
        #setup comunication arrays to other workers + 1 from world
        self.swarm, self.swarmChi = [],[]
        for i in range(self.cpu_tot+1):
            self.swarm.append(Array('d',nu.ones(k_max * 3 + 2 + 4) + nu.nan))
            self.swarmChi.append(Value('d',nu.inf))
        #check if topology is in list
        if not top in ['all', 'ring', 'cliques', 'square']:
            raise ValueError('Topology is not in list.')
        if top == 'all':
            self.comm = self.All(mpicomm)
        elif top == 'ring':
            self.comm = self.Ring(mpicomm)
        elif top == 'cliques':
            self.comm = self.Cliques(mpicomm)
        elif top == 'square':
            self.comm = self.Square(mpicomm)

    def All(self, comm):
        pass
    def Ring(self, comm):
        pass
    def Cliques(self, comm):
        pass
    def Square(self, comm):
        pass


if __name__ == '__main__':
    import Age_date as ag
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        data,info,weight,dust = ag.iterp_spec(1,lam_min=4000, lam_max=8000)
        data_len = nu.array(data.shape[0],dtype=int)
        comm.Bcast(data_len, root=0)
        comm.Bcast(data, root=0)
    else:
        data_len = nu.zeros(1,dtype=int)
        comm.Bcast( data_len, root=0)
        data = nu.zeros((data_len, 2))
        comm.Bcast(data, root=0)

    Top = Topologies(comm)
    fun = MC_func(data)
    fun.autosetup()
    root_run(fun.send_class, Top, itter=10**6, k_max=16)
