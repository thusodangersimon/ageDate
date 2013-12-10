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
Sandbox develoment programs
"""
import likelihood_class as lik
import numpy as nu
import pylab as lab
from matplotlib.animation import FuncAnimation
import cPickle as pik

def make_chi(flux,spec,t,z,del_index):
    chi = nu.zeros_like(t)
    d = nu.vstack((t.ravel(),z.ravel())).T
    for i in range(d.shape[0]):
        index = nu.unravel_index(i,t.shape)
        chi[index] = nu.sum((spec[i,del_index]-flux)**2)/float(len(flux)-3)
    return chi

class anim(object):

    def __init__(self, t, z, chi, wave, flux):
        #set all things needed for making animation
        self.t = t
        self.z = z
        self.wave = wave
        self.flux = flux
        self.chi = chi
        self.fig = lab.figure()
        self.plt_spec = self.fig.add_subplot(211)
        self.plt_chi = self.fig.add_subplot(212)
        self.plt_chi.set_xlabel('$log_{10}(age)$')
        self.plt_chi.set_ylabel('$log_{10}(Z)$')
        self.plt_spec.set_xlabel('$\lambda$ (A)')
        self.plt_spec.set_ylabel('Normalized flus')
        self.plt_spec.set_xlim((2000,10000))
        self.fig.canvas.draw()

    def make_im(self,j):        
        #generator that does plotting
        i = self.chi[j]
        self.plt_chi.clear()
        self.plt_chi.set_xlabel('$log_{10}(age)$')
        self.plt_chi.set_ylabel('$log_{10}(Z)$')
        self.plt_chi.pcolormesh(self.t,self.z,i[0])
        self.plt_spec.set_title('Total Information content is %2.2f'%i[1])
        #plot spectrum
        if len(self.plt_spec.lines) > 0:
            self.plt_spec.lines.pop(0)
        self.plt_spec.plot(self.wave[i[2]],self.flux[i[2]],'b.')
    
        #fig.canvas.draw()
def get_information(data,spec):
    #calculates information content of data from spec
    ###make information 
    norm = nu.ones_like(spec[:,0])
    #find normalization
    for i in range(1,len(spec)):
        norm[i] = nu.sum(data[:,1]*spec[i,:])/nu.sum(spec[i,:]**2)
    #replace nans with 0
    norm[nu.isnan(norm)] = 0.
    #get normalization for data
    #make probabity mass function matrix
    pmf = nu.zeros_like(spec)
    for i in xrange(spec.shape[1]):
        p = nu.copy(spec[:,i]*norm)
        #histogram
        h=nu.histogram(p,bins=nu.sort(p))
        H = []
        for j in h[0]:
            H.append(j)
        H[-1] /= 2.
        H.append(H[-1])
        unsorted = H/nu.float64(sum(H))
        pmf[:,i] = unsorted[nu.argsort(nu.argsort(p))]

    #set minimun prob
    pmf[pmf == 0] = 10**-99
    #find infomation content of data
    H = []
    for i in xrange(spec.shape[1]):
        sorted_spec = nu.sort(spec[:,i]*norm)
        arg_sort = nu.argsort(spec[:,i]*norm)
        j = nu.searchsorted(sorted_spec,data[:,1][i])
        if j == sorted_spec.shape[0]:
            H.append(pmf[arg_sort,i][j-1])
        else:
            H.append(pmf[arg_sort,i][j])
        
    return nu.asarray(H)

def shannon(p):
    return -nu.log10(p)*p

def mod_shannon(p):
    return -nu.log10(p)*(1-p)
        
def main(data,outmov):
    #entropy calculation with plots
    fun = lik.VESPA_fit(data,spec_lib='bc03')
    
    SSP = fun.SSP
    ages = fun._age_unq
    metal = nu.linspace(fun._metal_unq.min(),fun._metal_unq.max(),10)

    t,z = nu.meshgrid(ages,metal)
    spec = []
    d = nu.vstack((t.ravel(),z.ravel())).T
    for i in d:
        try:
            spec.append(SSP.get_sed(10**(i[0]-9),10**i[1]))
        except:
            spec.append(SSP.get_sed(round(10**(i[0]-9)),10**i[1]))

    #make array
    spec = nu.asarray(spec)
    #match wavelenth with data
    if not nu.all(nu.sort(SSP.sed_ls) == SSP.sed_ls):
        #check if sorted
        wave = SSP.sed_ls[::-1]
        spec = spec[:,::-1]
    else:
        wave = SSP.sed_ls
    new_spec = nu.zeros((len(d),len(data)))
    for i in xrange(len(d)):
        new_spec[i,:] = nu.interp(data[:,0],wave,spec[i,:])
    spec = new_spec
    H = get_information(data,spec)
    
    #make animation of how information changes likelihood
    #get spectra
    chi = []
    flux = data[:,1]
    #how i think the enropy should look -sum((1-p)*log(p))
    #tot_infor = nu.sum(mod_shannon(H))
    #H = mod_shannon(H)
    H = shannon(H)
    wave =data[:,0]
    del_index = flux == flux
    print 'Making images'
    for i in xrange(len(wave)):
        index = nu.nanargmin(H)
        H[index] = nu.nan
        del_index[index] = False
        chi.append([make_chi(flux[del_index],spec,t,z,del_index),nu.nansum(H),nu.copy(del_index)])
    pik.dump((chi,z,t,wave,flux),open('temp.pik','w'),2)
    print 'Saving animations as movie'
    #make animation
    an = anim(t, z, chi, wave, flux)
    ani = FuncAnimation(an.fig,an.make_im,frames = len(chi))
    ani.save(outmov+'.mp4')
    #lab.show()

#delyed rejection MCMC

def delayed_mcmc(fun, option,bins='1', burnin=5*10**3, birth_rate=0.5,max_iter=10**5, seed=None, fail_recover=True):
    '''Delayed rejection'''
    if True:
        active_param, sigma = {},{}
        param,chi = {},{}
        Nacept, Nreject = {},{}
        acept_rate, out_sigma = {},{}
        bayes_fact = {} #to calculate bayes factor
        #simulated anneling param
        T_cuurent = {}
        for i in fun.models.keys(): ####todo add random combination of models
                active_param[i], sigma[i] = fun.initalize_param(i)
        #start with random model
        #bins = nu.random.choice(fun.models.keys())

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
        eff = -999999
    while option.iter_stop:
        #show status of running code
        if T_cuurent[bins] % 501 == 0:
            show = ('acpt = %.2f,log lik = %e, bins = %s, steps = %i,ESS = %2.0f'
                    %(acept_rate[bins][-1],chi[bins][-1],bins, option.current,eff))
            print show
            sys.stdout.flush()
			
        #sample from proposal distiburtion        
        active_param[bins] = fun.proposal(active_param[bins], sigma[bins])
        #calculate new model and chi
        chi[bins].append(0.)
        chi[bins][-1] = fun.lik(active_param,bins) + fun.prior(active_param,bins)
        #just lik part
        a = (chi[bins][-1] - chi[bins][-2])/2.
        #simulated anneling
        #a /= SA(T_cuurent[bins],burnin,abs(T_start),T_stop)
        #put temperature on order of chi calue
        '''if T_start < chi[str(bins)][-1]:
            T_start = chi[str(bins)][-1]+0'''
        #metropolis hastings
        #print a
        if nu.exp(a) > nu.random.rand():
            #acepted
            param[bins].append(active_param[bins].copy())
            Nacept[bins] += 1
        else:
            #rejected
            #delayed rejection
            temp_param,temp_chi = active_param[bins].copy(), nu.copy(chi[bins][-1])
            print 'first', temp_chi
            for _i in range(50):
                temp_param,acc, temp_chi = delayed_rejection(
                    temp_param,temp_chi,param[bins][-1] ,chi[bins][-2]
                    , sigma[bins], fun)
                if acc:
                    active_param[bins] = temp_param.copy()
                    chi[bins][-1] = nu.copy(temp_chi)
                    break
            #if accepted break
            if acc:
                #accepted
                param[bins].append(active_param[bins].copy())
                Nacept[bins] += 1
               
            else:
                #still rejected
                param[bins].append(param[bins][-1].copy())
                active_param[bins] = param[bins][-1].copy()
                chi[bins][-1] = nu.copy(chi[bins][-2])
                Nreject[bins]+=1
        
        ###########################step stuff
        #t_step.append(Time.time())
        if T_cuurent[bins] < burnin + 5000 or acept_rate[bins][-1]<.11:
            #only tune step if in burn-in
            sigma[bins] =  fun.step_func(acept_rate[bins][-1] ,param[bins], sigma, bins)

        ##############################convergece assment
       
        ##############################house keeping
        #t_house.append(Time.time())
        j+=1
        option.current += 1
        acept_rate[bins].append(nu.copy(Nacept[bins]/(Nacept[bins]+Nreject[bins])))
        out_sigma[bins].append(sigma[bins][:])
        ####end
        if option.current > 10**5:
             option.iter_stop = False


    return param, chi, acept_rate , out_sigma, param.keys()


def gr_convergence(relevantHistoryEnd, relevantHistoryStart):
    """
    Gelman-Rubin Convergence
    Converged when sum(R <= 1.2) == nparam
    """
    start = relevantHistoryStart
    end = relevantHistoryEnd
    N = end - start
    if N==0:
        return  np.inf*np.ones(self.nchains)
    N = min(min([len(self.seqhist[c]) for c in range(self.nchains)]), N)
    seq = [self.seqhist[c][-N:] for c in range(self.nchains)]
    sequences = array(seq) #this becomes an array (nchains,samples,dimensions)
    variances  = var(sequences,axis = 1)#array(nchains,dim)
    means = mean(sequences, axis = 1)#array(nchains,dim)
    withinChainVariances = mean(variances, axis = 0)
    betweenChainVariances = var(means, axis = 0) * N
    varEstimate = (1 - 1.0/N) * withinChainVariances + (1.0/N) * betweenChainVariances
    R = sqrt(varEstimate/ withinChainVariances)
    return R

def delayed_rejection(xi, pxi,zi, zprob, sigma, fun):
    """(walk_state, walk_posterior,original_state,org_postier,step, lik_object) -> (params,accepted(bool),likihood)
    Generates a second proposal based on rejected proposal xi
    """
    #make step
    while True:
        #generate new point
        zdr = fun.proposal(xi,sigma)
        #check if in prior
        if nu.isfinite(fun.prior({'1':zdr},'1')):
            break
    #get next proposal
    #propphi_zdr = self._prop_phi([zdr])
    #calc lik
    zdrprob = fun.lik({'1':zdr},'1') + fun.prior({'1':zdr},'1')
    #acceptance prob for new param a(zdrprob[0],zprob)
    
    alpha2 = min(nu.exp(-(pxi-zdrprob)/2.)*(1-alpha1(zdrprob,zprob))/(1-alpha1(pxi, zprob)), 1)
    if nu.isnan(alpha2):
        alpha2 = 0.
    print pxi, zdrprob,zprob, alpha2
    #test acceptance
    if nu.random.rand() < alpha2:
        return zdr, 1, zdrprob
    else:
        return xi,0,pxi

def alpha1(old,new):
    '''returns acceptance problablity of min(1,old/new).
     old and new are log probs'''

    return min((1.,nu.exp((new-old)/2.)))

import ezgal as gal
from glob import glob

'''makes spectrum for use in fitting'''

def make_ssp(nparam,noise=False):
    '''Uses lik.VESPA.SSP (EZgal) to make a few spetra with linearly increasing metalicity
    for use of fitting. saves param array and spectrum for comparison. Noise will add
    gaussian noise with level of parameter
    '''
    #get ezgal wrapper
    SSP = get_SSP()
    SSP.is_matched =True
    metals = nu.log10(nu.float64(SSP.meta_data['met']))
    metals.sort()
    ages = nu.log10( SSP.sed_ages)
    ages = ages[nu.isfinite(ages)] 
    #make ssp with linerar increating metalicity
    metal,age,norm,ssp = [],[],[],[]
    age = nu.random.rand(nparam)*ages.ptp()+ages.min()
    age.sort()
    if nparam == 1:
        slope = 1.
    else:
        slope = metals.ptp() / age.ptp()
    b = metals.min()- slope*ages.min()
    for i in xrange(nparam):
        metal.append( slope * ages[i] + b)
        norm.append(nu.random.rand())
    metal = nu.round(metal,3)
    #normalize normalizations
    norm = nu.asarray(norm) / nu.sum(norm)
    #get ssp and add together
    out_spec = SSP.get_sed(10**(age[0]-9), 10**metal[0]) * norm[0]
    for i in xrange(1,nparam):
        out_spec += SSP.get_sed(10**(age[i]-9), 10**metal[i]) * norm[i]
    #select wavelength range
    if not isSortted(SSP.sed_ls):
        index = nu.argsort(SSP.sed_ls)
        out_spec  = out_spec[index]
        wave = SSP.sed_ls[index]
    else:
        wave = SSP.sed_ls
    index = nu.searchsorted(wave,[2000,10**4])
    out = nu.vstack((wave[index[0]:index[1]],out_spec[index[0]:index[1]])).T 
    #make parameter output
    return out, param_out(nparam,age,metal,norm)


def isSortted(x):
    '''Retruns True is x is sorted from smallest to largest'''
    for i in xrange(1,len(x)):
        if x[i-1] > x[i]:
            return False
    return True
    
def make_burst(nparam, noise=False):
    '''Uses Galvel or EZGAL.CSP to make a burst and delay it'''
    pass
    
def get_SSP(spec_lib='cb07',imf='salp',spec_lib_path='/home/thuso/Phd/stellar_models/ezgal/'):

    '''gets wrapper class for modes'''
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
    return SSP

def param_out(nparam,age,metal,norm):
    '''Turns params into something that can be read by vespa'''
    out = []
    for i in xrange(nparam):
        out.append([.00001,age[i],metal[i],norm[i]])
    return {str(nparam):{'gal':nu.asarray(out)}}
    
if __name__ == '__main__':
    '''test delayed rejection'''

    
    import mpi_top as hy
    import cPickle as pik
    import numpy as nu
    import os,sys

    data,param = make_ssp(1)
    '''while True:
        try:
            data,param = make_ssp(1)
            break
        except ValueError:
            print 'gerer' '''

    fun = lik.VESPA_fit(data,spec_lib='cb07',use_dust=False,use_losvd=False)
    fun.SSP.is_matched = True
    top =hy.Topologies('single')
    out = delayed_mcmc(fun,top,max_iter=1000)
    pik.dump((data,param,out),open('finished_%f.pik'%nu.random.rand(),'w'),2)
