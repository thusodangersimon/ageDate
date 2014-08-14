import emcee_lik
import emcee
import numpy as nu
import LRG_lik as lik
import mpi_top
import cPickle as pik
from os import path
from glob import glob
import mpi4py.MPI as mpi
import ipdb

'''Makes fake SSPs and CSP for fitting with different noise'''


def make_SSP_grid(db_path, save_path, num_gal=25):
    '''Burst model has tau=0 as SSP so uses tau=0 for all models'''
    #make fake data but correct wavelegth
    wave = make_wavelenght(3500, 9000)
    fake_data = {'fake':nu.vstack((wave, nu.ones_like(wave))).T}
    fun = lik.LRG_Tempering(fake_data, db_path, have_dust=True, have_losvd=True)
    # make 125 gal
    for gal in xrange(num_gal):
        param = fun.initalize_param(1)[0]
        param['tau'] = 0.
        param['redshift'] = 0.
        #param['age'] = gal
        #param[['metalicity', 'normalization', '$T_{bc}$', '$T_{ism}$','$\sigma$', '$V$']] = -3.9019, 33.501257 ,0, 0, 0, 0
        temp = fun.lik({'burst':{'fake':param}},'burst', True)
        for Lik, index, spec in temp:
            pass
        #lab.figure()
        #lab.title(str(gal))
        #lab.plot(spec[0])
        # put noise and save
        snr = nu.random.randint(30,90)
        noise = nu.random.randn(len(spec[0]))*spec[0].mean()/float(snr)
        out = nu.vstack((wave, spec[0] + noise, nu.abs(noise))).T
        pik.dump((param, snr, out), open(path.join(save_path,'%i_ssp.pik'%gal),
                                          'w'), 2)
    
def make_CSP_grid(db_path, save_path, num_gal=25):
    '''Any tau >0'''
    #make fake data but correct wavelegth
    wave = make_wavelenght(3500, 9000)
    fake_data = {'fake':nu.vstack((wave, nu.ones_like(wave))).T}
    fun = lik.LRG_Tempering(fake_data, db_path, have_dust=True, have_losvd=True)
    # make 125 gal
    for gal in xrange(num_gal):
        param = fun.initalize_param(1)[0]
        #param['tau'] = 0.
        param['redshift'] = 0.
        #param['age'] = gal
        #param[['metalicity', 'normalization', '$T_{bc}$', '$T_{ism}$','$\sigma$', '$V$']] = -3.9019, 33.501257 ,0, 0, 0, 0
        temp = fun.lik({'burst':{'fake':param}},'burst', True)
        for Lik, index, spec in temp:
            pass
        #lab.figure()
        #lab.title(str(gal))
        #lab.plot(spec[0])
        # put noise and save
        snr = nu.random.randint(30,90)
        noise = nu.random.randn(len(spec[0]))*spec[0].mean()/float(snr)
        out = nu.vstack((wave, spec[0] + noise, nu.abs(noise))).T
        pik.dump((param, snr, out), open(path.join(save_path,'%i_csp.pik'%gal),
                                          'w'), 2)

def make_wavelenght(lam_min, lam_max):
    '''makes random wavelenth coverage using inputs'''
    return nu.arange(lam_min, lam_max)

def do_fit(fit_dir, db_path):
    '''Fits data using emcee'''
    comm = mpi.COMM_WORLD
    pool = emcee_lik.MPIPool_stay_alive(loadbalance=True)
    files = glob(path.join(fit_dir, '*.pik'))
    # start fits
    for gal in files:
        comm.barrier()
        data = {}
        temp = pik.load(open(gal))
        data[gal] = temp[-1]
        data = comm.bcast(data, root=0)
        posterior = emcee_lik.LRG_emcee(data, db_path, have_dust=True,
                                            have_losvd=True)
        posterior.init()
        nwalkers = 4 *  posterior.ndim()
        if pool.is_master():
            pos0 = posterior.inital_pos(nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim() ,
                                            posterior, pool=pool)
            iterations = 10 * 10**3
            pos, prob = [], []
            i = 0
            for tpos, tprob, _ in sampler.sample(pos0, iterations=iterations):
                print '%i out of %i'%(i, iterations)
                pos.append(tpos)
                prob.append(tprob)
                i+=1
            pik.dump((temp,(pos, prob)), open(gal + '.res', 'w'), 2)
        else:
            pool.wait(posterior)
        
        
if __name__ == "__main__":
    '''makes fake spectra'''
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    if not path.exists('/home/thuso/Phd/experements/hierarical/LRG_Stack/stacked_real/Fake_SSP'):
        make_SSP_grid(db_path, '/home/thuso/Phd/experements/hierarical/LRG_Stack/stacked_real/Fake_SSP')
        make_CSP_grid(db_path, '/home/thuso/Phd/experements/hierarical/LRG_Stack/stacked_real/Fake_CSP')
    else:
        fit_dir = '/home/thuso/Phd/experements/hierarical/LRG_Stack/stacked_real/Fake_SSP'
        do_fit(fit_dir, db_path)
        fit_dir = '/home/thuso/Phd/experements/hierarical/LRG_Stack/stacked_real/Fake_CSP'
        do_fit(fit_dir, db_path)
