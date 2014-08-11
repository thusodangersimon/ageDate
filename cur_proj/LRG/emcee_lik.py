import numpy as nu
import emcee
from LRG_lik import Multi_LRG_burst
import pandas as pd
import ipdb
from mpi4py import MPI
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

class LRG_emcee(Multi_LRG_burst):
    '''makes my likelihood work with emcee sampler'''

    def __call__(self, param):
        '''makes ok for my input'''
        #print param
        #ipdb.set_trace()
        p = pd.DataFrame([param], columns=self._columns, index=[0])
        p = {self._bins: { self._gal: p}}
        # check prior first
        prior = self.prior(p, self._bins)
        for Prior, gal in prior:
            if not nu.isfinite(Prior):
                return -nu.inf
        #likihood
        lik = self.lik(p, self._bins , normalize=False)
        for Lik, index in lik:
            pass
        return Lik + Prior
        

    def ndim(self):
        '''returns number of dimensions'''
        inital = self.initalize_param(self.data.keys()[0])[1]
        return inital.shape[0]

    def init(self):
        '''sets all the things needed for emcee calling'''
        # coulnms
        self._columns = self.initalize_param(self.data.keys()[0])[0].columns
        self._columns = list(self._columns)
        # table name
        self._bins = self._table_name
        # galaxy name
        self._gal = self.data.keys()[0]
        
    def inital_pos(self, nwalkers):
        '''returns vector of valid positons for emcee
        returns parameters in shape (nwalkers, dim)'''
        
        out = []
        for walker in xrange(nwalkers):
            kill = False
            while True:
                inital = {self._bins:{self._gal:self.initalize_param(self.data.keys()[0])[0]}}
                prior = self.prior(inital, self._bins)
                for Prior, gal in prior:
                    # check if valid
                    if nu.isfinite(Prior):
                        kill = True
                        break
                if kill:
                    break
            out.append(nu.asarray(inital[self._bins][self._gal])[0])
        # vectorize
        return  nu.asarray(out)

class MPIPool_stay_alive(emcee.utils.MPIPool):
    '''
    Like emcee pool object, but has own instance of likeihood function.
    Otherwise I get database problems'''

    def wait(self, lik_fun=None):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if self.debug:
                print("Worker {0} got task {1} with tag {2}."
                      .format(self.rank, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, emcee.mpi_pool._close_pool_message):
                if self.debug:
                    print("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, emcee.mpi_pool._function_wrapper):
                # run local copy
                if lik_fun is None:
                    self.function = task
                else:
                    self.function = lik_fun
                if self.debug:
                    print("Worker {0} replaced its task function: {1}."
                          .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            result = self.function(task)
            if self.debug:
                print("Worker {0} sending answer {1} with tag {2}."
                      .format(self.rank, result, status.tag))
            self.comm.isend(result, dest=0, tag=status.tag)

        
if __name__ == '__main__':
    from glob import glob
    import cPickle as pik
    import mpi4py.MPI as mpi
    # get data from local dir
    files = glob('*.pik')
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    #emcee stuff
    for file in files:
        data = {}
        temp = pik.load(open(file))
        data[file] = temp[-1]
        posterior = LRG_emcee(data, db_path, have_dust=True, have_losvd=True)
        posterior.init()
        nwalkers = 2 *  posterior.ndim()
        # make inital position
        pos0 = posterior.inital_pos(nwalkers)
        #make sampler
        if mpi.COMM_WORLD.size > 1:
            pool = MPIPool_stay_alive(loadbalance=True)
            if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait(posterior)
                continue
            sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim() ,
                                            posterior, pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim() ,
                                            posterior, threads=10)
        # save sample line by lin

        #pos, prob, rstate= sampler.sample(pos0, iterations=1000)
        fn = file + '.csv'
        f = open(fn, "w")
        #f.close()
        iterations = 50 * 10**3
        i = 0
        autocorr = 0.
        accept = 0.
        for pos, prob, rstate in sampler.sample(pos0, iterations=iterations):
            print 'Postieror=%e acceptance=%2.1f autocorr=%2.1f %i out of %e'%(max(prob), 100*accept,
                                                                  autocorr
                                                            ,i ,iterations)
            if i % 100 == 0:
                autocorr = nu.nanmean(sampler.get_autocorr_time())

            accept = sampler.acceptance_fraction
            accept = nu.mean(accept)
            f.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
            f.write("\n")
            f.flush()
            i += 1
        f.close()
        pl.figure()
        for k in xrange(nwalkers):
            pl.plot(sampler.chain[k, :, 0])
        pl.xlabel("time")
        pl.savefig("eggbox_time.png")

        pl.figure(figsize=(8,8))
        x, y = sampler.flatchain[:,0], sampler.flatchain[:,1]
        pl.plot(x, y, "ok", ms=1, alpha=0.1)
        #pl.savefig("eggbox_2d.png")

        fig = pl.figure()
        ax = fig.add_subplot(111, projection="3d")

        for k in xrange(nwalkers):
            x, y = sampler.chain[k,:,0], sampler.chain[k,:,1]
            z = sampler.lnprobability[k,:]
            ax.scatter(x, y, z, marker="o", c="k", alpha=0.5, s=10)
        pl.show()
