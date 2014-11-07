import likelihood_class as lik
import numpy as np
import database_utils as util
import interp_utils as interp
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import itertools
import sys
from mpi4py import MPI as mpi
import scipy.stats as stats_dist
import spectra_utils as ag
import MC_utils as MC
import pandas as pd
import ipdb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
'''Likelyhood and functions needed for MCMC for LRGS'''



class Multi_LRG_burst(lik.Example_lik_class):
    '''Single core, LRG likelihood function'''
    def __init__(self, data, db_name='burst_dtau_10.db', have_dust=False,
                 have_losvd=False):
        self.has_dust = have_dust
        self.has_losvd = have_losvd
        self.db_name = db_name
        #check if data is right type
        
        self.data = {}
        #get mean data values to 1
        self.norm = {}
        self.norm_prior = {}
        for i in data:
            self.norm[i] = 1./data[i][:,1].mean()
            self.norm_prior[i] = np.log10(self.norm[i])
            self.data[i] = data[i].copy()
            self.data[i][:,1] *= self.norm[i]
            if self.data[i].shape[1] == 3:
                # Propagate the uncertany
                self.data[i][:,2] *= self.norm[i]
        self.db = util.numpy_sql(db_name)
        self._table_name = self.db.execute('select * from sqlite_master').fetchall()[0][1]
        # Tell which models are avalible and how many galaxies to fit
        self.models = {'burst': data.keys()}
        # Get param range (tau, age, metal)
        self.param_range = []
        for column in ['tau', 'age', 'metalicity']:
            self.param_range.append(np.sort(np.ravel(self.db.execute(
                'Select DISTINCT %s FROM %s'%(column,self._table_name)).fetchall())))
        self._hull = None
        # make resolution
        self._define_resolu()
        
    def _define_resolu(self):
        # resolution for CB07 and BC03 in km/s
        self.resolu = {}
        for gal in self.data:
            self.resolu[gal] = 3. * 299792.458 / self.data[gal][:,0].mean()

    def _make_hull(self):
        '''Make convex hull obj for telling if param is in range'''
        # Make all points in param space
        points = np.asarray([point for point in
                             itertools.product(*self.param_range)])
        self._hull = Delaunay(points)

    def is_in_hull(self, point):
        '''Checks if a point in in the param range Retuns bool'''
        if not isinstance(self._hull, Delaunay):
            self._make_hull()
        return self._hull.find_simplex(point) >= 0
    
    def lik(self, param, bins, return_model=False, normalize=True):
        '''Calculates log likelyhood for burst model'''
        for gal in param[bins]:
            #ipdb.set_trace()
            # get interp spectra
            #check if points are in range
            columns = ['tau', 'age', 'metalicity']
            if self.is_in_hull(param[bins][gal][columns]):
                # check if db works
                try:
                    self.db.execute('select * from sqlite_master').fetchall()[0][1]
                except: #ProgrammingError:
                    #make new database connection
                    self.db = util.numpy_sql(self.db_name)
                    
                spec = tri_lin_interp(self.db,
                    param[bins][gal][columns], self.param_range)
            else:
                if return_model:
                    yield -np.inf, gal, []
                else:
                    yield -np.inf, gal
                continue
            if normalize:
                model = {'wave':spec[:,0],0: spec[:,1]}
            else:
                model = {'wave':spec[:,0],
                     0: spec[:,1] * 10**param[bins][gal]['normalization'].iat[0]}
            # Redshift
            model['wave'] = ag.redshift(model['wave'],
                                        param[bins][gal]['redshift']) 
            # Dust
            if self.has_dust:
                columns = ['$T_{bc}$','$T_{ism}$']
                model = ag.dust(param[bins][gal][columns].iloc[0],
                                model)
        
            # LOSVD
            if self.has_losvd:
                # wave range for convolution
                wave_range = [self.data[gal][:,0].min(),
                              self.data[gal][:,0].max()]
                # check if resolution has been calculated
                columns = ['$\\sigma$','$V$','$h_3$','$h_4$']
                send_param = param[bins][gal][columns].iloc[0]
                model = ag.LOSVD(model, send_param,
                                   wave_range, self.resolu[gal])
            #match data wavelengths with model
            try:
                model = ag.data_match(self.data[gal], model)
            except:
                model = ag.data_match(self.data[gal], model, rebin=False)
            #calculate map for normalization
            norm = ag.normalize(self.data[gal], model[0])
            self.norm_prior[gal] = np.log10(norm)
            if normalize:
                model[0] *= norm
            # Calc liklihood
            if self.data[gal].shape[1] >= 3:
                # Has Uncertanty
                out_lik = stats_dist.norm.logpdf(
                self.data[gal][:,1], model[0], self.data[gal][:,2])
            else:
                #no uncertanty or bad entry
                out_lik = stats_dist.norm.logpdf(
                    self.data[gal][:,1], model[0])
            if return_model:
                yield out_lik.sum(), gal, model
            else:
                yield out_lik.sum(), gal

    def step_func(self, step_crit, param, step_size, itter):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray
        param should be a pandas.DataFrame
        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        diag = np.diagonal(step_size)
        
        '''if np.all(step_crit > 0.3) and np.all(diag < 8.):
            step_size *= 1.05
        elif np.all(step_crit < 0.2) and np.any(diag > 10**-6):
                step_size /= 1.05'''
        # Switch statement
        if step_crit < 0.001:
            # reduce by 90 percent
            step_size *= 0.1
        elif step_crit < 0.05:
            # reduce by 50 percent
            step_size *= 0.5
        elif step_crit < 0.2:
            # reduce by ten percent
            step_size *= 0.9
        elif step_crit > 0.95:
            # increase by factor of ten
            step_size *= 10.0
        elif step_crit > 0.75:
            # increase by double
            step_size *= 2.0
        elif step_crit > 0.5:
            # increase by ten percent
            step_size *= 1.1
        #cov matrix
        if itter % 500 == 0 and itter > 0.:
            
            Tstep_size = np.cov(np.vstack(param[-2000:]).T)
            #make sure not stuck
            index = np.isclose(Tstep_size, 0, rtol=1e-6) == False
            step_size[index] = Tstep_size[index]
        # make sure none of the diagonal elements are smaller than 10**-4
        diag = np.diagonal(step_size)
        index = np.where(diag < 10**-4)[0]
        for i in index:
            step_size[i,i] = 10**-4
        return step_size

    def initalize_param(self, bins):
        dtype = []
        param = []
        # make tau, age, and metal array
        dtype.append(('tau',float))
        dtype.append(('age',float))
        dtype.append(('metalicity', float))
        dtype.append(('normalization',float))
        dtype.append(('redshift', float))
        #uniform dist for everything except redshift
        param = [np.random.rand()*i.ptp() + i.min() for i in self.param_range]
        #norm
        param.append(np.random.rand()*np.mean(self.norm_prior.values())+25)
        #redshift
        param.append(0.)
        if self.has_dust:
            dtype.append((r'$T_{bc}$',float))
            dtype.append((r'$T_{ism}$', float))
            #uniform between 0 ,4
            param += [np.random.rand()*4 for i in range(2)]
        if self.has_losvd:
            dtype.append((r'$\sigma$', float))
            dtype.append((r'$V$', float))
            dtype.append((r'$h_3$', float))
            dtype.append((r'$h_4$', float))
            #sigma and v
            param += [np.random.rand()*3 for i in range(2)]
            # h3 and h4 off for now
            param += [0. for i in range(2)]
        # create array and assign values
        out_param = np.empty(1, dtype=dtype)
        for index,elmt in enumerate(param):
            out_param[0][index] = elmt
        out_param = pd.DataFrame(out_param)
        return out_param, np.eye(len(param))*.01
    
    def prior(self, param, bins):
        '''Calculates priors of all parameters'''
        # Uniform
        # tau, age, metalicity
        for gal in param[bins]:
            out_lik = 0.
            for i,ran in enumerate(self.param_range):
                if i ==1:
                    #age
                    if param[bins][gal].iloc[0][i] > ran.max():
                        out_lik += -np.inf
                    else:
                        out_lik += stats_dist.norm.logpdf(param[bins][gal].iloc[0][i], 9.65, 0.43)
                else:
                    #other
                    out_lik += stats_dist.uniform.logpdf(param[bins][gal].iloc[0][i],
                                                         ran.min(),ran.ptp())
                                
            # Redshift
            out_lik += stats_dist.uniform.logpdf(param[bins][gal]['redshift'],
                                                 0,.055)
            # make sure norm isn't too small
            norm = param[bins][gal]['normalization'] < -10
            
            if norm.bool():
                out_lik += -np.inf
            else:
                out_lik += stats_dist.norm.logpdf(param[bins][gal]['normalization'],
                                              self.norm_prior[gal], 10)
            
            if self.has_dust:
                out_lik += stats_dist.uniform.logpdf(param[bins][gal][['$T_{ism}$',
                                                '$T_{bc}$']],0,4).sum()
            if self.has_losvd:
                out_lik += stats_dist.uniform.logpdf(param[bins][gal]['$\\sigma$'],
                                                 0,3.2)
                out_lik += stats_dist.uniform.logpdf(param[bins][gal]['$V$'],0,4)
            yield out_lik, gal
    
    def model_prior(self, model):
        return 0.
    
    def proposal(self, Mu, Sigma):
        # get out of rec array
        out = {}
        for gal in Mu:
            mu = Mu[gal].values[0]
            try:
                out[gal] =  np.random.multivariate_normal(mu, Sigma[gal])
            except:
                pass
                #ipdb.set_trace()
            # put back into DataFrame
            out[gal] = pd.DataFrame(out[gal],Mu[gal].columns).T
            #set h3 and h4 to 0
            if self.has_losvd:
                out[gal]['$h_3$'] = 0.
                out[gal]['$h_4$'] = 0.
                out[gal]['$V$'] = 0.
        return pd.Panel(out)

    def exit_signal(self):
        '''Does any wrap ups before exiting'''
        pass

    def send_fitting_data(self):
        '''Send data to workers'''
        # Not needed with out mpi so return None
        return None
        
class LRG_mpi_lik(Multi_LRG_burst):
    '''Does LRG fitting and sends likelihood cal to different
    processors'''
    def __init__(self,  data, db_name='burst_dtau_10.db', have_dust=False,
                 have_losvd=False, use_mpi=False):
        # Set up like Muliti
        Multi_LRG_burst.__init__(self, data, db_name, have_dust, have_losvd)
        self._comm = mpi.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        if self._size > 1:
            # get workers
            if self._rank == 0:
                self.get_workers()
                self.lik = self.lik_root
                 
            else:
                self.calc_lik = self.lik
                self.lik = self.lik_worker
        else:
            self.get_workers()
            self._size = 5

    def lik_worker(self):
        '''Does lik calculation for sent galaxies, returns likelihood'''
        # keep calculating till run is over
        self._proc_name = mpi.Get_processor_name()
        while True:
            # Send ready signal
            self._comm.send([{'rank': self._rank,
                              'name':self._proc_name}], dest=0, tag=1)
            status = mpi.Status()
            recv = self._comm.recv(source=0, tag=mpi.ANY_TAG, status=status)
            # check if should quit
            if  status.tag == 2:
               #print 'trying to quit worker %i'%self._rank
               self._comm.isend([], dest=0, tag=2)
               sys.exit(0)
            # recived data
            if status.tag == 10:
                #Multi_LRG_burst.lik
                gal = recv[1]
                Lik = self.calc_lik({0:{gal:recv[0]}},0)
                out_lik = Lik.next()[0]
                self._comm.send((gal, out_lik, self.norm_prior[gal]),
                                dest=0, tag=10)
            if status.tag == 11:
                #Multi_LRG_burst.lik return model too
                gal = recv[1]
                Lik = self.calc_lik({0:{gal:recv[0]}}, 0, True)
                out_lik = Lik.next()
                self._comm.send((gal, out_lik[0], self.norm_prior[gal],out_lik[2]),
                                dest=0, tag=11)
            if status.tag == 5:
                #do nothing
                pass
            if status.tag == 3:
                # recive fitting data
                print "%i recived new data from root"%self._rank
                self.data, self.norm_prior, self.norm = recv
                # change resolution
                self._define_resolu()
                
                
    def lik_root(self, param, bins, return_model=False):
        '''Root lik calculator, manages workers for likelihood calcs'''
        # feed all queued jobs to workers
        # tag 1 codes for initialization.
        # tag 10 codes for requesting more data.
        # tag 11 codes for requesting more data but return model spectra
        # tag 5 codes for doing nothing
        # Tag 2 codes for a worker exiting.
        # Tag 3 codes to send/recive new fitting data
        index = -1
        #gal = param[bins].keys()[index]
        recv_num = []
        while True:
            status = mpi.Status()
            recv = self._comm.recv(source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG,
                                   status=status)
            if status.tag == 1:
                #send data to source
                if index+1 < len(param[bins]):
                    index += 1
                    gal = param[bins].keys()[index]
                    if not return_model:
                        self._comm.send((param[bins][gal],gal),
                                        dest=status.source, tag=10)
                    else:
                        self._comm.send((param[bins][gal],gal),
                                        dest=status.source, tag=11)
                    #print 'Send %s to %i'%(gal,status.source)
                else:
                    # send nothing
                    self._comm.send([], dest=status.source, tag=5)
                    
            if status.tag == 10:
                #recive likelyhoods
                recv_gal, recv_lik, recv_norm = recv
                self.norm_prior[recv_gal] = recv_norm
                #print 'Recived %s with chi %f'%(recv_gal,recv_lik)
                yield recv_lik, recv_gal
                recv_num.append(recv_gal)
                # Check if should exit
                if len(recv_num) == len(param[bins].keys()):
                    #print 'done'
                    break
                
            if status.tag == 11:
                recv_gal, recv_lik, recv_norm, recv_model = recv
                self.norm_prior[recv_gal] = recv_norm
                #print 'Recived %s with chi %f'%(recv_gal,recv_lik)
                yield recv_lik, recv_gal, recv_model
                recv_num.append(recv_gal)
                # Check if should exit
                if len(recv_num) == len(param[bins].keys()):
                    #print 'done'
                    break
                
            if status.tag == 2:
                # Worker crashed
                self.remove_workers(status.source)
                
    def send_fitting_data(self):
        '''Sends fitting data to workers from root'''
        # check if needed
        if self._comm.size < 2:
            return None
        index = -1
        recv_num = []
        # send to all
        while True:
            status = mpi.Status()
            recv = self._comm.recv(source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG,
                                   status=status)
            if not status.tag == 1:
                print "Didn't get right signal"
                raise
            if recv[0]['rank'] in recv_num:
                continue
            recv_num.append(recv[0]['rank'])
            # Send data
            #print self.data.keys()
            self._comm.send((self.data,self.norm_prior,self.norm), dest=status.source,
                             tag=3)
            # Check if all workers have data
            if np.all(np.sort(recv_num) == np.sort(self._workers)):
                # send a 5 so they listen again
                for worker in recv_num:
                    self._comm.send([], dest=worker, tag=5)
                break
        
    def get_workers(self):
        if self._comm.Get_size() > 1:
            self._workers = range(1,self._size)
        else:
            # single processing
            self._workers = ''

    def remove_workers(self, del_work):
        if del_work in self._workers:
            self._workers.pop(self._workers.index(del_work))
            print 'Worker %i has exited'%del_work

    def exit_signal(self):
        '''Sets varables needed for mpi'''
        if not self._rank == 0 or isinstance(self._workers, str):
            return None
        # send exit signal to all workers
        while len(self._workers) > 0:
            status = mpi.Status()
            recv = self._comm.recv(source=mpi.ANY_SOURCE, tag=mpi.ANY_TAG,
                                   status=status)
            #print recv
            if status.tag == 1:
                # Send Kill Signal
                print 'Killing worker %i on %s'%(recv[0]['rank'],recv[0]['name'])
                self._comm.send([], dest=status.source, tag=2)
                
            if status.tag == 2:
                # remove worker
                self.remove_workers(status.source)

                

class LRG_Tempering(LRG_mpi_lik):
    '''parallel Tempering liklihood'''
    def __init__(self, data, db_name='burst_dtau_10.db', have_dust=False,
                 have_losvd=False):
        # initalize from old class
        LRG_mpi_lik.__init__(self, data ,db_name, have_dust, have_losvd)
        # Check data should only have 1 gal
        assert len(data.keys()) == 1, 'Only 1 gal at a time'

    def initalize_temp(self, num_temp):
        '''Initalize different tempering spaced by delta_t'''
        # make temp data from 0=no to num ofcpu =high temp
        data = {}
        key = self.data.keys()[0]
        for cpu in range(num_temp-1):
            data[key+'_%03d'%cpu] = np.copy(self.data[key])
        self.data = data
        LRG_mpi_lik.__init__(self, data ,self.db_name, self.has_dust,
                             self.has_losvd)
        

                
            
def grid_search(point, param_range):
    '''Finds points that make a cube around input point and returns them with
    their spectra'''
    if isinstance(point, dict):
        points = point.get_values()[0]
    elif isinstance(point, (list, np.ndarray)):
        points = point
    else:
        raise TypeError('Array must be a list,dictionary or np.ndarray')
    index = []
    len_array = []
    on_plane = []
    for i in range(len(points)):
        len_array.append(len(param_range[i]))
        index.append(np.searchsorted(param_range[i],points[i]))
        on_plane.append(param_range[i] == points[i])
    index = np.asarray(index)
    len_array = np.asarray(len_array)
    # check if at on an edge
    if np.any(index == 0) or np.any(index == len_array):
        com_tupple = []
        for j, i in enumerate(index):
            if i == 0:
                # param is on lower edge return 1 points that are the same
                com_tupple.append((param_range[j][i],))
            elif i == len_array[j]:
                # param is on upper edge
                com_tupple.append((param_range[j][i],))
            else:
                com_tupple.append((param_range[j][i-1], param_range[j][i]))
        
    # check if on plane
    elif np.any(np.hstack(on_plane)):
        ipdb.set_trace()
        raise NotImplementedError
    # iterate around the point
    else:
        com_tupple = [(param_range[j][i-1], param_range[j][i])
                  for j,i in enumerate(index)]
    interp_points = np.asarray([p for p in
                             itertools.product(*com_tupple)])
    return interp_points


def tri_lin_interp(db, param, param_range):
    '''Does trilinear interoplation for spectra with points (tau,age,metal).
    Returns interpolated spectra or an array of inf if param is out of range'''
    table_name = db.execute('select * from sqlite_master').fetchall()[0][1]
    # get 8 nearest neighbors
    points = grid_search(param, param_range)
    spec = []
    wave = None
    for i, point in enumerate(points):
        spec.append(db.execute('''Select spec From %s WHERE tau=? AND age=? AND
        metalicity=?'''%table_name, point).fetchone()[0])
        if wave is None:
            wave = util.convert_array(spec[-1])[:,0]
        spec[-1] = util.convert_array(spec[-1])[:,1]
    # do interpolation
    # check if on edge or less than normal spectra
    if len(spec) == 4:
        # bi-linear inter is needed
        # find which param is the same
        arange = np.arange(np.asarray(param).shape[1])
        for i in arange:
            if np.all(points[:,i] == np.asarray(param)[0][i]):
                # cut it out
                points = points[:, arange != i]
                eval_points = np.asarray(param)[:, arange != i]
                break
        spec = interp.bilinear_interpolation(points[:,0], points[:,1],spec,
                                             eval_points[0,0],eval_points[0,1])
        
    else:
        #tri-linear is needed
        spec = griddata(points, spec, param)
    out_spec = np.vstack((wave, spec)).T
    return  out_spec
    
