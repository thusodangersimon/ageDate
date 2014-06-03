import likelihood_class as lik
import numpy as nu
import database_utils as util
import interp_utils as interp
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import itertools
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
        #check if data is right type
        
        self.data = data.copy()
        #get mean data values to 1
        self.norm = 1./nu.vstack(self.data.values())[:,1].mean()
        for i in data:
            self.data[i][:,1] *= self.norm
        self.db = util.numpy_sql(db_name)
        # Tell which models are avalible and how many galaxies to fit
        self.models = {'burst': data.keys()}
        # Get param range (tau, age, metal)
        self.param_range = []
        for column in ['tau', 'age', 'metalicity']:
            self.param_range.append(nu.sort(nu.ravel(self.db.execute(
                'Select DISTINCT %s FROM burst'%column).fetchall())))
        self._hull = None
        # resolution for CB07 and BC03 in km/s
        self.resolu = {}
        for gal in data:
            self.resolu[gal] = 3. * 299792.458 / data[gal][:,0].mean()

    def _make_hull(self):
        '''Make convex hull obj for telling if param is in range'''
        # Make all points in param space
        points = nu.asarray([point for point in
                             itertools.product(*self.param_range)])
        self._hull = Delaunay(points)

    def is_in_hull(self, point):
        '''Checks if a point in in the param range Retuns bool'''
        if not isinstance(self._hull, Delaunay):
            self._make_hull()
        return self._hull.find_simplex(point) >= 0
    
    def lik(self, param, bins, return_model=False):
        '''Calculates log likelyhood for burst model'''
        for gal in param[bins]:
            # get interp spectra
            #check if points are in range
            columns = ['tau', 'age', 'metalicity']
            if self.is_in_hull(param[bins][gal][columns]):
                spec = tri_lin_interp(self.db,
                    param[bins][gal][columns], self.param_range)
            else:
                if return_model:
                    yield -nu.inf, gal, []
                else:
                    yield -nu.inf, gal
                continue
            
            model = {'wave':spec[:,0],
                     0: spec[:,1] * 10**param[bins][gal]['normalization'].iat[0]}
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
            model = ag.data_match(self.data[gal], model)
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

    def step_func(self, step_crit, param, step_size, model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray
        param should be a pandas.DataFrame
        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        #ipdb.set_trace()
        for gal in step_size[model]:
            if step_crit > .30 and nu.all(step_size[model][gal].diagonal() < 8.):
                step_size[model][gal] *= 1.05
            elif step_crit < .2 and nu.any(step_size[model][gal].diagonal() > 10**-6):
                step_size[model][gal] /= 1.05
            #cov matrix
            if len(param) % 200 == 0 and len(param) > 0.:
                print 'here'
                temp = nu.cov(MC.list_dict_to(param[gal][-2000:],self._key_order).T)
                #make sure not stuck
                '''if nu.any(temp.diagonal() > 10**-6):
                    step_size[model][gal] = temp'''
        
        return step_size[model]

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
        param = [nu.random.rand()*i.ptp() + i.min() for i in self.param_range]
        #norm
        param.append(nu.random.rand()*nu.log10(self.norm)+15)
        #redshift
        param.append(0.)
        if self.has_dust:
            dtype.append((r'$T_{bc}$',float))
            dtype.append((r'$T_{ism}$', float))
            #uniform between 0 ,4
            param += [nu.random.rand()*4 for i in range(2)]
        if self.has_losvd:
            dtype.append((r'$\sigma$', float))
            dtype.append((r'$V$', float))
            dtype.append((r'$h_3$', float))
            dtype.append((r'$h_4$', float))
            #sigma and v
            param += [nu.random.rand()*3 for i in range(2)]
            # h3 and h4 off for now
            param += [0. for i in range(2)]
        # create array and assign values
        out_param = nu.empty(1, dtype=dtype)
        for index,elmt in enumerate(param):
            out_param[0][index] = elmt
        out_param = pd.DataFrame(out_param)
        return out_param, nu.eye(len(param))*.01
    
    def prior(self, param, bins):
        '''Calculates priors of all parameters'''
        # Uniform
        # tau, age, metalicity
        for gal in param[bins]:
            out_lik = nu.sum([stats_dist.uniform.logpdf(param[bins][gal].iloc[0][i],
                                                         ran.min(),ran.ptp())
                                for i,ran in enumerate(self.param_range)])
            out_lik += stats_dist.uniform.logpdf(param[bins][gal]['normalization'],
                                              -100, 200)
            #out_lik += redshift
            if self.has_dust:
                out_lik += stats_dist.uniform.logpdf(param[bins][gal][['$T_{ism}$',
                                                '$T_{bc}$']],0,4).sum()
            if self.has_losvd:
                out_lik += stats_dist.uniform.logpdf(param[bins][gal]['$\\sigma$'],
                                                 nu.log10(self.resolu[gal]),
                                                 3- nu.log10(self.resolu[gal]))
                out_lik += stats_dist.uniform.logpdf(param[bins][gal]['$V$'],0,4)
            yield out_lik, gal
    
    def model_prior(self, model):
        return 0.
    
    def proposal(self, Mu, Sigma):
        # get out of rec array
        
        out = {}
        for gal in Mu:
            mu = Mu[gal].values[0]
            out[gal] =  nu.random.multivariate_normal(mu, Sigma[gal])
            # put back into DataFrame
            out[gal] = pd.DataFrame(out[gal],Mu[gal].columns).T
        #set h3 and h4 to 0
        if self.has_losvd:
            out[gal]['$h_3$'] = 0.
            out[gal]['$h_4$'] = 0.
        return out

class LRG_mpi_lik(Multi_LRG_burst):
    '''Does LRG fitting and sends likelihood cal to different
    processors'''
    from mpi4py import MPI as mpi
        
def grid_search(point, param_range):
    '''Finds points that make a cube around input point and returns them with
    their spectra'''
    points = point.get_values()[0]
    index = []
    len_array = []
    on_plane = []
    for i in range(len(points)):
        len_array.append(len(param_range[i]))
        index.append(nu.searchsorted(param_range[i],points[i]))
        on_plane.append(param_range[i] == points[i])
    index = nu.asarray(index)
    len_array = nu.asarray(len_array)
    # check if at on an edge
    if nu.any(index == 0):
        ipdb.set_trace()
        raise NotImplementedError
    if nu.any(index == len_array):
        ipdb.set_trace()
        raise NotImplementedError
    # check if on plane
    if nu.any(nu.hstack(on_plane)):
        ipdb.set_trace()
        raise NotImplementedError
    # iterate around the point
    com_tupple = [(param_range[j][i-1], param_range[j][i])
                  for j,i in enumerate(index)]
    interp_points = nu.asarray([p for p in
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
    out_spec = nu.vstack((wave, griddata(points, spec, param))).T
    return  out_spec
    
