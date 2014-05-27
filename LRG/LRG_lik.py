import likelihood_class as lik
import numpy as nu
import database_utils as util
import interp_utils as interp
from scipy.spatial import Delaunay
import itertools

'''Likelyhood and functions needed for MCMC for LRGS'''

class Multi_LRG_burst(lik.Example_lik_class='burst_dtau_10.db'):

    def __init__(self, data, db_name=):

        self.data = data
        self.db = util.numpy_sql(db._name)
        self.models = {0: []}
        # Get param range (tau, age, metal)
        self.param_range = []
        for column in ['tau', 'age', 'metalicity']:
            self.param_range.append(nu.asarray(self.db.execute(
                'Select DISTINCT %s FROM burst'%column).fetchall()).ravel())
        self._hull = None
        

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
        return self._hull.find_simplex(points) >= 0
        
    def lik(self, param, bins):
        '''Calculates log likelyhood for burst model'''

        # get interp spectra

        # LOSVD

        # Dust

        # Calc liklihood

    def step_func(self, step_crit, param, step_size, model):
        pass

    def initalize_param(self, bins):
        return param, nu.eye(bins*4)
    
    def prior(self, param, bins):
            yield out_lik,i
    
    def model_prior(self, model):
        return 0.
    
    def proposal(self, Mu, Sigma):
        # get out of dict
        mu = nu.hstack(Mu.values())
        temp =  nu.random.multivariate_normal(mu, Sigma)
        out = Mu.copy()
        temp = nu.reshape(temp,(len(Mu.keys()), 4))
        for i in out.keys():
            out[i] = temp[i]
        return out

def tri_lin_interp(db, param, param_range):
    '''Does trilinear interoplation for spectra with points (tau,age,metal).
    Returns interpolated spectra or an array of inf if param is out of range'''

    # get 8 nearest neighbors

    # Do bilinear interp till interp is done

    # Check for nans or inf

    
