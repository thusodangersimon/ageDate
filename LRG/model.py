import numpy as nu
import database_utils as util
import pandas as pd
import itertools
from scipy.interpolate import griddata
'''Makes feducial model for testing'''


def random_burst_gal(db_path='/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'):
    '''gets num_gal of burst model spectra without any background model'''
    db = util.numpy_sql(db_path)
    param = get_points(db)
    data = tri_lin_interp(db, param[['tau','age','metalicity']], make_param_range(db))
    # Make noise

    # Change wavelength range
    data = data[nu.where(nu.logical_and(data[:,0]>=3200, data[:,0] <= 9500))]

    return data, param


def random_exp_gal(db_path='/home/thuso/Phd/experements/hierarical/LRG_Stack/exp_dtau_10.db'):
    '''gets num_gal of exponetal model spectra without any background model'''
    db = util.numpy_sql(db_path)
    param = get_points(db)
    data = tri_lin_interp(db, param, make_param_range(db))
    # Make noise

    return data, param


def get_points(db, has_dust=False, has_losvd=False):
    '''return points in range of parameters'''
    param_range = make_param_range(db)
    dtype = []
    param = []
    # make tau, age, and metal array
    dtype.append(('tau',float))
    dtype.append(('age',float))
    dtype.append(('metalicity', float))
    dtype.append(('redshift', float))
    #uniform dist for everything except redshift
    param = [nu.random.rand()*i.ptp() + i.min() for i in param_range]
    
    param.append(0.)
    if has_dust:
        dtype.append((r'$T_{bc}$',float))
        dtype.append((r'$T_{ism}$', float))
        #uniform between 0 ,4
        param += [nu.random.rand()*4 for i in range(2)]
    if has_losvd:
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
    return out_param

def make_param_range(db):
    '''Gets Unique values from spectral range'''

    table_name = db.execute('select * from sqlite_master').fetchall()[0][1]
    param_range = []
    for column in ['tau', 'age', 'metalicity']:
        param_range.append(nu.sort(nu.ravel(db.execute(
                'Select DISTINCT %s FROM %s'%(column, table_name)).fetchall())))
    return param_range

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
        raise NotImplementedError
    if nu.any(index == len_array):
        raise NotImplementedError
    # check if on plane
    if nu.any(nu.hstack(on_plane)):
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
    
