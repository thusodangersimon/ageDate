from Age_mltry import multi_main
import LRG_lik
from mpi4py import MPI as mpi
import cPickle as pik
import io
import pandas as pd
import pyfits as fits
import numpy as np

'''takes data from LRG.pik and processes the fits files into data for MCMC
fitting'''

def fits2data(fitsfile, sn):
    '''Returns nx3 data array from fits spectra'''
    if isinstance(fitsfile, str):
        # turn into buffer and load
        fitsfile = fits.open(io.BytesIO(fitsfile))
    step = fitsfile[0].data.shape[-1] * fitsfile[0].header['CD1_1']
    maxwave = fitsfile[0].header['CRVAL1'] + step
    wavelength = np.linspace(fitsfile[0].header['CRVAL1'], maxwave,
                             fitsfile[0].data.shape[-1])
    data = np.zeros((len(wavelength), 3))
    data[:,0] = wavelength
    data[:,1] = fitsfile[0].data[0,:][0]
    data[:,2] = np.median(fitsfile[0].data[0,:][0])/ sn
    return data

def load_data():
    # load database
    database = pik.load(open('LRG.pik'))
    header = ['Object_ID','RA', 'Declination', 'z_fin', 'sn', 'id',
                'spectra']
    # make into dataframe
    database = pd.DataFrame(database, columns=header, index=None)
    database[['RA', 'Declination', 'z_fin', 'sn','id']] = database[['RA',
                            'Declination', 'z_fin', 'sn', 'id']].astype(float)
    data = {}
    for _, gal in database.iterrows():
        data[gal.Object_ID] = fits2data(gal.spectra, gal.sn)

        
    pik.dump(data, open('spectra.pik','w'),2)

if __name__ == __main__:
    import cPickle as pik
    import Age_mltry as mltry
    import LRG_lik as lik
    import mpi_top
    from mpi4py import MPI as mpi
    
    data = pik.dump(data, open('spectra.pik','w'),2)
    mpi.COMM_WORLD.barrier()
    if mpi.Get_processor_name() == 'mightee.ast.uct.ac.za':
        db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    else:
        db_path = '/mnt/burst_dtau_10.db'
    #db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    fun = lik.Multi_LRG_burst(data, db_path, have_dust=True, have_losvd=True)
    top = mpi_top.Topologies('single')
    try:
        Param = mltry.multi_main(fun, top, fail_recover=True)
    except mltry.MCMCError:
        Param = mltry.multi_main(fun, top)

    pik.dump((Param, real_param, data), open('not_stacked_results.pik', 'w'), 2)    
