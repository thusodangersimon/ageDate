import LRG_lik as lik
import Age_mltry as mltry
import model
import mpi_top
from mpi4py import MPI as mpi
from numpy.random import randint
import cPickle as pik
## Called for fitting of LRG'''



def Single_LRG_model(db_path='/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'):
    '''Does fit for sigle LRG model. Plots results'''
    # make model
    #data, real_param = get_data(1, db_path)
    data, real_param =  model.make_noise_gal(20, db_path)
    data = {'SDSS 0000':data}
    #fit model
    fun = lik.Multi_LRG_burst(data, db_path)
    top = mpi_top.Topologies('single')
    try:
        out_class = mltry.multi_main(fun, top, fail_recover=True)
    except mltry.MCMCError:
        out_class = mltry.multi_main(fun, top)
    #plot marginal historgram
    return out_class, real_param, data
    

def Multiple_LRG_model(num_galm, db_path='/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'):
    '''Makes multiple models and fits them simultaneously'''
    data, real_param = get_data(num_gal, db_path)
    # run
    fun = lik.Multi_LRG_burst(data, db_path) #,have_dust=True,have_losvd=True)
    top = mpi_top.Topologies('single')
    out_class = mltry.multi_main(fun, top, max_iter=1000)
    return out_class, real_param, data


def open_mp_LRG_model(num_gal, db_path='/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'):
    '''fits multipule LRGs with multicores'''
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        data, real_param = get_data(num_gal, db_path)
    else:
        data = None
    data = comm.bcast(data, root=0)
    fun = lik.LRG_mpi_lik(data, db_path)
    # Workes just calc likelihoods
    if rank > 0:
        fun.lik_worker()
    else:
        top = mpi_top.Topologies('single')
        try:
            out_class = mltry.multi_main(fun, top, max_iter=5*10**5.,
                                          fail_recover=True)
        except mltry.MCMCError:
            pik.dump((data, real_param), open('save_param.pik', 'w'), 2)
            out_class = mltry.multi_main(fun, top, max_iter=5*10**5)
        return out_class, real_param, data



def get_data(num_gal, db_path):
    data, real_param = {}, {}
    for gal in range(num_gal):
        # Get SDSS id type name
        id = 'SDSS %04d'%randint(9999)
        if id in data:
            id = 'SDSS %04d'%randint(9999)
        data[id], real_param[id] = model.make_noise_gal(float(id[-2:]), db_path)
    return data, real_param

if __name__ == "__main__":
    #Single_LRG_model()
    import os
    models = 1
    #Param, real_param, data = Multiple_LRG_model(models)
    mpi.COMM_WORLD.barrier()
    t_multi = mpi.Wtime()
    #if mpi.Get_processor_name() == 'mightee.ast.uct.ac.za':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    #else:
        #db_path = '/mnt/burst_dtau_10.db'
    if not os.path.exists(db_path):
        print '%s rank %i cannot find db'%(mpi.Get_processor_name(),
                                           mpi.COMM_WORLD.rank)
    #Param, real_param, data = open_mp_LRG_model(models, db_path)
    Param, real_param, data = Single_LRG_model(db_path)
    t_multi -= mpi.Wtime()
    #import cPickle as pik
    pik.dump((Param, real_param, data), open('test.pik', 'w'), 2)
    #print 'Sigle time %f. Multi time %f'%(abs(t_single),abs(t_multi))
