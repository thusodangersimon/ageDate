import LRG_lik as lik
import Age_mltry as mltry
import model
import mpi_top
from numpy.random import randint
'''Called for fitting of LRG'''



def Single_LRG_model():
    '''Does fit for sigle LRG model. Plots results'''
    # make model
    data, real_param = get_data(1)
    #fit model
    fun = lik.Multi_LRG_burst(data,'/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db')
    top = mpi_top.Topologies('single')
    out_class =  mltry.multi_main(fun, top)
    #plot marginal historgram
    

def Multiple_LRG_model(num_gal):
    '''Makes multiple models and fits them simultaneously'''
    data, real_param = get_data(num_gal)
    # run
    fun = lik.Multi_LRG_burst(data,'/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db',have_dust=True,have_losvd=True)
    top = mpi_top.Topologies('single')
    out_class =  mltry.multi_main(fun, top)

    
def open_mp_LRG_model(cpus):
    '''fits multipule LRGs with multicores'''
    
    
def get_data(num_gal):
    data, real_param = {}, {}
    for gal in range(num_gal):
        # Get SDSS id type name
        id = 'SDSS %04d'%randint(9999)
        if id in data:
            id = 'SDSS %04d'%randint(9999)
        data[id], real_param[id] = model.random_burst_gal()
    return data, real_param

if __name__ == "__main__":
    #Single_LRG_model()
    Multiple_LRG_model(5)
