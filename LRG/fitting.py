import LRG_lik as lik
import Age_mltry as mltry
import model
import mpi_top 
'''Called for fitting of LRG'''



def Single_LRG_model():
    '''Does fit for sigle LRG model. Plots results'''
    # make model
    data, real_param = model.random_burst_gal(1)
    #fit model
    fun = lik.Multi_LRG_burst(data,'/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db')
    top = mpi_top.Topologies('single')
    out_class =  mltry.multi_main(fun, top)
    #plot marginal historgram
    


def Multiple_LRG_model():
    pass

if __name__ == "__main__":
    Single_LRG_model()
