from glob import glob
import numpy as np
import cPickle as pik
from Age_tempering import tempering_main, MCMCError
import LRG_lik as lik
import mpi_top
from mpi4py import MPI as mpi
import ipdb

data = {}
if mpi.COMM_WORLD.rank == 0:
    files = glob('*.pik')
    #make data array

    for i,file in enumerate(files):
        #print file, i
        temp = pik.load(open(file))
        data[file] = temp[-1]
        break
data = mpi.COMM_WORLD.bcast(data)


mpi.COMM_WORLD.barrier()
if mpi.Get_processor_name() == 'mightee.ast.uct.ac.za':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
else:
    db_path = '/mnt/burst_dtau_10.db'
db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
fun = lik.LRG_Tempering(data, db_path, have_dust=True, have_losvd=True)
top = mpi_top.Topologies('single')
#start mcmc
if  mpi.COMM_WORLD.size > 1:
    if mpi.COMM_WORLD.rank > 0:
        fun.lik_worker()
    else:
        try:
            Param = tempering_main(fun, top, fail_recover=True)
        except MCMCError:
            Param = tempering_main(fun, top)
        
        pik.dump((Param,  data), open('stacked_results.pik', 'w'), 2)    
        ipdb.set_trace()
