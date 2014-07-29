from glob import glob
import numpy as np
import cPickle as pik
import Age_mltry as mltry
import LRG_lik as lik
import mpi_top
from mpi4py import MPI as mpi

data = {}
if mpi.COMM_WORLD.rank == 0:
    files = glob('*.pik')
    #make data array

    for i,file in enumerate(files):
        print file,i
        temp = pik.load(open(file))
        data[file] = temp[-1]
data = mpi.COMM_WORLD.bcast(data)


mpi.COMM_WORLD.barrier()
if mpi.Get_processor_name() == 'mightee.ast.uct.ac.za':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
else:
    db_path = '/mnt/burst_dtau_10.db'
db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
fun = lik.LRG_mpi_lik(data, db_path, have_dust=True, have_losvd=True)
top = mpi_top.Topologies('single')
#start mcmc
if  mpi.COMM_WORLD.size > 1:
    if mpi.COMM_WORLD.rank > 0:
        fun.lik_worker()
    else:
        try:
            Param = mltry.multi_main(fun, top, fail_recover=True)
        except mltry.MCMCError:
            Param = mltry.multi_main(fun, top)

        pik.dump((Param, real_param, data), open('stacked_results.pik', 'w'), 2)    
