from glob import glob
import numpy as np
import cPickle as pik
import Age_mltry as mltry
import LRG_lik as lik
import mpi_top
from mpi4py import MPI as mpi

files = glob('*.pik')
#make data array
data = {}
for file in files:
    temp = pik.load(open(file))
    data[file] = temp[-1]


mpi.COMM_WORLD.barrier()
if mpi.Get_processor_name() == 'mightee.ast.uct.ac.za':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
else:
    db_path = '/mnt/burst_dtau_10.db'
db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
fun = lik.Multi_LRG_burst(data, db_path, have_dust=True, have_losvd=True)
top = mpi_top.Topologies('single')
try:
    Param = mltry.multi_main(fun, top, fail_recover=True)
except mltry.MCMCError:
    Param = mltry.multi_main(fun, top)

pik.dump((Param, real_param, data), open('stacked_results.pik', 'w'), 2)    
