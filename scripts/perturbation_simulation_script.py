
import numpy as np
import sys, os, time
from mpi4py import MPI

sys.path.append('../utils')
from perturbation_simul_utils import *


#output_filepath = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%d/perturbation-sims/edge-swap/control1' % FISH_ID
#multi_simu(500, output_filepath, Jpresz, None, None,
           **{'triplets': presz_sender_triplets, 'nedges': nedges, 'etype': 'control'})
    
    
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if size < 5: sys.exit(1)
    
FISH_ID = int(sys.argv[1])
Jfilepath = str(sys.argv[2])

f = np.load(Jfilepath)
J = f['J'].astype('float32')
f.close()

    
outputf = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%d/perturbation-sims/edge-swap' % FISH_ID
output_filepaths = [outputf + '/test', outputf + '/control1', outputf + '/control2', outputf + '/control3', outputf + '/control4']



rank_output = None
if rank < len(output_filepaths):
    rank_output = output_filepaths[rank]
if rank_output is not None:
    multi_simu(500, rank_output, J, 