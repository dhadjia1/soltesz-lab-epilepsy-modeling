

import numpy as np
from mpi4py import MPI
import time, logging, sys, os

sys.path.append('../utils')
from fc_utils import read_data, calc_fc_matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

fish_id = int(sys.argv[1])
hop = 1
window = 240

logging.basicConfig()
logger = logging.getLogger()

if rank == 0:
    input_filepath = '../../data/fish-%i-processed.npz' % fish_id
    traces = read_data(input_filepath)
else:
    traces = None

traces = comm.bcast(traces, root=0)
N, T = traces.shape

start_times = range(0, T - window, hop)
idxs        = range(len(start_times))

times_to_process, idxs_to_process = [], []
for i in range(rank, len(idxs), size):
    idxs_to_process.append(i)
    times_to_process.append(start_times[i])

logger.error('rank %i will process %i matrices', rank, len(idxs_to_process))
    
output_dir = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%i/gt' % fish_id
start = time.time()
for (citer, start_time) in enumerate(times_to_process):
    curr_idx = idxs_to_process[citer]
    data_chunk = traces[:,start_time:start_time+window]
    corr_mat   = calc_fc_matrix(data_chunk, todict=False)
    f = np.savez('%s/fc-%i.npz'% (output_dir, curr_idx), fc=corr_mat, start=start_time, window=window)

elapsed = time.time() - start
avg_time = elapsed / float(len(idxs_to_process))

comm.barrier()
logger.error('average time per matrix: %0.3f', avg_time)
