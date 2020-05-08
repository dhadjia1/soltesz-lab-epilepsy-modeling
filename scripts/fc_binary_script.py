import numpy as np
from mpi4py import MPI
import time, logging, sys, os

sys.path.append('../utils')
from fc_utils import read_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

fish_id = int(sys.argv[1])
hop = 1
window = 240

if rank == 0:
    input_filepath = '../data/fish-%i-processed.npz' % fish_id
    traces = read_data(input_filepath)
    N, T = traces.shape
else:
    N, T = None, None
    
N = comm.bcast(N, root=0)
T = comm.bcast(T, root=0)

logging.basicConfig()
logger = logging.getLogger()

start_times = range(0, T - window, hop)
idxs        = range(len(start_times))

times_to_process, idxs_to_process = [], []
for i in range(rank, len(idxs), size):
    idxs_to_process.append(i)
    times_to_process.append(start_times[i])

logger.error('rank %i will process %i matrices', rank, len(idxs_to_process))


input_filepath = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%i' % fish_id

for (citer, start_time) in enumerate(times_to_process):
    curr_idx = idxs_to_process[citer]
    gt_filepath = input_filepath + '/gt/fc-%i.npz' % curr_idx
    
    f = np.load(gt_filepath)
    gt_corr = f['fc'].astype('float32')
    start, window = f['start'], f['window']
    negative_idxs = np.where(gt_corr < 0.0)
    f.close()
    
    surr_filepath = input_filepath + '/surr/surr-fc-%i.npz' % curr_idx
    f = np.load(surr_filepath)
    mean_corr = f['mfc'].astype('float32')
    std_corr  = f['sfc'].astype('float32')
    f.close()
    
    threshold_corr  = mean_corr + 2. * std_corr
    subtracted_corr = gt_corr - threshold_corr
    
    binarized_corr = np.zeros_like(subtracted_corr, dtype='uint8')
    connected_idxs   = np.where(subtracted_corr > 0.)
    unconnected_idxs = np.where(subtracted_corr <= 0.)
    
    binarized_corr[connected_idxs]   = 1
    binarized_corr[unconnected_idxs] = 0
    binarized_corr[negative_idxs]    = 0
    binarized_corr[np.diag_indices(N)] = 0
    nedges = np.sum(binarized_corr)
   
    bin_filepath = input_filepath + '/bin/densities.npz' % curr_idx  
    f = np.savez(bin_filepath, binarized=binarized_corr, start=start, window=window, nedges=nedges)
    
