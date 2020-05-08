
import numpy as np
import sys, os, logging, time
from mpi4py import MPI

sys.path.append('../utils')
from fc_utils import read_data, calc_fc_matrix, get_single_IAAFT_surrogate, RunningStats

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def combine_rstats(a, b, dtype):
    combined = RunningStats.combine(a,b)
    return combined

mpi_op_combine_rstats = MPI.Op.Create(combine_rstats, commute=True)

fish_id = int(sys.argv[1])
hop     = 1
window  = 240
nsurr   = size

logging.basicConfig()
logger = logging.getLogger()

if rank == 0:
    input_filepath = '../../data/fish-%i-processed.npz' % fish_id
    traces = read_data(input_filepath)
else:
    traces = None

traces = comm.bcast(traces, root=0)
N, T   = traces.shape

    
output_filepath = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%i/surrogate' % fish_id

for (citer, i) in enumerate(range(0, T - window, hop)):
    if rank == 0:
        start = time.time()
        
    current_traces  = traces[:,i:i+window]
    shuffled_traces = np.empty(current_traces.shape, dtype='float32')
    for (ct_idx, ct) in enumerate(current_traces):
        shuffled_traces[ct_idx,:] = get_single_IAAFT_surrogate(ct, seed=citer*10000+rank, niters=25)
    comm.barrier()
                                                               
    rs   = RunningStats()
    corr = calc_fc_matrix(shuffled_traces, todict=False)
    rs.update(corr)

    all_stats = comm.reduce(rs, root=0, op=mpi_op_combine_rstats)
    if rank == 0:
        mean_xcorr = all_stats.mean()
        std_xcorr  = all_stats.standard_deviation()
        io_start   = time.time()
        f = np.savez('%s/surr-fc-%i.npz' % (output_filepath, citer), mfc=mean_xcorr, sfc=std_xcorr, start=i)
        io_elapsed = time.time() - io_start
        elapsed = time.time() - start
        logger.error('start time: %i. time elapsed: %0.3f seconds. io elapsed: %0.3f', i, elapsed, io_elapsed)
    comm.barrier()
    
