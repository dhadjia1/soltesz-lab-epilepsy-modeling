import numpy as np
from mpi4py import MPI
import sys, os
import logging

sys.path.append('../utils')


def norm(data):
    m = data.mean(axis=1)
    D = data - np.stack([m for _ in range(data.shape[1])]).T
    return D.astype('float32')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig()
logger = logging.getLogger()

input_filepath = sys.argv[1]
nsurr          = int(sys.argv[2])

if rank == 0:
    logger.error('num surrogate dsets: %d', nsurr)

if rank == 0:
    f = np.load(input_filepath)
    traces = f['traces']
    traces = norm(traces)
    original_eigenvalues = np.linalg.eigvals(np.corrcoef(traces))
    
else:
    traces = None
traces = comm.bcast(traces,root=0)
if rank == 0:
    logger.error('original eigenvalues calcualted..')

round_robin = []
for i in range(rank, nsurr, size):
    round_robin.append(i)
    
surr_evals = []
for (rridx,rr) in enumerate(round_robin):
    shuffled_traces = []
    for t in traces:
        np.random.shuffle(t)
        shuffled_traces.append(t)
        
    shuffled_traces = np.asarray(shuffled_traces,dtype='float32')
    surr_evals.append(np.linalg.eigvals(np.corrcoef(shuffled_traces)))
    comm.barrier()
    if rank == 0:
        logger.error('%d out of %d', (rridx+1), len(round_robin))
del traces

surr_evals = comm.gather(surr_evals, root=0)
if rank == 0:
    extended_surr_evals = []
    for se in surr_evals: extended_surr_evals.extend(se)
    extended_surr_evals = np.asarray(extended_surr_evals, dtype='float32')
    
    me, se = extended_surr_evals.mean(axis=0), extended_surr_evals.std(axis=0)
    synch_index = []
    for i in range(len(me)):
        if original_eigenvalues[i] > (me[i] + 2.0 * se[i]):
            synch_index.append( (original_eigenvalues[i] - me[i]) / (len(me) - me[i]) )
        else:
            synch_index.append(0.)
            
    synch_index = np.asarray(synch_index)
    logger.error('global synchronization: %0.3f', synch_index.max())
    
    
    