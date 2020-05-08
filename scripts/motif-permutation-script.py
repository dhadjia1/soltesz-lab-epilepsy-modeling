import numpy as np
import sys, os, time
from copy import deepcopy
from mpi4py import MPI

sys.path.append('../utils')
from futils import jthreshold, permute_weights
from graph_utils import generate_snap_unweighted_graph, run_motif_counting_algorithm



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Jfilepath   = str(sys.argv[1])
threshold   = int(sys.argv[2])
niterations = int(sys.argv[3])
fid         = int(sys.argv[4])
state       = str(sys.argv[5])

if rank == 0:
    f = np.load(Jfilepath)
    J = f['J'].astype('float32')
    f.close()
else:
    J = None
J = comm.bcast(J, root=0)
J = J.T
cwd = os.getcwd()

permutations_to_process = []
for i in range(rank, niterations, size):
    permutations_to_process.append(i)
    

analysis_filepath = '/mnt/e/dhh-soltesz-lab/zfish-proj/src-parallel/operation-figgeritout/comp-modeling/motif-analysis'
save_filepath     = analysis_filepath + '/f%i/%s/%s/permuted' % (fid, state, str(threshold))
for i in permutations_to_process:
    Jcopy = deepcopy(J)
    Jcopy = permute_weights(Jcopy, seed=i)
    sgraph_filepath = save_filepath + '/graphs/permuted-graph-%i.txt' % i
    motif_filepath  = save_filepath + '/motifs/permuted-motifs-%i.txt' % i
    
    
    Jth = jthreshold(Jcopy,threshold,above=True,pos=True,binarized=True)
    sgraph = generate_snap_unweighted_graph(Jth, directed=True, save=sgraph_filepath)
    run_motif_counting_algorithm(sgraph_filepath, motif_filepath, cwd)
    

    
#/mnt/f/dhh-soltesz-lab/zfish-modeling-outputs/f2/baseline/v1/training-188.npz
