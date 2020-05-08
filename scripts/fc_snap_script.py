import numpy as np
import sys, os, logging, time
from mpi4py import MPI

sys.path.append('../utils')
from graph_utils import data2percentile, generate_snap_unweighted_graph, get_node_centrality, get_eigenvector_centrality, get_clustering_coefficient
import snap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

fish_id = int(sys.argv[1])
save_snap = int(sys.argv[2])

nmatrices = int(sys.argv[3])
nmatrices_lst = range(nmatrices)

logging.basicConfig()
logger = logging.getLogger()

idxs_to_process = []
for i in range(rank, nmatrices, size):
    idxs_to_process.append(i)
logger.error('rank %i will process %i matrices', rank, len(idxs_to_process))

filepath = '/mnt/f/dhh-soltesz-lab/zfish-fc/f%i' % fish_id
for idx in idxs_to_process:
    binary_filepath = filepath + '/bin/bin-fc-%i.npz' % idx
    f = np.load(binary_filepath)
    binary = f['binarized']
    start  = f['start']
    window = f['window']
    f.close()
    
    nedges = None 
    if save_snap:
        snap_filepath = filepath + '/snap/sgraph-fc-%i.txt' % idx
        snap_graph = generate_snap_unweighted_graph(binary, directed=False)
        nedges = int(snap_graph.GetEdges())
        snap.SaveEdgeList(snap_graph, snap_filepath)

    dnids, deg_centrality = get_node_centrality(snap_graph)
    deg_centrality = deg_centrality[np.argsort(dnids)]
    
    enids, ev_centrality = get_eigenvector_centrality(snap_graph)
    ev_centrality = ev_centrality[np.argsort(enids)]

    cnids, ccs = get_clustering_coefficient(snap_graph)
    ccs = ccs[np.argsort(cnids)]
    
    metrics_filepath = filepath + '/graph-metrics/metrics-fc-%i.npz' % idx
    f = np.savez(metrics_filepath, degree=deg_centrality, eigenvector=ev_centrality, cc=ccs, nedges=nedges)
    
    
    
    
