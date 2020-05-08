import numpy as np
from mpi4py import MPI
import random
import time as clock
from copy import deepcopy
import sys, os, click, logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d

sys.path.append('/mnt/e/dhh-soltesz-lab/zfish-proj/src-parallel/operation-figgeritout/comp-modeling/utils')

import rnn_modeling_utils
from modeling_analysis_utils import load_neurons, load_baier_regions, LRsplit, build_traces, load_baier_connectome, build_neural_S

def list_find(f, lst):
    i = 0
    for x in lst:
        if f(x): return i
        else: i += 1
    return None

def mpi_excepthook(type, value, traceback):
        sys.stderr.flush()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)

sys_excepthook = sys.excepthook
#sys.excepthook = mpi_excepthook
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logging.basicConfig()
logger = logging.getLogger()


def get_best(save_directory, save_prefix, T):
    val_errors, train_errors = [], []
    for i in range(1, T):
        filepath = save_directory + '/' + save_prefix + '-%i.npz' % i
        try:
            f = np.load(filepath, allow_pickle=True)
            val_errors.append(f['validation_error'])
            train_errors.append(f['train_error'])
            f.close()
        except:
            train_errors.append(1e9)
    best_iter = np.argmin(train_errors) + 1
    if comm.rank == 0: print(best_iter, train_errors[best_iter - 1])
    
    filepath  = save_directory + '/' + save_prefix + '-%i.npz' % best_iter
    f = np.load(filepath, allow_pickle=True)
    J = f['J']
    f.close()
    return J.astype('float32')


def iron_bumps(traces, tb=20, db=20):
    idxs = []
    N, T = traces.shape
    for i in range(N):
        trace = traces[i,:]
        diff = np.diff(trace,1)
        
        trace_bad = np.where(abs(trace) > tb)[0]
        diff_bad  = np.where(abs(diff) > db)[0]
        if len(trace_bad) > 0 or len(diff_bad) > 0:
            idxs.append(i)
    return idxs

def run(save_directory, save_prefix, info_directory, traces, roi_coords, niters, S=None, J=None, comm=None, **kwargs):
    n1, n2 = kwargs['n1'], kwargs['n2']
    if n2 == -1: n2 = None
    t1, t2  = kwargs['t1'], kwargs['t2']
    fish_id = kwargs['fid']
    
    traces     = traces[n1:n2,:].astype('float32')
    roi_coords = roi_coords[n1:n2,:]
    tmax=None
    if fish_id== 2:
        tmax = 1370
    elif fish_id == 3:
        tmax = 1270
    elif fish_id == 5:
        tmax = 2190
    bad_idxs = iron_bumps(traces[:,10:tmax], tb=10, db=10)
    good_idxs = list(set(range(traces.shape[0])) - set(bad_idxs))
    
    roi_coords = roi_coords[good_idxs,:]
    traces = traces[good_idxs,t1:t2]
    traces_mean = np.mean(traces, axis=1)
    tmp_traces = []
    for i in range(traces.shape[0]):
        t = traces[i] - traces_mean[i]
        tmp_traces.append(t)
    traces = np.asarray(tmp_traces, dtype='float32')   

    if S is not None:
        S = S[n1:n2,:]; S = S[:,n1:n2]
        S = S[good_idxs,:]; S = S[:, good_idxs]
    N, T = traces.shape
    T = int(T*0.5)
    
    experimental_times = np.arange(0., T, 0.5)
    integration_times  = np.arange(0., T, 0.25)[:-1]
    interp = interp1d(experimental_times, traces, kind='cubic')
    traces_interp = interp(integration_times)
        
    if rank == 0:
        logger.error('N: %d', N)
        logger.error('max: %0.3f', np.max(traces_interp))
        logger.error('min: %0.3f', np.min(traces_interp))
#         plt.figure()
#         plt.plot(traces_interp[200:250,:].T)
#         plt.show()
    ## Actual modeling here

    Ytruth  = traces_interp
    state   = kwargs['state']
    version = kwargs['version'] - 1
    seed_add = 0
    if state == 'presz': seed_add = 1
    version_add = 1000000 * version
    tvseed  = (fish_id * 10 + seed_add) + version_add
    jseed   = fish_id * 100 + seed_add  + version_add
    ntseed  = fish_id * 1000 + seed_add + version_add
    cmseed  = fish_id * 10000 + version_add
    y0seed  = fish_id * 100000 + seed_add + version_add

    #####
    y0_range  = 0.5
    noise_std = 0.05
    pc    = kwargs['pc']
    alpha = 1.0
    
    if comm is not None:
        comm.barrier()


    rnn_model = rnn_modeling_utils.RNN_modeling(integration_times, Ytruth, 0.25, S=S, pc=pc, tau=1.5, alpha=alpha, g=1.25, comm=comm)
    if J is None:
        rnn_model.build_from_scratch(training_percent=None, tvseed=tvseed, jseed=jseed, ntseed=ntseed, cmseed=cmseed, y0seed=y0seed, 
                                     y0_range=y0_range, noise_trace_std=noise_std)
    else:
        rnn_model.hot_init(J, training_percent=None, tvseed=tvseed, ntseed=ntseed, cmseed=cmseed, y0seed=y0seed, y0_range=y0_range, 
                           noise_trace_std=noise_std, curr_iter=1)
        
    if comm.rank == 0:
        np.savez(info_directory, S=S, etimes=experimental_times, itimes=integration_times, 
                 Ytruth=Ytruth, mask=rnn_model._connectivity_mask, spatial=roi_coords, 
                 tvseed=tvseed, jseed=jseed, ntseed=ntseed, cmseed=cmseed, y0seed=y0seed, y0_range=y0_range, 
                 noise_std=noise_std, dt=rnn_model._dt, tau=rnn_model._tau, pc=rnn_model._pc, N=rnn_model._N, T=rnn_model._T)

    #rnn_model.train(save_directory, save_prefix, logger, niters=niters, verbose=True)
    
    
def mainrunner_load_mask(mask_filepath):
    all_spatial_ids, all_spatial_coords         = load_neurons(mask_filepath, region_name=['Telencephalon -', 
                                                                             'Rhombencephalon -', 
                                                                             'Mesencephalon -', 
                                                                             'Diencephalon -', 
                                                                             'Spinal Cord'], extend=True)

    ## remove duplicate ids
    unique_ids = []
    unique_coord_idxs = []
    for (idx,i) in enumerate(range(len(all_spatial_ids))):
        current_id = all_spatial_ids[i]
        if current_id not in unique_ids:
            unique_ids.append(current_id)
            unique_coord_idxs.append(idx)

    all_spatial_coords = np.asarray(all_spatial_coords, dtype='float32')[unique_coord_idxs,:]
    all_spatial_ids    = np.asarray(unique_ids, dtype='uint32')
    
    
    # build data structure thta maps mask id to neuron ids that are present in structural connectome
    midpoint                  = np.mean(all_spatial_coords, axis=0)
    atlas_dict, mask2idx_dict = load_baier_regions(mask_filepath)
    mask2idx_dict             = LRsplit(mask2idx_dict, midpoint, axis=0)
    return mask2idx_dict
    


@click.command()
@click.option('--output-dir', required=True, type=str)
@click.option('--date', required=True, type=str)
@click.option('--fish-id', required=True, type=int)
@click.option('--niters', required=True, type=int)
@click.option('--pc', required=True, type=float)
@click.option('--state', required=True, type=str)
@click.option('--version', default=1, type=int)
@click.option('--n1', default=0, type=int)
@click.option('--n2', default=-1, type=int)
def main(output_dir, date, fish_id, niters, pc, state, version, n1, n2):

    traces_full_filepath         = '../data/f090518-%iZbrain_IDs.npz' % fish_id
    mask_filepath                = '../data/f090518-%iZbrain_IDs.npz' % fish_id
     
    if rank == 0:
        mask2idx_dict = mainrunner_load_mask(mask_filepath)
        
      
        # extract neuron calcium traces if id present in sturcutral connectome
        ftraces = np.load(traces_full_filepath)
        traces  = ftraces['tracez']
        ftraces.close()
        roi_traces, roi_coords, mask_szs = build_traces(traces, mask2idx_dict)
        roi_traces = roi_traces.astype('float32')
        roi_coords = roi_coords.astype('float32')
        N, T = roi_traces.shape
        logger.error('traces extracted..')

        ###
        structural_connectome = load_baier_connectome()
        S = build_neural_S(mask2idx_dict, structural_connectome, N)
        logger.error('Structural connectome loaded..')

    else:
        roi_traces, roi_coords = None, None
        S = None
    #### MODELING #####
    
    S = comm.bcast(S, root=0)
    roi_traces = comm.bcast(roi_traces, root=0)
    roi_coords = comm.bcast(roi_coords, root=0)

    save_directory = '%s/zfish-modeling-outputs/f%i/%s/v%i/training-%s' % (output_dir, fish_id, state, version, date)
    save_prefix    = 'full-%s' % state
    info_directory = '%s/zfish-modeling-outputs/f%i/%s/v%i/additional-info-%s.npz' % (output_dir, fish_id, state, version, date)
    t1, t2 = None, None

    if state == 'baseline':
        t1, t2 = 10, 490
    elif state == 'presz':
        if fish_id== 2:
            t1, t2 = 650, 1370
        elif fish_id == 3:
            t1, t2 = 650, 1270
        elif fish_id == 5:
            t1, t2 = 650, 2190
    else:
        raise Exception('state must be baseline or presz') 


    #if not os.path.exists(save_directory):
    #    os.makedirs(save_directory)

    if state == 'presz' and rank == 0:
        temp_state = 'baseline'
        temp_save_directory = '%s/outputs/f%i/full/%s/v%i/training-%s' % (output_dir, fish_id, temp_state, version, date)
        temp_save_prefix    = 'full-%s' % temp_state
        J = get_best(temp_save_directory, temp_save_prefix, niters + 1)
    else:
        J = None
    if state == 'presz':
        J = comm.bcast(J, root=0)

    run_kwargs = {'n1': n1, 'n2': n2, 't1': t1, 't2': t2, 'state': state, 'fid': fish_id, 'version': version, 'pc': pc}
    run(save_directory, save_prefix, info_directory, roi_traces, roi_coords, niters, S=S, J=J, comm=comm, **run_kwargs)

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

