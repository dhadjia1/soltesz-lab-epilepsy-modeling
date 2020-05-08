import rnn_modeling_utils
import os, time, sys, logging
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def load_neurons(filepath, region_name=None, extend=False):
    f = None
    try:
        f = np.load(filepath,allow_pickle=True)
    except:
        raise Exception('Could not load file')
    mask_names = list(f['mask_name'])
    if region_name is None:
        cell_ids    = f['cell_ids']
        cell_coords = f['all_coords']
        
        cell_ids_lst = []
        for cid in cell_ids:
            cell_ids_lst.extend(cid)

        cell_coords = cell_coords[cell_ids_lst,:]

    elif type(region_name) is str:
        cell_ids_lst, cell_coords = _load_region(f, mask_names, region_name)
        
    elif type(region_name) is list:
        cell_ids_lst, cell_coords = [], []        
        for rn in region_name:
            region_cids, region_coords = _load_region(f, mask_names, rn)  
            cell_ids_lst.append(region_cids)
            cell_coords.append(region_coords)
           
        temp_ids, temp_coords = [], []
        if extend:
            for i in range(len(cell_ids_lst)):
                temp_ids.extend(cell_ids_lst[i])
                temp_coords.extend(cell_coords[i])
            cell_ids_lst = temp_ids
            cell_coords  = temp_coords
    else:
        raise Exception('region name argument not recognized. Must be None, str, or lst of strings')

    return cell_ids_lst, cell_coords
    
def _load_region(f, mask_names, region_name):
        if region_name not in mask_names:
            raise Exception('%s could not be found in mask names' % region_name)
        idx = mask_names.index(region_name)
        cell_ids_lst = f['cell_ids'][idx]
        cell_coords  = f['all_coords'][cell_ids_lst,:]
        return cell_ids_lst, cell_coords
    
def load_baier_regions(filepath):
    atlas_dict = zfish_baier_masks()
    ordered_keys = np.sort(list(atlas_dict.keys()))
        
    loaded_data = {}
    for key in ordered_keys:
        current_ids, current_coords = load_neurons(filepath, atlas_dict[key])
        id_set = []
        loaded_data[key] = {'ids': [], 'coords': []}
        for (t, current_id_grp) in enumerate(current_ids):
            for (tt, current_id) in enumerate(current_id_grp):
                if not current_id in id_set:
                    id_set.append(current_id)
                    loaded_data[key]['ids'].append(current_id)
                    loaded_data[key]['coords'].append(current_coords[t][tt])
    
    return atlas_dict, loaded_data                
    
def zfish_baier_masks():
    atlas_dict = {}
    atlas_dict[0] = []
    atlas_dict[2] = [ 'Diencephalon - Dopaminergic Cluster 7 - Caudal Hypothalamus',
                                          'Diencephalon - Hypothalamus - Caudal Hypothalamus Neural Cluster',
                                          'Diencephalon - Caudal Hypothalamus']
    atlas_dict[4] = ['Rhombencephalon - Cerebellum',
                                'Rhombencephalon - Cerebellum Gad1b Enriched Areas',
                                'Rhombencephalon - Olig2 enriched areas in cerebellum']
    atlas_dict[6] = ['Ganglia - Facial glossopharyngeal ganglion']
    atlas_dict[8] = ['Diencephalon - Habenula']
    atlas_dict[10] = ['Rhombencephalon - Inferior Olive']
    atlas_dict[12] = ['Diencephalon - Diffuse Nucleus of the Intermediate Hypothalamus',
                     'Diencephalon - Hypothalamus - Intermediate Hypothalamus Neural Cluster',
                     'Diencephalon - Intermediate Hypothalamus']
    atlas_dict[14] = ['Rhombencephalon - Lateral Reticular Nucleus']
    atlas_dict[16] = ['Rhombencephalon - Interpeduncular Nucleus']
    atlas_dict[18] = ['Rhombencephalon - VII Facial Motor and octavolateralis efferent neurons',
                      'Rhombencephalon - VII Facial Motor and octavolateralis efferent neurons']
    atlas_dict[20] = ['Rhombencephalon - 6.7FDhcrtR-Gal4 Stripe 1',
                      'Rhombencephalon - Gad1b Stripe 1',
                      'Rhombencephalon - Glyt2 Stripe 1',
                      'Rhombencephalon - Isl1 Stripe 1',
                      'Rhombencephalon - Vglut2 Stripe 1',
                      'Spinal Cord - Gad1b Stripe 1',
                      'Spinal Cord - Vglut2 Stripe 1']
    atlas_dict[22] = ['Rhombencephalon - 6.7FDhcrtR-Gal4 Stripe 2',
                      'Rhombencephalon - Gad1b Stripe 2',
                      'Rhombencephalon - Glyt2 Stripe 2',
                      'Rhombencephalon - Vglut2 Stripe 2',
                      'Spinal Cord - Gad1b Stripe 2',
                      'Spinal Cord - Vglut2 Stripe 2']
    atlas_dict[24] = ['Rhombencephalon - 6.7FDhcrtR-Gal4 Stripe 3',
                      'Rhombencephalon - Gad1b Stripe 3',
                      'Rhombencephalon - Glyt2 Stripe 3',
                      'Rhombencephalon - Vglut2 Stripe 3',
                      'Spinal Cord - Vglut2 Stripe 3']
    atlas_dict[26] = ['Rhombencephalon - 6.7FDhcrtR-Gal4 Stripe 4',
                      'Rhombencephalon - Vglut2 Stripe 4']
    atlas_dict[28] = []
    atlas_dict[30] = []
    atlas_dict[32] = ['Telencephalon - Olfactory Bulb',
                      'Telencephalon - Olfactory bulb dopaminergic neuron areas']

    atlas_dict[34] = ['Ganglia - Olfactory Epithelium']
    atlas_dict[36] = ['Telencephalon - Pallium']
    atlas_dict[38] = ['Diencephalon - Pituitary']
    atlas_dict[40] = ['Rhombencephalon - Lateral Reticular Nucleus']
    atlas_dict[42] = ['Diencephalon - Anterior group of the posterior tubercular vmat2 neurons',
                      'Diencephalon - Dopaminergic Cluster 1 - ventral thalamic and periventricular posterior tubercular DA neurons',
                      'Diencephalon - Dopaminergic Cluster 2 - posterior tuberculum',
                      'Diencephalon - Dopaminergic Cluster 4/5 - posterior tuberculum and hypothalamus',
                      'Diencephalon - Migrated Posterior Tubercular Area (M2)',
                      'Diencephalon - Posterior Tuberculum']
    atlas_dict[44] = ['Diencephalon - Preoptic Area',
                      'Diencephalon - Preoptic Otpb Cluster',
                      'Diencephalon - Preoptic area Vglut2 cluster',
                      'Diencephalon - Preoptic area posterior dopaminergic cluster',
                      'Diencephalon - Anterior preoptic dopaminergic cluster',
                      'Diencephalon - Oxtl Cluster 1 in Preoptic Area']
    atlas_dict[46] = ['Diencephalon - Pretectum',
                      'Diencephalon - Migrated Area of the Pretectum (M1)',
                      'Diencephalon - Anterior pretectum cluster of vmat2 Neurons']
    atlas_dict[48] = ['Rhombencephalon - Raphe - Inferior',
                      'Rhombencephalon - Raphe - Superior']
    atlas_dict[50] = []
    atlas_dict[52] = ['Diencephalon - Rostral Hypothalamus']
    atlas_dict[54] = ['Telencephalon - Subpallium']
    atlas_dict[56] = ['Mesencephalon - Tectum Stratum Periventriculare']
    atlas_dict[58] = ['Mesencephalon - Tegmentum']
    atlas_dict[60] = ['Diencephalon - Dorsal Thalamus',
                      'Diencephalon - Ventral Thalamus']
    atlas_dict[62] = ['Mesencephalon - Torus Longitudinalis']
    atlas_dict[64] = ['Mesencephalon - Torus Semicircularis']
    atlas_dict[66] = ['Ganglia - Trigeminal Ganglion']
    atlas_dict[68] = []
    atlas_dict[70] = []
    atlas_dict[72] = []
    
    return atlas_dict

def LRsplit(mask2idx_dict, midpoint, axis=0):
    new_mask2idx_dict = {}
    for key in np.sort(list(mask2idx_dict.keys())):
        Lids, Rids, Lcoords, Rcoords = [], [], [], []
        current_mask_spatial_coords = mask2idx_dict[key]['coords']
        current_mask_ids            = mask2idx_dict[key]['ids']
        if len(current_mask_spatial_coords) > 0:
            mask_centroid = np.mean(current_mask_spatial_coords, axis=1)

            for i in range(len(current_mask_spatial_coords)):
                current_coord = current_mask_spatial_coords[i]
                current_id    = current_mask_ids[i]

                if current_coord[axis] > midpoint[axis]:
                    Lids.append(current_id)
                    Lcoords.append(current_coord)
                else:
                    Rids.append(current_id)
                    Rcoords.append(current_coord)
        new_mask2idx_dict[key]    = {'coords': Lcoords, 'ids': Lids}
        new_mask2idx_dict[key+1] = {'coords': Rcoords, 'ids': Rids}
    return new_mask2idx_dict

def load_baier_connectome():
    import csv
    connectome_path = 'Baier_Connectome.csv'
    
    connectome = []
    with open(connectome_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in reader:
            if row_count > 0: 
                cropped_row = row[1:]
                float_row   = [np.float32(v) for v in cropped_row]
                connectome.append(float_row)
            row_count += 1
    return np.asarray(connectome, dtype='float32')
                      
def build_traces(traces, mask2idx_dict):
    roi_traces, roi_coords, mask_szs = [], [], []
    for key in np.sort(list(mask2idx_dict.keys())):
        current_mask = mask2idx_dict[key]
        current_ids    = current_mask['ids']
        current_coords = current_mask['coords']
        mask_szs.append(len(current_ids))
        if len(current_ids) > 0:
            roi_traces.extend(traces[current_ids,:])
            roi_coords.extend(current_coords)
    roi_traces = np.asarray(roi_traces)
    roi_coords = np.asarray(roi_coords)
    
    return roi_traces, roi_coords, mask_szs



def build_neural_S(mask2idx_dict, structural_connectome, N):
    S = np.zeros((N,N), dtype='float32')
    n1_count = 0
    for key1 in np.sort(list(mask2idx_dict.keys())):
        n1 = len(mask2idx_dict[key1]['ids'])
        n2_count = 0
        for key2 in np.sort(list(mask2idx_dict.keys())):
            n2 = len(list(mask2idx_dict[key2]['ids']))
            for i in range(n1_count, n1_count + n1):
                for j in range(n2_count, n2_count + n2):
                    if key1 == key2:
                        S[i,j] = np.float32(1.)
                    else:
                        S[i,j] = np.float32(structural_connectome[key1, key2])
            n2_count += n2
        n1_count += n1
    return S
    
def oasis_decon(traces):
    from oasis.functions import deconvolve

    decon, spikes = [], []
    count = 0
    for trace in traces:
        c, s, b, g, lam = deconvolve(trace, penalty=1, b_nonneg=False)
        if not np.any(c):
            c = np.random.normal(0., 0.1, size=(len(c)))
            count += 1
        decon.append(c)
        spikes.append(s)
    return np.asarray(decon), np.asarray(spikes)


def extract_top_J(filepath, prefix):
    filenames = os.listdir(filepath)
    full_filenames = [filepath + '/' + f for f in filenames]
    
    train_error_history = {}
    for full_filename in full_filenames:
        split_str = full_filename.split('-')[-1]
        iteration = int(split_str[:-4])
        
        f = np.load(full_filename, allow_pickle=True)
        train_error_history[iteration] = f['train_error']
        f.close()
        
    min_iter = min(train_error_history, key=train_error_history.get)
    best_filename = filepath + '/' + prefix + '-%i.npz' % min_iter
    training_err_idxs = np.sort(list(train_error_history.keys()))
    training_err_history_lst = []
    for idx in training_err_idxs:
        training_err_history_lst.append(train_error_history[idx])
        
    return training_err_history_lst, extract_training(best_filename)

def load_simulation_parameters(parameters_filepath):
    f = np.load(parameters_filepath, allow_pickle=True)
    S = f['S']
    times   = f['itimes']
    Ytruth  = f['Ytruth'].astype('float32')
    spatial = None
    if 'spatial' in f:
        spatial = f['spatial'].astype('float32')
    y0seed, y0_range, ntseed, noise_std = f['y0seed'], f['y0_range'], f['ntseed'], f['noise_std']
    tau, dt = f['tau'], f['dt']

    y0ran = np.random.RandomState(y0seed)
    y0 = y0ran.normal(0., y0_range, size=(Ytruth.shape[0],)).astype('float32')
    noiseran = np.random.RandomState(ntseed)
    H        = noiseran.normal(0., noise_std, size=Ytruth.shape).astype('float32')
    param_dict = {'S': S, 'times': times, 'y0': y0, 'H': H, 'Ytruth': Ytruth, 'spatial': spatial, 'tau': tau, 'dt': dt, 
                  'nstd': noise_std, 'y0range': y0_range}
    return param_dict

# def plot_ss(embedded, ax):
#     for i in range(embedded.shape[1] - 1):
#         ax.plot(embedded[0,i:i+2], embedded[1,i:i+2], color=plt.cm.inferno(255*i/embedded.shape[1])) 
        
def plot_example_traces(N, times, Ytruth, Ymodel, sz=4, idxs=None):
    if idxs is None:
        idxs = np.random.randint(0,N, size=(sz**2,))
    fig, ax = plt.subplots(sz,sz, figsize=(20, 10))
    for (i,ii) in enumerate(idxs):
        x = int(i/sz); y = int(i%sz)
        if Ytruth is not None:
            ax[x,y].plot(times, Ytruth[ii,:], label='truth', color='k')
        ax[x,y].plot(times, Ymodel[ii,:], label='modeled', color='r')
        ax[x,y].set_title(ii)
        if i < len(idxs)-1:
            ax[x,y].set_xticks([])
        #if i == 0 :
        #    ax[x,y].legend()
    return idxs

    
# def run_stability_analysis(save_filepath, niters, J, U, noises, duration, dt, hop, metric='correlation', min_ensemble_sz=3, noise_seed=1000000, parallel=False, verbose=False):
#     if not parallel:
#         _run_stability_analysis_serial(save_filepath, niters, J, U, noises, duration, dt, hop, noise_seed=noise_seed, verbose=verbose) 
#     elif parallel:
#         try:
#             from mpi4py import MPI
#         except:
#             print('Could not load MPI, running serial...')
#             _run_stability_analysis_serial(save_filepath, niters, J, U, noises, duration, dt, hop, noise_seed=noise_seed, verbose=verbose) 
#         _run_stability_analysis_parallel(save_filepath, niters, J, U, noises, duration, dt, hop, noise_seed, metric, min_ensemble_sz, verbose)        

                
# def _run_stability_analysis_parallel(save_filepath, niters, J, U, noises, duration, dt, hop, noise_seed, metric, min_ensemble_sz, verbose):
#     from mpi4py import MPI
#     import scipy.cluster.hierarchy as sch
#     from ensemble_utils import find_optimal_cutoff
    
    
#     logging.basicConfig(level=logging.DEBUG)
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()
  
#     if rank == 0:
#         tic = time.time()
        
#     tlength    = len(np.arange(0., duration, dt))
#     slicesz    = int(tlength / hop)
#     simulation_times = np.arange(0., duration * len(noises), 0.25)

#     full_information = []
#     N = J.shape[0]
    
    
#     # round robin distribution of iterations
#     iterations_to_process = []
#     for i in range(rank, niters, size):
#         iterations_to_process.append(i)
        
#     logging.info('rank %i ready to process %i iterations', rank, len(iterations_to_process))
     
#     for citer in iterations_to_process:
#         ctic = time.time()
#         noise_rnd   = np.random.RandomState(noise_seed + citer)
#         noise_trace = np.zeros((N, len(simulation_times)), dtype='float32')
#         for (idx, noise) in enumerate(noises):
#             st,sp = idx * tlength, (idx + 1) * tlength
#             noise_trace[:,st:sp] = noise_rnd.normal(0., noise, size=(N, sp-st))
#         noise_trace = noise_trace.astype('float32')
#         model_currents = rnn_modeling.solve_euler2(simulation_times, J, noise_trace, U, np.zeros((N,), dtype='float32'))
#         Ymodel         = np.matmul(J, np.tanh(model_currents))
#         del noise_trace
#         del model_currents
        
       
#         clustering_info = {}
#         for i in range(0, len(simulation_times), hop):
#             link = sch.linkage(Ymodel[:,i:i+hop], metric='euclidean', method='ward')
#             d    = sch.distance.pdist(Ymodel[:,i:i+hop], metric=metric).astype('float32')
#             dmax = d.max()
#             cutoff = find_optimal_cutoff(link, dmax, min_ensemble_sz=min_ensemble_sz)
#             del d
#             inds   = sch.fcluster(link, cutoff, 'distance')
#             k      = len(set(inds))
#             output_dict = {'k': k, 'idxs': returned_idxs, 'start': i, 'stop':i+hop }
#             clustering_info[i] = output_dict
            
#         full_information.append(clustering_info)
#         celapsed = time.time() - ctic
#         if verbose:
#             logging.info('It took %0.3f seconds for rank %i to complete iteration %i', celapsed, rank, citer)
            
#     combined_full_information = comm.gather(full_information, root=0)
#     if rank == 0:
#         full_information = []
#         for cfi in combined_full_information:
#             full_information.extend(cfi)
#         np.savez(save_filepath, data=full_information, noises=noises, seed=noise_seed, niter=niters, duration=duration, N=N, dt=dt, J=J, U=U, hop=hop, slicesz=slicesz)
#         full_elapsed = time.time() - tic
#         logging.info('It took %0.3f seconds to run %i iterations', full_elapsed, niters)

        
# def _run_stability_analysis_serial(save_filepath, niters, J, U, noises, duration, dt, hop, noise_seed, verbose):
#     tic = time.time()
#     tlength          = len(np.arange(0., duration, dt))
#     slicesz    = int(tlength / hop)
#     simulation_times = np.arange(0., duration * len(noises), 0.25)

#     full_information = []
#     N = J.shape[0]
#     for citer in range(niters):
#         noise_rnd   = np.random.RandomState(noise_seed + citer)
#         noise_trace = np.zeros((N, len(simulation_times)), dtype='float32')
#         for (idx, noise) in enumerate(noises):
#             st,sp = idx * tlength, (idx + 1) * tlength
#             noise_trace[:,st:sp] = noise_rnd.normal(0., noise, size=(N, sp-st))
#         noise_trace = noise_trace.astype('float32')
#         model_currents = rnn_modeling.solve_euler2(simulation_times, J, noise_trace, U, np.zeros((N,), dtype='float32'))
#         Ymodel         = np.matmul(J, np.tanh(model_currents))
#         del noise_trace
#         del model_currents
#         clustering_info = {}
#         for i in range(0, len(simulation_times), hop):
#             returned_idxs = clustering_idxs(Ymodel[:,i:i+hop])
#             k = len(set(returned_idxs))
#             output_dict = {'k': k, 'idxs': returned_idxs, 'start': i, 'stop':i+hop }
#             clustering_info[i] = output_dict

#         if citer > 0 and citer % 5 == 0 and verbose: print(citer) 
#         full_information.append(clustering_info)
#     elapsed = time.time() - tic
#     if verbose:
#         print('it took %0.3f seconds to complete %i iterations' % (elapsed, niters))
        
#     np.savez(save_filepath, data=full_information, noises=noises, seed=noise_seed, niter=niters, duration=duration, N=N, dt=dt, J=J, U=U, hop=hop, slicesz=slicesz)
           

            
# def sanalysis_rasterplot(full_filepath, vmax=15):
#     f = np.load(full_filepath, allow_pickle=True)
#     data    = f['data']
#     niters  = f['niter']
#     slicesz = f['slicesz']
#     noises  = f['noises']
#     f.close()
    
#     kss = []
#     for iteration in range(niters):
#         ks = []
#         current_clustering_info = data[iteration]
#         for start_time in np.sort(current_clustering_info.keys()):
#             ks.append(current_clustering_info[start_time]['k'])
#         kss.append(ks)
#     kss = np.asarray(kss)
    
#     plt.figure(figsize=(12,8))
#     plt.imshow(kss, cmap='inferno', vmin=0., vmax=vmax, aspect='auto')
#     plt.colorbar()
#     plt.xticks([])
    
#     for i in range(0, slicesz*len(noises), slicesz):
#         plt.plot(np.ones((niters,))*i, np.arange(niters), color='white', linestyle='dashed')
#     plt.show() 
    
# def sanalysis_kplot(full_filepaths, stat='mean'):
#     if type(full_filepaths) is not list:
#         raise Exception('Passed arguments must be of type list')
        
#     if stat != 'mean' and stat != 'var':
#         raise Exception('stat variable must be either mean or var')
        
#     data_lst, slicesz_lst, nnoises_lst, niters_lst = [], [], [], []
#     for i in range(len(full_filepaths)):
#         f = np.load(full_filepaths[i], allow_pickle=True)
#         data_lst.append(f['data'])
#         slicesz_lst.append(f['slicesz'])
#         nnoises_lst.append(len(f['noises']))
#         niters_lst.append(f['niter'])
#         f.close()
    
#     colors = ['k','r','b','g']
#     fig, ax = plt.subplots(figsize=(12,8))
#     for i in range(len(data_lst)):
#         curr_data = data_lst[i]
#         kss = []
#         for iteration in range(niters_lst[i]):
#             ks = []
#             current_clustering_info = curr_data[iteration]
#             for start_time in np.sort(current_clustering_info.keys()):
#                 ks.append(current_clustering_info[start_time]['k'])
#             kss.append(ks)
#         kss = np.asarray(kss)
     
#         means = []
#         stds  = []
#         for j in range(0, nnoises_lst[i] * slicesz_lst[i], slicesz_lst[i]):
#             s = kss[:,j:j+slicesz_lst[i]]
#             if stat == 'mean':
#                 means.append(np.mean(np.mean(s,axis=1)))
#                 stds.append(np.std(np.mean(s,axis=1)))
#             elif stat == 'var':
#                 means.append(np.mean(np.std(s,axis=1)))
#                 stds.append(np.std(np.std(s,axis=1)))
                          
#         means = np.asarray(means)
#         stds  = np.asarray(stds)
        
#         ax.plot(means, color=colors[i])
#         ax.scatter(range(len(means)), means, color=colors[i])
#         #ax.fill_between(range(len(means)), means - stds, means + stds, color=colors[i], alpha=0.1)
        
#     ax.set_xticks([])
      
        
 # mutual information

# def generate_mi_mats(save_prefix, J, U, noises, duration, dt, hop, noise_seed=1000000, parallel=False, verbose=False):  
#     if not parallel:
#         _generate_mi_mats_serial(save_prefix, J, U, noises, duration, dt, hop, noise_seed, verbose) 
#     elif parallel:
#         try:
#             from mpi4py import MPI
#         except:
#             print('Could not load MPI, running serial...')
#             _generate_mi_mats_serial(save_prefix, J, U, noises, duration, dt, hop, noise_seed, verbose)   
#         _generate_mi_mats_parallel(save_prefix, J, U, noises, duration, dt, hop, noise_seed, verbose)   


# def _calc_mi(data):
#     try:
#         import smite
#     except:
#         raise Exception('Could not load smite..')
        
#     N    = data.shape[0]
#     syms = [smite.symbolize(data[i,:],3) for i in range(N)]
#     mi_matrix = np.zeros((N,N))
#     for i in range(N):
#         for j in range(i):
#             TM = smite.symbolic_mutual_information(syms[i], syms[j])
#             mi_matrix[i,j] = TM; mi_matrix[j,i] = TM
#     return mi_matrix
    
# def _generate_mi_mats_serial(save_prefix, J, U, noises, duration, dt, hop, noise_seed, verbose):  
#     tic = time.time()
#     tlength    = len(np.arange(0., duration, dt))
#     slicesz    = int(tlength / hop)
#     simulation_times = np.arange(0., duration * len(noises), 0.25)

#     N = J.shape[0]
#     noise_rnd   = np.random.RandomState(noise_seed)
#     noise_trace = np.zeros((N, len(simulation_times)), dtype='float32')
#     for (idx, noise) in enumerate(noises):
#         st,sp = idx * tlength, (idx + 1) * tlength
#         noise_trace[:,st:sp] = noise_rnd.normal(0., noise, size=(N, sp-st))
#     noise_trace = noise_trace.astype('float32')
    
#     model_currents = rnn_modeling.solve_euler2(simulation_times, J, noise_trace, U, np.zeros((N,)))
#     Ymodel         = np.matmul(J, np.tanh(model_currents))
    
#     del model_currents
#     del noise_trace
    
#     for (ci, i) in enumerate(range(0, len(simulation_times), hop)):
#         current_tic = time.time()
#         mi_mat = _calc_mi(Ymodel[:,i:i+hop])
#         save_filepath = save_prefix + '-%i.npz' % ci
#         np.savez(save_filepath, mi=mi_mat, N=N, seed=noise_seed, duration=duration, dt=dt, hop=hop, noises=noises, slicesz=slicesz, start=i, stop=i+hop)
#         current_elapsed = time.time() - current_tic
#         if verbose:
#             print('It took %0.3f seconds to calculate the MI matrix' % current_elapsed)
            
#     elapsed = time.time() - tic
#     if verbose:
#         print('It took %0.3f seconds to calculate the MI matrix %i times' % (elapsed, len(mi_mats)))

# def _generate_mi_mats_parallel(save_prefix, J, U, noises, duration, dt, hop, noise_seed, verbose):  
#     from mpi4py import MPI
    
#     logging.basicConfig(level=logging.DEBUG)
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()
    
#     if verbose:
#         logging.info('rank %i ready to process..', rank)
              
#     if rank == 0:
#         tic = time.time()

#     tlength    = len(np.arange(0., duration, dt))
#     slicesz    = int(tlength / hop)
#     simulation_times = np.arange(0., duration * len(noises), 0.25)

#     N = J.shape[0]
#     noise_rnd   = np.random.RandomState(noise_seed)
#     noise_trace = np.zeros((N, len(simulation_times)), dtype='float32')
#     for (idx, noise) in enumerate(noises):
#         st,sp = idx * tlength, (idx + 1) * tlength
#         noise_trace[:,st:sp] = noise_rnd.normal(0., noise, size=(N, sp-st))
#     noise_trace = noise_trace.astype('float32')
    
#     model_currents = rnn_modeling.solve_euler2(simulation_times, J, noise_trace, U, np.zeros((N,)))
#     Ymodel         = np.matmul(J, np.tanh(model_currents))
#     del model_currents
#     del noise_trace
    
#     idxs_to_process = []
#     nmats = int(len(simulation_times) / hop)
#     for i in range(rank, nmats, size):
#         idxs_to_process.append(i)
    
#     for idx in idxs_to_process:
#         start, stop = idx * hop, (idx + 1) * hop
        
#         current_tic = time.time()
#         mi_mat = _calc_mi(Ymodel[:,start:stop])
#         save_filepath = save_prefix + '-%i.npz' % idx
#         np.savez(save_filepath, mi=mi_mat, N=N, seed=noise_seed, duration=duration, dt=dt, hop=hop, noises=noises, slicesz=slicesz, start=start, stop=stop)
#         current_elapsed = time.time() - current_tic
#         logging.info('It took %0.3f seconds to process matrix %i', current_elapsed, idx)
     
#     if rank == 0:
#         elapsed = time.time() - tic
#         if verbose:
#             logging.info('It took %0.3f seconds to process %i matrices', elapsed, nmats)
        
    
        
        
        
        

