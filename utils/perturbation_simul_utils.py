
import numpy as np
from modeling_analysis_utils import load_simulation_parameters
from rnn_modeling_utils import solve_stochastic_euler
from futils import extract_weights, jthreshold
import time
from copy import deepcopy

def null_ids(J, idxs):
    from copy import deepcopy
    Jcopy = deepcopy(J)
    Jcopy[:,idxs] = 0.
    #Jcopy[idxs,:] = 0.
    return Jcopy.astype('float32')


def multi_ablation(niterations, output_filepath, J, **kwargs):
    save = kwargs.get('save', True)
    for niter in range(niterations):
        tic = time.time()
        Y, Ja = ablation_expr(J, **kwargs)
        if save:
            elapsed = time.time() - tic
            np.savez(output_filepath + '/simulation-%d.npz' % niter, Y=Y)
            print('iteration %d took %0.3f seconds' % (niter, elapsed))
        
        
def ablation_expr(J, **kwargs):
    if 'parameters_filepath' not in kwargs:
        return None, None
    parameters_filepath = kwargs['parameters_filepath']
    N    = kwargs['N']
    idxs = kwargs['idxs']
    
    if N is not None:
        np.random.shuffle(idxs)
        picked = idxs[:N]
    else:
        picked = idxs
        
    Ja     = null_ids(J, picked)
    
    simulation_kwargs = load_simulation_parameters(parameters_filepath)
    simulation_kwargs['J'] = Ja

    currents    = solve_stochastic_euler(**simulation_kwargs)
    activations = np.tanh(currents)
    Ymodel      = np.matmul(Ja, activations)
    return Ymodel, currents, Ja
    

def perturbation_expr(J, **kwargs):
    if 'perturbation' not in kwargs:
        return None
    if 'parameters_filepath' not in kwargs:
        return None
    
    simulation_kwargs = load_simulation_parameters(kwargs['parameters_filepath'])
    simulation_kwargs['J'] = J
    
    
    H = simulation_kwargs['H']
    perturbation_params = kwargs['perturbation']
    nids = np.asarray(perturbation_params['nids'], dtype='uint32')
    ton  = perturbation_params['ton']
    toff = perturbation_params['toff']
    current_direction = perturbation_params['direction']
    
    for i in range(len(ton)):
        on, off, direction = int(ton[i]), int(toff[i]), current_direction[i]
        if on >= H.shape[1] or off >= H.shape[1]:
            print('on/off (%d, %d) time out of bounds, ignoring...' % (on, off))
            continue
        if direction == 'up':
            H[nids,on:off] = 1e9
        elif direction == 'down':
            H[nids,on:off] = -1e9
        elif direction == 'noise_off':
            H[:,on:off] = 0.
        else:
            raise Exception('option %s not implemented' % direction)
            
    simulation_kwargs['H'] = H
    currents    = solve_stochastic_euler(**simulation_kwargs)
    activations = np.tanh(currents)
    Ymodel      = np.matmul(J, activations)
    return Ymodel, currents
    
    
    

def multi_edge_swap_expr(niterations, output_filepaths, J, expr_types, **kwargs):
    save = kwargs.get('save', True)
    for niter in range(niterations):
        tic = time.time()
        swapped = None
        for nexpr in range(len(expr_types)):
            kwargs['expr_type'] = expr_types[nexpr]
            kwargs['toswap'] = swapped
            Y, _, swapped = edge_swap_expr(J, **kwargs)
            print(expr_types[nexpr])
            if save:
                elapsed = time.time() - tic
                np.savez(output_filepaths[nexpr] + '/simulation-%d.npz' % niter, Y=Y)
                print('iteration %d took %0.3f seconds' % (niter, elapsed))
           
def edge_swap_expr(J, **kwargs):
    if 'parameters_filepath' not in kwargs:
       return None, None

    parameters_filepath = kwargs['parameters_filepath']
    expr_type = kwargs.get('expr_type', 'random')
    nedges = kwargs.get('nedges', 100)
    tover  = kwargs.get('tover', 90)
    tunder = kwargs.get('tunder', 10)
    
    Ja = deepcopy(J)   
    edges2swap = []
    if expr_type == 'random':
        Jth_over = jthreshold(Ja, tover, binarized=True, pos=True, above=True).T
        Jth_over_srcs, Jth_over_dsts = Jth_over.nonzero()
        Jth_over_srcdst = list(zip(Jth_over_srcs, Jth_over_dsts))
        
        np.random.shuffle(Jth_over_srcdst)
        edges2swap = Jth_over_srcdst[:nedges]
       
    
    elif expr_type == 'bifan':
        triplet_edges = kwargs['triplet_edges']
        bifan_edges   = kwargs['bifan_edges']
        np.random.shuffle(bifan_edges)
        
        for edge in bifan_edges:
            if edge not in edges2swap:
                edges2swap.append(edge)      
        if len(edges2swap) >= nedges:
            edges2swap = edges2swap[0:nedges]
        else:
            np.random.shuffle(triplet_edges)
            count = len(edges2swap)
            for trip in triplet_edges:
                if trip not in edges2swap: 
                    edges2swap.append(trip)
                    count += 1
                if count == nedges: break
                        
    elif expr_type == 'triplets':
        triplet_edges = kwargs['triplet_edges']
        bifan_edges   = kwargs['bifan_edges']
        
        np.random.shuffle(triplet_edges)
        count = 0
        for edge in triplet_edges:
            if edge not in bifan_edges:
                edges2swap.append(edge)
                count += 1
            if count == nedges: break
            
    toswapwith = kwargs.get('toswap', None)
    if toswapwith is None:

        Jth_under = jthreshold(Ja, tunder, binarized=True, pos=True, above=False).T
        Jth_under_srcs, Jth_under_dsts = Jth_under.nonzero()
        Jth_under_srcdst = list(zip(Jth_under_srcs, Jth_under_dsts))       
        np.random.shuffle(Jth_under_srcdst)    
        toswapwith = Jth_under_srcdst[:nedges]
    
    swap = kwargs.get('swap', True)
    if swap:
        for edge_num in range(nedges):
            src1, dst1 = edges2swap[edge_num]
            src2, dst2 = toswapwith[edge_num]
            Ja[dst1,src1], Ja[dst2,src2] = Ja[dst2,src2], Ja[dst1,src1] # double check
    else:
        for edge_num in range(nedges):
            src, dst = edges2swap[edge_num]
            Ja[dst, src] = 0.
            
    simulation_kwargs = load_simulation_parameters(parameters_filepath)
    simulation_kwargs['J'] = Ja

    currents    = solve_stochastic_euler(**simulation_kwargs)
    activations = np.tanh(currents)
    Ymodel      = np.matmul(Ja, activations)
            
    return Ymodel, currents, Ja, toswapwith


def acquire_simulation_statistics(input_filepath, output_filepath, niterations):
    bins = np.linspace(-1.,1.,200)
    testing_hists = []
    for i in range(niterations):
        try:
            f = np.load(input_filepath + '/simulation-%d.npz'%i)
            Y = f['Y'].astype('float32')
            f.close()
            corr = np.corrcoef(Y).astype('float32')
            corr_weights = extract_weights(corr, directed=False)
            hist, edges = np.histogram(corr_weights, bins=bins)
            testing_hists.append(hist)
        except:
            print('error: %d' % i)
        if i % 10 == 0: print(i)
        
    testing_hists = np.asarray(testing_hists, dtype='float32')
    testing_median = np.median(testing_hists,axis=0)
    testing_mean  = np.mean(testing_hists,axis=0)
    testing_std   = np.std(testing_hists,axis=0)
    np.savez(output_filepath, med=testing_median, mean=testing_mean, std=testing_std, edges=edges)
    
      
    
def get_dists(Y, bins):
    Ycorr    = np.corrcoef(Y)
    Ycorr[np.diag_indices(Ycorr.shape[0])] = 0.
    Yweights = extract_weights(Ycorr, directed=False)
    Yweights = Yweights[Yweights > 0.]

    density, edges = np.histogram(Yweights, bins=bins, density=False)
    cumul = np.cumsum(density)
    cumul = cumul / cumul[-1]
    return cumul, density, edges

def plot_diagnostics(Y, base, pre, plot_cumul=False):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1,3,figsize=(14,6))
    ax[1].imshow(np.corrcoef(Y), cmap='inferno', vmin=-1., vmax=1.)
    ax[1].set_title('new')
    ax[0].imshow(np.corrcoef(base), cmap='inferno', vmin=-1., vmax=1.)
    ax[0].set_title('baseline')
    ax[2].imshow(np.corrcoef(pre), cmap='inferno', vmin=-1., vmax=1.)
    ax[2].set_title('presz')
    ax[1].set_xticks([]); ax[0].set_xticks([]); ax[2].set_xticks([])
    ax[1].set_yticks([]); ax[0].set_yticks([]); ax[2].set_yticks([])
    plt.show()

    if plot_cumul:
        bins = np.linspace(0,1,100)
        ycumul, _, yedges = get_dists(Y, bins)
        bcumul, _, bedges = get_dists(base, bins)
        pcumul, _, pedges = get_dists(pre, bins)

        plt.figure(figsize=(12,8))
        plt.plot(yedges[1:], ycumul, color='r')
        plt.plot(bedges[1:], bcumul, color='k', linestyle='--')
        plt.plot(pedges[1:], pcumul, color='r', linestyle='--')
        plt.show() 