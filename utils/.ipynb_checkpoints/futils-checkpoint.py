import numpy as np
from copy import deepcopy

def read_data(filepath):
    f = np.load(filepath)
    control_traces = f['control_traces'].astype('float32')
    presz_traces   = f['presz_traces'].astype('float32')
    sz_traces      = f['sz_traces'].astype('float32')
    postsz_traces  = f['postsz_traces'].astype('float32')
    f.close()
    traces = np.concatenate((control_traces.T, presz_traces.T, sz_traces.T, postsz_traces.T)).T
    return traces.astype('float32')

def read_metrics_filepath(metrics_filepath, nmatrices, idxs, verbose=False):
    degs, evcs, ccs = [], [], []
    valid_idxs = None
    for i in range(0, nmatrices):
        if i % 100 == 0 and verbose: print(i)
        f = np.load('%s/metrics-fc-%i.npz' % (metrics_filepath, i))
        if i == 0 and idxs is not None:
            valid_idxs = list(set(range(len(f['degree']))) - set(idxs))
        degs.append(np.asarray(f['degree'])[valid_idxs])
        evcs.append(np.asarray(f['eigenvector'])[valid_idxs])
        ccs.append(np.asarray(f['cc'])[valid_idxs])
        f.close()
    degs = np.asarray(degs, dtype='float32')
    evcs = np.asarray(evcs, dtype='float32')    
    ccs  = np.asarray(ccs, dtype='float32')
    return degs, evcs, ccs

def calc_fc_matrix(data, todict=False):
    data  = data.astype('float32')
    xcorr = np.corrcoef(data).astype('float32')
    if not todict:
        return xcorr
    else:
        N, _ = xcorr.shape
        d = {n:[] for n in range(N)}
        for i in range(N):
            for j in range(i):
                d[i].append(xcorr[i,j])
        return d  
    
def get_hilbert(X):
    from scipy.signal import hilbert
    return hilbert(X)

def get_phases(X):
    return np.unwrap(np.angle(X))

def get_synch(phases):
    import cmath
    N = len(phases)
    mean_phase = np.mean(phases) # variance arosss
    synch = 1./float(N) * np.sum([cmath.exp(1j*(tp-mean_phase)) for tp in phases])
    return np.absolute(synch)


def pca_analysis(X, n_components=2, kernel='rbf',pca=None):
    Xtransformed = None
    if pca is None:
        pca = KernelPCA(n_components=n_components, kernel=kernel)
        Xtransformed = pca.fit_transform(X.T).T
    else:
        try:
            Xtransformed = pca.transform(X.T).T
        except:
            raise Exception('passed pca object invalid..')
    return pca, Xtransformed
    
def data2percentile(data, method='average'):
    from scipy.stats import rankdata
    ranked_data = rankdata(data, method=method)
    return ranked_data / float(len(data))

def permute_weights(J, seed=0):
    newJ = np.zeros_like(J).astype('float32')
    srcs,dsts = np.nonzero(J)
    locs, weights = [], []
    for (src, dst) in list(zip(srcs,dsts)):
        locs.append((src,dst)); weights.append(J[src,dst])
        
    rs = np.random.RandomState(seed=seed)
    rs.shuffle(locs)
    for t, (src,dst) in enumerate(locs):
        newJ[src,dst] = weights[t]
        
    return newJ

def get_cutoff(J, percentile_cutoff):
    srcs, dsts = J.nonzero()
    weights_lst = []
    for (src, dst) in list(zip(srcs, dsts)):
        weights_lst.append(J[src,dst])
    cutoff_val = np.percentile(weights_lst, percentile_cutoff)
    return cutoff_val

def jthreshold(J, percentile_cutoff, above=True, pos=True, binarized=False, cutoff_val=None):
    Jcopy = deepcopy(J)
    if not pos:
        Jcopy *= -1
    
    Jcopy = np.clip(Jcopy, 0., np.max(Jcopy))
    if cutoff_val is None:
        cutoff_val = get_cutoff(Jcopy, percentile_cutoff)
    if above:
        Jcopy[Jcopy < cutoff_val]  = 0
        if binarized:
            Jcopy[Jcopy >= cutoff_val] = 1
            Jcopy = Jcopy.astype('uint8')
        else:
            Jcopy = Jcopy.astype('float32')     
    else:
        ps,pd = np.where(Jcopy >= cutoff_val)
        zs,zd = np.where((Jcopy < cutoff_val)&(Jcopy > 0.0))
        Jcopy[ps,pd] = 0
        if binarized:
            Jcopy[zs,zd]=1
            Jcopy = Jcopy.astype('uint8')
        else:
            Jcopy = Jcopy.astype('float32')            
        Jcopy[np.diag_indices(Jcopy.shape[0])] = 0

    return Jcopy

def extract_weights(J, directed=True, nonzero=False):
    Jcopy = deepcopy(J)
    if not directed:
        Jcopy = np.triu(Jcopy)
      
    srcs, dsts = Jcopy.nonzero()
    srcdst = list(zip(srcs, dsts))
    weights_lst = [J[src,dst] for (src,dst) in srcdst]
    return np.asarray(weights_lst, dtype='float32')


def get_ei(w):
    w = np.asarray(w)
    pos = np.sum(w[w > 0])
    neg = np.sum(w[w < 0])
    return abs(pos / neg)


def gaussian_mixture_modeling(ei, Nmax=11, minn=0., maxx=1.):
    from sklearn.mixture import GaussianMixture
    ei = np.asarray(ei, dtype='float32')    
    N = np.arange(1,Nmax)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(ei.reshape(-1,1))
    model_AIC = [cm.aic(ei.reshape(-1,1)) for cm in models]


    model_bestM = models[np.argmin(model_AIC)]

    x = np.linspace(minn,maxx,1000)
    logprob = model_bestM.score_samples(x.reshape(-1,1))
    resp    = model_bestM.predict_proba(x.reshape(-1,1))
    pdf     = np.exp(logprob)
    pdf_individual = resp * pdf[:,np.newaxis]


    model_labels = model_bestM.fit_predict(ei.reshape(-1,1)).reshape(-1,)
    return x, pdf_individual, model_labels


def ensemble_detection(X, run_sam=False, **kwargs):
    import umap
    import hdbscan
    from sklearn.cluster import DBSCAN
    metric   = kwargs.get('metric', 'euclidean')
    k        = kwargs.get('k', 5)
    min_dist = kwargs.get('min_dist', 0.05)

    umap_data, cluster_labels, sam = None, None, None
    if run_sam:
        from SAM import SAM

        N, T = X.shape
        counts = (X, np.arange(T), np.arange(N))
        
        npcs = kwargs.get('npcs', 5)
        resolution = kwargs.get('resolution', 2.0)
        stopping_condition = kwargs.get('stopping_condition', 5e-4)
        max_iter = kwargs.get('max_iter', 25)
        
        sam  = SAM(counts)
        sam.run(verbose=False, projection='umap', k=k, npcs=npcs, preprocessing='Normalizer', distance=metric, 
                stopping_condition=stopping_condition, max_iter=max_iter,
                proj_kwargs={'metric': metric, 'n_neighbors': k, 'min_dist': min_dist})
        param = kwargs.get('resolution', 1.0)
        umap_data = sam.adata.obsm['X_umap']    
        sam.clustering(X=None, param=param, method='leiden')
        cluster_labels = sam.adata.obs['leiden_clusters']
        cluster_labels = [cluster_labels.iloc[i] for i in range(N)]
         
    else:

        umapy = umap.UMAP(n_components=2, min_dist=min_dist, n_neighbors=k)
        umap_data = umapy.fit_transform(X)
        
        clustering = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clustering.fit_predict(umap_data)
        
    return sam, umap_data, cluster_labels

def find_optimal_cutoff(Z, dmax, min_ensemble_sz=3):
    cutoff = 0.01
    found = False
    while not found:
        ind = sch.fcluster(Z, cutoff*dmax, 'distance')
        szs = get_ensemble_sz(ind)
        if np.min(szs) >= min_ensemble_sz: 
            found = True
        cutoff += 0.01  
    return cutoff*dmax

def hierarchical_clustering(X, k, metric='euclidean', method='ward'):
    import scipy.cluster.hierarchy as sch
    link = sch.linkage(X, metric=metric, method=method)
    inds   = sch.fcluster(link, k, 'maxclust')
    return inds
        
def get_single_IAAFT_surrogate(ts, n_iterations = 10, seed = None):
    """
    Returns single iterative amplitude adjusted FT surrogate.
    n_iterations - number of iterations of the algorithm.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate IAAFT surrogates).
    """

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    xf = np.fft.rfft(ts, axis = 0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
    del xf

    return _compute_IAAFT_surrogates([ None, n_iterations, ts, angle])[-1]

def _compute_IAAFT_surrogates(a):
    i, n_iters, data, angle = a

    xf = np.fft.rfft(data, axis = 0)
    xf_amps = np.abs(xf)
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)

    # starting point
    R = _compute_AAFT_surrogates([None, data, angle])[-1]

    # iterate
    for _ in range(n_iters):
        r_fft = np.fft.rfft(R, axis = 0)
        r_phases = r_fft / np.abs(r_fft)

        s = np.fft.irfft(xf_amps * r_phases, n = data.shape[0], axis = 0)

        ranks = s.argsort(axis = 0).argsort(axis = 0)
        R = sorted_original[ranks]

    return (i, R)

def _compute_AAFT_surrogates(a):
    i, data, angle = a

    # create Gaussian data
    gaussian = np.random.randn(data.shape[0])
    gaussian.sort(axis = 0)

    # rescale data
    ranks = data.argsort(axis = 0).argsort(axis = 0)
    rescaled_data = gaussian[ranks]

    # transform the time series to Fourier domain
    xf = np.fft.rfft(rescaled_data, axis = 0)
     
    # randomise the time series with random phases     
    cxf = xf * np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = np.fft.irfft(cxf, n = data.shape[0], axis = 0)

    # rescale back to amplitude distribution of original data
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)
    ranks = ft_surr.argsort(axis = 0).argsort(axis = 0)

    rescaled_data = sorted_original[ranks]
    
    return (i, rescaled_data)

# class RunningdJ(object):
#     def __init__(self,N):
#         self.N = N
#         self.dJ = np.empty_like((N,N), dtype='float32')
#     def clear(self):
#         self.J = np.empty_like((self.N, self.N), dtype='float32')        
    
#     def getJ(self):
#         J = np.empty_like((self.N, self.N), dtype='float32')
#         for (pos,idx) in enumerate(self.idxs):
#             J[idx,:] = self.J[pos]
#         return J
    
#     @classmethod
#     def combine(cls, a, br, idxs):
#         combined = cls(a.N)
#         combined.dJ[idxs] =     = a.J.extend(br)
#         return combined       
        
                
class RunningStats(object):

    def __init__(self):
        self.n = 0
        self.m1 = 0.

    def clear(self):
        self.n = 0
        self.m1 = 0.
       

    def update(self, x):
        n1 = self.n
        self.n += 1
        n = self.n
        delta = x - self.m1
        delta_n = delta / n
        self.m1 += delta_n

    def mean(self):
        return self.m1

    def variance(self):
        return self.m1 / (self.n - 1.0)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    
    @classmethod
    def combine(cls, a, b):
        combined = cls()
        
        combined.n = a.n + b.n    
        combined.m1 = (a.n*a.m1 + b.n*b.m1) / combined.n;
        
        return combined
