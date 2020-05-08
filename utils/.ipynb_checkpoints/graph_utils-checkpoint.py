import numpy as np
import sys, os
import snap
import networkx as nx
import igraph as ig

def generate_snap_unweighted_graph(A, directed=False, save=None):
    N, _ = A.shape
    nids = np.arange(N)
    
    graph = None
    if directed:
        graph = snap.TNGraph.New()
    else:
        A = np.triu(A)
        graph = snap.TUNGraph.New()
    
    for nid in nids: graph.AddNode(int(nid))
    srcs,dsts = np.where(A == 1)
    for (src, dst) in list(zip(srcs,dsts)):
        if src == dst: continue
        graph.AddEdge(int(src), int(dst))  
    if save is not None:
        try:
            snap.SaveEdgeList(graph, save)
        except:
            print('Error, could not save to %s' % save)

    return graph


def generate_nx_graph(A, directed=False, weighted=False):
    graph = None
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
        A = np.triu(A)
    graph.add_nodes_from(range(A.shape[0]))
    
    srcs,dsts = np.nonzero(A)
    for (src, dst) in list(zip(srcs,dsts)):
        if src == dst: continue
        if weighted:
            graph.add_edge(src, dst, weight=A[src, dst])
        else:
            graph.add_edge(src, dst)       
    return graph

def generate_igraph(A, directed=False):
    if not directed:
        A = np.triu(A)
    sources, targets = A.nonzero()
    weights = A[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(A.shape[0])
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except BaseException:
        print('base exception excepted..')
        pass
    return g, weights

def get_node_centrality(snap_graph):
    nids, deg_centr = [], []
    for NI in snap_graph.Nodes():
        centr = snap.GetDegreeCentr(snap_graph, NI.GetId())
        nids.append(NI.GetId())
        deg_centr.append(centr)
        
    return np.asarray(nids, dtype='uint32'), np.asarray(deg_centr, dtype='float32')


def get_eigenvector_centrality(graph, gtype='snap'):
    nids, ev_centr = [], []
    if gtype == 'snap':
        NIdEigenH = snap.TIntFltH()
        snap.GetEigenVectorCentr(snap_graph, NIdEigenH)
        for item in NIdEigenH:
            nids.append(item)
            ev_centr.append(NIdEigenH[item])
    elif gtype == 'nx':
        centrality = nx.eigenvector_centrality(graph)
        nids, ev_centr = [], []
        for node in np.sort(list(centrality.keys())):
            nids.append(node)
            ev_centr.append(centrality[node])
            
    return np.asarray(nids, dtype='uint32'), np.asarray(ev_centr, dtype='float32')


def get_clustering_coefficient(graph, gtype='snap'):
    nids, ccs = [], []
    if gtype == 'snap':
        NIdCCfH = snap.TIntFltH()
        snap.GetNodeClustCf(graph, NIdCCfH)
        for item in NIdCCfH:
            nids.append(item)
            ccs.append(NIdCCfH[item])
    elif gtype == 'nx':
        cc_output = nx.clustering(graph)
        for nid in np.sort(list(cc_output.keys())):
            nids.append(nid)
            ccs.append(cc_output[nid])
    return np.asarray(nids, dtype='uint32'), np.asarray(ccs, dtype='float32')

          
def get_transitivity(graph, gtype='nx'):
    t = None
    if gtype == 'nx':
        t = nx.transitivity(graph)
    elif gtype == 'snap':
        t = 1.
    return np.float32(t)


def get_outgoing_degrees(J, threshold, pos=True):
    from futils import jthreshold
    Jth_over = jthreshold(J, threshold, binarized=True, pos=pos, above=True)
    outgoing_degrees = []
    for i in range(J.shape[0]):
        outgoing = Jth_over[:,i]
        outgoing_degrees.append(np.sum(outgoing))
    outgoing_degrees = np.asarray(outgoing_degrees, dtype='uint32')   
    return outgoing_degrees


def extract_outgoing_hubs(J, jthreshold, hub_percentile, N, idxs=None):
    outgoing_degrees = get_outgoing_degrees(J, jthreshold)
    cutoff = np.quantile(outgoing_degrees, hub_percentile)
    outgoing_hub_idxs = np.where(outgoing_degrees > cutoff)[0]
    if idxs is None:
        valid_nids = outgoing_hub_idxs
    else:
        valid_nids = []
        for nid in outgoing_hub_idxs:
            if nid not in idxs: 
                valid_nids.append(nid)
    
    return valid_nids, outgoing_degrees

def louvain_clustering(adjacency, directed=False):
    import community 
    
    graph = generate_nx_graph(adjacency, directed=directed, weighted=True)
    part = community.best_partition(graph, weight='weight')
    modularity = community.modularity(part, graph)
    return part, modularity
        
              
def leiden_clustering(adjacency, res=1.0, directed=False, binary=None, part=None):
    import leidenalg
    
    g, weights = None, None
    if binary is None:
        g, weights= generate_igraph(adjacency,directed=directed)
    else:
        g, weights = generate_igraph(binary, directed=directed) 
    if part is None:
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res)
        part = part.membership

    if binary is not None:
        g, weights = generate_igraph(adjacency, directed=directed)
    modularity = g.modularity(part, weights=weights)
    return part, modularity 


def run_motif_counting_algorithm(input_filepath, output_filepath, cwd):
    program_filepath = '/mnt/e/dhh-soltesz-lab/snap-higher-order/examples/motifs'
    os.chdir(program_filepath)
    os.system('./motifs -i:%s -m:3 -d:N -o:%s' % (input_filepath, output_filepath))
    os.chdir(cwd)
    
    
def read_motif_counts(input_filepath):
    f = open(input_filepath, 'r')
    lcount = 0
    count_dict = {}
    for line in f.readlines():
        if lcount == 0:
            lcount += 1
            continue
        line = line.strip('\n').split('\t')
        mid, mcount = int(line[0]), int(line[-1])
        count_dict[mid] = mcount
    return count_dict


def run_higher_order_analysis(input_filepath, output_filepath, cwd, motif):
    os.chdir('/mnt/e/dhh-soltesz-lab/snap-higher-order/examples/motifcluster')
    os.system('./motifclustermain -i:%s -m:%s -o:%s' % (input_filepath, motif, output_filepath))
    os.chdir(cwd)
    
def plot_higher_order(input_filepath, spatial_coords, view, idxs=None):
    import matplotlib.pyplot as plt
    
    nids1, nids2 = [], []
    f = open(input_filepath, 'r')
    line = f.readline().strip('\n').split('\t')
    for l in line: nids1.append(int(l))
    line = f.readline().strip('\n').split('\t')
    for l in line: nids2.append(int(l))
        
    if idxs is not None:
        idxs = np.asarray(idxs, dtype='uint32')
        nids1 = idxs[nids1]
        nids2 = idxs[nids2]
    
    fig = plt.figure(figsize=(18,8))
    ax = plt.axes(projection='3d')
    ax.scatter(*spatial_coords[nids1,:].T, color='r')
    ax.scatter(*spatial_coords[nids2,:].T, color='b')
    ax.scatter(*spatial_coords.T, alpha=0.1, color='k')
    ax.view_init(*view)
    
    
def filter_triplets(triplets, idxsA, idxsB, idxsC=None):
    filtered_triplets = []
    for (x,y,z) in triplets:
        if x not in idxsA: continue
        if idxsC is None:
            if y in idxsB and z in idxsB:
                if not ((x,y,z) in filtered_triplets or (x,z,y) in filtered_triplets):
                    filtered_triplets.append([x,y,z])
        else:
            if (y in idxsB and z in idxsC) or (y in idxsC and z in idxsB):
                if not( (x,y,z) in filtered_triplets or (x,z,y) in filtered_triplets):
                       filtered_triplets.append([x,y,z])
                       
    return np.asarray(filtered_triplets, dtype='uint32')

def get_contagious_edges(triplets):
    motif_edges = []    
    for (x,y,z) in triplets:
        if [x,y] not in motif_edges:
            motif_edges.append([x,y])
        if [x,z] not in motif_edges: 
            motif_edges.append([x,z])
    return motif_edges

def get_bifan_edges(triplets):
    
    bifan_edges = []    
    valid_tails = []
    for _,y,z in triplets:
        if [y,z] not in valid_tails and [z,y] not in valid_tails: valid_tails.append([y,z])
    valid_tails = np.asarray(valid_tails, dtype='uint32')
    
    for (i,(x,y,z)) in enumerate(triplets):
        locs = np.where((valid_tails == (y,z))|(valid_tails==(z,y)))[0]
        for (x2,y2,z2) in triplets[locs]:
            if x == x2: continue
            if (y == y2 and z == z2) or (y == z2 and z == y2):
                if [x,y] not in bifan_edges:
                    bifan_edges.append([x,y])
                if [x,y2] not in bifan_edges:
                    bifan_edges.append([x,y2])
                if [x2,y] not in bifan_edges:
                    bifan_edges.append([x2,y])
                if [x2,y2] not in bifan_edges:
                    bifan_edges.append([x2,y2])
    return bifan_edges

    
    
def extract_motifs(J, motif='send'):
    triplets = []   
    srcs, dsts = J.nonzero()
    if motif =='send':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            locs = np.where(srcs == src)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if dst == dst2: continue
                if J[dst,dst2] or J[dst2,dst] or J[dst,src] or J[dst2,src]: continue
                triplets.append((src,dst,dst2))
            if i % 5000 == 0:
                print(i,len(srcs))
            i += 1
            
    ## do 'out'
    elif motif == 'receive':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            locs = np.where(dsts == dst)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if src == src2: continue
                if J[dst,src] or J[dst, src2] or J[src,src2] or J[src2,src]: continue
                triplets.append((dst, src, src2))
            if i % 5000 == 0:
                print(i, len(srcs))
            i += 1
    elif motif == 'recurrent':
        i = 0
        for (src, dst) in list(zip(srcs, dsts)):
            if J[dst, src]: continue

            locs = np.where(srcs == dst)[0]
            tsrcs, tdsts = srcs[locs], dsts[locs]
            for (src2, dst2) in list(zip(tsrcs, tdsts)):
                if src == src2: continue
                if J[dst2, src2]: continue
                locs2 = np.where(srcs == dst2)[0]
                ttsrcs, ttdsts = srcs[locs2], dsts[locs2]
                for (src3, dst3) in list(zip(ttsrcs, ttdsts)):
                    if dst3 != src: continue
                    if J[dst3,src3]: continue
                    triplets.append((src,dst,dst2))
            if i % 5000 == 0: 
                print(i, len(srcs))
            i += 1
                    
    else:
        print('motif argument not recognized')
    return triplets


def plot_motif_statistics(baseline, presz):
    import matplotlib.pyplot as plt
    
    motif_jump = presz - baseline
    bins = np.linspace(0, 10000, 20)
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(np.sort(baseline)[:], color='k')
    ax[0].plot(np.sort(presz)[:], color='r')
    ax[0].set_yscale('log')
    ax[1].hist([baseline, presz], color=['k', 'r'], bins=bins, rwidth=0.65)
    ax[1].set_yscale('log')
    plt.show()

    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(np.sort(motif_jump))
    ax[1].hist(motif_jump, color='k', rwidth=0.65)
    ax[1].set_yscale('log')
    plt.show()

    sizes = []
    colors = []
    for jump in motif_jump:
        ajump = abs(jump)
        if ajump < 10: sizes.append(0.5)
        elif ajump >= 10 and ajump < 100: sizes.append(1.0)
        elif ajump >= 100 and ajump < 1000: sizes.append(5.0)
        else: sizes.append(20.0)
        if jump > 0: colors.append('r')
        else: colors.append('k')

    plt.figure(figsize=(12,8))
    plt.scatter(baseline, presz, color=colors, s=sizes, alpha=1.0)
    plt.plot([i for i in range(15000)], [i for i in range(15000)], color='k', linestyle='--')
    plt.xscale('log'); plt.yscale('log')
    plt.title(np.corrcoef(baseline, presz)[0][1] ** 2)
    plt.show()
    

def nid_motif_participation(triplets,N, loc='full'):
    participation = [0 for _ in range(N)]
    for (x,y,z) in triplets:
        if loc == 'head':
            participation[x] += 1
        elif loc == 'tail':
            participation[y] += 1
            participation[z] += 1
        else:
            participation[x] += 1
            participation[y] += 1
            participation[z] += 1
            
    return np.asarray(participation, dtype='uint32')


