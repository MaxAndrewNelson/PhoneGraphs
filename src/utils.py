import numpy as np
import math
import random
import sys
import jenkspy
import json
from kneed import KneeLocator
from itertools import chain
from collections import defaultdict
from scipy.stats import entropy
# from scipy import io
from scipy.linalg import fractional_matrix_power
from scipy.spatial import distance
from matplotlib import pyplot as plt
#from sknetwork.clustering import modularity

MAP_1 = {
    'B': 'b',
    'D': 'd',
    'F': 'f',
    'G': 'g',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'P': 'p',
    'R': 'ɹ',
    'S': 's',
    'T': 't',
    'V': 'v',
    'W': 'w',
    'Y': 'j',
    'Z': 'z',
    'AA': 'ɑ',
    'AE': 'æ',
    'AH': 'ʌ',
    'AO': 'ɔ',
    'AW': 'aʊ',
    'AY': 'aɪ',
    'EH': 'ɛ',
    'ER': 'ɝ',
    'EY': 'eɪ',
    'IH': 'ɪ',
    'IY': 'i',
    'OW': 'o',
    'OY': 'ɔɪ',
    'UH': 'ʊ',
    'UW': 'u',
    'CH': 'tʃ',
    'DH': 'ð',
    'HH': 'h',
    'JH': 'dʒ',
    'NG': 'ŋ',
    'SH': 'ʃ',
    'TH': 'θ',
    'ZH': 'ʒ'
}


def make_symmetric_KLD(contexts):
    KLD_graph = make_KLD_matrix(contexts)

    symmetric_KLD_graph = np.zeros(KLD_graph.shape)
    for row in range(KLD_graph.shape[0]):
        for col in range(KLD_graph.shape[1]):
            symmetric_KLD_graph[col, row] += KLD_graph[col, row] + KLD_graph[row, col]
    graph = (np.exp(-symmetric_KLD_graph))

    return graph

# def plot_mat(M, d):
#     phones = [d[i] for i in range(len(list(d.keys())))]
#     fig, ax = plt.subplots()
#     im = ax.imshow(M, cmap='Purples', interpolation='nearest')
#     fig.colorbar(im)
#     ax.set_xticks(np.arange(len(phones)))
#     ax.set_yticks(np.arange(len(phones)))
#     ax.set_xticklabels(phones)
#     ax.set_yticklabels(phones) 

#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             f = '{:.3f}'.format(M[i,j])
#             if f != '0.000':
#                 text = ax.text(j, i, f,
#                        ha="center", va="center", color="black", size=8)

#     plt.show()

def get_corpus_data(open_file, window=1):
    raw_data = []
    for line in open_file:
        line = line.rstrip().split('\t')[0]
        line = window*['<s>'] + line.split(' ') + window*['<e>']
        line = [x for x in line if x != '']
        raw_data.append(line)
    return raw_data

# def build_context_matrix(corpus, window, separate_lens = False):
#     contexts = []
#     for line in corpus:
#         for symbol in line:
#             if symbol not in contexts:
#                 contexts.append(symbol)


#     phones = [x for x in contexts if '<' not in x]
#     phone2ix = {p:i for i,p in enumerate(phones)}
#     contexts2ix = {c:i for i,c in enumerate(contexts)}

#     ''' cooccurrences is (phones, contextual symbols, environment),
#     environment is at least Left, Right, but if separate lens will be
#     left 1, left 2, ... left window, right 1, right 2 , ... '''
#     if separate_lens:
#         cooccurrences = np.zeros((len(phones), len(contexts), 2*window))
#     else:
#         cooccurrences = np.zeros((len(phones), len(contexts), 2))

#     for line in corpus:
#         for i in range(window, len(line)-window): #don't loop over padding symbols
#             curr_symbol = line[i]
#             for w in range(window):
#                 left_context = line[i-(w+1)]
#                 right_context = line[i+w+1]
#                 if separate_lens:
#                     ### left context, envs 0 through window ###
#                     cooccurrences[phone2ix[curr_symbol], contexts2ix[left_context], w] += 1
#                     ### right context, envs window through -1 ###
#                     cooccurrences[phone2ix[curr_symbol], contexts2ix[right_context], window+w] += 1
#                 else:
#                     ### left context, env 0 ###
#                     cooccurrences[phone2ix[curr_symbol], contexts2ix[left_context], 0] += 1
#                     ### right context, env 1 ###
#                     cooccurrences[phone2ix[curr_symbol], contexts2ix[right_context], 1] += 1

#     ### remove the syllable boundary from the phoneme set ###
    
#     try: #remove boundaries if data is syllabified
#         boundary_pos = phone2ix['.']
#         cooccurrences = np.delete(cooccurrences, boundary_pos, axis=0)

#         for phone in phone2ix:
#             if phone2ix[phone] > boundary_pos:
#                 phone2ix[phone] = phone2ix[phone]-1    
#     except: 
#         pass

    # return cooccurrences, phone2ix, contexts2ix

def pmi(df, positive=True):
    df = df+1
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

# def similarity_matrix(cooc):
#     #Fold matrix along paths of length 2
#     sim_mat = np.zeros((cooc.shape[0], cooc.shape[0]))
#     for p1 in range(sim_mat.shape[0]):
#         p1_envs = cooc[p1,:]
#         for p2 in range(sim_mat.shape[1]):
#             p2_envs = cooc[p2,:]
#             for pos in range(p1_envs.size):
#                 if (p1_envs[pos] != 0) and (p2_envs[pos] != 0):
#                     sim_mat[p1,p2] += min(p1_envs[pos], p2_envs[pos])
#               # sim_mat[p1,p2] += p1_envs[pos] + p2_envs[pos]
#     np.fill_diagonal(sim_mat, 0)
#     return(sim_mat)

def make_KLD_matrix(context_matrix):
    context_matrix += 0.0001 #very basic smoothing, TODO: something more principled
    KLDs = np.zeros((context_matrix.shape[0], context_matrix.shape[0]))
    for row in range(KLDs.shape[0]):
        for col in range(KLDs.shape[1]):
            KLDs[row][col] = entropy(context_matrix[row,:], context_matrix[col,:])
    return(KLDs)

def undirected_edge_norm(mat):
    denoms = np.zeros(mat.shape)
    for row in range(mat.shape[0]):
        sum1 = np.sum(mat[row])
        for col in range(mat.shape[1]):
            sum2 = np.sum(mat[col])
            denoms[row, col] = sum1 + sum2
    return mat/denoms

def one_mode_project(A):
    '''
    input biadjacency A matrix m x n
    output adjacency B matrix m x m

    one-mode bipartite projection as described in Banjeree et al
    changed so that it sums weights rather than being binary 
    '''
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            for k in range(A.shape[1]):
                if A[i][k] != 0 and A[j][k] != 0: 
                    B[i][j] += (A[i][k] + A[j][k])
    return(B)

def make_bt_graph(mat):
    denoms = np.expand_dims(np.sum(mat, axis=1), axis=1)
    normed = mat/denoms

    distances = np.zeros((mat.shape[0], mat.shape[0]))
    for row in range(mat.shape[0]):
        for col in range(mat.shape[0]):
            root = np.sqrt(normed[row] * normed[col])
            distances[row, col] = np.sum(root)

    return(distances)

def norm_adj(G):
    """Computes the normalized adjacency matrix of a given graph"""

    n = G.shape[0]
    D = np.zeros((1,n))
    for i in range(n):
        D[0,i] = math.sqrt(G[i,:].sum())

    temp = np.dot(D.T, np.ones((1,n)))
    horizontal = G / temp
    normalized_adjacency_matrix = horizontal / (temp.T)
    # gc.collect()

    return normalized_adjacency_matrix

def decomp(A):
    evalues, evectors = np.linalg.eig(A)

    ## need to sort if using linalg.eig on non-symmetric ###
    evalue_ixs = evalues.argsort()
    #eigenvalues, smallest to largest
    evalues = evalues[evalue_ixs]
    #eigenvectors, ordered with paired evalues
    evectors = evectors[:, evalue_ixs]

    return(evalues, evectors)

def reflect_diagonal(A):
    ''' takes in triangular matrix, makes it a square'''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                if A[i, j] == 0:
                    A[i, j] = A[j, i]
    return(A)

def adj_to_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D-A

def norm_laplacian(A):
    "take adjacency, return normalized laplacian s.t. L = D^(-.5)L D^(-.5)"
    D = np.diag(np.sum(graph, axis=1))
    L = D - A
    D_neg_half = fractional_matrix_power(D, -.5)
    
    return np.matmul(np.matmul(D_neg_half, L), D_neg_half)

def modularity(A, c):
    m = np.sum(A)/2

    Sigma = 0
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            sum_term1 = A[row,col]
            sum_term2 = (np.sum(A[:,row]/2) + np.sum(A[col, :]/2))
            sum_term2 /= (2*m)
            kronecker = float(c[row] == c[col])
            Sigma += (sum_term1 - sum_term2) * kronecker

    return (1/(2*m)) * Sigma

def conductance(A, c):
    '''
    conductance of a cut c (bit array) given weighted adjacency matrix A
    '''
    ixs = np.where(c==1)[0]
    temp = A[ixs,:] #cut out rows 
    subgraph = temp[:,ixs] #cut out columns
    cutsize = temp.sum() - subgraph.sum()
    denominator = min(temp.sum(),A.sum()-temp.sum())
    conductance = cutsize / denominator

    return conductance

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def classify(point, breaks):
    for brk in range(len(breaks)-1):
        if point == breaks[brk] and brk==0:
            return(int(brk))
        elif point == breaks[brk]:
            return(int(brk-1))
        elif point > breaks[brk] and point < breaks[brk+1]:
            return(int(brk))
        elif point == breaks[brk+1] and brk+2 == len(breaks):
            return(int(brk))
    print('Warning: point not assigned')
    print(point, breaks)
def clustering_coef(A, k):
    '''Eq. 4 Kalna and Higham, k is position of node we are computing the coef for'''
    w3 = np.linalg.matrix_power(A, 3)
    numerator = w3[k,k]
    eTwk = np.sum(A[k])**2
    wk22 = np.inner(A[k], A[k])

    return numerator / (eTwk - wk22)

def average_coef(A):
    coefs = []
    for i in range(A.shape[0]):
        coefs.append(clustering_coef(A, i))
    return np.mean(coefs)

def get_eigengap(vals):
    num_zeros = 0
    gaps = []
    for e in vals:
        print(e, math.isclose(e, 0, abs_tol=1e-3))
        if math.isclose(e, 0, abs_tol=1e-3):
            num_zeros += 1

def build_dot_graph(C):
    normed = np.zeros(C.shape)
    for row in range(C.shape[0]):
        denom = np.linalg.norm(C[row])
        normed[row] = C[row]/denom
    
    dot_graph = np.zeros((C.shape[0], C.shape[0]))
    for row in range(dot_graph.shape[0]):
        for col in range(dot_graph.shape[1]):
            dot_graph[row, col] = np.dot(normed[row], normed[col])

    return(dot_graph)

# class BadCluster(Exception): 
#     pass

class arguments():
    def __init__(self, js, k):
        '''
        has an attribute for every key in js[k]
        '''

        data = open(js, 'r').read()
        as_dict = json.loads(data)[k]
        
        for key in as_dict:
            setattr(self, key, as_dict[key])

def plot_graph(A):
    import networkx as nx

    G = nx.from_numpy_matrix(A)

    pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    plt.show()

def plot_mat(M, d, round=False):

    phones = [d[i] for i in range(len(list(d.keys())))]

    ### for context graph clean up ###
    for k,p in enumerate(phones):
        if p[2:] in MAP_1.keys():
            phones[k] = p[:2] + MAP_1[p[2:]]


    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(M, cmap='Purples', interpolation='nearest')
    fig.colorbar(im)
    ax.set_xticks(np.arange(len(phones)))
    ax.set_yticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, size=8, rotation=90)
    ax.set_yticklabels(phones, size=8) 

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            f = '{:.2f}'.format(M[i,j])
            if round:
                if f != '0.000':
                    text = ax.text(j, i, f,
                           ha="center", va="center", color="black", size=8)
            else:
                text = ax.text(j, i, f,
                        ha="center", va="center", color="black", size=8)

    plt.show()

def remove_empty_columns(M, c2ix):
    ix2c = {i:c for c,i in c2ix.items()}

    #environments (columns) in which no phonemes occur
    bad_cols = np.where(np.sum(M, axis=0) == 0)[0] #these are environments in which NOTHING occurs
    
    fixed_c2ix = {}

    new_ix = 0
    for old_ix in range(len(ix2c.keys())):
        if old_ix not in bad_cols:
            fixed_c2ix[ix2c[old_ix]] = new_ix
            new_ix += 1

    good_cols = np.where(np.sum(M, axis=0) != 0)[0]
    new_M = M[:, good_cols]


    return(new_M, fixed_c2ix)

def get_corpus_data(open_file, window=1):
    raw_data = []
    for line in open_file:
        line = line.rstrip()
        line = window*['<s>'] + line.split(' ') + window*['<e>']
        raw_data.append(line)
    return raw_data

def build_context_matrix(corpus, window, separate_lens = True):
    base_contexts = []
    for line in corpus:
        for symbol in line:
            if symbol not in base_contexts:
                base_contexts.append(symbol)


    phones = [x for x in base_contexts if '<' not in x]
    phone2ix = {p:i for i,p in enumerate(phones)}

    contexts = []
    for p in base_contexts:
        if separate_lens:
            for direction in ('L','R'):
                for i in range(window):
                    contexts.append(str(i+1) + direction + p)
        else:
            for direction in ('L', 'R'):
                contexts.append(direction + p)

    contexts2ix = {c:i for i,c in enumerate(contexts)}
    
    ''' cooccurrences is (phones, contextual symbols, environment),
    environment is at least Left, Right, but if separate lens will be
    left 1, left 2, ... left window, right 1, right 2 , ... '''
    
    cooccurrence = np.zeros((len(phones), len(contexts)))

    # if separate_lens:
    #     cooccurrences = np.zeros((len(phones), len(contexts), 2*window))
    # else:
    #     cooccurrences = np.zeros((len(phones), len(contexts), 2))

    for line in corpus:
        for i in range(window, len(line)-window): #don't loop over padding symbols
            curr_symbol = line[i]
            for w in range(window):
                if separate_lens:
                    left_context = str(w+1) +'L' + line[i-(w+1)]
                    right_context = str(w+1) + 'R' + line[i+w+1]

                    cooccurrence[phone2ix[curr_symbol], contexts2ix[left_context]] += 1
                    cooccurrence[phone2ix[curr_symbol], contexts2ix[right_context]] += 1

                    ### left context, envs 0 through window ###
                    # cooccurrences[phone2ix[curr_symbol], contexts2ix[left_context], w] += 1
                    ### right context, envs window through -1 ###
                    # cooccurrences[phone2ix[curr_symbol], contexts2ix[right_context], window+w] += 1
                else:
                    left_context = 'L' + line[i-(w+1)]
                    right_context = 'R' + line[i+w+1]
                    ### left context, env 0 ###
                    cooccurrence[phone2ix[curr_symbol], contexts2ix[left_context]] += 1
                    cooccurrence[phone2ix[curr_symbol], contexts2ix[right_context]] += 1

    ### remove the syllable boundary from the phoneme set ###
    try: #remove boundaries if data is syllabified
        boundary_pos = phone2ix['.']
        cooccurrence = np.delete(cooccurrence, boundary_pos, axis=0)

        for phone in phone2ix:
            if phone2ix[phone] > boundary_pos:
                phone2ix[phone] = phone2ix[phone]-1    
    except: 
        pass

    ##reshape the context matrix if its 3d (if separate lens = True)
    # cooccurrences = cooccurrences.reshape(len(phones),-1)

    return cooccurrence, phone2ix, contexts2ix

def pmi(df, positive=True):
    # df = df+1
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

# def similarity_matrix(cooc):
#     #Fold matrix along paths of length 2
#     sim_mat = np.zeros((cooc.shape[0], cooc.shape[0]))
#     for p1 in range(sim_mat.shape[0]):
#         p1_envs = cooc[p1,:]
#         for p2 in range(sim_mat.shape[1]):
#             p2_envs = cooc[p2,:]
#             for pos in range(p1_envs.size):
#                 if (p1_envs[pos] != 0) and (p2_envs[pos] != 0):
#                     sim_mat[p1,p2] += min(p1_envs[pos], p2_envs[pos])
#               # sim_mat[p1,p2] += p1_envs[pos] + p2_envs[pos]
#     np.fill_diagonal(sim_mat, 0)
#     return(sim_mat)

def make_KLD_matrix(context_matrix):
    context_matrix += 0.0001 #very basic smoothing, TODO: something more principled
    KLDs = np.zeros((context_matrix.shape[0], context_matrix.shape[0]))
    for row in range(KLDs.shape[0]):
        for col in range(KLDs.shape[1]):
            KLDs[row][col] = entropy(context_matrix[row,:], context_matrix[col,:])
    return(KLDs)

def undirected_edge_norm(mat):
    denoms = np.zeros(mat.shape)
    for row in range(mat.shape[0]):
        sum1 = np.sum(mat[row])
        for col in range(mat.shape[1]):
            sum2 = np.sum(mat[col])
            denoms[row, col] = sum1 + sum2
    return mat/denoms

def one_mode_project(A):
    '''
    input biadjacency A matrix m x n
    output adjacency B matrix m x m

    one-mode bipartite projection as described in Banjeree et al
    changed so that it sums weights rather than being binary 
    '''
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(i, A.shape[0]):
            for k in range(A.shape[1]):
                if A[i][k] != 0 and A[j][k] != 0: 
                    B[i][j] += (A[i][k] + A[j][k])
    return(B)

def make_bt_graph(mat):
    denoms = np.expand_dims(np.sum(mat, axis=1), axis=1)
    normed = mat/denoms

    distances = np.zeros((mat.shape[0], mat.shape[0]))
    for row in range(mat.shape[0]):
        for col in range(mat.shape[0]):
            root = np.sqrt(normed[row] * normed[col])
            distances[row, col] = np.sum(root)

    return(distances)

def norm_adj(G):
    """Computes the normalized adjacency matrix of a given graph"""

    n = G.shape[0]
    D = np.zeros((1,n))
    for i in range(n):
        D[0,i] = math.sqrt(G[i,:].sum())

    temp = np.dot(D.T, np.ones((1,n)))
    horizontal = G / temp
    normalized_adjacency_matrix = horizontal / (temp.T)
    # gc.collect()

    return normalized_adjacency_matrix

def decomp(A):
    evalues, evectors = np.linalg.eig(A)

    ## need to sort if using linalg.eig on non-symmetric ###
    evalue_ixs = evalues.argsort()
    #eigenvalues, smallest to largest
    evalues = evalues[evalue_ixs]
    #eigenvectors, ordered with paired evalues
    evectors = evectors[:, evalue_ixs]

    return(evalues, evectors)

def reflect_diagonal(A):
    ''' takes in triangular matrix, makes it a square'''
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                if A[i, j] == 0:
                    A[i, j] = A[j, i]
    return(A)

def adj_to_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    return D-A

def norm_laplacian(A):
    "take adjacency, return normalized laplacian s.t. L = D^(-.5)L D^(-.5)"
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    D_neg_half = fractional_matrix_power(D, -.5)
    
    return np.matmul(np.matmul(D_neg_half, L), D_neg_half)

def modularity(A, c):
    m = np.sum(A)/2

    Sigma = 0
    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            sum_term1 = A[row,col]
            sum_term2 = (np.sum(A[:,row]/2) + np.sum(A[col, :]/2))
            sum_term2 /= (2*m)
            kronecker = float(c[row] == c[col])
            Sigma += (sum_term1 - sum_term2) * kronecker

    return (1/(2*m)) * Sigma

def conductance(A, c):
    '''
    conductance of a cut c (bit array) given weighted adjacency matrix A
    '''
    ixs = np.where(c==1)[0]
    temp = A[ixs,:] #cut out rows 
    subgraph = temp[:,ixs] #cut out columns
    cutsize = temp.sum() - subgraph.sum()
    denominator = min(temp.sum(),A.sum()-temp.sum())
    conductance = cutsize / denominator

    return conductance

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenkspy.jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def classify(point, breaks):
    for brk in range(len(breaks)-1):
        if point == breaks[brk] and brk==0:
            return(int(brk))
        elif point == breaks[brk]:
            return(int(brk-1))
        elif point > breaks[brk] and point < breaks[brk+1]:
            return(int(brk))
        elif point == breaks[brk+1] and brk+2 == len(breaks):
            return(int(brk))
    print('Warning: point not assigned')
    print(point, breaks)
def clustering_coef(A, k):
    '''Eq. 4 Kalna and Higham, k is position of node we are computing the coef for'''
    w3 = np.linalg.matrix_power(A, 3)
    numerator = w3[k,k]
    eTwk = np.sum(A[k])**2
    wk22 = np.inner(A[k], A[k])

    return numerator / (eTwk - wk22)

def average_coef(A):
    coefs = []
    for i in range(A.shape[0]):
        coefs.append(clustering_coef(A, i))
    return np.mean(coefs)

def get_eigengap(vals):
    num_zeros = 0
    gaps = []
    for e in vals:
        print(e, math.isclose(e, 0, abs_tol=1e-3))
        if math.isclose(e, 0, abs_tol=1e-3):
            num_zeros += 1

def build_dot_graph(C):
    normed = np.zeros(C.shape)
    for row in range(C.shape[0]):
        denom = np.linalg.norm(C[row])
        normed[row] = C[row]/denom
    
    dot_graph = np.zeros((C.shape[0], C.shape[0]))
    for row in range(dot_graph.shape[0]):
        for col in range(dot_graph.shape[1]):
            dot_graph[row, col] = np.dot(normed[row], normed[col])

    return(dot_graph)

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def partition(laplacian, adjacency, ix2id, args, NOT_LAPLACE = False):#max_cuts=2, use_jenks=True, plot_cuts=False,):
    '''
    Binary or n-ary cut on the Fiedler vector, using Jenks breaks with kneedle 
    algorithm of GMM with optimal BIC for cut
    '''
    max_cuts = args.max_cuts
    use_jenks = args.use_jenks
    plot_cuts = args.plot_cuts


    evals, evecs = decomp(laplacian)

    if NOT_LAPLACE:
        curr_reps = evecs[:,-1]
    else:
        curr_reps = evecs[:,1]

    all_assignments = []
    agg_mods = []
    agg_conds = []

    if args.sign_cut:
        assignments = (curr_reps > 0) * 1

    elif args.use_jenks: 

        if args.max_cuts > 2:
            Gs = [0.]
            ns = [1]
            for i in range(2,min(curr_reps.shape[0], args.max_cuts+1)):
                gvf = goodness_of_variance_fit(curr_reps, i)
                Gs.append(gvf)
                ns.append(i)
            try:
                kneedle = KneeLocator(ns, Gs, S=1.0, curve="concave", direction="increasing")        
                n_clusters = kneedle.elbow
            except:
                n_clusters = 2

        else:
            n_clusters = 2

        if curr_reps.shape[0] > 2:
            breaks = jenkspy.jenks_breaks(curr_reps, n_clusters)
            assignments = [classify(x, breaks) for x in curr_reps]
        else:
            assignments = np.array([0,1])

    else:
        from sklearn import mixture
        curr_reps = curr_reps.reshape(-1,1)

        Gs = []
        ns = []
        #as many clusters as max cuts, unless that is more clusters than phones in the class
        for i in range(1,min(args.max_cuts,curr_reps.shape[0])):
            g = mixture.GaussianMixture(n_components=(i+1))
            g.fit(curr_reps)
            Gs.append(g.bic(curr_reps))
            ns.append(i+1)

        n_clusters = ns[np.argmin(Gs)]
        g = mixture.GaussianMixture(n_components=n_clusters)
        #g = mixture.GaussianMixture(n_components=2)
        model = g.fit(curr_reps)
        assignments = model.predict(curr_reps)

    # print(assignments)
    # print("{} clusters".format(n_clusters))

    # curr_conds = []
    # curr_mods = []
    # avg_clustering_coeffs = []
    # curr_mean_pos = []
    for k in range(len(list(set(assignments)))): #for every cluster
        membership = np.zeros((len(ix2id),))
        for a in range(len(assignments)):
            if assignments[a] == k:
                membership[a] += 1
        all_assignments.append(membership)

    #     mean_location = np.sum(membership*curr_reps.reshape(-1)) / np.sum(membership)
    #     curr_mean_pos.append(mean_location)

    #     #conductance of cuts
    #     curr_conds.append(conductance(adjacency, membership.astype(int)))
    #     #modularity of cuts
    #     curr_mods.append(modularity(adjacency, membership.astype(int)))

    # agg_mods.append(curr_mods)
    # agg_conds.append(curr_conds)
    # curr_conds = ["{:.3f}".format(x) for x in curr_conds]
    # curr_mods = ["{:.3f}".format(x) for x in curr_mods]
    # avg_clustering_coeffs = ["{:.3f}".format(x) for x in avg_clustering_coeffs]

    if plot_cuts:
        fig, ax = plt.subplots(2,1)
        colors = {0:"red", 1:"blue", 2:"green", 3:"purple", 4:"yellow", 5:"black", 6:"orange", 7:"black", 8:"grey"}
        ax[0].set_title("Embedding dim {}".format(e))
        for c in range(curr_reps.shape[0]):
            ax[0].scatter(curr_reps[c], 1, color=colors[assignments[c]])
            ax[0].annotate(ix2id[c], xy=(evecs[c,e]-0.003, 1.002), fontsize=6)
        ax[0].axes.yaxis.set_ticks([])

        ### annotate conductances ###
        for c in range(len(curr_mean_pos)):
            ax[0].annotate(curr_conds[c], xy=(curr_mean_pos[c]-0.025, .995), color='red')
            #ax[0].annotate(avg_clustering_coeffs[c], xy=(curr_mean_pos[c]-0.025, .99))
            ax[0].annotate(curr_mods[c], xy=(curr_mean_pos[c]-0.025, .99), color='green')
        ax[1].plot(ns, Gs)
        ax[1].set_xlabel('Number of clusters')
        if use_jenks:
            ax[1].set_ylabel('GVF')
        else:
            ax[1].set_ylabel('BIC')
        ax[1].axvline(x=n_clusters, color='black', linestyle=':')

        plt.show()

    return(assignments, all_assignments)


  