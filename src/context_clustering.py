from utils import *
# import networkx as nx
# import pandas as pd
# import itertools

def create_subgraph(G, cut, ix2phone):
        ''' Makes a new graph G given cuts, a 1D array which is a binary cut vector
        cut vectors '''

        ### cut out the subgraph ###
        ixs = np.where(cut==1)[0]
        subgraph = G[ixs, :] #slice rows
        subgraph = subgraph[:, ixs] #slice cols

        ### update ix2phone to reflect new node positions ###
        new_phones = [ix2phone[ix] for ix in ixs] 
        i2p = {i:p for i,p in enumerate(new_phones)}

        return subgraph, i2p

# def calculate_mean_and_variance(X, n):
#     '''
#     CONNORS FN
#     Calculate the mean and variance of a cluster
#     '''
#     my_sum = 0
#     sumsq = 0

#     sorted_X = sorted(X)

#     median = sorted_X[len(sorted_X) // 2]
#     for item in sorted_X:
#         my_sum += item - median
#         sumsq += (item - median) * (item - median)
#     mean = my_sum / n + median

#     if n > 1:
#         variance = (sumsq - my_sum * my_sum / n) / (n - 1)
#     else:
#         variance = 0

#     return mean, variance

# def compute_bic(kmeans,X):
#     """
#     Computes the BIC metric for a given clusters

#     Parameters:
#     -----------------------------------------
#     kmeans:  List of clustering object from scikit learn

#     X     :  multidimension np array of data points

#     Returns:
#     -----------------------------------------
#     BIC value
#     """
#     # assign centers and labels
#     centers = [kmeans.cluster_centers_]
#     labels  = kmeans.labels_
#     #number of clusters
#     m = kmeans.n_clusters
#     # size of the clusters
#     n = np.bincount(labels)
#     #size of data set
#     N, d = X.shape

#     #compute variance for all clusters beforehand
#     cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
#              'euclidean')**2) for i in range(m)])

#     const_term = 0.5 * m * np.log(N) * (d+1)

#     BIC = np.sum([n[i] * np.log(n[i]) -
#                n[i] * np.log(N) -
#              ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
#              ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

#     return(BIC)

def cut_contexts(C, cut, ix2p):
    ixs = np.where(cut==1)[0]

    new_phones = [ix2p[ix] for ix in ixs]
    i2p = {i:p for i,p in enumerate(new_phones)}

    return C[ixs, :], i2p


# def make_symmetric_KLD(contexts):
#     KLD_graph = make_KLD_matrix(contexts)

#     symmetric_KLD_graph = np.zeros(KLD_graph.shape)
#     for row in range(KLD_graph.shape[0]):
#         for col in range(KLD_graph.shape[1]):
#             symmetric_KLD_graph[col, row] += KLD_graph[col, row] + KLD_graph[row, col]
#     graph = (np.exp(-symmetric_KLD_graph))

#     return graph

def contexts_to_phonemes(context_ixs, contexts, ix2id, args):
    PPMIt = args.PPMI_threshold
    ppmi = np.nan_to_num(pmi(contexts))

    above_threshold = (ppmi > PPMIt) * context_ixs
    responsible_contexts = np.sum(above_threshold, axis=1)
    context_ixs = np.where(responsible_contexts != 0)[0].tolist()
    phoneme_assignments = np.sum(above_threshold, axis=0)
    phone_ixs = np.where(phoneme_assignments != 0)[0].tolist()

    ### print this to look at the phones
    phones = [ix2phone[x] for x in phone_ixs]
    # cs = [ix2id[x] for x in context_ixs[0].tolist()]

    phoneme_membership = np.zeros((contexts.shape[1]))
    np.put(phoneme_membership, phone_ixs, 1)

    return phone_ixs, context_ixs

def recursive_partition(contexts, ix2id, id2ix, full_G, full_contexts, all_classes, args):
    

    ### Make the graph from the context cooccurrence matrix ###
    if args.graph_type == "KLD":
        # print("\tGraph: KLD")
        G = make_symmetric_KLD(contexts)
        np.fill_diagonal(baseG, 0)
    elif args.graph_type == 'one-mode':
        # print("\tGraph: one-mode")
        G = one_mode_project(contexts)
    else:
        # print("\tGraph: covariance")
        G = np.matmul(contexts, contexts.transpose())
    np.fill_diagonal(G, 0)

    ### Make the adjacency matrix ###
    degree = np.diag(np.sum(G, axis=1))


    if args.mod_matrix:
        if args.norm_lap:
            print("Bad parameters: normalized modularity not yet implemented \n\tStopping")
            sys.exit()
        else:
            print(np.sum(G))
            kk = np.outer(np.sum(G, axis=1), np.sum(G, axis=1))/(np.sum(G)*2)
            laplace = G - kk #not actually laplace, modularity, but keeping it for code compatability
    else:
        if args.norm_lap:
             laplace = np.identity(degree.shape[0]) - np.matmul(np.linalg.inv(degree), G)
        else:
             laplace = degree - G


    ### Partition the graph by the laplacian ###
    cut_indicator, partitions = partition(laplace, G, ix2id, args, NOT_LAPLACE = args.mod_matrix)

    # if len(ix2id.keys()) > 1:
    #     draw_graph(nx.from_numpy_matrix(G), ix2id, 0.1, cut=cut_indicator)

    ### Apply partitioning recursively ###
    for part in partitions:
        new_contexts, i2p = cut_contexts(contexts, part, ix2id)
        cuts = [id2ix[x] for x in list(i2p.values())] #index of the cut in the original graph
        cut_vector = np.zeros((len(id2ix),1))
        np.put(cut_vector, cuts, 1)
        #subgraph, subix2phone = create_subgraph(graph, part, ix2phone)
        if modularity(G, cut_vector) > args.mod_threshold:
            membership_vector, responsible_cs = contexts_to_phonemes(cut_vector, full_contexts, ix2id, args)
            all_classes.add(membership_vector, responsible_cs)
            if new_contexts.shape[0] > 1:
                recursive_partition(new_contexts, i2p, id2ix, full_G, full_contexts, all_classes, args)
        else:
            print('------------------')
            return True 

def spectral_clustering(contexts, context2ix, args):

    if args.graph_type == "KLD":
        G = make_symmetric_KLD(contexts)
    elif args.graph_type == 'one-mode':
        G = one_mode_project(contexts)
    else:
        G = np.matmul(contexts, contexts.transpose()) 

    degree = np.diag(np.sum(G, axis=1))
    if args.mod_matrix:
        if args.norm_lap:
            print("Bad parameters: normalized modularity not yet implemented \n\tStopping")
            sys.exit()
        else:
            kk = np.outer(np.sum(G, axis=1), np.sum(G, axis=1))/(np.sum(G)*2)
            laplace = G - kk #not actually laplace, modularity, but keeping it for code compatability
    else:
        if args.norm_lap:
            unnormed = degree - G
  
            laplace = np.matmul(np.linalg.inv(degree), unnormed)
        
            # plt.imshow(laplace)
            # plt.show()
            # sys.exit()
        else:
             laplace = degree - G

    from sklearn.cluster import SpectralClustering, KMeans
    
    if args.eigengap: #nclusters and n embeddings set by eigengap
        evals, evecs = decomp(laplace)

        min_embs = args.min_embs
        gaps = np.diff(evals[1:])
        ### using minimum gap position ###
        largest_gap = np.argmax(gaps[min_embs:]) + min_embs + 1
        ### no minimum gap position ###
        #largest_gap = np.argmax(gaps)+1 
        n = largest_gap
 

        embedded = evecs[:,1:n+1]
        best_model = KMeans(n_clusters=n).fit(embedded)
        #print("{} clusters".format(n))

    else: #KMeans using BIC
        bics = []
        ns = []
        models = []
        for n in range(5,G.shape[1],2):
            evals, evecs = decomp(laplace)
            embedded = evecs[:,1:n+1]

            model = KMeans(n_clusters=n).fit(embedded)
            bics.append(compute_bic(model, embedded))
            ns.append(n)
            models.append(model)

        best_ix = np.argmax(bics)
        best_n = ns[best_ix]
        best_model = models[best_ix]

        print("{} clusters".format(best_n))


    num_nodes = G.shape[0]
    as_cuts = [] #will hold binary vectors indicating the cuts for each community

    for group_ix in range(n):
        cut_vector = np.zeros((num_nodes,1))
        for k, assignment in enumerate(best_model.labels_):
            if assignment == group_ix:
                cut_vector[k] = 1
        membership_vector, responsible_cs = contexts_to_phonemes(cut_vector, contexts, context2ix, args)
        all_classes.add(membership_vector, responsible_cs)


# def context_louvain(contexts, context2id, params):
#     import community
#     import networkx as nx

#     G = make_symmetric_KLD(contexts)
#     np.fill_diagonal(G, 0)

#     nxG = nx.from_numpy_matrix(G)
#     partition = community.best_partition(nxG)

#     num_nodes = G.shape[0]
#     num_groups = len(set(list(partition.values())))
#     as_cuts = [] #will hold binary vectors indicating the cuts for each community
#     for group_ix in range(num_groups):
#         cut_vector = np.zeros((num_nodes,1))
#         for node_pos in partition.keys():
#             if partition[node_pos] == group_ix:
#                 cut_vector[node_pos] = 1
#         #curr_part is not the binary cut indicator vector
#         membership_vector, responsible_cs = contexts_to_phonemes(cut_vector, contexts, context2id, params)
#         all_classes.add(membership_vector, responsible_cs)


class memberships():
    def __init__(self):
        self.mems = []
        self.contexts = []
    def add(self, x, y):
        self.mems.append((x, y))


# def draw_graph(G, labels, threshold, cut=np.zeros([1,1])):
#     G = nx.relabel_nodes(G, labels)


#     elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > threshold]
#     esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= threshold]

#     if np.sum(cut) != 0:
#         colors = ["lightcoral" if x == 1 else "seagreen" for x in cut]
#     else:
#         colors = 'lightcoral'

#     pos = nx.spring_layout(G, k=1)  # positions for all nodes

#     # nodes
#     nx.draw_networkx_nodes(G, pos, node_size=350, node_color=colors)

#     # edges
#     nx.draw_networkx_edges(
#         G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="black", 
#     )
#     nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)

#     # labels
#     #nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")

#     plt.axis("off")
#     plt.show()

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

if __name__ == '__main__':
    # params = {
    #     "method":"recurse",
    #     "eigengap":True, #ignored if method is anything but spectral
    #     "window" : 2, #3 for polish, 2 for english 
    #     "min_embs" : 4,
    #     "unweighted" : False,
    #     "use_jenks" : True, #jenks with kneedle algorithm else GMM with BIC
    #     "oned_plots" : True,
    #     "PPMI_threshold" : 0.00,
    #     "separate_lens" : True, 
    #     "norm_lap" : True,
    #     "max_cuts" : 2, 
    #     "plot_cuts" : False, 
    #     "mod_threshold" : -100,
    #     "graph_type": "cov",
    #     "mod_matrix": False, #partition with modularity matrix instead of laplacian
    #     "sign_cut": True,
    #     "PMI": False
    #     }
    # output_file = open("./English_Onsets/RC_weird_test_1.txt", 'w', encoding='utf-8')

    file = open(sys.argv[1], "r")
    class_file = open(sys.argv[2], 'w')
    args = arguments(sys.argv[3], "class")


    # file = open('Polish_Onsets/Polish_onset_type_more_cropped.txt', 'r', encoding='UTF-8')
    raw_data = get_corpus_data(file, window=args.window)
    context_matrix, phone2ix, contexts2ix = build_context_matrix(raw_data, args.window, separate_lens=args.separate_lens)

    context_matrix, contexts2ix = remove_empty_columns(context_matrix, contexts2ix)
    ix2context = {k:v for v,k in contexts2ix.items()}

    ix2phone = {v:k for k,v in phone2ix.items()}
    context_matrix = context_matrix.reshape((context_matrix.shape[0], -1)).transpose()
    
    if args.PMI:
        context_matrix = pmi(context_matrix)

    ix2context = {v:k for k,v in contexts2ix.items()}


    if args.graph_type == "KLD":
        baseG = make_symmetric_KLD(context_matrix)
        np.fill_diagonal(baseG, 0)
    elif args.graph_type == 'one-mode':
        baseG = one_mode_project(context_matrix)
    else:
        baseG = np.matmul(context_matrix, context_matrix.transpose())
        degree = np.diag(np.sum(baseG, axis=1))
        baseG = np.matmul(np.linalg.inv(degree), baseG)
        np.fill_diagonal(baseG, 0)

    all_classes = memberships()


    print(f"\tMethod: {args.method}. Graph: {args.graph_type}")
    if args.method == "recurse":
        recursive_partition(context_matrix, ix2context, contexts2ix, baseG, context_matrix, all_classes, args)
    elif args.method == 'louvain':
        context_louvain(context_matrix, contexts2ix, args)
    elif args.method == 'spectral':
        spectral_clustering(context_matrix, contexts2ix, args)
    else:
        print("Error incorrect method. Please specify 'recurse', 'louvain', or 'spectral'")


    classes_to_contexts = {}
    for x in all_classes.mems:
        if x[0] != []:
            if frozenset([ix2phone[p] for p in x[0]]) not in classes_to_contexts:
                classes_to_contexts[frozenset([ix2phone[p] for p in x[0]])] = [ix2context[c] for c in x[1]]

    classes_as_phones = [list(s) for s in classes_to_contexts.keys()]

    ## add in all singleton classes
    # for x in phone2ix.keys():
    #     if [x] not in classes_to_contexts.keys():
    #         classes_to_contexts[[x]] = "singleton"

    # print('\nDONE CLUSTERING')
    # print('----------------------------')

    # for p in phone2ix:
    #     print(p)
    #     output_file.write(p + '\n')

    print('\t{} classes found'.format(len(classes_to_contexts.keys())))

    print(f"\tWriting classes to {sys.argv[2]}")

    for k,v in classes_to_contexts.items():
        class_file.write(' '.join(list(k)) + '\t' + ' '.join(v) + '\n')
        # print(list(k), v)

    
  