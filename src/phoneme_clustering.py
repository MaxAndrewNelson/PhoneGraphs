# import numpy as np
# import math
# import random
# import sys
# import jenkspy
# import itertools
# from kneed import KneeLocator
# from itertools import chain
# from collections import defaultdict
# from scipy.stats import entropy
# # from scipy import io
# from scipy.linalg import fractional_matrix_power
# from matplotlib import pyplot as plt
from utils import *
# import json


if __name__ == '__main__':
    '''
    Args: 

    input file

    output file

    config

    '''
    file = open(sys.argv[1], "r")
    class_file = open(sys.argv[2], 'w')
    args = arguments(sys.argv[3]["class"])

    raw_data = get_corpus_data(file, window=args.window)
    context_matrix, phone2ix, contexts2ix = build_context_matrix(raw_data, args.window, separate_lens=True)
    context_matrix, contexts2ix = remove_empty_columns(context_matrix, contexts2ix)
    ix2phone = {v:k for k,v in phone2ix.items()}
    context_matrix = context_matrix.reshape((context_matrix.shape[0], -1))



    if args.graph_type == "KLD":
        print("\tConstructing KLD graph")
        graph = make_symmetric_KLD(context_matrix)
        np.fill_diagonal(graph, 0)

    else:
        print("\tConstructing covariance graph")
        graph = np.matmul(context_matrix, context_matrix.transpose()) 
        np.fill_diagonal(graph, 0)

    degree = np.diag(np.sum(graph, axis=1))
    lap = np.identity(degree.shape[0]) - np.matmul(np.linalg.inv(degree), graph) #this is working better for parupa
    
    # plot_mat(lap, ix2phone)
    # normed = lap * -1 #normalized adjacency matrix
    # np.fill_diagonal(normed, 0)
    
    evals, evecs = decomp(lap)

    if args.eigengap: #nclusters and n embeddings set by eigengap
        gaps = np.diff(evals[1:])
        ### using minimum gap position ###
        largest_gap = np.argmax(gaps[args.min_embs:]) + args.min_embs + 1
        ### no minimum gap position ###
        #largest_gap = np.argmax(gaps)+1 
        n = largest_gap

        num_embs = n
        # embedded = evecs[:,1:n+1]
        # best_model = KMeans(n_clusters=n).fit(embedded)
        print("{} embedding dimensions with {} minimum".format(n, args.min_embs))
    else:
        num_embs = args.num_embs


    # plt.plot(evals)
    # plt.show()

    # sys.exit()

    phone2feats = {k:[] for k in phone2ix.keys()}

    agg_mods = []
    agg_conds = []

    all_assignments = []
    for e in range(1, num_embs): #how deep into embeddings to go?
        curr_reps = evecs[:,e].reshape(evecs.shape[0], 1)
        if args.use_jenks: 
            print("Clustering evec {}".format(e))
            Gs = [0.]
            ns = [1]
            for i in range(2,evecs.shape[0]):
                gvf = goodness_of_variance_fit(curr_reps, i)
                Gs.append(gvf)
                ns.append(i)

            kneedle = KneeLocator(ns, Gs, S=1.0, curve="concave", direction="increasing")        
            n_clusters = kneedle.elbow

            breaks = jenkspy.jenks_breaks(evecs[:,e], n_clusters)
            assignments = [classify(x, breaks) for x in evecs[:,e]]


        else:
            from sklearn import mixture
            print("Clustering evec {}".format(e))
            Gs = []
            ns = []
            for i in range(1,args.max_clusters):
                g = mixture.GaussianMixture(n_components=(i+1), 
                    covariance_type='full')
                g.fit(curr_reps)
                Gs.append(g.bic(curr_reps))
                ns.append(i+1)

            n_clusters = ns[np.argmin(Gs)]
            g = mixture.GaussianMixture(n_components=n_clusters, 
                covariance_type='full', 
                n_init = 25, 
                init_params = 'random', 
                tol = 5e-4)
            #g = mixture.GaussianMixture(n_components=2)
            model = g.fit(curr_reps)
            assignments = model.predict(curr_reps)

        # print(assignments)
        # print("{} clusters".format(n_clusters))

        curr_conds = []
        curr_mods = []
        avg_clustering_coeffs = []
        curr_mean_pos = []
        for k in range(len(list(set(assignments)))): #for every cluster
            membership = np.zeros((len(ix2phone),))
            for a in range(len(assignments)):
                if assignments[a] == k:
                    membership[a] += 1
            all_assignments.append(membership)
            mean_location = np.sum(membership*curr_reps.reshape(-1)) / np.sum(membership)
            curr_mean_pos.append(mean_location)

            #conductance of cuts
            # curr_conds.append(conductance(normed, membership.astype(int)))
            #modularity of cuts
            # curr_mods.append(modularity(normed, membership.astype(int)))

        agg_mods.append(curr_mods)
        agg_conds.append(curr_conds)
        curr_conds = ["{:.3f}".format(x) for x in curr_conds]
        curr_mods = ["{:.3f}".format(x) for x in curr_mods]
        avg_clustering_coeffs = ["{:.3f}".format(x) for x in avg_clustering_coeffs]

        if args.oned_plots:
            fig, ax = plt.subplots(2,1)
            colors = {0:"red", 1:"blue", 2:"green", 3:"purple", 4:"yellow", 5:"black", 6:"orange", 7:"black", 8:"grey"}
            ax[0].set_title("Embedding dim {}".format(e))
            for c in range(curr_reps.shape[0]):
                ax[0].scatter(curr_reps[c], 1, color=colors[assignments[c]])
                ax[0].annotate(ix2phone[c], xy=(evecs[c,e]-0.003, 1.002), fontsize=6)
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


    all_phones = set(phone2ix.keys())
    classes_as_phones = []
    for a in all_assignments:
        ixs = np.where(a == 1)[0]
        classes_as_phones.append(frozenset([ix2phone[x] for x in ixs]))
        #classes_as_phones.append(all_phones.difference(set([ix2phone[x] for x in ixs])))

    intersected_classes = list(set(classes_as_phones))

    ### checks for conjunctions up to r natural classes ###
    # r = 3
    # for k in range(2,r+1):
    #     set_pairs = itertools.combinations(classes_as_phones, k)
    #     for p in set_pairs:
    #         new_set = set.intersection(*p)
    #         if new_set not in intersected_classes:
    #             intersected_classes.append(new_set)

    # class_file = open('parupa.txt', 'w', encoding='utf-8')
    n_classes = len(intersected_classes)
    for cc in intersected_classes:
        # print(' '.join(list(cc)))
        class_file.write(' '.join(list(cc)) + '\n')

    print("learned {} classes".format(n_classes))









