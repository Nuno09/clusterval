"""
Function evaluate does hierarchical clustering and validation of it's results.
Validation can be done with external or internal indexes, or both.


"""


from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster, cophenet
import random
from clusterval.external import calculate_external
from clusterval.internal import calculate_internal, cvnn
from statistics import mean
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import os
import shutil

external_indexes = ['R', 'AR', 'FM', 'J', 'AW', 'VD', 'H', 'H\'', 'F', 'VI', 'MS']
internal_indexes = ['CVNN', 'XB*', 'S_Dbw', 'DB*', 'S', 'SD']
min_indexes = ['VD', 'VI', 'MS', 'CVNN', 'XB*', 'S_Dbw', 'DB*', 'SD']
indexes = ['R', 'AR', 'FM', 'J', 'AW', 'VD', 'H',
           'H\'', 'F', 'VI', 'MS', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']

map_indexes = {'R': 'Rand', 'AR': 'Ajusted Rand', 'FM': 'Fowlkes & Mallows', 'J': 'Jaccard', 'AW': 'Adjusted Wallace',
               'VD': 'Van Dongen', 'H': 'Huberts', 'H\'': 'Huberts Normalized', 'F': 'F-Measure',
               'VI': 'Variation of Information', 'MS': 'Minkowski Score', 'CVNN': 'CVNN', 'XB*':'XB*',
               'S_Dbw': 'S_Dbw', 'DB*': 'DB*', 'S':'S', 'SD':'SD', 'all':'all', 'external':'external', 'internal':'internal'}



def evaluate(data, min_k=2, max_k=8, link='ward',M=100, index='all'):

    """

    :param data: A dataset, for now only doing lists
    :param min_k: Minimum number of clusters. Default is 2
    :param max_k: Maximum number of clusters. Default is 8.
    :param method: Method to evaluate distance between data points, possible: wards, complete, single, average and
    centroid
    :param M: Number of bootstrap samples. Default is 100.
    :param index: Allows to choose which indexes results you want to see.
    :return: Table with validation indexes mean values for each k.
    """


    flag_external = 0
    flag_internal = 0

    if index == 'all':
        flag_external = 1
        flag_internal = 1

    elif index == 'external' or index in external_indexes:
        flag_external = 1

    elif index == 'internal' or index in internal_indexes:
        flag_internal = 1


    else:
        raise ValueError(
            "Provide index equal to one of: \n\'R\', \'AR\', \'FM\', \'J\', \'AW\', \'VD\', \'H\', \'H\\'\', \'F\',"
            " \'VI\', \'MS\', \'CVNN\', \'XB**\', \'S_Dbw\', \'DB*\', \'S\', \'SD\', or choose \'external\', \'internal\' or \'all\' ")

    print('\n* Linkage criteria is: ' + link + '\n')
    print('* Minimum number of clusters to test: ' + str(min_k) + '\n')
    print('* Maximum number of clusters to test: ' + str(max_k) + '\n')
    print('* Number of bootstrap samples generated: ' + str(M) + '\n')

    if (index in indexes) or (index == 'all'):
      print('\n* Validation Indices calculated: ' + str(index) + '\n\n')
    elif index == 'external':
        print('\n* Validation Indices calculated: ' + str(external_indexes) + '\n\n')

    elif index == 'internal':
        print('\n* Validation Indices calculated: ' + str(internal_indexes) + '\n\n')



    results = {k: {} for k in range(min_k, max_k + 1)}
    #copy of results dict is better, but was not working
    mean_results_external = {k: [] for k in range(min_k, max_k + 1)}
    mean_results_internal = {k: [] for k in range(min_k, max_k + 1)}
    trees = defaultdict(dict)


    #probably to many creations of the kind
    for k in range(min_k, max_k + 1):
        if (index in external_indexes) or (index in internal_indexes):
            results[k][index] = []
        elif index == 'external':
            for i in external_indexes:
                results[k][i] = []
        elif index == 'internal':
            for j in internal_indexes:
                results[k][j] = []

        else:
            for i in external_indexes:
                results[k][i] = []
            for j in internal_indexes:
                results[k][j] = []

    Z = linkage(data, link)


    for k in range(min_k, max_k + 1):


        #map tree into the dataset
        clusters = cluster_indices(fcluster(Z, t=k, criterion='maxclust'), [i for i in range(0, len(data))])
        trees[k] = clusters


        if flag_external == 1:
            #external validation step
            for i in range(M):
                sample = random.sample(data, int(3/4*len(data)))

                Z_sample = linkage(sample, link)

                clusters_sample = cluster_indices(fcluster(Z_sample, t=k, criterion='maxclust'), [i for i in range(0, len(sample))])
                external = calculate_external(clusters, clusters_sample)

                #in case there is only one metric to evaluate
                if index in external_indexes:
                    results[k][index].append(external[index])

                else:
                     for key, metric in external.items():
                        results[k][key].append(metric)

    if flag_internal == 1:
        distances = distance_dict(data)
        #internal validation step
        if index in internal_indexes:
            if index == 'CVNN':
                cvnn_index = cvnn(trees, distances)
                for k in cvnn_index.keys():
                    results[k][index].append(cvnn_index[k])
            else:
                for k, partition in trees.items():
                    internal = calculate_internal(distances, partition, c_max=trees[max_k], index=index)

                results[k][index].append(internal[index])
        else:
            cvnn_index = cvnn(trees, distances)
            for k in cvnn_index.keys():
                results[k]['CVNN'].append(cvnn_index[k])
            for k, partition in trees.items():
                internal = calculate_internal(distances, partition, c_max=trees[max_k])

                for key, metric in internal.items():
                    results[k][key].append(metric)



    for k, keys in results.items():
        for key, value in keys.items():
            if key in external_indexes:
                mean_results_external[k].append(mean(value))

            else:
                mean_results_internal[k].append(value[0])

    # dictionary with the choices of "best" k and the "best" value
    choices_dict = defaultdict(list)


    # dendrogram plot
    if os.path.exists('../Results'):
        shutil.rmtree('../Results')

    try:
        os.makedirs('../Results')
    except OSError:
        print('Error: Creating directory. ' + './Results')

    filename_dn = '../Results/ClusteringDendrogram_' + str(link) + '_' + str(map_indexes[index]) + '.pdf'

    with PdfPages(filename_dn) as pp:
        fig = plt.figure(figsize=(40, 20))
        plt.title('Hierarchical Clustering Dendrogram - Linkage: %s, Metrics: %s' % (link, map_indexes[index]), fontsize=30)
        plt.xlabel('data point index', labelpad=20, fontsize=30)
        plt.ylabel('distance', labelpad=10, fontsize=30)
        plt.xticks(size=40)
        plt.yticks(size=40)
        dendrogram(

            Z,
            # truncate_mode = 'lastp',
            # p=6,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=15.,  # font size for the x axis labels
        )

        #c, coph_dists = cophenet(Z, data)
        #txt = 'Cophenetic Correlation Coefficient: ' + str(c)
        #fig.text(0.1, 0.01, txt, fontsize=30)

        pp.savefig(fig)


    print("* Among all indices: \n")

    count_choice = defaultdict(int)
    # table with all mean values of the metrics for every k
    if flag_external == 1:
        choices_dict = choose(choices_dict, mean_results_external, index, 'external')

    if flag_internal == 1:
        choices_dict = choose(choices_dict, mean_results_internal, index, 'internal')

    for values in choices_dict.values():
        count_choice[str(values[0])] += 1

    for k_num in sorted(count_choice):
        print('* ' + str(count_choice[k_num]) + ' proposed ' + k_num + ' as the best number of clusters \n')

    print('\t\t\t***** Conclusion *****\t\t\t\n')

    max_value = 0
    final_k = 0
    for key, value in count_choice.items():
        if value > max_value:
            max_value = value
            final_k = key

    print('* According to the majority rule, the best number of clusters is ' + str(final_k) + '\n')

    if flag_external == 1:
        results_final_external = visualize(mean_results_external, index, 'external')
        print('* External indices: \n')
        print(results_final_external)

    if flag_internal == 1:
        results_final_internal = visualize(mean_results_internal, index, 'internal')
        print('\n* Internal indices: \n')
        print(results_final_internal)

    print('\n* The best partition is:\n')

    final_clusters = fcluster(Z, t=final_k, criterion='maxclust')
    print(final_clusters)

    for nc in range(1, int(final_k) + 1):
        filename_cluster = '../Results/Cluster' + str(nc)+ '.csv'
        count = 0
        with open(filename_cluster, mode='a+') as cluster_file:
            csv_writer = csv.writer(cluster_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for pos, el in enumerate(final_clusters):
                if el == nc:
                    count+=1
                    csv_writer.writerow(data[pos])
        os.rename(filename_cluster, '../Results/Cluster' + str(nc) + '_' + str(count) + 'elements.csv')



def cluster_indices(cluster_assignments,idx):
    import numpy as np
    n = cluster_assignments.max()
    clusters = []
    for cluster_number in range(1, n + 1):
        aux = np.where(cluster_assignments == cluster_number)[0].tolist()
        if aux:
            cluster = list(idx[i] for i in aux)
            clusters.append(cluster)
    return clusters

def visualize(results, index, method):
    from tabulate import tabulate

    if method == 'external':
        head = external_indexes
    else:
        head = internal_indexes

    if index in head:
        vis = tabulate(results.values(), headers=[index], showindex=results.keys())
    else:
        vis = tabulate(results.values(), headers=head, showindex=results.keys())

    return vis

def choose(choices_dict, results, index, method):
    import math

    k_number = 0

    if method == 'external':
        metrics = external_indexes
    else:
        metrics = internal_indexes

    #in case of only one metric
    if (index in metrics):
        if index in min_indexes:
            min_value = math.inf
            for k, v in results.items():
                if v[0] < min_value:
                    min_value = v[0]
                    k_number = k
            choices_dict[index] = [k_number, min_value]

        else:
            max_value = -math.inf
            for k, v in results.items():
                if v[0] > max_value:
                    max_value = v[0]
                    k_number = k
            choices_dict[index] = [k_number, max_value]

    else:

        for pos, metric in enumerate(metrics):
            if metric in min_indexes:
                min_value = math.inf
                for k,v in results.items():
                    if v[pos] < min_value:
                        min_value = v[pos]
                        k_number = k
                choices_dict[metric] = [k_number, min_value]

            else:
                max_value = -math.inf
                for k,v in results.items():
                    if v[pos] > max_value:
                        max_value = v[pos]
                        k_number = k
                choices_dict[metric] = [k_number, max_value]



    return choices_dict

def distance_dict(data):
    """
    Calculate the accumulative distance considering all features, between each pair of observations
    :param data:
    :return:
    """

    from itertools import combinations

    comb = list(combinations([i for i in range(0, len(data))], 2))
    dist_dict = defaultdict(float)

    for pair in comb:
        dist = 0
        for i, j in zip(data[pair[0]], data[pair[1]]):
            dist += abs(float(i) - float(j))
        dist_dict[pair] = dist

    return dist_dict


if __name__ == '__main__':

    import csv

    filename = "/home/nuno/Desktop/Iris.csv"

    data = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
    k = 2

    dataset = []

    with open(filename, 'r') as file:
        for line in csv.reader(file):
            dataset.append(line[1:-1])

    dataset = dataset[1:]

    evaluate(dataset, link='single')