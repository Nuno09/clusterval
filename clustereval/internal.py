"""
Calculate internal validation. Dataset with pairs and their distances should be given.

"""
def calculate_internal(data, clusters, k):
    from scipy.cluster.hierarchy import cophenet
    import pandas as pd
    import csv
    from collections import defaultdict


    distance_dict = defaultdict(dict)

    if isinstance(data, list):
        for l in data:
            if int(l[0]) < int(l[2]):
                v1 = l[0]
                v2 = l[1]
            else:
                v2 = l[0]
                v1 = l[1]
            score = l[2]
            distance_dict[(v1, v2)] = score

    elif isinstance(data, dict):
        distance_dict = data

    elif isinstance(data, pd.core.frame.DataFrame):
        fields = data.keys()
        dict_aux = data.to_dict('index')
        for key,value in dict_aux.items():
            if int(value[fields[0]]) < int(value[fields[1]]):
                v1 = value[fields[0]]
                v2 = value[fields[1]]
            else:
                v2 = value[fields[0]]
                v1 = value[fields[1]]
            score = value[fields[2]]
            distance_dict[(v1, v2)] = score


    elif data[-3:] == 'csv':
        with open(data, "r") as file:
            r = csv.DictReader(file) #just a reader that can map information of each row into a dict
            fields = r.fieldnames
            for row in r:
                if int(row[fields[0]]) < int(row[fields[1]]):
                    v1 = row[fields[0]]
                    v2 = row[fields[1]]
                else:
                    v2 = row[fields[0]]
                    v1 = row[fields[1]]
                score = row[fields[2]]

                distance_dict[(v1, v2)] = score

    cvnn_idx = cvnn(clusters, distance_dict, k)

    #XB**
    xb = xb_improved(clusters, distance_dict, k)

    #S_Bbw
    sdbw = s_dbw(clusters, distance_dict, k)

    #DB**

    #S

    return {'CVNN' : cvnn_idx, 'XB**' : xb, 'S_Dbw' : sdbw}


def cvnn(clusters, data, k):

    """

    Metric based on the calculation of intercluster separaration and intracluster compatness.
    Lower value of this metric indicates a better clustering result.

    """
    import itertools

    dict_clusters = {i: clusters[i] for i in range(len(clusters))}
    separation = []
    compactness = 0
    for idx, cluster in dict_clusters.items():
        n_i = len(cluster)

        #intracluster compactness
        pairs = list(itertools.combinations(cluster, 2))
        distance_i = pairwise_distance(pairs, data)
        compactness += (2 / (n_i*(n_i - 1))) * distance_i

        #intercluster separation
        sum_of_weights = 0
        for el in cluster:
            count_nn = 0
            nn = getknn(data, el, k)
            for n in nn:
                for m in n:
                    if m != el and m not in cluster:
                        count_nn += 1
            sum_of_weights += count_nn / k

        separation.append(sum_of_weights / n_i)


    return (max(separation) / len(clusters)) + (compactness / len(clusters))


def getknn(data, el, k):
    nn = []
    pairs = {key : value for key, value in data.items() if el in key}
    for k_aux in range(k):
        key_min = min(pairs.keys(), key=(lambda x: pairs[x]))
        nn.append(key_min)
        del pairs[key_min]
    return nn


def pairwise_distance(pairs, data):
    sum_of_distances = 0
    for pair in pairs:
        if pair not in data.keys():
            pair = (pair[1], pair[0])
        sum_of_distances += float(data[pair])

    return sum_of_distances

def xb_improved(clusters, data, k):
    '''
    The Xie-Beni improved index (XB**) defines the intercluster separation as the minimum square distance between
    cluster centers, and the intracluster compactness as the maximum square distance between each data object and its
    cluster center. The optimal cluster number is reached when the minimum of XB** is found

    :param clusters:
    :param data:
    :return:
    '''
    import math
    import itertools

    centroids = cluster_centroids(clusters, data)
    max_dist_elements = []
    min_dist_clusters = []
    max_diff_clusters = []
    for i, cluster in enumerate(clusters):
        if i+1 < len(centroids):
            max_diff_clusters.append(centroids[i] - centroids[i+1])
        sum_distances_k = 0
        n_k = len(cluster)
        for key, el in data.items():
            dist = float(el) - centroids[i]
            dist2 = math.sqrt(math.pow(dist, 2))
            sum_distances_k += dist2 / n_k
        max_dist_elements.append(sum_distances_k)

    for d in list(itertools.combinations(centroids, 2)):
        dist_c = math.sqrt(math.pow((d[0] - d[1]), 2))
        min_dist_clusters.append(dist_c)

    return (max(max_dist_elements) + max(max_diff_clusters)) / min(min_dist_clusters)



def cluster_centroids(clusters, data):
    import itertools

    cluster_centroids_lst = []
    for cluster in clusters:
        sum_of_scores = 0
        pairs = list(itertools.combinations(cluster, 2))
        for pair in pairs:
            if pair not in data.keys():
                pair = (pair[1], pair[0])
            sum_of_scores += float(data[pair])

        cluster_centroids_lst.append(sum_of_scores / len(pairs))

    return cluster_centroids_lst

def s_dbw(clusters, data, k):
    '''
    The S Dbw index (S Dbw) takes density into account to measure the intercluster separation.
    The basic idea is that for each pair of cluster centers, at least one of their densities should be larger
    than the density of their midpoint. The intracluster compactness is based on variances of cluster objects
    The index is the summation of these two terms and the minimum value of S Dbw indicates the
    optimal cluster number.


    :param clusters:
    :param data:
    :param k:
    :return:
    '''

    return scat(clusters, data) + dens_bw(clusters, data)

def scat(clusters, data):
    from itertools import combinations
    from statistics import variance
    import math

    sum_variances = 0
    variance_d = variance(float(el) for key, el in data.items())
    variance_d = math.sqrt(variance_d*variance_d)
    n_c = len(clusters)
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        variance_list = []
        pairs = list(combinations(cluster, 2))
        for pair in pairs:
            if pair not in data.keys():
                pair = (pair[1], pair[0])
            variance_list.append(float(data[pair]))
        variance_i = variance(variance_list)

        sum_variances += math.sqrt(variance_i*variance_i) / variance_d

    return sum_variances / n_c


def dens_bw(clusters, data):
    from statistics import stdev
    from collections import defaultdict

    n_c = len(clusters)
    avg_std_deviation = avg_stdev(data, clusters)
    centroids = cluster_centroids(clusters, data)
    result_sum = 0

    #dict with tuples of clusters
    pairs = defaultdict()
    for i, c1 in enumerate(clusters):
        for j in range(i + 1, n_c):
            pairs[i,j] = (c1, clusters[j])

    for key,tup in pairs.items():
        dens_ij = density(data, tup, avg_std_deviation, centroids[key[0]], centroids[key[1]])
        dens_i = density(data, tup[0], avg_std_deviation, centroids[key[0]])
        dens_j = density(data, tup[0], avg_std_deviation, centroids[key[1]])

        if max(dens_i, dens_j) != 0:
            result_sum += (dens_ij / max(dens_i, dens_j))
        else:
            result_sum += 10

    return result_sum / n_c * (n_c - 1)


def density(data, tup, avgstdev, c1, c2=None):
    import itertools

    if c2 != None:
        u_ij = (c1 + c2) / 2
        tup = list(itertools.chain.from_iterable(tup))
    else:
        u_ij = c1
    sum_density = 0
    pairs_in_clusters = list(itertools.combinations(tup, 2))
    for pair in pairs_in_clusters:
        if pair not in data.keys():
            pair = (pair[1],pair[0])
        distance = abs(float(data[pair]) - u_ij)
        if distance <= avgstdev:
            sum_density += 1

    return sum_density

def avg_stdev(data, clusters):
    import itertools
    from statistics import stdev

    sum_stdev = 0
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        tupls = list(itertools.combinations(cluster, 2))
        stdev_list = []
        for tupl in tupls:
            if tupl not in data.keys():
                tupl = (tupl[1], tupl[0])
            stdev_list.append(float(data[tupl]))
        sum_stdev += stdev(stdev_list)

    return sum_stdev / len(clusters)


if __name__ == '__main__':
    from collections import defaultdict
    from tabulate import tabulate
    import math
    import pandas as pd
    import csv
    import itertools

    dicio_statistics = defaultdict()
    indexes = ('CVNN', 'XB**', 'S_Dbw')
    for index in indexes:
        dicio_statistics[index] = []

    file = '/home/nuno/Documentos/IST/Tese/AliClu/Code/dataframe.csv'
    k = 2
    clusters = [['0', '1', '2', '3', '17', '5', '6', '7', '8', '9', '11', '14', '18', '20', '21', '22', '23'],
               ['10', '12', '13', '15', '16', '4', '19'], ['24', '25', '26', '27', '28', '29']]

    #create list and dictionary out of csv
    with open(file, 'r') as csv_file:
        lst = []
        d = defaultdict()
        r = csv.reader(csv_file, delimiter=',')
        next(r)
        for row in r:
            lst.append(row)
            if int(row[0]) < int(row[1]):
                d[(row[0], row[1])] = row[2]
            else:
                d[(row[1], row[0])] = row[2]



    #create dataframe out of csv
    df = pd.DataFrame(lst, columns=['patient1', 'patient2', 'score'])

    metrics = calculate_internal(df, clusters, k)
    # print(computed_indexes)
    for index in indexes:
        dicio_statistics[index].append(metrics[index])

    print(tabulate([list(dicio_statistics.values())], headers=list(dicio_statistics.keys())))
