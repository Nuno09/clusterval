"""
Calculate internal validation. Dataset with pairs and their distances should be given.

"""
import  itertools
from collections import defaultdict
import math

def calculate_internal(distance_dict, clustering, indices=['all']):
    """

    :param distance_dict: dictionary with distance between pairs
    :param clustering: disctionary with clustering results for a range of k number of clusters
    :param index: str, which index to calculate
    :return: dictionary with indices values for a range of k number of clusters
    """
    indices_funcs = {'CVNN': cvnn, 'XB*': xb_improved, 'S_Dbw': s_dbw, 'DB*': db_improved, 'S': silhouette, 'SD': sd}
    results = defaultdict(dict)

    if isinstance(indices, str):
        indices = [x.strip() for x in indices.split(',')]

    for index in indices:
        if index == 'all' or index == 'internal':
            for cvi, func in indices_funcs.items():
                results[cvi] = func(clustering, distance_dict)
        elif index in indices_funcs.keys():
            results[index] = indices_funcs[index](clustering, distance_dict)


    return results


def cvnn(clustering, data):

    """

    Metric based on the calculation of intercluster separaration and intracluster compatness.
    Lower value of this metric indicates a better clustering result.

    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: CVNN index
    """

    comp = defaultdict(float)
    sep = defaultdict(float)
    sum_of_objects = 0

    for k, clusters in clustering.items():
        separation = []
        compactness = 0
        for cluster in clusters:
            n_i = len(cluster)
            sum_of_objects += n_i * (n_i - 1)
            if n_i != 0:
                #intracluster compactness
                pairs = list(itertools.combinations(cluster, 2))
                distance_i = pairwise_distance(pairs, data)
                if distance_i != 0:
                    compactness += distance_i #Method-Independent Indices for Cluster Validation and Estimating the Number of Clusters - chapter 26 Handbook of Cluster Analysis


                # intercluster separation
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
            else:
                sum_of_objects = 1
                compactness += 10
                separation.append(10)

        comp[k] = compactness / sum_of_objects
        sep[k] = max(separation)

    maxcomp = max(i for i in comp.values())
    maxsep = max(i for i in sep.values())


    if maxcomp == 0 or maxsep == 0:
        maxcomp = 1
        maxsep = 1

    comp_final = {key: value / maxcomp for key, value in comp.items()}
    sep_final = {key: value / maxsep for key, value in sep.items()}

    cvnn_index = {key: sep_final[key] + comp_final[key] for key in comp.keys()}

    return cvnn_index


def getknn(data, el, k):
    nn = []
    pairs = {key : value for key, value in data.items() if el in key}
    for k_aux in range(0, k):
        try:
            key_min = min(pairs.keys(), key=(lambda x: pairs[x]))
            nn.append(key_min)
            del pairs[key_min]
        except:
            raise ValueError("There is empty clusters being generated, please set max_k to a lower values (default=8)")
    return nn


def pairwise_distance(pairs, data):
    sum_of_distances = 0
    for pair in pairs:
        if pair not in data.keys():
            pair = (pair[1], pair[0])
        sum_of_distances += float(data[pair])

    return sum_of_distances

def xb_improved(clustering, data): #BIB: New indices for cluster validity assessment - Kim
    '''
    The Xie-Beni improved index (XB*) defines the intercluster separation as the minimum square distance between
    cluster centers, and the intracluster compactness as the maximum square distance between each data object and its
    cluster center. The optimal cluster number is reached when the minimum of XB** is found

    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: Xie Beni improved index.
    '''

    xb_improved_index = defaultdict(float)
    for k, clusters in clustering.items():
        centroids = cluster_centroids(clusters, data)
        max_dist_elements = []
        min_dist_clusters = []
        for i, cluster in enumerate(clusters):
            sum_distances_k = 0
            n_k = len(cluster)
            if n_k != 0:
                for key, el in data.items():
                    dist = float(el) - centroids[i]
                    dist2 = math.sqrt(math.pow(dist, 2))
                    sum_distances_k += dist2 / n_k
                max_dist_elements.append(sum_distances_k)

        if len(centroids) == 1:
            min_dist_clusters.append(0)
        else:
            for d in list(itertools.combinations(centroids, 2)):
                dist_c = math.sqrt(math.pow((d[0] - d[1]), 2))
                min_dist_clusters.append(dist_c)

        #if min(min_dist_clusters) == 0:
         #   min_dis = 1
        #else:
        try:
            min_dis = min(i for i in min_dist_clusters if i > 0)
        except:
            raise ValueError("There is empty clusters being generated, please set max_k to a lower values (default=8)")

        xb_improved_index[k] = max(max_dist_elements) / min_dis

    return xb_improved_index



def cluster_centroids(clusters, data):

    cluster_centroids_lst = []
    for cluster in clusters:
        if len(cluster) <= 1:
            cluster_centroids_lst.append(0)
            continue
        sum_of_scores = 0
        pairs = list(itertools.combinations(cluster, 2))
        if pairs:
            for pair in pairs:
                if pair not in data.keys():
                    pair = (pair[1], pair[0])
                sum_of_scores += float(data[pair])

            cluster_centroids_lst.append(sum_of_scores / len(pairs))

    return cluster_centroids_lst

def s_dbw(clustering, data):
    '''
    The S Dbw index (S Dbw) takes density into account to measure the intercluster separation.
    The basic idea is that for each pair of cluster centers, at least one of their densities should be larger
    than the density of their midpoint. The intracluster compactness is based on variances of cluster objects
    The index is the summation of these two terms and the minimum value of S Dbw indicates the
    optimal cluster number.


    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: S_Dbw index.
    '''

    s_dbw_index = defaultdict(float)
    for k, clusters in clustering.items():
        s_dbw_index[k] = scat(clusters, data) + dens_bw(clusters, data)

    return s_dbw_index

def scat(clusters, data):
    #scat improved from BIB: New indices for cluster validity assessment - Kim
    from statistics import variance

    centroids = cluster_centroids(clusters, data)
    variance_clusters = []
    variance_d = variance(float(el) for key, el in data.items())

    for ci, cluster in enumerate(clusters):
        if len(cluster) < 3:
            pairs = list(itertools.combinations(cluster, 2))
            for pair in pairs:
                if pair not in data.keys():
                    pair = (pair[1], pair[0])
                variance_i = math.pow((float(data[pair]) - centroids[ci]), 2)
                variance_clusters.append(variance_i / variance_d)
            continue
        variance_list = []
        pairs = list(itertools.combinations(cluster, 2))
        for pair in pairs:
            if pair not in data.keys():
                pair = (pair[1], pair[0])
            variance_list.append(float(data[pair]))
        variance_i = variance(variance_list, centroids[ci])

        variance_clusters.append(variance_i / variance_d)

    return max(variance_clusters)


def dens_bw(clusters, data):

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
        dens_i = density(data, tup, avg_std_deviation, centroids[key[0]])
        dens_j = density(data, tup, avg_std_deviation, centroids[key[1]])

        if max(dens_i, dens_j) != 0:
            result_sum += (dens_ij / max(dens_i, dens_j))
        else:
            result_sum += 10

    return result_sum / n_c * (n_c - 1)


def density(data, tup, avgstdev, c1, c2=None):


    if c2 != None:
        u_ij = (c1 + c2) / 2
    else:
        u_ij = c1
    tup = list(itertools.chain.from_iterable(tup))
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

def db_improved(clustering, data):
    '''
    This index is obtained by averaging all cluster similarities. A smaller value indicates a better clustering.

    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: Davies-Bouldin improved index.
    '''

    db_improved_index = defaultdict(float)
    for k, clusters in clustering.items():
        n_c = len(clusters)

        centroids = cluster_centroids(clusters, data)
        sum_of_similarities = 0
        for i, cluster in enumerate(clusters):
            array_of_similarities = []
            distance_centroids = []
            if len(cluster) == 1:
                s_i = 0
            else:
                s_i = similarity(cluster, data, centroids[i])
            for j, clusterJ in enumerate(clusters):
                if j != i:
                    if len(clusterJ) == 1:
                        s_j = 0
                    else:
                        s_j = similarity(clusterJ, data, centroids[j])
                    array_of_similarities.append(s_i + s_j)
                    distance_centroids.append(abs(centroids[i]-centroids[j]))
            if len(centroids) == 1:
                distance_centroids.append(0)
                array_of_similarities.append(0)
            min_dst_centroids = min(i for i in distance_centroids if i > 0)
            #if min_dst_centroids == 0:
                #min_dst_centroids = 1
            sum_of_similarities += (max(array_of_similarities) / min_dst_centroids)

        db_improved_index[k] = sum_of_similarities / n_c

    return db_improved_index

def similarity(cluster, data, centroid):
    if len(cluster) == 0:
        sim = 0
    else:
        pairs_in_cluster = list(itertools.combinations(cluster, 2))
        result = 0
        for tup in pairs_in_cluster:
            if tup not in data.keys():
                tup = (tup[1], tup[0])
            result += abs(float(data[tup]) - centroid)
        sim = result / len(cluster)
    
    return sim

def silhouette(clustering, data):
    '''
     Validates the clustering performance based on the pairwise difference of between and within cluster distances.
     The optimal cluster number is determined by maximizing the value of this index.

    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: Silhouette improved index.
    '''

    silhouette_index = defaultdict(float)
    for k, clusters in clustering.items():
        n_c = len(clusters)
        sum_clusters_diff = 0
        for i, cluster in enumerate(clusters):
            if len(cluster) != 0:
                sum_pairwise = 0
                for el in cluster:
                    b = silhouette_b([x for y, x in enumerate(clusters) if i != y], el, data)
                    a = silhouette_a(cluster, el, data)
                    if max(a,b) == 0:
                        sum_pairwise += (b - a)
                    else:
                        sum_pairwise += (b - a) / max(a,b)
                sum_clusters_diff += sum_pairwise / len(cluster)

        silhouette_index[k] = sum_clusters_diff / n_c

    return silhouette_index


def silhouette_a(cluster, el, data):
    n_i = len(cluster)
    sum_dist = 0
    for c_i in cluster:
        if c_i == el:
            continue
        pair = (el, c_i)
        if pair not in data.keys():
            pair = (pair[1], pair[0])
        sum_dist += float(data[pair])
    if n_i <= 1:
        result = sum_dist
    else:
        result = sum_dist / (n_i - 1)

    return result

def silhouette_b(clusters, el, data):
    array_of_between_clusters = []
    if len(clusters) == 0:
        array_of_between_clusters.append(0)
    for cluster in clusters:
        sum_dist_within = 0
        n_j = len(cluster)
        if n_j == 0:
            array_of_between_clusters.append(0)
        for c_j in cluster:
            pair = (el, c_j)
            if pair not in data.keys():
                pair = (pair[1], pair[0])
            sum_dist_within += float(data[pair])

        array_of_between_clusters.append(sum_dist_within / n_j)

    return min(array_of_between_clusters)


def sd(clustering, data):
    '''
    Based on the concepts of average scattering, which indicates the compactness between clusters
    and the total separation of clusters, which indicates the separation between the items of a cluster.
    The optimal cluster number is determined by minimizing the value of this index.

    :param clusters: Clusters represented in List format.
    :param data: Dataset represented as a distance matrix.
    :return: Davies-Bouldin improved index.
    '''

    sd_index = defaultdict(float)
    for k, clusters in clustering.items():

        centroids = cluster_centroids(clusters, data)

        #Maximum number of input clusters.
        c_max = clustering[list(clustering)[-1]]

        sd_index[k] = dis(centroids) + scat(clusters, data)*dis(cluster_centroids(c_max, data))

    return sd_index


def dis(centroids):
    '''
    Total separation between clusters.
    :param centroids: list
    Clusters centroids
    :return: float
    '''
    centroids_pairs = list(itertools.combinations(centroids, 2))
    dist_lst = [abs(pair[0]-pair[1]) for pair in centroids_pairs]

    d_max = max(dist_lst)
    d_min = min(i for i in dist_lst if i > 0)

    total = sum(dist_lst)

    return (d_max / d_min) * total



