"""
Calculate internal validation. distance_dictset with pairs and their distances should be given.

"""
import itertools
from collections import defaultdict
import math
import clusterval
from scipy.spatial.distance import euclidean
import numpy as np

def calculate_internal(data, distance_dict, clustering, indices=['all']):
    """
    :param data: dataset being evaluated
    :param distance_dict: dictionary with distance between pairs
    :param clustering: dictionary with clustering results and centroids for a range of k number of clusters
    :param index: str, which index to calculate
    :return: dictionary with indices values for a range of k number of clusters
    """

    indices_funcs = {'CVNN': cvnn, 'XB': xb_improved, 'S_Dbw': s_dbw, 'DB': db_improved, 'S': silhouette, 'SD': sd,
                     'PBM': pbm, 'Dunn': dunn}
    results = defaultdict(dict)

    if isinstance(indices, str):
        indices = [x.strip() for x in indices.split(',')]



    for index in indices:
        if index == 'all' or index == 'internal':
            for cvi, func in indices_funcs.items():
                results[cvi] = func(clustering, data, distance_dict)
        elif index in indices_funcs.keys():
            results[index] = indices_funcs[index](clustering, data, distance_dict)


    return results


def cvnn(clustering, data, distance_dict):

    """

    Metric based on the calculation of intercluster separaration and intracluster compatness.
    Lower value of this metric indicates a better clustering result.

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: CVNN index
    """
    comp = defaultdict(float)
    sep = defaultdict(float)


    #nearest neighbours values to test
    nn_values = [1, 5, 10, 15, 20]

    #calculate nearest neighbours for k=[1,5,10,15,20]
    nn_history = getknn(distance_dict, nn_values)

    for k, clusters in clustering.items():

        compactness = cvnn_compactness(distance_dict, clusters['clusters'])
        separation = cvnn_separation(clusters['clusters'], nn_history, nn_values)

        comp[k] = compactness
        sep[k] = separation


    maxcomp = max(i for i in comp.values())
    maxsep = max(i for i in sep.values())


    if maxcomp == 0 or maxsep == 0:
        maxcomp = 1
        maxsep = 1

    comp_final = {key: value / maxcomp for key, value in comp.items()}
    sep_final = {key: value / maxsep for key, value in sep.items()}

    cvnn_index = {key: sep_final[key] + comp_final[key] for key in comp.keys()}


    return cvnn_index


def getknn(distance_dict, nn_values):

    nn_history = {'1': defaultdict(list), '5': defaultdict(list), '10': defaultdict(list), '15': defaultdict(list), '20': defaultdict(list)}

    #get first row index of distance_dict
    first_el = next(iter(distance_dict))[0]
    elements = [first_el]
    for key in distance_dict.keys():
        if key[0] != first_el:
            break
        else:
            elements.append(key[1])

    for k in nn_values:
        for el in elements:
            nn = []
            pairs = {key : value for key, value in distance_dict.items() if el in key}
            for k_aux in range(0, k):
                try:
                    key_min = min(pairs.keys(), key=(lambda x: pairs[x]))
                    if key_min[0] == el:
                        nn.append(key_min[1])
                    else:
                        nn.append(key_min[0])
                    del pairs[key_min]
                except:
                    raise ValueError("There is empty clusters being generated, please set max_k to a lower values (default=8)")
            nn_history[str(k)][str(el)] = nn

    return nn_history


def cvnn_compactness(distance_dict, clusters):
    compactness = 0
    sum_of_objects = 0
    for cluster in clusters:
        n_i = len(cluster)
        if n_i == 0:
            raise ValueError("Empty clusters being formed")
        sum_of_objects += n_i * (n_i - 1)
        # intracluster compactness
        pairs = list(itertools.combinations(cluster, 2))
        distance_i = pairwise_distance(pairs, distance_dict)
        compactness += distance_i


    return compactness / sum_of_objects


def cvnn_separation(clusters, nn_history, nn_values):
    best_k_value = []
    for k in nn_values:
        separation = []
        for cluster in clusters:
            n_i = len(cluster)
            if n_i == 0:
                raise ValueError("Empty clusters being formed")
            # intercluster separation
            sum_of_weights = 0
            for el in cluster:
                count_nn = 0
                for n in nn_history[str(k)][str(el)]:
                    if n not in cluster:
                        count_nn += 1
                sum_of_weights += count_nn / k

            separation.append(sum_of_weights / n_i)

        best_k_value.append(max(separation))

    return min(best_k_value)

def pairwise_distance(pairs, distance_dict):
    sum_of_distances = 0
    for pair in pairs:
        if pair not in distance_dict.keys():
            pair = (pair[1], pair[0])
        sum_of_distances += float(distance_dict[pair])
    return sum_of_distances

def xb_improved(clustering, data, distance_dict): #BIB: New indices for cluster validity assessment - Kim
    '''
    The Xie-Beni improved index (XB) defines the intercluster separation as the minimum square distance between
    cluster centers, and the intracluster compactness as the maximum square distance between each distance_dict object and its
    cluster center. The optimal cluster number is reached when the minimum of XB is found

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: Xie Beni improved index.
    '''

    xb_improved_index = defaultdict(float)
    for k, clusters in clustering.items():
        max_dist_elements = []
        min_dist_clusters = []
        n_c = len(clusters['clusters'])
        for i, cluster in enumerate(clusters['clusters']):
            sum_distances_k = 0
            n_k = len(cluster)
            if n_k == 0:
                raise ValueError("empty clusters being formed, please set a lower value for max_k")
            for el in cluster:
                dist = euclidean(data[el], clusters['centroids'][i])
                dist2 = math.sqrt(math.pow(dist, 2))
                sum_distances_k += dist2

            max_dist_elements.append(sum_distances_k / n_k)

            if i == (n_c - 1):
                continue
            else:
                for j in range(i + 1, n_c):
                    dist_centroids = euclidean(clusters['centroids'][i], clusters['centroids'][j])
                    min_dist_clusters.append(math.sqrt(math.pow(dist_centroids, 2)))


        xb_improved_index[k] = max(max_dist_elements) / min(min_dist_clusters)

    return xb_improved_index



def s_dbw(clustering, data, distance_dict):
    '''
    The S Dbw index (S Dbw) takes density into account to measure the intercluster separation.
    The basic idea is that for each pair of cluster centers, at least one of their densities should be larger
    than the density of their midpoint. The intracluster compactness is based on variances of cluster objects
    The index is the summation of these two terms and the minimum value of S Dbw indicates the
    optimal cluster number.


    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: S_Dbw index.
    '''

    s_dbw_index = defaultdict(float)

    centroid_dataset = np.mean(data, axis=0)
    variance_d = calculate_variance(data, [i for i in range(len(data))], centroid_dataset)

    for k, clusters in clustering.items():
        s_dbw_index[k] = scat(data, clusters['clusters'], clusters['centroids'], variance_d)\
                         + dens_bw(clusters['clusters'], data, distance_dict, clusters['centroids'])

    return s_dbw_index

def calculate_variance(data, cluster, mean):

    variance = 0.0
    for el in cluster:
        dist = euclidean(data[el], mean)
        variance += math.pow(dist, 2)

    return variance / len(cluster)

def scat(data, clusters, centroids, variance_d):
    #scat improved from BIB: New indices for cluster validity assessment - Kim

    variance_clusters = []

    for ci, cluster in enumerate(clusters):
        variance_ci = calculate_variance(data, cluster, centroids[ci])
        variance_clusters.append(variance_ci/variance_d)

    return max(variance_clusters)


def dens_bw(clusters, data, distance_dict, centroids):

    n_c = len(clusters)
    avg_std_deviation = avg_stdev(data, clusters, centroids)

    result_sum = 0.0

    for i, cluster in enumerate(clusters):
        if i == (n_c - 1):
            break
        for j in range(i+1, n_c):
            u_ij = [(c_i + c_j)/2 for c_i,c_j in zip(centroids[i],centroids[j])]
            n_ij = cluster + clusters[j]
            dens_ij = density(data,n_ij, u_ij, avg_std_deviation)
            dens_i = density(data,n_ij, centroids[i], avg_std_deviation)
            dens_j = density(data,n_ij, centroids[j], avg_std_deviation)

            res = dens_ij / max(dens_i, dens_j)
            result_sum += res

    return result_sum / (n_c * (n_c - 1))


def density(data, n_ij, center, avg_stdev):

    sum_density = 0
    for el in n_ij:
        if euclidean(data[el], center) <= avg_stdev:
            sum_density += 1

    return sum_density

def avg_stdev(data, clusters, centroids):

    average_stdev = 0.0
    for i, cluster in enumerate(clusters):
        stdev = 0.0
        for el in cluster:
            dist = euclidean(data[el], centroids[i])
            stdev += math.pow(dist, 2)
        stdev = math.sqrt((stdev / len(cluster)))
        average_stdev += stdev

    return average_stdev / len(clusters)

def db_improved(clustering, data, distance_dict):
    '''
    This index is obtained by averaging all cluster similarities. A smaller value indicates a better clustering.

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: Davies-Bouldin improved index.
    '''

    db_improved_index = defaultdict(float)
    for k, clusters in clustering.items():
        n_c = len(clusters['clusters'])
        sum_of_similarities = 0
        for i, cluster in enumerate(clusters['clusters']):
            if i == (n_c - 1):
                continue
            rij = []
            distance_centroids = []
            if len(cluster) == 0:
                raise ValueError("Empty clusters being formed. Set a lower value for max_k")

            elif len(cluster) == 1:
                s_i = 0
            else:
                s_i = similarity(cluster, data, clusters['centroids'][i])
            for j in range(i+1,n_c):
                if len(clusters['clusters'][j]) == 0:
                    raise ValueError("Empty clusters being formed. Set a lower value for max_k")

                elif len(clusters['clusters'][j]) == 1:
                    s_j = 0
                else:
                    s_j = similarity(clusters['clusters'][j], data, clusters['centroids'][j])

                dissimilarity = euclidean(clusters['centroids'][i], clusters['centroids'][j])

                rij.append((s_i + s_j)/dissimilarity)


            sum_of_similarities += (max(rij))

        db_improved_index[k] = sum_of_similarities / n_c

    return db_improved_index

def similarity(cluster, data, centroid):

    result = 0
    for el in cluster:
        result += euclidean(data[el], centroid)
    sim = result / len(cluster)
    
    return sim

def silhouette(clustering, data, distance_dict):
    '''
     Validates the clustering performance based on the pairwise difference of between and within cluster distances.
     The optimal cluster number is determined by maximizing the value of this index.

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: Silhouette index.
    '''

    silhouette_index = defaultdict(float)
    for k, clusters in clustering.items():
        n_c = len(clusters['clusters'])
        sum_clusters_diff = 0
        for i, cluster in enumerate(clusters['clusters']):
            if len(cluster) != 0:
                sum_pairwise = 0
                for el in cluster:
                    b = silhouette_b([x for y, x in enumerate(clusters['clusters']) if i != y], el, distance_dict)
                    a = silhouette_a(cluster, el, distance_dict)
                    if max(a,b) == 0:
                        sum_pairwise += (b - a)
                    else:
                        sum_pairwise += (b - a) / max(a,b)
                sum_clusters_diff += sum_pairwise / len(cluster)

        silhouette_index[k] = sum_clusters_diff / n_c

    return silhouette_index


def silhouette_a(cluster, el, distance_dict):
    n_i = len(cluster)
    sum_dist = 0
    for c_i in cluster:
        if c_i == el:
            continue
        pair = (el, c_i)
        if pair not in distance_dict.keys():
            pair = (pair[1], pair[0])
        sum_dist += float(distance_dict[pair])
    if n_i <= 1:
        result = sum_dist
    else:
        result = sum_dist / (n_i - 1)

    return result

def silhouette_b(clusters, el, distance_dict):
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
            if pair not in distance_dict.keys():
                pair = (pair[1], pair[0])
            sum_dist_within += float(distance_dict[pair])

        array_of_between_clusters.append(sum_dist_within / n_j)

    return min(array_of_between_clusters)


def sd(clustering, data, distance_dict):
    '''
    Based on the concepts of average scattering, which indicates the compactness between clusters
    and the total separation of clusters, which indicates the separation between the items of a cluster.
    The optimal cluster number is determined by minimizing the value of this index.

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: SD index.
    '''

    sd_index = defaultdict(float)

    centroid_dataset = np.mean(data, axis=0)
    variance_d = calculate_variance(data, [i for i in range(len(data))], centroid_dataset)

    # Maximum number of input clusters.
    c_max = clustering[next(reversed(clustering))]['centroids']

    for k, clusters in clustering.items():

        sd_index[k] = dis(clusters['centroids']) + scat(data, clusters['clusters'], clusters['centroids'], variance_d)*dis(c_max)

    return sd_index


def dis(centers):
    '''
    Total separation between clusters.
    :param centers: list
    Clusters centroids
    :return: float
    '''
    centroids_pairs = list(itertools.combinations(centers, 2))
    dist_lst = [abs(pair[0]-pair[1]) for pair in centroids_pairs]

    d_max = max(dist_lst)
    d_min = min(i for i in dist_lst if i > 0)

    total = sum(dist_lst)

    return (d_max / d_min) * total


def pbm(clustering, data, distance_dict):
    '''
    The PBM index (acronym constituted of the initals of the names of its authors,
    Pakhira, Bandyopadhyay and Maulik) is calculated using the distances between
    the points and their cluster centers and the distances between the cluster centers
    themselves.

    :param clustering: dictionary with clustering results and respective centroids for each k simulation.
    :param data: dataset being analysed
    :param distance_dict: distance_dictset represented as a distance matrix.
    :return: PBM index.
    '''

    pbm_index = defaultdict(float)

    center_distance_dictset = sum(pair for k, pair in distance_dict.items()) / len(distance_dict)
    e_t = sum(abs(pair - center_distance_dictset) for k,pair in distance_dict.items())
    e_w = 0
    for k, clusters in clustering.items():
        comb_clusters = itertools.combinations(clusters['centroids'], 2)
        max_dist_clusters = max(abs(dst[0] - dst[1]) for dst in comb_clusters)
        for i, cluster in enumerate(clusters['clusters']):
            sum_dist2center = 0
            comb_pairs = list(itertools.combinations(cluster, 2))
            for pair in comb_pairs:
                if pair not in distance_dict:
                    pair = (pair[1], pair[0])
                d = abs(float(distance_dict[pair]) - clusters['centroids'][i])
                sum_dist2center += d
            e_w += sum_dist2center

        #if clause for the case with cluster with only one element
        if e_w == 0:
            dist2center = 1
        else:
            dist2center = e_t/e_w

        pbm_index[k] = math.pow((1/len(clusters['clusters'])) * dist2center * max_dist_clusters, 2)

    return pbm_index


def dunn(clustering, data, distance_dict):
    '''
        Dunn's calculates the minimum distance between clusters to measure the intercluster separation and the maximum
        diameter among all clusters to measure the intracluster compactness.

        :param clustering: dictionary with clustering results and respective centroids for each k simulation.
        :param data: dataset being analysed
        :param distance_dict: distance_dictset represented as a distance matrix.
        :return: Dunn index.
        '''

    dunn_index = defaultdict(float)

    for k, clusters in clustering.items():
        
        size = len(clusters['clusters'])
        #for the case in which there is only one object in the cluster
        if size < 2:
            dunn_index[k] = 0

        else:

            #step for calculating max diameter of clustering
            diameter_all = []
            for c in clusters['clusters']:
                if len(c) >= 2:
                    pairs = list(itertools.combinations(c, 2))
                    distances = []
                    for pair in pairs:
                        if pair not in distance_dict.keys():
                            pair = (pair[1], pair[0])
                        distances.append(float(distance_dict[pair]))

                    diameter_all.append(max(distances))

            max_diameter = max(diameter_all)

            compare_clusters = []
            for i in range(size):
                list_of_evals = []
                if i+1 < size:
                    for j in range(i+1, size):
                        dissimilarity = math.inf
                        pairs = list(itertools.product(clusters[i], clusters[j]))
                        for pair in pairs:
                            if pair not in distance_dict.keys():
                                pair = (pair[1], pair[0])
                            dissimilarity_aux = float(distance_dict[pair])
                            if dissimilarity_aux < dissimilarity:
                                dissimilarity = dissimilarity_aux

                        list_of_evals.append(dissimilarity/max_diameter)

                    compare_clusters.append(min(list_of_evals))


            dunn_index[k] = min(compare_clusters)

    return dunn_index
