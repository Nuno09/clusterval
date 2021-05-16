"""
Function evaluate does hierarchical clustering and validation of it's results.
Validation can be done with external or internal indices, or both.


"""


from scipy.cluster.hierarchy import dendrogram, fcluster, cut_tree
from fastcluster import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from statistics import mean
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math
import random

from clusterval.internal import calculate_internal
from clusterval.external import calculate_external


class Clusterval:
    """
    min_k: int, optional
    minimum number of clusters to test. Default 2.

    max_k: int, optional
    maximum number of clusters to test. Default 8.

    algorithm: str, optional
    clustering algorithm to use. Accepts 'single', 'ward'(default), 'complete', 'centroid', 'average' and 'kmeans

    bootstrap_samples: int, optional
    number of bootstrap samples simulated. Default 250

    index: list of str or str, optional
    what indices to be calculated. Accepts 'all' to calculate all, 'internal' to calculate only internal indices,
     'external' to calculate external indices, or one of ['AR', 'FM', 'J', 'AW', 'VD', 'H', 'F', 'VI', 'K', 'Phi', 'RT', 'SS',
     'CVNN', 'XB', 'S_Dbw', 'DB', 'S', 'SD', 'PBM', 'Dunn']. Default 'all'.
    """

    def __init__(self, min_k=2, max_k=8, algorithm='ward', bootstrap_samples=250, index='all'):

        external_indices = ['AR', 'FM', 'J', 'AW', 'VD', 'H', 'F', 'VI', 'K', 'Phi', 'RT', 'SS']

        internal_indices = ['CVNN', 'XB', 'S_Dbw', 'DB', 'S', 'SD', 'PBM', 'Dunn']

        min_indices = ['VD', 'VI', 'MS', 'CVNN', 'XB', 'S_Dbw', 'DB', 'SD']

        idx_distance_matrix = ['XB', 'S_Dbw', 'DB', 'SD', 'PBM']

        indices = {'AR': ['AR'], 'FM': ['FM'], 'J': ['J'], 'AW': ['AW'], 'VD': ['VD'], 'H': ['H'],
                   'F': ['F'], 'K': ['K'], 'Phi': ['Phi'], 'RT': ['RT'], 'SS': ['SS'],
                   'VI': ['VI'], 'CVNN': ['CVNN'], 'XB': ['XB'],
                   'S_Dbw': ['S_Dbw'], 'DB': ['DB'], 'S': ['S'], 'SD': ['SD'],'PBM': ['PBM'], 'Dunn': ['Dunn'],
                   'all': external_indices + internal_indices, 'external': external_indices,
                   'internal': internal_indices
                   }


        if isinstance(index, str):
            index = [x.strip() for x in index.split(',')]

        for idx in index:
            if idx not in indices.keys():
                raise ValueError('{0} is not a valid index value, please check help(clusterval.Clusterval) to see acceptables indices'.format(
                    idx))

        self._external_indices = external_indices
        self._internal_indices = internal_indices
        self._min_indices = min_indices
        self._indices = indices
        self._idx_distance_matrix = idx_distance_matrix

        self.min_k = min_k
        self.max_k = max_k
        self.algorithm = algorithm
        self.bootstrap_samples = bootstrap_samples
        self.index = index

        columns = []
        for i in self.index:
            columns = columns + self._indices[i]
        self.output_df = pd.DataFrame(
            index=range(self.min_k, self.max_k + 1),
            columns=columns,
            dtype=np.float64
        )
        self.final_k = None
        self.count_choice = None
        self.final_clusters = None
        self.Z= None

        self.long_info = None

    def __repr__(self):
        args = ['{}={}'.format(' ' + key, value) for key,value in self.__dict__.items() if key not in ['output_df',
                                                                                                       '_external_indices',
                                                                                                       '_internal_indices',
                                                                                                       '_min_indices',
                                                                                                       '_indices',
                                                                                                       '_data',
                                                                                                       'final_k',
                                                                                                       'count_choice',
                                                                                                       'final_clusters',
                                                                                                       'long_info',
                                                                                                       'Z',
                                                                                                       '_idx_distance_matrix']]
        args = ','.join(args)
        return 'Clusterval(' + str(args) + ')\nfinal_k = {}'.format(self.final_k)

    def __str__(self):
        args = ['{} is {}'.format(' ' + key, value) for key, value in self.__dict__.items() if key not in ['output_df',
                                                                                                       '_external_indices',
                                                                                                       '_internal_indices',
                                                                                                       '_min_indices',
                                                                                                       '_indices',
                                                                                                       '_data',
                                                                                                       'final_k',
                                                                                                       'count_choice',
                                                                                                       'final_clusters',
                                                                                                        'long_info',
                                                                                                           'Z',
                                                                                                           '_idx_distance_matrix']]
        args = ';\n'.join(args)
        return 'Clusterval: \n' + str(args) + '\n'


    def _cluster_indices(self, cluster_assignments, idx):
        '''
        Transform cluster memebership array into array of clusters
        :param cluster_assignments: array
        :param idx: array with indices
        :return: array of clusters
        '''

        n = int(cluster_assignments.max())
        clusters = []
        for cluster_number in range(0, n + 1):
            aux = np.where(cluster_assignments == cluster_number)[0].tolist()
            if aux:
                cluster = list(idx[i] for i in aux)
                clusters.append(cluster)
        return clusters

    def _distance_dict(self, data):
        """
        Build dictionary that maps pairs to distance values
        :param data: ndarray m * (m - 1)) // 2 like returned by pdist
        :return: dictionary of tuples
        """
        from itertools import combinations
        from sympy import Symbol, solve

        diss_matrix = False
        #check if distances already calculated
        if data.ndim == 1:
            pairwise_distances = [i for i in data]

            x = Symbol('x')
            n = solve(x ** 2 - x - 2*data.shape[0])[1]

            diss_matrix = True

        else:
            pairwise_distances = pdist(data)
            n = len(data)

        comb = list(combinations([i for i in range(0, n)], 2))
        dist_dict = defaultdict(float)

        for pair, dist in zip(comb,pairwise_distances):
            dist_dict[pair] = dist

        return dist_dict, diss_matrix

    def _choose(self, choices_dict):

        for metrics in self.index:
            for metric in self._indices[metrics]:
                if metric in self.output_df.columns.values:
                    if metric in self._min_indices:
                        value = self.output_df.loc[self.output_df[metric].idxmin()]

                    else:
                        value = self.output_df.loc[self.output_df[metric].idxmax()]
                    choices_dict[metric] = [value.name, value[metric]]

        return choices_dict




    def evaluate(self, data):
        """
        Perform hierarchical clustering or Kmeans clustering on the data and calculate the validation indices
        :param data: array-like, n_samples x n_features or 1-d dissimilarity array
        Dataset to cluster
        :return: Clusterval object
        """

        columns = []
        for i in self.index:
            columns = columns + self._indices[i]
        self.output_df = pd.DataFrame(
            index=range(self.min_k, self.max_k + 1),
            columns=columns,
            dtype=np.float64
        )
        #convert data to numpy array
        data = np.array(data)
        #build dictionary with pairwise distances
        distances, diss_matrix = self._distance_dict(data)

        for metrics in self.index:
            for idx in self._indices[metrics]:
                if (diss_matrix) and (idx in self._idx_distance_matrix):
                    self.output_df = self.output_df.drop([idx], axis=1)

        clustering = defaultdict(dict)
        # dictionary with all mean values of the metrics for every k
        choices_dict = defaultdict(list)
        self.count_choice = defaultdict(int)
        results = {k: {} for k in range(self.min_k, self.max_k + 1)}

        # build dictionaries that hold the calculations

        for k in range(self.min_k, self.max_k + 1):
            for metrics in self.index:
                for idx in self._indices[metrics]:
                    if (diss_matrix) and (idx in self._idx_distance_matrix):
                        continue

                    else:
                        results[k][idx] = []
        if not any(self.output_df.columns.values):
            raise  ValueError('No CVIs being evaluated. If inputing distance matrix, please be aware that SD, XB, S_Dbw, PBM and DB are not possible to calculate.')

        if self.algorithm in ['ward', 'single', 'complete', 'average', 'centroid']:
            self.Z = linkage(data, method=self.algorithm)


        for k in range(self.min_k, self.max_k + 1):

            if self.algorithm in ['ward', 'single', 'complete', 'average', 'centroid']:
                # builds a list of the hierarchical clusters

                # with cut_tree
                partition = cut_tree(self.Z, n_clusters=k)
                clusters = self._cluster_indices(partition, [i for i in range(0, len(data))])
                clf = NearestCentroid()

                if not diss_matrix:
                    try:
                        clf.fit(data, list(itertools.chain.from_iterable(partition)))

                    #if only one cluster formed, try with fcluster - this happened with centroid
                    except ValueError as e:
                        if isinstance(e, ValueError):
                            partition = fcluster(self.Z, t=k, criterion='maxclust')
                            clusters = self._cluster_indices(partition, [i for i in range(0, len(data))])
                            clf.fit(data, list(itertools.chain(partition)))

                    centroids = clf.centroids_
                else:
                    centroids = None



            elif self.algorithm == 'kmeans':

                if diss_matrix:
                    raise ValueError('k-means does not work with only pairwise distances as input. Please set another algorithm.')

                partition = KMeans(n_clusters=k, random_state=0).fit(data)
                clusters = self._cluster_indices(partition.labels_, [i for i in range(0, len(data))])
                centroids = partition.cluster_centers_

            else:
                raise ValueError(self.algorithm + ' is not an acceptable clustering algorithm, please choose \'single\','
                                                  ' \'ward\', \'centroid\', \'complete\', \'average\' or \'kmeans\'')


            # dictionary of clustering of each 'k', to be used in internal validation
            clustering[k] = {'clusters': clusters, 'centroids': centroids}
            if (self.index[0] in ['all', 'external']) or (set(self.index).intersection(self._external_indices)):

                # external validation step
                for i in range(self.bootstrap_samples):
                    sample = random.sample(list(data), int(3 / 4 * len(data)))
                    #check if combinations are good in case of data == distance matrix
                    if diss_matrix:
                        NN = len(sample)
                        N = int(math.ceil(math.sqrt(NN*2)))
                        comb = (N*(N-1)//2)
                        if comb != NN:
                            sample = sample[:-(abs(comb - NN))]

                    if self.algorithm in ['ward', 'single', 'complete', 'average', 'centroid']:

                        Z_sample = linkage(sample, method=self.algorithm)

                        #clusters_sample = self._cluster_indices(fcluster(Z_sample, t=k, criterion='maxclust'), [i for i in range(0, len(sample))])

                        clusters_sample = self._cluster_indices(cut_tree(Z_sample, k), [i for i in range(0, len(sample))])

                    elif diss_matrix:
                        raise ValueError('k-means does not work with only pairwise distances as input. Please set another algorithm.')

                    else:
                        clusters_sample = self._cluster_indices(KMeans(n_clusters=k, random_state=0).fit(sample).labels_, [i for i in range(0, len(sample))])

                    external = calculate_external(clusters, clusters_sample, indices=self.index)

                    for key, metric in external.items():
                        if (key in self.index) or (self.index):
                            results[k][key].append(metric)

        if (self.index[0] in ['all', 'internal']) or (set(self.index).intersection(self._internal_indices)):
            # dictionary of distance between pairs of data points, e.g, {(pair1,pair2) : dist(pair1,pair2)}

            # internal validation step
            if diss_matrix:
                internal = calculate_internal(data, distances, clustering, indices=['CVNN','S','Dunn'])
            else:
                internal = calculate_internal(data, distances, clustering, indices=self.index)

            for cvi, k_clust in internal.items():
                for n_c, val in k_clust.items():
                    results[n_c][cvi] = val

        for k, keys in results.items():
            for key, value in keys.items():
                if key in self._external_indices:
                    self.output_df.loc[k, key] = mean(value)

                else:
                    self.output_df.loc[k, key] = value

        choices_dict = self._choose(choices_dict)
        for values in choices_dict.values():
            self.count_choice[str(values[0])] += 1

        max_value = 0
        for key, value in self.count_choice.items():
            if value > max_value:
                max_value = value
                final_k = key
        self.final_k = int(final_k)

        if self.algorithm in ['ward', 'single', 'complete', 'average', 'centroid']:
            #self.final_clusters = fcluster(self.Z, t=self.final_k, criterion='maxclust')
            self.final_clusters = np.concatenate(np.asarray(cut_tree(self.Z, self.final_k)))
        else:
            self.final_clusters = KMeans(n_clusters=self.final_k, random_state=0).fit(data).labels_
        self.long_info = self.print_results()

        return self

    def print_results(self):
        '''
        print detailed information on the clustering results conclusion
        :return: str with the results
        '''

        output_str = '* Minimum number of clusters to test: ' + str(self.min_k) + '\n'
        output_str += '* Maximum number of clusters to test: ' + str(self.max_k) + '\n'
        output_str +='* Number of bootstrap samples generated: ' + str(self.bootstrap_samples) + '\n'
        output_str += '* Clustering algorithm used: ' + str(self.algorithm) + '\n'

        if self.index not in ['all', 'internal', 'external']:
            idx = list(self.output_df.columns.values)
        else:
            idx = self.index
        output_str +='\n* Validation Indices calculated: ' + str(idx) + '\n\n'


        output_str +="* Among all indices: \n\n"

        output_str +='\n\n* According to the majority rule, the best number of clusters is ' + str(self.final_k) + '\n\n\n'

        for k_num in sorted(self.count_choice):
            output_str +='* ' + str(self.count_choice[k_num]) + ' proposed ' + k_num + ' as the best number of clusters \n\n'

        output_str +='\t\t\t***** Conclusion *****\t\t\t\n'

        #Display all dataframe
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        output_str +=str(self.output_df) + '\n'

        output_str +='\n* The best partition is:\n'
        output_str +=str(self.final_clusters) + '\n'

        return output_str

    def _calculate_dendrogram(self, labels=None):
        '''
        Calculate dendrogram for object linkage product
        :param labels: array-like, shape 1 x len(data.shape[0])
        :return: dendrogram
        '''

        dend = dendrogram(

            self.Z,
            # truncate_mode = 'lastp',
            # p=6,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=10,  # font size for the x axis labels
            labels=labels
        )

        return dend

    def plot(self, labels=None):
        '''
        print the hierarchical clustering dendrogram using matplotlib
        :return: matplotlib image of dendrogram
        '''

        fig = plt.figure(figsize=(40, 20))
        plt.title('Hierarchical Clustering Dendrogram - Linkage: %s, Metrics: %s' % (self.algorithm, str(self.index)),
                  fontsize=30)
        plt.xlabel('data point index', labelpad=20, fontsize=30)
        plt.ylabel('distance', labelpad=10, fontsize=30)
        plt.xticks(size=40)
        plt.yticks(size=40)
        self._calculate_dendrogram(labels)

        plt.show()

