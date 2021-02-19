"""
Function evaluate does hierarchical clustering and validation of it's results.
Validation can be done with external or internal indices, or both.


"""


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import random
from statistics import mean
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from clusterval.internal import calculate_internal
from clusterval.external import calculate_external


class Clusterval:
    """
    min_k: int, optional
    minimum number of clusters to test. Default 2.

    max_k: int, optional
    maximum number of clusters to test. Default 8.

    link: str, optional
    linkage method to use. Accepts 'single', 'ward'(default), 'complete', 'centroid', 'average'

    bootstrap_samples: int, optional
    number of bootstrap samples simulated. Default 250

    index: list of str or str, optional
    what indices to be calculated. Accepts 'all' to calculate all, or one of [R', 'AR', 'FM', 'J', 'AW', 'VD', 'H',
           'H\'', 'F', 'VI', 'MS', 'CVNN', 'XB**', 'S_Dbw', 'DB*', 'S', 'SD']. Default 'all'.
    """

    def __init__(self, min_k=2, max_k=8, link='ward', bootstrap_samples=250, index='all'):

        external_indices = ['R', 'AR', 'FM', 'J', 'AW', 'VD', 'H', 'H\'', 'F', 'VI', 'MS', 'CD', 'K', 'McNemar', 'Phi', 'RT']
        internal_indices = ['CVNN', 'XB*', 'S_Dbw', 'DB*', 'S', 'SD', 'PBM', 'Dunn']
        min_indices = ['VD', 'VI', 'MS', 'CVNN', 'XB*', 'S_Dbw', 'DB*', 'SD']
        indices = {'R': ['R'], 'AR': ['AR'], 'FM': ['FM'], 'J': ['J'], 'AW': ['AW'],
                   'VD': ['VD'], 'H': ['H'], 'H\'': ['H\''], 'F': ['F'],
                   'VI': ['VI'], 'MS': ['MS'], 'CVNN': ['CVNN'], 'XB*': ['XB*'],
                   'S_Dbw': ['S_Dbw'], 'DB*': ['DB*'], 'S': ['S'], 'SD': ['SD'],
                   'all': external_indices + internal_indices, 'external': external_indices,
                   'internal': internal_indices, 'CD': ['CD'], 'K': ['K'], 'McNemar': ['McNemar'], 'Phi': ['Phi'],
                   'RT': ['RT'], 'PBM': ['PBM'], 'Dunn': ['Dunn']}


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

        self.min_k = min_k
        self.max_k = max_k
        self.link = link
        self.bootstrap_samples = bootstrap_samples
        self.index = index

        columns = [self._indices[i][0] for i in self.index]

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
                                                                                                       'Z']]
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
                                                                                                           'Z']]
        args = ';\n'.join(args)
        return 'Clusterval: \n' + str(args) + '\n'


    def _cluster_indices(self, cluster_assignments, idx):
        '''
        Transform cluster memebership array into array of clusters
        :param cluster_assignments: array
        :param idx: array with indices
        :return: array of clusters
        '''
        import numpy as np
        n = cluster_assignments.max()
        clusters = []
        for cluster_number in range(0, n + 1):
            aux = np.where(cluster_assignments == cluster_number)[0].tolist()
            if aux:
                cluster = list(idx[i] for i in aux)
                clusters.append(cluster)
        return clusters

    def _distance_dict(self, data):
        """
        Calculate the accumulative distance considering all features, between each pair of observations
        :param data: list
        :return: dictionary of tuples
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

    def _choose(self, choices_dict):

        for metrics in self.index:
            for metric in self._indices[metrics]:
                if metric in self._min_indices:
                    value = self.output_df.loc[self.output_df[metric].idxmin()]

                else:
                    value = self.output_df.loc[self.output_df[metric].idxmax()]
                choices_dict[metric] = [value.name, value[metric]]

        return choices_dict

    def evaluate(self, data, labels=None):
        """
        Perform hierarchical clustering on the dataset and calculate the validation indices
        :param data: array-like, n_samples x n_features
        Dataset to cluster
        :return: Clusterval object
        """
        clustering = defaultdict(dict)
        # dictionary with all mean values of the metrics for every k
        choices_dict = defaultdict(list)
        self.count_choice = defaultdict(int)

        results = {k: {} for k in range(self.min_k, self.max_k + 1)}
        # build dictionaries that hold the calculations
        for k in range(self.min_k, self.max_k + 1):
            for metrics in self.index:
                for index in self._indices[metrics]:
                    results[k][index] = []


        self.Z = linkage(data, self.link)

        for k in range(self.min_k, self.max_k + 1):

            # builds a list of the clusters
            clusters = self._cluster_indices(fcluster(self.Z, t=k, criterion='maxclust'), [i for i in range(0, len(data))])
            # dictionary of clustering of each 'k', to be used in internal validation
            clustering[k] = clusters
            if (self.index[0] in ['all', 'external']) or (set(self.index).intersection(self._external_indices)):

                # external validation step
                for i in range(self.bootstrap_samples):
                    sample = random.sample(list(data), int(3 / 4 * len(data)))

                    Z_sample = linkage(sample, self.link)

                    clusters_sample = self._cluster_indices(fcluster(Z_sample, t=k, criterion='maxclust'),
                                                      [i for i in range(0, len(sample))])
                    external = calculate_external(clusters, clusters_sample, indices=self.index)

                    for key, metric in external.items():
                        if (key in self.index) or (self.index):
                            results[k][key].append(metric)

        if (self.index[0] in ['all', 'internal']) or (set(self.index).intersection(self._internal_indices)):
            # dictionary of distance between pairs of data points, e.g, {(pair1,pair2) : dist(pair1,pair2)}
            distances = self._distance_dict(data)
            # internal validation step
            internal = calculate_internal(distances, clustering, indices=self.index)
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
        self.final_k = final_k
        self.final_clusters = fcluster(self.Z, t=final_k, criterion='maxclust')
        self.long_info = self.print_results()


        return self

    def print_results(self):
        '''
        print detailed information on the clustering results conclusion
        :return: str with the results
        '''
        output_str = '\n* Linkage criteria is: ' + self.link + '\n'
        output_str += '* Minimum number of clusters to test: ' + str(self.min_k) + '\n'
        output_str += '* Maximum number of clusters to test: ' + str(self.max_k) + '\n'
        output_str +='* Number of bootstrap samples generated: ' + str(self.bootstrap_samples) + '\n'

        output_str +='\n* Validation Indices calculated: ' + str(self.index) + '\n\n'

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
            leaf_font_size=10.,  # font size for the x axis labels
            labels=labels
        )

        return dend

    def plot(self, labels=None):
        '''
        print the hierarchical clustering dendrogram using matplotlib
        :return: matplotlib image of dendrogram
        '''

        fig = plt.figure(figsize=(40, 20))
        plt.title('Hierarchical Clustering Dendrogram - Linkage: %s, Metrics: %s' % (self.link, str(self.index)),
                  fontsize=30)
        plt.xlabel('data point index', labelpad=20, fontsize=30)
        plt.ylabel('distance', labelpad=10, fontsize=30)
        plt.xticks(size=40)
        plt.yticks(size=40)
        self._calculate_dendrogram(labels)

        plt.show()


if __name__=='__main__':
    from sklearn.datasets import load_iris, make_blobs
    import re
    c = Clusterval()
    data = load_iris()['data']
    blobs, _ = make_blobs(n_samples=500, centers=4, n_features=5, random_state=0)

    synthetic_dim2_9 = []
    pattern = re.compile(r'^\s+')
    with open('/home/nuno/Documentos/Datasets/data_dim_k=9_txt/dim2.txt', 'r') as dim2:
        for line in dim2:
            re_new = re.sub(pattern, '', line)
            new = ''
            for i, el in enumerate(re_new[:-1]):
                if (el != ' ') and (re_new[i+1] == ' '):
                    new = new + el + ','
                elif el != ' ':
                    new += el

            new = new.split(',')
            new = [float(item) for item in new]
            synthetic_dim2_9.append(new)

