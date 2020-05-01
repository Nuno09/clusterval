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
        pass

    elif isinstance(data, dict):
        pass

    elif isinstance(data, pd.core.frame.DataFrame):
        fields = data.keys()
        dict_aux = data.to_dict('index')
        for key,value in dict_aux.items():
            v1 = value[fields[0]]
            v2 = value[fields[1]]
            score = value[fields[2]]
            distance_dict[(v1, v2)] = score


    elif data[-3:] == 'csv':
        with open(data, "r") as file:
            r = csv.DictReader(file) #just a reader that can map information of each row into a dict
            fields = r.fieldnames
            for row in r:
                v1 = row[fields[0]]
                v2 = row[fields[1]]
                score = row[fields[2]]
                distance_dict[(v1, v2)] = score

    metric = cvnn(clusters, distance_dict, k)
    #### PUT I WORKING IN ALICLU ###############
    return metric


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
        if int(pair[0]) > int(pair[1]):
            pair = (pair[1], pair[0])
        sum_of_distances += float(data[pair])

    return sum_of_distances


if __name__ == '__main__':
    from sklearn.utils import resample
    from collections import defaultdict
    from statistics import mean
    from tabulate import tabulate
    import math
    import pandas as pd
    import csv

    dicio_statistics = defaultdict(str)
    indexes = ('cvnn')
    for index in indexes:
        dicio_statistics[index] = []

    file = '/home/nuno/Documentos/IST/Tese/AliClu/Code/dataframe.csv'
    k = 2
    clusters = [['0', '1', '2', '3', '17', '5', '6', '7', '8', '9', '11', '14', '18', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
               ['10', '12', '13', '15', '16', '4', '19']]

    with open(file, 'r') as csv_file:
        lst = []
        r = csv.reader(csv_file, delimiter=',')
        for row in r:
            lst.append(row)

    df = pd.DataFrame(lst, columns=['patient1', 'patient2', 'score'])

    print(calculate_internal(file, clusters, k))


    # print(computed_indexes)
    #for index in indexes:
        #dicio_statistics[index].append(calculate_internal(file, clusters, k))

    #print(tabulate([list(dicio_statistics.values())], headers=list(dicio_statistics.keys())))
