# -*- coding: utf-8 -*-

from tests.context import clusterval
import pytest
import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage


iris = load_iris()
data, _ = make_blobs(n_samples=500, centers=4, n_features=5, random_state=0)

c = clusterval.Clusterval(min_k=2, max_k=8, index='S_Dbw,S')
c.evaluate(data)

def test_basic_run():

    aclust = AgglomerativeClustering(n_clusters=4)
    y = aclust.fit_predict(data)
    clusters = c._cluster_indices(y, [i for i in range(0, len(data))])
    clustering = {4: clusters}
    r = clusterval.calculate_internal(c._distance_dict(data), clustering, indices='S_Dbw, S')

    assert r['S_Dbw'][4] == c.output_df.loc[4, 'S_Dbw']
    assert r['S'][4] == c.output_df.loc[4, 'S']


def test_index_input():
    c1 = clusterval.Clusterval(index='CVNN')
    c2 = clusterval.Clusterval(index=['internal'])
    c3 = clusterval.Clusterval(index=['J', 'AW', 'S'])
    with pytest.raises(ValueError) as excinfo:
        clusterval.Clusterval(index=['MM','SD'])
    exception_msg = excinfo.value.args[0]

    assert c1.index == ['CVNN']
    assert c2.index == ['internal']
    assert c3.index == ['J', 'AW', 'S']
    assert exception_msg == 'MM is not a valid index value, please check help(clusterval.Clusterval) to see acceptables indices'

def test_method_mix():
    c4 = clusterval.Clusterval(index=['J', 'SD'])
    assert list(c4.output_df.columns) == ['J', 'SD']

def test_distances_pairs():
    c5 = clusterval.Clusterval()
    distance = c5._distance_dict(data)
    for pair in distance:
        opposite_pair = (pair[1], pair[0])
        assert opposite_pair not in distance.keys()

def test_dendrogram_plotting():
    c.Z = linkage(iris['data'], c.link)
    labels = np.random.rand(1, iris['data'].shape[0])[0]
    c.dendrogram = c._calculate_dendrogram(labels)
    assert len(labels) == len(c.dendrogram['ivl'])



