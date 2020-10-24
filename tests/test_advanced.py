# -*- coding: utf-8 -*-
from context import clusterval
from sklearn.datasets import load_iris

data = load_iris()['data']


def test_basic_run():
    val = clusterval.evaluate(list(data)
    assert val
