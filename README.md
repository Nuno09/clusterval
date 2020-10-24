# clusterval

For validating clustering results

## Motivation

This package was made to facilitate the process of clustering of a dataset. The user needs only to specifiy the dataset
in the form of a list and hierarchical clustering will be performed as well as evaluation of the results, through the 
use of CVIs (Clustering Validation Indices). `clusterval` outputs the best partition of the data, a dendrogram and `k`,
the number of clusters. 


## Installation

You can get the stable version from PyPI:

```
pip install clusterval
```

Or the development version from GitHub:

```
pip install git+https://github.com/Nuno09/clusterval.git
```

## Basic usage

<span>1.</span> Load libraries.

```python
from clusterval import evaluate
from sklearn.datasets import make_blobs

```
<span>2.</span> Create some synthetic data. The data will be clustered around 4 centers.

```python
data, _ = make_blobs(n_samples=500, centers=4, n_features=5, random_state=0)
```

<span>3.</span> Use `clusterval` to determine the optimal number of clusters. The code below will partition the data 
into 2-8 clusters using hierarchical aglomerative clustering, with ward criteria, then calculate various CVIs across the 
results.

```python
evaluate(data)


```
<span>4.</span> The user can also change some execution parameters. For example, linkage criteria, range of `k` to test,
bootstrap simulations and CVI to use.

```python
data, _ = make_blobs(n_samples=700, centers=10, n_features=5, random_state=0)
evaluate(data, min_k=5, max_k=15, link='single', M=200, index='CVNN')
```



