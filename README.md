# clusterval

For validating clustering results

## Motivation

This package was made to facilitate the process of clustering of a dataset. The user needs only to specifiy the dataset or the pairwise distances
in the form of a list-like structure and clustering will be performed as well as evaluation of the results, through the 
use of CVIs (Clustering Validation Indices). `clusterval` outputs the best partition of the data, a dendrogram and `k`,
the number of clusters. Clustering algorithms available are: 'single', 'complete', 'ward', 'centroid', 'average' and 'kmeans.


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
from clusterval import Clusterval
from sklearn.datasets import load_iris, make_blobs

```
<span>2.</span> Let's use the iris dataset

```python
data = load_iris()['data']
```

<span>3.</span> Use `clusterval` to determine the optimal number of clusters. The code below will create a Clusterval 
object that for an input dataset will partition the data 
into 2-8 clusters using hierarchical aglomerative clustering, with ward criteria, then calculate various CVIs across the 
results.

```python
c = Clusterval()
c.evaluate(data)

Outupt:
    Clusterval(min_k=2, max_k=8, algorithm=ward, bootstrap_samples=250, index=['all'])
    final_k = 2

```
<span>4.</span> If user wishes more information on the resulting clustering just type below command. 

```python
print(c.long_info)

Long output:

* Minimum number of clusters to test: 2
* Maximum number of clusters to test: 8
* Number of bootstrap samples generated: 250
* Clustering algorithm used: ward

* Validation Indices calculated: ['AR', 'FM', 'J', 'AW', 'VD', 'H', 'F', 'VI', 'K', 'Phi', 'RT', 'SS', 'CVNN', 'XB', 'SDbw', 'DB', 'S', 'SD', 'PBM', 'Dunn']

* Among all indices: 



* According to the majority rule, the best number of clusters is 2


* 15 proposed 2 as the best number of clusters 

* 3 proposed 3 as the best number of clusters 

* 1 proposed 6 as the best number of clusters 

* 1 proposed 8 as the best number of clusters 

			***** Conclusion *****			
         AR        FM         J        AW        VD         H         F        VI         K           Phi        RT        SS      CVNN        XB      SDbw        DB         S         SD        PBM  \
2  0.999426  0.526738  0.356955  0.000331  0.385982  0.000283  0.526045  1.312383  0.527433  2.974318e-06  0.333611  0.217291  1.000000  0.261539  0.752460  0.191376  0.722234  10.714331  20.485220   
3  1.000172  0.379244  0.233104 -0.000791  0.525643 -0.000891  0.378001  1.972355  0.380492 -9.809481e-06  0.356957  0.131954  0.974122  0.414386  0.791627  0.388819  0.560392  10.764615  25.473608   
4  1.000080  0.285500  0.166032 -0.000607  0.586732 -0.000677  0.284666  2.511133  0.286337 -8.672063e-06  0.418338  0.090562  1.233481  0.610347  0.798949  0.448981  0.458409  12.500304  19.111932   
5  1.000070  0.263834  0.150461 -0.000815  0.591750 -0.000972  0.261470  2.676828  0.266223 -1.296032e-05  0.434558  0.081374  1.552875  0.578900  0.699964  0.527533  0.436093  15.188085  18.932200   
6  1.000042  0.192872  0.106553 -0.000079  0.675911 -0.000065  0.192512  3.096166  0.193233 -8.572647e-07  0.524317  0.056289  1.518029  0.767859  0.695073  0.580975  0.367109  18.297956  16.657177   
7  1.000036  0.170423  0.092730 -0.001502  0.687232 -0.001626  0.169647  3.246090  0.171204 -2.903045e-05  0.554676  0.048633  1.479109  0.884422  0.743185  0.737307  0.335528  21.942174  13.619411   
8  1.000032  0.155010  0.083478 -0.000446  0.694161 -0.000496  0.154014  3.349153  0.156014 -9.595088e-06  0.581493  0.043571  1.455532  0.884422  0.701268  0.681691  0.370146  22.229196  11.636157   

       Dunn  
2  0.338909  
3  0.112795  
4  0.123508  
5  0.123508  
6  0.131081  
7  0.131081  
8  0.150756  

* The best partition is:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1]



```

<span>4.</span> It's also possible to visualize the hierarchical clustering.
Note: in linux systems installation of library "python3-tk" might be needed.

```python
c.plot()

```

<span>5.</span> The user can also change some execution parameters. For example, clustering algorithm, range of `k` to test,
bootstrap simulations and CVI to use.

```python
data, _ = make_blobs(n_samples=700, centers=10, n_features=5, random_state=0)
c = Clusterval(min_k=5, max_k=15, algorithm='kmeans', bootstrap_samples=200, index='CVNN')
c.evaluate(data)

Output:
    Clusterval(min_k=5, max_k=15, algorithm=kmeans, bootstrap_samples=200, index=['CVNN'])
    final_k = 10

```




