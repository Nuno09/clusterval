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

* Validation Indices calculated: ['AR', 'FM', 'J', 'AW', 'VD', 'H', 'F', 'VI', 'K', 'Phi', 'RT', 'SS', 'CVNN', 'XB', 'S_Dbw', 'DB', 'S', 'SD', 'PBM', 'Dunn']

* Among all indices: 



* According to the majority rule, the best number of clusters is 2


* 16 proposed 2 as the best number of clusters 

* 3 proposed 3 as the best number of clusters 

* 1 proposed 6 as the best number of clusters 

			***** Conclusion *****			
         AR        FM         J        AW        VD         H         F        VI         K           Phi        RT        SS      CVNN        XB     S_Dbw        DB         S         SD        PBM  \
2  0.999418  0.527024  0.357176  0.002762  4847.136  0.002467  0.526306  0.567622  0.527744  2.569224e-10  0.334478  0.217442  1.000000  0.261539  0.752460  0.191376  0.722234  10.714331  20.485220   
3  1.000214  0.335324  0.196524 -0.118590  6597.472 -0.169992  0.328440  0.842813  0.342358 -1.847535e-08  0.258682  0.108985  0.974122  0.414386  0.791627  0.388819  0.560392  10.764615  25.473608   
4  1.000134  0.204772  0.104434 -0.172468  7372.736 -0.346519  0.189049  1.042504  0.221836 -4.145184e-08  0.180963  0.055107  1.233481  0.610347  0.798949  0.448981  0.458409  12.500304  19.111932   
5  1.000124  0.186969  0.091307 -0.151693  7400.512 -0.341933  0.167287  1.038021  0.208999 -4.292749e-08  0.177830  0.047846  1.552875  0.578900  0.699964  0.527533  0.436093  15.188085  18.932200   
6  1.000107  0.105998  0.045157 -0.157418  8437.856 -0.493076  0.086388  1.080764  0.130108 -7.128094e-08  0.113579  0.023104  1.518029  0.767859  0.695073  0.580975  0.367109  18.297956  16.657177   
7  1.000102  0.087835  0.034890 -0.136587  8622.656 -0.501461  0.067409  0.958526  0.114513 -7.923592e-08  0.103693  0.017757  1.479109  0.884422  0.743185  0.737307  0.335528  21.942174  13.619411   
8  1.000098  0.074925  0.028118 -0.122833  8714.944 -0.511240  0.054689  0.768727  0.102685 -8.697985e-08  0.095137  0.014261  1.455532  0.884422  0.701268  0.681691  0.370146  22.229196  11.636157   

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




