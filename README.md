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
2  0.999434  0.527996  0.358020  0.002798  0.384946  0.002484  0.527220  0.566928  0.528773  2.587677e-10  0.334478  0.218069  1.000000  0.261539  0.752460  0.191376  0.722234  10.714331  20.485220   
3  1.000212  0.334907  0.196145 -0.117986  0.525982 -0.169768  0.327915  0.842555  0.342052 -1.846363e-08  0.258682  0.108751  0.974122  0.414386  0.791627  0.388819  0.560392  10.764615  25.473608   
4  1.000135  0.205800  0.105216 -0.174317  0.585786 -0.347196  0.190330  1.039881  0.222574 -4.141532e-08  0.180963  0.055543  1.233481  0.610347  0.798949  0.448981  0.458409  12.500304  19.111932   
5  1.000124  0.187061  0.091395 -0.152053  0.590107 -0.342123  0.167430  1.041136  0.209033 -4.292699e-08  0.177830  0.047895  1.552875  0.578900  0.699964  0.527533  0.436093  15.188085  18.932200   
6  1.000106  0.104764  0.044471 -0.156233  0.675554 -0.492585  0.085139  1.093407  0.128947 -7.145318e-08  0.113579  0.022744  1.518029  0.767859  0.695073  0.580975  0.367109  18.297956  16.657177   
7  1.000102  0.087725  0.034876 -0.137154  0.686821 -0.502020  0.067385  0.974156  0.114281 -7.921781e-08  0.103693  0.017750  1.479109  0.884422  0.743185  0.737307  0.335528  21.942174  13.619411   
8  1.000099  0.075781  0.028551 -0.123934  0.692946 -0.512168  0.055506  0.738652  0.103503 -8.674453e-08  0.095137  0.014484  1.455532  0.884422  0.701268  0.681691  0.370146  22.229196  11.636157   

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




