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
clusterval = Clusterval()
eval = clusterval.evaluate(data, algorithm='hierarchical')
print(eval.final_k)

Outupt:
    2

```
<span>4.</span> If user wishes more information on the resulting clustering just type below command. 
```python
print(eval.long_info)

Long output:

* Linkage criteria is: ward
* Minimum number of clusters to test: 2
* Maximum number of clusters to test: 8
* Number of bootstrap samples generated: 250
* Clustering algorithm used: hierarchical

* Validation Indices calculated: ['all']

* Among all indices: 



* According to the majority rule, the best number of clusters is 2



* 9 proposed 2 as the best number of clusters 

* 1 proposed 3 as the best number of clusters 

* 1 proposed 4 as the best number of clusters 

* 1 proposed 6 as the best number of clusters 

* 3 proposed 7 as the best number of clusters 

* 2 proposed 8 as the best number of clusters 

                        ***** Conclusion *****                  
          R        AR        FM         J        AW        VD         H        H'         F        VI        MS      CVNN           XB       S_Dbw         DB         S          SD
2  0.499887 -0.000501  0.526324  0.356564 -0.000561  0.386554 -0.000227 -0.000504  0.525636  1.894603  1.954966  1.000000  5.080366e+02    1.897584   48.132701  0.718247  134.967568
3  0.526773  0.000608  0.380229  0.233861  0.000535  0.523875  0.053546  0.000612  0.378990  2.844118  1.921184  0.302994  9.241778e+03    6.908961  253.595738  0.571105   72.912823
4  0.591175 -0.000077  0.284451  0.165264 -0.000064  0.588232  0.182350 -0.000077  0.283545  3.633834  1.766034  0.248039  4.697564e+04   10.728314  394.716423  0.461579  147.191587
5  0.606510  0.000315  0.264490  0.150870  0.000240  0.591286  0.213019  0.000328  0.262078  3.859444  1.819040  0.255457  9.814790e+04   28.080596  460.129162  0.438943  117.491911
6  0.687755 -0.000125  0.192927  0.106590 -0.000173  0.676982  0.375510 -0.000123  0.192562  4.468536  1.631676  0.279451  9.814790e+04   40.387317  236.150343  0.367344  117.080171
7  0.714658  0.001512  0.172814  0.094147  0.001332  0.683714  0.429317  0.001527  0.172012  4.667950  1.648638  0.458824  3.653942e+05   41.579133  462.896728  0.339015  426.822086
8  0.735017 -0.000115  0.155692  0.083893 -0.000092  0.691946  0.470035 -0.000117  0.154729  4.819567  1.648585  1.032990  1.320844e+06  136.399093  360.873310  0.372359  477.025711

* The best partition is:
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]



```
<span>4.</span> The user can also change some execution parameters. For example, linkage criteria, range of `k` to test,
bootstrap simulations and CVI to use.
When evaluating one can also choose kmeans algorithm instead of hierachical.

```python
data, _ = make_blobs(n_samples=700, centers=10, n_features=5, random_state=0)
clusterval = Clusterval(min_k=5, max_k=15, link='single', bootstrap_samples=200, index='CVNN')
eval = clusterval.evaluate(data, algorithm='kmeans')
print(eval.final_k)

Output:
    10

```
<span>5.</span> It's also possible to visualize the hierarchical clustering.
Note: in linux systems installation of library "python3-tk" might be needed.

```python
eval.plot()

```



