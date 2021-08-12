# Unsupervised-Learning

Unsupervised Learning Implementation for DBScans, KMeans, and KMedoids.

### Description

This project contains implementation of `Unsupervised Learning` algorithms from scratch in Python. It consists of many different algorithm, each with their own advantages and disadvantages, comparatively to others.

Unsupervised Learning can be used for both `clustering` and `association`. Among all algorithms, there are three algorithms used for the `clustering` problem. They are `DBScan`, `KMeans`, and `KMedoids`.

`DBScan`, or `Density-Based Spatial Clustering of Applications with Noise` is an algorithm that clusters data by expanding each data points and gather as much neighbors as possible within the range of `Epsilon`. If the data points contained in the group exceed the number of `Minimum Points`, then it can be clustered together.

`KMeans` is another algorithm used to primarily cluster continuous data. Unlike `DBScan`, the user of this algorithm will determine the number of clusters with the parameter `k`. The model will then randomly pick `k` number of data as `sentroids`. Each data is clustered to the closest `sentroid`, and the `sentroid` will update itself with the `mean` value of all data that it currently contains, until the algorithm converges.

`KMedoids` is an algorithm similar to `KMeans`. The difference between the two is that `KMedoids` uses concrete data from the dataset instead of taking average value of the data in a cluster like `KMeans`. The `medoids` will be updated everytime the algorithm found a more suitable data in the dataset that produces better (lower) error, until the algorithm converges.

The dataset used for this project is the `Mall Customer` dataset that can be found [here](https://www.kaggle.com/roshansharma/mall-customers-clustering-analysis). It contains both numerical and categorical features, and does not contain any labels in it.

### Guide

-   Go to the home directory of the project through `Terminal` or `Command Prompt` with the `cd <Unsupervised-Learning_directory>`. Don't get too deep to the `src` folder!
-   Enter the command needed to run an algorithm. The example of the command will be given below.
    -   `DBScan`: `python src\main.py --dbscan --data data\dataset.csv --eps 10 --min_pts 3`
    -   `KMeans`: `python src\main.py --kmeans --data data\dataset.csv --k_cen 5`
    -   `KMedoids`: `python src\main.py --kmedoids --data data\dataset.csv --k_med 3`
-   Keep in mind that the first argument (aside from file name) should be the type of the algorithm!
-   All other values can be modified.
-   If the data wants to be changed, put the data in the `data` folder. One thing to note is the preprocessing part of the data needs to be changed too. The corresponding file can be found in `src/preprocess_un.py`.
-   The model will automatically fit the data. After it's done, the model will pick a generate random data based on the dataset and cluster the data.
-   The main output of the prediction will be: `Clusters and Data Counts` or `Sentroids` and `Medoids`, `Input Data`, and `Input Data Cluster`.

### Creator

-   Richard Rivaldo / 13519185

### Possible Improvements

-   Better accuracy
-   Better design of algorithm
-   Clusters Visualization

### References

-   DBScan

    -   [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
    -   [DBSCAN Clustering (from scratch)](https://medium.com/@darkprogrammerpb/dbscan-clustering-from-scratch-199c0d8e8da1)
    -   [DBSCAN From Scratch (Almost)](https://medium.com/analytics-vidhya/dbscan-from-scratch-almost-b02096600c14)
    -   [DBSCAN with Python](https://towardsdatascience.com/dbscan-with-python-743162371dca)
    -   [Implementing DBSCAN Clustering from scratch in Python](http://madhugnadig.com/articles/machine-learning/2017/09/13/implementing-dbscan-from-scratch-in-python-machine-learning.html)
    -   [ML from Scratch](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/dbscan.py)
    -   [Scikit-Learn: Predicting new points with DBSCAN](https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan)
    -   [Understanding DBSCAN Algorithm and Implementation from Scratch](https://towardsdatascience.com/understanding-dbscan-algorithm-and-implementation-from-scratch-c256289479c5)

-   KMeans

    -   [Build K-Means from scratch in Python](https://dev.to/rishitdagli/build-k-means-from-scratch-in-python-2140)
    -   [Clustering categorical data](https://datascience.stackexchange.com/questions/13273/clustering-categorical-data)
    -   [Develop a K Mean Clustering Algorithm from Scratch in Python and Use It for Dimensional Reduction](https://regenerativetoday.com/develop-a-k-mean-clustering-algorithm-from-scratch-in-python-and-use-it-for-dimensional-reduction/)
    -   [How to program the kmeans algorithm in Python from scratch](https://anderfernandez.com/en/blog/kmeans-algorithm-python/)
    -   [Implementing K-means Clustering from Scratch - in Python](https://mmuratarat.github.io/2019-07-23/kmeans_from_scratch)
    -   [K-Means Clustering](https://dev.to/akhildraju/k-means-clustering-45ph)
    -   [K-Means Clustering Algorithm from Scratch](https://www.machinelearningplus.com/predictive-modeling/k-means-clustering/)
    -   [K-Means Clustering From Scratch in Python [Algorithm Explained]](https://www.askpython.com/python/examples/k-means-clustering-from-scratch)
    -   [K-Means from Scratch in Python](https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/)

-   KMedoids

    -   [K-medoids Clustering](https://iq.opengenus.org/k-medoids-clustering/)
    -   [K-Medoids Clustering on Iris Data Set](https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05)
    -   [ML | K-Medoids clustering with solved example](https://www.geeksforgeeks.org/ml-k-medoids-clustering-with-example/)
    -   [K Medoids PAM with Python](https://www.youtube.com/watch?v=L1ykPtlonAU)
    -   [K_Medoids Shen Xu Deu](https://github.com/shenxudeu/K_Medoids/blob/master/k_medoids.py)
    -   [Partitioning_Around_Medoids ML From Scratch](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/partitioning_around_medoids.py)
    -   [#SuperNaive: K-medoids clustering](https://medium.com/analytics-vidhya/supernaive-k-medoids-clustering-31db7bfc5075)
