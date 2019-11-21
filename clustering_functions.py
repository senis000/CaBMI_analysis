import numpy as np
import h5py
import time
import pdb
import community
#from networkx.algorithms import community
import networkx as nx
import markov_clustering as mc
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac
from utils_clustering import *

def agglomerative_clustering_elbow_plot(mat):
    """
    Generates an elbow plot to help select the number of clusters for
    agglomerative clustering. The metric of choice is normalized
    cross-correlation.
    Inputs:
        MAT: The (neurons x frames) calcium activity matrix
    """

    z = hac.linkage(mat, method='average', metric=normalized_cc)
    plt.figure()
    plt.plot(range(1, len(z)+1), z[::-1,2])
    plt.xlabel('k')
    plt.ylabel('Cluster distance')
    plt.title('Elbow Plot of Calcium Activity with Agglomerative Clustering')
    plt.show(block=True)

def agglomerative_clustering(mat):
    """
    Runs agglomerative clustering with time-shifted Pearson correlation as the
    metric.
    Inputs:
        MAT: The (neurons x frames) calcium activity matrix
    Output:
        CLUSTERING: A sklearn AgglomerativeClustering object
    """

    clustering = AgglomerativeClustering(
        affinity=normalized_cc_mat, linkage='average', n_clusters=2
        ).fit(mat)
    return clustering

def community_louvain(distance_mat):
    """
    Runs the Louvain community detection algorithm on the input distance matrix.
    Inputs:
        DISTANCE_MAT: A (neurons x neurons) numpy matrix calculated by some
            distance metric.
    Output:
        PARTITION: A dictionary where the keys are zero-indexed, numbered
            communities, and the values are the array of vertices belonging in
            the community.
    """

    G = nx.from_numpy_matrix(distance_mat)
    partition = community.best_partition(G)
    return partition

def markov_clustering(distance_mat, inflation):
    """
    Runs the Markov Clustering algorithm on the input distance matrix.
    Inputs:
        DISTANCE_MAT: A (neurons x neurons) numpy matrix calculated by some
            distance metric.
        INFLATION: An int; the Hadamarde power to take during the inflation step.
            In general, values from 1.1 to 10.0 can be tried, with higher
            values generally resulting in more clusters. Inflation boosts the
            probabilities of intra-cluster walks and demotes inter-cluster walks.
    Outputs:
        CLUSTERS: A (neurons x neurons) numpy matrix of the final remaining
            clusters.
        Q: A float between [-1,1]; the modularity score associated with this
            clustering. Modularity measures the density of in-cluster edges
            to out-of-cluster edges. Specifically, it is the fraction of edges
            that fall within the clusters minus the expected fraction if edges
            were randomly distributed.
    """

    G = nx.from_numpy_matrix(distance_mat)
    sparse_G = nx.to_scipy_sparse_matrix(G)
    result = mc.run_mcl(sparse_G, inflation=inflation)
    clusters = mc.get_clusters(result)
    Q = mc.modularity(matrix=result, clusters=clusters)
    return clusters, Q

def dbscan(distance_mat, epsilon, min_samples):
    """
    Runs DBSCAN on the input distance matrix.
    Inputs:
        DISTANCE_MAT: A (neurons x neurons) numpy matrix calculated by some
            distance metric.
        EPSILON: A float; the neighborhood radius for core points.
        MIN_SAMPLES: An int; the minumum number of samples within epsilon-
            distance of a point considered to be a core point.
    Outputs:
        DBSC: The sklearn DBSCAN object
    """
    dbsc = DBSCAN(
        eps=eps, min_samples=min_samples, metric='precomputed'
        ).fit(distance_mat)
    return dbsc
