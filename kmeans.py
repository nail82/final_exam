"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: kmeans.py

This module implements logic for k-means clustering.
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Looks like I need these matrices:
  sample pattern matrix
    augmented coords have the current cluster in 0th entry
    0th entry is empty (just for consistency)

  cluster matrix
    0th entry is the 0 vector
    columns are the current cluster centers
      augmented mean vector has the number of currently
        assigned patterns in the 0th entry

Algorithm:
Init mean vectors
for i in range(1,len(pattern_vecs))
  closest <- inf
  for j in (1,len(current_centers))
    d <- np.dot(patter_vec[1:,i].T, current_centers[1:,j])
    if d < closest
      closest = d
      patter_vecs[0,i] <- j

There is a clustering pass through the patterns
  Each pattern gets compared to each current_centers
There is an assignment pass through the patterns
  new_centers gets a sum of each pattern assigned & 0th element
    of new_centers gets incremented for each assigned pattern
Mean calculation (new_centers)
mean_diff <- abs(new_mean - current_mean)
if mean_diff.any() > threshold, keep going

(keep track of the length of the mean_diff vector for error plotting)

"""

def init_cluster_centers(patterns, k):
    """Create a cluster center matrix.  The 0th
    column of the matrix is zeros and unused.
    The 0th row holds the number of patterns assigned
    to this cluster.

    Args:
      patterns - The augmented pattern matrix.  The initial cluster
        centers are the first k column vectors from patterns.

      k - The number of clusters.  The returned matrix will
        have k+1 columns.

    Returns:
      A d+1 by k+1 matrix
    """
    d = patterns.shape[0]-1
    pattern_ids = range(1,patterns.shape[1])
    np.random.shuffle(pattern_ids)
    centers = np.matrix(np.zeros((d+1, k+1)))
    for i in range(1, k+1):
        patt_id = pattern_ids[i]
        centers[1:,i] = patterns[1:,patt_id]
    return centers

def augment_patterns(patterns):
    """Add a row of zeros for the cluster id and an
    empty column in the 0th position."""
    zero_row = np.zeros((1, patterns.shape[1]))
    patterns = np.vstack((zero_row, patterns))
    zero_col = np.zeros((patterns.shape[0], 1))
    patterns = np.hstack((zero_col, patterns))
    return patterns

def cluster_patterns(patterns, centers):
    cluster_assignments = np.zeros((1, patterns.shape[1]))
    n = patterns.shape[1]
    k = centers.shape[1]
    for pat_index in range(1, n):
        closest = np.inf
        for ctr_index in range(1, k):
            v = patterns[1:,pat_index] - centers[1:,ctr_index]
            distance = np.dot(v.T, v)[0,0]
            if distance < closest:
                closest = distance
                cluster_assignments[0,pat_index] = ctr_index
    return cluster_assignments

def calculate_new_means(patterns, k):
    new_centers = np.matrix(np.zeros((patterns.shape[0], k+1)))
    for i in range(1, patterns.shape[1]):
        cluster = patterns[0,i]
        new_centers[1:,cluster] += patterns[1:,i]
        new_centers[0,cluster] += 1

    # Find the new means
    for i in range(1, new_centers.shape[1]):
        if new_centers[0,i] == 0:
            # This center didn't collect any patterns, so
            # just give it one of the patterns
            new_centers[1:,i] = patterns[1:,i]
        else:
            new_centers[1:,i] = new_centers[1:,i] / new_centers[0,i]
    return new_centers

def plot_patterns_and_centers(patterns, centers):
    """Plot 2D patterns and cluster centers.

    Args:
      patterns - Patterns labeled with cluster assignment.
      centers - Cluster centers.
    """
    for i in range(1, patterns.shape[1]):
        plt.scatter(patterns[1,i], patterns[2,i])

    for i in range(1, centers.shape[1]):
        plt.scatter(centers[1,i], centers[2,i], marker='+', color='red', s = 80)
    plt.show()
