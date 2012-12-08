#!/usr/bin/env python2.7
"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: problem5.py

This is the executable for problem 5.
"""
from __future__ import print_function
import kmeans as km
import make_data as md
import numpy as np
import os

def main():
    fnm = 'prob5.data'
    patterns = md.read_data(fnm).T
    patterns = km.augment_patterns(patterns)
    k = 4 # Number of clusters. Figure out where to put this
    current_centers = km.init_cluster_centers(patterns, k)

    iteration_count = 0
    errors = np.matrix((1, k))
    while True:
        cluster_assignments = km.cluster_patterns(
            patterns, current_centers)
        patterns[0,] = cluster_assignments
        km.plot_patterns_and_centers(patterns, current_centers)
        new_means = km.calculate_new_means(patterns, k)
        mean_diff = current_centers[1:,1:] - new_means[1:,1:]
        errors = (mean_diff.T * mean_diff).diagonal()
        iteration_count += 1
        if (errors > .1).any():
            current_centers = new_means
        else:
            break
    print(current_centers[0,:])
    print(errors)
    print('Converged in', iteration_count,'iterations.')

if __name__ == '__main__':
    main()
