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

def main():
    fnm = 'prob5.data'
    patterns = md.read_data(fnm).T
    patterns = km.augment_patterns(patterns)
    d = patterns.shape[0]-1 # Dimensions of the data
    k = 4 # Number of clusters. Figure out where to put this
    current_centers = km.init_cluster_centers(patterns, k)
    cluster_assignments = km.cluster_patterns(patterns, current_centers)
    patterns[0,] = cluster_assignments
    new_means = km.calculate_new_means(patterns, k)


if __name__ == '__main__':
    main()
