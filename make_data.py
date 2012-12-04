"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: make_data.py<3>

This module is for reading and creating data
for the Final Exam.
"""
from __future__ import print_function
import numpy as np

def read_data(fnm):
    """Read data from the csv."""
    fh = open(fnm, 'r')
    data = np.genfromtxt(fh, dtype=float, delimiter=',')
    return np.matrix(data)

def write_data1(fnm, points):
    """Write that data to a csv."""
    fh = open(fnm, 'w')
    for p in points:
        p.tofile(fh, sep=',')
        fh.write('\n')

def fifty_points():
    """Creates a linearly separable, two category
    data set."""
    x1 = np.random.randn(50) + 2.5
    y1 = np.random.randn(50) + 2.5
    x2 = np.random.randn(50) + 7.5
    y2 = np.random.randn(50) + 7.5
    one = np.ndarray(50);
    two = np.ndarray(50);
    one.fill(1)
    two.fill(2)
    omega1 = np.column_stack((one,x1,y1))
    omega2 = np.column_stack((two,x2,y2))
    data = np.concatenate((omega1,omega2))
    return np.matrix(data)
