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
