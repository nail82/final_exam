"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: batch_perceptron.py

This modulue implements the batch perceptron algorithm.
"""
from __future__ import print_function
import numpy as np
import itertools as it

def separate(data):
    """Use the batch perceptron algorithm to calculate
    the decision function weight vector fo a two class problem.

    Args:
        data: A list of labeled sample vectors. The zeroth element
            of each sample must identify the category this sample
            belongs to.

    Returns:
        A tuple.  The first element is the number of
        iterations to find the separating weight vector
        and the second element is the weight vector
        itself.
    """
    ldata = data.copy()
    aug_data = np.matrix(map(augment_sample, ldata))
    n = aug_data.shape[0]
    wt_dimensions = aug_data.shape[1]

    # Intial weigth vector
    w = np.matrix(np.zeros(wt_dimensions)).T
    total_count = 0
    while True:
        total_count += 1
        w_delta = np.matrix(np.zeros(wt_dimensions)).T
        # Check the classification of each sample
        for i in range(0,len(aug_data)):
            yk = aug_data[i].T
            v = (w.T*yk)[0,0]
            if v <= 0:
                w_delta += yk
        print w_delta
        break


def augment_sample(y):
    """Augment a pattern vector and negate if the
    pattern is a member of omega2"""
    rtn = np.asarray(y).squeeze()
    if rtn[0] > 1:
        rtn[0] = 1
        rtn = -rtn
    return rtn

def test():
    pass

if __name__ == '__main__':
    test()
