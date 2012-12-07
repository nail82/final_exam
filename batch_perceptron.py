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
import make_data as md
import matplotlib.pyplot as plt

def separate(data):
    """Use the batch perceptron algorithm to calculate
    the decision function weight vector for a two class problem.

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
    # Learning rate
    eta = 1
    # Main loop for the batch perceptron
    while True:
        total_count += 1
        w_delta = np.matrix(np.zeros(wt_dimensions)).T
        # Check the classification of each sample
        mis_class_count = 0
        for i in range(0,len(aug_data)):
            yk = aug_data[i].T
            v = (w.T*yk)[0,0]
            if v <= 0:
                # This sample is mis-classified, so add it to
                # the weight update vector.
                mis_class_count += 1
                w_delta += yk
        # Update the weight vector
        w = w + (eta*w_delta)
        if mis_class_count == 0:
            break
    return (total_count, w)


def augment_sample(y):
    """Augment a pattern vector and negate if the
    pattern is a member of omega2"""
    rtn = np.asarray(y).squeeze()
    if rtn[0] > 1:
        rtn[0] = 1
        rtn = -rtn
    return rtn

def plot_solution(w, data, mytitle, catlabel):
    """
    Scatter plot data and the separation line described
    by the weight vector w.

    Args:
        w: The weight vector describing a solution line.
        data: A three dimensional data matrix. The first
            element of each row is assumed to be the category
            assignment of the sample.
        mytitle: Plot title
        catlabel: A dictionary with category names.

    Note: This function assumes two category data.
    """
    f = get_solution_func(np.array(w))
    adata = np.array(data)

    minx = min(adata[0:,1])
    maxx = max(adata[0:,1])
    rng = maxx-minx
    x = np.array((minx-rng, maxx+rng))
    y = np.array(list(it.imap(f, x)))
    mincat = min(adata[0:,0])
    maxcat = max(adata[0:,0])
    colors = dict(mincat="blue", maxcat="red")
    # Find the data for each category
    i = (adata[0:,0] == mincat)
    j = (adata[0:,0] == maxcat)
    mincat_x = adata[i,1]
    mincat_y = adata[i,2]
    maxcat_x = adata[j,1]
    maxcat_y = adata[j,2]

    # Plot the solution line
    plt.plot(x,y, color="black", label="Decision Boundary")
    plt.scatter(mincat_x, mincat_y, color=colors['mincat'],
                label=catlabel['mincat'])
    plt.scatter(maxcat_x, maxcat_y, color=colors['maxcat'],
                label=catlabel['maxcat'])
    plt.legend()
    plt.title(mytitle, fontsize=18)
    plt.xlabel("X1", fontsize=16)
    plt.ylabel("X2", fontsize=16)
    plt.show()

def get_solution_func(w):
    """
    Resolve a 2d weight vector into a plottable
    function in the form of y = mx + b
    """
    if w[2] == 0:
        raise Exception('Error: slope is infinte')
    return (lambda x1: (-w[0]/w[2]) + (-w[1]/w[2])*x1)
