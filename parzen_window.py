"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: parzen_window.py

This module defines the functions to create window
functions for the Parzen window approach for estimating
density functions.

Each function created takes as arguments x and xi where
x is the argument to the density function p(x) and xi
is one sample from the set of sample patterns.
"""

import numpy as np
def zero_one(h):
    """This function creates the zero-one window function.
    Equation 9, page 164, Duda et al."""
    d = 1./h # d is the denominator of the window function.
    def phi(x, xi):
        v = np.abs(d * (x-xi))
        if v <= 0.5:
            return 1
        else:
            return 0
    return phi

def gauss(h):
    """This function creates a Gaussian window function,
    Equation 26, page 168, from Duda, et al."""
    a = 1./np.sqrt(2*np.pi)
    b = -.5 * (1./h)**2
    def phi(x, xi):
        return a * np.exp(b * (x-xi)**2)
    return phi

def density_function(phi, S):
    """Create a density function from a window function phi
    and a list of samples X.
    Args:
      phi - A window function
      S - A numpy array of sample patterns

    Return:
      An estimated density function, p
    """
    n_inv = 1./len(S)
    def p(x):
        return n_inv * np.sum([phi(x, xi) for xi in S])
    return p
