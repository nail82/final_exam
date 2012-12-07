"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: parzen_window.py

This module defines the functions for the Parzen window
approach for estimating density functions.
"""

import numpy as np
def zero_one(h):
    """A factory to create the zero-one window function.

    This function creates the zero-one window function.
    Equation 9, page 164, Duda et al.

    Arguments to the window function:
      x - The density function argument.  This is
        point where we want an estimate of the density
        function, ie p(x).

      xi - A single sample from a set of samples that
        we are using to estimate the density.
    """
    d = 1./h # d is the denominator of the window function.
    def phi(x, xi):
        v = np.abs(d * (x-xi))
        if v <= 0.5:
            return 1
        else:
            return 0
    return phi

def gauss(h):
    """A factory to create the Gauss window function.

    This function creates the Gaussian window function,
    Equation 26, page 168, from Duda, et al.

    Arguments to the window function:
      x - The density function argument.  This is
        point where we want an estimate of the density
        function, ie p(x).

      xi - A single sample from a set of samples that
        we are using to estimate the density.
    """
    a = 1./np.sqrt(2*np.pi) # `One over root two pi'
    b = -.5 * (1./h)**2 # square of the inverse window size
    def phi(x, xi):
        return a * np.exp(b * (x-xi)**2)
    return phi

def density_function(phi, S):
    """Factory function to create a density function from a window
    function phi and a list of samples S.

    Args:
      phi - A window function
      S - A numpy array of sample patterns

    Return:
      An estimated density function, p
    """
    n_inv = 1./len(S) # `One over n, where n is the number of samples
    def p(x):
        """Return the scalar estimate of the density
        function at x, utilizing the window function phi.
        """
        return n_inv * np.sum([phi(x, xi) for xi in S])
    return p
