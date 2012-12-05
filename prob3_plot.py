"""


Plot the decision boundary for the Bayesian classifier
developed in problem 3.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import make_data as md

def main():
    fnm = 'prob3.data'
    data = md.read(fnm)
    D1 = data[0:8,].T
    D2 = data[8:,].T

    u1 = np.matrix((np.mean(D1[0,:]), np.mean(D1[1,:]))).T
    u2 = np.matrix((np.mean(D2[0,:]), np.mean(D2[1,:]))).T

    sigma1 = np.asmatrix(np.cov(D1, bias=1))
    sigma2 = np.asmatrix(np.cov(D1, bias=1))

    g1 = discrim_func(u1, sigma1)
    g2 = discrim_func(u2, sigma2)



def discrim_func(u, sigma):
    """Create a discriminant function for an arbitrary normal
    distribution with multivariate parameters u and sigma.  NOTE:
    Assumes the priors are equal.

    The argument to the returned function is expected to have
    the same shape as u.
    """
    sigma_inverse = sigma.I
    det = np.linalg.det(sigma)
    def g(x):
        rtn = (-.5*(x-u).T * sigma_inverse * (x-u)) -.5* np.log(det)
        return rtn[0,0]
    return g


if __name__ == '__main__':
    main()
