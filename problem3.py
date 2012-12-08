#!/usr/bin/env python2.7
"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: prob3_plot.py

Plot the decision boundary for the Bayesian classifier
developed in problem 3.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import make_data as md

def main():
    fnm = 'prob3.data'
    data = md.read_data(fnm)
    D1 = data[0:8,].T
    D2 = data[8:,].T

    u1 = np.matrix((np.mean(D1[0,:]), np.mean(D1[1,:]))).T
    u2 = np.matrix((np.mean(D2[0,:]), np.mean(D2[1,:]))).T

    sigma1 = np.asmatrix(np.cov(D1, bias=1))
    sigma2 = np.asmatrix(np.cov(D1, bias=1))

    g1 = discrim_func(u1, sigma1)
    g2 = discrim_func(u2, sigma2)

    steps = 100
    x = np.linspace(-2,2,steps)
    y = np.linspace(-6,6,steps)

    X,Y = np.meshgrid(x,y)
    z = [g1(X[r,c], Y[r,c]) - g2(X[r,c], Y[r,c])
         for r in range(0,steps) for c in range(0,steps)]
    Z = np.array(z)
    px = X.ravel()
    py = Y.ravel()
    pz = Z.ravel()
    gridsize = 50
    plot = plt.subplot(111)
    plt.hexbin(px,py,C=pz, gridsize=gridsize, cmap=cm.jet, bins=None)
    cb = plt.colorbar()
    cb.set_label('g1 minus g2')
    return plot

def discrim_func(u, sigma):
    """Create a discriminant function for an arbitrary normal
    distribution with multivariate parameters u and sigma.  NOTE:
    Assumes the priors are equal.

    The argument to the returned function is expected to have
    the same shape as u.
    """
    sigma_inverse = sigma.I
    det = np.linalg.det(sigma)
    d = u.shape[0]
    second_term = -.5 * np.log(det)
    def g(x):
        assert(x.shape[0] == d)
        x_minus_u = x-u
        first_term = np.zeros(x.shape[1])
        for i in range(0, x.shape[1]):
            v = x_minus_u[:,i]
            first_term[i] = v.T * sigma_inverse * v * -.5
        return (first_term + second_term)
    return g

def multi_norm(u, sigma):
    """Return the density function for a multivariate normal
    distribution with mean u and covariance matrix sigma"""

    d = u.shape[0]
    sigma_I = sigma.I

    sig = np.sqrt(np.linalg.det(sigma))
    first_term = (
        ((2.*np.pi)**(-d/2.)) *
        (np.linalg.det(sigma)**(-1./2.)))

    def f(x):
        assert(x.shape[0] == d)
        x_minus_u = x-u
        second_term = np.zeros(x.shape[1])
        for i in range(0, x_minus_u.shape[1]):
            v = x_minus_u[:,i]
            second_term[i] = (np.exp(v.T * sigma_I * v * -.5))
        second_term = np.array(second_term)
        return (first_term * second_term)

    return f



if __name__ == '__main__':
    main()
