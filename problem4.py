#!/usr/bin/env python2.7
"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: problem4.py

This executable is for Problem 4 for the final exam.
It uses the Parzen window approach to calculate
and estimated density function and also to plot
the estimate.
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import parzen_window as pw
import make_data as md

def main():
    fnm = 'p4.data'
    data = np.array(md.read_data(fnm)).squeeze()
    h = 1 # Window size

    # Create plots for the zero-one window function
    phi = pw.zero_one(h)
    print("Zero-one window function...")
    plot_data(phi, data)

    # Create plots for the Gauss window function
    phi =  pw.gauss(h)
    print("Gauss window function...")
    plot_data(phi, data, sample_sizes)

def plot_data(phi, data)
    """This function calculates a new density function
    for each sample size.  It then evaluates the density
    function for a range of values and finally plots the
    results."""
    x = np.linspace(-4,10,100) # x values for plotting
    sample_sizes = [100, 1000, 10000]
    for ss in sample_sizes:
        sample = data[0:ss]
        p = pw.density_function(phi, sample)
        # Evaluate the density funtion for values of x,
        # using the zero-one window function.
        print("Plotting density for a sample size of", ss)
        y = np.array([p(xi) for xi in x])
        plt.plot(x,y)
    plt.show()


if __name__ == '__main__':
    main()
