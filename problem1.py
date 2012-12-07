#!/usr/bin/env python2.7
"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: problem1.py

This executable uses the batch perceptron algorithm
to separate a two-category case.
"""
from __future__ import print_function
import batch_perceptron as bp
import make_data as md

def main():
    p1a_fnm = 'prob1a.data'
    p1b_fnm = 'prob1b.data'

    # Separate omega 1 and 2
    p1a_data = md.read_data(p1a_fnm)
    (tc, w) = bp.separate(p1a_data)
    print("Separated omega 1 and omega 2 in", tc, "iterations")
    bp.plot_solution(w, p1a_data, "Problem 1a",
                     dict(mincat="Omega 1", maxcat="Omega 2"))

    # Separate omega 2 and 3
    p1b_data = md.read_data(p1b_fnm)
    (tc, w) = bp.separate(p1b_data)
    print("Separated omega 2 and omega 3 in", tc, "iterations")
    bp.plot_solution(w, p1b_data, "Problem 1b",
                     dict(mincat="Omega 2", maxcat="Omega 3"))

if __name__ == '__main__':
    main()
