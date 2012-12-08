"""
Ted Satcher
CS 640
Fall 2012

Final Exam

File: kmeans.py

This module implements logic for k-means clustering.
"""

"""
Looks like I need these matrices:
  sample pattern matrix
    augmented coords have the current cluster in 0th entry
    0th entry is empty (maybe just for ease of thinking about it)

  cluster matrix
    0th entry is the 0 vector
    columns are the current cluster centers
      augmented mean vector has the number of currently
        assigned patterns in the 0th entry

Algorithm:
Init mean vectors
for i in range(1,len(pattern_vecs))
  closest <- inf
  for j in (1,len(current_centers))
    d <- np.dot(patter_vec[1:,i].T, current_centers[1:,j])
    if d < closest
      closest = d
      patter_vecs[0,i] <- j

There is a clustering pass through the patterns
  Each pattern gets compared to each current_centers
There is an assignment pass through the patterns
  new_centers gets a sum of each pattern assigned & 0th element
    of new_centers gets incremented for each assigned pattern
Mean calculation (new_centers)
mean_diff <- abs(new_mean - current_mean)
if mean_diff.any() > threshold, keep going

(keep track of the length of the mean_diff vector for error plotting)








"""
