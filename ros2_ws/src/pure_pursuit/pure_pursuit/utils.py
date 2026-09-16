import numpy as np

# helper functions
def pt_to_pt_distance (pt1,pt2):
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance

# returns -1 if num is negative, 1 otherwise
def sgn (num):
  if num >= 0:
    return 1
  else:
    return -1