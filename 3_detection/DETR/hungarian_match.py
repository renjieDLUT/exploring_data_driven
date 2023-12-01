from scipy.optimize import linear_sum_assignment
import numpy as np

# [ 3, 5, 6, 7
#   1, 8, 4, 2
#   9, 0, 1, 5]
matrix=np.asarray([[3, 5, 6, 7],[1, 8, 4, 2],[9, 0, 1, 5]])

# (array([0, 1, 2]), array([0, 3, 1]))
ret=linear_sum_assignment(matrix)

# [3 2 0]
ret=matrix[ret[0],ret[1]]


