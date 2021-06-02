import numpy as np


def distance2(A, B):

    M = A.shape[0]
    N = B.shape[0]

    A_dots = (A*A).sum(axis=1).reshape((M, 1))*np.ones(shape=(1, N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2*A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)
