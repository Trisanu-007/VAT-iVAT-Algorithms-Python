import numpy as np


def inciVAT(RV, RiV_old, point_to_remove_index):

    N = N = np.max(RV.shape)

    RiV = np.zeros((N, N))
    RiV[0:point_to_remove_index-2, 0:point_to_remove_index -
        2] = RiV_old[0:point_to_remove_index-2, 0:point_to_remove_index-2]

    for r in range(point_to_remove_index, N):
        c = np.linspace(0, r-1, r, dtype=int)
        y = np.min(RV[r, c])
        i = np.argmin(RV[r, c])
        RiV[r, c] = y
        cnei = c[c != i]
        RiV[r, cnei] = np.max(np.array(
            [RiV[r, cnei], RiV[i, cnei]])) if cnei.shape[0] != 0 else np.empty(0)
        RiV[c, r] = RiV[r, c].T

    return RiV
