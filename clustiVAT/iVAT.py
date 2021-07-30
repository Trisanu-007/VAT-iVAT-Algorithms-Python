from VAT import VAT
import numpy as np


def iVAT(*args):
    if len(args) == 1:
        VATFlag = 0
    else:
        VATFlag = args[1]
    R = args[0]
    N = np.max(R.shape)
    reordering_mat = np.zeros(N, dtype=int).T
    reordering_mat[0] = 0

    if VATFlag:
        RV = R
    else:
        RV, C, I, RI, cut = VAT(R)
    RiV = np.zeros((N, N))
    for r in range(1, N):
        c = np.linspace(0, r-1, r, dtype=int)
        y = np.min(RV[r, c])
        i = np.argmin(RV[r, c])
        reordering_mat[r] = i
        RiV[r, c] = y
        cnei = c[c != i]
        RiV[r, cnei] = np.max(np.array(
            [RiV[r, cnei], RiV[i, cnei]]), axis=0) if cnei.shape[0] != 0 else np.empty(0)
        RiV[c, r] = RiV[r, c].T

    return RiV, RV, reordering_mat


# This is used for debugging purposes only
if __name__ == "__main__":
    f = open("rv.txt", "r")
    lst = []
    for line in f:
        strd = line.strip().split(',')
        lst.append([float(x) for x in strd])

    arr = np.array(lst)
    RiV, RV, reordering_mat = iVAT(arr, 1)
    np.savetxt("riv.xls", RiV, fmt="%e")
    print(RiV[-1, -2])
    print(reordering_mat)
