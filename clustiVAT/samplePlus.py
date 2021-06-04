import numpy as np
from distance2 import distance2


def samplePlus(X, cp):
    rows, cols = np.shape(X)

    m = np.ones((cp, 1), dtype=int)
    d = distance2(np.array([X[0, :]]), X).T
    c1, c2 = d.shape
    d = np.reshape(d, (c1*c2))

    Rp = d
    # Rp = np.array([d[:, 0]])
    Rp = np.array([d]).T

    for i in range(1, cp):
        d = np.minimum(d, Rp[:, i-1])
        m[i, 0] = np.argmax(d, axis=0)
        Rp = np.hstack((Rp, distance2(X[m[i], :], X).T))

    return m, Rp

# For debugging
if __name__ == "__main__":
    X = np.array([[16, 2, 3, 13], [5, 11, 10, 8],
                 [9, 7, 6, 12], [4, 14, 15, 1]])

    m, Rp = samplePlus(X, 5)
    print(Rp)
    print(m)
