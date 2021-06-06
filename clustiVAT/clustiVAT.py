from distance2 import distance2
import numpy as np
from samplePlus import samplePlus
from VAT import VAT


def clustiVAT(x, cp, ns):

    rows, cols = x.shape

    m, rp = samplePlus(x, cp)

    ind = np.argmin(rp, axis=1)

    smp = np.empty((0, 0))

    for i in range(0, cp):
        s = np.where(ind == i)
        nt = np.ceil(ns*np.max(s.shape)/rows)
        idx = np.ceil(np.random.rand(nt, 1)*np.max(s.shape))
        smp = np.vstack((smp, s[idx]))

    rs = distance2(x[smp, :], x[smp, :])

    rv, C, I, ri, cut = VAT(rs)

    return rv, C, I, ri, cut, smp
