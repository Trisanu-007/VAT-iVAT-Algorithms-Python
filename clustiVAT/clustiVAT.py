from distance2 import distance2
import numpy as np
from samplePlus import samplePlus
from VAT import VAT


def clustiVAT(x, cp, ns):

    rows, cols = x.shape

    m, rp = samplePlus(x, cp)

    ind = np.argmin(rp, axis=1)

    smp = np.empty((0, 1), dtype=int)

    for i in range(0, cp):
        s = np.where(ind == i)[0].astype(int)
        nt = np.ceil(ns*np.max(s.shape)/rows).astype(int)
        idx = np.ceil(np.random.rand(nt, 1)*np.max(s.shape)).astype(int)
        smp = np.vstack((smp, s[idx-1]))

    r_smp, c_smp = smp.shape
    smp = np.reshape(smp, (r_smp*c_smp))
    rs = distance2(x[smp, :], x[smp, :])

    rv, C, I, ri, cut = VAT(rs)

    return rv, C, I, ri, cut, smp
