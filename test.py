import numpy as np
from distance2 import distance2
from dunns_index import dunns_index

f = open("datalabels.txt", "r")
lst = []
for line in f:
    strd = line.strip().split(',')
    lst.append([float(x) for x in strd])

arr = np.array(lst)
print(arr)
dist_matrix = distance2(arr[:, 0:2], arr[:, 0:2])
print(dist_matrix.shape)
np.savetxt("distmatrix.txt", dist_matrix, fmt="%e")

dunn = dunns_index(4, dist_matrix, arr[:, 2])
f.close()
