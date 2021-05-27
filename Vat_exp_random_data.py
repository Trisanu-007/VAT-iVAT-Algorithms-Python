import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import mode

from data_generate_non_cs import data_generate_non_cs
from distance2 import distance2
from iVAT import iVAT
from VAT import VAT

total_num_of_points = 2000
num_clusters = 4

odds_matrix = np.ones((1, 4), dtype=int)

data_matrix_with_labels, mean_matrix, var_matrix = data_generate_non_cs(
    num_clusters, odds_matrix, total_num_of_points)

colors_1 = np.array(cm.get_cmap().colors)
colors = np.zeros((num_clusters, 3))

for i in range(1, num_clusters+1):
    colors[i-1, :] = colors_1[int(
        np.ceil(max(colors_1.shape)*i/num_clusters)-1), :]

p1 = plt.figure(1)
plt.title(label="Ground truth scatter plot")
for i in range(1, num_clusters+1):
    cluster_index = np.array(np.where(data_matrix_with_labels[:, -1] == i))
    plt.plot(data_matrix_with_labels[cluster_index, 0],
             data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)

# plt.axis("equal")
# plt.show()

##############  VAT Algorithm ####################

x = data_matrix_with_labels
n, p = x.shape

tic = time.time()

pi_true = x[:, -1]
x = x[:, 0:-1]

rs = distance2(x, x)
rv, C, I, RI, cut = VAT(rs)
RiV, RV, reordering_mat = iVAT(rv, 1)

toc = time.time()
print("Time elapsed : ", str(toc-tic))

x1, y1 = cut.shape
cut = cut.reshape((x1*y1,))
cuts, ind = -np.sort(-cut), np.argsort(-cut)
ind = np.sort(ind[0:num_clusters-1])

Pi = np.zeros((n,))
Pi[I[0:ind[1]]] = 1
Pi[I[ind[-1]:-1]] = num_clusters

for k in range(1, num_clusters-1):
    Pi[I[ind[k-1]:ind[k]-1]] = k

p2 = plt.figure(2)
plt.rcParams["figure.autolayout"] = True
plt.imshow(rv, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])
plt.title(label="VAT reordered dissimilarity matrix image")


p3 = plt.figure(3)
plt.rcParams["figure.autolayout"] = True
plt.imshow(RiV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])
plt.title(label="iVAT dissimilarity matrix image")


p4 = plt.figure(4)
for i in range(0, np.max(I.shape)-1):
    x_cor = np.hstack((x[I[i], 0], x[I[C[i]], 0]))
    y_cor = np.hstack((x[I[i], 1], x[I[C[i]], 1]))
    plt.plot(x_cor, y_cor, 'b')

for i in range(np.max(ind.shape)):
    x_cor = np.hstack((x[I[ind[i]], 0], x[I[C[ind[i]]], 0]))
    y_cor = np.hstack((x[I[ind[i]], 1], x[I[C[ind[i]]], 1]))
    plt.plot(x_cor, y_cor, 'g')

p5 = plt.figure(5)
plt.plot(x[I, 0], x[I, 1], 'r.')
plt.title(label="MST of the dataset")

# Thorough debugging left from here:
# cluster_matrix_mod = np.zeros((total_num_of_points))
# length_partition = np.zeros((num_clusters))

# for i in range(0, num_clusters):
#     length_partition[i] = np.max(np.where(Pi == i).shape)

# length_partition_sorted, length_partition_sorted_idx = - \
#     np.sort(-length_partition), np.argsort(-length_partition)
# index_remaining = np.linspace(0, num_clusters, num_clusters-1, dtype=int)

# for i in range(0, num_clusters):

#     original_idx = length_partition_sorted_idx[i]
#     partition = np.where(Pi == original_idx)
#     proposed_idx = mode(pi_true(partition))

#     if np.sum(index_remaining == proposed_idx) != 0:
#         cluster_matrix_mod[np.where(Pi == original_idx)] = proposed_idx
#     else:
#         cluster_matrix_mod[np.where(Pi == original_idx)] = index_remaining[0]

#     index_remaining = np.delete(
#         index_remaining, index_remaining == proposed_idx)

# p6 = plt.figure(6)
# for i in range(0, num_clusters):
#     if i == 0:
#         partition = I[0:ind[i]]
#     elif i == num_clusters-1:
#         partition = I[ind[i-1]:np.max(I.shape)]
#     else:
#         partition = I[ind[i-1]:ind[i]-1]

#     plt.plot(x[partition, 0], x[partition, 1], '.', colors[i, :])


# plt.title(label="VAT generated partition of the dataset")
# plt.show()
