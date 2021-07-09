from matplotlib import pyplot as plt
from dunns_index import dunns_index
import numpy as np
from CS_data_generate import cs_data_generate
from matplotlib import cm


num_clusters = 3
num_points = 100

DI = 0
# data_matrix_with_labels=0
while DI < 1:
    odds_matrix = np.ceil(
        num_clusters * np.random.rand(1, num_clusters)).astype(int)
    data_matrix_with_labels, dist_matrix = cs_data_generate(
        number_of_clusters=num_clusters, odds_matrix=odds_matrix, total_no_of_points=num_points)

    DI = dunns_index(num_clusters, dist_matrix, data_matrix_with_labels[:, 2])


colors_1 = np.array(cm.get_cmap().colors)
colors = np.zeros((num_clusters, 3))

for i in range(1, num_clusters+1):
    colors[i-1, :] = colors_1[int(
        np.ceil(max(colors_1.shape)*i/num_clusters)-1), :]

p1 = plt.figure(1)
for i in range(1, num_clusters+1):
    cluster_index = np.array(np.where(data_matrix_with_labels[:, -1] == i))
    plt.plot(data_matrix_with_labels[cluster_index, 0],
             data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)
