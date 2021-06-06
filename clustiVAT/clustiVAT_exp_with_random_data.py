import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from data_generate import data_generate

total_no_of_points = 1000
clusters = 4
odds_matrix = np.array(
    [np.ceil(clusters*np.random.rand(clusters))]).astype(int)
print(odds_matrix.shape)
colors_1 = np.array(cm.get_cmap().colors)
colors = np.zeros((clusters, 3))

for i in range(1, clusters+1):
    colors[i-1, :] = colors_1[int(
        np.ceil(max(colors_1.shape)*i/clusters)-1), :]

data_matrix_with_labels, mean_matrix, var_matrix = data_generate(
    number_of_clusters=clusters, odds_matrix=odds_matrix, total_no_of_points=total_no_of_points)


p1 = plt.figure(1)
plt.title(label="Ground truth scatter plot")
for i in range(1, clusters+1):
    cluster_index = np.array(np.where(data_matrix_with_labels[:, -1] == i))
    plt.plot(data_matrix_with_labels[cluster_index, 0],
             data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)

plt.show()
