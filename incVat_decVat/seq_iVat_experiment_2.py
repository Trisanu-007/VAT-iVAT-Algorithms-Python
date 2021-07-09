import cv2
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from CS_data_generate import cs_data_generate
from deciVAT import deciVAT
from decVAT import decVAT
from dunns_index import dunns_index
from inciVat import inciVAT
from incVat import incVAT


def length(mat):
    return np.max(mat.shape)


num_clusters = 3
num_points = 100

DI = 0
data_matrix_with_labels = 0
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

# Add cv2 support


[N, M] = dist_matrix.shape
I = np.array([1, 2])
RV = dist_matrix[1:2, 1:2]
RiV = RV
d = dist_matrix[0, 1]
C = np.array([1, 1])
RI = np.array([1, 2])
RiV_index = [[0, 2], [2, 0]]

p2 = plt.figure(2)
for j in range(num_clusters):
    cluster_index = np.where(data_matrix_with_labels(np.sort(I), 2) == j)
    plt.plot(data_matrix_with_labels[cluster_index, 0],
             data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)


p3 = plt.figure(3)
plt.rcParams["figure.autolayout"] = True
plt.imshow(RiV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])

for i in range(3, N):
    distance_previous_points = dist_matrix[i, I]
    [RV, C, I, RI, d, new_point_location] = incVAT(
        RV, C, I, RI, d, distance_previous_points)

    RiV = inciVAT(RV, RiV, new_point_location)

    p4 = plt.figure(4)
    for j in range(num_clusters):
        cluster_index = np.where(data_matrix_with_labels(np.sort(I), 2) == j)
        plt.plot(data_matrix_with_labels[cluster_index, 0],
                 data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)

    p5 = plt.figure(5)
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(RiV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])


while np.max(I.shape) > 3:

    point_to_remove = I(np.random.rand(length(I), length(I)))
    iVAT_point_to_remove_index = np.where(I == point_to_remove)

    data_matrix_with_labels[iVAT_point_to_remove_index, :] = np.empty()

    RV, C, I, RI, d = decVAT(RV, C, I, RI, d, point_to_remove)

    RiV = deciVAT(RV, RiV, iVAT_point_to_remove_index)

    p6 = plt.figure(6)
    for j in range(num_clusters):
        cluster_index = np.where(data_matrix_with_labels(np.sort(I), 2) == j)
        plt.plot(data_matrix_with_labels[cluster_index, 0],
                 data_matrix_with_labels[cluster_index, 1], marker='o', color=colors[i-1, :], markersize=1)

    p7 = plt.figure(7)
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(RiV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])
