from distance2 import distance2
import numpy as np
from numpy import random as ran


def data_generate(number_of_clusters, odds_matrix, total_no_of_points):

    mean_x_matrix = 1000*ran.randn(1, number_of_clusters)
    mean_Y_matrix = 1000*ran.randn(1, number_of_clusters)

    var_x_matrix = 30*np.abs(ran.randn(1, number_of_clusters))
    var_y_matrix = 30*np.abs(ran.randn(1, number_of_clusters))

    x_cor = int(np.ceil(total_no_of_points/np.sum(odds_matrix))
                )*np.sum(odds_matrix)

    data_matrix_with_labels = np.zeros((x_cor, 3))

    l = 0
    while l <= max(np.shape(data_matrix_with_labels)):
        for j in range(number_of_clusters):
            for _ in range(odds_matrix[0][j]):
                data_matrix_with_labels[l:] = [mean_x_matrix[:, j] + var_x_matrix[:, j] * ran.randn(
                    1), mean_Y_matrix[:, j] + var_y_matrix[:, j] * ran.randn(1), int(j+1)]

                l += 1

    mean_matrix = np.array([mean_x_matrix, mean_Y_matrix])
    var_matrix = np.array([var_x_matrix, var_y_matrix])

    return data_matrix_with_labels, mean_matrix, var_matrix
