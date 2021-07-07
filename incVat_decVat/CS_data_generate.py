import numpy as np
from numpy.core.numeric import ones
from numpy.random import randn as ran

# [NOTE] CLusters are marked as 0,1,2..... and not as 1,2,3....


def cs_data_generate(number_of_clusters, odds_matrix, total_no_of_points):

    mean_x_matrix = 1000*ran(1, number_of_clusters)
    mean_Y_matrix = 1000*ran(1, number_of_clusters)

    var_x_matrix = 100*np.abs(ran(1, number_of_clusters))
    var_y_matrix = 100*np.abs(ran(1, number_of_clusters))

    x_cor = int(np.ceil(total_no_of_points/np.sum(odds_matrix))
                )*np.sum(odds_matrix)

    data_matrix_with_labels = np.zeros((x_cor, 3))

    l = 0
    while l <= max(np.shape(data_matrix_with_labels)):
        for j in range(number_of_clusters):
            for _ in range(odds_matrix[0][j]):
                data_matrix_with_labels[l:] = [mean_x_matrix[:, j] + var_x_matrix[:, j] * ran(
                    1), mean_Y_matrix[:, j] + var_y_matrix[:, j] * ran(1), int(j)]

                l += 1

    random_permutation = np.random.permutation(
        np.max(data_matrix_with_labels.shape))
    data_matrix_with_labels = data_matrix_with_labels[random_permutation, :]

    dist_matrix = np.zeros(
        (np.max(data_matrix_with_labels.shape), np.max(data_matrix_with_labels.shape)))
    leng, wid = dist_matrix.shape

    for l in range(leng):
        diff_vector = data_matrix_with_labels[:, 0:1] - \
            np.array([data_matrix_with_labels[l, 0]*np.ones((leng, 1)),
                     data_matrix_with_labels[l, 1]*np.ones((leng, 1))])
        dist_matrix[l, :] = np.abs(
            diff_vector[:, 0] + np.complex(0, 1)*diff_vector[:, 1])

    return data_matrix_with_labels, dist_matrix
