import numpy as np


def dunns_index(cluster_number, dist_matrix, indexes):

    denominator = np.empty((0, 1))

    for num in range(1, cluster_number+1):
        ind_i = np.array(np.where(indexes == num))
        ind_j = np.array(np.where(indexes != num))
        c = np.ix_(ind_i[0], ind_j[0])
        temp = dist_matrix[c]
        temp = temp.flatten('F')
        temp = np.array([temp])
        denominator = np.vstack((denominator, temp.T))

    min_num = np.min(denominator)
    dist_rnum, dist_cnum = dist_matrix.shape
    neg_obs = np.zeros((dist_rnum, dist_cnum))

    for num in range(1, cluster_number+1):
        idx = np.array(np.where(indexes == num))
        neg_obs[np.ix_(idx[0], idx[0])] = 1

    max_num = np.max(neg_obs*dist_matrix)

    dunns_index = min_num/max_num
    #print("Dunns index: " + str(dunns_index))
    return dunns_index
