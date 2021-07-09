import time

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
from VAT import VAT
from iVAT import iVAT


def length(mat):
    return np.max(mat.shape)


num_clusters = 3
num_points = 100

DI = 0
dist_matrix = np.array([0, 0])
while DI < 0.2 or DI > 0.3:
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


[N, M] = dist_matrix.shape
I = np.array([1, 2])
RV = dist_matrix[1:2, 1:2]
RiV = RV
d = dist_matrix[0, 1]
C = np.array([1, 1])
RI = np.array([1, 2])
RiV_index = [[0, 2], [2, 0]]

time_incVAT = []
time_inciVAT = []
time_VAT = []
time_iVAT = []

for i in range(3, N+1):

    tic = time.time()
    distance_previous_points = dist_matrix[i, I]
    [RV, C, I, RI, d, new_point_location] = incVAT(
        RV, C, I, RI, d, distance_previous_points)
    toc = time.time()
    time_incVAT.append(toc-tic)

    tic = time.time()
    RiV_old = RiV
    RiV = inciVAT(RV, RiV_old, new_point_location)
    toc = time.time()
    time_inciVAT.append(toc-tic)

    tic = time.time()
    RV_vat, C_vat, I_vat, RI_vat, d_vat = VAT(dist_matrix[0: i, 1: i])
    toc = time.time()
    time_VAT.append(toc-tic)

    tic = time.time()
    RiV_vat = iVAT(RV_vat)
    toc = time.time()
    time_iVAT.append(toc-tic)


p1 = plt.figure(1)
plt.rcParams["figure.autolayout"] = True
plt.imshow(RV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])

p2 = plt.figure(2)
plt.rcParams["figure.autolayout"] = True
plt.imshow(RiV, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])

p3 = plt.figure(3)
plt.rcParams["figure.autolayout"] = True
plt.imshow(RiV_vat, cmap=cm.get_cmap('gray'), extent=[-1, 1, -1, 1])

p4 = plt.figure(4)
x1 = [i for i in range(1, len(time_incVAT)+1)]
x2 = [i for i in range(1, len(time_VAT)+1)]
plt.plot(x1, time_incVAT, 'b', x2, time_VAT, 'r')

p5 = plt.figure(5)
x1 = [i for i in range(1, len(time_inciVAT)+1)]
x2 = [i for i in range(1, len(time_iVAT)+1)]
plt.plot(x1, time_inciVAT, 'b', x2, time_iVAT, 'r')

p6 = plt.figure(6)
x1 = [i for i in range(1, len(time_incVAT)+1)]
x2 = [i for i in range(1, len(time_VAT)+1)]
x3 = [i for i in range(1, len(time_inciVAT)+1)]
x4 = [i for i in range(1, len(time_iVAT)+1)]
plt.plot(x1, time_incVAT, 'b', x2, time_VAT, 'r',
         x3, time_inciVAT, 'g', x4, time_iVAT, 'm')


p6 = plt.figure(6)
plt.plot(x1, time_incVAT + time_inciVAT, 'b', x2, time_VAT+time_iVAT, 'r')
