import numpy as np
from numpy.core.fromnumeric import shape


def length(mat):
    return np.max(mat.shape)


def ismember(mat1, mat2):
    return np.isin(mat1, mat2)


def incVAT(RV, C, I, RI, d, distance_previous_points):

    I_old = I
    C_old = C
    new_point_index = np.max(I.shape)+1
    new_point_location = np.max(I.shape)+1

    for j in range(np.max(I.shape)):

        value, index = np.min(distance_previous_points[0:j]), np.argmin(
            distance_previous_points[0:j])
        if value < d[j]:
            new_point_location = j+1
            break
        else:
            [_, index] = np.argmin(distance_previous_points)

    remaining_points = I[new_point_location:-1]
    remaining_points_old_points_method = remaining_points
    remaining_points_location_in_RV =  np.max(RV.shape)
    remaining_points_old_points_method_location_in_RV = remaining_points_location_in_RV
    included_old_points = np.empty()
    included_old_points_location_in_RV = np.empty()
    pointer_last_point = new_point_location-1
    d_remaining = d[new_point_location-1:-1]
    C_remaining = C[new_point_location: -1]

    I = np.array([I[0:new_point_location], new_point_index])
    d = np.array([d[0:new_point_location-1],
                 np.min(distance_previous_points[0:new_point_location])])
    RV_reordering = np.linspace(0, new_point_location)
    C = np.array([C[0:new_point_location], index])

    method = np.empty()
    for k in range(np.max(remaining_points.shape)):

        min_dist_old_points = d_remaining[0]
        closest_old_points = remaining_points_old_points_method[0]
        closest_old_points_location_RV = remaining_points_location_in_RV[0]
        [_, closest_point_C_remaining_old_points] = np.isin(
            I_old[C_remaining[0]], I)

        dist_new_point = distance_previous_points[remaining_points_location_in_RV]
        [min_dist_new_point, index] = np.min(
            dist_new_point), np.argmin(dist_new_point)
        closest_new_point_location_RV = remaining_points_location_in_RV[index]

        closest_new_point = remaining_points[index]
        closest_point_C_remaining_new_point = new_point_location

        dist_included_old_points = RV[
            included_old_points_location_in_RV, remaining_points_location_in_RV]

        if np.max(included_old_points_location_in_RV.shape) == 1:

            [value1, index1] = min(dist_included_old_points)
            [_, closest_point_C_included_old_points] = np.isin(
                included_old_points, I)

        else:
            [value, index] = min(dist_included_old_points)
            [value1, index1] = min(value)
            [_, closest_point_C_included_old_points] = np.isin(
                included_old_points(index[index1]), I)

        min_dist_included_old_points = value1
        closest_included_old_points_location_RV = remaining_points_location_in_RV[index1]
        closest_included_old_points = remaining_points[index1]

        if np.shape(min_dist_included_old_points) == (0, 0):
            [min_dist_all, min_dist_method] = np.min(np.array([min_dist_old_points, min_dist_new_point])), np.argmin(
                np.array([min_dist_old_points, min_dist_new_point]))
        else:
            [min_dist_all, min_dist_method] = np.min(np.array(
                [min_dist_old_points, min_dist_new_point, min_dist_included_old_points])), np.argmin(np.array(
                    [min_dist_old_points, min_dist_new_point, min_dist_included_old_points]))

        if min_dist_method == 1:
            method = np.array([method, 1])
            I = np.array([I, closest_old_points])
            d = np.array([d, min_dist_old_points])
            C = np.array([C, closest_point_C_remaining_old_points])

            RV_reordering = np.array(
                [RV_reordering, closest_old_points_location_RV])
            remaining_points[remaining_points ==
                             closest_old_points] = np.empty()
            remaining_points_old_points_method[remaining_points_old_points_method ==
                                               closest_old_points] = np.empty()
            remaining_points_old_points_method_location_in_RV[remaining_points_old_points_method_location_in_RV ==
                                                              closest_old_points_location_RV] = np.empty()
            remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                            closest_old_points_location_RV] = np.empty()
            pointer_last_point = pointer_last_point+1
            d_remaining[0] = np.empty()
            C_remaining[0] = np.empty()
            if np.max(remaining_points_old_points_method.shape) > 0:
                while np.isin(remaining_points_old_points_method[0], I):
                    pointer_last_point = pointer_last_point+1
                    d_remaining[0] = np.empty()
                    C_remaining[0] = np.empty()
                    remaining_points_old_points_method(1) = np.empty()
                    remaining_points_old_points_method_location_in_RV(1) = np.empty()

                    if np.max(remaining_points_old_points_method.shape) == 0:
                        break
        if min_dist_method == 2:
            method = np.array([method, 2])
            I = np.array([I, closest_old_points])
            d = np.array([d, min_dist_old_points])
            C = np.array([C, closest_point_C_remaining_old_points])

            if closest_new_point == remaining_points[0]:
                if length(remaining_points_old_points_method) > 0:
                    while ismember(remaining_points_old_points_method[0], I):
                        pointer_last_point = pointer_last_point+1
                        d_remaining[0] = np.empty()
                        C_remaining[0] = np.empty()

                        included_old_points(included_old_points == remaining_points_old_points_method[0]) = np.empty()
                        included_old_points_location_in_RV(included_old_points_location_in_RV == remaining_points_old_points_method_location_in_RV(1)) = np.empty()

                        remaining_points_old_points_method[0] = np.empty()
                        remaining_points_old_points_method_location_in_RV[0] = np.empty(
                        )
                        if length(remaining_points_old_points_method) == 0:
                            break
            else:
                included_old_points = np.array(
                    [included_old_points, closest_new_point])
                included_old_points_location_in_RV = np.array(
                    [included_old_points_location_in_RV, closest_new_point_location_RV])

            RV_reordering = np.array(
                [RV_reordering, closest_new_point_location_RV])
            remaining_points(remaining_points == closest_new_point) = np.empty()
            remaining_points_location_in_RV(remaining_points_location_in_RV == closest_new_point_location_RV) = np.empty()

        if min_dist_method == 3:
            method = np.array([method, 3])
            I = np.array([I, closest_old_points])
            d = np.array([d, min_dist_old_points])
            C = np.array([C, closest_point_C_remaining_old_points])

            if length(remaining_points_old_points_method) > 0:
                while ismember(remaining_points_old_points_method[0], I):
                    pointer_last_point = pointer_last_point+1
                    d_remaining[0] = np.empty()
                    C_remaining[0] = np.empty()

                    included_old_points(included_old_points == remaining_points_old_points_method[0]) = np.empty()
                    included_old_points_location_in_RV(included_old_points_location_in_RV == remaining_points_old_points_method_location_in_RV(1)) = np.empty()

                    remaining_points_old_points_method[0] = np.empty()
                    remaining_points_old_points_method_location_in_RV[0] = np.empty(
                    )
                    if length(remaining_points_old_points_method) == 0:
                        break
            else:
                included_old_points = np.array(
                    [included_old_points, closest_new_point])
                included_old_points_location_in_RV = np.array(
                    [included_old_points_location_in_RV, closest_new_point_location_RV])

            RV_reordering = np.array(
                [RV_reordering, closest_new_point_location_RV])
            remaining_points(remaining_points == closest_new_point) = np.empty()
            remaining_points_location_in_RV(remaining_points_location_in_RV == closest_new_point_location_RV) = np.empty()

    RV_old = RV
    RV = RV[RV_reordering, RV_reordering]

    row_to_insert = distance_previous_points[RV_reordering]
    row_to_insert = np.array(
        [row_to_insert[0:new_point_location], 0, row_to_insert[new_point_location:-1]])
    RV = [
        RV[0:new_point_location, 1:new_point_location -
            1], (row_to_insert[0:new_point_location-1]).T, RV[0:new_point_location, new_point_location:-1], row_to_insert,
        RV[new_point_location:-1, 0:new_point_location], (row_to_insert[new_point_location:-1]).T, RV[new_point_location:-1, new_point_location:-1]]
    [_, RI] = np.sort(I), np.argsort(I)

    return RV, C, I, RI, d, new_point_location
