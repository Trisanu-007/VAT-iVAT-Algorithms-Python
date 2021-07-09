import numpy as np
from numpy.ma.core import where


end = -1


def decVAT(RV, C, I, RI, d, point_to_remove):

    point_to_remove_index = np.where(I == point_to_remove)
    removed_points_associates_index = np.empty((0, 1))

    for j in range(0, np.max(point_to_remove_index)):
        removed_points_associates_index = np.hstack(
            removed_points_associates_index, np.where(C[1:-1] == point_to_remove_index[j])+1)

    removed_points_associates = I[removed_points_associates_index]

    if(removed_points_associates.size == (0, 0)):
        I[point_to_remove] = np.empty((0, 1))

        C[point_to_remove] = np.empty((0, 1))
        index = np.where(C > point_to_remove_index)
        C[index] = C[index] - 1

        d[point_to_remove_index-1] = np.empty((0, 1))

        RV[point_to_remove_index, :] = np.empty((0, 1))
        RV[:, point_to_remove_index] = np.empty((0, 1))

        RI = np.argsort(I)

    else:
        I_old = I

        if point_to_remove == I_old(0):
            remaining_points = I[2:-1]
            remaining_points_old_points_method = remaining_points
            remaining_points_location_in_RV = np.linspace(
                3, max(RV.size))  # Check step

            remaining_points_old_points_method_location_in_RV = remaining_points_location_in_RV
            included_old_points = np.empty((0, 1))
            included_old_points_location_in_RV = np.empty((0, 1))
            pointer_last_point = 2
            d_remaining = d[1: end]
            C_remaining = C[2: end]

            I = I[2]
            d = np.empty((0, 1))
            RV_reordering = 2
            C = 1

            idx = np.where(removed_points_associates == I)
            removed_points_associates[idx] = np.empty((0, 1))
            removed_points_associates_index[idx] = np.empty((0, 1))

        else:
            remaining_points = I[point_to_remove_index+1:end]
            remaining_points_old_points_method = remaining_points
            remaining_points_location_in_RV = np.linspace(
                point_to_remove_index+1, max(RV.size))
            remaining_points_old_points_method_location_in_RV = remaining_points_location_in_RV
            included_old_points = np.empty((0, 1))
            included_old_points_location_in_RV = np.empty((0, 1))
            pointer_last_point = point_to_remove_index
            d_remaining = d[point_to_remove_index: end]
            C_remaining = C[point_to_remove_index+1: end]

            I = I[0: point_to_remove_index-1]
            d = d[0: point_to_remove_index-2]
            RV_reordering = np.linspace(1, point_to_remove_index-1)
            C = C[0: point_to_remove_index-1]

        for k in range(np.max(remaining_points.size)):
            if removed_points_associates.size != 0:
                if remaining_points_old_points_method[0] == removed_points_associates[0]:
                    dist_remaining_points = RV[RV_reordering,
                                               remaining_points_location_in_RV]
                    len, be = dist_remaining_points.size
                    if len == 1:
                        min_dist_remaining_points, index1 = np.min[dist_remaining_points], np.argmin(
                            dist_remaining_points)
                        closest_point_index_remaining_points = remaining_points_location_in_RV[
                            index1]
                        closest_point_remaining_points = I_old[
                            remaining_points_location_in_RV[index1]]
                        closest_point_C_index_remaining_points = np.isin(
                            I_old[[RV_reordering[0]]], I)
                    else:

                        value, index = np.min(dist_remaining_points), np.argmin(
                            dist_remaining_points)
                        min_dist_remaining_points, index1 = np.min(
                            value), np.argmin(value)
                        closest_point_index_remaining_points = remaining_points_location_in_RV[
                            index1]
                        closest_point_remaining_points = I_old[
                            remaining_points_location_in_RV[index1]]
                        _, closest_point_C_index_remaining_points = np.isin(
                            I_old[RV_reordering[index[index1]]], I)

                    if np.isin(closest_point_remaining_points, removed_points_associates):
                        I = np.array([I, closest_point_remaining_points])
                        d = np.array([d, min_dist_remaining_points])
                        C = np.array(
                            [C, closest_point_C_index_remaining_points])
                        RV_reordering = np.array(
                            [RV_reordering, closest_point_index_remaining_points])
                        remaining_points[remaining_points ==
                                         closest_point_remaining_points] = np.empty()
                        remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                        closest_point_index_remaining_points] = np.empty()

                        if closest_point_remaining_points == removed_points_associates(0):

                            remaining_points_old_points_method[remaining_points_old_points_method ==
                                                               closest_point_remaining_points] = np.empty()
                            remaining_points_old_points_method_location_in_RV[
                                remaining_points_old_points_method_location_in_RV == closest_point_index_remaining_points] = np.empty()
                            pointer_last_point = pointer_last_point+1
                            d_remaining[0] = np.empty()
                            C_remaining[0] = np.empty()

                            if np.max(remaining_points_old_points_method.size) > 0:
                                while(np.isin(remaining_points_old_points_method[0], I)):
                                    pointer_last_point = pointer_last_point+1
                                    d_remaining[0] = np.empty()
                                    C_remaining[0] = np.empty()
                                    remaining_points_old_points_method[0] = np.empty(
                                    )
                                    remaining_points_old_points_method_location_in_RV[0] = np.empty(
                                    )

                                    if(np.max(remaining_points_old_points_method.size) == 0):
                                        break
                        idx = np.where(removed_points_associates ==
                                       closest_point_remaining_points)
                        removed_points_associates[idx] = np.empty()
                        removed_points_associates_index[idx] = np.empty()

                    else:
                        I = np.array([I, closest_point_remaining_points])
                        d = np.array([d, min_dist_remaining_points])
                        C = np.array(
                            [C, closest_point_C_index_remaining_points])

                        included_old_point = np.empty()
                        included_old_points = np.array(
                            [included_old_point, closest_point_remaining_points])
                        included_old_points_location_in_RV = np.array(
                            [included_old_points_location_in_RV, closest_point_index_remaining_points])

                        RV_reordering = np.array(
                            [RV_reordering, closest_point_index_remaining_points])
                        if not np.isin(closest_point_remaining_points):
                            remaining_points[remaining_points ==
                                             closest_point_remaining_points] = np.empty()
                            remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                            closest_point_index_remaining_points] = np.empty()

                else:
                    min_dist_old_points = d_remaining[0]
                    closest_old_points = remaining_points_old_points_method[0]
                    closest_old_points_location_RV = remaining_points_location_in_RV[
                        0]
                    _, closest_point_C_remaining_old_points = np.isin(
                        I_old(C_remaining[0]), I)

                    dist_included_old_points = RV[
                        included_old_points_location_in_RV, remaining_points_location_in_RV]
                    if np.max(included_old_points_location_in_RV.size) == 1:
                        value1, index1 = min(dist_included_old_points), np.argmin(
                            dist_included_old_points)
                        _, closest_point_C_included_old_points = np.isin(
                            included_old_points, I)
                    else:
                        value, index = min(dist_included_old_points), np.argmin(
                            dist_included_old_points)
                        value1, index1 = min(value), np.argmin(value)
                        _, closest_point_C_included_old_points = np.isin(
                            included_old_points[index[index1]], I)

                    min_dist_included_old_points = value1
                    closest_included_old_points_location_RV = remaining_points_location_in_RV[
                        index1]
                    closest_included_old_points = remaining_points[index1]

                    if min_dist_included_old_points.size == 0:
                        min_dist_all = min_dist_old_points
                        min_dist_method = 1
                    else:
                        [min_dist_all, min_dist_method] = min(
                            np.array([min_dist_old_points, np.inf, min_dist_included_old_points]))
                    if min_dist_method == 0:
                        I = np.array([I, closest_old_points])
                        d = np.array([d, min_dist_old_points])
                        C = np.array([C, closest_point_C_remaining_old_points])
                        RV_reordering = np.array(
                            [RV_reordering, closest_old_points_location_RV])
                        remaining_points[remaining_points ==
                                         closest_old_points] = np.empty()
                        remaining_points_old_points_method[remaining_points_old_points_method == closest_old_points] = np.empty(
                        )
                        remaining_points_old_points_method_location_in_RV[
                            remaining_points_old_points_method_location_in_RV == closest_old_points_location_RV] = np.empty()
                        remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                        closest_old_points_location_RV] = np.empty()
                        pointer_last_point = pointer_last_point+1
                        d_remaining[0] = np.empty()
                        C_remaining[0] = np.empty()
                        if np.max(remaining_points_old_points_method.size) > 0:
                            while np.isin(remaining_points_old_points_method(0), I):
                                pointer_last_point = pointer_last_point+1
                                d_remaining[0] = np.empty()
                                C_remaining[0] = np.empty()
                                remaining_points_old_points_method[0] = np.empty(
                                )
                                remaining_points_old_points_method_location_in_RV[0] = np.empty(
                                )
                                if max(remaining_points_old_points_method.size) == 0:
                                    break

                    if min_dist_method == 1:
                        print('Error')
                        break
                    if min_dist_method == 2:
                        I = np.array([I, closest_included_old_points])
                        d = np.array([d, min_dist_included_old_points])
                        C = np.array([C, closest_point_C_included_old_points])
                        if closest_included_old_points == remaining_points[0]:
                            if np.max(remaining_points_old_points_method.size) > 0:
                                while np.isin(remaining_points_old_points_method[0], I):
                                    pointer_last_point = pointer_last_point+1
                                    d_remaining[0] = np.empty()
                                    C_remaining[0] = np.empty()

                                    included_old_points[included_old_points ==
                                                        remaining_points_old_points_method[0]] = np.empty()
                                    included_old_points_location_in_RV[included_old_points_location_in_RV ==
                                                                       remaining_points_old_points_method_location_in_RV[0]] = np.empty()

                                    remaining_points_old_points_method[0] = np.empty(
                                    )
                                    remaining_points_old_points_method_location_in_RV[0] = np.empty(
                                    )

                            if np.max(remaining_points_old_points_method.size) == 0:
                                break

                        else:
                            included_old_points = np.array(
                                [included_old_points, closest_included_old_points])
                            included_old_points_location_in_RV = np.array(
                                [included_old_points_location_in_RV, closest_included_old_points_location_RV])

                        RV_reordering = np.array(
                            [RV_reordering, closest_included_old_points_location_RV])

                        if np.max(closest_included_old_points.size) != 0:
                            remaining_points[remaining_points ==
                                             closest_included_old_points] = np.empty()
                            remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                            closest_included_old_points_location_RV] = np.empty()

            else:
                min_dist_old_points = d_remaining[0]
                closest_old_points = remaining_points_old_points_method[0]
                closest_old_points_location_RV = remaining_points_location_in_RV[0]
                _, closest_point_C_remaining_old_points = np.isin(
                    I_old(C_remaining[0]), I)

                dist_included_old_points = RV[
                    included_old_points_location_in_RV, remaining_points_location_in_RV]
                if np.max(included_old_points_location_in_RV.size) == 1:
                    value1, index1 = min(dist_included_old_points), np.argmin(
                        dist_included_old_points)
                    _, closest_point_C_included_old_points = np.isin(
                        included_old_points, I)
                else:
                    value, index = min(dist_included_old_points), np.argmin(
                        dist_included_old_points)
                    value1, index1 = min(value), np.argmin(value)
                    _, closest_point_C_included_old_points = np.isin(
                        included_old_points[index[index1]], I)

                min_dist_included_old_points = value1
                closest_included_old_points_location_RV = remaining_points_location_in_RV[
                    index1]
                closest_included_old_points = remaining_points[index1]

                if min_dist_included_old_points.size == 0:
                    min_dist_all = min_dist_old_points
                    min_dist_method = 1
                else:
                    [min_dist_all, min_dist_method] = min(
                        np.array([min_dist_old_points, np.inf, min_dist_included_old_points]))
                if min_dist_method == 0:
                    I = np.array([I, closest_old_points])
                    d = np.array([d, min_dist_old_points])
                    C = np.array([C, closest_point_C_remaining_old_points])
                    RV_reordering = np.array(
                        [RV_reordering, closest_old_points_location_RV])
                    remaining_points[remaining_points ==
                                     closest_old_points] = np.empty()
                    remaining_points_old_points_method[remaining_points_old_points_method == closest_old_points] = np.empty(
                    )
                    remaining_points_old_points_method_location_in_RV[
                        remaining_points_old_points_method_location_in_RV == closest_old_points_location_RV] = np.empty()
                    remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                    closest_old_points_location_RV] = np.empty()
                    pointer_last_point = pointer_last_point+1
                    d_remaining[0] = np.empty()
                    C_remaining[0] = np.empty()
                    if np.max(remaining_points_old_points_method.size) > 0:
                        while np.isin(remaining_points_old_points_method[0], I):
                            pointer_last_point = pointer_last_point+1
                            d_remaining[0] = np.empty()
                            C_remaining[0] = np.empty()
                            remaining_points_old_points_method[0] = np.empty()
                            remaining_points_old_points_method_location_in_RV[0] = np.empty(
                            )
                            if max(remaining_points_old_points_method.size) == 0:
                                break

                if min_dist_method == 1:
                    print('Error')
                    break
                if min_dist_method == 2:
                    I = np.array([I, closest_included_old_points])
                    d = np.array([d, min_dist_included_old_points])
                    C = np.array([C, closest_point_C_included_old_points])
                    if closest_included_old_points == remaining_points[0]:
                        if np.max(remaining_points_old_points_method.size) > 0:
                            while np.isin(remaining_points_old_points_method[0], I):
                                pointer_last_point = pointer_last_point+1
                                d_remaining[0] = np.empty()
                                C_remaining[0] = np.empty()

                                included_old_points[included_old_points ==
                                                    remaining_points_old_points_method[0]] = np.empty()
                                included_old_points_location_in_RV[included_old_points_location_in_RV ==
                                                                   remaining_points_old_points_method_location_in_RV[0]] = np.empty()

                                remaining_points_old_points_method[0] = np.empty(
                                )
                                remaining_points_old_points_method_location_in_RV[0] = np.empty(
                                )

                        if np.max(remaining_points_old_points_method.shape) == 0:
                            break

                    else:
                        included_old_points = np.array(
                            [included_old_points, closest_included_old_points])
                        included_old_points_location_in_RV = np.array(
                            [included_old_points_location_in_RV, closest_included_old_points_location_RV])

                    RV_reordering = np.array(
                        [RV_reordering, closest_included_old_points_location_RV])

                    if np.max(closest_included_old_points.size) != 0:
                        remaining_points[remaining_points ==
                                         closest_included_old_points] = np.empty()
                        remaining_points_location_in_RV[remaining_points_location_in_RV ==
                                                        closest_included_old_points_location_RV] = np.empty()

        RV = RV[RV_reordering, RV_reordering]
        _, RI = np.sort(I), np.argsort(I)

    return RV, C, I, RI, d
