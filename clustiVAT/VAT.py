import numpy as np


def VAT(mat):
    rows, cols = mat.shape
    K = np.linspace(0, rows-1, rows, dtype=int)
    J = K

    max_val_eachcol, max_val_index_eachcol = np.amax(
        mat, axis=0), np.argmax(mat, axis=0)
    max_val_new, max_val_ind = np.amax(
        max_val_eachcol), np.argmax(max_val_eachcol)
    # np.where(
    #     max_val_eachcol == np.amax(max_val_eachcol))

    I = max_val_index_eachcol[max_val_ind]
    J = np.delete(J, I)

    y = np.min(mat[I, J])
    j = np.argmin(mat[I, J])

    I = np.hstack((I, J[j]))
    J = np.delete(J, J == J[j])

    C = np.array([1, 1])
    cut = np.zeros((rows, 1))
    cut[1] = y

    for r in range(3, rows):
        y, i = np.amin(mat[np.ix_(I, J)], axis=0), np.argmin(
            mat[np.ix_(I, J)], axis=0)
        try:
            y, j = np.amin(y), np.argmin(y)
        except ValueError:
            pass
            
        I = np.hstack((I, J[j]))
        J = np.delete(J, J == J[j])
        C = np.hstack((C, i[j]))
        cut[r-1] = y

    y, i = np.min(mat[np.ix_(I, J)]), np.argmin(np.ix_(mat[I, J]))
    I = np.hstack((I, J))
    C = np.hstack((C, i))
    cut[rows-1] = y

    RI = np.zeros((rows))
    for r in range(0, rows):
        RI[I[r]] = r

    RV = mat[np.ix_(I, I)]

    return RV, C, I, RI, cut


# For debugging purposes
if __name__ == "__main__":
    f = open("rs.txt", "r")
    lst = []
    for line in f:
        strd = line.strip().split(',')
        lst.append([float(x) for x in strd])

    arr = np.array(lst)
    rv, c, i, ri, cut = VAT(arr)
