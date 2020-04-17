"""
Calculate external validation indexes

"""



def calculate(partition_a, partition_b):
    import numpy as np
    import math
    # size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    # contigency table
    ct = np.zeros((R + 1, C + 1))
    # fill the contigency table
    for i in range(0, R + 1):
        for j in range(0, C):
            if (i in range(0, R)):
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                ct[i][j] = n_common_elements
            else:
                ct[i][j] = ct[:, j].sum()

        ct[i][j + 1] = ct[i].sum()
    N = ct[R][C]
    # condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(ct[0:R][:, range(0, C)] ** 2)
    sum_R_squared = np.sum(ct[0:R, C] ** 2)
    sum_R = np.sum(ct[0:R, C])
    sum_C_squared = np.sum(ct[R, 0:C] ** 2)
    sum_C = np.sum(ct[R, 0:C])
    # computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    VD_i = 0
    VD_j = 0
    flag_columns = 0
    for i in range(0, R):
        VD_i += max(ct[i, 0:C])
        for j in range(0, C):
            if flag_columns == 0:
                VD_j += max(ct[0:R, j])

            a = a + ct[i][j] * (ct[i][j] - 1)

        flag_columns = 1

    a = a / 2
    # computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared - sum_all_squared) / 2
    # computing the number of pair in different cluster in partition A but in the same cluster in partition B
    c = (sum_C_squared - sum_all_squared) / 2
    # computing the number of pairs in different cluster both in partition A and partition B
    d = (N ** 2 + sum_all_squared - (sum_R_squared + sum_C_squared)) / 2

    # Rand Index
    rand_index = (a + d) / (a + b + c + d)

    # Adjusted Rand Index
    nc = ((sum_R_squared - sum_R) * (sum_C_squared - sum_C)) / (2 * N * (N - 1))
    nd = (sum_R_squared - sum_R + sum_C_squared - sum_C) / 4
    if (nd == nc):
        adjusted_rand_index = 0
    else:
        adjusted_rand_index = (a - nc) / (nd - nc)

    # Fowlks and Mallows
    if ((a + b) == 0 or (a + c) == 0):
        FM = 0
    else:
        FM = a / math.sqrt((a + b) * (a + c))


    # Jaccard
    if (a + b + c == 0):
        jaccard = 1
    else:
        jaccard = a / (a + b + c)

    # Adjusted Wallace
    if ((a + b) == 0):
        wallace = 0
    else:
        wallace = a / (a + b)
    SID_B = 1 - ((sum_C_squared - sum_C) / (N * (N - 1)))
    if ((SID_B) == 0):
        adjusted_wallace = 0
    else:
        adjusted_wallace = (wallace - (1 - SID_B)) / (1 - (1 - SID_B))

    # Van Dongen
    van_dongen = (2 * N - VD_i - VD_j)

    return [rand_index, adjusted_rand_index, FM, jaccard, adjusted_wallace, van_dongen]


if __name__ == '__main__':
    cluster = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
               [10, 12, 13, 15, 16, 17, 19]]
    partition = [[0, 1, 2, 3, 4, 7, 8, 9, 14, 18, 21, 23, 25, 26, 27, 28], [10, 11, 13, 16, 17, 19]]
    print(calculate(cluster, partition))
