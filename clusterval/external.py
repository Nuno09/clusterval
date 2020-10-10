"""
Calculate external validation. Two partitions should be given.

"""



def calculate_external(partition_a, partition_b):
    import math
    import numpy as np

    # size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    # contigency table
    ct = np.zeros((R + 1, C + 1))
    # fill the contigency table
    for i in range(0, R + 1):
        for j in range(0, C):
            if i in range(0, R):
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
    for i in range(0, R):
        for j in range(0, C):
            a = a + ct[i][j] * (ct[i][j] - 1)
    a = a / 2
    # computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared - sum_all_squared) / 2
    # computing the number of pair in different cluster in partition A but in the same cluster in partition B
    c = (sum_C_squared - sum_all_squared) / 2
    # computing the number of pairs in different cluster both in partition A and partition B
    d = (N ** 2 + sum_all_squared - (sum_R_squared + sum_C_squared)) / 2

    M = (a + b + c + d)
    #print(M)
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

    # Van Dongen ------> low value is best
    VD_i = 0
    VD_j = 0

    for i in range(0, R):
        VD_i += max(ct[i, 0:C])
    for j in range(0, C):
        VD_j += max(ct[0:R, j])

    van_dongen = ((2*N) - VD_i - VD_j) / (2*N)

    #Huberts statistic
    hubert = (M - 2*b - 2*c) / M

    #Huberts statistics norrmalized
    aux_hub1 = (a+b)*(a+c)
    aux_hub2 = (a+b)*(a+c)*(d+b)*(d+c)
    if aux_hub2 == 0:
        hub_normalized = 0
    else:
        hub_normalized = (M*a - aux_hub1) / (math.sqrt(aux_hub2))

    #F-Measure
    if a + b + c == 0:
        f_measure = 0
    else:
        f_measure = 2*a / (2*a + b + c)

    #Variation of information - lower value is better
    mutual_info = 0
    entropy_C = 0
    entropy_P = 0
    flag_entropy = 0
    for i in range(R):
        # calculation for entropy
        p_i = ct[i][-1] / N
        if p_i != 0:
            entropy_C = entropy_C + (p_i * math.log2(p_i))
        for j in range(C):
            if flag_entropy == 0:
                # calculation for entropy
                p = ct[-1][j] / N
                if p!= 0:
                    entropy_P = entropy_P + (p * math.log2(p))
            p_ij = ct[i][j] / N
            p_j = ct[-1][j] / N
            if not 0 in [p_ij, p_i, p_j]:
                mutual_info = mutual_info + (p_ij * math.log2(p_ij / (p_i*p_j)))
        flag_entropy = 1

    VI = -entropy_C - entropy_P - (2*mutual_info)

    #Minkowski

    if c == 0:
        MS = 0
    else:
        MS = math.sqrt(b + c + 2 * a) / math.sqrt(c)

    return {'R': rand_index, 'AR': adjusted_rand_index, 'FM': FM, 'J': jaccard, 'AW': adjusted_wallace, 'VD': van_dongen,
            'H': hubert, 'H\'': hub_normalized, 'F': f_measure, 'VI': VI, 'MS': MS}


if __name__ == '__main__':
    from sklearn.utils import resample
    from collections import defaultdict
    from statistics import mean
    from tabulate import tabulate
    import math

    dicio_statistics = defaultdict(str)
    indexes = ('rand', 'adjusted', 'FM', 'jaccard', 'adjusted_wallace', 'van_dongen', 'huberts', 'huberts_normalized', 'F-Measure', 'VI', 'Minkowski')

    for index in indexes:
        dicio_statistics[index] = []

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    cluster = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
               [10, 12, 13, 15, 16, 17, 19]]


    for i in range(50):
        boot1 = resample(data, replace=False, n_samples=11)
        boot2 = resample(data, replace=False, n_samples=18)
        boot = [boot1, boot2]
        computed_indexes = calculate_external(cluster, boot)

        # print(computed_indexes)
        for pos, index in enumerate(indexes):
            dicio_statistics[index].append(computed_indexes[pos])

    for k,v in dicio_statistics.items():
        dicio_statistics[k] = mean(v)

    print(tabulate([list(dicio_statistics.values())], headers=list(dicio_statistics.keys())))

