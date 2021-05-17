"""
Calculate external validation. Two partitions should be given.

"""
import math
import numpy as np


def calculate_external(partition_a, partition_b, indices=['all']):

    from collections import defaultdict

    # size of contigency table
    R = len(partition_a)
    C = len(partition_b)

    # contigency table
    contigency_table = np.zeros((R + 1, C + 1))
    # fill the contigency table
    for i in range(0, R + 1):
        for j in range(0, C):
            if i in range(0, R):
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                contigency_table[i][j] = n_common_elements
            else:
                contigency_table[i][j] = contigency_table[:, j].sum()

        contigency_table[i][j + 1] = contigency_table[i].sum()

    N = contigency_table[R][C]
    # condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(contigency_table[0:R][:, range(0, C)] ** 2)
    sum_R_squared = np.sum(contigency_table[0:R, C] ** 2)
    sum_R = np.sum(contigency_table[0:R, C])
    sum_C_squared = np.sum(contigency_table[R, 0:C] ** 2)
    sum_C = np.sum(contigency_table[R, 0:C])
    # computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    for i in range(0, R):
        for j in range(0, C):
            a += contigency_table[i][j] * (contigency_table[i][j] - 1)
    a = a / 2
    # computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = ((N**2) + sum_all_squared - (sum_R_squared + sum_C_squared)) / 2
    # computing the number of pair in different cluster in partition A but in the same cluster in partition B
    c = (sum_C_squared - sum_all_squared) / 2
    # computing the number of pairs in different cluster both in partition A and partition B
    d = (sum_R_squared - sum_all_squared    ) / 2

    M = (a + b + c + d)

    indices_funcs = {'AR': adjusted_rand, 'FM': fowlkes_mallows, 'J': jaccard, 'AW': adjusted_wallace,
                     'VD': van_dongen, 'H': hubert_normalized, 'F': f_measure,
                     'VI': variation_information, 'K': Kulczynski,
                     'Phi': Phi, 'RT': Rogers_Tanimoto, 'SS': Sokal_Sneath}
    results = defaultdict()

    if isinstance(indices, str):
        indices = [x.strip() for x in indices.split(',')]

    for index in indices:
        if index == 'all' or index == 'external':
            for cvi, func in indices_funcs.items():
                if cvi in ['AR', 'AW']:
                    results[cvi] = func(a, b, c, d, sum_C, sum_R_squared, sum_C_squared, sum_all_squared, N)
                elif cvi in ['VD', 'VI']:
                    results[cvi] = func(contigency_table, R, C, N)
                else:
                    results[cvi] = func(a, b, c, d, M)
        elif index in indices_funcs.keys():
            if index in ['AR', 'AW']:
                results[index] = indices_funcs[index](a, b, c, d, sum_C,  sum_R_squared, sum_C_squared, sum_all_squared, N)
            elif index in ['VD', 'VI']:
                results[index] = indices_funcs[index](contigency_table, R, C, N)
            else:
                results[index] = indices_funcs[index](a, b, c, d, M)

    return results



def adjusted_rand(a, b , c, d, sum_C, sum_R_squared, sum_C_squared, sum_all_squared, N):
    nc = ((N*(math.pow(N, 2) + 1)) - ((N + 1)*sum_R_squared) - ((N + 1)*sum_C_squared) + (sum_all_squared / N)) / 2*(N - 1)

    adjusted_rand_index = (a + d - nc) / (a + b + c + d - nc)

    return adjusted_rand_index

def fowlkes_mallows(a, b, c, d, M):
    if ((a + b) == 0 or (a + c) == 0):
        FM = 0
    else:
        FM = a/math.sqrt(((a + b)) * ((a + c)))

    return FM

def jaccard(a, b, c, d, M):
    if (a + b + c == 0):
        jaccard = 1
    else:
        jaccard = a / (a + b + c)

    return jaccard

def adjusted_wallace(a, b, c, d, sum_C, sum_R_squared, sum_C_squared, sum_all_squared, N):
    if ((a + b) == 0):
        wallace = 0
    else:
        wallace = a / (a + b)
    SID_B = 1 - ((sum_C_squared - sum_C) / (N * (N - 1)))
    if ((SID_B) == 0):
        adjusted_wallace_index = 0
    else:
        adjusted_wallace_index = (wallace - (1 - SID_B)) / (1 - (1 - SID_B))

    return adjusted_wallace_index

def van_dongen(contigency_table, R, C, N):

    VD_i = np.sum(max(contigency_table[i, 0:C]) for i in range(R))
    VD_j = np.sum(max(contigency_table[0: R, j]) for j in range(C))

    van_dongen_index = ((2 * N) - VD_i - VD_j)/ (2*N)
    
    return van_dongen_index


def hubert_normalized(a, b, c, d, M):

    aux_hub1 = (a + b) * (a + c)
    aux_hub2 = (a + b) * (a + c) * (d + b) * (d + c)
    if aux_hub2 == 0:
        hub_normalized = float(0)
    else:
        hub_normalized = (M * a - aux_hub1) / (math.sqrt(aux_hub2))

    return hub_normalized

def f_measure(a, b, c, d, M):
    if a + b + c == 0:
        f_measure_index = float(0)
    else:
        f_measure_index = (2*a) / ((2*a) + b + c)

    return f_measure_index

def variation_information(contigency_table, R, C, N):
    mutual_info = 0
    mutual_info_aux = 0
    entropy_C = 0
    entropy_P = 0
    flag_entropy = 0
    for i in range(R):
        # calculation for entropy
        p_i = contigency_table[i][-1] / N
        if p_i != 0:
            entropy_C += (p_i * math.log10(p_i))
        for j in range(C):
            if flag_entropy == 0:
                # calculation for entropy
                p = contigency_table[-1][j] / N
                if p != 0:
                    entropy_P += (p * math.log10(p))
            p_ij = contigency_table[i][j] / N
            p_j = contigency_table[-1][j] / N
            if not 0 in [p_ij, p_i, p_j]:
                mutual_info_aux += (p_ij * math.log10(p_ij / (p_i * p_j)))
        flag_entropy = 1
        mutual_info += mutual_info_aux

    VI = -entropy_C - entropy_P - (2 * mutual_info)

    return VI


def Czekanowski_Dice(a, b, c, d, M):

    return 2*a / 2*a + b + c

def Kulczynski(a, b, c, d, M):

    c = (a / (a + c)) + (a / (a + b))

    return c/2


def Phi(a, b, c, d, M):

    c1 = (a * d) - (b * c)
    c2 = (a + b)*(a + c)*(b + d)*(c + d)

    if c2 == 0:
        c = 0
    else:
        c = c1 / c2
    return c

def Rogers_Tanimoto(a, b, c, d, M):

    return (a + d) / (a + d + 2*(b + c))


def Sokal_Sneath(a,b,c,d,M):

    return a / (a + 2*(b+c))


