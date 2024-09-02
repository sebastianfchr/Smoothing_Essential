import numpy as np 

def equal_n_of_n(mats1, mats2):
    # print(len(mats1))
    # print(len(mats2))
    # print("===========")
    assert(len(mats1) == len(mats2))

    num_same = 0
    for i in range(len(mats1)):
        m = mats1[i]
        for j in range(len(mats2)):  
            # print(mats1[i] == mats2[j])
            if( np.allclose(m, mats2[j])):
                num_same += 1
                mats2 = mats2[:j]+mats2[j+1:] # list-op cut out j-th element
                print(i,j)
                break
            
    # print(num_same, "!!", len(mats1), "..", len(mats2))
    print(num_same)
    return num_same == len(mats1)


############### TEMP_REM BEGIN ###############
def all_permutations(arr):
    if len(arr) == 1:
        return [arr] # empty list of results
    else:
        resultList = []
        for i in range(len(arr)):
            resultList += [ [arr[i]] + perm for perm in all_permutations(arr[:i]+arr[i+1:])]
        return resultList
    
############### TEMP_REM END ###############

def permuation_matrix(perm):
    ret = np.zeros((len(perm), len(perm)))
    for (i, p) in zip(range(len(perm)),perm):
        ret[p][i] = 1
    return np.stack(ret)


# =====================================================
def get_path_edges(path):
    taken_edges = [] # take only those on the path
    for i in range(len(path)-1):
        taken_edges.append((i, i+1))
    return taken_edges
    
def filter_matrix_subset(original_mat, path):
    taken_edges = get_path_edges(path)
    for i in range(original_mat.shape[0]):
        for j in range(original_mat.shape[1]):
            original_mat[i,j] = original_mat[i,j] if (i,j) in taken_edges else 0
    
    return original_mat



def extract_edge_sequence(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def inverse_permutation(perm):
    perm = np.stack(perm)
    invperm = np.empty_like(perm)
    invperm[perm] = np.arange(0, len(perm)) # perm gives 0->2? so give it 2->0!
    return invperm
