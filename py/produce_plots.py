import sys
sys.path.append('../')
from build.pybound.smfuncs import smooth_dijkstra, set_smoothing_factor
from utils.plotting_utils import *
import networkx as nx 

import matplotlib.pyplot as plt

from utils.plot_graphs import *
from utils.comparison_utils import equal_n_of_n, all_permutations, permuation_matrix


def inverse_permutation(perm):
    """ Returns: Permutation that results in the identity when applied to input permutation 'perm' """
    perm = np.stack(perm)
    invperm = np.empty_like(perm)
    invperm[perm] = np.arange(0, len(perm)) 
    return invperm


# pick a smoothing factor for Dijkstra
set_smoothing_factor(2)
    
# construct directed "grid" matrices to showcase algorithm
m = 3
mat1 = np.zeros([m*m, m*m])
mat2 = np.zeros([m*m, m*m])
nodes = np.stack(list(range(m*m)))
inds_per_aligned_row = np.stack([nodes[i*m:i*m+m] for i in range(m)])

across_edges = np.empty([0,2]).astype(int)
# cross edges fill
for i in range(len(inds_per_aligned_row)-1):
    acrosslinks = np.stack([inds_per_aligned_row[i][:-1],inds_per_aligned_row[i+1][1:]]).T
    across_edges = np.append(across_edges, acrosslinks, axis=0)

perpendicular_edges = np.empty([0,2]).astype(int)
# right and downward edges fill
for i in range(len(inds_per_aligned_row)):
    inds_consec = np.stack(list(range(m)))
    indpairs_consec = np.stack([inds_consec[:-1], inds_consec[1:]]).T
    perpendicular_edges = np.append(perpendicular_edges, inds_per_aligned_row[i][indpairs_consec], axis=0)
    perpendicular_edges = np.append(perpendicular_edges, inds_per_aligned_row[:,i][indpairs_consec], axis=0)


# align edges in an m,m grid sequentially
pos_dict = {}
for i in range(m*m):
    row = i//m
    col = i%m
    pos_dict[i] = [col,row]

nfrom_r0 = 0
nto_r0 = m*m-1

for v1,v2 in perpendicular_edges: mat1[v1][v2] = 1
mat2[:] = mat1[:]
for v1,v2 in across_edges: mat2[v1][v2] = 1


for mat in [mat1, mat2]:

    [deriv_mat, deriv_increments_per_run, distances_per_iteration, paths, contributions] = smooth_dijkstra(mat, nfrom_r0, nto_r0)
    deriv_mat = np.stack(deriv_mat)
    deriv_increments_per_run = np.stack(deriv_increments_per_run)

    plot_complete(mat1, deriv_mat, pos_dict=pos_dict, figsize=(14,3))
    plt.subplots_adjust(left=0, right=1, wspace=0)

plt.show()