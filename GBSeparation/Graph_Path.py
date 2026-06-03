# Copyright (c) 2017-2019, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017-2019, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.3.2"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def add_nodes(G, base_node, indices, distance, threshold):

    """
    Adds a set of nodes and weighted edges based on pairs of indices
    between base_node and all entries in indices. Each node pair shares an
    edge with weight equal to the distance between both nodes.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object to which all nodes/edges will be added.
    base_node : int
        Base node's id to be added. All other nodes will be paired with
        base_node to form different edges.
    indices : list or array
        Set of nodes indices to be paired with base_node.
    distance : list or array
        Set of distances between all nodes in 'indices' and base_node.
    threshold : float
        Edge distance threshold. All edges with distance larger than
        'threshold' will not be added to G.

    """

    for c in np.arange(len(indices)):
        if distance[c] <= threshold:
            # If the distance between vertices is less than a given
            # threshold, add edge (i[0], i[c]) to Graph.
            G.add_weighted_edges_from([(base_node, indices[c],
                                        distance[c])])


def array_to_graph(arr, base_id, kpairs=3, knn=30, nbrs_threshold=0.1,
                   nbrs_threshold_step=0.02, graph_threshold=np.inf, n_jobs=-1):

    """
    Converts a numpy.array of points coordinates into a weighted, symmetric
    scipy.sparse CSR adjacency matrix.

    Mirrors the original NetworkX traversal exactly, but
    accumulates (row, col, weight) triples instead of mutating a NetworkX
    graph. Row/col indices are the point cloud indices (0..N-1); the root is
    the last point. Duplicate edges keep the minimum weight; self-loops are
    skipped.
    """

    N = arr.shape[0]
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            leaf_size=15, n_jobs=n_jobs).fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    indices = indices.astype(np.int32)
    print('NN done')

    rows, cols, weights = [], [], []

    def _add_edges(base_node, nn_idx, dd_idx):
        for i in range(len(nn_idx)):
            v = int(nn_idx[i])
            if v == base_node:
                continue
            if dd_idx[i] <= graph_threshold:
                rows.append(base_node)
                cols.append(v)
                weights.append(float(dd_idx[i]))

    visited = np.zeros(N, dtype=bool)
    current_idx = indices[-1]
    _add_edges(base_id, indices[-1], distances[-1])
    visited[current_idx] = True

    unprocessed_idx = np.flatnonzero(~visited)
    temp_nbrs_threshold = nbrs_threshold
    k = 1
    previous_test = 0
    pbar = tqdm(total=N, desc="Processing points")
    try:
        while unprocessed_idx.shape[0] > 0:
            k += 1
            test = unprocessed_idx.shape[0]
            if k % 10 == 0:
                if test == previous_test:
                    break
                previous_test = test

            if len(current_idx) > 0:
                nn = indices[current_idx]
                dd = distances[current_idx]
                mask1 = ~visited[nn]
                nntemp = []
                for i, g in enumerate(current_idx):
                    nn_idx = nn[i][mask1[i]][0:kpairs+1]
                    dd_idx = dd[i][mask1[i]][0:kpairs+1]
                    nntemp.append(nn_idx)
                    _add_edges(int(g), nn_idx, dd_idx)
                if nntemp and any(len(t) for t in nntemp):
                    current_idx = np.unique(np.concatenate(nntemp)).astype(np.intp)
                else:
                    current_idx = np.empty(0, dtype=np.intp)
                temp_nbrs_threshold = nbrs_threshold
            else:
                unprocessed_nn = indices[unprocessed_idx]
                unprocessed_dd = distances[unprocessed_idx]
                mask1 = visited[unprocessed_nn]
                mask2 = unprocessed_dd < temp_nbrs_threshold
                mask = mask1 & mask2
                temp_idx = np.unique(np.where(mask)[0])
                current_idx = unprocessed_idx[temp_idx]
                nn = indices[current_idx]
                dd = distances[current_idx]
                mask = visited[nn]
                for i, g in enumerate(current_idx):
                    nn_idx = nn[i][mask[i]][0:kpairs+1]
                    dd_idx = dd[i][mask[i]][0:kpairs+1]
                    _add_edges(int(g), nn_idx, dd_idx)
                if len(current_idx) == 0:
                    temp_nbrs_threshold += nbrs_threshold_step

            visited[current_idx] = True
            if len(current_idx) > 0:
                unprocessed_idx = unprocessed_idx[~visited[unprocessed_idx]]
            pbar.update(len(current_idx))
    finally:
        pbar.close()

    # Build symmetric CSR: add both directions, then dedup duplicate (i,j)
    # entries by keeping the minimum weight (scipy's constructor would sum
    # them otherwise).
    #
    # Memory note: avoid `rows + cols` Python list concatenation — for large
    # trees it briefly doubles the Python-object heap before np.array() can
    # free it.  Instead convert each list to numpy first, then np.concatenate
    # (which operates on compact arrays and never materialises a doubled Python
    # list).
    r_fwd = np.array(rows,    dtype=np.int32)
    c_fwd = np.array(cols,    dtype=np.int32)
    w_fwd = np.array(weights, dtype=np.float32)
    del rows, cols, weights          # release Python-object lists immediately

    rows_arr = np.concatenate((r_fwd, c_fwd))   # symmetric: fwd + rev
    cols_arr = np.concatenate((c_fwd, r_fwd))
    wts_arr  = np.concatenate((w_fwd, w_fwd))
    del r_fwd, c_fwd, w_fwd         # release per-direction arrays

    order = np.lexsort((wts_arr, cols_arr, rows_arr))
    r2, c2, d2 = rows_arr[order], cols_arr[order], wts_arr[order]
    del rows_arr, cols_arr, wts_arr, order   # free before CSR allocation

    mask = np.empty(len(r2), dtype=bool)
    if len(r2) > 0:
        mask[0] = True
        mask[1:] = (r2[1:] != r2[:-1]) | (c2[1:] != c2[:-1])
    G = scipy.sparse.csr_matrix((d2[mask], (r2[mask], c2[mask])), shape=(N, N))
    return G


def reconstruct_path(pred, u):
    """Walk predecessor array from u to root (-1). Returns [root, ..., u]."""
    path = []
    while u != -1:
        path.append(int(u))
        u = int(pred[u])
    return path[::-1]


def extract_path_info(G, base_id, return_path=False):

    """
    Extracts shortest path information.

    For a scipy.sparse CSR graph, uses scipy.sparse.csgraph.dijkstra:
      - return_path=True  -> (dist_array, pred_array), both indexed by node id.
        pred is int32 with -1 marking root/unreachable (scipy's -9999 sentinel
        remapped to -1 to match reconstruct_path's terminator).
      - return_path=False -> dist_array.
    """

    if return_path:
        dist, pred = csgraph_dijkstra(
            G, directed=False, indices=base_id,
            return_predecessors=True)
        pred = pred.astype(np.int32)
        pred[pred < 0] = -1
        return dist, pred
    dist = csgraph_dijkstra(G, directed=False, indices=base_id)
    return dist
