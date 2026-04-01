import numpy as np
from tqdm import tqdm

def extract_final_wood_old(pcd, base_id, path_dis, path_list, init_wood_ids, G, max_iter=100):

    """
    Final wood points extraction by region growth.

    Parameters
    ----------
    pcd : array
        Three-dimensional point cloud of a single tree.
    base_id : int
        Index of base id (root) in the graph.
    path_dis : list
        Shortest path distance from all nodes in G to root node.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        root node.
    init_wood_ids : list
        Index of init wood points.
    G : networkx graph
        The original graph construed on single tree point cloud.
    max_iter : int
        The max number of iterations in growing.

    Returns
    -------
    wood_mask : array
        Boolean mask where 'True' represents wood points.

    """

    print("Starting final wood extraction...")
    
    # detect every int_wood_ids shortest path toward root.
    temp_ids = []
    for i in tqdm(init_wood_ids, desc="Building initial paths"):
        for ids in path_list[i]:
            temp_ids.append(ids)
    current_idx = np.unique(temp_ids)
    print(f"Initial wood points identified: {len(current_idx)}")

    final_wood_mask = np.zeros(pcd.shape[0], dtype=bool)
    final_wood_mask[current_idx] = True

    # Looping while there are still indices in current_idx to process.
    iteration = 0
    print("Starting region growth iterations...")
    while (len(current_idx) > 0) & (iteration < max_iter):
        temp_idx = []
        for i in current_idx:
            for key, value in G[i].items():
                if(final_wood_mask[key] == False
                        and path_dis[key] <= path_dis[i]):
                    final_wood_mask[key] = True
                    temp_idx.append(key)
        current_idx = np.array(temp_idx)
        iteration += 1
        print(f"Iteration {iteration}: Added {len(temp_idx)} new points")

    print("Applying neighborhood smoothing...")
    idx_base = np.arange(pcd.shape[0], dtype=int)
    wood_ids = idx_base[final_wood_mask]
    for i in tqdm(wood_ids, desc="Smoothing"):
        if (i == base_id):
            continue
        pre_dis = path_dis[i] - path_dis[path_list[i][-2]]
        for key, value in G[i].items():
            if (final_wood_mask[key] == False
                    and value['weight'] < 2 * pre_dis):
                final_wood_mask[key] = True

    print("Extracting tree stump points...")
    stump = []
    for key, value in G[base_id].items():
        stump.append(key)
    for i, (point) in enumerate(pcd):
        if (point[2] < pcd[base_id][2]):
            stump.append(i)
    final_wood_mask[stump] = True
    print(f"Stump points added: {len(stump)}")

    total_wood = np.sum(final_wood_mask)
    print(f"Final wood extraction complete. Total wood points: {total_wood}")

    return final_wood_mask

import numpy as np
from collections import deque
from itertools import chain
from typing import Iterable, Dict, List, Set, Hashable

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is unavailable
    def tqdm(x, **kwargs):
        return x


def extract_final_wood(
    pcd: np.ndarray,
    base_id: int,
    path_dis,
    path_list: Dict[int, List[int]],
    init_wood_ids: Iterable[int],
    G,
    max_iter: int = 100,              # kept for API compatibility (no longer the bottleneck)
    verbose: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Final wood points extraction by region growth (fast version).

    Improvements:
      • Uses a single-queue BFS region growth (each node processed at most once).
      • Precomputes predecessor step distance (pre_dis) once for smoothing pass.
      • Pre-extracts adjacency lists (neighbors + weights) for tight loops.
      • Uses set-union for initial path expansion.

    Parameters
    ----------
    pcd : (N, 3) np.ndarray
        Three-dimensional point cloud of a single tree.
    base_id : int
        Index of base id (root) in the graph.
    path_dis : dict[int->float] or 1D np.ndarray
        Shortest path distance from all nodes in G to root node.
    path_list : dict[int -> list[int]]
        Dictionary: root-to-node path (inclusive) for each node id.
        Must have length >= 1; for most nodes length >= 2 (to compute predecessor).
    init_wood_ids : iterable[int]
        Initial wood points (node ids).
    G : networkx.Graph
        Graph built on the single-tree point cloud.
        Assumes nodes correspond 1:1 with row indices in `pcd`.
    max_iter : int
        Kept for API compatibility. BFS growth no longer needs it (terminates when queue empties).
    verbose : bool
        Print key milestones and counts.
    show_progress : bool
        Show tqdm bars for long phases.

    Returns
    -------
    final_wood_mask : (N,) np.ndarray of bool
        Boolean mask where True represents wood points.
    """

    N = pcd.shape[0]

    # ---- Quick sanity check for node labeling ----
    # We assume node ids in G line up with 0..N-1 so arrays index directly by node id.
    # If this is not true in your data, relabel the graph before calling this function:
    # G = nx.convert_node_labels_to_integers(G, ordering='sorted') and reorder pcd/path_dis accordingly.
    try:
        nodes = list(G.nodes)
        if len(nodes) != N or min(nodes) != 0 or max(nodes) != N - 1:
            print("Warning: G nodes do not appear to be 0..N-1. Ensure pcd, path_dis, path_list match G's labeling.")
    except Exception:
        pass

    # ---- Coerce path_dis to a fast indexable array ----
    if isinstance(path_dis, dict):
        path_dis_arr = np.zeros(N, dtype=float)
        for u, d in path_dis.items():
            path_dis_arr[u] = d
    else:
        # already an array / list aligned with node ids
        path_dis_arr = np.asarray(path_dis, dtype=float)
        if path_dis_arr.shape[0] != N:
            raise ValueError("path_dis length does not match pcd.shape[0]")

    # -----------------------------
    # 1) Build initial index set
    # -----------------------------
    print("Starting final wood extraction...")
    # All nodes on shortest paths for each init wood id (set-union is faster than list+unique)
    if show_progress:
        iterator = tqdm(init_wood_ids, desc="Building initial paths")
    else:
        iterator = init_wood_ids

    initial_set: Set[int] = set()
    for i in iterator:
        # path_list[i] is the full path from root to i; extend the set
        initial_set.update(path_list[i])

    current_idx = np.fromiter(initial_set, dtype=np.int64, count=len(initial_set))
    print(f"Initial wood points identified: {current_idx.size}")

    # Initialize mask and queue for BFS region growth
    final_wood_mask = np.zeros(N, dtype=bool)
    if current_idx.size > 0:
        final_wood_mask[current_idx] = True

    # -----------------------------
    # 2) Pre-extract adjacency once
    # -----------------------------
    # neighbors[u] : list of neighbor node ids
    # weights[u]   : list of edge weights aligned with neighbors[u]
    # This removes attribute dict lookups from hot loops.
    neighbors: List[List[int]] = [[] for _ in range(N)]
    weights: List[List[float]] = [[] for _ in range(N)]
    for u in range(N):
        # G[u] is adjacency dict: neighbor -> edge_attr_dict
        # Keep both neighbor id and its 'weight'
        adu = G[u]
        if not adu:
            continue
        # Note: assume 'weight' exists for all edges (as used downstream)
        nbs = []
        wts = []
        for v, d in adu.items():
            nbs.append(v)
            wts.append(d.get("weight", 1.0))
        neighbors[u] = nbs
        weights[u] = wts

    # ---------------------------------
    # 3) Region growth with a single BFS
    # ---------------------------------
    print("Starting region growth (BFS)...")
    # Each node will be enqueued at most once (when first turned True)
    q = deque(int(u) for u in current_idx)

    # We preserve the same rule: expand to neighbor `key` if
    #   final_wood_mask[key] == False and path_dis[key] <= path_dis[i]
    # Using BFS eliminates the need for multiple global iterations.
    steps = 0
    added_total = 0
    while q:
        i = q.popleft()
        pi = path_dis_arr[i]
        # expand neighbors
        for key in neighbors[i]:
            if not final_wood_mask[key] and path_dis_arr[key] <= pi:
                final_wood_mask[key] = True
                q.append(key)
                added_total += 1
        steps += 1

    print(f"Region growth done. Visited seeds: {steps}, newly added: {added_total}")

    # ----------------------------------------------
    # 4) Precompute predecessor step distances (pre_dis)
    # ----------------------------------------------
    # pre_dis[i] = path_dis[i] - path_dis[ predecessor(i) ]
    # For base_id or nodes without a predecessor (path length < 2), set to 0.
    pre_dis = np.zeros(N, dtype=float)
    # Only compute for nodes with a valid predecessor
    for i in range(N):
        if i == base_id:
            continue
        # Be defensive: path_list[i] may be size 1, but in the usual pipeline it’s >=2
        pl = path_list[i]
        if len(pl) >= 2:
            pre = pl[-2]
            pre_dis[i] = path_dis_arr[i] - path_dis_arr[pre]
        else:
            pre_dis[i] = 0.0

    # ----------------------------------------------
    # 5) Neighborhood smoothing (single pass)
    # ----------------------------------------------
    print("Applying neighborhood smoothing...")
    wood_ids = np.flatnonzero(final_wood_mask)
    if show_progress:
        iterator = tqdm(wood_ids, desc="Smoothing")
    else:
        iterator = wood_ids

    added_smooth = 0
    twice_pre_dis = 2.0 * pre_dis  # vector for fast lookup
    for i in iterator:
        if i == base_id:
            continue
        thr = twice_pre_dis[i]
        # Consider neighbors if edge weight < 2 * pre_dis[i]
        # (Only flip those not already True)
        nb = neighbors[i]
        wt = weights[i]
        for k, w in zip(nb, wt):
            if not final_wood_mask[k] and w < thr:
                final_wood_mask[k] = True
                added_smooth += 1

    print(f"Smoothing added: {added_smooth} points")

    # ----------------------------------------------
    # 6) Extract tree stump points
    # ----------------------------------------------
    print("Extracting tree stump points...")
    stump = []

    # (a) neighbors of base_id
    stump.extend(neighbors[base_id])

    # (b) points below the base's z
    base_z = pcd[base_id, 2]
    below = np.flatnonzero(pcd[:, 2] < base_z)
    stump.extend(below.tolist())

    # Update mask
    if stump:
        final_wood_mask[np.asarray(stump, dtype=int)] = True

    print(f"Stump points added: {len(stump)}")

    total_wood = int(final_wood_mask.sum())
    print(f"Final wood extraction complete. Total wood points: {total_wood}")

    return final_wood_mask