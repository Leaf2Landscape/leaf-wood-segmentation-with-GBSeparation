import numpy as np
import scipy.sparse
from collections import deque
from typing import Iterable, Dict, List, Set
from GBSeparation.Graph_Path import reconstruct_path

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
    path_list : dict[int -> list[int]] or np.ndarray of int (predecessor array from scipy Dijkstra)
        Dictionary: root-to-node path (inclusive) for each node id.
        Must have length >= 1; for most nodes length >= 2 (to compute predecessor).
    init_wood_ids : iterable[int]
        Initial wood points (node ids).
    G : networkx.Graph or scipy.sparse.csr_matrix (CSR adjacency)
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

    # CSR mode: `path_list` carries the predecessor int array (pred); adjacency
    # comes from CSR rows. NetworkX mode keeps the original dict/path behaviour.
    use_csr = scipy.sparse.issparse(G)
    pred = path_list if use_csr else None

    # ---- Quick sanity check for node labeling ----
    # We assume node ids in G line up with 0..N-1 so arrays index directly by node id.
    # If this is not true in your data, relabel the graph before calling this function:
    # G = nx.convert_node_labels_to_integers(G, ordering='sorted') and reorder pcd/path_dis accordingly.
    if not use_csr:
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
        # full path from root to i; extend the set
        if use_csr:
            initial_set.update(reconstruct_path(pred, i))
        else:
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
    if use_csr:
        indptr = G.indptr
        cols = G.indices
        data = G.data
        for u in range(N):
            s, e = indptr[u], indptr[u + 1]
            if s == e:
                continue
            neighbors[u] = cols[s:e].tolist()
            weights[u] = data[s:e].tolist()
    else:
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
        if use_csr:
            pre = int(pred[i])
            if pre >= 0:
                pre_dis[i] = path_dis_arr[i] - path_dis_arr[pre]
            else:
                pre_dis[i] = 0.0
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