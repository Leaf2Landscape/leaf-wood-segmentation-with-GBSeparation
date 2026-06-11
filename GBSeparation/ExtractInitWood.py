import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.sparse
from GBSeparation.Components_classify import getAngle3D
from GBSeparation.Components_classify import components_classify

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not available
    def tqdm(x, **kwargs):
        return x


def extract_init_wood(
    pcd,
    G,
    base_id,
    path_dis: dict,
    path_list: dict,
    split_interval=(0.1, 0.2, 0.3, 0.5, 1.0),
    max_angle=np.pi,
    t_linearity=0.9,
    t_error=0.2,
    min_component_size: int = 1,
    classify_parallel: bool = True,
    max_workers: int = None,
    verbose: bool = True,
):
    """
    Fast & robust wood cluster extraction.

    Pipeline:
      1) Cut edges by distance and angle criteria.
      2) For each split interval:
         - Compute bin-induced components via a single-pass, bin-constrained BFS.
         - (Optionally) parallelize classification of components.

    Parameters
    ----------
    pcd : np.ndarray
        Point cloud positions; indexable by node id (pcd[u]).
    G : networkx.Graph or scipy.sparse.csr_matrix (CSR adjacency)
        Undirected graph; should match pcd/path_dis/path_list node ids.
    base_id : hashable
        Node id to exclude from edge cutting logic.
    path_dis : dict[node -> float]
        Shortest path length to node.
    path_list : dict[node -> list[node]] or np.ndarray of int (predecessor array from scipy Dijkstra)
        Predecessor path list for each node (assumed length >= 2 where used).
    split_interval : sequence of float, default (0.1, 0.2, 0.3, 0.5, 1.0)
        Bin widths for multi-scale segmentation.
    max_angle : float
        Maximum acceptable spatial angle between predecessor vectors.
    t_linearity : float
        Threshold passed into components_classify.
    t_error : float
        Threshold passed into components_classify.
    min_component_size : int, default 1
        Skip components smaller than this size.
    classify_parallel : bool, default True
        If True, classify components in parallel (ThreadPool).
        Use True when components_classify is NumPy-heavy (releases GIL).
    max_workers : int | None
        Workers for the thread pool. Default uses Python's heuristic.
    verbose : bool
        Print timings and counters.

    Returns
    -------
    np.ndarray
        Unique node ids that were classified as wood clusters.
    """

    # CSR mode: path_dis is a distance array indexed by node id and `path_list`
    # carries the predecessor int array (pred). NetworkX mode keeps the old
    # dict-based access for backward compatibility.
    use_csr = scipy.sparse.issparse(G)
    pred = path_list if use_csr else None

    # -----------------------------------------
    # 1) Cut edges by distance and angle guards
    # -----------------------------------------
    print("cut edges...")
    t0 = time.perf_counter()

    if use_csr:
        # Iterate the upper triangle of the symmetric CSR once; build a filtered
        # CSR rather than mutating in place. Removed edges are dropped from both
        # directions so the downstream BFS sees a consistent graph.
        G_coo = G.tocoo()
        keep_mask = np.ones(len(G_coo.data), dtype=bool)
        rr, cc, dd = G_coo.row, G_coo.col, G_coo.data
        removed = 0
        for idx in tqdm(range(len(dd)), total=len(dd), desc="Processing edges"):
            u = int(rr[idx])
            v = int(cc[idx])
            if u == base_id or v == base_id:
                continue
            pre_u = int(pred[u])
            pre_v = int(pred[v])
            if pre_u < 0 or pre_v < 0:
                continue
            pre_u_dis = path_dis[u] - path_dis[pre_u]
            pre_v_dis = path_dis[v] - path_dis[pre_v]
            pre_u_vec = pcd[u] - pcd[pre_u]
            pre_v_vec = pcd[v] - pcd[pre_v]
            if (dd[idx] > 2 * min(pre_u_dis, pre_v_dis)) or (getAngle3D(pre_u_vec, pre_v_vec) > max_angle):
                keep_mask[idx] = False
                removed += 1
        G = scipy.sparse.csr_matrix(
            (dd[keep_mask], (rr[keep_mask], cc[keep_mask])), shape=G.shape)
        G.eliminate_zeros()
        print(f"  removed edges: {removed} in {time.perf_counter() - t0:.3f}s")
    else:
        remove_edge_list = []

        # NOTE: assumes path_list[u] has length >= 2 for all non-base nodes used here.
        for (u, v, d) in tqdm(G.edges(data=True), total=G.number_of_edges(), desc="Processing edges"):
            if u == base_id or v == base_id:
                continue

            pre_u = path_list[u][-2]
            pre_v = path_list[v][-2]

            pre_u_dis = path_dis[u] - path_dis[pre_u]
            pre_v_dis = path_dis[v] - path_dis[pre_v]
            pre_u_vec = pcd[u] - pcd[pre_u]
            pre_v_vec = pcd[v] - pcd[pre_v]

            # NOTE: expects an existing getAngle3D(pre_u_vec, pre_v_vec) -> float
            if (d["weight"] > 2 * min(pre_u_dis, pre_v_dis)) or (getAngle3D(pre_u_vec, pre_v_vec) > max_angle):
                remove_edge_list.append((u, v))

        G.remove_edges_from(remove_edge_list)
        print(f"  removed edges: {len(remove_edge_list)} in {time.perf_counter() - t0:.3f}s")

    # ---------------------------------------------------------
    # Helper: Single-pass, bin-constrained components (per interval)
    # ---------------------------------------------------------
    def components_per_interval(G_, path_dis_: dict, interval: float, min_size: int):
        """
        Compute components such that nodes connect ONLY if they are in the same bin:
            bin_idx[u] = floor(path_dis[u] / interval)

        This runs in O(|V| + |E|) per interval without constructing subgraphs.
        Returns a list of Python sets (node ids).
        """
        if use_csr:
            N = G_.shape[0]
            indptr = G_.indptr
            cols = G_.indices
            bin_idx = np.floor(np.asarray(path_dis_) / interval).astype(np.int64)
            visited = np.zeros(N, dtype=bool)
            comps = []
            for u in range(N):
                if visited[u]:
                    continue
                b = bin_idx[u]
                stack = [u]
                visited[u] = True
                comp = {u}
                while stack:
                    x = stack.pop()
                    for y in cols[indptr[x]:indptr[x + 1]]:
                        y = int(y)
                        if (not visited[y]) and (bin_idx[y] == b):
                            visited[y] = True
                            comp.add(y)
                            stack.append(y)
                if len(comp) >= min_size:
                    comps.append(comp)
            return comps

        # Compute bin index for each node once per interval
        # dict is robust to non-integer node ids, and fast enough
        bin_idx = {u: math.floor(path_dis_[u] / interval) for u in G_.nodes}

        visited = set()
        comps = []

        # Traverse every node once
        for u in G_.nodes:
            if u in visited:
                continue
            b = bin_idx[u]
            # DFS/LIFO
            stack = [u]
            visited.add(u)
            comp = {u}

            while stack:
                x = stack.pop()
                # Traverse neighbors but only remain inside the same bin
                for y in G_[x]:
                    if (y not in visited) and (bin_idx.get(y) == b):
                        visited.add(y)
                        comp.add(y)
                        stack.append(y)

            if len(comp) >= min_size:
                comps.append(comp)

        return comps

    # ---------------------------------------------------------
    # Helper: classify one component (wraps your existing API)
    # ---------------------------------------------------------
    def classify_one(comp, interval):
        """
        components_classify expects a list of components; we pass [comp].
        Return normalized iterable of (label, ids) entries.
        """
        result = components_classify(
            pcd,
            [comp],
            path_list,
            t_linearity=t_linearity,
            t_error=t_error,
            split_interval=interval,
        )
        # Normalize: result may be list-of-[label, ids] or a single pair
        # We assume your original contract: a list of entries even for single input
        return result

    # ------------------------------
    # 2) Multi-scale segmentation
    # ------------------------------
    init_wood_ids = []
    for interval in split_interval:
        print(f"interval: {interval}")

        # (a) Compute bin-induced components in one pass
        t_comp = time.perf_counter()
        components = components_per_interval(G, path_dis, interval, min_component_size)
        print(f"  components: {len(components)} (found in {time.perf_counter() - t_comp:.3f}s)")

        if not components:
            continue

        # (b) Sort by size (largest first helps load balance if you later batch or tweak)
        components.sort(key=len, reverse=True)

        # (c) Classification (parallel or sequential)
        t_cls = time.perf_counter()
        classify_results = []

        if classify_parallel:
            # Threads are the safest & usually fastest when components_classify is NumPy-heavy.
            # If it's pure Python and CPU-bound, consider switching to processes with a top-level
            # initializer to avoid pickling big inputs (see earlier notes).
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(classify_one, comp, interval) for comp in components]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"  Classifying interval {interval}"):
                    classify_results.append(f.result())
        else:
            # Sequential (for debugging or low component counts)
            for comp in tqdm(components, desc=f"  Classifying interval {interval}"):
                classify_results.append(classify_one(comp, interval))

        # (d) Integrate results
        kept = 0
        for res in classify_results:
            # res is expected to be a list of [label, ids] for the single component we passed
            for label, ids in res:
                if label != 0:
                    init_wood_ids.extend(ids)
                    kept += len(ids)

        print(f"  kept ids: {kept} (classification {time.perf_counter() - t_cls:.3f}s)")

    init_wood_ids = np.unique(init_wood_ids)
    print(f"Total wood IDs identified: {len(init_wood_ids)}")
    # Return G alongside init_wood_ids so callers always receive the edge-cut
    # graph. For CSR the cut version is a new object (local rebind inside this
    # function); for NetworkX G was mutated in-place and is the same object.
    return init_wood_ids, G