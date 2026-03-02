import math
import numpy as np
from GBSeparation.Components_classify import getAngle3D
from GBSeparation.Components_classify import components_classify
import networkx as nx
from GBSeparation.Visualization import graph_cluster, graph_cluster2, show_clusters
from tqdm import tqdm

def extract_init_wood_old(pcd, G, base_id, path_dis, path_list, split_interval=[0.1,0.2,0.3,0.5,1],
                     max_angle=np.pi):
    """
    Clustering by cut the edges of graph G based on shortest path length and single edge length.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object from which to split.
    path_dis : dictionary
        the key is the point ID, the value is the shortest path length.
    split_interval : float
        split interval on shortest path length.
    max_angle : float
        The max acceptable spatial angle of two vectors.
    Returns
    -------

    """

    # precursor distance/direction-based segmentation.
    print("cut edges...")
    remove_edge_list = []
    for (u, v, d) in tqdm(G.edges(data=True), total=G.number_of_edges(), desc="Processing edges"):
        if (u == base_id or v == base_id):
            continue
        pre_u_dis = path_dis[u] - path_dis[path_list[u][-2]]
        pre_v_dis = path_dis[v] - path_dis[path_list[v][-2]]
        pre_u_vec = pcd[u] - pcd[path_list[u][-2]]
        pre_v_vec = pcd[v] - pcd[path_list[v][-2]]
        if (d['weight'] > 2 * min(pre_u_dis, pre_v_dis)
                or getAngle3D(pre_u_vec, pre_v_vec) > max_angle):
            remove_edge_list.append([u, v])
    G.remove_edges_from(remove_edge_list)

    # multi-scale segmentation.
    interval_dicts = []
    for i in range(len(split_interval)):
        interval_dicts.append({})
    for id, dis in tqdm(path_dis.items(), total=len(path_dis), desc="Building intervals"):
        for i, interval in enumerate(split_interval):
            f = math.floor(dis / interval)
            if f in interval_dicts[i]:
                interval_dicts[i][f].append(id)
            else:
                interval_dicts[i][f] = [id]

    init_wood_ids = []
    for i, interval_dict in enumerate(interval_dicts):
        print(f"interval: {split_interval[i]}")

        def _components_per_interval(G, path_dis, interval):
            """Return bin-induced components for one interval in a single pass."""
            # Compute the bin index for each node once
            # (dict lookup is fine; if nodes are 0..N-1, you can use arrays)
            bin_idx = {u: math.floor(path_dis[u] / interval) for u in G.nodes}
            
            visited = set()
            components = []

            for u in G.nodes:
                if u in visited:
                    continue
                b = bin_idx[u]
                comp = {u}
                visited.add(u)
                stack = [u]

                while stack:
                    x = stack.pop()
                    # Traverse only neighbors that stay in the same bin b
                    for y in G[x]:
                        if y not in visited and bin_idx.get(y) == b:
                            visited.add(y)
                            comp.add(y)
                            stack.append(y)

                components.append(comp)

            return components
        
        components = _components_per_interval(G, path_dis, split_interval[i])
        print(f"components: {len(components)}")

        # recognition of wood clusters with linear/cylindrical shape in a individual scale.
        classify_components = components_classify(pcd, components, path_list, t_linearity=0.9,
                                                  t_error=0.2, split_interval=split_interval[i])
        for classify_component in classify_components:
            if (classify_component[0] != 0):
                for elm in classify_component[1]:
                    init_wood_ids.append(elm)

    init_wood_ids = np.unique(init_wood_ids)
    return init_wood_ids


import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not available
    def tqdm(x, **kwargs):
        return x


def extract_init_wood(
    pcd,
    G: nx.Graph,
    base_id,
    path_dis: dict,
    path_list: dict,
    split_interval=(0.1, 0.2, 0.3, 0.5, 1.0),
    max_angle=np.pi,
    t_linearity=0.9,
    t_error=0.2,
    min_component_size: int = 2,
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
    G : networkx.Graph
        Undirected graph; should match pcd/path_dis/path_list node ids.
    base_id : hashable
        Node id to exclude from edge cutting logic.
    path_dis : dict[node -> float]
        Shortest path length to node.
    path_list : dict[node -> list[node]]
        Predecessor path list for each node (assumed length >= 2 where used).
    split_interval : sequence of float, default (0.1, 0.2, 0.3, 0.5, 1.0)
        Bin widths for multi-scale segmentation.
    max_angle : float
        Maximum acceptable spatial angle between predecessor vectors.
    t_linearity : float
        Threshold passed into components_classify.
    t_error : float
        Threshold passed into components_classify.
    min_component_size : int, default 2
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

    # -----------------------------------------
    # 1) Cut edges by distance and angle guards
    # -----------------------------------------
    print("cut edges...")
    t0 = time.perf_counter()
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
    def components_per_interval(G_: nx.Graph, path_dis_: dict, interval: float, min_size: int):
        """
        Compute components such that nodes connect ONLY if they are in the same bin:
            bin_idx[u] = floor(path_dis[u] / interval)

        This runs in O(|V| + |E|) per interval without constructing subgraphs.
        Returns a list of Python sets (node ids).
        """
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
    return init_wood_ids