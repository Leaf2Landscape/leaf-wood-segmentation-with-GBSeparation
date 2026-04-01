"""GBSeparation - Graph-Based leaf-wood separation for single tree point clouds.

Based on Tian et al. (2022) - doi: 10.1109/TGRS.2022.3218603
"""

from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood
from GBSeparation.Accuracy_evaluation import evaluate_indicators

__version__ = "0.1.0"


def separate(pcd, lower_h=0.0, upper_h=0.2, kpairs=3, knn=300,
             split_interval=None, max_angle_factor=0.15):
    """Run leaf-wood separation on a single tree point cloud.

    Parameters
    ----------
    pcd : numpy.ndarray
        (N, 3) array of x, y, z coordinates. Growth direction must be parallel to Z.
    lower_h : float
        Relative lower height for root point fitting.
    upper_h : float
        Relative upper height for root point fitting.
    kpairs : int
        Number of point pairs for graph edge construction.
    knn : int
        Number of nearest neighbors for graph construction.
    split_interval : list of float or None
        Multi-scale segmentation intervals. Default: [0.1, 0.2, 0.3, 0.5, 1]
    max_angle_factor : float
        Factor of pi for max angle threshold. Default: 0.15 (i.e. 0.15*pi)

    Returns
    -------
    wood : numpy.ndarray
        (M, 3) array of wood points.
    leaf : numpy.ndarray
        (K, 3) array of leaf points.
    """
    import numpy as np

    if split_interval is None:
        split_interval = [0.1, 0.2, 0.3, 0.5, 1]

    pcd = np.asarray(pcd, dtype=np.float32)
    tree_height = pcd[:, 2].max() - pcd[:, 2].min()

    root, _ = getRootPt(pcd, lower_h=lower_h, upper_h=upper_h)
    pcd = np.append(pcd, root, axis=0)
    root_id = pcd.shape[0] - 1

    G = array_to_graph(pcd, root_id, kpairs=kpairs, knn=knn,
                       nbrs_threshold=tree_height / 30,
                       nbrs_threshold_step=tree_height / 60)

    path_dis, path_list = extract_path_info(G, root_id, return_path=True)

    init_wood_ids = extract_init_wood(pcd, G, root_id, path_dis, path_list,
                                      split_interval=split_interval,
                                      max_angle=max_angle_factor * np.pi)

    final_wood_mask = extract_final_wood(pcd, root_id, path_dis, path_list,
                                         init_wood_ids, G)

    # Remove the inserted root point
    final_wood_mask[-1] = False
    wood = pcd[final_wood_mask]
    final_wood_mask[-1] = True
    leaf = pcd[~final_wood_mask]

    return wood, leaf
