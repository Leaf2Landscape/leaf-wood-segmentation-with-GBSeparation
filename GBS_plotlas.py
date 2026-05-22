import os
import sys
import argparse
import numpy as np
import laspy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood

# LAS standard class 2 = ground; sentinel value written for ground / failed / unprocessed points.
GROUND_CLASS = 2
SENTINEL = 255
MIN_COMPONENT_POINTS = 10


def build_groups(las):
    """Pick a grouping strategy by priority and return {key: global_indices_array}."""
    dim_names = [dim.name for dim in las.point_format.dimensions]
    n_points = len(las.x)

    if 'component_id' in dim_names:
        keys = np.asarray(las['component_id'])
    elif 'tree_id' in dim_names and 'stem_id' in dim_names:
        tree = np.asarray(las['tree_id'])
        stem = np.asarray(las['stem_id'])
        keys = np.stack((tree, stem), axis=1)
    elif 'tree_id' in dim_names:
        keys = np.asarray(las['tree_id'])
    else:
        print("Error: no grouping dimension found. Need 'component_id', 'tree_id', "
              "or 'tree_id'+'stem_id'.")
        print("Available dimensions: " + ", ".join(dim_names))
        sys.exit(1)

    all_idx = np.arange(n_points)
    groups = {}
    if keys.ndim == 1:
        for value in np.unique(keys):
            groups[int(value)] = all_idx[keys == value]
    else:
        for value in np.unique(keys, axis=0):
            mask = np.all(keys == value, axis=1)
            groups[(int(value[0]), int(value[1]))] = all_idx[mask]
    return groups


def grouping_strategy_name(las):
    dim_names = [dim.name for dim in las.point_format.dimensions]
    if 'component_id' in dim_names:
        return "component_id"
    if 'tree_id' in dim_names and 'stem_id' in dim_names:
        return "(tree_id, stem_id)"
    if 'tree_id' in dim_names:
        return "tree_id"
    return None


def _gbs_worker(args):
    """args = (xyz_array_float32, global_indices_array)"""
    xyz_array, global_indices = args
    if len(global_indices) < MIN_COMPONENT_POINTS:
        print("Warning: component with %d points (< %d) skipped."
              % (len(global_indices), MIN_COMPONENT_POINTS))
        return global_indices, np.full(len(global_indices), SENTINEL, dtype=np.uint8)

    try:
        xyz = np.asarray(xyz_array, dtype=np.float32)
        treeHeight = np.max(xyz[:, 2]) - np.min(xyz[:, 2])
        root, _ = getRootPt(xyz, lower_h=0.0, upper_h=0.2)
        xyz = np.append(xyz, root, axis=0)
        root_id = xyz.shape[0] - 1
        G = array_to_graph(xyz, root_id, kpairs=3, knn=300,
                           nbrs_threshold=treeHeight / 30,
                           nbrs_threshold_step=treeHeight / 60)
        path_dis, path_list = extract_path_info(G, root_id, return_path=True)
        init_wood_ids = extract_init_wood(xyz, G, root_id, path_dis, path_list,
                                          split_interval=[0.1, 0.2, 0.3, 0.5, 1],
                                          max_angle=0.15 * np.pi)
        final_wood_mask = extract_final_wood(xyz, root_id, path_dis, path_list,
                                             init_wood_ids, G)
        final_wood_mask[-1] = False
        # pipeline returns True=wood; output convention is 1=leaf, 0=wood.
        leaf_wood = (~final_wood_mask[:-1]).astype(np.uint8)
        return global_indices, leaf_wood
    except Exception as exc:
        print("Warning: component failed (%d points): %s"
              % (len(global_indices), exc))
        return global_indices, np.full(len(global_indices), SENTINEL, dtype=np.uint8)


def run_dry_run(las, args):
    print("Input file: %s (readable)" % args.input_file)
    dim_names = [dim.name for dim in las.point_format.dimensions]
    print("Dimensions present: " + ", ".join(dim_names))

    strategy = grouping_strategy_name(las)
    if strategy is None:
        print("Error: no grouping dimension found. Need 'component_id', 'tree_id', "
              "or 'tree_id'+'stem_id'.")
        sys.exit(1)
    print("Grouping strategy: %s" % strategy)

    classification = np.asarray(las.classification)
    ground_mask = classification == GROUND_CLASS

    groups = build_groups(las)
    print("Components found: %d" % len(groups))
    for key in sorted(groups, key=lambda k: str(k)):
        print("  %s: %d points" % (key, len(groups[key])))

    print("Ground points (Classification == %d): %d" % (GROUND_CLASS, int(np.sum(ground_mask))))
    print("Output path that would be written: %s" % args.output_file)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Run GBSeparation leaf/wood classification on a forest-plot .las/.laz file.")
    parser.add_argument("--input_file", required=True, help="path to input .las/.laz")
    parser.add_argument("--output_file", required=True, help="path to output .las/.laz")
    parser.add_argument("--dry_run", action="store_true",
                        help="inspect input and exit without writing any files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="number of parallel processes (default: os.cpu_count())")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the 'leaf_wood' dimension if it already exists")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("Error: input file not found or not readable: %s" % args.input_file)
        sys.exit(1)
    try:
        las = laspy.read(args.input_file)
    except Exception as exc:
        print("Error: could not read input file: %s" % exc)
        sys.exit(1)

    if args.dry_run:
        run_dry_run(las, args)

    dim_names = [dim.name for dim in las.point_format.dimensions]
    if 'leaf_wood' in dim_names:
        if not args.overwrite:
            print("Error: 'leaf_wood' dimension already exists. Use --overwrite to replace it.")
            sys.exit(1)

    n_points = len(las.x)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    classification = np.asarray(las.classification)
    ground_mask = classification == GROUND_CLASS

    groups = build_groups(las)
    # Ground filter wins over component membership: drop ground points from each group.
    non_ground = ~ground_mask
    groups = {key: idx[non_ground[idx]] for key, idx in groups.items()}
    groups = {key: idx for key, idx in groups.items() if len(idx) > 0}

    leaf_wood_out = np.full(n_points, SENTINEL, dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_gbs_worker, (xyz[idx], idx)): key for key, idx in groups.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing components"):
            global_idx, lw = future.result()
            leaf_wood_out[global_idx] = lw

    new_las = laspy.LasData(header=las.header)
    new_las.points = las.points.copy()
    if 'leaf_wood' not in dim_names:
        leaf_wood_dim = laspy.ExtraBytesParams(
            name="leaf_wood", type=np.uint8,
            description="1=leaf, 0=wood, 255=ground/unprocessed")
        new_las.add_extra_dims([leaf_wood_dim])
    new_las.leaf_wood = leaf_wood_out
    new_las.write(args.output_file)


if __name__ == '__main__':
    main()
