import os
import sys
import argparse
import multiprocessing as mp
import numpy as np
import laspy
from tqdm import tqdm
from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood

GROUND_CLASS = 2
_LW_DTYPE = np.int8   # 1=leaf, 0=wood, -1=other/ground/failed
SENTINEL = np.int8(-1)
MIN_COMPONENT_POINTS = 10
_MAX_POOL_RETRIES = 5


def build_groups(las):
    """
    Pick a grouping strategy and return:
      (groups {key: global_indices}, strategy_name, other_mask)

    other_mask is True for points that should not be processed:
      - component_id <= 0 (non-vegetation / unclassified segments)
      - all other strategies: other_mask is all-False (caller adds ground on top)
    """
    dim_names = [dim.name for dim in las.point_format.dimensions]
    n_points = len(las.x)
    all_idx = np.arange(n_points)
    other_mask = np.zeros(n_points, dtype=bool)

    if 'component_id' in dim_names:
        component_ids = np.asarray(las['component_id'])
        other_mask = component_ids <= 0
        groups = {}
        for value in np.unique(component_ids[~other_mask]):
            groups[int(value)] = all_idx[component_ids == value]
        return groups, 'component_id', other_mask

    if 'tree_id' in dim_names and 'stem_id' in dim_names:
        tree = np.asarray(las['tree_id'])
        stem = np.asarray(las['stem_id'])
        keys = np.stack((tree, stem), axis=1)
        groups = {}
        for value in np.unique(keys, axis=0):
            mask = np.all(keys == value, axis=1)
            groups[(int(value[0]), int(value[1]))] = all_idx[mask]
        return groups, '(tree_id, stem_id)', other_mask

    if 'tree_id' in dim_names:
        keys = np.asarray(las['tree_id'])
        groups = {}
        for value in np.unique(keys):
            groups[int(value)] = all_idx[keys == value]
        return groups, 'tree_id', other_mask

    print("Error: no grouping dimension found. Need 'component_id', 'tree_id', "
          "or 'tree_id'+'stem_id'.")
    print("Available dimensions: " + ", ".join(dim_names))
    sys.exit(1)


def _make_output_header(las, overwrite):
    """Build output LasHeader with foliage_type (int8) extra dim."""
    dim_names = [dim.name for dim in las.point_format.dimensions]
    if 'foliage_type' in dim_names and not overwrite:
        print("Error: 'foliage_type' dimension already exists. Use --overwrite to replace it.")
        sys.exit(1)

    existing_extra = [
        laspy.ExtraBytesParams(name=d.name, type=d.type)
        for d in las.point_format.extra_dims
        if d.name != 'foliage_type'
    ]
    existing_extra.append(laspy.ExtraBytesParams(
        name="foliage_type", type=np.int8,
        description="1=leaf, 0=wood, -1=other/ground/failed"))

    new_header = laspy.LasHeader(
        point_format=las.header.point_format,
        version=las.header.version,
        extra_dims=existing_extra,
    )
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    return new_header


def _write_chunk(writer, src_points, src_fmt, indices, lw_values, out_fmt):
    """Write a subset of source points with foliage_type values to a LasWriter."""
    sub = src_points[indices]
    buf = np.zeros(len(indices), dtype=out_fmt.dtype)
    for name in src_fmt.dimension_names:
        if name in buf.dtype.names:
            buf[name] = sub[name]
    buf['foliage_type'] = lw_values
    writer.write_points(laspy.PackedPointRecord(buf, out_fmt))


def _gbs_worker(args):
    """args = (task_id, xyz_float32, global_indices)"""
    task_id, xyz_array, global_indices = args
    if len(global_indices) < MIN_COMPONENT_POINTS:
        print("Warning: component with %d points (< %d) skipped."
              % (len(global_indices), MIN_COMPONENT_POINTS))
        return task_id, global_indices, np.full(len(global_indices), SENTINEL, dtype=_LW_DTYPE)
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
        # pipeline: True=wood; output: 1=leaf, 0=wood
        foliage_type = (~final_wood_mask[:-1]).astype(_LW_DTYPE)
        return task_id, global_indices, foliage_type
    except Exception as exc:
        print("Warning: component failed (%d points): %s" % (len(global_indices), exc))
        return task_id, global_indices, np.full(len(global_indices), SENTINEL, dtype=_LW_DTYPE)


def run_dry_run(las, args):
    print("Input file: %s (readable)" % args.input_file)
    dim_names = [dim.name for dim in las.point_format.dimensions]
    print("Dimensions present: " + ", ".join(dim_names))

    groups, strategy, other_mask = build_groups(las)
    print("Grouping strategy: %s" % strategy)

    classification = np.asarray(las.classification)
    ground_mask = classification == GROUND_CLASS

    print("Components found: %d" % len(groups))
    for key in sorted(groups, key=lambda k: str(k)):
        print("  %s: %d points" % (key, len(groups[key])))

    print("Ground points (Classification == %d): %d" % (GROUND_CLASS, int(np.sum(ground_mask))))
    n_other = int(np.sum(other_mask & ~ground_mask))
    if n_other:
        print("Other non-vegetation points (component_id <= 0): %d" % n_other)
    print("Output path that would be written: %s" % args.output_file)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Run GBSeparation leaf/wood classification on a forest-plot .las/.laz file.")
    parser.add_argument("--input_file", required=True, help="path to input .las/.laz")
    parser.add_argument("--output_file", required=True, help="path to output .las/.laz")
    parser.add_argument("--dry_run", action="store_true",
                        help="inspect input and exit without writing any files")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8),
                        help="number of parallel processes (default: min(cpu_count, 8))")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the 'foliage_type' dimension if it already exists in the input")
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

    new_header = _make_output_header(las, args.overwrite)
    out_fmt = new_header.point_format
    src_fmt = las.point_format

    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    classification = np.asarray(las.classification)
    ground_mask = classification == GROUND_CLASS

    groups, _strategy, other_mask = build_groups(las)
    # Ground wins over component membership.
    other_mask = other_mask | ground_mask
    non_other = ~other_mask
    groups = {key: idx[non_other[idx]] for key, idx in groups.items()}
    groups = {key: idx for key, idx in groups.items() if len(idx) > 0}

    # Largest components first: processed while workers are memory-fresh.
    sorted_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    tasks = [(i, xyz[idx], idx) for i, (_, idx) in enumerate(sorted_items)]

    other_indices = np.where(other_mask)[0]

    # Stream results to output as they arrive. Point order in the output differs
    # from the input: other/ground first, then components in completion order.
    done_ids: set = set()
    with laspy.LasWriter(args.output_file, header=new_header) as writer:
        if len(other_indices):
            other_lw = np.full(len(other_indices), SENTINEL, dtype=_LW_DTYPE)
            _write_chunk(writer, las.points, src_fmt, other_indices, other_lw, out_fmt)

        with tqdm(total=len(tasks), desc="Processing components") as pbar:
            remaining = list(tasks)
            for attempt in range(_MAX_POOL_RETRIES):
                try:
                    with mp.Pool(processes=min(args.workers, len(remaining)),
                                 maxtasksperchild=10) as pool:
                        for task_id, global_idx, lw in pool.imap_unordered(
                                _gbs_worker, remaining, chunksize=1):
                            _write_chunk(writer, las.points, src_fmt, global_idx, lw, out_fmt)
                            done_ids.add(task_id)
                            pbar.update()
                    break
                except Exception as exc:
                    remaining = [t for t in remaining if t[0] not in done_ids]
                    if not remaining:
                        break
                    if attempt + 1 < _MAX_POOL_RETRIES:
                        print("\nPool crashed (%s); restarting with %d remaining components."
                              % (exc, len(remaining)), file=sys.stderr)
                    else:
                        print("\nPool crashed %d times; writing %d unprocessed component(s) as %d."
                              % (_MAX_POOL_RETRIES, len(remaining), SENTINEL), file=sys.stderr)

            # Guarantee all points appear in the output even after repeated crashes.
            unprocessed = [t for t in tasks if t[0] not in done_ids]
            for task_id, _, idx in unprocessed:
                _write_chunk(writer, las.points, src_fmt, idx,
                             np.full(len(idx), SENTINEL, dtype=_LW_DTYPE), out_fmt)
                pbar.update()


if __name__ == '__main__':
    main()
