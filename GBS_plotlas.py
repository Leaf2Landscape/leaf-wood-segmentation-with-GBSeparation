import os
import sys
import argparse
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
_WRITE_CHUNK = 100_000


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
        unique_vals = np.unique(component_ids[~other_mask])
        groups = {}
        for value in tqdm(unique_vals, desc="Indexing components", unit="comp"):
            groups[int(value)] = all_idx[component_ids == value]
        return groups, 'component_id', other_mask

    if 'tree_id' in dim_names and 'stem_id' in dim_names:
        tree = np.asarray(las['tree_id'])
        stem = np.asarray(las['stem_id'])
        keys = np.stack((tree, stem), axis=1)
        unique_vals = np.unique(keys, axis=0)
        groups = {}
        for value in tqdm(unique_vals, desc="Indexing components", unit="comp"):
            mask = np.all(keys == value, axis=1)
            groups[(int(value[0]), int(value[1]))] = all_idx[mask]
        return groups, '(tree_id, stem_id)', other_mask

    if 'tree_id' in dim_names:
        keys = np.asarray(las['tree_id'])
        unique_vals = np.unique(keys)
        groups = {}
        for value in tqdm(unique_vals, desc="Indexing components", unit="comp"):
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

    # Use the integer format ID so LasHeader gets its own fresh PointFormat object
    # rather than sharing (and mutating) the source's PointFormat via object reference.
    new_header = laspy.LasHeader(
        point_format=las.header.point_format.id,
        version=las.header.version,
    )
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    extra_dims = [
        laspy.ExtraBytesParams(name=d.name, type=d.dtype)
        for d in las.point_format.extra_dimensions
        if d.name != 'foliage_type'
    ]
    extra_dims.append(laspy.ExtraBytesParams(
        name="foliage_type", type=np.int8,
        description="1=leaf,0=wood,-1=other"))
    new_header.add_extra_dims(extra_dims)
    return new_header


def _write_chunk(writer, src_points, indices, lw_values, out_fmt):
    """Write a subset of source points with foliage_type values to a LasWriter."""
    sub = src_points[indices]
    buf = np.zeros(len(indices), dtype=out_fmt.dtype())
    # Use the raw underlying array dtype names (bit_fields, classification_flags, etc.)
    # rather than the decoded PointFormat dimension names, which differ and may be mutated.
    src_raw = sub.array
    for name in src_raw.dtype.names:
        if name in buf.dtype.names:
            buf[name] = src_raw[name]
    buf['foliage_type'] = lw_values
    writer.write_points(laspy.PackedPointRecord(buf, out_fmt))


def _gbs_worker(task_id, global_indices, xyz):
    if len(global_indices) < MIN_COMPONENT_POINTS:
        print("Warning: component with %d points (< %d) skipped."
              % (len(global_indices), MIN_COMPONENT_POINTS))
        return task_id, global_indices, np.full(len(global_indices), SENTINEL, dtype=_LW_DTYPE)
    try:
        xyz = xyz[global_indices].astype(np.float32)
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
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the 'foliage_type' dimension if it already exists in the input")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("Error: input file not found or not readable: %s" % args.input_file)
        sys.exit(1)

    file_size_gb = os.path.getsize(args.input_file) / 1e9
    print("Reading %.2f GB: %s" % (file_size_gb, args.input_file))
    try:
        las = laspy.read(args.input_file)
    except Exception as exc:
        print("Error: could not read input file: %s" % exc)
        sys.exit(1)
    n_pts = len(las.x)
    print("  Loaded %s points." % f"{n_pts:,}")

    if args.dry_run:
        run_dry_run(las, args)

    new_header = _make_output_header(las, args.overwrite)
    out_fmt = new_header.point_format

    print("Extracting XYZ coordinates ...")
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    classification = np.asarray(las.classification)
    ground_mask = classification == GROUND_CLASS

    groups, strategy, other_mask = build_groups(las)
    # Ground wins over component membership.
    other_mask = other_mask | ground_mask
    non_other = ~other_mask
    groups = {key: idx[non_other[idx]] for key, idx in groups.items()}
    groups = {key: idx for key, idx in groups.items() if len(idx) > 0}

    n_veg = int(np.sum(non_other))
    n_ground = int(np.sum(ground_mask))
    n_unclassified = int(np.sum(other_mask)) - n_ground
    print("Grouping strategy : %s" % strategy)
    print("Components        : %d" % len(groups))
    print("Vegetation pts    : %s" % f"{n_veg:,}")
    print("Ground pts        : %s" % f"{n_ground:,}")
    if n_unclassified > 0:
        print("Unclassified pts  : %s" % f"{n_unclassified:,}")

    sorted_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    tasks = [(i, idx) for i, (_, idx) in enumerate(sorted_items)]

    other_indices = np.where(other_mask)[0]
    non_other_indices = np.where(non_other)[0]

    # Extract veg_xyz now and immediately free the full xyz and mask arrays.
    # Avoids a ~3 GB veg_points copy later by keeping las.points (already loaded)
    # and writing via original global indices instead.
    print("Extracting vegetation coordinates (%s pts) ..." % f"{len(non_other_indices):,}")
    veg_xyz = xyz[non_other_indices]
    del xyz, other_mask, non_other

    # Remap task global indices to local veg_xyz indices.
    # non_other_indices is sorted (np.where output), so searchsorted is exact.
    tasks = [(tid, np.searchsorted(non_other_indices, global_idx), global_idx)
             for tid, global_idx in tqdm(tasks, desc="Remapping indices", unit="comp")]
    del non_other_indices

    with open(args.output_file, "wb") as _out_fh, \
         laspy.LasWriter(_out_fh, header=new_header) as writer:
        # Stream non-classified points in chunks.
        n_other = len(other_indices)
        with tqdm(total=n_other, desc="Writing non-classified",
                  unit="pts", unit_scale=True) as pbar:
            for start in range(0, n_other, _WRITE_CHUNK):
                chunk = other_indices[start:start + _WRITE_CHUNK]
                _write_chunk(writer, las.points, chunk,
                             np.full(len(chunk), SENTINEL, dtype=_LW_DTYPE), out_fmt)
                pbar.update(len(chunk))
        del other_indices

        n_leaf = n_wood = n_failed = 0
        with tqdm(total=len(tasks), desc="Processing components") as pbar:
            for task_id, local_idx, global_idx in tasks:
                _, _, lw = _gbs_worker(task_id, local_idx, veg_xyz)
                n_leaf += int(np.sum(lw == 1))
                n_wood += int(np.sum(lw == 0))
                n_failed += int(np.sum(lw == SENTINEL))
                pbar.set_postfix(sz=len(global_idx), leaf=n_leaf,
                                 wood=n_wood, skip=n_failed)
                _write_chunk(writer, las.points, global_idx, lw, out_fmt)
                pbar.update()

    print("Done.")
    print("  Leaf pts  : %s" % f"{n_leaf:,}")
    print("  Wood pts  : %s" % f"{n_wood:,}")
    if n_failed:
        print("  Skipped   : %s" % f"{n_failed:,}")
    print("Output: %s" % args.output_file)


if __name__ == '__main__':
    main()
