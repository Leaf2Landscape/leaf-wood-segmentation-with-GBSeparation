import os
import sys
import argparse
from contextlib import nullcontext, contextmanager
from multiprocessing import Pool, cpu_count
import joblib
import numpy as np
import laspy
from tqdm import tqdm
from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood

def _available_cpus():
    """CPU count respecting SLURM allocation, falling back to multiprocessing.cpu_count().

    SLURM_CPUS_PER_NODE can be a plain integer or compressed like "16(x2),8";
    we take the first numeric token which represents this node's allocation.
    """
    slurm = os.environ.get('SLURM_CPUS_PER_NODE', '')
    if slurm:
        try:
            return int(slurm.split('(')[0].split(',')[0])
        except ValueError:
            pass
    return cpu_count()


GROUND_CLASS = 2
_LW_DTYPE = np.int8   # 1=leaf, 0=wood, -1=other/ground/failed
SENTINEL = np.int8(-1)
MIN_COMPONENT_POINTS = 10
_WRITE_CHUNK = 100_000
VOXEL_SIZE = 0.02     # metres — grid resolution used to downsample each component before GBS

# Module-level globals shared with worker processes via fork.
# Set in main() before Pool creation; never written by workers.
_VEG_XYZ = None
_TASK_LIST = None
_THREADS_PER_WORKER = -1   # -1 = all cores (sequential path only)
_SILENCE_IO = False         # True in multi-process mode to prevent terminal write contention


@contextmanager
def _silence():
    """Redirect stdout and stderr to /dev/null for the duration of the block.

    GBSeparation library calls (array_to_graph, extract_init_wood, …) write
    verbose output — nested tqdm bars, 'NN done', cut-edge counts — directly
    to stdout/stderr.  In forked parallel workers these writes share the same
    terminal file descriptor as the parent process's tqdm progress bar.  On
    some systems the resulting contention blocks the write call, which prevents
    the worker from ever reaching its return statement and returning the result.
    Suppressing the output inside parallel workers fixes the hang.
    """
    old_out, old_err = sys.stdout, sys.stderr
    try:
        devnull = open(os.devnull, 'w')
        sys.stdout = sys.stderr = devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


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
    other_mask = np.zeros(n_points, dtype=bool)

    if 'component_id' in dim_names:
        component_ids = np.asarray(las['component_id'])
        other_mask = component_ids <= 0
        veg_idx = np.where(~other_mask)[0]
        veg_cids = component_ids[veg_idx]
        print("Indexing %s components (%s pts) ..."
              % (f"{np.unique(veg_cids).size:,}", f"{len(veg_idx):,}"))
        order = np.argsort(veg_cids, kind='stable')
        sorted_cids = veg_cids[order]
        sorted_idx = veg_idx[order]
        splits = np.where(np.diff(sorted_cids))[0] + 1
        boundaries = np.r_[0, splits]
        groups = {int(sorted_cids[b]): grp
                  for b, grp in zip(boundaries, np.split(sorted_idx, splits))}
        return groups, 'component_id', other_mask

    if 'tree_id' in dim_names and 'stem_id' in dim_names:
        tree = np.asarray(las['tree_id'])
        stem = np.asarray(las['stem_id'])
        print("Indexing components by (tree_id, stem_id) (%s pts) ..." % f"{n_points:,}")
        order = np.lexsort((stem, tree))
        sorted_tree = tree[order]
        sorted_stem = stem[order]
        splits = np.where(
            (np.diff(sorted_tree) != 0) | (np.diff(sorted_stem) != 0)
        )[0] + 1
        boundaries = np.r_[0, splits]
        groups = {(int(sorted_tree[b]), int(sorted_stem[b])): grp
                  for b, grp in zip(boundaries, np.split(order, splits))}
        return groups, '(tree_id, stem_id)', other_mask

    if 'tree_id' in dim_names:
        keys = np.asarray(las['tree_id'])
        print("Indexing components by tree_id (%s pts) ..." % f"{n_points:,}")
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        splits = np.where(np.diff(sorted_keys))[0] + 1
        boundaries = np.r_[0, splits]
        groups = {int(sorted_keys[b]): grp
                  for b, grp in zip(boundaries, np.split(order, splits))}
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


def _voxel_downsample(xyz, voxel_size):
    """
    Voxel-grid downsample xyz. Pure numpy, no Python loops.

    Returns
    -------
    rep_indices      : (M,) indices into xyz of one representative per occupied voxel
    voxel_assignment : (N,) for each point in xyz, its index into rep_indices
    """
    voxel_ijk = np.floor(xyz / voxel_size).astype(np.int64)
    mn = voxel_ijk.min(axis=0)
    voxel_ijk -= mn
    dims = voxel_ijk.max(axis=0) + 1
    keys = (voxel_ijk[:, 0] * (dims[1] * dims[2])
            + voxel_ijk[:, 1] * dims[2]
            + voxel_ijk[:, 2])

    sort_order = np.argsort(keys, kind='stable')
    sorted_keys = keys[sort_order]

    is_first = np.empty(len(sorted_keys), dtype=bool)
    is_first[0] = True
    is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]

    rep_indices = sort_order[is_first]          # one representative per voxel
    unique_keys = sorted_keys[is_first]
    voxel_assignment = np.searchsorted(unique_keys, keys)  # each point → rep index
    return rep_indices, voxel_assignment


def _gbs_worker(task_idx):
    """
    Worker entry point for both sequential and multiprocessing execution.
    Reads _VEG_XYZ, _TASK_LIST, _THREADS_PER_WORKER, _SILENCE_IO from module globals
    (set in main() before any Pool is created; inherited via fork on Linux).
    Returns (task_id, foliage_type); caller looks up global_idx via _TASK_LIST[task_id].
    """
    task_id, comp_key, local_idx, _ = _TASK_LIST[task_idx]
    n_pts = len(local_idx)
    if n_pts < MIN_COMPONENT_POINTS:
        print("Warning: component %s with %d points (< %d) skipped."
              % (comp_key, n_pts, MIN_COMPONENT_POINTS))
        return task_id, np.full(n_pts, SENTINEL, dtype=_LW_DTYPE)
    try:
        xyz_comp = _VEG_XYZ[local_idx].astype(np.float32)

        # Downsample to VOXEL_SIZE grid; voxel_assignment maps every original
        # point back to its representative's label after classification.
        rep_indices, voxel_assignment = _voxel_downsample(xyz_comp, VOXEL_SIZE)
        if len(rep_indices) >= 30:
            # Normal path: classify the downsampled cloud, map labels back via voxel assignment.
            xyz_sub = xyz_comp[rep_indices]
            use_voxel = True
        elif n_pts >= 30:
            # Downsampled cloud is too sparse for knn=30 but the original has enough points.
            # Classify the original directly; labels map 1-to-1, no voxel assignment needed.
            print("Warning: component %s downsampled to %d pts (< knn=30); "
                  "classifying %d original pts instead." % (comp_key, len(rep_indices), n_pts))
            xyz_sub = xyz_comp
            use_voxel = False
        else:
            print("Warning: component %s %d pts (< knn=30) — skipped." % (comp_key, n_pts))
            return task_id, np.full(n_pts, SENTINEL, dtype=_LW_DTYPE)

        # Use the largest spatial dimension (X, Y, or Z) to derive graph thresholds.
        # Using only Z (height) gives near-zero thresholds for flat/horizontal structures,
        # making array_to_graph's bridging loop iterate hundreds of thousands of times.
        treeHeight = float((np.max(xyz_sub, axis=0) - np.min(xyz_sub, axis=0)).max())
        root, _ = getRootPt(xyz_sub, lower_h=0.0, upper_h=0.2)
        # circleFit degenerates (divide-by-zero) when the low-height slice points
        # are collinear; fall back to the XY centroid of the slice in that case.
        if not np.isfinite(root).all():
            low_mask = xyz_sub[:, 2] < (xyz_sub[:, 2].min() + 0.2)
            src = xyz_sub[low_mask] if low_mask.any() else xyz_sub
            root = np.array([[src[:, 0].mean(), src[:, 1].mean(),
                               float(xyz_sub[:, 2].min())]], dtype=np.float32)
        xyz_sub = np.append(xyz_sub, root, axis=0)
        root_id = xyz_sub.shape[0] - 1
        n_vox = len(xyz_sub) - 1  # -1 to exclude root point
        print("  comp %s: %d pts → %d voxels" % (comp_key, n_pts, n_vox), flush=True)
        # Silence library stdout/stderr in parallel workers to prevent blocked
        # terminal writes from hanging the worker before it can return its result.
        _io_ctx = _silence() if _SILENCE_IO else nullcontext()
        with _io_ctx:
            if _THREADS_PER_WORKER == -1:
                # Sequential path: bare call, all cores via sklearn default.
                G = array_to_graph(xyz_sub, root_id, kpairs=3, knn=30,
                                   nbrs_threshold=treeHeight / 30,
                                   nbrs_threshold_step=treeHeight / 60,
                                   n_jobs=-1)
            else:
                # Multi-process path: joblib threading backend avoids the loky
                # restriction on spawning sub-processes inside forked workers.
                with joblib.parallel_backend('threading', n_jobs=_THREADS_PER_WORKER):
                    G = array_to_graph(xyz_sub, root_id, kpairs=3, knn=30,
                                       nbrs_threshold=treeHeight / 30,
                                       nbrs_threshold_step=treeHeight / 60,
                                       n_jobs=_THREADS_PER_WORKER)
            path_dis, path_list = extract_path_info(G, root_id, return_path=True)
            _max_workers = None if _THREADS_PER_WORKER == -1 else _THREADS_PER_WORKER
            init_wood_ids = extract_init_wood(xyz_sub, G, root_id, path_dis, path_list,
                                              split_interval=[0.1, 0.2, 0.3, 0.5, 1],
                                              max_angle=0.15 * np.pi,
                                              classify_parallel=True,
                                              max_workers=_max_workers)
            final_wood_mask = extract_final_wood(xyz_sub, root_id, path_dis, path_list,
                                                 init_wood_ids, G)
        final_wood_mask[-1] = False
        # pipeline: True=wood; output: 1=leaf, 0=wood
        rep_labels = (~final_wood_mask[:-1]).astype(_LW_DTYPE)

        foliage_type = rep_labels[voxel_assignment] if use_voxel else rep_labels
        return task_id, foliage_type
    except Exception as exc:
        print("Warning: component %s failed (%d points): %s" % (comp_key, n_pts, exc))
        return task_id, np.full(n_pts, SENTINEL, dtype=_LW_DTYPE)


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
    parser.add_argument("--workers", type=int, default=1,
                        help="number of components to classify in parallel (default: 1). "
                             "Requires Linux/WSL (uses fork). Recommended: 2–4.")
    parser.add_argument("--threads", type=int, default=0,
                        help="threads per worker for KNN and component classification "
                             "(default: 0 = auto: cpu_count()//workers). "
                             "Has no effect when --workers 1.")
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
    tasks = [(i, key, idx) for i, (key, idx) in enumerate(sorted_items)]

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
    tasks = [(tid, key, np.searchsorted(non_other_indices, global_idx), global_idx)
             for tid, key, global_idx in tqdm(tasks, desc="Remapping indices", unit="comp")]
    del non_other_indices

    # Set module-level globals before any Pool is created so worker processes
    # inherit them via fork without pickling large arrays.
    global _VEG_XYZ, _TASK_LIST, _THREADS_PER_WORKER, _SILENCE_IO
    _VEG_XYZ = veg_xyz
    _TASK_LIST = tasks
    workers = args.workers
    _SILENCE_IO = workers > 1
    if workers > 1:
        _THREADS_PER_WORKER = args.threads if args.threads > 0 else max(1, _available_cpus() // workers)
        print("Parallel mode: %d workers, %d threads/worker" % (workers, _THREADS_PER_WORKER))
    # workers == 1: _THREADS_PER_WORKER stays -1; sequential path is unchanged.

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
        ctx = Pool(workers) if workers > 1 else nullcontext()
        with ctx as pool, tqdm(total=len(tasks), desc="Processing components") as pbar:
            results = (pool.imap_unordered(_gbs_worker, range(len(tasks)))
                       if pool is not None else map(_gbs_worker, range(len(tasks))))
            try:
                for task_id, lw in results:
                    _, _, _, global_idx = tasks[task_id]
                    n_leaf += int(np.sum(lw == 1))
                    n_wood += int(np.sum(lw == 0))
                    n_failed += int(np.sum(lw == SENTINEL))
                    pbar.set_postfix(sz=len(lw), leaf=n_leaf, wood=n_wood, skip=n_failed)
                    _write_chunk(writer, las.points, global_idx, lw, out_fmt)
                    pbar.update()
            except Exception as exc:
                print("\nERROR: worker process crashed unexpectedly: %s" % exc)
                print("Output written up to this point; remaining components unclassified.")

    print("Done.")
    print("  Leaf pts  : %s" % f"{n_leaf:,}")
    print("  Wood pts  : %s" % f"{n_wood:,}")
    if n_failed:
        print("  Skipped   : %s" % f"{n_failed:,}")
    print("Output: %s" % args.output_file)


if __name__ == '__main__':
    main()
