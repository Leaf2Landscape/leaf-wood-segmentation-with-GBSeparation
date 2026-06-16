import os
import sys
import argparse
import tempfile
from contextlib import nullcontext, contextmanager
from multiprocessing import Pool, cpu_count
import joblib
import numpy as np
import laspy
from tqdm import tqdm
import threadpoolctl
from GBSeparation.Graph_Path import array_to_graph, extract_path_info
from GBSeparation.LS_circle import getRootPt
from GBSeparation.ExtractInitWood import extract_init_wood
from GBSeparation.ExtractFinalWood import extract_final_wood

def _available_cpus():
    """CPU count respecting SLURM allocation, falling back to multiprocessing.cpu_count().

    Priority:
      1. SLURM_CPUS_PER_TASK  — CPUs allocated to this task (--cpus-per-task); always a plain int.
      2. SLURM_CPUS_PER_NODE  — node allocation; may be compressed like "16(x2),8"; take first token.
      3. multiprocessing.cpu_count() — OS fallback (may return node-wide count on shared HPC nodes).
    """
    task_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if task_cpus:
        try:
            return int(task_cpus)
        except ValueError:
            pass
    node_cpus = os.environ.get('SLURM_CPUS_PER_NODE', '')
    if node_cpus:
        try:
            return int(node_cpus.split('(')[0].split(',')[0])
        except ValueError:
            pass
    return cpu_count()


GROUND_CLASS = 2
_LW_DTYPE = np.int8   # 1=leaf, 0=wood, 2=understorey, -1=other/ground/failed
SENTINEL = np.int8(-1)
UNDERSTOREY_VALUE = np.int8(2)
MIN_COMPONENT_POINTS = 100
_WRITE_CHUNK = 100_000
_STREAM_CHUNK = 1_000_000   # points per chunk in the streaming filter pass
VOXEL_SIZE = 0.02     # metres — grid resolution used to downsample each component before GBS
DTM_CLEARANCE = 0.2   # metres — points whose height above the --dtm surface is below this are dropped

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


def _build_dtm_interpolators(dtm_path):
    """Load DTM and build LinearNDInterpolator + NearestNDInterpolator.

    Both are built upfront so the streaming pass can query them per chunk
    without reloading the DTM each time.  Returns (lin, nn).
    """
    import open3d as o3d
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pcd = o3d.io.read_point_cloud(dtm_path)
    dtm = np.asarray(pcd.points)
    if dtm.size == 0:
        dtm = np.asarray(o3d.io.read_triangle_mesh(dtm_path).vertices)
    if dtm.size == 0:
        print("Error: DTM file has no points or vertices: %s" % dtm_path)
        sys.exit(1)
    print("  DTM loaded: %s surface points." % f"{len(dtm):,}")
    lin = LinearNDInterpolator(dtm[:, :2], dtm[:, 2])
    nn  = NearestNDInterpolator(dtm[:, :2], dtm[:, 2])
    return lin, nn


def _apply_dtm_mask_chunk(dtm_interps, x, y, z):
    """Return boolean mask (True = within DTM_CLEARANCE of the surface) for one chunk."""
    lin, nn = dtm_interps
    surf_z = lin(x, y)
    outside = np.isnan(surf_z)
    if outside.any():
        surf_z[outside] = nn(x[outside], y[outside])
    return (z - surf_z) < DTM_CLEARANCE


def dtm_ground_mask(xyz, dtm_path):
    """Return a boolean mask (True = within DTM_CLEARANCE of the DTM surface).

    Used only by run_dry_run (which already holds the full xyz in memory).
    The main pipeline uses _build_dtm_interpolators + _apply_dtm_mask_chunk
    instead, to avoid loading 148 M query points into a single interpolation call.
    """
    lin, nn = _build_dtm_interpolators(dtm_path)
    surf_z = lin(xyz[:, 0], xyz[:, 1])
    outside = np.isnan(surf_z)
    if outside.any():
        surf_z[outside] = nn(xyz[outside, 0], xyz[outside, 1])
    return (xyz[:, 2] - surf_z) < DTM_CLEARANCE


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


def _make_output_header(source_header, source_pf, overwrite):
    """Build output LasHeader with foliage_type (int8) extra dim.

    Accepts the source LasHeader and its PointFormat separately so callers
    can supply them from either a LasData object or a LasReader.
    """
    dim_names = [dim.name for dim in source_pf.dimensions]
    if 'foliage_type' in dim_names and not overwrite:
        print("Error: 'foliage_type' dimension already exists. Use --overwrite to replace it.")
        sys.exit(1)

    # Use the integer format ID so LasHeader gets its own fresh PointFormat object
    # rather than sharing (and mutating) the source's PointFormat via object reference.
    new_header = laspy.LasHeader(
        point_format=source_header.point_format.id,
        version=source_header.version,
    )
    new_header.offsets = source_header.offsets
    new_header.scales = source_header.scales
    extra_dims = [
        laspy.ExtraBytesParams(name=d.name, type=d.dtype)
        for d in source_pf.extra_dimensions
        if d.name != 'foliage_type'
    ]
    extra_dims.append(laspy.ExtraBytesParams(
        name="foliage_type", type=np.int8,
        description="1=leaf,0=wood,2=under,-1=other"))
    new_header.add_extra_dims(extra_dims)
    return new_header


def _make_temp_header(source_header, source_pf):
    """Build a LasHeader for the streaming-filter temp file.

    Same format as the input (preserving component_id, classification, etc.)
    but without adding foliage_type — the temp file is an intermediate that
    the GBS phase reads; foliage_type is written to the final output.
    """
    new_header = laspy.LasHeader(
        point_format=source_header.point_format.id,
        version=source_header.version,
    )
    new_header.offsets = source_header.offsets
    new_header.scales = source_header.scales
    extra_dims = [
        laspy.ExtraBytesParams(name=d.name, type=d.dtype)
        for d in source_pf.extra_dimensions
        if d.name != 'foliage_type'
    ]
    if extra_dims:
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


def _copy_chunk(writer, src_chunk, indices, dest_fmt):
    """Copy a subset of src_chunk to writer using dest_fmt field matching.

    Like _write_chunk but does not write foliage_type — used to populate the
    temp file during the streaming filter pass.
    """
    sub = src_chunk[indices]
    buf = np.zeros(len(indices), dtype=dest_fmt.dtype())
    src_raw = sub.array
    for name in src_raw.dtype.names:
        if name in buf.dtype.names:
            buf[name] = src_raw[name]
    writer.write_points(laspy.PackedPointRecord(buf, dest_fmt))


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


def _worker_init():
    # When maxtasksperchild restarts a worker, fork happens while the parent's
    # "Processing components" tqdm bar may hold its internal RLock.  The child
    # inherits the lock in a held state with no thread to release it, so any
    # tqdm bar creation in the worker (e.g. inside array_to_graph) deadlocks.
    # Replace tqdm's global lock with a fresh instance to fix the race.
    tqdm.set_lock(type(tqdm.get_lock())())
    # Cap BLAS (OpenBLAS/MKL) internal thread pools to _THREADS_PER_WORKER.
    # Without this limit each forked worker inherits the full cpu_count() default
    # (22 on this machine), so workers × joblib_threads × BLAS_threads threads are
    # spawned simultaneously, exhausting WSL2's per-process thread limit (~99%).
    # threadpoolctl works post-fork on the child side; env-var approaches must be
    # set pre-fork and do not help when maxtasksperchild respawns workers mid-run.
    if _THREADS_PER_WORKER > 0:
        threadpoolctl.threadpool_limits(limits=_THREADS_PER_WORKER, user_api='blas')


def _gbs_worker(task_idx):
    """
    Worker entry point for both sequential and multiprocessing execution.
    Reads _VEG_XYZ, _TASK_LIST, _THREADS_PER_WORKER, _SILENCE_IO from module globals
    set in main() before any Pool is created.
    """
    import traceback as _tb
    task_id, comp_key, local_idx, global_idx = _TASK_LIST[task_idx]
    n_pts = len(local_idx)
    if n_pts < MIN_COMPONENT_POINTS:
        return task_id, np.full(n_pts, SENTINEL, dtype=_LW_DTYPE)
    try:
        ctx = _silence() if _SILENCE_IO else nullcontext()
        with ctx:
            if _THREADS_PER_WORKER > 0:
                threadpoolctl.threadpool_limits(limits=_THREADS_PER_WORKER, user_api='blas')
            xyz_comp = _VEG_XYZ[local_idx].astype(np.float32)

            rep_indices, voxel_assignment = _voxel_downsample(xyz_comp, VOXEL_SIZE)
            if len(rep_indices) >= MIN_COMPONENT_POINTS:
                xyz_sub = xyz_comp[rep_indices]
            else:
                xyz_sub = xyz_comp

            treeHeight = float((np.max(xyz_sub, axis=0) - np.min(xyz_sub, axis=0)).max())
            root, _ = getRootPt(xyz_sub, lower_h=0.0, upper_h=0.2)
            if root is None:
                low_mask = xyz_sub[:, 2] < (xyz_sub[:, 2].min() + 0.2)
                src = xyz_sub[low_mask] if low_mask.any() else xyz_sub
                root = np.array([[float(src[:, 0].mean()),
                                   float(src[:, 1].mean()),
                                   float(xyz_sub[:, 2].min())]], dtype=np.float32)
            xyz_sub = np.append(xyz_sub, root, axis=0)
            root_id = xyz_sub.shape[0] - 1
            n_vox = len(xyz_sub) - 1  # -1 to exclude root point

            G = array_to_graph(xyz_sub, root_id, kpairs=3, knn=30,
                               nbrs_threshold=0.15,
                               nbrs_threshold_step=0.05)
            path_dis, pred = extract_path_info(G, root_id, return_path=True)

            init_wood_ids, G = extract_init_wood(xyz_sub, G, root_id, path_dis, pred,
                                                 split_interval=[0.1, 0.2, 0.3, 0.5, 1],
                                                 max_angle=0.15 * np.pi)
            final_wood_mask = extract_final_wood(xyz_sub, root_id, path_dis, pred,
                                                 init_wood_ids, G)
            final_wood_mask = final_wood_mask[:n_vox]  # drop the appended root

            if len(rep_indices) >= MIN_COMPONENT_POINTS:
                lw = np.where(final_wood_mask[voxel_assignment], 0, 1).astype(_LW_DTYPE)
            else:
                lw = np.where(final_wood_mask, 0, 1).astype(_LW_DTYPE)
            return task_id, lw
    except Exception as exc:
        # Write to tqdm so error appears below the
        # progress bar (which uses carriage-return and would stomp a
        # multi-line traceback written directly to the shared terminal fd).
        tb_str = _tb.format_exc().rstrip()
        tqdm.write("Warning: component %s failed (%d points): %s\n%s"
                   % (comp_key, n_pts, exc, tb_str))
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
    if strategy == 'component_id':
        n_tree_comps = sum(1 for idx in groups.values() if np.all(classification[idx] == 5))
        print("  -> Tree components (classification == 5): %d" % n_tree_comps)
        print("  -> Understorey components (other): %d" % (len(groups) - n_tree_comps))

    print("Ground points (Classification == %d): %d" % (GROUND_CLASS, int(np.sum(ground_mask))))
    if args.dtm is not None:
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        dtm_mask = dtm_ground_mask(xyz, args.dtm)
        print("DTM-filtered points (within %.2f m of %s): %d"
              % (DTM_CLEARANCE, args.dtm, int(np.sum(dtm_mask & ~ground_mask))))
    n_other = int(np.sum(other_mask & ~ground_mask))
    if n_other:
        print("Other non-vegetation points (component_id <= 0): %d" % n_other)
    print("Output path that would be written: %s" % args.output_file)
    sys.exit(0)


def _run_gbs_pipeline(las, args, writer, out_fmt):
    """Run GBS classification on already-filtered las data, writing results to writer."""
    classification = np.asarray(las.classification)
    # Ground points were removed in the streaming pass; ground_mask is kept for
    # structural compatibility with the understorey / other separation logic.
    ground_mask = classification == GROUND_CLASS

    groups, strategy, other_mask = build_groups(las)
    other_mask = other_mask | ground_mask
    non_other = ~other_mask
    groups = {key: idx[non_other[idx]] for key, idx in groups.items()}
    groups = {key: idx for key, idx in groups.items() if len(idx) > 0}

    # For component_id strategy: only run GBS on tree components (classification == 5).
    # All others (understorey) are written directly with foliage_type = UNDERSTOREY_VALUE.
    understorey_indices = np.empty(0, dtype=np.intp)
    if strategy == 'component_id':
        tree_groups = {}
        under_parts = []
        for key, idx in groups.items():
            if np.all(classification[idx] == 5):
                tree_groups[key] = idx
            else:
                under_parts.append(idx)
        understorey_indices = np.concatenate(under_parts) if under_parts else np.empty(0, dtype=np.intp)
        groups = tree_groups
        if len(understorey_indices):
            non_other[understorey_indices] = False

    n_gbs_comps = sum(1 for idx in groups.values() if len(idx) >= MIN_COMPONENT_POINTS)
    n_veg = int(np.sum(non_other))
    n_ground = int(np.sum(ground_mask))
    n_unclassified = int(np.sum(other_mask)) - n_ground
    n_understorey = len(understorey_indices)
    print("Grouping strategy : %s" % strategy)
    print("Vegetation pts    : %s" % f"{n_veg:,}")
    if n_understorey > 0:
        print("Understorey pts   : %s" % f"{n_understorey:,}")
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
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
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
        _THREADS_PER_WORKER = args.threads if args.threads > 0 else max(1, _available_cpus() // (workers * 2))
        print("Parallel mode: %d workers, %d threads/worker" % (workers, _THREADS_PER_WORKER))
    # workers == 1: _THREADS_PER_WORKER stays -1; sequential path is unchanged.

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

    # Write understorey components directly — no GBS classification.
    n_under = len(understorey_indices)
    if n_under:
        with tqdm(total=n_under, desc="Writing understorey",
                  unit="pts", unit_scale=True) as pbar:
            for start in range(0, n_under, _WRITE_CHUNK):
                chunk = understorey_indices[start:start + _WRITE_CHUNK]
                _write_chunk(writer, las.points, chunk,
                             np.full(len(chunk), UNDERSTOREY_VALUE, dtype=_LW_DTYPE), out_fmt)
                pbar.update(len(chunk))
    del understorey_indices
    print("Components        : %d" % n_gbs_comps)

    n_leaf = n_wood = n_failed = 0
    ctx = Pool(workers, maxtasksperchild=16, initializer=_worker_init) if workers > 1 else nullcontext()
    with ctx as pool, tqdm(total=len(tasks), desc="Processing components") as pbar:
        results = (pool.imap_unordered(_gbs_worker, range(len(tasks)))
                   if pool is not None else map(_gbs_worker, range(len(tasks))))
        try:
            for task_id, lw in results:
                _, comp_key, _, global_idx = tasks[task_id]
                n_pts = len(lw)
                n_leaf_c = int(np.sum(lw == 1))
                n_wood_c = int(np.sum(lw == 0))
                n_fail_c = int(np.sum(lw == SENTINEL))
                n_leaf += n_leaf_c
                n_wood += n_wood_c
                n_failed += n_fail_c
                pbar.set_postfix(sz=n_pts, leaf=n_leaf, wood=n_wood, skip=n_failed)
                _write_chunk(writer, las.points, global_idx, lw, out_fmt)
                if _SILENCE_IO:
                    pct = int(100 * n_leaf_c / n_pts) if n_pts else 0
                    suffix = ("  [%d failed]" % n_fail_c) if n_fail_c else ""
                    tqdm.write("  comp %s: %s pts → leaf=%d%% wood=%s%s"
                               % (comp_key, f"{n_pts:,}", pct, f"{n_wood_c:,}", suffix))
                pbar.update()
        except Exception as exc:
            print("\nERROR: worker process crashed unexpectedly: %s" % exc)
            print("Output written up to this point; remaining components unclassified.")

    print("Done.")
    print("  Leaf pts  : %s" % f"{n_leaf:,}")
    print("  Wood pts  : %s" % f"{n_wood:,}")
    if n_failed:
        print("  Skipped   : %s" % f"{n_failed:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GBSeparation leaf/wood classification on a forest-plot .las/.laz file.")
    parser.add_argument("--input_file", required=True, help="path to input .las/.laz")
    parser.add_argument("--output_file", required=True, help="path to output .las/.laz")
    parser.add_argument("--dtm", default=None,
                        help="optional DTM ground surface (.ply/.las/...); points within "
                             "%.2f m of the surface are dropped (written as -1)." % DTM_CLEARANCE)
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

    if args.dtm is not None and not os.path.isfile(args.dtm):
        print("Error: DTM file not found or not readable: %s" % args.dtm)
        sys.exit(1)

    # Build DTM interpolators upfront: one triangulation over the ~1.8 M DTM
    # vertices, queried once per streaming chunk rather than over all 148 M
    # input points in a single call.
    dtm_interps = None
    if args.dtm is not None:
        print("Filtering points within %.2f m of DTM: %s" % (DTM_CLEARANCE, args.dtm))
        dtm_interps = _build_dtm_interpolators(args.dtm)

    file_size_gb = os.path.getsize(args.input_file) / 1e9
    print("Reading %.2f GB: %s" % (file_size_gb, args.input_file))

    if args.dry_run:
        try:
            las = laspy.read(args.input_file)
        except Exception as exc:
            print("Error: could not read input file: %s" % exc)
            sys.exit(1)
        print("  Loaded %s points." % f"{len(las.x):,}")
        run_dry_run(las, args)

    # Read just the file header (no point data) to build output and temp formats.
    try:
        with laspy.LasReader(open(args.input_file, 'rb')) as hdr_reader:
            src_header = hdr_reader.header
            src_pf     = hdr_reader.header.point_format
            n_pts_total = src_header.point_count
            new_header  = _make_output_header(src_header, src_pf, args.overwrite)
            temp_header = _make_temp_header(src_header, src_pf)
    except Exception as exc:
        print("Error: could not read input file: %s" % exc)
        sys.exit(1)

    out_fmt  = new_header.point_format
    temp_fmt = temp_header.point_format

    # Temp file holds surviving (non-ground, non-DTM) points for the GBS phase.
    temp_fd, temp_path = tempfile.mkstemp(suffix='.las', prefix='gbs_filter_')
    os.close(temp_fd)
    n_ground = n_dtm = n_surviving = 0

    try:
        with open(args.output_file, "wb") as out_fh, \
             laspy.LasWriter(out_fh, header=new_header) as writer:

            # ── Phase 1: streaming filter ─────────────────────────────────────
            # Stream the input in _STREAM_CHUNK-point chunks.  For each chunk:
            #   • ground + DTM-near points  → written to output immediately as SENTINEL
            #   • surviving points          → written to temp file for the GBS phase
            # Peak memory during this phase: O(_STREAM_CHUNK), not O(all points).
            print("Streaming %s pts in %s-pt chunks ..."
                  % (f"{n_pts_total:,}" if n_pts_total else "?",
                     f"{_STREAM_CHUNK:,}"))
            with laspy.LasReader(open(args.input_file, 'rb')) as reader, \
                 open(temp_path, 'wb') as temp_fh, \
                 laspy.LasWriter(temp_fh, header=temp_header) as temp_writer:

                tqdm_total = n_pts_total if n_pts_total > 0 else None
                with tqdm(total=tqdm_total, desc="Filtering",
                          unit="pts", unit_scale=True) as pbar:
                    for chunk in reader.chunk_iterator(_STREAM_CHUNK):
                        x   = np.asarray(chunk.x,              dtype=np.float64)
                        y   = np.asarray(chunk.y,              dtype=np.float64)
                        z   = np.asarray(chunk.z,              dtype=np.float64)
                        cls = np.asarray(chunk.classification,  dtype=np.uint8)

                        gnd      = cls == GROUND_CLASS
                        dtm_near = (_apply_dtm_mask_chunk(dtm_interps, x, y, z)
                                    if dtm_interps is not None
                                    else np.zeros(len(x), dtype=bool))
                        drop = gnd | dtm_near

                        n_ground   += int(gnd.sum())
                        n_dtm      += int((dtm_near & ~gnd).sum())
                        n_surviving += int((~drop).sum())

                        drop_idx = np.where(drop)[0]
                        if len(drop_idx):
                            _write_chunk(writer, chunk, drop_idx,
                                         np.full(len(drop_idx), SENTINEL, dtype=_LW_DTYPE),
                                         out_fmt)

                        keep_idx = np.where(~drop)[0]
                        if len(keep_idx):
                            _copy_chunk(temp_writer, chunk, keep_idx, temp_fmt)

                        pbar.update(len(x))

            print("Ground pts        : %s" % f"{n_ground:,}")
            if dtm_interps is not None:
                print("DTM-filtered pts  : %s" % f"{n_dtm:,}")
            print("Surviving pts     : %s" % f"{n_surviving:,}")

            # ── Phase 2: GBS classification ───────────────────────────────────
            # Load only the surviving (vegetation) points — much smaller than
            # the original file — and run the full GBS pipeline on them.
            print("Loading filtered points (%s pts) ..." % f"{n_surviving:,}")
            try:
                las = laspy.read(temp_path)
            except Exception as exc:
                print("Error: could not read temp file: %s" % exc)
                raise
            print("  Loaded %s points." % f"{len(las.x):,}")

            _run_gbs_pipeline(las, args, writer, out_fmt)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("Output: %s" % args.output_file)


if __name__ == '__main__':
    main()
