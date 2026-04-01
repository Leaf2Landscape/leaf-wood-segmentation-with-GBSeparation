"""CLI entry point: python -m GBSeparation input_dir output_dir"""

import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="GBSeparation: leaf-wood separation for single tree point clouds")
    parser.add_argument("input", help="Input file or directory of point clouds")
    parser.add_argument("output", help="Output directory for separated wood/leaf files")
    parser.add_argument("--lower-h", type=float, default=0.0,
                        help="Lower height for root fitting (default: 0.0)")
    parser.add_argument("--upper-h", type=float, default=0.2,
                        help="Upper height for root fitting (default: 0.2)")
    parser.add_argument("--kpairs", type=int, default=3)
    parser.add_argument("--knn", type=int, default=300)
    parser.add_argument("--split-interval", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5, 1])
    parser.add_argument("--max-angle-factor", type=float, default=0.15,
                        help="Factor of pi for max angle (default: 0.15)")
    args = parser.parse_args()

    from GBSeparation import separate

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
        input_dir = os.path.dirname(args.input)
    else:
        input_dir = args.input
        files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]

    for filepath in files:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        print(f"Processing {filename}...")
        try:
            if ext in ('.txt', '.xyz'):
                pcd = np.loadtxt(filepath, usecols=[0, 1, 2], dtype=np.float32)
            elif ext == '.las':
                import laspy
                las = laspy.read(filepath)
                pcd = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
            elif ext in ('.pcd', '.ply'):
                import open3d as o3d
                cloud = o3d.io.read_point_cloud(filepath, format=ext[1:])
                pcd = np.asarray(cloud.points, dtype=np.float32)
            else:
                print(f"  Skipping unsupported format: {ext}")
                continue

            wood, leaf = separate(
                pcd,
                lower_h=args.lower_h,
                upper_h=args.upper_h,
                kpairs=args.kpairs,
                knn=args.knn,
                split_interval=args.split_interval,
                max_angle_factor=args.max_angle_factor,
            )

            np.savetxt(os.path.join(args.output, f"{name}_wood.txt"), wood, fmt='%1.6f')
            np.savetxt(os.path.join(args.output, f"{name}_leaf.txt"), leaf, fmt='%1.6f')
            print(f"  wood: {len(wood)} pts, leaf: {len(leaf)} pts")

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
