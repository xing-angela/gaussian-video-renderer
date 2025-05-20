import os
import shutil
from tqdm import tqdm

# Directory containing timestamped subdirectories
SOURCE_ROOT = "./../gaussian-data/brics_baby"  # <-- Replace with your data path
OUTPUT_DIR = "./../gaussian-data/brics_baby_pcd"

# Iteration priority
ITER_PRIORITIES = ["45000", "30000", "15000"]

def find_point_cloud(frame_path):
    pc_dir = os.path.join(frame_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        return None

    for iteration in ITER_PRIORITIES:
        candidate = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")
        if os.path.isfile(candidate):
            return candidate

    return None

def extract_point_clouds(source_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Gather timestamp folders
    frame_dirs = sorted([
        os.path.join(source_root, d) for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d)) and d.startswith("timestamp")
    ])

    for idx, frame_dir in enumerate(tqdm(frame_dirs, desc="Extracting point clouds")):
        ply_path = find_point_cloud(frame_dir)
        if ply_path is None:
            print(f"Warning: No valid point cloud found in {frame_dir}")
            continue

        target_name = f"frame_{idx:04d}.ply"
        target_path = os.path.join(output_dir, target_name)

        shutil.copyfile(ply_path, target_path)

    print(f"\nDone. Extracted {len(os.listdir(output_dir))} frames to {output_dir}")

if __name__ == "__main__":
    extract_point_clouds(SOURCE_ROOT, OUTPUT_DIR)
