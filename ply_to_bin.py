import os, json
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import argparse
import util_gau

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def load_ply(path, max_sh_degree=2):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    print(extra_f_names)
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    ret = np.concatenate([xyz, rots, scales, opacities, shs], axis=-1)
    return np.ascontiguousarray(ret)

def convert_folder(ply_files, avail_timestamps, out_dir, max_sh_degree):
    os.makedirs(out_dir, exist_ok=True)
    items = []

    for i, ply_path in enumerate(tqdm(ply_files, desc="Converting PLY→BIN")):
        # gaus = util_gau.load_ply(ply_path, max_sh_degree)
        flat = load_ply(ply_path, max_sh_degree)
        # flat = util_gau.GaussianData.flat(gaus)
        
        # flat = load_ply(ply_path, max_sh_degree) 
        assert flat.dtype == np.float32 and flat.flags['C_CONTIGUOUS']
        n = flat.size

        timestamp = avail_timestamps[i]
        bin_name = f"{timestamp:06d}.bin"
        
        # subdir
        bin_subdir = "bin"

        # Full path for file writing
        bin_dir_full = os.path.join(out_dir, bin_subdir)
        os.makedirs(bin_dir_full, exist_ok=True)
        bin_path_full = os.path.join(bin_dir_full, bin_name)
        with open(bin_path_full, "wb") as f:
            f.write(flat.tobytes(order="C"))
        
        # Relative path for index file
        relative_path = os.path.join(bin_subdir, bin_name)

        items.append({
            "index": i,
            "path": relative_path, 
            "dtype": "float32",
            "count": int(n)
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Input .ply file base directory")
    parser.add_argument("--outdir", required=True, help="Output .bin file base directory")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--max_sh_degree", type=int, default=1, help="Max SH degree")
    args = parser.parse_args()

    timestamps = sorted(os.listdir(args.src), key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
    ply_files = []
    avail_timestamps = []

    for timestamp in timestamps:
        idx = int(timestamp.split("_")[-1])
        if idx < args.start:
            continue

        if not os.path.exists(os.path.join(args.src, timestamp, "point_cloud")):
            print("Skipping", timestamp)
            continue

        iteration = searchForMaxIteration(os.path.join(args.src, timestamp, "point_cloud"))
        ply_path = os.path.join(args.src, timestamp, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        ply_files.append(ply_path)
        avail_timestamps.append(idx)

    convert_folder(ply_files, avail_timestamps, args.outdir, args.max_sh_degree)