import os
import re
import json
import time
import threading
import numpy as np
import util
from util_gau import GaussianData
from renderer_cuda import gaus_cuda_from_cpu
# from concurrent.futures import ThreadPoolExecutor


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.findall(r'\d+|\D+', os.path.basename(s))]

def _read_timestamps_index(folder_path, default_bin_dir="bin", index_name="timestamps.idx"):
    """
    Read the index produced by your converter:
      {
        "version": 1,
        "dtype": "float32",
        "timestamps": [
          {"index": i, "path": "bin/000000.bin", "dtype": "float32", "count": N*D},
          ...
        ]
      }

    Returns: (files:list[str], dtype:str, counts:list[int]|None)
    """
    idx_path = os.path.join(folder_path, index_name)
    if not os.path.isfile(idx_path):
        # Fallback: list *.bin
        bin_dir = os.path.join(folder_path, default_bin_dir)
        if os.path.isdir(bin_dir):
            files = [os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith(".bin")]
        else:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".bin")]
        files.sort(key=_natural_key)
        return files, "float32", None

    with open(idx_path, "r") as f:
        meta = json.load(f)

    # if meta.get("version", 1) != 1:
    #     raise RuntimeError(f"Unsupported timestamps.idx version: {meta.get('version')}")
    dtype = meta.get("dtype", "float32")
    items = sorted(meta.get("timestamps", []), key=lambda it: it.get("index", 0))

    files, counts = [], []
    for it in items:
        p = it.get("path")
        files.append(p)
        # full_path = os.path.join(folder_path, p)
        # files.append(full_path)
        counts.append(int(it.get("count", 0)) if "count" in it else -1)

    return files, dtype, counts


def _infer_sh_dim_from_file(path, base=11):
    """Infer the SH dimension from a .bin file."""
    c = np.fromfile(path, dtype=np.float32).size

    for s in range(0, 65):
        dim = 3*(s+1)**2
        if c % (base + dim) == 0:
            return dim

class _FrameCache:
    """Forward-only (looping) frame cache with background prefetch for .bin frames."""
    def __init__(self, frame_paths, feature_dim, sh_dim, cache_ahead=5, use_memmap=False):
        self.paths = frame_paths
        self.N = len(frame_paths)
        self.D = feature_dim
        self.SH = sh_dim
        self.cache_ahead = max(0, cache_ahead)
        self.use_memmap = use_memmap

        # idx -> (GaussianData, flat_array) ; both share memory when possible
        self._have = {}
        self._order = []  # maintain recent order for predictable eviction
        self._cur_idx = 0

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self, start_idx=0):
        with self._lock:
            self._cur_idx = start_idx % self.N if self.N > 0 else 0
            self._stop = False
        if self.N > 0:
            self._th.start()

    def stop(self):
        with self._lock:
            self._stop = True
            self._cond.notify_all()
        if self._th.is_alive():
            self._th.join(timeout=1.0)
        # self._executor.shutdown(wait=False)

    def set_current_index(self, idx):
        with self._lock:
            if self.N == 0:
                return
            self._cur_idx = idx % self.N
            self._cond.notify_all()

    def get_gaussian(self, idx, block=False, timeout=0.5):
        """Return GaussianData if ready."""
        g, _ = self._get(idx, want_gauss=True, want_flat=False, block=block, timeout=timeout)
        return g

    def get_flat(self, idx, block=False, timeout=0.5):
        """Return flat (N, D) float32 if ready."""
        _, f = self._get(idx, want_gauss=False, want_flat=True, block=block, timeout=timeout)
        return f

    def _get(self, idx, want_gauss, want_flat, block, timeout):
        if self.N == 0:
            return None, None
        idx = idx % self.N
        t0 = time.time()
        while True:
            with self._lock:
                entry = self._have.get(idx, None)
                if entry is not None:
                    # mark recent
                    if self._order and self._order[-1] != idx:
                        try:
                            self._order.remove(idx)
                        except ValueError:
                            pass
                        self._order.append(idx)
                    g, f = entry
                    g = g if want_gauss else None
                    f = f if want_flat else None
                    return g, f
                if not block:
                    return None, None
                remaining = max(0.0, timeout - (time.time() - t0))
                if remaining == 0:
                    return None, None
                self._cond.wait(remaining)

    # ---- Background Thread ----
    def _run(self):
        while True:
            with self._lock:
                if self._stop or self.N == 0:
                    return
                cur = self._cur_idx
                want = [(cur + i) % self.N for i in range(self.cache_ahead + 1)]
                # Evict anything not wanted
                to_evict = [k for k in self._order if k not in want]
                for k in to_evict:
                    self._have.pop(k, None)
                    try:
                        self._order.remove(k)
                    except ValueError:
                        pass
                to_load = [i for i in want if i not in self._have]

            # Load missing frames outside the lock (in forward order)
            for idx in to_load:
                g, f = self._load_bin_as_gaussian(self.paths[idx])
                with self._lock:
                    # If advanced, next loop will evict if necessary
                    self._have[idx] = (g, f)
                    self._order.append(idx)
                    self._cond.notify_all()

            time.sleep(0.001)

    # ---- I/O ----
    def _load_bin_as_gaussian(self, path):
        # Read float32 and reshape to (N, D)
        if self.use_memmap:
            mm = np.memmap(path, dtype=np.float32, mode='r')
            if mm.size % self.D != 0:
                raise RuntimeError(f"Bad frame size in {path}: total={mm.size}, D={self.D}")
            arr = np.reshape(mm, (-1, self.D))
        else:
            arr = np.fromfile(path, dtype=np.float32)
            if arr.size % self.D != 0:
                raise RuntimeError(f"Bad frame size in {path}: total={arr.size}, D={self.D}")
            arr = arr.reshape((-1, self.D))

        xyz = arr[:, 0:3]
        rot = arr[:, 3:7]
        scale = arr[:, 7:10]
        opacity = arr[:, 10:11]
        sh = arr[:, 11:11 + self.SH]
        g = GaussianData(xyz=xyz, rot=rot, scale=scale, opacity=opacity, sh=sh)

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return g, arr


class GaussianVideo:
    """
      Load .bin frames on-demand with a forward-only prefetch cache (current + K next).
      Expose GaussianData in the same format as before.
      Keep a small GPU ring (gaussian_cache) for SSBO/CUDA objects.
    """

    def __init__(self, folder_path, fps=30.0, renderer_type=0, cache_ahead=5, use_memmap=False):
        self.fps = fps
        self.playback_speed = 1.0
        self.frame_interval = 1.0 / (fps * self.playback_speed)
        self.last_update_time = time.time()
        self.current_frame_idx = 0
        self.prev_frame_idx = 0
        self.paused = False
        self.renderer_type = renderer_type  # 0=OpenGL, 1=CUDA
        self.program = None
        self.frame_step_size = 1

        # GPU-side small ring of uploaded buffers (id, frame_idx)
        self.gaussian_cache = np.array([None] * 15, dtype=object)

        # Disk frames
        self.folder_path = folder_path
        self.num_frames = 0

        # Layout info for .bin
        self.sh_dim = None
        self.feature_dim = None

        # Cache / loader
        self._cache = None
        self._frame_paths = []

        self._init_frames(cache_ahead=cache_ahead, use_memmap=use_memmap)

    # -------------- Setup --------------
    def _init_frames(self, cache_ahead, use_memmap):
        if not self.folder_path:
            self.num_frames = 0
            self._frame_paths = []
            self._cache = _FrameCache([], self.feature_dim, self.sh_dim,
                                    cache_ahead=cache_ahead, use_memmap=use_memmap)
            return

        files, dtype, counts = _read_timestamps_index(self.folder_path)
        if not files:
            self.num_frames = 0
            self._frame_paths = []
            self._cache = _FrameCache([], self.feature_dim, self.sh_dim,
                                    cache_ahead=cache_ahead, use_memmap=use_memmap)
            return

        if dtype != "float32":
            raise RuntimeError(f"Expected float32 in timestamps.idx, got {dtype}")

        # Get the SH dimension from the first file
        sh_dim = _infer_sh_dim_from_file(files[0])
        self.sh_dim = sh_dim
        self.feature_dim = 3 + 4 + 3 + 1 + self.sh_dim  # xyz + rot + scale + opacity + sh

        # Sanity: each frame's element count should be divisible by feature_dim
        if counts is not None and len(counts) == len(files):
            for i, (c, p) in enumerate(zip(counts, files)):
                if c > 0 and (c % self.feature_dim) != 0:
                    raise RuntimeError(
                        f"Frame {i} has count={c} not divisible by feature_dim={self.feature_dim} ({p})"
                    )

        self._frame_paths = files
        self.num_frames = len(files)

        self._cache = _FrameCache(self._frame_paths, self.feature_dim, self.sh_dim,
                                cache_ahead=cache_ahead, use_memmap=use_memmap)
        self._cache.start(start_idx=0)

    # -------------- GPU Uploads --------------
    def upload_to_gpu(self, idx, block=False):
        """Ensure frame idx is uploaded to GPU; return (gpu_obj, idx)."""
        if self.num_frames == 0:
            return None

        cache_idx = idx % len(self.gaussian_cache)
        prev_entry = self.gaussian_cache[cache_idx]

        # Fast path: if slot already has this frame, reuse it
        if prev_entry is not None and prev_entry[1] == idx:
            return prev_entry

        gpu_data = None

        if self.renderer_type == 0:
            # OpenGL: need a flat (N, D) buffer
            flat = self._cache.get_flat(idx, block=block, timeout=0.25 if block else 0.0)
            if flat is None:
                # Not ready yet -> keep previous GPU buffer (caller can continue drawing old frame)
                return prev_entry

            assert self.program is not None, "OpenGL program not set"
            buffer_id = prev_entry[0] if prev_entry else None
            buffer_id = util.set_storage_buffer_data(
                self.program, "gaussian_data", flat, bind_idx=0, buffer_id=buffer_id
            )
            gpu_data = buffer_id

        elif self.renderer_type == 1:
            # CUDA: need GaussianData
            gaus = self._cache.get_gaussian(idx, block=block, timeout=0.25 if block else 0.0)
            if gaus is None:
                return prev_entry
            gpu_data = gaus_cuda_from_cpu(gaus)

        self.gaussian_cache[cache_idx] = (gpu_data, idx)
        return self.gaussian_cache[cache_idx]

    def upload_frames_to_gpu(self, indices):
        if len(indices) > len(self.gaussian_cache):
            print("Cannot cache all gaussian frames at once; will overwrite ring slots.")
        for idx in indices:
            self.upload_to_gpu(idx, block=False)

    # -------------- Playback --------------
    def update(self):
        """Advance to the next frame if enough time has passed. Returns True if frame index changed."""
        if self.num_frames == 0 or self.paused:
            return False
        now = time.time()
        if now - self.last_update_time >= self.frame_interval:
            self.prev_frame_idx = self.current_frame_idx
            self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
            self.last_update_time = now
            self._cache.set_current_index(self.current_frame_idx)
            return True
        return False

    def get_current_frame_cpu(self, block=False):
        if self.num_frames == 0:
            return None
        return self._cache.get_gaussian(self.current_frame_idx, block=block, timeout=0.25 if block else 0.0)

    def get_current_frame_gpu(self, force_load=False):
        if self.num_frames == 0:
            return None

        # Try to reuse the slot; if not matching, attempt a non-blocking upload
        entry = self.gaussian_cache[self.current_frame_idx % len(self.gaussian_cache)]
        if entry is None or entry[1] != self.current_frame_idx:
            entry = self.upload_to_gpu(self.current_frame_idx, block=False)
        return entry[0] if entry else None

    def toggle_pause(self):
        self.paused = not self.paused
        self.last_update_time = time.time()

    def step_forward(self):
        if self.num_frames == 0:
            return
        self.prev_frame_idx = self.current_frame_idx
        self.current_frame_idx = (self.current_frame_idx + self.frame_step_size) % self.num_frames
        self.last_update_time = time.time()
        self._cache.set_current_index(self.current_frame_idx)

    def set_speed(self, speed):
        self.playback_speed = max(0.1, speed)
        self.frame_interval = 1.0 / (self.fps * self.playback_speed)

    def set_program(self, program):
        self.program = program

    def get_total_frame_memory_gb(self):
        """Approximate memory of cached frames (current + ahead)."""
        total = 0
        cache = self._cache
        if cache is None:
            return 0.0
        with cache._lock:
            for _, (_, flat) in cache._have.items():
                total += 0 if flat is None else flat.nbytes
        return total / (1024 ** 3)
