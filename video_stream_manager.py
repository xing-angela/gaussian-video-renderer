import os
import re
import time
import threading
import numpy as np
import util
from util_gau import GaussianData
from renderer_cuda import gaus_cuda_from_cpu

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.findall(r'\d+|\D+', os.path.basename(s))]

def _read_timestamps_index(folder_path, default_bin_dir="bin"):
    """
    Gets all .bin files
    """
    bin_dir = os.path.join(folder_path, default_bin_dir)
    if os.path.isdir(bin_dir):
        files = [os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith(".bin")]
    else:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".bin")]
    files.sort(key=_natural_key)
    # in the files list, skip every other file
    # files = files[::2]
    return files, "float32"


def _infer_sh_dim_from_file(path, base=11):
    """Infer the SH dimension from a .bin file."""
    c = np.fromfile(path, dtype=np.float32).size

    for s in range(0, 65):
        dim = 3*(s+1)**2
        if c % (base + dim) == 0:
            return dim

class _FrameCache:
    """Frame cache with background prefetch for .bin frames."""
    def __init__(self, frame_paths, feature_dim, sh_dim, cache_ahead=5):
        self.paths = frame_paths
        self.N = len(frame_paths)
        self.D = feature_dim
        self.SH = sh_dim
        self.cache_ahead = max(0, cache_ahead)

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
        t1 = time.perf_counter()
        arr = np.fromfile(path, dtype=np.float32)
        # print(f"file read: {(time.perf_counter() - t1) * 1e3} ms")
        if arr.size % self.D != 0:
            raise RuntimeError(f"Bad frame size in {path}: total={arr.size}, D={self.D}")
        arr = arr.reshape((-1, self.D))

        xyz = arr[:, 0:3]
        rot = arr[:, 3:7]
        scale = arr[:, 7:10]
        opacity = arr[:, 10:11]
        sh = arr[:, 11:11 + self.SH]
        g = GaussianData(xyz=xyz, rot=rot, scale=scale, opacity=opacity, sh=sh)
        # print(f"full funtion: {(time.time() - start_time) * 1e3} ms")
        # print(f"# gaussians: {len(g.xyz)}")

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
        self.gaussian_cache = np.array([None] * 90, dtype=object)

        # Disk frames
        self.folder_path = folder_path
        self.num_frames = 0

        # Layout info for .bin
        self.sh_dim = None
        self.feature_dim = None

        # Cache / loader
        self._cache = None
        self._frame_paths = []

        self._init_frames(cache_ahead=cache_ahead)

    # -------------- Setup --------------
    def _init_frames(self, cache_ahead):
        if not self.folder_path:
            self.num_frames = 0
            self._frame_paths = []
            self._cache = _FrameCache([], self.feature_dim, self.sh_dim,
                                    cache_ahead=cache_ahead)
            return

        files, dtype = _read_timestamps_index(self.folder_path)
        if not files:
            self.num_frames = 0
            self._frame_paths = []
            self._cache = _FrameCache([], self.feature_dim, self.sh_dim,
                                    cache_ahead=cache_ahead)
            return

        if dtype != "float32":
            raise RuntimeError(f"Expected float32 in timestamps.idx, got {dtype}")

        # Get the SH dimension from the first file
        sh_dim = _infer_sh_dim_from_file(files[0])
        self.sh_dim = sh_dim
        self.feature_dim = 3 + 4 + 3 + 1 + self.sh_dim  # xyz + rot + scale + opacity + sh

        self._frame_paths = files
        self.num_frames = len(files)

        self._cache = _FrameCache(self._frame_paths, self.feature_dim, self.sh_dim,
                                cache_ahead=cache_ahead)
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
            flat = self._cache.get_flat(idx, block=block, timeout=0.25 if block else 0.0)
            if flat is None:
                # keep previous GPU buffer
                return prev_entry

            buffer_id = prev_entry[0] if prev_entry else None

            t1 = time.perf_counter()
            buffer_id = util.set_storage_buffer_data(
                self.program, "gaussian_data", flat, bind_idx=0, buffer_id=buffer_id
            )
            t2 = time.perf_counter()
            # print(f"upload time for frame {idx}: {(t2 - t1) * 1e3} ms")
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
            print("Cannot cache all gaussian frames.")
        for idx in indices:
            self.upload_to_gpu(idx, block=False)

    # -------------- Playback --------------
    def update(self):
        """Advance to the next frame if enough time has passed."""
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
        if force_load:
            entry = self.upload_to_gpu(self.current_frame_idx, block=True)
            return entry[0] if entry else None
        
        # try to reuse cached slot
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
        return 0.0
