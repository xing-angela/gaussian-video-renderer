import os
import time
import util
import util_gau
import collections
import numpy as np
from renderer_cuda import gaus_cuda_from_cpu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class GaussianVideo:

    def __init__(self, folder_path, fps=30.0, renderer_type=0):
        self.num_frames = 0
        self.frames = self.load_frames(folder_path)
        self.fps = fps
        self.playback_speed = 1.0  # multiplier
        self.frame_interval = 1.0 / (fps * self.playback_speed)
        self.last_update_time = time.time()
        self.current_frame_idx = 0
        self.paused = False
        self.gaussian_cache = np.array([None] * fps) # opengl buffer ids or cuda gaussians
        self.renderer_type = renderer_type # opengl = 0, cuda = 1
        self.program = None
        self.frame_step_size = 1
        self.prev_frame_idx = 0

    def load_frames(self, folder_path):
        if not folder_path:
            self.num_frames = 0
            self.frames = []
            return

        frame_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.ply')
        ])
        self.num_frames = len(frame_files)

        with ThreadPoolExecutor() as executor:
            self.frames = list(tqdm(executor.map(util_gau.load_ply, frame_files), total=len(frame_files)))

    def upload_to_gpu(self, idx):
        cache_idx = idx % len(self.gaussian_cache)
        print("uploading frame " + str(idx) + " to cache idx " + str(cache_idx))
        gaus = self.frames[idx]
        gpu_data = None
        prev_entry = self.gaussian_cache[cache_idx]
        if prev_entry is None or prev_entry[1] != idx:
            if (self.renderer_type == 0):
                assert self.program != None
                gaussian_data = gaus.flat()
                buffer_id = prev_entry[0] if prev_entry is not None else None
                buffer_id = util.set_storage_buffer_data(
                    self.program, "gaussian_data", gaussian_data,
                    bind_idx=0, buffer_id=buffer_id
                )
                gpu_data = buffer_id
            elif (self.renderer_type == 1):
                gpu_data = gaus_cuda_from_cpu(gaus)
            self.gaussian_cache[cache_idx] = (gpu_data, idx)

        return (gpu_data, idx)

    def upload_frames_to_gpu(self, indices):
        if len(indices) > len(self.gaussian_cache):
            print("Cannot cache all gaussian frames")
        
        for idx in indices:
            self.upload_to_gpu(idx)

    def update(self):
        """Advance to the next frame if enough time has passed."""
        now = time.time()
        frame_changed = False
        if now - self.last_update_time >= self.frame_interval:
            self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
            self.last_update_time = now
            frame_changed = True

        return frame_changed

    def prefetch_frames(self, num_frames, load_current=True):
        # Preload current + next N frames into cache
        start = 0 if load_current else 1
        total_frames = self.num_frames

        indices = [(self.current_frame_idx + i) % total_frames for i in range(start, num_frames + 1)]
        for idx in indices:
            self.upload_to_gpu(idx)

    def get_current_frame_cpu(self):
        if self.num_frames == 0:
            return None

        return self.frames[self.current_frame_idx]

    def get_current_frame_gpu(self, force_load=False):
        if self.num_frames == 0:
            return None

        cache_entry = None
        if force_load:
            cache_entry = self.upload_to_gpu(self.current_frame_idx)
        else:
            cache_entry = self.gaussian_cache[self.current_frame_idx % len(self.gaussian_cache)]
            assert cache_entry is not None, "cached frame is None"
            assert self.current_frame_idx == cache_entry[1], "cached frame is different from current frame"

        return cache_entry[0]

    def reset(self):
        self.current_frame_idx = 0
        self.last_update_time = time.time()

    def toggle_pause(self):
        self.paused = not self.paused
        self.last_update_time = time.time()

    def step_forward(self):
        self.current_frame_idx = (self.current_frame_idx + self.frame_step_size) % self.num_frames
        self.last_update_time = time.time()

    def step_backward(self):
        self.current_frame_idx = (self.current_frame_idx - self.frame_step_size) % self.num_frames
        self.last_update_time = time.time()

    def set_frame(self, idx):
        self.current_frame_idx = max(0, min(idx, self.num_frames - 1))
        self.last_update_time = time.time()

    def set_speed(self, speed):
        self.playback_speed = max(0.1, speed)
        self.frame_interval = 1.0 / (self.fps * self.playback_speed)

    def set_program(self, program):
        self.program = program

    def get_total_frame_memory_gb(self):
        total_bytes = sum(frame.flat().nbytes for frame in self.frames)
        return total_bytes / (1024 ** 3)    