# 4D Gaussian Splatting Video Viewer

This project is a custom video-capable viewer built on top of the excellent [Tiny Gaussian Splatting Viewer]([https://github.com/ashawkey/tiny-gs-viewer](https://github.com/limacv/GaussianSplattingViewer)) by Li Ma. The original viewer supports OpenGL and CUDA backends for static 3D Gaussian splat scenes. Our extension enables interactive playback and exploration of **4D Gaussian splats** (i.e., time-varying sequences).

---

## 🔍 Description

We build a custom viewer for our **4D Gaussian splats** on top of the Tiny Gaussian Splatting Viewer by Li Ma, which originally supports OpenGL and CUDA backends for static scenes.

Our viewer extends this functionality to enable:

- **Interactive video camera control**
- **Real-time playback** at arbitrary frame rates
- **Frame-by-frame navigation** through dynamic sequences

Rendering is performed using **OpenGL shaders** and **Shader Storage Buffer Objects (SSBOs)**. Each video frame is represented as a precomputed Gaussian point cloud stored in `.ply` format. To accelerate playback, we cache a flattened, OpenGL-ready version of each frame on the CPU.

At runtime, when advancing to a new frame, the viewer copies this preprocessed buffer to the GPU for rendering. Between frame updates, the current buffer remains active, allowing continuous rendering and smooth scene exploration.

---
