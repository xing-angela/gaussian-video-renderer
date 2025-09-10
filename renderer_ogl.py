from OpenGL import GL as gl
import util
import util_gau
import numpy as np

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()

    def update_preloaded_gaussian_data(self, gaus: util_gau.GaussianData, metadata=0):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()

        vendor = gl.glGetString(gl.GL_VENDOR)
        renderer = gl.glGetString(gl.GL_RENDERER)
        version = gl.glGetString(gl.GL_VERSION)

        print("OpenGL Vendor:", vendor.decode())
        print("OpenGL Renderer:", renderer.decode())
        print("OpenGL Version:", version.decode())

        # sphere culling
        self.culling_enabled = False
        self.sphere_center = np.array([0.0, 0.0, 0.0])
        self.culling_radius = 2.0**2

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def update_preloaded_gaussian_data(self, gaus: util_gau.GaussianData, metadata=0):
        self.gaussians = gaus
        # load gaussian geometry
        if (metadata == 0):
            gaussian_data = gaus.flat()
            self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                            bind_idx=0,
                                                            buffer_id=self.gau_bufferid)
        else:
            self.gau_bufferid = metadata
            util.bind_storage_buffer(0, metadata)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def set_sphere_culling(self, center: np.ndarray, radius: float, enabled: bool):
        self.culling_enabled = enabled
        self.sphere_center = center
        self.culling_radius = radius**2

    def draw(self):
        gl.glUseProgram(self.program)

        # sphere culling
        util.set_uniform_1int(self.program, int(self.culling_enabled), "u_culling_enabled")
        if self.culling_enabled:
            util.set_uniform_v3(self.program, self.sphere_center, "u_sphere_center")
            util.set_uniform_1f(self.program, self.culling_radius, "u_sphere_radius_sq")

        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)


def create_sphere_vertices(radius, rings, sectors):
    """
    Generates vertices for a wireframe sphere.
    This corrected version builds a list of line-segment endpoints.
    """
    lines = []
    # Generate horizontal lines (circles of latitude)
    for i in range(rings + 1):
        phi = np.pi * i / rings
        for j in range(sectors):
            theta1 = 2 * np.pi * j / sectors
            theta2 = 2 * np.pi * (j + 1) / sectors

            # Vertex 1 of the line segment
            x1 = radius * np.sin(phi) * np.cos(theta1)
            y1 = radius * np.cos(phi)
            z1 = radius * np.sin(phi) * np.sin(theta1)
            
            # Vertex 2 of the line segment
            x2 = radius * np.sin(phi) * np.cos(theta2)
            y2 = radius * np.cos(phi)
            z2 = radius * np.sin(phi) * np.sin(theta2)
            
            lines.append([x1, y1, z1])
            lines.append([x2, y2, z2])

    # Generate vertical lines (lines of longitude)
    for j in range(sectors + 1):
        theta = 2 * np.pi * j / sectors
        for i in range(rings):
            phi1 = np.pi * i / rings
            phi2 = np.pi * (i + 1) / rings
            
            # Vertex 1 of the line segment
            x1 = radius * np.sin(phi1) * np.cos(theta)
            y1 = radius * np.cos(phi1)
            z1 = radius * np.sin(phi1) * np.sin(theta)

            # Vertex 2 of the line segment
            x2 = radius * np.sin(phi2) * np.cos(theta)
            y2 = radius * np.cos(phi2)
            z2 = radius * np.sin(phi2) * np.sin(theta)

            lines.append([x1, y1, z1])
            lines.append([x2, y2, z2])
            
    return np.array(lines, dtype=np.float32)