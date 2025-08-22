import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
# from video_manager import GaussianVideo
from video_stream_manager import GaussianVideo
# from video_stream_manager2 import GaussianVideo
from tkinter import filedialog
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from PIL import Image
import gc
import torch

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_camera = util.Camera(720, 1280)
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

g_video_total_memory = 0.0

# g_video = GaussianVideo(None, fps=12)
g_video = GaussianVideo(None, fps=30, renderer_type=g_renderer_idx, cache_ahead=5)

def impl_glfw_init():
    window_name = "Panopticon"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def load_texture(path):
    img = Image.open(path)
    img_data = img.convert("RGBA").tobytes()

    width, height = img.size

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data
    )
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture

def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, g_video_total_memory, \
        g_render_mode, g_render_mode_tables, g_video

    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()

    play_tex = load_texture("assets/play.png")
    pause_tex = load_texture("assets/pause.png")
    forward_tex = load_texture("assets/forward.png")
    upload_tex = load_texture("assets/open_folder.png")
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        raise ImportError("Forcing OpenGL")
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
        print("Using OpenGL")
    else:
        g_renderer_idx = BACKEND_CUDA
        print("Using CUDA")

    g_renderer = g_renderer_list[g_renderer_idx]
    g_video.renderer_type = g_renderer_idx
    if (g_renderer_idx == BACKEND_OGL):
        g_video.set_program(g_renderer_list[BACKEND_OGL].program)

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()

        # caching
        if g_video and g_video.num_frames > 0:
            frame_changed = False
            if not g_video.paused:
                frame_changed = g_video.update()  # advances index + pokes prefetcher

            # Ensure we have a GPU buffer for the *current* frame.
            # First try non-blocking (keeps drawing previous frame if load not ready yet),
            # then fall back to a short blocking upload if needed.
            gaussians_gpu = g_video.get_current_frame_gpu(force_load=False)
            if gaussians_gpu is None or frame_changed:
                gaussians_gpu = g_video.get_current_frame_gpu(force_load=True)

            # CPU GaussianData
            gaussians_cpu = g_video.get_current_frame_cpu(block=False)

            if gaussians_cpu is not None and gaussians_gpu is not None:
                g_renderer.update_preloaded_gaussian_data(gaussians_cpu, metadata=gaussians_gpu)
                if g_auto_sort or frame_changed:
                    g_renderer.sort_and_update(g_camera)

        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            # Not yet integrated with vid
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    g_video.renderer_type = g_renderer_idx
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
                # imgui.text(f"mem = {g_video_total_memory:.3f}")

                # changed, g_renderer.reduce_updates = imgui.checkbox(
                #         "reduce updates", g_renderer.reduce_updates,
                #     )

                # imgui.text(f"# of Gaus = {len(gaussians)}")
                # if imgui.button(label='open ply'):
                #     file_path = filedialog.askopenfilename(title="open ply",
                #         initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                #         filetypes=[('ply file', '.ply')]
                #         )
                #     if file_path:
                #         try:
                #             gaussians = util_gau.load_ply(file_path)
                #             g_renderer.update_gaussian_data(gaussians)
                #             g_renderer.sort_and_update(g_camera)
                #         except RuntimeError as e:
                #             pass

                if imgui.button(label='open video folder'):
                    folder_path = filedialog.askdirectory(title="Open Gaussian Video Folder")
                    if folder_path:
                        # Rebuild the video with cache
                        new_video = GaussianVideo(
                            folder_path,
                            fps=12,
                            renderer_type=g_renderer_idx,
                            cache_ahead=5,
                            use_memmap=False
                        )

                        if g_renderer_idx == BACKEND_OGL:
                            new_video.set_program(g_renderer_list[BACKEND_OGL].program)

                        # wait for background thread to finish loading first frame
                        first_frame_cpu = new_video.get_current_frame_cpu(block=True)
                        if first_frame_cpu is not None:
                            new_video.upload_to_gpu(0, block=True)

                        # swap in
                        g_video = new_video


                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
                # sort button
                # if imgui.button(label='sort Gaussians'):
                #     g_renderer.sort_and_update(g_camera)
                # imgui.same_line()
                # changed, g_auto_sort = imgui.checkbox(
                #         "auto sort", g_auto_sort,
                #     )
                # if g_auto_sort:
                #     g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )

                imgui.end()

        # place at bottom of main viewport
        viewport = imgui.get_main_viewport()
        win_pos = (viewport.pos.x, viewport.pos.y + viewport.size.y - 65)
        win_size = (viewport.size.x, 65)

        imgui.set_next_window_position(win_pos[0], win_pos[1])
        imgui.set_next_window_size(win_size[0], win_size[1])

        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)

        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
        )

        if imgui.begin("VideoControlBar", True, flags):
            if g_video.paused:
                if imgui.image_button(play_tex, 16, 16):
                    g_video.toggle_pause()
            else:
                if imgui.image_button(pause_tex, 16, 16):
                    g_video.toggle_pause()

            imgui.same_line()
            changed_frame = False
            imgui.same_line()
            if imgui.image_button(forward_tex, 16, 16):
                g_video.step_forward()
                changed_frame = True

            imgui.same_line()
            progress = g_video.current_frame_idx / max(1, g_video.num_frames - 1)
            bar_width = imgui.get_content_region_available_width() - 100
            bar_height = 20

            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 1.0, 1.0, 1.0, 1.0)
            imgui.progress_bar(progress, size=(bar_width, bar_height))
            imgui.pop_style_color()

            # if changed:
            #     g_video.set_frame(new_idx)
            #     changed_frame = True
            if changed_frame:
                gaussians = g_video.get_current_frame_cpu()
                gaussians_gpu = g_video.get_current_frame_gpu()
                g_renderer.update_preloaded_gaussian_data(gaussians, metadata=gaussians_gpu)
                g_renderer.sort_and_update(g_camera) 

            imgui.same_line()
            imgui.text(f"{g_video.current_frame_idx}/{g_video.num_frames}")

            imgui.same_line()
            if imgui.image_button(upload_tex, 16, 16):
                folder_path = filedialog.askdirectory(title="Open Gaussian Video Folder")
                if folder_path:
                    # Rebuild the video with cache
                    new_video = GaussianVideo(
                        folder_path,
                        fps=12,
                        renderer_type=g_renderer_idx,
                        cache_ahead=5, 
                        use_memmap=False 
                    )
                    if g_renderer_idx == BACKEND_OGL:
                        new_video.set_program(g_renderer_list[BACKEND_OGL].program)

                    # swap in
                    g_video = new_video


            for i, mode_name in enumerate(g_render_mode_tables):
                if i > 0:
                    imgui.same_line()

                # compute whether this button should be highlighted BEFORE changing g_render_mode
                is_active = (i == g_render_mode)

                if is_active:
                    imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.7, 0.2, 0.5)  # active green

                if imgui.button(mode_name):
                    g_render_mode = i
                    g_renderer.set_render_mod(g_render_mode - 4)

                if is_active:
                    imgui.pop_style_color()

        imgui.end()
        imgui.pop_style_var(1)

        # Ensure autosorting for paused video (so there are no artifacts)
        if g_video.paused:
            g_renderer.sort_and_update(g_camera)
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

    # Clean up
    del g_video
    del g_renderer
    del g_renderer_list
    del g_camera
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        gl.glDeleteTextures([play_tex, pause_tex, forward_tex, upload_tex])
    except Exception:
        pass


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="Panopticon editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
