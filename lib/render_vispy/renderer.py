import os
import os.path as osp

import cv2
import numpy as np
import vispy
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import quat2mat
from vispy import app, gloo

# import torch
# import copy
import OpenGL.GL as gl
from lib.render_vispy.frustum import Camera3D
from lib.render_vispy.model3d import Model3D, load_models  # noqa

os.environ["PYOPENGL_PLATFORM"] = "egl"

cur_dir = osp.dirname(osp.abspath(__file__))

# app backends: glfw, pyglet, egl
# gl backends: gl2, pyopengl2, gl+
app_backend = "egl"
gl_backend = "gl2"  # "pyopengl2"  # speed: 'gl+' < 'gl2' < 'pyopengl2'
vispy.use(app=app_backend, gl=gl_backend)
print("vispy uses app: {}, gl: {}".format(app_backend, gl_backend))


def shader_from_path(shader_filename):
    shader_path = osp.join(cur_dir, "./shader", shader_filename)
    assert osp.exists(shader_path)
    with open(shader_path, "r") as f:
        return f.read()


def singleton(cls):
    instances = {}

    def get_instance(size, cam, model_paths=None, scale_to_meter=1.0, gpu_id=None):
        if cls not in instances:
            instances[cls] = cls(size, cam, model_paths, scale_to_meter, gpu_id)
        return instances[cls]

    return get_instance


@singleton  # Don't throw GL context into trash when having more than one Renderer instance
class Renderer(app.Canvas):
    """
    NOTE: internally convert RGB to BGR
    """

    def __init__(self, size, cam, model_paths=None, scale_to_meter=1.0, gpu_id=None):
        """
        size: (width, height)
        """
        app.Canvas.__init__(self, show=False, size=size)
        width, height = size
        self.height = height
        self.width = width
        self.shape = (height, width)  # height, width

        # OpenGL is right-hand with (x+ right, y+ up and z- is forward)
        # OpenCV is right-hand with (x+ right, y- up and z+ is forward)
        # We define everything in OUR left-hand global system (x+ is right, y+ is up, z+ is forward)
        # We therefore must flip Y for OpenCV and Z for OpenGL for every operation
        self.opengl_zdir_neg = np.eye(4, dtype=np.float32)
        # self.opengl_zdir_neg[2, 2] = -1
        self.opengl_zdir_neg[1, 1], self.opengl_zdir_neg[2, 2] = -1, -1

        self.set_cam(cam)
        self.setup_views()

        # Set up shader programs
        # fmt: off
        _vertex_code_pointcloud = shader_from_path("point_cloud.vs")
        _fragment_code_pointcloud = shader_from_path("point_cloud.frag")  # varying

        _vertex_code_colored = shader_from_path("colored.vs")
        _fragment_code_colored = shader_from_path("colored.frag")

        # use colored vertex shader
        _fragment_code_bbox = shader_from_path("bbox.frag")

        _vertex_code_textured = shader_from_path("textured.vs")
        _fragment_code_textured = shader_from_path("textured.frag")

        _vertex_code_background = shader_from_path("background.vs")
        _fragment_code_background = shader_from_path("background.frag")
        self.program_pcl = gloo.Program(_vertex_code_pointcloud, _fragment_code_pointcloud)
        self.program_col = gloo.Program(_vertex_code_colored, _fragment_code_colored)
        self.program_bbox = gloo.Program(_vertex_code_colored, _fragment_code_bbox)
        self.program_tex = gloo.Program(_vertex_code_textured, _fragment_code_textured)
        self.program_bg = gloo.Program(_vertex_code_background, _fragment_code_background)
        # fmt: on

        # Texture where we render the color/depth and its FBO
        self.col_tex = gloo.Texture2D(shape=self.shape + (3,))
        self.fbo = gloo.FrameBuffer(self.col_tex, gloo.RenderBuffer(self.shape))
        self.fbo.activate()
        # gloo.set_state(depth_test=True, blend=False, cull_face=True)
        gloo.set_state(depth_test=True, blend=False, cull_face=False)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        # gl.glDisable(gl.GL_LINE_SMOOTH)
        gloo.set_clear_color((0.0, 0.0, 0.0))
        gloo.set_viewport(0, 0, *self.size)

        # Set up background render quad in NDC
        quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        tex = [[0, 1], [1, 1], [1, 0], [0, 0]]
        vertices_type = [
            ("a_position", np.float32, 2),
            ("a_texcoord", np.float32, 2),
        ]
        collated = np.asarray(list(zip(quad, tex)), vertices_type)
        self.bg_vbuffer = gloo.VertexBuffer(collated)
        self.bg_ibuffer = gloo.IndexBuffer([0, 1, 2, 0, 2, 3])

        self.models = None
        if model_paths is not None:
            self._load_models(model_paths, scale_to_meter=scale_to_meter)

    def _load_models(self, model_paths, scale_to_meter=1.0):
        self.models = load_models(model_paths, scale_to_meter=scale_to_meter)

    def set_cam(self, cam, clip_near=0.1, clip_far=100.0):
        self.cam = cam
        self.clip_near = clip_near
        self.clip_far = clip_far
        self.mat_proj = self.projective_matrix(cam, 0, 0, self.shape[1], self.shape[0], clip_near, clip_far)

    def clear(self, color=True, depth=True):
        gloo.clear(color=color, depth=depth)

    def setup_views(self):
        self.view = dict()

        self.view["back"] = np.eye(4)
        self.view["back"][:3, :3] = axangle2mat(axis=[1, 0, 0], angle=15 * np.pi / 180)
        self.view["back"][:3, 3] = [0, -2.0, -3.25]

        self.view["center"] = np.eye(4)

        self.view["front"] = np.eye(4)
        self.view["front"][:3, :3] = axangle2mat(axis=[1, 0, 0], angle=9 * np.pi / 180)
        self.view["front"][:3, 3] = [0, 0, 3.25]

        self.view["show"] = np.eye(4)
        self.view["show"][:3, :3] = axangle2mat(axis=[1, 0, 0], angle=5 * np.pi / 180) @ axangle2mat(
            axis=[0, 1, 0], angle=-15 * np.pi / 180
        )
        self.view["show"][:3, 3] = [-3.5, -1, -5]
        self.used_view = "center"

    def finish(self, only_color=False, to_255=False):
        # NOTE: the colors in Model3D were converted into BGR, so the rgb loaded here is BGR
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_FLOAT)
        # Read buffer and flip X
        rgb = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape + (3,))[::-1, :]
        if to_255:
            rgb = (rgb * 255 + 0.5).astype(np.uint8)
        if only_color:
            return rgb

        im = gl.glReadPixels(
            0,
            0,
            self.size[0],
            self.size[1],
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
        )
        # Read buffer and flip X
        dep = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape + (1,))[::-1, :]

        # Convert z-buffer to depth map
        mult = (self.clip_near * self.clip_far) / (self.clip_near - self.clip_far)
        addi = self.clip_far / (self.clip_near - self.clip_far)
        bg = dep == 1
        dep = mult / (dep + addi)
        dep[bg] = 0
        return rgb, np.squeeze(dep)

    def compute_rotation(self, eye_point, look_point):
        up = [0, 1, 0]
        if eye_point[0] == 0 and eye_point[1] != 0 and eye_point[2] == 0:
            up = [0, 0, -1]
        rot = np.zeros((3, 3))
        rot[2] = look_point - eye_point
        rot[2] /= np.linalg.norm(rot[2])
        rot[0] = np.cross(rot[2], up)
        rot[0] /= np.linalg.norm(rot[0])
        rot[1] = np.cross(rot[0], -rot[2])
        return rot.T

    def _validate_pose(self, pose, rot_type="mat"):
        if rot_type == "mat":
            res = pose
            if pose.shape[0] == 3:
                res = np.concatenate(
                    (
                        pose,
                        np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4),
                    ),
                    axis=0,
                )
        elif rot_type == "quat":
            res = np.eye(4)
            res[:3, :3] = quat2mat(pose[:4])
            res[:3, 3] = pose[4:7]
        else:
            raise ValueError(f"wrong rot_type: {rot_type}")
        return res  # 4x4

    def draw_detection_boundingbox(
        self,
        pose,
        extents,
        view="center",
        is_gt=False,
        thickness=1.5,
        centroid=0,
        rot_type="mat",
    ):
        """
        centroid: [0,0,0]
        """
        assert view in ["front", "top", "back", "show", "center"]
        pose = self._validate_pose(pose, rot_type=rot_type)
        xsize, ysize, zsize = extents
        # fmt: off
        bb = np.asarray([[-xsize / 2, ysize / 2, zsize / 2], [xsize / 2, ysize / 2, zsize / 2],
                         [-xsize / 2, -ysize / 2, zsize / 2], [xsize / 2, -ysize / 2, zsize / 2],
                         [-xsize / 2, ysize / 2, -zsize / 2], [xsize / 2, ysize / 2, -zsize / 2],
                         [-xsize / 2, -ysize / 2, -zsize / 2], [xsize / 2, -ysize / 2, -zsize / 2]])
        # Set up rendering data
        bb += centroid
        if is_gt:
            colors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        else:
            colors = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
        # fmt: on
        indices = [
            0,
            1,
            0,
            2,
            3,
            1,
            3,
            2,
            4,
            5,
            4,
            6,
            7,
            5,
            7,
            6,
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
        ]

        vertices_type = [
            ("a_position", np.float32, 3),
            ("a_color", np.float32, 3),
        ]
        collated = np.asarray(list(zip(bb, colors)), vertices_type)

        self.program_bbox.bind(gloo.VertexBuffer(collated))

        # Flip from our system and .T since OpenGL is column-major
        self.program_bbox["u_model"] = (self.opengl_zdir_neg.dot(pose)).T
        self.program_bbox["u_view"] = self.view[view].T
        self.program_bbox["u_projection"] = self.mat_proj

        gloo.set_line_width(width=thickness)
        self.program_bbox.draw("lines", gloo.IndexBuffer(indices))
        gloo.set_line_width(width=1.0)

    def draw_camera(
        self,
        pose=None,
        color=[0, 1, 0],
        scaler=1.0,
        view="center",
        rot_type="mat",
    ):
        if pose is None:
            pose = np.eye(4)
        else:
            pose = self._validate_pose(pose)

        assert view in ["front", "top", "back", "show"]

        cam = Camera3D(color=color, scaler=scaler)

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        # Flip from our system and .T since OpenGL is column-major
        mv = (self.opengl_zdir_neg.dot(pose)).T

        self.program_bbox.bind(cam.vertex_buffer)

        self.program_bbox["u_model"] = mv
        self.program_bbox["u_view"] = self.view[view].T
        self.program_bbox["u_projection"] = self.mat_proj

        gloo.set_line_width(width=2.5)
        self.program_bbox.draw("lines", cam.index_buffer)
        gloo.set_line_width(width=1.0)

    def draw_pointcloud(self, points, colors=None, s_color=None, radius=1.0, view="center"):

        assert view in ["center", "front", "top", "back", "show"]

        points = np.copy(points)

        if colors is None:
            if s_color is None:
                colors = np.asarray(
                    np.ones((points.shape[0], 1)) * [1, 0, 0, 1],
                    dtype=np.float32,
                )
            else:
                colors = np.asarray(
                    np.ones((points.shape[0], 1)) * [1, 0, 0, 1],
                    dtype=np.float32,
                )

        radius = np.ones((points.shape[0]), dtype=np.float32) * radius
        # fmt: off
        data = np.zeros(points.shape[0], [("a_position", np.float32, 3), ("a_bg_color", np.float32, 4),
                                          ("a_fg_color", np.float32, 4), ("a_size", np.float32, 1)])
        # fmt: on

        data["a_position"] = points
        data["a_size"] = radius
        data["a_fg_color"] = 0, 0, 0, 0.5
        data["a_fg_color"] = colors

        self.program_pcl.bind(gloo.VertexBuffer(data))
        self.program_pcl["u_linewidth"] = 0.75
        self.program_pcl["u_antialias"] = 1.00
        self.program_pcl["u_model"] = self.opengl_zdir_neg.dot(np.eye(4)).T
        self.program_pcl["u_view"] = self.view[view].T
        self.program_pcl["u_projection"] = self.mat_proj

        self.program_pcl.draw("points")

    def draw_background(self, image):
        """bgr image."""
        self.program_bg["u_tex"] = gloo.Texture2D(image)
        self.program_bg.bind(self.bg_vbuffer)
        self.program_bg.draw("triangles", self.bg_ibuffer)
        gloo.clear(color=False, depth=True)  # Clear depth

    def draw_model(
        self,
        model_or_id,
        pose,
        ambient=0.5,
        specular=0,
        shininess=1,
        light_dir=(0, 0, -1),
        light_col=(1, 1, 1),
        view="center",
        rot_type="mat",
    ):
        """
        pose: 4x4
        """
        assert view in ["front", "top", "back", "center", "show"]
        pose = self._validate_pose(pose, rot_type=rot_type)
        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        # Flip from our system and .T since OpenGL is column-major
        m = (self.opengl_zdir_neg.dot(pose)).T

        if isinstance(model_or_id, int):
            # how to avoid re-binding for existing models???
            assert self.models is not None, self.models
            model = self.models[model_or_id]
        else:
            model = model_or_id

        used_program = self.program_col
        if model.texcoord is not None:
            used_program = self.program_tex
            used_program["u_tex"] = model.texture

        used_program.bind(model.vertex_buffer)
        used_program["u_view"] = self.view[view].T
        used_program["u_projection"] = self.mat_proj
        used_program["u_light_dir"] = light_dir
        used_program["u_light_col"] = light_col
        used_program["u_ambient"] = ambient
        used_program["u_specular"] = specular
        used_program["u_shininess"] = shininess

        used_program["u_model"] = m

        used_program.draw("triangles", model.index_buffer)

    def show_scene(self, points, colors, models, poses):
        angular_err, trans_err = np.inf, np.inf

        camera_locations = np.asarray([[7.0, 7.0, 10.0]])
        lookat_locations = np.asarray([[0.0, 0.0, 20.0]])

        # for pose in poses:
        #    lookat = pose[:3, 3].copy()
        #    position = lookat.copy()
        #    position[0] = 0.

        #    camera_locations.append(position)
        #    lookat_locations.append(lookat)

        view = np.eye(4)
        steps = 300

        for path_idx in range(len(camera_locations)):
            cam_loc, lookat_loc = (
                camera_locations[path_idx],
                lookat_locations[path_idx],
            )

            t_start = np.copy(view[:3, 3])
            t_end = cam_loc

            for i in range(steps):

                view[:3, :3] = np.eye(3)

                eye_pos = t_start + ((t_end - t_start) * (i / steps))

                rot = self.compute_rotation(eye_pos, lookat_loc).T
                view[:3, :3] = rot
                view[:3, 3] = -rot @ eye_pos

                print(view)

                self.view["show"] = view.copy()

                flip = np.eye(4)
                self.view["show"] = flip.dot(self.view["show"])

                self.clear()
                self.draw_pointcloud(points, colors, view="show")

                for moodel, pose in zip(models, poses):
                    self.draw_model(moodel, pose, view="show")

                cv2.imshow("show", self.finish()[0])
                cv2.waitKey(10)

    def projective_matrix(self, cam, x0, y0, w, h, nc, fc):

        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        # fmt: off
        proj = np.array([
            [2 * cam[0, 0] / w, -2 * cam[0, 1] / w, (-2 * cam[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * cam[1, 1] / h, (2 * cam[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ])
        # fmt: on

        # Compensate for the flipped image
        return proj.T


if __name__ == "__main__":
    from transforms3d.axangles import axangle2mat
    import time
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    import ref

    ref_key = "lm_full"
    data_ref = ref.__dict__[ref_key]

    classes = data_ref.objects
    model_paths = data_ref.model_paths
    # models = load_models()

    cam = data_ref.camera_matrix
    width = data_ref.width
    height = data_ref.height

    obj_ids = [i for i in range(len(model_paths))]

    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([0, 0, 0.7], dtype=np.float32)
    pose = np.hstack([R, t.reshape((3, 1))])
    pose1 = np.hstack([R, 0.1 + t.reshape((3, 1))])
    pose2 = np.hstack([R, t.reshape((3, 1)) - 0.1])
    pose3 = np.hstack([R, t.reshape((3, 1)) - 0.05])
    pose4 = np.hstack([R, t.reshape((3, 1)) + 0.05])

    # tensor_kwargs = {'device': torch.device('cuda'), 'dtype': torch.float32}
    # image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    ren = Renderer(
        size=(width, height),
        cam=cam,
        model_paths=model_paths,
        scale_to_meter=data_ref.vertex_scale,
    )

    # rendering
    runs = 0
    t_render = 0
    for j in tqdm(range(200)):
        for obj_id, cls_name in enumerate(classes):
            t0 = time.perf_counter()

            poses = [pose, pose1, pose2, pose3, pose4]
            obj_ids = [obj_id, obj_id, obj_id, obj_id, obj_id]
            # poses = [pose]
            # obj_ids = [obj_id]
            ren.clear()
            # ren.draw_background(background)
            for i in range(len(poses)):
                # cur_model = models[obj_ids[i]]
                cur_model = ren.models[obj_ids[i]]
                ren.draw_model(obj_ids[i], poses[i])
                ren.draw_detection_boundingbox(
                    poses[i],
                    [cur_model.xsize, cur_model.ysize, cur_model.zsize],
                )
            im, depth = ren.finish()
            # ren.finish(image_tensor=image_tensor)

            # im = (im.cpu().numpy() + 0.5).astype(np.uint8)
            t_render += time.perf_counter() - t0
            runs += 1
            if False:
                fig = plt.figure(frameon=False, dpi=200)
                plt.subplot(1, 2, 1)
                # im = (image_tensor[:, :, :3].cpu().numpy() + 0.5).astype('uint8')
                plt.imshow(im[:, :, [2, 1, 0]])  # rgb
                plt.axis("off")
                plt.title("{} color".format(cls_name))

                plt.subplot(1, 2, 2)
                plt.imshow(depth)
                plt.axis("off")
                plt.title("{} depth".format(cls_name))

                plt.show()
    print("{}s {}fps".format(t_render / runs, runs / t_render))
    # 5 objs: 0.01251343576113383s 79.91410345557973fps
    # 1 obj:  0.00796224382718404s 125.59273763833781fps
