import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from pyrender.constants import RenderFlags

from lib.pysixd import misc, renderer


class Renderer(renderer.Renderer):
    def __init__(self, width, height, mode="rgb+depth", bg_color=(0.0, 0.0, 0.0, 0.0)):
        """Constructor.

        :param width: Width of the rendered image.
        :param height: Height of the rendered image.
        :param mode: Rendering mode ('rgb+depth', 'rgb', 'depth').
        :param bg_color: Color of the background (R, G, B, A).
        """
        self.mode = mode
        # self.shading = shading
        self.bg_color = bg_color

        # Indicators whether to render RGB and/or depth image.
        self.render_rgb = self.mode in ["rgb", "rgb+depth"]
        self.render_depth = self.mode in ["depth", "rgb+depth"]

        # Structures to store object models and related info.
        self.models = {}
        self.mesh_nodes = {}  # store mesh nodes
        self.model_bbox_corners = {}
        # self.model_textures = {}

        self.height = height
        self.width = width

        self.r = pyrender.OffscreenRenderer(self.width, self.height)
        # yz_flip
        self.cv_to_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype="float32")

        self.camera = None
        self.cam_node = None
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=bg_color)

    def set_ambient_light(self, light):
        self.scene.ambient_light(light)

    def add_object(self, obj_id, model_path, **kwargs):
        """See base class."""
        # Color of the object model (the original color saved with the object model
        # will be used if None).
        surf_color = None
        if "surf_color" in kwargs:
            surf_color = kwargs["surf_color"]

        # Load the object model.
        obj_mesh = trimesh.load(model_path)
        mesh = pyrender.Mesh.from_trimesh(obj_mesh)
        self.models[obj_id] = mesh

        # Calculate the 3D bounding box of the model (will be used to set the near
        # and far clipping plane).
        pts = np.array(obj_mesh.vertices)
        bb = misc.calc_3d_bbox(pts[:, 0], pts[:, 1], pts[:, 2])
        self.model_bbox_corners[obj_id] = np.array(
            [
                [bb[0], bb[1], bb[2]],
                [bb[0], bb[1], bb[2] + bb[5]],
                [bb[0], bb[1] + bb[4], bb[2]],
                [bb[0], bb[1] + bb[4], bb[2] + bb[5]],
                [bb[0] + bb[3], bb[1], bb[2]],
                [bb[0] + bb[3], bb[1], bb[2] + bb[5]],
                [bb[0] + bb[3], bb[1] + bb[4], bb[2]],
                [bb[0] + bb[3], bb[1] + bb[4], bb[2] + bb[5]],
            ]
        )

        # Use the specified uniform surface color.
        if surf_color is not None:
            colors = np.tile(list(surf_color) + [1.0], [pts.shape[0], 1])
            mesh = pyrender.Mesh.from_points(obj_mesh.vertices, colors=colors)
            self.models[obj_id] = mesh

    def set_mesh_node_pose(self, obj_id, pose=np.eye(4)):
        # pose: pose_gl
        # must be set after cam_node and loading mesh
        if obj_id not in self.mesh_nodes:
            mesh = self.models[obj_id]
            mesh_node = self.scene.add(mesh, pose=pose, parent_node=self.cam_node)  # Object pose parent is cam
            self.mesh_nodes[obj_id] = mesh_node
        else:
            self.scene.set_pose(self.mesh_nodes[obj_id], pose)

    def remove_object(self, obj_id):
        """See base class."""
        del self.models[obj_id]
        del self.model_bbox_corners[obj_id]
        if obj_id in self.mesh_nodes:
            del self.mesh_nodes[obj_id]

    def set_cam(self, fx, fy, cx, cy, znear=0.01, zfar=10000.0):
        if self.camera is None and self.cam_node is None:
            self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
            self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        else:
            self.camera.fx = fx
            self.camera.fy = fy
            self.camera.cx = cx
            self.camera.cy = cy

    def render_object(self, obj_id, R, t, fx, fy, cx, cy, shading="flat"):
        """See base class."""
        pose_cv = np.eye(4, dtype=np.float32)
        pose_cv[:3, :3], pose_cv[:3, 3] = R, t.squeeze()
        pose_gl = self.cv_to_gl.dot(pose_cv)  # OpenCV to OpenGL camera system.

        # Calculate the near and far clipping plane from the 3D bounding box.
        bbox_corners = self.model_bbox_corners[obj_id]
        bbox_corners_ht = np.concatenate((bbox_corners, np.ones((bbox_corners.shape[0], 1))), axis=1).transpose()
        bbox_corners_eye_z = pose_cv[2, :].reshape((1, 4)).dot(bbox_corners_ht)
        clip_near = bbox_corners_eye_z.min()
        clip_far = bbox_corners_eye_z.max()

        self.set_cam(fx=fx, fy=fy, cx=cx, cy=cy, znear=clip_near, zfar=clip_far)
        self.set_mesh_node_pose(obj_id, pose_gl)

        ren_flags = RenderFlags.SKIP_CULL_FACES  # | RenderFlags.RGBA
        if shading == "flat":
            ren_flags = ren_flags | RenderFlags.FLAT
        color, depth = self.r.render(self.scene, flags=ren_flags)  # depth: float
        # import ipdb; ipdb.set_trace()
        if self.mode == "rgb":
            return {"rgb": color}
        elif self.mode == "depth":
            return {"depth": depth}
        elif self.mode == "rgb+depth":
            return {"rgb": color, "depth": depth}


if __name__ == "__main__":
    W = 640
    H = 480
    fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    ren = Renderer(W, H)

    R = np.eye(3).flatten().tolist()  # Identity.
    t = [0.0, 0.0, 300.0]
