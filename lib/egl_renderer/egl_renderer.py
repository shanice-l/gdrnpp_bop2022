import ctypes
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from pprint import pprint
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import OpenGL.GL as GL
import torch
from PIL import Image
import pyassimp
from pyassimp import load, release
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, qinverse, qmult

cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, cur_dir)
from . import CppEGLRenderer
from .glutils.meshutil import (
    homotrans,
    lookat,
    mat2rotmat,
    mat2xyz,
    perspective,
    quat2rotmat,
    safemat2quat,
    xyz2mat,
    loadTexture,
    im2Texture,
    shader_from_path,
    load_mesh_pyassimp,
    load_mesh_sixd,
    get_vertices_extent,
)

from .glutils.egl_offscreen_context import OffscreenContext
from lib.utils import logger


class EGLRenderer(object):
    def __init__(
        self,
        model_paths,
        K=None,
        texture_paths=None,
        model_colors=None,
        width=640,
        height=480,
        gpu_id=None,
        render_marker=False,
        robot="panda_arm",
        vertex_scale=1.0,
        znear=0.25,
        zfar=6.0,
        model_loadfn=None,
        use_cache=False,
        cad_model_colors=None,
    ):
        if model_loadfn == "pyassimp":
            self.model_load_fn = load_mesh_pyassimp
        elif model_loadfn == "pysixd":
            self.model_load_fn = load_mesh_sixd
        else:
            self.model_load_fn = load_mesh_sixd  # default using pysixd .ply loader
        self.use_cache = use_cache

        if gpu_id is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(gpu_id):
                cuda_device_idx = torch.cuda.current_device()
        self._context = OffscreenContext(gpu_id=cuda_device_idx)
        self.render_marker = render_marker
        self.VAOs = []
        self.VBOs = []
        self.materials = []
        self.textures = []
        self.is_textured = []
        self.is_materialed = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height

        self.znear = znear
        self.zfar = zfar
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        self.instances = []
        self.robot = robot
        if len(self.robot) > 3:
            self._offset_map = self.load_offset()

        self.r = CppEGLRenderer.CppEGLRenderer(width, height, cuda_device_idx)
        self.r.init()
        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders

        self.shaders = shaders
        self.colors = [
            [0.9, 0, 0],
            [0.6, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0],
        ]
        self.is_cad_list = []

        shader_types = {
            "shader_bbox": ("shader_bbox.vs", "shader_bbox.frag"),
            "shader_textureless_texture": (
                "shader_textureless_texture.vs",
                "shader_textureless_texture.frag",
            ),
            "shader_material": ("shader_material.vs", "shader_material.frag"),
            # "shader_bg": ("background.vs", "background.frag"),
        }
        self.shaders_dict = {}
        for _s_type in shader_types:
            self.shaders_dict[_s_type] = {
                "vertex": self.shaders.compileShader(
                    shader_from_path(shader_types[_s_type][0]),
                    GL.GL_VERTEX_SHADER,
                ),
                "fragment": self.shaders.compileShader(
                    shader_from_path(shader_types[_s_type][1]),
                    GL.GL_FRAGMENT_SHADER,
                ),
            }

        self.shader_programs = {}
        for _s_type in shader_types:
            self.shader_programs[_s_type] = self.shaders.compileProgram(
                self.shaders_dict[_s_type]["vertex"],
                self.shaders_dict[_s_type]["fragment"],
            )
        # self.texUnitUniform = GL.glGetUniformLocation(self.shader_programs['shader'], "uTexture")
        self.texUnitUniform = GL.glGetUniformLocation(self.shader_programs["shader_textureless_texture"], "uTexture")

        self.lightpos = [0, 0, 0]
        self.lightcolor = [1, 1, 1]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)
        # print("fbo {}, color_tex {}, color_tex_2 {}, color_tex_3 {}, color_tex_4 {}, color_tex_5 {}, depth_tex {}".format(
        #     int(self.fbo), int(self.color_tex), int(self.color_tex_2), int(self.color_tex_3),
        #     int(self.color_tex_4), int(self.color_tex_5), int(self.depth_tex)))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH24_STENCIL8,
            self.width,
            self.height,
            0,
            GL.GL_DEPTH_STENCIL,
            GL.GL_UNSIGNED_INT_24_8,
            None,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.color_tex,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_TEXTURE_2D,
            self.color_tex_2,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_TEXTURE_2D,
            self.color_tex_3,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT3,
            GL.GL_TEXTURE_2D,
            self.color_tex_4,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT4,
            GL.GL_TEXTURE_2D,
            self.color_tex_5,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_STENCIL_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.depth_tex,
            0,
        )
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(
            5,
            [
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_COLOR_ATTACHMENT1,
                GL.GL_COLOR_ATTACHMENT2,
                GL.GL_COLOR_ATTACHMENT3,
                GL.GL_COLOR_ATTACHMENT4,
            ],
        )

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.grid = self.generate_grid()

        # self.bg_VAO, self.bg_indices = self.set_bg_buffers()

        self.is_rotating = False  # added mouse interaction
        if model_colors is None:  # init render stuff
            class_colors_all = [((x + 1) * 10, (x + 1) * 10, (x + 1) * 10) for x in range(len(model_paths))]
            model_colors = [np.array(class_colors_all[i]) / 255.0 for i in range(len(model_paths))]
        if texture_paths is None:
            texture_paths = ["" for i in range(len(model_paths))]
        if cad_model_colors is not None:
            assert len(cad_model_colors) == len(model_paths)
        self.load_objects(
            model_paths,
            texture_paths,
            model_colors,
            vertex_scale=vertex_scale,
            cad_model_colors=cad_model_colors,
        )
        self.set_camera_default()
        if K is not None:
            self.set_projection_matrix(K, width, height, znear, zfar)

    def generate_grid(self):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)
        vertexData = []
        for i in np.arange(-1, 1, 0.05):  # 160
            vertexData.append([i, 0, -1, 0, 0, 0, 0, 0])
            vertexData.append([i, 0, 1, 0, 0, 0, 0, 0])
            vertexData.append([1, 0, i, 0, 0, 0, 0, 0])
            vertexData.append([-1, 0, i, 0, 0, 0, 0, 0])
        vertexData = np.array(vertexData).astype(np.float32) * 3
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertexData.nbytes,
            vertexData,
            GL.GL_STATIC_DRAW,
        )

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shader_programs["shader_simple"], "aPosition")
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 8 * 4, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return VAO

    # def set_bg_buffers(self):
    #     # TODO: make it work
    #     # Set up background render quad in NDC

    #     # fmt: off
    #     # quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    #     # tex = [[0, 1], [1, 1], [1, 0], [0, 0]]
    #     quad = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
    #     tex  = [[ 0,  0], [ 0, 1], [1, 1], [1,  0]]
    #     # fmt: on
    #     vertices = np.array(quad, dtype=np.float32)
    #     texcoord = np.array(tex, dtype=np.float32)
    #     vertexData = np.concatenate([vertices, texcoord], axis=-1).astype(np.float32)
    #     # indices = np.array([0, 1, 2, 0, 2, 3], np.int32)
    #     indices = np.array([0, 1, 3, 0, 2, 3], np.int32)

    #     VAO = GL.glGenVertexArrays(1)
    #     GL.glBindVertexArray(VAO)
    #     # Need VBO for triangle vertices and texture UV coordinates
    #     VBO = GL.glGenBuffers(1)
    #     GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    #     GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

    #     # enable array and set up data
    #     _shader_type = "shader_bg"
    #     positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
    #     coordsAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aTexcoord")

    #     GL.glEnableVertexAttribArray(0)
    #     GL.glEnableVertexAttribArray(1)
    #     # index, size, type, normalized, stride=vertexData.shape[1]*4, pointer
    #     GL.glVertexAttribPointer(positionAttrib, 2, GL.GL_FLOAT, GL.GL_FALSE, 4*4, None)  # 0
    #     GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 4*4, ctypes.c_void_p(2*4)) # 2*4=8

    #     GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    #     GL.glBindVertexArray(0)

    #     return VAO, indices

    def extent_to_bbox3d(self, xsize, ysize, zsize, is_gt=False):
        # yapf: disable
        bb = np.asarray([[-xsize / 2,  ysize / 2,  zsize / 2],
                         [ xsize / 2,  ysize / 2,  zsize / 2],
                         [-xsize / 2, -ysize / 2,  zsize / 2],
                         [ xsize / 2, -ysize / 2,  zsize / 2],
                         [-xsize / 2,  ysize / 2, -zsize / 2],
                         [ xsize / 2,  ysize / 2, -zsize / 2],
                         [-xsize / 2, -ysize / 2, -zsize / 2],
                         [ xsize / 2, -ysize / 2, -zsize / 2]])
        # Set up rendering data
        if is_gt:
            colors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        else:
            colors = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
        # yapf: enable
        """
            0 -------- 1
           /|         /|
          2 -------- 3 .
          | |        | |
          . 4 -------- 5
          |/         |/
          6 -------- 7
        """
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
        indices = np.array(indices, dtype=np.int32)

        vertices = np.array(bb, dtype=np.float32)
        normals = np.zeros_like(vertices)
        colors = np.array(colors, dtype=np.float32)
        vertexData = np.concatenate([vertices, normals, colors], axis=-1).astype(np.float32)

        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertexData.nbytes,
            vertexData,
            GL.GL_STATIC_DRAW,
        )

        # enable array and set up data
        _shader_type = "shader_bbox"
        positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
        # normalAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aNormal")
        colorAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aColor")

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(2)
        # index, size, type, normalized, stride=vertexData.shape[1]*4, pointer
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 9 * 4, None)  # 0
        GL.glVertexAttribPointer(
            colorAttrib,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            9 * 4,
            ctypes.c_void_p(6 * 4),
        )  # 6*4=24

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return VAO, indices

    def load_object(self, obj_path, texture_path="", vertex_scale=1.0):
        assert osp.exists(obj_path), obj_path
        is_texture = False
        is_materialed = False
        is_cad = False
        if texture_path != "":
            is_texture = True
            logger.info("texture path: {}".format(texture_path))
            texture = loadTexture(texture_path)
            self.textures.append(texture)
        self.is_textured.append(is_texture)
        if obj_path.endswith("DAE"):
            is_materialed = True
            vertices, faces, materials = self.load_robot_mesh(obj_path)  # return list of vertices, faces, materials
            self.materials.append(materials)
            self.textures.append("")  # dummy
        self.is_materialed.append(is_materialed)
        if is_materialed:
            for idx in range(len(vertices)):
                vertexData = vertices[idx].astype(np.float32)
                face = faces[idx]
                VAO = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(VAO)

                # Need VBO for triangle vertices and texture UV coordinates
                VBO = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER,
                    vertexData.nbytes,
                    vertexData,
                    GL.GL_STATIC_DRAW,
                )
                positionAttrib = GL.glGetAttribLocation(self.shader_programs["shader_material"], "aPosition")
                normalAttrib = GL.glGetAttribLocation(self.shader_programs["shader_material"], "aNormal")

                GL.glEnableVertexAttribArray(0)
                GL.glEnableVertexAttribArray(1)

                GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
                GL.glVertexAttribPointer(
                    normalAttrib,
                    3,
                    GL.GL_FLOAT,
                    GL.GL_FALSE,
                    24,
                    ctypes.c_void_p(12),
                )

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindVertexArray(0)

                self.VAOs.append(VAO)
                self.VBOs.append(VBO)
                self.faces.append(face)
            self.objects.append(obj_path)
            self.poses_rot.append(np.eye(4))
            self.poses_trans.append(np.eye(4))

        else:
            _shader_type = "shader_textureless_texture"
            logger.info(obj_path)
            mesh = self.model_load_fn(
                obj_path,
                vertex_scale=vertex_scale,
                is_textured=is_texture,
                use_cache=self.use_cache,
            )
            is_cad = mesh["is_cad"]
            logger.info("is_textured: {} | is_cad: {} | is_materialed: {}".format(is_texture, is_cad, is_materialed))
            # pprint(mesh)
            # check materials
            logger.info("{}".format(list(mesh.keys())))
            mat_diffuse, mat_specular, mat_ambient, mat_shininess = [
                mesh[_k]
                for _k in [
                    "uMatDiffuse",
                    "uMatSpecular",
                    "uMatAmbient",
                    "uMatShininess",
                ]
            ]
            self.materials.append([np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess])])

            faces = mesh["faces"]
            logger.info("colors: {}".format(mesh["colors"].max()))
            vertices = np.concatenate(
                [
                    mesh["vertices"],
                    mesh["normals"],
                    mesh["colors"],
                    mesh["texturecoords"],
                ],
                axis=-1,
            )  # ply models

            vertexData = vertices.astype(np.float32)
            # print(vertexData.shape, faces.shape) #..x8, ..x3
            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                vertexData.nbytes,
                vertexData,
                GL.GL_STATIC_DRAW,
            )

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
            normalAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aNormal")
            colorAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aColor")
            coordsAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aTexcoord")

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)
            GL.glEnableVertexAttribArray(3)  # added

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 11 * 4, None)  # 0
            GL.glVertexAttribPointer(
                normalAttrib,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                11 * 4,
                ctypes.c_void_p(3 * 4),
            )  # 3*4=12
            GL.glVertexAttribPointer(
                colorAttrib,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                11 * 4,
                ctypes.c_void_p(6 * 4),
            )  # 6*4=24
            GL.glVertexAttribPointer(
                coordsAttrib,
                2,
                GL.GL_FLOAT,
                GL.GL_TRUE,
                11 * 4,
                ctypes.c_void_p(9 * 4),
            )  # 9*4 = 36

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

            self.VAOs.append(VAO)
            self.VBOs.append(VBO)
            self.faces.append(faces)
            self.objects.append(obj_path)
            self.poses_rot.append(np.eye(4))
            self.poses_trans.append(np.eye(4))
        self.is_cad_list.append(is_cad)

    def load_offset(self):
        cur_path = osp.abspath(osp.dirname(__file__))
        offset_file = osp.join(cur_path, "robotPose", self.robot + "_models", "center_offset.txt")
        model_file = osp.join(cur_path, "robotPose", self.robot + "_models", "models.txt")
        with open(model_file, "r+") as file:
            content = file.readlines()
            model_paths = [path.strip().split("/")[-1] for path in content]
        offset = np.loadtxt(offset_file).astype(np.float32)
        offset_map = {}
        for i in range(offset.shape[0]):
            offset_map[model_paths[i]] = offset[i, :]
        # extent max - min in mesh, center = (max + min)/2
        return offset_map

    def load_robot_mesh(self, collada_path):
        # load collada file and return vertices, faces, materials
        mesh_file = collada_path.strip().split("/")[-1]  # for offset the robot mesh
        scene = load(collada_path)  # load collada
        offset = self._offset_map[mesh_file]
        return self.recursive_load(scene.rootnode, [], [], [], offset)

    def recursive_load(self, node, vertices, faces, materials, offset):
        if node.meshes:
            transform = node.transformation
            for idx, mesh in enumerate(node.meshes):
                # pprint(vars(mesh))
                if mesh.faces.shape[-1] != 3:  # ignore boundLineSet
                    continue
                mat = mesh.material
                pprint(vars(mat))
                mat_diffuse = np.array(mat.properties["diffuse"])[:3]

                if "specular" in mat.properties:
                    mat_specular = np.array(mat.properties["specular"])[:3]
                else:
                    mat_specular = [0.5, 0.5, 0.5]
                    mat_diffuse = [0.8, 0.8, 0.8]

                if "ambient" in mat.properties:
                    mat_ambient = np.array(mat.properties["ambient"])[:3]  # phong shader
                else:
                    mat_ambient = [0, 0, 0]

                if "shininess" in mat.properties:
                    mat_shininess = max(mat.properties["shininess"], 1)  # avoid the 0 shininess
                else:
                    mat_shininess = 1

                mesh_vertex = homotrans(transform, mesh.vertices) - offset  # subtract the offset
                mesh_normals = transform[:3, :3].dot(mesh.normals.transpose()).transpose()  # normal stays the same
                vertices.append(np.concatenate([mesh_vertex, mesh_normals], axis=-1))
                faces.append(mesh.faces)
                materials.append(np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess]))
                # concat speed, render speed, bind & unbind, memory
        for child in node.children:
            self.recursive_load(child, vertices, faces, materials, offset)
        return vertices, faces, materials

    def load_objects(
        self,
        obj_paths,
        texture_paths,
        colors=[[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]],
        vertex_scale=1.0,
        cad_model_colors=None,
    ):
        self.colors = colors
        self.cad_model_colors = cad_model_colors
        self.is_cad_list = []
        for i in tqdm(range(len(obj_paths))):
            self.load_object(obj_paths[i], texture_paths[i], vertex_scale=vertex_scale)
            if i == 0:
                self.instances.append(0)
            else:
                self.instances.append(self.instances[-1] + len(self.materials[i - 1]))  # offset

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_camera_default(self):
        self.V = np.eye(4)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_projection_matrix(self, K, width, height, znear, zfar):
        """set projection matrix according to real camera intrinsics."""
        fx = K[0, 0]
        fy = K[1, 1]
        u0 = K[0, 2]
        v0 = K[1, 2]
        L = -u0 * znear / fx
        R = +(width - u0) * znear / fx
        T = -v0 * znear / fy
        B = +(height - v0) * znear / fy

        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * znear / (R - L)
        P[1, 1] = 2 * znear / (T - B)
        P[2, 0] = (R + L) / (L - R)
        P[2, 1] = (T + B) / (B - T)
        P[2, 2] = (zfar + znear) / (zfar - znear)
        P[2, 3] = 1.0
        P[3, 2] = (2 * zfar * znear) / (znear - zfar)
        self.P = P

    def set_light_color(self, color):
        self.lightcolor = color

    def draw_bg(self, im):
        texture_id = im2Texture(im, flip_v=True)
        # draw texture
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glBegin(GL.GL_QUADS)
        # fmt: off
        GL.glTexCoord2f(0, 0)
        GL.glVertex2f(-1, -1)
        GL.glTexCoord2f(0, 1)
        GL.glVertex2f(-1, 1)
        GL.glTexCoord2f(1, 1)
        GL.glVertex2f(1, 1)
        GL.glTexCoord2f(1, 0)
        GL.glVertex2f(1, -1)
        # fmt: on
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        # GL.glBindVertexArray(0)
        # GL.glUseProgram(0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # clear depth
        GL.glEnable(GL.GL_DEPTH_TEST)

        # _shader_type = 'shader_bg'
        # shader = self.shader_programs[_shader_type]

        # GL.glEnable(GL.GL_TEXTURE_2D)
        # GL.glBegin(GL.GL_QUADS)
        # GL.glUseProgram(shader)
        # # whether fixed-point data values should be normalized ( GL_TRUE ) or converted directly as fixed-point values ( GL_FALSE )
        # try:
        #     GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
        #     GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)  # self.instances[index]
        #     # GL.glUniform1i(self.texUnitUniform, 0)
        #     GL.glUniform1i(GL.glGetUniformLocation(shader, "uTexture"), 0)
        #     GL.glBindVertexArray(self.bg_VAO) # Activate array
        #     # draw triangles
        #     GL.glDrawElements(GL.GL_TRIANGLES, len(self.bg_indices), GL.GL_UNSIGNED_INT, self.bg_indices)
        # except:
        #     logger.warn('err in draw bg')
        # finally:
        #     GL.glEnd()
        #     GL.glDisable(GL.GL_TEXTURE_2D)

        #     GL.glBindVertexArray(0)
        #     GL.glUseProgram(0)
        #     GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth

    def render(
        self,
        obj_ids,
        poses,
        K=None,
        to_bgr=True,
        to_255=True,
        rot_type="mat",
        instance_colors=None,
        light_pos=None,
        light_color=None,
        image_tensor=None,
        seg_tensor=None,
        normal_tensor=None,
        pc_obj_tensor=None,
        pc_cam_tensor=None,
        phong={"ambient": 0.4, "diffuse": 0.8, "specular": 0.3},
        extents=None,
        gt_extents=None,
        background=None,
    ):
        # get un-occluded instance masks by rendering one by one
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]
        if isinstance(poses, np.ndarray):
            poses = [poses]
        if K is not None:
            self.set_projection_matrix(
                K,
                width=self.width,
                height=self.height,
                znear=self.znear,
                zfar=self.zfar,
            )
        if light_pos is not None:
            self.set_light_pos(light_pos)
        if light_color is not None:
            self.set_light_color(light_color)
        if instance_colors is not None:
            assert len(instance_colors) == len(obj_ids)
        else:
            instance_colors = self.colors
        if extents is not None:
            assert len(extents) == len(obj_ids)
        if gt_extents is not None:
            assert len(gt_extents) == len(obj_ids)
        self.set_poses(poses, rot_type=rot_type)

        # self.lightpos = np.random.uniform(-1, 1, 3)
        # frame = 0
        GL.glClearColor(0, 0, 0, 1)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
        if background is not None:
            self.draw_bg(background)

        if self.render_marker:
            # render some grid and directions
            GL.glUseProgram(self.shader_programs["shader_simple"])
            GL.glBindVertexArray(self.grid)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.shader_programs["shader_simple"], "V"),
                1,
                GL.GL_TRUE,
                self.V,
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.shader_programs["shader_simple"], "uProj"),
                1,
                GL.GL_FALSE,
                self.P,
            )
            GL.glDrawElements(
                GL.GL_LINES,
                160,
                GL.GL_UNSIGNED_INT,
                np.arange(160, dtype=np.int),
            )
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            # end rendering markers

        # render 3d bboxes ================================================================================
        if extents is not None:
            thickness = 1.5
            GL.glLineWidth(thickness)
            _shader_name = "shader_bbox"
            shader = self.shader_programs[_shader_name]
            for i, extent in enumerate(extents):
                GL.glUseProgram(shader)
                _vertexData, _indices = self.extent_to_bbox3d(extent[0], extent[1], extent[2], is_gt=False)
                GL.glBindVertexArray(_vertexData)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glDrawElements(GL.GL_LINES, len(_indices), GL.GL_UNSIGNED_INT, _indices)
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
            GL.glLineWidth(1.0)

            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth of 3d bboxes

        if gt_extents is not None:
            thickness = 1.5
            GL.glLineWidth(thickness)
            _shader_name = "shader_bbox"
            shader = self.shader_programs[_shader_name]
            for i, gt_extent in enumerate(gt_extents):
                GL.glUseProgram(shader)
                _vertexData, _indices = self.extent_to_bbox3d(gt_extent[0], gt_extent[1], gt_extent[2], is_gt=True)
                GL.glBindVertexArray(_vertexData)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glDrawElements(GL.GL_LINES, len(_indices), GL.GL_UNSIGNED_INT, _indices)
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth of 3d bboxes
        # size = 0
        for i in range(len(obj_ids)):  ##################################
            index = obj_ids[i]
            is_texture = self.is_textured[index]  # index
            is_materialed = self.is_materialed[index]
            # active shader program
            if is_materialed:  # for mesh in the robot mesh
                num = len(self.materials[index])
                for idx in range(num):
                    # the materials stored in vertex attribute instead of uniforms to avoid bind & unbind
                    shader = self.shader_programs["shader_material"]
                    GL.glUseProgram(shader)
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "V"),
                        1,
                        GL.GL_TRUE,
                        self.V,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "uProj"),
                        1,
                        GL.GL_FALSE,
                        self.P,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "pose_trans"),
                        1,
                        GL.GL_FALSE,
                        self.poses_trans[i],
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "pose_rot"),
                        1,
                        GL.GL_TRUE,
                        self.poses_rot[i],
                    )
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "uLightPosition"), *self.lightpos)
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "instance_color"), *instance_colors[index])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "uLightColor"), *self.lightcolor)
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatDiffuse"), *self.materials[index][idx][:3])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatSpecular"), *self.materials[index][idx][3:6])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatAmbient"), *self.materials[index][idx][6:9])
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uMatShininess"),
                        self.materials[index][idx][-1],
                    )

                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightAmbientWeight"),
                        phong["ambient"],
                    )
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightDiffuseWeight"),
                        phong["diffuse"],
                    )
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightSpecularWeight"),
                        phong["specular"],
                    )

                    try:
                        GL.glBindVertexArray(self.VAOs[self.instances[index] + idx])
                        GL.glDrawElements(
                            GL.GL_TRIANGLES,
                            self.faces[self.instances[index] + idx].size,
                            GL.GL_UNSIGNED_INT,
                            self.faces[self.instances[index] + idx],
                        )
                    finally:
                        GL.glBindVertexArray(0)
                        GL.glUseProgram(0)
            else:  #####################################################################################
                _shader_type = "shader_textureless_texture"
                shader = self.shader_programs[_shader_type]

                GL.glUseProgram(shader)
                # whether fixed-point data values should be normalized ( GL_TRUE ) or converted directly as fixed-point values ( GL_FALSE )
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glUniform3f(GL.glGetUniformLocation(shader, "uLightPosition"), *self.lightpos)
                GL.glUniform3f(GL.glGetUniformLocation(shader, "instance_color"), *instance_colors[index])
                GL.glUniform3f(GL.glGetUniformLocation(shader, "uLightColor"), *self.lightcolor)
                GL.glUniform1i(
                    GL.glGetUniformLocation(shader, "uUseTexture"),
                    int(is_texture),
                )

                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightAmbientWeight"),
                    phong["ambient"],
                )
                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightDiffuseWeight"),
                    phong["diffuse"],
                )
                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightSpecularWeight"),
                    phong["specular"],
                )

                try:
                    if is_texture:
                        GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
                        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[index])  # self.instances[index]
                        # GL.glUniform1i(self.texUnitUniform, 0)
                        GL.glUniform1i(GL.glGetUniformLocation(shader, "uTexture"), 0)
                        GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatDiffuse"), *self.materials[index][0][:3])
                        GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatSpecular"), *self.materials[index][0][3:6])
                        GL.glUniform3f(GL.glGetUniformLocation(shader, "uMatAmbient"), *self.materials[index][0][6:9])
                        GL.glUniform1f(
                            GL.glGetUniformLocation(shader, "uMatShininess"),
                            self.materials[index][0][-1],
                        )
                    GL.glBindVertexArray(self.VAOs[self.instances[index]])  # Activate array
                    # draw triangles
                    GL.glDrawElements(
                        GL.GL_TRIANGLES,
                        self.faces[self.instances[index]].size,
                        GL.GL_UNSIGNED_INT,
                        self.faces[self.instances[index]],
                    )
                except:
                    logger.warn("err in render")
                finally:
                    GL.glBindVertexArray(0)
                    GL.glUseProgram(0)

        # draw done

        GL.glDisable(GL.GL_DEPTH_TEST)
        # mapping
        # print('color_tex: {} seg_tex: {}'.format(int(self.color_tex), int(self.color_tex_3)))  # 1, 3
        if image_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex),
                int(self.width),
                int(self.height),
                image_tensor.data_ptr(),
            )
            image_tensor.data = torch.flip(image_tensor, (0,))
            if to_bgr:
                image_tensor.data[:, :, :3] = image_tensor.data[:, :, [2, 1, 0]]
            if to_255:
                image_tensor.data = image_tensor.data * 255
        if seg_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_3),
                int(self.width),
                int(self.height),
                seg_tensor.data_ptr(),
            )
            seg_tensor.data = torch.flip(seg_tensor, (0,))
        # print(np.unique(seg_tensor.cpu().numpy()))
        if normal_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_2),
                int(self.width),
                int(self.height),
                normal_tensor.data_ptr(),
            )
        if pc_obj_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_4),
                int(self.width),
                int(self.height),
                pc_obj_tensor.data_ptr(),
            )
            pc_obj_tensor.data = torch.flip(pc_obj_tensor, (0,))
        if pc_cam_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_5),
                int(self.width),
                int(self.height),
                pc_cam_tensor.data_ptr(),
            )
            pc_cam_tensor.data = torch.flip(pc_cam_tensor, (0,))
            # depth is pc_cam_tensor[:,:,2]
        """
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #frame = np.frombuffer(frame,dtype = np.float32).reshape(self.width, self.height, 4)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        #normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #normal = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #normal = normal[::-1, ]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        #pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        # seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)

        #pc = np.stack([pc,pc, pc, np.ones(pc.shape)], axis = -1)
        #pc = pc[::-1, ]
        #pc = (1-pc) * 10

        # points in object coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        pc2 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc2 = pc2.reshape(self.height, self.width, 4)[::-1, :]
        pc2 = pc2[:,:,:3]

        # points in camera coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
        pc3 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc3 = pc3.reshape(self.height, self.width, 4)[::-1, :]
        pc3 = pc3[:,:,:3]

        return [frame, seg, pc2, pc3]
        """

    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses, rot_type="mat"):
        assert rot_type in ["mat", "quat"], rot_type
        if rot_type == "quat":
            self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[:4])) for item in poses]
            self.poses_trans = [np.ascontiguousarray(xyz2mat(item[4:7])) for item in poses]
        elif rot_type == "mat":
            self.poses_rot = [np.ascontiguousarray(mat2rotmat(item[:3, :3])) for item in poses]
            self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3, 3])) for item in poses]
        else:
            raise ValueError("wrong rot_type: {}".format(rot_type))

    def set_allocentric_poses(self, poses):
        self.poses_rot = []
        self.poses_trans = []
        for pose in poses:
            x, y, z = pose[:3]
            quat_input = pose[3:]
            dx = np.arctan2(x, -z)
            dy = np.arctan2(y, -z)
            # print(dx, dy)
            quat = euler2quat(-dy, -dx, 0, axes="sxyz")
            quat = qmult(quat, quat_input)
            self.poses_rot.append(np.ascontiguousarray(quat2rotmat(quat)))
            self.poses_trans.append(np.ascontiguousarray(xyz2mat(pose[:3])))

    def close(self):
        # logger.info(self.glstring)
        self.clean()
        self._context.close()
        # TODO: handle errors
        self.r.release()

    def clean(self):
        GL.glDeleteTextures(
            [
                self.color_tex,
                self.color_tex_2,
                self.color_tex_3,
                self.color_tex_4,
                self.depth_tex,
            ]
        )
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None

        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def get_num_instances(self):
        return len(self.instances)

    def get_poses(self):
        # quat + trans
        mat = [self.V.dot(self.poses_trans[i].T).dot(self.poses_rot[i]).T for i in range(self.get_num_instances())]
        poses = [np.concatenate([safemat2quat(item[:3, :3].T), mat2xyz(item)]) for item in mat]
        return poses

    def get_egocentric_poses(self):
        return self.get_poses()

    def get_allocentric_poses(self):
        poses = self.get_poses()
        poses_allocentric = []
        for pose in poses:
            dx = np.arctan2(pose[4], -pose[6])
            dy = np.arctan2(pose[5], -pose[6])
            quat = euler2quat(-dy, -dx, 0, axes="sxyz")
            quat = qmult(qinverse(quat), pose[:4])
            poses_allocentric.append(np.concatenate([quat, pose[4:7]]))
            # print(quat, pose[:4], pose[4:7])
        return poses_allocentric

    def get_centers(self):
        centers = []
        for i in range(len(self.poses_trans)):
            pose_trans = self.poses_trans[i]
            proj = self.P.T.dot(self.V.dot(pose_trans.T).dot(np.array([0, 0, 0, 1])))
            proj /= proj[-1]
            centers.append(proj[:2])
        centers = np.array(centers)
        centers = (centers + 1) / 2.0
        centers[:, 1] = 1 - centers[:, 1]
        centers = centers[:, ::-1]  # in y, x order
        return centers


def test_ycb_render():
    # from robotPose.robot_pykdl import *
    MAX_NUM_OBJECTS = 3

    model_path = sys.argv[1]
    robot_name = ""  # sys.argv[2]
    print("robot name", robot_name)
    width = 640  # 800
    height = 480  # 600
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 214.08], [0, 0, 1]])
    camera_extrinsics = np.array(
        [
            [-0.211719, 0.97654, -0.0393032, 0.377451],
            [0.166697, -0.00354316, -0.986002, 0.374476],
            [-0.96301, -0.215307, -0.162036, 1.87315],
            [0, 0, 0, 1],
        ]
    )

    if robot_name == "baxter":
        models = ["S0", "S1", "E0", "E1", "W0", "W1", "W2"]
        # models = ['E1']
        obj_paths = ["robotPose/{}_models/{}.DAE".format(robot_name, item) for item in models]
        colors = [[0.1 * (idx + 1), 0, 0] for idx in range(len(models))]
        texture_paths = ["" for item in models]
    elif robot_name == "panda_arm":
        models = [
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "finger",
            "finger",
        ]
        # models = ['link4']
        obj_paths = ["robotPose/{}_models/{}.DAE".format(robot_name, item) for item in models]
        colors = [[0, 0.1 * (idx + 1), 0] for idx in range(len(models))]
        texture_paths = ["" for item in models]
    else:
        models = ["003_cracker_box", "002_master_chef_can", "011_banana"]
        colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]]

        # obj_paths = [
        #     "{}/models/{}/textured_simple.obj".format(model_path, item)
        #     for item in models
        # ]
        obj_paths = ["{}/models/{}/textured.obj".format(model_path, item) for item in models]
        texture_paths = ["{}/models/{}/texture_map.png".format(model_path, item) for item in models]
    print("obj_paths ", obj_paths)
    print("texture_paths ", texture_paths)
    renderer = EGLRenderer(
        model_paths=obj_paths,
        texture_paths=texture_paths,
        model_colors=colors,
        width=width,
        height=height,
        render_marker=True,
        robot=robot_name,
        use_cache=True,
    )
    # mat = pose2mat(pose)
    pose = np.array(
        [
            -0.025801208,
            0.08432201,
            0.004528991,
            0.9992879,
            -0.0021458883,
            0.0304758,
            0.022142926,
        ]
    )
    pose2 = np.array(
        [
            -0.56162935,
            0.05060109,
            -0.028915625,
            0.6582951,
            0.03479896,
            -0.036391996,
            -0.75107396,
        ]
    )
    pose3 = np.array(
        [
            0.22380374,
            0.019853603,
            0.12159989,
            -0.40458265,
            -0.036644224,
            -0.6464779,
            0.64578354,
        ]
    )

    theta = 0
    z = 1
    fix_pos = [np.sin(theta), z, np.cos(theta)]
    renderer.set_camera(fix_pos, [0, 0, 0], [0, 1, 0])
    fix_pos = np.zeros(3)
    poses = [pose, pose2, pose3]
    cls_indexes = [0, 1, 2]
    if robot_name == "baxter" or robot_name == "panda_arm":
        import scipy.io as sio

        robot = robot_kinematics(robot_name)
        poses = []
        if robot_name == "baxter":
            base_link = "right_arm_mount"
        else:
            base_link = "panda_link0"
        pose, joint = robot.gen_rand_pose(base_link)
        cls_indexes = range(len(models))
        pose = robot.offset_pose_center(pose, dir="off", base_link=base_link)  # print pose_hand
        # pose = np.load('%s.npy'%robot_name)
        for i in range(len(pose)):
            pose_i = pose[i]
            quat = mat2quat(pose_i[:3, :3])
            trans = pose_i[:3, 3]
            poses.append(np.hstack((quat, trans)))

        renderer.set_poses(poses, rot_type="quat")
        renderer.V = camera_extrinsics
        renderer.set_projection_matrix(K, width, height, 0.0001, 6)
        fix_pos = renderer.V[:3, 3].reshape([1, 3]).copy()
    renderer.set_light_pos([2, 2, 2])
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    import time

    num_iter = 0
    start = time.perf_counter()
    while True:
        num_iter += 1
        renderer.render(
            cls_indexes,
            poses=poses,
            rot_type="quat",
            image_tensor=image_tensor,
            seg_tensor=seg_tensor,
        )
        frame = [image_tensor.cpu().numpy() / 255.0, seg_tensor.cpu().numpy()]
        centers = renderer.get_centers()
        for center in centers:
            x = int(center[1] * width)
            y = int(center[0] * height)
            frame[0][y - 2 : y + 2, x - 2 : x + 2, :] = 1
            frame[1][y - 2 : y + 2, x - 2 : x + 2, :] = 1
        if len(sys.argv) > 2 and sys.argv[2] == "headless":
            # print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1:
                break
        else:
            if True:
                import matplotlib.pyplot as plt

                plt.imshow(np.concatenate(frame, axis=1)[:, :, [2, 1, 0]])
                plt.show()
            else:
                cv2.imshow("test", np.concatenate(frame, axis=1))
                q = cv2.waitKey(16)
                if q == ord("w"):
                    z += 0.05
                elif q == ord("s"):
                    z -= 0.05
                elif q == ord("a"):
                    theta -= 0.1
                elif q == ord("d"):
                    theta += 0.1
                elif q == ord("p"):
                    Image.fromarray((frame[0][:, :, :3] * 255).astype(np.uint8)).save("test.png")
                elif q == ord("q"):
                    break
                elif q == ord("r"):  # rotate
                    pose[3:] = qmult(axangle2quat([0, 0, 1], 5 / 180.0 * np.pi), pose[3:])
                    pose2[3:] = qmult(axangle2quat([0, 0, 1], 5 / 180.0 * np.pi), pose2[3:])
                    pose3[3:] = qmult(axangle2quat([0, 0, 1], 5 / 180.0 * np.pi), pose3[3:])
                    poses = [pose, pose2, pose3]

        cam_pos = fix_pos + np.array([np.sin(theta), z, np.cos(theta)])
        if robot_name == "baxter" or robot_name == "panda_arm":
            renderer.V[:3, 3] = np.array(cam_pos)
        else:
            cam_pos = fix_pos + np.array([np.sin(theta), z, np.cos(theta)])
            renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
        # renderer.set_light_pos(cam_pos)
    dt = time.perf_counter() - start
    print("iters: {}, {}s, {} fps".format(num_iter, dt, num_iter / dt))
    # iters: 1000, 6.252699375152588s, 159.93092582922978 fps

    renderer.close()


if __name__ == "__main__":
    import random
    import glob
    import time
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from tqdm import tqdm
    from transforms3d.axangles import axangle2mat
    import matplotlib.pyplot as plt
    from lib.pysixd import inout

    random.seed(0)
    # test_ycb_render()
    # exit(0)

    width = 640
    height = 480
    znear = 0.25
    zfar = 6.0
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    idx2class = {
        1: "ape",
        2: "benchvise",
        3: "bowl",
        4: "camera",
        5: "can",
        6: "cat",
        7: "cup",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    classes = idx2class.values()
    classes = sorted(classes)

    model_root = "datasets/BOP_DATASETS/lm/models/"
    model_paths = [osp.join(model_root, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in idx2class]
    models = [inout.load_ply(model_path, vertex_scale=0.001) for model_path in model_paths]
    extents = [get_vertices_extent(model["pts"]) for model in models]

    renderer = EGLRenderer(
        model_paths,
        K,
        width=width,
        height=height,
        render_marker=False,
        vertex_scale=0.001,
        use_cache=True,
    )
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    instance_mask_tensors = [torch.empty((height, width, 4), **tensor_kwargs).detach() for i in range(10)]
    pc_obj_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    pc_cam_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    # render target pose
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([0, 0, 0.7], dtype=np.float32)
    pose = np.hstack([R, t.reshape((3, 1))])
    pose1 = np.hstack([R, 0.1 + t.reshape((3, 1))])
    pose2 = np.hstack([R, t.reshape((3, 1)) - 0.1])
    pose3 = np.hstack([R, t.reshape((3, 1)) - 0.05])
    pose4 = np.hstack([R, t.reshape((3, 1)) + 0.05])
    # renderer.set_poses([pose])

    bg_images = glob.glob("datasets/coco/train2017/*.jpg")
    num_bg_imgs = len(bg_images)

    # rendering
    runs = 0
    t_render = 0
    # without copy to cpu, it is faster than meshrenderer: 0.0008892447471618652s 1124.549797107741fps
    # 5 objects, render instance masks: 0.0023294403235117594s 429.2876661860326fps
    # 5 objects, without instance masks: 0.0010711719353993733s 933.5569453909957fps
    # when copy to cpu: 0.002706778923670451s 369.4428057109217fps
    for j in tqdm(range(1000)):
        for obj_id, cls_name in enumerate(classes):
            t0 = time.perf_counter()
            light_pos = np.random.uniform(-0.5, 0.5, 3)
            intensity = np.random.uniform(0.8, 2)
            light_color = intensity * np.random.uniform(0.9, 1.1, 3)
            poses = [pose, pose1, pose2, pose3, pose4]
            obj_ids = [obj_id, obj_id, obj_id, obj_id, obj_id]
            gt_extents = [extents[_obj_id] for _obj_id in obj_ids]
            # light_color = None
            # light_pos = (0, 0, 0)
            """
            bg_path = bg_images[random.randint(0, num_bg_imgs - 1)]
            bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            bg_img = cv2.resize(bg_img, (width, height))
            renderer.render(obj_ids, poses=poses,
                    image_tensor=image_tensor,
                    seg_tensor=None, rot_type='mat', pc_cam_tensor=None,
                    light_pos=light_pos, light_color=light_color,
                    extents=gt_extents,
                    background=bg_img[:,:, [2, 1, 0]])
            renderer.render(obj_ids, poses=poses,
                    image_tensor=None,
                    seg_tensor=seg_tensor, rot_type='mat', pc_cam_tensor=pc_cam_tensor,
                    light_pos=light_pos, light_color=light_color,
                    extents=None,
                    background=None)
            """
            renderer.render(
                obj_ids,
                poses=poses,
                image_tensor=image_tensor,
                seg_tensor=seg_tensor,
                rot_type="mat",
                pc_cam_tensor=pc_cam_tensor,
                light_pos=light_pos,
                light_color=light_color,
                extents=None,
                background=None,
            )
            for i in range(len(poses)):
                renderer.render(
                    obj_ids[i],
                    poses=poses[i],
                    image_tensor=None,
                    seg_tensor=instance_mask_tensors[i],
                    rot_type="mat",
                    pc_cam_tensor=None,
                    light_pos=None,
                    light_color=None,
                )
            im = image_tensor[:, :, :3]
            # im = (im.cpu().numpy() + 0.5).astype(np.uint8)
            t_render += time.perf_counter() - t0
            runs += 1
            # torch.save(im, 'im_{}.pth'.format(cls_name))
            if False:
                im = (im.cpu().numpy() + 0.5).astype(np.uint8)  # bgr
                seg = (seg_tensor[:, :, 0].cpu().numpy() * 255 + 0.5).astype(np.uint8)
                masks = [
                    (ins_mask[:, :, 0].cpu().numpy() * 255 + 0.5).astype(np.uint8)
                    for ins_mask in instance_mask_tensors[: len(poses)]
                ]
                print("seg unique: ", np.unique(seg))
                # fig = plt.figure()
                # plt.imshow(bg_img[:,:, [2, 1, 0]])
                # plt.show()

                fig = plt.figure(frameon=False, dpi=200)
                plt.subplot(2, 2, 1)
                plt.imshow(im[:, :, [2, 1, 0]])  # rgb
                plt.axis("off")
                plt.title("{} color".format(cls_name))

                plt.subplot(2, 2, 2)
                plt.imshow(seg)
                plt.axis("off")
                plt.title("{} seg".format(cls_name))

                depth = pc_cam_tensor[:, :, 2].cpu().numpy()
                depth_save = (depth * 1000).astype(np.uint16)
                cv2.imwrite("depth_{}.png".format(cls_name), depth_save)
                plt.subplot(2, 2, 3)
                plt.imshow(depth)
                plt.axis("off")
                plt.title("{} depth".format(cls_name))

                img_vis = vis_image_mask_bbox_cv2(im, masks, bboxes=None, labels=None)
                plt.subplot(2, 2, 4)
                plt.imshow(img_vis[:, :, [2, 1, 0]])
                plt.axis("off")
                plt.title("{} instance masks".format(cls_name))

                plt.show()

    print("{}s {}fps".format(t_render / runs, runs / t_render))
    renderer.close()
