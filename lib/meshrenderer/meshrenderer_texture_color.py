# -*- coding: utf-8 -*-
# flake8: noqa
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import numpy as np
from OpenGL import GL
from OpenGL.GL import *  # noqa:F403

_CUR_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(_CUR_DIR, "../.."))
import lib.meshrenderer.gl_utils as gu
from lib.pysixd import misc
from lib.utils import logger

_RENDER_COLOR_ONLY = 0
_RENDER_PHONG_VERTEX = 1
_RENDER_TEXTURED = 2


__all__ = ["Renderer"]


class Renderer(object):
    """reconstruction models (with vertex color or texture) change shader
    phong/color during runtime."""

    MAX_FBO_WIDTH = 2000
    MAX_FBO_HEIGHT = 2000

    def __init__(
        self,
        models_cad_files,
        K=None,
        texture_paths=None,
        samples=1,
        vertex_tmp_store_folder=".cache",
        vertex_scale=1.0,
        height=480,
        width=640,
        near=0.25,
        far=6.0,
        offset_uv=0.0,
        render_uv=False,
        render_normalized_coords=False,
        render_nocs=False,
        use_cache=True,
        model_infos=None,
        cad_model_colors=None,
        recalculate_normals=False,
        model_load_fn="pysixd",
        # gpu_id=0,
    ):
        """vertex_scale: factor for point coordinates
        offset_uv: +0.5 to get more accurate pnp results,
                    0 to be consistent with bop_renderer
        """
        self._samples = samples
        self._context = gu.OffscreenContext()

        self.height = height
        self.width = width
        self.near = near
        self.far = far
        self.offset_uv = offset_uv

        self.K = K
        self.render_uv = render_uv
        self.render_normalized_coords = render_normalized_coords
        self.render_nocs = render_nocs

        # FBO
        W, H = Renderer.MAX_FBO_WIDTH, Renderer.MAX_FBO_HEIGHT
        self._fbo = gu.Framebuffer(
            {
                GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                GL_DEPTH_ATTACHMENT: gu.Renderbuffer(GL_DEPTH_COMPONENT32F, W, H),
            }
        )

        self._fbo_depth = gu.Framebuffer(
            {
                GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                GL_DEPTH_ATTACHMENT: gu.Renderbuffer(GL_DEPTH_COMPONENT32F, W, H),
            }
        )
        glNamedFramebufferDrawBuffers(
            self._fbo.id,
            2,
            np.array((GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1), dtype=np.uint32),
        )
        glNamedFramebufferDrawBuffers(
            self._fbo_depth.id,
            2,
            np.array((GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1), dtype=np.uint32),
        )

        if self._samples > 1:
            self._render_fbo = gu.Framebuffer(
                {
                    GL_COLOR_ATTACHMENT0: gu.TextureMultisample(self._samples, GL_RGB8, W, H, True),
                    GL_COLOR_ATTACHMENT1: gu.TextureMultisample(self._samples, GL_R32F, W, H, True),
                    GL_DEPTH_STENCIL_ATTACHMENT: gu.RenderbufferMultisample(self._samples, GL_DEPTH32F_STENCIL8, W, H),
                }
            )
            glNamedFramebufferDrawBuffers(
                self._render_fbo.id,
                2,
                np.array(
                    (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1),
                    dtype=np.uint32,
                ),
            )

        self._fbo.bind()

        # VAO
        # maybe assign uv or normalized coords to color
        if model_load_fn == "pysixd":
            attributes = gu.geo.load_meshes_sixd(
                models_cad_files,
                vertex_tmp_store_folder,
                recalculate_normals=False,
                render_uv=render_uv,
                render_normalized_coords=render_normalized_coords,
                render_nocs=render_nocs,
                use_cache=use_cache,
                model_infos=model_infos,
                cad_model_colors=cad_model_colors,
                texture_paths=texture_paths,
            )
        elif model_load_fn == "pyassimp":
            attributes = gu.geo.load_meshes(
                models_cad_files,
                vertex_tmp_store_folder,
                recalculate_normals=recalculate_normals,
                use_cache=use_cache,
            )

        vertices = []
        indices = []
        self.textures = []
        self.is_cad_list = []
        self.is_textured_list = []
        for _i, attributes_dict in enumerate(attributes):
            vertex, normal, color, faces, texture_uv = [
                attributes_dict[_k]
                for _k in (
                    "vertices",
                    "normals",
                    "colors",
                    "faces",
                    "texture_uv",
                )
            ]
            indices.append(faces.flatten())
            self.is_cad_list.append(attributes_dict["is_cad"])
            self.is_textured_list.append(attributes_dict["is_textured"])

            if attributes_dict["is_textured"]:
                texture_path = texture_paths[_i]
                assert osp.exists(texture_path), texture_path
                self.textures.append(gu.loadTexture(texture_path))
            else:
                self.textures.append(None)

            if color.max() > 1.1:  # in range [0, 1]
                color = color / 255.0
            vertices.append(np.hstack((vertex * vertex_scale, normal, color, texture_uv)).flatten())

        indices = np.hstack(indices).astype(np.uint32)
        vertices = np.hstack(vertices).astype(np.float32)

        self.vao = gu.VAO(
            {
                (gu.Vertexbuffer(vertices), 0, 11 * 4): [
                    (0, 3, GL_FLOAT, GL_FALSE, 0 * 4),  # a_position
                    (1, 3, GL_FLOAT, GL_FALSE, 3 * 4),  # a_normal
                    (2, 3, GL_FLOAT, GL_FALSE, 6 * 4),  # a_color
                    (3, 2, GL_FLOAT, GL_TRUE, 9 * 4),  # a_texcoord
                ]
            },
            gu.EBO(indices),
        )
        self.vao.bind()

        # IBO
        vertex_count = [np.prod(vert["faces"].shape) for vert in attributes]
        instance_count = np.ones(len(attributes))
        first_index = [sum(vertex_count[:i]) for i in range(len(vertex_count))]

        vertex_sizes = [vert["vertices"].shape[0] for vert in attributes]
        base_vertex = [sum(vertex_sizes[:i]) for i in range(len(vertex_sizes))]
        base_instance = np.zeros(len(attributes))

        self.ibo = gu.IBO(
            vertex_count,
            instance_count,
            first_index,
            base_vertex,
            base_instance,
        )
        self.ibo.bind()

        gu.Shader.shader_folder = osp.join(_CUR_DIR, "shader")
        # if clamp:
        #     shader = gu.Shader('depth_shader_phong.vs', 'depth_shader_phong_clamped.frag')
        # else:
        textureless_texture_shader = gu.Shader(
            "depth_shader_textureless_texture.vs",
            "depth_shader_textureless_texture.frag",
        )
        textureless_texture_shader.compile_and_use()

        self.shaders = {"textureless_texture": textureless_texture_shader}
        self.used_shader = "textureless_texture"

        self._scene_buffer = gu.ShaderStorage(
            0,
            gu.Camera(offset_u=self.offset_uv, offset_v=self.offset_uv).data,
            True,
        )
        self._scene_buffer.bind()

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def set_light_pose(self, direction=(0.4, 0.4, 0.4)):
        glUniform3f(0, direction[0], direction[1], direction[2])

    def set_ambient_light(self, a=0.4):
        glUniform1f(1, a)

    def set_diffuse_light(self, a=0.8):
        glUniform1f(2, a)

    def set_specular_light(self, a=0.3):
        glUniform1f(3, a)

    def set_render_type(self, value=_RENDER_PHONG_VERTEX):
        """
        0: color only
        1: phong vertex color
        2: textured
        """
        glUniform1i(4, value)

    def render(
        self,
        obj_id,
        R,
        t,
        K=None,
        W=640,
        H=480,
        near=None,
        far=None,
        random_light=False,
        phong={"ambient": 0.4, "diffuse": 0.8, "specular": 0.3},
        to_255=True,
        light_pose=None,
    ):
        """
        obj_id: the idx of models_cad_files, 0-based
        ----------
        Return:
            (bgr, depth)
        """
        if self.width != W:
            self.width = W
        if self.height != H:
            self.height = H
        if K is not None:
            self.K = K
        if near is not None:
            self.near = near
        if far is not None:
            self.far = far

        assert self.width <= Renderer.MAX_FBO_WIDTH and self.height <= Renderer.MAX_FBO_HEIGHT

        if self._samples > 1:
            self._render_fbo.bind()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glViewport(0, 0, self.width, self.height)

        render_type = _RENDER_PHONG_VERTEX
        if self.is_textured_list[obj_id]:
            render_type = _RENDER_TEXTURED

            GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[obj_id])
            # shader = self.shaders[self.used_shader]
            GL.glUniform1i(5, 0)  # u_texture

        if (not random_light) and (light_pose is None) and (render_type != _RENDER_TEXTURED):
            render_type = _RENDER_COLOR_ONLY
        self.set_render_type(render_type)

        camera = gu.Camera(offset_u=self.offset_uv, offset_v=self.offset_uv)
        camera.realCamera(self.width, self.height, self.K, R, t, self.near, self.far)
        self._scene_buffer.update(camera.data)

        # print(phong)
        if random_light:
            self.set_light_pose(1000.0 * np.random.uniform(-1, 1, 3))  # this should be more random
            # self.set_light_pose(1000.0 * np.random.random(3))  # it is in [0,1], AAE default
            # originally there is not + 0.1 * (2 * np.random.rand() - 1)) for ambient
            self.set_ambient_light(phong["ambient"] + 0.1 * (2 * np.random.rand() - 1))
            self.set_diffuse_light(phong["diffuse"] + 0.1 * (2 * np.random.rand() - 1))
            self.set_specular_light(phong["specular"] + 0.1 * (2 * np.random.rand() - 1))
            # self.set_ambient_light(phong['ambient'])
            # self.set_diffuse_light(0.7)
            # self.set_specular_light(0.3)
        else:
            if light_pose is not None:
                # NOTE: fixed light (or assign random light outside)
                # default is (0.4, 0.4, 0.4)
                self.set_light_pose(np.array(light_pose) * 1000)
            else:
                self.set_light_pose(np.array([1.0, 1.0, 1.0]) * 1000)
            self.set_ambient_light(phong["ambient"])
            self.set_diffuse_light(phong["diffuse"])
            self.set_specular_light(phong["specular"])

        glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(obj_id * 4 * 5))

        # if self._samples > 1:
        #     for i in range(2):
        #         glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0 + i)
        #         glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0 + i)
        #         glBlitNamedFramebuffer(self._render_fbo.id, self._fbo.id, 0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        #     self._fbo.bind()

        if self._samples > 1:
            self._fbo.bind()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(obj_id * 4 * 5))

            glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0)
            glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
            glBlitNamedFramebuffer(
                self._render_fbo.id,
                self._fbo.id,
                0,
                0,
                self.width,
                self.height,
                0,
                0,
                self.width,
                self.height,
                GL_COLOR_BUFFER_BIT,
                GL_NEAREST,
            )

            glNamedFramebufferDrawBuffers(
                self._fbo.id,
                2,
                np.array(
                    (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1),
                    dtype=np.uint32,
                ),
            )

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
        if to_255:
            bgr_flipped = np.frombuffer(
                glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE),
                dtype=np.uint8,
            ).reshape(self.height, self.width, 3)
        else:
            bgr_flipped = np.frombuffer(
                glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_FLOAT),
                dtype=np.float32,
            ).reshape(self.height, self.width, 3)
        bgr = np.flipud(bgr_flipped).copy()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
        depth_flipped = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_FLOAT).reshape(self.height, self.width)
        depth = np.flipud(depth_flipped).copy()

        return bgr, depth

    def render_many(
        self,
        obj_ids,
        Rs,
        ts,
        K=None,
        W=640,
        H=480,
        near=None,
        far=None,
        random_light=False,
        phong={"ambient": 0.4, "diffuse": 0.8, "specular": 0.3},
        to_255=True,
        # to_bgr=True,
        light_pose=None,
        with_mask=True,
    ):
        """
        TODO: how to get masks
        """
        from lib.utils.mask_utils import mask2bbox_xyxy

        if self.width != W:
            self.width = W
        if self.height != H:
            self.height = H
        if K is not None:
            self.K = K
        if near is not None:
            self.near = near
        if far is not None:
            self.far = far
        assert self.width <= Renderer.MAX_FBO_WIDTH and self.height <= Renderer.MAX_FBO_HEIGHT

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.width, self.height)

        if random_light:
            self.set_light_pose(1000.0 * np.random.uniform(-1, 1, 3))
            # self.set_light_pose(1000.0 * np.random.random(3))  # original
            self.set_ambient_light(phong["ambient"] + 0.1 * (2 * np.random.rand() - 1))
            self.set_diffuse_light(phong["diffuse"] + 0.1 * (2 * np.random.rand() - 1))
            self.set_specular_light(phong["specular"] + 0.1 * (2 * np.random.rand() - 1))
            # self.set_ambient_light(phong['ambient'])
            # self.set_diffuse_light(0.7)
            # self.set_specular_light(0.3)
        else:
            if light_pose is not None:
                # self.set_light_pose(np.array([400.0, 400.0, 400]))
                self.set_light_pose(np.array(light_pose) * 1000)
            else:
                self.set_light_pose(np.array([400.0, 400.0, 400]))
            self.set_ambient_light(phong["ambient"])
            self.set_diffuse_light(phong["diffuse"])
            self.set_specular_light(phong["specular"])

        bbs = []  # xyxy, NOTE: original AAE is xywh
        if with_mask:
            masks = []
        else:
            masks = None
        for i in range(len(obj_ids)):
            o = int(obj_ids[i])
            R = Rs[i]
            t = ts[i]
            camera = gu.Camera(offset_u=self.offset_uv, offset_v=self.offset_uv)
            camera.realCamera(self.width, self.height, self.K, R, t, self.near, self.far)
            self._scene_buffer.update(camera.data)

            render_type = _RENDER_PHONG_VERTEX
            if self.is_textured_list[o]:
                render_type = _RENDER_TEXTURED
            if (not random_light) and (light_pose is None) and (render_type != _RENDER_TEXTURED):
                render_type = _RENDER_COLOR_ONLY
            self.set_render_type(render_type)

            if self.is_textured_list[o]:
                GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[o])
                # shader = self.shaders[self.used_shader]
                GL.glUniform1i(5, 0)  # u_texture

            self._fbo.bind()
            # logger.info('{}'.format(type(o)))
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(o * 4 * 5))

            self._fbo_depth.bind()
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(o * 4 * 5))

            glNamedFramebufferReadBuffer(self._fbo_depth.id, GL_COLOR_ATTACHMENT1)
            depth_flipped = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_FLOAT).reshape(
                self.height, self.width
            )
            depth = np.flipud(depth_flipped).copy()

            mask = (depth > 0).astype(np.uint8)
            if with_mask:
                masks.append(mask)
            obj_bb_xyxy = mask2bbox_xyxy(mask)
            bbs.append(obj_bb_xyxy)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo.id)
        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
        if to_255:
            bgr_flipped = np.frombuffer(
                glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE),
                dtype=np.uint8,
            ).reshape(self.height, self.width, 3)
        else:
            bgr_flipped = np.frombuffer(
                glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_FLOAT),
                dtype=np.float32,
            ).reshape(self.height, self.width, 3)
        bgr = np.flipud(bgr_flipped).copy()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
        depth_flipped = glReadPixels(0, 0, self.width, self.height, GL_RED, GL_FLOAT).reshape(self.height, self.width)
        depth = np.flipud(depth_flipped).copy()
        # we can get semantic segmentation via the {t_z}s, however in most cases, we don't need that
        return bgr, depth, bbs, masks

    def close(self):
        self._context.close()
        # self.r.release()


def test_render_many():
    import math
    import time
    from tqdm import tqdm
    from lib.vis_utils.image import (
        grid_show,
        vis_image_mask_bbox_cv2,
        vis_image_bboxes_cv2,
    )
    from lib.pysixd import view_sampler, transform

    model_dir = "datasets/BOP_DATASETS/lm/models"
    K = np.array(
        [
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0],
        ]
    )
    width = 640
    height = 480
    ZNEAR = 0.25
    ZFAR = 6.0
    IDX2CLASS = {
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
    CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}
    classes = IDX2CLASS.values()
    classes = sorted(classes)
    models_cad_files = [osp.join(model_dir, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in IDX2CLASS]
    # obj_ids = [CLASS2IDX[cls_name] for cls_name in classes]
    renderer = Renderer(models_cad_files, K=K, samples=1, vertex_scale=0.001)
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    min_objs = 5
    max_objs = 6
    azimuth_range = (0, 2 * math.pi)
    elev_range = (0 * math.pi, 0.5 * math.pi)
    min_n_views = 1000
    radius = 0.7
    all_views, _ = view_sampler.sample_views(min_n_views, radius, azimuth_range, elev_range)
    rotations = view_sampler.sample_rotations_phere(
        min_n_views=min_n_views,
        radius=0.7,
        azimuth_range=azimuth_range,
        elev_range=elev_range,
        num_cyclo=36,
    )
    # rotations = view_sampler.sample_rotations_sphere_and_inplane(min_n_views, -90, (0, 360))
    print("num of sampled rotations: {}".format(len(rotations)))
    print(type(rotations))
    runs = 200
    t1 = time.perf_counter()
    dt_sample = 0
    dt_render = 0
    for i in tqdm(range(runs)):
        t_sample_0 = time.perf_counter()
        Rs = []
        ts = []
        N = np.random.randint(min_objs, max_objs + 1)

        obj_ids = np.random.choice(len(models_cad_files), N)
        ts = []
        ts_norm = []
        Rs = []

        for i in range(N):
            rand_k = random.randint(0, len(rotations) - 1)
            R = rotations[rand_k]
            success = False
            trial = 0
            while not success:
                tz = np.random.triangular(radius - radius / 3, radius, radius + radius / 3)
                tx = np.random.uniform(-0.35 * tz * width / K[0, 0], 0.35 * tz * width / K[0, 0])
                ty = np.random.uniform(-0.35 * tz * height / K[1, 1], 0.35 * tz * height / K[1, 1])
                t = np.array([tx, ty, tz])
                # print('tx range: ', -0.35 * tz * width / K[0, 0], 0.35 * tz * width / K[0, 0])
                # print('ty range: ', -0.35 * tz * height / K[1, 1], 0.35 * tz * height / K[1, 1])
                # print(t)
                # R = transform.random_rotation_matrix()[:3, :3]
                # R = v['R']
                t_norm = t / np.linalg.norm(t)
                if len(ts_norm) > 0 and np.any(np.dot(np.array(ts_norm), t_norm.reshape(3, 1)) > 0.99):
                    success = False  # too close
                    trial += 1
                    if trial >= 1000:
                        ts_norm.append(t_norm)
                        ts.append(t)
                        Rs.append(R)
                        break
                else:
                    ts_norm.append(t_norm)
                    ts.append(t)
                    Rs.append(R)
                    success = True
        dt_sample += time.perf_counter() - t_sample_0
        t_render_0 = time.perf_counter()
        random_light = bool(random.randint(0, 1))
        bgr, depth, bbs, masks = renderer.render_many(obj_ids, Rs, ts, random_light=random_light, with_mask=True)
        dt_render += time.perf_counter() - t_render_0
        labels = [IDX2CLASS[1 + obj_id] for obj_id in obj_ids]
        if False:  # show
            img_vis = vis_image_mask_bbox_cv2(
                bgr,
                masks,
                bboxes=np.array(bbs),
                labels=labels,
                font_scale=0.5,
                text_color="green",
            )
            img_vis = vis_image_bboxes_cv2(
                bgr,
                bboxes=np.array(bbs),
                labels=labels,
                font_scale=0.5,
                text_color="green",
            )
            grid_show(
                [img_vis[:, :, [2, 1, 0]], depth],
                [
                    "img_vis_randlight:{}, num objs:{}".format(random_light, len(obj_ids)),
                    "depth",
                ],
                row=1,
                col=2,
            )
    t_total = dt_sample + dt_render
    print(
        "t sample {}s, t_render: {}s t_total: {}s, fps: {}".format(
            dt_sample / runs, dt_render / runs, t_total / runs, runs / t_total
        )
    )
    """
    t sample 0.00034781177411787213s, t_render: 0.018074159582029098s t_total: 0.01842197135614697s, fps: 54.283006995683124
    """


def test_render_ycbv():
    model_dir = "datasets/BOP_DATASETS/ycbv/models"

    K = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
    width = 640
    height = 480
    ZNEAR = 0.25
    ZFAR = 6.0
    IDX2CLASS = {
        1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
        2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
        3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
        4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
        5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
        6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
        7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
        8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
        9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
        10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
        11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
        12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
        13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
        14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
        15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
        16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
        17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
        18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
        19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
        20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
        21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
    }
    CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}
    classes = IDX2CLASS.values()
    classes = sorted(classes)
    # image_tensor = torch.cuda.FloatTensor(height, width, 3).detach()
    # depth_tensor = torch.cuda.FloatTensor(height, width, 1).detach()
    models_cad_files = [osp.join(model_dir, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in IDX2CLASS]
    texture_files = [osp.join(model_dir, "obj_{:06d}.png".format(cls_idx)) for cls_idx in IDX2CLASS]

    renderer = Renderer(
        models_cad_files,
        K=K,
        texture_paths=texture_files,
        samples=1,
        vertex_scale=0.001,
    )
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([0, 0, 0.7], dtype=np.float32)

    # benchmark speed
    if True:
        run = 0
        start = time.perf_counter()
        for i in tqdm(range(300)):
            for obj_id, cls_name in enumerate(classes):
                bgr, depth = renderer.render(
                    obj_id,
                    W=width,
                    H=height,
                    K=K,
                    R=R,
                    t=t,
                    near=ZNEAR,
                    far=ZFAR,
                    to_255=True,
                )
                run += 1
        dt = time.perf_counter() - start
        print("total runs: {}, time: {}s, speed: {} fps".format(run, dt, run / dt))
        # total runs: 45000, time: 86.256765127182s, speed: 521.6982103797815 fps

    if True:
        for i in range(1):
            for obj_id, cls_name in enumerate(classes):
                # obj_id = 0
                # if cls_name != 'driller': continue
                print(obj_id, cls_name)
                use_light = True  # bool(random.randint(0, 1))
                random_light = False  # bool(random.randint(0, 1))
                print("use_light", use_light)
                print("random_light", random_light)
                if random_light:
                    light_pose = np.random.uniform(-1, 1, 3)
                else:
                    light_pose = np.ones(3)
                print("light_pose: ", light_pose)
                bgr, depth = renderer.render(
                    obj_id,
                    W=width,
                    H=height,
                    K=K,
                    R=R,
                    t=t,
                    near=ZNEAR,
                    far=ZFAR,
                    to_255=True,
                    random_light=random_light,
                    light_pose=light_pose,
                    # image_tensor=image_tensor,
                    # depth_tensor=depth_tensor,
                    # light_pose=(0.4, 0.4, 0.4) if use_light else None,
                )
                # mmcv.imwrite(bgr, 'tmp_opengl_color_{}.png'.format(cls_name))
                # mmcv.imwrite((depth * 1000).astype(np.uint16), 'opengl_depth_{}.png'.format(cls_name))
                # bgr = image_tensor[:, :, :3]
                # bgr = (bgr.cpu().numpy() + 0.5).astype(np.uint8)

                # depth = depth_tensor[:,:, 0]
                # depth = depth.cpu().numpy()

                print(bgr.shape, bgr.min(), bgr.max(), bgr.dtype)
                fig = plt.figure(frameon=False, dpi=200)
                plt.subplot(1, 2, 1)
                plt.imshow(bgr[:, :, [2, 1, 0]])
                plt.axis("off")
                if random_light:
                    plt.title("{} phong(random)".format(cls_name))
                elif use_light:
                    plt.title("{} phong".format(cls_name))
                else:
                    plt.title("{} color".format(cls_name))

                plt.subplot(1, 2, 2)
                plt.imshow(depth)
                plt.axis("off")
                plt.title("{} depth".format(cls_name))

                # plt.savefig('tmp_{}.png'.format(cls_name))
                plt.show()


def test_render_many_ycbv():
    import math
    import time
    from tqdm import tqdm
    from lib.vis_utils.image import (
        grid_show,
        vis_image_mask_bbox_cv2,
        vis_image_bboxes_cv2,
    )
    from lib.pysixd import view_sampler, transform

    SHOW = False
    model_dir = "datasets/BOP_DATASETS/ycbv/models"
    K = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
    width = 640
    height = 480
    ZNEAR = 0.25
    ZFAR = 6.0
    IDX2CLASS = {
        1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
        2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
        3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
        4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
        5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
        6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
        7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
        8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
        9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
        10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
        11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
        12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
        13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
        14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
        15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
        16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
        17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
        18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
        19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
        20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
        21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
    }
    CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}
    classes = IDX2CLASS.values()
    classes = sorted(classes)
    models_cad_files = [osp.join(model_dir, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in IDX2CLASS]
    texture_files = [osp.join(model_dir, "obj_{:06d}.png".format(cls_idx)) for cls_idx in IDX2CLASS]
    # obj_ids = [CLASS2IDX[cls_name] for cls_name in classes]
    renderer = Renderer(
        models_cad_files,
        K=K,
        texture_paths=texture_files,
        samples=1,
        vertex_scale=0.001,
    )
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    min_objs = 5
    max_objs = 6
    azimuth_range = (0, 2 * math.pi)
    elev_range = (0 * math.pi, 0.5 * math.pi)
    min_n_views = 1000
    radius = 0.7
    all_views, _ = view_sampler.sample_views(min_n_views, radius, azimuth_range, elev_range)
    rotations = view_sampler.sample_rotations_phere(
        min_n_views=min_n_views,
        radius=0.7,
        azimuth_range=azimuth_range,
        elev_range=elev_range,
        num_cyclo=36,
    )
    # rotations = view_sampler.sample_rotations_sphere_and_inplane(min_n_views, -90, (0, 360))
    print("num of sampled rotations: {}".format(len(rotations)))
    print(type(rotations))
    runs = 200
    t1 = time.perf_counter()
    dt_sample = 0
    dt_render = 0
    for i in tqdm(range(runs)):
        t_sample_0 = time.perf_counter()
        Rs = []
        ts = []
        N = np.random.randint(min_objs, max_objs + 1)

        obj_ids = np.random.choice(len(models_cad_files), N)
        ts = []
        ts_norm = []
        Rs = []

        for i in range(N):
            rand_k = random.randint(0, len(rotations) - 1)
            R = rotations[rand_k]
            success = False
            trial = 0
            while not success:
                tz = np.random.triangular(radius - radius / 3, radius, radius + radius / 3)
                tx = np.random.uniform(-0.35 * tz * width / K[0, 0], 0.35 * tz * width / K[0, 0])
                ty = np.random.uniform(-0.35 * tz * height / K[1, 1], 0.35 * tz * height / K[1, 1])
                t = np.array([tx, ty, tz])
                # print('tx range: ', -0.35 * tz * width / K[0, 0], 0.35 * tz * width / K[0, 0])
                # print('ty range: ', -0.35 * tz * height / K[1, 1], 0.35 * tz * height / K[1, 1])
                # print(t)
                # R = transform.random_rotation_matrix()[:3, :3]
                # R = v['R']
                t_norm = t / np.linalg.norm(t)
                if len(ts_norm) > 0 and np.any(np.dot(np.array(ts_norm), t_norm.reshape(3, 1)) > 0.99):
                    success = False  # too close
                    trial += 1
                    if trial >= 1000:
                        ts_norm.append(t_norm)
                        ts.append(t)
                        Rs.append(R)
                        break
                else:
                    ts_norm.append(t_norm)
                    ts.append(t)
                    Rs.append(R)
                    success = True
        dt_sample += time.perf_counter() - t_sample_0
        t_render_0 = time.perf_counter()
        random_light = bool(random.randint(0, 1))
        bgr, depth, bbs, masks = renderer.render_many(obj_ids, Rs, ts, random_light=random_light, with_mask=True)
        dt_render += time.perf_counter() - t_render_0
        labels = [IDX2CLASS[1 + obj_id] for obj_id in obj_ids]
        if SHOW:
            img_vis = vis_image_mask_bbox_cv2(
                bgr,
                masks,
                bboxes=np.array(bbs),
                labels=labels,
                font_scale=0.5,
                text_color="green",
            )
            img_vis = vis_image_bboxes_cv2(
                bgr,
                bboxes=np.array(bbs),
                labels=labels,
                font_scale=0.5,
                text_color="green",
            )
            grid_show(
                [img_vis[:, :, [2, 1, 0]], depth],
                [
                    "img_vis_randlight:{}, num objs:{}".format(random_light, len(obj_ids)),
                    "depth",
                ],
                row=1,
                col=2,
            )
    t_total = dt_sample + dt_render
    print(
        "t sample {}s, t_render: {}s t_total: {}s, fps: {}".format(
            dt_sample / runs, dt_render / runs, t_total / runs, runs / t_total
        )
    )
    """
    t sample 0.0031757269194349646s, t_render: 0.024196420814841985s t_total: 0.02737214773427695s, fps: 36.53348687533728
    """


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import mmcv
    import time
    from tqdm import tqdm
    import random
    from transforms3d.axangles import axangle2mat

    # test_render_many()
    # test_render_ycbv()
    # test_render_many_ycbv()
    # exit(0)

    model_dir = "datasets/BOP_DATASETS/lm/models"

    K = np.array(
        [
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0],
        ]
    )
    width = 640
    height = 480
    ZNEAR = 0.25
    ZFAR = 6.0
    IDX2CLASS = {
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
    CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}
    classes = IDX2CLASS.values()
    classes = sorted(classes)
    # image_tensor = torch.cuda.FloatTensor(height, width, 3).detach()
    # depth_tensor = torch.cuda.FloatTensor(height, width, 1).detach()
    models_cad_files = [osp.join(model_dir, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in IDX2CLASS]
    # obj_ids = [CLASS2IDX[cls_name] for cls_name in classes]
    renderer = Renderer(models_cad_files, K=K, samples=1, vertex_scale=0.001)
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([0, 0, 0.7], dtype=np.float32)

    # benchmark speed
    if True:
        run = 0
        start = time.perf_counter()
        for i in tqdm(range(300)):
            for obj_id, cls_name in enumerate(classes):
                bgr, depth = renderer.render(
                    obj_id,
                    W=width,
                    H=height,
                    K=K,
                    R=R,
                    t=t,
                    near=ZNEAR,
                    far=ZFAR,
                    to_255=True,
                )
                run += 1
        dt = time.perf_counter() - start
        print("total runs: {}, time: {}s, speed: {} fps".format(run, dt, run / dt))
        # total runs: 45000, time: 86.256765127182s, speed: 521.6982103797815 fps

    if True:
        for i in range(1):
            for obj_id, cls_name in enumerate(classes):
                # obj_id = 0
                # if cls_name != 'driller': continue
                print(obj_id, cls_name)
                use_light = True  # bool(random.randint(0, 1))
                random_light = False  # bool(random.randint(0, 1))
                print("use_light", use_light)
                print("random_light", random_light)
                light_pose = np.random.uniform(-1, 1, 3)
                print("light_pose: ", light_pose)
                bgr, depth = renderer.render(
                    obj_id,
                    W=width,
                    H=height,
                    K=K,
                    R=R,
                    t=t,
                    near=ZNEAR,
                    far=ZFAR,
                    to_255=True,
                    random_light=random_light,
                    light_pose=light_pose,
                    # image_tensor=image_tensor,
                    # depth_tensor=depth_tensor,
                    # light_pose=(0.4, 0.4, 0.4) if use_light else None,
                )
                # mmcv.imwrite(bgr, 'tmp_opengl_color_{}.png'.format(cls_name))
                # mmcv.imwrite((depth * 1000).astype(np.uint16), 'opengl_depth_{}.png'.format(cls_name))
                # bgr = image_tensor[:, :, :3]
                # bgr = (bgr.cpu().numpy() + 0.5).astype(np.uint8)

                # depth = depth_tensor[:,:, 0]
                # depth = depth.cpu().numpy()

                print(bgr.shape, bgr.min(), bgr.max(), bgr.dtype)
                fig = plt.figure(frameon=False, dpi=200)
                plt.subplot(1, 2, 1)
                plt.imshow(bgr[:, :, [2, 1, 0]])
                plt.axis("off")
                if random_light:
                    plt.title("{} phong(random)".format(cls_name))
                elif use_light:
                    plt.title("{} phong".format(cls_name))
                else:
                    plt.title("{} color".format(cls_name))

                plt.subplot(1, 2, 2)
                plt.imshow(depth)
                plt.axis("off")
                plt.title("{} depth".format(cls_name))

                # plt.savefig('tmp_{}.png'.format(cls_name))
                plt.show()
