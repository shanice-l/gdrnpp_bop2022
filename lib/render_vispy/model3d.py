import hashlib
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import pyassimp
import pyassimp.postprocess

# import optimesh
import scipy
from meshplex import MeshTri
from plyfile import PlyData, PlyElement
from scipy.spatial.distance import pdist
from skimage import measure
from tqdm import tqdm
from vispy import gloo

from lib.pysixd import inout
from lib.utils.utils import iprint

"""
modified
support model_loadfn to load objects
* pyassimp
* pysixd: the bop toolkit implementation
* plydata: the original implementation
"""


def _fast_add_at(target, idx, vals):
    # https://github.com/ml31415/numpy-groupies/issues/24
    # https://github.com/numpy/numpy/issues/5922
    return target + np.bincount(idx, weights=vals, minlength=target.shape[0])


class Model3D:
    """"""

    def __init__(
        self,
        file_to_load=None,
        center=False,
        scale_to_meter=1.0,
        finalize=True,
    ):
        """
        finalize:
        """
        self.scale_to_meter = scale_to_meter
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None
        if file_to_load:
            self.load(file_to_load, center, scale_to_meter)
        if finalize:
            self.finalize()

    def _compute_bbox(self):

        self.bb = []
        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        avgx = np.average(self.vertices[:, 0])
        avgy = np.average(self.vertices[:, 1])
        avgz = np.average(self.vertices[:, 2])
        self.bb.append([minx, miny, minz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, maxz])
        self.bb.append([avgx, avgy, avgz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        self.diameter = max(pdist(self.bb, "euclidean"))

        # Set up rendering data
        # fmt: off
        colors = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
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
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)
        # self.bb_vbuffer = gloo.VertexBuffer(collated)
        # self.bb_ibuffer = gloo.IndexBuffer(indices)
        self.bb_vbuffer = collated
        self.bb_ibuffer = indices

    def load(
        self,
        path,
        center=False,
        scale_to_meter=1.0,
        flip_opengl=False,
        texture_path=None,
    ):

        suffix = osp.basename(path).split(".")[-1]
        if suffix.lower() == "ply":
            self.load_ply(path)
        elif suffix.lower() == "obj":
            self.load_obj(path, tex=texture_path)
        else:
            raise ValueError("Cannot load models with ending {}".format(suffix))

        self.scale_to_meter = scale_to_meter
        self.vertices *= self.scale_to_meter

        if center:
            xmin, xmax = np.amin(self.vertices[:, 0]), np.amax(self.vertices[:, 0])
            ymin, ymax = np.amin(self.vertices[:, 1]), np.amax(self.vertices[:, 1])
            zmin, zmax = np.amin(self.vertices[:, 2]), np.amax(self.vertices[:, 2])

            self.vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
            self.vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
            self.vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

        if flip_opengl:
            self.yz_flip = np.eye(4, dtype=np.float32)
            # self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1
            # self.yz_flip[0, 0], self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1, -1
            # self.vertices = np.matmul(self.yz_flip[:3, :3], self.vertices.T).T
            # self.vertices[:, 1] *= -1

    def finalize(self, center=False):
        xmin, xmax = np.amin(self.vertices[:, 0]), np.amax(self.vertices[:, 0])
        ymin, ymax = np.amin(self.vertices[:, 1]), np.amax(self.vertices[:, 1])
        zmin, zmax = np.amin(self.vertices[:, 2]), np.amax(self.vertices[:, 2])

        self.xsize = xmax - xmin
        self.ysize = ymax - ymin
        self.zsize = zmax - zmin
        if center:
            # fmt: off
            self.vertices[:, 0] -= (xmax + xmin) * 0.5
            self.vertices[:, 1] -= (ymax + ymin) * 0.5
            self.vertices[:, 2] -= (zmax + zmin) * 0.5
            # fmt: on
        self.centroid = np.mean(self.vertices, 0)

        self._compute_bbox()

        if self.colors is None:  # gray color
            self.colors = 0.5 * np.ones((self.vertices.shape[0], 3))

        if self.texcoord is not None:
            vertices_type = [
                ("a_position", np.float32, 3),
                ("a_texcoord", np.float32, 2),
            ]
            self.collated = np.asarray(list(zip(self.vertices, self.texcoord)), vertices_type)
        else:
            vertices_type = [
                ("a_position", np.float32, 3),
                ("a_color", np.float32, 3),
            ]
            self.collated = np.asarray(list(zip(self.vertices, self.colors)), vertices_type)

        # self.vertex_buffer = gloo.VertexBuffer(self.collated)
        # self.index_buffer = gloo.IndexBuffer(self.indices.flatten())
        self.vertex_buffer = self.collated
        self.index_buffer = self.indices.tolist()

    def normalize(self, scale):
        xmin, xmax = np.amin(self.vertices[:, 0]), np.amax(self.vertices[:, 0])
        ymin, ymax = np.amin(self.vertices[:, 1]), np.amax(self.vertices[:, 1])
        zmin, zmax = np.amin(self.vertices[:, 2]), np.amax(self.vertices[:, 2])

        self.xsize = xmax - xmin
        self.ysize = ymax - ymin
        self.zsize = zmax - zmin

        self.vertices[:, 0] -= (xmax + xmin) * 0.5
        self.vertices[:, 1] -= (ymax + ymin) * 0.5
        self.vertices[:, 2] -= (zmax + zmin) * 0.5

        self.vertices = (self.vertices / np.max(self.vertices, axis=0)) * scale

        # print(scale, np.max(self.vertices, axis=0), np.min(self.vertices, axis=0))

    def load_obj(self, path, tex=None):
        """Loads a Wavefront OBJ file."""
        self.path = path

        scene = pyassimp.load(path, processing=pyassimp.postprocess.aiProcess_Triangulate)

        self.vertices = np.asarray([]).reshape(0, 3)
        self.normals = np.asarray([]).reshape(0, 3)
        self.texcoord = np.asarray([]).reshape(0, 2)
        self.indices = np.asarray([], dtype=np.uint32).reshape(0, 3)

        for mesh_id, mesh in enumerate(scene.meshes):

            iprint(
                mesh.texturecoords.shape,
                mesh.vertices.shape,
                mesh.faces.shape,
                mesh.normals.shape,
                self.vertices.shape[0],
            )

            if mesh.texturecoords.shape[0] == 0 or mesh.texturecoords.shape[1] != mesh.vertices.shape[0]:
                continue
            # print(self.indices.shape, (self.vertices.shape[0] + np.asarray(mesh.faces, dtype=np.uint32)).shape)
            self.indices = np.concatenate(
                [
                    self.indices,
                    self.vertices.shape[0] + np.asarray(mesh.faces, dtype=np.uint32),
                ]
            )
            self.vertices = np.concatenate([self.vertices, mesh.vertices])
            self.normals = np.concatenate([self.normals, mesh.normals])
            if mesh.texturecoords.shape[0] == 0:
                self.texcoord = np.concatenate([self.texcoord, np.zeros((mesh.vertisces.shape[0], 2))])
            else:
                self.texcoord = np.concatenate([self.texcoord, mesh.texturecoords[0, :, :2]])

            # print(mesh_id, self.indices[:-10])

        if tex is not None:
            image = cv2.flip(cv2.imread(tex, cv2.IMREAD_UNCHANGED), 0)

            if False:
                cv2.imshow("tex", image)
                cv2.waitKey()
            # self.texture = gloo.Texture2D(image, resizable=False)
            self.texture = image

        iprint(np.min(self.texcoord, axis=0), np.max(self.texcoord, axis=0))
        # if (np.max(self.texcoord, axis=0) > 1).any() or (np.min(self.texcoord, axis=0) < 0).any():
        #    self.texcoord -= np.min(self.texcoord, axis=0)
        #    self.texcoord /= np.max(self.texcoord, axis=0)

        #    self.texcoord = [2.5, 1.5] * self.texcoord
        if self.vertices.shape[0] < 100:
            return False

        return True

    def load_ply(self, path):
        data = PlyData.read(path)
        self.vertices = np.zeros((data["vertex"].count, 3))
        self.vertices[:, 0] = np.array(data["vertex"]["x"])
        self.vertices[:, 1] = np.array(data["vertex"]["y"])
        self.vertices[:, 2] = np.array(data["vertex"]["z"])
        self.indices = np.asarray(list(data["face"]["vertex_indices"]), np.uint32)

        # Look for texture map as jpg or png
        filename = osp.basename(path)
        abs_path = path[: path.find(filename)]
        tex_to_load = None
        if osp.exists(abs_path + filename[:-4] + ".jpg"):
            tex_to_load = abs_path + filename[:-4] + ".jpg"
        elif osp.exists(abs_path + filename[:-4] + ".png"):
            tex_to_load = abs_path + filename[:-4] + ".png"

        # Try to read out texture coordinates
        if tex_to_load is not None:
            iprint("Loading {} with texture {}".format(osp.normpath(filename), tex_to_load))
            # Must be flipped because of OpenGL
            image = cv2.flip(cv2.imread(tex_to_load, cv2.IMREAD_UNCHANGED), 0)
            # self.texture = gloo.Texture2D(image)
            self.texture = image

            # If texcoords are face-wise
            if "texcoord" in str(data):
                self.texcoord = np.asarray(list(data["face"]["texcoord"]))
                # Check same face count
                assert self.indices.shape[0] == self.texcoord.shape[0]
                temp = np.zeros((data["vertex"].count, 2))
                temp[self.indices.flatten()] = self.texcoord.reshape((-1, 2))
                self.texcoord = temp

            # If texcoords are vertex-wise
            elif "texture_u" in str(data):
                self.texcoord = np.zeros((data["vertex"].count, 2))
                self.texcoord[:, 0] = np.array(data["vertex"]["texture_u"])
                self.texcoord[:, 1] = np.array(data["vertex"]["texture_v"])

        # If no texture coords loaded, fall back to vertex colors
        if self.texcoord is None:
            self.colors = 0.5 * np.ones((data["vertex"].count, 3))
            if "blue" in str(data):
                iprint("Loading {} with vertex colors".format(filename))
                self.colors[:, 0] = np.array(data["vertex"]["blue"])
                self.colors[:, 1] = np.array(data["vertex"]["green"])
                self.colors[:, 2] = np.array(data["vertex"]["red"])
                self.colors /= 255.0
            else:
                iprint("Loading {} without any colors!!".format(filename))

    def _smooth_laplacian(self, vertices, faces, iterations):
        mesh = MeshTri(vertices, faces)
        # move interior points into average of their neighbors
        num_neighbors = np.zeros(len(mesh.node_coords), dtype=int)

        idx = mesh.edges["nodes"]
        num_neighbors = _fast_add_at(num_neighbors, idx, np.ones(idx.shape, dtype=int))

        new_points = np.zeros(mesh.node_coords.shape)
        new_points = _fast_add_at(new_points, idx[:, 0], mesh.node_coords[idx[:, 1]])
        new_points = _fast_add_at(new_points, idx[:, 1], mesh.node_coords[idx[:, 0]])

        new_points /= num_neighbors[:, None]
        idx = mesh.is_boundary_node
        new_points[idx] = mesh.node_coords[idx]

        return new_points

    # Takes sdf and extends to return a Model3D
    def load_from_tsdf(
        self,
        sdf,
        extends=[1, 1, 1],
        spacing=(2.0, 2.0, 2.0),
        step_size=2,
        laplacian_smoothing=False,
        color=None,
        image_colors=None,
        pose=None,
        cam=None,
        points_and_colors=None,
    ):
        # Use marching cubes to obtain the surface mesh of these ellipsoids

        verts, faces, normals, _ = measure.marching_cubes_lewiner(sdf, 0, spacing=spacing, step_size=step_size)

        # quenze it between 0 and 1
        for i in range(3):
            verts[:, i] = verts[:, i] - np.min(verts[:, i])
            verts[:, i] /= np.max(verts[:, i])

        # tsdf is upside down
        # verts[:, 1] *= -1

        # scale mesh to correct size of gt
        verts = verts * np.asarray(extends)

        # load vertices, faces, normals and finalize
        self.indices = np.asarray(faces, dtype=np.uint32)
        self.vertices = np.asarray(verts)

        xmin, xmax = np.amin(self.vertices[:, 0]), np.amax(self.vertices[:, 0])
        ymin, ymax = np.amin(self.vertices[:, 1]), np.amax(self.vertices[:, 1])
        zmin, zmax = np.amin(self.vertices[:, 2]), np.amax(self.vertices[:, 2])

        self.xsize = xmax - xmin
        self.ysize = ymax - ymin
        self.zsize = zmax - zmin

        self.vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
        self.vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
        self.vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

        self.normals = np.asarray(normals)

        if image_colors is not None and pose is not None and cam is not None:
            verts_up_down = self.vertices.copy()

            vertices = np.matmul(pose[:3, :3], verts_up_down.T).T + pose[:3, 3]

            rgb = image_colors[:, :, ::-1].copy()

            # take scene colors to make it look better

            # version two
            img_pt = np.matmul(cam, vertices.T).T
            img_pt[:, :2] /= img_pt[:, 2:3]
            img_pt = np.asarray(img_pt[:, :2], dtype=np.int32)

            color = []
            for p in img_pt:

                if p[0] > 0 and p[0] < image_colors.shape[1] and p[1] > 0 and p[1] < image_colors.shape[0]:
                    color.append(rgb[p[1], p[0]])
                else:
                    color.append([1, 0, 0])
            self.colors = np.asarray(color)
        elif points_and_colors is not None and pose is not None:
            verts_up_down = self.vertices.copy()
            vertices = np.matmul(pose[:3, :3], verts_up_down.T).T + pose[:3, 3]

            near_points = []
            near_colors = []
            for point, color in zip(points_and_colors[0], points_and_colors[1]):
                centroid = pose[:3, 3]

                if np.linalg.norm(point - centroid) < 6.0:
                    near_points.append(point)
                    near_colors.append(color)

            near_points = np.asarray(near_points)
            near_colors = np.asarray(near_colors)

            dist = scipy.spatial.distance.cdist(vertices, near_points)
            colors_idx = np.argmin(dist, axis=1)
            self.colors = near_colors[colors_idx][:, :3]

        else:
            if color is None:
                self.colors = 0.5 * np.ones((self.vertices.shape[0], 3))
            else:
                self.colors = color * np.ones((self.vertices.shape[0], 3))

        # if laplacian_smoothing:
        #     self.vertices = self._smooth_laplacian(self.vertices, self.indices, 50)

        self.finalize(center=False)

    def save_ply(self, path):
        header = "ply\nformat ascii 1.0\n"
        header += "element vertex {}\n".format(self.vertices.shape[0])
        header += (
            "property float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar red\n"
            "property uchar green\nproperty uchar blue\nproperty float32 nx\n"
            "property float32 ny\nproperty float32 nz\n"
        )

        header += "element face {}\n".format(self.indices.shape[0])
        header += "property list uchar uint vertex_indices\n"
        header += "end_header\n"
        # fmt: off
        body = ""
        for i in range(self.vertices.shape[0]):
            body += "{:.5f} {:.5f} {:.5f} {} {} {} {:.5f} {:.5f} {:.5f}\n".format(
                self.vertices[i, 0],
                self.vertices[i, 1],
                self.vertices[i, 2],
                int(self.colors[i, 2] * 255),
                int(self.colors[i, 1] * 255),
                int(self.colors[i, 0] * 255),
                self.normals[i, 0],
                self.normals[i, 1],
                self.normals[i, 2],
            )

        for i in range(self.indices.shape[0]):
            body += "3 {:d} {:d} {:d}\n".format(
                int(self.indices[i, 0]), int(self.indices[i, 1]), int(self.indices[i, 2]))
        # fmt: on
        to_write = header + body
        with open(path, "w") as ply:
            ply.write(to_write)

    def save_ply_without_normals(self, path):
        header = "ply\nformat ascii 1.0\n"
        header += "element vertex {}\n".format(self.vertices.shape[0])
        header += (
            "property float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar red\n"
            "property uchar green\nproperty uchar blue\n"
        )

        header += "element face {}\n".format(self.indices.shape[0])
        header += "property list uchar uint vertex_indices\n"
        header += "end_header\n"

        body = ""
        for i in range(self.vertices.shape[0]):
            body += "{:.5f} {:.5f} {:.5f} {} {} {}\n".format(
                self.vertices[i, 0],
                self.vertices[i, 1],
                self.vertices[i, 2],
                int(self.colors[i, 2] * 255),
                int(self.colors[i, 1] * 255),
                int(self.colors[i, 0] * 255),
            )

        for i in range(self.indices.shape[0]):
            body += "3 {:d} {:d} {:d}\n".format(
                int(self.indices[i, 0]),
                int(self.indices[i, 1]),
                int(self.indices[i, 2]),
            )

        to_write = header + body
        with open(path, "w") as ply:
            ply.write(to_write)


def load_models(
    model_paths,
    scale_to_meter=0.001,
    cache_dir=".cache",
    texture_paths=None,
    center=False,
    use_cache=True,
):
    hashed_file_name = hashlib.md5(("{}".format(model_paths)).encode("utf-8")).hexdigest()
    cache_path = osp.normpath(osp.join(cache_dir, "models_{}.pkl".format(hashed_file_name)))
    if use_cache and osp.exists(cache_path):
        iprint("loading models from {}".format(cache_path))
        models = mmcv.load(cache_path)
    else:
        if texture_paths is not None:
            assert len(model_paths) == len(texture_paths), f"{len(model_paths)} != {len(texture_paths)}"
        models = []
        # objs = data_ref.objects
        # model_paths = data_ref.model_paths
        for i, model_path in enumerate(tqdm(model_paths)):
            model = Model3D(finalize=False)
            if texture_paths is not None:
                texture_path = texture_paths[i]
            else:
                texture_path = None
            model.load(
                model_path,
                scale_to_meter=scale_to_meter,
                flip_opengl=True,
                texture_path=texture_path,
            )
            model.finalize(center=center)
            models.append(model)

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(models, cache_path)
        iprint("models were dumped to {}".format(cache_path))
    # convert buffers to gloo buffers
    for model in tqdm(models):
        model.vertex_buffer = gloo.VertexBuffer(model.vertex_buffer)
        # import ipdb; ipdb.set_trace()
        # NOTE: save index_buffer in list, load as uint32 np.array
        model.index_buffer = gloo.IndexBuffer(model.index_buffer)
        if model.bb_vbuffer is not None:
            model.bb_vbuffer = gloo.VertexBuffer(model.bb_vbuffer)
        if model.bb_ibuffer is not None:
            model.bb_ibuffer = gloo.IndexBuffer(model.bb_ibuffer)
        if model.texture is not None:
            model.texture = gloo.Texture2D(model.texture)
    return models


if __name__ == "__main__":
    import ref

    ref_key = "lm_full"

    data_ref = ref.__dict__[ref_key]
    models = load_models(data_ref.model_paths)
    # import ipdb; ipdb.set_trace()
