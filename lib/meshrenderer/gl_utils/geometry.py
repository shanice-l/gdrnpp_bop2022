# -*- coding: utf-8 -*-
# flake8: noqa
import os
import os.path as osp
import numpy as np
import hashlib
import mmcv
import pyassimp
from tqdm import tqdm
from lib.utils import logger
from lib.pysixd import inout
from PIL import Image


def load(filename):
    scene = pyassimp.load(
        filename,
        processing=pyassimp.postprocess.aiProcess_GenUVCoords | pyassimp.postprocess.aiProcess_Triangulate,
    )
    mesh = scene.meshes[0]
    return mesh.vertices, mesh.normals, mesh.texturecoords[0, :, :2]


def load_meshes_sixd(
    obj_files,
    vertex_tmp_store_folder=".cache",
    recalculate_normals=False,
    render_uv=False,
    render_normalized_coords=False,
    render_nocs=False,
    use_cache=True,
    model_infos=None,
    cad_model_colors=None,
    texture_paths=None,
):
    """load mesh attributes with pysixd.inout.load_ply.

    :param obj_files:
    :param vertex_tmp_store_folder:
    :param recalculate_normals:
    :param render_uv:
    :param render_normalized_coords:
    :return:
    """
    mmcv.mkdir_or_exist(vertex_tmp_store_folder)
    hashed_file_name = (
        hashlib.md5(
            (
                "".join(obj_files)
                + "load_meshes_sixd"
                + str(recalculate_normals)
                + str(render_uv)
                + str(render_normalized_coords)
                + str(render_nocs)
                + f"{cad_model_colors}"
                + f"{texture_paths}"
            ).encode("utf-8")
        ).hexdigest()
        + "_load_meshes_sixd.pkl"
    )

    out_file = os.path.join(vertex_tmp_store_folder, hashed_file_name)
    if osp.exists(out_file) and use_cache:
        logger.info("loading {}".format(out_file))
        if render_uv:
            logger.info("render uv")
        if render_normalized_coords:
            logger.info("render normalized coords")
        if render_nocs:
            logger.info("render nocs")
        return mmcv.load(out_file)
    else:
        if cad_model_colors is not None:
            # list of tuple
            assert len(cad_model_colors) == len(obj_files)
        attributes = []
        for model_i, model_path in enumerate(tqdm(obj_files)):
            logger.info("loading {}".format(model_path))
            model = inout.load_ply(model_path)
            vertices = np.array(model["pts"]).astype(np.float32)
            num_pts = vertices.shape[0]
            if recalculate_normals:
                normals = calc_normals(vertices)
            else:
                normals = np.array(model["normals"]).astype(np.float32)

            assert (
                int(render_uv + render_normalized_coords + render_nocs) <= 1
            ), "render_uv, render_normalized_coords, render_nocs can not be True the same time"
            if render_uv:
                logger.info("render uv")
                model["colors"] = np.zeros((num_pts, 3), np.float32)
                model["colors"][:, 1:] = model["texture_uv"]
            if render_normalized_coords:  # each axis normalized within [0, 1]
                logger.info("render normalized coords")
                # assert model_infos is not None
                normalizedCoords = np.copy(vertices)
                if model_infos is None:
                    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
                    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
                    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
                else:
                    xmin, xmax = (
                        model_infos[model_i]["xmin"],
                        model_infos[model_i]["xmax"],
                    )
                    ymin, ymax = (
                        model_infos[model_i]["ymin"],
                        model_infos[model_i]["ymax"],
                    )
                    zmin, zmax = (
                        model_infos[model_i]["zmin"],
                        model_infos[model_i]["zmax"],
                    )
                # normalize every axis to [0, 1]
                normalizedCoords[:, 0] = (normalizedCoords[:, 0] - xmin) / (xmax - xmin)
                normalizedCoords[:, 1] = (normalizedCoords[:, 1] - ymin) / (ymax - ymin)
                normalizedCoords[:, 2] = (normalizedCoords[:, 2] - zmin) / (zmax - zmin)
                model["colors"] = normalizedCoords
            if render_nocs:  # diagnal normalized to 1, and min corner moved to (0,0,0)
                logger.info("render nocs")
                # Centering and scaling to fit the unit box
                nocs = np.copy(vertices)
                if model_infos is None:
                    xmin, xmax = nocs[:, 0].min(), nocs[:, 0].max()
                    ymin, ymax = nocs[:, 1].min(), nocs[:, 1].max()
                    zmin, zmax = nocs[:, 2].min(), nocs[:, 2].max()
                    diagonal = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
                else:
                    xmin, xmax = (
                        model_infos[model_i]["xmin"],
                        model_infos[model_i]["xmax"],
                    )
                    ymin, ymax = (
                        model_infos[model_i]["ymin"],
                        model_infos[model_i]["ymax"],
                    )
                    zmin, zmax = (
                        model_infos[model_i]["zmin"],
                        model_infos[model_i]["zmax"],
                    )
                    diagonal = model_infos[model_i]["diagonal"]
                # # move (xmin, ymin, zmin) to origin, model centered at the 3b bbox center
                nocs[:, 0] -= xmin
                nocs[:, 1] -= ymin
                nocs[:, 2] -= zmin
                # scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin)
                # unit diagonal
                nocs /= diagonal
                model["colors"] = nocs

            faces = np.array(model["faces"]).astype(np.uint32)
            is_cad = False
            is_textured = False

            if "colors" in model:  # model with vertex color
                # NOTE: hack
                logger.info("colors max: {}".format(model["colors"].max()))
                if model["colors"].max() < 1.1:  # in range [0, 1]
                    logger.info("colors in [0, 1]")
                    colors = np.array(model["colors"]).astype(np.float32)
                else:
                    colors = np.array(model["colors"]).astype(np.uint32)

            elif "texture_uv" in model:  # textured model
                is_textured = True

                # UV texture coordinates.
                texture_uv = model["texture_uv"]

                colors = np.zeros((vertices.shape[0], 3), np.float32)  # dummy colors

            else:  # model without vertex color or texture
                is_cad = True
                colors = np.zeros((vertices.shape[0], 3), np.float32)  # dummy colors
                # TODO: assign colors for cad models here
                if cad_model_colors is None:
                    colors[:, 0] = 223.0 / 255
                    colors[:, 1] = 214.0 / 255
                    colors[:, 2] = 205.0 / 255
                else:
                    colors[:, 0] = cad_model_colors[model_i][0] / 255
                    colors[:, 1] = cad_model_colors[model_i][1] / 255
                    colors[:, 2] = cad_model_colors[model_i][2] / 255

            #############################
            if not is_textured:
                # Set UV texture coordinates to dummy values.
                texture_uv = np.zeros((vertices.shape[0], 2), np.float32)

            attributes.append(
                dict(
                    vertices=vertices,
                    normals=normals,
                    colors=colors,
                    faces=faces,
                    is_cad=is_cad,
                    is_textured=is_textured,
                    texture_uv=texture_uv,
                )
            )
        mmcv.dump(attributes, out_file)
        return attributes


def load_meshes(
    obj_files,
    vertex_tmp_store_folder=".cache",
    recalculate_normals=False,
    use_cache=True,
):
    mmcv.mkdir_or_exist(vertex_tmp_store_folder)
    hashed_file_name = (
        hashlib.md5(("".join(obj_files) + "load_meshes" + str(recalculate_normals)).encode("utf-8")).hexdigest()
        + "_load_meshes.pkl"
    )

    out_file = os.path.join(vertex_tmp_store_folder, hashed_file_name)
    if os.path.exists(out_file) and use_cache:
        return mmcv.load(out_file)
    else:
        attributes = []
        for model_path in tqdm(obj_files):
            assert osp.exists(model_path), model_path
            logger.info("loading {}".format(osp.normpath(model_path)))
            # import ipdb;ipdb.set_trace()
            """
            a possible bug of not properly triangulation caused by libassimp-dev
            Ubuntu 16.04: libassimp-dev 3.2 is OK
            Ubuntu 18.04: libassimp-dev 4.1 is buggy.
                ```
                sudo apt remove libassimp-dev
                download assimp 3.2 from github, follow the INSTALL guide to build
                add environment variable to ~/.bashrc:
                export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Downloads/assimp-3.2/lib
            """
            scene = pyassimp.load(
                model_path,
                processing=pyassimp.postprocess.aiProcess_Triangulate,
            )
            mesh = scene.meshes[0]
            vertices = []
            for face in mesh.faces:
                vertices.extend([mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]])
            vertices = np.array(vertices)
            # vertices = mesh.vertices
            normals = calc_normals(vertices) if recalculate_normals else mesh.normals
            result = {
                "vertices": vertices,
                "normals": normals,
                "faces": mesh.faces,
            }
            if sum(mesh.colors.shape) > 0:
                result["colors"] = mesh.colors[0, :, :3]
                is_cad = False
            else:
                logger.warning("no colors available")
                is_cad = True
                result["colors"] = np.zeros((vertices.shape[0], 3), np.float32)
            result["is_cad"] = is_cad
            pyassimp.release(scene)
            attributes.append(result)
        mmcv.dump(attributes, out_file)
        return attributes


def calc_normals(vertices):
    normals = np.empty_like(vertices)
    N = vertices.shape[0]
    for i in range(0, N - 1, 3):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        v3 = vertices[i + 2]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        normal = np.zeros(3) if norm == 0 else normal / norm
        normals[i] = normal
        normals[i + 1] = normal
        normals[i + 2] = normal
    return normals


# src: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.2.ibl_specular_textured/ibl_specular_textured.cpp
def sphere(x_segments, y_segments):

    N = (x_segments + 1) * (y_segments + 1)
    positions = np.empty((N, 3), dtype=np.float32)
    uv = np.empty((N, 2), dtype=np.float32)
    normals = np.empty((N, 3), dtype=np.float32)

    i = 0
    for y in range(y_segments + 1):
        for x in range(x_segments + 1):
            xSegment = float(x) / float(x_segments)
            ySegment = float(y) / float(y_segments)
            xPos = np.cos(xSegment * 2.0 * np.pi) * np.sin(ySegment * np.pi)
            yPos = np.cos(ySegment * np.pi)
            zPos = np.sin(xSegment * 2.0 * np.pi) * np.sin(ySegment * np.pi)

            positions[i] = (xPos, yPos, zPos)
            uv[i] = (xSegment, ySegment)
            normals[i] = (xPos, yPos, zPos)
            i += 1

    indices = []
    oddRow = False
    for y in range(y_segments):
        if not oddRow:
            for x in range(x_segments + 1):
                indices.append(y * (x_segments + 1) + x)
                indices.append((y + 1) * (x_segments + 1) + x)
        else:
            for x in reversed(range(x_segments + 1)):
                indices.append((y + 1) * (x_segments + 1) + x)
                indices.append(y * (x_segments + 1) + x)
        oddRow = not oddRow
    indices = np.array(indices, dtype=np.uint32)

    return positions, uv, normals, indices


def cube():
    positions = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return positions, uv, normals


def cube2(min, max):
    positions = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    normals = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    uv = np.array(
        [
            [min, min],
            [max, max],
            [max, min],
            [max, max],
            [min, min],
            [min, max],
            [min, min],
            [max, min],
            [max, max],
            [max, max],
            [min, max],
            [min, min],
            [max, min],
            [max, max],
            [min, max],
            [min, max],
            [min, min],
            [max, min],
            [max, min],
            [min, max],
            [max, max],
            [min, max],
            [max, min],
            [min, min],
            [min, max],
            [max, max],
            [max, min],
            [max, min],
            [min, min],
            [min, max],
            [min, max],
            [max, min],
            [max, max],
            [max, min],
            [min, max],
            [min, min],
        ],
        dtype=np.float32,
    )
    return positions, uv, normals


def quad(reverse_uv=False):
    positions = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    if reverse_uv:
        uv = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    else:
        uv = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return positions, uv


cube_vertices_texture = np.array(
    [
        -0.5,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.5,
        -0.5,
        -0.5,
        1.0,
        0.0,
        0.5,
        0.5,
        -0.5,
        1.0,
        1.0,
        0.5,
        0.5,
        -0.5,
        1.0,
        1.0,
        -0.5,
        0.5,
        -0.5,
        0.0,
        1.0,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        0.0,
        -0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        0.5,
        -0.5,
        0.5,
        1.0,
        0.0,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        -0.5,
        0.5,
        0.5,
        0.0,
        1.0,
        -0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        -0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        -0.5,
        0.5,
        -0.5,
        1.0,
        1.0,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        -0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        -0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        0.5,
        0.5,
        -0.5,
        1.0,
        1.0,
        0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        0.5,
        -0.5,
        -0.5,
        1.0,
        1.0,
        0.5,
        -0.5,
        0.5,
        1.0,
        0.0,
        0.5,
        -0.5,
        0.5,
        1.0,
        0.0,
        -0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        1.0,
        -0.5,
        0.5,
        -0.5,
        0.0,
        1.0,
        0.5,
        0.5,
        -0.5,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
        -0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        -0.5,
        0.5,
        -0.5,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)


def quad_bitangent():
    verts = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    uv = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    edge1 = verts[1] - verts[0]
    edge2 = verts[2] - verts[0]
    deltaUV1 = uv[1] - uv[0]
    deltaUV2 = uv[2] - uv[0]

    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

    tangent1 = f * np.array(
        [
            deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0],
            deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1],
            deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2],
        ],
        dtype=np.float32,
    )
    tangent1 /= np.linalg.norm(tangent1)

    bitangent1 = f * np.array(
        [
            -deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0],
            -deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1],
            -deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2],
        ],
        dtype=np.float32,
    )
    bitangent1 /= np.linalg.norm(bitangent1)

    edge1 = verts[2] - verts[0]
    edge2 = verts[3] - verts[0]
    deltaUV1 = uv[2] - uv[0]
    deltaUV2 = uv[3] - uv[0]

    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

    tangent2 = f * np.array(
        [
            deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0],
            deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1],
            deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2],
        ],
        dtype=np.float32,
    )
    tangent2 /= np.linalg.norm(tangent2)

    bitangent2 = f * np.array(
        [
            -deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0],
            -deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1],
            -deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2],
        ],
        dtype=np.float32,
    )
    bitangent2 /= np.linalg.norm(bitangent2)

    return np.array(
        [
            verts[0][0],
            verts[0][1],
            verts[0][2],
            uv[0][0],
            uv[0][1],
            normal[0],
            normal[1],
            normal[2],
            tangent1[0],
            tangent1[1],
            tangent1[2],
            bitangent1[0],
            bitangent1[1],
            bitangent1[2],
            verts[1][0],
            verts[1][1],
            verts[1][2],
            uv[1][0],
            uv[1][1],
            normal[0],
            normal[1],
            normal[2],
            tangent1[0],
            tangent1[1],
            tangent1[2],
            bitangent1[0],
            bitangent1[1],
            bitangent1[2],
            verts[2][0],
            verts[2][1],
            verts[2][2],
            uv[2][0],
            uv[2][1],
            normal[0],
            normal[1],
            normal[2],
            tangent1[0],
            tangent1[1],
            tangent1[2],
            bitangent1[0],
            bitangent1[1],
            bitangent1[2],
            verts[0][0],
            verts[0][1],
            verts[0][2],
            uv[0][0],
            uv[0][1],
            normal[0],
            normal[1],
            normal[2],
            tangent2[0],
            tangent2[1],
            tangent2[2],
            bitangent2[0],
            bitangent2[1],
            bitangent2[2],
            verts[2][0],
            verts[2][1],
            verts[2][2],
            uv[2][0],
            uv[2][1],
            normal[0],
            normal[1],
            normal[2],
            tangent2[0],
            tangent2[1],
            tangent2[2],
            bitangent2[0],
            bitangent2[1],
            bitangent2[2],
            verts[3][0],
            verts[3][1],
            verts[3][2],
            uv[3][0],
            uv[3][1],
            normal[0],
            normal[1],
            normal[2],
            tangent2[0],
            tangent2[1],
            tangent2[2],
            bitangent2[0],
            bitangent2[1],
            bitangent2[2],
        ],
        dtype=np.float32,
    )


quad_vert_tex_normal_tangent_bitangent = np.array(
    [
        -1,
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        -1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        -1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
    ],
    dtype=np.float32,
)
