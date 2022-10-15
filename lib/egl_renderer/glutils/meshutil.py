"""3D mesh manipulation utilities."""

from builtins import str
from collections import OrderedDict
import numpy as np
from transforms3d import quaternions
from transforms3d.quaternions import axangle2quat, mat2quat
import os.path as osp
import pyassimp
import pprint
import hashlib
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
from lib.utils import logger
from lib.pysixd import inout


def get_vertices_extent(vertices):
    xmin, xmax = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
    ymin, ymax = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
    zmin, zmax = np.amin(vertices[:, 2]), np.amax(vertices[:, 2])

    xsize = xmax - xmin
    ysize = ymax - ymin
    zsize = zmax - zmin
    return xsize, ysize, zsize


def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum matrix."""
    assert right != left
    assert bottom != top
    assert znear != zfar

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    """Create perspective projection matrix.

    fovy: deg
    """
    assert znear != zfar
    fovy_rad = fovy / 180.0 * np.pi
    h = np.tan(fovy_rad / 2.0) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(anorm(v, axis=axis, keepdims=True), eps)


def lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt modelview matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    M = np.eye(4, dtype=np.float32)
    R = M[:3, :3]
    R[:] = [side, up, -forward]
    M[:3, 3] = -(R.dot(eye))
    return M


def sample_view(min_dist, max_dist=None):
    """Sample random camera position.

    Sample origin directed camera position in given distance range from
    the origin. ModelView matrix is returned.
    """
    if max_dist is None:
        max_dist = min_dist
    dist = np.random.uniform(min_dist, max_dist)
    eye = np.random.normal(size=3)
    eye = normalize(eye) * dist
    return lookat(eye)


def homotrans(M, p):
    p = np.asarray(p)
    if p.shape[-1] == M.shape[1] - 1:
        p = np.append(p, np.ones_like(p[..., :1]), -1)
    p = np.dot(p, M.T)
    return p[..., :-1] / p[..., -1:]


def _parse_vertex_tuple(s):
    """Parse vertex indices in '/' separated form (like 'i/j/k', 'i//k'.

    ...).
    """
    vt = [0, 0, 0]
    for i, c in enumerate(s.split("/")):
        if c:
            vt[i] = int(c)
    return tuple(vt)


def _unify_rows(a):
    """Unify lengths of each row of a."""
    lens = np.fromiter(map(len, a), np.int32)
    if not (lens[0] == lens).all():
        out = np.zeros((len(a), lens.max()), np.float32)
        for i, row in enumerate(a):
            out[i, : lens[i]] = row
    else:
        out = np.float32(a)
    return out


def loadTexture(path):
    from PIL import Image
    import OpenGL.GL as GL

    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    if img.mode != "RGB":
        # print('convert {} to RGB'.format(img.mode))
        img = img.convert("RGB")
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture_id = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)  # bind texture
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB,
        width,
        height,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        img_data,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture_id


def im2Texture(im, flip_v=False):
    from PIL import Image
    import OpenGL.GL as GL

    im_pil = Image.fromarray(im)
    if flip_v:
        im_pil = im_pil.transpose(Image.FLIP_TOP_BOTTOM)
    if im_pil.mode != "RGB":
        print("convert {} to RGB".format(im_pil.mode))
        im_pil = im_pil.convert("RGB")
    img_data = np.fromstring(im_pil.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = im_pil.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture_id = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)  # bind texture
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB,
        width,
        height,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        img_data,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture_id


def shader_from_path(shader_filename):
    shader_path = osp.join(cur_dir, "../shader", shader_filename)
    assert osp.exists(shader_path)
    with open(shader_path, "r") as f:
        return f.read()


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


def load_mesh_pyassimp(
    model_path,
    recalculate_normals=False,
    vertex_scale=1.0,
    is_textured=False,
    cad_model_color=None,
    use_cache=True,
    cache_dir=".cache",
    verbose=True,
):
    hashed_file_name = (
        hashlib.md5(
            (
                "{}_{}_{}_{}_{}_{}".format(
                    model_path,
                    "load_meshe_pyassimp",
                    recalculate_normals,
                    vertex_scale,
                    is_textured,
                    cad_model_color,
                )
            ).encode("utf-8")
        ).hexdigest()
        + "_load_mesh_pyassimp.pkl"
    )
    mmcv.mkdir_or_exist(cache_dir)
    cache_file = osp.join(cache_dir, hashed_file_name)
    if use_cache and osp.exists(cache_file):
        logger.info("{} loaded cache file: {}".format(model_path, cache_file))
        return mmcv.load(cache_file)
    scene = pyassimp.load(
        model_path
    )  # ,processing=pyassimp.postprocess.aiProcess_GenUVCoords | pyassimp.postprocess.aiProcess_Triangulate)  # load collada
    mesh = scene.meshes[0]
    # pprint(vars(mesh))
    print(mesh.__dict__.keys())

    # check materials
    mat = mesh.material
    # pprint(vars(mat))
    print(mat.__dict__.keys())
    # default values in pyassimp, ambient:0.05, diffuse: 0.6, specular: 0.6
    if "diffuse" in mat.properties.keys() and mat.properties["diffuse"] != 0:
        uMatDiffuse = np.array(mat.properties["diffuse"])[:3]
    else:
        uMatDiffuse = [0.8, 0.8, 0.8]
    if "specular" in mat.properties.keys() and mat.properties["specular"] != 0:
        uMatSpecular = np.array(mat.properties["specular"])[:3]
    else:
        uMatSpecular = [0.5, 0.5, 0.5]
    if "ambient" in mat.properties.keys() and mat.properties["ambient"] != 0:
        uMatAmbient = np.array(mat.properties["ambient"])[:3]  # phong shader
    else:
        uMatAmbient = [0, 0, 0]
    if "shininess" in mat.properties.keys() and mat.properties["shininess"] != 0:
        uMatShininess = max(mat.properties["shininess"], 1)  # avoid the 0 shininess
    else:
        uMatShininess = 1
    vertices = mesh.vertices * vertex_scale
    if recalculate_normals:
        normals = calc_normals(vertices)
    else:
        normals = mesh.normals
    if sum(normals.shape) == 0:
        normals = calc_normals(vertices)
    # import pdb; pdb.set_trace();
    result = dict(
        vertices=vertices,
        normals=normals,
        faces=mesh.faces,
        uMatDiffuse=uMatDiffuse,
        uMatSpecular=uMatSpecular,
        uMatAmbient=uMatAmbient,
        uMatShininess=uMatShininess,
    )
    if is_textured:
        result["colors"] = np.zeros((vertices.shape[0], 3), np.float32)
        if sum(mesh.texturecoords.shape) > 0:
            result["texturecoords"] = mesh.texturecoords[0, :, :2]
        else:
            logger.warn("can not load texturecoords with pyassimp")  # pyassimp does not load ply texture_uv
    else:
        result["texturecoords"] = np.zeros((vertices.shape[0], 2), np.float32)
        if sum(mesh.colors.shape) > 0:
            result["colors"] = mesh.colors[0, :, :3]
        else:
            if verbose:
                logger.warn("can not load colors with pyassimp. (ignore this if the model is textured)")

    if not is_textured and "colors" not in result:
        # no vert color and texture
        is_cad = True
        colors = np.zeros((vertices.shape[0], 3), np.float32)  # dummy colors
        if cad_model_color is None:
            colors[:, 0] = 223.0 / 255
            colors[:, 1] = 214.0 / 255
            colors[:, 2] = 205.0 / 255
        else:
            colors[:, 0] = cad_model_color[0] / 255
            colors[:, 1] = cad_model_color[1] / 255
            colors[:, 2] = cad_model_color[2] / 255
        result["colors"] = colors
    else:
        is_cad = False
    result["is_cad"] = is_cad

    pyassimp.release(scene)
    # if model_path.endswith('.obj'):
    #     ply_path = model_path.replace('.obj', '.ply')
    #     if osp.exists(ply_path):
    #         for key in ['uMatDiffuse', 'uMatSpecular', 'uMatAmbient', 'uMatShininess']:
    #             print('before: ', key, result[key])
    #         logger.info('assign light properties by loading {}'.format(ply_path))
    #         _res = load_mesh_pyassimp(ply_path)
    #         for key in ['uMatDiffuse', 'uMatSpecular', 'uMatAmbient', 'uMatShininess']:
    #             result[key] = _res[key]
    #         for key in ['uMatDiffuse', 'uMatSpecular', 'uMatAmbient', 'uMatShininess']:
    #             print('after: ', key, result[key])
    mmcv.dump(result, cache_file)
    return result


def load_mesh_sixd(
    model_path,
    recalculate_normals=False,
    vertex_scale=1.0,
    is_textured=False,
    render_uv=False,
    render_normalized_coords=False,
    render_nocs=False,
    model_info=None,
    cad_model_color=None,
    use_cache=True,
    cache_dir=".cache",
):
    mmcv.mkdir_or_exist(cache_dir)
    if model_path.endswith(".obj"):
        logger.warn(".obj file, load with pyassimp")
        return load_mesh_pyassimp(
            model_path,
            recalculate_normals=recalculate_normals,
            vertex_scale=vertex_scale,
            is_textured=is_textured,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
    #####################################
    hashed_file_name = (
        hashlib.md5(
            (
                "{}_{}_{}_{}_{}".format(
                    model_path,
                    "load_mesh_pysixd",
                    recalculate_normals,
                    vertex_scale,
                    is_textured,
                )
            ).encode("utf-8")
        ).hexdigest()
        + "_load_mesh_pysixd.pkl"
    )
    cache_file = osp.join(cache_dir, hashed_file_name)
    if use_cache and osp.exists(cache_file):
        logger.info("{} loaded cache file: {}".format(model_path, cache_file))
        return mmcv.load(cache_file)

    attributes = {}
    logger.info("loading {}".format(model_path))
    model = inout.load_ply(model_path)
    vertices = np.array(model["pts"]).astype(np.float32) * vertex_scale
    # import pdb; pdb.set_trace();
    num_pts = vertices.shape[0]
    if recalculate_normals or "normals" not in model:
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
        # assert model_info is not None
        normalizedCoords = np.copy(vertices)
        if model_info is None:
            xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
            ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
            zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
        else:
            xmin, xmax = model_info["xmin"], model_info["xmax"]
            ymin, ymax = model_info["ymin"], model_info["ymax"]
            zmin, zmax = model_info["zmin"], model_info["zmax"]
        # normalize every axis to [0, 1]
        normalizedCoords[:, 0] = (normalizedCoords[:, 0] - xmin) / (xmax - xmin)
        normalizedCoords[:, 1] = (normalizedCoords[:, 1] - ymin) / (ymax - ymin)
        normalizedCoords[:, 2] = (normalizedCoords[:, 2] - zmin) / (zmax - zmin)
        model["colors"] = normalizedCoords
    if render_nocs:  # diagnal normalized to 1, and min corner moved to (0,0,0)
        logger.info("render nocs")
        # Centering and scaling to fit the unit box
        nocs = np.copy(vertices)
        if model_info is None:
            xmin, xmax = nocs[:, 0].min(), nocs[:, 0].max()
            ymin, ymax = nocs[:, 1].min(), nocs[:, 1].max()
            zmin, zmax = nocs[:, 2].min(), nocs[:, 2].max()
            diagonal = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        else:
            xmin, xmax = model_info["xmin"], model_info["xmax"]
            ymin, ymax = model_info["ymin"], model_info["ymax"]
            zmin, zmax = model_info["zmin"], model_info["zmax"]
            diagonal = model_info["diagonal"]
        # # move (xmin, ymin, zmin) to origin, model centered at the 3b bbox center
        nocs[:, 0] -= xmin
        nocs[:, 1] -= ymin
        nocs[:, 2] -= zmin
        # scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin)
        # unit diagonal
        nocs /= diagonal
        model["colors"] = nocs

    faces = np.array(model["faces"]).astype(np.uint32)
    if "colors" in model:
        # NOTE: hack
        logger.info("colors max: {}".format(model["colors"].max()))
        if model["colors"].max() > 1.1:
            logger.info("make colors in [0, 1]")
            colors = np.array(model["colors"]).astype(np.float32) / 255.0
        else:  # in range [0, 1]
            colors = np.array(model["colors"]).astype(np.float32)
        attributes.update(vertices=vertices, normals=normals, colors=colors, faces=faces)
    else:
        attributes.update(vertices=vertices, normals=normals, faces=faces)
    if "texture_uv" in model and is_textured:
        attributes["texturecoords"] = model["texture_uv"]
        attributes["colors"] = np.zeros((vertices.shape[0], 3), np.float32)
    else:
        attributes["texturecoords"] = np.zeros((vertices.shape[0], 2), np.float32)

    if not is_textured and "colors" not in model:
        # no vert color and texture
        is_cad = True
        colors = np.zeros((vertices.shape[0], 3), np.float32)  # dummy colors
        if cad_model_color is None:
            colors[:, 0] = 223.0 / 255
            colors[:, 1] = 214.0 / 255
            colors[:, 2] = 205.0 / 255
        else:
            colors[:, 0] = cad_model_color[0] / 255
            colors[:, 1] = cad_model_color[1] / 255
            colors[:, 2] = cad_model_color[2] / 255
        attributes["colors"] = colors
    else:
        is_cad = False
    attributes["is_cad"] = is_cad

    result = load_mesh_pyassimp(
        model_path,
        recalculate_normals=recalculate_normals,
        vertex_scale=vertex_scale,
        is_textured=False,
        use_cache=False,
        verbose=False,
    )
    attributes.update(
        uMatDiffuse=result["uMatDiffuse"],
        uMatSpecular=result["uMatSpecular"],
        uMatAmbient=result["uMatAmbient"],
        uMatShininess=result["uMatShininess"],
    )
    mmcv.dump(attributes, cache_file)
    return attributes


def load_obj(fn):
    """Load 3d mesh form .obj' file.

    Args:
      fn: Input file name or file-like object.

    Returns:
      dictionary with the following keys (some of which may be missing):
        position: np.float32, (n, 3) array, vertex positions
        uv: np.float32, (n, 2) array, vertex uv coordinates
        normal: np.float32, (n, 3) array, vertex uv normals
        face: np.int32, (k*3,) traingular face indices
    """
    position = [np.zeros(3, dtype=np.float32)]
    normal = [np.zeros(3, dtype=np.float32)]
    uv = [np.zeros(2, dtype=np.float32)]

    tuple2idx = OrderedDict()
    trinagle_indices = []

    input_file = open(fn) if isinstance(fn, str) else fn
    for line in input_file:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        line = line.split(" ", 1)
        tag = line[0]
        if len(line) > 1:
            line = line[1]
        else:
            line = ""
        if tag == "v":
            position.append(np.fromstring(line, sep=" "))
        elif tag == "vt":
            uv.append(np.fromstring(line, sep=" "))
        elif tag == "vn":
            normal.append(np.fromstring(line, sep=" "))
        elif tag == "f":
            output_face_indices = []
            for chunk in line.split():
                # tuple order: pos_idx, uv_idx, normal_idx
                vt = _parse_vertex_tuple(chunk)
                if vt not in tuple2idx:  # create a new output vertex?
                    tuple2idx[vt] = len(tuple2idx)
                output_face_indices.append(tuple2idx[vt])
            # generate face triangles
            for i in range(1, len(output_face_indices) - 1):
                for vi in [0, i, i + 1]:
                    trinagle_indices.append(output_face_indices[vi])

    outputs = {}
    outputs["face"] = np.int32(trinagle_indices)
    pos_idx, uv_idx, normal_idx = np.int32(list(tuple2idx)).T
    if np.any(pos_idx):
        outputs["position"] = _unify_rows(position)[pos_idx]
    if np.any(uv_idx):
        outputs["uv"] = _unify_rows(uv)[uv_idx]
    if np.any(normal_idx):
        outputs["normal"] = _unify_rows(normal)[normal_idx]
    return outputs


def normalize_mesh(mesh):
    """Scale mesh to fit into -1..1 cube."""
    mesh = dict(mesh)
    pos = mesh["position"][:, :3].copy()
    pos -= (pos.max(0) + pos.min(0)) / 2.0
    pos /= np.abs(pos).max()
    mesh["position"] = pos
    return mesh


def quat2rotmat(quat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = quaternions.quat2mat(quat)
    return quat_mat


def mat2rotmat(mat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = mat
    return quat_mat


def xyz2mat(xyz):
    trans_mat = np.eye(4)
    trans_mat[-1, :3] = xyz
    return trans_mat


def mat2xyz(mat):
    xyz = mat[-1, :3]
    xyz[np.isnan(xyz)] = 0
    return xyz


def safemat2quat(mat):
    quat = np.array([1, 0, 0, 0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat
