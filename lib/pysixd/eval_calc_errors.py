# from https://raw.githubusercontent.com/DLR-RM/AugmentedAutoencoder/master/sixd_toolkit_extensions/eval_calc_errors.py
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
# --------------------------
# modified

# NOTE: Calculates error of 6D object pose estimates.
import os
import os.path as osp
import sys
import glob

# import time

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from lib.pysixd import inout, pose_error, misc, pose_error_more
from lib.pysixd.dataset_params import get_dataset_params
from lib.utils import logger
from lib.meshrenderer import meshrenderer_color


# def eval_calc_errors(eval_args, eval_dir):
def eval_calc_errors(
    eval_dir,
    error_types=["vsd", "re", "te"],
    dataset="linemod",
    cam_type="primesense",
    n_top=1,
    vsd_delta=15,
    vsd_tau=20,
    vsd_cost="step",
    method="",
):
    """
    eval_dir:
    error_types: ['vsd', 'adi', 'add', 'cou', 're', 'te']
    dataset:  {hinterstoisser/linemod, tless, tudlight, rutgers, tejani, doumanoglou}
    cam_type: 'primesense'
    n_top: Top N pose estimates (with the highest score) to be evaluated for each object in each image.
            0 = all estimates, -1 = given by the number of GT poses

    vsd_delta: Tolerance used for estimation of the visibility masks.
    vsd_tau: Misalignment tolerance. mm
    vsd_cost: Pixel-wise matching cost:
            "step": Used for SIXD Challenge 2017. It is easier to interpret.
            'tlinear': Used in the original definition of VSD in:
                    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
    method: algorithm name
    --------------
    """
    # Results for which the errors will be calculated
    # -------------------------------------------------------------------------------
    # result_base = '/path/to/results/'
    # result_paths = [
    #     osp.join(result_base, 'hodan-iros15_hinterstoisser'),
    #     # osp.join(result_base, 'hodan-iros15_tless_primesense'),
    # ]
    if dataset in ["linemod", "hinterstoisser"]:
        test_type = ""
    else:
        test_type = cam_type

    result_path = eval_dir
    logger.info("Processing: {}".format(result_path))
    # Other paths
    # -------------------------------------------------------------------------------
    for error_type in error_types:
        # Mask of path to the output file with calculated errors
        # errors_mpath = osp.join('{result_path}', '..', '..', 'eval', '{result_name}',
        #                      '{error_sign}', 'errors_{scene_id:02d}.yml')
        errors_mpath = "{result_path}/{error_sign}/errors_{scene_id:02d}.yml"

        error_sign = "error:" + error_type + "_ntop:" + str(n_top)  # Error signature
        if error_type == "vsd":
            error_sign += "_delta:{}_tau:{}_cost:{}".format(vsd_delta, vsd_tau, vsd_cost)
        # Error calculation
        # -------------------------------------------------------------------------------
        # Select data type
        if dataset == "tless":
            cam_type = test_type
            if error_type in ["adi", "add"]:
                model_type = "cad_subdivided"
            else:
                model_type = "cad"
        else:
            model_type = ""
            cam_type = ""
        # Load dataset parameters
        dp = get_dataset_params(
            dataset,
            model_type=model_type,
            test_type=test_type,
            cam_type=cam_type,
        )
        # Load object models
        if error_type in ["vsd", "add", "adi", "cou", "proj", "projamb"]:
            logger.info("Loading object models...")
            models_cad_files = []
            models = {}
            for obj_id in range(1, dp["obj_count"] + 1):
                models[obj_id] = inout.load_ply(dp["model_mpath"].format(obj_id))
                models_cad_files.append(dp["model_mpath"].format(obj_id))

            logger.info("Initializing renderer")  # the models are in mm
            # TODO: different renderer should be used when using cad models
            renderer = meshrenderer_color.Renderer(
                models_cad_files,
                K=None,
                samples=1,
                vertex_tmp_store_folder=".",
                vertex_scale=1.0,
                height=480,
                width=640,
                near=100,
                far=10000,
                use_cache=True,
            )

        # test_sensor = pjoin(dp["base_path"], dp["test_dir"])
        # Directories with results for individual scenes
        scene_dirs = sorted(
            [d for d in glob.glob(osp.join(result_path, "*")) if osp.isdir(d) and osp.basename(d).isdigit()]
        )
        logger.info("scene dirs: {}".format(scene_dirs))
        for scene_dir in scene_dirs:
            scene_id = int(osp.basename(scene_dir))

            # Load info and GT poses for the current scene
            scene_info = inout.load_info(dp["scene_info_mpath"].format(scene_id))
            scene_gt = inout.load_gt(dp["scene_gt_mpath"].format(scene_id))

            res_paths = sorted(glob.glob(osp.join(scene_dir, "*.yml")))  # NOTE: maybe using test idx file

            errs = []
            im_id = -1
            depth_im = None
            for res_id, res_path in enumerate(res_paths):
                # t = time.perf_counter()
                # Parse image ID and object ID from the filename
                filename = osp.basename(res_path).split(".")[0]
                im_id_prev = im_id
                im_id, obj_id = map(int, filename.split("_"))

                if res_id % 10 == 0:
                    dataset_str = dataset
                    if test_type != "":
                        dataset_str += " - {}".format(test_type)
                    logger.info(
                        "Calculating error: {}, {}, {}, scene_id:{}, img_id:{}, obj_id:{}".format(
                            error_type,
                            method,
                            dataset_str,
                            scene_id,
                            im_id,
                            obj_id,
                        )
                    )

                # Load depth image if VSD is selected
                if error_type == "vsd" and im_id != im_id_prev:
                    depth_path = dp["test_depth_mpath"].format(scene_id, im_id)
                    # depth_im = inout.load_depth(depth_path)
                    depth_im = inout.load_depth2(depth_path)  # Faster
                    depth_im *= dp["cam"]["depth_scale"]  # to [mm]

                # Load camera matrix
                if error_type in ["vsd", "cou", "proj", "projamb"]:
                    K = scene_info[im_id]["cam_K"]

                # Load pose estimates
                res = inout.load_results_sixd17(res_path)
                ests = res["ests"]

                # Sort the estimates by score (in descending order)
                ests_sorted = sorted(enumerate(ests), key=lambda x: x[1]["score"], reverse=True)

                # Select the required number of top estimated poses
                if n_top == 0:  # All estimates are considered
                    n_top_curr = None
                elif n_top == -1:  # Given by the number of GT poses
                    n_gt = sum([gt["obj_id"] == obj_id for gt in scene_gt[im_id]])
                    n_top_curr = n_gt
                else:
                    n_top_curr = n_top
                ests_sorted = ests_sorted[slice(0, n_top_curr)]

                for est_id, est in ests_sorted:
                    # est_errs = []
                    R_e = est["R"]
                    t_e = est["t"]

                    errs_gts = {}  # Errors w.r.t. GT poses of the same object
                    for gt_id, gt in enumerate(scene_gt[im_id]):
                        if gt["obj_id"] != obj_id:
                            continue

                        e = -1.0
                        R_g = gt["cam_R_m2c"]
                        t_g = gt["cam_t_m2c"]

                        if error_type == "vsd":
                            e = pose_error_more.vsd(
                                R_e,
                                t_e,
                                R_g,
                                t_g,
                                models[obj_id],
                                depth_im,
                                K,
                                vsd_delta,
                                vsd_tau,
                                vsd_cost,
                                renderer=renderer,
                            )
                        elif error_type == "add":
                            e = pose_error.add(R_e, t_e, R_g, t_g, models[obj_id])
                        elif error_type == "adi":
                            e = pose_error.adi(R_e, t_e, R_g, t_g, models[obj_id])
                        elif error_type == "proj":
                            e = pose_error.arp_2d(R_e, t_e, R_g, t_g, models[obj_id], K)
                        elif error_type == "projamb":  # NOTE: seems not available
                            e = pose_error.arpi_2d(R_e, t_e, R_g, t_g, models[obj_id], K)
                        elif error_type == "cou":
                            e = pose_error_more.cou(
                                R_e,
                                t_e,
                                R_g,
                                t_g,
                                models[obj_id],
                                dp["test_im_size"],
                                K,
                                renderer=renderer,
                            )
                        elif error_type == "re":
                            e = pose_error.re(R_e, R_g)
                        elif error_type == "te":
                            e = pose_error.te(t_e, t_g)

                        errs_gts[gt_id] = e

                    errs.append(
                        {
                            "im_id": im_id,
                            "obj_id": obj_id,
                            "est_id": est_id,
                            "score": est["score"],
                            "errors": errs_gts,
                        }
                    )
                # print('Evaluation time: {}s'.format(time.perf_counter() - t))

            logger.info("Saving errors...")
            errors_path = errors_mpath.format(
                result_path=result_path,
                error_sign=error_sign,
                scene_id=scene_id,
            )

            misc.ensure_dir(os.path.dirname(errors_path))
            inout.save_errors(errors_path, errs)

            logger.info("")
    logger.info("Done.")
    return True


if __name__ == "__main__":
    # from lib.pysixd import renderer
    import numpy as np
    from lib.transforms3d.axangles import axangle2mat
    import matplotlib.pyplot as plt
    from lib.vis_utils.image import grid_show

    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # linemod
    models_cad_files = [
        osp.join(
            cur_dir,
            "../../data/SIXD_DATASETS/LM6d_origin/models/obj_{:02d}.ply".format(obj_id),
        )
        for obj_id in range(1, 15 + 1)
    ]
    renderer = meshrenderer_color.Renderer(
        models_cad_files,
        K,
        samples=1,
        vertex_tmp_store_folder=".",
        vertex_scale=1.0,
        height=480,
        width=640,
        near=100,
        far=10000,
        use_cache=True,
        model_infos=None,
    )
    models = {}
    for obj_id in range(1, 15 + 1):
        models[obj_id] = inout.load_ply(
            osp.join(
                cur_dir,
                "../../data/SIXD_DATASETS/LM6d_origin/models/obj_{:02d}.ply".format(obj_id),
            )
        )

    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([0, 0, 0.7], dtype=np.float32)
    R_est = R
    t_est = t * 1000  # mm
    # depth_est = renderer.render(models[1], (640, 480), K, R_est, t_est, clip_near=100,
    #                             clip_far=10000, mode='depth')

    bgr, depth_est = renderer.render(0, R_est, t_est, K=None, W=640, H=480, near=100, far=10000, to_255=True)
    print(depth_est.min(), depth_est.max())

    #
    grid_show([depth_est / 1000], ["depth_est_glumpy"], row=1, col=1)
