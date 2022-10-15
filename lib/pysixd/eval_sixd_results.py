import os.path as osp
import sys
import argparse

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, "../.."))

from lib.pysixd.eval_calc_errors import eval_calc_errors
from lib.pysixd.eval_loc import match_and_eval_performance_scores
from lib.pysixd import eval_plots, latex_report
from lib.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Eval sixd results")
    parser.add_argument("--eval_dir", help="root dir of sixd eval results")
    parser.add_argument(
        "--eval_types",
        type=str,
        nargs="+",
        choices=["vsd", "add", "adi", "re", "te", "cou"],
        help="eval types",
    )
    parser.add_argument(
        "--dataset",
        help="dataset name: {hinterstoisser/linemod, tless, tudlight, rutgers, tejani, doumanoglou}",
    )
    parser.add_argument(
        "--cam_type",
        default="",
        help="cam_type: primesense | canon, for linemod, it is null",
    )
    parser.add_argument(
        "--n_top",
        type=int,
        default=1,
        help="Top N pose estimates (with the highest score) to be evaluated for each object in each image. "
        "0 = all estimates, -1 = given by the number of GT poses",
    )
    parser.add_argument(
        "--calc_errors",
        type=int,
        default=1,
        help="whether to calculate errors",
    )
    parser.add_argument("--eval_errors", type=int, default=1, help="whether to evaluate errors")
    # vsd parameters
    parser.add_argument(
        "--vsd_delta",
        default=15,
        type=float,
        help="Tolerance used for estimation of the visibility masks.",
    )
    parser.add_argument("--vsd_tau", default=20, type=float, help="Misalignment tolerance.")
    parser.add_argument(
        "--vsd_cost",
        default="step",
        help="step | tlinear. Pixel-wise matching cost: "
        ' "step": Used for SIXD Challenge 2017. It is easier to interpret.'
        ' "tlinear": Used in the original definition of VSD in:'
        " Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016",
    )
    # thresholds
    parser.add_argument("--vsd_thres", default=0.3, type=float, help="error threshold of vsd")
    parser.add_argument("--cou_thres", default=0.5, type=float, help="threshold of cou")
    parser.add_argument("--re_thres", default=5.0, type=float, help="threshold of re, in deg")
    parser.add_argument("--te_thres", default=5.0, type=float, help="threshold of te, in cm")
    parser.add_argument("--add_factor", default=0.1, type=float, help="threshold factor of add")
    parser.add_argument("--adi_factor", default=0.1, type=float, help="threshold factor of adi")
    parser.add_argument("--method", default="cdpn", help="method name")
    parser.add_argument("--image_subset", default="bb8", help="test set: bb8 | sixd_v1 | None")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.set_logger_dir(args.eval_dir, action="k")
    if args.calc_errors:
        eval_calc_errors(
            args.eval_dir,
            args.eval_types,
            dataset=args.dataset,
            cam_type=args.cam_type,
            n_top=args.n_top,
            vsd_delta=args.vsd_delta,
            vsd_tau=args.vsd_tau,
            vsd_cost=args.vsd_cost,
            method=args.method,
        )
    if args.eval_errors:
        match_and_eval_performance_scores(
            args.eval_dir,
            error_types=args.eval_types,
            error_thresh={
                "vsd": args.vsd_thres,
                "cou": args.cou_thres,
                "te": args.te_thres,  # cm
                "re": args.re_thres,  # deg
            },
            error_thresh_fact={"add": args.add_factor, "adi": args.adi_factor},
            dataset=args.dataset,
            cam_type=args.cam_type,
            n_top=args.n_top,
            vsd_delta=args.vsd_delta,
            vsd_tau=args.vsd_tau,
            vsd_cost=args.vsd_cost,
            method=args.method,
            image_subset=args.image_subset,
        )
        if args.dataset == "linemod":
            scene_ids = [i for i in range(1, 15 + 1) if i not in [3, 7]]
            obj_ids = [i for i in range(1, 15 + 1) if i not in [3, 7]]
        for obj_id in obj_ids:
            # AUC_vsd
            eval_plots.plot_vsd_err_hist(
                eval_dir=args.eval_dir,
                scene_ids=scene_ids,
                obj_id=obj_id,
                dataset_name=args.dataset,
                top_n=args.n_top,
                delta=args.vsd_delta,
                tau=args.vsd_tau,
                cost=args.vsd_cost,
                cam_type=args.cam_type,
            )
        logger.info("generating latex report")
        report = latex_report.Report(eval_dir=args.eval_dir, log_dir=args.eval_dir)
        # report.write_configuration(train_cfg_file_path, eval_cfg_file_path)
        report.merge_all_tex_files()
        report.include_all_figures()
        logger.info("save report pdf")
        report.save(open_pdf=True)
        logger.info("done")


if __name__ == "__main__":
    main()
