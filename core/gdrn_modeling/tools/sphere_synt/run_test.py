import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description="run sphere synt test")
parser.add_argument(
    "--cfg",
    default="configs/gdrn_sphere_synt/a6_cPnP_sphere.py",
    help="cfg path",
)
parser.add_argument(
    "--ckpt",
    default="output/gdrn_sphere_synt/a6_cPnP_sphere/model_final.pth",
    help="ckpt path",
)
# parser.add_argument("--noise_sigma", default=0.0, type=float, 'noise sigma')
# parser.add_argument("--outlier", default=0.1, type=float, 'outlier ratio')
# parser.add_argument("--use_pnp", default=False, action="store_true")
args = parser.parse_args()

print(args.cfg)
print(args.ckpt)

for outlier in [0.1, 0.3]:
    # for outlier in [0.3]:
    for noise_level in range(0, 31):
        noise_sigma = noise_level * 0.002  # [0, 0.06]
        for use_pnp in [True, False]:
            print(
                "outlier: ",
                outlier,
                "noise sigma: ",
                noise_sigma,
                "use_pnp:",
                use_pnp,
            )
            cmd = "./core/gdrn_sphere_synt/test_gdrn_sphere_synt.sh {} 0 {} INPUT.XYZ_NOISE_SIGMA_TEST={} TEST.USE_PNP={} INPUT.MIN_XYZ_OUTLIER_TEST={} INPUT.MAX_XYZ_OUTLIER_TEST={}".format(
                args.cfg, args.ckpt, noise_sigma, use_pnp, outlier, outlier
            )
            print(cmd)
            os.system(cmd)
