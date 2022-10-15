#
set -e
set -u
set -x

EVAL_DIR=$HOME/PoseEst/cdpn_results/exp/three_steps_training/old_coor/CDPN_rot_trans/2019-04-05T11:06:11.127909/160/results_sixd_format/

python lib/pysixd/eval_sixd_results.py \
    --eval_dir $EVAL_DIR \
    --eval_types vsd \
    --dataset linemod \
    --n_top 1 \
    --vsd_delta 15 \
    --vsd_tau 20 \
    --vsd_cost step \
    --vsd_thres 0.3 \
    --cou_thres 0.5 \
    --re_thres 5.0 \
    --te_thres 5.0 \
    --add_factor 0.1 \
    --adi_factor 0.1 \
    --method cdpn \
    --calc_errors 1 \
    --eval_errors 1
