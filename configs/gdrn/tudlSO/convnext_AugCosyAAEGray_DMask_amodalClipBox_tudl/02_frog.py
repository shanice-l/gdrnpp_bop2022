_base_ = "./03_can.py"
OUTPUT_DIR = "output/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/frog"
DATASETS = dict(TRAIN=("tudl_frog_train_real",), TRAIN2=("tudl_frog_train_pbr",))
