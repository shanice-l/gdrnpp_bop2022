_base_ = "./03_can.py"
OUTPUT_DIR = "output/gdrn/tudlSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_tudl/dragon"
DATASETS = dict(TRAIN=("tudl_dragon_train_real",), TRAIN2=("tudl_dragon_train_pbr",))
