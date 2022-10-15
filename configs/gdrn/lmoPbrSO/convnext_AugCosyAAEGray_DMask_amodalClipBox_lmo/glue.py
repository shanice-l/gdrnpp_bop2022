_base_ = "./ape.py"
OUTPUT_DIR = "output/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/glue"
DATASETS = dict(TRAIN=("lmo_glue_train_pbr",), TEST=("lmo_bop_test",))
