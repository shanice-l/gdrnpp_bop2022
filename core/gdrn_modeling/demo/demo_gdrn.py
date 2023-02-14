# inference with detector, gdrn, and refiner
from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os

import cv2


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names


if __name__ == "__main__":
    image_paths = get_image_list("../../../datasets/BOP_DATASETS/lmo/test/000001/rgb", "../../../datasets/BOP_DATASETS/lmo/test/000001/depth")
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                       config_file_path="../../../configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test.py",
                       ckpt_file_path="../../../output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test/model_final.pth",
                       fuse=True,
                       fp16=False
                     )
    gdrn_predictor = GdrnPredictor(
        config_file_path="../../../configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py",
        ckpt_file_path="../../../output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_final.pth",
        camera_json_path="../../../datasets/BOP_DATASETS/lmo/camera.json",
        path_to_obj_models="../../../datasets/BOP_DATASETS/lmo/models"
    )

    for rgb_img, depth_img in image_paths:
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=depth_img)
        out_dict = gdrn_predictor.inference(data_dict)
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=rgb_img)

