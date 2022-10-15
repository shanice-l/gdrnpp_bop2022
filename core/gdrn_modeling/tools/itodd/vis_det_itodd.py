from PIL import Image, ImageDraw
from skimage import io
import mmcv
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))


json_det = mmcv.load("datasets/BOP_DATASETS/itodd/challenge2022-524061_itodd-test_bop.json")
for id, list in json_det.items():
    image_path = "datasets/BOP_DATASETS/itodd/test/000001/gray/" + "{:06d}.tif".format(int(id[2:]))
    image = io.imread(image_path)
    # image = Image.open(image_path)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for bbox in list:
        box = bbox["bbox_est"]
        obj_id = bbox["obj_id"]
        # 坐标参数依次是左上角、右上角、右下角、左下角，outline里面是RGB参数：红、绿、蓝
        draw.polygon(
            [
                (box[0], box[1]),
                (box[0] + box[2], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (box[0], box[1] + box[3]),
            ],
            outline=255,
        )
        image.save("itodd_vis_det/" + id[2:] + "_" + obj_id + ".png")
