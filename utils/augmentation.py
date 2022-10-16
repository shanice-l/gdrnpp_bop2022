import os.path
import random
from copy import deepcopy

import numpy as np
import PIL
import torch
from crops import crop_to_aspect_ratio, get_K_crop_resize
from PIL import ImageEnhance, ImageFilter
from torchvision.datasets import ImageFolder

F = torch.nn.functional


def to_pil(im):
    if isinstance(im, PIL.Image.Image):
        return im
    elif isinstance(im, torch.Tensor):
        return PIL.Image.fromarray(np.asarray(im))
    elif isinstance(im, np.ndarray):
        return PIL.Image.fromarray(im)
    else:
        raise ValueError('Type not supported', type(im))


def to_torch_uint8(im):
    if isinstance(im, PIL.Image.Image):
        im = torch.as_tensor(np.asarray(im).astype(np.uint8))
    elif isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        im = torch.as_tensor(im)
    else:
        raise ValueError('Type not supported', type(im))
    if im.dim() == 3:
        assert im.shape[-1] in {1, 3}
    return im


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im):
        im = to_pil(im)
        k = random.randint(*self.factor_interval)
        im = im.filter(ImageFilter.GaussianBlur(k))
        return im


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im):
        im = to_pil(im)
        if random.random() <= self.p:
            im = self._pillow_fn(im).enhance(factor=random.uniform(*self.factor_interval))
        return im


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0., 50.)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval)


class BackgroundAugmentation:
    def __init__(self, image_dataset, p):
        self.image_dataset = image_dataset
        self.p = p

    def get_bg_image(self, idx):
        return self.image_dataset[idx]

    def __call__(self, im, mask):
        if random.random() <= self.p:
            im = to_torch_uint8(im)
            mask = to_torch_uint8(mask)
            h, w, c = im.shape
            im_bg = self.get_bg_image(random.randint(0, len(self.image_dataset) - 1))
            im_bg = to_pil(im_bg)
            im_bg = torch.as_tensor(np.copy(np.asarray(im_bg.resize((w, h)))))
            mask_bg = mask == 0
            im[mask_bg] = im_bg[mask_bg]
        return im


class VOCBackgroundAugmentation(BackgroundAugmentation):
    def __init__(self, voc_root, p=0.3):
        image_dataset = ImageFolder(voc_root)
        super().__init__(image_dataset=image_dataset, p=p)

    def get_bg_image(self, idx):
        return self.image_dataset[idx][0]

def make_detections_from_segmentation(masks):
    detections = []
    if masks.dim() == 4:
        assert masks.shape[0] == 1
        masks = masks.squeeze(0)

    for mask_n in masks:
        dets_n = dict()
        for uniq in torch.unique(mask_n, sorted=True):
            ids = np.where((mask_n == uniq).cpu().numpy())
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            dets_n[int(uniq.item())] = torch.tensor([x1, y1, x2, y2]).to(mask_n.device)
        detections.append(dets_n)
    return detections


class CropResizeToAspectAugmentation:
    def __init__(self, resize=(480, 640)):
        self.resize = (min(resize), max(resize))
        self.aspect = max(resize) / min(resize)

    def __call__(self, im, mask, depth, obs):
        im = to_torch_uint8(im)
        mask = to_torch_uint8(mask)
        depth = torch.from_numpy(depth)
        obs['orig_camera'] = deepcopy(obs['camera'])
        assert im.shape[-1] == 3
        h, w = im.shape[:2]
        if (h, w) == self.resize:
            obs['orig_camera']['crop_resize_bbox'] = (0, 0, w-1, h-1)
            return im, mask, depth.numpy(), obs

        images = (torch.as_tensor(im).float() / 255).unsqueeze(0).permute(0, 3, 1, 2)
        masks = torch.as_tensor(mask).unsqueeze(0).unsqueeze(0).float()
        depth = depth.unsqueeze(0).unsqueeze(0)
        K = torch.tensor(obs['camera']['K']).unsqueeze(0)

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w/h, self.aspect):
            x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
            w = images.shape[-1]
            r = self.aspect
            h = w * 1/r
            box_size = (h, w)
            h, w = min(box_size), max(box_size)
            x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
            box = torch.tensor([x1, y1, x2, y2])
            images, masks, K, depth = crop_to_aspect_ratio(images, box, masks=masks, K=K, depths=depth)

        # Resize to target size
        x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
        h_input, w_input = images.shape[-2], images.shape[-1]
        h_output, w_output = min(self.resize), max(self.resize)
        box_size = (h_input, w_input)
        h, w = min(box_size), max(box_size)
        x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
        box = torch.tensor([x1, y1, x2, y2])
        images = F.interpolate(images, size=(h_output, w_output), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(h_output, w_output), mode='bilinear', align_corners=False)
        masks = F.interpolate(masks, size=(h_output, w_output), mode='nearest')
        obs['orig_camera']['crop_resize_bbox'] = tuple(box.tolist())
        K = get_K_crop_resize(K, box.unsqueeze(0), orig_size=(h_input, w_input), crop_resize=(h_output, w_output))

        # Update the bounding box annotations
        dets_gt = make_detections_from_segmentation(masks)[0]
        for n, obj in enumerate(obs['objects']):
            if 'bbox' in obj and obj['id_in_segm'] in dets_gt:
                obj['bbox'] = dets_gt[obj['id_in_segm']].numpy()

        im = (images[0].permute(1, 2, 0) * 255).to(torch.uint8)
        mask = masks[0, 0].to(torch.uint8)
        depth = depth[0, 0].numpy()
        obs['camera']['K'] = K.squeeze(0).numpy()
        obs['camera']['resolution'] = (h_output, w_output)
        return im, mask, depth, obs
