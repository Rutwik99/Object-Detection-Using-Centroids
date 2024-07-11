import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from scipy.ndimage import binary_erosion

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # CSE 527
        # Compute centroid for each object (i.e, one centroid for one annotation id)
        # target['centroid'] should be filled with tensor containing centroids with
        # shape ( num_annos x 2)
        ##############################################################################
        segmentations = [obj["segmentation"] for obj in anno]
        # print("aa", convert_coco_poly_to_centroids(segmentations, h, w, boxes))
        target["centroids"] = convert_coco_poly_to_centroids(segmentations, h, w, boxes)[keep]
        # print('ConvertCocoPolysToMask', keep, len(target['labels']), len(target['boxes']), len(target['centroids']))
        ##############################################################################
        return image, target


################## TODO CSE527 STUDENT CODE STARTS HERE ################################################################
# Read implementation of ConvertCocoPolysToMask and understand the input and outputs
# Complete the convert_coco_poly_to_centroids() method
# refer: https://cocodataset.org/#format-data and COCO API : https://github.com/cocodataset/cocoapi/tree/master


def convert_coco_poly_to_centroids(segmentations, height, width, bboxes):
    """Converts COCO polygon annotations to centroids.
    Args:
        segmentations: A list of COCO polygon annotations.
        height: The height of the image.
        width: The width of the image.

    Returns:
        A PyTorch tensor containing the centroids of the objects in the image.
    """
    masks = convert_coco_poly_to_mask(segmentations, height, width)
    centroids = []
    for mask in masks:
        kernel = np.ones((8, 8))
        new_mask = binary_erosion(mask, kernel, 3)

        if (new_mask == 1).sum() == 0:
            y_centroid, x_centroid = np.argwhere(mask.numpy() == 1).sum(0) / (mask.numpy() == 1).sum()
        else:
            y_centroid, x_centroid = np.argwhere(new_mask == 1).sum(0) / (new_mask == 1).sum()

        centroids.append([x_centroid, y_centroid])
    return torch.tensor(centroids).float()

################## STUDENT CODE ENDS HERE #######################################################################

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # TODO CSE527: PUT BACK TRANSFORMS (students do not need to add the transofrms)
    if image_set == 'train':
        return T.Compose([normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
