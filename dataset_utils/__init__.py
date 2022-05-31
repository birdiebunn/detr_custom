from .coco import build_coco
import torch
import torchvision

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
