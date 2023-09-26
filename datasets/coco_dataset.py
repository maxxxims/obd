import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils.utils import transform
from collections import defaultdict

class _objects:
    def __init__(self):
        self.boxes = []
        self.labels = []



class MSCoCoDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder: str, label_file: str, split: str):
        """
        data_folder/
        ....data/
        ....labels.json

        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.label_file = label_file

        assert self.label_file.split('.')[-1] == 'json'
        self.images = []
        object = defaultdict(_objects)

        with open(label_file, 'r') as j:
            labels_json = json.load(j)

        # read images paths
        for img in labels_json['images']:
            self.images.append(f'{data_folder}/data/{img["file_name"]}')

        for annotation in labels_json['annotations']:
            object[annotation['image_id']].boxes.append(annotation['bbox'])
            object[annotation['image_id']].labels.append(annotation['category_id'])
        self.objects = []
        for key in object.keys():
            self.objects.append(object[key].__dict__)

        del object
        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor([0] * len(objects['boxes']))  # (n_objects)


        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, 1 # tensor (N, 3, 300, 300), 3 lists of N tensors each
