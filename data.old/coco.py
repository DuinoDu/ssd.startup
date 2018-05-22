# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2


# https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py

CATEGORY_ID = []

class CocoDetection(data.Dataset):

    def __init__(self, coco_root, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.coco_root = coco_root

        train = ['train2014', 'annotations/instances_train2014.json']
        val = ['val2014', 'annotations/instances_val2014.json']

        self.coco1 = COCO(os.path.join(coco_root, train[1]))
        self.coco2 = COCO(os.path.join(coco_root, val[1]))

        self.imgpath1 = os.path.join(coco_root, train[0])
        self.imgpath2 = os.path.join(coco_root, val[0])

        #self.ids = list(self.coco.imgs.keys())
        self.ids1 = list(set([x['image_id'] for x in self.coco1.anns.values()]))
        self.ids2 = list(set([x['image_id'] for x in self.coco2.anns.values()]))[:35000]

        self.transform = transform
        self.target_transform = target_transform

        global CATEGORY_ID
        CATEGORY_ID = sorted([cat['id'] for cat in self.coco1.loadCats(self.coco1.getCatIds())])

    def __getitem__(self, index):
        if index < len(self.ids1):
            coco = self.coco1
            img_id = self.ids1[index]
            root = self.imgpath1
        else:
            coco = self.coco2
            img_id = self.ids2[index-len(self.ids1)]
            root = self.imgpath2
        ann_ids = coco.getAnnIds(imgIds=img_id)

        try:
            assert len(ann_ids) > 0
        except Exception as e:
            print("non ann_ids:", img_id)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(root, path))
        img = img[:,:,(2,1,0)] # BGR to RGB
        height, width, channels = img.shape
        target = coco.loadAnns(ann_ids)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            img, bbox, label = self.transform(img, target[:,:4], target[:,4])
            target = np.concatenate((bbox, np.expand_dims(label, axis=1)), 1)

        return img, target

    def __len__(self):
        return len(self.ids1) + len(self.ids2)


class AnnotationTransform(object):
    def __call__(self, target, width, height):
        """Convert coco target to boundingbox target.

        :target: (dict_in list), coco_json format, [x, y, w, h]
        :returns:
            res: (np.ndarray), [len(num_objs, 5)], [xmin, ymin, xmax, ymax, label]

        """
        res = np.zeros((len(target), 5)).astype(float)
        for ind, obj in enumerate(target):
            res[ind, 0] = max(0, obj['bbox'][0] - obj['bbox'][2]/2)
            res[ind, 1] = max(0, obj['bbox'][1] - obj['bbox'][3]/2)
            res[ind, 2] = min(width, obj['bbox'][0] + obj['bbox'][2]/2)
            res[ind, 3] = min(height, obj['bbox'][1] + obj['bbox'][3]/2)
            res[ind, 4] = CATEGORY_ID.index(obj['category_id'])
        res[:,0] /= width
        res[:,1] /= height
        res[:,2] /= width
        res[:,3] /= height
        return res


def detection_collate(batch):
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def make_dataloader(root, augment,
                    batch_size, num_workers, shuffle=True, pin_memory=True):
    dataset = CocoDetection(root, augment, AnnotationTransform())
    return data.DataLoader(dataset, batch_size, num_workers=num_workers,
                           shuffle=shuffle, collate_fn=detection_collate, pin_memory=pin_memory)


if __name__ == "__main__":

    coco_root = os.environ['HOME'] + '/data/coco'
    augment = None

    dataset = CocoDetection(coco_root, augment, AnnotationTransform())
    print(len(dataset))
    for i in range(len(dataset)):
        x = dataset[i]
        print("{}/{}".format(i, len(dataset)))
