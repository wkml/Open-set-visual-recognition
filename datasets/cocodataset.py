import os
import sys
import json
import numpy as np
from PIL import Image
import torch.utils.data as data

from pycocotools.coco import COCO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class COCO2014(data.Dataset):
    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform

        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())

        with open('./data/coco/category.json', 'r') as load_category:
            self.category_map = json.load(load_category)

        # labels : numpy.ndarray, shape->(len(coco), 80)
        # value range->(0 means label don't exist, 1 means label exist)
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
        self.labels = np.array(self.labels)
        self.labels[self.labels < 1] = 0

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)

        return input, self.labels[index]

    def __len__(self):
        return len(self.ids)


# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)


def getLabelVector(categories, category_map):
    label = np.zeros(80)
    label.fill(-1)
    for c in categories:
        label[category_map[str(c)] - 1] = 1.0
    return label



