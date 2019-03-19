
import os, re, io
import numpy as np
from PIL import Image

import utils as ut

import torch as th
from torch.utils.data import Dataset

class WiderTrain(Dataset):
    def __init__(self, img_dir, anno_path, transform=None, min_size=10):
        self.imgs_dir = img_dir
        self.anno_dict = self.__read_anno(anno_path, min_size)
        self.anno_keys = list(self.anno_dict.keys()) # the fname
        self.anno_values = list(self.anno_dict.values()) #it is the float box
        self.transform = transform
        self.imgs_list = [] # list of io.BytesIO object
        self.loaded = False

    def __read_anno(self, path, min_size):
        # annotation: float [ymin, xmin, ymax, xmax]
        annotation = dict()
        k = ''
        for line in open(path).readlines():
            line = line.rstrip()
            if re.match('.*.jpg', line):
                k = line
                annotation[k] = []
                continue
            elif len(line.split()) >= 4:
                box = np.array([float(x) for x in line.split()])
                box = box[[1, 0, 3, 2]]
                #box must larger le min_size
                if box[2] < 0: box[2] = -box[2]
                if box[3] < 0: box[3] = -box[3]
                mask = np.sqrt(box[2]*box[3]) > min_size
                if mask:
                    box[[2, 3]] = box[[0, 1]] + box[[2, 3]]
                    annotation[k].append(box.tolist())
        for k in list(annotation.keys()):
            if len(annotation[k]) == 0:
                del annotation[k]
            else:
                boxes = np.stack(annotation[k])
                annotation[k] = boxes
        return annotation

    def __len__(self):
        return len(self.anno_dict)

    def __getitem__(self, index):
        if not self.loaded:
            self.load()
        im = Image.open(self.imgs_list[index]).convert('RGB')

        boxes = self.anno_values[index]
        if self.transform is not None:
            im, boxes = self.transform(image=im, boxes=boxes)
        im = np.array(im).astype(np.float32)
        im = th.tensor(im)
        boxes = th.tensor(boxes).float()
        return {'image': im,
                'boxes': boxes}

    def load(self):
        print('WIDERFACE loading...')
        for fname in self.anno_keys:
            img_path = os.path.join(self.imgs_dir, fname)
            raw_data = open(img_path,'rb').read()
            raw_data = io.BytesIO(raw_data)
            self.imgs_list.append(raw_data)
        self.loaded = True
        print('WIDERFACE loaded.')

    def collate_fn(self, samples):
        imgs = th.stack([s['image'] for s in samples], dim=0)
        boxes_list = [s['boxes'] for s in samples]
        boxes_num = [s.shape[0] for s in boxes_list]
        max_num = max(boxes_num)
        boxes = [ut.pad_boxes(b, max_num) for b in boxes_list]
        boxes = th.stack(boxes, dim=0)
        return {'image': imgs,
                'boxes': boxes}
