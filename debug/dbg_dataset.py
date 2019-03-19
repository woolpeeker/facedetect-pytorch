
import json
import random
import matplotlib.pyplot as plt
import utils as ut
plt.interactive(False)

import numpy as np

import torch as th
from torch.utils.data import DataLoader

from dataset.widerface import WiderTrain
from dataset import transform

cfg = json.load(open('config.json'))

np.random.seed(0)
random.seed(0)
th.random.manual_seed(0)


def trans_fn(sample):
    image, boxes =sample['image'], sample['boxes']
    sel_box = random.choice(boxes)
    to_size = random.randrange(12, 32)
    img, boxes = transform.select_crop_face(image, boxes, cfg['train']['image_shape'], sel_box, to_size)
    return img, boxes

wider_train_dataset = WiderTrain(img_dir=cfg['wider_train']['image_dir'],
                                 anno_path=cfg['wider_train']['txt_path'],
                                 transform=trans_fn)
dataloader = DataLoader(wider_train_dataset, batch_size=2, collate_fn=wider_train_dataset.collate_fn)

def debug_train_dataset():
    for i, sample in enumerate(dataloader):
        imgs = sample['image']
        boxes_list =sample['boxes']
        for i in range(imgs.size(0)):
            img, boxes = imgs[i], boxes_list[i]
            img = ut.draw_boxes(img, boxes_list[i])
            plt.imshow(img)
            plt.show()
            pass

def debug_encode_and_decode():
    from model.module import MultiAnchor
    image_shape = (cfg['train']['image_shape'],cfg['train']['image_shape'])
    anchor_obj = MultiAnchor(image_shape, (8,8,8), (16, 24, 36))

    for sample_idx, sample in enumerate(dataloader):
        imgs = sample['image']
        boxes = sample['boxes']
        iou_list, offset_list = anchor_obj.encode(boxes)
        label_list = [iou>0.5 for iou in iou_list]
        decode_boxes_list = anchor_obj.decode(label_list, offset_list)
        for i in range(len(imgs)):
            img = imgs[i]
            decode_boxes = decode_boxes_list[i].numpy()
            img = ut.draw_boxes(img, decode_boxes)
            plt.imshow(img)
            plt.show()

if __name__ == '__main__':
    debug_encode_and_decode()