
import os, json, random

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from dataset.widerface import WiderTrain
from dataset import transform

from model.detector import Detector
import utils as ut

cfg_path = 'config.json'

np.random.seed(0)
random.seed(0)
th.random.manual_seed(0)

def train():
    cfg = json.load(open(cfg_path))
    enable_cuda = cfg['train']['enable_cuda']
    gpus = cfg['train']['GPUS']
    device = 'cuda:%d'%gpus[0] if enable_cuda else 'cpu'
    enable_multi_gpus = enable_cuda and len(gpus) > 1

    batch_size = cfg['train']['batch_size']
    batch_size = len(gpus) * batch_size if enable_multi_gpus else batch_size

    def transform_fn(image, boxes):
        sel_box_idx = random.randrange(0, len(boxes))
        to_size = random.choice(cfg['network']['anchor_sizes'])
        to_size = random.randint(round(to_size*0.7), round(to_size*1.3))
        img, boxes = transform.select_crop_face(image, boxes, cfg['train']['image_shape'], sel_box_idx, to_size)
        return img, boxes

    wider_train_dataset = WiderTrain(img_dir=cfg['wider_train']['image_dir'],
                                     anno_path=cfg['wider_train']['txt_path'],
                                     transform=transform_fn)
    dataloader = DataLoader(wider_train_dataset,
                            batch_size=batch_size,
                            collate_fn=wider_train_dataset.collate_fn,
                            num_workers=12, drop_last=True, shuffle=True)

    print('model is being building...')
    detector = Detector(cfg=cfg)
    if enable_multi_gpus:
        detector = th.nn.DataParallel(detector, device_ids=gpus)
    detector.to(device)
    detector.train(True)
    optimizer = th.optim.Adam(detector.parameters())
    print('model is builded.')

    train_step = 0
    for epoch in range(50):
        for i, sample in enumerate(dataloader):
            optimizer.zero_grad()
            sample['image'] = sample['image'].to(device)
            sample['boxes'] = sample['boxes'].to(device)
            loss = detector(sample)
            if enable_multi_gpus:
                loss = sum(loss)
            loss.backward()
            optimizer.step()
            lr = _lr_adjust(train_step)
            ut.adjust_learning_rate(optimizer, lr)
            #if i % 100 == 0:
            print("epoch %d, step %d, total_step %d, loss %.3f" % (epoch, i, train_step, loss.item()))
            i += 1
            train_step += 1

def _lr_adjust(step):
    if step < 2e4: return 1e-2
    if step < 10e4: return 1e-4
    else: return 1e-4 * (0.7 ** ((step - 10e4) / 10000))

if __name__ == '__main__':
    train()