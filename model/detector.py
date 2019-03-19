
import torch as th
import torch.nn as nn
import utils as ut

from model.module import MultiAnchor
from model.net import AlphaNet

class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.anchor_sizes = cfg['network']['anchor_sizes']
        self.train_image_shape = cfg['train']['image_shape']
        self.net = AlphaNet(len(self.anchor_sizes))
        self.anchor_strides = self.net.strides
        self.train_anchor_obj = MultiAnchor([self.train_image_shape]*2,
                                            self.anchor_strides,
                                            self.anchor_sizes)

    def forward(self, sample):
        if self.training:
            return self.forward_train(sample)
        else:
            pass

    def forward_train(self, sample):
        images = sample['image'].permute([0, 3, 1, 2])
        boxes = sample['boxes']
        gt_iou, gt_offset = self.train_anchor_obj.encode(boxes)
        predictions = self.net(images)
        pred_log = predictions['log']
        pred_cls = predictions['cls']
        pred_loc = predictions['loc']
        loss = self.loss_fn(gt_iou, gt_offset, pred_log, pred_loc)
        return loss



    def loss_fn(self, gt_iou_list, gt_offset_list, pred_log_list, pred_loc_list):
        batch_size = self.cfg['train']['batch_size']
        log_loss_list = []
        loc_loss_list = []
        for i in range(len(self.anchor_sizes)):
            iou = gt_iou_list[i]
            gt_label = th.where(iou > self.cfg['train']['iou_thres'],
                                th.ones_like(iou, dtype=th.long),
                                th.zeros_like(iou, dtype=th.long))
            gt_label.requires_grad=False
            log_loss = th.nn.functional.cross_entropy(pred_log_list[i], gt_label, reduction='none')
            log_loss = th.sum(log_loss)/batch_size
            log_loss_list.append(log_loss)

            gt_offset = gt_offset_list[i]
            gt_offset.requires_grad = False
            loc_loss = th.nn.functional.mse_loss(pred_loc_list[i], gt_offset_list[i], reduction='none')
            loc_loss = th.sum(loc_loss)/batch_size
            loc_loss_list.append(loc_loss)
        total_loss = sum(log_loss_list) + sum(loc_loss_list)
        return total_loss
