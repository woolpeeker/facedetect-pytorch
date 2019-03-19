
import torch as th
import numpy as np

from PIL import Image

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4]. [ymin,xmin, ymax, xmax]
      box_b: (tensor) bounding boxes, Shape: [B,4]. [ymin,xmin, ymax, xmax]
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    #here min_yx is the max of the ymin_xmin and max_yx is the min of the ymax_xmax
    min_yx = th.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                    box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_yx = th.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                    box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    inter = th.clamp((max_yx - min_yx), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]

def yxhw_to_yxyx(yxhw):
    #convert the yxhw format anchors to yxyx format
    half_hw = yxhw[..., 2:4]/2
    yxyx = [yxhw[..., 0:2] - half_hw,
            yxhw[..., 0:2] + half_hw]
    yxyx = th.cat(yxyx, dim=-1)
    return yxyx

def yxyx_to_yxhw(yxyx):
    #convert the yxyx format anchors to yxhw format
    hw = yxyx[..., 2:4] - yxyx[..., 0:2]
    yxhw = [yxyx[..., :2] + hw/2, hw]
    yxhw = th.cat(yxhw, dim=-1)
    return yxhw

def tensor_to_image(tensor, permute=True):
    if len(tensor.shape)==3:
        if permute: tensor = tensor.permute([1, 2, 0])
        img = tensor.numpy().astype(np.uint8)
        img = Image.fromarray(img, mode='RGB')
        return img
    if len(tensor.shape)==4:
        if permute: tensor = tensor.permute([0, 2, 3, 1])
        tensors = th.unbind(tensor, dim=0)
        imgs = [t.numpy().astype(np.uint8) for t in tensors]
        imgs = [Image.fromarray(img, mode='RGB') for img in imgs]
        return imgs


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def boxes_precision_recall(gt_boxes, pred_boxes, iou_thres=0.5):
    """
    :param gt_boxes:   a list of boxes tensors. length batch_size
    :param pred_boxes:  a list of boxes tensors. length batch_size
    :param iou_thres:
    :return:  a tensor [batch_size, 2]
    """
    assert len(gt_boxes.shape) == len(pred_boxes.shape) == 4
    batch_size = gt_boxes.size(0)
    pr_list = []
    for i in range(batch_size):
        iou_mat = jaccard(pred_boxes[i], gt_boxes[i])
        precision = th.max(iou_mat, dim=0) > iou_thres
        precision = th.sum(precision) / precision.size(0)
        recall = th.max(iou_mat, dim=1) > iou_thres
        recall = th.sum(recall) / recall.size(0)
        pr_list.append(th.tensor([precision, recall]))
    return th.stack(pr_list, dim=0)

def nms_batch(batch_boxes, batch_scores, scores_thres=0.5, nms_thres=0.5):
    """
    :param batch_boxes: a list of boxes tensors. length batch_size
    :param batch_scores: a list of boxes tensors. length batch_size
    :param scores_thres:
    :param nms_thres:
    :return:  a list of keep index tensors. length batch_size
    """
    assert(len(batch_boxes) == len(batch_scores))
    batch_size = len(batch_boxes)
    keep_list = [nms(batch_size[i], batch_scores[i], scores_thres, nms_thres)
                 for i in range(batch_size)]
    return keep_list

def nms(boxes, scores, scores_thres=0.5, nms_thres=0.5):
    # return the boxes index, not boolean mask
    keep = (scores_thres > scores_thres).nonzero().flatten()
    keep_boxes = boxes[keep]
    keep_scores = scores[keep]
    nms_keep = _nms(keep_boxes, keep_scores, nms_thres=nms_thres)
    keep = keep[nms_keep]
    return keep

def _nms(bboxes, scores, nms_thres=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    order = th.argsort(scores, descending=True)    # 降序排列
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)
        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= nms_thres).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值
    return th.tensor(keep)  # Pytorch的索引值为LongTensor

def tensors_to_device(ts, device):
    """
    move a list of tensors in list or dict to device
    :param obj: tensor, list, or dict
    :param device:
    :return: nothing
    """
    if isinstance(ts, list):
        for i,t in enumerate(ts):
            if isinstance(t, th.Tensor):
                ts[i] = ts[i].to(device)
    else:
        raise TypeError('input must be a list object.')

def pad_boxes(boxes, pad_to):
    """
    pad a boxes tensor to the pad_to length
    :param boxes: tensor [num_boxes, 4]
    :param pad_to: scalar
    :return: padded boxes
    """
    if pad_to < boxes.shape[0]:
        raise ValueError('pad target length is smaller than original length.')
    pad_len = pad_to-boxes.shape[0]
    pad_tensor = th.zeros([pad_len, 4], dtype=boxes.dtype)
    boxes = th.cat([boxes, pad_tensor], dim=0)
    return boxes