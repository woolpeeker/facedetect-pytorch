
import utils as ut

from PIL import Image, ImageOps
import numpy as np
import random

__all__ = ['select_crop_face',
           'random_crop']

def select_crop_face(img, boxes, out_shape, sel_box_idx, to_size):
    '''random crop the image, the output image must contain the selected box,
    and resize the image to make the box to_size. crop fisrtly, resize second.
    :param im: PIL.Image
    :param boxes: int [[ymin, xmin, ymax, xmax],...]
    :param out_shape: a scalar
    :param sel_box: [ymin, xmin, ymax, xmax]
    :param to_size: a scalar
    :return: (cropped_image, cropped_boxes)
    '''
    sel_box = boxes[sel_box_idx]
    original_sel_box_size = np.sqrt((sel_box[2]-sel_box[0]) * (sel_box[3]-sel_box[1]))
    ratio = to_size / original_sel_box_size
    ratio_h = ratio * random.uniform(0.85, 1.15)
    ratio_w = ratio * random.uniform(0.85, 1.15)
    resized_h = int(ratio_h * img.size[1])
    resized_w = int(ratio_w * img.size[0])
    resized_img = img.resize((resized_w, resized_h), Image.BILINEAR)
    resized_boxes = ut.resize_bboxes((ratio_h,ratio_w), boxes)
    sel_box = resized_boxes[sel_box_idx]

    dst_h = out_shape
    dst_w = out_shape
    dst_y = np.random.uniform(low=np.maximum(0, sel_box[2] - dst_h),
                              high=np.minimum(sel_box[0], resized_h - dst_h))
    dst_x = np.random.uniform(low=np.maximum(0, sel_box[3] - dst_w),
                              high=np.minimum(sel_box[1], resized_w - dst_w))
    cropped_img, cropped_boxes = ut.crop_image(resized_img, [dst_y, dst_x, dst_y+dst_h, dst_x+dst_w], resized_boxes)
    mask = ut.bboxes_filter_center(cropped_boxes, (dst_h, dst_w))
    cropped_boxes = cropped_boxes[mask]
    if len(cropped_boxes)==0:
        raise ValueError('boxes is empty.')
    return cropped_img, cropped_boxes

def random_crop(img, boxes, out_shape, scale):
    '''
    random crop the image
    :param im: Pillow Image
    :param boxes: int boxes
    :param out_shape: a scalar
    :param scale: a scalar
    :return: (cropeed_image, cropped_boxes)
    '''
    resized_h = int(scale * img.shape[1])
    resized_w = int(scale * img.shape[0])
    resized_img = img.resize((resized_w, resized_h), Image.BILINEAR)
    padded_img, padded_boxes = ut.pad_image(resized_img, boxes, (out_shape, out_shape))

    dst_y = random.randint(0, padded_img.shape[1] - out_shape)
    dst_x = random.randint(0, padded_img.shape[0] - out_shape)
    dst_h = out_shape
    dst_w = out_shape
    cropped_img, cropped_boxes = ut.crop_image(padded_img,
                                               [dst_y, dst_x, dst_y+dst_h, dst_x+dst_w],
                                               padded_boxes)
    mask = ut.bboxes_filter_center(cropped_boxes, (dst_h, dst_w))
    cropped_boxes = cropped_boxes[mask]
    return cropped_img, cropped_boxes