import torch.utils.data as data
import torch
import numpy as np
import random
import cv2
from PIL import Image
from torch.utils.data.sampler import Sampler

from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from roi_data_layer.minibatch import get_minibatch, get_minibatch


class GeneralTestLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, training=True, normalize=None):
        self._roidb = roidb
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio


    def __getitem__(self, index):
        index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])  # (H, W, scale)
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)

        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        # gt_boxes = torch.FloatTensor([1,1,1,1,1])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)