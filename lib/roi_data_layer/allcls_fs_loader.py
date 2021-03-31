import torch.utils.data as data
import torch
import numpy as np
import random
import cv2
import os
from PIL import Image
from pathlib import Path
from torch.utils.data.sampler import Sampler
from scipy.misc import imread
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.blob import prep_im_for_blob, im_list_to_blob


class ALLCLSFSLoader(data.Dataset):
    def __init__(self, imdb, roidb, ratio_list, ratio_index, support_dir,
                batch_size, num_classes, num_shot=5, training=True, normalize=None):
        self._imdb = imdb
        self._roidb = roidb
        self._num_classes = num_classes
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

        self.support_pool = [[] for i in range(self._num_classes)]
        self._label_to_cls_name = dict(list(zip(list(range(self._num_classes)), self._imdb.classes)))
        for _label in range(1, self._num_classes):
            cls_name = self._label_to_cls_name[_label]
            cls_dir = os.path.join(support_dir, cls_name)
            support_im_paths = [str(_p) for _p in list(Path(cls_dir).glob('*.jpg'))]
            if len(support_im_paths) == 0:
                raise Exception(f'support data not found in {cls_dir}')
            self.support_pool[_label].extend(support_im_paths)

        self.support_im_size = 320
        self.testing_shot = num_shot


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

        all_cls_gt_boxes = gt_boxes.clone()

        cur_cls_id_list = []
        for i in range(gt_boxes.size(0)):
            if gt_boxes[i, 4] not in cur_cls_id_list:
                cur_cls_id_list.append(gt_boxes[i, 4])
        random.seed(0)
        chosen_cls = random.sample(cur_cls_id_list, k=1)[0]

        new_gt_boxes = []
        for i in range(gt_boxes.size(0)):
            if gt_boxes[i, 4] == chosen_cls:
                new_gt_boxes.append([gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 2], gt_boxes[i, 3], chosen_cls])
        gt_boxes = torch.from_numpy(np.asarray(new_gt_boxes))

        num_boxes = 0

        # get supports
        support_data_all = np.zeros((self.testing_shot, 3, self.support_im_size, self.support_im_size), dtype=np.float32)
        current_gt_class_id = int(gt_boxes[0][4])
        pool = self.support_pool[current_gt_class_id]

        random.seed(index)
        selected_supports = random.sample(pool, k=self.testing_shot)
        
        for i, _path in enumerate(selected_supports):
            support_im = imread(_path)[:,:,::-1]  # rgb -> bgr
            target_size = np.min(support_im.shape[0:2])  # don't change the size
            support_im, _ = prep_im_for_blob(support_im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            _h, _w = support_im.shape[0], support_im.shape[1]
            if _h > _w:
                resize_scale = float(self.support_im_size) / float(_h)
                unfit_size = int(_w * resize_scale)
                support_im = cv2.resize(support_im, (unfit_size, self.support_im_size), interpolation=cv2.INTER_LINEAR)
            else:
                resize_scale = float(self.support_im_size) / float(_w)
                unfit_size = int(_h * resize_scale)
                support_im = cv2.resize(support_im, (self.support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
            h, w = support_im.shape[0], support_im.shape[1]
            support_data_all[i, :, :h, :w] = np.transpose(support_im, (2, 0, 1)) 
        supports = torch.from_numpy(support_data_all)


        return data, im_info, gt_boxes, num_boxes, supports, all_cls_gt_boxes

    def __len__(self):
        return len(self._roidb)