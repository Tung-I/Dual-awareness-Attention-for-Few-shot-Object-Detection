"""The data layer used during training to train a Fast R-CNN network.
"""
import numpy as np
import random
import time
import pdb
import cv2
import torch.utils.data as data
import torch
import os
from pathlib import Path
from PIL import Image
from scipy.misc import imread

from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob

from pycocotools.coco import COCO


class InferenceLoader(data.Dataset):
    def __init__(self, epi_random_seed, imdb, roidb, ratio_list, ratio_index, support_dir,
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
        self.epi_random_seed = epi_random_seed
        #############################################################################
        # roidb:
        # {'width': 640, 'height': 484, 'boxes': array([[ 58, 152, 268, 243]], dtype=uint16), 
        # 'gt_classes': array([79], dtype=int32), flipped': False, 'seg_areas': array([12328.567], dtype=float32),
        # 'img_id': 565198, 'image': '/home/tungi/FSOD/data/coco/images/val2014/COCO_val2014_000000565198.jpg', 
        # 'max_classes': array([79]), 'max_overlaps': array([1.], dtype=float32), 'need_crop': 0}

        # name_to_coco_cls_ind = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7,
        #  	'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
        # 	'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24,
        # 	'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35,
        # 	'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42,
        # 	'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 
        # 	'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 
        # 	'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72,
        # 	'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 
        # 	'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}
        #############################################################################

        self.support_im_size = 320
        self.testing_shot = num_shot

        self.support_pool = [[] for i in range(self._num_classes)]
        self._label_to_cls_name = dict(list(zip(list(range(self._num_classes)), self._imdb.classes)))
        for _label in range(1, self._num_classes):
            cls_name = self._label_to_cls_name[_label]
            cls_dir = os.path.join(support_dir, cls_name)
            support_im_paths = [str(_p) for _p in list(Path(cls_dir).glob('*.jpg'))]
            if len(support_im_paths) == 0:
                raise Exception(f'support data not found in {cls_dir}')
            random.seed(epi_random_seed)  # fix the shots
            support_im_paths = random.sample(support_im_paths, k=self.testing_shot)
            self.support_pool[_label].extend(support_im_paths)


    def __getitem__(self, index):
        # testing
        index_ratio = index
        # though it is called minibatch, in fact it contains only one img here
        minibatch_db = [self._roidb[index_ratio]]

        # load query
        blobs = get_minibatch(minibatch_db)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])  # (H, W, scale)
        data_height, data_width = data.size(1), data.size(2)
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        num_boxes = gt_boxes.size(0)
        
        # get supports
        support_data_all = np.zeros((self.testing_shot, 3, self.support_im_size, self.support_im_size), dtype=np.float32)
        current_gt_class_id = int(gt_boxes[0][4])
        selected_supports = self.support_pool[current_gt_class_id]
        
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


        return data, im_info, gt_boxes, num_boxes, supports

    def __len__(self):
        return len(self._roidb)