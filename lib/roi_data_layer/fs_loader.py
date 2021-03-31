import torch.utils.data as data
import torch
import numpy as np
import random
import time
import pdb
import cv2
from PIL import Image
from torch.utils.data.sampler import Sampler

from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from roi_data_layer.minibatch import get_minibatch, get_minibatch


class FewShotLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None, num_way=2, num_shot=5):
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

        self.support_size_threshold = 64
        self.support_im_size = 320
        self.support_way = num_way
        self.support_shot = num_shot

        ##############################
        # given the ratio_list, we want to make the ratio same for each batch.
        # ex. [0.5, 0.5, 0.7, 0.8, 1.5, 1.6, 2., 2.] -> [0.5, 0.5, 0.7, 0.7, 1.6, 1.6, 2., 2.]
        ##############################
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

        ##############################
        # prepare few shot support pool
        ##############################
        threshold = self.support_size_threshold
        self.support_db = []  # list[class_idx] = [box_info_1, box_info_2, ...]
        for i in range(self._num_classes):
            self.support_db.append([])

        for roidb_idx, _roidb in enumerate(self._roidb):
            if _roidb['flipped'] == True:
                continue
            gt_inds = np.where((_roidb['gt_classes'] != 0) & np.all(_roidb['gt_overlaps'].toarray() > -1.0, axis=1))[0]
            boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            boxes[:, 0:4] = _roidb['boxes'][gt_inds, :]  # note: boxes have not been scaled
            boxes[:, 4] = _roidb['gt_classes'][gt_inds]
            for i in range(len(gt_inds)):
                box = boxes[i, 0:4]
                box_cls_idx = int(boxes[i, 4])
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                if box_w < threshold or box_h < threshold or box_w > 2*box_h or box_h > 2*box_w:
                    continue 
                _info = {'roidb_idx': roidb_idx, 'box': box}
                self.support_db[box_cls_idx].append(_info)


    def __getitem__(self, index):
        ##############################
        # in training, for example, the index feed to dataloader may be n1
        # but the img ID may be n2, due to the order of imgs is sorted by ratios 
        # so we need to get the true index by self.ratio_index[index]
        ##############################
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # though it is called minibatch, in fact it contains only one img here
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db)  # [n_box, 5]
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])  # (H, W, scale)
        data_height, data_width = data.size(1), data.size(2)  # [1, h, w, c]
        gt_boxes = blobs['gt_boxes']

        #################
        # support data
        #################
        target_size = self.support_im_size
        support_data_all = np.zeros((self.support_way * self.support_shot, 3, target_size, target_size), dtype=np.float32)
        # positive supports
        cls_in_query = []
        for i in range(gt_boxes.shape[0]):
            _cls = gt_boxes[i, 4]
            cls_in_query.append(_cls)
        cls_in_query = list(set(cls_in_query))
        pos_cls_idx = int(random.sample(cls_in_query, k=1)[0])
        support_dbs = random.sample(self.support_db[pos_cls_idx], k=self.support_shot)
        for i, sup_db in enumerate(support_dbs):
            support_roidb = self._roidb[sup_db['roidb_idx']]
            support_box = sup_db['box']
            support_blob = get_minibatch([support_roidb])
            support_im = support_blob['data'][0]
            support_im_h, support_im_w = support_im.shape[0], support_im.shape[1]
            support_scale = support_blob['im_info'][0][2]
            support_box = (support_box * support_scale).astype(np.int16)
            support_im_final = np.zeros((3, target_size, target_size), dtype=np.float32)
            x_min, y_min = support_box[0], support_box[1]
            x_max, y_max = support_box[2], support_box[3]
            box_h, box_w = y_max - y_min, x_max - x_min
            support_im_cropped = support_im[y_min:y_max+1, x_min:x_max+1, :]
            if box_h > box_w:
                resize_scale = float(target_size) / float(box_h)
                unfit_size = int(box_w * resize_scale)
                support_im_cropped = cv2.resize(support_im_cropped, (unfit_size, target_size), interpolation=cv2.INTER_LINEAR)
            else:
                resize_scale = float(target_size) / float(box_w)
                unfit_size = int(box_h * resize_scale)
                support_im_cropped = cv2.resize(support_im_cropped, (target_size, unfit_size), interpolation=cv2.INTER_LINEAR)

            cropped_h = support_im_cropped.shape[0]
            cropped_w = support_im_cropped.shape[1]
            support_im_final[:, :cropped_h, :cropped_w] = np.transpose(support_im_cropped, (2, 0, 1))
            support_data_all[i] = support_im_final 

        # negative supports
        if self.support_way != 1:
            neg_cls = []
            for cls_id in range(1, self._num_classes):
                if cls_id not in cls_in_query:
                    neg_cls.append(cls_id)
            neg_cls_idx = random.sample(neg_cls, k=1)[0]
            neg_support_dbs = random.sample(self.support_db[neg_cls_idx], k=self.support_shot)

            for i, sup_db in enumerate(neg_support_dbs):
                support_roidb = self._roidb[sup_db['roidb_idx']]
                support_box = sup_db['box']
                support_blob = get_minibatch([support_roidb])
                support_im = support_blob['data'][0]
                support_im_h, support_im_w = support_im.shape[0], support_im.shape[1]
                support_scale = support_blob['im_info'][0][2]
                support_box = (support_box * support_scale).astype(np.int16)
                support_im_final = np.zeros((3, target_size, target_size), dtype=np.float32)
                x_min, y_min = support_box[0], support_box[1]
                x_max, y_max = support_box[2], support_box[3]
                box_h, box_w = y_max - y_min, x_max - x_min
                support_im_cropped = support_im[y_min:y_max+1, x_min:x_max+1, :]
                if box_h > box_w:
                    resize_scale = float(target_size) / float(box_h)
                    unfit_size = int(box_w * resize_scale)
                    support_im_cropped = cv2.resize(support_im_cropped, (unfit_size, target_size), interpolation=cv2.INTER_LINEAR)
                else:
                    resize_scale = float(target_size) / float(box_w)
                    unfit_size = int(box_h * resize_scale)
                    support_im_cropped = cv2.resize(support_im_cropped, (target_size, unfit_size), interpolation=cv2.INTER_LINEAR)

                cropped_h = support_im_cropped.shape[0]
                cropped_w = support_im_cropped.shape[1]
                support_im_final[:, :cropped_h, :cropped_w] = np.transpose(support_im_cropped, (2, 0, 1))
                support_data_all[i + self.support_shot] = support_im_final 

        support = torch.from_numpy(support_data_all)
    
        #################
        # query data
        #################
        # padding the input image to fixed size for each group 
        # if the image need to crop, crop to the target size.
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        ratio = self.ratio_list_batch[index]
        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # no need to crop, or after cropping
        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                                data_width, 3).zero_()
            padding_data[:data_height, :, :] = data[0]
            # update im_info [[H, W, scale]]
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                                int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        # filt boxes
        fs_gt_boxes = []
        for i in range(gt_boxes.shape[0]):
            if gt_boxes[i, 4] == pos_cls_idx:
                fs_gt_boxes += [gt_boxes[i]]
        fs_gt_boxes = torch.stack(fs_gt_boxes, 0)
        fs_gt_boxes[:, 4] = 1.
       
        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

        #
        not_keep = (fs_gt_boxes[:,0] == fs_gt_boxes[:,2]) | (fs_gt_boxes[:,1] == fs_gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        fs_gt_boxes_padding = torch.FloatTensor(self.max_num_box, fs_gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            fs_gt_boxes = fs_gt_boxes[keep]
            num_boxes = min(fs_gt_boxes.size(0), self.max_num_box)
            fs_gt_boxes_padding[:num_boxes,:] = fs_gt_boxes[:num_boxes]
        else:
            num_boxes = 0

        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        
        # to sum up, data in diffenent batches may have different size, 
        # but will have same size in the same batch

        return padding_data, im_info, fs_gt_boxes_padding, num_boxes, support_data_all, gt_boxes_padding
        

    def __len__(self):
        return len(self._roidb)


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data