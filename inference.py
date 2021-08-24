import os
import numpy as np
import argparse
import time
import pickle
import cv2
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from matplotlib import pyplot as plt
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.inference_loader import InferenceLoader
from roi_data_layer.general_test_loader import GeneralTestLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.fsod_logger import FSODInferenceLogger
from utils import *


if __name__ == '__main__':

    args = parse_args()
    print(args)
    cfg_from_file(args.cfg_file)
    cfg_from_list(args.set_cfgs)

    # prepare roidb
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    CWD = os.getcwd()
    support_dir = os.path.join(CWD, 'data/supports', args.sup_dir)

    # load dir
    input_dir = os.path.join(args.load_dir, "train/checkpoints")
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'model_{}_{}.pth'.format(args.checkepoch, args.checkpoint))

    # initilize the network
    classes = ['fg', 'bg']
    model = get_model(args.net, pretrained=False, way=args.way, shot=args.shot, classes=classes)
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    model.load_state_dict(checkpoint['model'])
    if args.mGPUs:
        model = model.module
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    cfg.CUDA = True
    model.cuda()
    model.eval()

    # initilize the tensor holders
    holders = prepare_var(support=True)
    im_data = holders[0]
    im_info = holders[1]
    num_boxes = holders[2]
    gt_boxes = holders[3]
    support_ims = holders[4]

    # prepare holder for predicted boxes
    start = time.time()
    max_per_image = 100
    thresh = 0.05
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                for _ in range(imdb.num_classes)]
    _t = {'im_detect': time.time(), 'misc': time.time()}

    model.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    dataset = InferenceLoader(0, imdb, roidb, ratio_list, ratio_index, support_dir, 
                            1, len(imdb._classes), num_shot=args.shot, training=False, normalize=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    for i in tqdm(range(num_images)):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            support_ims.resize_(data[4].size()).copy_(data[4])


        det_tic = time.time()
        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims)
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev

            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)


        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        # re-scale boxes to the origin img scale
        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        

        for j in range(1, imdb.num_classes):
            if j != gt_boxes[0, 0, 4]:
                all_boxes[j][i] = empty_array
                continue
            inds = torch.nonzero(scores[:,1]>thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:,1][inds]
                cls_boxes = pred_boxes[inds, :]
                cls_dets = NMS(cls_boxes, cls_scores)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        # if args.imlog:
        #     origin_im = im_data[0].permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]
        #     origin_im = origin_im - origin_im.min()
        #     origin_im /= origin_im.max()
        #     gt_im = origin_im.copy()
        #     pt_im = origin_im.copy()
        #     np_gt_boxes = gt_boxes[0]
        #     for n in range(np_gt_boxes.shape[0]):
        #         box = np_gt_boxes[n].clone()
        #         cv2.rectangle(gt_im, (box[0], box[1]), (box[2], box[3]), (0.1, 1, 0.1), 2)
        #     plt.imshow(gt_im)
        #     plt.show()
        #     sup_im = support_ims[0][0].permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1]
        #     sup_im = sup_im - sup_im.min()
        #     sup_im /= sup_im.max()
        #     plt.imshow(sup_im)
        #     plt.show()
        #     raise Exception(' ')

            # raise Exception(' ')
            # cv2.rectangle(im, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (20, 255, 20), 2)
            # tb_logger.write(i, gt, support_ims, predict, save_im=True)

    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
    sys.stdout.flush()

    output_dir = os.path.join(CWD, 'inference_output', args.eval_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
