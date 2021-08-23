import argparse
import torch
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob
from model.roi_layers import nms
from model.utils.config import cfg
from torch.autograd import Variable
from model.framework.faster_rcnn import FasterRCNN
from model.framework.fsod import FSOD
from model.framework.meta import METARCNN
from model.framework.fgn import FGN
from model.framework.dana import DAnARCNN
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # net and dataset
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res101', default='res50', type=str)
    parser.add_argument('--flip', dest='use_flip', help='use flipped data or not', default=False, action='store_true')
    # optimizer
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch', default=1000, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    # train&finetuning setting
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=8, type=int)
    parser.add_argument('--ls', dest='large_scale', help='whether use large imag scale', action='store_true')                      
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=16, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=12, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="models", type=str)
    parser.add_argument('--ascale', dest='ascale', help='number of anchor scale', default=4, type=int)
    # parser.add_argument('--ft', dest='finetune', help='finetune mode', default=False, action='store_true')
    parser.add_argument('--eval', dest='eval', help='evaluation mode', default=False, action='store_true')
    parser.add_argument('--onc', dest='old_n_classes', help='number of classes of the source domain', default=81, type=int)
    # inference setting
    parser.add_argument('--eval_dir', dest='eval_dir', help='output directory of evaluation', default=None, type=str)
    # few shot
    parser.add_argument('--fs', dest='fewshot', help='few-shot setting', default=False, action='store_true')
    parser.add_argument('--way', dest='way', help='num of support way', default=2, type=int)
    parser.add_argument('--shot', dest='shot', help='num of support shot', default=5, type=int)
    parser.add_argument('--sup_dir', dest='sup_dir', help='directory of support images', default='all', type=str) 
    # load checkpoints
    parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true', default=False)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="models", type=str)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model', default=0, type=int)
    # logger
    parser.add_argument('--dlog', dest='dlog', help='disable the logger', default=False, action='store_true')
    parser.add_argument('--imlog', dest='imlog', help='save im in the logger', default=False, action='store_true')


    args = parser.parse_args()

    # parse dataset
    if args.ascale == 3:
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.ascale == 4:
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    else:
        raise Exception(f'invalid anchor scale {args.ascale}')
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train"
        args.imdbval_name = "coco_2014_minival"
    elif args.dataset == "coco2017":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
    elif args.dataset == "coco_base":
        args.imdb_name = "coco_60_set1"
    elif args.dataset == "coco_ft":
        args.imdb_name = "coco_ft"
    elif args.dataset == "0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "val2014_novel":
        args.imdbval_name = "coco_20_set1"
    elif args.dataset == "val2014_base":
        args.imdbval_name = "coco_20_set2"

    else:
        raise Exception(f'dataset {args.dataset} not defined')
    args.cfg_file = "cfgs/res50.yml"
    return args

def get_model(name, pretrained=True, way=2, shot=3, classes=[]):
    if name == 'frcnn':
        model = FasterRCNN(classes, pretrained=pretrained)
    elif name == 'fsod':
        model = FSOD(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'meta':
        model = METARCNN(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'fgn':
        model = FGN(classes, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'cisa':
        model = CISARCNN(classes, 'concat', 256, 256, pretrained=pretrained, num_way=way, num_shot=shot)
    elif name == 'DAnA':
        model = DAnARCNN(classes, 'concat', 256, 256, pretrained=pretrained, num_way=way, num_shot=shot)
    else:
        raise Exception(f"network {name} is not defined")
    model.create_architecture()
    # model.cuda()
    # if eval:
    #     model.eval()
    return model
        

def create_annotation(nd_dir, cls_names, cls_im_inds, dump_path):
    clsname2ind = {'cube':1, 'can':2, 'box':3, 'bottle':4}
    data_categories = []
    for name in cls_names:   
        dic = {}
        dic['supercategory'] = 'None'
        dic['id'] = clsname2ind[name]
        dic['name'] = name
        data_categories.append(dic)
    data_images = []
    data_annotations = []
    for cls, inds in zip(cls_names, cls_im_inds):
        for ind in inds:
            im_file_name = str(ind).zfill(6) + '.jpg'
            dic = {}
            dic['license'] = 1
            dic['file_name'] = im_file_name
            dic['coco_url'] = 'http://farm3.staticflickr.com/2253/1755223462_fabbeb8dc3_z.jpg'
            dic['height'] = 256
            dic['width'] = 256
            dic['date_captured'] = '2013-11-15 13:55:22'
            dic['id'] = ind
            data_images.append(dic)
            
            ann_file_name = str(ind).zfill(6) + '.npy'
            boxes = np.load(os.path.join(nd_dir, ann_file_name), allow_pickle=True)
            for j in range(boxes.shape[0]):
                box = boxes[j]
                dic = {}
                dic['segmentation'] = [[184.05]]
                dic['area'] = 1.28
                dic['iscrowd'] = 0
                dic['image_id'] = ind
                dic['bbox'] = [int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])]
                dic['category_id'] = clsname2ind[cls]
                dic['id'] = int(str(ind)+str(j))
                data_annotations.append(dic)

    coco_json_path = '/home/tony/datasets/coco/annotations/instances_minival2014.json'
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
   
    new_dict = {}
    new_dict['info'] = data['info']
    new_dict['images'] = data_images
    new_dict['licenses'] = data['licenses']
    new_dict['annotations'] = data_annotations
    new_dict['categories'] = data_categories
    with open(dump_path, 'w') as f:
        json.dump(new_dict, f)


def generate_pseudo_label(output_dir, sp_dir, q_im_path, model, num_shot):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    q_im = np.asarray(Image.open(q_im_path))[:, :, :3]
    if num_shot > 1:
        final_dets = None
        for i in range(num_shot): 
            sp_im_path = os.path.join(sp_dir, f'shot_{i+1}.jpg')
            sp_im = np.asarray(Image.open(sp_im_path))[:, :, :3]
            cls_dets = run_detection(sp_im, q_im, model)
            if final_dets is not None:
                final_dets = torch.cat((final_dets, cls_dets), 0)
            else:
                final_dets = cls_dets
        _, order = torch.sort(final_dets[:, 4], 0, True)
        final_dets = final_dets[order]
        keep = nms(final_dets[:, :4], final_dets[:, 4], cfg.TEST.NMS)
        final_dets = final_dets[keep.view(-1).long()]
    else:
        sp_im_path = os.path.join(sp_dir, 'shot_1.jpg')
        sp_im = np.asarray(Image.open(sp_im_path))[:, :, :3]
        final_dets = run_detection(sp_im, q_im, model)
    return final_dets


def support_im_preprocess(im_list, cfg, support_im_size):
    n_of_shot = len(im_list)
    support_data_all = np.zeros((n_of_shot, 3, support_im_size, support_im_size), dtype=np.float32)
    for i, im in enumerate(im_list):
        im = im[:,:,::-1]  # rgb -> bgr
        target_size = np.min(im.shape[0:2])  # don't change the size
        im, _ = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        _h, _w = im.shape[0], im.shape[1]
        if _h > _w:
            resize_scale = float(support_im_size) / float(_h)
            unfit_size = int(_w * resize_scale)
            im = cv2.resize(im, (unfit_size, support_im_size), interpolation=cv2.INTER_LINEAR)
        else:
            resize_scale = float(support_im_size) / float(_w)
            unfit_size = int(_h * resize_scale)
            im = cv2.resize(im, (support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
        h, w = im.shape[0], im.shape[1]
        support_data_all[i, :, :h, :w] = np.transpose(im, (2, 0, 1))
    support_data = torch.from_numpy(support_data_all).unsqueeze(0)
    
    return support_data

def query_im_preprocess(im_data, cfg):
    target_size = cfg.TRAIN.SCALES[0]
    im_data, im_scale = prep_im_for_blob(im_data, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    im_data = torch.from_numpy(im_data)
    im_info = np.array([[im_data.shape[0], im_data.shape[1], im_scale]], dtype=np.float32)
    im_info = torch.from_numpy(im_info)
    gt_boxes = torch.from_numpy(np.array([0]))
    num_boxes = torch.from_numpy(np.array([0]))
    query = im_data.permute(2, 0, 1).contiguous().unsqueeze(0)
    
    return query, im_info, gt_boxes, num_boxes

def run_detection(sp_im, q_im, model):
    support_data = support_im_preprocess([sp_im], cfg, 320)
    query_data, im_info, gt_boxes, num_boxes = query_im_preprocess(q_im, cfg)
    data = [query_data, im_info, gt_boxes, num_boxes, support_data]
    im_data, im_info, num_boxes, gt_boxes, support_ims = prepare_var(support=True)
    with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        support_ims.resize_(data[4].size()).copy_(data[4])
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    box_deltas = bbox_pred.data
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    pred_boxes /= data[1][0][2].item()
    # do nms
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    thresh = 0.05
    inds = torch.nonzero(scores[:,1]>thresh).view(-1)
    cls_scores = scores[:,1][inds]
    cls_boxes = pred_boxes[inds, :]
    cls_dets = NMS(cls_boxes, cls_scores)

    return cls_dets

def prepare_var(support=False):
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if support:
        support_ims = torch.FloatTensor(1)
        support_ims = support_ims.cuda()
        support_ims = Variable(support_ims)
        return [im_data, im_info, num_boxes, gt_boxes, support_ims]
    else:
        return [im_data, im_info, num_boxes, gt_boxes]

def plot_box(im, boxes, thres=0.5):
    # boxes[n] = [x1, y1, x2, y2, score]
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if box[4] > thres:
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (20, 255, 20), 2)
    return im 

def NMS(boxes, scores):
    _, order = torch.sort(scores, 0, True)
    dets = torch.cat((boxes, scores.unsqueeze(1)), 1)[order]
    keep = nms(boxes[order, :], scores[order], cfg.TEST.NMS)
    dets = dets[keep.view(-1).long()]
    return dets