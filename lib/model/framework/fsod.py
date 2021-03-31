import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import time

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.framework.resnet import resnet50


class _fsodRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, n_way=2, n_shot=5, g=True, l=True, p=True):
        super(_fsodRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        # for proposal-target matching
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # pooling or align
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        # few shot rcnn head
        self.global_relation = g
        self.local_correlation = l
        self.patch_relation = p
        self.pool_feat_dim = 1024
        self.soft_gamma = 1e1
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.avgpool_fc = nn.AvgPool2d(7)
        self.patch_avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
        dim_in = self.pool_feat_dim
        if self.global_relation:
            self.global_fc_1 = nn.Linear(dim_in*2, dim_in)
            self.global_fc_2 = nn.Linear(dim_in, dim_in)
            self.global_cls_score = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)
            init.normal_(self.global_fc_1.weight, std=0.01)
            init.constant_(self.global_fc_1.bias, 0)
            init.normal_(self.global_fc_2.weight, std=0.01)
            init.constant_(self.global_fc_2.bias, 0)
            init.normal_(self.global_cls_score.weight, std=0.01)
            init.constant_(self.global_cls_score.bias, 0)

        if self.local_correlation:
            self.corr_conv = nn.Conv2d(dim_in, dim_in, 1, padding=0, bias=False)
            #self.bbox_pred_cor = nn.Linear(dim_in, 4 * 2)
            self.corr_cls_score = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)
            init.normal_(self.corr_conv.weight, std=0.01)
            init.normal_(self.corr_cls_score.weight, std=0.01)
            init.constant_(self.corr_cls_score.bias, 0)
        
        if self.patch_relation: 
            self.patch_conv_1 = nn.Conv2d(2*dim_in, int(dim_in/4), 1, padding=0, bias=False)
            self.patch_conv_2 = nn.Conv2d(int(dim_in/4), int(dim_in/4), 3, padding=0, bias=False)
            self.patch_conv_3 = nn.Conv2d(int(dim_in/4), dim_in, 1, padding=0, bias=False)
            self.patch_cls_score = nn.Linear(dim_in, 2)
            init.normal_(self.patch_conv_1.weight, std=0.01)
            init.normal_(self.patch_conv_2.weight, std=0.01)
            init.normal_(self.patch_conv_3.weight, std=0.01)
            init.normal_(self.patch_cls_score.weight, std=0.01)
            init.constant_(self.patch_cls_score.bias, 0)
        
        # few shot settings
        self.n_way = n_way
        self.n_shot = n_shot

    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_ims, all_cls_gt_boxes=None):
        if self.training:
            self.num_of_rois = cfg.TRAIN.BATCH_SIZE
        else:
            self.num_of_rois = cfg.TEST.RPN_POST_NMS_TOP_N 
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feature extraction
        base_feat = self.RCNN_base(im_data)
        if self.training:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)
            support_feats = support_feats.view(-1, self.n_way*self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].mean(1)
            neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].mean(1)
            pos_support_feat = self.avgpool(pos_support_feat)  # [B, 1024, 7, 7]
            neg_support_feat = self.avgpool(neg_support_feat)
        else:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].mean(1)
            pos_support_feat = self.avgpool(pos_support_feat)

        # attention RPN
        heatmaps = []
        for kernel, feat in zip(pos_support_feat.chunk(pos_support_feat.size(0), dim=0), base_feat.chunk(base_feat.size(0), dim=0)):
            kernel = kernel.view(pos_support_feat.size(1), 1, pos_support_feat.size(2), pos_support_feat.size(3))
            heatmap = F.conv2d(feat, kernel, groups=1024)  # [1, 1024, h, w]
            heatmap = heatmap.squeeze()
            heatmaps += [heatmap]
        correlation_feat = torch.stack(heatmaps, 0)


        ## rois [B, RPN_POST_NMS_TOP_N(2000), 5], 5 is [batch_num, x1, y1, x2, y2]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(correlation_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            ## rois [B, rois_per_image(128), 5]
                ### 5 is [batch_num, x1, y1, x2, y2]
            ## rois_label [B, 128]
            ## rois_target [B, 128, 4]
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)

        # do roi pooling based on predicted rois, pooled_feat = [B*128, 1024, 7, 7]
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # rcnn head
        if self.training:
            bbox_pred, cls_prob, cls_score_all = self.rcnn_head(pooled_feat, pos_support_feat)
            _, neg_cls_prob, neg_cls_score_all = self.rcnn_head(pooled_feat, neg_support_feat)
            cls_prob = torch.cat([cls_prob, neg_cls_prob], dim=0)
            cls_score_all = torch.cat([cls_score_all, neg_cls_score_all], dim=0)
            neg_rois_label = torch.zeros_like(rois_label)
            rois_label = torch.cat([rois_label, neg_rois_label], dim=0)
        else:
            bbox_pred, cls_prob, cls_score_all = self.rcnn_head(pooled_feat, pos_support_feat)

        # losses
        if self.training:
            ## bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            ## classification loss, 2-way, 1:2:1
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            bg_inds = (rois_label == 0).nonzero().squeeze(-1)
            cls_score_softmax = torch.nn.functional.softmax(cls_score_all, dim=1)
            bg_cls_score_softmax = cls_score_softmax[bg_inds, :]
            bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(rois_label.shape[0] * 0.25)))
            bg_num_1 = max(1, min(fg_inds.shape[0], bg_num_0))
            _sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 1], descending=True)
            real_bg_inds = bg_inds[sorted_bg_inds]  # sort the real_bg_inds
            real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(rois_label.shape[0] * 0.5)][:bg_num_0]  # pos support
            real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(rois_label.shape[0] * 0.5)][:bg_num_1]  # neg_support
            topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
            RCNN_loss_cls = F.cross_entropy(cls_score_all[topk_inds], rois_label[topk_inds])
        else:
            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def rcnn_head(self, pooled_feat, support_pooled_feat):
        # box regression
        bbox_pred = self.RCNN_bbox_pred(self._head_to_tail(pooled_feat))  # [B*128, 4]
        # classification
        ## global relation
        if self.global_relation:
            global_feats = []
            current_b = 0
            for i, feat in enumerate(pooled_feat.chunk(pooled_feat.size(0), dim=0)):
                if i == (current_b + 1) * self.num_of_rois:
                    current_b += 1
                current_support_feat = support_pooled_feat[current_b].unsqueeze(0)
                concat_feat = torch.cat((feat, current_support_feat), 1)  # [1, 2c, 7, 7]
                global_feats += [concat_feat]
            global_feats = torch.cat(global_feats, 0)  # [B*128, 2c, 7, 7]
            global_feats = self.avgpool_fc(global_feats).squeeze(3).squeeze(2)  # [B*128, 2c]
            out_fc = F.relu(self.global_fc_1(global_feats), inplace=True)
            out_fc = F.relu(self.global_fc_2(out_fc), inplace=True)  # [B*128, c]
            global_cls_score = self.global_cls_score(out_fc)  # [B*128, 2]
        ## local correlation
        if self.local_correlation:
            corr_rois = self.corr_conv(pooled_feat)  # [B*128, c, 7, 7]
            corr_support = self.corr_conv(support_pooled_feat)  # [B, c, 7, 7]
            heatmaps = []
            for b, grouped_feat in enumerate(corr_rois.chunk(corr_support.size(0), dim=0)):
                kernel = corr_support[b].unsqueeze(0)  # [1, c, 7, 7]
                kernel = kernel.view(kernel.size(1), 1, kernel.size(2), kernel.size(3))  # [c, 1, 7, 7]
                heatmap = F.conv2d(grouped_feat, kernel, groups=self.pool_feat_dim)  # [128, c, 1, 1]
                heatmap = heatmap.squeeze()
                heatmaps += [heatmap]
            out_corr = torch.cat(heatmaps, 0)  # [B*128, c]
            corr_cls_score = self.corr_cls_score(out_corr)  # [B*128, 2]
        ## patch relation
        if self.patch_relation:
            batch_size = support_pooled_feat.size(0)
            patch_feats = []
            for b, roi_feats in enumerate(pooled_feat.chunk(batch_size, dim=0)):
                current_support_feat = support_pooled_feat[b].expand(roi_feats.size(0), -1, -1, -1)  # [128, c, 7, 7]
                patch = torch.cat((roi_feats, current_support_feat), 1)  # [128, 2c, 7, 7]
                patch_feats += [patch]
            x = torch.cat(patch_feats, dim=0)
            x = F.relu(self.patch_conv_1(x), inplace=True)
            x = self.patch_avgpool(x)
            x = F.relu(self.patch_conv_2(x), inplace=True) # 3x3
            x = F.relu(self.patch_conv_3(x), inplace=True) # 3x3
            x = self.patch_avgpool(x) # 1x1
            x = x.squeeze(3).squeeze(2)
            patch_cls_score = self.patch_cls_score(x)

        # score and prob
        if self.patch_relation:
            cls_score_all = (global_cls_score + corr_cls_score + patch_cls_score) / self.soft_gamma
        else:
            cls_score_all = (global_cls_score + corr_cls_score) / self.soft_gamma
        cls_prob = F.softmax(cls_score_all, 1)  # [B*128, 1]

        return bbox_pred, cls_prob, cls_score_all 

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    
class FSOD(_fsodRCNN):
  def __init__(self, classes, num_layers=50, pretrained=False, num_way=2, num_shot=5, g=True, l=True, p=True):
    self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained

    _fsodRCNN.__init__(self, classes, n_way=num_way, n_shot=num_shot, g=g, l=l, p=p)

  def _init_modules(self):
    resnet = resnet50()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet. (base -> top -> head)
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)  # 1024 -> 2048

    # build rcnn head
    # self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    self.RCNN_bbox_pred = nn.Linear(2048, 4)


    # Fix blocks 
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)  # [128, 2048]
    return fc7