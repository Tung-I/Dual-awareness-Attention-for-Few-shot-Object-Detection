import torch.nn as nn
import torch
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.ops import nms
from model.retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from model.retinanet.anchors import Anchors
from model.retinanet import losses

from model.retinanet.model import PyramidFeatures, RegressionModel, ClassificationModel


class aa_retinanet(nn.Module):

    def __init__(self, num_classes, block, layers, n_head=1, attention_type='concat', shot_mode='mean', 
                    num_way=2, num_shot=5, pos_encoding=True, pretrained=False):
        super(aa_retinanet, self).__init__()
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.pretrained = pretrained
        self.inplanes = 64
        self.n_head = n_head
        self.attention_type = attention_type
        self.shot_mode = shot_mode
        self.num_shot = num_shot
        self.pos_encoding = pos_encoding
        self.support_im_size = 320
    
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            self.load_state_dict({k:v for k,v in state_dict.items() if k in self.state_dict()})

            def set_bn_fix(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for p in m.parameters(): p.requires_grad=False
            self.apply(set_bn_fix)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])  # [512, 1024, 2048]
        self.fpn_dim = 256
        attention_output_dim = 256 if attention_type == 'product' else 512
        self.regressionModel = RegressionModel(attention_output_dim)
        self.classificationModel = ClassificationModel(attention_output_dim, num_classes=num_classes)
        self.anchors = Anchors([4, 5, 6, 7])
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

        self.resnet_base = nn.Sequential(
            self.conv1, 
            self.bn1, 
            self.relu, 
            self.maxpool
        )

        # querys, keys
        Q_list = []
        K_list = []
        self.d_k = 64
        for i in range(self.n_head):
            Q_weight = nn.Linear(self.fpn_dim, self.d_k)
            K_weight = nn.Linear(self.fpn_dim, self.d_k)
            init.normal_(Q_weight.weight, std=0.01)
            init.constant_(Q_weight.bias, 0)
            init.normal_(K_weight.weight, std=0.01)
            init.constant_(K_weight.bias, 0)
            Q_list.append(Q_weight)
            K_list.append(K_weight)
        self.pyramid_Q_layers = nn.ModuleList(Q_list)
        self.pyramid_K_layers = nn.ModuleList(K_list)
        if self.pos_encoding:
            pel_4 = PositionalEncoding(d_model=256, max_len=20*20)
            pel_5 = PositionalEncoding(d_model=256, max_len=10*10)
            pel_6 = PositionalEncoding(d_model=256, max_len=5*5)
            pel_7 = PositionalEncoding(d_model=256, max_len=3*3)
        self.pos_encoding_layers = nn.ModuleList([pel_4, pel_5, pel_6, pel_7])
        if n_head != 1:
            self.multihead_layer = nn.Linear(n_head * feature_size, feature_size)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def create_architecture(self):
        pass

    def attention_module(self, fpn_q_feats, fpn_s_feats):
        batch_size = fpn_q_feats[0].size(0)
        output_pyramid_attentive_feat = []
        for p in range(len(fpn_q_feats)):
            multihead_support_feat = []
            q_feat = fpn_q_feats[p]
            feat_h = q_feat.size(2)
            feat_w = q_feat.size(3)
            q_feat = q_feat.view(batch_size, self.fpn_dim, -1).permute(0, 2, 1).contiguous()  # [B, *, 256]
            s_feat_nshot = fpn_s_feats[p].view(batch_size, self.num_shot, self.fpn_dim, -1).transpose(2, 3).contiguous()  # [B, S, *, 256]

            for n in range(self.n_head):
                attention_q_mat = self.pyramid_Q_layers[n](q_feat)
                multishot_support_feat = []
                for i in range(self.num_shot):
                    s_feat = s_feat_nshot[:, i, :, :]
                    if self.pos_encoding:
                        s_feat = self.pos_encoding_layers[p](s_feat)
                    attention_k_mat = self.pyramid_K_layers[n](s_feat)
                    attention_mask = torch.bmm(attention_q_mat, attention_k_mat.transpose(1, 2)) / math.sqrt(self.d_k)
                    attention_mask = F.softmax(attention_mask, dim=2)
                    multishot_support_feat += [torch.bmm(attention_mask, s_feat)]

                multishot_support_feat = torch.stack(multishot_support_feat, 0)
                if self.shot_mode == 'max':
                    multihead_support_feat += [torch.max(multishot_support_feat, 0)[0]]
                elif self.shot_mode == 'mean':
                    multihead_support_feat += [torch.mean(multishot_support_feat, 0)]
            masked_support_feat = torch.cat(multihead_support_feat, 2)
            if self.n_head != 1:
                masked_support_feat = F.relu(self.multihead_layer(masked_support_feat))  # [B, hw, 256]
            masked_support_feat = masked_support_feat.transpose(1, 2).view(batch_size, self.fpn_dim, feat_h, feat_w)

            if self.attention_type == 'concat':
                output_pyramid_attentive_feat.append(torch.cat([fpn_q_feats[p], masked_support_feat], 1))
            elif self.attention_type == 'product':
                output_pyramid_attentive_feat.append(fpn_q_feats[p] * masked_support_feat)
        
        return output_pyramid_attentive_feat

    
    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_ims, all_cls_gt_boxes):

        im_batch = im_data
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        support_ims = support_ims.data
        batch_size = im_batch.size(0)

        x = self.resnet_base(im_batch)
        pos_sup_ims = support_ims[:, :self.num_shot, :, :, :].contiguous()
        pos_sup_feats = self.resnet_base(pos_sup_ims.view(
                                        batch_size * self.num_shot, 3, self.support_im_size, self.support_im_size))  # [B*S, 64, 80, 80]
        if self.training:
            neg_sup_ims = support_ims[:, self.num_shot:, :, :, :].contiguous()
            neg_sup_feats = self.resnet_base(neg_sup_ims.view(
                                            batch_size * self.num_shot, 3, self.support_im_size, self.support_im_size))  # [B*S, 64, 80, 80]
    
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        pos_s1 = self.layer1(pos_sup_feats)
        pos_s2 = self.layer2(pos_s1)
        pos_s3 = self.layer3(pos_s2)
        pos_s4 = self.layer4(pos_s3)
        if self.training:
            neg_s1 = self.layer1(neg_sup_feats)
            neg_s2 = self.layer2(neg_s1)
            neg_s3 = self.layer3(neg_s2)
            neg_s4 = self.layer4(neg_s3)

        features = self.fpn([x2, x3, x4])[1:]
        pos_s_features = self.fpn([pos_s2, pos_s3, pos_s4])[1:]
        if self.training:
            neg_s_features = self.fpn([neg_s2, neg_s3, neg_s4])[1:]

        correlation_features = self.attention_module(features, pos_s_features)
        if self.training:
            neg_correlation_features = self.attention_module(features, neg_s_features)

        regression = torch.cat([self.regressionModel(feat) for feat in correlation_features], dim=1)  # [1, n_proposal, 4]
        classification = torch.cat([self.classificationModel(feat) for feat in correlation_features], dim=1)  # [1, n_proposal, n_class]
        if self.training:
            neg_regression = torch.cat([self.regressionModel(feat) for feat in neg_correlation_features], dim=1)
            neg_classification = torch.cat([self.classificationModel(feat) for feat in neg_correlation_features], dim=1)

        anchors = self.anchors(im_batch)

        rpn_loss_cls = torch.zeros(1).cuda()
        rpn_loss_bbox = torch.zeros(1).cuda()
        RCNN_loss_cls = torch.zeros(1).cuda()
        RCNN_loss_bbox = torch.zeros(1).cuda()
        rois_label = torch.zeros(10).cuda()
        rois = None
        cls_prob = None
        bbox_pred = None

        if self.training:
            pos_RCNN_loss_cls, RCNN_loss_bbox = self.focalLoss(classification, regression, anchors, gt_boxes)
            empty_gt_boxes = torch.zeros_like(gt_boxes) - 1.
            neg_RCNN_loss_cls, _ = self.focalLoss(neg_classification, neg_regression, anchors, empty_gt_boxes)
            RCNN_loss_cls = (pos_RCNN_loss_cls + neg_RCNN_loss_cls) / 2

            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, im_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()


            # scores = torch.squeeze(classification[:, :, 1])
            scores = F.softmax(classification[0], 1)[:, 1]
            scores_over_thresh = (scores > 0.5)
            if scores_over_thresh.sum() == 0:
                return None, None, np.array([0., 0., 0., 0.])

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))  # [n_predict]
            finalAnchorBoxesIndexesValue = torch.tensor([1] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))  # [n_predict]
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))  # [n_predict, 4]

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=1024, max_len=49):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return x


def AARetinaNet(num_classes, n_head, attention_type, shot_mode, num_way=2, num_shot=5, pos_encoding=True, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
    """
    model = aa_retinanet(num_classes, Bottleneck, [3, 4, 6, 3], n_head=n_head, attention_type=attention_type, 
                        shot_mode=shot_mode, num_way=num_way, num_shot=num_shot, pos_encoding=pos_encoding, pretrained=pretrained)
    return model