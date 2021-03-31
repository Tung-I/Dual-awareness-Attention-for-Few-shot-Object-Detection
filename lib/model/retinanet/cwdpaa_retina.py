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


class cwdpaa_retinanet(nn.Module):

    def __init__(self, num_classes, block, layers, attention_type='concat', reduce_dim=400, gamma=0.1, 
                    num_way=2, num_shot=5, pos_encoding=True, pretrained=False):
        super(cwdpaa_retinanet, self).__init__()
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.pretrained = pretrained
        self.inplanes = 64
        self.attention_type = attention_type
        self.num_shot = num_shot
        self.pos_encoding = pos_encoding
        self.support_im_size = 320
        self.reduce_dim = reduce_dim
        self.channel_gamma = gamma
        self.unary_gamma = 0.1
    
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

        attention_output_dim = 256 if self.attention_type == 'product' else 512
        if self.attention_type == 'product':
            self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], feature_size=attention_output_dim)  # [512, 1024, 2048]
        else:
            self.fpn = PyramidFeatures(fpn_sizes[0]*2, fpn_sizes[1]*2, fpn_sizes[2]*2, feature_size=attention_output_dim)

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
        unary_list = []
        adapt_q_list = []
        adapt_k_list = []
        channel_k_list = []
        self.fpn_dims = [512, 1024, 2048]
        for fpn_dim in self.fpn_dims:
            unary_layer = nn.Linear(fpn_dim, 1)
            init.normal_(unary_layer.weight, std=0.01)
            init.constant_(unary_layer.bias, 0)

            adapt_q_layer = nn.Linear(fpn_dim, reduce_dim)
            init.normal_(adapt_q_layer.weight, std=0.01)
            init.constant_(adapt_q_layer.bias, 0)

            adapt_k_layer = nn.Linear(fpn_dim, reduce_dim)
            init.normal_(adapt_k_layer.weight, std=0.01)
            init.constant_(adapt_k_layer.bias, 0)

            channel_k_layer = nn.Linear(fpn_dim, 1)
            init.normal_(channel_k_layer.weight, std=0.01)
            init.constant_(channel_k_layer.bias, 0)

            unary_list.append(unary_layer)
            adapt_q_list.append(adapt_q_layer)
            adapt_k_list.append(adapt_k_layer)
            channel_k_list.append(channel_k_layer)
        self.unary_layers = nn.ModuleList(unary_list)
        self.adapt_Q_layers = nn.ModuleList(adapt_q_list)
        self.adapt_K_layers = nn.ModuleList(adapt_k_list)
        self.channel_K_layers = nn.ModuleList(channel_k_list)
        if self.pos_encoding:
            pel_3 = PositionalEncoding(d_model=512, max_len=40*40)
            pel_4 = PositionalEncoding(d_model=1024, max_len=20*20)
            pel_5 = PositionalEncoding(d_model=2048, max_len=10*10)
        self.pos_encoding_layers = nn.ModuleList([pel_3, pel_4, pel_5])

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
            q_feat = fpn_q_feats[p]
            cur_feat_dim = q_feat.size(1)
            feat_h = q_feat.size(2)
            feat_w = q_feat.size(3)
            q_feat = q_feat.view(batch_size, cur_feat_dim, -1).permute(0, 2, 1).contiguous()  # [B, *, c]
            s_feat_nshot = fpn_s_feats[p].view(batch_size, self.num_shot, cur_feat_dim, -1).transpose(2, 3).contiguous()  # [B, S, *, c]

            dense_support_feature = []
            q_matrix = self.adapt_Q_layers[p](q_feat)  # [B, hw, c']
            q_matrix = q_matrix - q_matrix.mean(1, keepdim=True)
            for i in range(self.num_shot):
                s_feat = s_feat_nshot[:, i, :, :]
                if self.pos_encoding:
                    s_feat = self.pos_encoding_layers[p](s_feat)

                # support channel enhance
                support_spatial_weight = self.channel_K_layers[p](s_feat)  # [B, nn, 1]
                support_spatial_weight = F.softmax(support_spatial_weight, 1)
                support_channel_global = torch.bmm(support_spatial_weight.transpose(1, 2), s_feat)  # [B, 1, c]
                s_feat = s_feat + self.channel_gamma * support_channel_global

                # support adaptive attention
                k_matrix = self.adapt_K_layers[p](s_feat)  # [B, nn, c']
                k_matrix = k_matrix - k_matrix.mean(1, keepdim=True)
                support_adaptive_attention_weight = torch.bmm(q_matrix, k_matrix.transpose(1, 2)) / math.sqrt(self.reduce_dim)  # [B, hw, nn]
                support_adaptive_attention_weight = F.softmax(support_adaptive_attention_weight, dim=2)
                unary_term = self.unary_layers[p](s_feat)  # [B, nn, 1]
                unary_term = F.softmax(unary_term, dim=1)
                support_adaptive_attention_weight = support_adaptive_attention_weight + self.unary_gamma * unary_term.transpose(1, 2)  # [B, hw, nn]
                support_adaptive_attention_feature = torch.bmm(support_adaptive_attention_weight, s_feat)  # [B, hw, c]

                dense_support_feature += [support_adaptive_attention_feature]

            dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)  # [B, hw, c]
            dense_support_feature = dense_support_feature.transpose(1, 2).view(batch_size, cur_feat_dim, feat_h, feat_w)

            if self.attention_type == 'concat':
                output_pyramid_attentive_feat.append(torch.cat([fpn_q_feats[p], dense_support_feature], 1))
            elif self.attention_type == 'product':
                output_pyramid_attentive_feat.append(fpn_q_feats[p] * dense_support_feature)

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
    
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        pos_s1 = self.layer1(pos_sup_feats)
        pos_s2 = self.layer2(pos_s1)
        pos_s3 = self.layer3(pos_s2)
        pos_s4 = self.layer4(pos_s3)

        correlation_features = self.attention_module([x2, x3, x4], [pos_s2, pos_s3, pos_s4])
        # ignore the output of layer 3
        fpn_output_features = self.fpn(correlation_features)[1:]
        
        regression = torch.cat([self.regressionModel(feat) for feat in fpn_output_features], dim=1)  # [1, n_proposal, 4]
        classification = torch.cat([self.classificationModel(feat) for feat in fpn_output_features], dim=1)  # [1, n_proposal, n_class]

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
            RCNN_loss_cls, RCNN_loss_bbox = self.focalLoss(classification, regression, anchors, gt_boxes)

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
            scores_over_thresh = (scores >= 0.5)
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


def CWDPAARetinaNet(num_classes, attention_type, reduce_dim, gamma, num_way=2, num_shot=5, pos_encoding=True, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
    """
    model = cwdpaa_retinanet(num_classes, Bottleneck, [3, 4, 6, 3], attention_type=attention_type, reduce_dim=reduce_dim, gamma=gamma,
                        num_way=num_way, num_shot=num_shot, pos_encoding=pos_encoding, pretrained=pretrained)
    return model