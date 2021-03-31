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


class tripleA_retinanet(nn.Module):

    def __init__(self, num_classes, block, layers, n_head=1, attention_type='concat', shot_mode='mean', 
                    num_way=2, num_shot=5, pos_encoding=True, pretrained=False, gamma=0.1):
        super(tripleA_retinanet, self).__init__()
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.pretrained = pretrained
        self.inplanes = 64
        self.n_head = n_head
        self.attention_type = attention_type
        self.shot_mode = shot_mode
        self.num_shot = num_shot
        self.pos_encoding = pos_encoding
        self.support_im_size = 320
        self.gamma = gamma
    
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
        Q_list = []
        K_list = []
        M_list = []
        self.d_k = 128
        self.fpn_dims = [512, 1024, 2048]
        for fpn_dim in self.fpn_dims:
            Q_weight = nn.Linear(fpn_dim, self.d_k)
            K_weight = nn.Linear(fpn_dim, self.d_k)
            M_weight = nn.Linear(fpn_dim, 1)
            init.normal_(Q_weight.weight, std=0.01)
            init.constant_(Q_weight.bias, 0)
            init.normal_(K_weight.weight, std=0.01)
            init.constant_(K_weight.bias, 0)
            init.normal_(M_weight.weight, std=0.01)
            init.constant_(M_weight.bias, 0)
            Q_list.append(Q_weight)
            K_list.append(K_weight)
            M_list.append(M_weight)
        self.pyramid_Q_layers = nn.ModuleList(Q_list)
        self.pyramid_K_layers = nn.ModuleList(K_list)
        self.pyramid_M_layers = nn.ModuleList(M_list)
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
            q_feat = q_feat.view(batch_size, cur_feat_dim, -1).permute(0, 2, 1).contiguous()  # [B, *, 256]
            s_feat_nshot = fpn_s_feats[p].view(batch_size, self.num_shot, cur_feat_dim, -1).transpose(2, 3).contiguous()  # [B, S, *, 256]

            attention_q_mat = self.pyramid_Q_layers[p](q_feat)
            multishot_support_feat = []
            for i in range(self.num_shot):
                s_feat = s_feat_nshot[:, i, :, :]
                if self.pos_encoding:
                    s_feat = self.pos_encoding_layers[p](s_feat)

                attention_k_mat = self.pyramid_K_layers[p](s_feat)
                attention_q_mat = attention_q_mat - attention_q_mat.mean(1, keepdim=True)
                attention_k_mat = attention_k_mat - attention_k_mat.mean(1, keepdim=True)
                attention_mask = torch.bmm(attention_q_mat, attention_k_mat.transpose(1, 2)) / math.sqrt(self.d_k)
                attention_mask = F.softmax(attention_mask, dim=2)
                unary_term = self.pyramid_M_layers[p](s_feat)
                unary_term = F.softmax(unary_term, dim=1)
                attention_mask = attention_mask + unary_term.transpose(1, 2) * self.gamma
                multishot_support_feat += [torch.bmm(attention_mask, s_feat)]

            multishot_support_feat = torch.stack(multishot_support_feat, 0)
            if self.shot_mode == 'max':
                masked_support_feat = torch.max(multishot_support_feat, 0)[0]
            elif self.shot_mode == 'mean':
                masked_support_feat = torch.mean(multishot_support_feat, 0)

            masked_support_feat = masked_support_feat.transpose(1, 2).view(batch_size, cur_feat_dim, feat_h, feat_w)

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


def TripleARetinaNet(num_classes, n_head, attention_type, shot_mode, num_way=2, num_shot=5, pos_encoding=True, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
    """
    model = tripleA_retinanet(num_classes, Bottleneck, [3, 4, 6, 3], n_head=n_head, attention_type=attention_type, 
                        shot_mode=shot_mode, num_way=num_way, num_shot=num_shot, pos_encoding=pos_encoding, pretrained=pretrained)
    return model