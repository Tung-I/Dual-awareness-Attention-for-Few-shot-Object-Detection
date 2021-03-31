import torch
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class FSODInferenceLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write(self, save_step, gt, support, predict, save_im=False):
        # self._add_scalars(save_step, train_log)
        if save_im:
            self._add_images(save_step, gt, support, predict)
  
    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_images(self, save_step, gt, support, predict):
        # gt = gt.cpu()
        # support = support.cpu()
        # predict = predict.cpu()
        # H, W = gt[0].size(1), gt[0].size(2)
        # support_img = support[i].permute(1, 2, 0).numpy()
        # support_img = support_img[:, :, ::-1].copy()
        gt_grid = make_grid(gt, nrow=1, normalize=True, scale_each=True, pad_value=1)
        support_grid = make_grid(support, nrow=1, normalize=True, scale_each=True, pad_value=1)
        pred_grid = make_grid(predict, nrow=1, normalize=True, scale_each=True, pad_value=1)

        grid = torch.cat((gt_grid, support_grid, pred_grid), dim=-1)
        self.writer.add_image('gt&pred', grid, save_step)


class FSODLogger:
    def __init__(self, log_dir, train_shot=5):
        self.writer = SummaryWriter(log_dir)
        self.train_shot = train_shot

    def write(self, save_step, train_log, query=None, support=None, boxes=None, save_im=False):
        self._add_scalars(save_step, train_log)
        if save_im:
            self._add_images(save_step, query, support, boxes)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, save_step, train_log):
      for key in train_log.keys():
        self.writer.add_scalars(key, {'train': train_log[key]}, save_step)

    def _add_images(self, save_step, query, supports, boxes):
        query = query.cpu()
        support = supports[:, 0, :, :, :].cpu()
        neg_support = supports[:, self.train_shot, :, :, :].cpu()
        boxes = boxes.cpu()
        query_ims = []
        support_ims = []
        neg_support_ims = []
        H, W = query[0].size(1), query[0].size(2)
        for i in range(query.size(0)):
            query_im = query[i].permute(1, 2, 0).numpy()
            support_im = support[i].permute(1, 2, 0).numpy()
            neg_support_im = neg_support[i].permute(1, 2, 0).numpy()
            query_im = query_im[:, :, ::-1].copy()
            support_im = support_im[:, :, ::-1].copy()
            neg_support_im = neg_support_im[:, :, ::-1].copy()
            boxes_of_one_img = boxes[i]

            for ii in range(boxes_of_one_img.size(0)):
                box = boxes_of_one_img[ii]
                if box[4] == 0:
                    continue
                x = box[0]
                y = box[1]
                w = box[2] - box[0]
                h = box[3] - box[1]
                query_im = cv2.rectangle(np.array(query_im), (int(x), int(y)), (int(x+w), int(y+h)), (220, 0, 50), 2)

            query_im = torch.from_numpy(query_im).permute(2,0,1)
            support_im = cv2.resize(support_im, (W, H), interpolation=cv2.INTER_LINEAR)
            neg_support_im = cv2.resize(neg_support_im, (W, H), interpolation=cv2.INTER_LINEAR)
            support_im = torch.from_numpy(support_im).permute(2,0,1)
            neg_support_im = torch.from_numpy(neg_support_im).permute(2,0,1)

            query_ims += [query_im]
            support_ims += [support_im]
            neg_support_ims += [neg_support_im]

        query_ims = torch.stack(query_ims, 0)
        support_ims = torch.stack(support_ims, 0)
        neg_support_ims = torch.stack(neg_support_ims, 0)

        train_query = make_grid(query_ims, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_support = make_grid(support_ims, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_support_2 = make_grid(neg_support_ims, nrow=1, normalize=True, scale_each=True, pad_value=1)
        grid = torch.cat((train_query, train_support, train_support_2), dim=-1)
        self.writer.add_image('train', grid, save_step)

class BaseLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write(self, save_step, gt, support, predict):
        # self._add_scalars(save_step, train_log)
        self._add_images(save_step, gt, support, predict)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_images(self, save_step, gt, support, predict):
        # gt = gt.cpu()
        # support = support.cpu()
        # predict = predict.cpu()
        # H, W = gt[0].size(1), gt[0].size(2)
        # support_img = support[i].permute(1, 2, 0).numpy()
        # support_img = support_img[:, :, ::-1].copy()
        gt_grid = make_grid(gt, nrow=1, normalize=True, scale_each=True, pad_value=1)
        support_grid = make_grid(support, nrow=1, normalize=True, scale_each=True, pad_value=1)
        pred_grid = make_grid(predict, nrow=1, normalize=True, scale_each=True, pad_value=1)

        gt_pred_grid = torch.cat((gt_grid, pred_grid), dim=-1)
        self.writer.add_image('gt&pred', gt_pred_grid)
        self.writer.add_image('support', support_grid)

