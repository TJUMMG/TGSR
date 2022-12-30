# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

from pioneer.utils.processing_utils import iou as get_iou


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox, dataset):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        # acquire TIR parameter
        self.dataset = dataset
        self.weight = cfg.TRACK[self.dataset]['WEIGHT']
        self.sigma = cfg.TRACK[self.dataset]['SIGMA']
        self.iou_threshold = cfg.TRACK[self.dataset]['IOU']

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop, crop_scare, _ = self.get_subwindow_scale(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        # 取出backbone的最后两块特征,
        # SiamRPN++的backbone feature channel 依次为 512，1024，2048,
        # SiamMask的backbone feature channel 依次为 64, 256, 512，1024
        z_backbone, _ = self.model.template(z_crop)

        # 保存目标在x_crop上的坐标
        self.traj_len_delta = 0
        self.pos_on_x_crop = torch.zeros([1, 7+self.traj_len_delta, 2]).cuda()
        self.wh_on_x_crop = torch.zeros([1, 7+self.traj_len_delta, 2]).cuda()
        self.idx = 1


        # 计算出目标在z_crop上的坐标(x,y,w,h), 并初始化IoUNet
        bbox_list = [127./2-crop_scare[0], 127./2-crop_scare[1], crop_scare[0], crop_scare[1]]
        z_bbox = torch.tensor(bbox_list).unsqueeze(0).cuda()

        self.z_feat = self.model.iou_pred.get_modulation(z_backbone[0:2], z_bbox)


    def track(self, img, gt=0):
        """
        args:
            img(np.ndarray): BGR image
            gt(np.ndarray): 4BBox:[x,y,w,h]  8BBox
        return:
            bbox(list):[x, y, width, height]
        """
        sigma = self.sigma
        weight = self.weight

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop, crop_scare, crop_bbox = self.get_subwindow_scale(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        self.crop_scare = crop_scare   # [scare_w, scare_y] 实际图像大小/预先固定大小
        self.pre_center_pos = self.center_pos

        # 计算当前帧目标中心在搜索区域的位置,gt
        dy = (gt[1]-self.center_pos[1])/crop_scare[1]
        dx = (gt[0]-self.center_pos[0])/crop_scare[0]
        gt_cx_x_crop = 255/2 + dx
        gt_cy_x_crop = 255/2 + dy
        gt_w_x_crop = gt[2] / crop_scare[0]
        gt_h_x_crop = gt[3] / crop_scare[1]


        # 进行轨迹预测
        if self.idx > 7+self.traj_len_delta:
            # 判断是否运动幅度过大
            cx_v = torch.zeros([6+self.traj_len_delta]).cuda()
            cy_v = torch.zeros([6+self.traj_len_delta]).cuda()
            for i in range(6+self.traj_len_delta):
                cx_v[i] = self.pos_on_x_crop[0, i + 1, 0] - self.pos_on_x_crop[0, i, 0]
            for i in range(6+self.traj_len_delta):
                cy_v[i] = self.pos_on_x_crop[0, i + 1, 1] - self.pos_on_x_crop[0, i, 1]
            if torch.mean(cx_v) < 60 and torch.mean(cy_v) < 60:
                cx_ = self.pos_on_x_crop[0,1:4+3+self.traj_len_delta,0]
                cy_ = self.pos_on_x_crop[0,1:4+3+self.traj_len_delta,1]
                w_ = self.wh_on_x_crop[0,1:4+3+self.traj_len_delta,0]
                h_ = self.wh_on_x_crop[0,1:4+3+self.traj_len_delta,1]
                data = torch.stack((cx_, cy_, cx_v, cy_v, w_, h_)).transpose(1,0).unsqueeze(0)
                pred_pos_cx, pred_pos_cy = self.model.pred(data)

                # 作用在搜索区域上
                dx_ = (pred_pos_cx - 255 / 2).detach().cpu().numpy()
                dy_ = (pred_pos_cy - 255 / 2).detach().cpu().numpy()
                # 注意在mgrid的输出中，坐标x_grid实际是负责h方向，坐标y_grid实际是负责h方向！
                # 这与通常的x与w对应不同！
                # 可记做(x_grid,y_grid) 与 (h,w) 相对
                x_grid, y_grid = np.mgrid[-127:127:255j, -127:127:255j]
                z = 1 / (2 * np.pi * (sigma ** 2)) * \
                    np.exp(-((x_grid - dy_) ** 2 + (y_grid - dx_) ** 2) / (2 * sigma ** 2))

                z = z / np.max(z)
                z = torch.tensor(z).cuda().type(torch.float)
            else:
                z = torch.ones([255, 255]).cuda().type(torch.float)
        else:
            z = torch.ones([255, 255]).cuda().type(torch.float)

        outputs = self.model.track(x_crop * (1 + weight * z))

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # cfg.TRACK.PENALTY_K
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR #cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        original_bbox = bbox

        # 计算当前帧目标中心在搜索区域的位置，跟踪结果
        dy = (self.center_pos[1]-self.pre_center_pos[1])/self.crop_scare[1]
        dx = (self.center_pos[0]-self.pre_center_pos[0])/self.crop_scare[0]
        tr_cx_x_crop = float(255/2 + dx)
        tr_cy_x_crop = float(255/2 + dy)
        tr_w_x_crop = bbox[2] / self.crop_scare[0]
        tr_h_x_crop = bbox[3] / self.crop_scare[1]

        self.pos_on_x_crop[0,self.idx % (7+self.traj_len_delta), 0] = torch.tensor(tr_cx_x_crop)
        self.pos_on_x_crop[0,self.idx % (7+self.traj_len_delta), 1] = torch.tensor(tr_cy_x_crop)
        self.wh_on_x_crop[0,self.idx % (7+self.traj_len_delta), 0] = torch.tensor(tr_w_x_crop)
        self.wh_on_x_crop[0,self.idx % (7+self.traj_len_delta), 1] = torch.tensor(tr_h_x_crop)

        # iou_pred
        bbox_list = [tr_cx_x_crop-tr_w_x_crop/2, tr_cy_x_crop-tr_h_x_crop/2, tr_w_x_crop, tr_h_x_crop]
        x_bbox = torch.tensor(bbox_list).reshape(1,1,4).float().cuda()
        x_feat = self.model.iou_pred.get_iou_feat(outputs['x_backbone_feat'][0:2]) # outputs['x_backbone_feat']
        iou = self.model.iou_pred.predict_iou(self.z_feat, x_feat, x_bbox)

        # Refine BBox
        do_refine = False # sign
        if iou < self.iou_threshold:
        
            # 使用Refine Moudule
            bbox_for_refine = torch.tensor(bbox_list).to(x_crop.device).unsqueeze(0)
            bbox_x_crop = self.model.refine_module.forward_track(
                self.model.zf[-1], outputs['x_f'][-1], bbox_for_refine) # new_xf outputs['x_f']
            x, y, w, h = bbox_x_crop[0], bbox_x_crop[1],bbox_x_crop[2],bbox_x_crop[3]
            cx, cy = x + w/2, y + h/2
        
            # 重新预测一下iou
            x_bbox = torch.tensor(bbox_x_crop).reshape(1, 1, 4).float().cuda()
            pred_refine_iou = self.model.iou_pred.predict_iou(self.z_feat, x_feat, x_bbox) # x_feat
        
        
            if pred_refine_iou > iou:
                do_refine = True
                # 从x_crop转变到img
                cx_img = (cx - float(255 / 2)) * self.crop_scare[0] + self.pre_center_pos[0]
                cy_img = (cy - float(255 / 2)) * self.crop_scare[1] + self.pre_center_pos[1]
                w_img = w * self.crop_scare[0]
                h_img = h * self.crop_scare[1]
        
                bbox = [cx_img - w_img / 2,
                        cy_img - h_img / 2,
                        w_img,
                        h_img]
                self.center_pos = np.array([cx_img, cy_img])
                self.size = np.array([w_img, h_img])
        
                # for traj pred, 重新计算当前帧目标中心在搜索区域的位置，跟踪结果
                dy = (self.center_pos[1] - self.pre_center_pos[1]) / self.crop_scare[1]
                dx = (self.center_pos[0] - self.pre_center_pos[0]) / self.crop_scare[0]
                tr_cx_x_crop = float(255 / 2 + dx)
                tr_cy_x_crop = float(255 / 2 + dy)
                tr_w_x_crop = bbox[2] / self.crop_scare[0]
                tr_h_x_crop = bbox[3] / self.crop_scare[1]
        
                self.pos_on_x_crop[0, self.idx % 7, 0] = torch.tensor(tr_cx_x_crop)
                self.pos_on_x_crop[0, self.idx % 7, 1] = torch.tensor(tr_cy_x_crop)
                self.wh_on_x_crop[0, self.idx % 7, 0] = torch.tensor(tr_w_x_crop)
                self.wh_on_x_crop[0, self.idx % 7, 1] = torch.tensor(tr_h_x_crop)


        self.idx += 1 # 使用轨迹预测与后期修复时

        # 获取最终输出结果和GT的真实IoU
        gt_tensor = torch.tensor(gt).to(x_crop.device).reshape([1, -1])
        bbox_tensor = torch.tensor(original_bbox).to(x_crop.device).reshape([1, -1])
        gt_original_iou = get_iou(gt_tensor, bbox_tensor)


        if do_refine == True:
            # 进行了Refine

            refine_bbox_on_img = bbox
            # 获取最终输出结果和GT的真实IoU
            gt_tensor = torch.tensor(gt).to(x_crop.device).reshape([1, -1])
            bbox_tensor = torch.tensor(refine_bbox_on_img).to(x_crop.device).reshape([1, -1])
            gt_refine_iou = get_iou(gt_tensor, bbox_tensor).item()
            pred_refine_iou = pred_refine_iou.item()

        else:
            refine_bbox_on_img = -1
            pred_refine_iou = -1
            gt_refine_iou = -1

        return {
                'bbox': bbox,
                'best_score': best_score,
                'crop_bbox': crop_bbox,
                'bbox_on_crop': [tr_cx_x_crop-tr_w_x_crop/2,
                                 tr_cy_x_crop-tr_h_x_crop/2,
                                 tr_w_x_crop, tr_h_x_crop],
                'gt_on_crop': [gt_cx_x_crop-gt_w_x_crop/2,
                               gt_cy_x_crop-gt_h_x_crop/2,
                               gt_w_x_crop, gt_h_x_crop],
                'gt_iou': gt_original_iou.item(),
                'pred_iou': iou.item(),
                'is_refine': do_refine,
                'refine_bbox_on_img': refine_bbox_on_img,
                'original_bbox': original_bbox,
                'pred_refine_iou': pred_refine_iou,
                'gt_refine_iou': gt_refine_iou
               }