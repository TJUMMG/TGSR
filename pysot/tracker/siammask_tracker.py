# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

from pioneer.utils.processing_utils import iou as get_iou



# GRU
class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

    def iou_anchor(self, bbox):
        # bbox = [5.,10.,20.,30.]
        x, y, w, h = bbox
        bbox1 = np.array([x - w / 10, y - h / 10, w, h])
        bbox2 = np.array([x, y - h / 10, w, h])
        bbox3 = np.array([x + w / 10, y - h / 10, w, h])
        bbox4 = np.array([x - w / 10, y, w, h])
        bbox5 = np.array([x + w / 10, y, w, h])
        bbox6 = np.array([x - w / 10, y + h / 10, w, h])
        bbox7 = np.array([x, y + h / 10, w, h])
        bbox8 = np.array([x + w / 10, y + h / 10, w, h])

        bbox_new = np.array([bbox1, bbox2, bbox3, bbox4, bbox5, bbox6, bbox7, bbox8])
        # [x,y,w,h] to [x1,y1,x2,y2]
        bbox_new[:, 2:4] = bbox_new[:, 0:2] + bbox_new[:, 2:4]
        for i in range(bbox_new.shape[0]):
            if bbox_new[i, 0] < 0:
                bbox_new[i, 0] = 0
            if bbox_new[i, 1] < 0:
                bbox_new[i, 1] = 0
            if bbox_new[i, 2] > 255:
                bbox_new[i, 2] = 255
            if bbox_new[i, 3] > 255:
                bbox_new[i, 3] = 255

        # [x1,y1,x2,y2] to [x,y,w,h]
        bbox_new[:, 2:4] = bbox_new[:, 2:4] - bbox_new[:, 0:2]
        return bbox_new

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
        self.pos_on_x_crop = torch.zeros([1, 7, 2]).cuda()
        self.wh_on_x_crop = torch.zeros([1, 7, 2]).cuda()
        self.idx = 1


        # 计算出目标在z_crop上的坐标(x,y,w,h), 并初始化IoUNet
        bbox_list = [127./2-crop_scare[0], 127./2-crop_scare[1], crop_scare[0], crop_scare[1]]
        z_bbox = torch.tensor(bbox_list).unsqueeze(0).cuda()

        # trainable_params = []
        # trainable_params += [{'params': self.model.iou_pred.iou_predictor.parameters(),
        #                       'lr': 6e-5}]
        # trainable_params += [{'params': self.model.iou_pred.fc3_rt.parameters(),
        #                       'lr': 6e-5}]
        # trainable_params += [{'params': self.model.iou_pred.fc4_rt.parameters(),
        #                       'lr': 6e-5}]
        # optimizer = torch.optim.Adam(trainable_params)
        # # 优化两次
        # for i in range(2):
        #     optimizer.zero_grad()
        #     loss = self.model.iou_pred.init(z_backbone[2:4], z_bbox)
        #     if i==0:
        #         loss.backward(retain_graph=True)
        #     else:
        #         loss.backward()
        #     optimizer.step()

        self.z_feat = self.model.iou_pred.get_modulation(z_backbone[2:4], z_bbox)

    def track(self, img, gt):
        """
        args:
            img(np.ndarray): BGR image
            gt: [cx, cy, w, h]
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
        s_x = round(s_x)


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
        if self.idx > 7:
            # 判断是否运动幅度过大
            cx_v = torch.zeros([6]).cuda()
            cy_v = torch.zeros([6]).cuda()
            for i in range(6):
                cx_v[i] = self.pos_on_x_crop[0, i + 1, 0] - self.pos_on_x_crop[0, i, 0]
            for i in range(6):
                cy_v[i] = self.pos_on_x_crop[0, i + 1, 1] - self.pos_on_x_crop[0, i, 1]
            if torch.mean(cx_v) < 60 and torch.mean(cy_v) < 60:
                cx_ = self.pos_on_x_crop[0,1:4+3,0]
                cy_ = self.pos_on_x_crop[0,1:4+3,1]
                w_ = self.wh_on_x_crop[0,1:4+3,0]
                h_ = self.wh_on_x_crop[0,1:4+3,1]
                data = torch.stack((cx_, cy_,cx_v,cy_v,w_,h_)).transpose(1,0).unsqueeze(0)
                pred_pos_cx, pred_pos_cy = self.model.pred(data)

                # 作用在搜索区域上
                dx_ = (pred_pos_cx - 255 / 2).detach().cpu().numpy()
                dy_ = (pred_pos_cy - 255 / 2).detach().cpu().numpy()
                # 注意在mgrid的输出中，坐标x_grid实际是负责h方向，坐标y_grid实际是负责h方向！
                # 这与通常的x与w对应不同！
                # 可记做(x_grid,y_grid) 与 (h,w) 相对
                x_grid, y_grid = np.mgrid[-127:127:255j, -127:127:255j]
                z = 1 / (2 * np.pi * (sigma ** 2)) \
                    * np.exp(-((x_grid - dy_) ** 2 + (y_grid - dx_) ** 2) / (2 * sigma ** 2))
                z = z / np.max(z)
                z = torch.tensor(z).cuda().type(torch.float)
            else:
                z = torch.ones([255, 255]).cuda().type(torch.float)
        else:
            z = torch.ones([255, 255]).cuda().type(torch.float)



        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]


        outputs = self.model.track(x_crop*(1+weight*z))

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
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

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

        self.pos_on_x_crop[0,self.idx % 7, 0] = torch.tensor(tr_cx_x_crop)
        self.pos_on_x_crop[0,self.idx % 7, 1] = torch.tensor(tr_cy_x_crop)
        self.wh_on_x_crop[0,self.idx % 7, 0] = torch.tensor(tr_w_x_crop)
        self.wh_on_x_crop[0,self.idx % 7, 1] = torch.tensor(tr_h_x_crop)

        # iou_pred
        bbox_list = [tr_cx_x_crop-tr_w_x_crop/2, tr_cy_x_crop-tr_h_x_crop/2, tr_w_x_crop, tr_h_x_crop]
        x_bbox = torch.tensor(bbox_list).reshape(1,1,4).float().cuda()
        x_feat = self.model.iou_pred.get_iou_feat(outputs['x_backbone_feat'][2:4])
        iou = self.model.iou_pred.predict_iou(self.z_feat, x_feat, x_bbox)

        # Refine BBox
        do_refine = False # sign
        if iou < self.iou_threshold:
            # 使用Refine Moudule
            bbox_for_refine = torch.tensor(bbox_list).to(x_crop.device).unsqueeze(0)
            bbox_x_crop = self.model.refine_module.forward_track(self.model.zf, outputs['x_f'], bbox_for_refine)
            x, y, w, h = bbox_x_crop[0], bbox_x_crop[1],bbox_x_crop[2],bbox_x_crop[3]
            cx, cy = x + w/2, y + h/2

            # 重新预测一下iou
            x_bbox = torch.tensor(bbox_x_crop).reshape(1, 1, 4).float().cuda()
            pred_refine_iou = self.model.iou_pred.predict_iou(self.z_feat, x_feat, x_bbox) 

            if pred_refine_iou > iou:
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
                # print('refine', self.idx)

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

        self.idx += 1

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

        # processing mask
        # np.unravel_index(ind_max, A.shape)返回索引对应的坐标值
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()

        return {
                'bbox': bbox,  # bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
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
