# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck

from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from pioneer.core.traj_predict import Predict_FC, Predict_GRU, Predict_LSTM, Predict_STLSTM
# from pioneer.traj_predict_train import Predict_FC, Predict_GRU, Predict_LSTM, Predict_STLSTM

from pioneer.core.AtomIoUNet import AtomIoUNet
from pioneer.core.RefineNet import RefineNet
from pioneer.core.DR_Loss import SigmoidDRLoss


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        # 轨迹预测
        # self.pred = Predict_STLSTM(batchsize=1)  # STLSTM 需指定batchsize
        self.pred = Predict_GRU()

        # iou_pred, input_dim参数针对ResNet50
        self.iou_pred = AtomIoUNet(input_dim=(512, 1024), pred_input_dim=(256,256), pred_inter_dim=(256, 256))

        # Refine bbox
        self.refine_module = RefineNet(input_channels=256)

        # PrRoiPool
        self.pr_roi_pool = PrRoIPool2D(7, 7, 31/255)

    def template(self, z):
        zf = self.backbone(z)
        zf_backbone = zf.copy()
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        # 为将提取x_crop特征集成于此处，
        # 添加判断zf是否为模板特征的条件
        if zf[-1].shape[-1] == 7:
            self.zf = zf
        return zf_backbone, zf

    def track(self, x):
        xf = self.backbone(x)
        xf_backbone = xf.copy()
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        # factor = 0.001
        # gmask_ = torch.nn.functional.interpolate(gmask.reshape([1, 1, 255, 255]), (31, 31))
        # xf = [xf_ * (1 + factor * gmask_) for xf_ in xf]

        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'x_backbone_feat': xf_backbone,
                'x_f': xf
               }


    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs

    def forward_iou(self,data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_bbox = data['template_bbox'].cuda().unsqueeze(0)
        proposals_bbox = data['proposals_bbox'].cuda().unsqueeze(0)
        proposals_iou = data['proposals_iou'].cuda()
        search_bbox = data['search_bbox'].cuda()

        # # simulate the trajectory prediction
        # BatchSize = template.shape[0]
        # search_bbox_center = (search_bbox[:, :2] + search_bbox[:, 2:]/2).cpu()
        # # offset is uniform random para between [-8, 8]
        # offset_w = torch.rand([BatchSize])
        # offset_h = torch.rand([BatchSize])
        # delta_center_w = search_bbox_center[:, 0] -255/2 + (offset_w * 16) - 8
        # delta_center_h = search_bbox_center[:, 1] -255/2 + (offset_h * 16) - 8
        #
        # # 注意在mgrid的输出中，坐标x实际是负责h方向，坐标y实际是负责h方向！
        # # 这与通常的x与w对应不同！
        # x, y = np.mgrid[-127:127:255j, -127:127:255j]
        # x_tensor = torch.tensor(x).repeat([BatchSize,1,1]) - delta_center_h.reshape([BatchSize,1,1])
        # y_tensor = torch.tensor(y).repeat([BatchSize,1,1]) - delta_center_w.reshape([BatchSize,1,1])
        #
        # sigma = 120
        # guass_mask = 1 / (2 * np.pi * (sigma ** 2)) * \
        #     torch.exp(-((x_tensor) ** 2 + (y_tensor) ** 2) / (2 * sigma ** 2))
        #
        # wight = 0.005
        # guass_mask = (guass_mask/torch.max(guass_mask)).to(search.device).float().unsqueeze(1)
        # search_weighted = search*(1+wight*guass_mask)



        zf = self.backbone(template)
        xf = self.backbone(search)  # search_weighted
        # z_feat3, z_feat4 = zf[:2]
        # x_feat3, x_feat4 = xf[:2]

        if cfg.MASK.MASK:
            # 仅用于判断是否为SiamMask
            pred_iou = self.iou_pred(zf[2:4], xf[2:4], template_bbox, proposals_bbox)
        else:
            # SiamRPN++
            pred_iou = self.iou_pred(zf[:2],xf[:2],template_bbox,proposals_bbox)

        loss = nn.MSELoss()
        loss_mse = loss(pred_iou, proposals_iou)

        # 将回归问题转换成分类问题,分类的编号从1开始
        proposals_iou = proposals_iou.reshape([-1])
        proposals_iou_class_1 = torch.ones_like(proposals_iou)   # iou <= 0.5
        proposals_iou_class_2 = torch.ones_like(proposals_iou) * 2   # iou > 0.5
        proposals_iou_class = torch.where(proposals_iou>0.5, proposals_iou_class_2, proposals_iou_class_1)

        pred_iou_ = pred_iou.reshape([-1,1])
        # 考虑DRLoss中对pred_iou使用sigmod()将范围变到[0,1],
        # 此处将概率预测范围变为[-0.5,0.5]
        pred_iou_class = torch.cat([-(pred_iou_-0.5), pred_iou_-0.5],dim=1)

        # 应用DRLoss
        DRLoss = SigmoidDRLoss()
        loss_dr = DRLoss(pred_iou_class, proposals_iou_class)


        return {'loss': 3*loss_mse + loss_dr,
                'loss_mse': loss_mse,
                'loss_drloss': loss_dr}, \
               pred_iou

    def forward_refine(self,data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_bbox = data['template_bbox'].cuda().unsqueeze(0)
        proposals_bbox = data['proposals_bbox'].cuda().unsqueeze(0)
        proposals_iou = data['proposals_iou'].cuda()
        search_bbox = data['search_bbox'].cuda()

        _, xf = self.template(search) # search_weighted
        _, zf = self.template(template)

        proposals_bbox_ = proposals_bbox[0,:,0,:]

        if cfg.MASK.MASK:
            # SiamMask:
            output = self.refine_module.forward_train(zf, xf,
                                                      proposals_bbox_,
                                                      search_bbox)
        else:
            # SiamRPN++
            output = self.refine_module.forward_train(zf[-1], xf[-1],
                                                      proposals_bbox_,
                                                      search_bbox)

        return{'loss': output['loss_reg'] + output['loss_iou'],
               'loss_reg': output['loss_reg'],
               'loss_iou': output['loss_iou']}
