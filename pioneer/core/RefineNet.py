import torch
import torch.nn as nn
import torch.nn.functional as F
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from pioneer.utils.processing_utils import iou as get_iou

class RefineNet(nn.Module):
    def __init__(self, input_channels):
        super(RefineNet, self).__init__()
        # self.z_conv1 = nn.Conv2d(input_channels*2, 256, 3, 1, 1)
        # self.z_conv2 = nn.Conv2d(256, 128, 3, 1, 1)
        # self.z_x_conv3 = nn.Conv2d(128, 64, 3, 1)
        # self.z_x_conv4 = nn.Conv2d(64, 32, 3, 1)
        # self.z_x_conv5 = nn.Conv2d(32, 16, 3, 1)
        # self.z_x_conv6 = nn.Conv2d(16, 1, 3, 1)

        self.z_conv = nn.Sequential(
            nn.Conv2d(input_channels * 2, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

        self.z_x_fc = nn.Sequential(
            nn.Linear(25*25, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )

        self.z_y_fc = nn.Sequential(
            nn.Linear(25*25, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )

        self.pr_roi_pool = PrRoIPool2D(7, 7, 31/255)

    def get_pred_delta(self, z_f, r_f):
        B, C, H, W = z_f.shape
        net_input = torch.cat([z_f, r_f], dim=1)
        transition = self.z_conv(net_input)
        # transition_w = self.z_w_conv(transition)
        # pred_delta_w = torch.sigmoid(self.z_w_fc(transition_w.reshape([B,-1])))
        # transition_h = self.z_h_conv(transition)
        # pred_delta_h = torch.sigmoid(self.z_w_fc(transition_h.reshape([B,-1])))
        # transition_x = self.z_x_conv(transition)
        # pred_delta_x = torch.sigmoid(self.z_x_fc(transition_x.reshape([B,-1])))
        # transition_y = self.z_y_conv(transition)
        # pred_delta_y = torch.sigmoid(self.z_y_fc(transition_y.reshape([B,-1])))
        pred_delta_x = torch.sigmoid(self.z_x_fc(transition.reshape([B, -1])))
        pred_delta_y = torch.sigmoid(self.z_y_fc(transition.reshape([B, -1])))

        return torch.cat([pred_delta_x, pred_delta_y], dim=1)


    def refine_bbox(self, pred_delta, bbox):
        '''

        Args:
            pred_delta: [pred_delta_x, pred_delta_y, pred_delta_w, pred_delta_h]
            bbox: tensor [x, y, w, h]

        Returns: the refine bbox refer to RPN [refine_x, refine_y, refine_w, refine_h]

        '''
        # pred_delta_x, pred_delta_y, pred_delta_w, pred_delta_h = pred_delta.split([1]*4,dim=1)
        # x, y, w, h = bbox.split([1]*4,dim=1)  # 拆分
        # refine_x = pred_delta_x * w + x
        # refine_y = pred_delta_y * h + y
        # refine_w = torch.exp(pred_delta_w) * w
        # refine_h = torch.exp(pred_delta_h) * h

        pred_delta_x, pred_delta_y = pred_delta.split([1]*2,dim=1)
        x, y, w, h = bbox.split([1]*4,dim=1)  # 拆分
        refine_x = pred_delta_x * w + x
        refine_y = pred_delta_y * h + y
        refine_w = w
        refine_h = h
        return torch.cat([refine_x, refine_y, refine_w, refine_h],dim=1)

    def get_true_delta(self, gt_bbox, bbox):

        x, y, w, h = bbox.split([1] * 4, dim=1)  # 拆分
        gt_x, gt_y, gt_w, gt_h = gt_bbox.split([1] * 4, dim=1)
        delta_x = (gt_x - x) / w
        delta_y = (gt_y - y) / h
        delta_w = torch.log(gt_w / w)
        delta_h = torch.log(gt_h / h)
        return torch.cat([delta_x, delta_y, delta_w, delta_h],dim=1)

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


    def forward_train_(self,z_f, x_f, bbox, gt_bbox):
        # z_f, r_f
        # bbox [B,4] [x,y,w,h]
        batch_size = bbox.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(bbox.device)
        bbox_for_roi = bbox.clone()
        bbox_for_roi[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
        roi = torch.cat((batch_index, bbox_for_roi), dim=1)

        r_f = self.pr_roi_pool(x_f, roi)
        pred_delta = self.get_pred_delta(z_f, r_f)
        refine_bbox = self.refine_bbox(pred_delta, bbox)
        ture_delta = self.get_true_delta(gt_bbox, bbox)
        loss_reg = F.smooth_l1_loss(pred_delta, ture_delta[:,:2])
        iou = get_iou(gt_bbox, refine_bbox)
        loss_iou = F.mse_loss(iou, torch.ones([batch_size]).to(iou.device))
        # loss = loss_iou + loss_reg

        return {'loss_reg': loss_reg,
                'loss_iou': loss_iou}

    def forward_train(self, z_f, x_f, bbox, gt_bbox):
        # resp_z, resp_r
        # bbox [B,4] [x,y,w,h]
        batch_size = bbox.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(z_f.device)
        bbox_for_roi = bbox.clone()
        bbox_for_roi[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
        roi = torch.cat((batch_index, bbox_for_roi), dim=1)
        r_f = self.pr_roi_pool(x_f, roi)

        resp_z = self.xcorr_depthwise(x_f, z_f)
        resp_r = self.xcorr_depthwise(x_f, r_f)

        pred_delta = self.get_pred_delta(resp_z, resp_r)
        ture_delta = self.get_true_delta(gt_bbox, bbox)
        refine_bbox = self.refine_bbox(pred_delta, bbox)
        loss_reg = F.smooth_l1_loss(pred_delta, ture_delta[:,:2])

        iou = get_iou(gt_bbox, refine_bbox)
        loss_iou = F.mse_loss(iou, torch.ones([batch_size]).to(iou.device))
        # loss = loss_iou + loss_reg

        return {'loss_reg': loss_reg,
                'loss_iou': loss_iou}

    def forward_track_(self, z_f, x_f, bbox):
        # z_f, r_f
        '''

        Args:
            z_f: template feature [1,256,7,7]
            x_f: template feature [1,256,31,31]
            bbox: list [x,y,w,h]

        Returns: refine bbox list [x,y,w,h]

        '''
        batch_size = bbox.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(z_f.device)
        bbox_for_roi = bbox.clone()
        bbox_for_roi[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
        roi = torch.cat((batch_index, bbox_for_roi), dim=1)

        r_f = self.pr_roi_pool(x_f, roi)
        pred_delta = self.get_pred_delta(z_f, r_f)
        refine_bbox = self.refine_bbox(pred_delta, bbox)

        return refine_bbox.tolist()[0]

    def forward_track(self, z_f, x_f, bbox):
        # resp_z, resp_r
        batch_size = bbox.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(z_f.device)
        bbox_for_roi = bbox.clone()
        bbox_for_roi[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
        roi = torch.cat((batch_index, bbox_for_roi), dim=1)
        r_f = self.pr_roi_pool(x_f, roi)

        resp_z = self.xcorr_depthwise(x_f, z_f)
        resp_r = self.xcorr_depthwise(x_f, r_f)

        pred_delta = self.get_pred_delta(resp_z, resp_r)
        refine_bbox = self.refine_bbox(pred_delta, bbox)

        return refine_bbox.tolist()[0]




if __name__  == '__main__':
    net = RefineNet(256).cuda(0)
    z_f = torch.randn([3, 256, 7, 7]).cuda(0)
    x_f = torch.randn([3, 256, 31, 31]).cuda(0)
    r_f = torch.randn([3, 256, 7, 7]).cuda(0)
    # pred_delta = net.get_pred_delta(z_f, r_f)

    bbox = torch.rand([3,4]).cuda(0)
    gt_bbox = torch.rand([3, 4]).cuda(0)
    loss = net.forward_train(z_f, x_f, bbox, gt_bbox)

    z_f_track = torch.randn([1, 256, 7, 7]).cuda(0)
    x_f_track = torch.randn([1, 256, 31, 31]).cuda(0)
    bbox_track = torch.rand([1, 4]).cuda(0)
    refine = net.forward_track(z_f_track, x_f_track, bbox_track)


    print(loss)

