import torch.nn as nn
import torch
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

import pioneer.utils.processing_utils as prutils


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class LinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes, input_sz, bias=True, batch_norm=True, relu=True):
        super().__init__()
        self.linear = nn.Linear(in_planes*input_sz*input_sz, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)

class AtomIoUNet(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=(128,256), pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        super().__init__()
        # _r for reference, _t for test
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)

        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(3, 3, 15/127)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 31/255)

        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 15/127)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 31/255)

        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

        self.iou_predictor = nn.Linear(pred_inter_dim[0]+pred_inter_dim[1], 1, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1).
                # So we use the same initialization here.
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

        assert bb1.dim() == 3
        assert proposals2.dim() == 4

        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]

        # Extract first train sample
        feat1 = [f[0,...] if f.dim()==5 else f.reshape(-1, num_sequences, *f.shape[-3:])[0,...] for f in feat1]
        bb1 = bb1[0,...]

        # Get modulation vector
        modulation = self.get_modulation(feat1, bb1)

        iou_feat = self.get_iou_feat(feat2)

        modulation = [f.reshape(1, num_sequences, -1).repeat(num_images, 1, 1).reshape(num_sequences*num_images, -1) for f in modulation]

        proposals2 = proposals2.reshape(num_sequences*num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.reshape(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * fc34_3_r.reshape(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.reshape(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.reshape(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)

        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (batch, 4)."""

        feat3_r, feat4_r = feat

        c3_r = self.conv3_1r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        # torch.arange(start, end)的结果并不包含end, Tensor.to()关于type和device的类型转换
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)  # [batch_index, x0,y0,x1,y1] dim[batch,5]

        roi3r = self.prroi_pool3r(c3_r, roi1)

        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [f.reshape(-1, *f.shape[-3:]) if f.dim()==5 else f for f in feat2]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t

    def _generate_proposals(self, box):
        # 来自pytracking/ltr/data/processing.py
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = 10 # self.proposal_params['boxes_per_frame']
        min_iou = 0.01
        sigma_factor = [0.01, 0.05, 0.1, 0.2, 0.3]

        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=min_iou,
                                                             sigma_factor=sigma_factor)

        # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1

        # Map to [-1, 1]
        gt_iou = gt_iou/10
        return proposals, gt_iou

    def init(self, feat, bb):
        '''
        利用两次梯度下降，使网络参数更适合特定目标
        Args:
            feat: z_crop的backbone特征提取最后两个
            bb: 初始帧bbox，相对于z_crop

        Returns: loss

        '''

        z_feat = self.get_modulation(feat, bb)
        proposals, gt_iou = self._generate_proposals(bb[0].cpu())
        x_feat = self.get_iou_feat(feat)
        pred_iou = self.predict_iou(z_feat, x_feat, proposals.cuda().unsqueeze(0))

        objective = nn.MSELoss()
        loss = objective(gt_iou.cuda(), pred_iou)
        return loss




if __name__ == '__main__':
    test_iou = AtomIoUNet(input_dim=(512, 1024), pred_input_dim=(256,256), pred_inter_dim=(256, 256))
    test_iou.load_state_dict(torch.load('/media/HardDisk_new/wh/second_code/pysot-master_0526/AtomIoUNet50.pth'))
    test_iou.cuda().eval()
    # print('success')
    # for name, para in test.conv3_1r.named_parameters():
    #     print(name, para)

    z_feature_3 = torch.randn([1,512,15,15]).cuda()
    z_feature_4 = torch.randn([1, 1024, 15, 15]).cuda()
    z_bbox_list = [63 - 68. / 2, 63 - 59. / 2, 68., 59.]
    z_bbox = torch.tensor(z_bbox_list).unsqueeze(0).cuda()

    x_feature_3 = torch.randn([1, 512, 31, 31]).cuda()
    x_feature_4 = torch.randn([1, 1024, 31, 31]).cuda()
    x_bbox_list = [108.-68./2, 123.-59./2, 68., 59.]
    x_bbox = torch.tensor(x_bbox_list).reshape(1,1,4).cuda()

    z_feat = test_iou.get_modulation([z_feature_3, z_feature_4], z_bbox)
    x_feat = test_iou.get_iou_feat([x_feature_3, x_feature_4])
    iou_pred = test_iou.predict_iou(z_feat, x_feat, x_bbox)
    print(iou_pred)

