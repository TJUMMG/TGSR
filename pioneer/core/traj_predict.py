import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import warnings

from pioneer.core.STLSTM import STLSTM

class Predict_FC(nn.Module):
    def __init__(self):
        super(Predict_FC, self).__init__()
        self.pred_cx = torch.nn.Sequential(
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.pred_cy = torch.nn.Sequential(
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, center_post):
        '''

        :param center_post: [BatchSize, Seq_len, 6]  (cx,cy,vx,vy,gtw,gth)
        :return: [cx,cy]
        '''
        cx_concat = center_post[:, :, 0]
        cy_concat = center_post[:, :, 1]
        cx_pred = self.pred_cx(cx_concat)
        cy_pred = self.pred_cy(cy_concat)

        return cx_pred,cy_pred

class Predict_GRU(nn.Module):
    def __init__(self):
        super(Predict_GRU, self).__init__()
        self.pred_cx = torch.nn.GRU(4, 10, 2, batch_first=True)
        self.cx_linear = torch.nn.Linear(10, 1)
        self.pred_cy = torch.nn.GRU(4, 10, 2, batch_first=True)
        self.cy_linear = torch.nn.Linear(10, 1)

    def forward(self, center_post):
        '''
        :param center_post: [BatchSize, Seq_len, 6]  (cx,cy,vx,vy,gtw,gth)
        :return: [cx,cy]
        '''
        cx_concat = center_post[:,:, 0]
        cy_concat = center_post[:,:, 1]
        vx_concat = center_post[:,:, 2]
        vy_concat = center_post[:,:, 3]
        w_concat = center_post[:, :, 4]
        h_concat = center_post[:, :, 5]
        input_x = torch.stack((cx_concat, vx_concat, w_concat, h_concat), dim=-1)
        input_y = torch.stack((cy_concat, vy_concat, w_concat, h_concat), dim=-1)
        _, hn_x = self.pred_cx(torch.tensor(input_x))
        _, hn_y = self.pred_cy(torch.tensor(input_y))
        cx_pred = self.cx_linear(hn_x[-1])
        cy_pred = self.cy_linear(hn_y[-1])

        return cx_pred,cy_pred

class Predict_LSTM(nn.Module):
    def __init__(self):
        super(Predict_LSTM, self).__init__()
        self.pred_cx = torch.nn.LSTM(4, 10, 2, batch_first=True)
        self.cx_linear = torch.nn.Linear(10, 1)
        self.pred_cy = torch.nn.LSTM(4, 10, 2, batch_first=True)
        self.cy_linear = torch.nn.Linear(10, 1)

    def forward(self, center_post):
        '''

        :param center_post: [BatchSize, Seq_len, 6]  (cx,cy,vx,vy,gtw,gth)
        :return: [cx,cy]
        '''
        cx_concat = center_post[:,:, 0]
        cy_concat = center_post[:,:, 1]
        vx_concat = center_post[:,:, 2]
        vy_concat = center_post[:,:, 3]
        w_concat = center_post[:, :, 4]
        h_concat = center_post[:, :, 5]
        input_x = torch.stack((cx_concat, vx_concat, w_concat, h_concat), dim=-1)
        input_y = torch.stack((cy_concat, vy_concat, w_concat, h_concat), dim=-1)
        _, (hn_x,cn_x) = self.pred_cx(torch.tensor(input_x))
        _, (hn_y,cn_y) = self.pred_cy(torch.tensor(input_y))
        cx_pred = self.cx_linear(hn_x[-1])
        cy_pred = self.cy_linear(hn_y[-1])

        return cx_pred,cy_pred

class Predict_STLSTM(nn.Module):
    def __init__(self, batchsize=12):
        super(Predict_STLSTM, self).__init__()
        self.pred_cx = STLSTM(4, 10, 2, batchsize)
        self.cx_linear = torch.nn.Linear(10, 1)
        self.pred_cy = STLSTM(4, 10, 2, batchsize)
        self.cy_linear = torch.nn.Linear(10, 1)

    def forward(self, center_post):
        '''
        :param center_post: [BatchSize, Seq_len, 6]  (cx,cy,vx,vy,gtw,gth)
        :return: [cx,cy]
        '''
        cx_concat = center_post[:,:, 0]
        cy_concat = center_post[:,:, 1]
        vx_concat = center_post[:,:, 2]
        vy_concat = center_post[:,:, 3]
        w_concat = center_post[:, :, 4]
        h_concat = center_post[:, :, 5]
        input_x = torch.stack((cx_concat, vx_concat, w_concat, h_concat), dim=-1)
        input_y = torch.stack((cy_concat, vy_concat, w_concat, h_concat), dim=-1)
        hn_x, cn_x, mn_x = self.pred_cx(torch.tensor(input_x))
        hn_y, cn_y, mn_y = self.pred_cy(torch.tensor(input_y))
        cx_pred = self.cx_linear(hn_x)
        cy_pred = self.cy_linear(hn_y)

        return cx_pred, cy_pred

class Data(Dataset):
    def __init__(self, data_path, seq_len):
        # 载入数据
        with open(data_path, 'r') as file:
            data = file.readlines()
        self.data_name = []
        self.data_cx = []
        self.data_cy = []
        self.data_gtw = []
        self.data_gth = []
        for data_i in data:
            # name, cx, cy = data_i.split()
            name, gtcx, gtcy, gtw, gth, trcx, trcy, trw, trh = data_i.split(',')
            self.data_name.append(name)
            self.data_cx.append(gtcx)
            self.data_cy.append(gtcy)
            self.data_gtw.append(gtw)
            self.data_gth.append(gth)

        self.data_lens = len(self.data_cx)

        # 定义序列长度
        self.seq_len = seq_len

    def __getitem__(self, index):

        while True:
            # 对index做一些限制
            if index > (self.data_lens - (self.seq_len + 2)):
                index = self.data_lens - (self.seq_len + 2)
            if self.data_name[index] != self.data_name[index + self.seq_len + 1]:
                index = index - (self.seq_len + 1)

            # 判断是否运动幅度过大
            cx = np.array([float(i) for i in self.data_cx[index:index + self.seq_len + 2]])
            cy = np.array([float(i) for i in self.data_cy[index:index + self.seq_len + 2]])
            gtw = np.array([float(i) for i in self.data_gtw[index:index + self.seq_len + 2]])
            gth = np.array([float(i) for i in self.data_gth[index:index + self.seq_len + 2]])
            cx_v = np.zeros([self.seq_len])
            cy_v = np.zeros([self.seq_len])
            for i in range(len(cx_v) - 1):
                cx_v[i] = cx[i + 1] - cx[i]
            for i in range(len(cy_v) - 1):
                cy_v[i] = cy[i + 1] - cy[i]
            if np.mean(cx_v) < 60 and np.mean(cy_v) < 60:
                cx_ = cx[1:self.seq_len + 1]
                cy_ = cy[1:self.seq_len + 1]
                gtw = gtw[1:self.seq_len + 1]
                gth = gth[1:self.seq_len + 1]
                data = np.stack((cx_, cy_, cx_v, cy_v, gtw, gth)).T.astype(np.float32)
                label = np.array([cx[self.seq_len + 1], cy[self.seq_len + 1]]).astype(np.float32)
                break
            else: # 重新选
                index = int(np.random.choice(self.data_lens - (self.seq_len + 2), 1))

        return data, label

    def __len__(self):
        return self.data_lens


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # set the parameter
    # torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    training_data_path = '../data/cxcy_uav123.txt'
    batch_num = 12
    seq_len = 6
    epoch_num = 20
    model_save_path = './model_test'
    # pretrain_model_path = './model_lstm2_wh_uav123/checkpoint_19.pth' # None or path
    pretrain_model_path = None  # None or path
    FLAG = 'Eval'  # Train or Eval

    # define predict net and load pretrained model
    pred = Predict_STLSTM().cuda()
    if pretrain_model_path != None:
        pred.load_state_dict(torch.load(pretrain_model_path))

    # Dataloader and Loss function
    dataset = Data(data_path=training_data_path, seq_len=seq_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_num)
    criterion = torch.nn.MSELoss(reduction='mean')

    if FLAG == 'Train':
        pred.train()
        for epoch in range(epoch_num):
            # 用于查看loss
            loss_data = torch.zeros([1])
            for idx, (batch_data, batch_label) in enumerate(dataloader):
                cx_pred, cy_pred = pred(batch_data.cuda())

                loss_x = criterion(cx_pred, batch_label[:, 0].cuda())
                loss_y = criterion(cy_pred, batch_label[:, 1].cuda())
                loss = loss_x + loss_y

                # 手动定义一个随epoch衰减的lr
                lrs = np.logspace(-7, -5, epoch_num)
                lr = float(lrs[epoch_num - epoch - 1])

                optim = torch.optim.SGD(params=pred.parameters(), lr=lr)
                optim.zero_grad()
                loss.backward()
                optim.step()

                loss_data += loss

            print(epoch, loss_data / (dataset.data_lens / batch_num))

            # save model
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(pred.state_dict(),
                       os.path.join(model_save_path, 'checkpoint_{}.pth'.format(epoch+1)))
    else:
        # 测试
        pred.eval()
        loss_ = torch.zeros([1])
        for idx, (batch_data, batch_label) in enumerate(dataloader):
            cx_pred, cy_pred = pred(batch_data.cuda())
            loss_x = criterion(cx_pred, batch_label[:, 0].cuda())
            loss_y = criterion(cy_pred, batch_label[:, 1].cuda())
            loss = loss_x + loss_y
            loss_ += loss

        print(loss_ / (dataset.data_lens / batch_num))