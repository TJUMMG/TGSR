import torch
import torch.nn as nn

class STLSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(STLSTMCell, self).__init__()
        self.conv_x = nn.Linear(input_size,hidden_size*7)
        self.conv_h = nn.Linear(hidden_size,hidden_size*4)
        self.conv_m = nn.Linear(hidden_size,hidden_size*3)
        self.conv_o = nn.Linear(hidden_size*2,hidden_size)
        self.conv_last = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input, hidden_state, cell_state, memory_state):
        # input [batch, input_size]
        x_concat = self.conv_x(input)
        h_concat = self.conv_h(hidden_state)
        m_concat = self.conv_m(memory_state)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.hidden_size, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_size, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_size, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)
        c_t = f_t * cell_state + i_t * g_t

        i_t_m = torch.sigmoid(i_x + i_m)
        f_t_m = torch.sigmoid(f_x + f_m)
        g_t_m = torch.tanh(g_x + g_m)
        m_t = f_t_m * memory_state + i_t_m * g_t_m

        mem = torch.cat((c_t, m_t), -1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_t = o_t * torch.tanh(self.conv_last(mem))

        return h_t, c_t, m_t

class STLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layer, batchsize):
        super(STLSTM, self).__init__()
        # 构建网络层
        self.num_layer = num_layer
        cell_list=[]
        cell_list.append(STLSTMCell(input_size, hidden_size))
        for i in range(num_layer-1):
            cell_list.append(STLSTMCell(hidden_size,hidden_size))
        self.cell_list = nn.ModuleList(cell_list)

        self.hidden_state = torch.zeros(batchsize, hidden_size)
        self.cell_state = torch.zeros(batchsize, hidden_size)
        self.memory_state = torch.zeros(batchsize, hidden_size)

    def forward(self, inputs):

        device = inputs.device

        self.hidden_state = self.hidden_state.to(device=device)
        self.cell_state = self.cell_state.to(device=device)
        self.memory_state = self.memory_state.to(device=device)

        # 原方案使用对self.hidden_state循环赋值，可以测试，但无法训练
        h0 = [self.hidden_state[:, :]]
        h1 = [self.hidden_state[:, :]]
        c0 = [self.cell_state[:, :]]
        c1 = [self.cell_state[:, :]]
        m = [self.memory_state]

        for seq in range(inputs.shape[1]):
            input = inputs[:, seq, :]
            # 对单个输入的计算input[batch,seq_len,input_size]
            for i in range(self.num_layer):
                if i == 0:
                    a,b,x = \
                        self.cell_list[i](input, h0[-1], c0[-1], m[-1])
                    h0.append(a), c0.append(b), m.append(x)
                else:
                    a,b,x = \
                        self.cell_list[i](h0[-1], h1[-1], c1[-1], m[-1])
                    h1.append(a), c1.append(b), m.append(x)

        return h1[-1], c1[-1], m[-1]


    def forward_(self, inputs):

        # 原方案
        for seq in range(inputs.shape[1]):
            input = inputs[:, seq, :]
            # 对单个输入的计算input[batch,seq_len,input_size]
            for i in range(self.num_layer):
                if i == 0:
                    self.hidden_state[:, i, :], self.cell_state[:, i, :], self.memory_state = \
                        self.cell_list[i](input, self.hidden_state[:, i, :], self.cell_state[:, i, :],
                                          self.memory_state)
                else:
                    self.hidden_state[:, i, :], self.cell_state[:, i, :], self.memory_state = \
                        self.cell_list[i](self.hidden_state[:, i, :], self.hidden_state[:, i, :], self.cell_state[:, i, :],
                                          self.memory_state)

        return self.hidden_state, self.cell_state, self.memory_state


if __name__ == '__main__':
    batchsize =10
    # stlstm_cell = STLSTMCell(4, 10)
    # x = torch.randn([1, 4])
    # h = torch.randn([1, 10])
    # c = torch.randn([1, 10])
    # m = torch.randn([1, 10])
    #
    # h_t, c_t, m_t = stlstm_cell(x, h, c, m)

    stlstm = STLSTM(4, 10, 2, batchsize)
    # stlstm.cuda()
    x = torch.randn([batchsize, 7, 4])
    a = stlstm(x)
    a =1





