import numpy as np
import torch
from torch.utils.data import DataLoader
from datasload import LoadData


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, middle_channels=6, out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels, middle_channels, 1),
            torch.nn.BatchNorm1d(middle_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(middle_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(torch.nn.Sequential):
    def __init__(self, layer_num, growth_rate, in_channels, middele_channels=6):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels + i * growth_rate, middele_channels, growth_rate)
            self.add_module('denselayer%d' % (i), layer)


class Transition(torch.nn.Sequential):
    def __init__(self, channels):
        super(Transition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm1d(channels))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv1d(channels, channels // 2, 3, padding=1))
        self.add_module('Avgpool', torch.nn.AvgPool1d(2))


class MultiAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        '''
        :param embed_dim: 嵌入特征个数
        :param num_heads: scale dot-product attention层数
        '''
        super(MultiAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w_q = [torch.nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        self.w_k = [torch.nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        self.w_v = [torch.nn.Linear(embed_dim, embed_dim) for i in range(num_heads)]
        self.w_o = torch.nn.Linear(embed_dim * num_heads, embed_dim)
        self.softmax = torch.nn.Softmax()

    def single_head(self, q, k, v, head_idx):
        '''scale dot-scale attention '''
        q = self.w_q[head_idx](q)
        k = self.w_k[head_idx](k)
        v = self.w_v[head_idx](v)
        out = torch.matmul(torch.matmul(q, k.permute(0, 2, 1)), v) / self.embed_dim
        return out

    def forward(self, q, k, v):
        output = []
        for i in range(self.num_heads):
            out = self.single_head(q, k, v, i)
            output.append(out)
        output = torch.cat(output, dim=2)
        output = self.w_o(output)
        print(output.shape)
        return output


class DenseNet(torch.nn.Module):
    def __init__(self, layer_num=(6, 12, 24, 16), growth_rate=32, init_features=64, in_channels=3, middele_channels=6,
                 classes=2, epicenter_dis=[100]):
        super(DenseNet, self).__init__()
        self.epicenter_distance = epicenter_dis
        self.feature_channel_num = init_features
        self.conv = torch.nn.Conv1d(in_channels, self.feature_channel_num, 7, 2)
        self.norm = torch.nn.BatchNorm1d(self.feature_channel_num)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(3, 3)

        self.DenseBlock1 = DenseBlock(layer_num[0], growth_rate, self.feature_channel_num, middele_channels)
        self.feature_channel_num = self.feature_channel_num + layer_num[0] * growth_rate
        self.Transition1 = Transition(self.feature_channel_num)

        self.DenseBlock2 = DenseBlock(layer_num[1], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[1] * growth_rate
        self.Transition2 = Transition(self.feature_channel_num)

        self.DenseBlock3 = DenseBlock(layer_num[2], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[2] * growth_rate
        self.Transition3 = Transition(self.feature_channel_num)

        self.DenseBlock4 = DenseBlock(layer_num[3], growth_rate, self.feature_channel_num // 2, middele_channels)
        self.feature_channel_num = self.feature_channel_num // 2 + layer_num[3] * growth_rate

        # 多头注意力机制
        self.MultiAttn = MultiAttention(self.feature_channel_num, num_heads=8)

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num + 1, self.feature_channel_num),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num, classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)

        x = self.DenseBlock2(x)
        x = self.Transition2(x)

        x = self.DenseBlock3(x)
        x = self.Transition3(x)

        x = self.DenseBlock4(x)

        x = self.avgpool(x)

        # 融合震中距特征融合震中距特征
        # dis = torch.tensor(np.array(self.epicenter_distance))
        # dis = torch.flatten(dis)
        # dis = dis.resize(1,1,1)
        # print(x.shape)
        #x = torch.flatten(x)
        x = x.view(len(x),-1)
        x = torch.cat((x, self.epicenter_distance), dim=1)
        x = self.classifer(torch.sigmoid(x))

        return x


if __name__ == '__main__':
    print("……………………………………………………………………test modul……………………………………………………………………………………………………")
    # input = torch.randn(size=(1,3,1000))
    dislist = torch.randn(size=(64, 1))
    model = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=32, in_channels=3, classes=2,epicenter_dis=dislist)
    # output = model(input)
    # print(model)
    # print("……………………………………………………………………summary……………………………………………………………………………………………………")
    #
    # summary(model, (3,1000), device='cpu')

    eval_dataloader =torch.utils.data.DataLoader(LoadData("data_set"), shuffle=True, batch_size=64)
    for k, data in enumerate(eval_dataloader):
        stream, label = data
        output = model(stream.to(torch.device("cpu")))
        print(output)