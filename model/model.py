import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .agc_layer import AGCLayer as InputEncoding
from .selectscale_hc import SelectscaleHyperConv
from .selectframe_tc import SelectframeTemConv
from .utils import *
from graph.nturgbd import *
from graph.kinetics import *



class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class SelectSTHCBlock(nn.Module):
    '''Selective Spatial Temporal Hypergraph Convolution Block.'''
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, stride=1, pool_channels=None, residual=True):
        super(SelectSTHCBlock, self).__init__()
        self.stride = stride
        self.gcn1 = SelectscaleHyperConv(in_channels, out_channels, A, G_part, G_body, num_point, num_frame)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU()
        if self.stride == 2:
                    self.pool = SelectframeTemConv(*pool_channels)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, N):
        res = self.residual(x)
        x = self.gcn1(x)
        x = self.tcn1(x)
        
        # NM, C, T, V = x.size()
        # if self.stride == 2:
        #     x, indices = self.pool(x, N)

        #     T_new = (T+1)//2
        #     pre_res = res.view(N, NM//N, C, T, V)
        #     res = torch.empty((N, NM//N, C, T_new, V), device=x.device)
        #     for i in range(N):
        #         res[i] = pre_res[i,:,:,indices[i],:]
        #     res = res.view(NM, C, T_new, V)

        x = x + res
        return self.relu(x)


class Model(nn.Module):
    '''Selective-HCN Model.'''
    def __init__(self, num_class=120, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.hypergraph = Hypergraph()

        A = self.graph.A
        G_part = self.hypergraph.G_part
        G_body = self.hypergraph.G_body
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = InputEncoding(3, 64, A, residual=False)
        self.l2 = SelectSTHCBlock(64, 64, A, G_part, G_body, num_point, 300)
        # self.l3 = SelectSTHCBlock(64, 64, A, G_part, G_body, num_point, 300)
        self.l4 = SelectSTHCBlock(64, 64, A, G_part, G_body, num_point, 300)
        self.l5 = SelectSTHCBlock(64, 128, A, G_part, G_body, num_point, 300, stride=2, pool_channels=[128, 300])
        self.l6 = SelectSTHCBlock(128, 128, A, G_part, G_body, num_point, 150)
        # self.l7 = SelectSTHCBlock(128, 128, A, G_part, G_body, num_point, 150)
        self.l8 = SelectSTHCBlock(128, 256, A, G_part, G_body, num_point, 150, stride=2, pool_channels=[256, 150])
        self.l9 = SelectSTHCBlock(256, 256, A, G_part, G_body, num_point, 75)
        self.l10 = SelectSTHCBlock(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x, N)
        x = self.l3(x, N)
        x = self.l4(x, N)
        x = self.l5(x, N)
        x = self.l6(x, N)
        x = self.l7(x, N)
        x = self.l8(x, N)
        x = self.l9(x, N)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
