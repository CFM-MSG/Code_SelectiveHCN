import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import *

class SelectscaleHyperConv(nn.Module):
    '''Select-scale Hypergraph Convolution (SHC).'''
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, coff_embedding=4, num_subset=3):
        super(SelectscaleHyperConv, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

        self.G_part = nn.Parameter(torch.from_numpy(G_part.astype(np.float32)))
        self.G_body = nn.Parameter(torch.from_numpy(G_body.astype(np.float32)))
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.GloAvgMat = torch.ones(25, 25) / 25
        self.num_subset = num_subset
        self.num_point = num_point

        self.conv_joint = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_joint.append(nn.Conv2d(in_channels, out_channels, 1))
        
        self.PE = PositionalEncoding(in_channels, num_point, num_frame, 'spatial')
        self.conv_theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_nonlocal = nn.Conv2d(in_channels, out_channels, 1)
        
        self.conv_part = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_body = nn.Conv2d(in_channels, out_channels, 1)
        '''self.conv_avg = nn.Conv2d(in_channels, out_channels, 1)'''

        # selective-scale.
        self.num_branch = 3
        d = int(out_channels / 2)
        self.fc = nn.Linear(out_channels, d)
        self.fc_branch = nn.ModuleList([])
        for i in range(3):
            self.fc_branch.append(
                nn.Linear(d, out_channels)
            )
        self.softmax = nn.Softmax(dim=1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            # elif isinstance(m, nn.Linear):
            #     fc_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_joint[i], self.num_subset)

    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = torch.eye(self.num_point).to(device=A.device) * D_list_12
        A = torch.matmul(A, D_12)
        return A

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        G_part = self.G_part.cuda(x.get_device())
        G_body = self.G_body.cuda(x.get_device())
        Avg = self.GloAvgMat.cuda(x.get_device())

        x_joint = None
        for i in range(self.num_subset):
            x_temp = x.view(N, C * T, V)
            x_temp = self.conv_joint[i](torch.matmul(x_temp, self.norm(A[i])).view(N, C, T, V))
            x_joint = x_temp + x_joint if x_joint is not None else x_temp

        x_withPE = self.PE(x)
        theta = self.conv_theta(x_withPE).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        phi = self.conv_phi(x_withPE).view(N, self.inter_c * T, V)
        att_map = self.soft(torch.matmul(theta, phi) / theta.size(-1))  # N V V
        x_nonlocal = x.view(N, C * T, V)
        x_nonlocal = torch.matmul(x_nonlocal, att_map).view(N, C, T, V)
        x_nonlocal = self.conv_nonlocal(x_nonlocal)

        x_part = x.view(N, C * T, V)
        x_part = torch.matmul(x_part, self.norm(G_part)).view(N, C, T, V)
        x_part = self.conv_part(x_part)
        x_body = x.view(N, C * T, V)
        x_body = torch.matmul(x_body, self.norm(G_body)).view(N, C, T, V)
        x_body = self.conv_body(x_body)
        """
        x_avg = x.view(N, C * T, V)
        x_avg = torch.matmul(x_avg, Avg).view(N, C, T, V)
        x_avg = self.conv_avg(x_avg)
        y = x_joint + x_part + x_body + x_nonlocal + x_avg
        """

        # selective-scale.
        x_joint = x_joint.unsqueeze_(dim=1)
        x_part = x_part.unsqueeze_(dim=1)
        x_body = x_body.unsqueeze_(dim=1)
        
        x_local = torch.cat([x_joint, x_part, x_body], dim=1)
        x_sum = torch.sum(x_local, dim=1)
        glo_avg = x_sum.mean(-1).mean(-1)
        feature_z = self.fc(glo_avg)

        attention_vectors = None
        for i, fc in enumerate(self.fc_branch):
            vector = fc(feature_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        x_local_selected = (x_local * attention_vectors).sum(dim=1)
        
        y = x_local_selected + x_nonlocal
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)