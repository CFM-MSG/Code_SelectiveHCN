import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class SelectframeTemConv(nn.Module):
    '''Selective-frame Temporal Convolution (STC).'''
    def __init__(self,in_spa,in_tem,ratio=0.5):
        super(SelectframeTemConv, self).__init__()
        self.in_spa = in_spa
        self.in_tem = in_tem
        self.ratio = ratio
        self.ch_reduce = nn.Sequential(
                            nn.Conv2d(in_spa, 1, 1),
                            nn.BatchNorm2d(1),
                            nn.ReLU(),
                        )
        self.spa_reduce = nn.Sequential(
                            nn.Conv2d(25, 1, 1),
                            nn.BatchNorm2d(1),
                            nn.ReLU(),
                        )

        self.fc_tem = nn.Sequential(
                nn.Linear(in_tem, in_tem),
                nn.Tanh(),
                nn.Linear(in_tem, in_tem),
                nn.Tanh(),
                nn.Linear(in_tem, in_tem),
        )
        self.drop = nn.Dropout(p=0.8)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, x_in, N):
        NM, C, T, V = x_in.size()
        x_in = x_in.view(N, NM//N, C, T, V)
        x = x_in
        x = x.mean(dim=1)  # N C T V

        x = self.ch_reduce(x)
        x = x.permute(0,3,2,1).contiguous()
        x = self.spa_reduce(x)
        x = x.view(N, -1)
        
        # attention.
        x = self.fc_tem(x)
        scores = self.sigmoid(x)  # N T
        
        # Top-k selection.
        T_new = int(T*self.ratio)
        top_scores, indices = scores.topk(T_new, dim=1, largest=True, sorted=False)
        top_scores = top_scores.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        if T_new < T:
            x_out = torch.empty((N, NM//N, C, T_new, V), device=x_in.device)
            for i in range(N):
                x_out[i] = x_in[i,:,:,indices[i],:]
        else:
            x_out = x_in
            
        x_out = (x_out*top_scores).view(NM, C, T_new, V)
        
        return x_out, indices


