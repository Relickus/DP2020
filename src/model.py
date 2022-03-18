import torch.nn as nn
import torch
from src.dataset import C
import torch.nn.functional as F
import numpy as np


class MyNetwork(nn.Module):
    def __init__(self, dropout=False):
        super(MyNetwork,self).__init__()
        self.drop = dropout

        self.conv1 = nn.Conv3d( in_channels=C, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d( in_channels=32, out_channels=16, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d( in_channels=16, out_channels=16, kernel_size=3)
        self.bn3 = nn.BatchNorm3d(16)

        # FC layers 128 -> 64 -> 1
        FC1_INSHAPE = 1152 # explicit def. of input shape, see https://discuss.pytorch.org/t/expected-object-of-backend-cuda-but-got-backend-cpu-for-argument-2-mat2-err/46475/2

        self.fc1 = nn.Linear(in_features=FC1_INSHAPE,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc3 = nn.Linear(in_features=64,out_features=1)

    def forward(self,input):

        # Conv layers
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.max_pool3d(F.relu(x),2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool3d(F.relu(x),2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool3d(F.relu(x),2)

        # flatten
        x = x.view(-1, np.prod(x.shape[1:]))

        # FC layers
        x = self.fc1(x)
        if self.drop: x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        if self.drop: x = F.dropout(x, p=0.1)
        x = self.fc3(x)

        # output sigmoid
        x = torch.sigmoid(x)

        return x.squeeze()

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device


class ISR(torch.nn.Module):

    def forward(self, preds):
        A = (preds / 1-preds).sum()
        return A / (1+A)

class NoisyOr(torch.nn.Module):
    """
    MIL aggregation layer. Nonparametric.
    """
    def forward(self, preds):
        return 1 - np.prod(1-preds)

class Max(torch.nn.Module):
    """
    Max aggregation layer
    """

    def forward(self, preds):
        return max(preds)


class NoisyAnd(torch.nn.Module):
    """
    MIL aggregation layer. Hyperparams: a. Trainable params: b.

    An implementation of MIL layer by Kraus et al.
    Takes N (variable batch size) predictions as input, one for each patch and outputs a
    scalar prediction aggregating the input predictions.
    """

    def __init__(self, a, b_init, clip_b, train_a ):
        """
        a: A scalar constant hyperparameter (default value 10).
        b: A trainable scalar parameter of size (num_classes) controlling the slope of the aggregation sigmoid function

        """
        super(NoisyAnd,self).__init__()
        self.a = nn.Parameter(torch.ones(1)*a, requires_grad=train_a)

        # init B to 0.5 if not given another value
        if b_init == None:
            #b_init = torch.rand(1)
            b_init = torch.ones(1)*0.5
        # init b to a random number from [0,1]
        self.b = nn.Parameter(torch.ones(1)*b_init, requires_grad=True)
        self.clip_b = clip_b

    def forward(self, preds):
        """
        Formula taken from the original paper.
        """
        if self.clip_b:
            eps = 1e-12
            self.b.data = self.b.clamp(0+eps,1-eps).data

        A = torch.sigmoid(self.a*preds.mean() - (self.a * self.b))
        B = torch.sigmoid(-self.a*self.b)
        C = torch.sigmoid(self.a - self.a*self.b)

        Y = (A - B) / (C - B)
        return Y

    def extra_repr(self):
        return f"NoisyAnd a={self.a}, b={self.b} of shape {self.b.shape}"

class MILNetwork(nn.Module):
    def __init__(self, a=10, b_init=None, clip_b=False, transfer_from_model=None, train_a=False):
        super(MILNetwork,self).__init__()

        # base model network
        self.basenet = MyNetwork()
        self.nand = NoisyAnd(a=a,b_init=b_init,clip_b=clip_b, train_a=train_a)

        # if transfer learning, assign the base model and freeze its parameters
        if transfer_from_model is not None:
            self.basenet = transfer_from_model
            self.basenet.requires_grad_(False)
            # only NAnd.B requires grad
            if isinstance(self.nand,NoisyAnd):
                if train_a:
                    assert sum([p.requires_grad for p in self.parameters()]) == 2
                else:
                    assert sum([p.requires_grad for p in self.parameters()]) == 1



    def forward(self,femur):
        #output patch predictions
        y_p_hat = self.basenet(femur)
        # aggregate the patch predictions
        Y_I_hat = self.nand(y_p_hat)

        return Y_I_hat, y_p_hat


    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device
