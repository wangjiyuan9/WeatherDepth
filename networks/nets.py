from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from .hr_decoder import HR_DepthDecoder
from .mpvit import *


class DeepNet(nn.Module):
    def __init__(self,type,weights_init= "pretrained",num_layers=18,num_pose_frames=2,scales=range(4)):
        super(DeepNet, self).__init__()
        self.type = type
        self.num_layers=num_layers
        self.weights_init=weights_init
        self.num_pose_frames=num_pose_frames
        self.scales = scales
        if self.type =='mpvitnet':
            self.encoder = mpvit_small()
            self.decoder = HR_DepthDecoder()
 
        else:
            print("wrong type of the networks, only depthnet and posenet")
            
    def forward(self, inputs):
        if self.type =='mpvitnet': 
            self.outputs = self.decoder(self.encoder(inputs))
        return self.outputs
