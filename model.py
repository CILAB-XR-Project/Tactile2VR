import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model_component import TactileVREncoder, PoseDecoder


class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, height, width, depth, channel, lim=[0., 1., 0., 1., 0., 1.], temperature=None, data_format='NCHWD'):
        super(SpatialSoftmax3D, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.
        pos_y, pos_x, pos_z = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height),
            np.linspace(lim[4], lim[5], self.depth))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width * self.depth)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width * self.depth)).float()
        pos_z = torch.from_numpy(pos_z.reshape(self.height * self.width * self.depth)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)
            
    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWDC':
            assert feature.shape[1] == self.height and feature.shape[2] == self.width and feature.shape[3] == self.depth
            feature = feature.transpose(1, 4).tranpose(2, 4).tranpose(3,4).reshape(-1, self.height * self.width * self.depth) #NCHWD
        else:
            feature = feature.reshape(-1, self.height * self.width * self.depth)

        softmax_attention = feature
        #softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        heatmap = softmax_attention.reshape(-1, self.channel, self.height, self.width, self.depth)

        eps = 1e-6
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xyz.reshape(-1, self.channel, 3)
        return feature_keypoints, heatmap

        
class Tactile2PoseFeatureModel(nn.Module):
    def __init__(self, config):
        super(Tactile2PoseFeatureModel, self).__init__()
        
        self.window_size = config.WINDOW_SIZE
        self.action_list= config.ACTION_LIST
        self.vr_kp_len = config.VR_KP_LEN
        
        self.feature_encoder = TactileVREncoder(self.window_size, self.vr_kp_len)
        self.feature_decoder = PoseDecoder()
        
        
        self.count_parameters()
        
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')    
        
    def forward(self, tactile_left, tactile_right, keypoint_vr):
        multimodel_feature = self.feature_encoder(tactile_left, tactile_right)
        current_pose = self.feature_decoder(multimodel_feature)
        
        return current_pose
