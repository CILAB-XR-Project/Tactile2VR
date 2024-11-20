import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model_component import TactileVREncoder, PoseDecoder
from config import Tactile2PoseConfig
from dataloader import generate_heatmap, get_tactile_dataloaders
from visualize import plot3Dheatmap
from PIL import Image
from const import VR_INDEXS

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
        multimodel_feature = self.feature_encoder(tactile_left, tactile_right ,keypoint_vr)
        current_pose = self.feature_decoder(multimodel_feature)
        
        return current_pose
    
class Tactile2PoseVRHeatmap(nn.Module):
    def __init__(self, config):
        super(Tactile2PoseVRHeatmap, self).__init__()

        conv_0_channel = config.WINDOW_SIZE
        self.action_list= config.ACTION_LIST
        self.vr_kp_len = config.VR_KP_LEN

        self.conv_0 = nn.Sequential(
            nn.Conv2d(conv_0_channel, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))

        
 
        self.conv3D_vr_heatmap_0 = nn.Sequential(
            nn.Conv3d(self.vr_kp_len, 8, kernel_size=(3,3,3),padding=(1,2,1)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8))
        
        self.conv3D_vr_heatmap_1 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=2))
        
        self.conv3D_vr_heatmap_2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32)) 


        self.conv3D_0 = nn.Sequential(
            nn.Conv3d(289, 257, kernel_size=(3, 5, 5), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(257))

        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(257, 128, kernel_size=(4, 5, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))

        self.conv3DTrans_0 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))

        self.conv3D_2 = nn.Sequential(
            nn.Conv3d(64, 19, kernel_size=(3, 3, 5), padding=1),
            nn.LeakyReLU())

        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, tactile_left, tactile_right, feature_vr_heatmap):
        tactile = torch.cat((tactile_left, tactile_right), dim=-1)
        feature = self.conv_0(tactile)
        feature = self.conv_1(feature)
        feature = self.conv_2(feature)
        feature = self.conv_3(feature)
        
        feature = feature.unsqueeze(-1)
        feature = feature.repeat(1, 1, 1, 1, 9)
        
        layer = torch.zeros(feature.shape[0], 1, feature.shape[2], feature.shape[3], feature.shape[4])
        layer = layer.to(self.dummy_param.device)
        for i in range(layer.shape[4]):
            layer[:, :, :, :, i] = i
        layer = layer / (layer.shape[4] - 1)
        
        feature_vr = self.conv3D_vr_heatmap_0(feature_vr_heatmap)
        feature_vr = self.conv3D_vr_heatmap_1(feature_vr)
        feature_vr = self.conv3D_vr_heatmap_2(feature_vr)

        output = torch.cat((feature, feature_vr, layer), axis=1)

        output = self.conv3D_0(output)
        output = self.conv3D_1(output)
        output = self.conv3DTrans_0(output)
        output = self.conv3D_2(output)
        return output

class Tactile2PoseVRLinear(nn.Module):
    def __init__(self, config):
        super(Tactile2PoseVRLinear, self).__init__()

        conv_0_channel = config.WINDOW_SIZE
        self.action_list= config.ACTION_LIST
        self.vr_kp_len = config.VR_KP_LEN

        self.conv_0 = nn.Sequential(
            nn.Conv2d(conv_0_channel, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))


        # 3*3:9 -> 32 -> 64 -> 80
        self.vr_mlps_0 = nn.Sequential(
            nn.Linear(self.vr_kp_len*3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 88),
            nn.LeakyReLU())
        
        self.vr_conv_0 = nn.Sequential(
            nn.Conv2d(config.WINDOW_SIZE, 16, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))
        
        self.vr_conv_1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        
        self.vr_conv_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        
        
        
        self.conv3D_0 = nn.Sequential(
            nn.Conv3d(289, 257, kernel_size=(3, 5, 5), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(257)) 

        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(257, 128, kernel_size=(4, 5, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))

        self.conv3DTrans_0 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))

        self.conv3D_2 = nn.Sequential(
            nn.Conv3d(64, 19, kernel_size=(3, 3, 5), padding=1),
            nn.LeakyReLU())

        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, tactile_left, tactile_right, keypoint_vr):
        tactile = torch.cat((tactile_left, tactile_right), dim=-1)
        feature = self.conv_0(tactile)
        feature = self.conv_1(feature)
        feature = self.conv_2(feature)
        feature = self.conv_3(feature)
 
        
        feature_kp_vr = keypoint_vr.view(keypoint_vr.shape[0]*keypoint_vr.shape[1], -1)
        feature_kp_vr = self.vr_mlps_0(feature_kp_vr)
        feature_kp_vr = feature_kp_vr.view(keypoint_vr.shape[0], keypoint_vr.shape[1], feature.shape[2], feature.shape[3])
        feature_kp_vr = self.vr_conv_0(feature_kp_vr)
        feature_kp_vr = self.vr_conv_1(feature_kp_vr)
        feature_kp_vr = self.vr_conv_2(feature_kp_vr) #32 * 8 * 11
        feature = torch.cat((feature, feature_kp_vr), dim=1) # 288 * 8 * 11
        
        
        feature = feature.unsqueeze(-1)
        feature = feature.repeat(1, 1, 1, 1, 9)

        layer = torch.zeros(feature.shape[0], 1, feature.shape[2], feature.shape[3], feature.shape[4])
        layer = layer.to(self.dummy_param.device)
        for i in range(layer.shape[4]):
            layer[:, :, :, :, i] = i
        layer = layer / (layer.shape[4] - 1)

        output = torch.cat((feature, layer), axis=1)

        output = self.conv3D_0(output)
        output = self.conv3D_1(output)
        output = self.conv3DTrans_0(output)
        output = self.conv3D_2(output)
        return output




if __name__ == "__main__":
    config = Tactile2PoseConfig
    data_dir = "/app/raid/isaac/InsoleData/dataset/"
    loaders, datasets, mappings = get_tactile_dataloaders(data_dir, config)
    train_dataloader, _,_ = loaders
    data_iter= iter(train_dataloader)
    next_data = next(data_iter)
    tactile_left, tactile_right, keypoint, action_idx = next_data
    keypoint_vr = keypoint[:,-1,VR_INDEXS, :]
    # keypoint_vr = keypoint
    keypoint_vr_heatmap = generate_heatmap(keypoint_vr)
    # keypoint_vr_heatmap = heatmap_from_keypoint(keypoint_vr[0])
    
    keypoint_vr_heatmap [keypoint_vr_heatmap  < 0.01] = 0
    keypoint_vr_heatmap [keypoint_vr_heatmap  > 0.01] *= 100
    # img = plot3Dheatmap(keypoint_vr_heatmap[-1,-1].detach().cpu().numpy())
    # img = Image.fromarray(img)
    # img.save("heatmap.png")
    
    model = tactile2pose_vr_3dheatmap(config.WINDOW_SIZE, config.VR_KP_LEN)
    batch_size = 32
    window_size = 20
    output = model(tactile_left.float(),tactile_right.float(), keypoint_vr_heatmap.float())
    print(output.size())


