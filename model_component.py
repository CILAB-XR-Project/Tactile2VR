import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder for train multimodal feature
#Done
class TactileEncoder(nn.Module): # window_size * 32 * 22 -> 256*8*10 
    def __init__(self, window_size):
        super(TactileEncoder, self).__init__()
        conv_0_channel = window_size
        
        #left tactile encoding module
        self.left_conv_0 = nn.Sequential(
            nn.Conv2d(conv_0_channel, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))# 32 * 32 * 22

        self.left_conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))# 64 * 16 * 11

        self.left_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)) # 128 * 16 * 11

        self.left_conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))# 256 * 8 * 5

        #right tactile encoding module
        self.right_conv_0 = nn.Sequential(
            nn.Conv2d(conv_0_channel, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))# 32 * 32 * 22

        self.right_conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 64 * 16 * 11

        self.right_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)) # 128 * 16 * 11

        self.right_conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))  # 256 * 8 * 5

    def forward(self, left_tactile, right_tactile):
        feature_left = self.left_conv_0(left_tactile)
        feature_left = self.left_conv_1(feature_left)
        feature_left = self.left_conv_2(feature_left)
        feature_left = self.left_conv_3(feature_left)
        
        feature_right = self.right_conv_0(right_tactile)
        feature_right = self.right_conv_1(feature_right)
        feature_right = self.right_conv_2(feature_right)
        feature_right = self.right_conv_3(feature_right)
        
        multimodal_feature = torch.cat((feature_left, feature_right), dim=-1) # 256 * 8 * 10
        return multimodal_feature

    
class PoseDecoder(nn.Module):# 256*8*10 -> 257*8*10*9 -> 19*20*20*18
    def __init__(self):
        super(PoseDecoder, self).__init__()
        
        self.conv3D_0 = nn.Sequential(
            nn.Conv3d(257, 257, kernel_size=(3,5,4),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(257))# 257*8*8*8
        
        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(257, 128, kernel_size=(4,4,4),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))# 128*7*7*7
        
        self.conv3DTrans_0 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(2,2,2),stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))# 64*16*20*9
        
        self.conv3D_2 = nn.Sequential(
            nn.Conv3d(64, 19, kernel_size=(3,3,5),padding=1),
            nn.LeakyReLU())# 19*20*20*18
        
        self.dummy_param = nn.Parameter(torch.empty(0))
                            
        
    def forward(self, feature):
        feature = feature.unsqueeze(-1)
        feature = feature.repeat(1, 1, 1, 1, 9)

        layer = torch.zeros(feature.shape[0], 1, feature.shape[2], feature.shape[3], feature.shape[4])
        layer = layer.to(self.dummy_param.device)
        for i in range(layer.shape[4]):
            layer[:,:,:,:,i] = i
        layer = layer/(layer.shape[4]-1)

        output = torch.cat((feature, layer), axis=1)

        output = self.conv3D_0(output)
        output = self.conv3D_1(output)
        output = self.conv3DTrans_0(output)
        output = self.conv3D_2(output)
        return output


