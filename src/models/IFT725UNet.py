# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel

#'''
#TODO

#Ajouter du code ici pour faire fonctionner le réseau IFT725UNet.  Un réseau inspiré de UNet
#mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

#'''

class IFT725UNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True, residuts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        """
        Builds AlexNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
            residuts: whether or not apply residual to block number i
        """
        super(IFT725UNet, self).__init__()
        in_channels = 1  # gray image
        self.residus = residuts
        
        self.conv_encoder1 = self._contracting_block(in_channels=in_channels, out_channels=64)
        self.residual_shortcut1 = self._residual_shortcut_block(in_channels=in_channels, out_channels=64) ##
        self.max_pool_encoder1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv_encoder2 = self._contracting_block(64, 128)
        self.residual_shortcut2 = self._residual_shortcut_block(in_channels=64, out_channels=128) ##
        self.max_pool_encoder2 = nn.MaxPool2d(kernel_size=2)

        self.conv_encoder3 = self._contracting_block(128, 256)
        self.residual_shortcut3 = self._residual_shortcut_block(in_channels=128, out_channels=256) ##
        self.max_pool_encoder3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv_encoder4 = self._contracting_block(256, 512)
        self.residual_shortcut4 = self._residual_shortcut_block(in_channels=256, out_channels=512) ##
        self.max_pool_encoder4 = nn.MaxPool2d(kernel_size=2)
        
        

        # Transitional block # Couche Dense
        kernel_size = 3 
        in_channels_trans = 512
        out_channels_trans = 256
        
        self.conv_encoder_trans_1 = nn.Conv2d(kernel_size=3, in_channels=in_channels_trans, out_channels=out_channels_trans, padding=1)
        self.conv_encoder_trans_2 = nn.ReLU()
        self.conv_encoder_trans_3 = nn.BatchNorm2d(out_channels_trans)
        
        self.conv_encoder_trans_4 = nn.Conv2d(kernel_size=3, in_channels=in_channels_trans+out_channels_trans, out_channels=out_channels_trans, padding=1)
        self.conv_encoder_trans_5 = nn.ReLU()
        self.conv_encoder_trans_6 = nn.BatchNorm2d(out_channels_trans)          
        
        self.residual_shortcut_transitional = self._residual_shortcut_block(in_channels=512, out_channels=1024)
        self.convTranspose_transitional = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1) ## kernel_size was = 3 in UNET

        # Decode
        self.residual_shortcut_encoder_decoder4 = self._residual_shortcut_block(in_channels=512, out_channels=1024)
        self.conv_decoder4 = self._expansive_block(1024, 512)
        self.residual_shortcut_decoder4 = self._residual_shortcut_block(in_channels=1024, out_channels=512)
        self.convTranspose_decoder4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                               output_padding=1)                               

        self.residual_shortcut_encoder_decoder3 = self._residual_shortcut_block(in_channels=256, out_channels=512)
        self.conv_decoder3 = self._expansive_block(512, 256)
        self.residual_shortcut_decoder3 = self._residual_shortcut_block(in_channels=512, out_channels=256)
        self.convTranspose_decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                               output_padding=1)

        self.residual_shortcut_encoder_decoder2 = self._residual_shortcut_block(in_channels=128, out_channels=256)
        self.conv_decoder2 = self._expansive_block(256, 128)
        self.residual_shortcut_decoder2 = self._residual_shortcut_block(in_channels=256, out_channels=128)
        self.convTranspose_decoder2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                               output_padding=1)

        self.residual_shortcut_encoder_decoder1 = self._residual_shortcut_block(in_channels=64, out_channels=128)
        self.final_layer = self._final_block(128, 64, num_classes)
        self.residual_shortcut_final_layer= self._residual_shortcut_block(in_channels=128, out_channels=num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encoder1      --block1
        encode_block1 = self.conv_encoder1(x)
        if self.residus[0] == 1:
            encode_block1 += self.residual_shortcut1(x)
        encode_pool1 = self.max_pool_encoder1(encode_block1)

        # Encoder2      --block2
        encode_block2 = self.conv_encoder2(encode_pool1)
        if self.residus[1] == 1:
            encode_block2 += self.residual_shortcut2(encode_pool1)
        encode_pool2 = self.max_pool_encoder2(encode_block2)
        
        # Encoder3      --block3
        encode_block3 = self.conv_encoder3(encode_pool2)
        if self.residus[2] == 1:
            encode_block3 += self.residual_shortcut3(encode_pool2) ##
        encode_pool3 = self.max_pool_encoder3(encode_block3)
        
        # Encoder4      --block4
        encode_block4 = self.conv_encoder4(encode_pool3) 
        if self.residus[3] == 1:
            encode_block4 += self.residual_shortcut4(encode_pool3) ##
        encode_pool4 = self.max_pool_encoder4(encode_block4)

        # Transitional block    
        encode_block_trans_1 = self.conv_encoder_trans_1(encode_pool4)
        encode_block_trans_1 = self.conv_encoder_trans_2(encode_block_trans_1)
        encode_block_trans_1 = self.conv_encoder_trans_3(encode_block_trans_1)
        
        encode_block_trans_1 = torch.cat((encode_pool4, encode_block_trans_1), 1) # Concatenation
        
        encode_block_trans_2 = self.conv_encoder_trans_4(encode_block_trans_1)
        encode_block_trans_2 = self.conv_encoder_trans_5(encode_block_trans_2)
        encode_block_trans_2 = self.conv_encoder_trans_6(encode_block_trans_2)

        middle_block = torch.cat((encode_block_trans_1, encode_block_trans_2), 1) # Concatenation
        
        convTranspose_transitional = self.convTranspose_transitional(middle_block) 
        # Decoder4     --block5
        decode_block4 = torch.cat((convTranspose_transitional, encode_block4), 1) 
        if self.residus[4] == 1:
            decode_block4 += self.residual_shortcut_encoder_decoder4(encode_block4)

                        #--block6
        cat_layer3 = self.conv_decoder4(decode_block4) 
        if self.residus[5] == 1:
            cat_layer3 += self.residual_shortcut_decoder4(decode_block4)
        convTranspose_decoder4 = self.convTranspose_decoder4(cat_layer3)
        
        
        # Decoder3      --block7
        decode_block3 = torch.cat((convTranspose_decoder4, encode_block3), 1)
        if self.residus[6] == 1:
            decode_block3 += self.residual_shortcut_encoder_decoder3(encode_block3)
        
                        #--block8
        cat_layer2 = self.conv_decoder3(decode_block3) 
        if self.residus[7] == 1:
            cat_layer2 += self.residual_shortcut_decoder3(decode_block3)
        convTranspose_decoder3 = self.convTranspose_decoder3(cat_layer2)
        
        # Decoder2      --block9
        decode_block2 = torch.cat((convTranspose_decoder3, encode_block2), 1) 
        if self.residus[8] == 1:
            decode_block2 += self.residual_shortcut_encoder_decoder2(encode_block2)
        
                        #--block10
        cat_layer1 = self.conv_decoder2(decode_block2)
        if self.residus[9] == 1:
            cat_layer1 += self.residual_shortcut_decoder2(decode_block2)
        convTranspose_decoder2 = self.convTranspose_decoder2(cat_layer1)
        
        # Decoder1      --block11
        decode_block1 = torch.cat((convTranspose_decoder2, encode_block1), 1) 
        if self.residus[10] == 1:
            decode_block1 += self.residual_shortcut_encoder_decoder1(encode_block1)
        
                        #--block12
        final_layer = self.final_layer(decode_block1)
        if self.residus[11] == 1:
            final_layer += self.residual_shortcut_final_layer(decode_block1)
        
        
        return final_layer

    def _contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return block

    def _expansive_block(self, in_channels, mid_channels, kernel_size=3):
        """
        Building block of the expansive part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
        )
        return block

    def _final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the UNet model
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block
    
    def _residual_shortcut_block(self, in_channels, out_channels, kernel_size=1):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(out_channels)
        )
        return block
