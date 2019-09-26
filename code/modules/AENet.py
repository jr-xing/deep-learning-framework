#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:10:16 2019

@author: jrxing
"""
# Copied from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
import torch
from torch import nn
def getAENET(config):
    net_type = config.get('type', 'AECV2D')
    if net_type == 'AECV2D':
        return AECV2D(config)
    elif net_type == 'AECV3D':
        return AECV3D(config)
    elif net_type == 'AECRCV3D':
        return AECRCV3D(config)
    elif net_type == 'AEFC':
        return AEFC(config)
    else:
        raise ValueError(f'Unsupported network type: {net_type}')

class AECV2D(nn.Module):
    def __init__(self, config):
        super(AECV2D, self).__init__()
        self.imgDim = 2
#        self.encoder = nn.Sequential(
#            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
#            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#        )
#        self.decoder = nn.Sequential(
#            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0),  # b, 16, 5, 5
#            nn.ReLU(True),
#            nn.ConvTranspose2d(16, 8, 5, stride=1, padding=1),  # b, 8, 15, 15
#            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#            nn.Tanh()
#        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.Conv2d(8, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )

#    def __init__(self, config):
#        super(AE, self).__init__()
        
#        decLayers = [inputLayer]
#        for decLayerIdx in range(1, decLayerNum + 1):
#            layer = tf.keras.layers.Dense(np.round(inputSize/2**(decLayerNum)), \
#                                 activation='relu',  \
#                                 name='dec-'+str(decLayerIdx))\
#                                 (decLayers[decLayerIdx-1])
#            decLayers.append(layer)
#            
#        middleLayer = tf.keras.layers.Dense(midLayerSize, name='middle')(decLayers[-1])
#        incLayers = [middleLayer]
#        
#        for incLayerIdx in range(incLayerNum, 0, -1):
#            layer = tf.keras.layers.Dense(np.round(inputSize/2**(incLayerNum)), \
#                                 activation='relu',  \
#                                 name='inc-'+str(incLayerNum - incLayerIdx + 1))\
#                                 (incLayers[incLayerNum - incLayerIdx])
#            incLayers.append(layer)
#        outputLayer = tf.keras.layers.Dense(inputSize, activation='tanh', name='outputs')(incLayers[-1])
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AECV3D(nn.Module):
    def __init__(self, config):
        super(AECV3D, self).__init__()
        self.imgDim = 3
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get3DNet(paras):
    if paras == None or paras['structure'] == 'default':        
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(16, 32, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv3d(32, 32, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.Conv3d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )
    elif paras['structure'] == 'default_BN':
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(16, 32, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm3d(32),
            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv3d(32, 32, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm3d(32),
            nn.ReLU(True)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.BatchNorm3d(16),
            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.Conv3d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )
    elif paras['structure'] == 'debug':
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(8, 16, 3, stride=1, padding=1),  # b, 8, 3, 3            
            nn.ReLU(True)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )
    elif paras['structure'] == 'decreasing':
        # Pooling + double channel
        encoder_layers = []
        decoder_layers = []
        downlayer_num = paras.get('decreasing_layer_num', 3)
        batchNorm = paras.get('batchNorm', True)
        root_feature_num = paras.get('root_feature_num', 32)
        # Down layers
        encoder_layers.append(nn.Conv3d(1, root_feature_num, 3, stride = 1, padding = 1))
        for layerIdx in range(downlayer_num):
            in_feature_num = (2**layerIdx)*root_feature_num
            out_feature_num = in_feature_num*2
            encoder_layers.append(nn.Conv3d(in_feature_num, out_feature_num, 3, stride = 1, padding = 1))
            if batchNorm:
                encoder_layers.append(nn.BatchNorm3d(out_feature_num))
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.MaxPool3d(2, stride=2))
        
        # Middle Layer


        # Up layers
        for layerIdx in range(downlayer_num):
#            print(layerIdx)
            in_feature_num = (2**(downlayer_num-layerIdx))*root_feature_num
            out_feature_num = int(in_feature_num/2)
#            print(in_feature_num)
            decoder_layers.append(nn.ConvTranspose3d(in_feature_num, out_feature_num, 2, stride=2, padding=0))  # b, 16, 5, 5
            if batchNorm:
                decoder_layers.append(nn.BatchNorm3d(out_feature_num))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.Conv3d(out_feature_num, 1, 3, stride = 1, padding = 1))
        
        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)
    return encoder, decoder


#from modules.Layers import CoordConv3d
#class AECRCV3D(nn.Module):
#    def __init__(self, config):
#        super(AECRCV3D, self).__init__()
#        self.encoder = nn.Sequential(
#            CoordConv3d(1, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
#            nn.ReLU(True),
#            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
#            nn.Conv3d(16, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
#            nn.ReLU(True),
##            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
#            nn.Conv3d(16, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
#            nn.ReLU(True)
#        )
#        self.decoder = nn.Sequential(
#            nn.ConvTranspose3d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
#            nn.ReLU(True),
##            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),  # b, 16, 5, 5
#            nn.Conv3d(8, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
#            nn.ReLU(True),
#            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
#            nn.Tanh(),
#        )
#        
#    def forward(self, x):
#        x = self.encoder(x)
#        x = self.decoder(x)
#        return x

class AEFC(nn.Module):
    def __init__(self, config):
        super(AEFC, self).__init__()
        data_size = config['data_size']
        self.encoder = nn.Sequential(
            nn.Linear(data_size, int(data_size/8)),         nn.ReLU(True),
            nn.Linear(int(data_size/8), int(data_size/16)),  nn.ReLU(True), 
            )
        self.decoder = nn.Sequential(
            nn.Linear(int(data_size/16), int(data_size/8)),  nn.ReLU(True),
            nn.Linear(int(data_size/8), data_size),         nn.Tanh(),
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__  == '__main__':    
    import numpy as np
    config = {'type': 'AECV3D',
              'paras': {
              'structure': 'decreasing',
              'batchNorm': True,
              'root_feature_num':16}}
    # config =  = {'type':'AECV3D','paras':{'name':'debug'}}
    ae = getAENET(config)
#    ae = AE(config = None)
    data_shape = (100, 1, 32, 32, 32)
#    data_shape = (1, 1, 100, 128, 128)
    data = np.random.rand(*data_shape).astype(np.float32)
    output = ae.forward(torch.from_numpy(data)).detach().numpy()
    print(output.shape)
    

    
    