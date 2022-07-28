
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        # pass
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x += residual
        x = self.relu(x)
        return x

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128))
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        # self.res1 = residual_block(in_channels=64)
        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.relu1 = nn.ReLU()
        # self.res1 = residual_block(in_channels=128)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.relu2 = nn.ReLU()
        # self.res2 = residual_block(in_channels=256)



        
        self.res1 = residual_block(in_channels=128)
        self.myconv1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 16,128,15,15
                             nn.BatchNorm2d(256),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.ReLU(),
                             )
        self.res2 = residual_block(in_channels=256) 
        self.myconv2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
                             nn.BatchNorm2d(512),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.ReLU(),
                             )
        
        self.res3 = residual_block(in_channels=512)
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256,128), nn.ReLU())
        self.fc3 = nn.Linear(128, num_out)

        self.activation = nn.ReLU()
        # pass
        # self.tempconv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        # self.batchnorm1 = nn.BatchNorm2d(128)
        # self.maxpool = nn.MaxPool2d
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # pass


        # print(x.shape) # torch.Size([16, 3, 32, 32])
        x = self.stem_conv(x)
        x = self.activation(x)
        # print('after self.activation(x):', x.shape) # torch.Size([16, 64, 32, 32])
        x = self.res1(x)
        # print('after self.res1(x):', x.shape) # torch.Size([16, 64, 32, 32])
        x = self.myconv1(x)
        # print('after self.myconv1(x):', x.shape) 
        x = self.res2(x)
        # print('after self.res2(x):', x.shape) 
        x = self.myconv2(x)
        x = self.res3(x)
        # print('after self.myconv2(x):', x.shape) 
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # x = x.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out