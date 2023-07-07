import torch.nn as nn
import torch.nn.functional as F


NO_GROUPS = 4

'''
- Block class contains 3 or less convolution layers. 
- conv1 is either a 3x3 convolution or a 3x3 depthwise separable convolution followed by either of BN, GN or LN normalization, activation function, 
 and a dropout layer
- conv2 is a 3x3 convolution layer and followed by either of BN, GN or LN normalization activation function,
 and a dropout layer
- conv3 is either a dilated convolution with dilation=2 or a strided convolution with stride=2 followed by either of BN, GN or LN normalization, activation function, 
 and a dropout layer

'''
class Block(nn.Module):
    def __init__(self, input_channel, output_channel, padding=0, norm='bn', drop=0.01, dilated_conv=False, depth_sep_conv=False):

        super(Block, self).__init__()

        if depth_sep_conv:
            '''
            Depthwise Separable Convolution
            '''
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 3, groups=input_channel, padding=padding),
                nn.Conv2d(input_channel, output_channel, 1)
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_channel)

        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_channel)

        self.drop2 = nn.Dropout2d(drop)

        if dilated_conv:
            '''
            Dilated Convolution
            '''
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=padding+1, dilation=2)
        else:
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=padding, stride=2)

        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, output_channel)

        self.drop3 = nn.Dropout2d(drop)


    '''
    Depending on the model requirement, Convolution block with number of layers is applied to the input image
    '''
    def __call__(self, x, layers=2):

        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)

        x = self.drop1(x)


        if layers >= 2:
            x = self.conv2(x)

            x = self.n2(x)
            x = F.relu(x)
            x = self.drop2(x)

        if layers == 3:
            x = self.conv3(x)

            x = self.n3(x)
            x = F.relu(x)
            x = self.drop3(x)


        return x

'''
class TransitionBlock(nn.Module):
    def __init__(self, output_channel, pooling=True):
        super(TransitionBlock, self).__init__()

        self.tconv = nn.Conv2d(output_channel, int(output_channel/2), 1)

        self.pooling = pooling

        self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x):
        x = self.tconv(x)

        if self.pooling:
            x = self.pool(x)

        return x
        
'''


class S9Model(nn.Module):
    def __init__(self, base_channels, norm='bn', drop=0.01):
        super(S9Model, self).__init__()

        '''
        In the first convolution block, the Dilated Convolution is applied (3rd layer) to increase receptive field.
        '''
        self.block1 = Block(3, base_channels, padding=1, norm=norm, drop=drop, dilated_conv=True)
        #self.tblock1 = TransitionBlock(base_channels, pooling=False)
        ''' RF: 1 => 3 > 5 > 9 '''

        '''
        In the second convolution block, the Depthwise Separable Convolution is applied (1st layer) to reduce param count.
        '''
        self.block2 = Block(base_channels, base_channels*2, padding=1, norm=norm, drop=drop, dilated_conv=False, depth_sep_conv=True)
        #self.tblock2 = TransitionBlock(base_channels*2, pooling=False)
        ''' RF: 9 => 11 > 13 > 15 '''

        self.block3 = Block(base_channels*2, base_channels*4, padding=1, norm=norm, drop=drop, dilated_conv=False, depth_sep_conv=True)
        #self.tblock3 = TransitionBlock(base_channels*2, pooling=False)
        ''' RF: 15 => 19 > 23 > 27 '''

        self.block4 = Block(base_channels*4, base_channels*2, padding=1, norm=norm, drop=drop, dilated_conv=False)
        ''' RF: 27 => 35 > 43 '''

        self.block4LastLayer = nn.Conv2d(base_channels*2, base_channels*2, 3)
        ''' RF: 43 => 51 '''

        self.gap = nn.AvgPool2d(6)
        self.linear = nn.Conv2d(base_channels*2, 10, 1)

    def forward(self, x):
        x = self.block1(x, layers=3)

        x = self.block2(x, layers=3)

        x = self.block3(x, layers=3)

        x = self.block4(x, layers=2)

        x = self.block4LastLayer(x)

        x = self.gap(x)

        x = self.linear(x)

        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)
