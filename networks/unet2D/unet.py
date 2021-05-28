import torch.nn as nn
import torch
import torch.nn.functional as F
class DownConv2DBlock(nn.Module):

    """
    input_channels, output_channels - number of input and output channels, respectively
    conv_kernel_size - kernel size for down convolution. Original U-Net default is 3x3 convolution.
    """
    def __init__(self, input_channels, output_channels, use_dropout, legacy, conv_kernel_size=3):
        super(DownConv2DBlock, self).__init__()

        if legacy:
            block = [
                nn.Conv2d(input_channels, output_channels, conv_kernel_size, padding=1),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, conv_kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(output_channels)
            ]
        else:
            block = [
                nn.Conv2d(input_channels, output_channels, conv_kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, conv_kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            ]

        if use_dropout:
            block.insert(2, nn.Dropout2d())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class DownConvWithMaxPool2DBlock(nn.Module):

    """
    input_channels, output_channels - number of input and output channels, respectively
    conv_kernel_size - kernel size for down convolution. Original U-Net default is 3x3 convolution.
    """
    def __init__(self, input_channels, output_channels, use_dropout, legacy, conv_kernel_size=3):
        super(DownConvWithMaxPool2DBlock, self).__init__()
        self.down_conv = DownConv2DBlock(input_channels, output_channels, use_dropout, legacy, conv_kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down_conv(x)
        x_skip = x # TODO: Is this correct place for skip?
        x = self.max_pool(x)
        return x, x_skip

class BottomBlock(nn.Module):

    """
    input_channels - number of input and output channels
    conv_kernel_size - kernel size for down convolution. Original U-Net default is 3x3 convolution.
    """
    def __init__(self, input_channels, use_dropout, legacy, conv_kernel_size=3):
        super(BottomBlock, self).__init__()
        hidden_channels = input_channels * 2
        block = [
            DownConv2DBlock(input_channels, hidden_channels, use_dropout, legacy, conv_kernel_size),
            nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size=2, stride=2)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class UpConv2DBlock(nn.Module):

    """
    input_channels, output_channels - number of input and output channels, respectively
    conv_kernel_size - kernel size for down convolution. Original U-Net default is 3x3 convolution.
    """
    def __init__(self, input_channels, use_dropout, legacy, conv_kernel_size=3):
        super(UpConv2DBlock, self).__init__()
        hidden_channels = input_channels // 2
        block = [
            DownConv2DBlock(input_channels, hidden_channels, use_dropout, legacy, conv_kernel_size),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

"""
U-Net [1] implementation.

[1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation.
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
"""
class Unet2D(nn.Module):

    def __init__(self, n_classes, n_input_channels, relu_type='normal', use_dropout=False, legacy=True):
        super(Unet2D, self).__init__()

        # Contracting path
        self.block1 = DownConvWithMaxPool2DBlock(n_input_channels, 64, use_dropout=use_dropout, legacy=legacy)
        self.block2 = DownConvWithMaxPool2DBlock(64, 128, use_dropout=use_dropout, legacy=legacy)
        self.block3 = DownConvWithMaxPool2DBlock(128, 256, use_dropout=use_dropout, legacy=legacy)
        self.block4 = DownConvWithMaxPool2DBlock(256, 512, use_dropout=use_dropout, legacy=legacy)

        self.bottom_block = BottomBlock(512, use_dropout=use_dropout, legacy=legacy)
      
        # Expansive path
        self.up_block1 = UpConv2DBlock(1024, use_dropout=use_dropout, legacy=legacy)
        self.up_block2 = UpConv2DBlock(512, use_dropout=use_dropout, legacy=legacy)
        self.up_block3 = UpConv2DBlock(256, use_dropout=use_dropout, legacy=legacy)

        # Final block
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_3 = nn.Conv2d(64, n_classes, 1)

        if relu_type == 'normal':
            self.relu = nn.ReLU()
        elif relu_type == 'leaky':
            self.relu = nn.LeakyReLU()
        else:
            raise Exception('ReLU type should be either normal or leaky')
        
    # applies negative padding to image from skip connection.
    # not needed for padded convolutions
    def crop_skip_connected(self, skipped, up_sampled):
        #padding = (skipped.shape[2] - up_sampled.shape[2]) // 2
        return skipped#F.pad(skipped, (padding, -padding, -padding, padding))

    # Forward for Encoder (contracting path)
    def encoder_forward(self, x):
        x, x_1 = self.block1(x)
        x, x_2 = self.block2(x)
        x, x_3 = self.block3(x)
        x, x_4 = self.block4(x)

        x = self.bottom_block(x)
        
        return x, x_1, x_2, x_3, x_4

    # Forward for Decoder (expansive path)
    # If classify=True, will execute the final convolutional layer. Otherwise, it will be skipped.
    def decoder_forward(self, x, x_1, x_2, x_3, x_4, classify=True):

        x = torch.cat([x_4, x], dim=1)
        x = self.up_block1(x)

        x = torch.cat([x_3, x], dim=1)
        x = self.up_block2(x)

        x = torch.cat([x_2, x], dim=1)
        x = self.up_block3(x)

        x = torch.cat([x_1, x], dim=1)

        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv9_2(x)
        x = self.relu(x)

        if classify:
            x = self.predict(x)

        return x

    def predict(self, x):
        x = self.conv9_3(x)
        #x = self.relu(x) # relu here as well? Not according to paper.
        return x

    # Forward pass is split into two, to be able to branch off of the latent space
    def forward(self, x):
        x, x_1, x_2, x_3, x_4 = self.encoder_forward(x)
        x = self.decoder_forward(x, x_1, x_2, x_3, x_4)

        return x

    def toggle_encoder_weights(self, disable):
        blocks = [self.block1, self.block2, self.block3, self.block4, self.bottom_block]
        for block in blocks:
            for _, param in block.named_parameters():
                param.requires_grad = not disable

    def toggle_decoder_weights(self, disable):
        blocks = [self.up_block1, self.up_block2, self.up_block3, self.conv9, self.conv9_2, self.conv9_3]
        for block in blocks:
            for _, param in block.named_parameters():
                param.requires_grad = not disable