from .util import DownSample, ConvBnActivate, UpSample, FinalConvolution, CatBlock
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, num_out_classes=2, input_channels=1, init_feat_channels=32, testing=False):
        super().__init__()

        self.testing = testing

        # Encoder layers definitions
        self.down_sample = DownSample()

        self.init_conv = ConvBnActivate(input_channels, init_feat_channels, init_feat_channels*2)
        self.down_conv1 = ConvBnActivate(init_feat_channels*2, init_feat_channels*2, init_feat_channels*4)
        self.down_conv2 = ConvBnActivate(init_feat_channels*4, init_feat_channels*4, init_feat_channels*8)
        self.down_conv3 = ConvBnActivate(init_feat_channels*8, init_feat_channels*8, init_feat_channels*16)

        # Decoder layers definitions
        self.up_sample1 = UpSample(init_feat_channels*16, init_feat_channels*16)
        self.up_conv1   = ConvBnActivate(init_feat_channels*(16+8), init_feat_channels*8, init_feat_channels*8)

        self.up_sample2 = UpSample(init_feat_channels*8, init_feat_channels*8)
        self.up_conv2   = ConvBnActivate(init_feat_channels*(8+4), init_feat_channels*4, init_feat_channels*4)

        self.up_sample3 = UpSample(init_feat_channels*4, init_feat_channels*4)
        self.up_conv3   = ConvBnActivate(init_feat_channels*(4+2), init_feat_channels*2, init_feat_channels*2)

        self.final_conv = FinalConvolution(init_feat_channels*2, num_out_classes)

        # Softmax
        self.softmax = nn.Softmax(dim=1)    # 多分类问题用soft-max函数作为输出
        self.sigmoid = nn.Sigmoid()         # 二分类问题用sigmoid函数作为输出，二分类下和softmax等价

    def forward(self, image):
        # Encoder Part #
        # B x  1 x Z x Y x X
        layer_init = self.init_conv(image)

        # B x 64 x Z x Y x X
        max_pool1  = self.down_sample(layer_init)
        # B x 64 x Z//2 x Y//2 x X//2
        layer_down2 = self.down_conv1(max_pool1)

        # B x 128 x Z//2 x Y//2 x X//2
        max_pool2   = self.down_sample(layer_down2)
        # B x 128 x Z//4 x Y//4 x X//4
        layer_down3 = self.down_conv2(max_pool2)

        # B x 256 x Z//4 x Y//4 x X//4
        max_pool_3  = self.down_sample(layer_down3)
        # B x 256 x Z//8 x Y//8 x X//8
        layer_down4 = self.down_conv3(max_pool_3)
        # B x 512 x Z//8 x Y//8 x X//8

        # Decoder part #
        layer_up1 = self.up_sample1(layer_down4)
        # B x 512 x Z//4 x Y//4 x X//4
        cat_block1 = CatBlock(layer_down3, layer_up1)
        # B x (256+512) x Z//4 x Y//4 x X//4
        layer_conv_up1 = self.up_conv1(cat_block1)
        # B x 256 x Z//4 x Y//4 x X//4

        layer_up2 = self.up_sample2(layer_conv_up1)
        # B x 256 x Z//2 x Y//2 x X//2
        cat_block2 = CatBlock(layer_down2, layer_up2)
        # B x (128+256) x Z//2 x Y//2 x X//2
        layer_conv_up2 = self.up_conv2(cat_block2)
        # B x 128 x Z//2 x Y//2 x X//2

        layer_up3 = self.up_sample3(layer_conv_up2)
        # B x 128 x Z x Y x X
        cat_block3 = CatBlock(layer_init, layer_up3)
        # B x (64+128) x Z x Y x X
        layer_conv_up3 = self.up_conv3(cat_block3)

        # B x 64 x Z x Y x X
        final_layer = self.final_conv(layer_conv_up3)
        # B x 2 x Z x Y x X
        if self.testing:
            final_layer = self.sigmoid(final_layer)

        return final_layer
