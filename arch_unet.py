import paddle
import paddle.nn as nn
from paddle.nn import initializer as init


def initialize_weights(module, scale=1):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv3D):
            # initializer = init.KaimingNormal(fan_in=0)
            import torch
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            # initializer(m.weight)
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2DTranspose) or isinstance(
                m, nn.Conv3DTranspose):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2D) or isinstance(
                m, nn.BatchNorm3D):
            init.constant_(m.weight, 1)
            init.constant_(m.bias.data, 0.0)


class UpsampleCat(nn.Layer):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc

        self.deconv = nn.Conv2DTranspose(in_nc, out_nc, 2, 2, 0, False)
        # initialize_weights(self, 0.1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return paddle.concat([x1, x2], axis=1)


def conv_func(x, conv, blindspot=False):
    size = conv._kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad = nn.Pad2D(padding=(0, 0, ofs, 0), value=0)
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot=False):
    if blindspot:
        pad = nn.Pad2D(padding=(0, 0, 1, 0), value=0)
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x

class UNet(nn.Layer):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.act = nn.LeakyReLU(negative_slope=0.2)

        # Encoder part
        self.enc_conv0 = nn.Conv2D(self.in_nc, self.n_feature, 3, 1, 1)
        self.enc_conv1 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool1 = nn.MaxPool2D(2)

        self.enc_conv2 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool2 = nn.MaxPool2D(2)

        self.enc_conv3 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool3 = nn.MaxPool2D(2)

        self.enc_conv4 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool4 = nn.MaxPool2D(2)

        self.enc_conv5 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)
        self.pool5 = nn.MaxPool2D(2)

        self.enc_conv6 = nn.Conv2D(self.n_feature, self.n_feature, 3, 1, 1)

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = nn.Conv2D(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv5b = nn.Conv2D(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = nn.Conv2D(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv4b = nn.Conv2D(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = nn.Conv2D(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv3b = nn.Conv2D(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = nn.Conv2D(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv2b = nn.Conv2D(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = nn.Conv2D(self.n_feature * 2 + self.in_nc, 96, 3, 1,
                                    1)
        self.dec_conv1b = nn.Conv2D(96, 96, 3, 1, 1)

        self.nin_a = nn.Conv2D(96, 96, 1, 1, 0)
        self.nin_b = nn.Conv2D(96, 96, 1, 1, 0)

        self.nin_c = nn.Conv2D(96, self.out_nc, 1, 1, 0)
        initialize_weights(self.nin_c, 0.1)

    def forward(self, x):
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0))
        x = self.act(conv_func(x, self.enc_conv1))
        x = pool_func(x, self.pool1)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2))
        x = pool_func(x, self.pool2)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3))
        x = pool_func(x, self.pool3)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4))
        x = pool_func(x, self.pool4)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5))
        x = pool_func(x, self.pool5)

        x = self.act(conv_func(x, self.enc_conv6))

        # Decoder part
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a))
        x = self.act(conv_func(x, self.dec_conv5b))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a))
        x = self.act(conv_func(x, self.dec_conv4b))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a))
        x = self.act(conv_func(x, self.dec_conv3b))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a))
        x = self.act(conv_func(x, self.dec_conv2b))

        x = self.up1(x, pool0)

        # Output stage

        x = self.act(conv_func(x, self.dec_conv1a))
        x = self.act(conv_func(x, self.dec_conv1b))
        x = self.act(conv_func(x, self.nin_a))
        x = self.act(conv_func(x, self.nin_b))
        x = conv_func(x, self.nin_c)
        return x


if __name__ == "__main__":
    import numpy as np
    x = paddle.to_tensor(np.zeros((10, 3, 32, 32), dtype=np.float32))
    print(x.shape)
    net = UNet(in_nc=3, out_nc=3)
    y = net(x)
    print(y.shape)
