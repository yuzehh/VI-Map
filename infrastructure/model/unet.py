import numpy as np
import torch
import torch.nn as nn

up = nn.Upsample(scale_factor=2)
# def up(t: torch.Tensor):
#     return torch.repeat_interleave(torch.repeat_interleave(t, 2, dim=-1), 2, dim=-2)


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


# # # for debug
# class UNet(nn.Module):
#     def __init__(self, cnl_in=4):
#         super().__init__()
#         self.p = nn.Parameter(torch.ones((4, 4+16, 400, 800)).float()*0, requires_grad=True)
#         # self.p = nn.Parameter(torch.Tensor(0.09*np.random.random((4, 4+16, 400, 800))).float(), requires_grad=True)
#     def forward(self, img):
#         semantic, embedding = torch.split(self.p, [4, 16], 1)
#         return semantic, embedding


class UNet(nn.Module):
    """
    MPINet that takes in arbitrary shape image
    """

    def __init__(self, cnl_in=4):
        super().__init__()
        self.down = nn.MaxPool2d(2, ceil_mode=True)
        self.up = up
        self.down1 = conv(cnl_in, 32, 7)
        self.down1b = conv(32, 32, 7)
        self.down2 = conv(32, 64, 5)
        self.down2b = conv(64, 64, 5)
        self.down3 = conv(64, 128, 3)
        self.down3b = conv(128, 128, 3)
        self.down4 = conv(128, 256, 3)
        self.down4b = conv(256, 256, 3)
        self.down5 = conv(256, 512, 3)
        self.down5b = conv(512, 512, 3)
        self.down6 = conv(512, 512, 3)
        self.down6b = conv(512, 512, 3)
        self.mid1 = conv(512, 512, 3)
        self.mid2 = conv(512, 512, 3)
        self.up6 = conv(1024, 512, 3)
        self.up6b = conv(512, 512, 3)
        self.up5 = conv(1024, 512, 3)
        self.up5b = conv(512, 512, 3)
        self.up4 = conv(768, 256, 3)
        self.up4b = conv(256, 256, 3)
        self.up3 = conv(384, 128, 3)
        self.up3b = conv(128, 128, 3)
        self.up2 = conv(192, 64, 3)
        self.up2b = conv(64, 64, 3)
        self.post1 = conv(96, 64, 3)
        # self.post2 = conv(64, 4+16+16, 3)
        self.post2 = conv(64, 5+16, 3, isReLU=False)
        self.up1 = conv(64, 64, 3)
        self.up1b = conv(64, 64, 3)
        self.initial_weights()

    @staticmethod
    def shapeto(x, tar):
        return [x[..., :tar.shape[-2], :tar.shape[-1]], tar]

    def forward(self, img):
        if hasattr(torch.cuda, 'empty_cache'):
	        torch.cuda.empty_cache()
        down1 = self.down1b(self.down1(img))
        down2 = self.down2b(self.down2(self.down(down1)))
        down3 = self.down3b(self.down3(self.down(down2)))
        down4 = self.down4b(self.down4(self.down(down3)))
        down5 = self.down5b(self.down5(self.down(down4)))
        down6 = self.down6b(self.down6(self.down(down5)))
        x = self.up(self.mid2(self.mid1(self.down(down6))))
        x = self.up(self.up6b(self.up6(torch.cat(self.shapeto(x, down6), dim=1))))
        x = self.up(self.up5b(self.up5(torch.cat(self.shapeto(x, down5), dim=1))))
        x = self.up(self.up4b(self.up4(torch.cat(self.shapeto(x, down4), dim=1))))
        x = self.up(self.up3b(self.up3(torch.cat(self.shapeto(x, down3), dim=1))))
        x = self.up(self.up2b(self.up2(torch.cat(self.shapeto(x, down2), dim=1))))
        x = self.post2(self.post1(torch.cat(self.shapeto(x, down1), dim=1)))
        # semantic, embedding, lanembedding = torch.split(x, [4,16,16], 1)
        semantic, embedding = torch.split(x, [5,16], 1)
        # semantic = torch.sigmoid(semantic)
        # semantic = torch.softmax(semantic, 1)
        # s = 1 - semantic[:,0,:,:].clone()
        # se = torch.unsqueeze(s,1)
        # sem = torch.concat((se,semantic[:,1:,:,:].clone()), dim=1)
        return semantic, embedding

    def initial_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
