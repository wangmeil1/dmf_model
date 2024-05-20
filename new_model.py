import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models as models
from resnext import ResNeXt101
import math


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        rgb_mean = (0.485, 0.456, 0.406)
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.229, 0.224, 0.225)
        self.std = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class BaseA(nn.Module):
    def __init__(self):
        super(BaseA, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63438
        self.mean[0, 1, 0, 0] = 0.59396
        self.mean[0, 2, 0, 0] = 0.58369
        self.std[0, 0, 0, 0] = 0.16195
        self.std[0, 1, 0, 0] = 0.16937
        self.std[0, 2, 0, 0] = 0.17564

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class BaseITS(nn.Module):
    def __init__(self):
        super(BaseITS, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63542
        self.mean[0, 1, 0, 0] = 0.59579
        self.mean[0, 2, 0, 0] = 0.58550
        self.std[0, 0, 0, 0] = 0.14470
        self.std[0, 1, 0, 0] = 0.14850
        self.std[0, 2, 0, 0] = 0.15348

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class Base_OHAZE(nn.Module):
    def __init__(self):
        super(Base_OHAZE, self).__init__()
        rgb_mean = (0.47421, 0.50878, 0.56789)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.10168, 0.10488, 0.11524)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)

        rgb_mean = (0.35851, 0.35316, 0.34425)
        self.mean_out = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.16391, 0.16174, 0.17148)
        self.std_out = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


# class new_model(Base):
#     def __init__(self, num_features=48):
#         super().__init__()
#         self.conv00 = nn.Sequential(nn.Conv2d(3, num_features // 3, kernel_size=1, padding=0),
#                                     nn.SELU(),
#                                     nn.AvgPool2d(2),
#                                     nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#                                     nn.SELU(),
#                                     )
#         self.conv01 = nn.Sequential(nn.Conv2d(4, num_features // 3, kernel_size=1, padding=0),
#                                     nn.SELU(),
#                                     nn.AvgPool2d(2),
#                                     nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#                                     nn.SELU(),
#                                     )
#         # self.conv_p = nn.Conv2d(3, num_features//3, kernel_size=3, padding=1)
#         # self.conv0 = nn.Sequential(
#         #     nn.Dropout(0.2),
#         #     nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#         #     nn.SELU(),
#         # )
#         self.conv1 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#             nn.SELU(),
#         )
#         self.conv11 = nn.Sequential(
#             nn.Conv2d(2*num_features // 3, num_features // 3, kernel_size=1, padding=0),
#             nn.SELU(),
#         )
#         self.conv20 = nn.Sequential(
#             nn.Conv2d(7, num_features // 3, kernel_size=1, padding=0),
#             nn.SELU(),
#             nn.AvgPool2d(2),
#             nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#             nn.SELU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(num_features // 3, num_features // 3, kernel_size=3, padding=1),
#             nn.SELU(),
#         )
#         self.conv21 = nn.Sequential(
#             nn.Conv2d(2 * num_features // 3, num_features // 3, kernel_size=1, padding=0),
#             nn.SELU(),
#         )
#         self.conv_t = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Conv2d(8 + 2 * num_features // 3, 16, kernel_size=1, padding=0),
#             nn.SELU(),
#             nn.Conv2d(16, 8, kernel_size=3, padding=1),
#             nn.SELU(),
#             nn.Conv2d(8, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.j1 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(6 + 2 * num_features // 3, 12, kernel_size=3, padding=1),
#             nn.SELU(),
#             # nn.Conv2d(12, 6, kernel_size=3, padding=1),
#             # nn.SELU(),
#             nn.Conv2d(12, 3, kernel_size=1, padding=0),
#         )
#         self.j2 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(6 + 2 * num_features // 3, 12, kernel_size=3, padding=1),
#             nn.SELU(),
#             # nn.Conv2d(12, 6, kernel_size=3, padding=1),
#             # nn.SELU(),
#             nn.Conv2d(12, 3, kernel_size=1, padding=0),
#         )
#         self.j3 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(6 + 2 * num_features // 3, 12, kernel_size=3, padding=1),
#             nn.SELU(),
#             # nn.Conv2d(12, 6, kernel_size=3, padding=1),
#             # nn.SELU(),
#             nn.Conv2d(12, 3, kernel_size=1, padding=0),
#         )
#         self.j4 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(6 + 2 * num_features // 3, 12, kernel_size=3, padding=1),
#             nn.SELU(),
#             # nn.Conv2d(12, 6, kernel_size=3, padding=1),
#             # nn.SELU(),
#             nn.Conv2d(12, 3, kernel_size=1, padding=0),
#         )
#         self.attention_fusion = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(24 + 2 * num_features // 3, num_features, kernel_size=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 15, kernel_size=1)
#         )
#         self.conv_y = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(24, 24, kernel_size=1, padding=0),
#             nn.SELU(),
#         )
#         self.reshape_a = Reshape_a(num_features // 3)
#         self.reshape_t = Reshape_t(num_features // 3)
#         self.a_transformer = A_Transformer(num_features // 3)
#         self.t_transformer0 = T_Transformer(num_features // 3)
#         self.p_transformer1 = P_Transformer(num_features // 3)
#         # self.L_t = nn.Linear(64*64, 256)
#
#     def forward(self, x):
#         x0 = x
#         x = (x0 - self.mean) / self.std
#         xm00 = self.conv00(x)
#         # x_a = self.conv0(xm00)
#         # x_a = torch.cat([x_a, xm00], dim=1)
#         # x_t = torch.cat([x_t, xm10], dim=1)
#         x_a0, l, w, pad_l, pad_w = self.reshape_a(xm00)
#         a, a_feature, a_feature_local = self.a_transformer(x_a0)
#         a = a.unsqueeze(-1)
#         l_o = (l - pad_l)
#         w_o = (w - pad_w)
#         # l_o2 = l - pad_l
#         # w_o2 = w - pad_w
#         a_conv = a.detach().repeat(1, 1, x.shape[2], x.shape[3])
#         x_at = torch.cat((x, a_conv), dim=1)
#         xm10 = self.conv01(x_at)
#         x_t = self.conv1(xm10)
#         x_t0 = self.conv11(torch.cat([x_t, xm10], dim=1))
#         x_t0 = self.reshape_t(x_t0)
#         t0 = self.t_transformer0(x_t0, a_feature.detach(), a_feature_local, l, w)
#         # x_conv_t0 = self.conv_t0(x)
#         t0 = t0[:, :, :l_o, :w_o]
#         t0 = torch.cat((t0, x_t, xm10), dim=1)
#         t0 = self.conv_t(t0)
#         t0 = F.upsample(t0, size=x0.size()[2:], mode='bilinear')
#         p0 = ((x0 - a * (1 - t0)) / t0.clamp(min=1e-8)).clamp(min=0., max=1.)
#
#         # xp = self.conv_p(x)
#         x_t2 = torch.cat([p0.detach(), x, t0.detach()], dim=1)
#         x_t20 = self.conv20(x_t2)
#         x_t21 = self.conv2(x_t20)
#         x_t22 = torch.cat([x_t20, x_t21], dim=1)
#         x_t22 = self.conv21(x_t22)
#         x_t2 = self.reshape_t(x_t22)
#
#         y = self.p_transformer1(x_t2, a_feature.detach(), l, w)
#         y = y[:, :, :l_o, :w_o]
#         y = self.conv_y(y)
#         y1, y2, y3, y4 = torch.split(y, 6, dim=1)
#         y = torch.cat((y, x_t21, x_t20), 1)
#         y1 = torch.cat((y1, x_t21, x_t20), 1)
#         y2 = torch.cat((y2, x_t21, x_t20), 1)
#         y3 = torch.cat((y3, x_t21, x_t20), 1)
#         y4 = torch.cat((y4, x_t21, x_t20), 1)
#
#         log_x0 = torch.log(x0.clamp(min=1e-8))
#         log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))
#
#         # J0 = (I - A0 * (1 - T0)) / T0
#         x_phy = p0
#
#         # J2 = I * exp(R2)
#         r1 = F.upsample(self.j1(y1), size=x0.size()[2:], mode='bilinear')
#         x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)
#
#         # J2 = I + R2
#         r2 = F.upsample(self.j2(y2), size=x0.size()[2:], mode='bilinear')
#         x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)
#
#         #
#         r3 = F.upsample(self.j3(y3), size=x0.size()[2:], mode='bilinear')
#         x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)
#
#         # J4 = log(1 + I * R4)
#         r4 = F.upsample(self.j4(y4), size=x0.size()[2:], mode='bilinear')
#         # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
#         x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)
#
#         attention_fusion = F.upsample(self.attention_fusion(y), size=x0.size()[2:], mode='bilinear')
#         x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
#                          torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
#                                       x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
#         x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
#                          torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
#                                       x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
#         x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
#                          torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
#                                       x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
#         x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)
#
#         if self.training:
#             return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t0, a.view(x.size(0), -1)
#         else:
#             return x_fusion


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Reshape_t(nn.Module):
    def __init__(self, num_features):
        super(Reshape_t, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        x = x.view(x.size(0), self.num_features, x.size(2), x.size(3))
        return x

class new_model(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d'):
        super(new_model, self).__init__()
        self.num_features = num_features
        self.cnn01 = nn.Sequential(
                                      nn.Conv2d(64, num_features//2, kernel_size=3, dilation=2, padding=2),
                                      nn.SELU(),
                                      nn.Dropout(0.2),
                                      nn.Conv2d(num_features//2, num_features//2, kernel_size=5, dilation=4, padding=8),
                                      nn.Dropout(0.2),
                                      nn.SELU(),
                                      nn.Conv2d(num_features//2, num_features, kernel_size=5, dilation=8, padding=16),
                                      nn.Dropout(0.2),
                                      nn.SELU(),
                                      )
        self.cnn02 = nn.Sequential(nn.Conv2d(3, num_features//6, kernel_size=1, padding=0),
                                    nn.SELU(),
                                    nn.Conv2d(num_features//6, num_features//6, kernel_size=3, padding=1, stride=2),
                                    nn.SELU(),
                                    nn.Conv2d(num_features // 6, num_features // 4, kernel_size=3, padding=1, stride=2),
                                   nn.SELU(),
                                   nn.Dropout(0.2),
                                   nn.Conv2d(num_features//4, num_features//3, kernel_size=3, padding=1, stride=2),
                                   nn.Dropout(0.2),
                                   nn.SELU(),
                                   nn.Conv2d(num_features//3, num_features//2, kernel_size=3, padding=1, stride=2),
                                   nn.Dropout(0.2),
                                   nn.SELU(),
                                   nn.Conv2d(num_features//2, num_features, kernel_size=3, padding=1, stride=2),
                                   )
        self.reshape_t = Reshape_t(num_features//3)
        self.p_transformer_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=2, dim_feedforward=num_features*2,batch_first=True)
        self.p_transformer = nn.TransformerEncoder(self.p_transformer_layer, num_layers=3)
        self.positional_encoding = PositionalEncoding(num_features)
        self.conv1 = nn.Sequential(nn.Conv2d(1+3, 64, kernel_size=7,stride=2, padding=3))
        # resnext = ResNeXt101()
        #
        # self.layer0 = resnext.layer0
        # self.layer1 = resnext.layer1
        # self.layer2 = resnext.layer2
        # self.layer3 = resnext.layer3
        # self.layer4 = resnext.layer4

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=True)
        del backbone.fc
        self.backbone = backbone
        self.backbone2 = backbone

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1)
        )

        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 5, kernel_size=1)
        )

        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 5, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 5, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 5, kernel_size=1)
        )
        self.attention4 = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 5, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 5, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1)
        )
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=1), nn.SELU(),)
        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):
        x = (x0 - self.mean) / self.std
        backbone = self.backbone

        # x_t_p = self.cnn02(x)
        # x_t_p = x_t_p.reshape(x_t_p.size(0), x_t_p.size(1), x_t_p.size(2)*x_t_p.size(3))
        # x_t_p = x_t_p.transpose(1, 2)
        # x_t_p = self.positional_encoding(x_t_p)
        # x_t_p = self.p_transformer(x_t_p)
        # x_t_p = self.L_pre(x_t_p)
        # x_t_p = torch.mean(x_t_p, dim=1, keepdim=True).unsqueeze(-1)

        layer0 = backbone.conv1(x)
        # x_t_p_r = x_t_p.repeat(1, 1, x.size(2), x.size(3)).detach()
        # layer0 = torch.cat((x, x_t_p_r), 1)
        # layer0 = self.conv1(layer0)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)
        x_t_d = self.cnn01(layer0)
        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        # layere0 = backbone_e.conv1(x)
        # layere0 = backbone_e.bn1(layere0)
        # layere0 = backbone_e.relu(layere0)
        # layere0 = backbone_e.maxpool(layere0)

        # layere1 = backbone_e.layer1(layere0)

        # layer0 = self.layer0(x)
        # layer1 = self.layer1(layer0)
        # layer2 = self.layer2(layer1)
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        x_t = self.down5(x_t_d)
        x_t = F.upsample(x_t, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4, x_t), 1)

        n, c, h, w = down1.size()

        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 5, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :] + \
                x_t * attention_phy[:, 4, :, :, :]
        l0_feature = f_phy.detach()
        f_phy = self.refine(f_phy) + f_phy

        down1 = self.cnn_feature(torch.cat((down1, l0_feature), 1))
        concat = torch.cat((down1, down2, down3, down4, x_t), 1)
        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 5, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :] + \
             x_t * attention1[:, 4, :, :, :]
        f1 = self.refine(f1) + f1

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 5, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :] + \
             x_t * attention2[:, 4, :, :, :]
        f2 = self.refine(f2) + f2

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 5, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :] + \
             x_t * attention3[:, 4, :, :, :]
        f3 = self.refine(f3) + f3

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 5, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
             down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :] + \
             x_t * attention4[:, 4, :, :, :]
        f4 = self.refine(f4) + f4

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(f_phy)
        a = torch.sigmoid(a)
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)

        # J2 = I * exp(R2)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)

        #
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)

        attention_fusion = F.upsample(self.attention_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                      x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
        x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                      x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
        x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                      x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)

        if self.training:
            return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4, t, a.view(x.size(0), -1)
        else:
            return x_fusion


# model2
# class new_model_OHAZE(Base_OHAZE):
#     def __init__(self, num_features=64, arch='resnext101_32x8d'):
#         super(new_model_OHAZE, self).__init__()
#         self.num_features = num_features
#
#         # resnext = ResNeXt101Syn()
#         # self.layer0 = resnext.layer0
#         # self.layer1 = resnext.layer1
#         # self.layer2 = resnext.layer2
#         # self.layer3 = resnext.layer3
#         # self.layer4 = resnext.layer4
#
#         assert arch in ['resnet50', 'resnet101',
#                         'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
#         backbone = models.__dict__[arch](pretrained=True)
#         del backbone.fc
#         self.backbone = backbone
#
#         self.down0 = nn.Sequential(
#             nn.Conv2d(64, num_features, kernel_size=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
#         )
#         self.down1 = nn.Sequential(
#             nn.Conv2d(256, num_features, kernel_size=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
#         )
#         self.down2 = nn.Sequential(
#             nn.Conv2d(512, num_features, kernel_size=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
#         )
#         self.down4 = nn.Sequential(
#             nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
#         )
#         # self.cnn01 = nn.Sequential(
#         #     nn.Conv2d(64, num_features // 2, kernel_size=3, dilation=2, padding=2),
#         #     nn.SELU(),
#         #     nn.Dropout(0.2),
#         #     nn.Conv2d(num_features // 2, num_features // 2, kernel_size=5, dilation=4, padding=8),
#         #     nn.Dropout(0.2),
#         #     nn.SELU(),
#         #     nn.Conv2d(num_features // 2, num_features, kernel_size=5, dilation=8, padding=16),
#         #     nn.Dropout(0.2),
#         #     nn.SELU(),
#         # )
#         self.cnn02 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
#                                    nn.SELU(),
#                                    )
#         # self.p_transformer_layer_a = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256 * 2,
#         #                                                         batch_first=True)
#         # self.p_transformer_a = nn.TransformerEncoder(self.p_transformer_layer_a, num_layers=3)
#         # self.p_transformer_layer_b = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256 * 2,
#         #                                                         batch_first=True)
#         # self.p_transformer_b = nn.TransformerEncoder(self.p_transformer_layer_b, num_layers=3)
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128 * 2,
#                                                                     batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)
#         self.positional_encoding = PositionalEncoding(128)
#         # self.L_a = nn.Linear(256, 128 * num_features // 2)
#         # self.L_b = nn.Linear(256, 128 * num_features // 2)
#         self.x_t_conv = nn.Sequential(
#             nn.Conv2d(128, self.num_features, kernel_size=1), nn.SELU()
#         )
#         self.fuse3 = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
#         )
#         self.fuse2 = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
#         )
#         self.fuse1 = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
#         )
#         self.fuse0 = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
#         )
#         self.fuse_x_t = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
#         )
#         self.fuse3_attention = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
#         )
#         self.fuse2_attention = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
#         )
#         self.fuse1_attention = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
#         )
#         self.fuse0_attention = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
#         )
#         self.x_t_attention = nn.Sequential(
#             nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
#         )
#
#         self.p0 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#         self.p1 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#         self.p2_0 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#         self.p2_1 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#         self.p3_0 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#         self.p3_1 = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 3, kernel_size=1)
#         )
#
#         self.attentional_fusion = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
#             nn.Conv2d(num_features // 2, 12, kernel_size=3, padding=1)
#         )
#
#         # self.vgg = VGGF()
#
#         for m in self.modules():
#             if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
#                 m.inplace = True
#
#     def forward(self, x0):
#         x = (x0 - self.mean_in) / self.std_in
#
#         backbone = self.backbone
#
#         layer0 = backbone.conv1(x)
#         layer0 = backbone.bn1(layer0)
#         layer0 = backbone.relu(layer0)
#
#         t_layer = layer0
#         t_layer = self.cnn02(t_layer)
#         t_layer = F.interpolate(t_layer, size=(16, 16), mode='bilinear').reshape(t_layer.size(0), t_layer.size(1), 256).transpose(1, 2)
#         t_layer = self.positional_encoding(t_layer)
#         t_layer = self.transformer_encoder(t_layer).transpose(1, 2).reshape(t_layer.size(0), t_layer.size(2), 16, 16)
#         # t_layer_a = F.interpolate(t_layer, size=(64, 128), mode='bilinear').transpose(1, 2).flatten(-2)
#         # t_layer_b = F.interpolate(t_layer, size=(128, 64), mode='bilinear').transpose(-1, -2).transpose(1, 2).flatten(-2)
#         # t_layer_a = self.positional_encoding(t_layer_a)
#         # t_layer_b = self.positional_encoding(t_layer_b)
#         # t_layer_a = self.p_transformer_a(t_layer_a)
#         # t_layer_b = self.p_transformer_b(t_layer_b)
#         # t_layer_a = self.L_a(t_layer_a).reshape(t_layer_a.size(0), t_layer_a.size(1), 128, self.num_features//2).transpose(1, 2).transpose(1, 3)
#         # t_layer_b = self.L_b(t_layer_b).reshape(t_layer_b.size(0), t_layer_b.size(1), 128, self.num_features//2).transpose(1, 3)
#
#         layer0 = backbone.maxpool(layer0)
#
#         layer1 = backbone.layer1(layer0)
#         layer2 = backbone.layer2(layer1)
#         layer3 = backbone.layer3(layer2)
#         layer4 = backbone.layer4(layer3)
#
#         # x_t = self.cnn01(layer0)
#
#         down0 = self.down0(layer0)
#         down1 = self.down1(layer1)
#         down2 = self.down2(layer2)
#         down3 = self.down3(layer3)
#         down4 = self.down4(layer4)
#
#         t_layer = self.x_t_conv(t_layer)
#         x_t = F.interpolate(t_layer, size=down4.size()[2:], mode='bilinear')
#         x_t_attention = self.x_t_attention(torch.cat((down4, x_t), 1))
#         down4 = x_t + self.fuse_x_t(torch.cat((x_t, x_t_attention * down4), 1))
#
#         down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
#         fuse3_attention = self.fuse3_attention(torch.cat((down4, down3), 1))
#         f = down4 + self.fuse3(torch.cat((down4, fuse3_attention * down3), 1))
#
#         f = F.upsample(f, size=down2.size()[2:], mode='bilinear')
#         fuse2_attention = self.fuse2_attention(torch.cat((f, down2), 1))
#         f = f + self.fuse2(torch.cat((f, fuse2_attention * down2), 1))
#
#         f = F.upsample(f, size=down1.size()[2:], mode='bilinear')
#         fuse1_attention = self.fuse1_attention(torch.cat((f, down1), 1))
#         f = f + self.fuse1(torch.cat((f, fuse1_attention * down1), 1))
#
#         f = F.upsample(f, size=down0.size()[2:], mode='bilinear')
#         fuse0_attention = self.fuse0_attention(torch.cat((f, down0), 1))
#         f = f + self.fuse0(torch.cat((f, fuse0_attention * down0), 1))
#
#         # t_layer_a = F.interpolate(t_layer_a, size=down0.size()[2:], mode='bilinear')
#         # t_layer_b = F.interpolate(t_layer_b, size=down0.size()[2:], mode='bilinear')
#         # x_t = torch.cat((t_layer_a, t_layer_b), 1)
#         # t_layer = torch.cat((t_layer_a, t_layer_b), 1)
#         # x_t_attention = self.x_t_attention(torch.cat((f, x_t), 1))
#         # f = f + self.fuse_x_t(torch.cat((f, x_t_attention * x_t), 1))
#
#         log_x0 = torch.log(x0.clamp(min=1e-8))
#         log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))
#
#         x_p0 = torch.exp(log_x0 + F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0, max=1)
#
#         x_p1 = ((x + F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out) \
#             .clamp(min=0., max=1.)
#
#         log_x_p2_0 = torch.log(
#             ((x + F.upsample(self.p2_0(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)
#             .clamp(min=1e-8))
#         x_p2 = torch.exp(log_x_p2_0 + F.upsample(self.p2_1(f), size=x0.size()[2:], mode='bilinear')) \
#             .clamp(min=0., max=1.)
#
#         log_x_p3_0 = torch.exp(log_log_x0_inverse + F.upsample(self.p3_0(f), size=x0.size()[2:], mode='bilinear'))
#         x_p3 = torch.exp(-log_x_p3_0 + F.upsample(self.p3_1(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0,
#                                                                                                             max=1)
#
#         attention_fusion = F.upsample(self.attentional_fusion(f), size=x0.size()[2:], mode='bilinear')
#         x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 4, :, :], 1) * torch.stack(
#             (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
#                               torch.sum(F.softmax(attention_fusion[:, 4: 8, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
#                                                                                                      x_p1[:, 1, :, :],
#                                                                                                      x_p2[:, 1, :, :],
#                                                                                                      x_p3[:, 1, :, :]),
#                                                                                                     1), 1, True),
#                               torch.sum(F.softmax(attention_fusion[:, 8:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
#                                                                                                    x_p1[:, 2, :, :],
#                                                                                                    x_p2[:, 2, :, :],
#                                                                                                    x_p3[:, 2, :, :]),
#                                                                                                   1), 1, True)),
#                              1).clamp(min=0, max=1)
#
#         if self.training:
#             return x_fusion, x_p0, x_p1, x_p2, x_p3
#         else:
#             return x_fusion


# model1
class new_model_OHAZE(Base_OHAZE):
    def __init__(self, num_features=64, arch='resnext101_32x8d'):
        super(new_model_OHAZE, self).__init__()
        self.num_features = num_features

        # resnext = ResNeXt101Syn()
        # self.layer0 = resnext.layer0
        # self.layer1 = resnext.layer1
        # self.layer2 = resnext.layer2
        # self.layer3 = resnext.layer3
        # self.layer4 = resnext.layer4

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        backbone = models.__dict__[arch](pretrained=True)
        del backbone.fc
        self.backbone = backbone

        self.down0 = nn.Sequential(
            nn.Conv2d(64, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )
        # self.cnn01 = nn.Sequential(
        #     nn.Conv2d(64, num_features // 2, kernel_size=3, dilation=2, padding=2),
        #     nn.SELU(),
        #     nn.Dropout(0.2),
        #     nn.Conv2d(num_features // 2, num_features // 2, kernel_size=5, dilation=4, padding=8),
        #     nn.Dropout(0.2),
        #     nn.SELU(),
        #     nn.Conv2d(num_features // 2, num_features, kernel_size=5, dilation=8, padding=16),
        #     nn.Dropout(0.2),
        #     nn.SELU(),
        # )
        self.cnn02 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1),
                                   nn.SELU(),
                                   )
        self.p_transformer_layer_a = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256 * 2,
                                                                batch_first=True)
        self.p_transformer_a = nn.TransformerEncoder(self.p_transformer_layer_a, num_layers=3)
        self.p_transformer_layer_b = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256 * 2,
                                                                batch_first=True)
        self.p_transformer_b = nn.TransformerEncoder(self.p_transformer_layer_b, num_layers=3)
        self.positional_encoding = PositionalEncoding(256)
        self.L_a = nn.Linear(256, 128 * num_features // 2)
        self.L_b = nn.Linear(256, 128 * num_features // 2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse0 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse_x_t = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        self.fuse3_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse2_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse1_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.fuse0_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )
        self.x_t_attention = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.Sigmoid()
        )

        self.p0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p2_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_0 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.p3_1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attentional_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 12, kernel_size=3, padding=1)
        )

        # self.vgg = VGGF()

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0):
        x = (x0 - self.mean_in) / self.std_in

        backbone = self.backbone

        layer0 = backbone.conv1(x)
        layer0 = backbone.bn1(layer0)
        layer0 = backbone.relu(layer0)

        t_layer = layer0
        t_layer = self.cnn02(t_layer)
        t_layer_a = F.interpolate(t_layer, size=(64, 128), mode='bilinear').transpose(1, 2).flatten(-2)
        t_layer_b = F.interpolate(t_layer, size=(128, 64), mode='bilinear').transpose(-1, -2).transpose(1, 2).flatten(-2)
        t_layer_a = self.positional_encoding(t_layer_a)
        t_layer_b = self.positional_encoding(t_layer_b)
        t_layer_a = self.p_transformer_a(t_layer_a)
        t_layer_b = self.p_transformer_b(t_layer_b)
        t_layer_a = self.L_a(t_layer_a).reshape(t_layer_a.size(0), t_layer_a.size(1), 128, self.num_features//2).transpose(1, 2).transpose(1, 3)
        t_layer_b = self.L_b(t_layer_b).reshape(t_layer_b.size(0), t_layer_b.size(1), 128, self.num_features//2).transpose(1, 3)

        layer0 = backbone.maxpool(layer0)

        layer1 = backbone.layer1(layer0)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        # x_t = self.cnn01(layer0)

        down0 = self.down0(layer0)
        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        fuse3_attention = self.fuse3_attention(torch.cat((down4, down3), 1))
        f = down4 + self.fuse3(torch.cat((down4, fuse3_attention * down3), 1))

        f = F.upsample(f, size=down2.size()[2:], mode='bilinear')
        fuse2_attention = self.fuse2_attention(torch.cat((f, down2), 1))
        f = f + self.fuse2(torch.cat((f, fuse2_attention * down2), 1))

        f = F.upsample(f, size=down1.size()[2:], mode='bilinear')
        fuse1_attention = self.fuse1_attention(torch.cat((f, down1), 1))
        f = f + self.fuse1(torch.cat((f, fuse1_attention * down1), 1))

        f = F.upsample(f, size=down0.size()[2:], mode='bilinear')
        fuse0_attention = self.fuse0_attention(torch.cat((f, down0), 1))
        f = f + self.fuse0(torch.cat((f, fuse0_attention * down0), 1))

        t_layer_a = F.interpolate(t_layer_a, size=down0.size()[2:], mode='bilinear')
        t_layer_b = F.interpolate(t_layer_b, size=down0.size()[2:], mode='bilinear')
        x_t = torch.cat((t_layer_a, t_layer_b), 1)
        # t_layer = torch.cat((t_layer_a, t_layer_b), 1)
        x_t_attention = self.x_t_attention(torch.cat((f, x_t), 1))
        f = f + self.fuse_x_t(torch.cat((f, x_t_attention * x_t), 1))

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        x_p0 = torch.exp(log_x0 + F.upsample(self.p0(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0, max=1)

        x_p1 = ((x + F.upsample(self.p1(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out) \
            .clamp(min=0., max=1.)

        log_x_p2_0 = torch.log(
            ((x + F.upsample(self.p2_0(f), size=x0.size()[2:], mode='bilinear')) * self.std_out + self.mean_out)
            .clamp(min=1e-8))
        x_p2 = torch.exp(log_x_p2_0 + F.upsample(self.p2_1(f), size=x0.size()[2:], mode='bilinear')) \
            .clamp(min=0., max=1.)

        log_x_p3_0 = torch.exp(log_log_x0_inverse + F.upsample(self.p3_0(f), size=x0.size()[2:], mode='bilinear'))
        x_p3 = torch.exp(-log_x_p3_0 + F.upsample(self.p3_1(f), size=x0.size()[2:], mode='bilinear')).clamp(min=0,
                                                                                                            max=1)

        attention_fusion = F.upsample(self.attentional_fusion(f), size=x0.size()[2:], mode='bilinear')
        x_fusion = torch.cat((torch.sum(F.softmax(attention_fusion[:, : 4, :, :], 1) * torch.stack(
            (x_p0[:, 0, :, :], x_p1[:, 0, :, :], x_p2[:, 0, :, :], x_p3[:, 0, :, :]), 1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 4: 8, :, :], 1) * torch.stack((x_p0[:, 1, :, :],
                                                                                                     x_p1[:, 1, :, :],
                                                                                                     x_p2[:, 1, :, :],
                                                                                                     x_p3[:, 1, :, :]),
                                                                                                    1), 1, True),
                              torch.sum(F.softmax(attention_fusion[:, 8:, :, :], 1) * torch.stack((x_p0[:, 2, :, :],
                                                                                                   x_p1[:, 2, :, :],
                                                                                                   x_p2[:, 2, :, :],
                                                                                                   x_p3[:, 2, :, :]),
                                                                                                  1), 1, True)),
                             1).clamp(min=0, max=1)

        if self.training:
            return x_fusion, x_p0, x_p1, x_p2, x_p3
        else:
            return x_fusion
