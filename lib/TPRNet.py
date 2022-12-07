import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.Res2Net_v1b import res2net50_v1b_26w_4s

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))  # [B, 14*14, 384]
        fea_1_16 = self.encoderlayer(fea_1_16)
        return fea_1_16


class Transformer_Decoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super(Transformer_Decoder, self).__init__()
        self.token_trans = token_Transformer(embed_dim, depth, num_heads, mlp_ratio=3.)
        self.fc_96_384 = nn.Linear(96, 384)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False)
        self.pre_1_16 = nn.Linear(384, 1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, z):
        x5 = x
        y4 = self.downsample2(y)
        z3 = self.downsample4(z)
        feat_t = torch.cat([x5, y4, z3], 1)  # [B, 96, 11, 11]
        B, Ct, Ht, Wt = feat_t.shape
        feat_t = feat_t.view(B, Ct, -1).transpose(1, 2)
        feat_t = self.fc_96_384(feat_t)  # [B, 11*11, 384]
        Tt = self.token_trans(feat_t)
        mask_x = self.pre_1_16(Tt)
        mask_x = mask_x.transpose(1, 2).reshape(B, 1, Ht, Wt)
        return mask_x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class RCU(nn.Module):
    def __init__(self, channel, subchannel):
        super(RCU, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)
        self.conv1_32 = nn.Conv2d(1, 32, 3, padding=1)
        self.query_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.query_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.query_conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        y1 = y
        if self.group == 1:
            x_cat = torch.cat((x, y1), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1),
                              1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1,
                               xs[16], y1, xs[17], y1, xs[18], y1, xs[19], y1, xs[20], y1, xs[21], y1, xs[22], y1,
                               xs[23], y1,
                               xs[24], y1, xs[25], y1, xs[26], y1, xs[27], y1, xs[28], y1, xs[29], y1, xs[30], y1,
                               xs[31], y1),
                              1)
        else:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1,
                               xs[16], y1, xs[17], y1, xs[18], y1, xs[19], y1, xs[20], y1, xs[21], y1, xs[22], y1,
                               xs[23], y1,
                               xs[24], y1, xs[25], y1, xs[26], y1, xs[27], y1, xs[28], y1, xs[29], y1, xs[30], y1,
                               xs[31], y1,
                               xs[32], y1, xs[33], y1, xs[34], y1, xs[35], y1, xs[36], y1, xs[37], y1, xs[38], y1,
                               xs[39], y1,
                               xs[40], y1, xs[41], y1, xs[42], y1, xs[43], y1, xs[44], y1, xs[45], y1, xs[46], y1,
                               xs[47], y1,
                               xs[48], y1, xs[49], y1, xs[50], y1, xs[51], y1, xs[52], y1, xs[53], y1, xs[54], y1,
                               xs[55], y1,
                               xs[56], y1, xs[57], y1, xs[58], y1, xs[59], y1, xs[60], y1, xs[61], y1, xs[62], y1,
                               xs[63], y1),
                              1)

        x_cat = self.conv(x_cat)
        # --------------并发特征---------------#
        x1_co = x_cat
        x2_co = y
        x2_co = self.conv1_32(x2_co)

        B1, C1, H1, W1 = x1_co.size()

        x_query1 = self.query_conv1(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]
        x_key1 = self.key_conv1(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]

        x_query2 = self.query_conv2(x1_co).view(B1, -1, W1 * H1)
        x_key2 = self.key_conv2(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]

        B2, C2, H2, W2 = x2_co.size()
        y_query = self.query_conv3(x2_co).view(B2, -1, W2 * H2)  # [b, c, hw]
        y_key = self.key_conv3(x2_co).view(B2, -1, W2 * H2)  # [b, c, hw]

        x_hw = torch.bmm(x_query1.permute(0, 2, 1), x_key2)  # [b, h1w1, h1w1]
        x_hw = F.softmax(x_hw, dim=-1)  # [b, h1w1, h1w1]

        x_c = torch.bmm(x_key1, x_query2.permute(0, 2, 1))  # [b, c1, c1]
        x_c = F.softmax(x_c, dim=-1)  # [b, c1, c1]

        xy_hw = torch.bmm(y_query, x_hw)  # [b, c2, h1w1]
        xy_c = torch.bmm(x_c.permute(0, 2, 1), y_key)  # [b, c1, h2w2]

        xy_hw = xy_hw.view(B1, 32, H1, W1)  # [b, 32, h, w]
        xy_c = xy_c.view(B2, 32, H2, W2)  # [b, 32, h, w]

        out_final = torch.cat([xy_hw, xy_c], 1)  # [b, 64, h, w]
        out_final = self.conv6(out_final)

        x = x + out_final
        y = y + self.score(x)
        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = RCU(channel, 8)

    def forward(self, x, y):
        y = -1 * (torch.sigmoid(y)) + 1
        x, y = self.weak_gra(x, y)
        return y

class SIEM(nn.Module):
    def __init__(self, channel):
        super(SIEM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.conv_1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_4 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_5 = BasicConv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv1_32 = nn.Conv2d(1, 32, 1)
        self.conv256_32 = nn.Conv2d(256, 32, 1)
        self.conv64_32 = nn.Conv2d(64, 32, 1)

    def forward(self, x1, x2, y):
        x1 = self.conv64_32(x1)
        x2 = self.conv256_32(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv_5(x)  # 88 88
        y = self.conv1_32(y)  # 44 44

        left1 = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)  # 88
        right1 = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)  # 44

        left = self.conv_1(left1 * x)
        right = self.conv_2(right1 * y)
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear', align_corners=True)

        x_co = left + right

        atten = self.avgpool(x_co)
        atten = torch.sigmoid(self.conv_atten(atten))

        out = torch.mul(x_co, atten) + x_co

        out = self.conv_3(out)
        out = self.conv_4(out)

        return out


class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- Res2Net Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Transformer_Decoder ----
        self.TD = Transformer_Decoder(384, 4, 6)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)
        # -----SIEM--------
        self.siem = SIEM(channel)
        self.conv32_1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        # Receptive Field Block
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        # Transformer_Decoder
        S_g = self.TD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=32, mode='bilinear')  # Sup-1 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ----stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=1, mode='bilinear')
        ra5_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra5_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ----stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra4_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra4_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ----stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra3_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---SIEM ----#
        S_2 = self.siem(x, x1, S_3)
        S_2_pred = F.interpolate(self.conv32_1(S_2), scale_factor=4, mode='bilinear')
        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, S_2_pred


if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
