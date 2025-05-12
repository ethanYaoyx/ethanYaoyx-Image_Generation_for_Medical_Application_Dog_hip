import torch
import torch.nn as nn
from mmcv.ops import DeformConv2dPack
from timm.models.vision_transformer import Block
from mmpose.registry import MODELS
from .hrnet import HRNet


class DeformableFusionLayer(nn.Module):
    """Fusion layer using Deformable Convolution."""
    def __init__(self, in_channels, out_channels):
        super(DeformableFusionLayer, self).__init__()
        self.deform_conv = DeformConv2dPack(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        return self.relu(x)


class TransformerFusionLayer(nn.Module):
    """Fusion layer using Vision Transformer."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super(TransformerFusionLayer, self).__init__()
        self.vit_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)  # Flatten spatial dimensions to (B, HW, C)
        x = self.vit_block(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # Restore spatial dimensions
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = x.mean((2, 3))  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y


@MODELS.register_module()
class CustomHRNet(HRNet):
    """Custom HRNet with enhanced Multi-scale Feature Fusion."""
    def __init__(self, extra, in_channels=3, se_reduction=16, **kwargs):
        self.se_reduction = se_reduction
        super(CustomHRNet, self).__init__(extra, in_channels=in_channels, **kwargs)

        # Define final_conv to adjust output channels to match RTMCCHead
        self.final_conv = nn.Conv2d(
            in_channels=64,  # Stage 4 output channels
            out_channels=256,  # Expected by RTMCCHead
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_fuse_layers(self):
        """Override `_make_fuse_layers` to include Deformable Convolution and Transformer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsampling using Deformable Convolution
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.Upsample(scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsampling using Transformer Fusion Layer
                    fuse_layer.append(
                        TransformerFusionLayer(dim=in_channels[j], num_heads=4))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Override `_make_layer` to include SE blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(
            block(in_channels, out_channels, stride=stride, downsample=downsample)
        )
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        # Add SE block after residual blocks
        layers.append(SEBlock(out_channels, reduction=self.se_reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward with shape debugging."""
        # Stem net
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 1
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        if self.training:
            print(f"Stage 2 Outputs: {[y.shape for y in y_list]}")

        # Stage 2
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        if self.training:
            print(f"Stage 3 Outputs: {[y.shape for y in y_list]}")

        # Stage 3
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        if self.training:
            print(f"Stage 4 Outputs: {[y.shape for y in y_list]}")

        # Ensure correct output shape
        final_output = y_list[0]  # Use the first branch as an example
        final_output = self.final_conv(final_output)  # Adjust channels to 256

        print(f"Final output shape: {final_output.shape}")
        return final_output
