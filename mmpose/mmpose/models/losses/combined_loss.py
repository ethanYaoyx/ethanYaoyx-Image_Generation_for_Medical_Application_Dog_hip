import torch
import torch.nn as nn
from mmpose.registry import MODELS
from mmpose.models.losses import KLDiscretLoss, WingLoss  # 确保已导入这两个损失类

@MODELS.register_module()
class CombinedLoss(nn.Module):
    def __init__(self, kld_params=None, wing_params=None, weights=[0.5, 0.5]):
        super(CombinedLoss, self).__init__()
        
        # 默认参数设置
        if kld_params is None:
            kld_params = {'use_target_weight': True, 'beta': 10.0, 'label_softmax': True}
        if wing_params is None:
            wing_params = {'omega': 10, 'epsilon': 2, 'use_target_weight': True}
        
        # 初始化子损失函数
        self.kld_loss = KLDiscretLoss(**kld_params)
        self.wing_loss = WingLoss(**wing_params)
        
        # 损失权重
        self.weights = weights

    def forward(self, pred, target, weight=None):
        # 计算每个损失的值
        kld_loss_value = self.kld_loss(pred, target, weight)
        wing_loss_value = self.wing_loss(pred, target, weight)
        
        # 组合损失值
        total_loss = self.weights[0] * kld_loss_value + self.weights[1] * wing_loss_value
        return total_loss