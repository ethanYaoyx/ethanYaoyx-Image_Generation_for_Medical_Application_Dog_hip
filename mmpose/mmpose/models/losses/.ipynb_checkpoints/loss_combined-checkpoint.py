# loss_combined.py
import torch
import torch.nn as nn
from mmpose.registry import MODELS
import torch.nn.functional as F
from mmpose.registry import MODELS
from .classification_loss import BCELoss, KLDiscretLoss 

@MODELS.register_module()
class BCEKLDCombinedLoss(nn.Module):
    """Custom loss combining Binary Cross Entropy (BCE) and KL Divergence (KLD) loss.

    Args:
        bce_weight (float): Weight for the BCE loss. Default: 1.0.
        kld_weight (float): Weight for the KLD loss. Default: 1.0.
        use_target_weight (bool): Option to use weighted loss for BCE. Default: True.
        beta (float): Temperature factor for softmax in KLD loss. Default: 1.0.
        label_softmax (bool): Whether to apply softmax on labels in KLD loss. Default: True.
    """

    def __init__(self, bce_weight=1.0, kld_weight=1.0, use_target_weight=True, beta=1.0, label_softmax=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.kld_weight = kld_weight

        # Initialize BCE and KLD loss functions from classification_loss.py
        self.bce_loss_fn = BCELoss(use_target_weight=use_target_weight)
        self.kld_loss_fn = KLDiscretLoss(beta=beta, label_softmax=label_softmax, use_target_weight=use_target_weight)

    def forward(self, pred, target, target_weight=None):
        """Compute combined BCE and KLD loss.

        Args:
            pred (tuple[Tensor, Tensor]): Predicted output (pred_x, pred_y).
            target (tuple[Tensor, Tensor]): Ground truth (target_x, target_y).
            target_weight (Tensor, optional): Weights for different keypoints. Defaults to None.

        Returns:
            Tensor: Combined loss value.
        """
        # Unpack the predicted and target outputs
        pred_x, pred_y = pred
        target_x, target_y = target

        
        pred_x = torch.sigmoid(pred_x)
        pred_y = torch.sigmoid(pred_y)
        
        # Compute BCE loss for both x and y separately
        bce_loss_x = self.bce_loss_fn(pred_x, target_x, target_weight)
        bce_loss_y = self.bce_loss_fn(pred_y, target_y, target_weight)

        bce_loss = bce_loss_x + bce_loss_y

        # Compute KLD loss for both x and y
        kld_loss = self.kld_loss_fn(pred, target, target_weight)

        # Combine BCE and KLD losses with respective weights
        total_loss = self.bce_weight * bce_loss + self.kld_weight * kld_loss

        # Print the individual and total losses for debugging
       
        print(f"BCE Loss: {bce_loss.item():.4f}, KLD Loss: {kld_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        return total_loss