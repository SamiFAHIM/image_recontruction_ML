import torch
import torch.nn as nn
class Weight_Decay_Loss(nn.Module):
    def __init__(self, loss, lambda_l1=1e-4, lambda_l2=01e-4):
        """
        Custom loss function with L1 and L2 regularization (weight decay).
        
        Args:
            loss (nn.Module): Base loss function (e.g., MSELoss).
            lambda_l1 (float): Coefficient for L1 regularization.
            lambda_l2 (float): Coefficient for L2 regularization.
        """
        super(Weight_Decay_Loss, self).__init__()
        self.loss = loss  # Base loss function (e.g., MSE)
        self.lambda_l1 = lambda_l1  # Coefficient for L1 regularization
        self.lambda_l2 = lambda_l2  # Coefficient for L2 regularization

    def forward(self, x, y, net):
        """
        Compute the loss with L1 and L2 regularization.
        
        Args:
            x (torch.Tensor): Predictions from the model.
            y (torch.Tensor): Ground truth targets.
            net (nn.Module): The model to regularize.
        
        Returns:
            torch.Tensor: Total loss (base loss + regularization terms).
        """
        # Compute the base loss (e.g., MSE)
        base_loss = self.loss(x, y)

        # Compute the L1 and L2 regularization terms
        l1_reg = 0.0
        l2_reg = 0.0
        for param in net.parameters():
            if param.requires_grad:  # Only regularize trainable parameters
                l1_reg += torch.norm(param, 1)  # L1 norm
                l2_reg += torch.norm(param, 2) ** 2  # Square of L2 norm

        # Combine the base loss with regularization terms
        total_loss = base_loss + self.lambda_l1 * l1_reg + self.lambda_l2 * l2_reg
        return total_loss