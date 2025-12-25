import torch
import torch.nn as nn

class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss:
    Weights samples based on their difficulty. Hard samples (high loss/low prob) get higher weights.
    Formula: Loss = (1 - p_t)^gamma * CTC_Loss
    where p_t = exp(-CTC_Loss) is an approximation of the likelihood.
    """
    def __init__(self, blank=0, gamma=2.0, reduction='mean', zero_infinity=True):
        super(FocalCTCLoss, self).__init__()
        # Must set reduction='none' to apply weights per-sample before averaging
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 1. Calculate standard CTC loss for each sample in the batch
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Calculate Focal Weights
        with torch.no_grad():
            # Convert loss back to probability space: p ~ exp(-loss)
            prob = torch.exp(-ctc_loss)
            # Weight = (1 - p)^gamma
            # If p is high (easy sample), weight -> 0
            # If p is low (hard sample), weight -> 1
            weights = (1.0 - prob) ** self.gamma
            
        # 3. Apply weights
        loss = weights * ctc_loss
        
        # 4. Apply reduction (Mean or Sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss