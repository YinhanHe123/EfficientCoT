import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningLoss(nn.Module):
    """Loss function for reasoning similarity"""
    def __init__(self):
        super().__init__()

    def forward(self, ground_truth_embeds, contemp_embeds):
        """
        Calculate similarity-based loss between ground truth and contemplation embeddings

        Args:
            ground_truth_embeds: Embeddings of ground truth reasoning
            contemp_embeds: Embeddings of contemplation tokens

        Returns:
            Tensor: Loss value (lower means more similar)
        """
        # Normalize embeddings
        ground_truth_embeds = F.normalize(ground_truth_embeds, p=2, dim=1)
        contemp_embeds = F.normalize(contemp_embeds, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.sum(ground_truth_embeds * contemp_embeds, dim=1)

        # Loss = 1 - similarity (lower value means higher similarity)
        loss = 1 - similarity

        return loss.mean()

class AnswerLoss(nn.Module):
    """Loss function for answer accuracy"""
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, target_ids):
        """
        Calculate cross-entropy loss on answer tokens

        Args:
            logits: Predicted token logits from model (B, L, V)
            target_ids: Target token IDs (B, L)

        Returns:
            Tensor: Loss value
        """
        # Shift logits and targets for teacher forcing
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # Calculate loss
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss

class CombinedLoss(nn.Module):
    """Combined loss function with weighted reasoning and answer components"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.reasoning_loss = ReasoningLoss()

    def forward(self, reason_gt_embeds, contemp_embeds, ans_loss):
        """
        Calculate combined loss

        Args:
            reason_gt_embeds: Ground truth reasoning embeddings
            contemp_embeds: Contemplation token embeddings
            ans_loss: Answer loss value

        Returns:
            Tensor: Combined loss value
        """
        # Calculate reasoning loss
        l_reason = self.reasoning_loss(reason_gt_embeds, contemp_embeds)

        # Combine losses with weighting
        total_loss = self.alpha * l_reason + (1 - self.alpha) * ans_loss

        return total_loss, l_reason, ans_loss