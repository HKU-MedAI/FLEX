import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    """
    InfoNCE loss for self-supervised learning.
    
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
    and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        """
        Args:
            temperature: Logits are divided by temperature before calculating the cross entropy.
            reduction: Reduction method applied to the output.
                Value must be one of ['none', 'sum', 'mean'].
            negative_mode: Determines how the negative_keys are handled.
                Value must be one of ['paired', 'unpaired'].
                If 'paired', then each query sample is paired with a number of negative keys.
                If 'unpaired', then the negative keys are all unrelated to any positive key.
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        """
        Calculate InfoNCE loss
        
        Args:
            query: (N, D) Tensor with query samples (e.g. embeddings of the input)
            positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input)
            negative_keys: Tensor with negative samples (e.g. embeddings of other inputs)
                If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor
                If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor
                If None, then the negative keys are the positive keys for the other samples
                
        Returns:
            InfoNCE loss value
        """
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, 
             reduction='mean', negative_mode='unpaired'):
    """
    InfoNCE loss function
    
    Args:
        query: Query embeddings
        positive_key: Positive key embeddings
        negative_keys: Negative key embeddings
        temperature: Temperature parameter for scaling logits
        reduction: Reduction method for the loss
        negative_mode: Mode for handling negative keys
        
    Returns:
        InfoNCE loss value
    """
    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    """Transpose the last two dimensions of a tensor"""
    return x.transpose(-2, -1)


def normalize(*xs):
    """Normalize tensors along the last dimension"""
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
