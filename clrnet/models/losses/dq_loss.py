import torch
import torch.nn as nn
import torch.nn.functional as F


def dequity_loss_weight(p, eta=1.0, gamma=5.0):
    """Calculate the dequity loss weight.
    Args:
        p (float): The probability of the sample.
        eta (float): The parameter to control the weight.
        gamma (float): The parameter to control the weight.
    Returns:
        float: The dequity loss weight.
    """
    return (eta + (1 - p) ** gamma) / (eta + 1)
