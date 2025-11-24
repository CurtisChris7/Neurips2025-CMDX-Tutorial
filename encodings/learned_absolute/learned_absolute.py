import torch
import torch.nn as nn

class LearnedAbsoluteEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.abs_params = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        pos = self.abs_params[:seq_len, :].unsqueeze(0)
        return x + pos
