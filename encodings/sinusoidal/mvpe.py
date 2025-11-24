import torch
import torch.nn as nn

class MVPE(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000.0, k=1000.0):
        super().__init__()

        # Compute position (row) and dimension (column) indices
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = base ** (-torch.arange(0, d_model).float() / d_model)  # (d_model,)

        # Outer product → (max_len, d_model)
        angle_rates = position * div_term  # broadcasting

        # Apply sin to even, cos to odd
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle_rates[:, 0::2] * k)
        pe[:, 1::2] = torch.cos(angle_rates[:, 1::2] * k)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
