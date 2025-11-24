import torch
import torch.nn as nn

class AxialEncoding(nn.Module):
    def __init__(self, d_out, context_length, n1, n2, d1, d2):
        super().__init__()
        
        assert (d1 + d2 == d_out), \
            "d1 + d2 must equal d_out"
        assert (n1 * n2 == context_length), \
            "n1 + n2 must equal the total context length"
        
        self.d1, self.d2 = d1, d2
        self.n1, self.n2 = n1, n2

        self.params1 = nn.Parameter(torch.rand(n1,d1))
        self.params2 = nn.Parameter(torch.rand(n2,d2))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        i = torch.arange(num_tokens)
        r = i % self.n1
        s = i // self.n1

        pe1 = self.params1[r]
        pe2 = self.params2[s]

        P = torch.cat([pe1, pe2], dim=-1)
        return x + P.unsqueeze(0)
