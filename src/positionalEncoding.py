import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freq_bands, include_input=True):
        """
        Args:
            input_dim: Dimensionality of the input (e.g., 3 for (x, y, z)).
            num_freq_bands: Number of frequency bands for encoding.
            include_input: Whether to include the original input in the output.
        """
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_freq_bands = num_freq_bands
        self.include_input = include_input

        self.freq_bands = 2 ** torch.arange(num_freq_bands, dtype=torch.float32)

    def forward(self, x):
        """
        Args:
            x: Input positions of shape (N, input_dim).
        Returns:
            Encoded positions of shape (N, output_dim).
        """
        encoded = [x] if self.include_input else []

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))

        return torch.cat(encoded, dim=-1)