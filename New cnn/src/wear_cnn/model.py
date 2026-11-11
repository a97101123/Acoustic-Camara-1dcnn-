from __future__ import annotations


def resolve_device(torch, device: str) -> str:
    """Resolve automatic device selection without importing PyTorch eagerly."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(torch, input_channels: int, num_classes: int, options):
    from .configuration import model_settings

    model_settings(options)
    if type(num_classes) is not int or num_classes not in (2, 3):
        raise ValueError("num_classes must be 2 or 3")
    nn = torch.nn

    class ChannelLayerNorm(nn.LayerNorm):
        def forward(self, x):
            return super().forward(x.transpose(1, 2)).transpose(1, 2)

    def normalization(channels):
        epsilon = options["normalization_epsilon"]
        if options["normalization"] == "layer":
            return ChannelLayerNorm(channels, eps=epsilon)
        if options["normalization"] == "group":
            return nn.GroupNorm(options["norm_groups"], channels, eps=epsilon)
        return nn.Identity()

    class FeatureCNN(nn.Module):
        def __init__(self):
            super().__init__()
            first, second = options["filters"]
            kernel = options["kernel_size"]
            self.conv1 = nn.Conv1d(input_channels, first, kernel, padding=kernel - 1)
            self.norm1 = normalization(first)
            self.conv2 = nn.Conv1d(first, second, kernel, padding=kernel - 1)
            self.norm2 = normalization(second)
            self.dropout = nn.Dropout(options["dropout"])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(second, num_classes)

        def forward(self, x):
            # Left padding only -> causal convolution.
            x = self.conv1(x)[:, :, : x.shape[-1]]
            x = self.norm1(torch.relu(x))
            x = self.conv2(x)[:, :, : x.shape[-1]]
            x = self.norm2(torch.relu(x))
            return self.head(self.pool(self.dropout(x)).squeeze(-1))

    return FeatureCNN()
