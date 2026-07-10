# pumml/src/models/pumml_model.py
#
# CNN architecture from the paper
#
# Architecture:
#   Input  : (N, 3, 36, 36)  three-channel pileup image
#   Conv1  : 10 filters, 6x6, stride 2, zero-pad 2  -> (N, 10, 18, 18)
#   ReLU
#   Conv2  : 10 filters, 6x6, stride 2, zero-pad 2  -> (N, 10,  9,  9)
#   ReLU
#   Conv3  : 1  filter,  1x1, stride 1, no pad       -> (N,  1,  9,  9)
#   ReLU
#   Output : (N, 1, 9, 9)  predicted neutral LV pT image
#
# Total parameters: 4,711  (intentionally tiny, paper Section 2)
#
# Input channel order (must match dataset.py and compare.py):
#   channel 0 (RED)   : all neutral pT      (upsampled 9->36)
#   channel 1 (GREEN) : charged pileup pT   (36x36)
#   channel 2 (BLUE)  : charged LV pT       (36x36)
#
# Output:
#   9x9 predicted neutral LV pT image (before upsampling)

import torch
import torch.nn as nn

class PUMMLNet(nn.Module):
    """
    PUMML convolutional neural network.
    Paper-exact architecture: 2 strided conv layers + 1x1 projection.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Conv1: 6x6 filter, stride 2, pad 2 -> 36->18
            nn.ZeroPad2d(2),
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=6,
                stride=2,
                padding=0,    # padding handled by ZeroPad2d above
                bias=True,
            ),
            nn.ReLU(),

            # Conv2: 6x6 filter, stride 2, pad 2 -> 18->9
            nn.ZeroPad2d(2),
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=6,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),

            # Conv3: 1x1 projection -> 10 channels -> 1 channel
            nn.Conv2d(
                in_channels=10,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
        )

        # He-uniform initialisation (paper Section 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, 36, 36)  stacked input channels

        Returns
        -------
        (N, 1, 9, 9)  predicted neutral LV pT image
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper: squeeze output to (N, 9, 9) and clip to >= 0.
        ReLU already ensures non-negative, but explicit clip guards
        against floating-point edge cases.
        """
        with torch.no_grad():
            out = self.forward(x)          # (N, 1, 9, 9)
            out = out.squeeze(1)           # (N, 9, 9)
            return torch.clamp(out, min=0.0)

    @staticmethod
    def count_parameters() -> int:
        """Return total number of trainable parameters."""
        model = PUMMLNet()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PUMMLNet()
    print(model)
    n = model.count_parameters()
    print(f"\nTotal trainable parameters: {n:,}")
    print(f"Paper reports: 4,711")

    # quick shape check
    x = torch.zeros(4, 3, 36, 36)
    y = model(x)
    print(f"\nInput  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (4, 1, 9, 9), f"Unexpected output shape: {y.shape}"
    print("Shape check passed.")