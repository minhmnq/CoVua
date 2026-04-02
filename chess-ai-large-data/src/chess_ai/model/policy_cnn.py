from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class ChessPolicyCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        trunk_channels: int = 128,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.policy_head(x)
