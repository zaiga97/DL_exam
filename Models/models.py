import torch
from torch import nn
from torch.nn import functional as F


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class LREquilizer(nn.Module):
    def __init__(self, in_dim: int, kernel_size: int, alpha: float = 1.):
        super().__init__()
        self.scale = (alpha / (in_dim * (kernel_size ** 2))) ** 0.5

    def forward(self, x: torch.Tensor):
        return x * self.scale


class EQConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.convBlock = nn.Sequential(
            LREquilizer(in_dim=in_dim, kernel_size=3),
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convBlock(x)


class DoubleEQConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, padding: int = 1, kernel_size2: int = 3,
                 padding2: int = 1):
        super().__init__()
        self.convBlock = nn.Sequential(
            EQConvBlock(in_dim, out_dim, kernel_size, padding),
            EQConvBlock(out_dim, out_dim, kernel_size2, padding2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convBlock(x)


class Generator(nn.Module):
    def __init__(self, input_size: int = 64, hidden_channels: int = 128):
        super().__init__()
        self.max_depth = 5
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.input_layer = nn.ConvTranspose2d(input_size, hidden_channels, 8, 1, 0)

        self.blocks = nn.ModuleList([
            DoubleEQConvBlock(hidden_channels, hidden_channels),  # hidden_channels * 8 * 8
            DoubleEQConvBlock(hidden_channels, hidden_channels),  # hidden_channels * 16 * 16
            DoubleEQConvBlock(hidden_channels, hidden_channels // 2),  # hidden_channels//2 * 32 * 32
            DoubleEQConvBlock(hidden_channels // 2, hidden_channels // 4),  # hidden_channels//4 * 64 * 64
            DoubleEQConvBlock(hidden_channels // 4, hidden_channels // 8),  # hidden_channels//8 * 128 * 128
        ])
        self.out_blocks = nn.ModuleList([
            EQConvBlock(hidden_channels, 3, kernel_size=1, padding=0),
            EQConvBlock(hidden_channels, 3, kernel_size=1, padding=0),
            EQConvBlock(hidden_channels // 2, 3, kernel_size=1, padding=0),
            EQConvBlock(hidden_channels // 4, 3, kernel_size=1, padding=0),
            EQConvBlock(hidden_channels // 8, 3, kernel_size=1, padding=0)
        ])

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, level: int, alpha: float = 1) -> torch.Tensor:
        x = self.input_layer(z)
        x = self.blocks[0](x)
        if level == 0:
            return torch.tanh(self.out_blocks[0](x))

        xo = torch.Tensor()
        for i in range(1, level + 1):
            xo = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.blocks[i](xo)

        if alpha < 1:
            return torch.tanh(((1 - alpha) * self.out_blocks[level - 1](xo) + alpha * self.out_blocks[level](x)))
        return torch.tanh(self.out_blocks[level](x))


class Discriminator(nn.Module):
    def __init__(self, hidden_channels: int = 128):
        super().__init__()
        self.max_depth = 5
        self.hidden_channels = hidden_channels
        self.output_layer = nn.Linear(hidden_channels, 1)
        self.blocks = nn.ModuleList([
            DoubleEQConvBlock(hidden_channels, hidden_channels, kernel_size2=8, padding2=0),  # hidden_channels * 8 * 8
            DoubleEQConvBlock(hidden_channels, hidden_channels),  # hidden_channels * 16 * 16
            DoubleEQConvBlock(hidden_channels // 2, hidden_channels),  # hidden_channels * 32 * 32
            DoubleEQConvBlock(hidden_channels // 4, hidden_channels // 2),  # hidden_channels//2 * 64 * 64
            DoubleEQConvBlock(hidden_channels // 8, hidden_channels // 4),  # hidden_channels//4 * 128 * 128
        ])
        self.in_blocks = nn.ModuleList([
            EQConvBlock(3, hidden_channels, kernel_size=1, padding=0),
            EQConvBlock(3, hidden_channels, kernel_size=1, padding=0),
            EQConvBlock(3, hidden_channels // 2, kernel_size=1, padding=0),
            EQConvBlock(3, hidden_channels // 4, kernel_size=1, padding=0),
            EQConvBlock(3, hidden_channels // 8, kernel_size=1, padding=0)
        ])

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, level: int, alpha: float = 1) -> torch.Tensor:

        # Read the image
        if level > 0 and alpha < 1:
            xd = self.in_blocks[level - 1](F.avg_pool2d(x, kernel_size=2, stride=2))
            x = self.blocks[level](self.in_blocks[level](x))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = ((1 - alpha) * xd) + (alpha * x)
            level -= 1
        else:
            x = self.in_blocks[level](x)

        # Pass through the nn
        for i in range(level, -1, -1):
            x = self.blocks[i](x)
            if i > 0:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                pass

        return self.output_layer(x.squeeze(2).squeeze(2))
