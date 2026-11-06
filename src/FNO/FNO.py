"""
This code is modified based on Zongyi Li's original implementation of Fourier Neural Operators.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, num_channels, modes=16, width=64, initial_step=10, shift_equivariant=True):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)

        shift_equivariant: if True (default), the coordinate grid is not
        concatenated to the input, so the model is translation-equivariant.
        If False, the grid (1 extra feature) is concatenated before lifting,
        matching the original FNO paper's setup.
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.shift_equivariant = shift_equivariant
        in_features = initial_step * num_channels + (0 if shift_equivariant else 1)
        self.fc0 = nn.Linear(
            in_features, self.width
        )  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, t*v]
        if not self.shift_equivariant:
            x = torch.cat((x, grid), dim=-1)  # noqa: PLW2901
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # pad the domain if input is non-periodic
        # x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            # Number of Fourier modes to multiply, at most floor(N/2) + 1
            modes1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(
        self, num_channels, modes1=12, modes2=12, width=20, initial_step=10, shift_equivariant=True,
        phase_mode="xy",
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)

        shift_equivariant: if True (default), no (or partial, see phase_mode)
        positional encoding is used. If False, the full (x, y) grid (2 extra
        features) is concatenated before lifting, and phase_mode is ignored.

        phase_mode (only used when shift_equivariant=True): "xy" (default)
        omits the grid entirely, so the model is equivariant to translations
        in both x and y. "x" concatenates the y-coordinate only (1 extra
        feature), so the model is equivariant to translations in x only —
        matching IMNO2dx for a fair comparison on non-x-periodic data.
        """
        if phase_mode not in ("xy", "x"):
            raise ValueError(f"phase_mode must be 'xy' or 'x', got {phase_mode!r}")

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.shift_equivariant = shift_equivariant
        self.phase_mode = phase_mode
        if not shift_equivariant:
            extra = 2
        elif phase_mode == "x":
            extra = 1
        else:
            extra = 0
        in_features = initial_step * num_channels + extra
        self.fc0 = nn.Linear(in_features, self.width)

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2
        )
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        if not self.shift_equivariant:
            x = torch.cat((x, grid), dim=-1)  # noqa: PLW2901
        elif self.phase_mode == "x":
            x = torch.cat((x, grid[..., 1:2]), dim=-1)  # noqa: PLW2901  (y-coord only)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., : -self.padding, : -self.padding]  # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)
