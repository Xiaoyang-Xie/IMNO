# RNO1d and RNO2d: GRU-style Recurrent Neural Operators, 1D and 2D.
# shift_equivariant=True (default): grid unused. False: grid concatenated
# before lifting. Forward: pred, h = model(inp, grid, h).

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

class SpectralConv1d(nn.Module):
    """
    1D spectral convolution (truncated-mode FFT filter).

    x: (B, Cin, X)
    out: (B, Cout, X)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)

        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(x_ft: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, Cin, K) x (Cin, Cout, K) -> (B, Cout, K)
        return torch.einsum("bik,iok->bok", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, X = x.shape
        # rFFT along last axis
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, Cin, K_full)
        K_full = x_ft.shape[-1]
        K = min(self.modes1, K_full)

        out_ft = torch.zeros(
            B, self.out_channels, K_full, device=x.device, dtype=torch.cfloat
        )
        out_ft[..., :K] = self.compl_mul1d(x_ft[..., :K], self.weights[..., :K])
        out = torch.fft.irfft(out_ft, n=X, dim=-1)
        return out


class FNOBlock1d(nn.Module):
    """GELU applied to the sum of pointwise and spectral convolutions."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.w = nn.Conv1d(width, width, 1)
        self.k = SpectralConv1d(width, width, modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, X)
        return F.gelu(self.w(x) + self.k(x))


# -------------------------
# RNO Cell (GRU-style on function space)
# -------------------------

class RNOCell1d(nn.Module):
    """
    GRU-like recurrent cell on 1D function space:
      z = sigmoid( f1(x) + f2(h) + b1 )
      r = sigmoid( f3(x) + f4(h) + b2 )
      h~ = selu   ( f5(x) + f6(r*h) + b3 )
      h' = (1-z)*h + z*h~
    """

    def __init__(self, modes: int, width: int):
        super().__init__()

        # for each gate: input block + hidden block
        self.fx_z = FNOBlock1d(width, modes)
        self.fh_z = FNOBlock1d(width, modes)

        self.fx_r = FNOBlock1d(width, modes)
        self.fh_r = FNOBlock1d(width, modes)

        self.fx_h = FNOBlock1d(width, modes)
        self.fh_h = FNOBlock1d(width, modes)

        # scalar biases broadcast to (B,C,X)
        self.b_z = nn.Parameter(torch.randn(1, width, 1) * 0.01)
        self.b_r = nn.Parameter(torch.randn(1, width, 1) * 0.01)
        self.b_h = nn.Parameter(torch.randn(1, width, 1) * 0.01)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x,h: (B, C, X)
        z = torch.sigmoid(self.fx_z(x) + self.fh_z(h) + self.b_z)
        r = torch.sigmoid(self.fx_r(x) + self.fh_r(h) + self.b_r)
        h_tilde = F.selu(self.fx_h(x) + self.fh_h(r * h) + self.b_h)
        h_next = (1.0 - z) * h + z * h_tilde
        return h_next


# -------------------------
# RNO model (train.py compatible)
# -------------------------

class RNO1d(nn.Module):
    """
    1D RNO, called once per autoregressive step from train.py's training loop.

    Call:
      pred, h = model(inp, grid, h)

    Shapes:
      inp  : (B, X, initial_step*num_channels)
      grid : (B, X, 1)
      h    : (B, width, X) or None
      pred : (B, X, 1, num_channels)
    """

    def __init__(
        self,
        num_channels: int,
        initial_step: int,
        width: int = 64,
        modes: int = 24,
        lifting_hidden: int = 0,
        projection_hidden: int = 128,
        shift_equivariant: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.initial_step = int(initial_step)
        self.width = int(width)
        self.modes = int(modes)
        self.shift_equivariant = shift_equivariant

        # Input features = initial_step*num_channels (+1 for grid if not shift_equivariant)
        in_features = self.initial_step * self.num_channels + (0 if shift_equivariant else 1)

        # Lifting: pointwise MLP (Linear applied over last dim)
        # (B,X,F) -> (B,X,width)
        if lifting_hidden and lifting_hidden > 0:
            self.lift = nn.Sequential(
                nn.Linear(in_features, lifting_hidden),
                nn.GELU(),
                nn.Linear(lifting_hidden, self.width),
            )
        else:
            self.lift = nn.Linear(in_features, self.width)

        # Recurrent cell on lifted function (B,width,X)
        self.cell = RNOCell1d(modes=self.modes, width=self.width)

        # Projection head: (B,X,width) -> (B,X,num_channels)
        self.proj = nn.Sequential(
            nn.Linear(self.width, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, self.num_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ):
        """
        x:    (B, X, initial_step*num_channels)
        grid: (B, X, 1)
        h:    (B, width, X) or None
        """
        assert x.ndim == 3, f"expected x (B,X,F), got {tuple(x.shape)}"
        assert grid.ndim == 3 and grid.shape[-1] == 1, f"expected grid (B,X,1), got {tuple(grid.shape)}"
        B, Xn, Fdim = x.shape
        assert grid.shape[0] == B and grid.shape[1] == Xn

        # concat coordinate (only when not shift-equivariant)
        xin = x if self.shift_equivariant else torch.cat([x, grid], dim=-1)  # (B, X, F[+1])
        u = self.lift(xin)                     # (B, X, width)
        u = u.permute(0, 2, 1).contiguous()    # (B, width, X)

        if h is None:
            h = torch.zeros(B, self.width, Xn, device=u.device, dtype=u.dtype)

        h = self.cell(u, h)                    # (B, width, X)

        # predict next step from current hidden state
        y = h.permute(0, 2, 1).contiguous()    # (B, X, width)
        y = self.proj(y)                       # (B, X, num_channels)
        y = y.unsqueeze(-2)                    # (B, X, 1, num_channels)

        return y, h


# -------------------------
# 2D
# -------------------------

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution via rfft2.
    x: (B, Cin, H, W) -> (B, Cout, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def compl_mul2d(self, inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        K = x_ft.size(-1)
        out_ft = torch.zeros(B, self.out_channels, H, K, device=x.device, dtype=torch.cfloat)
        mx = min(self.modes_x, H)
        my = min(self.modes_y, K)
        if mx > 0 and my > 0:
            out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
            out_ft[:, :, -mx:, :my] = self.compl_mul2d(x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    """Single 2D FNO layer: GELU( W(x) + K(x) ), x: (B, C, H, W)."""

    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.w = nn.Conv2d(width, width, 1)
        self.k = SpectralConv2d(width, width, modes_x, modes_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.w(x) + self.k(x))


class RNOCell2d(nn.Module):
    """
    GRU-style recurrent cell on 2D function space.
    x, h: (B, C, H, W)
    """

    def __init__(self, modes_x: int, modes_y: int, width: int):
        super().__init__()
        self.fx_z = FNOBlock2d(width, modes_x, modes_y)
        self.fh_z = FNOBlock2d(width, modes_x, modes_y)

        self.fx_r = FNOBlock2d(width, modes_x, modes_y)
        self.fh_r = FNOBlock2d(width, modes_x, modes_y)

        self.fx_h = FNOBlock2d(width, modes_x, modes_y)
        self.fh_h = FNOBlock2d(width, modes_x, modes_y)

        self.b_z = nn.Parameter(torch.randn(1, width, 1, 1) * 0.01)
        self.b_r = nn.Parameter(torch.randn(1, width, 1, 1) * 0.01)
        self.b_h = nn.Parameter(torch.randn(1, width, 1, 1) * 0.01)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.fx_z(x) + self.fh_z(h) + self.b_z)
        r = torch.sigmoid(self.fx_r(x) + self.fh_r(h) + self.b_r)
        h_tilde = F.selu(self.fx_h(x) + self.fh_h(r * h) + self.b_h)
        return (1.0 - z) * h + z * h_tilde


class RNO2d(nn.Module):
    """
    2D Recurrent Neural Operator (GRU-style on function space).

    Call:
      pred, h = model(x, grid, h)

    Shapes:
      x    : (B, H, W, initial_step * num_channels)
      grid : (B, H, W, 2)
      h    : (B, width, H, W) or None
      pred : (B, H, W, 1, num_channels)
    """

    def __init__(
        self,
        num_channels: int,
        initial_step: int,
        width: int = 64,
        modes_x: int = 12,
        modes_y: int = 12,
        projection_hidden: int = 128,
        shift_equivariant: bool = True,
        phase_mode: str = "xy",
    ):
        super().__init__()
        if phase_mode not in ("xy", "x"):
            raise ValueError(f"phase_mode must be 'xy' or 'x', got {phase_mode!r}")
        self.num_channels = int(num_channels)
        self.initial_step = int(initial_step)
        self.width = int(width)
        self.shift_equivariant = shift_equivariant
        self.phase_mode = phase_mode

        # shift_equivariant=True (default): grid unused, or partial per
        # phase_mode. phase_mode="xy" (default): equivariant to translations
        # in both x and y. phase_mode="x": the y-coordinate only (1 extra
        # feature) is concatenated, so the model is equivariant to
        # translations in x only. shift_equivariant=False: full (x, y) grid
        # (2 extra features) concatenated before lifting; phase_mode ignored.
        if not shift_equivariant:
            extra = 2
        elif phase_mode == "x":
            extra = 1
        else:
            extra = 0
        in_features = self.initial_step * self.num_channels + extra
        self.lift = nn.Linear(in_features, self.width)
        self.cell = RNOCell2d(modes_x, modes_y, self.width)
        self.proj = nn.Sequential(
            nn.Linear(self.width, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, self.num_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ):
        """
        x    : (B, H, W, initial_step * num_channels)
        grid : (B, H, W, 2)
        h    : (B, width, H, W) or None
        """
        assert x.ndim == 4, f"expected (B,H,W,F), got {tuple(x.shape)}"
        B, H, W, _ = x.shape

        if not self.shift_equivariant:
            xin = torch.cat([x, grid], dim=-1)  # (B, H, W, F+2)
        elif self.phase_mode == "x":
            xin = torch.cat([x, grid[..., 1:2]], dim=-1)  # (B, H, W, F+1), y-coord only
        else:
            xin = x
        u = self.lift(xin)                       # (B, H, W, width)
        u = u.permute(0, 3, 1, 2).contiguous()   # (B, width, H, W)

        if h is None:
            h = torch.zeros(B, self.width, H, W, device=u.device, dtype=u.dtype)

        h = self.cell(u, h)                      # (B, width, H, W)

        y = h.permute(0, 2, 3, 1).contiguous()   # (B, H, W, width)
        y = self.proj(y)                         # (B, H, W, num_channels)
        y = y.unsqueeze(-2)                      # (B, H, W, 1, num_channels)

        return y, h
