from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ODEFunc(nn.Module):
    """dh/dt = f(h)"""
    def __init__(self, dim: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.GELU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -------------------------- 1D building blocks --------------------------

class Encode1d(nn.Module):
    """Encode (B, C, L) -> (B, hidden_states) using low-frequency rFFT features."""
    def __init__(self, in_channels: int, modes1: int, hidden_states: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_states = hidden_states
        self.modes1 = modes1

        # (C * 2*modes1) -> hidden_states
        self.encoder = nn.Linear(in_channels * (2 * modes1), hidden_states, bias=False)
        self.ln = nn.LayerNorm(hidden_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "x must be (B, C, L)"
        B, C, L = x.shape
        assert C == self.in_channels, f"in_channels mismatch: {C} vs {self.in_channels}"

        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, K_full)
        K_full = x_ft.size(-1)

        k = min(self.modes1, K_full)
        ft_low = x_ft[..., :k]  # (B, C, k)

        if k < self.modes1:
            ft_low = F.pad(ft_low, (0, self.modes1 - k))

        feats = torch.cat([ft_low.real, ft_low.imag], dim=-1)  # (B, C, 2*modes1)
        feats = feats.reshape(B, -1)                           # (B, C*2*modes1)

        z = self.encoder(feats)  # (B, H)
        z = self.ln(z)
        return z


class SpectralConv1d(nn.Module):
    """1D Fourier layer: rFFT -> linear on the first ``modes1`` modes -> irFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (B, in, K) x (in, out, K) -> (B, out, K)
        return torch.einsum("bik,iok->bok", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, in, K_full)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)


class FourierDecoder1d(nn.Module):
    """
    Decode h -> low-frequency Fourier coefficients -> irfft(n=L).
    Resolution-free: output length controlled by n.
    If apply_gauge_fix, out_channel 0's k=1 mode is constrained to be real
    and non-negative (removes the phase gauge freedom on that mode).
    """
    def __init__(
        self,
        hidden_states: int,
        out_channels: int,
        modes1: int,
        mlp_width: int = 128,
        apply_gauge_fix: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.apply_gauge_fix = apply_gauge_fix
        self.net = nn.Sequential(
            nn.Linear(hidden_states, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, out_channels * modes1 * 2),  # real+imag
        )

    def forward(self, h: torch.Tensor, n: int) -> torch.Tensor:
        assert h.dim() == 2, "h must be (B, H)"
        B = h.size(0)

        coeff_ri_raw = self.net(h).view(B, self.out_channels, self.modes1, 2)
        coeff_ri = coeff_ri_raw.clone()
        if self.apply_gauge_fix:
            # torch.abs / torch.zeros_like are differentiable, so this doesn't block backprop.
            real = coeff_ri_raw[:, 0, 1, 0]
            coeff_ri[:, 0, 1, 0] = torch.abs(real)
            coeff_ri[:, 0, 1, 1] = torch.zeros_like(real)
        coeff = torch.complex(coeff_ri[..., 0], coeff_ri[..., 1])  # (B, C, modes1)

        k_full = n // 2 + 1
        k = min(self.modes1, k_full)

        out_ft = torch.zeros(B, self.out_channels, k_full, device=h.device, dtype=torch.cfloat)
        out_ft[:, :, :k] = coeff[:, :, :k]

        u = torch.fft.irfft(out_ft, n=n, dim=-1)  # (B, C, n)
        return u


# -------------------------- 2D building blocks --------------------------

class Encode2d(nn.Module):
    """Encode (B, C, H, W) -> (B, hidden_states) using low-frequency rFFT2 features."""
    def __init__(self, in_channels: int, modes1: int, hidden_states: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_states = hidden_states
        self.modes1 = modes1
        # x axis needs 2*modes1 points (+/-), y axis needs modes1 points (rFFT half-spectrum).
        self.encoder = nn.Linear(in_channels * (4 * modes1 * modes1), hidden_states, bias=False)
        self.ln = nn.LayerNorm(hidden_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # positive x-frequencies [0:modes1] and negative x-frequencies [-modes1:], y kept at [0:modes1]
        kx, ky = self.modes1, self.modes1
        ft_low = torch.cat(
            [
                x_ft[..., :kx, :ky],
                x_ft[..., -kx:, :ky],
            ],
            dim=-2,
        )  # (B, C, 2*modes1, modes1)

        feats = torch.view_as_real(ft_low).reshape(B, -1)
        return self.ln(self.encoder(feats))


class SpectralConv2d(nn.Module):
    """
    2D spectral conv (rfft2, then a linear map on the low-frequency block, then irfft2).
    modes_x: modes kept along the first spatial-frequency axis (full FFT axis, +/-).
    modes_y: modes kept along the second spatial-frequency axis (rFFT half axis).
    """
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def compl_mul2d(self, inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        H, K = x.size(-2), x.size(-1) // 2 + 1
        out_ft = torch.zeros(B, self.out_channels, H, K, device=x.device, dtype=torch.cfloat)
        mx = min(self.modes_x, H)
        my = min(self.modes_y, K)
        if mx > 0 and my > 0:
            out_ft[:, :, :mx, :my] = self.compl_mul2d(
                x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my]
            )
            out_ft[:, :, -mx:, :my] = self.compl_mul2d(
                x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my]
            )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))


class FourierDecoder2d(nn.Module):
    """
    Decode h -> low-frequency 2D Fourier coefficients -> irfft2(s=n).
    gauge_fix_x/gauge_fix_y independently constrain out_channel 0's (kx=1,ky=0)
    and (kx=0,ky=1) modes to be real and non-negative.
    """
    def __init__(
        self,
        hidden_states: int,
        out_channels: int,
        modes1: int,
        mlp_width: int = 128,
        gauge_fix_x: bool = False,
        gauge_fix_y: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.gauge_fix_x = gauge_fix_x
        self.gauge_fix_y = gauge_fix_y
        self.net = nn.Sequential(
            nn.Linear(hidden_states, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, out_channels * (modes1 * modes1 * 4)),
        )

    def forward(self, h: torch.Tensor, n: tuple[int, int]) -> torch.Tensor:
        B = h.size(0)
        H, W = int(n[0]), int(n[1])

        coeff_ri = self.net(h).view(B, self.out_channels, 2 * self.modes1, self.modes1, 2)
        coeff_ri = coeff_ri.clone()

        if self.gauge_fix_x:
            real_10 = coeff_ri[:, 0, 1, 0, 0].clone()
            coeff_ri[:, 0, 1, 0, 0] = torch.abs(real_10)
            coeff_ri[:, 0, 1, 0, 1] = 0.0

        if self.gauge_fix_y:
            real_01 = coeff_ri[:, 0, 0, 1, 0].clone()
            coeff_ri[:, 0, 0, 1, 0] = torch.abs(real_01)
            coeff_ri[:, 0, 0, 1, 1] = 0.0

        coeff = torch.complex(coeff_ri[..., 0], coeff_ri[..., 1])

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=h.device, dtype=torch.cfloat)

        kx, ky = self.modes1, self.modes1
        out_ft[:, :, :kx, :ky] = coeff[:, :, :kx, :ky]
        out_ft[:, :, -kx:, :ky] = coeff[:, :, kx:, :ky]

        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))
