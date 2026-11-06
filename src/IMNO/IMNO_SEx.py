from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.IMNO.common import Encode2d, SpectralConv2d, FourierDecoder2d, ODEFunc


def phase_align_x(u: torch.Tensor, ref_channel: int = 0, eps: float = 1e-12):
    """
    u: (B, H, W, C)
    Phase alignment only in x direction (k=(1,0) mode of ref_channel).
    returns:
      uhat: (B, H, W, C)
      phi:  (B,)  -- phi_x only
    """
    assert u.dim() == 4, "u must be (B, H, W, C)"
    B, H, W, C = u.shape

    u_ref = u[..., ref_channel]
    A_ref = torch.fft.rfft2(u_ref, dim=(-2, -1))
    Hf = A_ref.shape[-2]

    if H >= 2:
        a10 = A_ref[..., 1, 0]
    else:
        a10 = torch.zeros(B, dtype=torch.cfloat, device=u.device)

    phi_x = torch.atan2(a10.imag, a10.real + eps)

    x = u.permute(0, 3, 1, 2).contiguous()
    A = torch.fft.rfft2(x, dim=(-2, -1))

    kx = (torch.fft.fftfreq(Hf, d=1.0, device=u.device) * Hf).to(dtype=A.real.dtype)
    kx = kx.view(1, 1, Hf, 1)
    phase = torch.exp(-1j * kx * phi_x.view(B, 1, 1, 1))
    A = A * phase
    uhat = torch.fft.irfft2(A, s=(H, W), dim=(-2, -1))
    uhat = uhat.permute(0, 2, 3, 1).contiguous()
    return uhat, phi_x


def add_phase_x(uhat: torch.Tensor, phi_x: torch.Tensor):
    """
    uhat: (B, H, W, C)
    phi_x: (B,)
    """
    assert uhat.dim() == 4, "uhat must be (B, H, W, C)"
    assert phi_x.dim() == 1, "phi_x must be (B,)"
    B, H, W, C = uhat.shape
    assert phi_x.size(0) == B

    x = uhat.permute(0, 3, 1, 2).contiguous()
    A = torch.fft.rfft2(x, dim=(-2, -1))
    Hf = A.shape[-2]

    kx = (torch.fft.fftfreq(Hf, d=1.0, device=uhat.device) * Hf).to(dtype=A.real.dtype)
    kx = kx.view(1, 1, Hf, 1)
    phase = torch.exp(1j * kx * phi_x.view(B, 1, 1, 1))
    A = A * phase
    u = torch.fft.irfft2(A, s=(H, W), dim=(-2, -1))
    u = u.permute(0, 2, 3, 1).contiguous()
    return u


class ResidualStepNO2d(nn.Module):
    """Single-step residual dynamics; phase feature in x direction only."""

    def __init__(
        self,
        num_channels: int,
        width: int = 64,
        modes: int = 16,
        modes2: int = 12,
        hidden_states: int = 64,
        grid_dim: int = 2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.width = width
        self.hidden_states = hidden_states
        self.grid_dim = grid_dim

        # num_channels + 2: phase_x (aligned) + pos_y (unaligned positional embedding)
        self.fc0 = nn.Linear(num_channels + 2, width)

        self.film0 = nn.Linear(hidden_states, 2 * width)
        self.film1 = nn.Linear(hidden_states, 2 * width)
        self.film2 = nn.Linear(hidden_states, 2 * width)
        self.film3 = nn.Linear(hidden_states, 2 * width)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.conv0 = SpectralConv2d(width, width, modes, modes2)
        self.conv1 = SpectralConv2d(width, width, modes, modes2)
        self.conv2 = SpectralConv2d(width, width, modes, modes2)
        self.conv3 = SpectralConv2d(width, width, modes, modes2)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, v: torch.Tensor, grid: torch.Tensor, h: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """
        v:    (B,H,W,C) or (B,H,W,1,C)
        grid: (B,H,W,grid_dim)
        h:    (B,H_lat)
        Phi:  (B,)  -- phi_x only
        return: (B,H,W,1,C)
        """
        if v.dim() == 5:
            v = v.squeeze(-2)
        assert v.dim() == 4, "v must be (B,H,W,C) or (B,H,W,1,C)"
        assert h.dim() == 2, "h must be (B,H)"
        assert grid.dim() == 4, "grid must be (B,H,W,grid_dim)"
        assert Phi.dim() == 1, "Phi must be (B,)"

        B, H, W, C = v.shape
        device = v.device

        ii = torch.arange(H, device=device, dtype=v.dtype).view(1, H, 1)
        jj = torch.arange(W, device=device, dtype=v.dtype).view(1, 1, W)
        phase_x = torch.cos(Phi.view(B, 1, 1) + 2 * torch.pi * ii / H).expand(B, H, W)
        # y direction: no phase alignment, use pure positional embedding (phi_y = 0)
        pos_y = torch.cos(2 * torch.pi * jj / W).expand(B, H, W)
        phase_feat = torch.stack([phase_x, pos_y], dim=-1)  # (B, H, W, 2)

        x = torch.cat([v, phase_feat], dim=-1)  # (B, H, W, C+2)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        def film(layer, feat):
            gam, bet = layer(h).chunk(2, dim=-1)
            gam = gam.unsqueeze(-1).unsqueeze(-1)
            bet = bet.unsqueeze(-1).unsqueeze(-1)
            gamma = 1.0 + torch.tanh(gam)
            return gamma * feat + bet

        x = F.gelu(film(self.film0, self.w0(x) + self.conv0(x)))
        x = F.gelu(film(self.film1, self.w1(x) + self.conv1(x)))
        x = F.gelu(film(self.film2, self.w2(x) + self.conv2(x)))
        x = film(self.film3, self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


class IMNO2dx(nn.Module):
    """
    2D IMNO with phase alignment in x direction only.
    Designed for single-channel (num_channels=1) 2D PDEs.

    State: (h_t, v_t, Phi_t)
      - Phi_t: (B,) accumulated phi_x in original frame
      - v_t: residual in canonical frame
    """

    def __init__(
        self,
        num_channels: int = 1,
        width: int = 64,
        modes: int = 24,
        modes2: int = 12,
        modes3: int = 8,
        modes4: int = 12,
        initial_step: int = 1,
        hidden_states: int = 64,
        grid_dim: int = 2,
        dt: float = 1.0,
        ode_method: Literal["euler", "rk4"] = "euler",
        ode_substeps: int = 1,
        ode_hidden: int = 128,
        ode_depth: int = 2,
        ref_channel: int = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.initial_step = initial_step
        self.effective_channels = num_channels * initial_step
        self.hidden_states = hidden_states
        self.grid_dim = grid_dim
        self.ref_channel = ref_channel

        self.dt = float(dt)
        self.ode_method = ode_method
        self.ode_substeps = int(ode_substeps)

        self.h_encoder = Encode2d(width, modes3, hidden_states)
        self.h_decoders = nn.ModuleList(
            FourierDecoder2d(
                hidden_states,
                initial_step,
                modes4,
                mlp_width=ode_hidden,
                gauge_fix_x=(c == ref_channel),
            )
            for c in range(num_channels)
        )

        self.fc0_h = nn.Linear(self.effective_channels, width)

        self.h_ode = ODEFunc(hidden_states, hidden=ode_hidden, depth=ode_depth)

        self.v_model = ResidualStepNO2d(
            num_channels=self.effective_channels,
            width=width,
            modes=modes,
            modes2=modes2,
            hidden_states=hidden_states,
            grid_dim=grid_dim,
        )

        dphi_mlp_width = 200
        self.dphi_mlp = nn.Sequential(
            nn.Linear(hidden_states, dphi_mlp_width),
            nn.GELU(),
            nn.Linear(dphi_mlp_width, dphi_mlp_width),
            nn.GELU(),
            nn.Linear(dphi_mlp_width, 1),
        )

    def _decode_h_to_uhat(self, h: torch.Tensor, n: tuple[int, int]) -> torch.Tensor:
        """Returns (B, T*C, H, W)."""
        T = self.initial_step
        C = self.num_channels
        decoded = [dec(h, n) for dec in self.h_decoders]  # each: (B, T, H, W)

        if T == 1:
            return torch.cat(decoded, dim=1)  # (B, C, H, W)

        chunks = [torch.stack([decoded[c][:, t] for c in range(C)], dim=1) for t in range(T)]
        return torch.cat(chunks, dim=1)  # (B, T*C, H, W)

    def _ode_step(self, h: torch.Tensor) -> torch.Tensor:
        dt = self.dt / max(self.ode_substeps, 1)

        def f(x):
            return self.h_ode(x)

        x = h
        for _ in range(max(self.ode_substeps, 1)):
            if self.ode_method == "euler":
                x = x + dt * f(x)
            elif self.ode_method == "rk4":
                k1 = f(x)
                k2 = f(x + 0.5 * dt * k1)
                k3 = f(x + 0.5 * dt * k2)
                k4 = f(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown ode_method: {self.ode_method}")
        return x

    def _predict_dphi(self, h_next: torch.Tensor) -> torch.Tensor:
        """Returns (B,) scalar dPhi_x."""
        return self.dphi_mlp(h_next).squeeze(-1)

    def _init_from_window(self, x: torch.Tensor, grid: torch.Tensor):
        """
        x: (B, H, W, T*C)
        returns:
            h_t:   (B, H_lat)
            v_t:   (B, H, W, 1, effective_C)
            Phi_t: (B,)
        """
        B, H, W, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"

        u_t = x
        uhat_t, Phi_t = phase_align_x(u_t, ref_channel=self.ref_channel)

        xh = self.fc0_h(uhat_t)
        xh = xh.permute(0, 3, 1, 2)
        h_t = self.h_encoder(xh)

        uhat_m_t = self._decode_h_to_uhat(h_t, (H, W)).permute(0, 2, 3, 1)
        u_m_t = add_phase_x(uhat_m_t, Phi_t)
        v_t = (u_t - u_m_t).unsqueeze(-2)

        return h_t, v_t, Phi_t

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        Phi: Optional[torch.Tensor] = None,
    ):
        """
        x:   (B, H, W, T*C)
        grid:(B, H, W, grid_dim)
        h:   (B, H_lat)
        v:   (B, H, W, effective_C) or (B, H, W, 1, effective_C)
        Phi: (B,)  -- phi_x only
        Returns:
            u_next:   (B, H, W, 1, effective_C)
            h_next:   (B, H_lat)
            v_next:   (B, H, W, 1, effective_C)
            Phi_next: (B,)
        """
        assert x.dim() == 4, "x must be (B, H, W, T*C)"
        B, H, W, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"
        assert grid.dim() == 4, "grid must be (B, H, W, grid_dim)"

        if (h is None) or (v is None) or (Phi is None):
            h_t, v_t, Phi_t = self._init_from_window(x, grid)
        else:
            h_t, v_t, Phi_t = h, v, Phi

        h_next = self._ode_step(h_t)
        v_next = self.v_model(v_t, grid, h_next, Phi_t)

        uhat_m = self._decode_h_to_uhat(h_next, (H, W)).permute(0, 2, 3, 1)
        dPhi = self._predict_dphi(h_next)
        Phi_next = Phi_t + dPhi
        u_next = v_next + add_phase_x(uhat_m, Phi_next).unsqueeze(-2)

        return u_next, h_next, v_next, Phi_next
