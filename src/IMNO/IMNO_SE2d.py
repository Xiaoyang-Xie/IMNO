from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.IMNO.common import Encode2d, SpectralConv2d, FourierDecoder2d, ODEFunc


def resolve_phase_ref_channels(num_channels: int, ref_channel: int) -> tuple[int, int]:
    ch_fallback = max(0, min(ref_channel, num_channels - 1))
    ref_ch_x = 2 if num_channels > 2 else ch_fallback
    ref_ch_y = 3 if num_channels > 3 else ch_fallback
    return ref_ch_x, ref_ch_y


def phase_align2d(u: torch.Tensor, ref_channel: int = 0, eps: float = 1e-12):
    """
    u: (B, H, W, C)
    returns:
      uhat: (B, H, W, C)  -- remove phases of (kx,ky)=(1,0) and (0,1)
      phi:  (B, 2)        -- (phi_x, phi_y)
    """
    assert u.dim() == 4, "u must be (B, H, W, C)"
    B, H, W, C = u.shape

    ref_ch_x, ref_ch_y = resolve_phase_ref_channels(C, ref_channel)

    A_ref_x = torch.fft.rfft2(u[..., ref_ch_x], dim=(-2, -1))
    A_ref_y = torch.fft.rfft2(u[..., ref_ch_y], dim=(-2, -1))
    Hf, K = A_ref_x.shape[-2], A_ref_x.shape[-1]

    if H >= 2 and K >= 1:
        a10 = A_ref_x[..., 1, 0]
    else:
        a10 = torch.zeros(B, dtype=torch.cfloat, device=u.device)
    if H >= 1 and K >= 2:
        a01 = A_ref_y[..., 0, 1]
    else:
        a01 = torch.zeros(B, dtype=torch.cfloat, device=u.device)

    phi_x = torch.atan2(a10.imag, a10.real + eps)
    phi_y = torch.atan2(a01.imag, a01.real + eps)

    x = u.permute(0, 3, 1, 2).contiguous()
    A = torch.fft.rfft2(x, dim=(-2, -1))

    # In rfft2, the last axis is half-spectrum (ky>=0), but the -2 axis keeps the full FFT order.
    # Therefore kx must be signed wave numbers in FFT index order.
    kx = (torch.fft.fftfreq(Hf, d=1.0, device=u.device) * Hf).to(dtype=A.real.dtype)
    kx = kx.view(1, 1, Hf, 1)
    ky = torch.arange(K, device=u.device, dtype=A.real.dtype).view(1, 1, 1, K)
    phase = torch.exp(-1j * (kx * phi_x.view(B, 1, 1, 1) + ky * phi_y.view(B, 1, 1, 1)))
    A = A * phase
    uhat = torch.fft.irfft2(A, s=(H, W), dim=(-2, -1))
    uhat = uhat.permute(0, 2, 3, 1).contiguous()
    phi = torch.stack([phi_x, phi_y], dim=-1)
    return uhat, phi


def add_phase2d(uhat: torch.Tensor, phi: torch.Tensor):
    """
    uhat: (B, H, W, C)
    phi:  (B, 2)  -- (phi_x, phi_y)
    """
    assert uhat.dim() == 4, "uhat must be (B, H, W, C)"
    assert phi.dim() == 2 and phi.size(-1) == 2, "phi must be (B, 2)"
    B, H, W, C = uhat.shape
    assert phi.size(0) == B

    x = uhat.permute(0, 3, 1, 2).contiguous()
    A = torch.fft.rfft2(x, dim=(-2, -1))
    Hf, K = A.shape[-2], A.shape[-1]
    phi_x, phi_y = phi[:, 0], phi[:, 1]

    kx = (torch.fft.fftfreq(Hf, d=1.0, device=uhat.device) * Hf).to(dtype=A.real.dtype)
    kx = kx.view(1, 1, Hf, 1)
    ky = torch.arange(K, device=uhat.device, dtype=A.real.dtype).view(1, 1, 1, K)
    phase = torch.exp(1j * (kx * phi_x.view(B, 1, 1, 1) + ky * phi_y.view(B, 1, 1, 1)))
    A = A * phase
    u = torch.fft.irfft2(A, s=(H, W), dim=(-2, -1))
    u = u.permute(0, 2, 3, 1).contiguous()
    return u

class ResidualStepNO2d(nn.Module):
    """Single-step residual dynamics on 2D fields."""

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
        v: (B,H,W,C) or (B,H,W,1,C)
        grid: (B,H,W,grid_dim)
        h: (B,H_lat)
        Phi: (B, 2)
        return: (B,H,W,1,C)
        """
        if v.dim() == 5:
            v = v.squeeze(-2)
        assert v.dim() == 4, "v must be (B,H,W,C) or (B,H,W,1,C)"
        assert h.dim() == 2, "h must be (B,H)"
        assert grid.dim() == 4, "grid must be (B,H,W,grid_dim)"
        assert Phi.dim() == 2 and Phi.size(-1) == 2, "Phi must be (B,2)"

        B, H, W, C = v.shape
        device = v.device

        ii = torch.arange(H, device=device, dtype=v.dtype).view(1, H, 1)
        jj = torch.arange(W, device=device, dtype=v.dtype).view(1, 1, W)
        # phase_x: (B,H,1) -> expand to (B,H,W)
        phase_x = torch.cos(Phi[:, 0].view(B, 1, 1) + 2 * torch.pi * ii / H).expand(B, H, W)
        # phase_y: (B,1,W) -> expand to (B,H,W)
        phase_y = torch.cos(Phi[:, 1].view(B, 1, 1) + 2 * torch.pi * jj / W).expand(B, H, W)
        # (B,H,W,2) channel-wise concat
        phase_feat = torch.stack([phase_x, phase_y], dim=-1)

        x = torch.cat([v, phase_feat], dim=-1)
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


class IMNO2d(nn.Module):
    """
    State: (h_t, v_t, Phi_t); default single-frame window initial_step=1,
    physical channels are usually 1 or 4 (num_channels). modes/modes2
    match FNO2d (full x axis / half y axis).
    """

    def __init__(
        self,
        num_channels: int = 4,
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
        ref_channel: int = -1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.initial_step = initial_step
        self.effective_channels = num_channels * initial_step
        self.hidden_states = hidden_states
        self.grid_dim = grid_dim

        self.dt = float(dt)
        self.ode_method = ode_method
        self.ode_substeps = int(ode_substeps)
        self.ref_channel = ref_channel

        self.h_encoder = Encode2d(width, modes3, hidden_states)
        # gauge-fix the decoder for whichever channel phase_align2d actually
        # used as the x/y phase reference (see resolve_phase_ref_channels) --
        # that channel's own k=1 mode is zero by construction once its phase
        # has been subtracted out, so pinning it there is a real constraint,
        # not an arbitrary one.
        ref_ch_x, ref_ch_y = resolve_phase_ref_channels(num_channels, ref_channel)
        self.h_decoders = nn.ModuleList(
            FourierDecoder2d(
                hidden_states,
                initial_step,
                modes4,
                mlp_width=ode_hidden,
                gauge_fix_x=(c == ref_ch_x),
                gauge_fix_y=(c == ref_ch_y),
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
            nn.Linear(dphi_mlp_width, 2),
        )

    def _decode_h_to_uhat(self, h: torch.Tensor, n: tuple[int, int]) -> torch.Tensor:
        """(B, T*C, H, W), flattened in t*C+c order."""
        T = self.initial_step
        C = self.num_channels
        decoded = [dec(h, n) for dec in self.h_decoders]  # each: (B,T,H,W)

        if T == 1:
            return torch.cat(decoded, dim=1)  # (B,C,H,W)

        chunks = [torch.stack([decoded[c][:, t] for c in range(C)], dim=1) for t in range(T)]
        return torch.cat(chunks, dim=1)  # (B,T*C,H,W)

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

    def _predict_dphi_from_uhat_m(self, h_next: torch.Tensor) -> torch.Tensor:
        assert h_next.dim() == 2, "h_next must be (B, H)"
        B, Hdim = h_next.shape
        assert Hdim == self.hidden_states
        return self.dphi_mlp(h_next)

    def _init_from_window(self, x: torch.Tensor, grid: torch.Tensor):
        """
        x: (B, H, W, T*C)
        returns:
            h_t:   (B,H_lat)
            v_t:   (B,H,W,1,effective_C)
            Phi_t: (B,2)
        """
        B, H, W, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"

        u_t = x

        uhat_t, Phi_t = phase_align2d(u_t, ref_channel=self.ref_channel)

        xh = self.fc0_h(uhat_t)
        xh = xh.permute(0, 3, 1, 2)
        h_t = self.h_encoder(xh)

        uhat_m_t = self._decode_h_to_uhat(h_t, (H, W)).permute(0, 2, 3, 1)

        u_m_t = add_phase2d(uhat_m_t, Phi_t)
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
        h:   (B,H_lat)
        v:   (B,H,W,effective_C) or (B,H,W,1,effective_C)
        Phi: (B,2)
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
        dPhi = self._predict_dphi_from_uhat_m(h_next)
        Phi_next = Phi_t + dPhi
        u_next = v_next + add_phase2d(uhat_m, Phi_next).unsqueeze(-2)

        return u_next, h_next, v_next, Phi_next
