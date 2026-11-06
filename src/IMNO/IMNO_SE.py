from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.IMNO.common import Encode1d, SpectralConv1d, FourierDecoder1d, ODEFunc


def phase_align(u: torch.Tensor, ref_channel: int = 0, eps: float = 1e-12):
    """
    u: (B, L, C)
    returns:
      uhat: (B, L, C)  -- remove phase of k=1 mode (measured on ref_channel)
      phi:  (B,)       -- extracted phase
    """
    assert u.dim() == 3, "u must be (B, L, C)"
    B, L, C = u.shape

    # reference channel for phase
    u_ref = u[:, :, ref_channel]                 # (B, L)
    a = torch.fft.rfft(u_ref, dim=1)             # (B, K)
    a1 = a[:, 1]                                  # k=1

    # stable phase: if |a1| tiny, atan2 still defined but noisy; eps helps avoid 0/0 gradients
    phi = torch.atan2(a1.imag, a1.real + eps)    # (B,)

    # apply exp(-i k phi) to ALL channels
    A = torch.fft.rfft(u, dim=1)                 # (B, K, C)
    k = torch.arange(A.size(1), device=u.device, dtype=A.real.dtype).view(1, -1, 1)  # (1,K,1)
    phase = torch.exp(-1j * k * phi.view(-1, 1, 1))                                  # (B,K,1)
    uhat = torch.fft.irfft(A * phase, n=L, dim=1)                                     # (B,L,C)
    return uhat, phi


def add_phase(uhat: torch.Tensor, phi: torch.Tensor):
    """
    uhat: (B, L, C)
    phi:  (B,)
    return:
      u: (B, L, C)  -- apply exp(+i k phi)
    """
    assert uhat.dim() == 3, "uhat must be (B, L, C)"
    assert phi.dim() == 1, "phi must be (B,)"
    B, L, C = uhat.shape
    assert phi.size(0) == B

    A = torch.fft.rfft(uhat, dim=1)              # (B, K, C)
    k = torch.arange(A.size(1), device=uhat.device, dtype=A.real.dtype).view(1, -1, 1)
    phase = torch.exp(1j * k * phi.view(-1, 1, 1))
    u = torch.fft.irfft(A * phase, n=L, dim=1)   # (B, L, C)
    return u

class ResidualStepNO1d(nn.Module):
    """
    Single-step residual dynamics:
        v_{t+1} = Phi(v_t, grid; h_t or h_{t+1})
    v is a function (B,L,1,C) (no window).
    """
    def __init__(self, num_channels: int, width: int = 64, modes: int = 16, hidden_states: int = 64, grid_dim: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.width = width
        self.modes1 = modes
        self.hidden_states = hidden_states
        self.grid_dim = grid_dim

        # input feature per point is [v (C), phase_feat (1)], not grid directly
        self.fc0 = nn.Linear(num_channels + 1, width)

        # FiLM (scale + shift) conditioning of h into each block
        self.film0 = nn.Linear(hidden_states, 2 * width)
        self.film1 = nn.Linear(hidden_states, 2 * width)
        self.film2 = nn.Linear(hidden_states, 2 * width)
        self.film3 = nn.Linear(hidden_states, 2 * width)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, v: torch.Tensor, grid: torch.Tensor, h: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """
        v: (B,L,1,C) or (B,L,C)
        grid: (B,L,grid_dim)
        h: (B,H)
        return: (B,L,1,C)
        """
        if v.dim() == 4:
            v = v.squeeze(-2)  # (B,L,C)
        assert v.dim() == 3, "v must be (B,L,C) or (B,L,1,C)"
        assert h.dim() == 2, "h must be (B,H)"
        assert grid.dim() == 3, "grid must be (B,L,grid_dim)"

        # per-point phase feature: cos(Phi + 2*pi*l/L)
        B, L, C = v.shape
        device = v.device
        l = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        Phi_expanded = Phi.unsqueeze(-1)  # (B, 1)
        phase_feat = torch.cos(Phi_expanded + 2 * torch.pi * l / L)  # (B, L)
        x = torch.cat([v, phase_feat.unsqueeze(-1)], dim=-1)  # (B, L, C+1)
        x = self.fc0(x)                # (B,L,width)
        x = x.permute(0, 2, 1)         # (B,width,L)

        def film(layer, feat):
            gam, bet = layer(h).chunk(2, dim=-1)   # each (B, width)
            gam = gam.unsqueeze(-1); bet = bet.unsqueeze(-1)  # (B, width, 1)
            gamma = 1.0 + torch.tanh(gam)
            return gamma * feat + bet

        x = F.gelu(film(self.film0, self.w0(x) + self.conv0(x)))
        x = F.gelu(film(self.film1, self.w1(x) + self.conv1(x)))
        x = F.gelu(film(self.film2, self.w2(x) + self.conv2(x)))
        x = film(self.film3, self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 1)         # (B,L,width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                # (B,L,C)
        return x.unsqueeze(-2)         # (B,L,1,C)


# -------------------------- IMNO --------------------------

class IMNO1d(nn.Module):
    """
    State: (h_t, v_t, Phi_t)
      - Phi_t: accumulated global phase in original frame
      - v_t: residual in canonical frame
    """
    def __init__(
        self,
        num_channels: int,
        width: int = 64,
        modes: int = 24,
        modes3: int = 8,
        modes4: int = 12,
        initial_step: int = 10,
        hidden_states: int = 64,
        grid_dim: int = 1,
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
        # When initial_step > 1, T and C are stacked on the channel dim, so
        # effective_channels = T*C is treated exactly like C everywhere below.
        self.effective_channels = num_channels * initial_step
        self.hidden_states = hidden_states
        self.modes1 = modes
        self.grid_dim = grid_dim

        self.dt = float(dt)
        self.ode_method = ode_method
        self.ode_substeps = int(ode_substeps)
        self.ref_channel = ref_channel

        # encoder/decoder (both operate on the effective_channels dim)
        self.h_encoder = Encode1d(width, modes3, hidden_states)
        if num_channels == 1:
            # Single physical channel: all T*C effective channels share one
            # decoder, gauge fixing is always on.
            self.h_decoder = FourierDecoder1d(
                hidden_states, self.effective_channels, modes4, mlp_width=ode_hidden, apply_gauge_fix=True
            )
        else:
            # Multiple physical channels: one decoder per physical channel,
            # gauge fixing only on the decoder matching ref_channel.
            self.h_decoders = nn.ModuleList(
                FourierDecoder1d(
                    hidden_states,
                    initial_step,
                    modes4,
                    mlp_width=ode_hidden,
                    apply_gauge_fix=(c == ref_channel),
                )
                for c in range(num_channels)
            )

        # Input dim = effective_channels (C when T=1, T*C when T>1)
        self.fc0_h = nn.Linear(self.effective_channels, width)

        # ODE
        self.h_ode = ODEFunc(hidden_states, hidden=ode_hidden, depth=ode_depth)

        # residual model (input/output on the effective_channels dim)
        self.v_model = ResidualStepNO1d(
            num_channels=self.effective_channels,
            width=width,
            modes=modes,
            hidden_states=hidden_states,
            grid_dim=grid_dim,
        )

        # MLP to predict dPhi from h_next
        dphi_mlp_width = 200
        self.dphi_mlp = nn.Sequential(
            nn.Linear(hidden_states, dphi_mlp_width),
            nn.GELU(),
            nn.Linear(dphi_mlp_width, dphi_mlp_width),
            nn.GELU(),
            nn.Linear(dphi_mlp_width, 1),
        )

    def _ode_step(self, h: torch.Tensor) -> torch.Tensor:
        dt = self.dt / max(self.ode_substeps, 1)

        def f(x): return self.h_ode(x)

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
        """
        Predict dPhi from uhat_m using MLP.
        
        Args:
            h_next: (B, H)  latent state at next time step
            
        Returns:
            dPhi: (B,)  predicted phase shift
        """
        assert h_next.dim() == 2, "h_next must be (B, H)"
        B, H = h_next.shape
        assert H == self.hidden_states
        
        # Predict dPhi
        dPhi = self.dphi_mlp(h_next).squeeze(-1)  # (B,)
        return dPhi

    def _decode_h_to_uhat(self, h: torch.Tensor, n: int) -> torch.Tensor:
        """Multi-channel branch: decode per channel then concat to (B, T*C, L), matching the single-decoder layout."""
        T = self.initial_step
        C = self.num_channels
        decoded = [dec(h, n) for dec in self.h_decoders]

        if T == 1:
            return torch.cat(decoded, dim=1)

        chunks = [torch.stack([decoded[c][:, t] for c in range(C)], dim=1) for t in range(T)]
        return torch.cat(chunks, dim=1)

    def _decode(self, h: torch.Tensor, n: int) -> torch.Tensor:
        if self.num_channels == 1:
            return self.h_decoder(h, n=n)
        return self._decode_h_to_uhat(h, n)

    def _init_from_window(self, x: torch.Tensor, grid: torch.Tensor):
        """
        x: (B, L, T*C)
        When initial_step > 1, T and C are stacked on the channel dim, so
        u_t = x, shape (B,L,TC), is treated exactly like C elsewhere.
        returns:
            h_t:   (B,H)
            v_t:   (B,L,1,effective_C)   residual in canonical frame
            Phi_t: (B,)
        """
        B, L, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"

        # Use x directly as u_t: T*C channels, treated exactly like C.
        u_t = x  # (B, L, TC)

        # align to canonical
        uhat_t, Phi_t = phase_align(u_t, ref_channel=self.ref_channel)  # (B,L,TC), (B,)

        # encode from canonical uhat_t
        xh = uhat_t                                # (B,L,TC)
        xh = self.fc0_h(xh)                # (B,L,width)
        xh = xh.permute(0, 2, 1)           # (B,width,L)
        h_t = self.h_encoder(xh)           # (B,H)

        # residual in canonical frame
        uhat_m_t = self._decode(h_t, L).permute(0, 2, 1)  # (B,L,TC)

        u_m_t = add_phase(uhat_m_t, Phi_t)
        v_t = (u_t - u_m_t).unsqueeze(-2)               # (B,L,1,TC)

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
        Args:
            x:   (B, L, T*C)  used if any of (h,v,Phi) is None
            grid:(B, L, grid_dim)
            h:   (B,H)
            v:   (B,L,effective_C)   canonical residual (effective_C = C or T*C)
            Phi: (B,)       accumulated phase in original frame

        Returns:
            u_next:   (B,L,1,effective_C)  original frame (effective_C = C or T*C, matches input)
            h_next:   (B,H)
            v_next:   (B,L,1,effective_C)  canonical residual
            Phi_next: (B,)
        """
        assert x.dim() == 3, "x must be (B, L, T*C)"
        B, L, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"
        assert grid.dim() == 3, "grid must be (B, L, grid_dim)"

        if (h is None) or (v is None) or (Phi is None):
            h_t, v_t, Phi_t = self._init_from_window(x, grid)
        else:
            h_t, v_t, Phi_t = h, v, Phi

        # step latent
        h_next = self._ode_step(h_t)  # (B,H)

        # residual step
        v_next = self.v_model(v_t, grid, h_next, Phi_t)

        # manifold part in canonical frame
        uhat_m = self._decode(h_next, L).permute(0, 2, 1)  # (B,L,effective_C)
        dPhi = self._predict_dphi_from_uhat_m(h_next)
        Phi_next = Phi_t + dPhi
        u_next = v_next + add_phase(uhat_m, Phi_next).unsqueeze(-2)  # (B,L,1,effective_C)

        return u_next, h_next, v_next, Phi_next
