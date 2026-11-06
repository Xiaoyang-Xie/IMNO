"""
IMNO1d and IMNO2d: the plain (non-shift-equivariant) IMNO models, 1D and 2D,
in one file, mirroring FNO/FNO.py's and RNO/RNO.py's layout.
"""
from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.IMNO.common import (
    Encode1d, SpectralConv1d, FourierDecoder1d,
    Encode2d, SpectralConv2d, FourierDecoder2d,
    ODEFunc,
)


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

        self.fc0 = nn.Linear(num_channels + grid_dim, width)

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

    def forward(self, v: torch.Tensor, grid: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
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

        g = grid[..., : self.grid_dim]
        x = torch.cat([v, g], dim=-1)   # (B,L,C+grid_dim)
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
    IMNO rollout state is (h_t, v_t) only (single-step).
    forward(x, grid, h=None, v=None) returns (u_next, h_next, v_next).
    """
    def __init__(
        self,
        num_channels: int,
        width: int = 64,
        modes: int = 24,
        modes3: int = 6,                # for h encoder
        modes4: int = 8,                # for h decoder
        initial_step: int = 10,        # kept for interface / init from window
        hidden_states: int = 64,
        grid_dim: int = 1,
        # ODE
        dt: float = 1.0,
        ode_method: Literal["euler", "rk4"] = "euler",
        ode_substeps: int = 1,
        ode_hidden: int = 128,
        ode_depth: int = 2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.initial_step = initial_step
        self.hidden_states = hidden_states
        self.modes1 = modes
        self.grid_dim = grid_dim

        self.dt = float(dt)
        self.ode_method = ode_method
        self.ode_substeps = int(ode_substeps)

        # manifold encoder/decoder
        self.h_encoder = Encode1d(width, modes3, hidden_states)
        self.h_decoder = FourierDecoder1d(hidden_states, num_channels, modes4, mlp_width=ode_hidden)
        self.fc0_h = nn.Linear(num_channels + grid_dim, width)

        # neural ODE
        self.h_ode = ODEFunc(hidden_states, hidden=ode_hidden, depth=ode_depth)

        # residual single-step operator
        self.v_model = ResidualStepNO1d(
            num_channels=num_channels,
            width=width,
            modes=modes,
            hidden_states=hidden_states,
            grid_dim=grid_dim,
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

    def _init_from_window(self, x: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize single-step state (h_t, v_t) from the input window.

        Args:
            x:    (B, L, T*C)
            grid: (B, L, grid_dim)

        Returns:
            h_t: (B, H)
            v_t: (B, L, 1, C)
        """
        B, L, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"

        u_win = x.view(B, L, T, C)  # (B,L,T,C)

        # --- compute h_t from LIFTED features (width), not raw u ---

        u_t = u_win[:, :, -1, :]  # (B,L,C)
        xh = torch.cat([u_t, grid[..., : self.grid_dim]], dim=-1)  # (B,L,C+grid_dim)
        xh = self.fc0_h(xh)                # (B,L,width)
        xh = xh.permute(0, 2, 1)           # (B,width,L)
        h_t = self.h_encoder(xh)           # (B,H)

        # --- compute residual v ---
        u_m_t = self.h_decoder(h_t, n=L).permute(0, 2, 1)  # (B,L,C)
        v_t = (u_t - u_m_t).unsqueeze(-2)                  # (B,L,1,C)
        return h_t, v_t


    def forward(self, x: torch.Tensor, grid: torch.Tensor, h: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None):
        """
        Args:
            x:    (B, L, T*C)  only used when (h is None) or (v is None)
            grid: (B, L, grid_dim)
            h:    None or (B, H)
            v:    None or (B, L, 1, C)

        Returns:
            u_next: (B, L, 1, C)
            h_next: (B, H)
            v_next: (B, L, 1, C)  
        """
        assert x.dim() == 3, "x must be (B, L, T*C)"
        B, L, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"
        assert grid.dim() == 3, "grid must be (B, L, grid_dim)"

        # init if needed
        if (h is None) or (v is None):
            h_t, v_t = self._init_from_window(x,grid)
        else:
            h_t, v_t = h, v

        # step h
        h_next = self._ode_step(h_t)  # (B,H)

        # step v (conditioned on h_next)
        v_next = self.v_model(v_t, grid, h_next)  # (B,L,1,C)

        # compose u
        u_m_next = self.h_decoder(h_next, n=L)  # (B,C,L)
        u_m_next = u_m_next.permute(0, 2, 1).unsqueeze(-2)  # (B,L,1,C)
        u_next = u_m_next + v_next

        return u_next, h_next, v_next


# -------------------------- 2D --------------------------

class ResidualStepNO2d(nn.Module):
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

        self.fc0 = nn.Linear(num_channels + grid_dim, width)

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

    def forward(self, v: torch.Tensor, grid: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if v.dim() == 5:
            v = v.squeeze(-2)
        assert v.dim() == 4, "v must be (B,H,W,C) or (B,H,W,1,C)"
        assert h.dim() == 2, "h must be (B,H)"
        assert grid.dim() == 4, "grid must be (B,H,W,grid_dim)"

        g = grid[..., : self.grid_dim]
        x = torch.cat([v, g], dim=-1)
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

        self.h_encoder = Encode2d(width, modes3, hidden_states)
        self.h_decoders = nn.ModuleList(
            FourierDecoder2d(hidden_states, initial_step, modes4, mlp_width=ode_hidden)
            for _ in range(num_channels)
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

    def _decode_h_to_u(self, h: torch.Tensor, n: tuple[int, int]) -> torch.Tensor:
        T = self.initial_step
        C = self.num_channels
        decoded = [dec(h, n) for dec in self.h_decoders]

        if T == 1:
            return torch.cat(decoded, dim=1)

        chunks = [torch.stack([decoded[c][:, t] for c in range(C)], dim=1) for t in range(T)]
        return torch.cat(chunks, dim=1)

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

    def _init_from_window(self, x: torch.Tensor, grid: torch.Tensor):
        B, H, W, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"

        u_t = x
        xh = self.fc0_h(u_t)
        xh = xh.permute(0, 3, 1, 2)
        h_t = self.h_encoder(xh)

        u_m_t = self._decode_h_to_u(h_t, (H, W)).permute(0, 2, 3, 1)
        v_t = (u_t - u_m_t).unsqueeze(-2)

        return h_t, v_t

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        assert x.dim() == 4, "x must be (B, H, W, T*C)"
        B, H, W, TC = x.shape
        T = self.initial_step
        C = self.num_channels
        assert TC == T * C, f"Expected {T*C}, got {TC}"
        assert grid.dim() == 4, "grid must be (B, H, W, grid_dim)"

        if (h is None) or (v is None):
            h_t, v_t = self._init_from_window(x, grid)
        else:
            h_t, v_t = h, v

        h_next = self._ode_step(h_t)
        v_next = self.v_model(v_t, grid, h_next)

        u_m_next = self._decode_h_to_u(h_next, (H, W)).permute(0, 2, 3, 1).unsqueeze(-2)
        u_next = v_next + u_m_next

        return u_next, h_next, v_next