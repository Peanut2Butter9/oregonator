"""Numerical integration of the stiff Oregonator and measurement of its attractor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from .model import Params, jacobian, rhs


def simulate(p: Params, u0, t_end: float, n: int = 4001, rtol: float = 1e-9, atol: float = 1e-14):
    """Integrate with Radau (implicit, the system is stiff) and sample on a uniform grid."""
    sol = solve_ivp(
        rhs, (0.0, t_end), u0, args=(p,), method="Radau",
        jac=lambda t, u, p: jacobian(u, p), rtol=rtol, atol=atol, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    t = np.linspace(0.0, t_end, n)
    return t, sol.sol(t)


@dataclass
class Attractor:
    f: float
    oscillates: bool
    log_x_min: float
    log_x_max: float
    period: float  # nan when the trajectory settles to E2*
    u_end: np.ndarray


def attractor(p: Params, u0, t_end: float = 60.0, transient: float = 30.0, n: int = 120001) -> Attractor:
    """Run past the transient and report the range of log10 x and the period of y's peaks."""
    t, u = simulate(p, u0, t_end, n)
    keep = t >= transient
    log_x = np.log10(np.clip(u[0, keep], 1e-300, None))
    log_y = np.log10(np.clip(u[1, keep], 1e-300, None))
    oscillates = log_x.max() - log_x.min() > 1e-3
    period = np.nan
    if oscillates:
        peaks, _ = find_peaks(log_y, prominence=0.25 * (log_y.max() - log_y.min()))
        if len(peaks) >= 2:
            period = float(np.diff(t[keep][peaks]).mean())
    return Attractor(p.f, bool(oscillates), float(log_x.min()), float(log_x.max()), period, u[:, -1])
