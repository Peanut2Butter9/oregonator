"""Hopf points, transversality and the linear period at E2*."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .model import Params, char_coeffs, eigenvalues, routh_hurwitz


def hopf_points(p: Params, f_min: float = 1e-3, f_max: float = 5.0, n: int = 5000) -> list[float]:
    """Values of f where AB - C changes sign, i.e. where E2* has eigenvalues +-i omega."""
    fs = np.linspace(f_min, f_max, n)
    h = np.array([routh_hurwitz(p.with_f(f)) for f in fs])
    idx = np.where(np.diff(np.sign(h)) != 0)[0]
    return [brentq(lambda f: routh_hurwitz(p.with_f(f)), fs[i], fs[i + 1], xtol=1e-13) for i in idx]


def leading_real_part(p: Params) -> float:
    """Real part of the complex pair (the eigenvalue other than -A at the Hopf point)."""
    ev = eigenvalues(p)
    complex_pair = ev[np.abs(ev.imag) > 0]
    if complex_pair.size:
        return complex_pair.real.max()
    return np.sort(ev.real)[-1]


def transversality(p: Params, f_h: float, h: float = 1e-6) -> float:
    """d Re(lambda) / df of the crossing pair at the Hopf point f_h (must be nonzero)."""
    return (leading_real_part(p.with_f(f_h + h)) - leading_real_part(p.with_f(f_h - h))) / (2 * h)


def hopf_frequency(p: Params) -> float:
    """omega = sqrt(C / A): the imaginary part of the crossing pair when AB = C."""
    A, _, C = char_coeffs(p)
    return float(np.sqrt(C / A))


def linear_period(p: Params) -> float:
    """2 pi / omega. Only the period of small oscillations at a Hopf point."""
    return 2 * np.pi / hopf_frequency(p)
