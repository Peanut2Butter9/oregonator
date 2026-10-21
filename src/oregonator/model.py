"""Dimensionless three-variable Oregonator of the Belousov-Zhabotinsky reaction.

    dx/dtau = (q y - x y + x (1 - x)) / eps
    dy/dtau = (-q y - x y + 2 f z) / delta
    dz/dtau = x - z

x, y, z are the scaled concentrations of HBrO2, Br- and Ce(IV); f is the
stoichiometric factor used as the bifurcation parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    eps: float = 5e-5
    delta: float = 2e-4
    q: float = 8e-4
    f: float = 0.4

    def with_f(self, f: float) -> "Params":
        return replace(self, f=float(f))


# Values used for the simulations in the original slides.
SLIDE_PARAMS = Params()
SLIDE_U0 = (1.27e-7, 6e-7, 5e-3)

# Rate constants listed in the slides (1/(M s)).
SLIDE_RATES = {"k0": 0.4, "k2": 2e9, "k3": 2.0, "k4": 4e7, "k5": 1e4}


def from_rate_constants(k0, k2, k3, k4, k5, A, B, f) -> Params:
    """Scaled parameters from rate constants and the BrO3- (A) and BrMA (B) concentrations.

    The scaling X0 = k5 A / (2 k4), Y0 = k5 A / k2, Z0 = (k5 A)^2 / (k4 k0 B),
    T0 = 1 / (k0 B) gives eps = k0 B / (k5 A); the slides wrote its inverse.
    """
    return Params(
        eps=k0 * B / (k5 * A),
        delta=2 * k0 * k4 * B / (k2 * k5 * A),
        q=2 * k3 * k4 / (k2 * k5),
        f=f,
    )


def rhs(t, u, p: Params):
    x, y, z = u
    return [
        (p.q * y - x * y + x * (1 - x)) / p.eps,
        (-p.q * y - x * y + 2 * p.f * z) / p.delta,
        x - z,
    ]


def jacobian(u, p: Params) -> np.ndarray:
    x, y, z = u
    return np.array([
        [(1 - 2 * x - y) / p.eps, (p.q - x) / p.eps, 0.0],
        [-y / p.delta, -(p.q + x) / p.delta, 2 * p.f / p.delta],
        [1.0, 0.0, -1.0],
    ])


def equilibrium(p: Params) -> np.ndarray:
    """The positive equilibrium E2* = (x*, y*, x*).

    x* is the positive root of x^2 - (1 - 2f - q) x - q (1 + 2f) = 0. The product
    of the roots is -q (1 + 2f) < 0, so E2* exists for every f > 0 and q > 0.
    """
    b = 1 - 2 * p.f - p.q
    x = (b + np.sqrt(b * b + 4 * p.q * (1 + 2 * p.f))) / 2
    return np.array([x, 2 * p.f * x / (p.q + x), x])


def char_coeffs(p: Params) -> tuple[float, float, float]:
    """A, B, C of the characteristic polynomial lambda^3 + A lambda^2 + B lambda + C at E2*."""
    x = equilibrium(p)[0]
    eps, delta, q, f = p.eps, p.delta, p.q, p.f
    a = (x * x + q * (x + 2 * f)) / (q + x)
    A = a / eps + (q + x) / delta + 1
    B = a / eps + (q + x) / delta + (a * (q + x) + 2 * f * x * (q - x) / (q + x)) / (eps * delta)
    C = (x * x + q * (1 + 2 * f)) / (eps * delta)
    return A, B, C


def char_coeffs_origin(p: Params) -> tuple[float, float, float]:
    """A, B, C at E1* = (0, 0, 0). C < 0, so E1* always has a positive real eigenvalue."""
    eps, delta, q, f = p.eps, p.delta, p.q, p.f
    return (
        (eps * (delta + q) - delta) / (eps * delta),
        (q * (eps - 1) - delta) / (eps * delta),
        -q * (1 + 2 * f) / (eps * delta),
    )


def routh_hurwitz(p: Params) -> float:
    """AB - C at E2*: positive means stable, negative unstable (A, C > 0 always)."""
    A, B, C = char_coeffs(p)
    return A * B - C


def eigenvalues(p: Params, u=None) -> np.ndarray:
    u = equilibrium(p) if u is None else u
    return np.linalg.eigvals(jacobian(u, p))
