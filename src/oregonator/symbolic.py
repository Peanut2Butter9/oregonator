"""Symbolic derivation (SymPy) of the scaling, equilibria and characteristic polynomials."""
from __future__ import annotations

import sympy as sp

x, y, z, eps, delta, q, f, lam = sp.symbols("x y z epsilon delta q f lambda", real=True)


def vector_field() -> sp.Matrix:
    return sp.Matrix([
        (q * y - x * y + x * (1 - x)) / eps,
        (-q * y - x * y + 2 * f * z) / delta,
        x - z,
    ])


def jacobian() -> sp.Matrix:
    return vector_field().jacobian([x, y, z])


def scaled_system():
    """Apply the slides' scaling to the dimensional mass-action system.

    Returns the right-hand sides of dx/dtau, dy/dtau, dz/dtau in terms of the
    rate constants, A = [BrO3-] and B = [BrMA].
    """
    k0, k2, k3, k4, k5, A, B, F = sp.symbols("k0 k2 k3 k4 k5 A B f", positive=True)
    X0 = k5 * A / (2 * k4)
    Y0 = k5 * A / k2
    Z0 = (k5 * A) ** 2 / (k4 * k0 * B)
    T0 = 1 / (k0 * B)
    X, Y, Z = X0 * x, Y0 * y, Z0 * z
    dX = k3 * A * Y - k2 * X * Y + k5 * A * X - 2 * k4 * X**2
    dY = -k3 * A * Y - k2 * X * Y + F * k0 * B * Z
    dZ = 2 * k5 * A * X - k0 * B * Z
    return [sp.simplify(T0 / X0 * dX), sp.simplify(T0 / Y0 * dY), sp.simplify(T0 / Z0 * dZ)]


def equilibrium_quadratic() -> sp.Expr:
    """Polynomial in x whose positive root is x* (after y = 2fx/(q+x), z = x)."""
    ys = 2 * f * x / (q + x)
    return sp.expand(sp.numer(sp.together(vector_field()[0].subs({y: ys, z: x}) * eps / x)))


def x_star() -> sp.Expr:
    b = 1 - 2 * f - q
    return (b + sp.sqrt(b**2 + 4 * q * (1 + 2 * f))) / 2


def char_poly(point) -> sp.Poly:
    J = jacobian().subs(point)
    return sp.Poly(sp.expand((lam * sp.eye(3) - J).det()), lam)
