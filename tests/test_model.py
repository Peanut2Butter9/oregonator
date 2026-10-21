import numpy as np
import pytest
import sympy as sp

from oregonator import symbolic as S
from oregonator.analysis import hopf_frequency, hopf_points, transversality
from oregonator.model import (
    SLIDE_PARAMS,
    SLIDE_RATES,
    SLIDE_U0,
    Params,
    char_coeffs,
    char_coeffs_origin,
    eigenvalues,
    equilibrium,
    from_rate_constants,
    rhs,
    routh_hurwitz,
)
from oregonator.simulate import attractor, simulate

RNG = np.random.default_rng(0)
RANDOM_PARAMS = [
    Params(eps=10 ** RNG.uniform(-5, -2), delta=10 ** RNG.uniform(-5, -2), q=10 ** RNG.uniform(-6, -3), f=RNG.uniform(0.05, 3))
    for _ in range(20)
]


def subs(p: Params):
    return {S.eps: p.eps, S.delta: p.delta, S.q: p.q, S.f: p.f}


def test_scaling_gives_eps_as_k0B_over_k5A():
    dx, dy, dz = S.scaled_system()
    k0, k2, k3, k4, k5, A, B = sp.symbols("k0 k2 k3 k4 k5 A B", positive=True)
    q = 2 * k4 * k3 / (k2 * k5)
    inner = q * S.y - S.x * S.y + S.x * (1 - S.x)
    # dx/dtau = inner / eps with eps = k0 B / (k5 A), not k5 A / (k0 B) as in the slides
    assert sp.simplify(dx - inner / (k0 * B / (k5 * A))) == 0
    assert sp.simplify(dz - (S.x - S.z)) == 0
    delta = 2 * k0 * k4 * B / (k2 * k5 * A)
    F = sp.Symbol("f", positive=True)
    assert sp.simplify(dy - (-q * S.y - S.x * S.y + 2 * F * S.z) / delta) == 0


def test_rate_constants_in_slides_give_q_8e_6():
    p = from_rate_constants(**SLIDE_RATES, A=1.0, B=1.0, f=0.4)
    assert p.q == pytest.approx(8e-6)


def test_equilibrium_quadratic_uses_2f():
    quad = S.equilibrium_quadratic()
    expected = -S.x**2 + (1 - 2 * S.f - S.q) * S.x + S.q * (1 + 2 * S.f)
    assert sp.expand(quad - expected) == 0


@pytest.mark.parametrize("p", RANDOM_PARAMS[:5] + [SLIDE_PARAMS.with_f(0.1), SLIDE_PARAMS.with_f(0.4)])
def test_equilibrium_is_a_fixed_point(p):
    assert np.allclose(rhs(0, equilibrium(p), p), 0, atol=1e-6 / min(p.eps, p.delta))


def test_slide_x_star_formula_is_not_an_equilibrium():
    p = SLIDE_PARAMS.with_f(0.1)
    D = 1 + p.q**2 + 4 * p.f**2 + 2 * p.q + 12 * p.q * p.f - 4 * p.f
    x_slide = (1 - p.f - p.q + np.sqrt(D)) / 2
    assert x_slide == pytest.approx(0.8504, abs=1e-4)
    assert equilibrium(p)[0] == pytest.approx(0.8004, abs=1e-4)
    assert abs(rhs(0, [x_slide, 2 * p.f * x_slide / (p.q + x_slide), x_slide], p)[0]) > 1e2


def test_e2_exists_for_every_f():
    # the slides' existence condition 1 - 2f - q > 0 is not needed
    for f in [0.6, 1.0, 2.0, 10.0]:
        assert np.all(equilibrium(SLIDE_PARAMS.with_f(f)) > 0)


def test_origin_char_poly():
    poly = S.char_poly({S.x: 0, S.y: 0, S.z: 0})
    for p in RANDOM_PARAMS[:5]:
        _, a2, a1, a0 = [float(c.subs(subs(p))) for c in poly.all_coeffs()]
        A, B, C = char_coeffs_origin(p)
        assert (a2, a1, a0) == pytest.approx((A, B, C), rel=1e-12)
        # the slides' lambda coefficient (q(eps + 1) - delta) / (eps delta) is wrong
        assert B != pytest.approx((p.q * (p.eps + 1) - p.delta) / (p.eps * p.delta), rel=1e-6)
        assert C < 0


@pytest.mark.parametrize("p", RANDOM_PARAMS)
def test_e2_char_coeffs_match_jacobian(p):
    A, B, C = char_coeffs(p)
    c = np.poly(np.linalg.eigvals(np.array(S.jacobian().subs(subs(p)).subs(
        {S.x: equilibrium(p)[0], S.y: equilibrium(p)[1], S.z: equilibrium(p)[2]}), dtype=float)))
    assert c[1:].real == pytest.approx([A, B, C], rel=1e-8)
    assert A > 0 and C > 0


def test_hopf_points_for_slide_parameters():
    lo, hi = hopf_points(SLIDE_PARAMS)
    assert lo == pytest.approx(0.25126, abs=1e-5)
    assert hi == pytest.approx(1.16268, abs=1e-5)
    assert transversality(SLIDE_PARAMS, lo) > 0
    assert transversality(SLIDE_PARAMS, hi) < 0
    ev = eigenvalues(SLIDE_PARAMS.with_f(lo))
    assert np.abs(ev.imag).max() == pytest.approx(hopf_frequency(SLIDE_PARAMS.with_f(lo)), rel=1e-6)


def test_unstable_beyond_one_half():
    # the slides claim instability only for 1/4 < f < 1/2
    assert routh_hurwitz(SLIDE_PARAMS.with_f(1.0)) < 0
    assert routh_hurwitz(SLIDE_PARAMS.with_f(0.1)) > 0
    assert routh_hurwitz(SLIDE_PARAMS.with_f(1.3)) > 0


def test_one_half_is_the_small_q_limit():
    upper = [hopf_points(Params(5e-5, 2e-4, q), f_max=3, n=20000)[1] for q in (8e-6, 8e-10)]
    assert upper[0] == pytest.approx(0.692, abs=1e-3)
    assert upper[1] == pytest.approx(0.502, abs=1e-3)


def test_f_0_1_settles_to_e2():
    t, u = simulate(SLIDE_PARAMS.with_f(0.1), SLIDE_U0, 10.0, 11)
    assert u[:, -1] == pytest.approx([0.8004, 0.1998, 0.8004], abs=2e-3)


def test_f_0_4_oscillates_with_period_about_5():
    a = attractor(SLIDE_PARAMS.with_f(0.4), SLIDE_U0, t_end=40, transient=15, n=80001)
    assert a.oscillates
    assert a.period == pytest.approx(5.06, abs=0.05)
