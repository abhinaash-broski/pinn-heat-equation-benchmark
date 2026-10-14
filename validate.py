"""Independent, CPU-only verification of the benchmark results (runs in ~5 s).

What it checks, using only the committed CSVs in computed-solutions/:
  1. Reproduces the README error table (rel. L2 and L-inf) for every solver.
  2. Re-derives the ground truth independently. The IC is a product of 1D
     triangles, so the 2D/3D solution is a product of 1D Fourier series; this
     gives a 4001-term reference in milliseconds instead of a triple sum.
  3. Scores a "do nothing" baseline u(x,t) = u0(x) so the errors have context.
  4. Confirms the FDM time-sampling offset in the notebook (see README).
  5. Runs a time-aligned explicit-FDM grid-refinement study and plots
     accuracy vs. wall-clock cost against the trained PINNs.

Usage:  python validate.py            (writes plots/accuracy_vs_cost.png)
"""
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "computed-solutions")
ALPHA, NX = 1.1644e-4, 15
X = np.linspace(0, 1, NX)
T = np.arange(1, 61, dtype=float)
COLS = {1: ["x"], 2: ["x", "y"], 3: ["x", "y", "z"]}
METHODS = ["explicit", "implicit", "crank_nicolson"]

# values published in README.md (rel. L2 in %), used as regression targets
README_REL_L2 = {
    1: {"explicit": 0.751, "implicit": 0.788, "crank_nicolson": 0.770, "pinn": 0.276},
    2: {"explicit": 1.212, "implicit": 1.268, "crank_nicolson": 1.240, "pinn": 0.845},
    3: {"explicit": 1.646, "implicit": 1.723, "crank_nicolson": 1.685, "pinn": 1.583},
}


def tri(xs):
    return np.where(xs <= 0.5, 2 * xs, 2 * (1 - xs))


def outer_power(v, dim):
    out = v
    for _ in range(dim - 1):
        out = np.multiply.outer(out, v)
    return out


def series_1d(xs, t, n_terms):
    n = np.arange(1, n_terms + 1)
    b = 8 / (n**2 * np.pi**2) * np.sin(n * np.pi / 2)
    return (b * np.exp(-ALPHA * (n * np.pi) ** 2 * t) * np.sin(np.outer(xs, n) * np.pi)).sum(1)


U1_EXACT = np.array([series_1d(X, t, 4001) for t in T])


def exact(dim):
    """(60, 15, ..., 15) array; separable product of the 1D solution."""
    subs = "ijk"[:dim]
    return np.einsum(",".join(f"t{s}" for s in subs) + f"->t{subs}", *([U1_EXACT] * dim))


def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def load_grid(path, dim, method=None):
    """Load a solution CSV into a (time, x[, y[, z]]) array indexed by coordinate."""
    df = pd.read_csv(path)
    if method is not None:
        df = df[df["method"] == method]
    ti = np.rint(df["time"].values).astype(int) - 1
    idx = [np.rint(df[c].values * (NX - 1)).astype(int) for c in COLS[dim]]
    out = np.full((60,) + (NX,) * dim, np.nan)
    out[(ti, *idx)] = df["temperature"].values
    if np.isnan(out).any():
        raise ValueError(f"{path}: grid not fully populated")
    return out


def pinn_file(dim):
    (path,) = glob.glob(os.path.join(DATA, f"pinn_solutions_{dim}d_*.csv"))
    train_s = float(re.search(r"time_([\d.]+)s", path).group(1))
    return path, train_s


def fdm_explicit(dim, nx, dt):
    """Time-aligned explicit FDM on an nx^dim grid, sampled on the 15^dim grid at t=1..60."""
    xs = np.linspace(0, 1, nx)
    h = xs[1] - xs[0]
    r = ALPHA * dt / h**2
    assert dim * r <= 0.5, "explicit stability limit violated"
    u = outer_power(tri(xs), dim)
    for ax in range(dim):
        u[(slice(None),) * ax + (0,)] = 0.0
        u[(slice(None),) * ax + (-1,)] = 0.0
    inner = (slice(1, -1),) * dim
    stride = (nx - 1) // (NX - 1)
    sample = (slice(None, None, stride),) * dim
    steps_per_second = int(round(1 / dt))
    snaps = []
    t0 = time.perf_counter()
    for step in range(1, 60 * steps_per_second + 1):
        lap = -2 * dim * u[inner]
        for ax in range(dim):
            lo, hi = [slice(1, -1)] * dim, [slice(1, -1)] * dim
            lo[ax], hi[ax] = slice(0, -2), slice(2, None)
            lap = lap + u[tuple(lo)] + u[tuple(hi)]
        u[inner] += r * lap
        if step % steps_per_second == 0:
            snaps.append(u[sample].copy())
    return np.array(snaps), time.perf_counter() - t0


def stable_dt(dim, nx):
    h = 1 / (nx - 1)
    return 1 / max(1, int(np.ceil(dim * ALPHA / (0.5 * h * h))))


def main():
    failures = []

    print("1-3. Reported errors, independent ground truth, trivial baseline")
    print(f"{'dim':<4}{'method':<16}{'rel L2 %':>10}{'README %':>10}{'L-inf':>9}")
    pinn_points = {}
    for dim in (1, 2, 3):
        truth = exact(dim)
        ana = load_grid(os.path.join(DATA, f"analytical_solutions_{dim}d.csv"), dim)
        err_ref = rel_l2(ana, truth)
        if err_ref > 1e-4:
            failures.append(f"{dim}D analytical CSV deviates from exact solution ({err_ref:.1e})")
        rows = [(m, load_grid(os.path.join(DATA, f"fdm_solutions_{dim}d.csv"), dim, m)) for m in METHODS]
        path, train_s = pinn_file(dim)
        rows.append(("pinn", load_grid(path, dim)))
        for name, pred in rows:
            e = 100 * rel_l2(pred, truth)
            if abs(e - README_REL_L2[dim][name]) > 0.0015:
                failures.append(f"{dim}D {name}: {e:.4f}% != README {README_REL_L2[dim][name]}%")
            print(f"{dim}D  {name:<16}{e:>10.3f}{README_REL_L2[dim][name]:>10.3f}"
                  f"{np.abs(pred - truth).max():>9.4f}")
        pinn_points[dim] = (train_s, 100 * rel_l2(rows[-1][1], truth))
        base = 100 * rel_l2(np.broadcast_to(outer_power(tri(X), dim), truth.shape), truth)
        print(f"{dim}D  {'u = u0 baseline':<16}{base:>10.3f}{'':>10}   "
              f"(analytical CSV vs 4001-term truth: {err_ref:.1e})")

    print("\n4. FDM time sampling in the notebook (dt = 0.6 s)")
    dt0, n_steps = 0.6, 100
    eval_idx = [min(int(np.round(t / dt0)), n_steps - 1) for t in T]
    actual = np.array([(i + 1) * dt0 for i in eval_idx])
    u = tri(X)
    r = ALPHA * dt0 / (X[1] - X[0]) ** 2
    snaps = []
    for _ in range(n_steps):
        u = u.copy()
        u[1:-1] += r * (u[2:] - 2 * u[1:-1] + u[:-2])
        snaps.append(u)
    stored = load_grid(os.path.join(DATA, "fdm_solutions_1d.csv"), 1, "explicit")
    reproduced = np.allclose(np.array([snaps[i] for i in eval_idx]), stored)
    print(f"   rows labelled t = 1, 2, 3, 4 s are really t = {', '.join(f'{a:g}' for a in actual[:4])} s "
          f"(mean offset {np.mean(actual - T):+.2f} s); reproduced from CSV: {reproduced}")

    print("\n5. Time-aligned explicit FDM, grid refinement (errors on the same 15^d evaluation grid)")
    print(f"{'dim':<4}{'grid':>6}{'dt (s)':>9}{'rel L2 %':>10}{'time (s)':>11}")
    refinement = {1: [15, 29, 57, 113, 225], 2: [15, 29, 57, 113], 3: [15, 29, 57]}
    fdm_curves = {}
    for dim, grids in refinement.items():
        truth = exact(dim)
        curve = []
        for nx in grids:
            dt = stable_dt(dim, nx)
            best = np.inf
            for _ in range(3):  # best-of-3 timing; these runs are milliseconds
                sol, elapsed = fdm_explicit(dim, nx, dt)
                best = min(best, elapsed)
            e = 100 * rel_l2(sol, truth)
            curve.append((nx, best, e))
            print(f"{dim}D  {nx:>5}^{dim}{dt:>8.3f}{e:>10.4f}{best:>11.4f}")
        fdm_curves[dim] = curve
        pinn_s, pinn_e = pinn_points[dim]
        match = next(((nx, s) for nx, s, e in curve if e <= pinn_e), None)
        if match:
            print(f"    -> FDM matches the PINN ({pinn_e:.3f}%) at {match[0]}^{dim} in {match[1]:.4f} s "
                  f"vs {pinn_s:.0f} s of PINN training (~{pinn_s / max(match[1], 1e-4):.0e}x faster)")

    plot(fdm_curves, pinn_points)

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll README error figures reproduced from the committed CSVs.")
    return 0


def plot(fdm_curves, pinn_points):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fdm_c, pinn_c, ink, muted = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, dim in zip(axes, (1, 2, 3)):
        nx, secs, errs = zip(*fdm_curves[dim])
        secs = np.maximum(secs, 1e-4)
        ax.plot(secs, errs, "-o", color=fdm_c, lw=2, ms=7, label="Explicit FDM (refined grid)")
        for n, s, e in zip(nx, secs, errs):
            ax.annotate(f"{n}$^{dim}$", (s, e), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=muted)
        ps, pe = pinn_points[dim]
        ax.plot([ps], [pe], "D", color=pinn_c, ms=9, label="PINN (best of search, GPU)")
        ax.annotate(f"PINN {ps:.0f} s", (ps, pe), textcoords="offset points", xytext=(-10, 8),
                    ha="right", fontsize=9, color=ink)
        ax.axhline(pe, color=pinn_c, lw=1, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{dim}D heat equation", color=ink)
        ax.set_xlabel("wall-clock time (s, log)", color=muted)
        ax.grid(True, which="major", color="#e4e3dc", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("relative L2 error, % (log)", color=muted)
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Accuracy vs. cost: FDM on a finer grid reaches PINN accuracy in milliseconds",
                 color=ink, fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "plots", "accuracy_vs_cost.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    sys.exit(main())
