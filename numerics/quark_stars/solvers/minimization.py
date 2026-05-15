"""Shared multi-start minimization helper for QMD order-parameter landscapes.

Designed for minimising the QMD grand potential over (sigma, Delta) but
works for any bounded scalar-valued objective in N dimensions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# Methods that accept the `bounds` keyword in scipy.optimize.minimize.
_BOUNDS_SUPPORTING_METHODS = frozenset({"L-BFGS-B", "TNC", "SLSQP", "trust-constr"})

# Maximum number of best grid points promoted to local minimization starts.
_DEFAULT_GRID_TOP_K = 5


@dataclass(frozen=True)
class MinimizationResult:
    """Outcome of find_global_minimum."""

    x: np.ndarray
    fun: float
    success: bool
    message: str
    n_attempts: int
    n_successful: int


def find_global_minimum(
    objective,
    bounds,
    initial_guesses=None,
    grid_shape=None,
    method: str = "L-BFGS-B",
    options: dict | None = None,
) -> MinimizationResult:
    """Multi-start bounded minimization over an N-dimensional domain.

    Parameters
    ----------
    objective:
        Callable ``f(x: np.ndarray) -> float``.  Returns the value to
        minimise.  nan / inf values are treated as invalid.
    bounds:
        Sequence of ``(lower, upper)`` pairs, one per dimension.
    initial_guesses:
        Optional list of 1-D array-like starting points.  Each is clipped
        to the bounds before use.
    grid_shape:
        Optional tuple of ints ``(n1, n2, …)``.  A uniform grid with
        ``n_i`` points along each axis is evaluated; the ``_DEFAULT_GRID_TOP_K``
        best finite-valued grid points are added as extra starting guesses.
    method:
        scipy minimize method.  Any method that does not natively support
        bounds is silently replaced by ``"L-BFGS-B"``.
    options:
        Passed verbatim to ``scipy.optimize.minimize`` as ``options``.

    Returns
    -------
    MinimizationResult with the best solution found.  ``success`` is True
    only if at least one scipy run reported success and returned a finite
    value.  If every attempt fails, the best finite point seen during grid
    evaluation (or the domain midpoint) is returned with ``success=False``.
    """
    bounds = list(bounds)
    ndim = len(bounds)
    lower = np.array([lo for lo, _ in bounds], dtype=float)
    upper = np.array([hi for _, hi in bounds], dtype=float)

    # Fall back to a bounds-aware method if the requested one does not support bounds.
    actual_method = method if method in _BOUNDS_SUPPORTING_METHODS else "L-BFGS-B"

    # ------------------------------------------------------------------ #
    # Collect starting points                                              #
    # ------------------------------------------------------------------ #
    starts: list[np.ndarray] = []

    if initial_guesses is not None:
        for guess in initial_guesses:
            starts.append(np.clip(np.asarray(guess, dtype=float), lower, upper))

    if grid_shape is not None:
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, grid_shape)]
        grid_pts = np.array(list(itertools.product(*axes)))
        grid_vals: list[tuple[float, np.ndarray]] = []
        for pt in grid_pts:
            try:
                val = float(objective(pt))
            except Exception:
                val = np.inf
            if np.isfinite(val):
                grid_vals.append((val, pt.copy()))
        grid_vals.sort(key=lambda t: t[0])
        for _, pt in grid_vals[:_DEFAULT_GRID_TOP_K]:
            starts.append(pt)

    # Fallback: domain midpoint if nothing else was provided.
    if not starts:
        starts.append((lower + upper) / 2.0)
    else:
        unique_starts: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for start in starts:
            key = tuple(np.round(start, decimals=10))
            if key in seen:
                continue
            seen.add(key)
            unique_starts.append(start)
        starts = unique_starts

    # ------------------------------------------------------------------ #
    # Local minimization from every starting point                        #
    # ------------------------------------------------------------------ #
    best_x: np.ndarray = starts[0]
    best_fun: float = np.inf
    best_success: bool = False
    best_message: str = "No minimization attempted."
    n_attempts = 0
    n_successful = 0

    for start in starts:
        n_attempts += 1
        try:
            result = minimize(
                objective,
                start,
                method=actual_method,
                bounds=bounds,
                options=options,
            )
        except Exception as exc:
            best_message = str(exc)
            continue

        fun = float(result.fun) if np.isfinite(result.fun) else np.inf
        if not np.isfinite(fun):
            continue

        if result.success:
            n_successful += 1

        if fun < best_fun:
            best_fun = fun
            best_x = np.asarray(result.x, dtype=float)
            best_success = bool(result.success)
            best_message = getattr(result, "message", str(getattr(result, "status", "")))

    if best_fun == np.inf:
        # No finite result from any local run — fall back to best grid point.
        if grid_shape is not None and grid_vals:
            best_fun, best_x = grid_vals[0]
            best_message = "Local minimization failed; returning best grid point."
        else:
            best_message = "No finite value found anywhere in the domain."

    return MinimizationResult(
        x=best_x,
        fun=best_fun,
        success=best_success,
        message=best_message,
        n_attempts=n_attempts,
        n_successful=n_successful,
    )


# --------------------------------------------------------------------------- #
# Self-test                                                                     #
# --------------------------------------------------------------------------- #

def _run_self_test() -> None:
    """Minimise f(x,y) = (x-2)^2 + (y+1)^2 on [0,5]×[-3,3].  Expected: (2, -1)."""
    result = find_global_minimum(
        lambda x: (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2,
        bounds=[(0.0, 5.0), (-3.0, 3.0)],
        grid_shape=(10, 10),
    )
    tol = 1.0e-4
    assert result.success, f"Self-test: minimization did not converge. message={result.message!r}"
    assert abs(result.x[0] - 2.0) < tol, f"Self-test: x[0]={result.x[0]:.6f}, expected 2.0"
    assert abs(result.x[1] + 1.0) < tol, f"Self-test: x[1]={result.x[1]:.6f}, expected -1.0"
    assert abs(result.fun) < tol, f"Self-test: f={result.fun:.6e}, expected 0.0"
    print(
        f"Self-test passed: x=({result.x[0]:.6f}, {result.x[1]:.6f})  "
        f"f={result.fun:.2e}  "
        f"attempts={result.n_attempts}  successful={result.n_successful}"
    )


if __name__ == "__main__":
    _run_self_test()
