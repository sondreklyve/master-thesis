"""Maxwell equal-pressure construction for first-order phase transitions."""

from __future__ import annotations

import numpy as np


def _arr_slice(array: np.ndarray, index_list: list[int]) -> np.ndarray:
    if not index_list:
        return array
    if len(index_list) == 1:
        return array[index_list[0]:]
    return np.concatenate((array[: index_list[0] + 1], array[index_list[1]:]))


def maxwell_construct(
    mu_q: np.ndarray,
    pressure: np.ndarray,
    energy_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Equal-pressure Maxwell construction along the mu_q axis.

    Finds the pressure loop in P(mu_q): locates mu_1 < mu_2 where
    P(mu_1) = P(mu_2) = P_transition, then replaces all points with
    mu_1 < mu_q < mu_2 with a single discontinuous jump at P_transition.

    Returns
    -------
    stable_p, stable_e : arrays with the unstable loop removed
    maxwell_indices    : [i1, i2] indices of the transition endpoints in the
                         *input* arrays (empty list if no transition found)
    """
    mu_q = np.asarray(mu_q, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    energy_density = np.asarray(energy_density, dtype=float)

    dp = np.gradient(pressure, mu_q)

    # Find indices where dP/dmu_q changes sign (pressure loop)
    sign_changes = []
    for i in range(1, len(dp)):
        if dp[i - 1] * dp[i] < 0.0:
            sign_changes.append(i)

    if len(sign_changes) < 2:
        # No complete loop — return as-is
        return pressure, energy_density, []

    # The loop spans from the first local maximum to the first local minimum
    # after it.  sign_changes[0] is the falling edge (max), sign_changes[1]
    # is the rising edge (min).
    i_max = sign_changes[0]  # dP/dmu changes from + to -
    i_min = sign_changes[1]  # dP/dmu changes from - to +

    # P at the peak and trough of the loop
    p_peak  = float(pressure[i_max])
    p_trough = float(pressure[i_min])

    # Equal-pressure transition: find P_t in [p_trough, p_peak] where the
    # "area" above and below the flat line cancels.  Equivalently, we seek
    # P_t such that mu_q(P_t, upper branch) = mu_q(P_t, lower branch) when
    # integrating around the loop — i.e., the standard Maxwell criterion
    # reduces to finding where the two branches of mu_q(P) intersect.
    #
    # Since we have mu_q(P) on the rising branch [0..i_max] and again on the
    # falling branch [i_min..end], the transition pressure is where the
    # pre-loop and post-loop P values are equal: we scan the pre-loop
    # pressure array and match against the post-loop pressure array.

    p_pre  = pressure[:i_max + 1]   # monotone rising up to peak
    p_post = pressure[i_min:]        # monotone rising after trough

    # Intersection: largest P in pre-loop that also appears in post-loop.
    # Find the crossover: last index in pre where p_pre <= p_post[0] gives
    # a lower bound; the intersection is where p_pre == some p_post value.
    # Use linear interpolation to find exact crossing.

    # For each point in p_pre (above p_trough), find whether the same
    # pressure exists in p_post.  The transition pressure P_t satisfies:
    # there exist i1 in pre-loop and i2 in post-loop with P(i1) == P(i2).
    # We want the *largest* such pressure in [p_trough, p_peak].

    # Valid search range: [max(p_post[0], 0), p_peak]
    p_lo = max(float(p_post[0]), 0.0)
    p_hi = float(p_pre[-1])  # == p_peak

    if p_lo >= p_hi:
        # Degenerate: transition at the boundary
        p_transition = p_lo
    else:
        # Walk p_pre from high to low; find first value also in p_post range
        # i.e., p_pre[j] >= p_post[0].  The boundary crossing is the transition.
        # Use bisection: find P_t such that
        #   mu_q_pre(P_t)  ==  mu_q_post(P_t)
        # where mu_q_pre and mu_q_post are the pre- and post-loop branches.

        # mu_q as function of P on pre-loop branch (monotone in P)
        sort_pre  = np.argsort(p_pre)
        sort_post = np.argsort(p_post)
        p_pre_s   = p_pre[sort_pre]
        mu_pre_s  = mu_q[:i_max + 1][sort_pre]
        p_post_s  = p_post[sort_post]
        mu_post_s = mu_q[i_min:][sort_post]

        # Restrict to the overlapping P range
        p_lo_common = max(float(p_pre_s[0]),  float(p_post_s[0]))
        p_hi_common = min(float(p_pre_s[-1]), float(p_post_s[-1]))

        if p_lo_common >= p_hi_common:
            p_transition = p_lo_common
        else:
            # Evaluate mu_pre(P) - mu_post(P) over a fine grid and find zero
            n_grid = 500
            p_grid = np.linspace(p_lo_common, p_hi_common, n_grid)
            mu_pre_grid  = np.interp(p_grid, p_pre_s,  mu_pre_s)
            mu_post_grid = np.interp(p_grid, p_post_s, mu_post_s)
            diff = mu_pre_grid - mu_post_grid  # should cross zero

            # Find sign change in diff
            crossings = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
            if crossings.size == 0:
                # No crossing found — use midpoint as fallback
                p_transition = 0.5 * (p_lo_common + p_hi_common)
            else:
                # Linear interpolation to the first crossing
                j = int(crossings[-1])  # use last crossing (stable branch)
                frac = diff[j] / (diff[j] - diff[j + 1]) if diff[j] != diff[j + 1] else 0.5
                p_transition = float(p_grid[j] + frac * (p_grid[j + 1] - p_grid[j]))

    # Find i1: last index in pre-loop where P <= p_transition
    pre_mask = pressure[:i_max + 1] <= p_transition + 1.0e-10
    if not pre_mask.any():
        i1 = 0
    else:
        i1 = int(np.where(pre_mask)[0][-1])

    # Find i2: first index in post-loop where P >= p_transition
    post_mask = pressure[i_min:] >= p_transition - 1.0e-10
    if not post_mask.any():
        i2 = len(pressure) - 1
    else:
        i2 = int(np.where(post_mask)[0][0]) + i_min

    # Build stable arrays: keep [0..i1] and [i2..end]
    stable_p = np.concatenate((pressure[:i1 + 1], pressure[i2:]))
    stable_e = np.concatenate((energy_density[:i1 + 1], energy_density[i2:]))

    return stable_p, stable_e, [i1, i2]
