#!/usr/bin/env python3
"""
Gerver's Sofa Problem — Brute-Force Python Implementation

The Moving Sofa Problem asks: what is the largest area of a 2-D shape that
can be slid around a right-angle corner in a hallway of unit width?

This module:
  1. Represents a candidate sofa as a discrete set of (x, y) grid points
     satisfying  0 < x < max_width  and  0 < y < max_width.
  2. Simulates the sofa moving through an L-shaped corridor by rotating it
     from 0 to π/2 and, at each angle, searching for a valid translation
     that keeps every point inside the corridor.
  3. Uses the *rotating-hallway intersection* method to find the maximal
     sofa shape: starting from all grid points, iteratively removes any
     point that makes it impossible to position the sofa in the corridor
     at some rotation angle.
  4. Reports the approximate maximum area and produces a visualisation.

L-shaped corridor definition (unit width W = 1):
  • Horizontal hallway : all x,    0 ≤ y ≤ W
  • Vertical   hallway : 0 ≤ x ≤ W,  y ≥ 0
  A point (p, q) is *inside* the corridor when
      (0 ≤ q ≤ W)  OR  (0 ≤ p ≤ W  AND  q ≥ 0)
  and *outside* (inner obstacle) when  p < 0  AND  q > W.

Gerver's theoretical maximum area ≈ 2.2195.
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Corridor geometry
# ---------------------------------------------------------------------------
CORRIDOR_WIDTH: float = 1.0


def is_in_corridor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a boolean array: True where each (x[i], y[i]) is in the L-corridor.

    The L-corridor is the union of:
      • Horizontal hallway: any x,   0 ≤ y ≤ CORRIDOR_WIDTH
      • Vertical   hallway: 0 ≤ x ≤ CORRIDOR_WIDTH,  y ≥ 0

    Points with x < 0 and y > CORRIDOR_WIDTH are in the inner obstacle and
    are *not* in the corridor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    in_horizontal = (y >= 0.0) & (y <= CORRIDOR_WIDTH)
    in_vertical = (x >= 0.0) & (x <= CORRIDOR_WIDTH) & (y >= 0.0)
    return in_horizontal | in_vertical


# ---------------------------------------------------------------------------
# Translation feasibility — sorted-sweep algorithm
# ---------------------------------------------------------------------------

def find_feasible_translation(
    rx: np.ndarray,
    ry: np.ndarray,
) -> tuple[bool, float, float]:
    """Decide whether a translation (tx, ty) exists that places *every*
    rotated point inside the L-corridor.

    Algorithm (O(n log n))
    ----------------------
    Sort points by ry descending and try each split k = 0 … n where the
    top-k points are placed in the **vertical** hallway and the remaining
    n−k points are placed in the **horizontal** hallway.

    For a given split k the constraints are:

    * Floor:  every ry_i + ty ≥ 0  ⟹  ty ≥ −min(ry)
    * Horizontal (low) points:  ry_{(k)} − min(ry) ≤ W
      (the n−k lowest points span at most W in the vertical direction)
    * Vertical   (high) points: max(rx_{top-k}) − min(rx_{top-k}) ≤ W
      (the top-k points span at most W in the horizontal direction)

    The first split k that satisfies all three gives a feasible (tx, ty).

    Returns
    -------
    (feasible, tx, ty)
    """
    W = CORRIDOR_WIDTH
    n = len(rx)
    if n == 0:
        return True, 0.0, 0.0

    # Sort by ry descending; ry_s[0] = max ry, ry_s[-1] = min ry
    idx = np.argsort(ry)[::-1]
    ry_s = ry[idx]
    rx_s = rx[idx]
    ry_min = float(ry_s[-1])
    ty_floor = -ry_min  # minimum ty so that all ry + ty ≥ 0

    rx_max_k = -np.inf
    rx_min_k = np.inf

    for k in range(n + 1):
        if k > 0:
            rx_val = float(rx_s[k - 1])
            if rx_val > rx_max_k:
                rx_max_k = rx_val
            if rx_val < rx_min_k:
                rx_min_k = rx_val

        if k == 0:
            # All points in horizontal hallway: need ry-extent ≤ W
            if float(ry_s[0]) - ry_min <= W + 1e-9:
                return True, 0.0, ty_floor

        elif k < n:
            # Low points (indices k … n-1) must fit in horizontal hallway.
            # Their extent: ry_s[k] (max of the low group) − ry_min ≤ W
            if float(ry_s[k]) - ry_min > W + 1e-9:
                continue  # low group too tall; try more high points

            # High points (indices 0 … k-1) must fit in vertical hallway.
            if rx_max_k - rx_min_k <= W + 1e-9:
                # ty: just above the boundary so the k-th point is high
                ty = max(W - float(ry_s[k - 1]) + 1e-9, ty_floor)
                tx = 0.5 * W - 0.5 * (rx_max_k + rx_min_k)
                return True, tx, ty

        else:  # k == n: all points in vertical hallway
            if rx_max_k - rx_min_k <= W + 1e-9:
                ty = W + 1e-9 + ty_floor  # lift all points above W
                tx = 0.5 * W - 0.5 * (rx_max_k + rx_min_k)
                return True, tx, ty

    return False, 0.0, 0.0


# ---------------------------------------------------------------------------
# Maximum-coverage computation for a single angle
# ---------------------------------------------------------------------------

def _max_coverage_mask(rx: np.ndarray, ry: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return the boolean mask of the **maximum** subset of points that can
    be placed inside the L-corridor with a single translation (tx, ty).

    When the full point set is infeasible this function finds the largest
    feasible subset by:

    1. Sweeping over all "interesting" ty values (one per point crossing the
       horizontal ceiling at W).
    2. For each ty: all "low" points (ry + ty ∈ [0, W]) are automatically
       covered; for "high" points (ry + ty > W) we use a sliding-window scan
       over their rx values to find the window of width W containing the most
       high points.
    3. The ty giving the highest total count wins.

    Returns
    -------
    (mask, tx, ty)
        mask – boolean array, True for each kept point.
    """
    W = CORRIDOR_WIDTH
    n = len(rx)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0

    ty_floor = -float(np.min(ry))

    # Interesting ty values: ty_floor, plus one for each point "just becoming high"
    ty_candidates: list[float] = [ty_floor]
    for ry_i in ry:
        t = W - float(ry_i) + 1e-10
        if t >= ty_floor - 1e-12:
            ty_candidates.append(t)

    best_count = -1
    best_mask = np.zeros(n, dtype=bool)
    best_tx = 0.0
    best_ty = ty_floor

    for ty in ty_candidates:
        ry_shifted = ry + ty
        if np.any(ry_shifted < -1e-9):
            continue

        low = (ry_shifted >= -1e-9) & (ry_shifted <= W + 1e-9)
        high = ry_shifted > W + 1e-9
        n_low = int(np.sum(low))

        if not np.any(high):
            count = n_low
            if count > best_count:
                best_count = count
                best_mask = low.copy()
                best_tx, best_ty = 0.0, ty
            continue

        # Sliding-window over rx values of high points
        high_idx = np.where(high)[0]
        rx_h = rx[high_idx]
        order = np.argsort(rx_h)
        rx_sorted = rx_h[order]
        m = len(rx_sorted)

        j = 0
        bj, bi = 0, 0
        max_win = 0
        for i in range(m):
            while rx_sorted[i] - rx_sorted[j] > W + 1e-9:
                j += 1
            win = i - j + 1
            if win > max_win:
                max_win = win
                bj, bi = j, i

        count = n_low + max_win
        if count > best_count:
            best_count = count
            best_ty = ty
            best_tx = 0.5 * W - 0.5 * (rx_sorted[bj] + rx_sorted[bi])
            win_local = np.zeros(m, dtype=bool)
            win_local[order[bj : bi + 1]] = True
            mask = low.copy()
            mask[high_idx[win_local]] = True
            best_mask = mask

    return best_mask, best_tx, best_ty


# ---------------------------------------------------------------------------
# Single-shape feasibility check
# ---------------------------------------------------------------------------

def sofa_can_pass(
    points: np.ndarray,
    num_angles: int = 90,
) -> bool:
    """Return True if the sofa (given as an (N, 2) array of points) can
    navigate the L-shaped corner by rotating from 0 to π/2.

    At each discrete rotation angle θ the sofa is rotated by θ and then
    ``find_feasible_translation`` is called to decide whether a valid
    placement inside the corridor exists.

    Parameters
    ----------
    points :
        (N, 2) array of sofa point coordinates.  All coordinates must satisfy
        x > 0 and y > 0.
    num_angles :
        Number of equally-spaced rotation angles in [0, π/2] to check.

    Returns
    -------
    bool
        True only if the sofa fits in the corridor at *every* tested angle.
    """
    if len(points) == 0:
        return False

    for theta in np.linspace(0.0, np.pi / 2.0, num_angles + 1):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rx = points[:, 0] * cos_t - points[:, 1] * sin_t
        ry = points[:, 0] * sin_t + points[:, 1] * cos_t
        feasible, _, _ = find_feasible_translation(rx, ry)
        if not feasible:
            return False
    return True


# ---------------------------------------------------------------------------
# Rotating-hallway intersection — maximal sofa finder
# ---------------------------------------------------------------------------

def rotating_hallway_sofa(
    max_width: float = 2.0,
    resolution: int = 30,
    num_angles: int = 90,
) -> tuple[float, np.ndarray]:
    """Find the approximate maximal sofa using the rotating-hallway method.

    Algorithm
    ---------
    1. Build a grid of candidate points inside (0, max_width)².
    2. Repeat until convergence (the valid set no longer changes):
       For each rotation angle θ ∈ [0, π/2]:
         a. Rotate all currently-valid points by θ.
         b. Call ``find_feasible_translation``.  If feasible, every valid
            point survives this angle.
         c. If infeasible, call ``_max_coverage_mask`` to find the largest
            subset that *does* fit at θ and mark the rest as invalid.
    3. The surviving grid points are the approximate maximal sofa.

    Using ``_max_coverage_mask`` (rather than one-point-at-a-time greedy
    removal) ensures that at each angle we retain the maximum possible
    coverage, and iterating to convergence handles the coupling between
    different angles.

    Parameters
    ----------
    max_width :
        Grid spans (0, max_width) in both x and y.
    resolution :
        Number of grid cells per unit length; total grid size is roughly
        (max_width * resolution)².
    num_angles :
        Number of rotation angles in [0, π/2].

    Returns
    -------
    (area, sofa_points)
        area        – approximate sofa area (grid-point count × cell area).
        sofa_points – (M, 2) array of surviving grid point coordinates.
    """
    n = int(max_width * resolution)
    dx = max_width / n

    # Interior grid points: 0 < x < max_width, 0 < y < max_width
    coords = np.arange(1, n) * dx
    XX, YY = np.meshgrid(coords, coords)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])  # (N, 2)

    valid = np.ones(len(all_pts), dtype=bool)
    angles = np.linspace(0.0, np.pi / 2.0, num_angles + 1)

    # Iterate until the valid set converges
    for _iteration in range(20):
        prev_count = int(np.sum(valid))

        for theta in angles:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            vpts = all_pts[valid]
            if len(vpts) == 0:
                break

            rx = vpts[:, 0] * cos_t - vpts[:, 1] * sin_t
            ry = vpts[:, 0] * sin_t + vpts[:, 1] * cos_t

            feasible, _, _ = find_feasible_translation(rx, ry)
            if feasible:
                continue  # all current points survive this angle

            # Find the maximum feasible subset at this angle and
            # invalidate points outside it.
            keep_local, _, _ = _max_coverage_mask(rx, ry)
            valid_idx = np.where(valid)[0]
            valid[valid_idx[~keep_local]] = False

        if int(np.sum(valid)) == prev_count:
            break  # converged — no points removed in this pass

    sofa_points = all_pts[valid]
    area = float(len(sofa_points)) * dx * dx
    return area, sofa_points


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_corridor(ax: "plt.Axes") -> None:
    """Draw the L-shaped corridor outline on *ax*."""
    W = CORRIDOR_WIDTH
    corridor_verts = [
        (-2.5, 0.0), (W, 0.0), (W, 3.0),
        (0.0, 3.0), (0.0, W), (-2.5, W),
    ]
    ax.add_patch(plt.Polygon(
        corridor_verts, closed=True,
        facecolor="lightyellow", edgecolor="black", linewidth=2, zorder=1,
    ))
    ax.plot(0.0, W, "ro", markersize=5, label="inner corner", zorder=3)


def visualize_sofa(
    sofa_points: np.ndarray,
    max_width: float,
    area: float,
    save_path: str = "sofa_result.png",
) -> None:
    """Produce a three-panel figure.

    • Left  : the computed sofa shape (grid points in sofa frame).
    • Centre: the sofa placed in the horizontal hallway (θ = 0).
    • Right : the sofa placed in the vertical hallway (θ = π/2).

    The figure is saved to *save_path* and also displayed interactively.
    """
    W = CORRIDOR_WIDTH
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- Left panel: sofa shape in its own frame -----------------------
    ax1 = axes[0]
    if len(sofa_points) > 0:
        ax1.scatter(
            sofa_points[:, 0], sofa_points[:, 1],
            c="steelblue", s=4, alpha=0.7, linewidths=0,
        )
    ax1.set_xlim(0, max_width)
    ax1.set_ylim(0, max_width)
    ax1.set_aspect("equal")
    ax1.set_title(f"Sofa shape (sofa frame)\nArea ≈ {area:.4f}  (Gerver ≈ 2.2195)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)

    # Helper: place sofa into corridor at angle theta and draw it
    def _draw_sofa_in_corridor(ax: "plt.Axes", theta: float, title: str) -> None:
        _draw_corridor(ax)
        if len(sofa_points) == 0:
            return
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rx = sofa_points[:, 0] * cos_t - sofa_points[:, 1] * sin_t
        ry = sofa_points[:, 0] * sin_t + sofa_points[:, 1] * cos_t
        _, tx, ty = find_feasible_translation(rx, ry)
        ax.scatter(
            rx + tx, ry + ty,
            c="steelblue", s=4, alpha=0.7, linewidths=0, zorder=2,
            label="sofa",
        )
        ax.set_xlim(-2.5, 2.0)
        ax.set_ylim(-0.5, 3.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (corridor)")
        ax.set_ylabel("y (corridor)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # ---- Centre: sofa in corridor at θ = 0 (horizontal hallway) --------
    _draw_sofa_in_corridor(axes[1], 0.0,
                           "Sofa in corridor  θ = 0°\n(entering horizontal hallway)")

    # ---- Right: sofa in corridor at θ = π/2 (vertical hallway) ---------
    _draw_sofa_in_corridor(axes[2], np.pi / 2,
                           "Sofa in corridor  θ = 90°\n(exiting into vertical hallway)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Figure saved to '{save_path}'.")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Brute-force approximation of the maximum sofa area "
            "(Gerver's Sofa Problem)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-width", type=float, default=3.0,
        help="Upper bound for x and y coordinates of sofa grid points.",
    )
    p.add_argument(
        "--resolution", type=int, default=30,
        help="Grid points per unit length (higher → finer grid, slower).",
    )
    p.add_argument(
        "--num-angles", type=int, default=90,
        help="Number of rotation angles sampled in [0, π/2].",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip the matplotlib visualisation.",
    )
    p.add_argument(
        "--save", type=str, default="sofa_result.png",
        help="File path for the output figure.",
    )
    return p


def main(argv=None) -> int:
    """Entry point for CLI usage: ``python sofa.py [options]``."""
    args = _build_parser().parse_args(argv)

    grid_n = int(args.max_width * args.resolution)
    print("=" * 60)
    print("Gerver's Sofa Problem — Brute-Force Python Solution")
    print("=" * 60)
    print(f"  max_width  : {args.max_width}")
    print(f"  resolution : {args.resolution} pts/unit  →  grid ≈ {grid_n}×{grid_n}")
    print(f"  num_angles : {args.num_angles + 1}")
    print()

    area, sofa_pts = rotating_hallway_sofa(
        max_width=args.max_width,
        resolution=args.resolution,
        num_angles=args.num_angles,
    )

    print(f"  Sofa grid points : {len(sofa_pts)}")
    print(f"  Computed area    : {area:.6f}")
    print(f"  Gerver's limit   : ~2.2195")
    print(f"  Ratio            : {area / 2.2195:.4f}")

    # Verify the result: the computed sofa must be able to navigate the corner.
    verified = sofa_can_pass(sofa_pts, num_angles=args.num_angles)
    print(f"  Can pass corner  : {verified}")
    print()

    if not args.no_plot:
        visualize_sofa(sofa_pts, args.max_width, area, save_path=args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
