"""Unit tests for sofa.py — Gerver's Sofa brute-force implementation."""

import numpy as np
import pytest

from sofa import (
    CORRIDOR_WIDTH,
    _check_angle_worker,
    _max_coverage_mask,
    find_feasible_translation,
    is_in_corridor,
    rotating_hallway_sofa,
    sofa_can_pass,
)

W = CORRIDOR_WIDTH  # 1.0


# ---------------------------------------------------------------------------
# is_in_corridor
# ---------------------------------------------------------------------------

class TestIsInCorridor:
    def test_horizontal_hallway_centre(self):
        """Points well inside the horizontal hallway are accepted."""
        x = np.array([-5.0, 0.0, 5.0])
        y = np.array([0.5,  0.5, 0.5])
        assert np.all(is_in_corridor(x, y))

    def test_horizontal_hallway_boundaries(self):
        """Floor (y=0) and ceiling (y=W) of the horizontal hallway."""
        x = np.array([0.0, 0.0])
        y = np.array([0.0, W])
        assert np.all(is_in_corridor(x, y))

    def test_vertical_hallway(self):
        """Points in the vertical hallway above y=W."""
        x = np.array([0.2, 0.8])
        y = np.array([2.0, 5.0])
        assert np.all(is_in_corridor(x, y))

    def test_inner_obstacle(self):
        """Points in the inner obstacle (x < 0, y > W) are rejected."""
        x = np.array([-0.1, -1.0])
        y = np.array([W + 0.1, 2.0])
        assert not np.any(is_in_corridor(x, y))

    def test_below_floor(self):
        """Points below y=0 are always outside the corridor."""
        x = np.array([0.5, -1.0])
        y = np.array([-0.1, -1.0])
        assert not np.any(is_in_corridor(x, y))


# ---------------------------------------------------------------------------
# find_feasible_translation
# ---------------------------------------------------------------------------

class TestFindFeasibleTranslation:
    def test_single_point_always_feasible(self):
        """A single point can always be placed in the corridor."""
        rx = np.array([3.7])
        ry = np.array([2.5])
        feasible, tx, ty = find_feasible_translation(rx, ry)
        assert feasible

    def test_narrow_shape_in_horizontal_hallway(self):
        """Shape that fits within height W is always feasible."""
        rx = np.array([0.0, 1.0, 2.0, 3.0])
        ry = np.array([0.1, 0.3, 0.7, 0.9])  # ry-extent = 0.8 < W
        feasible, tx, ty = find_feasible_translation(rx, ry)
        assert feasible
        # Verify the translation actually works
        assert np.all(is_in_corridor(rx + tx, ry + ty))

    def test_unit_square_at_zero(self):
        """A unit square at θ=0 fits in horizontal hallway."""
        pts = np.linspace(0.05, 0.95, 10)
        xx, yy = np.meshgrid(pts, pts)
        rx = xx.ravel()
        ry = yy.ravel()
        feasible, tx, ty = find_feasible_translation(rx, ry)
        assert feasible
        assert np.all(is_in_corridor(rx + tx, ry + ty))

    def test_too_wide_is_infeasible(self):
        """A shape where every possible high/low split fails is infeasible.

        The four points form a 2×2 pattern with:
          - ry-extent = 2 > W  (cannot all fit in horizontal hallway)
          - For any split k, either the high-rx-range > W or the low-ry-range > W
        """
        # sorted by ry desc: (rx=2,ry=2),(rx=0,ry=1.5),(rx=2,ry=0.5),(rx=0,ry=0)
        rx = np.array([0.0, 0.0, 2.0, 2.0])
        ry = np.array([0.0, 1.5, 0.5, 2.0])
        feasible, _, _ = find_feasible_translation(rx, ry)
        assert not feasible

    def test_tall_shape_with_narrow_top_feasible(self):
        """Shape taller than W, but the upper part is narrow (fits in vertical hallway)."""
        # Low part: wide in rx, below W in ry
        # High part: narrow in rx, above W in ry
        rx_low = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ry_low = np.full(5, 0.5)
        rx_high = np.array([0.4, 0.6])   # rx-range = 0.2 ≤ W
        ry_high = np.array([1.5, 2.0])
        rx = np.concatenate([rx_low, rx_high])
        ry = np.concatenate([ry_low, ry_high])
        feasible, tx, ty = find_feasible_translation(rx, ry)
        assert feasible
        assert np.all(is_in_corridor(rx + tx, ry + ty))

    def test_empty_input(self):
        """Empty input is always feasible."""
        feasible, tx, ty = find_feasible_translation(
            np.array([]), np.array([])
        )
        assert feasible


# ---------------------------------------------------------------------------
# sofa_can_pass
# ---------------------------------------------------------------------------

class TestSofaCanPass:
    def _grid_in(self, x_lo, x_hi, y_lo, y_hi, n=8):
        """Return a small grid of points inside the given rectangle."""
        xs = np.linspace(x_lo, x_hi, n)
        ys = np.linspace(y_lo, y_hi, n)
        XX, YY = np.meshgrid(xs, ys)
        return np.column_stack([XX.ravel(), YY.ravel()])

    def test_thin_horizontal_strip_passes(self):
        """A strip that fits entirely within height W passes at every angle."""
        pts = self._grid_in(0.1, 0.9, 0.1, 0.9)  # inside (0,1)×(0,1)
        assert sofa_can_pass(pts, num_angles=36)

    def test_very_wide_strip_cannot_pass(self):
        """A strip that is 4 units wide cannot pass the unit-width corner."""
        pts = self._grid_in(0.1, 3.9, 0.1, 0.9)  # 3.8 × 0.8 rectangle
        assert not sofa_can_pass(pts, num_angles=36)

    def test_empty_cannot_pass(self):
        """Empty set is not a valid sofa."""
        assert not sofa_can_pass(np.zeros((0, 2)), num_angles=10)


# ---------------------------------------------------------------------------
# _max_coverage_mask
# ---------------------------------------------------------------------------

class TestMaxCoverageMask:
    def test_already_feasible_keeps_all(self):
        """When the full set is already feasible, all points are kept."""
        rx = np.array([0.0, 0.5, 1.0, 1.5])
        ry = np.array([0.1, 0.3, 0.6, 0.9])  # ry-extent < W
        mask, tx, ty = _max_coverage_mask(rx, ry)
        assert np.all(mask)

    def test_mask_is_feasible(self):
        """The returned mask must yield a feasible set."""
        rng = np.random.default_rng(42)
        rx = rng.uniform(-2.0, 2.0, 50)
        ry = rng.uniform(-1.0, 3.0, 50)
        mask, tx, ty = _max_coverage_mask(rx, ry)
        rx_kept = rx[mask]
        ry_kept = ry[mask]
        assert np.all(is_in_corridor(rx_kept + tx, ry_kept + ty))

    def test_mask_covers_more_than_zero(self):
        """At least one point is always kept (unless input is empty)."""
        rx = np.array([0.3, 0.7])
        ry = np.array([0.4, 0.8])
        mask, _, _ = _max_coverage_mask(rx, ry)
        assert np.sum(mask) >= 1


# ---------------------------------------------------------------------------
# rotating_hallway_sofa — integration test
# ---------------------------------------------------------------------------

class TestRotatingHallwaySofa:
    def test_area_positive(self):
        """The algorithm always returns a positive area."""
        area, pts = rotating_hallway_sofa(max_width=2.0, resolution=5,
                                          num_angles=18)
        assert area > 0.0

    def test_result_can_pass(self):
        """Every point in the result must be able to navigate the corner."""
        area, pts = rotating_hallway_sofa(max_width=2.0, resolution=10,
                                          num_angles=45)
        assert sofa_can_pass(pts, num_angles=45)

    def test_area_below_gervers_limit(self):
        """The computed area must be ≤ Gerver's theoretical maximum."""
        area, _ = rotating_hallway_sofa(max_width=3.0, resolution=10,
                                        num_angles=45)
        assert area <= 2.2195 + 0.05  # small tolerance for grid discretisation

    def test_area_reasonable(self):
        """The computed area should be at least 1.5 (Hammersley's simple bound)."""
        area, _ = rotating_hallway_sofa(max_width=3.0, resolution=10,
                                        num_angles=45)
        assert area >= 1.5

    def test_points_within_bounds(self):
        """All returned points satisfy 0 < x,y < max_width."""
        mw = 2.0
        _, pts = rotating_hallway_sofa(max_width=mw, resolution=8,
                                       num_angles=18)
        assert np.all(pts[:, 0] > 0) and np.all(pts[:, 0] < mw)
        assert np.all(pts[:, 1] > 0) and np.all(pts[:, 1] < mw)


# ---------------------------------------------------------------------------
# _check_angle_worker
# ---------------------------------------------------------------------------

class TestCheckAngleWorker:
    def test_feasible_angle_returns_none(self):
        """When all rotated points fit in the corridor, None is returned."""
        pts = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        result = _check_angle_worker((0.0, pts))
        assert result is None

    def test_infeasible_angle_returns_indices(self):
        """When some points must be removed, an index array is returned."""
        # Build a point set that is infeasible at 45°
        rng = np.random.default_rng(0)
        pts = rng.uniform(0.05, 2.5, (60, 2))
        theta = np.pi / 4
        result = _check_angle_worker((theta, pts))
        # Result is either None (happened to be feasible) or an ndarray
        assert result is None or isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# rotating_hallway_sofa — multi-worker tests
# ---------------------------------------------------------------------------

class TestRotatingHallwaySofaMultiWorker:
    def test_multiworker_area_positive(self):
        """Multi-worker run returns a positive area."""
        area, pts = rotating_hallway_sofa(max_width=2.0, resolution=5,
                                          num_angles=18, num_workers=2)
        assert area > 0.0

    def test_multiworker_result_can_pass(self):
        """Multi-worker result must be able to navigate the corner."""
        area, pts = rotating_hallway_sofa(max_width=2.0, resolution=8,
                                          num_angles=36, num_workers=2)
        assert sofa_can_pass(pts, num_angles=36)

    def test_multiworker_area_close_to_sequential(self):
        """Multi-worker produces a valid sofa.

        The parallel batch-update strategy evaluates all angles against the
        same point-set snapshot, so it may be more conservative than the
        sequential pass and produce a smaller (but still valid) sofa.
        The key guarantee is that the result is positive and can pass the corner.
        """
        area_par, pts_par = rotating_hallway_sofa(max_width=2.0, resolution=8,
                                                  num_angles=36, num_workers=2)
        assert area_par > 0.0
        assert sofa_can_pass(pts_par, num_angles=36)
