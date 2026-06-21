"""Unit tests for the person follower control law (pure, no ROS/OpenCV)."""

import pytest

from ros2_person_follower.control import compute_target_velocity, exponential_smooth

DEAD_ZONE = 0.05
MAX_LIN = 0.3
MAX_ANG = 0.8


def _velocity(error_x, size_ratio, target_box_ratio):
    return compute_target_velocity(
        error_x,
        size_ratio,
        target_box_ratio,
        dead_zone=DEAD_ZONE,
        max_linear_speed=MAX_LIN,
        max_angular_speed=MAX_ANG,
    )


def test_centered_on_target_is_stationary():
    assert _velocity(0.0, 0.30, 0.30) == (0.0, 0.0)


def test_error_within_dead_zone_does_not_rotate():
    vx, wz = _velocity(0.04, 0.30, 0.30)
    assert (vx, wz) == (0.0, 0.0)


def test_rotates_toward_person():
    vx, wz = _velocity(0.2, 0.30, 0.30)
    assert wz == pytest.approx(-0.2 * 2.0 * MAX_ANG)
    assert vx == 0.0


def test_drives_forward_when_person_too_small():
    vx, _ = _velocity(0.0, 0.0, 0.30)
    assert vx == pytest.approx(0.30 * 2.0 * MAX_LIN)
    assert 0.0 < vx <= MAX_LIN


def test_linear_speed_is_clamped():
    vx, _ = _velocity(0.0, 0.0, 0.9)
    assert vx == pytest.approx(MAX_LIN)


def test_backs_up_when_person_too_close():
    vx, _ = _velocity(0.0, 0.8, 0.30)
    assert vx == pytest.approx(-MAX_LIN)


def test_small_size_error_is_ignored():
    # |size_error| <= 0.03 must not move forward/back.
    vx, _ = _velocity(0.0, 0.29, 0.30)
    assert vx == 0.0


def test_exponential_smooth_snaps_small_values_to_zero():
    assert exponential_smooth(0.0, 0.01, 0.25) == 0.0
    assert exponential_smooth(0.0, 1.0, 0.25) == pytest.approx(0.25)
