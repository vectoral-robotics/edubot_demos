"""Unit tests for the color follower control law (pure, no ROS/OpenCV)."""

import pytest

from ros2_color_follower.control import compute_target_velocity, exponential_smooth

DEAD_ZONE = 0.04
MAX_LIN = 0.3
MAX_ANG = 1.0


def _velocity(error_x, radius_ratio, target_radius_ratio, omni=False):
    return compute_target_velocity(
        error_x,
        radius_ratio,
        target_radius_ratio,
        omni=omni,
        dead_zone=DEAD_ZONE,
        max_linear_speed=MAX_LIN,
        max_angular_speed=MAX_ANG,
    )


def test_centered_on_target_is_stationary():
    assert _velocity(0.0, 0.12, 0.12) == (0.0, 0.0, 0.0)


def test_error_within_dead_zone_does_not_rotate():
    vx, vy, wz = _velocity(0.03, 0.12, 0.12)
    assert (vx, vy, wz) == (0.0, 0.0, 0.0)


def test_differential_rotates_toward_target():
    vx, vy, wz = _velocity(0.2, 0.12, 0.12)
    assert wz == pytest.approx(-0.2 * 2.0 * MAX_ANG)
    assert vx == 0.0 and vy == 0.0


def test_omni_strafes_and_rotates():
    vx, vy, wz = _velocity(0.2, 0.12, 0.12, omni=True)
    assert vy == pytest.approx(-0.2 * 2.0 * MAX_LIN)
    assert wz == pytest.approx(-0.2 * 0.8 * MAX_ANG)
    assert vx == 0.0


def test_drives_forward_when_ball_too_small():
    vx, _, _ = _velocity(0.0, 0.0, 0.12)
    expected = (0.12**2) * 30.0 * MAX_LIN + 0.12 * 1.5 * MAX_LIN
    assert vx == pytest.approx(expected)
    assert 0.0 < vx <= MAX_LIN


def test_linear_speed_is_clamped():
    vx, _, _ = _velocity(0.0, 0.0, 0.9)
    assert vx == pytest.approx(MAX_LIN)


def test_backs_up_when_ball_too_close():
    vx, _, _ = _velocity(0.0, 0.5, 0.12)
    assert vx == pytest.approx(-MAX_LIN)


@pytest.mark.parametrize(
    ("prev", "target", "alpha", "expected"),
    [
        (0.0, 1.0, 0.25, 0.25),
        (0.25, 1.0, 0.25, 0.4375),
        (0.0, 0.01, 0.25, 0.0),  # below dead-band -> snapped to zero
    ],
)
def test_exponential_smooth(prev, target, alpha, expected):
    assert exponential_smooth(prev, target, alpha) == pytest.approx(expected)
