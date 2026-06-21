"""Pure control law for the color follower.

Kept free of ROS and OpenCV so it can be unit-tested without hardware. The node
delegates its velocity computation here; the math matches the original inline
control loop exactly.
"""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_target_velocity(
    error_x: float,
    radius_ratio: float,
    target_radius_ratio: float,
    *,
    omni: bool,
    dead_zone: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> tuple[float, float, float]:
    """Map the tracked ball offset/size to body velocities ``(vx, vy, wz)``.

    Args:
        error_x: horizontal ball offset in ``[-0.5, 0.5]`` (0 = centered).
        radius_ratio: measured ball diameter / frame width.
        target_radius_ratio: desired ball diameter / frame width.
        omni: True for omnidirectional (strafe) control, False for differential.
        dead_zone: ``|error_x|`` below this produces no rotation/strafe.
        max_linear_speed: linear speed clamp [m/s].
        max_angular_speed: angular speed clamp [rad/s].
    """
    vx = 0.0
    vy = 0.0
    wz = 0.0

    if omni:
        # Omnidirectional: strafe sideways + gentle rotation.
        if abs(error_x) > dead_zone:
            vy = -error_x * 2.0 * max_linear_speed
            wz = -error_x * 0.8 * max_angular_speed
    else:
        # Differential: rotate to center the ball.
        if abs(error_x) > dead_zone:
            wz = -error_x * 2.0 * max_angular_speed

    # Forward/backward: drive based on apparent ball size (quadratic response).
    size_error = target_radius_ratio - radius_ratio
    if abs(size_error) > 0.02:
        sign = 1.0 if size_error > 0 else -1.0
        vx = _clamp(
            sign * (size_error**2) * 30.0 * max_linear_speed + size_error * 1.5 * max_linear_speed,
            -max_linear_speed,
            max_linear_speed,
        )

    return vx, vy, wz


def exponential_smooth(
    previous: float, target: float, alpha: float, *, deadband: float = 0.01
) -> float:
    """Exponentially smooth ``previous`` toward ``target``; snap tiny values to 0."""
    smoothed = alpha * target + (1.0 - alpha) * previous
    return 0.0 if abs(smoothed) < deadband else smoothed
