"""Pure control law for the person follower.

Kept free of ROS and OpenCV so it can be unit-tested without hardware. The node
delegates its velocity computation here; the math matches the original inline
control loop exactly.
"""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_target_velocity(
    error_x: float,
    size_ratio: float,
    target_box_ratio: float,
    *,
    dead_zone: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> tuple[float, float]:
    """Map the detected person offset/size to body velocities ``(vx, wz)``.

    Args:
        error_x: horizontal person-center offset in ``[-0.5, 0.5]`` (0 = centered).
        size_ratio: detected bounding-box width / frame width.
        target_box_ratio: desired bounding-box width / frame width.
        dead_zone: ``|error_x|`` below this produces no rotation.
        max_linear_speed: linear speed clamp [m/s].
        max_angular_speed: angular speed clamp [rad/s].
    """
    vx = 0.0
    wz = 0.0

    # Angular: steer toward the person.
    if abs(error_x) > dead_zone:
        wz = -error_x * 2.0 * max_angular_speed

    # Linear: drive based on person size in frame.
    size_error = target_box_ratio - size_ratio
    if abs(size_error) > 0.03:
        vx = _clamp(
            size_error * 2.0 * max_linear_speed,
            -max_linear_speed,
            max_linear_speed,
        )

    return vx, wz


def exponential_smooth(
    previous: float, target: float, alpha: float, *, deadband: float = 0.01
) -> float:
    """Exponentially smooth ``previous`` toward ``target``; snap tiny values to 0."""
    smoothed = alpha * target + (1.0 - alpha) * previous
    return 0.0 if abs(smoothed) < deadband else smoothed
