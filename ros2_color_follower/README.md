# ros2_color_follower

A lightweight ROS 2 demo that makes the robot follow a colored ball using
hybrid HSV color detection + CamShift tracking. No neural network required —
runs at full frame rate with minimal CPU load.

## How it works

1. **HSV Detection** – Finds the ball by color range + circularity filter.
2. **CamShift Tracking** – Builds a hue histogram of the ball and tracks it
   frame-to-frame via back-projection. Adapts to lighting changes automatically.
3. **Re-verification** – Periodically re-checks with full HSV detection to
   prevent drift. Resets if tracking fails.
4. **Smooth control** – 20 Hz control loop with exponential smoothing for
   fluid robot motion.

## Build

```bash
cd /workspace
colcon build --packages-select ros2_color_follower
source install/setup.bash
```

## Run

**Differential drive (default):**
```bash
ros2 launch ros2_color_follower color_follower.launch.py
```

**Omnidirectional drive:**
```bash
ros2 launch ros2_color_follower color_follower.launch.py omni:=true
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `input_topic` | `/image` | Camera image topic |
| `output_topic` | `/follower/image` | Annotated output image topic |
| `cmd_vel_topic` | `/cmd_vel` | Velocity command topic |
| `omni` | `false` | Enable omnidirectional mode (strafing) |
| `hsv_low_h` | `25` | HSV lower hue bound |
| `hsv_low_s` | `40` | HSV lower saturation bound |
| `hsv_low_v` | `40` | HSV lower value bound |
| `hsv_high_h` | `95` | HSV upper hue bound |
| `hsv_high_s` | `255` | HSV upper saturation bound |
| `hsv_high_v` | `255` | HSV upper value bound |
| `min_area` | `200` | Minimum contour area (in downscaled image) |
| `min_circularity` | `0.40` | Minimum circularity to accept as ball |
| `max_linear_speed` | `0.3` | Maximum forward/backward speed (m/s) |
| `max_angular_speed` | `1.0` | Maximum rotation speed (rad/s) |
| `target_radius_ratio` | `0.12` | Target ball size relative to frame width |
| `dead_zone` | `0.04` | Ignore small positional errors |
| `smoothing` | `0.25` | Smoothing factor (0=sluggish, 1=instant) |
| `control_hz` | `20.0` | Control loop frequency |
| `video_hz` | `15.0` | Annotated video output frequency |
| `lost_timeout_sec` | `1.5` | Seconds before declaring ball lost |

## Tuning HSV for your ball color

The default HSV range covers bright/lime green. To tune for a different color:

```bash
# Example: track a red ball
ros2 launch ros2_color_follower color_follower.launch.py \
  hsv_low_h:=0 hsv_high_h:=10 hsv_low_s:=100 hsv_low_v:=100

# Example: track an orange ball
ros2 launch ros2_color_follower color_follower.launch.py \
  hsv_low_h:=10 hsv_high_h:=25 hsv_low_s:=100 hsv_low_v:=100
```

Use the `/follower/image` topic in the dashboard to see what the tracker sees.
