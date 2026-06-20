# edubot_demos

Standalone ROS 2 demo packages for the [EduBot](https://github.com/vectoral-robotics) robot — by Vectoral.

## What it is

A collection of self-contained, education-oriented demos that show off the robot
and serve as starting points for your own packages. They are **not** part of the
core robot stack — on a deployed robot the dev container seeds them into the user
workspace (`/workspace/src`), where they can be freely edited (existing files are
never overwritten).

| Package | What it does |
|---|---|
| `ros2_color_follower` | Follows a colored ball using HSV blob tracking |
| `ros2_person_follower` | Follows a person using a lightweight detector |
| `ros2_yolo_stream` | Runs YOLO object detection on the camera stream |

Each demo subscribes to a camera image topic and publishes `/cmd_vel`, so it
drives the robot directly.

## Installation

Requires ROS 2 Humble. On a deployed EduBot the demos already appear in
`/workspace/src`. To build them yourself:

```bash
cd ~/ros2_ws/src
git clone https://github.com/vectoral-robotics/edubot_demos.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ros2_color_follower ros2_person_follower ros2_yolo_stream
source install/setup.bash
```

## Usage

Each package ships a launch file with tunable arguments:

```bash
ros2 launch ros2_color_follower color_follower.launch.py
ros2 launch ros2_person_follower person_follower.launch.py
ros2 launch ros2_yolo_stream yolo_stream.launch.py
```

See the individual launch files for arguments (HSV thresholds, speed limits,
confidence thresholds, input/output topics, …). The robot stack
([`edubot_bringup`](https://github.com/vectoral-robotics/edubot_bringup)) and a
camera must be running for the demos to have an image to act on.

## Contributing

- Work on a short-lived feature branch and open a pull request against `main`
  (which is protected); changes land via PR review.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org)
  with the demo as scope, e.g. `feat(color_follower): …`. See `CLAUDE.md`.
- These are Python packages with linting/formatting via **ruff**. Install the
  git hooks once after cloning:

  ```bash
  pip install pre-commit && pre-commit install
  ```

## License

PolyForm Perimeter 1.0.0 (source-available) — see [LICENSE](LICENSE).
