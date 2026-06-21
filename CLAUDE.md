# edubot_demos — Claude guidelines

Collection of standalone ROS 2 demo packages for EduBot: `ros2_color_follower`,
`ros2_person_follower`, `ros2_yolo_stream`. Seeded into the user workspace by the
dev container in `edubot_dashboard`; not part of the core ROS 2 stack.

The conventions match the EduBot reference repo `edubot_hardware`, with one
difference: **these are demos, so there is no colcon build in CI.**

## Language

Everything is written in **English** — code, comments, docstrings, READMEs,
commit messages, config comments, identifiers. This holds even when a chat with
the maintainer is in another language.

## Naming: OmniBot → EduBot

The project was formerly called **OmniBot**; it is now **EduBot**. Always use
`EduBot`/`edubot`. Fix any `OmniBot`/`omnibot` leftovers.

## Commits

All commits MUST follow the [Conventional Commits](https://www.conventionalcommits.org) spec.
Enforced by the `commitizen` commit-msg hook.

    <type>(<optional scope>): <short summary>

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
Imperative, lower case, no trailing period, summary < ~72 chars. The scope names
the affected demo package, e.g. `feat(color_follower): add strafe mode`.

## Metadata

- Maintainer / contact: **Vectoral**, **info@vectoral.ch** (in every
  `package.xml` and `setup.py`). Author attribution may stay personal.
- License: **PolyForm Perimeter 1.0.0** — keep manifests, setup.py and README consistent.

## Development environment

Dev tooling is managed with **uv** at the repo root; it covers all three
packages. The ROS build itself stays `colcon`/`ament_python` per package.

```bash
uv sync                                         # create .venv with dev tools
uv run pre-commit install --install-hooks       # git hooks (once per clone)
uv run pre-commit install --hook-type commit-msg
```

The dev env pins Python **3.10** (Humble) via `.python-version`. A machine
without ROS can run ruff, pytest and commitizen, but not anything importing
`rclpy`/`cv2`/`cv_bridge` (the nodes) — those run on the robot or in a container.

## Linting, formatting, tests

Before every push (pre-commit does 1–2 automatically):

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest                 # control-law tests (bare; colcon test-compatible)
uv run pytest --cov=ros2_color_follower.control --cov=ros2_person_follower.control \
              --cov-report=term-missing
```

- ruff rule sets: `E,F,W,I,B,UP,SIM,RUF`, `ignore = ["E501"]`, line length 99.
- The vision/inference code (`*_node.py`, OpenCV/DNN) is not unit-tested. The
  pure control law is extracted into each package's `control.py` and tested
  under `<package>/test/`.

## Versioning & releases

`commitizen` derives the next version from the commit history and bumps it in all
three packages' `package.xml` + `setup.py`, and updates `CHANGELOG.md`:

```bash
uv run cz bump
```

## Architecture note

Each follower node keeps perception (OpenCV/DNN) and ROS plumbing in
`*_node.py`, and the pure velocity control law in `control.py`
(`compute_target_velocity`, `exponential_smooth`). Keep new pure logic in
`control.py` so it stays testable.
