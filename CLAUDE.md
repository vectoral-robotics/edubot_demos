# edubot_demos — Claude guidelines

Collection of standalone ROS2 demo packages for EduBot (color follower, person
follower, YOLO stream). Seeded into the user workspace by the dev container in
`edubot_dashboard`; not part of the core ROS2 stack.

These guidelines will grow over time. For now the most important rule:

## Commits

All commits MUST follow the [Conventional Commits](https://www.conventionalcommits.org) spec.

Format:

    <type>(<optional scope>): <short summary>

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

- Imperative mood ("add", not "added").
- Summary under ~72 characters, lower case, no trailing period.
- Scope names the affected demo package (e.g. `color_follower`).

Example:

    feat(color_follower): add omnidirectional strafe mode
