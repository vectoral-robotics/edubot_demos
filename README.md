# EduBot Demos

This folder contains versioned demo templates.

When the Code Server container starts, it copies missing demo files and folders
to:

```text
/workspace/src
```

Existing content in the shared workspace is not overwritten by container
restarts or repository updates.

For ROS 2 demo packages that should be built by users, prefer putting the actual
packages directly in this folder so they are seeded into `/workspace/src`.
