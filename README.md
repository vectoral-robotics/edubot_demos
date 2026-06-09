# OmniBot Demos

This folder contains versioned demo templates.

When the Code Server container starts, it copies missing demo files and folders
to:

```text
/workspace/demos
```

Existing content in the shared workspace is not overwritten by container
restarts or repository updates.

For ROS 2 demo packages that should be built by users, prefer putting the actual
packages under `/workspace/src` after copying or creating them there directly.
