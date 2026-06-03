# Models

Local model assets used by extractor backends live here.

The current `privte_behavior_v1` config expects MediaPipe Tasks model files at:

```text
models/mediapipe/hand_landmarker.task
models/mediapipe/face_landmarker.task
models/mediapipe/pose_landmarker_lite.task
```

The default YOLO device detector weight is expected at:

```text
models/yolo/yolo11n.pt
```

Model files are external assets and should not be committed unless the release
policy explicitly allows it. `.task` files in `models/mediapipe/` are ignored by
Git, and `.pt` files in `models/yolo/` are ignored by Git. The algorithm can
still run dependency-free plumbing checks with
`--allow-metadata-fallback`, but that mode intentionally produces
insufficient-evidence output rather than behavior proxy features.

On Ubuntu/WSL, MediaPipe Tasks may also need system OpenGL/EGL runtime
libraries. If initialization fails with `libGLESv2.so.2`, install one of the
Ubuntu packages that provides it, for example:

```bash
sudo apt-get update
sudo apt-get install -y libgles2
```

If the runtime later reports a missing `libEGL.so`, install `libegl1` as well.
