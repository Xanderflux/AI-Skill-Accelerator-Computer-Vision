# Pose Tracking with YOLOv8 + Supervision

This project demonstrates **human pose detection and tracking** in videos using [Ultralytics YOLOv8](https://docs.ultralytics.com/) and the [Supervision](https://github.com/roboflow/supervision) library.  

It processes an input video, detects people, estimates their body keypoints, tracks them across frames, and overlays annotations (skeleton, bounding boxes, traces) on the video. The final annotated video is saved to disk.

---

## 🚀 Features
- Pose estimation with **YOLOv8-pose**  
- **Keypoint detection** (joints, limbs)  
- **Object tracking** across frames with ByteTrack  
- Visual **annotations**:
  - Bounding boxes
  - Skeleton edges
  - Joint vertices
  - Motion traces  
- Save results to a new video file  
- Live display of processed frames  

---

## 🧩 How the Libraries Work Together

Ultralytics (YOLOv8)

Provides the deep learning model (yolov8n-pose.pt) that detects humans and estimates their pose keypoints.

Converts each frame into raw predictions: bounding boxes, keypoint coordinates, and confidence scores.

Supervision

Converts YOLO’s raw predictions into structured objects (Detections, KeyPoints).

Handles object tracking across frames using ByteTrack.

Provides annotation utilities to visualize results:

Skeleton edges between keypoints

Dots on joints

Bounding boxes around people

Motion traces over time

👉 In short: Ultralytics does the detection, Supervision makes the results usable and visually interpretable.

## 📦 Requirements

Install dependencies (Python 3.9+ recommended):

```bash
uv add ultralytics supervision
```

## Run object detection and tracking

```bash
uv run python tracking.py
```

## Run object detection with pose tracking

```
uv run python pose_detector.py
```
