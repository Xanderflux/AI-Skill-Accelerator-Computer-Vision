import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load YOLO model
# This time we are using the standard object detection model (not pose).
# "yolov8n.pt" → YOLOv8 nano (smallest version) pretrained on COCO dataset.
model = YOLO("yolov8n.pt")

# Annotators from Supervision
box_annotator = sv.BoxAnnotator()  # Draws bounding boxes around detected objects
trace_annotator = sv.TraceAnnotator(
    thickness=2, trace_length=30
)  # Draws motion trails behind moving objects
label_annotator = sv.LabelAnnotator()  # Adds labels (e.g., class names, IDs)

# Tracker
# Uses ByteTrack to assign consistent IDs to objects across frames
# so you can follow the same object as it moves through the video.
tracker = sv.ByteTrack()


def callback(frame: np.ndarray) -> np.ndarray:
    """
    Runs YOLO detection + tracking + annotation on a single frame
    and returns the annotated frame.
    """
    results = model(frame)[0]  # Run object detection on the frame
    detections = sv.Detections.from_ultralytics(results)  # Convert to Supervision detections
    detections = tracker.update_with_detections(detections)  # Update tracker with current detections

    # Extract class names if available (sometimes results might not have them)
    class_names = detections.data.get("class_name", [])

    # Build labels like "ID 3 | person" for each tracked object
    labels = []
    if len(class_names) > 0:
        labels = [
            f"ID {tracker_id} | {class_name}"
            for class_name, tracker_id in zip(class_names, detections.tracker_id)
        ]

    # Apply annotations
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)  # Draw bounding boxes

    if labels:
        # Add labels only if we have them
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )

    # Draw motion traces (paths of objects over time)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

    return annotated_frame


def track_object_with_traces(input_path: str, output_path: str):
    """
    Reads a video, applies YOLO object detection + tracking + annotation frame by frame,
    and writes the result to a new video file.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties for output writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer (saves output with same FPS, width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the frame with YOLO + tracking + annotations
        annotated_frame = callback(frame)

        # Show the result in a window (live visualization)
        cv2.imshow("YOLO Tracking with Traces", annotated_frame)

        # Save the processed frame to output video
        out.write(annotated_frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Saved video to: {output_path}")


if __name__ == "__main__":
    # Run tracker on input video and save annotated result
    track_object_with_traces("man_utd.mp4", "tracking_result.mp4")
