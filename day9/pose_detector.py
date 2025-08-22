import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n-pose.pt")

# Annotators
edge_annotator = sv.EdgeAnnotator() # draws lines between keypoints (e.g., skeleton joints)
vertex_annotator = sv.VertexAnnotator() #draws dots at keypoints (e.g., joints themselves).
box_annotator = sv.BoxAnnotator() # draws bounding boxes around detected objects.
trace_annotator = sv.TraceAnnotator() #draws the trajectory of moving objects across frames.

# Tracker
#Uses ByteTrack, a lightweight object tracking algorithm.
# It assigns consistent IDs to objects across frames so you can track the same person 
# throughout the video
tracker = sv.ByteTrack()


def callback(frame: np.ndarray) -> np.ndarray:
    results = model(frame)[0] #Runs pose detection on the input frame and gets first batch output
    key_points = sv.KeyPoints.from_ultralytics(results) #Extracts human pose keypoints (like head, shoulders, knees) in a format Supervision understands.
    detections = key_points.as_detections()  #Converts keypoints to detection objects so they can be tracked.
    detections = tracker.update_with_detections(detections) #Tracker assigns IDs and tracks across frames.
    annotated_frame = edge_annotator.annotate(frame.copy(), key_points=key_points)  #Copy the frame (so original isn’t modified) and Draw skeleton edges (lines between joints).
    annotated_frame = vertex_annotator.annotate(annotated_frame, key_points=key_points) #Draw vertices (dots at joints).
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections) #Draw bounding boxes around detected humans.
    return trace_annotator.annotate(annotated_frame, detections=detections) #Draw motion traces (path of movement) and Return the final annotated frame.


def track_objects_with_poses(source_path: str, target_path: str):

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {source_path}")

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save result
    #Creates a video writer to save processed frames.
    #Uses "mp4v" codec → outputs .mp4.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    #Reads frame-by-frame.
    #If no frame is left (end of video), exit loop.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Runs the YOLO model + tracking + annotation.
        annotated_frame = callback(frame)

        # Show in a window
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Save to file
        out.write(annotated_frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Saved video to: {target_path}")



if __name__ == "__main__":
    track_objects_with_poses("man_utd.mp4", "pose_result.mp4")
