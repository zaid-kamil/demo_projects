import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=detections)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(1)
    return annotated_frame

sv.process_video(
    source_path=r"people-walking.mp4",
    target_path="result.mp4",
    callback=callback
)