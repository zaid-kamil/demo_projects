from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
video_path = r"C:\Users\ZAID\Documents\minor_projects\objection_detection_app\test_video.mp4"
cap = cv2.VideoCapture(video_path)

# video length
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_frame = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    current_frame += 1
    if success:
        # Run YOLOv5 inference on the frame
        results = model(frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the progress bar
        cv2.rectangle(annotated_frame, (0, height-5), (int(width * current_frame / length), height), (0, 200, 255), -1)
        # % of video completed
        per_remaining = round((current_frame / length) * 100, 2)
        # Display the progress percentage
        cv2.putText(annotated_frame, f"{per_remaining}%", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
