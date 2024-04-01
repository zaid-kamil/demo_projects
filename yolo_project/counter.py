from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO("yolov8n.pt")
video_path = r"C:\Users\ZAID\Videos\Los Angeles.mp4"
cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points horizontally in the middle of the frame full width and 50px height
hp = h-200
region_points = [(0, hp), (w, hp), (w, hp + 50), (0, hp + 50)]


# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True,)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()