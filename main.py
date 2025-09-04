from ultralytics import YOLO
import cv2
import torch
import numpy as np
from sort.sort import Sort

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt").to(device)

person = [0]

person_track = Sort(max_age=30)
video_path = "vd1.mp4"
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
region_points = np.array(
    [[199, 583], [250, 719], [799, 719], [953, 443],
     [689, 332], [594, 286], [270, 398], [299, 454], [185, 506]],
    np.int32
).reshape((-1, 1, 2))

tracking_person = []
while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break

    # frame = cv2.resize(frame, (1920, 1080))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(region_points, dtype=np.int32)], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    results = model(frame)[0]

    for result in results.boxes:

        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        score = float(result.conf[0])
        cls = int(result.cls[0].item())


        if cls in person:
            tracking_person.append([x1, y1, x2, y2, score])
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            inside = cv2.pointPolygonTest(region_points, (cx, cy), False)
            for track in tracking_person:
                x1, y1, x2, y2, track_id = track
                # cv2.putText(frame, f"{int(track_id)}", (int(x1), int(y1) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if inside >= 0:
                color = (0, 0, 255)
                cv2.putText(frame, "IN REGION", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    cv2.polylines(frame, [region_points], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("result", frame)
    cv2.waitKey(2)

cap.release()
video_writer.release()
cv2.destroyAllWindows()


