import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    phone_detected = False

    for box in results[0].boxes:

        cls = int(box.cls[0])
        class_name = model.names[cls]

        if class_name == "cell phone":
            phone_detected = True

    if phone_detected:

        cv2.putText(
            annotated_frame,
            "PHONE DETECTED!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Phone Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()