import cv2
import time
import os
from datetime import datetime

from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO

from logger import log_event
from face_recognition_module import verify_student
# ==================================
# INITIALIZATION
# ==================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

detector = FaceMeshDetector(
    maxFaces=2,
    minDetectionCon=0.7,
    minTrackCon=0.7
)

model = YOLO("yolov8n.pt")

# ==================================
# FOLDERS
# ==================================

os.makedirs("evidence", exist_ok=True)

# ==================================
# VARIABLES
# ==================================

risk_score = 0
# ==========================
# YOLO CACHE
# ==========================

frame_count = 0
cached_results = None
mesh_frame_count = 0
cached_faces = None
face_missing_start = None
looking_away_start = None
phone_start_time = None

last_absent_log = 0
last_multiple_log = 0
last_away_log = 0
last_phone_log = 0
last_unknown_log = 0
unknown_start_time = None
last_screenshot_time = 0

COOLDOWN = 10
SCREENSHOT_COOLDOWN = 10

# ==================================
# EVIDENCE FUNCTION
# ==================================

def save_evidence(frame, violation):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"evidence/{violation}_{timestamp}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(
        f"Evidence Saved: {filename}"
    )

# ==================================
# MAIN LOOP
# ==================================
frame_count = 0
cached_results = None
while True:

    success = cap.grab()

    if not success:
        break

    success, frame = cap.retrieve()

    if not success:
       break

    if not success:
        break

    current_time = time.time()
        # ==========================
    # FACE RECOGNITION
    # ==========================

    verified, similarity, bbox = verify_student(frame)

    if bbox is not None:

        x1, y1, x2, y2 = bbox

        if verified:

            cv2.putText(
                frame,
                "IDENTITY VERIFIED",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Reset timer because correct student is back
            unknown_start_time = None

        else:

            cv2.putText(
                frame,
                "UNKNOWN PERSON",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if unknown_start_time is None:
                unknown_start_time = current_time

            unknown_duration = current_time - unknown_start_time
            if unknown_duration > 3:

                if current_time - last_unknown_log > COOLDOWN:

                    log_event("Unknown Person Detected")

                    risk_score += 60

                    save_evidence(
                        frame,
                        "UNKNOWN_PERSON"
                    )

                    last_unknown_log = current_time

            cv2.putText(
                frame,
                f"Unknown: {int(unknown_duration)} sec",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    else:
        unknown_start_time = None
    # ==================================
    # PHONE DETECTION (YOLO)
    # ==================================

    frame_count += 1

    if frame_count % 5 == 0 or cached_results is None:
        cached_results = model(
            frame,
            imgsz=320,
            conf=0.5,
            verbose=False
        )

    results = cached_results

    phone_detected = False

    for box in results[0].boxes:

        cls = int(box.cls[0])
        class_name = model.names[cls]

        if class_name == "cell phone":

            phone_detected = True

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "PHONE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # ==================================
    # PHONE LOGIC
    # ==================================

    if phone_detected:

        if phone_start_time is None:
            phone_start_time = current_time

        phone_duration = (
            current_time - phone_start_time
        )

        cv2.putText(
            frame,
            f"Phone: {int(phone_duration)} sec",
            (20, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if phone_duration > 3:

            cv2.putText(
                frame,
                "PHONE DETECTED!",
                (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            if current_time - last_phone_log > COOLDOWN:

                log_event(
                    "Mobile Phone Detected"
                )

                if (
                    current_time
                    - last_screenshot_time
                    > SCREENSHOT_COOLDOWN
                ):
                    save_evidence(
                        frame,
                        "PHONE"
                    )
                    last_screenshot_time = (
                        current_time
                    )

                risk_score += 40
                last_phone_log = current_time

    else:
        phone_start_time = None

    # ==================================
    # FACE MESH
    # ==================================

    mesh_frame_count += 1

    if mesh_frame_count % 3 == 0 or cached_faces is None:

        frame, cached_faces = detector.findFaceMesh(
           frame,
           draw=False
    )

    faces = cached_faces
    face_count = (
        len(faces)
        if faces
        else 0
    )

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ==================================
    # MULTIPLE FACE
    # ==================================

    if face_count > 1:

        cv2.putText(
            frame,
            "MULTIPLE FACES!",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if (
            current_time
            - last_multiple_log
            > COOLDOWN
        ):

            log_event(
                "Multiple Faces Detected"
            )

            if (
                current_time
                - last_screenshot_time
                > SCREENSHOT_COOLDOWN
            ):
                save_evidence(
                    frame,
                    "MULTI_FACE"
                )
                last_screenshot_time = (
                    current_time
                )

            risk_score += 50
            last_multiple_log = current_time

    # ==================================
    # STUDENT ABSENT
    # ==================================

    if face_count == 0:

        if face_missing_start is None:
            face_missing_start = current_time

        absent_time = (
            current_time
            - face_missing_start
        )

        cv2.putText(
            frame,
            f"No Face: {int(absent_time)} sec",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if absent_time > 5:

            cv2.putText(
                frame,
                "STUDENT ABSENT!",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if (
                current_time
                - last_absent_log
                > COOLDOWN
            ):

                log_event(
                    "Student Absent"
                )

                if (
                    current_time
                    - last_screenshot_time
                    > SCREENSHOT_COOLDOWN
                ):
                    save_evidence(
                        frame,
                        "ABSENT"
                    )
                    last_screenshot_time = (
                        current_time
                    )

                risk_score += 30
                last_absent_log = current_time

    else:
        face_missing_start = None

    # ==================================
    # LOOKING AWAY
    # ==================================

    if face_count == 1:

        face = faces[0]

        left_point = face[130]
        right_point = face[359]
        nose_point = face[1]

        face_width = (
            right_point[0]
            - left_point[0]
        )

        if face_width != 0:

            ratio = (
                (nose_point[0]
                - left_point[0])
                / face_width
            )

            direction = "CENTER"

            if ratio < 0.45:
                direction = "LEFT"

            elif ratio > 0.55:
                direction = "RIGHT"

            cv2.putText(
                frame,
                f"LOOKING: {direction}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            if direction in [
                "LEFT",
                "RIGHT"
            ]:

                if looking_away_start is None:
                    looking_away_start = (
                        current_time
                    )

                away_time = (
                    current_time
                    - looking_away_start
                )

                cv2.putText(
                    frame,
                    f"Away: {int(away_time)} sec",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                if away_time > 5:

                    cv2.putText(
                        frame,
                        "SUSPICIOUS BEHAVIOR",
                        (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    if (
                        current_time
                        - last_away_log
                        > COOLDOWN
                    ):

                        log_event(
                            "Looking Away"
                        )

                        if (
                            current_time
                            - last_screenshot_time
                            > SCREENSHOT_COOLDOWN
                        ):
                            save_evidence(
                                frame,
                                "LOOKING_AWAY"
                            )
                            last_screenshot_time = (
                                current_time
                            )

                        risk_score += 20
                        last_away_log = current_time

            else:
                looking_away_start = None

    # ==================================
    # RISK LEVEL
    # ==================================

    if risk_score < 100:
        risk_level = "LOW"

    elif risk_score < 300:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    cv2.putText(
        frame,
        f"Risk Score: {risk_score}",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Risk Level: {risk_level}",
        (20, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "AI Exam Proctoring System",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

# ==================================
# CLEANUP
# ==================================

cap.release()
cv2.destroyAllWindows()