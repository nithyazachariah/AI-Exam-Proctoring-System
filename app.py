import cv2
import time
from logger import log_event

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

# Variables
face_missing_start = None

# Cooldown settings
last_multiple_log_time = 0
last_absence_log_time = 0
LOG_COOLDOWN = 10  # seconds

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    face_count = len(faces)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Display face count
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ====================================
    # MULTIPLE FACE DETECTION
    # ====================================
    if face_count > 1:

        current_time = time.time()

        if current_time - last_multiple_log_time > LOG_COOLDOWN:
            log_event("Multiple Faces Detected")
            last_multiple_log_time = current_time

        cv2.putText(
            frame,
            "WARNING: MULTIPLE FACES!",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ====================================
    # FACE ABSENCE DETECTION
    # ====================================
    if face_count == 0:

        if face_missing_start is None:
            face_missing_start = time.time()

        elapsed = time.time() - face_missing_start

        cv2.putText(
            frame,
            f"No Face: {int(elapsed)} sec",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if elapsed > 5:

            current_time = time.time()

            if current_time - last_absence_log_time > LOG_COOLDOWN:
                log_event("Student Absent")
                last_absence_log_time = current_time

            cv2.putText(
                frame,
                "STUDENT ABSENT!",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    else:
        face_missing_start = None

    # Show video
    cv2.imshow(
        "AI Exam Proctor",
        frame
    )

    # ESC key exits
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()