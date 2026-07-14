import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ------------------------------------
# Load Face Recognition Model
# ------------------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# ------------------------------------
# Load Registered Student Image
# ------------------------------------
student_img = cv2.imread("data/student.jpg")

if student_img is None:
    print("Student image not found!")
    exit()

student_faces = app.get(student_img)

if len(student_faces) == 0:
    print("No face detected in registered image!")
    exit()

student_embedding = student_faces[0].embedding

print("Student face loaded successfully!")

# ------------------------------------
# Start Webcam
# ------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam!")
    exit()

print("\nPress ESC or Q to Exit\n")

# ------------------------------------
# Main Loop
# ------------------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    faces = app.get(frame)

    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    for face in faces:

        embedding = face.embedding

        similarity = np.dot(
            student_embedding,
            embedding
        ) / (
            np.linalg.norm(student_embedding)
            * np.linalg.norm(embedding)
        )

        if similarity >= 0.60:

            status = "IDENTITY VERIFIED"
            color = (0, 255, 0)

        else:

            status = "UNKNOWN PERSON"
            color = (0, 0, 255)

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {similarity:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    cv2.imshow(
        "Face Verification",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:          # ESC
        break

    elif key == ord("q"):  # Q
        break

# ------------------------------------
# Cleanup
# ------------------------------------
cap.release()
cv2.destroyAllWindows()