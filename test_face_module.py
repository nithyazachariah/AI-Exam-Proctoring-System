import cv2
from face_recognition_module import verify_student

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    verified, similarity, bbox = verify_student(frame)

    if bbox is not None:

        x1, y1, x2, y2 = bbox

        if verified:

            color = (0, 255, 0)
            text = "IDENTITY VERIFIED"

        else:

            color = (0, 0, 255)
            text = "UNKNOWN PERSON"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {similarity:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Recognition Module Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()