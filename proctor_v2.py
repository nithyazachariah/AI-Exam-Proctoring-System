import cv2
import time
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)

looking_away_start = None

while True:

    success, img = cap.read()

    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

    direction = "NO FACE"

    if faces:

        face = faces[0]

        left_point = face[130]
        right_point = face[359]
        nose_point = face[1]

        face_width = right_point[0] - left_point[0]

        nose_ratio = (nose_point[0] - left_point[0]) / face_width

        direction = "CENTER"

        if nose_ratio < 0.45:
            direction = "LEFT"

        elif nose_ratio > 0.55:
            direction = "RIGHT"

        # Looking Away Logic
        if direction in ["LEFT", "RIGHT"]:

            if looking_away_start is None:
                looking_away_start = time.time()

            elapsed = time.time() - looking_away_start

            cv2.putText(
                img,
                f"Away: {int(elapsed)} sec",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if elapsed > 5:

                cv2.putText(
                    img,
                    "SUSPICIOUS BEHAVIOR",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        else:
            looking_away_start = None

    cv2.putText(
        img,
        f"LOOKING: {direction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Proctor V2", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()