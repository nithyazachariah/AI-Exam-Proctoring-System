import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)

while True:

    success, img = cap.read()

    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

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

        cv2.putText(
            img,
            f"LOOKING: {direction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Gaze Detection", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()