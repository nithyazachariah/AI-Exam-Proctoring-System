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

        cv2.putText(
            img,
            f"Landmarks: {len(face)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()