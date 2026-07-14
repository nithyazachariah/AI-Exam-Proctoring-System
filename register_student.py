import cv2
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)

print("\n")
print("===================================")
print(" STUDENT FACE REGISTRATION")
print("===================================")
print("Press 'S' to capture your face")
print("Press 'ESC' to exit")
print("===================================")

while True:

    success, frame = cap.read()

    if not success:
        break

    cv2.putText(
        frame,
        "Press S to Register",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Student Registration",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("s"):

        cv2.imwrite(
            "data/student.jpg",
            frame
        )

        print("\nStudent registered successfully!")
        print("Image saved as data/student.jpg")

        break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()