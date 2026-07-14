import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

# -----------------------------
# Initialize InsightFace
# -----------------------------
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0)

# -----------------------------
# Load Registered Student
# -----------------------------
student_img = cv2.imread("data/student.jpg")

if student_img is None:
    raise Exception("Registered student image not found!")

student_faces = app.get(student_img)

if len(student_faces) == 0:
    raise Exception("No face found in student image!")

student_embedding = student_faces[0].embedding

# -----------------------------
# Cache Variables
# -----------------------------
last_time = 0

cached_verified = False
cached_similarity = 0.0
cached_bbox = None

# -----------------------------
# Face Verification
# -----------------------------
def verify_student(frame):

    global last_time
    global cached_verified
    global cached_similarity
    global cached_bbox

    current_time = time.time()

    # Run face recognition only once every second
    if current_time - last_time > 1:

        last_time = current_time

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (480, 360))

        faces = app.get(small_frame)

        if len(faces) == 0:

            cached_verified = False
            cached_similarity = 0.0
            cached_bbox = None

        else:

            face = faces[0]

            embedding = face.embedding

            similarity = np.dot(
                student_embedding,
                embedding
            ) / (
                np.linalg.norm(student_embedding)
                * np.linalg.norm(embedding)
            )

            cached_verified = similarity >= 0.60
            cached_similarity = similarity

            bbox = face.bbox.astype(int)

            # Convert coordinates back to original frame
            scale_x = frame.shape[1] / 480
            scale_y = frame.shape[0] / 360

            bbox[0] = int(bbox[0] * scale_x)
            bbox[1] = int(bbox[1] * scale_y)
            bbox[2] = int(bbox[2] * scale_x)
            bbox[3] = int(bbox[3] * scale_y)

            cached_bbox = bbox

    return (
        cached_verified,
        cached_similarity,
        cached_bbox
    )