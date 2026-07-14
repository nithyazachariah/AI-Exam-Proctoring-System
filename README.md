# AI-Based Smart Exam Proctoring System

An **AI-powered Smart Exam Proctoring System** developed as an **Interdisciplinary Group Project** for the **Artificial Intelligence and Data Science (AI & DS)** course. The system uses **Computer Vision** and **Artificial Intelligence** to monitor online examinations in real time by detecting suspicious activities and generating detailed reports.

---

## Project Information

- **Project Title:** AI-Based Smart Exam Proctoring System
- **Course:** Artificial Intelligence and Data Science (AI & DS)
- **Project Type:** Interdisciplinary Group Project
- **Academic Year:** 2026

---

## Project Team

This project was developed by an interdisciplinary team consisting of students from **Electronics and Computer Engineering (ER)** and **Computer Science and Engineering (CSE)**.

| Team Member | Department |
|------------|------------|
| **Nithya Zachariah** | Electronics and Computer Engineering (ER) |
| **Nandana M** | Electronics and Computer Engineering (ER) |
| **Sreemayi K K** | Computer Science and Engineering (CSE) |
| **Sreya S** | Computer Science and Engineering (CSE) |

---

# Project Overview

The AI-Based Smart Exam Proctoring System is designed to automate the invigilation process during online examinations. It continuously monitors candidates through a webcam, detects suspicious activities using Artificial Intelligence and Computer Vision, records violations, captures evidence, and generates analytical reports for examination review.

---

# Features

- Student Face Registration
- Face Recognition & Identity Verification
- Mobile Phone Detection using YOLOv8
- Looking Away Detection using MediaPipe FaceMesh
- Student Absence Detection
- Multiple Face Detection
- Unknown Person Detection
- Automatic Evidence Screenshot Capture
- Event Logging (CSV)
- Risk Score Calculation
- Dashboard Generation
- PDF Report Generation

---

# Technologies Used

- Python 3.11
- OpenCV
- InsightFace
- YOLOv8 (Ultralytics)
- MediaPipe FaceMesh
- NumPy
- Pandas
- Matplotlib
- ReportLab

---

# Project Structure

```text
AI-Exam-Proctoring-System
│
├── data/
├── evidence/
├── logs/
├── reports/
├── models/
│
├── app.py
├── dashboard.py
├── face_recognition_module.py
├── face_verification.py
├── final_proctor.py
├── generate_report.py
├── logger.py
├── register_student.py
├── README.md
└── yolov8n.pt
```

---

# How to Run the Project

## Step 1: Clone the Repository

```bash
git clone https://github.com/nithyazachariah/AI-Exam-Proctoring-System.git
```

## Step 2: Navigate to the Project Folder

```bash
cd AI-Exam-Proctoring-System
```

## Step 3: Install the Required Libraries

```bash
pip install -r requirements.txt
```

## Step 4: Register the Student

```bash
python register_student.py
```

## Step 5: Start the AI Proctoring System

```bash
python final_proctor.py
```

## Step 6: Generate the Final Report

```bash
python generate_report.py
```

---

# Output Generated

The system automatically generates:

- Event Log (`logs/events.csv`)
- Evidence Images (`evidence/`)
- Summary Report (`reports/summary_report.csv`)
- Violation Graph (`reports/violation_graph.png`)
- PDF Report (`reports/exam_report.pdf`)

---

# Applications

- Online University Examinations
- Competitive Examinations
- Certification Assessments
- Recruitment Tests
- Remote Learning Platforms

---

# Future Enhancements

- Browser Tab Switching Detection
- Audio Monitoring
- Eye Gaze Tracking
- Anti-Spoofing Face Detection
- Live Web Dashboard
- Cloud-Based Report Storage

---

# Sample Outputs

The project provides:

- Student Registration
- Face Verification
- Mobile Phone Detection
- Looking Away Detection
- Multiple Face Detection
- Unknown Person Detection
- Evidence Capture
- Dashboard Visualization
- PDF Examination Report

---

# Acknowledgement

This project was developed as part of the **Artificial Intelligence and Data Science (AI & DS)** course. We sincerely thank our faculty members and institution for their valuable guidance and support throughout the development of this interdisciplinary project.

---

# License

This project is developed **for academic and educational purposes only**.

---

## If you find this project useful, consider giving it a Star on GitHub!