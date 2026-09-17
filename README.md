# 🏏 AI-Powered Decision Review System for Local Cricket

An AI-powered Computer Vision system designed to assist cricket decision-making for local, street, and indoor cricket environments.

The system analyzes cricket video footage and applies computer vision and deep learning techniques to assist with decisions such as:

- LBW
- No Ball
- Wide Ball
- Legal Ball

## 🚀 Project Overview

Traditional Decision Review Systems are expensive and primarily designed for professional cricket environments.

This project explores a more accessible AI-based approach for local cricket by combining object detection, object tracking, trajectory estimation, and deep learning-based decision classification.

The system processes video input to track the ball and analyze relevant events before generating a decision.

## 🧠 Key Features

- Cricket ball detection and tracking
- Real-time video processing
- LBW decision assistance
- No Ball detection
- Wide Ball detection
- Legal Ball detection
- Ball trajectory estimation
- Deep learning-based decision classification
- Web-based interface for interacting with the system

## 🛠️ Technologies Used

### Computer Vision
- Python
- OpenCV
- YOLOv11
- ByteTrack
- Kalman Filter

### Deep Learning
- ResNet18 3D
- Convolutional Neural Networks

### Backend
- Flask

### Frontend
- HTML
- CSS
- JavaScript

## 🔄 System Workflow

1. Input cricket video
2. Detect the cricket ball using YOLO
3. Track the detected ball across video frames
4. Apply trajectory estimation using Kalman filtering
5. Analyze the detected cricket event
6. Apply deep learning/rule-based decision logic
7. Generate the corresponding cricket decision

## 📁 Project Structure

```text
AI-Powered-Decision-Review-System-for-Local-Cricket/
│
├── index.html
├── style.css
├── script.js
├── README.md
└── .gitignore
