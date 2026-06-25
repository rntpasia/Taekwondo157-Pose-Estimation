# Taekwondo157-Pose-Estimation

# 🥋 Taekwondo Pose Estimation Using YOLO Pose

## Overview

This project implements a Human Pose Estimation system for Taekwondo using the Ultralytics YOLO Pose model. The application detects and visualizes human body keypoints from uploaded images, allowing analysis of Taekwondo movements specifically forms 1, 5, and 7.

The system was developed as part of a project exploring the application of computer vision techniques for keypoint detection and pose estimation in Taekwondo.

---

## Features

* Upload Taekwondo images through a Streamlit web interface
* Automatic human pose estimation using a trained YOLO Pose model
* Visualization of detected body keypoints and skeletal connections
* Bounding box detection for practitioners
* Adjustable confidence threshold
* Automatic model download from Google Drive during deployment
* Deployable on Streamlit Community Cloud

---

## Dataset

The model was trained using annotated images of Taekwondo practitioners performing:
* Form 1
* Form 5
* Form 7

Each image contains:
* Bounding box annotations
* Human pose keypoints
* Keypoint visibility information

---

## Streamlit Deployment

The application is configured for deployment on Streamlit Community Cloud.

---

## Model Download

To avoid GitHub's file size limitations, the trained model is hosted externally and automatically downloaded when the application starts.

The model is cached locally after the first download.

---

