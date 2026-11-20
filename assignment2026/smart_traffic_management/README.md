# Real-Time Video Analytics for Smart Traffic Management

This repository contains a full production-ready implementation skeleton for an AI-driven smart traffic monitoring and adaptive light signaling platform.

## 📂 Repository Structure & Zip Details

* `app.py`: Core application script handling YOLOv8 vehicle detection, frame tracking, and dynamic timing algorithms.
* `requirements.txt`: Python package dependency list.
* `sort.py`: Simple Online and Realtime Tracking script implementation for frame-to-frame vehicle correlation.
* `traffic_video.mp4`: Sample placeholder traffic video stream for inference testing.

## 🚀 Setup & Execution Instructions

1. **Extract Zip Contents:** Unpack your download archive locally.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Application:** Place your traffic footage as `traffic_video.mp4` and execute:
   ```bash
   python app.py
   ```
