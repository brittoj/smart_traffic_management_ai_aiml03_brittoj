import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort import Sort


class TrafficAnalyzer:
    def __init__(self, model_path):
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        # Initialize SORT tracker for tracking vehicles across frames
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        # Class IDs for vehicles in COCO dataset: car, motorbike, bus, truck
        self.vehicle_classes = [2, 3, 5, 7]

    def process_frame(self, frame):
        # 1. Object Detection
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in self.vehicle_classes and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

        # 2. Object Tracking
        if len(detections) > 0:
            tracked_objects = self.tracker.update(np.array(detections))
        else:
            tracked_objects = self.tracker.update(np.empty((0, 5)))

        # 3. Density Analytics & Visualization
        vehicle_count = len(tracked_objects)
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {obj_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame, vehicle_count

    def calculate_signal_timing(self, dynamic_count):
        # Adaptive light timing logic based on vehicle queues
        base_time = 15  # minimum green light duration
        added_time = min(dynamic_count * 2, 45)  # 2 seconds per vehicle up to 45s cap
        return base_time + added_time


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(BASE_DIR, "traffic_sample.mp4")
    model_path = os.path.join(BASE_DIR, "yolov8n.pt")

    analyzer = TrafficAnalyzer(model_path=model_path)
    cap = cv2.VideoCapture(r"C:\Users\BRITTO\Desktop\AIML2026\assignment2026\smart_traffic_management\traffic_sample.mp4")


    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, current_density = analyzer.process_frame(frame)
        signal_duration = analyzer.calculate_signal_timing(current_density)

        cv2.putText(
            processed_frame,
            f"Density: {current_density} Vehicles",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            processed_frame,
            f"Optimum Green: {signal_duration}s",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

        cv2.imshow('Smart Traffic Management System', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
