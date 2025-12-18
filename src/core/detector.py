from ultralytics import YOLO
import threading

"""
file: detector.py

This file contains the methods for initializing and interfacing with the YOLOv8n face detection.

We define the YOLO model as a singleton class with interface methods for performing inferences, and what not. 
"""

class YOLOService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.model = YOLO("models/yolov8n-pose.pt")
        return cls._instance
    
    def _Process_Image(self, image_bytes):
        #converts image bytes to OpenCV/PIL object
        return

    def _Detect_Image(self, processed_image):
        #detects faces from OpenCV/PIL object and returns an array of coordinates of the face centers
        return


