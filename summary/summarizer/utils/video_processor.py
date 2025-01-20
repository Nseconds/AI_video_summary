import cv2
import numpy as np
import torch
from transformers import pipeline
import whisper
from ultralytics import YOLO
import spacy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.yolo_model = YOLO('yolov8n.pt')
        self.nlp = spacy.load("en_core_web_sm")  # For NER

    def process_video(self):
        """Process the video and extract relevant data"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            frames = []
            motion_data = []
            object_data = []
            prev_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Store frame
                frames.append(frame)
                
                # Calculate motion if we have a previous frame
                if prev_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    motion_intensity = np.mean(np.abs(flow))
                    motion_data.append(motion_intensity)
                
                # Perform object detection
                results = self.yolo_model(frame)
                frame_objects = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].tolist()  # get box coordinates
                        conf = float(box.conf)     # get confidence
                        cls = int(box.cls)         # get class
                        name = self.yolo_model.names[cls]  # get class name
                        frame_objects.append({
                            'label': name,
                            'confidence': conf,
                            'bbox': b
                        })
                object_data.append(frame_objects)
                
                prev_frame = frame.copy()
                
            cap.release()
            return {
                'frames': frames,
                'motion_data': motion_data,
                'object_data': object_data
            }
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return None
