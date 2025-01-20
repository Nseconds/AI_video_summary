import cv2
import numpy as np
import torch
from transformers import pipeline
import whisper
from ultralytics import YOLO
import spacy

class EnhancedVideoSummarizer:
    def __init__(self, video_path):
        self.video_path = video_path
        # Initialize models
        self.whisper_model = whisper.load_model("base")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.yolo_model = YOLO('yolov8n.pt')
        self.nlp = spacy.load("en_core_web_sm")  # For NER
        
    def process_video(self):
        """Main processing pipeline"""
        try:
            # Get all different types of data
            textual_data = self._process_textual_data()
            visual_data = self._process_visual_data()
            sentiment_ner_data = self._process_sentiment_ner(textual_data['transcription'])
            highlights = self._generate_highlights(visual_data, textual_data)
            
            # Combine all data into final summary
            return self._generate_overall_summary(
                textual_data, 
                visual_data, 
                sentiment_ner_data, 
                highlights
            )
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def _process_textual_data(self):
        """Process all text-related data"""
        # Get transcription
        result = self.whisper_model.transcribe(self.video_path)
        transcription = result["text"]
        
        # Generate summary of transcription
        summary = self.summarizer(transcription, max_length=130, min_length=30)[0]['summary_text']
        
        return {
            'transcription': transcription,
            'summary': summary
        }

    def _process_visual_data(self):
        """Process all visual data including motion and object detection"""
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

    def _process_sentiment_ner(self, text):
        """Process sentiment and named entities"""
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Named Entity Recognition
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            
        return {
            'sentiment': sentiment,
            'entities': entities
        }

    def _generate_highlights(self, visual_data, textual_data):
        """Generate highlights based on combined visual and textual data"""
        highlights = []
        
        # Find frames with high motion
        motion_threshold = np.mean(visual_data['motion_data']) + np.std(visual_data['motion_data'])
        
        for i, motion in enumerate(visual_data['motion_data']):
            if motion > motion_threshold:
                # Get objects in this frame
                frame_objects = visual_data['object_data'][i]
                
                highlights.append({
                    'frame_index': i,
                    'motion_intensity': motion,
                    'objects': frame_objects
                })
        
        return highlights

    def _generate_overall_summary(self, textual_data, visual_data, sentiment_ner_data, highlights):
        """Generate final comprehensive summary"""
        summary_parts = []
        
        # 1. Add transcription summary
        summary_parts.append({
            'title': 'Video Content Summary',
            'content': textual_data['summary']
        })
        
        # 2. Add sentiment information
        summary_parts.append({
            'title': 'Overall Sentiment',
            'content': f"The video content appears to be {sentiment_ner_data['sentiment']['label']} "
                       f"(confidence: {sentiment_ner_data['sentiment']['score']:.2f})"
        })
        
        # 3. Add key entities
        if sentiment_ner_data['entities']:
            summary_parts.append({'title': 'Key Elements Mentioned', 'content': ''})
            entities_by_type = {}
            for entity in sentiment_ner_data['entities']:
                if entity['label'] not in entities_by_type:
                    entities_by_type[entity['label']] = []
                entities_by_type[entity['label']].append(entity['text'])
            
            for ent_type, ents in entities_by_type.items():
                unique_ents = list(set(ents))[:3]  # Take up to 3 unique entities of each type
                summary_parts[-1]['content'] += f"- {ent_type}: {', '.join(unique_ents)}\n"
        
        # 4. Add visual highlights
        if highlights:
            summary_parts.append({'title': 'Key Visual Moments', 'content': ''})
            for i, highlight in enumerate(highlights[:3]):  # Take top 3 highlights
                objects_str = ', '.join(set(obj['label'] for obj in highlight['objects']))
                summary_parts[-1]['content'] += f"- Moment {i+1}: High activity detected with {objects_str}\n"
        
        return summary_parts

# Usage example:
"""
summarizer = EnhancedVideoSummarizer('path_to_video.mp4')
summary = summarizer.process_video()
print(summary)
"""
