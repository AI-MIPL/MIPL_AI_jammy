import os
import cv2
import json
import shutil
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

class YOLOTrainer:
    def __init__(self):
        self.base_dir = os.path.join('static', 'yolo_data')
        self.model_dir = os.path.join('static', 'models', 'yolo')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Video frame extraction parameters
        self.frame_interval = 10  # Extract every 10th frame from videos

    def convert_annotations(self):
        """Convert JSON annotations to YOLO format"""
        json_files = [f for f in os.listdir('static/annotations') if f.endswith('.json')]
        
        for json_file in json_files:
            # Create matching text file path
            txt_path = os.path.join(self.base_dir, 'labels', 'train', 
                                   json_file.replace('.json', '.txt'))
            
            with open(os.path.join('static/annotations', json_file)) as f:
                data = json.load(f)
            
            img_path = os.path.join('static/uploads', json_file.replace('.json', '.jpg'))
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            with open(txt_path, 'w') as f:
                for ann in data:
                    # Convert to YOLO format (class cx cy w h)
                    x = ann['x']
                    y = ann['y']
                    w = ann['width']
                    h = ann['height']
                    
                    cx = (x + w/2) / img_width
                    cy = (y + h/2) / img_height
                    nw = w / img_width
                    nh = h / img_height
                    
                    f.write(f"{ann['class_id']} {cx} {cy} {nw} {nh}\n")

    def process_video(self, video_path):
        """Extract frames from video and create placeholder annotations"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0:
                img_path = os.path.join(self.base_dir, 'images', 'train', 
                                      f"video_frame_{saved_count}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Create empty annotation file
                txt_path = os.path.join(self.base_dir, 'labels', 'train',
                                      f"video_frame_{saved_count}.txt")
                open(txt_path, 'a').close()
                
                saved_count += 1
            frame_count += 1
            
        cap.release()

    def prepare_dataset(self):
        """Prepare YOLO dataset structure"""
        # Create directories
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val']:
            Path(os.path.join(self.base_dir, 'images', split)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.base_dir, 'labels', split)).mkdir(parents=True, exist_ok=True)

        # Convert existing annotations
        self.convert_annotations()
        
        # Move 20% of data to validation
        all_files = os.listdir(os.path.join(self.base_dir, 'images', 'train'))
        val_files = np.random.choice(all_files, int(0.2*len(all_files)), replace=False)
        
        for file in val_files:
            shutil.move(
                os.path.join(self.base_dir, 'images', 'train', file),
                os.path.join(self.base_dir, 'images', 'val', file)
            )
            shutil.move(
                os.path.join(self.base_dir, 'labels', 'train', file.replace('.jpg', '.txt')),
                os.path.join(self.base_dir, 'labels', 'val', file.replace('.jpg', '.txt'))
            )

    def train(self):
        """Main training function"""
        try:
            self.prepare_dataset()
            
            # Initialize YOLO model
            model = YOLO('yolov11n.pt')  # Using YOLOv11 nano variant
            
            # Training configuration
            results = model.train(
                data=os.path.join(self.base_dir, 'dataset.yaml'),
                epochs=50,
                imgsz=640,
                batch=8,
                project=self.model_dir,
                name='trained_model',
                exist_ok=True
            )
            
            print(f"Training complete. Model saved to {self.model_dir}")
            return True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False