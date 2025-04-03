import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target animal classes
TARGET_CLASSES = {
    'lion', 'tiger', 'leopard', 'cheetah', 'wolf', 'bear',
    'deer', 'bison', 'wild boar', 'hyena', 'jaguar',
    'dog', 'cat', 'human', 'person'
}

def load_model():
    """Load the object detection model and processor."""
    try:
        model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
        processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
        logger.info("Model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def process_image(image_path, model, processor, threshold=0.5):
    """Process an image and return detections."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        detections = []
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                label = model.config.id2label[label_id.item()].lower()
                if label in TARGET_CLASSES:
                    box = [round(coord, 2) for coord in box.tolist()]
                    detections.append({
                        'label': label,
                        'score': round(score.item(), 2),
                        'box': box,
                        'x': box[0],
                        'y': box[1],
                        'width': box[2] - box[0],
                        'height': box[3] - box[1]
                    })
        
        return detections
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def draw_annotations(image_path, annotations, output_path=None):
    """Draw annotations on an image and optionally save it."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        for anno in annotations:
            x, y, w, h = int(anno['x']), int(anno['y']), int(anno['width']), int(anno['height'])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{anno['label']} {anno.get('score', '')}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    except Exception as e:
        logger.error(f"Error drawing annotations: {e}")
        raise

def process_video(video_path, model, processor, output_path=None, frame_interval=5):
    """Process a video file and return or save detections."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_count = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process frame
            inputs = processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )
            
            frame_detections = []
            for result in results:
                for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    label = model.config.id2label[label_id.item()].lower()
                    if label in TARGET_CLASSES:
                        box = [round(coord, 2) for coord in box.tolist()]
                        frame_detections.append({
                            'frame': frame_count,
                            'label': label,
                            'score': round(score.item(), 2),
                            'box': box,
                            'x': box[0],
                            'y': box[1],
                            'width': box[2] - box[0],
                            'height': box[3] - box[1]
                        })
            
            all_detections.extend(frame_detections)
            
            if output_path:
                draw_annotations(frame, frame_detections)
        
        cap.release()
        return all_detections
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise