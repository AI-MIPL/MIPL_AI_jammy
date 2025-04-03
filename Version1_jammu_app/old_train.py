import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, processor, transform=None):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.processor = processor
        self.transform = transform
        
        # Get list of annotation files
        self.annotation_files = [f for f in os.listdir(annotations_dir) 
                               if f.endswith('.json')]
        
        # Verify corresponding images exist
        self.valid_samples = []
        for ann_file in self.annotation_files:
            image_file = ann_file.replace('.json', '.jpg')
            if os.path.exists(os.path.join(images_dir, image_file)):
                self.valid_samples.append((image_file, ann_file))
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        image_file, ann_file = self.valid_samples[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        # Load annotations
        with open(os.path.join(self.annotations_dir, ann_file)) as f:
            annotations = json.load(f)
        
        # Convert annotations to DETR format
        boxes = []
        labels = []
        for ann in annotations:
            boxes.append([ann['x'], ann['y'], 
                         ann['x'] + ann['width'], 
                         ann['y'] + ann['height']])
            labels.append(ann['class'])  # We'll need to map these to IDs
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # TODO: Create a class mapping (this should be consistent across all data)
        class_mapping = {
            'lion': 1, 'tiger': 2, 'leopard': 3, 'cheetah': 4, 
            'wolf': 5, 'bear': 6, 'deer': 7, 'bison': 8,
            'wild boar': 9, 'hyena': 10, 'jaguar': 11,
            'dog': 12, 'cat': 13, 'human': 14
        }
        
        labels = torch.as_tensor([class_mapping.get(label, 0) for label in labels], 
                                dtype=torch.int64)
        
        # Prepare for DETR
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply processor
        if self.processor:
            encoding = self.processor(images=image, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            target = encoding["labels"][0]  # Remove batch dimension
        
        return pixel_values, target

def train():
    try:
        logger.info("Starting training process...")
        
        # Initialize processor and model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                      num_labels=15,  # Adjust based on your classes
                                                      ignore_mismatched_sizes=True)
        
        # Set up dataset and dataloader
        dataset = AnimalDataset(
            annotations_dir='static/annotations',
            images_dir='static/uploads',
            processor=processor
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, (pixel_values, targets) in enumerate(dataloader):
                pixel_values = pixel_values.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, labels=targets)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save the trained model
        model.save_pretrained('static/models/animal_detector')
        processor.save_pretrained('static/models/animal_detector')
        logger.info("Training complete. Model saved.")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train()