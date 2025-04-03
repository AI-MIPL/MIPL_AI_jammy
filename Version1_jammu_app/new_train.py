from utils.trainer_model import YOLOTrainer

def train():
    trainer = YOLOTrainer()
    
    # Process new uploads
    for file in os.listdir('static/uploads'):
        if file.endswith(('.mp4', '.avi', '.mov')):
            trainer.process_video(os.path.join('static/uploads', file))
        elif file.endswith(('.jpg', '.png', '.jpeg')):
            # Images are already processed through annotation system
            pass
            
    # Start training
    if trainer.train():
        print("YOLO training completed successfully")
    else:
        print("YOLO training failed")