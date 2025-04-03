from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
from PIL import Image
import json
from werkzeug.utils import secure_filename
from utils.detection import load_model, process_video,process_image, draw_annotations
import torch

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ANNOTATION_FOLDER'] = 'static/annotations'
app.config['MODEL_FOLDER'] = 'static/models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATION_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Load detection model
model, processor = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    videos = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
              if f.endswith(('.mp4', '.avi', '.mov'))]
    frames = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', videos=videos, frames=frames)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('file')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    return redirect(url_for('index'))

@app.route('/annotate/<filename>')
def annotate(filename):
    return render_template('annotate.html', filename=filename)

@app.route('/auto_annotate/<filename>', methods=['POST'])
def auto_annotate(filename):
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5
        )
        
        annotations = []
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                label = model.config.id2label[label_id.item()].lower()
                box = [round(coord, 2) for coord in box.tolist()]
                
                annotations.append({
                    'x': box[0],
                    'y': box[1],
                    'width': box[2] - box[0],
                    'height': box[3] - box[1],
                    'class': label,
                    'score': round(score.item(), 2),
                    'auto': True  # Mark as auto-generated
                })
        
        return jsonify({'status': 'success', 'annotations': annotations})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data['filename']
    annotations = data['annotations']
    
    annotation_path = os.path.join(
        app.config['ANNOTATION_FOLDER'], 
        f"{os.path.splitext(filename)[0]}.json"
    )
    
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return jsonify({'status': 'success'})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # In a production environment, you would use Celery or similar for background tasks
        os.system('python train.py &')
        return jsonify({'status': 'training_started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_status')
def train_status():
    return render_template('train.html')

if __name__ == '__main__':
    app.run(debug=True)