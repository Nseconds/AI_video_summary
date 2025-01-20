from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import os
from config import Config
from utils.video_processor import VideoProcessor
from utils.enhanced_summarizer import EnhancedVideoSummarizer

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Process the video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        summarizer = EnhancedVideoSummarizer(video_path)
        summary = summarizer.process_video()
        
        return render_template('result.html', summary=summary)
    
    flash('File type not allowed')
    return redirect(request.url)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)
