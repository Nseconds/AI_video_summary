# Video Summarizer

## Description
A Flask application that processes video files to generate summaries using various machine learning models.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   conda create -n video-summarizer python=3.10
   conda activate video-summarizer
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables:
   ```bash
   export SECRET_KEY='your-secret-key-here'
   export UPLOAD_FOLDER='static/uploads'
   export MAX_CONTENT_LENGTH=16777216  # 16MB
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Upload a video file to process.

## Configuration
- Allowed file types: `mp4`, `avi`, `mov`, `mkv`
- Maximum file size: 16MB

## License
This project is licensed under the MIT License.
