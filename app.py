from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from main import initialize_sam2, extract_frames, process_video, get_user_input
from flask_socketio import SocketIO, emit
import threading
import webbrowser
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize SAM2 predictor
predictor = initialize_sam2()

browser_opened = False

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

def run_app():
    socketio.run(app, debug=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_video = request.files['video']
    if input_video:
        # Create uploads directory if it doesn't exist
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save the uploaded video
        video_path = os.path.join(uploads_dir, input_video.filename)
        input_video.save(video_path)

        # Get parameters from the form
        test_mode = request.form.get('test_mode') == 'on'
        test_duration = int(request.form.get('test_duration', 5))
        batch_size = int(request.form.get('batch_size', 250))

        def progress_callback(progress, status):
            socketio.emit('progress_update', {'progress': progress, 'status': status})

        # Extract frames (always extract all frames)
        frames_dir = os.path.splitext(video_path)[0] + "_frames"
        frame_count, fps = extract_frames(video_path, frames_dir, progress_callback)

        # Get the first frame for user input
        first_frame = cv2.imread(os.path.join(frames_dir, sorted(os.listdir(frames_dir))[0]))
        frame_base64 = get_user_input(first_frame)

        return jsonify({'status': 'awaiting_input', 'frame': frame_base64, 'video_path': video_path})
    else:
        return jsonify({'status': 'error', 'message': 'No video file uploaded'})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    data = request.json
    input_points = np.array(data.get('points', []))
    input_labels = np.array(data.get('labels', []))
    video_path = data.get('video_path')

    if not video_path:
        return jsonify({'status': 'error', 'message': 'Video path is missing'}), 400

    frames_dir = os.path.splitext(video_path)[0] + "_frames"
    output_video = os.path.splitext(video_path)[0] + "_masked.mp4"
    output_gif = os.path.splitext(video_path)[0] + "_masked.gif"

    test_mode = data.get('test_mode', False)
    test_duration = data.get('test_duration', 5)
    batch_size = data.get('batch_size', 250)

    def progress_callback(progress, status):
        socketio.emit('progress_update', {'progress': progress, 'status': status})

    def process_thread():
        try:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Process video (apply test_mode only during processing)
            _, gif_path = process_video(predictor, frames_dir, output_video, output_gif, fps, 
                                        input_points, input_labels,
                                        test_mode=test_mode, test_duration=test_duration, 
                                        batch_size=batch_size, progress_callback=progress_callback,
                                        socketio=socketio)
            socketio.emit('processing_complete', {'gif_path': gif_path})
        except Exception as e:
            progress_callback(-1, f"Error: {str(e)}")

    def frame_callback(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame_update', {'frame': frame_base64})

    # Start processing in a separate thread
    thread = threading.Thread(target=process_thread)
    thread.start()

    return jsonify({'status': 'processing'})

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    response = send_from_directory('uploads', filename, as_attachment=False)
    response.headers['Content-Type'] = 'video/mp4'
    return response

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('uploads', filename, as_attachment=True)

@socketio.on('quit')
def handle_quit():
    os._exit(0)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    threading.Timer(2, open_browser).start()
    run_app()
