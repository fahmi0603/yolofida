from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
import subprocess
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import request, jsonify, send_file

app = Flask(__name__)

# Buat folder jika belum ada
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model YOLO
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Simpan file upload
    input_path = os.path.join('uploads', file.filename)
    file.save(input_path)

    # Output awal
    output_raw = os.path.join('static', 'hasil_deteksi_raw.mp4')
    output_final = os.path.join('static', 'hasil_deteksi.mp4')

    # Deteksi video
    deteksi_video(input_path, output_raw)

    # Konversi ke format browser friendly (H.264)
    convert_to_h264(output_raw, output_final)

    return render_template('hasil.html', hasil_deteksi='hasil_deteksi.mp4')

def deteksi_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Masih raw
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        frame_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        out.write(frame_bgr)

    cap.release()
    out.release()

def convert_to_h264(input_path, output_path):
    """Konversi video agar bisa diputar di browser (H.264 + AAC)"""
    cmd = f"ffmpeg -y -i {input_path} -vcodec libx264 -acodec aac {output_path}"
    subprocess.run(cmd, shell=True)

@app.route('/kamera')
def kamera():
    return render_template('kamera.html')

def proses_kamera():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Tidak ada gambar'}), 400

    # Decode Base64 image
    encoded_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(encoded_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    # Jalankan deteksi YOLO
    results = model(frame)
    detected_frame = results[0].plot()  # sudah dalam RGB

    # Convert ke JPEG
    _, buffer = cv2.imencode('.jpg', detected_frame)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')
@app.route('/video_feed')
def video_feed():
    return Response(proses_kamera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
