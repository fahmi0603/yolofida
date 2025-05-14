from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import cv2
import os
import subprocess
import base64
import numpy as np

app = Flask(__name__)

# Buat folder jika belum ada
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load dua model YOLO
model1 = YOLO("model/best1.pt")
model2 = YOLO("model/best2.pt")

# Daftar label (kelas)
LABELS = [
    "Apel Busuk", "Apel Segar",
    "Mangga Busuk", "Mangga Segar",
    "Pisang Busuk", "Pisang Segar"
]

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

    input_path = os.path.join('uploads', file.filename)
    file.save(input_path)

    output_raw = os.path.join('static', 'hasil_deteksi_raw.mp4')
    output_final = os.path.join('static', 'hasil_deteksi.mp4')

    deteksi_video(input_path, output_raw)
    convert_to_h264(output_raw, output_final)

    return render_template('hasil.html', hasil_deteksi='hasil_deteksi.mp4')

def convert_to_h264(input_path, output_path):
    cmd = f"ffmpeg -y -i {input_path} -vcodec libx264 -acodec aac {output_path}"
    subprocess.run(cmd, shell=True)

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def gabungkan_prediksi(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results1 = model1(frame_rgb)[0]
    results2 = model2(frame_rgb)[0]

    semua_box = []

    for result, model_id in zip([results1, results2], ["model1", "model2"]):
        for box in result.boxes:
            semua_box.append({
                'xyxy': box.xyxy[0].tolist(),
                'conf': float(box.conf[0]),
                'cls': int(box.cls[0]),
                'model': model_id
            })

    selected = []
    for box in sorted(semua_box, key=lambda x: x['conf'], reverse=True):
        if all(iou(box['xyxy'], sel['xyxy']) < 0.5 for sel in selected):
            selected.append(box)

    for det in selected:
        x1, y1, x2, y2 = map(int, det['xyxy'])
        cls_id = det['cls']
        label = f"{LABELS[cls_id]} ({det['conf']:.2f})"
        color = (0, 255, 0) if 'Segar' in LABELS[cls_id] else (0, 0, 255)

        # Ukuran teks berdasarkan tinggi bounding box
        box_height = y2 - y1
        font_scale = max(0.4, min(0.6, box_height / 300))
        thickness = 1

        # Ukuran label teks
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Latar belakang teks
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
        cv2.rectangle(frame, (x1, y_text - text_height - 4), (x1 + text_width + 2, y_text + baseline - 4), color, -1)

        # Teks label (warna hitam agar kontras)
        cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Kotak deteksi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame

def deteksi_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_bgr = gabungkan_prediksi(frame)
        out.write(frame_bgr)

    cap.release()
    out.release()

@app.route('/kamera')
def kamera():
    return render_template('kamera.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_bgr = gabungkan_prediksi(frame)
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteksi_kamera', methods=['POST'])
def deteksi_kamera():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_bgr = gabungkan_prediksi(img)

    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
