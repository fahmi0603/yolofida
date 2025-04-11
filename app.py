from ultralytics import YOLO
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

# Buat folder yang diperlukan
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model YOLO
MODEL_PATH = "model/best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model tidak ditemukan: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/kamera')
def kamera():
    return render_template('kamera.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Simpan video asli ke folder uploads
    input_path = os.path.join('uploads', file.filename)
    file.save(input_path)

    # Simpan hasil deteksi ke folder static
    output_path = os.path.join('static', 'hasil_deteksi.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Jalankan deteksi
    deteksi_video(input_path, output_path)

    return render_template('hasil.html', hasil_deteksi='hasil_deteksi.mp4')

def deteksi_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        print("❌ Tidak bisa membaca dimensi video.")
        return

    # Gunakan codec yang aman untuk browser
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        result_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
        out.write(result_frame)

    cap.release()
    out.release()
    print(f"✅ Video tersimpan di {output_path}")

# Streaming kamera (jika dibutuhkan)
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        result_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
