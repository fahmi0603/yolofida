from ultralytics import YOLO
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

# Inisialisasi Flask
app = Flask(__name__)

# Pastikan folder 'uploads' dan 'static' ada
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model YOLO
MODEL_PATH = "model/best.pt"  # Pastikan path model benar
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model tidak ditemukan di: {MODEL_PATH}")

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
    
    # Simpan video ke folder uploads
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Jalankan deteksi YOLO pada video
    hasil_video = os.path.join('static', 'hasil_deteksi.mp4')
    deteksi_video(filepath, hasil_video)

    return render_template('hasil.html', hasil_deteksi='hasil_deteksi.mp4')

def deteksi_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Lebih kompatibel lintas platform
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Jika FPS 0, gunakan 30 FPS
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Menampilkan semua hasil deteksi tanpa filter kelas
        frame_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    cap.release()
    out.release()

# Fungsi untuk streaming kamera
def gen_frames():
    cap = cv2.VideoCapture(0)  # Jika pakai kamera eksternal, ubah ke 1 atau 2
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Menampilkan semua hasil deteksi tanpa filter kelas
        frame_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        # Perbaikan: Gunakan cv2.IMWRITE_JPEG_QUALITY untuk mengurangi lag
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/kamera')
def kamera():
    return render_template('kamera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🔥 Menjalankan Flask dengan Host & Port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway akan kasih PORT ini
    app.run(host='0.0.0.0', port=port)