from ultralytics import YOLO
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

# Inisialisasi Flask
app = Flask(__name__)  # perbaikan: gunakan __name_ bukan name

# Pastikan folder uploads dan static ada
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model YOLO
MODEL_PATH = "model/best.pt"
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

    # Arahkan ke halaman hasil yang menampilkan streaming
    return render_template('hasil.html', hasil_stream_path=filepath)

# Deteksi langsung dan stream hasil video
def gen_detected_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("❌ Gagal membuka video")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        frame_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/stream_deteksi')
def stream_deteksi():
    video_path = request.args.get('video')
    if not video_path or not os.path.exists(video_path):
        return "❌ File video tidak ditemukan", 404
    return Response(gen_detected_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Fungsi untuk streaming dari kamera langsung
def gen_frames():
    cap = cv2.VideoCapture(0)  # Ubah ke 1 jika pakai webcam eksternal
    if not cap.isOpened():
        print("❌ Kamera tidak bisa dibuka!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        frame_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

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

# Jalankan Flask
if __name__ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)