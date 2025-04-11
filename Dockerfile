# Gunakan base image python
FROM python:3.12-slim

# ✅ Install ffmpeg + dependency OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt dan install semua dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua isi project ke dalam container
COPY . .

# Jalankan aplikasi
CMD ["python", "app.py"]
