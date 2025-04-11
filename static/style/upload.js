let video = document.getElementById("videoStream");
let cameraContainer = document.getElementById("cameraContainer");
let stream = null;

document.getElementById("openCamera").addEventListener("click", function () {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (mediaStream) {
            stream = mediaStream; // Simpan stream agar bisa dimatikan nanti
            video.srcObject = mediaStream;
            cameraContainer.style.display = "block"; // Tampilkan video jika kamera berhasil dibuka
        })
        .catch(function (err) {
            alert("Kamera tidak dapat diakses: " + err.message);
        });
});

document.getElementById("closeCamera").addEventListener("click", function () {
    if (stream) {
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop()); // Hentikan semua track
        video.srcObject = null; // Hapus stream dari video
        cameraContainer.style.display = "none"; // Sembunyikan tampilan kamera
    }
});