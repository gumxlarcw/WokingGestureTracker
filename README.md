WokingGestureTracker — Work Tracker (Mouse/Keyboard) — README (Notepad)
==================================================
Skrip Python untuk memantau waktu kerja aktif berbasis deteksi tangan (MediaPipe Hands).
Aktif (WORKING) jika posisi pusat tangan terdeteksi berada di dalam ROI Keyboard atau ROI Mouse.
Menyediakan kalibrasi otomatis (k-means) untuk membentuk ROI dari pola pergerakan tangan.

Fitur Utama
--------------------------------------------------
- Deteksi wajah, pose (opsional tampilan), dan tangan (utama).
- Dua ROI: Keyboard & Mouse. Status WORKING bila salah satu terdeteksi.
- Auto-calibration (C): merekam pergerakan tangan lalu membentuk ROI via k-means.
- Manual ROI (K/M): pilih ROI Keyboard/Mouse dengan selektor interaktif.
- Timer kerja real-time + simpan hasil ke CSV.
- Tampilan overlay status, total waktu kerja (HH:MM:SS), dan panduan tombol.
- Opsi tampilan mirror (MIRROR_VIEW).

Dependensi
--------------------------------------------------
- Python 3.9+
- OpenCV (cv2)
- NumPy
- MediaPipe (pose, hands, face_detection)
- absl-py (untuk suppress log)
- (opsional) virtualenv

Instalasi Cepat (Contoh)
--------------------------------------------------
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install --upgrade pip wheel
pip install opencv-python mediapipe numpy absl-py

Konfigurasi (di awal file)
--------------------------------------------------
MIRROR_VIEW         : False  → set True jika ingin tampilan mirror kamera
AUTO_CALIB_DURATION : 6.0    → detik pengambilan sampel auto-kalibrasi
ROI_MARGIN          : 40     → margin piksel saat membuat ROI dari klaster

Cara Pakai
--------------------------------------------------
1) Jalankan skrip:
   python work_tracker.py

2) Jendela "Work Tracker - Mouse/Keyboard" akan terbuka.
   Tekan salah satu tombol berikut untuk menentukan ROI:
   - C  : Auto-ROI (kalibrasi otomatis). Gerakkan tangan di area keyboard & mouse
          selama durasi kalibrasi; sistem membentuk dua ROI dari pola titik.
   - K  : Pilih ROI Keyboard secara manual (drag seleksi pada jendela).
   - M  : Pilih ROI Mouse secara manual.
   - R  : Reset timer (total waktu kerja di-nol-kan).
   - S  : Simpan ringkasan sesi ke CSV (work_log.csv):
          start_time, end_time, total_work_time(HH:MM:SS)
   - Q  : Keluar dari aplikasi.

3) Status & timer:
   - Status "WORKING" bila pusat tangan berada di dalam ROI Keyboard/Mouse.
   - Status "IDLE" bila tidak ada tangan di ROI.
   - Total waktu kerja bertambah hanya saat status tetap WORKING.

Auto-Calibration (C) — Detail
--------------------------------------------------
- Sistem merekam titik pusat tangan (cx, cy) selama AUTO_CALIB_DURATION detik.
- Mengelompokkan dengan K-Means (k=2), klaster dengan centroid-x terbesar dianggap "Mouse",
  sisanya "Keyboard".
- ROI dibentuk dari bounding box tiap klaster + ROI_MARGIN.
- Pastikan Anda menggerakkan tangan di area keyboard & mouse secara natural saat kalibrasi.

Tampilan & Kontrol
--------------------------------------------------
- ROI Keyboard: kotak berwarna (label "Keyboard ROI")
- ROI Mouse   : kotak berwarna (label "Mouse ROI")
- Pusat tangan: titik kuning
- Overlay teks: Status, Work Time, dan bantuan tombol

Output CSV
--------------------------------------------------
File: work_log.csv
Format baris:
start_datetime_iso, end_datetime_iso, HH:MM:SS_total_work

Contoh:
2025-09-02T10:15:00,2025-09-02T11:05:42,00:36:18

Catatan Teknis
--------------------------------------------------
- Kamera dibuka dengan beberapa backend fallback: CAP_DSHOW, CAP_MSMF, dst.
- Resolusi default: 640x480 (dapat diubah via cv2.CAP_PROP_*).
- Pose & FaceDetection hanya untuk visual (tidak memengaruhi logika working),
  namun bisa dimanfaatkan di pengembangan berikutnya.
- Logika WORKING berbasis "hand center ∈ ROI".

Troubleshooting
--------------------------------------------------
- Kamera tidak terbuka:
  - Coba ganti index kamera (0/1/2) atau backend (DirectShow/MSMF).
  - Pastikan tidak ada aplikasi lain memakai kamera.
- ROI tidak akurat saat auto-calibrate:
  - Perpanjang AUTO_CALIB_DURATION (mis. 8–10 detik).
  - Naikkan ROI_MARGIN.
  - Pastikan pergerakan tangan jelas di area keyboard & mouse saat kalibrasi.
- FPS rendah:
  - Kecilkan resolusi frame.
  - Tutup aplikasi lain yang berat.
- Mirror view:
  - Set MIRROR_VIEW = True jika arah tampilan terbalik.

Roadmap (Saran)
--------------------------------------------------
- Simpan metrik per interval (mis. ke CSV/SQLite) untuk analitik detail.
- Peringatan mikro-break (mis. reminder tiap 30–45 menit WORKING).
- Analisis postur sederhana dari pose-landmarks (sudut leher/punggung).
- Multiple ROI (touchpad, pen tablet, dsb.).
- UI pengaturan ambang & kalibrasi.

Lisensi
--------------------------------------------------
Tambahkan lisensi (MIT/Apache-2.0) sesuai kebutuhan.

