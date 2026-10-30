# Retrolens - Hand Tracking Filter & Portal

Aplikasi filter kamera interaktif menggunakan gesture tangan (hand tracking) dengan MediaPipe dan OpenCV. Anda dapat membuat "portal" dengan jari Anda yang akan menerapkan berbagai filter menarik (Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon, Galaxy).

Ada juga **mode SNAP**: jentikkan jari di depan kamera, layar menyala putih kebiruan dari tangan Anda, lalu tubuh Anda hancur menjadi debu dari kepala ke bawah dan terbawa angin, menyisakan background kosong. Penjelasan lengkap ada di bagian [Mode SNAP](#mode-snap).

## Persyaratan Sistem
- Python 3.9 atau lebih baru
- Webcam
- Mikrofon (opsional, hanya untuk trigger suara di mode SNAP)

## Cara Install dan Menjalankan

### 1. Clone Repository
Pertama, clone repository ini ke komputer Anda dan masuk ke foldernya (ganti URL dengan link repository GitHub Anda):
```bash
git clone <URL_GITHUB_ANDA>
cd <NAMA_FOLDER_REPO>
```

### 2. Buat Virtual Environment (Opsional tapi Sangat Disarankan)
Gunakan virtual environment agar dependencies (library) tidak bentrok dengan project Python lainnya di komputer Anda.
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install semua library Python yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

Atau jika Anda ingin menginstallnya secara manual satu per satu:
```bash
pip install opencv-contrib-python mediapipe numpy sounddevice
```

**Catatan soal OpenCV:** cukup pasang satu paket OpenCV saja. `opencv-python` dan `opencv-contrib-python` sama-sama memakai nama modul `cv2` dan saling menimpa file jika terpasang bersamaan. MediaPipe 1.0 sudah mewajibkan `opencv-contrib-python` (isinya mencakup `opencv-python`), jadi project ini memakai paket itu. Jika keduanya sudah terlanjur terpasang:
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### 4. Pastikan Model MediaPipe Tersedia
Aplikasi ini membutuhkan dua file model dari MediaPipe yang seharusnya sudah ada di dalam repository ini:
1. `hand_landmarker.task` (Untuk mendeteksi titik pada tangan)
2. `selfie_segmenter.tflite` (Untuk filter Galaxy dan mode SNAP, memisahkan orang dari background)

Jika file tersebut belum ada, pastikan untuk meletakkannya di dalam folder yang sama dengan file `main.py`.

**Gesture model (one time download).** Hold gestures such as thumbs up (learn the room) and thumbs down (cancel) need `gesture_recognizer.task` next to `main.py`. It is not in the repository. Download it once:

```bash
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

If that link ever fails, get the file from the Models section of the official MediaPipe Gesture Recognizer page. Without it the app still runs, and only the snap gesture works.

### 5. Jalankan Aplikasi
Jalankan script utamanya dengan mengetik:
```bash
python main.py
```
*(Gunakan `python3 main.py` jika Anda menggunakan macOS/Linux dan tidak menggunakan virtual environment)*

Untuk langsung masuk ke mode SNAP:
```bash
python main.py --snap
```

## Cara Penggunaan Fitur

- **Membuka Portal:** Gunakan ujung telunjuk dan jempol dari **kedua** tangan Anda di depan kamera (total 4 jari). Sebuah portal berbentuk persegi empat akan terbentuk di antara keempat jari Anda, dan efek filter akan muncul di dalamnya.
- **Mengganti Filter:** Ada beberapa cara untuk mengganti filter yang sedang aktif:
  - Sentuhkan/dekatkan ujung jempol dan jari kelingking Anda.
  - Atau, dekatkan ujung telunjuk dari kedua tangan Anda.
- **Pindah ke Mode SNAP:** tekan **`m`**.
- **Menutup Aplikasi:** Pastikan jendela kamera/Retrolens sedang aktif (diklik), kemudian tekan tombol **`q`** pada keyboard Anda untuk keluar dari aplikasi. Menutup jendela dengan tombol merah juga bisa.

## Mode SNAP

### Mulai cepat
1. Jalankan `python main.py --snap`.
2. Duduk atau berdiri di depan kamera sampai kepala dan bahu terlihat.
3. Tempelkan ujung jempol ke ujung jari tengah, tahan sebentar, lalu jentikkan sampai jari tengah memukul telapak tangan.

Layar menyala putih, tubuh Anda hancur jadi debu, background terlihat selama 2 detik, lalu kamera kembali normal. Jika jentikan belum terdeteksi, tekan **`S`** atau **Spasi** untuk snap manual.

### Supaya background bersih
Saat Anda menghilang, aplikasi harus tahu seperti apa background di belakang Anda. Ada dua cara:

- **Otomatis.** Aplikasi terus mempelajari bagian background yang terlihat. Persentasenya tampil di pojok kiri atas, misalnya `BG OTOMATIS 82%`. Bagian yang belum pernah terlihat (biasanya tepat di belakang badan) diisi tebakan, jadi terlihat seperti noda. Bergeser sedikit ke kiri dan kanan sebelum snap akan menaikkan persentasenya.
- **Manual (hasil terbaik).** Tekan **`B`**, lalu keluar dari frame selama hitung mundur 3 detik. Aplikasi merekam 30 frame background kosong. Status berubah menjadi `BG TERSIMPAN`.

Rekaman background ditolak jika masih ada orang di frame atau jika gambarnya tidak cocok dengan kamera (misalnya kamera ditutup tangan). Kamera harus diam. Jika kamera bergeser setelah background direkam, aplikasi mendeteksinya saat snap, kembali ke background otomatis, dan meminta Anda menekan `B` lagi.

### Tombol

| Tombol | Fungsi |
| --- | --- |
| `M` | Ganti mode RETROLENS dan SNAP |
| `B` | Rekam background kosong (hitung mundur 3 detik) |
| `S` atau Spasi | Snap manual |
| `T` | Ganti mode trigger: GESTUR, SUARA, ATAU, DAN |
| `K` | Kalibrasi suara jentikan |
| `D` | Overlay debug |
| `V` | Mulai atau berhenti merekam layar |
| `R` | Paksa kembali ke kamera live |
| `Q` | Keluar |

Huruf besar dan kecil sama saja.

### Mode trigger
- **GESTUR** (bawaan): hanya gerakan tangan. Mikrofon tidak dipakai sama sekali.
- **SUARA**: hanya bunyi jentikan. Lakukan kalibrasi dulu.
- **ATAU**: gerakan atau bunyi, salah satu cukup.
- **DAN**: gerakan dan bunyi harus terjadi dalam 0.25 detik. Paling tahan salah picu, misalnya saat Anda banyak bergerak atau ada orang bertepuk tangan.

Bunyi lain yang tajam, seperti tepukan atau ketukan meja, juga bisa terdeteksi sebagai jentikan. Karena itu SUARA saja kurang andal; gunakan DAN jika butuh trigger suara.

### Kalibrasi suara
1. Tekan **`K`** di mode SNAP.
2. Diam selama 2 detik saat layar menulis "diam sebentar".
3. Jentikkan jari 3 kali dalam 6 detik.

Ambang suara disimpan ke `snap_kalibrasi.json` dan dimuat otomatis setiap kali aplikasi dibuka. Selama kalibrasi, jentikan tidak memicu efek. Ulangi kalibrasi jika Anda pindah ruangan atau ganti mikrofon.

### Izin kamera dan mikrofon di macOS
macOS memberi izin ke aplikasi terminal yang menjalankan Python (Terminal, iTerm, VS Code, dan sebagainya), bukan ke Python itu sendiri.

- Kamera: System Settings, Privacy & Security, Camera. Aktifkan untuk aplikasi terminal Anda.
- Mikrofon: System Settings, Privacy & Security, Microphone. Aktifkan untuk aplikasi terminal Anda.

Tutup dan buka lagi terminal setelah mengubah izin. Jika izin mikrofon ditolak, macOS mengirim suara kosong tanpa pesan error, jadi aplikasi menampilkan "Mikrofon diam" di layar. Jika mikrofon tidak ada atau terputus, aplikasi tetap berjalan dengan mode GESTUR.

### Overlay debug (`D`)
Menampilkan mask orang berwarna hijau, titik tangan 0, 4, 9, 12, nilai `cubit` dan `tengah` beserta state tiap tangan (DIAM, SIAP, JENTIK), level suara terhadap ambang, waktu proses tiap tahap dalam milidetik, dan waktu efek yang sedang berjalan.

Jika jentikan Anda sering tidak terdeteksi, perhatikan angkanya:
- `cubit` harus turun di bawah 0.35 selama 0.2 detik (state berubah menjadi SIAP).
- Saat lepas, `cubit` harus naik di atas 0.45 dan `tengah` turun minimal 30% dalam 0.12 detik.

Semua ambang ada di `CONFIG` di `snap_effect.py` (`cubit_masuk`, `cubit_lepas`, `tengah_turun_min`, dan lainnya). Nilai bawaan disetel dari rekaman jentikan sungguhan: 8 dari 8 jentikan terdeteksi dan tidak ada salah picu saat melambai, menunjuk, atau mencubit jempol dengan telunjuk.

### Rekam video (`V`)
Hasilnya tersimpan di `rekaman/snap_TANGGAL_JAM.mp4`. File berisi efek saja, tanpa teks di layar. Titik merah `REC` di pojok kanan atas menandakan sedang merekam. Frame ditulis mengikuti jam, jadi durasi efek di file sama dengan aslinya walaupun FPS naik turun.

### Tips performa
Proses efek di MacBook Apple Silicon hanya sekitar 10 sampai 15 ms per frame, jadi kecepatan biasanya dibatasi kamera. Di ruangan redup, webcam Mac bisa menurunkan frame rate ke sekitar 15 sampai 20 fps; cahaya yang lebih terang mengembalikannya ke 30 fps. Jika FPS di bawah 20 selama 2 detik, ukuran butir debu otomatis naik dari 2 ke 3 piksel mulai snap berikutnya.

### Mode uji tanpa webcam
```bash
# Pakai file video sebagai input kamera
python main.py --sumber "path/video.mp4" --snap

# Render 9 frame efek (detik 0 sampai 3.2) dari satu foto ke foto_lembar.png
python snap_effect.py --uji "path/foto.png"

# Sama, dengan foto background kosong dari sudut kamera yang sama
python snap_effect.py --uji "path/foto.png" --bg "path/kosong.png"

# Tes otomatis
python -m unittest discover -s tests -v
```

### Batasan
- Segmenter tidak bisa memilih satu orang. Semua orang yang terlihat ikut hancur.
- Selama efek, layar memakai frame beku dan background, bukan kamera live, karena Anda sebenarnya masih berdiri di depan kamera.
- Arah angin debu ke kiri atas. Ubah `debu_balik_horizontal` di `CONFIG` menjadi `True` untuk ke kanan atas.

---
**Catatan:** Pastikan ruangan memiliki pencahayaan yang cukup agar deteksi tangan dari kamera dapat bekerja dengan optimal.
