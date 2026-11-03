# Retrolens - Hand Tracking Filter & Portal

Aplikasi filter kamera interaktif menggunakan gesture tangan (hand tracking) dengan MediaPipe dan OpenCV. Anda dapat membuat "portal" dengan jari Anda yang akan menerapkan berbagai filter menarik (Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon, Galaxy).

There is also a **SNAP mode**: snap your fingers and you turn into dust and vanish, with the room behind you showing through. Snap again to come back. See [SNAP mode](#snap-mode).

## Quick start (macOS)

1. Open the **Terminal** app.
2. Go to the project folder. Type `cd `, drag the `python-handtrack` folder into the Terminal window, and press Enter.
3. Start the app:
   ```bash
   ./run.sh
   ```
   The first time it sets everything up by itself (a few minutes). After that it starts right away.
4. If macOS asks for camera access, click **Allow**. If the camera window stays black, open System Settings, Privacy & Security, Camera, turn on your Terminal app, quit Terminal, and run `./run.sh` again.
5. **Let it learn the room:** step out of the camera view until the top left says **Ready** in green.
6. **Vanish:** sit in front of the camera and snap your fingers (thumb and middle finger). Pressing **S** works too.
7. **Come back:** snap again, or show your open palm for 1 second.
8. **Quit:** press **Q** in the camera window.

Press **H** in the app for all gestures and keys. The full guide is in [SNAP mode](#snap-mode).

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

## SNAP mode

### What it does

Snap your fingers and you turn into dust and vanish. The room behind you shows through as if you were never there. Snap again or show your open palm to come back.

### One time setup

The easy way is `./run.sh` (see [Quick start](#quick-start-macos)): it does all the steps below by itself. To do them by hand:

1. Open Terminal in the project folder and activate the environment: `source venv/bin/activate`
2. Install packages: `pip install -r requirements.txt`
3. Download the gesture model into the project folder (one time):
   ```bash
   curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
   ```
   If that link ever fails, get the file from the Models section of the official MediaPipe Gesture Recognizer page. Without it the app still runs, but only the snap gesture works.
4. Allow camera and microphone for your Terminal app in System Settings, Privacy and Security. macOS gives the permission to the terminal app (Terminal, iTerm, VS Code), not to Python. Quit and reopen the terminal after changing it.
5. Start: `python main.py --snap`

### How to use

1. **Place the camera somewhere stable.** It must not move during use.
2. **Let it learn the room.** Step out of the frame for 2 seconds. The status shows "Ready" in green when it is done. You can also show a thumbs up for 1 second (or press B) and step out during the countdown.
3. **Sit or stand in front of the camera.**
4. **Snap your fingers** (thumb and middle finger). You vanish into dust.
5. **Walk around while invisible.** You stay invisible.
6. **Come back:** snap again, or hold your open palm toward the camera for 1 second.

Gestures:

* Snap: vanish, or come back when invisible
* Open palm, 1 s: come back
* Thumbs up, 1 s: learn the room again
* Peace sign, 1 s: next effect (Dust, Ash, Ember, Glitch)
* Fist, hold: freeze time during the effect
* Pointing up, hold: slow motion during the effect
* Rock sign (thumb, index, and pinky up), 1 s: show or hide help
* Thumbs down, 1 s: cancel and go back to normal

A ring around your hand fills up while you hold a gesture. When it is full, the action happens. Freeze and slow motion work once the white flash has faded, and they ease in and out instead of stopping at once.

Keys: M mode, S or Space snap, B learn room, 1 to 4 pick an effect, X next effect, E effects menu, H help, T trigger mode, K sound calibration, D debug, V record, Shift+V save frames, R reset, Q quit.

### Effects

Pick an effect with the number keys, cycle with X or the peace sign, or press E for a small menu that shows which one is on. You can switch while you are visible or invisible; the status pill shows the current one.

1. **Dust:** you crumble into film-like dust.
2. **Ash:** dark gray ash that falls slower and sinks.
3. **Ember:** a warm glowing edge, embers fading out.
4. **Glitch:** the Spider-Verse look. While you are visible, your body glitches in short bursts: it gets sliced and shifted, recolored in flat colors, doubled in see-through pink and cyan, and now and then the whole frame flashes color regions, halftone dots, or star bursts. The glitch changes 12 times a second ("on twos" at 24 fps), while the camera picture under it stays live. It follows the timing of the reference animation, starting with half a second of calm. Learn the room first (step out until it says Ready): until then the shifted pieces cover your body instead of showing the room behind you. Snapping with Glitch on glitches you out in under a second: your body breaks into recolored pieces, the lower part goes first, and a last strip across your eyes flickers before you are gone. Coming back plays it the other way, from a piece of your head to all of you, and ends on the plain camera picture. Fist and pointing up freeze and slow the glitch too.

### Snap by sound

Press T to choose what counts as a snap: gesture (the default, the microphone is not used), sound, gesture or sound, or gesture and sound (both within 0.25 s, the hardest to trigger by accident). Other sharp sounds such as clapping can also count as a snap, so sound alone is the least reliable.

For sound, press K once: stay quiet for 2 seconds, then snap 3 times within 6 seconds. The result is saved in `snap_kalibrasi.json` and loaded every time. Do it again after moving to another room or changing microphones. If macOS blocks the microphone it sends silence without an error, so the app shows "The microphone hears nothing".

### Recording

* **V** records the effect to `rekaman/snap_DATE_TIME.mp4` (H.264 when available). The status pill shows REC.
* **Shift+V** saves every shown frame as a lossless PNG in `rekaman/frames_DATE_TIME/`, for checking the effect frame by frame.

Recordings never contain the status pill, hints, help card, or debug views.

### If something looks wrong

* **"I still need to see the room behind you":** step out of the frame for 2 seconds, then snap again. A light blue tint shows which part is still missing.
* **"Camera moved":** keep the camera still and step out for a moment so it can relearn the room.
* **"I can't see behind you there":** you walked onto a part of the room the camera has never seen. That part fades back to the live picture until it is learned. Step aside for a moment so it can see it.
* **Snap not detected:** snap a little slower with your palm facing the camera, or press D to see the live values.
* **Low fps:** close other apps that use the camera, and use good lighting. In a dim room the Mac camera itself drops to about 15 to 20 fps.

### Debug views (D)

D cycles through: off, stats, matte, removal zone, coverage, gain map. Stats shows the hand values (`cubit` must drop below 0.35 for 0.2 s to get ready, then rise above 0.45 while `tengah` drops by at least 30% within 0.12 s), the recognized gesture and score per hand, the sound level, and the time per stage in milliseconds. The other views draw the person matte, the area that is replaced by the room (orange for shadows), the part of the room that has not been seen yet (red), and the exposure match.

All settings live in `CONFIG` in `snap/config.py`, and the app checks them on start.

### Speed

Measured on an Apple silicon MacBook at 1280x720, without drawing the window: 29 to 30 fps in every state, which is the camera's own rate, with the hand and person models running on a background thread. `python tools/benchmark.py CLIP` measures it again; the status pill shows the live value. If the frame rate stays below 24 fps for 2 seconds, the app steps down to cheaper settings one at a time (coarser matte, bigger dust, no dust blur, no shadow removal) and steps back up after 5 seconds above 28 fps. A change is used from the next vanish or reappear on, never in the middle of one.

### For developers

```bash
# Automatic tests
python -m unittest discover -s tests

# Use a video file as the camera (record clips with tools/record_clip.py, they are mirrored like the app)
python main.py --sumber "path/video.mp4" --snap

# Render the effect from a clip without a camera, then check it
python tools/render_offline.py uji_lokal/v2/sit_snap.mp4 --bg uji_lokal/v2/room_empty.mp4 --snap-at 3 --snap-at 10 --out uji_lokal/render/sit_snap
python tools/measure_quality.py uji_lokal/render/sit_snap

# Frame rate per state, the way the app runs
python tools/benchmark.py uji_lokal/v2/sit_snap.mp4 --bg uji_lokal/v2/room_empty.mp4

# Preview sheet of 9 effect frames from one photo (with an optional empty room photo from the same spot)
python snap_effect.py --uji "path/photo.png" --bg "path/empty.png"
```

Recording test clips: keep the camera in the same position for all clips, then run these one at a time:

1. `python tools/record_clip.py room_empty 10` (nobody in the frame)
2. `python tools/record_clip.py sit_still 12` (walk in, sit, stay still)
3. `python tools/record_clip.py sit_snap 12` (sit, snap twice)
4. `python tools/record_clip.py move_around 15` (move around the frame)
5. `python tools/record_clip.py leave_return 12` (leave for 3 s, come back)

### Known limits

1. The room behind you must be seen at least once. Areas never seen cannot be rebuilt, which is why the app asks you to step out instead of guessing.
2. The person segmenter cannot tell people apart. Everyone in the frame vanishes together.
3. The room must be still where you stand. Moving things behind you (a TV, a curtain in the wind) can show as a frozen patch while you are invisible.
4. Light changes during the effect are matched only from 0.7x to 1.4x brightness per color channel. Beyond that, the edge of the hidden area may become visible.
5. To keep the frame rate up, the picture is shown one camera frame (about 33 ms) late.
6. The quality measurements inside the hidden area have only been run on compressed test clips with a stand-in room. There the flicker check passes, right at its limit (the hidden area changes no more than the live camera itself), and the noise match does not pass yet. Both still need real `room_empty` and `sit_snap` recordings.

---
**Catatan:** Pastikan ruangan memiliki pencahayaan yang cukup agar deteksi tangan dari kamera dapat bekerja dengan optimal.
