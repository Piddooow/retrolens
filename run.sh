#!/bin/bash
# One command to try Retrolens SNAP: sets everything up the first time, then starts the app.
# Usage: ./run.sh            (extra options are passed on, for example ./run.sh --logs)
set -e
cd "$(dirname "$0")"

MODEL_URL="https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"

if [ ! -x venv/bin/python ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is needed. Install it from https://www.python.org/downloads/ and run ./run.sh again."
    exit 1
  fi
  echo "First run: creating the Python environment..."
  python3 -m venv venv
fi

if ! venv/bin/python -c "import cv2, mediapipe, numpy, sounddevice; cv2.ximgproc" >/dev/null 2>&1; then
  echo "Installing the packages (one time, takes a few minutes)..."
  venv/bin/python -m pip install --quiet --upgrade pip
  venv/bin/python -m pip install --quiet -r requirements.txt
fi

if [ ! -f gesture_recognizer.task ]; then
  echo "Downloading the gesture model (one time)..."
  if curl -fsSL --retry 2 -o gesture_recognizer.task.part "$MODEL_URL"; then
    mv gesture_recognizer.task.part gesture_recognizer.task
  else
    rm -f gesture_recognizer.task.part
    echo "Could not download the gesture model. The app still runs, but only the snap gesture works."
  fi
fi

echo "Starting. Step out of the frame until the top left says Ready, then come back and snap."
echo "Press Q in the camera window to quit."
exec venv/bin/python main.py --snap "$@"
