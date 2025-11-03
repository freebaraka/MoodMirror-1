# -*- coding: ascii -*-
import os
import sys
import cv2
import time
import json
import queue
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from deepface import DeepFace
from flask import Flask, jsonify
from PyQt5 import QtCore, QtGui, QtWidgets
from pymongo import MongoClient
import pyttsx3

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MoodMirror")

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5000

def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# -------------------------------------------------------
# Database
# -------------------------------------------------------
class Database:
    def __init__(self, host="localhost", port=27017, database="emoji_tracker", reconnect_attempts=3, reconnect_delay=1.0):
        self.host = host
        self.port = port
        self.database_name = database
        self.client = None
        self.db = None
        self.lock = threading.Lock()
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

    def connect(self):
        attempt = 0
        while attempt < self.reconnect_attempts:
            try:
                self.client = MongoClient(self.host, self.port, serverSelectionTimeoutMS=2000)
                self.client.server_info()
                self.db = self.client[self.database_name]
                logging.info("Connected to MongoDB database.")
                return
            except Exception as err:
                logging.warning("MongoDB connect attempt %d failed: %s", attempt + 1, err)
                self.client = None
                attempt += 1
                time.sleep(self.reconnect_delay)
        logging.error("Exceeded maximum MongoDB connect attempts.")

    def _ensure_connection(self):
        if self.client is None:
            self.connect()

    def log_face_mood(self, person_name, emotion, confidence=0.0, duration=0.0, timestamp=None):
        self._ensure_connection()
        if not self.db:
            return False
        if timestamp is None:
            timestamp = datetime.now()
        doc = {"person_name": person_name, "emotion": emotion, "confidence": float(confidence),
               "duration": float(duration), "timestamp": timestamp}
        try:
            with self.lock:
                self.db.face_moods.insert_one(doc)
            return True
        except Exception as e:
            logging.error("Failed to log face mood: %s", e)
            return False

    def close(self):
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed.")

# -------------------------------------------------------
# Threads: TTS, DBLogger, HttpServer
# -------------------------------------------------------
class TTS(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.engine = pyttsx3.init()
        self.queue = queue.Queue()
        self.running = True

    def run(self):
        while self.running:
            try:
                text = self.queue.get(timeout=1)
                if text:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except queue.Empty:
                continue

    def speak(self, text):
        self.queue.put(text)

    def stop(self):
        self.running = False
        try:
            self.engine.stop()
        except Exception:
            pass


class DBLogger(threading.Thread):
    def __init__(self, db):
        super().__init__(daemon=True)
        self.db = db
        self.queue = queue.Queue()
        self.running = True

    def enqueue_face_mood(self, person_name, emotion, confidence, duration):
        if self.db:
            self.queue.put((person_name, emotion, confidence, duration))

    def run(self):
        while self.running:
            try:
                person_name, emotion, conf, dur = self.queue.get(timeout=1)
                self.db.log_face_mood(person_name, emotion, conf, dur)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


class HttpServer(threading.Thread):
    def __init__(self, host, port, db):
        super().__init__(daemon=True)
        self.app = Flask(__name__)
        self.db = db
        self.host = host
        self.port = port
        self._build_app()

    def _build_app(self):
        @self.app.route("/api/recent_face_moods", methods=["GET"])
        def recent_face_moods():
            rows = self.db.db.face_moods.find().sort("timestamp", -1).limit(20)
            return jsonify(json.loads(json.dumps(list(rows), default=json_safe)))

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)

# -------------------------------------------------------
# Camera Worker
# -------------------------------------------------------
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture = None
        self.last_analysis = time.time()

    def run(self):
        self.capture = cv2.VideoCapture(0)
        self.running = True
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.05)
                continue

            emotion_data = {}
            now = time.time()
            if now - self.last_analysis > 3:  # analyze every 3 seconds
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion_data = result[0] if isinstance(result, list) else result
                    self.last_analysis = now
                except Exception as e:
                    logging.warning("DeepFace analysis failed: %s", e)

            self.frame_ready.emit(frame, emotion_data)

    def stop(self):
        self.running = False
        try:
            if self.capture:
                self.capture.release()
            self.wait(1000)
            cv2.destroyAllWindows()
        except Exception:
            pass

# -------------------------------------------------------
# Main Window - Camera + Mood Image + Text
# -------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.setMinimumSize(950, 700)

        # --- Layout ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Horizontal split: camera + mood image
        hbox = QtWidgets.QHBoxLayout()
        layout.addLayout(hbox, stretch=4)

        # Camera feed
        self.video_label = QtWidgets.QLabel("Starting camera...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        hbox.addWidget(self.video_label, stretch=3)

        # Mood image
        self.mood_image = QtWidgets.QLabel()
        self.mood_image.setAlignment(QtCore.Qt.AlignCenter)
        self.mood_image.setStyleSheet("background-color: #222;")
        hbox.addWidget(self.mood_image, stretch=1)

        # Emotion text below
        self.emotion_label = QtWidgets.QLabel("Detecting emotion...")
        self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.emotion_label.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan; margin-top: 10px;")
        layout.addWidget(self.emotion_label, stretch=1)

        # --- Threads ---
        self.db = Database()
        try:
            self.db.connect()
        except Exception:
            logger.exception("DB connect failed")
            self.db = None

        self.db_logger = DBLogger(self.db)
        self.db_logger.start()

        self.tts = TTS()
        self.tts.start()

        self.http_server = HttpServer(HTTP_HOST, HTTP_PORT, self.db)
        self.http_server.start()

        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.update_display)
        self.camera_worker.start()

        self.statusBar().showMessage("Camera running...")

        # Cache for last emotion to avoid repeating same TTS
        self.last_emotion = None

    def update_display(self, frame, emotion_data):
        """Display camera, text, and mood image"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

        if emotion_data:
            dominant = emotion_data.get("dominant_emotion", "Unknown").capitalize()
            confidence = emotion_data.get("emotion", {}).get(dominant.lower(), 0)
            self.emotion_label.setText(f"Detected Emotion: {dominant} ({confidence:.1f}%)")

            # Load corresponding mood image
            image_path = f"{dominant.lower()}.png"
            if os.path.exists(image_path):
                pixmap = QtGui.QPixmap(image_path)
                self.mood_image.setPixmap(
                    pixmap.scaled(self.mood_image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                )
            else:
                self.mood_image.clear()

            # Speak & log only if emotion changed
            if dominant != self.last_emotion:
                self.tts.speak(f"You look {dominant}")
                self.db_logger.enqueue_face_mood("User", dominant, float(confidence), 0)
                self.last_emotion = dominant

    def closeEvent(self, event):
        try:
            self.camera_worker.stop()
            self.db_logger.stop()
            self.tts.stop()
            if self.db:
                self.db.close()
        except Exception:
            pass
        event.accept()

# -------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
