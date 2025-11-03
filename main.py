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

# -------------------------------------------------------
# Utility: JSON-safe serialization for Flask responses
# -------------------------------------------------------
def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# -------------------------------------------------------
# MongoDB Database class (simplified)
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

    def log_mood(self, mood, confidence=0.0, duration=0.0, timestamp=None):
        self._ensure_connection()
        if not self.db:
            return False
        if timestamp is None:
            timestamp = datetime.now()
        doc = {"mood": mood, "confidence": float(confidence), "duration": float(duration), "timestamp": timestamp}
        try:
            with self.lock:
                self.db.moods.insert_one(doc)
            return True
        except Exception as e:
            logging.error("Failed to log mood: %s", e)
            return False

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

    def get_recent_face_moods(self, minutes=60, limit=1000):
        self._ensure_connection()
        if not self.db:
            return []
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            cursor = self.db.face_moods.find({"timestamp": {"$gte": cutoff}}).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logging.error("Failed to get recent face moods: %s", e)
            return []

    def close(self):
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed.")

# -------------------------------------------------------
# TTS (Text-to-Speech)
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

# -------------------------------------------------------
# Database Logger Thread
# -------------------------------------------------------
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

# -------------------------------------------------------
# Flask HTTP server
# -------------------------------------------------------
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
            rows = self.db.get_recent_face_moods(minutes=60)
            return jsonify(json.loads(json.dumps(rows, default=json_safe)))

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)

# -------------------------------------------------------
# Camera thread (simplified)
# -------------------------------------------------------
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture = None
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def run(self):
        self.capture = cv2.VideoCapture(0)
        self.running = True
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.05)

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
# GUI Main Window
# -------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mood Mirror")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setMinimumSize(800, 600)

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
        self.camera_worker.start()

    def closeEvent(self, event):
        """Stop all threads cleanly in correct order"""
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
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
