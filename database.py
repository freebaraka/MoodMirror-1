# -*- coding: ascii -*-
from pymongo import MongoClient
from datetime import datetime, timedelta
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)

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
                # Force a connection test
                self.client.server_info()

                self.db = self.client[self.database_name]

                # Collections will be created automatically when data is inserted
                logging.info("Connected to MongoDB database.")
                return
            except Exception as err:
                logging.warning("MongoDB connect attempt %d failed: %s", attempt + 1, err)
                self.client = None
                attempt += 1
                time.sleep(self.reconnect_delay)

        logging.error("Exceeded maximum MongoDB connect attempts; continuing without DB connection.")

    def _ensure_connection(self):
        if self.client is None:
            self.connect()

    def log_mood(self, mood, confidence=0.0, duration=0.0, timestamp=None):
        self._ensure_connection()
        if timestamp is None:
            timestamp = datetime.now()
        doc = {
            "mood": mood,
            "confidence": float(confidence),
            "duration": float(duration),
            "timestamp": timestamp
        }
        try:
            with self.lock:
                self.db.moods.insert_one(doc)
            logging.debug("Logged mood: %s", doc)
            return True
        except Exception as e:
            logging.error("Failed to log mood: %s", e)
            return False

    def log_face_mood(self, person_name, emotion, confidence=0.0, duration=0.0, timestamp=None):
        self._ensure_connection()
        if timestamp is None:
            timestamp = datetime.now()
        doc = {
            "person_name": person_name,
            "emotion": emotion,
            "confidence": float(confidence),
            "duration": float(duration),
            "timestamp": timestamp
        }
        try:
            with self.lock:
                self.db.face_moods.insert_one(doc)
            logging.debug("Logged face_mood: %s", doc)
            return True
        except Exception as e:
            logging.error("Failed to log face mood: %s", e)
            return False

    def get_recent_face_moods(self, minutes=60, limit=1000):
        self._ensure_connection()
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            cursor = (
                self.db.face_moods
                .find({"timestamp": {"$gte": cutoff}})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:
            logging.error("Failed to get recent face moods: %s", e)
            return []

    def close(self):
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed.")
