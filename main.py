import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import asyncio
import tensorflow as tf
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque
from threading import Lock

# Load TensorFlow model quietly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Register custom loss and metric functions to fix model loading issues
tf.keras.losses.mse = tf.keras.losses.MeanSquaredError()
tf.keras.metrics.mse = tf.keras.metrics.MeanSquaredError()


class AccelerometerReading(BaseModel):
    timestamp: float
    x: float
    y: float
    z: float


class AccelerometerBatch(BaseModel):
    device_id: str
    readings: List[AccelerometerReading]


class CollisionDetectionService:
    def __init__(self, config_path='config.json'):
        # Load configuration
        print("Initializing Collision Detection Service...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize parameters from config
        self.window_length = self.config.get('window_length', 60)  # in seconds
        self.segment_length = self.config.get(
            'segment_length', 20)  # in seconds
        self.overlap = self.config.get('overlap', 0.5)  # 50% overlap
        self.sampling_rate = self.config.get('sampling_rate', 10)  # Hz
        self.model_path = self.config.get(
            'model_path', 'collision_detection_model.h5')
        self.scaler_path = self.config.get('scaler_path', 'min_max_scaler.pkl')
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.1)

        # Derived parameters
        self.samples_per_window = self.window_length * self.sampling_rate
        self.samples_per_segment = self.segment_length * self.sampling_rate
        self.stride = int(self.samples_per_segment * (1 - self.overlap))

        # Load model and scaler
        print(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                'mse': tf.keras.losses.MeanSquaredError(),
                'MSE': tf.keras.losses.MeanSquaredError(),
            }
        )

        print(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

        # Initialize device buffers
        self.device_buffers = {}
        self.buffer_locks = {}

        print("Collision Detection Service initialized successfully!")

    def get_or_create_buffer(self, device_id):
        """Get the buffer for a device or create a new one if it doesn't exist"""
        if device_id not in self.device_buffers:
            self.device_buffers[device_id] = deque(
                maxlen=self.samples_per_window)
            self.buffer_locks[device_id] = Lock()
        return self.device_buffers[device_id], self.buffer_locks[device_id]

    def add_readings(self, device_id, readings):
        """Add new accelerometer readings to the device buffer"""
        buffer, lock = self.get_or_create_buffer(device_id)

        with lock:
            for reading in readings:
                buffer.append([reading.x, reading.y, reading.z])

    def preprocess_buffer(self, buffer):
        """Convert buffer to numpy array and scale it"""
        if len(buffer) < self.samples_per_segment:
            return None

        # Convert to numpy array
        data = np.array(list(buffer))

        # Scale the data
        scaled_data = self.scaler.transform(data)

        return scaled_data

    def create_segments(self, data):
        """Create overlapping segments from the preprocessed data"""
        if data is None or len(data) < self.samples_per_segment:
            return []

        segments = []
        for i in range(0, len(data) - self.samples_per_segment + 1, self.stride):
            segment = data[i:i + self.samples_per_segment]
            segments.append(segment)

        return np.array(segments)

    def detect_collision(self, device_id):
        """Run collision detection on the current buffer for a device"""
        buffer, lock = self.get_or_create_buffer(device_id)

        with lock:
            if len(buffer) < self.samples_per_segment:
                return {"collision_detected": False, "reason": "Insufficient data"}

            # Preprocess buffer
            preprocessed_data = self.preprocess_buffer(buffer)

            # Create segments
            segments = self.create_segments(preprocessed_data)

            if len(segments) == 0:
                return {"collision_detected": False, "reason": "No segments created"}

            # Run inference
            predictions = self.model.predict(segments)

            # Calculate reconstruction errors
            mse = np.mean(np.square(segments - predictions), axis=(1, 2))

            # Check if any segment exceeds threshold
            max_error = np.max(mse)
            collision_detected = bool(max_error > self.anomaly_threshold)

            result = {
                "collision_detected": collision_detected,
                "max_error": float(max_error),
                "threshold": float(self.anomaly_threshold),
                "segments_analyzed": int(len(segments)),
                "buffer_size": int(len(buffer))
            }

            return result


# Initialize FastAPI app
app = FastAPI(title="Vehicle Collision Detection System",
              description="API for real-time collision detection using accelerometer data",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize collision detection service
detection_service = None


@app.on_event("startup")
async def startup_event():
    global detection_service
    detection_service = CollisionDetectionService()

# Endpoint to receive accelerometer data


@app.post("/readings/")
async def add_readings(batch: AccelerometerBatch, background_tasks: BackgroundTasks):
    """
    Add new accelerometer readings and process them asynchronously
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Add readings to buffer
    detection_service.add_readings(batch.device_id, batch.readings)

    return {"status": "success", "message": f"Added {len(batch.readings)} readings for device {batch.device_id}"}

# Endpoint to check for collisions


@app.get("/detect/{device_id}")
async def detect_collision(device_id: str):
    """
    Run collision detection for a specific device
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Check if device exists
    if device_id not in detection_service.device_buffers:
        raise HTTPException(
            status_code=404, detail=f"Device {device_id} not found")

    # Run detection
    result = detection_service.detect_collision(device_id)

    return result

# Health check endpoint


@app.get("/health")
async def health_check():
    """
    Check if the service is running
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_path": detection_service.model_path,
        "window_length": detection_service.window_length,
        "segment_length": detection_service.segment_length,
        "threshold": detection_service.anomaly_threshold
    }

# Configuration endpoint


@app.get("/config")
async def get_config():
    """
    Get the current configuration
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return detection_service.config

# Device information endpoint


@app.get("/devices")
async def get_devices():
    """
    Get information about all connected devices
    """
    if not detection_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    devices = {}
    for device_id in detection_service.device_buffers:
        devices[device_id] = {
            "buffer_size": len(detection_service.device_buffers[device_id]),
            "max_buffer_size": detection_service.samples_per_window
        }

    return {"devices": devices}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
