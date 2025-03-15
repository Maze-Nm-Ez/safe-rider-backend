# Vehicle Collision Detection System

A real-time collision detection system that processes accelerometer data from vehicles to detect potential collisions using machine learning.

## Overview

This system provides a FastAPI-based REST API service that:

- Processes streaming accelerometer data from multiple vehicles
- Uses a TensorFlow-based autoencoder model for anomaly detection
- Provides real-time collision detection through a simple API interface
- Supports multiple devices simultaneously with thread-safe operations

## Features

- Real-time accelerometer data processing
- Sliding window-based analysis with configurable parameters
- Anomaly detection using deep learning
- Thread-safe device buffer management
- RESTful API with FastAPI
- CORS support for web integration
- Health monitoring endpoints
- Device status tracking

## Requirements

- Python 3.12+
- Dependencies (managed by Poetry):
  - TensorFlow 2.18.0+
  - FastAPI
  - NumPy
  - Pandas
  - Joblib
  - scikit-learn
  - uvicorn
  - pydantic

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd vehicle-collision-detection
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Modify `config.json` according to your needs:

```json
{
  "window_length": 60,
  "segment_length": 20,
  "overlap": 0.5,
  "sampling_rate": 10,
  "model_path": "collision_detection_model.h5",
  "scaler_path": "min_max_scaler.pkl",
  "anomaly_threshold": 0.1
}
```

## Usage

1. Start the server:

```bash
poetry run python main.py
```

The server will start on `http://localhost:8000`

2. API Endpoints:

- `POST /readings/`: Submit new accelerometer readings

```json
{
  "device_id": "vehicle123",
  "readings": [
    {
      "timestamp": 1234567890.123,
      "x": 0.1,
      "y": -0.2,
      "z": 9.8
    }
  ]
}
```

- `GET /detect/{device_id}`: Check for collisions
- `GET /health`: Check service health
- `GET /config`: Get current configuration
- `GET /devices`: List connected devices

## API Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration Parameters

- `window_length`: Duration of the analysis window in seconds
- `segment_length`: Length of each data segment in seconds
- `overlap`: Overlap between consecutive segments (0.0-1.0)
- `sampling_rate`: Expected data sampling rate in Hz
- `model_path`: Path to the TensorFlow model file
- `scaler_path`: Path to the scikit-learn scaler file
- `anomaly_threshold`: Threshold for collision detection

## Model Details

The system uses an autoencoder neural network to detect anomalies in the accelerometer data:

- Input: Segments of 3-axis accelerometer data
- Processing: Data normalization and segmentation
- Output: Reconstruction error-based anomaly detection

## Development

To run in development mode with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

## Authors

[Your Name/Organization]
