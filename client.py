import requests
import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime


class CollisionDetectionClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.device_id = f"device_{random.randint(1000, 9999)}"
        print(f"Initialized client with device ID: {self.device_id}")

    def send_data_batch(self, readings):
        """Send a batch of accelerometer readings to the API"""
        data = {
            "device_id": self.device_id,
            "readings": readings
        }

        response = requests.post(f"{self.api_url}/readings/", json=data)
        return response.json()

    def check_collision(self):
        """Check if a collision has been detected"""
        response = requests.get(f"{self.api_url}/detect/{self.device_id}")
        return response.json()

    def simulate_normal_driving(self, duration_seconds=10, frequency=10):
        """Simulate normal driving data and send it to the API"""
        readings = []
        start_time = time.time()

        # Generate simulated normal driving data
        for i in range(duration_seconds * frequency):
            current_time = start_time + (i / frequency)

            # Normal driving has low variance
            x = random.normalvariate(0, 0.2)
            y = random.normalvariate(0, 0.2)
            z = random.normalvariate(1, 0.2)  # Z typically around 1G

            readings.append({
                "timestamp": current_time,
                "x": x,
                "y": y,
                "z": z
            })

        # Send the data
        response = self.send_data_batch(readings)
        print(
            f"Sent {len(readings)} normal driving readings. Response: {response}")

        # Check for collision
        result = self.check_collision()
        print(f"Collision check result: {result}")

        return result

    def simulate_collision(self, duration_seconds=10, frequency=10, collision_time=5):
        """Simulate a driving scenario with a collision"""
        readings = []
        start_time = time.time()

        # Generate simulated data with collision
        for i in range(duration_seconds * frequency):
            current_time = start_time + (i / frequency)

            # Check if this is during the collision
            is_collision = (i >= collision_time * frequency) and (i <
                                                                  (collision_time + 0.5) * frequency)

            if is_collision:
                # Collision has high acceleration/deceleration
                x = random.normalvariate(0, 2.0) + random.choice([-5, 5])
                y = random.normalvariate(0, 2.0)
                z = random.normalvariate(1, 2.0)
            else:
                # Normal driving has low variance
                x = random.normalvariate(0, 0.2)
                y = random.normalvariate(0, 0.2)
                z = random.normalvariate(1, 0.2)  # Z typically around 1G

            readings.append({
                "timestamp": current_time,
                "x": x,
                "y": y,
                "z": z
            })

        # Send the data
        response = self.send_data_batch(readings)
        print(
            f"Sent {len(readings)} readings with collision at {collision_time}s. Response: {response}")

        # Check for collision
        result = self.check_collision()
        print(f"Collision check result: {result}")

        return result

    def load_and_send_csv_data(self, csv_path, batch_size=100):
        """Load data from CSV and send it to the API in batches"""
        # Load CSV
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_columns = ['timestamp', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            return False

        # Process in batches
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:min(i+batch_size, total_rows)]
            readings = batch_df[['timestamp', 'x', 'y', 'z']].to_dict(
                'records')

            # Send batch
            response = self.send_data_batch(readings)
            print(
                f"Sent batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size}. Response: {response}")

            # Pause briefly between batches
            time.sleep(0.1)

        # Check for collision
        result = self.check_collision()
        print(f"Collision check result after sending all data: {result}")

        return result


# Example usage
if __name__ == "__main__":
    # Create client
    client = CollisionDetectionClient()

    # Test normal driving
    print("\n--- Testing normal driving ---")
    client.simulate_normal_driving(duration_seconds=30)

    # Test collision
    print("\n--- Testing collision scenario ---")
    client.simulate_collision(duration_seconds=30, collision_time=15)

    # Alternatively, load data from CSV
    # client.load_and_send_csv_data("sample_data.csv")
