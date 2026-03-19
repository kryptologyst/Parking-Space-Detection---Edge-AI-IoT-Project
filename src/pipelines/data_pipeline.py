"""
Data pipeline and sensor simulation for parking space detection.

This module provides streaming I/O, sensor data simulation, and edge-constrained
data processing for real-time parking space detection.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Represents a single sensor reading from parking space sensors."""
    timestamp: float
    spot_id: str
    distance: float  # meters from ultrasonic sensor
    lighting: float  # lux level
    motion_detected: bool
    temperature: float  # ambient temperature
    humidity: float  # relative humidity
    occupancy: Optional[bool] = None  # ground truth (for training)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorReading":
        """Create from dictionary."""
        return cls(**data)

class SensorSimulator:
    """Simulates parking space sensors for testing and development."""
    
    def __init__(
        self,
        num_spots: int = 10,
        sampling_rate: float = 1.0,  # Hz
        noise_level: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Initialize sensor simulator.
        
        Args:
            num_spots: Number of parking spots to simulate
            sampling_rate: Data sampling rate in Hz
            noise_level: Amount of noise to add to sensor readings
            seed: Random seed for reproducibility
        """
        self.num_spots = num_spots
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
        
        # Initialize spot states
        self.spot_states = {
            f"spot_{i:02d}": {
                "occupied": self.rng.choice([True, False]),
                "last_motion": 0.0,
                "base_distance": self.rng.uniform(0.5, 2.0),
                "base_lighting": self.rng.uniform(200, 400),
            }
            for i in range(num_spots)
        }
        
        self.current_time = time.time()
    
    def generate_reading(self, spot_id: str) -> SensorReading:
        """
        Generate a sensor reading for a specific spot.
        
        Args:
            spot_id: Identifier for the parking spot
            
        Returns:
            Sensor reading with simulated data
        """
        if spot_id not in self.spot_states:
            raise ValueError(f"Unknown spot ID: {spot_id}")
        
        state = self.spot_states[spot_id]
        self.current_time = time.time()
        
        # Simulate occupancy changes (cars entering/leaving)
        if self.rng.random() < 0.01:  # 1% chance of state change per reading
            state["occupied"] = not state["occupied"]
            state["last_motion"] = self.current_time
        
        # Generate sensor readings based on occupancy
        if state["occupied"]:
            # Car present: shorter distance, lower lighting (shadow)
            distance = state["base_distance"] * 0.3 + self.rng.normal(0, 0.1)
            lighting = state["base_lighting"] * 0.6 + self.rng.normal(0, 20)
        else:
            # No car: normal distance, normal lighting
            distance = state["base_distance"] + self.rng.normal(0, 0.1)
            lighting = state["base_lighting"] + self.rng.normal(0, 30)
        
        # Add noise
        distance += self.rng.normal(0, self.noise_level * 0.1)
        lighting += self.rng.normal(0, self.noise_level * 50)
        
        # Ensure realistic bounds
        distance = max(0.1, min(distance, 5.0))
        lighting = max(0, min(lighting, 1000))
        
        # Motion detection (recent state change)
        motion_detected = (self.current_time - state["last_motion"]) < 5.0
        
        # Environmental conditions
        temperature = 20.0 + self.rng.normal(0, 2.0)  # Room temperature
        humidity = 50.0 + self.rng.normal(0, 10.0)  # Relative humidity
        
        return SensorReading(
            timestamp=self.current_time,
            spot_id=spot_id,
            distance=distance,
            lighting=lighting,
            motion_detected=motion_detected,
            temperature=temperature,
            humidity=humidity,
            occupancy=state["occupied"],
        )
    
    async def stream_readings(
        self,
        duration: Optional[float] = None,
    ) -> AsyncGenerator[SensorReading, None]:
        """
        Stream sensor readings asynchronously.
        
        Args:
            duration: Duration to stream (None for infinite)
            
        Yields:
            Sensor readings at the specified sampling rate
        """
        start_time = time.time()
        interval = 1.0 / self.sampling_rate
        
        while True:
            if duration and (time.time() - start_time) > duration:
                break
            
            # Generate readings for all spots
            for spot_id in self.spot_states.keys():
                reading = self.generate_reading(spot_id)
                yield reading
            
            # Wait for next sampling interval
            await asyncio.sleep(interval)

class StreamingDataset(Dataset):
    """Dataset that handles streaming sensor data with edge constraints."""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        sequence_length: int = 1,
        features: List[str] = None,
    ) -> None:
        """
        Initialize streaming dataset.
        
        Args:
            buffer_size: Maximum number of samples to keep in memory
            sequence_length: Length of input sequences (for time series)
            features: List of feature names to include
        """
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.features = features or ["distance", "lighting", "motion_detected", "hour"]
        
        self.buffer = deque(maxlen=buffer_size)
        self.lock = asyncio.Lock()
    
    def add_reading(self, reading: SensorReading) -> None:
        """Add a new sensor reading to the buffer."""
        # Convert reading to feature vector
        hour = time.localtime(reading.timestamp).tm_hour
        feature_vector = [
            reading.distance,
            reading.lighting,
            float(reading.motion_detected),
            float(hour),
        ]
        
        label = float(reading.occupancy) if reading.occupancy is not None else -1
        
        self.buffer.append({
            "features": feature_vector,
            "label": label,
            "timestamp": reading.timestamp,
            "spot_id": reading.spot_id,
        })
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        if idx >= len(self.buffer):
            raise IndexError("Index out of range")
        
        sample = self.buffer[idx]
        features = torch.FloatTensor(sample["features"])
        label = torch.LongTensor([int(sample["label"])])
        
        return features, label
    
    def get_latest_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the latest batch of samples."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        latest_samples = list(self.buffer)[-batch_size:]
        
        features = torch.FloatTensor([s["features"] for s in latest_samples])
        labels = torch.LongTensor([int(s["label"]) for s in latest_samples])
        
        return features, labels

class MQTTDataCollector:
    """Collects sensor data via MQTT for real-time processing."""
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "parking/sensors",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """
        Initialize MQTT data collector.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            topic_prefix: Topic prefix for sensor data
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        
        self.client = mqtt.Client()
        self.dataset = StreamingDataset()
        
        # Set up MQTT callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        if username and password:
            self.client.username_pw_set(username, password)
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """Handle MQTT connection."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to all sensor topics
            topic = f"{self.topic_prefix}/+"
            client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Code: {rc}")
    
    def _on_message(self, client, userdata, msg) -> None:
        """Handle incoming MQTT messages."""
        try:
            # Parse JSON message
            data = json.loads(msg.payload.decode())
            
            # Create sensor reading
            reading = SensorReading.from_dict(data)
            
            # Add to dataset
            self.dataset.add_reading(reading)
            
            logger.debug(f"Received reading from {reading.spot_id}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc) -> None:
        """Handle MQTT disconnection."""
        logger.info("Disconnected from MQTT broker")
    
    def connect(self) -> None:
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
    
    def get_dataset(self) -> StreamingDataset:
        """Get the streaming dataset."""
        return self.dataset

class EdgeDataProcessor:
    """Processes sensor data with edge device constraints."""
    
    def __init__(
        self,
        max_batch_size: int = 1,
        max_sequence_length: int = 1,
        enable_preprocessing: bool = True,
    ) -> None:
        """
        Initialize edge data processor.
        
        Args:
            max_batch_size: Maximum batch size for edge inference
            max_sequence_length: Maximum sequence length for processing
            enable_preprocessing: Whether to enable data preprocessing
        """
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.enable_preprocessing = enable_preprocessing
        
        # Preprocessing parameters
        self.feature_stats = {
            "distance": {"mean": 1.0, "std": 0.5},
            "lighting": {"mean": 300.0, "std": 100.0},
            "motion_detected": {"mean": 0.15, "std": 0.36},
            "hour": {"mean": 12.0, "std": 6.93},
        }
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features for edge inference.
        
        Args:
            features: Raw feature array
            
        Returns:
            Preprocessed feature array
        """
        if not self.enable_preprocessing:
            return features
        
        processed = features.copy()
        
        # Normalize features
        for i, feature_name in enumerate(["distance", "lighting", "motion_detected", "hour"]):
            if i < len(processed):
                stats = self.feature_stats[feature_name]
                processed[i] = (processed[i] - stats["mean"]) / stats["std"]
        
        return processed
    
    def process_batch(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process a batch of data for edge inference.
        
        Args:
            features: Input features
            labels: Optional labels
            
        Returns:
            Tuple of (processed_features, processed_labels)
        """
        # Ensure batch size constraint
        if features.size(0) > self.max_batch_size:
            features = features[:self.max_batch_size]
            if labels is not None:
                labels = labels[:self.max_batch_size]
        
        # Preprocess features
        if self.enable_preprocessing:
            processed_features = []
            for i in range(features.size(0)):
                feature_array = features[i].numpy()
                processed_array = self.preprocess_features(feature_array)
                processed_features.append(processed_array)
            
            features = torch.FloatTensor(processed_features)
        
        return features, labels
    
    def create_edge_batch(self, readings: List[SensorReading]) -> torch.Tensor:
        """
        Create a batch suitable for edge inference from sensor readings.
        
        Args:
            readings: List of sensor readings
            
        Returns:
            Batch tensor for inference
        """
        if not readings:
            return torch.empty(0, 4)
        
        # Limit batch size
        readings = readings[:self.max_batch_size]
        
        features = []
        for reading in readings:
            hour = time.localtime(reading.timestamp).tm_hour
            feature_vector = [
                reading.distance,
                reading.lighting,
                float(reading.motion_detected),
                float(hour),
            ]
            features.append(feature_vector)
        
        batch = torch.FloatTensor(features)
        processed_batch, _ = self.process_batch(batch)
        
        return processed_batch

class DataPipeline:
    """Complete data pipeline for parking space detection."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """
        Initialize data pipeline.
        
        Args:
            config: Pipeline configuration
            device: Device for processing
        """
        self.config = config
        self.device = device
        
        # Initialize components
        self.simulator = SensorSimulator(
            num_spots=config.get("num_spots", 10),
            sampling_rate=config.get("sampling_rate", 1.0),
            noise_level=config.get("noise_level", 0.1),
        )
        
        self.processor = EdgeDataProcessor(
            max_batch_size=config.get("max_batch_size", 1),
            max_sequence_length=config.get("max_sequence_length", 1),
            enable_preprocessing=config.get("enable_preprocessing", True),
        )
        
        self.dataset = StreamingDataset(
            buffer_size=config.get("buffer_size", 1000),
            sequence_length=config.get("sequence_length", 1),
        )
        
        # MQTT collector (optional)
        self.mqtt_collector = None
        if config.get("enable_mqtt", False):
            self.mqtt_collector = MQTTDataCollector(
                broker_host=config.get("mqtt_host", "localhost"),
                broker_port=config.get("mqtt_port", 1883),
                topic_prefix=config.get("mqtt_topic_prefix", "parking/sensors"),
            )
    
    async def start_streaming(self, duration: Optional[float] = None) -> None:
        """Start streaming sensor data."""
        logger.info("Starting data streaming pipeline...")
        
        if self.mqtt_collector:
            self.mqtt_collector.connect()
            self.dataset = self.mqtt_collector.get_dataset()
        
        # Stream from simulator
        async for reading in self.simulator.stream_readings(duration):
            self.dataset.add_reading(reading)
            
            # Process in real-time if we have enough data
            if len(self.dataset) >= self.config.get("min_samples_for_inference", 1):
                batch = self.processor.create_edge_batch([reading])
                yield batch, reading
    
    def get_training_data(self, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training data for model training."""
        logger.info(f"Generating {n_samples} training samples...")
        
        # Generate synthetic training data
        features, labels = self.simulator.generate_training_data(n_samples)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        # Process for edge constraints
        processed_features, processed_labels = self.processor.process_batch(
            features_tensor, labels_tensor
        )
        
        return processed_features, processed_labels
    
    def save_data(self, filepath: Path) -> None:
        """Save current dataset to file."""
        data = {
            "config": self.config,
            "samples": [sample for sample in self.dataset.buffer],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset to {filepath}")
    
    def load_data(self, filepath: Path) -> None:
        """Load dataset from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.config.update(data["config"])
        
        # Recreate dataset
        for sample in data["samples"]:
            reading = SensorReading(
                timestamp=sample["timestamp"],
                spot_id=sample["spot_id"],
                distance=sample["features"][0],
                lighting=sample["features"][1],
                motion_detected=bool(sample["features"][2]),
                temperature=20.0,  # Default values
                humidity=50.0,
                occupancy=sample["label"] != -1,
            )
            self.dataset.add_reading(reading)
        
        logger.info(f"Loaded dataset from {filepath}")

# Add missing method to SensorSimulator
def generate_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for model training."""
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Generate random spot
        spot_id = f"spot_{self.rng.randint(0, self.num_spots):02d}"
        reading = self.generate_reading(spot_id)
        
        hour = time.localtime(reading.timestamp).tm_hour
        feature_vector = [
            reading.distance,
            reading.lighting,
            float(reading.motion_detected),
            float(hour),
        ]
        
        features.append(feature_vector)
        labels.append(float(reading.occupancy))
    
    return np.array(features), np.array(labels)

# Monkey patch the method
SensorSimulator.generate_training_data = generate_training_data
