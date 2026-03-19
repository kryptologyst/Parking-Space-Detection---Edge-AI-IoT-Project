"""
Test suite for parking space detection system.

This module contains comprehensive tests for all components
of the parking space detection system.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import project modules
from src.models.parking_classifier import (
    ParkingSpaceClassifier,
    ParkingSensorDataset,
    generate_sensor_data,
    train_model,
    evaluate_model,
)
from src.models.optimization import ModelOptimizer, EdgeOptimizedModel
from src.pipelines.data_pipeline import SensorSimulator, StreamingDataset
from src.utils.evaluation import ModelEvaluator, ModelMetrics, EdgeMetrics
from src.export.deployment import ModelExporter, EdgeRuntime, DeviceConfig

class TestParkingSpaceClassifier:
    """Test cases for the parking space classifier."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])
        x = torch.randn(1, 4)
        output = model(x)
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
    
    def test_model_training(self):
        """Test model training."""
        # Generate small dataset
        features, labels = generate_sensor_data(n_samples=100)
        
        # Create datasets
        train_dataset = ParkingSensorDataset(features[:80], labels[:80])
        val_dataset = ParkingSensorDataset(features[80:], labels[80:])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Initialize model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        
        # Train model
        history = train_model(model, train_loader, val_loader, epochs=5)
        
        # Check training history
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert len(history["train_acc"]) == 5
        assert len(history["val_acc"]) == 5
        
        # Check that loss decreases
        assert history["train_loss"][-1] < history["train_loss"][0]
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Generate test data
        features, labels = generate_sensor_data(n_samples=50)
        test_dataset = ParkingSensorDataset(features, labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Initialize and train model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        
        # Train briefly
        train_dataset = ParkingSensorDataset(features, labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        train_model(model, train_loader, test_loader, epochs=3)
        
        # Evaluate model
        results = evaluate_model(model, test_loader)
        
        # Check results
        assert "accuracy" in results
        assert "classification_report" in results
        assert "confusion_matrix" in results
        assert 0 <= results["accuracy"] <= 1

class TestModelOptimization:
    """Test cases for model optimization."""
    
    def test_model_optimizer_initialization(self):
        """Test model optimizer initialization."""
        device = torch.device("cpu")
        optimizer = ModelOptimizer(device)
        assert optimizer.device == device
    
    def test_quantization(self):
        """Test model quantization."""
        device = torch.device("cpu")
        optimizer = ModelOptimizer(device)
        
        # Create model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        
        # Create calibration data
        features, labels = generate_sensor_data(n_samples=50)
        dataset = ParkingSensorDataset(features, labels)
        calibration_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Test PTQ quantization
        quantized_model, quantization_info = optimizer.quantize_model(
            model, calibration_loader, method="ptq"
        )
        
        assert quantized_model is not None
        assert "compression_ratio" in quantization_info
        assert quantization_info["compression_ratio"] > 1.0
    
    def test_pruning(self):
        """Test model pruning."""
        device = torch.device("cpu")
        optimizer = ModelOptimizer(device)
        
        # Create model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        original_params = sum(p.numel() for p in model.parameters())
        
        # Test pruning
        pruned_model, pruning_info = optimizer.prune_model(model, sparsity=0.3)
        
        assert pruned_model is not None
        assert "compression_ratio" in pruning_info
        assert pruning_info["compression_ratio"] > 1.0
    
    def test_distillation(self):
        """Test knowledge distillation."""
        device = torch.device("cpu")
        optimizer = ModelOptimizer(device)
        
        # Create teacher and student models
        teacher_model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])
        student_model = EdgeOptimizedModel(input_dim=4)
        
        # Create data
        features, labels = generate_sensor_data(n_samples=100)
        train_dataset = ParkingSensorDataset(features[:80], labels[:80])
        val_dataset = ParkingSensorDataset(features[80:], labels[80:])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Train teacher
        train_model(teacher_model, train_loader, val_loader, epochs=5)
        
        # Test distillation
        distilled_model, distillation_info = optimizer.distill_model(
            teacher_model, student_model, train_loader, val_loader, epochs=3
        )
        
        assert distilled_model is not None
        assert "compression_ratio" in distillation_info
        assert distillation_info["compression_ratio"] > 1.0

class TestDataPipeline:
    """Test cases for data pipeline."""
    
    def test_sensor_simulator(self):
        """Test sensor simulator."""
        simulator = SensorSimulator(num_spots=5, sampling_rate=1.0)
        
        # Test reading generation
        reading = simulator.generate_reading("spot_01")
        
        assert "timestamp" in reading
        assert "spot_id" in reading
        assert "distance" in reading
        assert "lighting" in reading
        assert "motion_detected" in reading
        assert "occupancy" in reading
        
        assert reading["spot_id"] == "spot_01"
        assert 0 <= reading["distance"] <= 5.0
        assert 0 <= reading["lighting"] <= 1000
    
    def test_streaming_dataset(self):
        """Test streaming dataset."""
        dataset = StreamingDataset(buffer_size=100)
        
        # Test adding readings
        from src.pipelines.data_pipeline import SensorReading
        
        reading = SensorReading(
            timestamp=time.time(),
            spot_id="spot_01",
            distance=1.0,
            lighting=300.0,
            motion_detected=False,
            temperature=20.0,
            humidity=50.0,
            occupancy=True
        )
        
        dataset.add_reading(reading)
        
        assert len(dataset) == 1
        
        # Test getting item
        features, label = dataset[0]
        assert features.shape == (4,)
        assert label.shape == (1,)

class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        device = torch.device("cpu")
        evaluator = ModelEvaluator(device)
        
        # Create model and data
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        features, labels = generate_sensor_data(n_samples=50)
        dataset = ParkingSensorDataset(features, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Test accuracy evaluation
        metrics = evaluator.evaluate_accuracy(model, test_loader)
        
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
    
    def test_edge_performance_evaluation(self):
        """Test edge performance evaluation."""
        device = torch.device("cpu")
        evaluator = ModelEvaluator(device)
        
        # Create model and data
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        features, labels = generate_sensor_data(n_samples=50)
        dataset = ParkingSensorDataset(features, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Test edge performance evaluation
        metrics = evaluator.evaluate_edge_performance(model, test_loader, num_runs=10)
        
        assert isinstance(metrics, EdgeMetrics)
        assert metrics.avg_latency_ms > 0
        assert metrics.throughput_fps > 0
        assert metrics.model_size_mb > 0

class TestDeployment:
    """Test cases for deployment utilities."""
    
    def test_model_exporter(self):
        """Test model exporter."""
        device = torch.device("cpu")
        exporter = ModelExporter(device)
        
        # Create model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        
        # Test ONNX export
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_model.onnx"
            
            export_info = exporter.export_to_onnx(model, input_shape=(4,), output_path=output_path)
            
            assert output_path.exists()
            assert "model_size_mb" in export_info
            assert export_info["model_size_mb"] > 0
    
    def test_device_config(self):
        """Test device configuration."""
        # Test getting device config
        config = DeviceConfig.get_config("raspberry_pi_4")
        
        assert "cpu" in config
        assert "memory" in config
        assert "supported_formats" in config
        assert "onnx" in config["supported_formats"]
        
        # Test listing devices
        devices = DeviceConfig.list_devices()
        assert "raspberry_pi_4" in devices
        assert "jetson_nano" in devices
        
        # Test optimal format
        optimal_format = DeviceConfig.get_optimal_format("raspberry_pi_4")
        assert optimal_format in ["onnx", "tflite"]

class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        features, labels = generate_sensor_data(n_samples=200)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create datasets
        train_dataset = ParkingSensorDataset(X_train, y_train)
        val_dataset = ParkingSensorDataset(X_val, y_val)
        test_dataset = ParkingSensorDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train model
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        history = train_model(model, train_loader, val_loader, epochs=5)
        
        # Evaluate model
        results = evaluate_model(model, test_loader)
        
        # Test optimization
        device = torch.device("cpu")
        optimizer = ModelOptimizer(device)
        
        # Test quantization
        quantized_model, quantization_info = optimizer.quantize_model(
            model, val_loader, method="ptq"
        )
        
        # Test evaluation
        evaluator = ModelEvaluator(device)
        accuracy_metrics = evaluator.evaluate_accuracy(model, test_loader)
        edge_metrics = evaluator.evaluate_edge_performance(model, test_loader, num_runs=10)
        
        # Assertions
        assert results["accuracy"] > 0.5  # Should be better than random
        assert quantization_info["compression_ratio"] > 1.0
        assert accuracy_metrics.accuracy > 0.5
        assert edge_metrics.avg_latency_ms > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
