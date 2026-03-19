# Parking Space Detection - Edge AI & IoT Project

## Quick Start Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd 0786_Parking_Space_Detection
pip install -r requirements.txt
```

2. **Run training pipeline:**
```bash
python scripts/train_and_evaluate.py
```

3. **Launch interactive demo:**
```bash
streamlit run demo/app.py
```

## Project Overview

This project implements a smart parking system using edge AI to detect available parking spaces in real-time. The system uses sensor data from ultrasonic sensors, lighting sensors, and motion detectors to classify parking spots as occupied or vacant.

### Key Features

- **Real-time sensor simulation** with realistic data patterns
- **Edge AI model inference** with performance monitoring  
- **Multiple model variants** (original, quantized, pruned, distilled)
- **Device constraint simulation** for different edge platforms
- **Live performance metrics** including latency, throughput, and memory usage
- **Interactive visualizations** of parking spot states and sensor data
- **Comprehensive evaluation** with accuracy and edge performance metrics
- **Deployment pipeline** for various edge runtime formats

### Supported Edge Devices

- **Raspberry Pi 4**: ARM Cortex-A72, 4GB RAM, ONNX/TFLite
- **Jetson Nano**: ARM Cortex-A57 + NVIDIA Maxwell, 4GB RAM, TensorRT
- **Jetson Xavier**: ARM Cortex-A78AE + NVIDIA Volta, 32GB RAM, OpenVINO
- **Android**: ARM/ARM64, 4-8GB RAM, TFLite/CoreML
- **iOS**: ARM64 + Apple GPU, 4-8GB RAM, CoreML
- **MCU**: ARM Cortex-M, 256KB-2MB RAM, TFLite

### Model Optimization Techniques

- **Quantization**: PTQ, QAT, Dynamic, Static (INT8, INT16, FP16)
- **Pruning**: Magnitude, Structured, Channel-wise
- **Distillation**: Knowledge distillation from teacher to student models
- **Architecture Optimization**: Lightweight models for edge deployment

## Usage Examples

### Training Models

```python
from src.models.parking_classifier import ParkingSpaceClassifier, train_model

# Create model
model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])

# Train model
history = train_model(model, train_loader, val_loader, epochs=50)
```

### Model Optimization

```python
from src.models.optimization import create_optimization_pipeline

# Create optimization pipeline
results = create_optimization_pipeline(
    original_model, train_loader, val_loader, device
)
```

### Deployment

```python
from src.export.deployment import DeploymentPipeline

# Deploy to Raspberry Pi
pipeline = DeploymentPipeline("raspberry_pi_4", Path("deployments"))
deployment_info = pipeline.deploy_model(model, input_shape=(4,))
```

### Evaluation

```python
from src.utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator(device)
accuracy_metrics = evaluator.evaluate_accuracy(model, test_loader)
edge_metrics = evaluator.evaluate_edge_performance(model, test_loader)
```

## Performance Metrics

### Model Quality
- Accuracy, Precision, Recall, F1-Score
- AUC Score, Confusion Matrix
- Classification Report

### Edge Performance
- Latency (P50, P95, P99)
- Throughput (FPS)
- Memory Usage (Peak RAM)
- Model Size (MB)
- Energy Consumption (Joules per inference)

### Robustness
- Noise tolerance
- JPEG compression resistance
- Blur resistance
- Packet loss handling
- Offline mode performance

### Communication
- Bandwidth usage (KB/s)
- MQTT QoS impact
- End-to-end latency
- Packet loss rate

## Interactive Demo

The Streamlit demo provides:

1. **Real-time sensor simulation** with configurable parameters
2. **Live parking spot visualization** with occupancy status
3. **Performance monitoring** with latency and accuracy metrics
4. **Device constraint simulation** for different edge platforms
5. **Interactive charts** showing sensor data and system performance

Launch the demo:
```bash
streamlit run demo/app.py
```

## Configuration

### Device Configurations

Device-specific configurations are stored in `configs/device/devices.yaml`:

```yaml
raspberry_pi_4:
  cpu: "ARM Cortex-A72"
  memory: "4GB"
  supported_formats: ["onnx", "tflite"]
  max_batch_size: 1
  target_latency_ms: 100
```

### Quantization Settings

Quantization configurations are in `configs/quant/quantization.yaml`:

```yaml
ptq:
  enabled: true
  backend: "qnnpack"
  calibration_samples: 100
  per_channel: true
```

### Communication Settings

Communication configurations are in `configs/comms/communication.yaml`:

```yaml
mqtt:
  enabled: true
  broker:
    host: "localhost"
    port: 1883
  topics:
    sensor_data: "parking/sensors/+/data"
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_parking_detection.py::TestParkingSpaceClassifier -v
pytest tests/test_parking_detection.py::TestModelOptimization -v
pytest tests/test_parking_detection.py::TestIntegration -v
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Automated quality checks

Setup pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### CI/CD

The project includes GitHub Actions workflows for:

- Code linting and formatting
- Unit testing with coverage
- Model export testing
- Security scanning
- Demo testing

## API Reference

### Core Classes

- `ParkingSpaceClassifier`: Main neural network model
- `ModelOptimizer`: Handles quantization, pruning, distillation
- `SensorSimulator`: Generates realistic sensor data
- `EdgeRuntime`: Executes models on edge devices
- `ModelEvaluator`: Comprehensive model evaluation
- `DeploymentPipeline`: Complete deployment workflow

### Configuration Classes

- `DeviceConfig`: Edge device specifications
- `ModelMetrics`: Model performance metrics
- `EdgeMetrics`: Edge performance metrics
- `RobustnessMetrics`: Model robustness metrics

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model export fails**: Check ONNX/TensorFlow installation
3. **Demo won't start**: Ensure Streamlit is installed and port 8501 is free
4. **Import errors**: Check Python path and module installation

### Performance Tips

1. **Use quantization** for smaller models
2. **Enable pruning** for reduced parameters
3. **Use appropriate device configs** for target hardware
4. **Optimize batch size** for your device constraints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

**This project is for research and educational purposes only. NOT FOR SAFETY-CRITICAL DEPLOYMENT.**

The system uses simulated sensor data and should not be used for real-world parking management without proper validation and safety measures.
