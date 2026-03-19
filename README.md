# Parking Space Detection - Edge AI & IoT Project

## Overview

This project implements a smart parking system that detects available parking spaces using sensor data (ultrasonic sensors, lighting, motion detection) and deploys optimized models to edge devices. The system uses binary classification to predict whether a parking spot is occupied or vacant in real-time.

**Project Type:** Edge Vision/IoT with Model Efficiency focus  
**Subtype:** Sensor-based binary classification with quantization and edge deployment

## ⚠️ DISCLAIMER

**This project is for research and educational purposes only. NOT FOR SAFETY-CRITICAL DEPLOYMENT.**

The system uses simulated sensor data and should not be used for real-world parking management without proper validation and safety measures.

## Features

- **Real-time sensor simulation** with realistic data patterns
- **Edge AI model inference** with performance monitoring
- **Multiple model variants** (original, quantized, pruned, distilled)
- **Device constraint simulation** for different edge platforms
- **Live performance metrics** including latency, throughput, and memory usage
- **Interactive visualizations** of parking spot states and sensor data
- **Comprehensive evaluation** with accuracy and edge performance metrics
- **Deployment pipeline** for various edge runtime formats

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Parking-Space-Detection---Edge-AI-IoT-Project.git
cd Parking-Space-Detection---Edge-AI-IoT-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
python src/models/parking_classifier.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model definitions and training
│   │   ├── parking_classifier.py # Main classifier model
│   │   └── optimization.py       # Model optimization utilities
│   ├── pipelines/                # Data processing pipelines
│   │   └── data_pipeline.py      # Sensor data pipeline
│   ├── export/                   # Model export and deployment
│   │   └── deployment.py          # Deployment utilities
│   ├── utils/                    # Utility functions
│   │   └── evaluation.py          # Evaluation metrics
│   └── comms/                    # Communication modules
├── configs/                      # Configuration files
│   ├── device/                   # Device configurations
│   ├── quant/                    # Quantization settings
│   └── comms/                    # Communication settings
├── data/                         # Data storage
│   ├── raw/                      # Raw sensor data
│   └── processed/                # Processed datasets
├── models/                       # Trained models
├── scripts/                      # Utility scripts
├── tests/                        # Test files
├── assets/                       # Generated assets
├── demo/                         # Interactive demo
└── docs/                         # Documentation
```

## Usage

### Training Models

Train the base model:
```bash
python src/models/parking_classifier.py
```

### Model Optimization

Apply optimization techniques:
```python
from src.models.optimization import create_optimization_pipeline

# Create optimization pipeline
results = create_optimization_pipeline(
    original_model, train_loader, val_loader, device
)
```

### Deployment

Deploy to specific edge devices:
```python
from src.export.deployment import DeploymentPipeline

# Deploy to Raspberry Pi
pipeline = DeploymentPipeline("raspberry_pi_4", Path("deployments"))
deployment_info = pipeline.deploy_model(model, input_shape=(4,))
```

### Evaluation

Run comprehensive evaluation:
```python
from src.utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator(device)
accuracy_metrics = evaluator.evaluate_accuracy(model, test_loader)
edge_metrics = evaluator.evaluate_edge_performance(model, test_loader)
```

## Configuration

### Device Configurations

The system supports various edge devices with different constraints:

- **Raspberry Pi 4**: ARM Cortex-A72, 4GB RAM, ONNX/TFLite
- **Jetson Nano**: ARM Cortex-A57 + NVIDIA Maxwell, 4GB RAM, TensorRT
- **Jetson Xavier**: ARM Cortex-A78AE + NVIDIA Volta, 32GB RAM, OpenVINO
- **Android**: ARM/ARM64, 4-8GB RAM, TFLite/CoreML
- **iOS**: ARM64 + Apple GPU, 4-8GB RAM, CoreML
- **MCU**: ARM Cortex-M, 256KB-2MB RAM, TFLite

### Model Optimization

The system supports multiple optimization techniques:

- **Quantization**: PTQ, QAT, Dynamic, Static (INT8, INT16, FP16)
- **Pruning**: Magnitude, Structured, Channel-wise
- **Distillation**: Knowledge distillation from teacher to student models
- **Architecture Optimization**: Lightweight models for edge deployment

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{parking_space_detection,
  title={Parking Space Detection - Edge AI & IoT Project},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Parking-Space-Detection---Edge-AI-IoT-Project}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- ONNX Runtime for model optimization
- TensorFlow Lite for mobile deployment
- Streamlit for the interactive demo
- The edge AI community for inspiration and best practices
# Parking-Space-Detection---Edge-AI-IoT-Project
