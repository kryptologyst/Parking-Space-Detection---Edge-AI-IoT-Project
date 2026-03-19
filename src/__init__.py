"""
Parking Space Detection - Edge AI & IoT Project

A comprehensive system for detecting parking space occupancy using sensor data
and deploying optimized models to edge devices.

This package provides:
- Neural network models for parking space classification
- Model optimization techniques (quantization, pruning, distillation)
- Data pipelines for sensor simulation and streaming
- Deployment utilities for various edge platforms
- Comprehensive evaluation metrics
- Interactive demo application

Author: AI Assistant
Version: 1.0.0
License: MIT

DISCLAIMER: This project is for research and educational purposes only.
NOT FOR SAFETY-CRITICAL DEPLOYMENT.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"
__license__ = "MIT"

# Import main modules
from . import models
from . import pipelines
from . import export
from . import utils

__all__ = [
    "models",
    "pipelines", 
    "export",
    "utils",
]
