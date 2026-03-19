"""
Model compilation and deployment utilities for edge devices.

This module provides tools to export PyTorch models to various edge runtime
formats including ONNX, TensorFlow Lite, CoreML, and OpenVINO.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Optional imports for different backends
try:
    import tensorflow as tf
    from tensorflow import lite as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    logger.warning("TensorFlow Lite not available")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logger.warning("CoreML Tools not available")

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logger.warning("OpenVINO not available")

class ModelExporter:
    """Handles model export to various edge runtime formats."""
    
    def __init__(self, device: torch.device) -> None:
        """Initialize model exporter."""
        self.device = device
        self.export_results = {}
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Dictionary containing export information
        """
        logger.info("Exporting model to ONNX...")
        
        model.eval()
        model = model.to("cpu")  # ONNX export works on CPU
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Get model size
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        export_info = {
            "format": "onnx",
            "opset_version": opset_version,
            "model_size_mb": model_size,
            "input_shape": input_shape,
            "output_path": str(output_path),
        }
        
        logger.info(f"ONNX export completed. Model size: {model_size:.2f} MB")
        return export_info
    
    def export_to_tflite(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        quantize: bool = True,
        optimization: str = "DEFAULT",
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save TFLite model
            quantize: Whether to quantize the model
            optimization: Optimization level
            
        Returns:
            Dictionary containing export information
        """
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available")
        
        logger.info("Exporting model to TensorFlow Lite...")
        
        # First export to ONNX, then convert to TensorFlow
        onnx_path = output_path.with_suffix(".onnx")
        onnx_info = self.export_to_onnx(model, input_shape, onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_model = self._onnx_to_tensorflow(onnx_path)
        
        # Convert TensorFlow to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model])
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        export_info = {
            "format": "tflite",
            "quantized": quantize,
            "optimization": optimization,
            "model_size_mb": model_size,
            "input_shape": input_shape,
            "output_path": str(output_path),
        }
        
        logger.info(f"TFLite export completed. Model size: {model_size:.2f} MB")
        return export_info
    
    def export_to_coreml(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        compute_units: str = "ALL",
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to CoreML format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save CoreML model
            compute_units: Compute units for CoreML
            
        Returns:
            Dictionary containing export information
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML Tools not available")
        
        logger.info("Exporting model to CoreML...")
        
        model.eval()
        model = model.to("cpu")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
            compute_units=getattr(ct.ComputeUnit, compute_units),
        )
        
        # Save CoreML model
        coreml_model.save(str(output_path))
        
        # Get model size
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        export_info = {
            "format": "coreml",
            "compute_units": compute_units,
            "model_size_mb": model_size,
            "input_shape": input_shape,
            "output_path": str(output_path),
        }
        
        logger.info(f"CoreML export completed. Model size: {model_size:.2f} MB")
        return export_info
    
    def export_to_openvino(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        precision: str = "FP32",
    ) -> Dict[str, Any]:
        """
        Export PyTorch model to OpenVINO format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save OpenVINO model
            precision: Model precision (FP32, FP16, INT8)
            
        Returns:
            Dictionary containing export information
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")
        
        logger.info("Exporting model to OpenVINO...")
        
        # First export to ONNX
        onnx_path = output_path.with_suffix(".onnx")
        onnx_info = self.export_to_onnx(model, input_shape, onnx_path)
        
        # Convert ONNX to OpenVINO
        core = Core()
        
        # Convert model
        model_ov = core.read_model(str(onnx_path))
        
        # Optimize for specific device
        compiled_model = core.compile_model(model_ov, "CPU")
        
        # Save OpenVINO model
        compiled_model.save(str(output_path))
        
        # Get model size
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        export_info = {
            "format": "openvino",
            "precision": precision,
            "model_size_mb": model_size,
            "input_shape": input_shape,
            "output_path": str(output_path),
        }
        
        logger.info(f"OpenVINO export completed. Model size: {model_size:.2f} MB")
        return export_info
    
    def _onnx_to_tensorflow(self, onnx_path: Path) -> Any:
        """Convert ONNX model to TensorFlow."""
        # This is a simplified conversion - in practice, you might need
        # more sophisticated conversion tools like onnx-tf
        raise NotImplementedError("ONNX to TensorFlow conversion not implemented")

class EdgeRuntime:
    """Runtime for executing models on edge devices."""
    
    def __init__(self, runtime_type: str, model_path: Path) -> None:
        """
        Initialize edge runtime.
        
        Args:
            runtime_type: Type of runtime ('onnx', 'tflite', 'coreml', 'openvino')
            model_path: Path to the model file
        """
        self.runtime_type = runtime_type
        self.model_path = model_path
        self.session = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model into the appropriate runtime."""
        if self.runtime_type == "onnx":
            self._load_onnx_model()
        elif self.runtime_type == "tflite":
            self._load_tflite_model()
        elif self.runtime_type == "coreml":
            self._load_coreml_model()
        elif self.runtime_type == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(f"Unknown runtime type: {self.runtime_type}")
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        providers = ["CPUExecutionProvider"]
        if ort.get_device() == "GPU":
            providers.insert(0, "CUDAExecutionProvider")
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _load_tflite_model(self) -> None:
        """Load TensorFlow Lite model."""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available")
        
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def _load_coreml_model(self) -> None:
        """Load CoreML model."""
        if not COREML_AVAILABLE:
            raise ImportError("CoreML Tools not available")
        
        self.model = ct.models.MLModel(str(self.model_path))
    
    def _load_openvino_model(self) -> None:
        """Load OpenVINO model."""
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")
        
        core = Core()
        self.compiled_model = core.compile_model(str(self.model_path), "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data array
            
        Returns:
            Prediction results
        """
        if self.runtime_type == "onnx":
            return self._predict_onnx(input_data)
        elif self.runtime_type == "tflite":
            return self._predict_tflite(input_data)
        elif self.runtime_type == "coreml":
            return self._predict_coreml(input_data)
        elif self.runtime_type == "openvino":
            return self._predict_openvino(input_data)
        else:
            raise ValueError(f"Unknown runtime type: {self.runtime_type}")
    
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        return self.session.run(
            [self.output_name],
            {self.input_name: input_data.astype(np.float32)}
        )[0]
    
    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """Run TensorFlow Lite inference."""
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_data.astype(np.float32)
        )
        
        self.interpreter.invoke()
        
        return self.interpreter.get_tensor(self.output_details[0]["index"])
    
    def _predict_coreml(self, input_data: np.ndarray) -> np.ndarray:
        """Run CoreML inference."""
        prediction = self.model.predict({"input": input_data})
        return prediction["output"]
    
    def _predict_openvino(self, input_data: np.ndarray) -> np.ndarray:
        """Run OpenVINO inference."""
        result = self.compiled_model([input_data.astype(np.float32)])
        return result[self.output_layer]

class DeviceConfig:
    """Configuration for different edge devices."""
    
    DEVICE_CONFIGS = {
        "raspberry_pi_4": {
            "cpu": "ARM Cortex-A72",
            "memory": "4GB",
            "storage": "32GB",
            "power": "5V/3A",
            "supported_formats": ["onnx", "tflite"],
            "max_batch_size": 1,
            "max_model_size_mb": 50,
            "target_latency_ms": 100,
        },
        "jetson_nano": {
            "cpu": "ARM Cortex-A57",
            "gpu": "NVIDIA Maxwell",
            "memory": "4GB",
            "storage": "16GB",
            "power": "5V/4A",
            "supported_formats": ["onnx", "tflite", "tensorrt"],
            "max_batch_size": 4,
            "max_model_size_mb": 200,
            "target_latency_ms": 50,
        },
        "jetson_xavier": {
            "cpu": "ARM Cortex-A78AE",
            "gpu": "NVIDIA Volta",
            "memory": "32GB",
            "storage": "64GB",
            "power": "19V/6.3A",
            "supported_formats": ["onnx", "tflite", "tensorrt", "openvino"],
            "max_batch_size": 8,
            "max_model_size_mb": 500,
            "target_latency_ms": 20,
        },
        "android": {
            "cpu": "ARM/ARM64",
            "gpu": "Adreno/Mali",
            "memory": "4-8GB",
            "storage": "64-256GB",
            "power": "Battery",
            "supported_formats": ["tflite", "coreml"],
            "max_batch_size": 1,
            "max_model_size_mb": 100,
            "target_latency_ms": 50,
        },
        "ios": {
            "cpu": "ARM64",
            "gpu": "Apple GPU",
            "memory": "4-8GB",
            "storage": "64-512GB",
            "power": "Battery",
            "supported_formats": ["coreml"],
            "max_batch_size": 1,
            "max_model_size_mb": 100,
            "target_latency_ms": 30,
        },
        "mcu": {
            "cpu": "ARM Cortex-M",
            "memory": "256KB-2MB",
            "storage": "1-16MB",
            "power": "3.3V/100mA",
            "supported_formats": ["tflite"],
            "max_batch_size": 1,
            "max_model_size_mb": 1,
            "target_latency_ms": 1000,
        },
    }
    
    @classmethod
    def get_config(cls, device_name: str) -> Dict[str, Any]:
        """Get configuration for a specific device."""
        if device_name not in cls.DEVICE_CONFIGS:
            raise ValueError(f"Unknown device: {device_name}")
        
        return cls.DEVICE_CONFIGS[device_name]
    
    @classmethod
    def list_devices(cls) -> List[str]:
        """List all available device configurations."""
        return list(cls.DEVICE_CONFIGS.keys())
    
    @classmethod
    def get_optimal_format(cls, device_name: str) -> str:
        """Get the optimal model format for a device."""
        config = cls.get_config(device_name)
        formats = config["supported_formats"]
        
        # Priority order for different formats
        priority = ["tensorrt", "openvino", "onnx", "tflite", "coreml"]
        
        for fmt in priority:
            if fmt in formats:
                return fmt
        
        return formats[0] if formats else "onnx"

class DeploymentPipeline:
    """Complete deployment pipeline for edge devices."""
    
    def __init__(self, device_name: str, output_dir: Path) -> None:
        """
        Initialize deployment pipeline.
        
        Args:
            device_name: Target device name
            output_dir: Output directory for models
        """
        self.device_name = device_name
        self.output_dir = output_dir
        self.device_config = DeviceConfig.get_config(device_name)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized deployment pipeline for {device_name}")
        logger.info(f"Device config: {self.device_config}")
    
    def deploy_model(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "parking_classifier",
    ) -> Dict[str, Any]:
        """
        Deploy model to target device.
        
        Args:
            model: PyTorch model to deploy
            input_shape: Input tensor shape
            model_name: Name for the deployed model
            
        Returns:
            Dictionary containing deployment information
        """
        logger.info(f"Deploying model to {self.device_name}...")
        
        deployment_info = {
            "device": self.device_name,
            "device_config": self.device_config,
            "model_name": model_name,
            "input_shape": input_shape,
            "exports": {},
        }
        
        # Export to all supported formats
        exporter = ModelExporter(torch.device("cpu"))
        
        for format_name in self.device_config["supported_formats"]:
            try:
                output_path = self.output_dir / f"{model_name}.{format_name}"
                
                if format_name == "onnx":
                    export_info = exporter.export_to_onnx(
                        model, input_shape, output_path
                    )
                elif format_name == "tflite":
                    export_info = exporter.export_to_tflite(
                        model, input_shape, output_path
                    )
                elif format_name == "coreml":
                    export_info = exporter.export_to_coreml(
                        model, input_shape, output_path
                    )
                elif format_name == "openvino":
                    export_info = exporter.export_to_openvino(
                        model, input_shape, output_path
                    )
                elif format_name == "tensorrt":
                    # TensorRT export would require additional setup
                    logger.warning("TensorRT export not implemented")
                    continue
                else:
                    logger.warning(f"Unknown format: {format_name}")
                    continue
                
                deployment_info["exports"][format_name] = export_info
                
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                continue
        
        # Test deployment
        optimal_format = DeviceConfig.get_optimal_format(self.device_name)
        if optimal_format in deployment_info["exports"]:
            self._test_deployment(
                deployment_info["exports"][optimal_format]["output_path"],
                optimal_format,
                input_shape,
            )
        
        logger.info(f"Deployment completed for {self.device_name}")
        return deployment_info
    
    def _test_deployment(
        self,
        model_path: str,
        format_name: str,
        input_shape: Tuple[int, ...],
    ) -> None:
        """Test the deployed model."""
        logger.info(f"Testing {format_name} deployment...")
        
        try:
            runtime = EdgeRuntime(format_name, Path(model_path))
            
            # Test with dummy data
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            start_time = time.time()
            prediction = runtime.predict(dummy_input)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            logger.info(f"Test inference completed:")
            logger.info(f"  Format: {format_name}")
            logger.info(f"  Latency: {latency:.2f} ms")
            logger.info(f"  Output shape: {prediction.shape}")
            
        except Exception as e:
            logger.error(f"Deployment test failed: {e}")
    
    def create_deployment_package(self, deployment_info: Dict[str, Any]) -> Path:
        """Create a deployment package with all necessary files."""
        package_dir = self.output_dir / "deployment_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy model files
        for format_name, export_info in deployment_info["exports"].items():
            model_path = Path(export_info["output_path"])
            package_dir.mkdir(exist_ok=True)
            
            # Copy model file
            import shutil
            shutil.copy2(model_path, package_dir / model_path.name)
        
        # Create deployment script
        self._create_deployment_script(package_dir, deployment_info)
        
        # Create configuration file
        config_path = package_dir / "deployment_config.yaml"
        OmegaConf.save(deployment_info, config_path)
        
        logger.info(f"Deployment package created at {package_dir}")
        return package_dir
    
    def _create_deployment_script(self, package_dir: Path, deployment_info: Dict[str, Any]) -> None:
        """Create a deployment script for the target device."""
        script_content = f'''#!/usr/bin/env python3
"""
Deployment script for {self.device_name}
Generated automatically by the deployment pipeline.
"""

import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main deployment function."""
    logger.info("Starting parking space detection on {self.device_name}")
    
    # Load model
    model_path = Path(__file__).parent / "parking_classifier.{DeviceConfig.get_optimal_format(self.device_name)}"
    
    if not model_path.exists():
        logger.error(f"Model file not found: {{model_path}}")
        return
    
    # Initialize runtime
    from src.export.runtime import EdgeRuntime
    
    runtime = EdgeRuntime("{DeviceConfig.get_optimal_format(self.device_name)}", model_path)
    
    # Run inference loop
    logger.info("Starting inference loop...")
    
    while True:
        try:
            # Get sensor data (replace with actual sensor reading)
            sensor_data = np.random.randn(1, 4).astype(np.float32)
            
            # Run inference
            prediction = runtime.predict(sensor_data)
            
            # Process results
            occupancy = prediction[0][1] > 0.5  # Assuming binary classification
            status = "Occupied" if occupancy else "Vacant"
            
            logger.info(f"Parking spot status: {{status}}")
            
            # Sleep for next reading
            import time
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            logger.info("Stopping inference loop...")
            break
        except Exception as e:
            logger.error(f"Inference error: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        script_path = package_dir / "deploy.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
