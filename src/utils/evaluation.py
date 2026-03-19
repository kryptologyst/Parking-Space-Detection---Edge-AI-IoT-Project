"""
Comprehensive evaluation metrics for edge AI models.

This module provides evaluation tools for both model accuracy and edge performance
metrics including latency, throughput, memory usage, and energy consumption.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class EdgeMetrics:
    """Container for edge performance metrics."""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    model_size_mb: float
    energy_per_inference_j: Optional[float] = None
    thermal_headroom_c: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class RobustnessMetrics:
    """Container for model robustness metrics."""
    noise_accuracy: float
    jpeg_accuracy: float
    blur_accuracy: float
    packet_loss_accuracy: float
    offline_mode_accuracy: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class CommunicationMetrics:
    """Container for communication metrics."""
    bandwidth_usage_kbps: float
    mqtt_qos_impact: float
    e2e_latency_ms: float
    packet_loss_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class ModelEvaluator:
    """Comprehensive model evaluation for edge AI systems."""
    
    def __init__(self, device: torch.device) -> None:
        """Initialize model evaluator."""
        self.device = device
        self.evaluation_results = {}
    
    def evaluate_accuracy(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
    ) -> ModelMetrics:
        """
        Evaluate model accuracy metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for logging
            
        Returns:
            ModelMetrics object containing accuracy metrics
        """
        logger.info(f"Evaluating accuracy for {model_name}...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="weighted")
        recall = recall_score(all_labels, all_predictions, average="weighted")
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        
        # Calculate AUC for binary classification
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probabilities, multi_class="ovr")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            confusion_matrix=cm.tolist(),
            classification_report=report,
        )
        
        logger.info(f"Accuracy evaluation completed for {model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return metrics
    
    def evaluate_edge_performance(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        num_runs: int = 100,
        model_name: str = "model",
    ) -> EdgeMetrics:
        """
        Evaluate edge performance metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            num_runs: Number of inference runs for timing
            model_name: Name of the model for logging
            
        Returns:
            EdgeMetrics object containing performance metrics
        """
        logger.info(f"Evaluating edge performance for {model_name}...")
        
        model.eval()
        model = model.to(self.device)
        
        # Get a sample batch
        sample_features, _ = next(iter(test_loader))
        sample_features = sample_features.to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_features)
        
        # Measure memory before inference
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        memory_before = self._get_memory_usage()
        
        # Benchmark inference time
        latencies = []
        memory_usage = []
        
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        
        for i in range(num_runs):
            # Measure memory during inference
            memory_during = self._get_memory_usage()
            memory_usage.append(memory_during)
            
            start_time = time.time()
            with torch.no_grad():
                _ = model(sample_features)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Calculate throughput
        batch_size = sample_features.size(0)
        throughput = batch_size / (avg_latency / 1000)  # Samples per second
        
        # Memory metrics
        peak_memory = max(memory_usage) - memory_before
        
        # Model size
        model_size = self._get_model_size(model)
        
        # Energy consumption (estimated)
        energy_per_inference = self._estimate_energy_consumption(avg_latency)
        
        metrics = EdgeMetrics(
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_fps=throughput,
            peak_memory_mb=peak_memory,
            model_size_mb=model_size,
            energy_per_inference_j=energy_per_inference,
        )
        
        logger.info(f"Edge performance evaluation completed for {model_name}:")
        logger.info(f"  Average latency: {avg_latency:.2f} ms")
        logger.info(f"  P95 latency: {p95_latency:.2f} ms")
        logger.info(f"  P99 latency: {p99_latency:.2f} ms")
        logger.info(f"  Throughput: {throughput:.1f} FPS")
        logger.info(f"  Peak memory: {peak_memory:.2f} MB")
        logger.info(f"  Model size: {model_size:.2f} MB")
        
        return metrics
    
    def evaluate_robustness(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
    ) -> RobustnessMetrics:
        """
        Evaluate model robustness to various perturbations.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for logging
            
        Returns:
            RobustnessMetrics object containing robustness metrics
        """
        logger.info(f"Evaluating robustness for {model_name}...")
        
        model.eval()
        
        # Test with noise
        noise_accuracy = self._test_with_noise(model, test_loader)
        
        # Test with JPEG compression (simulated)
        jpeg_accuracy = self._test_with_jpeg_compression(model, test_loader)
        
        # Test with blur (simulated)
        blur_accuracy = self._test_with_blur(model, test_loader)
        
        # Test with packet loss (simulated)
        packet_loss_accuracy = self._test_with_packet_loss(model, test_loader)
        
        # Test offline mode (no communication)
        offline_mode_accuracy = self._test_offline_mode(model, test_loader)
        
        metrics = RobustnessMetrics(
            noise_accuracy=noise_accuracy,
            jpeg_accuracy=jpeg_accuracy,
            blur_accuracy=blur_accuracy,
            packet_loss_accuracy=packet_loss_accuracy,
            offline_mode_accuracy=offline_mode_accuracy,
        )
        
        logger.info(f"Robustness evaluation completed for {model_name}:")
        logger.info(f"  Noise accuracy: {noise_accuracy:.4f}")
        logger.info(f"  JPEG accuracy: {jpeg_accuracy:.4f}")
        logger.info(f"  Blur accuracy: {blur_accuracy:.4f}")
        logger.info(f"  Packet loss accuracy: {packet_loss_accuracy:.4f}")
        logger.info(f"  Offline mode accuracy: {offline_mode_accuracy:.4f}")
        
        return metrics
    
    def evaluate_communication(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
    ) -> CommunicationMetrics:
        """
        Evaluate communication metrics for edge deployment.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for logging
            
        Returns:
            CommunicationMetrics object containing communication metrics
        """
        logger.info(f"Evaluating communication for {model_name}...")
        
        # Simulate MQTT communication
        bandwidth_usage = self._simulate_mqtt_bandwidth(test_loader)
        qos_impact = self._simulate_mqtt_qos_impact(test_loader)
        e2e_latency = self._simulate_e2e_latency(test_loader)
        packet_loss_rate = self._simulate_packet_loss(test_loader)
        
        metrics = CommunicationMetrics(
            bandwidth_usage_kbps=bandwidth_usage,
            mqtt_qos_impact=qos_impact,
            e2e_latency_ms=e2e_latency,
            packet_loss_rate=packet_loss_rate,
        )
        
        logger.info(f"Communication evaluation completed for {model_name}:")
        logger.info(f"  Bandwidth usage: {bandwidth_usage:.2f} KB/s")
        logger.info(f"  MQTT QoS impact: {qos_impact:.2f} ms")
        logger.info(f"  E2E latency: {e2e_latency:.2f} ms")
        logger.info(f"  Packet loss rate: {packet_loss_rate:.4f}")
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _get_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb
    
    def _estimate_energy_consumption(self, latency_ms: float) -> float:
        """Estimate energy consumption per inference in Joules."""
        # Rough estimation based on device type and latency
        if self.device.type == "cuda":
            # GPU: ~100W power consumption
            power_w = 100.0
        else:
            # CPU: ~10W power consumption
            power_w = 10.0
        
        # Convert latency to seconds and calculate energy
        latency_s = latency_ms / 1000.0
        energy_j = power_w * latency_s
        
        return energy_j
    
    def _test_with_noise(self, model: torch.nn.Module, test_loader: DataLoader) -> float:
        """Test model accuracy with added noise."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Add Gaussian noise
                noise = torch.randn_like(features) * 0.1
                noisy_features = features + noise
                
                outputs = model(noisy_features)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _test_with_jpeg_compression(self, model: torch.nn.Module, test_loader: DataLoader) -> float:
        """Test model accuracy with JPEG compression simulation."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Simulate JPEG compression by adding quantization noise
                quantization_noise = torch.randn_like(features) * 0.05
                compressed_features = features + quantization_noise
                
                outputs = model(compressed_features)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _test_with_blur(self, model: torch.nn.Module, test_loader: DataLoader) -> float:
        """Test model accuracy with blur simulation."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Simulate blur by smoothing features
                blurred_features = features * 0.8 + torch.mean(features, dim=1, keepdim=True) * 0.2
                
                outputs = model(blurred_features)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _test_with_packet_loss(self, model: torch.nn.Module, test_loader: DataLoader) -> float:
        """Test model accuracy with packet loss simulation."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Simulate packet loss by zeroing some features
                packet_loss_mask = torch.rand_like(features) > 0.1  # 10% packet loss
                features_with_loss = features * packet_loss_mask.float()
                
                outputs = model(features_with_loss)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _test_offline_mode(self, model: torch.nn.Module, test_loader: DataLoader) -> float:
        """Test model accuracy in offline mode (no communication)."""
        # In offline mode, we use cached/local data
        return self.evaluate_accuracy(model, test_loader).accuracy
    
    def _simulate_mqtt_bandwidth(self, test_loader: DataLoader) -> float:
        """Simulate MQTT bandwidth usage."""
        # Estimate bandwidth based on data size and frequency
        sample_features, _ = next(iter(test_loader))
        data_size_bytes = sample_features.numel() * 4  # 4 bytes per float32
        
        # Assume 1 Hz sampling rate
        bandwidth_kbps = (data_size_bytes * 8) / 1000  # Convert to kbps
        
        return bandwidth_kbps
    
    def _simulate_mqtt_qos_impact(self, test_loader: DataLoader) -> float:
        """Simulate MQTT QoS impact on latency."""
        # Different QoS levels have different latency impacts
        qos_latency_ms = {
            0: 5.0,   # At most once
            1: 10.0,  # At least once
            2: 15.0,  # Exactly once
        }
        
        # Use QoS 1 as default
        return qos_latency_ms[1]
    
    def _simulate_e2e_latency(self, test_loader: DataLoader) -> float:
        """Simulate end-to-end latency."""
        # Include processing, communication, and network latency
        processing_latency = 50.0  # ms
        network_latency = 20.0     # ms
        communication_latency = 10.0  # ms
        
        return processing_latency + network_latency + communication_latency
    
    def _simulate_packet_loss(self, test_loader: DataLoader) -> float:
        """Simulate packet loss rate."""
        # Typical packet loss rate for wireless networks
        return 0.01  # 1%

class EvaluationReport:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: Path) -> None:
        """Initialize evaluation report generator."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        model_results: Dict[str, ModelMetrics],
        edge_results: Dict[str, EdgeMetrics],
        robustness_results: Dict[str, RobustnessMetrics],
        communication_results: Dict[str, CommunicationMetrics],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_results: Model accuracy results
            edge_results: Edge performance results
            robustness_results: Robustness results
            communication_results: Communication results
            
        Returns:
            Dictionary containing the complete report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Create leaderboard
        leaderboard = self._create_leaderboard(
            model_results, edge_results, robustness_results
        )
        
        # Generate visualizations
        self._create_visualizations(
            model_results, edge_results, robustness_results
        )
        
        # Create summary report
        report = {
            "timestamp": time.time(),
            "leaderboard": leaderboard,
            "model_results": {k: v.to_dict() for k, v in model_results.items()},
            "edge_results": {k: v.to_dict() for k, v in edge_results.items()},
            "robustness_results": {k: v.to_dict() for k, v in robustness_results.items()},
            "communication_results": {k: v.to_dict() for k, v in communication_results.items()},
        }
        
        # Save report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report
    
    def _create_leaderboard(
        self,
        model_results: Dict[str, ModelMetrics],
        edge_results: Dict[str, EdgeMetrics],
        robustness_results: Dict[str, RobustnessMetrics],
    ) -> Dict[str, Any]:
        """Create performance leaderboard."""
        leaderboard = {
            "accuracy_ranking": [],
            "latency_ranking": [],
            "efficiency_ranking": [],
            "robustness_ranking": [],
        }
        
        # Accuracy ranking
        accuracy_scores = [(name, result.accuracy) for name, result in model_results.items()]
        accuracy_scores.sort(key=lambda x: x[1], reverse=True)
        leaderboard["accuracy_ranking"] = accuracy_scores
        
        # Latency ranking (lower is better)
        latency_scores = [(name, result.avg_latency_ms) for name, result in edge_results.items()]
        latency_scores.sort(key=lambda x: x[1])
        leaderboard["latency_ranking"] = latency_scores
        
        # Efficiency ranking (accuracy per MB)
        efficiency_scores = []
        for name in model_results.keys():
            if name in edge_results:
                accuracy = model_results[name].accuracy
                model_size = edge_results[name].model_size_mb
                efficiency = accuracy / model_size if model_size > 0 else 0
                efficiency_scores.append((name, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        leaderboard["efficiency_ranking"] = efficiency_scores
        
        # Robustness ranking
        robustness_scores = []
        for name in model_results.keys():
            if name in robustness_results:
                robustness = (
                    robustness_results[name].noise_accuracy +
                    robustness_results[name].jpeg_accuracy +
                    robustness_results[name].blur_accuracy +
                    robustness_results[name].packet_loss_accuracy +
                    robustness_results[name].offline_mode_accuracy
                ) / 5
                robustness_scores.append((name, robustness))
        
        robustness_scores.sort(key=lambda x: x[1], reverse=True)
        leaderboard["robustness_ranking"] = robustness_scores
        
        return leaderboard
    
    def _create_visualizations(
        self,
        model_results: Dict[str, ModelMetrics],
        edge_results: Dict[str, EdgeMetrics],
        robustness_results: Dict[str, RobustnessMetrics],
    ) -> None:
        """Create visualization plots."""
        # Accuracy vs Model Size plot
        self._plot_accuracy_vs_size(model_results, edge_results)
        
        # Latency distribution plot
        self._plot_latency_distribution(edge_results)
        
        # Robustness comparison plot
        self._plot_robustness_comparison(robustness_results)
        
        # Confusion matrices
        self._plot_confusion_matrices(model_results)
    
    def _plot_accuracy_vs_size(
        self,
        model_results: Dict[str, ModelMetrics],
        edge_results: Dict[str, EdgeMetrics],
    ) -> None:
        """Plot accuracy vs model size."""
        plt.figure(figsize=(10, 6))
        
        for name in model_results.keys():
            if name in edge_results:
                accuracy = model_results[name].accuracy
                model_size = edge_results[name].model_size_mb
                plt.scatter(model_size, accuracy, label=name, s=100)
        
        plt.xlabel("Model Size (MB)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Model Size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / "accuracy_vs_size.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_latency_distribution(self, edge_results: Dict[str, EdgeMetrics]) -> None:
        """Plot latency distribution."""
        plt.figure(figsize=(10, 6))
        
        models = list(edge_results.keys())
        avg_latencies = [edge_results[name].avg_latency_ms for name in models]
        p95_latencies = [edge_results[name].p95_latency_ms for name in models]
        p99_latencies = [edge_results[name].p99_latency_ms for name in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.bar(x - width, avg_latencies, width, label="Average", alpha=0.8)
        plt.bar(x, p95_latencies, width, label="P95", alpha=0.8)
        plt.bar(x + width, p99_latencies, width, label="P99", alpha=0.8)
        
        plt.xlabel("Models")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Distribution Comparison")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / "latency_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_robustness_comparison(self, robustness_results: Dict[str, RobustnessMetrics]) -> None:
        """Plot robustness comparison."""
        plt.figure(figsize=(12, 8))
        
        models = list(robustness_results.keys())
        metrics = ["noise_accuracy", "jpeg_accuracy", "blur_accuracy", "packet_loss_accuracy", "offline_mode_accuracy"]
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [getattr(robustness_results[name], metric) for name in models]
            plt.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
        
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Robustness Comparison")
        plt.xticks(x + width * 2, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / "robustness_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_confusion_matrices(self, model_results: Dict[str, ModelMetrics]) -> None:
        """Plot confusion matrices for all models."""
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(model_results.items()):
            cm = np.array(result.confusion_matrix)
            
            im = axes[i].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            axes[i].figure.colorbar(im, ax=axes[i])
            
            # Add text annotations
            thresh = cm.max() / 2.0
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    axes[i].text(
                        col, row, cm[row, col],
                        ha="center", va="center",
                        color="white" if cm[row, col] > thresh else "black"
                    )
            
            axes[i].set_title(f"{name} Confusion Matrix")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(["Vacant", "Occupied"])
            axes[i].set_yticklabels(["Vacant", "Occupied"])
        
        plt.tight_layout()
        plot_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
