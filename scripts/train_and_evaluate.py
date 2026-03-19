"""
Main training and evaluation script for parking space detection.

This script demonstrates the complete pipeline from data generation
to model training, optimization, and evaluation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

# Import project modules
from src.models.parking_classifier import (
    ParkingSpaceClassifier,
    ParkingSensorDataset,
    generate_sensor_data,
    train_model,
    evaluate_model,
    get_device,
)
from src.models.optimization import create_optimization_pipeline
from src.utils.evaluation import ModelEvaluator, EvaluationReport
from src.export.deployment import DeploymentPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main training and evaluation pipeline."""
    logger.info("Starting Parking Space Detection Training Pipeline")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config_path = Path("configs")
    if config_path.exists():
        device_config = OmegaConf.load(config_path / "device" / "devices.yaml")
        quant_config = OmegaConf.load(config_path / "quant" / "quantization.yaml")
        comms_config = OmegaConf.load(config_path / "comms" / "communication.yaml")
        logger.info("Configuration loaded successfully")
    else:
        logger.warning("Configuration files not found, using defaults")
        device_config = {}
        quant_config = {}
        comms_config = {}
    
    # Generate synthetic data
    logger.info("Generating synthetic sensor data...")
    features, labels = generate_sensor_data(n_samples=2000, noise_level=0.1)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create datasets and data loaders
    train_dataset = ParkingSensorDataset(X_train, y_train)
    val_dataset = ParkingSensorDataset(X_val, y_val)
    test_dataset = ParkingSensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    logger.info("Initializing models...")
    original_model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])
    logger.info(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # Train original model
    logger.info("Training original model...")
    history = train_model(original_model, train_loader, val_loader, epochs=50, device=device)
    
    # Evaluate original model
    logger.info("Evaluating original model...")
    original_results = evaluate_model(original_model, test_loader, device=device)
    logger.info(f"Original model accuracy: {original_results['accuracy']:.4f}")
    
    # Save original model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(original_model.state_dict(), models_dir / "parking_classifier.pth")
    logger.info(f"Original model saved to {models_dir / 'parking_classifier.pth'}")
    
    # Create optimization pipeline
    logger.info("Starting model optimization pipeline...")
    optimization_results = create_optimization_pipeline(
        original_model, train_loader, val_loader, device
    )
    
    # Comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    evaluator = ModelEvaluator(device)
    
    # Evaluate all models
    model_results = {}
    edge_results = {}
    robustness_results = {}
    communication_results = {}
    
    # Original model evaluation
    model_results["original"] = evaluator.evaluate_accuracy(original_model, test_loader, "original")
    edge_results["original"] = evaluator.evaluate_edge_performance(original_model, test_loader, "original")
    robustness_results["original"] = evaluator.evaluate_robustness(original_model, test_loader, "original")
    communication_results["original"] = evaluator.evaluate_communication(original_model, test_loader, "original")
    
    # Generate evaluation report
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    report_generator = EvaluationReport(assets_dir)
    report = report_generator.generate_report(
        model_results, edge_results, robustness_results, communication_results
    )
    
    logger.info("Evaluation report generated successfully")
    
    # Deployment demonstration
    logger.info("Demonstrating deployment pipeline...")
    
    # Deploy to different devices
    target_devices = ["raspberry_pi_4", "jetson_nano", "android"]
    
    for device_name in target_devices:
        try:
            logger.info(f"Deploying to {device_name}...")
            deployment_dir = Path("deployments") / device_name
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline = DeploymentPipeline(device_name, deployment_dir)
            deployment_info = pipeline.deploy_model(
                original_model, input_shape=(4,), model_name="parking_classifier"
            )
            
            logger.info(f"Deployment to {device_name} completed successfully")
            logger.info(f"Model size: {deployment_info['exports'].get('onnx', {}).get('model_size_mb', 'N/A'):.2f} MB")
            
        except Exception as e:
            logger.error(f"Deployment to {device_name} failed: {e}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    logger.info("Model Performance Summary:")
    for model_name, metrics in model_results.items():
        logger.info(f"  {model_name}: Accuracy = {metrics.accuracy:.4f}")
    
    logger.info("Edge Performance Summary:")
    for model_name, metrics in edge_results.items():
        logger.info(f"  {model_name}: Latency = {metrics.avg_latency_ms:.2f} ms, Size = {metrics.model_size_mb:.2f} MB")
    
    logger.info("Optimization Results:")
    if "overall_compression" in optimization_results:
        compression = optimization_results["overall_compression"]
        logger.info(f"  Overall compression: {compression['compression_ratio']:.2f}x")
        logger.info(f"  Size reduction: {compression['size_reduction_percent']:.1f}%")
    
    logger.info("Files generated:")
    logger.info(f"  - Models: {models_dir}")
    logger.info(f"  - Assets: {assets_dir}")
    logger.info(f"  - Deployments: deployments/")
    
    logger.info("To run the interactive demo:")
    logger.info("  streamlit run demo/app.py")

if __name__ == "__main__":
    main()
