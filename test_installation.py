"""
Simple installation test for the parking space detection system.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_installation():
    """Test that all components can be imported and basic functionality works."""
    print("Testing Parking Space Detection System Installation...")
    
    try:
        # Test PyTorch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test imports
        from src.models.parking_classifier import ParkingSpaceClassifier, generate_sensor_data
        from src.models.optimization import ModelOptimizer
        from src.pipelines.data_pipeline import SensorSimulator
        from src.utils.evaluation import ModelEvaluator
        from src.export.deployment import DeviceConfig
        
        print("✓ All modules imported successfully")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        # Test data generation
        features, labels = generate_sensor_data(n_samples=10)
        print(f"✓ Generated {len(features)} sensor samples")
        
        # Test model creation
        model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[16, 8])
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test model forward pass
        x = torch.randn(1, 4)
        output = model(x)
        print(f"✓ Model forward pass successful, output shape: {output.shape}")
        
        # Test sensor simulator
        simulator = SensorSimulator(num_spots=3)
        reading = simulator.generate_reading("spot_01")
        print(f"✓ Generated sensor reading for {reading['spot_id']}")
        
        # Test device config
        devices = DeviceConfig.list_devices()
        print(f"✓ Found {len(devices)} device configurations")
        
        # Test evaluator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = ModelEvaluator(device)
        print(f"✓ Created evaluator for device: {device}")
        
        print("\n🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train_and_evaluate.py")
        print("2. Launch demo: streamlit run demo/app.py")
        print("3. Run tests: pytest tests/ -v")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("\nPlease check your installation and dependencies.")
        return False

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
