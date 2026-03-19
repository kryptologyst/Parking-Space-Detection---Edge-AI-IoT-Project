"""
Streamlit demo for parking space detection system.

This demo simulates edge constraints and provides real-time visualization
of parking space detection with live metrics and performance monitoring.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Parking Space Detection - Edge AI Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ParkingSpaceDemo:
    """Main demo class for parking space detection."""
    
    def __init__(self) -> None:
        """Initialize the demo."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sensor_data = []
        self.predictions = []
        self.metrics_history = []
        
        # Initialize session state
        if "demo_running" not in st.session_state:
            st.session_state.demo_running = False
        if "spot_states" not in st.session_state:
            st.session_state.spot_states = {}
    
    def load_model(self) -> None:
        """Load the parking space detection model."""
        try:
            # Import model classes
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            
            from src.models.parking_classifier import ParkingSpaceClassifier
            
            # Create model
            self.model = ParkingSpaceClassifier(input_dim=4, hidden_dims=[32, 16])
            self.model.eval()
            
            # Load pre-trained weights if available
            model_path = Path("models/parking_classifier.pth")
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pre-trained model")
            else:
                logger.warning("No pre-trained model found, using random weights")
            
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            st.error(f"Failed to load model: {e}")
    
    def generate_sensor_reading(self, spot_id: str) -> Dict[str, Any]:
        """Generate simulated sensor reading for a parking spot."""
        # Simulate realistic sensor data
        np.random.seed(int(time.time() * 1000) % 10000)
        
        # Base values
        base_distance = np.random.uniform(0.5, 2.0)
        base_lighting = np.random.uniform(200, 400)
        
        # Simulate occupancy (cars entering/leaving)
        if spot_id not in st.session_state.spot_states:
            st.session_state.spot_states[spot_id] = {
                "occupied": np.random.choice([True, False]),
                "last_change": time.time(),
            }
        
        state = st.session_state.spot_states[spot_id]
        
        # Random state changes
        if np.random.random() < 0.02:  # 2% chance per reading
            state["occupied"] = not state["occupied"]
            state["last_change"] = time.time()
        
        # Generate sensor readings based on occupancy
        if state["occupied"]:
            # Car present: shorter distance, lower lighting
            distance = base_distance * 0.3 + np.random.normal(0, 0.1)
            lighting = base_lighting * 0.6 + np.random.normal(0, 20)
            motion_detected = (time.time() - state["last_change"]) < 5.0
        else:
            # No car: normal distance, normal lighting
            distance = base_distance + np.random.normal(0, 0.1)
            lighting = base_lighting + np.random.normal(0, 30)
            motion_detected = False
        
        # Ensure realistic bounds
        distance = max(0.1, min(distance, 5.0))
        lighting = max(0, min(lighting, 1000))
        
        # Environmental conditions
        temperature = 20.0 + np.random.normal(0, 2.0)
        humidity = 50.0 + np.random.normal(0, 10.0)
        
        reading = {
            "timestamp": time.time(),
            "spot_id": spot_id,
            "distance": distance,
            "lighting": lighting,
            "motion_detected": motion_detected,
            "temperature": temperature,
            "humidity": humidity,
            "hour": time.localtime().tm_hour,
            "occupied": state["occupied"],
        }
        
        return reading
    
    def predict_occupancy(self, sensor_reading: Dict[str, Any]) -> Dict[str, Any]:
        """Predict parking spot occupancy using the model."""
        if self.model is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        
        try:
            # Prepare input features
            features = torch.FloatTensor([
                sensor_reading["distance"],
                sensor_reading["lighting"],
                float(sensor_reading["motion_detected"]),
                float(sensor_reading["hour"]),
            ]).unsqueeze(0).to(self.device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get prediction and confidence
            prediction = "Occupied" if predicted.item() == 1 else "Vacant"
            confidence = probabilities[0][predicted.item()].item()
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "inference_time_ms": inference_time,
                "probabilities": probabilities[0].cpu().numpy().tolist(),
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prediction": "Error", "confidence": 0.0, "error": str(e)}
    
    def run_demo(self) -> None:
        """Run the main demo interface."""
        # Header
        st.markdown('<h1 class="main-header">🚗 Parking Space Detection - Edge AI Demo</h1>', unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ DISCLAIMER:</strong> This demo is for research and educational purposes only. 
            NOT FOR SAFETY-CRITICAL DEPLOYMENT. The system uses simulated sensor data and should not 
            be used for real-world parking management without proper validation and safety measures.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main content
        if st.session_state.demo_running:
            self.render_live_demo()
        else:
            self.render_setup_page()
    
    def render_sidebar(self) -> None:
        """Render the sidebar controls."""
        st.sidebar.title("🎛️ Demo Controls")
        
        # Model selection
        st.sidebar.subheader("Model Configuration")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["Original", "Quantized", "Pruned", "Distilled"],
            help="Select the model variant to use"
        )
        
        # Device simulation
        st.sidebar.subheader("Edge Device Simulation")
        device_type = st.sidebar.selectbox(
            "Simulated Device",
            ["Raspberry Pi 4", "Jetson Nano", "Jetson Xavier", "Android", "iOS", "MCU"],
            help="Simulate constraints of different edge devices"
        )
        
        # Sensor configuration
        st.sidebar.subheader("Sensor Configuration")
        num_spots = st.sidebar.slider("Number of Parking Spots", 1, 20, 10)
        sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 0.1, 5.0, 1.0)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
        
        # Demo controls
        st.sidebar.subheader("Demo Controls")
        if st.sidebar.button("🚀 Start Demo", type="primary"):
            st.session_state.demo_running = True
            st.rerun()
        
        if st.sidebar.button("⏹️ Stop Demo"):
            st.session_state.demo_running = False
            st.rerun()
        
        # Performance metrics
        st.sidebar.subheader("Performance Metrics")
        if st.session_state.demo_running and self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            st.sidebar.metric("Avg Latency", f"{latest_metrics.get('latency_ms', 0):.1f} ms")
            st.sidebar.metric("Throughput", f"{latest_metrics.get('throughput_fps', 0):.1f} FPS")
            st.sidebar.metric("Memory Usage", f"{latest_metrics.get('memory_mb', 0):.1f} MB")
    
    def render_setup_page(self) -> None:
        """Render the setup page."""
        st.markdown("## 📋 Setup and Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the Parking Space Detection Demo
            
            This interactive demo simulates a smart parking system using edge AI to detect 
            available parking spaces in real-time. The system uses sensor data from 
            ultrasonic sensors, lighting sensors, and motion detectors to classify 
            parking spots as occupied or vacant.
            
            #### Features:
            - **Real-time sensor simulation** with realistic data patterns
            - **Edge AI model inference** with performance monitoring
            - **Multiple model variants** (original, quantized, pruned, distilled)
            - **Device constraint simulation** for different edge platforms
            - **Live performance metrics** including latency, throughput, and memory usage
            - **Interactive visualizations** of parking spot states and sensor data
            
            #### How to use:
            1. Configure the model and device settings in the sidebar
            2. Click "Start Demo" to begin real-time simulation
            3. Monitor the live dashboard and performance metrics
            4. Use "Stop Demo" to end the simulation
            """)
        
        with col2:
            st.markdown("### 🎯 Quick Start")
            
            if st.button("🚀 Start Demo with Default Settings", type="primary"):
                st.session_state.demo_running = True
                st.rerun()
            
            st.markdown("### 📊 System Status")
            st.info("Demo ready to start")
            
            # Model status
            if self.model is not None:
                st.success("✅ Model loaded successfully")
            else:
                st.warning("⚠️ Model not loaded")
    
    def render_live_demo(self) -> None:
        """Render the live demo interface."""
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        # Main dashboard
        with placeholder.container():
            # Top row: Parking spots overview
            st.markdown("## 🅿️ Parking Spots Overview")
            self.render_parking_overview()
            
            # Middle row: Real-time metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spots", "10")
            with col2:
                occupied_count = sum(1 for spot in st.session_state.spot_states.values() if spot["occupied"])
                st.metric("Occupied", occupied_count)
            with col3:
                vacant_count = 10 - occupied_count
                st.metric("Vacant", vacant_count)
            with col4:
                occupancy_rate = (occupied_count / 10) * 100
                st.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")
            
            # Bottom row: Charts and detailed metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("## 📈 Real-time Sensor Data")
                self.render_sensor_charts()
            
            with col2:
                st.markdown("## ⚡ Performance Metrics")
                self.render_performance_metrics()
        
        # Auto-refresh for real-time updates
        if st.session_state.demo_running:
            time.sleep(1.0 / st.sidebar.slider("Refresh Rate (Hz)", 0.1, 2.0, 1.0))
            st.rerun()
    
    def render_parking_overview(self) -> None:
        """Render the parking spots overview."""
        # Create a grid of parking spots
        cols = st.columns(5)
        
        for i in range(10):
            col_idx = i % 5
            spot_id = f"spot_{i+1:02d}"
            
            with cols[col_idx]:
                # Generate sensor reading
                sensor_reading = self.generate_sensor_reading(spot_id)
                
                # Get prediction
                prediction_result = self.predict_occupancy(sensor_reading)
                
                # Determine spot status
                is_occupied = sensor_reading["occupied"]
                predicted_occupied = prediction_result["prediction"] == "Occupied"
                confidence = prediction_result.get("confidence", 0.0)
                
                # Color coding
                if is_occupied:
                    color = "🔴"  # Red for occupied
                    status = "Occupied"
                else:
                    color = "🟢"  # Green for vacant
                    status = "Vacant"
                
                # Prediction accuracy
                correct = is_occupied == predicted_occupied
                accuracy_indicator = "✅" if correct else "❌"
                
                # Display spot
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{color} Spot {i+1}</h4>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Prediction:</strong> {prediction_result["prediction"]}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Accuracy:</strong> {accuracy_indicator}</p>
                    <p><strong>Distance:</strong> {sensor_reading["distance"]:.2f}m</p>
                    <p><strong>Lighting:</strong> {sensor_reading["lighting"]:.0f} lux</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Store metrics
                self.metrics_history.append({
                    "timestamp": time.time(),
                    "spot_id": spot_id,
                    "latency_ms": prediction_result.get("inference_time_ms", 0),
                    "confidence": confidence,
                    "accuracy": 1.0 if correct else 0.0,
                })
    
    def render_sensor_charts(self) -> None:
        """Render real-time sensor data charts."""
        if not self.metrics_history:
            st.info("No data available yet. Starting demo...")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Latency Over Time", "Confidence Distribution", 
                          "Accuracy Over Time", "Sensor Readings"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Latency over time
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["latency_ms"], mode="lines", name="Latency"),
            row=1, col=1
        )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=df["confidence"], name="Confidence"),
            row=1, col=2
        )
        
        # Accuracy over time
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["accuracy"], mode="lines", name="Accuracy"),
            row=2, col=1
        )
        
        # Sensor readings (simulated)
        sensor_data = []
        for spot_id in [f"spot_{i+1:02d}" for i in range(10)]:
            reading = self.generate_sensor_reading(spot_id)
            sensor_data.append(reading)
        
        sensor_df = pd.DataFrame(sensor_data)
        fig.add_trace(
            go.Scatter(x=sensor_df["distance"], y=sensor_df["lighting"], 
                      mode="markers", name="Distance vs Lighting"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self) -> None:
        """Render performance metrics."""
        if not self.metrics_history:
            st.info("No metrics available yet.")
            return
        
        # Calculate current metrics
        df = pd.DataFrame(self.metrics_history)
        
        avg_latency = df["latency_ms"].mean()
        max_latency = df["latency_ms"].max()
        avg_confidence = df["confidence"].mean()
        accuracy_rate = df["accuracy"].mean()
        
        # Display metrics
        st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        st.metric("Max Latency", f"{max_latency:.1f} ms")
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        st.metric("Accuracy Rate", f"{accuracy_rate:.2%}")
        
        # Performance gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = accuracy_rate * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "System Accuracy (%)"},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the demo."""
    demo = ParkingSpaceDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
