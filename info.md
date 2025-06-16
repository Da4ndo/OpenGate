## OpenGate Detection Integration

A robust Home Assistant integration for gate detection using advanced texture pattern analysis.

### Features
- **Lighting Independent Detection**: Uses texture patterns instead of colors
- **Real-time Monitoring**: Live gate state detection via binary sensor
- **RTSP Camera Support**: Optimized for IP cameras with low latency
- **Perspective Correction**: Automatic correction for camera angles
- **Temporal Smoothing**: Reduces false positives through intelligent filtering

### What You Get
- **Binary Sensor**: `binary_sensor.gate_state` - Main gate state (ON = open, OFF = closed)
- **Confidence Sensor**: `sensor.gate_detection_confidence` - Detection confidence percentage
- **Similarity Sensor**: `sensor.gate_pattern_similarity` - Pattern similarity percentage

### Setup Required
1. Install the integration through HACS
2. Add the integration via Home Assistant UI
3. Configure your camera URL and detection parameters
4. **Important**: You must calibrate the system using the provided services:
   - `opengate_detection.set_roi` - Define the gate detection area
   - `opengate_detection.recalibrate` - Capture the closed gate pattern

### Supported Cameras
- RTSP IP Cameras (Hikvision, Dahua, etc.)
- HTTP/MJPEG streams  
- Local USB cameras

### Why This Integration?
Unlike color-based detection systems, OpenGate uses texture analysis which makes it:
- **Weather resistant** - Works in rain, snow, or fog
- **Shadow tolerant** - Not affected by changing shadows
- **Material agnostic** - Works with any gate material or color
- **Lighting independent** - Functions in day, night, or artificial lighting

Perfect for reliable gate automation and security monitoring! 