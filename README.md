# 🚪 OpenGate Detection - Home Assistant Integration

[![hacs][hacsbadge]][hacs]
[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

![Project Maintenance][maintenance-shield]

> **A robust Home Assistant integration for gate detection using advanced texture pattern analysis.**  
> Lighting-independent detection that adapts to different environmental conditions by analyzing texture patterns within the gate area.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🌅 Lighting Independent** | Uses texture patterns instead of color values |
| **🔧 Perspective Correction** | Automatic 4-point ROI perspective correction |
| **🎯 Adaptive Calibration** | Learns closed gate patterns from multiple samples |
| **⚡ Real-time Detection** | Fast texture analysis for live video streams |
| **🧠 Pattern Matching** | LBP, edge analysis, and statistical features |
| **📊 Temporal Smoothing** | Reduces false positives through history analysis |
| **🏠 HA Integration** | Native binary sensor and diagnostic entities |
| **📹 RTSP Support** | Optimized for IP cameras with low latency |

---

## 📦 Installation

### 🎪 **HACS (Recommended)**

**Quick Setup Steps:**
1. 🏠 Open HACS → Integrations
2. ⚙️ Three dots → Custom repositories  
3. ➕ Add: `https://github.com/Da4ndo/OpenGate`
4. 📂 Category: Integration
5. 🔍 Search & Install "OpenGate Detection"
6. 🔄 Restart Home Assistant

### 📁 **Manual Installation**

```bash
# Copy integration files
cp -r custom_components/opengate_detection /config/custom_components/
# Restart Home Assistant
```

---

## ⚙️ Configuration

### 🚀 **Initial Setup**

> **Path:** Settings → Devices & Services → Add Integration → OpenGate Detection

**Required Configuration:**
- **📹 Camera URL:** `rtsp://user:pass@192.168.1.100:554/stream1`
- **⏱️ Detection Interval:** `1.0` seconds
- **🎯 Confidence Threshold:** `0.8` (0.0-1.0)
- **📊 Pattern Similarity:** `0.85` (0.0-1.0)

### 📹 **Camera URL Configuration**

The camera URL is the most critical configuration parameter. Here's how to configure it correctly:

#### **🌐 RTSP Cameras** _(Most Common)_

**Format:** `rtsp://[username]:[password]@[ip_address]:[port]/[stream_path]`

| Camera Brand | URL Format Example |
|--------------|-------------------|
| **Hikvision** | `rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/101` |
| **Dahua** | `rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0` |
| **Reolink** | `rtsp://admin:password123@192.168.1.100:554/h264Preview_01_main` |
| **Tapo/TP-Link** | `rtsp://admin:password123@192.168.1.100:554/stream1` |
| **Generic IP Camera** | `rtsp://user:pass@192.168.1.100:554/stream1` |

#### **📡 HTTP/MJPEG Streams**

**Format:** `http://[ip_address]:[port]/[stream_path]`

- **MJPEG Stream:** `http://192.168.1.100:8080/video.mjpg`
- **HTTP Stream:** `http://admin:password@192.168.1.100/cgi-bin/mjpg/video.cgi`

#### **🔌 Local USB Cameras**

For locally connected USB cameras, use the device index:
- **Primary USB Camera:** `0`
- **Secondary USB Camera:** `1`
- **Third USB Camera:** `2`

#### **🔍 Finding Your Camera URL**

1. **Check Camera Manual/Web Interface:**
   - Login to camera's web interface
   - Look for "RTSP" or "Streaming" settings
   - Note the stream path and port

2. **Common RTSP Ports:**
   - **554** (standard RTSP port)
   - **8554** (alternative RTSP port)

3. **Test Your URL:**
   ```bash
   # Test with VLC or ffplay
   vlc rtsp://user:pass@192.168.1.100:554/stream1
   # or
   ffplay rtsp://user:pass@192.168.1.100:554/stream1
   ```

#### **⚠️ Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| **Authentication Failed** | Double-check username/password in camera settings |
| **Connection Timeout** | Verify IP address and port are correct |
| **Stream Not Found** | Check stream path in camera documentation |
| **Firewall Blocking** | Ensure RTSP port (554) is open |

### 🎯 **Calibration** ⚠️ **REQUIRED**

> **⚠️ Critical:** System requires calibration before accurate detection

```yaml
# Step 1: Set ROI (Region of Interest)
service: opengate_detection.set_roi
target:
  entity_id: binary_sensor.gate_state
data:
  points: "[[100,100], [300,100], [300,250], [100,250]]"

# Step 2: Calibrate (with gate CLOSED)
service: opengate_detection.recalibrate
target:
  entity_id: binary_sensor.gate_state
data:
  samples: 10
```

---

## 🎛️ Entities Created

| Entity | Type | Description |
|--------|------|-------------|
| `binary_sensor.gate_state` | **Binary Sensor** | 🟢 **ON** = Open / 🔴 **OFF** = Closed |
| `sensor.gate_detection_confidence` | **Sensor** | Detection confidence (0-100%) |
| `sensor.gate_pattern_similarity` | **Sensor** | Pattern similarity score (0-100%) |

**Available Attributes:**
- `confidence` - Detection confidence level
- `similarity_score` - Pattern matching score  
- `edge_density_ratio` - Edge analysis result
- `texture_variance_ratio` - Texture analysis result
- `last_detection` - Timestamp of last detection

---

## 📹 Camera Compatibility

### **🌐 RTSP Cameras** _(Recommended)_

| Brand | URL Format |
|-------|------------|
| **Hikvision** | `rtsp://user:pass@ip:554/Streaming/Channels/101` |
| **Dahua** | `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0` |
| **Generic** | `rtsp://user:pass@ip:554/stream1` |

### **📡 Alternative Sources**

- **HTTP/MJPEG:** `http://192.168.1.100:8080/stream`
- **USB Cameras:** `0`, `1`, `2`, etc.

---

## 🔬 How It Works

### **Texture Analysis Engine**

```mermaid
graph LR
    A[Camera Feed] --> B[ROI Extraction]
    B --> C[Texture Analysis]
    C --> D[Pattern Matching]
    D --> E[Temporal Smoothing]
    E --> F[Gate State]
```

**Analysis Features:**
1. **🔍 Local Binary Patterns (LBP)** - Texture pattern capture
2. **⚡ Edge Density Analysis** - Edge detection metrics  
3. **📊 Texture Variance** - Homogeneity analysis
4. **📈 Shannon Entropy** - Information content measurement
5. **📐 Gradient Analysis** - Directional texture features

### **Detection Logic**

> **✅ High Similarity** → Gate CLOSED  
> **❌ Low Similarity** → Gate OPEN  
> **📊 History Analysis** → Noise reduction

---

## 🏆 Advantages

| **vs Color Detection** | **OpenGate Advantage** |
|------------------------|------------------------|
| 🌅 **Lighting Changes** | ✅ **Lighting Independent** |
| 🌑 **Shadow Problems** | ✅ **Shadow Tolerant** |
| 🎨 **Material Dependent** | ✅ **Material Agnostic** |
| 📐 **Angle Sensitive** | ✅ **Perspective Robust** |
| 🌦️ **Weather Issues** | ✅ **Weather Resistant** |

---

## 🔧 Troubleshooting

<details>
<summary><strong>⚠️ Calibration Issues</strong></summary>

- ✅ **Gate Position:** Ensure gate is fully closed during calibration
- 📍 **ROI Points:** Verify points accurately cover the gate area  
- 💡 **Lighting:** Provide stable lighting during calibration
- 🔄 **Multiple Samples:** Use 10+ samples for better accuracy

</details>

<details>
<summary><strong>🚨 Detection Problems</strong></summary>

- 📡 **Connectivity:** Check camera stream quality and connection
- ⚖️ **Thresholds:** Adjust similarity/confidence if too sensitive
- 📍 **ROI Validation:** Verify ROI points match current camera view
- 🔄 **Recalibration:** Re-run calibration if environment changed

</details>

<details>
<summary><strong>⚡ Performance Issues</strong></summary>

- 🔄 **Detection Interval:** Increase interval (reduce frequency)
- 💻 **Resources:** Ensure adequate CPU/memory for image processing
- 📹 **Stream Quality:** Lower resolution if processing is slow
- 🎯 **ROI Size:** Smaller ROI = faster processing

</details>

---

## 👨‍💻 Development Info

**Built with Home Assistant Best Practices:**
- 🎛️ **Config Flow** - UI-based setup
- 📊 **DataUpdateCoordinator** - Efficient data fetching  
- 🏷️ **Device Registry** - Proper entity management
- 🔍 **Error Handling** - Comprehensive logging

---

## 🤝 Support & Contributing

| Resource | Link |
|----------|------|
| **🐛 Bug Reports** | [GitHub Issues](https://github.com/Da4ndo/OpenGate/issues) |
| **💬 Discussions** | [GitHub Discussions](https://github.com/Da4ndo/OpenGate/discussions) |
| **📖 Contributing** | [Contributing Guidelines](CONTRIBUTING.md) |
| **📄 License** | [BSD 3-Clause License](LICENSE) |

---

**Made with ❤️ by [@Da4ndo](https://github.com/Da4ndo)**

[hacs]: https://github.com/hacs/integration
[hacsbadge]: https://img.shields.io/badge/HACS-Custom-orange.svg?style=for-the-badge
[releases-shield]: https://img.shields.io/github/release/Da4ndo/OpenGate.svg?style=for-the-badge
[releases]: https://github.com/Da4ndo/OpenGate/releases
[commits-shield]: https://img.shields.io/github/commit-activity/y/Da4ndo/OpenGate.svg?style=for-the-badge
[commits]: https://github.com/Da4ndo/OpenGate/commits/main
[license-shield]: https://img.shields.io/github/license/Da4ndo/OpenGate.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-Da4ndo-blue.svg?style=for-the-badge 