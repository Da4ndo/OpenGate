"""Constants for the OpenGate Detection integration."""
from __future__ import annotations

from typing import Final

DOMAIN: Final = "opengate_detection"

# Default values
DEFAULT_NAME = "OpenGate Detection"
DEFAULT_DETECTION_INTERVAL = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_PATTERN_SIMILARITY = 0.85

# Configuration keys
CONF_CAMERA_URL = "camera_url"
CONF_DETECTION_INTERVAL = "detection_interval"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_PATTERN_SIMILARITY = "pattern_similarity"
CONF_ROI_POINTS = "roi_points"
CONF_CALIBRATION_DATA = "calibration_data"

# Entity attributes
ATTR_CONFIDENCE = "confidence"
ATTR_SIMILARITY_SCORE = "similarity_score"
ATTR_EDGE_DENSITY_RATIO = "edge_density_ratio"
ATTR_TEXTURE_VARIANCE_RATIO = "texture_variance_ratio"
ATTR_LAST_DETECTION = "last_detection"

# Entity device info
DEVICE_MANUFACTURER = "Da4ndo"
DEVICE_MODEL = "OpenGate Detection"
DEVICE_NAME = "OpenGate" 