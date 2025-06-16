"""
OpenGate Texture-Based Detection System adapted for Home Assistant.
Real-time gate state detection using texture pattern analysis.
"""

import cv2
import numpy as np
import pickle
import time
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import logging
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
	"""Result of gate detection analysis."""
	is_closed: bool
	confidence: float
	similarity_score: float
	edge_density_ratio: float
	texture_variance_ratio: float
	timestamp: float


class RTSPCamera:
	"""Optimized RTSP camera handler with reconnection and low-latency settings."""
	
	def __init__(self, camera_url: str):
		"""Initialize RTSP camera with URL."""
		self.camera_url = camera_url
		self.cap: Optional[cv2.VideoCapture] = None
		self.is_connected = False
		self.reconnect_count = 0
		self.max_reconnect_attempts = 5
		
		_LOGGER.info(f"Initializing camera: {self._sanitize_url(camera_url)}")
	
	def _sanitize_url(self, url: str) -> str:
		"""Remove credentials from URL for logging."""
		if isinstance(url, str) and '@' in url:
			parts = url.split('@')
			if len(parts) == 2:
				protocol_and_creds = parts[0]
				rest = parts[1]
				if '://' in protocol_and_creds:
					protocol = protocol_and_creds.split('://')[0]
					return f"{protocol}://***:***@{rest}"
		return str(url)
	
	def connect(self) -> bool:
		"""Connect to camera with optimized settings."""
		try:
			if self.is_connected and self.cap and self.cap.isOpened():
				return True
			
			self._disconnect()
			
			# Create VideoCapture with FFmpeg backend for RTSP
			if isinstance(self.camera_url, str) and self.camera_url.startswith('rtsp://'):
				self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
			else:
				# Local camera or file
				camera_index = int(self.camera_url) if self.camera_url.isdigit() else self.camera_url
				self.cap = cv2.VideoCapture(camera_index)
			
			if not self.cap.isOpened():
				_LOGGER.error("Failed to open camera stream")
				return False
			
			# Configure for optimal streaming
			self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
			self.cap.set(cv2.CAP_PROP_FPS, 30)
			
			# Test connection by reading a frame
			ret, frame = self.cap.read()
			if not ret or frame is None:
				_LOGGER.error("Failed to read initial frame")
				self._disconnect()
				return False
			
			self.is_connected = True
			self.reconnect_count = 0
			_LOGGER.info("Successfully connected to camera stream")
			
			return True
			
		except Exception as e:
			_LOGGER.error(f"Error connecting to camera: {e}")
			self._disconnect()
			return False
	
	def _disconnect(self) -> None:
		"""Disconnect from camera."""
		if self.cap:
			self.cap.release()
			self.cap = None
		self.is_connected = False
	
	def read_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
		"""Read frame with automatic reconnection on failure."""
		if not self.is_connected:
			if not self.connect():
				return False, None
		
		try:
			ret, frame = self.cap.read()
			
			if not ret or frame is None:
				_LOGGER.warning("Failed to read frame, attempting reconnection...")
				self.is_connected = False
				
				# Attempt reconnection
				if self.reconnect_count < self.max_reconnect_attempts:
					self.reconnect_count += 1
					time.sleep(1)  # Brief delay before reconnection
					
					if self.connect():
						ret, frame = self.cap.read()
						if ret and frame is not None:
							return True, frame
				
				_LOGGER.error("Max reconnection attempts reached")
				return False, None
			
			return True, frame
			
		except Exception as e:
			_LOGGER.error(f"Error reading frame: {e}")
			self.is_connected = False
			return False, None
	
	def release(self) -> None:
		"""Release camera resources."""
		_LOGGER.info("Releasing camera resources")
		self._disconnect()


class GateDetector:
	"""Real-time gate state detection using texture analysis."""
	
	def __init__(self, camera_url: str, config: Dict[str, Any], calibration_data: Dict[str, Any]):
		"""Initialize detector with configuration and calibration data."""
		self.camera_url = camera_url
		self.config = config
		self.calibration_data = calibration_data
		self.detection_history = deque(maxlen=config.get('smoothing_window', 5))
		self.camera = None
		
		# Validate calibration data
		if not self._validate_calibration():
			raise ValueError("Invalid or missing calibration data")
		
		_LOGGER.info("Gate detector initialized successfully")
	
	def _validate_calibration(self) -> bool:
		"""Validate that calibration data contains required fields."""
		required_fields = ['roi_points', 'closed_state_signature']
		
		if not all(field in self.calibration_data for field in required_fields):
			_LOGGER.error("Missing required calibration fields")
			return False
		
		if len(self.calibration_data['roi_points']) != 4:
			_LOGGER.error("Invalid ROI points in calibration data")
			return False
		
		return True
	
	def extract_roi(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
		"""Extract region of interest using perspective transformation."""
		if len(points) != 4:
			raise ValueError("Exactly 4 points required for ROI extraction")
		
		# Convert points to numpy array
		src_points = np.array(points, dtype=np.float32)
		
		# Define destination points for perspective correction
		width = max(
			np.linalg.norm(src_points[1] - src_points[0]),
			np.linalg.norm(src_points[2] - src_points[3])
		)
		height = max(
			np.linalg.norm(src_points[3] - src_points[0]),
			np.linalg.norm(src_points[2] - src_points[1])
		)
		
		dst_points = np.array([
			[0, 0],
			[width, 0],
			[width, height],
			[0, height]
		], dtype=np.float32)
		
		# Get perspective transformation matrix
		transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
		
		# Apply transformation
		roi = cv2.warpPerspective(image, transform_matrix, (int(width), int(height)))
		
		return roi
	
	def analyze_texture_pattern(self, roi: np.ndarray) -> Dict[str, Any]:
		"""Analyze texture patterns in the ROI to create a signature."""
		gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
		
		# Local Binary Pattern analysis
		lbp_radius = self.config.get('lbp_radius', 3)
		lbp_n_points = self.config.get('lbp_n_points', 24)
		lbp = local_binary_pattern(gray_roi, lbp_n_points, lbp_radius, method='uniform')
		lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_n_points + 2, range=(0, lbp_n_points + 2))
		lbp_hist = lbp_hist.astype(float)
		lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
		
		# Edge analysis
		edges = cv2.Canny(gray_roi, 50, 150)
		edge_density = np.sum(edges > 0) / edges.size
		
		# Texture homogeneity analysis
		block_size = self.config.get('block_size', 16)
		h, w = gray_roi.shape
		variances = []
		
		for y in range(0, h - block_size, block_size):
			for x in range(0, w - block_size, block_size):
				block = gray_roi[y:y+block_size, x:x+block_size]
				variances.append(np.var(block))
		
		texture_variance = np.mean(variances)
		texture_uniformity = 1.0 / (1.0 + texture_variance)
		
		# Statistical features
		entropy = shannon_entropy(gray_roi)
		mean_intensity = np.mean(gray_roi)
		std_intensity = np.std(gray_roi)
		
		# Gradient magnitude
		grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
		grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
		gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
		mean_gradient = np.mean(gradient_magnitude)
		
		pattern_signature = {
			'lbp_histogram': lbp_hist,
			'edge_density': edge_density,
			'texture_variance': texture_variance,
			'texture_uniformity': texture_uniformity,
			'entropy': entropy,
			'mean_intensity': mean_intensity,
			'std_intensity': std_intensity,
			'mean_gradient': mean_gradient,
			'roi_shape': gray_roi.shape
		}
		
		return pattern_signature
	
	def compare_signatures(self, current_sig: Dict[str, Any], reference_sig: Dict[str, Any]) -> Dict[str, float]:
		"""Compare current signature with reference (closed state) signature."""
		# Histogram comparison using Chi-Square distance
		hist_similarity = 0.0
		if 'lbp_histogram' in current_sig and 'lbp_histogram' in reference_sig:
			# Normalize histograms
			current_hist = current_sig['lbp_histogram']
			reference_hist = reference_sig['lbp_histogram']
			
			# Chi-square distance (lower is more similar)
			eps = 1e-10
			chi_square = 0.5 * np.sum(((current_hist - reference_hist) ** 2) / (current_hist + reference_hist + eps))
			hist_similarity = 1.0 / (1.0 + chi_square)  # Convert to similarity (0-1)
		
		# Edge density comparison
		edge_density_ratio = 1.0
		if 'edge_density' in current_sig and 'edge_density' in reference_sig:
			current_edge = current_sig['edge_density']
			reference_edge = reference_sig['edge_density']
			if reference_edge > 0:
				edge_density_ratio = min(current_edge / reference_edge, reference_edge / current_edge)
		
		# Texture variance comparison
		texture_variance_ratio = 1.0
		if 'texture_variance' in current_sig and 'texture_variance' in reference_sig:
			current_var = current_sig['texture_variance']
			reference_var = reference_sig['texture_variance']
			if reference_var > 0:
				texture_variance_ratio = min(current_var / reference_var, reference_var / current_var)
		
		# Overall similarity score (weighted combination)
		weights = {
			'histogram': 0.4,
			'edge_density': 0.3,
			'texture_variance': 0.3,
		}
		
		overall_similarity = (
			weights['histogram'] * hist_similarity +
			weights['edge_density'] * edge_density_ratio +
			weights['texture_variance'] * texture_variance_ratio
		)
		
		return {
			'overall_similarity': overall_similarity,
			'histogram_similarity': hist_similarity,
			'edge_density_ratio': edge_density_ratio,
			'texture_variance_ratio': texture_variance_ratio,
		}
	
	def detect_gate_state(self) -> DetectionResult:
		"""Detect gate state (open/closed) from current camera image."""
		try:
			# Initialize camera if needed
			if not self.camera:
				self.camera = RTSPCamera(self.camera_url)
			
			# Get current frame
			ret, frame = self.camera.read_frame()
			if not ret or frame is None:
				_LOGGER.error("Could not read frame from camera")
				return DetectionResult(False, 0.0, 0.0, 0.0, 0.0, time.time())
			
			# Extract ROI - convert to tuples if they're lists
			roi_points = self.calibration_data['roi_points']
			if roi_points and isinstance(roi_points[0], list):
				roi_points = [tuple(point) for point in roi_points]
			roi = self.extract_roi(frame, roi_points)
			
			# Analyze current texture pattern
			current_signature = self.analyze_texture_pattern(roi)
			
			# Compare with calibrated closed state
			reference_signature = self.calibration_data['closed_state_signature']
			comparison = self.compare_signatures(current_signature, reference_signature)
			
			# Determine if gate is closed based on similarity
			similarity_threshold = self.config.get('pattern_similarity', 0.85)
			is_closed = comparison['overall_similarity'] >= similarity_threshold
			
			# Calculate confidence
			confidence = comparison['overall_similarity']
			
			# Apply confidence threshold
			confidence_threshold = self.config.get('confidence_threshold', 0.8)
			if confidence < confidence_threshold:
				# Low confidence, use additional heuristics
				if comparison['edge_density_ratio'] < 0.5:
					is_closed = False
					confidence = max(confidence, 0.6)  # Boost confidence for clear open state
			
			result = DetectionResult(
				is_closed=is_closed,
				confidence=confidence,
				similarity_score=comparison['overall_similarity'],
				edge_density_ratio=comparison['edge_density_ratio'],
				texture_variance_ratio=comparison['texture_variance_ratio'],
				timestamp=time.time()
			)
			
			return result
			
		except Exception as e:
			_LOGGER.error(f"Error in gate detection: {e}")
			return DetectionResult(
				is_closed=False,
				confidence=0.0,
				similarity_score=0.0,
				edge_density_ratio=0.0,
				texture_variance_ratio=0.0,
				timestamp=time.time()
			)
	
	def get_smoothed_result(self, current_result: DetectionResult) -> DetectionResult:
		"""Apply temporal smoothing to detection results."""
		self.detection_history.append(current_result)
		
		if len(self.detection_history) < 3:
			return current_result
		
		# Count recent detections
		recent_closed = sum(1 for r in self.detection_history if r.is_closed)
		recent_open = len(self.detection_history) - recent_closed
		
		# Average confidence
		avg_confidence = np.mean([r.confidence for r in self.detection_history])
		avg_similarity = np.mean([r.similarity_score for r in self.detection_history])
		
		# Smoothed decision (majority vote with confidence weighting)
		if recent_closed > recent_open:
			smoothed_state = True
		elif recent_open > recent_closed:
			smoothed_state = False
		else:
			# Tie - use current detection
			smoothed_state = current_result.is_closed
		
		return DetectionResult(
			is_closed=smoothed_state,
			confidence=avg_confidence,
			similarity_score=avg_similarity,
			edge_density_ratio=current_result.edge_density_ratio,
			texture_variance_ratio=current_result.texture_variance_ratio,
			timestamp=current_result.timestamp
		)
	
	def cleanup(self):
		"""Clean up resources."""
		if self.camera:
			self.camera.release()
			self.camera = None 