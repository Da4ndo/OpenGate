"""Binary sensor platform for OpenGate Detection integration."""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from homeassistant.components.binary_sensor import (
	BinarySensorDeviceClass,
	BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
	CoordinatorEntity,
	DataUpdateCoordinator,
	UpdateFailed,
)
from homeassistant.helpers.device_registry import DeviceInfo

from .const import (
	DOMAIN,
	CONF_CAMERA_URL,
	CONF_DETECTION_INTERVAL,
	CONF_CONFIDENCE_THRESHOLD,
	CONF_PATTERN_SIMILARITY,
	ATTR_CONFIDENCE,
	ATTR_SIMILARITY_SCORE,
	ATTR_EDGE_DENSITY_RATIO,
	ATTR_TEXTURE_VARIANCE_RATIO,
	ATTR_LAST_DETECTION,
	DEVICE_MANUFACTURER,
	DEVICE_MODEL,
	DEVICE_NAME,
)
from .gate_detector import GateDetector, DetectionResult

_LOGGER = logging.getLogger(__name__)


class OpenGateDataUpdateCoordinator(DataUpdateCoordinator):
	"""Class to manage fetching data from the OpenGate detector."""

	def __init__(
		self,
		hass: HomeAssistant,
		camera_url: str,
		config: dict[str, Any],
		calibration_data: dict[str, Any],
	) -> None:
		"""Initialize."""
		self.camera_url = camera_url
		self.config = config
		self.calibration_data = calibration_data
		self.detector: GateDetector | None = None
		
		update_interval = timedelta(seconds=config.get(CONF_DETECTION_INTERVAL, 1.0))
		
		super().__init__(
			hass,
			_LOGGER,
			name=DOMAIN,
			update_interval=update_interval,
		)

	async def _async_update_data(self) -> DetectionResult:
		"""Update data via library."""
		try:
			if self.detector is None:
				# Initialize detector in executor
				self.detector = await self.hass.async_add_executor_job(
					self._init_detector
				)
			
			# Run detection in executor to avoid blocking
			result = await self.hass.async_add_executor_job(
				self._detect_gate_state
			)
			
			return result
			
		except Exception as exception:
			raise UpdateFailed(exception) from exception
	
	def _init_detector(self) -> GateDetector:
		"""Initialize the gate detector."""
		return GateDetector(
			self.camera_url,
			self.config,
			self.calibration_data
		)
	
	def _detect_gate_state(self) -> DetectionResult:
		"""Detect gate state and apply smoothing."""
		if self.detector is None:
			raise UpdateFailed("Detector not initialized")
		
		# Get raw detection result
		raw_result = self.detector.detect_gate_state()
		
		# Apply temporal smoothing
		smoothed_result = self.detector.get_smoothed_result(raw_result)
		
		return smoothed_result

	async def async_shutdown(self) -> None:
		"""Cleanup detector resources."""
		if self.detector:
			await self.hass.async_add_executor_job(self.detector.cleanup)


async def async_setup_entry(
	hass: HomeAssistant,
	config_entry: ConfigEntry,
	async_add_entities: AddEntitiesCallback,
) -> None:
	"""Set up the OpenGate binary sensor from a config entry."""
	camera_url = config_entry.data[CONF_CAMERA_URL]
	
	# Get calibration data (this would normally be loaded from storage)
	# For now, we'll use the existing calibration data file
	import pickle
	import os
	
	calibration_path = os.path.join(hass.config.path(), "custom_components", "opengate_detection", "calibration_data.pkl")
	if not os.path.exists(calibration_path):
		# Try fallback location
		calibration_path = os.path.join(hass.config.path(), "calibration_data.pkl")
	
	try:
		with open(calibration_path, 'rb') as f:
			calibration_data = pickle.load(f)
	except Exception as e:
		_LOGGER.error(f"Could not load calibration data: {e}")
		# Use dummy calibration data - user needs to configure this
		calibration_data = {
			'roi_points': [[100, 100], [200, 100], [200, 200], [100, 200]],
			'closed_state_signature': {
				'lbp_histogram': [0.1] * 26,
				'edge_density': 0.1,
				'texture_variance': 100.0,
			}
		}
	
	# Create coordinator
	coordinator = OpenGateDataUpdateCoordinator(
		hass,
		camera_url,
		config_entry.data,
		calibration_data,
	)

	# Store coordinator for sensor platform
	hass.data[DOMAIN][f"{config_entry.entry_id}_coordinator"] = coordinator

	# Fetch initial data so we have data when entities subscribe
	await coordinator.async_config_entry_first_refresh()

	# Create entity
	async_add_entities([OpenGateBinarySensor(coordinator, config_entry)])


class OpenGateBinarySensor(CoordinatorEntity, BinarySensorEntity):
	"""Representation of an OpenGate binary sensor."""

	def __init__(
		self,
		coordinator: OpenGateDataUpdateCoordinator,
		config_entry: ConfigEntry,
	) -> None:
		"""Initialize the binary sensor."""
		super().__init__(coordinator)
		self.config_entry = config_entry
		self._attr_unique_id = f"{config_entry.entry_id}_gate_state"
		self._attr_name = "Gate State"
		self._attr_device_class = BinarySensorDeviceClass.OPENING

	@property
	def device_info(self) -> DeviceInfo:
		"""Return device information."""
		return DeviceInfo(
			identifiers={(DOMAIN, self.config_entry.entry_id)},
			name=DEVICE_NAME,
			manufacturer=DEVICE_MANUFACTURER,
			model=DEVICE_MODEL,
			sw_version="1.0.0",
		)

	@property
	def is_on(self) -> bool | None:
		"""Return True if gate is open."""
		if self.coordinator.data is None:
			return None
		return not self.coordinator.data.is_closed  # Binary sensor is ON when gate is OPEN

	@property
	def extra_state_attributes(self) -> dict[str, Any]:
		"""Return the state attributes."""
		if self.coordinator.data is None:
			return {}
		
		data = self.coordinator.data
		return {
			ATTR_CONFIDENCE: round(data.confidence, 3),
			ATTR_SIMILARITY_SCORE: round(data.similarity_score, 3),
			ATTR_EDGE_DENSITY_RATIO: round(data.edge_density_ratio, 3),
			ATTR_TEXTURE_VARIANCE_RATIO: round(data.texture_variance_ratio, 3),
			ATTR_LAST_DETECTION: data.timestamp,
			"gate_status": "open" if not data.is_closed else "closed",
		}

	@property
	def available(self) -> bool:
		"""Return True if entity is available."""
		return self.coordinator.last_update_success 