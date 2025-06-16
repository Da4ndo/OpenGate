"""Sensor platform for OpenGate Detection integration."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.const import PERCENTAGE

from .const import (
	DOMAIN,
	ATTR_CONFIDENCE,
	ATTR_SIMILARITY_SCORE,
	DEVICE_MANUFACTURER,
	DEVICE_MODEL,
	DEVICE_NAME,
)
from .binary_sensor import OpenGateDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
	hass: HomeAssistant,
	config_entry: ConfigEntry,
	async_add_entities: AddEntitiesCallback,
) -> None:
	"""Set up the OpenGate sensors from a config entry."""
	# Get the coordinator from the binary sensor setup
	# This is a bit of a hack - in a real implementation, we'd share the coordinator
	# For now, we'll create the sensors only if the binary sensor exists
	coordinator = hass.data[DOMAIN].get(f"{config_entry.entry_id}_coordinator")
	
	if not coordinator:
		_LOGGER.warning("No coordinator found for OpenGate sensors")
		return

	# Create sensor entities
	entities = [
		OpenGateConfidenceSensor(coordinator, config_entry),
		OpenGateSimilaritySensor(coordinator, config_entry),
	]
	
	async_add_entities(entities)


class OpenGateBaseSensor(SensorEntity):
	"""Base class for OpenGate sensors."""

	def __init__(
		self,
		coordinator: OpenGateDataUpdateCoordinator,
		config_entry: ConfigEntry,
		sensor_type: str,
		name: str,
	) -> None:
		"""Initialize the sensor."""
		self.coordinator = coordinator
		self.config_entry = config_entry
		self.sensor_type = sensor_type
		self._attr_unique_id = f"{config_entry.entry_id}_{sensor_type}"
		self._attr_name = name

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
	def available(self) -> bool:
		"""Return True if entity is available."""
		return self.coordinator.last_update_success

	async def async_added_to_hass(self) -> None:
		"""Subscribe to coordinator updates."""
		await super().async_added_to_hass()
		self.async_on_remove(
			self.coordinator.async_add_listener(self.async_write_ha_state)
		)

	async def async_update(self) -> None:
		"""Update the entity."""
		await self.coordinator.async_request_refresh()


class OpenGateConfidenceSensor(OpenGateBaseSensor):
	"""Sensor for detection confidence."""

	def __init__(
		self,
		coordinator: OpenGateDataUpdateCoordinator,
		config_entry: ConfigEntry,
	) -> None:
		"""Initialize the confidence sensor."""
		super().__init__(coordinator, config_entry, "confidence", "Gate Detection Confidence")
		self._attr_native_unit_of_measurement = PERCENTAGE
		self._attr_device_class = SensorDeviceClass.POWER_FACTOR
		self._attr_suggested_display_precision = 1

	@property
	def native_value(self) -> float | None:
		"""Return the confidence value."""
		if self.coordinator.data is None:
			return None
		return round(self.coordinator.data.confidence * 100, 1)

	@property
	def icon(self) -> str:
		"""Return the icon for this sensor."""
		if self.coordinator.data is None:
			return "mdi:percent"
		
		confidence = self.coordinator.data.confidence
		if confidence >= 0.8:
			return "mdi:check-circle"
		elif confidence >= 0.6:
			return "mdi:alert-circle"
		else:
			return "mdi:close-circle"


class OpenGateSimilaritySensor(OpenGateBaseSensor):
	"""Sensor for pattern similarity score."""

	def __init__(
		self,
		coordinator: OpenGateDataUpdateCoordinator,
		config_entry: ConfigEntry,
	) -> None:
		"""Initialize the similarity sensor."""
		super().__init__(coordinator, config_entry, "similarity", "Gate Pattern Similarity")
		self._attr_native_unit_of_measurement = PERCENTAGE
		self._attr_device_class = SensorDeviceClass.POWER_FACTOR
		self._attr_suggested_display_precision = 1

	@property
	def native_value(self) -> float | None:
		"""Return the similarity value."""
		if self.coordinator.data is None:
			return None
		return round(self.coordinator.data.similarity_score * 100, 1)

	@property
	def icon(self) -> str:
		"""Return the icon for this sensor."""
		return "mdi:compare"

	@property
	def extra_state_attributes(self) -> dict[str, Any]:
		"""Return additional state attributes."""
		if self.coordinator.data is None:
			return {}
		
		data = self.coordinator.data
		return {
			"edge_density_ratio": round(data.edge_density_ratio, 3),
			"texture_variance_ratio": round(data.texture_variance_ratio, 3),
			"detection_timestamp": data.timestamp,
		} 