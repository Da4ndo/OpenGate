"""
OpenGate Texture-Based Detection Integration for Home Assistant.

This integration provides gate state detection using texture pattern analysis.
"""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]

CONFIG_SCHEMA = vol.Schema(
	{
		DOMAIN: vol.Schema(
			{
				vol.Optional("camera_url"): cv.string,
				vol.Optional("detection_interval", default=1.0): cv.positive_float,
				vol.Optional("confidence_threshold", default=0.8): cv.small_float,
				vol.Optional("pattern_similarity", default=0.85): cv.small_float,
			}
		)
	},
	extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
	"""Set up the OpenGate Detection integration."""
	_LOGGER.debug("Setting up OpenGate Detection integration")
	
	hass.data.setdefault(DOMAIN, {})
	
	# Store configuration
	if DOMAIN in config:
		hass.data[DOMAIN]["config"] = config[DOMAIN]
	
	return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
	"""Set up OpenGate Detection from a config entry."""
	_LOGGER.debug("Setting up OpenGate Detection config entry: %s", entry.entry_id)
	
	# Store entry data
	hass.data.setdefault(DOMAIN, {})
	hass.data[DOMAIN][entry.entry_id] = entry.data
	
	# Forward the setup to the sensor platform
	await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
	
	return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
	"""Unload a config entry."""
	_LOGGER.debug("Unloading OpenGate Detection config entry: %s", entry.entry_id)
	
	# Unload platforms
	unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
	
	if unload_ok:
		hass.data[DOMAIN].pop(entry.entry_id)
	
	return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
	"""Reload config entry."""
	await async_unload_entry(hass, entry)
	await async_setup_entry(hass, entry) 