"""Config flow for OpenGate Detection integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
import homeassistant.helpers.config_validation as cv

from .const import (
	DOMAIN,
	CONF_CAMERA_URL,
	CONF_DETECTION_INTERVAL,
	CONF_CONFIDENCE_THRESHOLD,
	CONF_PATTERN_SIMILARITY,
	DEFAULT_DETECTION_INTERVAL,
	DEFAULT_CONFIDENCE_THRESHOLD,
	DEFAULT_PATTERN_SIMILARITY,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
	{
		vol.Required(CONF_CAMERA_URL): cv.string,
		vol.Optional(
			CONF_DETECTION_INTERVAL, 
			default=DEFAULT_DETECTION_INTERVAL
		): cv.positive_float,
		vol.Optional(
			CONF_CONFIDENCE_THRESHOLD, 
			default=DEFAULT_CONFIDENCE_THRESHOLD
		): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
		vol.Optional(
			CONF_PATTERN_SIMILARITY, 
			default=DEFAULT_PATTERN_SIMILARITY
		): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
	}
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
	"""Validate the user input allows us to connect.
	
	Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
	"""
	# Validate camera URL format
	camera_url = data[CONF_CAMERA_URL]
	if not (camera_url.startswith('rtsp://') or camera_url.startswith('http://') or camera_url.startswith('https://') or camera_url.isdigit()):
		raise InvalidCameraURL
	
	# Return info that you want to store in the config entry.
	return {"title": f"OpenGate Detection ({camera_url})"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
	"""Handle a config flow for OpenGate Detection."""

	VERSION = 1

	async def async_step_user(
		self, user_input: dict[str, Any] | None = None
	) -> FlowResult:
		"""Handle the initial step."""
		errors: dict[str, str] = {}
		
		if user_input is not None:
			try:
				info = await validate_input(self.hass, user_input)
			except InvalidCameraURL:
				errors["base"] = "invalid_camera_url"
			except Exception:  # pylint: disable=broad-except
				_LOGGER.exception("Unexpected exception")
				errors["base"] = "unknown"
			else:
				return self.async_create_entry(title=info["title"], data=user_input)

		return self.async_show_form(
			step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
		)


class CannotConnect(HomeAssistantError):
	"""Error to indicate we cannot connect."""


class InvalidCameraURL(HomeAssistantError):
	"""Error to indicate there is invalid camera URL.""" 