"""Test the OpenGate Detection integration."""
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from custom_components.opengate_detection import async_setup, async_setup_entry
from custom_components.opengate_detection.const import DOMAIN


async def test_async_setup(hass: HomeAssistant):
    """Test async_setup."""
    config = {DOMAIN: {}}
    assert await async_setup(hass, config) is True
    assert DOMAIN in hass.data


async def test_async_setup_entry(hass: HomeAssistant):
    """Test async_setup_entry."""
    entry_data = {
        "camera_url": "0",
        "detection_interval": 1.0,
        "confidence_threshold": 0.8,
        "pattern_similarity": 0.85
    }
    
    config_entry = ConfigEntry(
        version=1,
        domain=DOMAIN,
        title="Test OpenGate",
        data=entry_data,
        source="test"
    )
    
    # Setup should succeed even without camera connection for testing
    assert await async_setup_entry(hass, config_entry) is True
    assert config_entry.entry_id in hass.data[DOMAIN] 