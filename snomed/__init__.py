"""
Module that hosts the API to access the SNOMED hierarchy.

Currently, only the Snowstorm API is supported, but work on adding FHIR/Snowstorm-Lite is underway.
"""

from .snowstorm import SnowstormAPI


__all__ = ["SnowstormAPI"]
