"""Convenience re-export for project config."""

from .config_setup import Config, config as config_instance

config = config_instance

__all__ = ["Config", "config"]
