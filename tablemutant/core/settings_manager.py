#!/usr/bin/env python3
"""
SettingsManager - Handles application settings persistence
"""

import json
import os
import platform
from pathlib import Path
from typing import Dict, Any


class SettingsManager:
    def __init__(self):
        self.settings_file = "settings.json"
        self.settings = self.load_settings()
        
    def get_models_dir(self) -> Path:
        """Get the appropriate models directory based on the platform."""
        system = platform.system()
        
        if system == "Linux":
            models_dir = Path.home() / '.tablemutant' / 'models'
        elif system == "Darwin":  # macOS
            models_dir = Path.home() / 'Library' / 'Application Support' / 'TableMutant' / 'models'
        elif system == "Windows":
            models_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'TableMutant' / 'models'
        else:
            # Fallback to home directory
            models_dir = Path.home() / '.tablemutant' / 'models'
        
        # Create directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Return default settings."""
        return {
            "model": "unsloth/medgemma-27b-text-it-GGUF",
            "models_directory": str(self.get_models_dir()),
            # New settings: default to local server endpoint
            "server_host": "http://localhost:8000",
            "auth_token": "",
            # Keep legacy key only for migration compatibility (not used by consumers anymore)
            "llamafile_port": 8000,
            "temperature": 0.7,
            "max_tokens": 2048
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or create default settings."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_settings = self.get_default_settings()
                    default_settings.update(loaded_settings)

                    # Migration: if server_host missing but legacy llamafile_port present, derive host
                    if "server_host" not in default_settings:
                        port = default_settings.get("llamafile_port", 8000)
                        try:
                            port = int(port)
                        except Exception:
                            port = 8000
                        default_settings["server_host"] = f"http://localhost:{port}"
                    # Ensure auth_token exists
                    if "auth_token" not in default_settings:
                        default_settings["auth_token"] = ""

                    # Persist any migrated keys
                    self.save_settings(default_settings)
                    return default_settings
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.get_default_settings()
        else:
            # Create default settings file
            default_settings = self.get_default_settings()
            self.save_settings(default_settings)
            return default_settings
    
    def save_settings(self, settings: Dict[str, Any]):
        """Save settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            self.settings = settings
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get(self, key: str, default=None):
        """Get a setting value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value."""
        self.settings[key] = value
        self.save_settings(self.settings)
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple settings at once."""
        self.settings.update(updates)
        self.save_settings(self.settings)