#!/usr/bin/env python3
"""
InitializationWindow - Handles application startup and initialization
"""

import asyncio
import os
import logging
from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN

# Get logger for this module
logger = logging.getLogger('tablemutant.ui.windows.initialization')


class InitializationWindow:
    def __init__(self, app):
        self.app = app
        self.init_box = None
        self.status_label = None
        self.init_progress = None
        self.init_log = None
        
    def create_content(self):
        """Create and return the initialization window content."""
        self.init_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        
        # Title
        title = toga.Label(
            "Initializing TableMutant...",
            style=Pack(margin=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Status label
        self.status_label = toga.Label(
            "Starting up...",
            style=Pack(margin=(0, 0, 10, 0))
        )
        
        # Progress bar
        self.init_progress = toga.ProgressBar(
            max=100,
            style=Pack(width=400, margin=(0, 0, 10, 0))
        )
        
        # Log text area
        self.init_log = toga.MultilineTextInput(
            readonly=True,
            style=Pack(width=600, height=300, margin=5)
        )
        
        self.init_box.add(title)
        self.init_box.add(self.status_label)
        self.init_box.add(self.init_progress)
        self.init_box.add(self.init_log)
        
        return self.init_box
    
    async def initialize_app(self):
        """Initialize the application components."""
        logger.debug("InitializationWindow.initialize_app")
        try:
            # Check if we need to download llamafile
            self.status_label.text = "Checking for llamafile..."
            self.init_progress.value = 10
            await asyncio.sleep(0.1)  # Allow UI to update
            
            # Download llamafile if needed
            llamafile_dir = Path.home() / '.tablemutant' / 'bin'
            existing_llamafiles = list(llamafile_dir.glob('llamafile-*')) if llamafile_dir.exists() else []
            
            if not existing_llamafiles:
                self.status_label.text = "Downloading llamafile..."
                self.init_progress.value = 20
                self.init_log.value += "Llamafile not found locally, downloading...\n"
                
                # Run download in thread to avoid blocking
                loop = asyncio.get_event_loop()
                try:
                    logger.debug("InitializationWindow downloading llamafile")
                    self.app.tm.model_manager.llamafile_path = await loop.run_in_executor(
                        None, self.app.tm.model_manager.download_llamafile
                    )
                    self.init_log.value += f"Downloaded llamafile to: {self.app.tm.model_manager.llamafile_path}\n"
                    logger.debug("InitializationWindow llamafile downloaded to: %s", self.app.tm.model_manager.llamafile_path)
                except Exception as e:
                    self.init_log.value += f"Error downloading llamafile: {e}\n"
                    logger.debug("InitializationWindow error downloading llamafile: %s", e)
                    raise
            else:
                self.app.tm.model_manager.llamafile_path = str(max(existing_llamafiles, key=os.path.getctime))
                self.init_log.value += f"Found existing llamafile: {self.app.tm.model_manager.llamafile_path}\n"
            
            self.init_progress.value = 50
            
            # Check Python dependencies
            self.status_label.text = "Checking dependencies..."
            self.init_log.value += "\nChecking required packages:\n"
            
            required_packages = ['polars', 'dspy', 'tqdm', 'chardet']
            for package in required_packages:
                try:
                    __import__(package)
                    self.init_log.value += f"✓ {package} installed\n"
                except ImportError:
                    self.init_log.value += f"✗ {package} NOT installed - please install with pip\n"
            
            self.init_progress.value = 80
            
            # Setup complete
            self.status_label.text = "Initialization complete!"
            self.init_progress.value = 100
            self.init_log.value += "\nReady to load table files!\n"
            
            # Wait a moment before transitioning
            await asyncio.sleep(1)
            
        except Exception as e:
            self.status_label.text = "Initialization failed!"
            self.init_log.value += f"\nERROR: {str(e)}\n"
            await asyncio.sleep(3)
        
        # Show file selection window
        logger.debug("InitializationWindow showing file selection window")
        self.app.show_file_selection_window()