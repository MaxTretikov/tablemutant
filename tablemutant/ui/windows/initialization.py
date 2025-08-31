#!/usr/bin/env python3
"""
InitializationWindow - Handles application startup and initialization
"""

import asyncio
import functools
import os
import logging
import threading
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
        self._progress_state = {"bytes": 0, "total": 1}
        self._progress_lock = threading.Lock()
        self._progress_done = None
        
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
    
    def _tick_progress(self):
        # Runs on the Toga event loop; safe to touch widgets
        with self._progress_lock:
            bytes_done = self._progress_state["bytes"]
            total = self._progress_state["total"]
            done = self._progress_done

        if done is None:
            self._progress_done = False
            self.init_log.value += "✗\nDownloading llamafile... "

        pct = 5 + int(65 * (bytes_done / (total or 1)))
        pct = max(5, min(70, pct))
        mb = bytes_done / 1_000_000
        tmb = total / 1_000_000

        self.init_progress.value = pct
        self.status_label.text = f"Downloading llamafile {mb:.1f}/{tmb:.1f} MB"

        # Re‑arm if the download is still running
        if not done:
            asyncio.get_running_loop().call_later(1/15, self._tick_progress)
    
    async def initialize_app(self):
        logger.debug("InitializationWindow.initialize_app")

        try:
            self.status_label.text = "Setting up llamafile..."
            self.init_progress.value = 5
            await asyncio.sleep(0.05)

            self.init_log.value += "Checking for llamafile... "

            # Kick off periodic UI polling on the main loop
            asyncio.get_running_loop().call_soon(self._tick_progress)

            def _progress(bytes_done: int, total: int):
                with self._progress_lock:
                    self._progress_state["bytes"] = bytes_done
                    self._progress_state["total"] = total or 1

            get_llama = functools.partial(self.app.tm.model_manager.download_llamafile, progress_cb=_progress)

            # Offload the blocking download to a worker thread
            self.app.tm.model_manager.llamafile_path = await asyncio.to_thread(get_llama)

            # Stop the UI poller
            with self._progress_lock:
                self._progress_done = True
                self.init_log.value += "✓"

            self.init_progress.value = 80
            self.status_label.text = "Checking dependencies..."
            self.init_log.value += "\nChecking required packages:\n"

            required_packages = ['polars', 'dspy', 'tqdm', 'chardet']
            for package in required_packages:
                try:
                    __import__(package)
                    self.init_log.value += f"✓ {package} installed\n"
                except ImportError:
                    self.init_log.value += f"✗ {package} NOT installed (install with pip)\n"

            self.init_progress.value = 95
            self.status_label.text = "Finalizing..."
            await asyncio.sleep(0.2)

            self.status_label.text = "Initialization complete!"
            self.init_progress.value = 100
            self.init_log.value += "\nReady to load table files!\n"
            await asyncio.sleep(0.5)

        except Exception as e:
            self.status_label.text = "Initialization failed!"
            self.init_log.value += f"\nERROR: {str(e)}\n"
            logger.exception("Initialization failed")
            await asyncio.sleep(2.0)
        finally:
            logger.debug("InitializationWindow showing file selection window")
            # Prevent duplicate window if initialize_app was triggered twice
            if not getattr(self.app, "_file_selector_shown", False):
                self.app._file_selector_shown = True
                self.app.show_file_selection_window()