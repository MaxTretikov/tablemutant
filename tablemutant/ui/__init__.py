#!/usr/bin/env python3
"""
TableMutant GUI - Main Toga application that coordinates different windows
"""

import asyncio
import os
import sys
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import polars as pl

from tablemutant.core import TableMutant
from tablemutant.core.settings_manager import SettingsManager

# Get logger for this module
logger = logging.getLogger('tablemutant.ui')

from tablemutant.ui.windows import (
    InitializationWindow,
    FileSelectionWindow,
    SourceSelectionWindow,
    OutputDefinitionWindow,
    ExampleWindow,
    ProcessingWindow,
    PreviewResultsWindow,
    SettingsWindow,
    EmbeddingProgressWindow
)

__all__ = [
    'TableMutantGUI',
]

class TableMutantGUI(toga.App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize settings manager first
        self.settings_manager = SettingsManager()
        
        logger.debug("TableMutantGUI.__init__")
        self.tm = TableMutant(settings_manager=self.settings_manager)
        self.current_df = None
        self.original_df = None  # Store original DataFrame
        self.header_rows = 1
        self.detected_headers = 1
        self.confidence = 0.0
        self.selected_columns = []
        self.rag_documents = []  # List of RAG document paths
        self.column_definitions = []
        self.preview_rows = 5  # Number of rows to preview (1-10)
        self.model_path = None
        self.table_path = None
        self.server_was_running = False
        
        # Window instances
        self.init_window = None
        self.file_window = None
        self.source_window = None
        self.output_window = None
        self.processing_window = None
        self.preview_window = None
        self.settings_window = None
        
    def startup(self):
        """Construct and show the Toga application."""
        logger.debug("TableMutantGUI.startup")
        self.main_window = toga.MainWindow(title=self.formal_name)
        
        # Create app menu with settings option
        self.create_app_menu()
        
        # Start with initialization window
        self.show_initialization_window()
        
    def create_app_menu(self):
        """Create application menu with settings option."""
        settings_command = toga.Command(
            self.show_settings,
            text='Settings...',
            tooltip='Application Settings',
            shortcut=toga.Key.MOD_1 + 'comma'  # Cmd+, on macOS, Ctrl+, on others
        )
        
        self.main_window.app.commands.add(settings_command)
        
    def show_settings(self, widget):
        """Show the settings window."""
        if not self.settings_window:
            self.settings_window = SettingsWindow(self)
        self.settings_window.show()
        
    def show_initialization_window(self):
        """Show initialization window with loading bar."""
        logger.debug("TableMutantGUI.show_initialization_window")
        self.init_window = InitializationWindow(self)
        self.main_window.content = self.init_window.create_content()
        self.main_window.show()
        
        # Run initialization in background
        asyncio.create_task(self.init_window.initialize_app())
    
    def show_file_selection_window(self):
        """Show window for selecting and previewing table file."""
        logger.debug("TableMutantGUI.show_file_selection_window")
        self.file_window = FileSelectionWindow(self)
        self.main_window.content = self.file_window.create_content()
    
    def show_source_selection_window(self):
        """Show window for selecting sources (columns and documents)."""
        self.source_window = SourceSelectionWindow(self)
        self.main_window.content = self.source_window.create_content()
    
    def show_output_definition_window(self):
        """Show window for defining output columns."""
        self.output_window = OutputDefinitionWindow(self)
        self.main_window.content = self.output_window.create_content()
    
    def show_processing_window(self, definitions, preview_only=True):
        """Show processing window with progress bar."""
        self.processing_window = ProcessingWindow(self)
        self.main_window.content = self.processing_window.create_content(preview_only)
        
        # Start processing in background
        asyncio.create_task(
            self.processing_window.process_generation(
                definitions=definitions,
                preview_only=preview_only
            )
        )
    
    def show_preview_results_window(self, df_preview, new_columns, definitions):
        """Show preview of results."""
        self.preview_window = PreviewResultsWindow(self)
        self.main_window.content = self.preview_window.create_content(df_preview, new_columns, definitions)
    
    # Utility methods for DataFrame management
    def detect_header_rows(self, df: pl.DataFrame) -> Tuple[int, float]:
        """Detect header rows using the HeaderProcessor."""
        return self.tm.table_processor.header_processor.detect_header_rows(df)
    
    def create_working_dataframe(self):
        """Create a working DataFrame using the HeaderProcessor."""
        if self.original_df is None or self.original_df.is_empty():
            return
        
        self.current_df = self.tm.table_processor.header_processor.create_working_dataframe(
            self.original_df, 
            self.header_rows
        )