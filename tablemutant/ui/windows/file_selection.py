#!/usr/bin/env python3
"""
FileSelectionWindow - Handles file selection and table preview
"""

import os

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class FileSelectionWindow:
    def __init__(self, app):
        self.app = app
        self.file_box = None
        self.file_label = None
        self.preview_container = None
        self.preview_table = None
        self.header_selector = None
        self.confidence_label = None
        self.model_input = None
        
    def _truncate_model_name(self, model_name, max_length=50):
        """Truncate model name from start if too long, keeping the end part."""
        if not model_name or len(model_name) <= max_length:
            return model_name
        return "..." + model_name[-(max_length-3):]
        
    def create_content(self):
        """Create and return the file selection window content."""
        self.file_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        # Title
        title = toga.Label(
            "Load Table File",
            style=Pack(padding=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # File selection section
        file_section = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
        self.file_label = toga.Label(
            "No file selected",
            style=Pack(padding=(0, 10, 0, 0), width=400)
        )
        load_button = toga.Button(
            "Browse...",
            on_press=self.load_table_file,
            style=Pack(padding=5)
        )
        file_section.add(load_button)
        file_section.add(self.file_label)
        
        # Preview section
        self.preview_label = toga.Label(
            "Table Preview (First 5 Rows):",
            style=Pack(padding=(10, 0, 5, 0), font_weight='bold')
        )
        
        # Create a ScrollContainer for horizontal scrolling
        self.preview_scroll_container = toga.ScrollContainer(
            style=Pack(
                flex=1,
                padding=5,
                height=250  # Fixed height for the preview area
            ),
            horizontal=True,  # Enable horizontal scrolling
            vertical=True     # Enable vertical scrolling as well
        )
        
        # Try to force scrollbar visibility using platform-specific approaches
        try:
            import sys
            if sys.platform == 'darwin':  # macOS
                # For macOS, try to access the native NSScrollView
                if hasattr(self.preview_scroll_container, '_impl') and hasattr(self.preview_scroll_container._impl, 'native'):
                    native_scroll = self.preview_scroll_container._impl.native
                    # Show scrollbars always
                    native_scroll.setHasHorizontalScroller_(True)
                    native_scroll.setHorizontalScrollerKnobStyle_(2)  # NSScrollerKnobStyleDefault
                    native_scroll.setScrollerStyle_(0)  # NSScrollerStyleLegacy (always visible)
            elif sys.platform == 'win32':  # Windows
                # For Windows, scrollbars are typically always visible by default
                pass
            elif sys.platform.startswith('linux'):  # Linux/GTK
                # For GTK, try to access the native widget
                if hasattr(self.preview_scroll_container, '_impl') and hasattr(self.preview_scroll_container._impl, 'native'):
                    native_scroll = self.preview_scroll_container._impl.native
                    # GTK-specific scrollbar policy
                    if hasattr(native_scroll, 'set_policy'):
                        from gi.repository import Gtk
                        native_scroll.set_policy(Gtk.PolicyType.ALWAYS, Gtk.PolicyType.AUTOMATIC)
        except Exception as e:
            # If platform-specific styling fails, continue with default behavior
            print(f"Could not apply platform-specific scrollbar styling: {e}")
        
        # Container for the table preview (inside the scroll container)
        self.preview_container = toga.Box(style=Pack())
        self.preview_scroll_container.content = self.preview_container
        self.preview_table = None
        
        # Header detection section (initially hidden)
        self.header_section = toga.Box(style=Pack(direction=ROW, padding=(20, 0, 10, 0)))
        header_label = toga.Label(
            "Number of header rows:",
            style=Pack(padding=(0, 10, 0, 0))
        )
        self.header_selector = toga.NumberInput(
            min=1,
            max=5,
            value=1,
            on_change=self.update_header_preview,
            style=Pack(width=60)
        )
        self.confidence_label = toga.Label(
            "",
            style=Pack(padding=(0, 10, 0, 0))
        )
        
        self.header_section.add(header_label)
        self.header_section.add(self.header_selector)
        self.header_section.add(self.confidence_label)
        
        # Model status section
        self.model_section = toga.Box(style=Pack(direction=ROW, padding=(10, 0, 20, 0)))
        model_label = toga.Label(
            "Model:",
            style=Pack(padding=(0, 10, 0, 0))
        )
        self.model_status = toga.Label(
            self._truncate_model_name(self.app.settings_manager.get('model', 'Not configured')),
            style=Pack(flex=1, padding=(0, 10, 0, 0))
        )
        settings_button = toga.Button(
            "Settings...",
            on_press=self.open_settings,
            style=Pack(padding=(0, 0, 0, 10))
        )
        self.model_section.add(model_label)
        self.model_section.add(self.model_status)
        self.model_section.add(settings_button)
        
        # Confirm button (initially hidden)
        self.confirm_button = toga.Button(
            "Next: Select Sources",
            on_press=self.confirm_file_selection,
            style=Pack(padding=(20, 0, 0, 0))
        )
        
        # Add all components
        self.file_box.add(title)
        self.file_box.add(file_section)
        # Preview components will be added after file is loaded
        
        return self.file_box
    
    async def load_table_file(self, widget):
        """Load and preview table file."""
        try:
            file_dialog = toga.OpenFileDialog(
                title="Select Table File",
                file_types=["csv", "parquet", "json"]
            )
            file_path = await self.app.main_window.dialog(file_dialog)
            
            if file_path:
                self.app.table_path = str(file_path)
                self.file_label.text = os.path.basename(self.app.table_path)
                
                # Load table and store original
                self.app.original_df = self.app.tm.table_processor.load_table(self.app.table_path)
                
                # Detect headers
                self.app.detected_headers, self.app.confidence = self.app.detect_header_rows(self.app.original_df)
                self.header_selector.value = self.app.detected_headers
                self.app.header_rows = self.app.detected_headers
                self.confidence_label.text = f"(Confidence: {self.app.confidence:.1%})"
                
                # Create working DataFrame
                self.app.create_working_dataframe()
                
                # Update preview
                self.update_preview()
                
                # Show the preview and additional controls now that a file is loaded
                if self.preview_label not in self.file_box.children:
                    self.file_box.add(self.preview_label)
                    self.file_box.add(self.preview_scroll_container)  # Add scroll container instead of preview_container
                    self.file_box.add(self.header_section)
                    self.file_box.add(self.model_section)
                    self.file_box.add(self.confirm_button)
                    
                    # Try to force scrollbar visibility after adding to the UI
                    self.app.main_window.app.add_background_task(
                        lambda app: self.force_scrollbar_visibility()
                    )
                
        except Exception as e:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Error",
                    message=f"Failed to load file: {str(e)}"
                )
            )

    def update_header_preview(self, widget):
        """Update preview when header rows change."""
        self.app.header_rows = int(self.header_selector.value)
        self.confidence_label.text = ""  # Clear confidence when user changes
        
        # Recreate working DataFrame with new header configuration
        self.app.create_working_dataframe()
        
        # DEBUG: Print current DataFrame info
        print("\n" + "="*60)
        print(f"DEBUG: Header selector changed to {self.app.header_rows} rows")
        if self.app.current_df is not None:
            print(f"Current DataFrame shape: {self.app.current_df.shape}")
            print(f"Column names ({len(self.app.current_df.columns)}):")
            for i, col in enumerate(self.app.current_df.columns):
                print(f"  [{i}] {col}")
            
            # Show first few rows of data
            if len(self.app.current_df) > 0:
                print(f"\nFirst 3 rows of data:")
                try:
                    preview_df = self.app.current_df.head(3)
                    print(preview_df)
                except Exception as e:
                    print(f"Error displaying preview: {e}")
            else:
                print("No data rows available")
        else:
            print("Current DataFrame is None")
        print("="*60)
        
        self.update_preview()
    
    def update_preview(self):
        """Update the table preview."""
        if self.app.current_df is None or self.app.current_df.is_empty():
            return
        
        # Remove old table if exists
        if self.preview_table:
            self.preview_container.remove(self.preview_table)
        
        # Show preview of current processed data
        preview_rows = 5
        total_preview_rows = min(preview_rows, len(self.app.current_df))
        
        # Use current DataFrame column names (already have [A], [B] format and combined headers)
        col_headers = list(self.app.current_df.columns)
        
        # Prepare data from current DataFrame
        data = []
        for i in range(total_preview_rows):
            row = []
            for col in self.app.current_df.columns:
                try:
                    val = self.app.current_df[col][i]
                    str_val = str(val) if val is not None else ""
                    if len(str_val) > 50:
                        str_val = str_val[:47] + "..."
                    row.append(str_val)
                except Exception:
                    row.append("")
            data.append(row)
        
        # Calculate total width needed for all columns
        # Each column gets a fixed width
        column_width = 150  # Fixed width for each column in pixels
        total_table_width = column_width * len(col_headers)
        
        # Create new table with fixed width
        self.preview_table = toga.Table(
            headings=col_headers,
            data=data,
            style=Pack(
                width=total_table_width,  # Set explicit width for the table
                flex=1  # Let the table expand vertically to fill available space
            )
        )
        
        # Update container width to match table width
        self.preview_container.style.width = total_table_width
        
        # Add to container
        self.preview_container.add(self.preview_table)
    
    def force_scrollbar_visibility(self):
        """Attempt to force scrollbar visibility after widget creation."""
        try:
            import sys
            if sys.platform == 'darwin':  # macOS
                # For macOS, access the native NSScrollView
                if hasattr(self.preview_scroll_container, '_impl') and hasattr(self.preview_scroll_container._impl, 'native'):
                    native_scroll = self.preview_scroll_container._impl.native
                    # Import required macOS constants
                    from rubicon.objc import ObjCClass
                    NSScroller = ObjCClass('NSScroller')
                    
                    # Show scrollbars always
                    native_scroll.setHasHorizontalScroller_(True)
                    native_scroll.setHasVerticalScroller_(True)
                    # Use legacy style (always visible)
                    native_scroll.setScrollerStyle_(0)  # NSScrollerStyleLegacy
                    # Force scrollers to be visible
                    if native_scroll.horizontalScroller:
                        native_scroll.horizontalScroller.setHidden_(False)
            elif sys.platform == 'win32':  # Windows
                # Windows typically shows scrollbars by default
                # But we can try to ensure they're visible
                if hasattr(self.preview_scroll_container, '_impl') and hasattr(self.preview_scroll_container._impl, 'native'):
                    # Windows-specific code would go here if needed
                    pass
            elif sys.platform.startswith('linux'):  # Linux/GTK
                # For GTK, set scrollbar policy
                if hasattr(self.preview_scroll_container, '_impl') and hasattr(self.preview_scroll_container._impl, 'native'):
                    native_scroll = self.preview_scroll_container._impl.native
                    # GTK-specific scrollbar policy
                    try:
                        from gi.repository import Gtk
                        native_scroll.set_policy(Gtk.PolicyType.ALWAYS, Gtk.PolicyType.AUTOMATIC)
                    except ImportError:
                        pass
        except Exception as e:
            print(f"Could not force scrollbar visibility: {e}")
    
    async def open_settings(self, widget):
        """Open settings dialog."""
        if not self.app.settings_window:
            from tablemutant.ui.windows.settings import SettingsWindow
            self.app.settings_window = SettingsWindow(self.app)
        self.app.settings_window.show()
    
    def refresh_model_status(self):
        """Refresh the model status display."""
        if hasattr(self, 'model_status'):
            self.model_status.text = self._truncate_model_name(self.app.settings_manager.get('model', 'Not configured'))
    
    async def confirm_file_selection(self, widget):
        """Confirm file selection and move to column selection."""
        if self.app.current_df is None or self.app.current_df.is_empty():
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Error",
                    message="Please select a table file first."
                )
            )
            return
        
        # Get model from settings
        model_path = self.app.settings_manager.get('model')
        if not model_path:
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Error", 
                    message="Please configure a model in Settings first."
                )
            )
            return
        
        self.app.model_path = model_path
        self.app.show_source_selection_window()