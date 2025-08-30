#!/usr/bin/env python3
"""
PreviewResultsWindow - Shows preview of generated results with detail view
"""

import asyncio
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class PreviewResultsWindow:
    def __init__(self, app):
        self.app = app
        self.preview_box = None
        self.detail_container = None
        self.current_headers = []
        self.current_data = []
        
    def create_content(self, df_preview, new_columns, definitions):
        """Create and return the preview results window content."""
        self.preview_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        
        # Title
        title = toga.Label(
            "Generation Preview",
            style=Pack(margin=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Create main content area with horizontal split
        main_content = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Left side - Table container (taking more space)
        table_section = toga.Box(style=Pack(direction=COLUMN, flex=3))
        
        # Prepare headers
        self.current_headers = []
        # Add source column headers
        for i in self.app.selected_columns:
            self.current_headers.append(f"[Source] {self.app.current_df.columns[i]}")
        # Add generated column headers
        for col_name in new_columns.keys():
            self.current_headers.append(f"[Generated] {col_name}")
        
        # Prepare data
        self.current_data = []
        for i in range(len(df_preview)):
            row = []
            # Add source values
            for col_idx in self.app.selected_columns:
                try:
                    val = df_preview[self.app.current_df.columns[col_idx]][i]
                    row.append(str(val) if val else "")
                except Exception:
                    row.append("")
            # Add generated values
            for values in new_columns.values():
                row.append(values[i] if i < len(values) else "")
            self.current_data.append(row)
        
        # Create table with flex to fill available space
        preview_table = toga.Table(
            headings=self.current_headers,
            data=self.current_data,
            style=Pack(
                flex=1
            ),
            on_select=self.on_row_select
        )
        
        # Add table to left section
        table_label = toga.Label(
            "Select a row to view details",
            style=Pack(margin=(0, 0, 5, 0), font_size=12)
        )
        table_section.add(table_label)
        table_section.add(preview_table)
        
        # Right side - Detail view (taking less space)
        detail_section = toga.Box(style=Pack(direction=COLUMN, flex=2, margin=(0, 0, 0, 10)))
        
        detail_label = toga.Label(
            "Row Details",
            style=Pack(margin=(0, 0, 10, 0), font_size=14, font_weight='bold')
        )
        
        # Create scrollable detail container
        detail_scroll = toga.ScrollContainer(
            style=Pack(
                flex=1,
                margin=5
            ),
            vertical=True
        )
        
        # Initial detail container (empty)
        self.detail_container = toga.Box(style=Pack(direction=COLUMN, margin=10))
        self.detail_container.add(toga.Label(
            "No row selected",
            style=Pack(margin=5, font_size=12)
        ))
        detail_scroll.content = self.detail_container
        
        detail_section.add(detail_label)
        detail_section.add(detail_scroll)
        
        # Add both sections to main content
        main_content.add(table_section)
        main_content.add(detail_section)
        
        # Navigation buttons
        nav_section = toga.Box(style=Pack(direction=ROW, margin=(20, 0, 0, 0)))
        back_button = toga.Button(
            "Back to Edit",
            on_press=self.back_to_output_definition,
            style=Pack(margin=5)
        )
        process_button = toga.Button(
            "Process Entire File",
            on_press=lambda w: self.handle_process_entire_file(definitions),
            style=Pack(margin=5)
        )
        nav_section.add(back_button)
        nav_section.add(toga.Box(style=Pack(flex=1)))
        nav_section.add(process_button)
        
        # Add all components
        self.preview_box.add(title)
        self.preview_box.add(main_content)
        self.preview_box.add(nav_section)
        
        # Store references for the selection handler
        self.preview_table = preview_table
        
        return self.preview_box
    
    async def on_row_select(self, widget):
        """Handle row selection and update detail view."""
        if widget.selection is None:
            return
        
        # Clear current detail container
        self.detail_container.clear()
        
        # Get selected row index - find the index of the selected row
        selected_row_index = -1
        for i, row in enumerate(self.current_data):
            if widget.selection == widget.data[i]:
                selected_row_index = i
                break
        
        if selected_row_index == -1:
            return
        
        # Add row number
        row_label = toga.Label(
            f"Row {selected_row_index + 1} of {len(self.current_data)}",
            style=Pack(margin=(0, 0, 10, 0), font_size=12, font_weight='bold')
        )
        self.detail_container.add(row_label)
        
        # Add each field with its full content
        for i, header in enumerate(self.current_headers):
            # Create a box for each field
            field_box = toga.Box(style=Pack(direction=COLUMN, margin=(0, 0, 15, 0)))
            
            # Header label
            header_label = toga.Label(
                header,
                style=Pack(
                    margin=(0, 0, 5, 0),
                    font_size=11,
                    font_weight='bold'
                )
            )
            field_box.add(header_label)
            
            # Content - use MultilineTextInput for better display of long text
            content = self.current_data[selected_row_index][i] if i < len(self.current_data[selected_row_index]) else ""
            
            if content and len(content) > 50:  # For longer content
                content_display = toga.MultilineTextInput(
                    value=content,
                    readonly=True,
                    style=Pack(
                        margin=5,
                        height=100  # Adjust height based on content
                    )
                )
            else:  # For shorter content
                content_display = toga.Label(
                    content if content else "(empty)",
                    style=Pack(
                        margin=5
                    )
                )
            
            field_box.add(content_display)
            
            # Add separator line (using a thin box)
            separator = toga.Box(
                style=Pack(
                    height=1,
                    margin=(5, 0, 0, 0)
                )
            )
            field_box.add(separator)
            
            self.detail_container.add(field_box)
        
        # Refresh the detail container
        self.detail_container.refresh()
    
    async def back_to_output_definition(self, widget):
        """Go back to output definition window."""
        self.app.show_output_definition_window()
    
    def handle_process_entire_file(self, definitions):
        """Handle the process entire file button press."""
        asyncio.create_task(self.process_entire_file(definitions))
    
    async def process_entire_file(self, definitions):
        """Process the entire file."""
        self.app.show_processing_window(definitions, preview_only=False)