#!/usr/bin/env python3
"""
OutputDefinitionWindow - Handles defining output columns
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .example import ExampleWindow
import asyncio


class OutputDefinitionWindow:
    def __init__(self, app):
        self.app = app
        self.output_box = None
        self.definitions_container = None
        self.definitions_box = None
        self.example_window = None
        
    def create_content(self):
        """Create and return the output definition window content."""
        self.output_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        
        # Title
        title = toga.Label(
            "Define Output Columns",
            style=Pack(margin=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Scrollable container for column definitions
        self.definitions_container = toga.ScrollContainer(
            style=Pack(flex=1)
        )
        self.definitions_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        
        # Load existing definitions or add initial one if none exist
        self.load_existing_definitions()
        
        # Add button
        add_button = toga.Button(
            "Add Another Column",
            on_press=self.add_column_definition_handler,
            style=Pack(margin=(10, 0, 10, 0))
        )
        
        # Preview rows section
        preview_section = toga.Box(style=Pack(direction=COLUMN, margin=(10, 0, 20, 0)))
        
        # Calculate actual available rows based on non-empty data
        max_preview_rows = self.get_max_preview_rows()
        
        # Ensure current preview_rows doesn't exceed available rows
        if self.app.preview_rows > max_preview_rows:
            self.app.preview_rows = max_preview_rows
        
        preview_label = toga.Label(
            f"Preview Rows: {self.app.preview_rows} (max: {max_preview_rows})",
            style=Pack(margin=(0, 0, 5, 0), font_weight='bold')
        )
        
        self.preview_rows_slider = toga.Slider(
            min=1,
            max=max_preview_rows,
            value=self.app.preview_rows,
            on_change=self.on_preview_rows_changed,
            style=Pack(width=300)
        )
        
        preview_section.add(preview_label)
        preview_section.add(self.preview_rows_slider)
        
        # Store reference to the label for updates
        self.preview_rows_label = preview_label
        
        # Navigation buttons
        nav_section = toga.Box(style=Pack(direction=ROW, margin=(20, 0, 0, 0)))
        back_button = toga.Button(
            "Back",
            on_press=self.back_to_source_selection,
            style=Pack(margin=5)
        )
        preview_button = toga.Button(
            "Preview Generation",
            on_press=self.preview_generation,
            style=Pack(margin=5)
        )
        nav_section.add(back_button)
        nav_section.add(toga.Box(style=Pack(flex=1)))
        nav_section.add(preview_button)
        
        # Add all components
        self.definitions_container.content = self.definitions_box
        self.output_box.add(title)
        self.output_box.add(self.definitions_container)
        self.output_box.add(add_button)
        self.output_box.add(preview_section)
        self.output_box.add(nav_section)
        
        # Refresh examples display for all existing definitions
        self.refresh_all_examples()
        
        return self.output_box
    
    def load_existing_definitions(self):
        """Load existing column definitions from app state or create initial empty one."""
        if self.app.column_definitions:
            # Store existing data before clearing
            existing_definitions = []
            for definition_data in self.app.column_definitions:
                # Extract the stored values
                headers_values = []
                if 'headers' in definition_data:
                    headers_values = [h.value if hasattr(h, 'value') else str(h) for h in definition_data['headers']]
                
                instructions_value = ""
                if 'instructions' in definition_data:
                    instructions_value = definition_data['instructions'].value if hasattr(definition_data['instructions'], 'value') else str(definition_data['instructions'])
                
                examples = definition_data.get('examples', {})
                
                existing_definitions.append({
                    'headers_values': headers_values,
                    'instructions_value': instructions_value,
                    'examples': examples
                })
            
            # Clear the UI-related definitions list
            self.app.column_definitions = []
            
            # Recreate definitions with stored data  
            for def_data in existing_definitions:
                # Create new definition
                self.add_column_definition()
                
                # Populate the new definition with stored values
                current_def = self.app.column_definitions[-1]
                
                # Set header values
                for i, header_input in enumerate(current_def['headers']):
                    if i < len(def_data['headers_values']):
                        header_input.value = def_data['headers_values'][i]
                
                # Set instructions value
                if def_data['instructions_value']:
                    current_def['instructions'].value = def_data['instructions_value']
                
                # Set examples
                current_def['examples'] = def_data['examples'].copy()
        else:
            # Add initial column definition if none exist
            self.add_column_definition()
    
    def refresh_all_examples(self):
        """Refresh the examples display for all column definitions."""
        for i, definition in enumerate(self.app.column_definitions):
            # Migrate old example format to new examples format if needed
            if 'example' in definition and definition['example'] and 'examples' not in definition:
                definition['examples'] = {'1': definition['example']}
                del definition['example']
            elif 'examples' not in definition:
                definition['examples'] = {}
                
            if 'examples_section' in definition:
                self.update_examples_display(i)
    
    def get_max_preview_rows(self):
        """Get the maximum number of preview rows based on non-empty data."""
        if self.app.current_df is None or self.app.current_df.is_empty() or not self.app.selected_columns:
            return 10  # Default fallback
        
        # Count non-empty rows in selected columns
        non_empty_count = self.app.tm.table_processor.count_non_empty_rows(
            self.app.current_df,
            self.app.selected_columns
        )
        
        # Return at least 1, but no more than the actual non-empty rows or 50 (reasonable upper limit)
        return max(1, min(non_empty_count, 50))
    
    def on_preview_rows_changed(self, widget):
        """Handle preview rows slider change."""
        max_rows = self.get_max_preview_rows()
        self.app.preview_rows = int(widget.value)
        self.preview_rows_label.text = f"Preview Rows: {self.app.preview_rows} (max: {max_rows})"
    
    def add_column_definition(self):
        """Add a new column definition widget."""
        definition_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        
        # Column header inputs
        header_label = toga.Label(
            f"Column Headers ({self.app.header_rows} row{'s' if self.app.header_rows > 1 else ''}):",
            style=Pack(margin=(0, 0, 5, 0), font_weight='bold')
        )
        definition_box.add(header_label)
        
        header_inputs = []
        for i in range(self.app.header_rows):
            header_input = toga.TextInput(
                placeholder=f"Header row {i + 1}",
                style=Pack(width=300, margin=(0, 0, 5, 0))
            )
            header_inputs.append(header_input)
            definition_box.add(header_input)
        
        # Instructions
        inst_label = toga.Label(
            "Generation Instructions:",
            style=Pack(margin=(10, 0, 5, 0), font_weight='bold')
        )
        instructions = toga.MultilineTextInput(
            placeholder="Describe how to generate this column based on the source columns...",
            style=Pack(width=500, height=100, margin=(0, 0, 10, 0))
        )
        
        definition_box.add(inst_label)
        definition_box.add(instructions)
        
        # Examples section (initially not added to parent)
        examples_section = toga.Box(style=Pack(direction=COLUMN, margin=(10, 0, 10, 0)))
        examples_label = toga.Label(
            "Examples:",
            style=Pack(margin=(0, 0, 5, 0), font_weight='bold')
        )
        examples_list_box = toga.Box(style=Pack(direction=COLUMN, margin=5, width=500))
        
        examples_section.add(examples_label)
        examples_section.add(examples_list_box)
        
        # Buttons section
        buttons_box = toga.Box(style=Pack(direction=ROW, margin=(0, 0, 10, 0)))
        
        # Capture the current index before adding the definition
        current_index = len(self.app.column_definitions)
        
        # Add example button
        example_button = toga.Button(
            "Add example",
            on_press=lambda w, idx=current_index: asyncio.create_task(self.add_example_for_next_row(idx)),
            style=Pack(margin=5)
        )
        buttons_box.add(example_button)
        
        # Remove button (if not the first definition)
        if len(self.app.column_definitions) > 0:
            remove_button = toga.Button(
                "Remove",
                on_press=lambda w: self.remove_column_definition(definition_box),
                style=Pack(margin=5)
            )
            buttons_box.add(remove_button)
        
        definition_box.add(buttons_box)
        
        # Store the definition
        self.app.column_definitions.append({
            'box': definition_box,
            'headers': header_inputs,
            'instructions': instructions,
            'examples': {},  # Will store examples as dict {row_number: example_text}
            'examples_section': examples_section,
            'examples_list_box': examples_list_box
        })
        
        self.definitions_box.add(definition_box)
    
    def update_examples_display(self, definition_index):
        """Update the examples display for a specific definition."""
        if definition_index >= len(self.app.column_definitions):
            return
            
        definition = self.app.column_definitions[definition_index]
        examples_section = definition.get('examples_section')
        examples_list_box = definition.get('examples_list_box')
        
        if not examples_section or not examples_list_box:
            return
        
        # Clear existing examples display
        for child in list(examples_list_box.children):
            examples_list_box.remove(child)
        
        examples = definition.get('examples', {})
        if examples:
            # Show the examples section by adding it to parent if not already present
            parent = definition['box']
            if examples_section not in parent.children:
                # Find where to insert (after instructions, before buttons)
                insert_index = len(parent.children) - 1  # Before buttons section
                parent.insert(insert_index, examples_section)
            
            # Create example items with edit and remove buttons (sorted by row number)
            for row_number in sorted(examples.keys(), key=int):
                self.add_example_item(
                    definition_index=definition_index,
                    row_number=int(row_number),
                    example_text=examples[row_number],
                    examples_list_box=examples_list_box
                )
        else:
            # Hide the examples section by removing it from parent
            parent = definition['box']
            if examples_section in parent.children:
                parent.remove(examples_section)
    
    def get_next_available_row(self, definition_index):
        """Get the next available row number for adding an example."""
        if definition_index >= len(self.app.column_definitions):
            return 1
            
        examples = self.app.column_definitions[definition_index].get('examples', {})
        if not examples:
            return 1
        
        # Find the lowest unused row number starting from 1
        used_rows = set(int(row) for row in examples.keys())
        next_row = 1
        while next_row in used_rows:
            next_row += 1
        
        # Make sure we don't exceed the available data rows
        if self.app.current_df is not None:
            max_rows = len(self.app.current_df)
            if next_row > max_rows:
                return None  # No more rows available
        
        return next_row
    
    def add_example_item(self, definition_index, row_number, example_text, examples_list_box):
        """Add an example item with edit and remove buttons."""
        # Create container for this item
        item_box = toga.Box(style=Pack(direction=ROW, margin=(5, 0, 5, 0)))
        
        # Content box for example text
        content_box = toga.Box(style=Pack(direction=COLUMN, flex=1, margin=(0, 10, 0, 0)))
        
        # Extract row info from example for display
        example_display = f"Row {row_number} example"
        if "Expected output: " in example_text:
            expected_output = example_text.split("Expected output: ")[-1]
            
            # Truncate at first newline OR 50 characters, whichever comes first
            newline_pos = expected_output.find('\n')
            if newline_pos != -1 and newline_pos < 50:
                # Newline comes before 50 chars
                expected_output = expected_output[:newline_pos] + "..."
            elif len(expected_output) > 50:
                # No newline in first 50 chars, so truncate at 50
                expected_output = expected_output[:50] + "..."
                
            example_display = f"Row {row_number}: {expected_output}"
        
        title_label = toga.Label(
            example_display,
            style=Pack(font_weight='bold')
        )
        subtitle_label = toga.Label(
            f"Example for row {row_number}",
            style=Pack(font_size=10, color='#666666')
        )
        
        content_box.add(title_label)
        content_box.add(subtitle_label)
        
        # Buttons box for edit and remove
        buttons_box = toga.Box(style=Pack(direction=ROW, margin=(0, 5, 0, 0)))
        
        # Edit button
        def create_edit_handler(def_idx, row_num):
            def handler(widget):
                self.show_example_window(def_idx, row_num)
            return handler
        
        edit_button = toga.Button(
            "Edit",
            on_press=create_edit_handler(definition_index, row_number),
            style=Pack(width=50, height=30, margin=(0, 2, 0, 0), background_color='#007acc', color='white')
        )
        
        # Remove button
        def create_remove_handler(def_idx, row_num):
            async def handler(widget):
                await self.remove_example_item(def_idx, row_num)
            return handler
        
        remove_button = toga.Button(
            "−",  # Unicode minus symbol
            on_press=create_remove_handler(definition_index, row_number),
            style=Pack(width=30, height=30, margin=(0, 0, 0, 0), background_color='#ff4444', color='white')
        )
        
        buttons_box.add(edit_button)
        buttons_box.add(remove_button)
        
        item_box.add(content_box)
        item_box.add(buttons_box)
        
        # Add to UI
        examples_list_box.add(item_box)
    
    async def remove_example_item(self, definition_index, row_number):
        """Remove an example from a specific definition."""
        # Remove the example from the definition
        examples = self.app.column_definitions[definition_index].get('examples', {})
        if str(row_number) in examples:
            del examples[str(row_number)]
        
        # Update the display
        self.update_examples_display(definition_index)

    async def add_column_definition_handler(self, widget):
        """Handler for add column button."""
        self.add_column_definition()
    
    def remove_column_definition(self, definition_box):
        """Remove a column definition."""
        # Find and remove from list
        for i, defn in enumerate(self.app.column_definitions):
            if defn['box'] == definition_box:
                self.app.column_definitions.pop(i)
                break
        
        # Remove from UI
        self.definitions_box.remove(definition_box)
    
    async def back_to_source_selection(self, widget):
        """Go back to source selection window."""
        self.app.show_source_selection_window()
    
    def format_examples_for_processing(self, examples_dict):
        """Format examples dictionary into a string for processing."""
        if not examples_dict:
            return ""
        
        formatted_examples = []
        for row_number in sorted(examples_dict.keys(), key=int):
            formatted_examples.append(f"Row {row_number}: {examples_dict[row_number]}")
        
        return "\n\n".join(formatted_examples)
    
    async def preview_generation(self, widget):
        """Preview the generation with selected number of rows."""
        # Validate definitions
        valid_definitions = []
        for defn in self.app.column_definitions:
            headers = [h.value for h in defn['headers'] if h.value]
            instructions = defn['instructions'].value
            
            if headers and instructions:
                # Format examples for processing
                examples_text = self.format_examples_for_processing(defn.get('examples', {}))
                
                valid_definitions.append({
                    'headers': headers,
                    'instructions': instructions,
                    'example': examples_text  # Use formatted examples for backward compatibility
                })
        
        if not valid_definitions:
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Error",
                    message="Please define at least one output column with headers and instructions."
                )
            )
            return
        
        # Show processing window
        self.app.show_processing_window(valid_definitions, preview_only=True)

    def show_example_window(self, column_index, row_number):
        """Show the example window for a specific column definition."""
        if not self.example_window:
            self.example_window = ExampleWindow(self.app)
        
        definition = self.app.column_definitions[column_index]
        self.example_window.show(definition, column_index, row_number) 

    async def add_example_for_next_row(self, column_index):
        """Add a new example for the next available row in a specific column definition."""
        next_row = self.get_next_available_row(column_index)
        if next_row is None:
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Error",
                    message="No more rows available to add an example."
                )
            )
            return
        
        self.show_example_window(column_index, next_row) 