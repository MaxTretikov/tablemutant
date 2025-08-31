#!/usr/bin/env python3
"""
TableProcessor - Handles table loading, encoding detection, and saving operations
"""

import logging
import os
from typing import List, Optional
import chardet
import polars as pl

# Get logger for this module
logger = logging.getLogger('tablemutant.core.table_processor')


class TableProcessor:
    def __init__(self):
        self.header_processor = HeaderProcessor()
        pass
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as f:
            # Read a sample of the file for detection
            sample = f.read(100000)  # Read first 100KB
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logger.debug("Detected encoding: %s (confidence: %.2f)", encoding, confidence)
            
            # If confidence is low, try some common encodings
            if confidence < 0.7:
                common_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16', 'utf-32']
                for enc in common_encodings:
                    try:
                        with open(file_path, 'r', encoding=enc) as test_file:
                            test_file.read(1000)  # Try reading first 1000 chars
                        return enc
                    except (UnicodeDecodeError, UnicodeError):
                        continue
            
            return encoding or 'utf-8'
    
    def detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect the delimiter used in a CSV file."""
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                # Read first few lines
                sample_lines = []
                for _ in range(5):
                    line = f.readline()
                    if line:
                        sample_lines.append(line)
                
                # Count occurrences of common delimiters
                delimiters = [',', ';', '\t', '|']
                delimiter_counts = {d: 0 for d in delimiters}
                
                for line in sample_lines:
                    for delimiter in delimiters:
                        delimiter_counts[delimiter] += line.count(delimiter)
                
                # Return the most common delimiter
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                if delimiter_counts[best_delimiter] > 0:
                    logger.debug("Detected delimiter: '%s'", best_delimiter)
                    return best_delimiter
                
                return ','  # Default to comma
        except:
            return ','
    
    def load_table(self, file_path: str) -> pl.DataFrame:
        """Load table with proper encoding detection."""
        if file_path.endswith('.parquet'):
            # Parquet files handle encoding internally
            return pl.read_parquet(file_path)
        elif file_path.endswith('.json'):
            # Detect encoding for JSON
            encoding = self.detect_encoding(file_path)
            # Read as text first, then parse
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return pl.read_json(f.read())
        else:
            # For CSV and other text formats
            encoding = self.detect_encoding(file_path)
            delimiter = self.detect_delimiter(file_path, encoding)
            
            # Try different approaches to read the CSV
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # First attempt: use detected encoding and delimiter, no auto-headers
                        return pl.read_csv(
                            file_path,
                            encoding=encoding,
                            separator=delimiter,
                            has_header=False,  # Disable auto-header detection
                            ignore_errors=True,
                            null_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a'],
                            truncate_ragged_lines=True
                        )
                    elif attempt == 1:
                        # Second attempt: try with infer_schema_length=0, no auto-headers
                        return pl.read_csv(
                            file_path,
                            encoding=encoding,
                            separator=delimiter,
                            has_header=False,  # Disable auto-header detection
                            ignore_errors=True,
                            infer_schema_length=0,  # Read all as strings
                            null_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a'],
                            truncate_ragged_lines=True
                        )
                    elif attempt == 2:
                        # Third attempt: manually read and parse (no headers)
                        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                            lines = f.readlines()
                        
                        if not lines:
                            raise ValueError("Empty file")
                        
                        # Don't parse header, treat all lines as data
                        data = []
                        max_cols = 0
                        
                        # First pass: find maximum number of columns
                        for line in lines:
                            if line.strip():
                                row = line.strip().split(delimiter)
                                max_cols = max(max_cols, len(row))
                        
                        # Second pass: parse all data with consistent column count
                        for line in lines:
                            if line.strip():
                                row = line.strip().split(delimiter)
                                # Pad or truncate row to match max columns
                                if len(row) < max_cols:
                                    row.extend([''] * (max_cols - len(row)))
                                elif len(row) > max_cols:
                                    row = row[:max_cols]
                                data.append(row)
                        
                        # Create DataFrame with generic column names
                        if data:
                            column_names = [f"column_{i}" for i in range(max_cols)]
                            df_dict = {col: [row[i] if i < len(row) else '' for row in data] for i, col in enumerate(column_names)}
                            return pl.DataFrame(df_dict)
                        else:
                            return pl.DataFrame()
                        
                except Exception as e:
                    if attempt < 2:
                        logger.debug("Attempt %d failed: %s", attempt + 1, e)
                        continue
                    else:
                        raise RuntimeError(f"Failed to load table after all attempts: {e}")
    
    def parse_column_indices(self, indices_str: str, total_columns: int) -> List[int]:
        """Parse column indices from string like '0-2,4,6-8'."""
        if not indices_str:
            return list(range(total_columns))
        
        indices = []
        parts = indices_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        
        # Filter valid indices
        indices = [i for i in indices if 0 <= i < total_columns]
        return sorted(set(indices))
    
    def count_non_empty_rows(self, df: pl.DataFrame, selected_columns: List[int] = None) -> int:
        """Count rows that have at least one non-empty value in the selected columns."""
        if df is None or df.is_empty():
            return 0
        
        # If no columns specified, use all columns
        if selected_columns is None:
            selected_columns = list(range(len(df.columns)))
        
        # Filter to valid column indices
        valid_columns = [i for i in selected_columns if 0 <= i < len(df.columns)]
        if not valid_columns:
            return 0
        
        # Get column names for the selected indices
        column_names = [df.columns[i] for i in valid_columns]
        
        non_empty_count = 0
        for i in range(len(df)):
            has_non_empty = False
            for col_name in column_names:
                try:
                    value = df[col_name][i]
                    # Check if value is non-empty (not None, not empty string, not just whitespace)
                    if value is not None:
                        str_value = str(value).strip()
                        if str_value and str_value.lower() not in ['null', 'na', 'n/a', '']:
                            has_non_empty = True
                            break
                except Exception:
                    continue
            
            if has_non_empty:
                non_empty_count += 1
        
        return non_empty_count
    
    def save_table(self, df: pl.DataFrame, output_path: Optional[str], new_column_name: str = None):
        """Save table with proper encoding or output to stdout."""
        if not output_path:
            # Output only the new column to stdout
            if new_column_name and new_column_name in df.columns:
                logger.info("\n" + "="*50 + " OUTPUT " + "="*50)
                for value in df[new_column_name]:
                    logger.info("%s", value if value else "")
                    logger.info("---")
            else:
                # Fallback: output entire CSV
                logger.info("\n" + "="*50 + " OUTPUT " + "="*50)
                logger.info("%s", df.write_csv(None))
            return
            
        try:
            if output_path.endswith('.csv'):
                # Save as UTF-8 with BOM for better compatibility
                df.write_csv(output_path, encoding='utf-8-sig')
            elif output_path.endswith('.parquet'):
                df.write_parquet(output_path)
            elif output_path.endswith('.json'):
                # Write JSON with UTF-8
                df.write_json(output_path)
            else:
                # Default to CSV with UTF-8
                if not output_path.endswith('.csv'):
                    output_path += '.csv'
                df.write_csv(output_path, encoding='utf-8-sig')
        except Exception as e:
            logger.debug("Error saving with UTF-8, trying latin-1: %s", e)
            # Fallback to latin-1 for maximum compatibility
            if output_path.endswith('.csv') or not output_path.endswith(('.parquet', '.json')):
                df.write_csv(output_path, encoding='latin-1') 


class HeaderProcessor:
    """Handles detection of header rows and creation of DataFrames with combined headers."""
    
    def __init__(self):
        pass
    
    def detect_header_rows(self, df: pl.DataFrame) -> tuple[int, float]:
        """
        Detect the number of header rows in a DataFrame.
        
        Args:
            df: The DataFrame to analyze
            
        Returns:
            Tuple of (header_rows, confidence) where header_rows is the detected
            number of header rows and confidence is a float between 0 and 1
        """
        if len(df) < 5:
            return 1, 0.5
        
        # Check for multi-level headers by analyzing data types consistency
        confidence = 0.8
        header_rows = 1
        
        # Get first few rows as strings
        first_rows = []
        for i in range(min(5, len(df))):
            row = []
            for col in df.columns:
                try:
                    val = df[col][i]
                    row.append(str(val) if val is not None else "")
                except Exception:
                    row.append("")
            first_rows.append(row)
        
        # Check if second row looks like headers (all strings, no numbers)
        if len(first_rows) > 1:
            second_row = first_rows[1]
            # Check first 5 columns or all if less
            cols_to_check = min(5, len(second_row))
            if cols_to_check > 0 and all(
                val and not val.replace('.', '').replace('-', '').isdigit() 
                for val in second_row[:cols_to_check] if val
            ):
                header_rows = 2
                confidence = 0.7
        
        return header_rows, confidence
    
    def create_working_dataframe(self, original_df: pl.DataFrame, header_rows: Optional[int] = None) -> pl.DataFrame:
        """
        Create a working DataFrame with multi-row headers combined as column names.
        
        Args:
            original_df: The original DataFrame loaded from file
            header_rows: Number of header rows to use. If None, auto-detects using detect_header_rows()
            
        Returns:
            A new DataFrame with combined header column names and data rows only
        """
        if original_df is None or original_df.is_empty():
            return pl.DataFrame()
        
        # Use provided header_rows or auto-detect
        if header_rows is None:
            header_rows, _ = self.detect_header_rows(original_df)
        
        # Ensure header_rows is within valid range
        header_rows = max(1, min(header_rows, len(original_df)))
        
        # Extract header rows
        header_data = []
        for h in range(header_rows):
            if h < len(original_df):
                row = []
                for col in original_df.columns:
                    try:
                        val = original_df[col][h]
                        row.append(str(val) if val is not None else "")
                    except Exception:
                        row.append("")
                header_data.append(row)
        
        # Create new column names from header rows
        new_columns = []
        for col_idx in range(len(original_df.columns)):
            # Get header values for this column
            header_parts = []
            for h in range(len(header_data)):
                if col_idx < len(header_data[h]):
                    part = header_data[h][col_idx]
                    if part:  # Only add non-empty parts
                        header_parts.append(part)
            
            # Create column name with index prefix
            letter = chr(ord('A') + col_idx) if col_idx < 26 else f"A{chr(ord('A') + col_idx - 26)}"
            if header_parts:
                col_name = f"[{letter}] " + " - ".join(header_parts)
            else:
                col_name = f"[{letter}] Column {col_idx + 1}"
            
            new_columns.append(col_name)
        
        # Create new DataFrame without header rows
        data_start = header_rows
        if data_start < len(original_df):
            # Get data rows
            data_rows = []
            for i in range(data_start, len(original_df)):
                row = []
                for col in original_df.columns:
                    try:
                        val = original_df[col][i]
                        row.append(val)
                    except Exception:
                        row.append(None)
                data_rows.append(row)
            
            # Create new DataFrame with proper column names
            return pl.DataFrame(
                data_rows,
                schema=new_columns,
                orient="row"
            )
        else:
            # No data rows, just create empty DataFrame with columns
            return pl.DataFrame(schema=new_columns) 