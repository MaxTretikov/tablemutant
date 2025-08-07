#!/usr/bin/env python3
"""
TableMutant Main Entry Point - Checks for CLI args and launches GUI if none provided
"""

import argparse
import sys
import os

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="TableMutant - Generate new columns in datasets using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a HuggingFace model (CLI mode)
  tablemutant --model TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \\
              --table data.csv \\
              --instructions "Summarize the content in columns 0 and 1"

  # Output to stdout (only new column values)
  tablemutant --model model.gguf \\
              --table data.csv \\
              --instructions "Extract key entities"

  # Launch GUI (no arguments)
  tablemutant
        """
    )
    
    parser.add_argument('--model', '-m',
                        help='HuggingFace model ID, path, or URL to GGUF file')
    parser.add_argument('--table', '-t',
                        help='Path to table file (CSV, Parquet, or JSON)')
    parser.add_argument('--columns', '-c',
                        help='Source column indices (e.g., "0-2,4,6-8"). Default: all columns')
    parser.add_argument('--instructions', '-i',
                        help='Instructions for generating the new column')
    parser.add_argument('--rag-source', '-r',
                        help='Optional PDF file for RAG context')
    parser.add_argument('--output', '-o',
                        help='Output file path (if not specified, outputs to stdout)')
    parser.add_argument('--output-column', '-n',
                        help='Name for the new column (default: generated_column)')
    parser.add_argument('--rows', type=int,
                        help='Number of rows to process (default: all rows)')
    parser.add_argument('--gui', action='store_true',
                        help='Force launch GUI mode')
    
    args = parser.parse_args()
    
    # Check if we should run in GUI mode
    if args.gui or not (args.model and args.table and args.instructions):
        # Launch GUI
        try:
            import toga
            from tablemutant.ui import TableMutantGUI
            
            app = TableMutantGUI('TableMutant', 'org.tablemutant.gui')
            return app.main_loop()
            
        except ImportError as e:
            print(f"Error: Cannot launch GUI - {e}")
            print("\nThe GUI requires Toga to be installed.")
            print("Install with: pip install toga")
            print("\nAlternatively, use CLI mode with required arguments:")
            print("  --model, --table, and --instructions")
            print("\nRun 'tablemutant --help' for more information.")
            sys.exit(1)
    else:
        # Run CLI mode
        try:
            from tablemutant.core import TableMutant
            
            tm = TableMutant()
            tm.run(args)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if 'tm' in locals():
                tm.cleanup()
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            if 'tm' in locals():
                tm.cleanup()
            sys.exit(1)


if __name__ == "__main__":
    main()