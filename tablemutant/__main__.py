#!/usr/bin/env python3
"""
TableMutant Main Entry Point - Checks for CLI args and launches GUI if none provided
"""

import os
import sys
import platform
import multiprocessing as mp

def _configure_mp_for_briefcase():
    """Configure multiprocessing for Briefcase-bundled apps to prevent GUI respawning."""
    # Make frozen bundles cooperate with multiprocessing
    try:
        mp.freeze_support()
    except Exception:
        pass

    if platform.system() == "Darwin":
        # 1) Prefer 'fork' to avoid re-exec of the GUI stub
        try:
            # Only set if not set yet; prevents RuntimeError if already set
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("fork")
        except Exception:
            # If 'fork' is not viable, fall back to spawn but make it headless
            pass

        # 2) If we end up on 'spawn' anyway, make sure the spawned interpreter
        # is *not* the GUI stub, to avoid a second app window.
        try:
            if mp.get_start_method() == "spawn":
                # Use the framework interpreter embedded by Briefcase if present;
                # else fall back to a system python.
                candidates = [
                    os.path.join(sys.prefix, "bin", "python3"),
                    "/usr/bin/python3"
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        mp.set_executable(cand)
                        break
        except Exception:
            pass

    # 3) Nudge common libs away from process-based parallelism
    os.environ.setdefault("JOBLIB_START_METHOD", "threading")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HF tokenizers
    # Optional; tame oversubscription while you're here
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Configure multiprocessing before any other imports
_configure_mp_for_briefcase()

import argparse
import logging

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_level='INFO'):
    """Setup logging configuration with the specified log level."""
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger for tablemutant
    logger = logging.getLogger('tablemutant')
    return logger

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
    parser.add_argument('--log-level', '-l',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='Set the logging level (default: INFO)')
    
    # Use parse_known_args to tolerate interpreter/launcher flags (e.g., -B, -I)
    args, _unknown = parser.parse_known_args()
    
    # Setup logging with the specified level
    setup_logging(args.log_level)
    
    # Check if we should run in GUI mode
    if args.gui or not (args.model and args.table and args.instructions):
        # Launch GUI
        try:
            # If this is a multiprocessing child, do not construct a GUI app
            try:
                if mp.current_process().name != "MainProcess":
                    return None
            except Exception:
                # Be conservative
                return None
            
            import toga
            from tablemutant.ui import TableMutantGUI
            
            # Setup logging for GUI mode (default to INFO level)
            if not hasattr(args, 'log_level') or not args.log_level:
                setup_logging('INFO')
            
            app = TableMutantGUI('TableMutant', 'org.tablemutant.gui')
            return app.main_loop()
            
        except ImportError as e:
            logger = logging.getLogger('tablemutant')
            logger.error("Cannot launch GUI - %s", e)
            logger.error("The GUI requires Toga to be installed.")
            logger.error("Install with: pip install toga")
            logger.error("Alternatively, use CLI mode with required arguments:")
            logger.error("  --model, --table, and --instructions")
            logger.error("Run 'tablemutant --help' for more information.")
            sys.exit(1)
    else:
        # Run CLI mode
        try:
            from tablemutant.core import TableMutant
            
            tm = TableMutant()
            tm.run(args)
        except KeyboardInterrupt:
            logger = logging.getLogger('tablemutant')
            logger.info("Interrupted by user")
            if 'tm' in locals():
                tm.cleanup()
            sys.exit(1)
        except Exception as e:
            logger = logging.getLogger('tablemutant')
            logger.error("Error: %s", e)
            if 'tm' in locals():
                tm.cleanup()
            sys.exit(1)


if __name__ == "__main__":
    main()