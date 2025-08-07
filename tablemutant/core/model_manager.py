#!/usr/bin/env python3
"""
ModelManager - Handles model downloading and llamafile operations
"""

import subprocess
import sys
import os
import signal
import atexit
import time
import threading
import urllib.request
import urllib.parse
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from tqdm import tqdm


class ModelManager:
    def __init__(self):
        self.llamafile_process = None
        self.llamafile_path = None
        self.model_path = None
        
    def validate_gguf(self, model_identifier: str) -> bool:
        """Validate that the model identifier points to a GGUF file."""
        # Check if it's a file path
        if os.path.exists(model_identifier):
            return model_identifier.endswith('.gguf')
        
        # Check if it's a URL
        parsed = urllib.parse.urlparse(model_identifier)
        if parsed.scheme in ['http', 'https']:
            return parsed.path.endswith('.gguf')
        
        # Check if it's a HuggingFace identifier
        # Format: username/model-name or username/model-name/filename.gguf
        parts = model_identifier.split('/')
        if len(parts) >= 2:
            # If filename is specified, check if it's GGUF
            if len(parts) > 2 and parts[-1].endswith('.gguf'):
                return True
            # Otherwise, we'll need to check the repo
            # For now, assume it needs to have GGUF in the name
            return 'gguf' in model_identifier.lower()
        
        return False
    
    def download_model(self, model_identifier: str) -> str:
        """Download model if it's not local. Returns path to local file."""
        # If it's already a local file, return it
        if os.path.exists(model_identifier):
            return os.path.abspath(model_identifier)
        
        # Create models directory if it doesn't exist
        models_dir = Path.home() / '.tablemutant' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle HuggingFace models
        if '/' in model_identifier and not model_identifier.startswith('http'):
            # Use huggingface-cli to download
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                
                parts = model_identifier.split('/')
                repo_id = '/'.join(parts[:2])
                
                # If specific file is mentioned
                if len(parts) > 2:
                    filename = parts[-1]
                else:
                    # Find GGUF file in repo
                    files = list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in repository {repo_id}")
                    filename = gguf_files[0]  # Take the first GGUF file
                
                print(f"Downloading {filename} from {repo_id}...")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=models_dir,
                    local_dir=models_dir / repo_id.replace('/', '_')
                )
                return local_path
                
            except ImportError:
                print("huggingface-hub not installed. Install with: pip install huggingface-hub")
                sys.exit(1)
            except Exception as e:
                print(f"Error downloading from HuggingFace: {e}")
                sys.exit(1)
        
        # Handle direct URLs
        elif model_identifier.startswith('http'):
            filename = os.path.basename(urllib.parse.urlparse(model_identifier).path)
            local_path = models_dir / filename
            
            if local_path.exists():
                print(f"Model already downloaded: {local_path}")
                return str(local_path)
            
            print(f"Downloading {model_identifier}...")
            
            # Use urllib instead of requests
            with urllib.request.urlopen(model_identifier) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(local_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return str(local_path)
        
        raise ValueError(f"Unknown model identifier format: {model_identifier}")
    
    def download_llamafile(self) -> str:
        """Download the latest llamafile if not present."""
        llamafile_dir = Path.home() / '.tablemutant' / 'bin'
        llamafile_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if llamafile already exists
        existing_llamafiles = list(llamafile_dir.glob('llamafile-*'))
        if existing_llamafiles:
            # Use the most recent one
            llamafile_path = max(existing_llamafiles, key=os.path.getctime)
            print(f"Using existing llamafile: {llamafile_path}")
            # Ensure it's executable
            os.chmod(llamafile_path, 0o755)
            return str(llamafile_path)
        
        # Get latest release info using urllib
        print("Fetching latest llamafile release...")
        with urllib.request.urlopen('https://api.github.com/repos/Mozilla-Ocho/llamafile/releases/latest') as response:
            release_data = json.loads(response.read().decode())
        
        # Find the llamafile asset (universal binary)
        llamafile_asset = None
        for asset in release_data['assets']:
            name = asset['name']
            # Look for the main llamafile binary (not .exe, .zip, etc.)
            if name.startswith('llamafile-') and not any(ext in name for ext in ['.zip', '.tar', '.exe', '.gz']):
                llamafile_asset = asset
                break
        
        if not llamafile_asset:
            raise ValueError("Could not find llamafile asset in latest release")
        
        # Download llamafile
        local_path = llamafile_dir / llamafile_asset['name']
        print(f"Downloading {llamafile_asset['name']}...")
        
        with urllib.request.urlopen(llamafile_asset['browser_download_url']) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Make executable
        os.chmod(local_path, 0o755)
        
        return str(local_path)
    
    def start_llamafile(self, model_path: str, port: int = 8000):
        """Start llamafile subprocess."""
        print(f"Starting llamafile with model: {model_path} on port {port}")
        
        # Ensure the llamafile binary is executable
        os.chmod(self.llamafile_path, 0o755)
        
        # Use shell=True for better compatibility
        cmd_str = f'"{self.llamafile_path}" -m "{model_path}" --port {port}'
        
        self.llamafile_process = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        # Only register signal handlers if we're in the main thread
        # Signal handlers can only be registered in the main thread
        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, lambda s, f: self.cleanup())
            signal.signal(signal.SIGTERM, lambda s, f: self.cleanup())
        
        # Wait for server to be ready
        print("Waiting for llamafile server to start...")
        max_retries = 30
        for i in range(max_retries):
            try:
                # Use urllib instead of requests
                with urllib.request.urlopen(f'http://localhost:{port}/v1/models') as response:
                    if response.status == 200:
                        print("Llamafile server is ready!")
                        break
            except (urllib.error.URLError, ConnectionRefusedError):
                pass
            
            if i == max_retries - 1:
                raise RuntimeError("Llamafile server failed to start")
            
            time.sleep(1)
    
    def cleanup(self):
        """Clean up llamafile process."""
        if self.llamafile_process:
            print("\nShutting down llamafile...")
            self.llamafile_process.terminate()
            try:
                self.llamafile_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.llamafile_process.kill()
            self.llamafile_process = None 