#!/usr/bin/env python3
"""
Checkpoint loader utility for direct checkpoint URL loading
Handles conversion between checkpoint URLs and HuggingFace model loading
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from urllib.parse import urlparse

import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


class CheckpointLoader:
    """Utility class for loading models from direct checkpoint URLs"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "olmo_checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def download_checkpoint(self, url: str, force_download: bool = False) -> Path:
        """Download checkpoint from URL to local cache"""
        # Parse URL to create cache path
        parsed = urlparse(url)
        cache_name = parsed.path.strip('/').replace('/', '_')
        cache_path = self.cache_dir / cache_name
        
        if cache_path.exists() and not force_download:
            self.logger.info(f"Using cached checkpoint: {cache_path}")
            return cache_path
            
        self.logger.info(f"Downloading checkpoint from: {url}")
        
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "checkpoint"
            
            # For olmo-checkpoints.org, we need to download the entire directory
            if "olmo-checkpoints.org" in url:
                self._download_olmo_checkpoint(url, temp_path)
            else:
                # Generic download (for single files)
                self._download_generic_checkpoint(url, temp_path)
            
            # Move to cache
            if cache_path.exists():
                shutil.rmtree(cache_path)
            shutil.move(str(temp_path), str(cache_path))
            
        self.logger.info(f"Checkpoint cached at: {cache_path}")
        return cache_path
        
    def _download_olmo_checkpoint(self, url: str, dest: Path):
        """Download OLMo checkpoint directory structure"""
        # OLMo checkpoints typically have these files
        checkpoint_files = [
            "model.pt",
            "config.yaml", 
            "tokenizer.json",
            "vocab.json",
            "merges.txt"
        ]
        
        dest.mkdir(parents=True, exist_ok=True)
        
        for filename in checkpoint_files:
            file_url = url.rstrip('/') + '/' + filename
            file_path = dest / filename
            
            try:
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                self.logger.info(f"Downloaded: {filename}")
                
            except requests.RequestException as e:
                self.logger.warning(f"Failed to download {filename}: {e}")
                # Some files might not exist, continue with others
                continue
                
    def _download_generic_checkpoint(self, url: str, dest: Path):
        """Download generic checkpoint file"""
        dest.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename = url.split('/')[-1] or "checkpoint"
        file_path = dest / filename
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    def load_model_from_checkpoint(self, 
                                 checkpoint_path: Path,
                                 model_name: str,
                                 **model_kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer from local checkpoint"""
        
        # Try to load as HuggingFace format first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
            return model, tokenizer
            
        except Exception as e:
            self.logger.info(f"Failed to load as HuggingFace format: {e}")
            
        # Try to load as OLMo checkpoint format
        try:
            return self._load_olmo_checkpoint(checkpoint_path, model_name, **model_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to load as OLMo checkpoint: {e}")
            raise
            
    def _load_olmo_checkpoint(self, 
                            checkpoint_path: Path, 
                            model_name: str,
                            **model_kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load OLMo checkpoint format"""
        # For OLMo checkpoints, we need to use the base model and load the state dict
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Load checkpoint state
        checkpoint_file = checkpoint_path / "model.pt"
        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            model.load_state_dict(state_dict, strict=False)
            self.logger.info("Loaded checkpoint state dict")
            
        # Load tokenizer (fallback to base model if not found in checkpoint)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        except:
            self.logger.info("Loading tokenizer from base model")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        return model, tokenizer
        
    def get_model_and_tokenizer(self,
                              model_name: str,
                              checkpoint_url: Optional[str] = None,
                              force_download: bool = False,
                              **model_kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Get model and tokenizer, either from HuggingFace or direct checkpoint URL
        
        Args:
            model_name: HuggingFace model name (used as fallback)
            checkpoint_url: Direct URL to checkpoint (optional)
            force_download: Force re-download of cached checkpoints
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        
        if checkpoint_url:
            self.logger.info(f"Loading model from checkpoint URL: {checkpoint_url}")
            checkpoint_path = self.download_checkpoint(checkpoint_url, force_download)
            return self.load_model_from_checkpoint(checkpoint_path, model_name, **model_kwargs)
        else:
            self.logger.info(f"Loading model from HuggingFace: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer


def create_checkpoint_revision_mapping() -> Dict[str, str]:
    """Create mapping from checkpoint URLs to HuggingFace revisions"""
    # This maps the checkpoint URLs you provided to potential HuggingFace revisions
    url_to_revision = {}
    
    # OLMo 1B checkpoints
    base_url = "https://olmo-checkpoints.org/ai2-llm/peteish1"
    steps = [0, 300, 10000, 20000, 23100, 30000, 40000, 50000, 60000, 66200, 70000, 
             80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000,
             170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000,
             260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000, 340000,
             350000, 360000, 370000, 380000, 390000, 400000, 410000, 420000, 430000,
             440000, 450000, 460000, 470000, 480000, 490000, 500000, 510000, 520000,
             530000, 540000, 550000, 560000, 570000, 580000, 590000, 600000, 610000,
             620000, 630000, 640000, 650000, 660000, 670000, 680000, 690000, 700000,
             710000, 720000, 730000, 740000, 750000, 760000, 770000, 780000, 790000,
             800000, 810000, 820000, 830000, 840000, 850000, 860000, 870000, 880000,
             890000, 900000, 910000, 920000, 930000, 940000, 950000, 960000, 970000,
             980000, 990000, 1000000, 1010000, 1020000, 1030000, 1040000, 1050000,
             1060000, 1070000, 1080000, 1090000, 1100000, 1110000, 1120000, 1130000,
             1140000, 1150000, 1160000, 1170000, 1180000, 1190000, 1200000, 1210000,
             1220000, 1230000, 1240000, 1250000, 1260000, 1270000, 1280000, 1290000,
             1300000, 1310000, 1320000, 1330000, 1340000, 1350000, 1360000, 1370000,
             1380000, 1390000, 1400000, 1410000, 1420000, 1430000, 1440000, 1450000,
             1460000, 1470000, 1480000, 1490000, 1500000, 1510000, 1520000, 1530000,
             1540000, 1550000, 1560000, 1570000, 1580000, 1590000, 1600000, 1610000,
             1620000, 1630000, 1640000, 1650000, 1660000, 1670000, 1680000, 1690000,
             1700000, 1710000, 1720000, 1730000, 1740000, 1750000, 1760000, 1770000,
             1780000, 1790000, 1800000, 1810000, 1820000, 1830000, 1840000, 1850000,
             1860000, 1870000, 1880000, 1890000, 1900000, 1907359]
             
    for step in steps:
        url = f"{base_url}/step{step}-unsharded/"
        # Estimate token count (roughly 2.1 tokens per step for 1B model)
        tokens = int(step * 2.1)
        revision = f"stage1-step{step}-tokens{tokens}B"
        url_to_revision[url] = revision
        
    return url_to_revision


if __name__ == "__main__":
    # Example usage
    loader = CheckpointLoader()
    
    # Test with a checkpoint URL
    url = "https://olmo-checkpoints.org/ai2-llm/peteish1/step100000-unsharded/"
    model_name = "allenai/OLMo-2-0425-1B"
    
    try:
        model, tokenizer = loader.get_model_and_tokenizer(
            model_name=model_name,
            checkpoint_url=url,
            torch_dtype=torch.float16
        )
        print("Successfully loaded model from checkpoint!")
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Falling back to HuggingFace model...")
        
        model, tokenizer = loader.get_model_and_tokenizer(
            model_name=model_name,
            torch_dtype=torch.float16
        )
        print("Successfully loaded model from HuggingFace!")