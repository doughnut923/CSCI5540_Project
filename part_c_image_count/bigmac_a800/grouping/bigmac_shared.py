"""Expose the previous validated module while preserving its own relative paths."""
import os
import importlib.util
from pathlib import Path
import sys

path = Path(os.environ['PARTC_PRIMARY']) / 'code/bigmac_qwen2vl.py'
spec = importlib.util.spec_from_file_location('mix_shared', path)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
smoke = shared.smoke
TimedBridge = shared.TimedBridge
from types import SimpleNamespace
import torch
import torch.distributed as dist
import time
import math
