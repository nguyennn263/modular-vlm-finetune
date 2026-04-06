"""
Simple logging setup for training and data processing.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create loggers
data_loader_logger = logging.getLogger('data_loader_logger')
model_builder_logger = logging.getLogger('model_builder_logger')
data_process_logger = logging.getLogger('data_process_logger')