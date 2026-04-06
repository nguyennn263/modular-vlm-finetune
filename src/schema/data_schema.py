from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class OneSample(BaseModel):
    """Data structure for a single data sample."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image_path: str  # Store path instead of loaded image for memory efficiency
    question: str
    answer: str  # Single answer for VLM training
    metadata: Optional[dict] = None
