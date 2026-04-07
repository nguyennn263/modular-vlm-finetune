from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


class OneSample(BaseModel):
    """Data structure for a single data sample."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image_path: str  # Store path instead of loaded image for memory efficiency
    question: str
    answer: str  # Single answer for VLM training
    metadata: Optional[dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return self.model_dump()

