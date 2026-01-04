from .registry import ModelRegistry, VISION_ENCODER_CONFIGS, get_vision_config
from .vision_encoders import (
    BaseVisionEncoder,
    InternViTEncoder,
    SiglipEncoder,
    CLIPEncoder,
    build_vision_encoder,
)
from .projectors import (
    MLPProjector,
    MLPGeLUProjector,
    LinearProjector,
    IdentityProjector,
    DownsampleProjector,
    build_projector,
)
from .vlm import VLMModel, create_vlm_model

__all__ = [
    # Registry
    "ModelRegistry",
    "VISION_ENCODER_CONFIGS",
    "get_vision_config",
    # Vision Encoders
    "BaseVisionEncoder",
    "InternViTEncoder", 
    "SiglipEncoder",
    "CLIPEncoder",
    "build_vision_encoder",
    # Projectors
    "MLPProjector",
    "MLPGeLUProjector",
    "LinearProjector",
    "IdentityProjector",
    "DownsampleProjector",
    "build_projector",
    # Main Model
    "VLMModel",
    "create_vlm_model",
]


# Legacy imports (deprecated - will be removed in future versions)
def __getattr__(name):
    """Lazy import for legacy components with deprecation warning"""
    legacy_components = ["VinternVLM", "VisionTower", "LegacyMLPProjector"]
    if name in legacy_components:
        import warnings
        warnings.warn(
            f"{name} is deprecated. Use the new modular system instead:\n"
            "  - VLMModel from src.models.vlm\n"
            "  - build_vision_encoder from src.models.vision_encoders\n"
            "  - build_projector from src.models.projectors",
            DeprecationWarning,
            stacklevel=2
        )
        from .legacy import VinternVLM, VisionTower, LegacyMLPProjector
        return {"VinternVLM": VinternVLM, "VisionTower": VisionTower, "LegacyMLPProjector": LegacyMLPProjector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
