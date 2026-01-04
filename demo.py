"""
Simple demo - Test modular VLM system
"""
import torch
from src.models.registry import ModelRegistry
from src.models.vision_encoders import build_vision_encoder
from src.models.projectors import build_projector

def demo_registry():
    """Demo model registry"""
    print("=" * 50)
    print("1. MODEL REGISTRY")
    print("=" * 50)
    
    print("\nAvailable LLMs:")
    for name, config in ModelRegistry.LLM_CONFIGS.items():
        print(f"- {name}: {config['model_name']}")
    
    print("\nAvailable Vision Encoders:")
    for name in ModelRegistry.VISION_ENCODER_CONFIGS.keys():
        print(f"- {name}")
    
    # List registered projectors
    print("\nAvailable Projectors:")
    for name in ModelRegistry._projectors.keys():
        print(f"- {name}")


def demo_build_components():
    """Demo building components"""
    print("\n" + "=" * 50)
    print("2. BUILD COMPONENTS")
    print("=" * 50)
    
    # Build projector (lightweight, no download needed) 
    print("\nBuilding MLP Projector...")
    projector = build_projector(
        projector_type="mlp",
        vision_hidden_size=1024,
        llm_hidden_size=896,
        num_hidden=2
    )
    print(f"Input: 1024 -> Output: 896")
    print(f"Parameters: {sum(p.numel() for p in projector.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 10, 1024)  # [batch, num_patches, hidden]
    output = projector(dummy_input)
    print(f"Forward: {dummy_input.shape} -> {output.shape}")
    
    # Build different projector types
    print("\n Building other projectors...")
    for ptype in ["linear", "mlp_gelu", "downsample"]:
        p = build_projector(ptype, 1024, 896)
        params = sum(x.numel() for x in p.parameters())
        print(f"{ptype}: {params:,} params")


def demo_config_swap():
    """Demo how to swap models via config"""
    print("\n" + "=" * 50)
    print("3. CONFIG-BASED MODEL SWAP")
    print("=" * 50)
    
    configs = [
        {"vision": "internvit", "llm": "qwen2-0.5b", "projector": "mlp"},
        {"vision": "siglip", "llm": "qwen2-1.5b", "projector": "mlp_gelu"},
        {"vision": "clip", "llm": "phi-2", "projector": "linear"},
    ]
    
    print("\nExample configurations:")
    for i, cfg in enumerate(configs, 1):
        llm_cfg = ModelRegistry.get_llm_config(cfg["llm"])
        vision_cfg = ModelRegistry.VISION_ENCODER_CONFIGS.get(cfg["vision"], {})
        
        print(f"\nConfig {i}:")
        print(f"Vision: {cfg['vision']} ({vision_cfg.get('model_name', 'N/A')})")
        print(f"LLM: {cfg['llm']} ({llm_cfg['model_name']})")
        print(f"Projector: {cfg['projector']}")
def demo_full_model(download=False):
    """Demo full VLM model (optional download)"""
    if not download:
        print("\n" + "=" * 50)
        print("4. FULL MODEL (skipped - set download=True)")
        print("=" * 50)
        print("\nTo test full model, run:")
        print("python demo.py --full")
        return
    
    print("\n" + "=" * 50)
    print("4. FULL VLM MODEL")
    print("=" * 50)
    
    from src.models.vlm import create_vlm_model
    
    config = {
        "vision_encoder_type": "internvit",
        "projector_type": "mlp",
        "llm_type": "qwen2-0.5b",
        "freeze_vision": True,
        "freeze_llm": False,
    }
    
    print("\nBuilding VLM model...")
    print(f" Config: {config}")
    
    model = create_vlm_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable: {trainable:,} ({100*trainable/total_params:.1f}%)")


if __name__ == "__main__":
    import sys
    
    demo_registry()
    demo_build_components()
    demo_config_swap()
    
    # Full model demo only if --full flag
    demo_full_model(download="--full" in sys.argv)
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)
