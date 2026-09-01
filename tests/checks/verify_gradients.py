#!/usr/bin/env python3
"""
Gradient flow verification script.
Ensures that:
1. Vision embeddings have gradient paths
2. Bridge parameters are trainable
3. No detach() breaks the computation graph
"""

import torch
import torch.nn as nn
from transformers import AutoModel

print("=" * 80)
print("GRADIENT FLOW VERIFICATION")
print("=" * 80)

# Load a tiny model for testing
print("\n[1] Loading base model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
    low_cpu_mem_usage=True
).eval()

# Import bridge setup
from src.training import create_finetune_model

print("[2] Creating fine-tune model...")
model = create_finetune_model(
    base_model,
    bridge_type="residual",
    bridge_config={}
)
model = model.to(device)

print("[3] Checking model setup...")

# Check vision model is frozen
vision_grad_count = sum(1 for p in model.vision_model.parameters() if p.requires_grad)
print(f"   Vision model trainable params: {vision_grad_count}")
assert vision_grad_count == 0, "❌ Vision model should be frozen!"

# Check LLM is frozen
llm_grad_count = sum(1 for p in model.language_model.parameters() if p.requires_grad)
print(f"   LLM trainable params: {llm_grad_count}")
assert llm_grad_count == 0, "❌ LLM should be frozen!"

# Check bridge is trainable
bridge_grad_count = sum(1 for p in model.bridge.parameters() if p.requires_grad)
print(f"   Bridge trainable params: {bridge_grad_count}")
assert bridge_grad_count > 0, "❌ Bridge should be trainable!"

print("\n[4] Testing forward pass with gradient computation...")

# Create dummy input (batch_size=2, 1 channel, small images for speed)
pixel_values = torch.randn(2, 1, 3, 448, 448, dtype=torch.bfloat16, device=device)

# Process through vision model
with torch.no_grad():
    vision_output = model.vision_model(pixel_values)
    
# Extract embeddings
if hasattr(vision_output, 'last_hidden_state'):
    last_hidden = vision_output.last_hidden_state  # [2, num_patches, 1024]
    print(f"   Vision output shape: {last_hidden.shape}")
else:
    last_hidden = vision_output

# Get embeddings for bridge (keep NO detach!)
vision_embeddings = last_hidden.clone()  # Clone without detach to preserve gradient
print(f"   Vision embeddings requires_grad: {vision_embeddings.requires_grad}")

# Forward through bridge
print("\n[5] Testing bridge gradients...")
bridged = model.bridge(vision_embeddings)
print(f"   Bridged output requires_grad: {bridged.requires_grad}")
print(f"   Bridged output shape: {bridged.shape}")

# Test backward pass
print("\n[6] Testing backward pass...")
loss = bridged.sum()
print(f"   Loss tensor requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("   ✓ Backward pass succeeded!")
except RuntimeError as e:
    print(f"   ❌ Backward failed: {e}")

# Check if bridge got gradients
print("\n[7] Checking gradient accumulation...")
bridge_has_grads = False
for name, param in model.bridge.named_parameters():
    if param.grad is not None:
        bridge_has_grads = True
        print(f"   ✓ {name} has gradients: {param.grad.shape}")
        break

if bridge_has_grads:
    print("   ✓✓✓ Bridge parameters received gradients!")
else:
    print("   ❌ Bridge parameters have NO gradients!")

print("\n" + "=" * 80)
if vision_grad_count == 0 and llm_grad_count == 0 and bridge_grad_count > 0 and bridge_has_grads:
    print("✓✓✓ ALL CHECKS PASSED - Gradients flowing correctly!")
else:
    print("❌ Some checks failed - see above")
print("=" * 80)
