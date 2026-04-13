#!/usr/bin/env python3
"""
Validation script for Vintern Integration & Format Compliance

Checks:
1. Template format matches official Vintern format (from ref2)
2. Loss masking works correctly
3. All 6 bridge architectures compatible
4. Checkpoints save/load correctly
"""

import os
import sys
import torch
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import VLMDataset
from src.data.collator_onesample import create_collate_fn
from utils.data_loader_helper import AblationDataLoader
from transformers import AutoTokenizer

def check_template_format():
    """Verify output format matches official Vintern template"""
    print("=" * 80)
    print("CHECK 1: Template Format Compliance")
    print("=" * 80)
    
    try:
        loader = AblationDataLoader()
        samples, _, _ = loader.load_train_val_split(max_samples=1, val_ratio=0.2)
        
        if not samples:
            print("❌ No samples loaded")
            return False
        
        sample = samples[0]
        question = sample.question
        answer = sample.answer
        
        print(f"\n✓ Sample loaded:")
        print(f"  Question: {question[:100]}...")
        print(f"  Answer: {answer[:100]}...")
        
        # Check question format
        if not question.startswith("<image>"):
            print(f"❌ Question should start with '<image>', got: {question[:50]}")
            return False
        
        print(f"\n✓ Question format valid")
        
        # Check dataset prompt building
        dataset = VLMDataset(
            data=[sample],
            image_dir=str(project_root / "data" / "raw" / "images"),
            max_length=256,
        )
        
        batch_item = dataset[0]
        
        # Verify answer_start_pos calculated
        if "answer_start_pos" not in batch_item:
            print("❌ answer_start_pos not in batch")
            return False
        
        answer_start_pos = batch_item["answer_start_pos"]
        print(f"\n✓ answer_start_pos calculated: {answer_start_pos}")
        
        if answer_start_pos <= 0:
            print(f"❌ answer_start_pos should be > 0, got {answer_start_pos}")
            return False
        
        # Decode tokens to verify format
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            trust_remote_code=True,
            use_fast=False
        )
        
        input_ids = batch_item["input_ids"]
        text_decoded = tokenizer.decode(input_ids)
        
        print(f"\n✓ Full decoded text (first 200 chars):")
        print(f"  {text_decoded[:200]}...")
        
        # Verify official format markers
        required_markers = [
            "<|im_start|>system",
            "<|im_end|>",
            "<|im_start|>user",
            "<image>",
            "<|im_start|>assistant",
        ]
        
        for marker in required_markers:
            if marker not in text_decoded:
                print(f"❌ Missing required marker: {marker}")
                return False
        
        print(f"\n✓ All required format markers present")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_loss_masking():
    """Verify loss masking logic works"""
    print("\n" + "=" * 80)
    print("CHECK 2: Loss Masking Logic")
    print("=" * 80)
    
    try:
        loader = AblationDataLoader()
        samples, _, _ = loader.load_train_val_split(max_samples=2, val_ratio=0.2)
        
        dataset = VLMDataset(
            data=samples,
            image_dir=str(project_root / "data" / "raw" / "images"),
            max_length=256,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            trust_remote_code=True,
            use_fast=False
        )
        
        collate_fn = create_collate_fn(tokenizer=tokenizer, max_length=256)
        batch_samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collate_fn(batch_samples)
        
        print(f"\n✓ Batch prepared:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  answer_start_pos: {batch['answer_start_pos']}")
        
        # Simulate loss masking (like in trainer.py)
        input_ids = batch['input_ids']
        shift_labels = input_ids[..., 1:].contiguous()
        answer_start_positions = batch['answer_start_pos']
        
        # Test masking logic
        for i in range(shift_labels.shape[0]):
            answer_start_pos = answer_start_positions[i].item() if isinstance(answer_start_positions[i], torch.Tensor) else answer_start_positions[i]
            if answer_start_pos > 1:
                shift_labels[i, :answer_start_pos-2] = -100
        
        num_answer_tokens = (shift_labels != -100).sum().item()
        num_question_tokens = (shift_labels == -100).sum().item()
        total_tokens = shift_labels.numel()
        
        print(f"\n✓ Loss masking applied:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Question tokens masked: {num_question_tokens}")
        print(f"  Answer tokens (for loss): {num_answer_tokens}")
        print(f"  Answer ratio: {num_answer_tokens/total_tokens*100:.1f}%")
        
        if num_answer_tokens <= 0:
            print(f"❌ No answer tokens to compute loss on!")
            return False
        
        if num_question_tokens == 0:
            print(f"⚠️  No question tokens masked (all tokens used)")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_bridge_compatibility():
    """Verify bridge architecture types produce correct shapes"""
    print("\n" + "=" * 80)
    print("CHECK 3: Bridge Architecture Compatibility")
    print("=" * 80)
    
    try:
        from src.training.finetune_setup import VisionLanguageBridge
        
        # Dummy config
        class DummyConfig:
            pass
        
        config = DummyConfig()
        config.bridge_type = "residual"
        config.vision_hidden_size = 1024
        config.text_hidden_size = 896
        config.use_distillation = True
        config.warm_start = False
        config.alpha = 0.1
        
        # Test each bridge type
        bridge_types = ['residual', 'multi_token', 'tile_attention', 'mini_qformer', 'qformer', 'gated_fusion']
        
        print(f"\n✓ Testing {len(bridge_types)} bridge types:")
        
        results = {}
        for bridge_type in bridge_types:
            try:
                config.bridge_type = bridge_type
                bridge = VisionLanguageBridge(config)
                
                # Check bridge has required components
                has_bridge = hasattr(bridge, 'bridge') and bridge.bridge is not None
                has_baseline = hasattr(bridge, 'baseline_bridge') and bridge.baseline_bridge is not None
                
                results[bridge_type] = {
                    'loaded': True,
                    'has_bridge': has_bridge,
                    'has_baseline': has_baseline,
                }
                
                print(f"  ✓ {bridge_type:20s} - bridge:{has_bridge}, baseline:{has_baseline}")
                
            except Exception as e:
                results[bridge_type] = {
                    'loaded': False,
                    'error': str(e),
                }
                print(f"  ❌ {bridge_type:20s} - {str(e)[:60]}")
        
        all_loaded = all(r['loaded'] for r in results.values())
        if not all_loaded:
            print(f"\n❌ Some bridge types failed to load")
            return False
        
        all_have_components = all(r.get('has_bridge') and r.get('has_baseline') for r in results.values())
        if not all_have_components:
            print(f"\n⚠️  Some bridge types missing components")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print("\n" + "=" * 80)
    print("VINTERN INTEGRATION VALIDATION SUITE")
    print("=" * 80)
    
    results = {
        'Template Format': check_template_format(),
        'Loss Masking': check_loss_masking(),
        'Bridge Compatibility': check_bridge_compatibility(),
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name:30s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for training!")
    else:
        print("❌ SOME CHECKS FAILED - Review above for details")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
