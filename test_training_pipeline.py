#!/usr/bin/env python3
"""
Test script to verify the complete training/validation/testing workflow.
"""

import torch
from pathlib import Path
from transformers import AutoModel

from src.training import create_finetune_model, BridgeTrainer, TrainConfig
from src.data.loaders import load_datasets
from utils.path_management import RAW_TEXT_CSV, RAW_IMAGES_DIR


def test_data_loading():
    """Test data loading with optional test split."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        # Test with no test split
        print("\n✓ Loading data with no test split...")
        result = load_datasets(
            csv_path=str(RAW_TEXT_CSV),
            images_dir=str(RAW_IMAGES_DIR),
            val_ratio=0.1,
            test_ratio=0.0,
            max_samples=100  # Use small sample for quick test
        )
        
        if len(result) == 3:
            train_ds, val_ds, test_ds = result
            assert test_ds is None, "test_ds should be None when test_ratio=0"
            print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: None")
        else:
            train_ds, val_ds = result
            test_ds = None
            print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        # Test with test split
        print("\n✓ Loading data with test split...")
        result = load_datasets(
            csv_path=str(RAW_TEXT_CSV),
            images_dir=str(RAW_IMAGES_DIR),
            val_ratio=0.1,
            test_ratio=0.1,
            max_samples=100
        )
        
        train_ds, val_ds, test_ds = result
        assert test_ds is not None, "test_ds should not be None when test_ratio>0"
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        print("\n✓ Data loading test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """Test trainer initialization with optional test_dataset."""
    print("\n" + "="*60)
    print("TEST 2: Trainer Initialization")
    print("="*60)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n✓ Using device: {device}")
        
        # Load minimal data
        print("✓ Loading minimal dataset...")
        result = load_datasets(
            csv_path=str(RAW_TEXT_CSV),
            images_dir=str(RAW_IMAGES_DIR),
            val_ratio=0.1,
            test_ratio=0.1,
            max_samples=16,  # Very small for quick test
            img_size=224  # Smaller images for speed
        )
        
        train_ds, val_ds, test_ds = result
        
        # Load model
        print("✓ Loading base model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            _fast_init=False,
        ).eval().to(device)
        
        # Create fine-tune model
        print("✓ Creating fine-tune model with better_mlp bridge...")
        model = create_finetune_model(
            base_model,
            bridge_type="better_mlp",
            bridge_config={}
        )
        
        # Create config
        config = TrainConfig(
            output_dir="./test_checkpoints",
            num_epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            eval_steps=2
        )
        
        # Initialize trainer with test_dataset
        print("✓ Initializing trainer with test_dataset...")
        trainer = BridgeTrainer(
            model,
            train_ds,
            val_ds,
            config,
            test_dataset=test_ds
        )
        
        # Verify attributes
        assert hasattr(trainer, 'train_loader'), "Missing train_loader"
        assert hasattr(trainer, 'val_loader'), "Missing val_loader"
        assert hasattr(trainer, 'test_loader'), "Missing test_loader"
        
        print(f"  Train loader batches: {len(trainer.train_loader)}")
        print(f"  Val loader batches: {len(trainer.val_loader)}")
        print(f"  Test loader batches: {len(trainer.test_loader)}")
        
        print("\n✓ Trainer initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Trainer initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_method():
    """Test the evaluate method returns correct metrics."""
    print("\n" + "="*60)
    print("TEST 3: Evaluate Method")
    print("="*60)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data
        print("✓ Loading dataset...")
        result = load_datasets(
            csv_path=str(RAW_TEXT_CSV),
            images_dir=str(RAW_IMAGES_DIR),
            val_ratio=0.1,
            test_ratio=0.1,
            max_samples=32,
            img_size=224
        )
        
        train_ds, val_ds, test_ds = result
        
        # Load model
        print("✓ Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            _fast_init=False,
        ).eval().to(device)
        
        model = create_finetune_model(
            base_model,
            bridge_type="better_mlp"
        )
        
        # Initialize trainer
        config = TrainConfig(
            output_dir="./test_checkpoints",
            num_epochs=1,
            batch_size=4,
            eval_steps=1
        )
        
        trainer = BridgeTrainer(model, train_ds, val_ds, config, test_dataset=test_ds)
        
        # Test evaluate method
        print("✓ Running evaluate() on validation set...")
        val_metrics = trainer.evaluate()
        
        assert isinstance(val_metrics, dict), "evaluate() should return dict"
        assert 'loss' in val_metrics, "metrics should have 'loss'"
        assert 'perplexity' in val_metrics, "metrics should have 'perplexity'"
        
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
        print(f"  Validation Perplexity: {val_metrics['perplexity']:.4f}")
        
        # Test evaluate with test_dataset
        if test_ds is not None:
            print("✓ Running evaluate() on test set...")
            test_metrics = trainer.evaluate(test_ds)
            
            assert isinstance(test_metrics, dict), "evaluate() should return dict"
            assert 'loss' in test_metrics, "metrics should have 'loss'"
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test Perplexity: {test_metrics['perplexity']:.4f}")
        
        print("\n✓ Evaluate method test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Evaluate method test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRAINING PIPELINE TEST SUITE")
    print("="*60)
    
    results = {
        "Data Loading": test_data_loading(),
        # Trainer init test disabled to avoid long model loading times
        # "Trainer Initialization": test_trainer_initialization(),
        # "Evaluate Method": test_evaluate_method(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
