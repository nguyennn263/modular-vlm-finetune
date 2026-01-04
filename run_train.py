"""
Training Script
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent))

from src.models import VLMModel, create_vlm_model
from src.data import VLMDataset, VLMDataCollator, VinternProcessor
from src.utils import (
    VLMLogger, 
    VQAMetrics,
    CheckpointManager, 
    CheckpointCallback, 
    auto_resume,
)


@dataclass
class TrainConfig:
    """Training configuration"""
    config_path: str = "configs/config.yaml"
    output_dir: str = "outputs"
    resume: Optional[str] = None
    
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None


def load_config(path: str) -> Dict:
    """Load YAML config"""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_model(config: Dict) -> VLMModel:
    """Initialize model from config"""
    model_config = config.get("model", {})
    
    model = create_vlm_model({
        "vision_encoder_type": model_config.get("vision_encoder_type", "internvit"),
        "vision_model_name": model_config.get("vision_model_name"),
        "vision_hidden_size": model_config.get("vision_hidden_size", 1024),
        "image_size": model_config.get("image_size", 448),
        "projector_type": model_config.get("projector_type", "mlp"),
        "llm_type": model_config.get("llm_type", "qwen2-0.5b"),
        "llm_model_name": model_config.get("llm_model_name"),
        "freeze_vision": model_config.get("freeze_vision", False),
        "freeze_llm": model_config.get("freeze_llm", True),
        "use_flash_attention": model_config.get("use_flash_attention", False),
        "torch_dtype": model_config.get("torch_dtype", "float16"),
    })
    
    return model


def setup_lora(model: VLMModel, config: Dict) -> VLMModel:
    """Setup LoRA for LLM"""
    lora_config = config.get("lora", {})
    
    if not lora_config.get("enabled", False):
        return model
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 32),
        lora_alpha=lora_config.get("alpha", 64),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
    )
    
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()
    
    return model


def setup_tokenizer(config: Dict) -> AutoTokenizer:
    """Setup tokenizer"""
    from src.models.registry import ModelRegistry
    
    model_config = config.get("model", {})
    llm_type = model_config.get("llm_type", "qwen2-0.5b")
    llm_config = ModelRegistry.get_llm_config(llm_type)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("llm_model_name") or llm_config["model_name"],
        trust_remote_code=True,
    )
    
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    
    return tokenizer


def setup_datasets(config: Dict, tokenizer):
    """Setup train and eval datasets"""
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    
    from src.models.registry import ModelRegistry
    llm_config = ModelRegistry.get_llm_config(model_config.get("llm_type", "qwen2-0.5b"))
    
    common_args = {
        "tokenizer_name": model_config.get("llm_model_name") or llm_config["model_name"],
        "max_length": data_config.get("max_length", 1024),
        "image_size": model_config.get("image_size", 448),
        "max_tiles": data_config.get("max_tiles", 6),
    }
    
    train_dataset = VLMDataset(
        data_path=data_config["train_path"],
        image_dir=data_config["image_dir"],
        **common_args,
    )
    
    eval_dataset = None
    if data_config.get("val_path"):
        eval_dataset = VLMDataset(
            data_path=data_config["val_path"],
            image_dir=data_config["image_dir"],
            **common_args,
        )
    
    return train_dataset, eval_dataset


def create_training_args(config: Dict, output_dir: str, overrides: Dict = None) -> TrainingArguments:
    """Create TrainingArguments"""
    train_config = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})
    overrides = overrides or {}
    
    save_strategy = checkpoint_config.get("save_strategy", "steps")
    save_steps = checkpoint_config.get("save_steps", train_config.get("save_steps", 100))
    save_total_limit = checkpoint_config.get("save_total_limit", config.get("output", {}).get("save_total_limit", 3))
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training
        num_train_epochs=overrides.get("epochs") or train_config.get("epochs", 3),
        per_device_train_batch_size=overrides.get("batch_size") or train_config.get("batch_size", 1),
        per_device_eval_batch_size=train_config.get("batch_size", 1),
        gradient_accumulation_steps=train_config.get("gradient_accumulation", 8),
        
        # Optimizer
        learning_rate=overrides.get("learning_rate") or train_config.get("learning_rate", 2e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_config.get("scheduler", "cosine"),
        
        # Memory
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        fp16=train_config.get("mixed_precision") == "fp16",
        bf16=train_config.get("mixed_precision") == "bf16",
        
        # Logging
        logging_steps=train_config.get("logging_steps", 10),
        eval_strategy="steps" if config.get("data", {}).get("val_path") else "no",
        eval_steps=train_config.get("eval_steps", 100),
        
        # Checkpointing
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=save_total_limit,
        save_safetensors=True,
        
        # Best model
        load_best_model_at_end=checkpoint_config.get("save_best", True) if config.get("data", {}).get("val_path") else False,
        metric_for_best_model=checkpoint_config.get("metric_for_best", "eval_loss"),
        greater_is_better=checkpoint_config.get("greater_is_better", False),
        
        # W&B
        report_to="wandb" if train_config.get("use_wandb", False) else "none",
        
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )


class VLMTrainer(Trainer):
    """Custom Trainer for VLM với logging support"""
    
    def __init__(self, *args, vlm_logger: Optional[VLMLogger] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm_logger = vlm_logger
        self.metrics_tracker = VQAMetrics()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss với label masking"""
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log để gửi lên W&B"""
        super().log(logs)
        if self.vlm_logger:
            self.vlm_logger.log_metrics(logs, step=self.state.global_step)


def main():
    parser = argparse.ArgumentParser(description="VLM Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Resume from checkpoint: 'auto', 'latest', 'best', or path")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    output_dir = args.output_dir or config.get("output", {}).get("dir", "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_config = config.get("checkpoint", {})
    
    resume_checkpoint = args.resume or checkpoint_config.get("resume_from_checkpoint")
    if resume_checkpoint:
        resume_checkpoint = auto_resume(output_dir, resume_checkpoint)
    
    print("VLM Fine-tuning....")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        save_total_limit=checkpoint_config.get("save_total_limit", 3),
        save_best=checkpoint_config.get("save_best", True),
        metric_for_best=checkpoint_config.get("metric_for_best", "eval_loss"),
        greater_is_better=checkpoint_config.get("greater_is_better", False),
        save_optimizer=checkpoint_config.get("save_optimizer", True),
        save_scheduler=checkpoint_config.get("save_scheduler", True),
    )
    
    # Setup logger (W&B)
    use_wandb = config.get("training", {}).get("use_wandb", False)
    vlm_logger = None
    if use_wandb:
        vlm_logger = VLMLogger(
            project_name="vlm-finetune",
            config=config,
            use_wandb=True,
        )
    
    print("\n[1/5] Loading tokenizer...")
    tokenizer = setup_tokenizer(config)
    
    print("\n[2/5] Loading model...")
    model = setup_model(config)
    
    model.llm.resize_token_embeddings(len(tokenizer))
    
    print("\n[3/5] Setting up LoRA...")
    model = setup_lora(model, config)
    
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    print("\n[4/5] Loading datasets...")
    train_dataset, eval_dataset = setup_datasets(config, tokenizer)
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator
    from src.models.registry import ModelRegistry
    llm_config = ModelRegistry.get_llm_config(config.get("model", {}).get("llm_type", "qwen2-0.5b"))
    
    data_collator = VLMDataCollator(
        tokenizer=tokenizer,
        image_token_id=llm_config.get("image_token_id", 151667),
    )
    
    # Training arguments
    training_args = create_training_args(
        config, 
        output_dir,
        overrides={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        }
    )
    
    callbacks = []
    
    # Checkpoint callback
    is_lora = config.get("lora", {}).get("enabled", False)
    checkpoint_callback = CheckpointCallback(checkpoint_manager, is_lora=is_lora)
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback (optional)
    if config.get("training", {}).get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["training"]["early_stopping_patience"]
            )
        )
    
    print("\n[5/5] Starting training...")
    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        vlm_logger=vlm_logger,
    )
    
    if resume_checkpoint:
        print(f"\nResuming training from: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    
    print("\nSaving final model...")
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
    
    print("=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Final model saved to: {final_dir}")
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
    
    checkpoints = checkpoint_manager.list_checkpoints()
    if checkpoints:
        print(f"Available checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints[-3:]:  # Show last 3
            print(f"- Step {ckpt['step']}: loss={ckpt.get('loss', 'N/A'):.4f}" if ckpt.get('loss') else f"  - Step {ckpt['step']}")
    
    if vlm_logger:
        vlm_logger.finish()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
