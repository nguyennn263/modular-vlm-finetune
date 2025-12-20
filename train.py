"""
VietVLM-Finetune: Training Script
Fine-tune Vision-Language Model cho tiếng Việt với LoRA/QLoRA
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from src.models import VinternVLM
from src.data import VLMDataset, VLMDataCollator
from src.utils import VLMLogger, VQAMetrics


def load_config(config_path: str) -> Dict:
    """Load YAML config"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_lora(model: VinternVLM, lora_config: Dict) -> VinternVLM:
    """Setup LoRA cho LLM component"""
    if not lora_config.get("enabled", False):
        return model
    
    # Cấu hình LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
    )
    
    # Apply LoRA chỉ cho LLM
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()
    
    return model


class VLMTrainer(Trainer):
    """Custom Trainer cho VLM với W&B logging"""
    
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


def create_training_args(config: Dict, output_dir: str) -> TrainingArguments:
    """Tạo TrainingArguments từ config"""
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training params
        num_train_epochs=training_config.get("epochs", 3),
        per_device_train_batch_size=data_config.get("train_batch_size", 4),
        per_device_eval_batch_size=data_config.get("eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation", 4),
        
        # Optimization
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("scheduler", "cosine"),
        
        # Memory optimization
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        bf16=training_config.get("bf16", True),
        
        # Logging
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=3,
        
        # W&B
        report_to="wandb" if training_config.get("use_wandb", True) else "none",
        
        # Others
        dataloader_num_workers=data_config.get("num_workers", 4),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )


def main():
    parser = argparse.ArgumentParser(description="VietVLM Fine-tuning")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configs
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    config = {**model_config, **data_config}
    
    print("=" * 50)
    print("VietVLM-Finetune Training")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VLMLogger(
        project_name="vietvlm-finetune",
        config=config,
        use_wandb=config.get("training", {}).get("use_wandb", True),
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"]["llm"]["model_name"],
        trust_remote_code=True,
    )
    
    # Thêm special tokens nếu cần
    special_tokens = model_config.get("special_tokens", {})
    if special_tokens.get("image_token"):
        tokenizer.add_special_tokens({
            "additional_special_tokens": [special_tokens["image_token"]]
        })
    
    # Initialize model
    print("\nLoading model...")
    model = VinternVLM(
        vision_model_name=model_config["model"]["vision_tower"]["model_name"],
        llm_model_name=model_config["model"]["llm"]["model_name"],
        projector_input_dim=model_config["model"]["projector"]["input_dim"],
        projector_output_dim=model_config["model"]["projector"]["output_dim"],
        freeze_vision=model_config["model"]["vision_tower"]["freeze"],
        image_token_id=special_tokens.get("image_token_id", 151667),
    )
    
    # Resize embeddings nếu thêm tokens
    model.llm.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA
    model = setup_lora(model, model_config.get("lora", {}))
    
    # Enable gradient checkpointing
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VLMDataset(
        data_path=data_config["data"]["train_path"],
        image_dir=data_config["data"]["image_dir"],
        tokenizer_name=model_config["model"]["llm"]["model_name"],
        max_length=model_config["model"]["llm"]["max_length"],
        image_size=model_config["model"]["vision_tower"]["image_size"],
        max_tiles=model_config["model"]["dynamic_resolution"]["max_tiles"],
    )
    
    eval_dataset = VLMDataset(
        data_path=data_config["data"]["val_path"],
        image_dir=data_config["data"]["image_dir"],
        tokenizer_name=model_config["model"]["llm"]["model_name"],
        max_length=model_config["model"]["llm"]["max_length"],
        image_size=model_config["model"]["vision_tower"]["image_size"],
        max_tiles=model_config["model"]["dynamic_resolution"]["max_tiles"],
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = VLMDataCollator(
        tokenizer=tokenizer,
        image_token_id=special_tokens.get("image_token_id", 151667),
    )
    
    # Training arguments
    training_args = create_training_args(config, str(output_dir))
    
    # Initialize trainer
    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        vlm_logger=logger,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Training
    print("\nStarting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    # Cleanup
    logger.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
