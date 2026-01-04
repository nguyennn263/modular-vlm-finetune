# CELL 1: Setup
"""
!pip install -q transformers>=4.40.0 peft>=0.10.0 accelerate>=0.27.0 pillow pyyaml

# Clone repo (or upload as dataset)
!git clone https://github.com/YOUR_USERNAME/VLM-Benchmark.git
%cd VLM-Benchmark
"""

# CELL 2: Quick Test
import sys
sys.path.insert(0, ".")

import torch
from src.models import VLMModel, create_vlm_model, ModelRegistry

# Check available models
print("Available LLMs:", ModelRegistry.list_llms())
print("Available Vision Encoders:", ModelRegistry.list_vision_encoders())
print("Available Projectors:", ModelRegistry.list_projectors())

# Test create model
print("\nCreating model...")
model = create_vlm_model({
    "vision_encoder_type": "internvit",
    "vision_model_name": "OpenGVLab/InternViT-300M-448px",
    "llm_type": "qwen2-0.5b",
    "projector_type": "mlp",
    "freeze_vision": False,
    "freeze_llm": True,
    "torch_dtype": "float16",
})

print(f"✓ Model created successfully!")
print(f"  Vision encoder: {type(model.vision_encoder).__name__}")
print(f"  Projector: {type(model.projector).__name__}")
print(f"  LLM: {type(model.llm).__name__}")

# Check VRAM
if torch.cuda.is_available():
    model = model.cuda()
    print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# CELL 3: Prepare Data
"""
# Upload your data to Kaggle as a dataset, then:
!mkdir -p data/images
!cp -r /kaggle/input/your-dataset/images/* data/images/
!cp /kaggle/input/your-dataset/train.json data/
!cp /kaggle/input/your-dataset/val.json data/
"""

# CELL 4: Training
# !python run_train.py --config configs/kaggle_t4.yaml

def train_on_kaggle():
    import yaml
    from pathlib import Path
    from transformers import AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    
    from src.models import create_vlm_model, ModelRegistry
    from src.data import VLMDataset, VLMDataCollator
    
    config = yaml.safe_load(open("configs/kaggle_t4.yaml"))
    
    model = create_vlm_model(config["model"])
    
    if config["lora"]["enabled"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
        )
        model.llm = get_peft_model(model.llm, peft_config)
    
    llm_config = ModelRegistry.get_llm_config(config["model"]["llm_type"])
    tokenizer = AutoTokenizer.from_pretrained(llm_config["model_name"], trust_remote_code=True)
    
    train_dataset = VLMDataset(
        data_path=config["data"]["train_path"],
        image_dir=config["data"]["image_dir"],
        tokenizer_name=llm_config["model_name"],
        max_length=config["data"]["max_length"],
        max_tiles=config["data"]["max_tiles"],
    )
    
    collator = VLMDataCollator(tokenizer=tokenizer, image_token_id=llm_config["image_token_id"])
    
    args = TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        learning_rate=config["training"]["learning_rate"],
        gradient_checkpointing=True,
        fp16=True,
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(f"{config['output']['dir']}/final")
    
    print("✓ Training complete!")

# train_on_kaggle()
