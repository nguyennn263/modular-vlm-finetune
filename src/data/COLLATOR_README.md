# Custom Collate Function for OneSample Objects

This module provides utilities for batching OneSample objects with PyTorch DataLoaders.

## Key Components

1. **custom_collate_fn()** - Core function that:
   - Loads images from image_path and stacks them into (B, 3, H, W) tensor
   - Tokenizes questions and answers (e.g., "Question: {q}\nAnswer: {a}")
   - Returns structured batch dict with pixel_values, input_ids, attention_mask

2. **create_collate_fn()** - Factory function for creating collate_fn with tokenizer

3. **OneSampleDataset** - PyTorch Dataset wrapper for lists of OneSample objects

4. **load_image()** - Helper to load and normalize images


## Usage with Trainer (Automatic) ✓

The BridgeTrainer automatically handles tokenizer loading and collate_fn setup:

```python
from src.training.trainer import BridgeTrainer, TrainConfig
from src.data.onesample_dataset import OneSampleDataset

# The trainer automatically:
# 1. Loads tokenizer from config.model_name using AutoTokenizer.from_pretrained()
# 2. Detects OneSample objects in datasets
# 3. Creates and applies the custom collate_fn
# 4. Passes data to forward_pass as pixel_values, input_ids, attention_mask

config = TrainConfig(
    model_name="5CD-AI/Vintern-1B-v3_5",  # Auto-loads tokenizer from this
    batch_size=8,
    num_epochs=10,
    # ... other settings
)

train_dataset = OneSampleDataset(train_samples)
val_dataset = OneSampleDataset(val_samples)

trainer = BridgeTrainer(model, train_dataset, val_dataset, config)
trainer.train()  # Everything works automatically!
```


## Manual Usage (Advanced)

If you need to use DataLoader directly without the trainer:

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.onesample_dataset import OneSampleDataset
from src.data.collator_onesample import create_collate_fn

# Load tokenizer (same as in notebook)
tokenizer = AutoTokenizer.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    trust_remote_code=True,
    use_fast=False
)

# Create dataset
train_dataset = OneSampleDataset(train_samples)

# Create collate function
collate_fn = create_collate_fn(tokenizer=tokenizer, image_size=(336, 336))

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

# Iterate
for batch in train_loader:
    images = batch['pixel_values']      # (B, 3, 336, 336)
    input_ids = batch['input_ids']      # Tokenized questions+answers
    attn = batch['attention_mask']      # Attention weights
```


## Batch Output Format

The collate_fn returns:

```python
{
    'pixel_values': torch.Tensor,           # (batch_size, 3, 336, 336)
    'input_ids': torch.Tensor,              # Tokenized text IDs
    'attention_mask': torch.Tensor,         # Attention mask for text
}
```

Text format: `"Question: {question}\nAnswer: {answer}"`


## Image Processing

- **Source**: image_path from OneSample
- **Resize**: 336x336 (configurable)
- **Normalize**: [0, 1] range
- **Format**: (3, H, W) tensor
- **Fallback**: Black image on load errors


## Integration with Notebook Pattern

This matches the notebook's tokenizer loading pattern:

**Notebook:**
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False
)
```

**Trainer:**
```python
tokenizer = AutoTokenizer.from_pretrained(
    self.config.model_name,  # Same parameters
    trust_remote_code=True,
    use_fast=False
)
```

The collate_fn then handles the rest automatically.
