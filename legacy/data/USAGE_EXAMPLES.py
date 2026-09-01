"""
Data loading utilities and examples.
Aligned with ref project's data loading structure.
"""

# Example 1: Load dataset using JSON path (simplest way)
from src.data.dataset import VLMDataset
from torch.utils.data import DataLoader

# Load directly from JSON file
dataset = VLMDataset(
    data="data/train.json",  # or path to any JSON file
    image_dir="data/raw/images"
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Example 2: Load using OneSample objects (for more control)
from src.data.data_actions import load_dataset_from_json

# Load data with helper function
train_samples = load_dataset_from_json(
    json_path="data/train.json",
    image_dir="data/raw/images"
)

val_samples = load_dataset_from_json(
    json_path="data/val.json",
    image_dir="data/raw/images"
)

# Create datasets from OneSample objects
train_dataset = VLMDataset(
    data=train_samples,  # Pass list of OneSample objects
    image_dir="data/raw/images"
)

val_dataset = VLMDataset(
    data=val_samples,
    image_dir="data/raw/images"
)


# Example 3: Load both train and val in one call
from src.data.data_actions import load_train_val_datasets

train_samples, val_samples = load_train_val_datasets(
    data_dir="data",
    image_dir="data/raw/images",
)

# Example 4: ConversationDataset for multi-turn conversations
from src.data.dataset import ConversationDataset

conversation_dataset = ConversationDataset(
    data="data/conversations.json",
    image_dir="data/raw/images"
)


# JSON Format Expected:
# For VLMDataset, train.json should have format:
# [
#     {
#         "image": "image_filename.jpg",
#         "question": "What is in this image?",
#         "answer": "This is a test image."
#     },
#     ...
# ]

# For ConversationDataset, conversations.json format:
# [
#     {
#         "image": "image_filename.jpg",
#         "conversations": [
#             {
#                 "role": "user",  # or "from": "human"
#                 "content": "<image>\nWhat is this?"
#             },
#             {
#                 "role": "assistant",  # or "from": "gpt"
#                 "content": "This is an image with..."
#             }
#         ]
#     },
#     ...
# ]

