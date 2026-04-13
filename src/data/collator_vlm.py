"""
Collate function for VLMDataset.
Handles batching of items that already have input_ids, attention_mask, etc.
"""
import torch
from typing import List, Dict, Any


def collate_vlm_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for VLMDataset batches.
    
    Each item in batch is a dict with:
    - input_ids: [seq_len]
    - attention_mask: [seq_len]
    - pixel_values: [H, W]
    - num_patches: int
    - question: str
    - answer: str
    - answer_start_pos: int
    
    Returns:
    - Batched tensors with padding
    """
    max_seq_len = max(item['input_ids'].shape[0] for item in batch)
    
    input_ids_padded = []
    attention_mask_padded = []
    pixel_values_list = []
    answer_start_pos_list = []
    questions = []
    answers = []
    
    for item in batch:
        # Pad input_ids and attention_mask to max_seq_len
        seq_len = item['input_ids'].shape[0]
        pad_len = max_seq_len - seq_len
        
        if pad_len > 0:
            input_ids_padded.append(
                torch.cat([item['input_ids'], torch.full((pad_len,), fill_value=0, dtype=item['input_ids'].dtype)])
            )
            attention_mask_padded.append(
                torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=item['attention_mask'].dtype)])
            )
        else:
            input_ids_padded.append(item['input_ids'])
            attention_mask_padded.append(item['attention_mask'])
        
        pixel_values_list.append(item['pixel_values'])
        answer_start_pos_list.append(item['answer_start_pos'])
        questions.append(item['question'])
        answers.append(item['answer'])
    
    return {
        'input_ids': torch.stack(input_ids_padded),
        'attention_mask': torch.stack(attention_mask_padded),
        'pixel_values': torch.stack(pixel_values_list),
        'answer_start_pos': torch.tensor(answer_start_pos_list, dtype=torch.long),
        'question': questions,
        'answer': answers,
    }
