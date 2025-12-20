"""
VLM Data Collator với Label Masking
Xử lý padding và masking loss cho multimodal data
"""
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class VLMDataCollator:
    """
    Data Collator cho Vision-Language Model
    - Padding multimodal inputs
    - Label Masking: chỉ tính loss trên phần Assistant response
    """
    
    tokenizer: PreTrainedTokenizer
    image_token_id: int = 151667
    ignore_index: int = -100
    pad_to_multiple_of: Optional[int] = 8
    
    # Markers để xác định vùng assistant response
    assistant_start: str = "<|im_start|>assistant"
    assistant_end: str = "<|im_end|>"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch với label masking"""
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
            "num_patches": [],
        }
        
        # Process từng sample
        for feat in features:
            input_ids = feat["input_ids"]
            attention_mask = feat["attention_mask"]
            
            # Tạo labels với masking
            labels = self._create_labels_with_masking(input_ids)
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
            
            if "pixel_values" in feat:
                batch["pixel_values"].append(feat["pixel_values"])
                batch["num_patches"].append(feat["num_patches"])
        
        # Pad text inputs
        batch = self._pad_sequences(batch)
        
        # Stack pixel values nếu có
        if batch["pixel_values"]:
            batch["pixel_values"] = self._pad_pixel_values(batch["pixel_values"])
        else:
            del batch["pixel_values"]
            del batch["num_patches"]
        
        return batch
    
    def _create_labels_with_masking(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Tạo labels với masking:
        - Set -100 cho prompt và image tokens
        - Chỉ giữ loss cho phần assistant response
        """
        labels = input_ids.clone()
        
        # Decode để tìm vị trí assistant
        text = self.tokenizer.decode(input_ids)
        
        # Tìm tất cả các vùng assistant
        assistant_ranges = self._find_assistant_ranges(text, input_ids)
        
        # Mặc định mask tất cả
        labels[:] = self.ignore_index
        
        # Unmask chỉ phần assistant response
        for start, end in assistant_ranges:
            labels[start:end] = input_ids[start:end]
        
        # Mask image tokens (nếu có)
        image_mask = input_ids == self.image_token_id
        labels[image_mask] = self.ignore_index
        
        return labels
    
    def _find_assistant_ranges(
        self, 
        text: str, 
        input_ids: torch.Tensor
    ) -> List[tuple]:
        """Tìm các vùng assistant response trong text"""
        ranges = []
        
        # Tìm tất cả positions của assistant markers
        start_marker = self.assistant_start
        end_marker = self.assistant_end
        
        pos = 0
        while True:
            # Tìm bắt đầu của assistant response
            start_pos = text.find(start_marker, pos)
            if start_pos == -1:
                break
            
            # Tìm kết thúc (sau newline của assistant header)
            content_start = text.find("\n", start_pos) + 1
            
            # Tìm end marker
            end_pos = text.find(end_marker, content_start)
            if end_pos == -1:
                end_pos = len(text)
            
            # Convert character positions sang token positions
            prefix_tokens = len(self.tokenizer.encode(
                text[:content_start], add_special_tokens=False
            ))
            content_tokens = len(self.tokenizer.encode(
                text[:end_pos], add_special_tokens=False
            ))
            
            ranges.append((prefix_tokens, content_tokens))
            pos = end_pos + len(end_marker)
        
        return ranges
    
    def _pad_sequences(self, batch: Dict) -> Dict:
        """Pad input_ids, attention_mask, labels"""
        
        # Tìm max length
        max_len = max(len(ids) for ids in batch["input_ids"])
        
        # Round up nếu cần
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        padded_input_ids = []
        padded_attention = []
        padded_labels = []
        
        pad_id = self.tokenizer.pad_token_id or 0
        
        for input_ids, attn, labels in zip(
            batch["input_ids"], 
            batch["attention_mask"], 
            batch["labels"]
        ):
            pad_len = max_len - len(input_ids)
            
            # Pad bên phải
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            )
            padded_attention.append(
                torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((pad_len,), self.ignore_index, dtype=labels.dtype)])
            )
        
        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention)
        batch["labels"] = torch.stack(padded_labels)
        
        return batch
    
    def _pad_pixel_values(self, pixel_values_list: List[torch.Tensor]) -> torch.Tensor:
        """Pad pixel values về cùng số patches"""
        max_patches = max(pv.shape[0] for pv in pixel_values_list)
        
        padded = []
        for pv in pixel_values_list:
            if pv.shape[0] < max_patches:
                pad_size = max_patches - pv.shape[0]
                pad = torch.zeros(
                    pad_size, *pv.shape[1:], dtype=pv.dtype
                )
                pv = torch.cat([pv, pad], dim=0)
            padded.append(pv)
        
        return torch.stack(padded)


def create_label_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    assistant_token: str = "assistant",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Utility function để tạo label mask
    Mask tất cả tokens trước và bao gồm assistant marker
    """
    labels = input_ids.clone()
    
    # Encode assistant token
    assistant_ids = tokenizer.encode(assistant_token, add_special_tokens=False)
    
    # Tìm vị trí của assistant token
    seq_len = len(input_ids)
    assistant_len = len(assistant_ids)
    
    found = False
    for i in range(seq_len - assistant_len + 1):
        if input_ids[i:i+assistant_len].tolist() == assistant_ids:
            # Mask từ đầu đến hết assistant marker + newline
            labels[:i+assistant_len+1] = ignore_index
            found = True
            break
    
    if not found:
        # Không tìm thấy assistant, mask tất cả
        labels[:] = ignore_index
    
    return labels
