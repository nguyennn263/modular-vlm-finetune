"""
VinternVLM: Kiến trúc chính cho Vision-Language Model
Kết hợp Vision Tower + Projector + LLM
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoConfig

from .vision_tower import VisionTower
from .projector import MLPProjector


class VinternVLM(nn.Module):
    """
    Vision-Language Model architecture theo InternVL2
    Components:
        - Vision Tower: InternViT để encode ảnh
        - Projector: 2-layer MLP để align vision/text spaces
        - LLM: Qwen2/BartPho cho text generation
    """
    
    def __init__(
        self,
        vision_model_name: str = "OpenGVLab/InternViT-300M-448px",
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        projector_input_dim: int = 1024,
        projector_output_dim: int = 2048,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        use_flash_attention: bool = True,
        image_token_id: int = 151667,
    ):
        super().__init__()
        
        self.image_token_id = image_token_id
        
        # Vision Tower
        self.vision_tower = VisionTower(
            model_name=vision_model_name,
            freeze=freeze_vision,
            use_flash_attention=use_flash_attention,
        )
        
        # MLP Projector
        self.projector = MLPProjector(
            input_dim=projector_input_dim,
            output_dim=projector_output_dim,
        )
        
        # LLM
        llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        if use_flash_attention:
            llm_config._attn_implementation = "flash_attention_2"
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            config=llm_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        if freeze_llm:
            self._freeze_llm()
    
    def _freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def get_vision_embeddings(
        self, 
        pixel_values: torch.Tensor,
        num_patches: List[int],
    ) -> List[torch.Tensor]:
        """
        Encode images và project sang LLM space
        Returns list of embeddings cho từng sample trong batch
        """
        # Vision encoding
        vision_features = self.vision_tower(pixel_values)
        
        # Project sang LLM space
        vision_embeds = self.projector(vision_features)
        
        # Split theo num_patches cho từng sample
        embeddings_list = []
        idx = 0
        for n in num_patches:
            # Tính số tokens thực tế
            seq_len = vision_embeds.shape[1] // pixel_values.shape[1]
            end_idx = idx + n * seq_len
            embeddings_list.append(vision_embeds[0, idx:end_idx])
            idx = end_idx
        
        return embeddings_list
    
    def merge_vision_text_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Chèn vision embeddings vào vị trí <image> token trong text
        """
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is None:
            return text_embeds
        
        B = input_ids.shape[0]
        
        # Get vision embeddings
        vision_embeds = self.vision_tower(pixel_values)
        vision_embeds = self.projector(vision_embeds)
        
        # Chèn vision vào vị trí <image>
        merged_embeds = []
        for b in range(B):
            # Tìm vị trí image token
            image_mask = input_ids[b] == self.image_token_id
            image_positions = torch.where(image_mask)[0]
            
            if len(image_positions) == 0:
                merged_embeds.append(text_embeds[b])
                continue
            
            # Số patches cho sample này
            n_patches = num_patches[b] if num_patches else pixel_values.shape[1]
            seq_len_per_patch = vision_embeds.shape[1] // pixel_values.shape[1]
            total_vision_tokens = n_patches * seq_len_per_patch
            
            # Lấy vision embedding cho sample này
            vis_emb = vision_embeds[b, :total_vision_tokens]
            
            # Build merged sequence
            parts = []
            prev_pos = 0
            
            for i, pos in enumerate(image_positions):
                # Text trước image token
                if pos > prev_pos:
                    parts.append(text_embeds[b, prev_pos:pos])
                
                # Chèn vision embeddings (thay thế cho 1 image token)
                if i == 0:  # Chỉ chèn vision một lần
                    parts.append(vis_emb)
                
                prev_pos = pos + 1
            
            # Text còn lại sau image tokens
            if prev_pos < text_embeds.shape[1]:
                parts.append(text_embeds[b, prev_pos:])
            
            merged_embeds.append(torch.cat(parts, dim=0))
        
        # Pad về cùng length
        max_len = max(e.shape[0] for e in merged_embeds)
        padded = torch.zeros(
            B, max_len, text_embeds.shape[-1],
            dtype=text_embeds.dtype, device=text_embeds.device
        )
        for b, emb in enumerate(merged_embeds):
            padded[b, :emb.shape[0]] = emb
        
        return padded
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass cho training và inference
        """
        # Merge vision + text embeddings
        inputs_embeds = self.merge_vision_text_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            num_patches=num_patches,
        )
        
        # Điều chỉnh attention mask nếu cần
        if attention_mask is not None and inputs_embeds.shape[1] != attention_mask.shape[1]:
            new_attn = torch.ones(
                inputs_embeds.shape[:2],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            new_attn[:, :attention_mask.shape[1]] = attention_mask
            attention_mask = new_attn
        
        # Điều chỉnh labels nếu cần
        if labels is not None and inputs_embeds.shape[1] != labels.shape[1]:
            new_labels = torch.full(
                inputs_embeds.shape[:2],
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            new_labels[:, :labels.shape[1]] = labels
            labels = new_labels
        
        # Forward qua LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches: Optional[List[int]] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """Generate text response"""
        inputs_embeds = self.merge_vision_text_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            num_patches=num_patches,
        )
        
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing cho memory efficiency"""
        self.llm.gradient_checkpointing_enable()
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load pretrained model"""
        import json
        config_path = f"{path}/config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        model = cls(**config, **kwargs)
        
        # Load weights
        state_dict = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model"""
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        config = {
            "image_token_id": self.image_token_id,
        }
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)
        
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
