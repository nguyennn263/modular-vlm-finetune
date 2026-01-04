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
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        
        self.vision_tower = VisionTower(
            model_name=vision_model_name,
            freeze=freeze_vision,
            use_flash_attention=use_flash_attention,
        )
        
        self.projector = MLPProjector(
            input_dim=projector_input_dim,
            output_dim=projector_output_dim,
        )
        
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
        """Freeze LLM parameters"""
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def encode_images(
        self, 
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to embeddings
        """
        B, N, C, H, W = pixel_values.shape
        
        pixel_values_flat = pixel_values.view(B * N, C, H, W)
        
        vision_features = self.vision_tower(pixel_values_flat)  # (B*N, seq_len, vision_dim)
        
        vision_embeds = self.projector(vision_features)  # (B*N, seq_len, llm_dim)
        
        # Reshape back to (B, N*seq_len, llm_dim)
        seq_len = vision_embeds.shape[1]
        vision_embeds = vision_embeds.view(B, N * seq_len, -1)
        
        return vision_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for training and inference
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, llm_dim)
        
        if pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)  # (B, N*seq_len, llm_dim)
            
            inputs_embeds = self._merge_embeddings(
                input_ids, text_embeds, vision_embeds
            )
            
            if inputs_embeds.shape[1] != attention_mask.shape[1]:
                new_attn_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                orig_len = min(attention_mask.shape[1], inputs_embeds.shape[1])
                new_attn_mask[:, :orig_len] = attention_mask[:, :orig_len]
                attention_mask = new_attn_mask
            
            if labels is not None and inputs_embeds.shape[1] != labels.shape[1]:
                new_labels = torch.full(
                    inputs_embeds.shape[:2],
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                orig_len = min(labels.shape[1], inputs_embeds.shape[1])
                new_labels[:, :orig_len] = labels[:, :orig_len]
                labels = new_labels
        else:
            inputs_embeds = text_embeds
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        return outputs
    
    def _merge_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge vision embeddings into text embeddings at <image> token positions
            input_ids: (B, T) input token IDs
            text_embeds: (B, T, llm_dim) text embeddings
            vision_embeds: (B, N*seq_len, llm_dim) vision embeddings
        
        Returns:
            merged: (B, T', llm_dim) merged embeddings where T' may differ from T
        """
        B = input_ids.shape[0]
        batch_embeds = []
        
        for b in range(B):
            image_mask = input_ids[b] == self.image_token_id
            image_positions = torch.where(image_mask)[0]
            
            if len(image_positions) == 0:
                batch_embeds.append(text_embeds[b])
                continue
            
            parts = []
            prev_pos = 0
            
            for i, pos in enumerate(image_positions):
                if pos > prev_pos:
                    parts.append(text_embeds[b, prev_pos:pos])
                
                if i == 0:  
                    parts.append(vision_embeds[b])
                
                prev_pos = pos + 1
            
            if prev_pos < text_embeds.shape[1]:
                parts.append(text_embeds[b, prev_pos:])
            
            merged = torch.cat(parts, dim=0)
            batch_embeds.append(merged)
        
        max_len = max(e.shape[0] for e in batch_embeds)
        padded = torch.zeros(
            B, max_len, text_embeds.shape[-1],
            dtype=text_embeds.dtype,
            device=text_embeds.device,
        )
        
        for b, emb in enumerate(batch_embeds):
            padded[b, :emb.shape[0]] = emb
        
        return padded
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text response given input and optionally images
        Args:
            input_ids: (B, T) token IDs
            pixel_values: (B, N, 3, 448, 448) optional image patches
            max_new_tokens: maximum tokens to generate
        Returns:
            generated_ids: (B, T') generated token IDs
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)
            
            inputs_embeds = self._merge_embeddings(
                input_ids, text_embeds, vision_embeds
            )
            
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        else:
            inputs_embeds = text_embeds
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )
        
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.llm.gradient_checkpointing_enable()
        self.vision_tower.vision_model.gradient_checkpointing = True
    
    def unfreeze_vision_tower(self):
        """Unfreeze vision tower for full fine-tuning"""
        self.vision_tower.unfreeze()
        for param in self.vision_tower.parameters():
            param.requires_grad = True
    
    def unfreeze_llm(self):
        """Unfreeze LLM for full fine-tuning (usually done via LoRA)"""
        for param in self.llm.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load pretrained model"""
        import json
        config_path = f"{path}/config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        model = cls(**config, **kwargs)
        
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
            json.dump(config, f, indent=2)
        
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
