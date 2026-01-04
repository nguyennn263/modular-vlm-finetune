"""
VLM Model: Main architecture combining Vision + Projector + LLM
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from .vision_encoders import build_vision_encoder
from .projectors import build_projector
from .registry import ModelRegistry, VISION_ENCODER_CONFIGS


class VLMModel(nn.Module):
    """
    Modular Vision-Language Model
    
    Có thể dễ dàng thay đổi:
    - Vision Encoder: internvit, siglip, clip
    - Projector: mlp, linear, downsample
    - LLM: qwen2, phi, vinallama
    """
    
    def __init__(
        self,
        vision_encoder_type: str = "internvit",
        vision_model_name: Optional[str] = None,
        vision_hidden_size: int = 1024,
        image_size: int = 448,
        
        projector_type: str = "mlp",
        
        llm_type: str = "qwen2-0.5b",
        llm_model_name: Optional[str] = None,
        
        freeze_vision: bool = False,
        freeze_llm: bool = True,
        use_flash_attention: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        
        image_token_id: Optional[int] = None,
    ):
        super().__init__()
        
        llm_config = ModelRegistry.get_llm_config(llm_type)
        llm_model_name = llm_model_name or llm_config["model_name"]
        llm_hidden_size = llm_config["hidden_size"]
        self.image_token_id = image_token_id or llm_config.get("image_token_id", 151667)
        
        # Build Vision Encoder
        self.vision_encoder = build_vision_encoder(
            encoder_type=vision_encoder_type,
            model_name=vision_model_name,
            hidden_size=vision_hidden_size,
            image_size=image_size,
            torch_dtype=torch_dtype,
            use_flash_attention=use_flash_attention if vision_encoder_type == "internvit" else False,
        )
        
        vision_hidden_size = self.vision_encoder.hidden_size
        
        # Build Projector
        self.projector = build_projector(
            projector_type=projector_type,
            input_dim=vision_hidden_size,
            output_dim=llm_hidden_size,
        )
        
        # Build LLM
        config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        if use_flash_attention:
            config._attn_implementation = "flash_attention_2"
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        
        if freeze_vision:
            self.vision_encoder.freeze()
        if freeze_llm:
            self._freeze_llm()
        
        # Store config
        self.config = {
            "vision_encoder_type": vision_encoder_type,
            "projector_type": projector_type,
            "llm_type": llm_type,
            "llm_model_name": llm_model_name,
            "image_token_id": self.image_token_id,
        }
    
    def _freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True
    
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through vision encoder + projector"""
        # Vision encoder
        vision_features = self.vision_encoder(pixel_values)
        # Project to LLM space
        vision_embeds = self.projector(vision_features)
        return vision_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass"""
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)
            
            inputs_embeds = self._merge_embeddings(input_ids, text_embeds, vision_embeds)
            
            if attention_mask is not None and inputs_embeds.shape[1] != attention_mask.shape[1]:
                attention_mask = self._adjust_mask(attention_mask, inputs_embeds.shape[1])
            
            if labels is not None and inputs_embeds.shape[1] != labels.shape[1]:
                labels = self._adjust_labels(labels, inputs_embeds.shape[1])
        else:
            inputs_embeds = text_embeds
        
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    
    def _merge_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Insert vision embeddings at <image> token positions"""
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
            
            batch_embeds.append(torch.cat(parts, dim=0))
        
        # Pad to same length
        max_len = max(e.shape[0] for e in batch_embeds)
        padded = torch.zeros(B, max_len, text_embeds.shape[-1], dtype=text_embeds.dtype, device=text_embeds.device)
        
        for b, emb in enumerate(batch_embeds):
            padded[b, :emb.shape[0]] = emb
        
        return padded
    
    def _adjust_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        new_mask = torch.ones(mask.shape[0], target_len, dtype=mask.dtype, device=mask.device)
        orig_len = min(mask.shape[1], target_len)
        new_mask[:, :orig_len] = mask[:, :orig_len]
        return new_mask
    
    def _adjust_labels(self, labels: torch.Tensor, target_len: int) -> torch.Tensor:
        new_labels = torch.full((labels.shape[0], target_len), -100, dtype=labels.dtype, device=labels.device)
        orig_len = min(labels.shape[1], target_len)
        new_labels[:, :orig_len] = labels[:, :orig_len]
        return new_labels
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        """Generate response"""
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)
            inputs_embeds = self._merge_embeddings(input_ids, text_embeds, vision_embeds)
        else:
            inputs_embeds = text_embeds
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    
    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()


def create_vlm_model(config: Dict[str, Any]) -> VLMModel:
    """Factory function to create VLM from config dict"""
    return VLMModel(
        vision_encoder_type=config.get("vision_encoder_type", "internvit"),
        vision_model_name=config.get("vision_model_name"),
        vision_hidden_size=config.get("vision_hidden_size", 1024),
        image_size=config.get("image_size", 448),
        projector_type=config.get("projector_type", "mlp"),
        llm_type=config.get("llm_type", "qwen2-0.5b"),
        llm_model_name=config.get("llm_model_name"),
        freeze_vision=config.get("freeze_vision", False),
        freeze_llm=config.get("freeze_llm", True),
        use_flash_attention=config.get("use_flash_attention", False),
        torch_dtype=getattr(torch, config.get("torch_dtype", "float16")),
        image_token_id=config.get("image_token_id"),
    )
