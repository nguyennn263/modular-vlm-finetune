# Vintern Integration Analysis & Refactoring Plan

## 1. Ref2 Codebase Structure Overview

### Architecture
- **Vision Model**: InternVisionModel (ViT-based)
- **Language Model**: Qwen2ForCausalLM or LlamaForCausalLM
- **Bridge**: MLP Layer (`self.mlp1`)
  ```python
  self.mlp1 = nn.Sequential(
      nn.LayerNorm(vit_hidden_size * downsample_ratio²),
      nn.Linear(vit_hidden_size * downsample_ratio², llm_hidden_size),
      nn.GELU(),
      nn.Linear(llm_hidden_size, llm_hidden_size)
  )
  ```

### Inference Flow (chat method)
1. **Conversation Template** (from `conversation.py`)
   ```
   <|im_start|>system
   {system_message}<|im_end|>
   <|im_start|>user
   <image>
   {question}<|im_end|>
   <|im_start|>assistant
   {answer}<|im_end|>
   ```

2. **Image Token Replacement**
   - `<image>` → `<img>[IMG_CONTEXT_TOKEN] × num_image_token × num_patches</img>`
   - Used during inference to mark image patch embeddings

3. **Forward Pass**
   - Extract vision embeddings via `extract_feature()`
   - Filter vision embeddings by `image_flags` (marks which positions had images)
   - Replace corresponding input embeddings with vision embeddings
   - Language model forward pass
   - CrossEntropyLoss on **all tokens** (no masking)

### Key Configs
- `template`: Conversation template name (e.g., 'internvl2_5')
- `system_message`: From template (Vietnamese for Vintern)
- `num_image_token`: Total image patch tokens
- `downsample_ratio`: Dimension reduction factor

---

## 2. Current Project Codebase Issues

### Problem 1: Output Format Mismatch ❌
**Ref2 (Correct)**:
```
<|im_start|>system
Bạn là một mô hình...Vintern...<|im_end|>
<|im_start|>user
<image>
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

**Project Using Wrong Placeholder**:
- Dataset.py: Uses question with `<image>\n` prefix already
- Should reconstruct: Remove `<image>\n` from question, then rebuild full template

### Problem 2: Loss Calculation ❌
**Ref2**: CrossEntropyLoss on all tokens (standard LLM training)
**Project**: Needs loss only on Answer part (VQA-style)
- ✅ Already implemented with `answer_start_pos` masking
- ⚠️ But format template changed, so positions may be off

### Problem 3: Image Inference Integration ❌
- Ref2 uses `image_flags` to mark which positions have images
- Project bridges are not yet compatible with this image marking scheme
- Project directly concatenates vision embeddings (no image_flags mechanism)

### Problem 4: Bridge Module Misalignment ❌
**Ref2**: Single fixed MLP bridge
**Project**: Multiple bridge types (Residual, MultiToken, TileAttention, etc.)
- Need to ensure all bridges output correct shape for distillation

---

## 3. Refactoring Steps for Project

### Step 1: Fix Template Format ✅ [DONE]
- ✅ dataset.py: Updated to use proper Vintern template
- ✅ collator_onesample.py: Updated to match dataset format
- ⚠️ Need to verify answer_start_pos is calculated correctly with new format

### Step 2: Align Loss Calculation
- Keep current VQA-style loss (only on Answer tokens)
- Verify masking works with `answer_start_pos` calculated from new template
- Add logging to track:
  - Total tokens in batch
  - Question tokens masked
  - Answer tokens used for loss

### Step 3: Image Flag Integration (Optional but Recommended)
- Add `image_flags` support to identify which positions have images
- Helps with inference mode visualization
- Not critical for training (can skip for now)

### Step 4: Bridge Architecture Compatibility
- Ensure all 6 bridge types output:
  - Shape: `[batch, seq_len, llm_hidden_size]` where seq_len=1 for pooled, >1 for patch-based
  - Dtype: matches model dtype (bfloat16)
  - Device: matches device (cuda/cpu)
- Test: Run sample forward pass with distillation loss

### Step 5: Validation
- Create test script to verify:
  - ✅ Template format matches ref2
  - ✅ Loss masking works correctly
  - ✅ All 6 experiments run without errors
  - ✅ Checkpoints save/load correctly

---

## 4. Key Changes Summary

### dataset.py Changes
- ✅ Add `system_message` from Vintern template
- ✅ Rebuild prompt format using official template
- ✅ Calculate `answer_start_pos` correctly with new format
- ✅ Remove hardcoded `<image>` prefix logic

### collator_onesample.py Changes
- ✅ Update prompt building to match dataset.py
- ✅ Use same system_message
- ✅ Calculate `answer_start_pos` per batch sample

### trainer.py (No changes needed)
- ✅ Loss masking already in place
- ⚠️ Verify masking still works with new template positions

### Experiments (Exp1-6)
- No changes needed to experiment files
- Bridge modules already compatible
- Just verify syntax and training flow

---

## 5. Testing Checklist

- [ ] Dataset produces correct token sequences
  - [ ] Verify `<|im_start|>` tokens present
  - [ ] Verify `system_message` included
  - [ ] Verify `answer_start_pos` > 0
  
- [ ] Loss computation
  - [ ] Check log: "CE Loss computed on X answer tokens"
  - [ ] Verify X > 0 (answer tokens being used)
  
- [ ] Training flow
  - [ ] Run Exp1 (ResidualBridge) for 1 epoch
  - [ ] Verify loss decreases
  - [ ] Check checkpoint saves
  
- [ ] All experiments
  - [ ] Exp2: MultiToken
  - [ ] Exp3: TileAttention
  - [ ] Exp4: MiniQFormer
  - [ ] Exp5: QFormer
  - [ ] Exp6: GatedFusion

---

## 6. Reference from Ref2

**Key Files Used**:
- `conversation.py`: Template definitions + `get_conv_template()`
- `modeling_internvl_chat.py`: InternVLChatModel forward() logic
- `configuration_internvl_chat.py`: Config structure

**Key Classes**:
- `Conversation`: Manages prompt templates
- `SeparatorStyle.MPT`: Template separator style
- `InternVLChatModel`: Main model with chat() and batch_chat() methods

---

## 7. Next Actions

1. **Done**: Update dataset.py and collator_onesample.py to use official Vintern template
2. **Next**: Verify template format by examining generated token sequences
3. **Then**: Run Exp1 to confirm loss calculation works
4. **Finally**: Run full ablation study (Exp1-6)

---

## 8. Format Reference (Official from Ref2)

```python
# From conversation.py (internvl2_5 template)
Conversation(
    name='internvl2_5',
    system_template='<|im_start|>system\n{system_message}',
    system_message='Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi là Vintern, được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ nhân tạo hữu ích và không gây hại.',
    roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
    sep_style=SeparatorStyle.MPT,
    sep='<|im_end|>\n',
)
```

**Final Template Output**:
```
<|im_start|>system
Bạn là một mô hình...Vintern...<|im_end|>
<|im_start|>user
<image>
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

**Loss Rule**: Calculate only on tokens after `<|im_start|>assistant\n`
