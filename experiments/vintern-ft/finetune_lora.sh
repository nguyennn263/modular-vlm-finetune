#!/bin/bash
# Vintern-1B-v3_5 2nd-stage fine-tune on AutoViVQA (grouped leak-free split).
#
# VERBATIM from the official Vintern cookbook recipe
#   (reference/repo_default_finetune_lora.sh + colab cells 35-38):
#   freeze backbone + MLP + LLM ; LoRA rank 16 on the LLM only ;
#   max_dynamic_patch 6 ; force_image_size 448 ; down_sample_ratio 0.5 ;
#   lr 4e-5 ; cosine ; warmup 0.03 ; wd 0.01 ; 1 epoch ; conv_style Hermes-2.
#
# Only deviations: paths, GPU count, and total-batch plumbing for a single
# 16 GB card. Recipe hyper-params are UNCHANGED.
set -x

GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}                     # cookbook total batch = 16
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

MODEL_PATH=${MODEL_PATH:-"./pretrained/Vintern-1B-v3_5"}
META_PATH=${META_PATH:-"./shell/data/meta_autovivqa.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"work_dirs/vintern_1b_v3_5_autovivqa_lora"}
EPOCHS=${EPOCHS:-1}
SEED=${SEED:-42}
RESUME_ARG=${RESUME_ARG:-}                       # e.g. "--resume_from_checkpoint <dir>/checkpoint-1000"

# --overwrite_output_dir would wipe a resume checkpoint we just copied in.
OVERWRITE=True
[ -n "$RESUME_ARG" ] && OVERWRITE=False

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1  # so `tee` actually captures progress instead of losing it to a full stdout buffer on kill
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "$OUTPUT_DIR"

# deepspeed is optional for a single-GPU LoRA run; drop it if unavailable
DS_ARG="--deepspeed zero_stage1_config.json"
if [ "${SKIP_DEEPSPEED:-0}" = "1" ] || [ ! -f "zero_stage1_config.json" ]; then
  DS_ARG=""
  echo "[finetune] deepspeed disabled (SKIP_DEEPSPEED=${SKIP_DEEPSPEED:-0}, config present=$([ -f zero_stage1_config.json ] && echo yes || echo no))"
fi

# save_only_model below is an infra fix, not a recipe change. Full checkpoints
# (model + fp32 optimizer moments, ~10GB each with 2 kept) were too large to
# reliably fetch off a killed 12h kernel over a flaky connection. Skips
# optimizer/scheduler/rng state on save -- trainer_state.json (global_step) is
# still written, so --resume_from_checkpoint still skips completed steps; only
# the optimizer's momentum resets across a session boundary. LoRA/lr/tiles/
# epoch unchanged.
torchrun \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir ${OVERWRITE} \
  ${RESUME_ARG} \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --seed ${SEED} \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --save_only_model True \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --max_seq_length 700 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  ${DS_ARG} \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
