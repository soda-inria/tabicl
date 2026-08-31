#!/bin/bash
# TabICLv2 pre-training -- REGRESSOR, Stage 2 (of 3). See train_v2_reg_stage1.sh for
# the full recipe.
#
# Stage 2: 40K steps on datasets with 400-10,240 samples (log-uniform), 79-81% of
# samples for training, max LR 1e-4, gradient clipping 10. Continues from the Stage 1
# checkpoint (weights only). FlashAttention-3 is enabled (--use_flash_attn3 True)
# for stages 2 & 3 only.
#
# Adjust the placeholder paths and --nproc_per_node / --n_jobs for your hardware.

NUM_GPUS=4                                   # the paper used 4 GPUs for pre-training
CKPT_DIR=/path/to/checkpoints/tabiclv2-reg/stage2
STAGE1_CKPT=/path/to/checkpoints/tabiclv2-reg/stage1/step-500000.ckpt

# Load the Stage 1 weights only on the very first launch. On later launches (e.g.
# restarts after a cluster time limit), CKPT_DIR already contains Stage 2
# checkpoints, and the trainer instead resumes from the latest one with full
# optimizer/scheduler state (--checkpoint_path would override that, so it must
# only be passed the first time).
RESUME_ARGS="--checkpoint_path $STAGE1_CKPT --only_load_model True"
if ls "$CKPT_DIR"/step-*.ckpt >/dev/null 2>&1; then
    RESUME_ARGS=""
fi

torchrun --standalone --nproc_per_node=$NUM_GPUS -m tabicl.train \
            --wandb_log False \
            --wandb_project TabICLv2 \
            --wandb_name tabiclv2_reg_stage2 \
            --device cuda \
            --dtype float32 \
            --np_seed 43 \
            --torch_seed 43 \
            --max_steps 40000 \
            --batch_size 64 \
            --micro_batch_size 1 \
            --lr 1e-4 \
            --muon True \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.01 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --regression_method quantile \
            --num_quantiles 999 \
            --prior_type graph_scm \
            --prior_device cpu \
            --n_jobs 16 \
            --batch_size_per_gp 1 \
            --min_features 1 \
            --max_features 100 \
            --min_seq_len 400 \
            --max_seq_len 10240 \
            --log_seq_len True \
            --min_train_size 0.79 \
            --max_train_size 0.81 \
            --seq_len_per_gp True \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --col_affine False \
            --col_feature_group same \
            --col_feature_group_size 3 \
            --col_target_aware True \
            --col_ssmax True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_rope_interleaved False \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_ssmax True \
            --ssmax_type qassmax-mlp-elementwise \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 True \
            --norm_type layernorm_nobias \
            --checkpoint_dir $CKPT_DIR \
            $RESUME_ARGS \
            --save_temp_every 50 \
            --save_perm_every 1000
