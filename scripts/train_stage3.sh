# This script is used to train TabICL for the third stage of the curriculum learning

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=1 /path/to/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage3 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 50 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 2e-6 \
            --scheduler constant \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cpu \
            --batch_size_per_gp 1 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 40000 \
            --max_seq_len 60000 \
            --log_seq_len True \
            --seq_len_per_gp True \
            --replay_small True \
            --min_train_size 0.5 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --freeze_col True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --freeze_row True \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /my/stage3/checkpoint/dir \
            --checkpoint_path /my/stage2/checkpoint/dir/step-{latest}.ckpt \
            --save_temp_every 1 \
            --save_perm_every 5 \
            --only_load_model True


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
python /path/to/tabicl/prior/genload.py \
    --save_dir /my/stage3/prior/dir \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 50 \
    --resume_from 0 \
    --batch_size 512 \
    --batch_size_per_gp 1 \
    --prior_type mix_scm \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --min_seq_len 40000 \
    --max_seq_len 60000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --replay_small True \
    --min_train_size 0.5 \
    --max_train_size 0.9 \
    --n_jobs -1 \
    --num_threads_per_generate 1 \
    --device cpu

# Loading from disk and training
torchrun --standalone --nproc_per_node=1 /path/to/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage3 \
            --wandb_dir /my/wandb/dir \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 50 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 2e-6 \
            --scheduler constant \
            --gradient_clipping 1.0 \
            --prior_dir /my/stage3/prior/dir \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --freeze_col True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --freeze_row True \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /my/stage3/checkpoint/dir \
            --checkpoint_path /my/stage2/checkpoint/dir/step-{latest}.ckpt \
            --save_temp_every 1 \
            --save_perm_every 5 \
            --only_load_model True