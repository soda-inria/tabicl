"""Define argument parser for TabICL training."""

import argparse

from tabicl.prior.graph_lib._config import PriorConfig


def str2bool(value):
    return value.lower() == "true"


def bool_or_str(value):
    """Parse ``"true"``/``"false"`` into a bool, otherwise pass the string through.

    Used for ``--col_feature_group``, which is either ``False`` (no grouping, the value
    stored in the checkpoint configs) or a named grouping mode (``"same"`` or ``"valid"``).
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    """Build an argument parser with all TabICL training arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all training, model, and
        checkpoint arguments.
    """
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabICL", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="Autocast data type used for training when --amp is on. 'float16' enables the gradient "
        "scaler; 'bfloat16' needs no loss scaling; 'float32' makes autocast a no-op. Parameters are "
        "always kept in float32.",
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--muon",
        default=False,
        type=str2bool,
        help="If True, use the Muon optimizer instead of AdamW. All parameters are placed in a "
        "single Muon group and updated via Newton-Schulz orthogonalization (non-2D parameters "
        "are reshaped to 2D for this).",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="First moment decay (AdamW beta1 / Muon momentum)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Second moment decay for AdamW (not used by Muon)")
    parser.add_argument(
        "--use_cautious_wd",
        default=False,
        type=str2bool,
        help="If True, use cautious weight decay for Muon (only decay where update and parameter share sign). "
        "Note: the released TabICLv2 checkpoints were trained with this left at False.",
    )
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type", default="mix_scm", type=str, help="Prior type: dummy, mlp_scm, tree_scm, mix_scm"
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of dataloader workers for on-the-fly prior generation. If <= 0, use the number of "
        "physical CPU cores. Ignored when loading pre-generated data from prior_dir.",
    )
    parser.add_argument(
        "--log_n_features",
        default=False,
        type=str2bool,
        help="If True, sample the number of features from a log-uniform distribution.",
    )
    parser.add_argument(
        "--regression_method",
        default=None,
        type=str,
        help="If None, train for classification. If 'quantile', train for quantile regression with a "
        "pinball loss (requires num_quantiles > 0). Other values are not supported.",
    )
    parser.add_argument(
        "--num_quantiles",
        type=int,
        default=999,
        help="Number of quantiles predicted for regression. Only used when regression_method is set.",
    )

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )
    parser.add_argument(
        "--ignore_d",
        default=True,
        type=str2bool,
        help="If True (the v2 default), ignore the per-dataset feature count d during training and "
        "pass d=None to the model. This lets the model's feature grouping operate over all (padded) "
        "columns and works with variable-feature priors such as graph_scm.",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument(
        "--col_affine",
        default=True,
        type=str2bool,
        help="If True (TabICLv1), the column embedder produces per-column affine weights/biases. "
        "TabICLv2 uses False.",
    )
    parser.add_argument(
        "--col_feature_group",
        default=False,
        type=bool_or_str,
        help="Feature grouping mode in the column embedder: False (no grouping, TabICLv1), "
        "'same' (repeated feature grouping through circular shifts, TabICLv2), or 'valid' "
        "(disjoint groups of consecutive columns, TabPFN-style).",
    )
    parser.add_argument(
        "--col_feature_group_size",
        default=3,
        type=int,
        help="Number of columns per feature group. Only used when --col_feature_group is not False.",
    )
    parser.add_argument(
        "--col_target_aware",
        default=False,
        type=str2bool,
        help="If True (TabICLv2), add target embeddings to the training rows of every column "
        "before the column set transformer. TabICLv1 uses False.",
    )
    parser.add_argument(
        "--col_ssmax",
        default=False,
        type=str2bool,
        help="Whether to use SSMax in the column embedder (the variant is chosen by --ssmax_type)",
    )
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument(
        "--row_rope_interleaved",
        default=True,
        type=str2bool,
        help="RoPE rotation layout in the row interactor: True = interleaved dimension pairs "
        "(TabICLv1), False = split-half rotation (TabICLv2). The two are not equivalent.",
    )
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # ICL Config
    parser.add_argument("--icl_num_blocks", type=int, default=12, help="Number of transformer blocks in ICL predictor")
    parser.add_argument("--icl_nhead", type=int, default=4, help="Number of attention heads in ICL predictor")
    parser.add_argument(
        "--icl_ssmax",
        default=False,
        type=str2bool,
        help="Whether to use SSMax in the ICL predictor (the variant is chosen by --ssmax_type)",
    )
    parser.add_argument("--freeze_icl", default=False, type=str2bool, help="Whether to freeze the ICL predictor")
    parser.add_argument(
        "--ssmax_type",
        default="ssmax",
        type=str,
        help="SSMax variant used where --col_ssmax/--icl_ssmax is True: 'ssmax', 'ssmax-mlp', "
        "'ssmax-mlp-elementwise', 'qassmax-mlp', or 'qassmax-mlp-elementwise' (TabICLv2).",
    )

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )
    parser.add_argument(
        "--norm_type",
        default="default",
        type=str,
        help="Normalization type: 'default' (LayerNorm with bias) or 'layernorm_nobias' (LayerNorm without bias, "
        "used by the TabICLv2 regression checkpoint; the v2 classifier uses the default).",
    )
    parser.add_argument(
        "--zero_init",
        default=True,
        type=str2bool,
        help="If True (TabICLv1), zero-initialize attention output projections and second feedforward "
        "linears so residual branches start as identity. TabICLv2 pre-training uses False "
        "(standard PyTorch initialization).",
    )
    parser.add_argument(
        "--use_flash_attn3",
        default=False,
        type=str2bool,
        help="If True, use FlashAttention-3 during training when it is installed (attention runs in fp16). "
        "The TabICLv2 recipe enables this only for pre-training stages 2 and 3.",
    )
    parser.add_argument(
        "--recompute",
        default=False,
        type=str2bool,
        help="If True, use gradient checkpointing (activation recomputation) to reduce memory at the cost of speed.",
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of temporary checkpoints to keep. Permanent checkpoints are not counted.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")

    ###########################################################################
    ###### Graph prior (graph_scm) config #####################################
    ###########################################################################
    # Adds the graph_scm PriorConfig options (e.g. --graph_noise, --filter_unpredictable_graphs,
    # --max_n_nodes, --cauchy_dag_offset, ...) using the same names as the standalone generator.
    # These are only used when --prior_type graph_scm and on-the-fly generation is enabled.
    PriorConfig.add_args_to_parser(parser)

    return parser
