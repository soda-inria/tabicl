"""Utility functions to load TabICL models from local paths or Hugging Face Hub."""
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
import torch

from tabicl.utils._multiton import _multiton

__all__ = ["_load_tabicl_model"]


def _load_tabicl_model(
    checkpoint_version: str,
    model_path=None,
    allow_auto_download: bool = True,
):
    """Load a model from a given path or download it if not available.

    It uses `model_path` and `checkpoint_version` to determine the source.

    - If `model_path` is specified and exists, it's used directly.
    - If `model_path` is specified but doesn't exist (and auto-download is enabled),
    the version specified by `checkpoint_version` is downloaded to `model_path`.
    - If `model_path` is None, the version specified by `checkpoint_version` is downloaded
    from Hugging Face Hub and cached in the default Hugging Face cache directory.

    Parameters
    ----------
    checkpoint : str or path, default='tabicl-classifier-v1.1-0506.ckpt'
        Specifies which version of the pre-trained model checkpoint to use when `model_path`
        is `None` or points to a non-existent file (and `allow_auto_download` is true).
        Checkpoints are downloaded from https://huggingface.co/jingang/TabICL-clf.
        Available versions:
        - `'tabicl-classifier-v1.1-0506.ckpt'` (Default): The latest best-performing version.
        - `'tabicl-classifier-v1-0208.ckpt'`: The version used in the original TabICL paper.
          Use this for reproducing paper results.
        - `'tabicl-classifier.ckpt'`: A legacy alias for `'tabicl-classifier-v1-0208.ckpt'`.
          Maintained for backward compatibility but its use is discouraged and it may be
          removed in a future release.

    model_path : Optional[str | Path] = None
        Path to the pre-trained model checkpoint file.
        - If provided and the file exists, it's loaded directly.
        - If provided but the file doesn't exist and `allow_auto_download` is true, the version
          specified by `checkpoint_version` is downloaded from Hugging Face Hub (repo: 'jingang/TabICL-clf')
          to this path.
        - If `None` (default), the version specified by `checkpoint_version` is downloaded from
          Hugging Face Hub (repo: 'jingang/TabICL-clf') and cached locally in the default
          Hugging Face cache directory (typically `~/.cache/huggingface/hub`).

    allow_auto_download: bool = True
        Whether to allow automatic download if the pretrained checkpoint cannot be
        found at the specified `model_path`.

    Returns
    -------
    model : TabICL
        The loaded TabICL model instance.

    model_path : Path
        The path from which the model was loaded (either the provided `model_path`
        or the cached/downloaded path).

    Raises
    ------
    AssertionError
        If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

    ValueError
        If a checkpoint cannot be found or downloaded based on the settings.
    """
    key = f"{checkpoint_version}|{model_path}|{allow_auto_download}"
    cached_model = _CachedTabICL(
        key=key,
        checkpoint_version=checkpoint_version,
        model_path=model_path,
        allow_auto_download=allow_auto_download,
    )
    return cached_model.load_from_checkpoint()


def _load_tabicl_model_inner(
    checkpoint_version: str,
    model_path=None,
    allow_auto_download: bool = True,
):
    from tabicl import TabICL

    repo_id = "jingang/TabICL-clf"
    filename = checkpoint_version

    ckpt_legacy = "tabicl-classifier.ckpt"
    ckpt_v1 = "tabicl-classifier-v1-0208.ckpt"
    ckpt_v1_1 = "tabicl-classifier-v1.1-0506.ckpt"

    if filename == ckpt_legacy:
        info_message = (
            f"INFO: You are using '{ckpt_legacy}'. This is a legacy alias for '{ckpt_v1}' "
            f"and is maintained for backward compatibility. It may be removed in a future release.\n"
            f"Please consider using '{ckpt_v1}' or the latest '{ckpt_v1_1}' directly.\n"
            f"'{ckpt_legacy}' (effectively '{ckpt_v1}') is the version "
            f"used in the original TabICL paper. For improved performance, consider using '{ckpt_v1_1}'.\n"
        )
    elif filename == ckpt_v1:
        info_message = (
            f"INFO: You are downloading '{ckpt_v1}', the version used in the original TabICL paper.\n"
            f"A newer version, '{ckpt_v1_1}', is available and offers improved performance.\n"
        )
    elif filename == ckpt_v1_1:
        info_message = (
            f"INFO: You are downloading '{ckpt_v1_1}', the latest best-performing version of TabICL.\n"
            f"To reproduce results from the original paper, please use '{ckpt_v1}'.\n"
        )
    else:
        raise ValueError(
            f"Invalid checkpoint version '{filename}'. Available ones are: '{ckpt_legacy}', '{ckpt_v1}', '{ckpt_v1_1}'."
        )

    if model_path is None:
        # Scenario 1: the model path is not provided, so download from HF Hub based on the checkpoint version
        try:
            model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
        except LocalEntryNotFoundError:
            if allow_auto_download:
                print(info_message)
                print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
            else:
                raise ValueError(
                    f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                    f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                )
        if model_path_:
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
    else:
        # Scenario 2: the model path is provided
        model_path_ = Path(model_path) if isinstance(model_path, str) else model_path
        if model_path_.exists():
            # Scenario 2a: the model path exists, load it directly
            checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:
            # Scenario 2b: the model path does not exist, download the checkpoint version to this path
            if allow_auto_download:
                print(info_message)
                print(
                    f"Checkpoint not found at '{model_path_}'.\n"
                    f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                )
                model_path_.parent.mkdir(parents=True, exist_ok=True)
                cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                Path(cache_path).rename(model_path_)
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                raise ValueError(
                    f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                    f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                    f"'{filename}' from Hugging Face Hub ({repo_id})."
                )

    assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
    assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

    model_path_ = model_path_
    model_ = TabICL(**checkpoint["config"])
    model_.load_state_dict(checkpoint["state_dict"])
    model_.eval()

    return model_, model_path_,



@_multiton
class _CachedTabICL:
    """Cached TabICL model, to ensure only one instance exists in memory.

    TabICL is pre-trained, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    """

    def __init__(
        self,
        key,
        checkpoint_version: str,
        model_path=None,
        allow_auto_download: bool = True,
    ):
        self.key = key
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download

        self.model = None
        self.model_path = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model, self.model_path

        self.model, self.model_path = _load_tabicl_model_inner(
            checkpoint_version=self.checkpoint_version,
            model_path=self.model_path,
            allow_auto_download=self.allow_auto_download,
        )
        return self.model, self.model_path
