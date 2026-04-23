"""Protocol-based experiment logging for TabICL fine-tuning."""

from __future__ import annotations

from typing import Any, Protocol


class FinetuningLogger(Protocol):
    """Protocol for fine-tuning experiment loggers."""

    def setup(self, config: dict[str, Any]) -> None:
        """Initialize the logger with run configuration."""
        ...

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Log per-step metrics (e.g., batch loss, learning rate)."""
        ...

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """Log per-epoch metrics (e.g., val metrics, mean loss)."""
        ...

    def finish(self) -> None:
        """Finalize the logger (e.g., close the wandb run)."""
        ...


class NullLogger:
    """No-op logger used when no experiment tracking is configured."""

    def setup(self, config: dict[str, Any]) -> None:
        del config

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        pass


class WandbLogger:
    """Weights & Biases experiment logger.

    Lazily imports ``wandb`` so the module can be imported even when the
    dependency is absent. Callers that want to fall back gracefully should
    catch ``ModuleNotFoundError`` from :meth:`setup`.
    """

    def __init__(
        self,
        project: str | None = None,
        run_name: str | None = None,
        entity: str | None = None,
        **wandb_kwargs: Any,
    ):
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.wandb_kwargs = wandb_kwargs
        self._run = None

    def setup(self, config: dict[str, Any]) -> None:
        try:
            import wandb  # noqa: PLC0415
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "WandbLogger requires the 'wandb' package. Install it with: pip install wandb"
            ) from None

        init_kwargs = dict(self.wandb_kwargs)
        if self.project:
            init_kwargs.setdefault("project", self.project)
        if self.run_name:
            init_kwargs.setdefault("name", self.run_name)
        if self.entity:
            init_kwargs.setdefault("entity", self.entity)
        init_kwargs.setdefault("config", config)
        self._run = wandb.init(**init_kwargs)
        wandb.define_metric("val/*", step_metric="train/epoch")

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        if self._run:
            self._run.log(metrics, step=step)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run:
            self._run.finish()
            self._run = None
