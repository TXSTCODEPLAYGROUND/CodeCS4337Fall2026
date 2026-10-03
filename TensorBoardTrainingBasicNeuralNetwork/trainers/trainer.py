"""The hand-written training loop: train, validate, checkpoint, and log to TensorBoard."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Handles the training loop, evaluation, checkpointing, metric history, and TensorBoard curves."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        run_dir: Path,
        device: torch.device,
        # Quoted: the docs build mocks torch, and a mocked class has no `|`.
        writer: "SummaryWriter | None" = None,
    ) -> None:
        """Set up the model, loss, optimizer, and history tracking.

        Args:
            model: The network to train. It is moved to ``device``.
            train_loader: Dataloader for the training set.
            val_loader: Dataloader for the validation set.
            config: Parsed experiment config. Must contain a ``training`` section
                with ``epochs`` and ``lr`` (and optionally ``weight_decay``).
            run_dir: Directory where checkpoints and history are written.
            device: Device used for training and evaluation.
            writer: TensorBoard writer for the curves, or ``None`` for no
                logging. The config's optional ``tensorboard`` section sets
                ``log_every_n_steps`` (batch loss, default 50) and
                ``histograms`` (weights and gradients, default ``True``).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.epochs = config["training"]["epochs"]
        self.run_dir = run_dir

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 0.0),
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_checkpoint: Path | None = None
        self.last_checkpoint: Path | None = None
        self.start_epoch = 0

        self.writer = writer
        tb_cfg = config.get("tensorboard", {})
        self.log_every_n_steps = tb_cfg.get("log_every_n_steps", 50)
        self.log_histograms = tb_cfg.get("histograms", True)
        self.global_step = 0
        if writer is not None:
            writer.add_custom_scalars(
                {
                    "Train vs val": {
                        "loss": ["Multiline", ["loss/train", "loss/val"]],
                        "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
                    }
                }
            )

    def _run_epoch(
        self, loader: DataLoader, train: bool, desc: str
    ) -> tuple[float, float]:
        """Run a single pass over ``loader``.

        Args:
            loader: Dataloader to iterate over.
            train: If True, gradients are computed and the optimizer is stepped;
                otherwise the model runs in eval mode without gradients.
            desc: Label shown on the progress bar.

        Returns:
            A tuple ``(mean_loss, accuracy)`` computed over all samples.
        """
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1
                    if self.writer and self.global_step % self.log_every_n_steps == 0:
                        self.writer.add_scalar(
                            "batch/train_loss", loss.item(), self.global_step
                        )

                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, loader: DataLoader, desc: str = "eval") -> tuple[float, float]:
        """Evaluate the model on ``loader`` without updating weights.

        Args:
            loader: Dataloader to evaluate on.
            desc: Label shown on the progress bar.

        Returns:
            A tuple ``(mean_loss, accuracy)``.
        """
        return self._run_epoch(loader, train=False, desc=desc)

    def fit(self) -> dict[str, list[float]]:
        """Train for the configured number of epochs.

        After each epoch the model is evaluated on the validation set. Whenever
        validation accuracy improves, the previous best checkpoint is replaced
        by ``best_epochXX_valaccY.YYYY.pt``. The previous last checkpoint is
        replaced by ``last_epochXX_valaccY.YYYY.pt`` after every epoch, so an
        interrupted run can be resumed (see :meth:`resume`). The metric
        history is saved as ``history.json`` at the end.

        After :meth:`resume`, epochs are numbered after the checkpoint's.

        With a TensorBoard writer, every epoch's losses, accuracies, learning
        rate, and (optionally) weight and gradient histograms are logged, and
        the training loss every ``log_every_n_steps`` batches.

        Returns:
            Per-epoch history with keys ``train_loss``, ``train_acc``,
            ``val_loss``, and ``val_acc``.
        """
        last_epoch = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch + 1, last_epoch + 1):
            train_loss, train_acc = self._run_epoch(
                self.train_loader, train=True, desc=f"train {epoch}"
            )
            val_loss, val_acc = self.evaluate(self.val_loader, desc=f"val {epoch}")

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            print(
                f"Epoch {epoch}/{last_epoch} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                if self.best_checkpoint is not None:
                    self.best_checkpoint.unlink(missing_ok=True)
                self.best_checkpoint = self.save_checkpoint(
                    self._checkpoint_name("best", epoch, val_acc), epoch, val_acc
                )
            if self.last_checkpoint is not None:
                self.last_checkpoint.unlink(missing_ok=True)
            self.last_checkpoint = self.save_checkpoint(
                self._checkpoint_name("last", epoch, val_acc), epoch, val_acc
            )

        self.save_history()
        return self.history

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Write one epoch's curves (and histograms) to TensorBoard.

        Train and validation values are logged as ``loss/train`` and
        ``loss/val`` (likewise ``accuracy/...``); the *Custom Scalars* tab
        draws each pair in one chart, so overfitting shows as the two lines
        drifting apart.

        Args:
            epoch: The epoch, used as the x-axis step.
            train_loss: Mean training loss of the epoch.
            train_acc: Training accuracy of the epoch.
            val_loss: Validation loss after the epoch.
            val_acc: Validation accuracy after the epoch.
        """
        if self.writer is None:
            return
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        self.writer.add_scalar("accuracy/train", train_acc, epoch)
        self.writer.add_scalar("accuracy/val", val_acc, epoch)
        self.writer.add_scalar(
            "learning_rate", self.optimizer.param_groups[0]["lr"], epoch
        )
        if self.log_histograms:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"weights/{name}", param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)
        # Write now, so a TensorBoard open during training shows every epoch.
        self.writer.flush()

    @staticmethod
    def _checkpoint_name(tag: str, epoch: int, val_acc: float) -> str:
        """Build a checkpoint filename that encodes its epoch and accuracy.

        Args:
            tag: Checkpoint kind, e.g. ``"best"`` or ``"last"``.
            epoch: Epoch at which the checkpoint was taken.
            val_acc: Validation accuracy at that epoch.

        Returns:
            A filename such as ``best_epoch07_valacc0.9123.pt``.
        """
        return f"{tag}_epoch{epoch:02d}_valacc{val_acc:.4f}.pt"

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float) -> Path:
        """Save model and optimizer state to ``run_dir / filename``.

        The config is stored inside the checkpoint so it can be reloaded
        without the original config file.

        Args:
            filename: Name of the checkpoint file.
            epoch: Epoch number stored alongside the weights.
            val_acc: Validation accuracy at ``epoch``.

        Returns:
            Path to the written checkpoint.
        """
        path = self.run_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "val_acc": val_acc,
                "best_val_acc": self.best_val_acc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Only the model state is restored; the optimizer state is ignored.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def resume(self, path: Path) -> None:
        """Continue from a checkpoint: restore the model, optimizer, and epoch count.

        :meth:`fit` then trains ``epochs`` more epochs, numbered after the
        checkpoint's. The optimizer continues with its saved state, including
        its learning rate. The best accuracy is tracked anew, so this run saves
        its own best checkpoint.

        Args:
            path: Path to a checkpoint written by :meth:`save_checkpoint`.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.global_step = self.start_epoch * len(self.train_loader)

    def save_history(self) -> None:
        """Write the per-epoch metric history to ``history.json`` in ``run_dir``."""
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
