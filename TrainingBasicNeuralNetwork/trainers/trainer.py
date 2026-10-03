"""The hand-written training loop: train, validate, checkpoint, and record history."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Handles the training loop, evaluation, checkpointing, and metric history."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        run_dir: Path,
        device: torch.device,
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

    def save_history(self) -> None:
        """Write the per-epoch metric history to ``history.json`` in ``run_dir``."""
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
