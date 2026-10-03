"""A callback that reports each epoch's validation accuracy to Optuna and stops bad trials."""

import lightning as L
import optuna


class OptunaPruning(L.Callback):
    """Report a validation metric to an Optuna trial after every epoch.

    Optuna's pruner compares the reported values with those of earlier trials
    at the same epoch. If this trial is clearly worse (e.g. below the median),
    the callback stops training and sets :attr:`pruned`; the caller then
    raises ``optuna.TrialPruned`` so Optuna records the trial as pruned, and
    the search spends its time on promising settings.

    The best value seen is kept in :attr:`best`, the score of the trial.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_acc") -> None:
        """Store the trial and the metric to report.

        Args:
            trial: The Optuna trial being trained.
            monitor: Logged metric to report, higher is better.
        """
        self.trial = trial
        self.monitor = monitor
        self.best = float("-inf")
        self.pruned: str | None = None

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Report the epoch's value; stop training if the pruner says so."""
        # Lightning runs 2 validation batches before training as a quick check.
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics[self.monitor].item()
        self.best = max(self.best, value)
        epoch = trainer.current_epoch
        self.trial.report(value, step=epoch)
        # Raising inside fit() would leave the DataLoader workers hanging: PyTorch
        # then waits 5 s per worker to shut them down before the next trial.
        if self.trial.should_prune():
            self.pruned = f"{self.monitor}={value:.4f} at epoch {epoch}"
            trainer.should_stop = True
