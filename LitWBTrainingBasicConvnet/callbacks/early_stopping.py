"""Early stopping that starts counting again when a run is resumed."""

from typing import Any

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason


class ResumableEarlyStopping(EarlyStopping):
    """Stop training when the monitored metric stops improving.

    The same as Lightning's ``EarlyStopping``, except when a run is resumed.
    The checkpoint stores the callback's state, and Lightning's version would
    restore all of it: a run that stopped early would stop again after one
    epoch, and a new ``patience`` in the config would be ignored. Here only the
    best score is restored, so the resumed run must still beat the earlier
    run's best; the epochs without improvement are counted from zero, with the
    config's ``patience``.
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the best score from a checkpoint, and nothing else.

        Args:
            state_dict: The state saved by ``EarlyStopping.state_dict``.
        """
        self.best_score = state_dict["best_score"]
        self.wait_count = 0
        self.stopped_epoch = 0
        self.stopping_reason = EarlyStoppingReason.NOT_STOPPED
        self.stopping_reason_message = None
