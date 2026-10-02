"""Lightning callbacks: extra behavior hooked into training and testing."""

from .wandb_predictions import LogTestPredictions

__all__ = ["LogTestPredictions"]
