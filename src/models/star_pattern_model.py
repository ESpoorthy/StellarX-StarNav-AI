"""
star_pattern_model.py
=====================
Defines the neural network architecture used for star pattern recognition.

Responsibility (planned)
------------------------
- Accept the extracted star feature representation as input.
- Produce a classification output identifying the observed star pattern,
  or a similarity embedding for retrieval-based matching.
- Expose a standard PyTorch ``nn.Module`` interface.

Architecture
------------
The specific architecture (CNN, MLP, graph network, transformer, or hybrid)
is to be determined during Phase 3 based on the chosen feature representation,
required accuracy, and inference latency constraints.

Implementation note
-------------------
All architecture hyperparameters (layer sizes, activation functions, dropout
rates, etc.) must be sourced from config.yaml.  No hard-coded values.
"""

from __future__ import annotations

from typing import Any


# NOTE: torch is imported lazily inside methods so this module can be imported
# without PyTorch installed (Phase 1 / Phase 2 work does not require it).
# Phase 3 will add the concrete architecture and require torch at runtime.


class StarPatternModel:
    """Neural network model for star pattern recognition.

    This class is a placeholder.  The architecture body will be implemented
    during Phase 3 once the input feature representation and output schema
    have been finalised.  It will then subclass ``torch.nn.Module``.

    Parameters
    ----------
    config:
        Model configuration dict, typically the ``model`` section of
        config.yaml.  Expected keys (all TBD):
        ``input_channels``, ``output_classes``, and any architecture-specific
        hyperparameters.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # TODO (Phase 3): import torch and torch.nn here, define all layer
        #   groups, and make this class subclass torch.nn.Module.
        #
        #   Example skeleton (to be replaced):
        #     import torch.nn as nn
        #     super().__init__()
        #     self.backbone = ...
        #     self.classifier = ...
        raise NotImplementedError(
            "StarPatternModel architecture is not yet defined. "
            "Implementation is planned for Phase 3."
        )

    def forward(self, x):
        """Run a forward pass through the network.

        Parameters
        ----------
        x:
            Input feature tensor (torch.Tensor).  Shape is TBD pending
            feature design (Phase 3).

        Returns
        -------
        torch.Tensor
            Raw logits or similarity scores.  Shape is TBD.

        Raises
        ------
        NotImplementedError
            Until the architecture is implemented in Phase 3.
        """
        # TODO (Phase 3): implement the forward pass.
        raise NotImplementedError("StarPatternModel.forward is not yet implemented.")


def build_model(config: dict) -> StarPatternModel:
    """Instantiate and return a StarPatternModel from a config dict.

    Parameters
    ----------
    config:
        Full project configuration dict (loaded from config.yaml).
        The ``model`` sub-dict is passed to StarPatternModel.

    Returns
    -------
    StarPatternModel
        Un-trained model instance.

    Raises
    ------
    NotImplementedError
        Until the model architecture is implemented in Phase 3.
    """
    # TODO (Phase 3): resolve architecture choice from config and instantiate.
    raise NotImplementedError("build_model is not yet implemented.")
