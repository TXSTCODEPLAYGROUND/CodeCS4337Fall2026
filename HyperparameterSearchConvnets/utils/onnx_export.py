"""Export a network to ONNX, e.g. to look at it in Netron."""

import warnings
from pathlib import Path

import torch
from torch import nn


def export_onnx(
    net: nn.Module, path: str | Path, in_channels: int = 1, image_size: int = 28
) -> Path:
    """Save a network as an ONNX graph that keeps every layer visible.

    The network is exported in eval mode, so dropout (which does nothing at
    test time) is left out. ``torch.onnx.export``'s own optimizer would also
    merge each BatchNorm into the convolution before it; here only constant
    folding is applied, so the BatchNorm layers stay in the graph, and
    convolutions built without a bias are saved without one.

    Args:
        net: The network, e.g. a
            :class:`~HyperparameterSearchConvnets.models.components.convnet.ConvNet`.
        path: Where to save the ``.onnx`` file.
        in_channels: Channels of the input images.
        image_size: Height and width of the input images.

    Returns:
        The path of the saved file.
    """
    import onnx_ir.passes.common as ir_passes
    from onnxscript import optimizer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = next(net.parameters()).device
    example = torch.randn(1, in_channels, image_size, image_size, device=device)
    was_training = net.training
    net.eval()
    try:
        with warnings.catch_warnings():
            # Printed by torch.export for every export: a PyTorch class being renamed.
            warnings.filterwarnings(
                "ignore", message=r".*treespec.*", category=FutureWarning
            )
            program = torch.onnx.export(
                net,
                (example,),
                input_names=["image"],
                output_names=["logits"],
                optimize=False,
                verbose=False,
            )
    finally:
        net.train(was_training)
    optimizer.fold_constants(program.model)
    # The exporter gives every convolution without a bias an all-zero one.
    parameters = dict(net.named_parameters())
    for node in program.model.graph:
        is_conv_with_bias = node.op_type == "Conv" and len(node.inputs) == 3
        if is_conv_with_bias and node.inputs[2].name not in parameters:
            node.resize_inputs(2)
    optimizer.remove_unused_nodes(program.model)
    ir_passes.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0)(
        program.model
    )
    program.save(path)
    return path
