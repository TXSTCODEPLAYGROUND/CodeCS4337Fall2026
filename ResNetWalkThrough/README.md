# ResNetWalkThrough

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/ResNetWalkThrough/ResNetWalkThrough.ipynb)

A tour of **ResNet** ([He et al., 2015](https://arxiv.org/abs/1512.03385)),
the network that made very deep ConvNets trainable with skip connections.
Nothing is trained: the notebook takes ResNets already trained on ImageNet
from `torchvision` and looks at them from every side.

Unlike the other projects, this one is **only a notebook**,
[`ResNetWalkThrough.ipynb`](ResNetWalkThrough.ipynb): no package, no
configs, no `python -m`. The ResNet-like blocks you trained yourself are in
[HyperparameterSearchConvnets](../HyperparameterSearchConvnets/README.md).

## What the notebook covers

1. **Setup**: one cell, nothing to clone in Colab.
2. **Example data**: [Imagenette](https://github.com/fastai/imagenette),
   10 easy ImageNet classes (about 95 MB, downloaded to `data/`).
3. **A pretrained ResNet-18** and its top-5 predictions on one image per class.
4. **The architecture**: the stem, the four stages, the blocks, and the head
   with [torchinfo](https://github.com/TylerYep/torchinfo), the first block of
   a stage (with and without a `downsample` shortcut), and the whole graph
   drawn with [torchview](https://github.com/mert-kurttutan/torchview).
5. **The graph in [Netron](https://netron.app)**: the network exported to
   ONNX (in `runs/ResNetWalkThrough/`) and opened in the browser.
6. **Inside the network** with [Captum](https://captum.ai):
   - the activations of every layer (the 8 most active channels per layer);
   - one residual block step by step: input, main path, shortcut, output;
   - **Grad-CAM** heatmaps: which regions of the image made the network
     choose the predicted class, and the true class.
7. **ResNet variants**: ResNet-18 to ResNet-152, ResNeXt, and Wide ResNet
   compared (blocks, depth, parameters, ImageNet accuracy), a bottleneck
   block, and ResNet-18 vs. ResNet-50 on 500 Imagenette images.

## Grad-CAM in one line

$$
\text{Grad-CAM} = \mathrm{ReLU}\left( \sum_{\text{channels}} \text{mean}(\text{gradient}) \times \text{activation} \right)
$$

The activation is the output of one layer (`layer4`: 512 maps of 7 × 7); the
gradient is the derivative of the class score with respect to that output,
so it has the same shape. Its average per channel says how much the class
relies on that channel; the activation says where the channel fires in this
image. The weighted sum is upsampled to 224 × 224 and overlaid on the image.

## Running it

**Colab:** click the badge above and run the cells from top to bottom. A GPU
runtime is faster but not required.

**Locally:** install the repo's requirements (see the
[main README](../README.md#running-locally)), then open the notebook from this
folder with the repo's `.venv` as the kernel. Two extras:

- The torchview graph needs the **Graphviz** program, which pip cannot
  install: `sudo apt install graphviz` (Linux), `brew install graphviz`
  (macOS), or the installer from [graphviz.org](https://graphviz.org/download/)
  (Windows). Colab already has it.
- Netron serves the graph on `localhost:8081`. If you run Cursor or VS Code on
  a remote machine over SSH, forward that port (Ports panel) to open it in
  your browser.

The first run downloads Imagenette and the pretrained weights (ResNet-18:
45 MB, ResNet-50: 98 MB, cached by PyTorch in `~/.cache/torch`).
