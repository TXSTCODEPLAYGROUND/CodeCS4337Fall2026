"""Sphinx configuration for the course code repository.

Heavy libraries (PyTorch, Lightning, torchvision, torchmetrics, Matplotlib,
NumPy, tqdm, python-dotenv) are mocked, so ``sphinx-build`` does not need them
installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "CS4337 Fall 2026"
author = "CS4337 Fall 2026"
copyright = "2026, CS4337 Fall 2026"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_baseurl = "https://txstcodeplayground.github.io/CodeCS4337Fall2026/"

# Import names, not pip names: python-dotenv is imported as ``dotenv``.
autodoc_mock_imports = [
    "lightning",
    "torch",
    "torchvision",
    "torchmetrics",
    "matplotlib",
    "numpy",
    "dotenv",
    "tqdm",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Keep constructor parameters on ``__init__``. A mocked ``nn.Module`` base
# would otherwise show ``ConvNet(*args, **kwargs)``.
autodoc_class_signature = "separated"
autodoc_typehints = "description"
autoclass_content = "class"

# Fold ``__init__`` parameters into the class description.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = False
