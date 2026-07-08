"""Neural emulator + simulation-based inference for THOR's biconical MgII wind.

Two halves, decoupled by the training library (library.h5):
  - data generation (THOR-coupled): prior -> sample -> thor_sim -> library
  - ML (THOR-independent):           library -> emulator -> npe -> infer

The ML submodules (emulator, npe) import torch/sbi lazily, so importing this
package without the `ml` extra installed is fine for the data-generation half.
"""

from .prior import Prior

__all__ = ["Prior"]
__version__ = "0.1.0"
