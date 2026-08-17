"""Post-hoc concept steering of a pre-trained model.

Run any module on its own to exercise it (``python -m steering.mrf``), or
``python -m steering.main`` for the full experiment.
"""
import torch


def resolve_device(name: str = 'auto') -> torch.device:
    """``'auto'`` picks CUDA, then Apple MPS, then CPU. Anything else is taken
    literally, so ``--device cpu`` still forces CPU on a GPU machine."""
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
