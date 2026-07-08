"""Pick a torch device: cuda (cluster) > mps (Apple Silicon) > cpu.

Note for MPS: some sbi/zuko normalizing-flow ops historically fall back to CPU.
The nets here are small, so if training is unstable on 'mps', force 'cpu'.
"""


def resolve_device(spec="auto"):
    import torch

    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
