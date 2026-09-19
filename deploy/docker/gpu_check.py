"""One-line GPU report for groove-start; exits non-zero when CUDA is unusable."""

import sys

import torch

if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False — no NVIDIA GPU visible to the container")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
flash = torch.backends.cuda.is_flash_attention_available() and cap >= (8, 0)
print(
    f"{name} {total:.0f} GiB sm{cap[0]}{cap[1]} bf16={torch.cuda.is_bf16_supported()} "
    f"flash={flash} torch={torch.__version__}"
)
