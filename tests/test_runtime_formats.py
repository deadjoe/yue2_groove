"""Exercise binary dependencies on the formats used by saved works and models."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file


@pytest.mark.parametrize(("extension", "subtype"), [("wav", "FLOAT"), ("flac", "PCM_24")])
def test_stereo_audio_roundtrip_and_concurrent_reads(tmp_path, extension, subtype):
    # Different channels expose interleaving/channel-order regressions.
    left = np.linspace(-0.9, 0.9, 4800, dtype=np.float32)
    audio = np.column_stack((left, -left * 0.5))
    path = tmp_path / f"audio.{extension}"
    sf.write(path, audio, 48000, subtype=subtype)

    def read(_):
        restored, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 48000
        assert restored.shape == audio.shape
        np.testing.assert_allclose(restored, audio, rtol=0, atol=2**-23)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(read, range(32)))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_safetensors_bfloat16_checkpoint_roundtrip(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable on this runner")
    tensors = {
        "weight": torch.arange(128, dtype=torch.float32).reshape(16, 8).to(torch.bfloat16),
        "bias": torch.linspace(-1, 1, 16),
    }
    path = tmp_path / "checkpoint.safetensors"
    save_file(tensors, path)
    restored = load_file(path, device=device)
    for name, expected in tensors.items():
        assert restored[name].device.type == device
        assert restored[name].dtype == expected.dtype
        assert torch.equal(restored[name].cpu(), expected)
