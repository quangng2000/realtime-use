"""Inference exports preserve the selected trained head and discard training state."""

import pytest
import torch

from realtime_use.config import ModelConfig
from realtime_use.export import export_exit, extract_exit
from realtime_use.infer import enhance_waveform, load_model
from realtime_use.model import Enhancer


@pytest.mark.parametrize("decoder_mode", ["shared", "per_exit"])
def test_export_trims_blocks_selects_decoder_and_matches_streaming(tmp_path, decoder_mode):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = ModelConfig(sample_rate=800, depth=3, exits=(1, 2, 3), width=4,
                             state_dim=2, expand=1, conv_kernel=2, decoder_mode=decoder_mode)
        source = Enhancer(config).eval()
        if decoder_mode == "per_exit":
            # Distinct heads detect accidentally exporting the final head instead of exit two.
            with torch.no_grad():
                for depth, decoder in source.decoders.items():
                    decoder.output.bias.fill_(0.1 * int(depth))
        checkpoint = tmp_path / "training.pt"
        torch.save({"model_config": config.to_dict(), "model_state": source.state_dict(),
                    "optimizer_state": {"sentinel": 7}, "discriminator_state": {"sentinel": 8}}, checkpoint)
        destination = export_exit(checkpoint, 2, tmp_path / "inference.pt")
        payload = torch.load(destination, weights_only=True)
        assert "optimizer_state" not in payload and "discriminator_state" not in payload
        assert payload["model_config"]["depth"] == 2
        assert payload["model_config"]["exits"] == [2]
        assert payload["model_config"]["decoder_mode"] == "shared"
        exported = load_model(destination)
        assert len(exported.blocks) == 2
        assert sum(p.numel() for p in exported.parameters()) == source.active_parameter_count(2)
        waveform = torch.randn(1, 111) * 0.1
        with torch.inference_mode():
            expected = enhance_waveform(source, waveform, 2)
            for streaming in (False, True):
                actual = enhance_waveform(exported, waveform, 2, streaming=streaming, chunk_samples=13)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        with pytest.raises(FileExistsError):
            export_exit(checkpoint, 2, destination)
    finally:
        torch.set_num_threads(previous_threads)


def test_export_requires_a_trained_exit():
    model = Enhancer(ModelConfig(sample_rate=800, depth=2, exits=(2,), width=4,
                                 state_dim=2, expand=1))
    with pytest.raises(ValueError, match="trained exits"):
        extract_exit(model, 1)
