"""Vendored ESPnet compatibility checks use synthetic arrays; never train."""
import numpy as np
import pytest

from realtime_use._vendor import espnet_detection as detection


@pytest.fixture(autouse=True)
def numpy_without_deprecated_alias(monkeypatch):
    # NumPy 2.x may expose np.bool again. Exercise the missing-alias behavior of
    # the production NumPy 1.26 environment without changing NumPy internals.
    class WithoutBoolAlias:
        def __getattr__(self, name):
            if name == 'bool':
                raise AttributeError("module 'numpy' has no attribute 'bool'")
            return getattr(np, name)
    monkeypatch.setattr(detection, 'np', WithoutBoolAlias())


@pytest.mark.parametrize('length', [1024, 1025, 4096])
@pytest.mark.parametrize('channels', [None, 2])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16])
def test_all_silent_full_frames_preserve_upstream_all_true_mask(length, channels, dtype):
    shape = (length,) if channels is None else (channels, length)
    audio = np.zeros(shape, dtype=dtype)
    before = audio.copy()
    mask = detection.detect_non_silence(audio)
    assert mask.shape == shape and mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, np.ones(shape, dtype=bool))
    np.testing.assert_array_equal(audio, before)


@pytest.mark.parametrize('length', [0, 1, 1023])
@pytest.mark.parametrize('channels', [None, 2])
@pytest.mark.parametrize('value', [0., .25])
def test_short_inputs_keep_shape_dtype_and_all_true_fallback(length, channels, value):
    shape = (length,) if channels is None else (channels, length)
    audio = np.full(shape, value, dtype=np.float32)
    mask = detection.detect_non_silence(audio)
    assert mask.shape == shape and mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, np.ones(shape, dtype=bool))


def test_nonzero_power_threshold_path_retains_expected_frame_mask():
    audio = np.array([[1., 1., 1., 1., 0., 0., 0., 0.,
                       2., 2., 2., 2., 0., 0., 0., 0.]])
    before = audio.copy()
    mask = detection.detect_non_silence(audio, frame_length=4, frame_shift=4)
    expected = np.repeat([[True, False, True, False]], 4, axis=1)
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(audio, before)
