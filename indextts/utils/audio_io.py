import numpy as np
import torch
import torchaudio

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def _tensor_to_numpy_channels_last(audio: torch.Tensor) -> np.ndarray:
    """Convert audio tensor to numpy array shaped (samples, channels)."""
    audio = audio.detach().cpu()
    if audio.ndim == 1:
        return audio.numpy()
    return audio.transpose(0, 1).contiguous().numpy()


def save_audio(output_path: str, audio: torch.Tensor, sampling_rate: int, subtype: str = "PCM_16"):
    """
    Save audio to disk preferring soundfile to avoid torchcodec dependency.

    Args:
        output_path: Destination path for the audio file.
        audio: Tensor shaped as (channels, samples) or (samples,).
        sampling_rate: Audio sampling rate.
        subtype: SoundFile subtype. Defaults to "PCM_16".
    """
    last_error = None
    if sf is not None:
        try:
            audio_np = _tensor_to_numpy_channels_last(audio)
            sf.write(output_path, audio_np, sampling_rate, subtype=subtype)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
    try:
        torchaudio.save(output_path, audio.cpu(), sampling_rate)
        return
    except Exception as exc:  # pragma: no cover
        if last_error is None:
            last_error = exc
        raise RuntimeError(
            "Failed to save audio using both soundfile and torchaudio. "
            "Please install the 'soundfile' package or ensure torchaudio is configured correctly."
        ) from last_error
