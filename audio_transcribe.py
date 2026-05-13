from utils import load_whisper
import numpy as np
import io
import soundfile as sf

def transcribe_audio(audio_bytes, sample_rate=16000):
    model = load_whisper()
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    result = model.transcribe(audio_np.astype(np.float32), language='en')
    return result['text']

def transcribe_audio_array(audio_np: np.ndarray, sample_rate: int = 16000):
    """Transcribe from raw numpy float32 array (for live recording)."""
    model = load_whisper()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sample_rate != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
    result = model.transcribe(audio_np.astype(np.float32), language='en')
    return result['text']