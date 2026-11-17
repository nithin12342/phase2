"""ASR transcript via Whisper. Lazy import; runs on the Colab GPU.

D-Vlog ships no transcript, so text features need ASR. The local sample mode
does not call this module.
"""

# The model loads once per process and stays on the GPU. Reloading per video
# wastes seconds and leaks memory, mirroring the encoder cache.
_MODEL = None
_MODEL_NAME = None


def transcribe(audio_path, model_name="base"):
    """Return {"text": str, "segments": list} for one audio file."""
    import whisper  # lazy: Colab only
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = whisper.load_model(model_name)
        _MODEL_NAME = model_name
    result = _MODEL.transcribe(str(audio_path))
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
    }


def token_counts(text):
    """Pure stdlib word counts used by the lexical features."""
    tokens = [t for t in text.lower().split() if t.strip()]
    return {"unique": len(set(tokens)), "total": len(tokens)}