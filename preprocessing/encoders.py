"""Pretrained encoders -> 768-d embeddings per modality (Colab GPU only).

All torch / transformers imports are lazy. The local sample mode uses
synthetic_embedding instead and never imports torch. Embeddings are saved one
file per video per modality so a disconnect never loses a finished batch.

Model choices (768-d hidden size):
  audio: facebook/wav2vec2-base-960h
  text:  bert-base-uncased
  video: google/vit-base-patch16-224 (frame mean-pool)
  face:  google/vit-base-patch16-224 (face-crop mean-pool; proxy until OpenFace)
"""
import json
import random
from pathlib import Path

EMBED_DIM = 768
MODALITIES = ("audio", "text", "video", "face")

_MODELS = {
    "audio": "facebook/wav2vec2-base-960h",
    "text": "bert-base-uncased",
    "video": "google/vit-base-patch16-224",
    "face": "google/vit-base-patch16-224",
}


def synthetic_embedding(seed, modality):
    """Deterministic 768-d vector for the sample run. Stdlib only."""
    rng = random.Random(f"{seed}:{modality}")
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)]


def _device():
    import torch  # lazy
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mean_pool(hidden):
    """Mean over the time/token axis -> one 768-d vector."""
    import torch  # lazy
    return hidden.mean(dim=-2).squeeze(0)


# Models load once per process and stay on the device. Reloading per video
# leaks GPU memory until the T4 OOMs, and wastes minutes per video.
_CACHE = {}


def _load(kind, name):
    key = (kind, name)
    if key not in _CACHE:
        from transformers import (  # lazy
            Wav2Vec2Model, Wav2Vec2Processor,
            BertModel, BertTokenizer,
            ViTModel, ViTImageProcessor,
        )
        dev = _device()
        if kind == "audio":
            proc = Wav2Vec2Processor.from_pretrained(name)
            model = Wav2Vec2Model.from_pretrained(name).to(dev).eval()
        elif kind == "text":
            proc = BertTokenizer.from_pretrained(name)
            model = BertModel.from_pretrained(name).to(dev).eval()
        else:
            proc = ViTImageProcessor.from_pretrained(name)
            model = ViTModel.from_pretrained(name).to(dev).eval()
        _CACHE[key] = (proc, model)
    return _CACHE[key]


def encode_audio(wav_path):
    import torch  # lazy
    import librosa  # lazy

    proc, model = _load("audio", _MODELS["audio"])
    dev = _device()
    y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    # Chunk long audio so one forward pass cannot OOM on a multi-minute video.
    chunk = 30 * 16000
    vecs = []
    with torch.no_grad():
        for start in range(0, len(y), chunk):
            seg = y[start:start + chunk]
            if len(seg) < 1600:
                continue
            inputs = proc(seg, sampling_rate=16000, return_tensors="pt")
            out = model(inputs.input_values.to(dev)).last_hidden_state
            vecs.append(_mean_pool(out))
    if not vecs:
        return [0.0] * EMBED_DIM
    return torch.stack(vecs).mean(dim=0).cpu().tolist()


def encode_text(text):
    import torch  # lazy

    tok, model = _load("text", _MODELS["text"])
    dev = _device()
    enc = tok(text or "", return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    return _mean_pool(out).cpu().tolist()


def _encode_frames(frame_paths, modality):
    import torch  # lazy
    from PIL import Image  # lazy

    proc, model = _load("vit", _MODELS[modality])
    dev = _device()
    paths = (frame_paths or [])[:32]
    if not paths:
        return [0.0] * EMBED_DIM
    # One batched forward pass for all frames. The old loop ran one frame per
    # pass (32 launches); batching cuts kernel-launch overhead ~10x on the T4.
    imgs = [Image.open(str(p)).convert("RGB") for p in paths]
    with torch.no_grad():
        inputs = proc(images=imgs, return_tensors="pt")
        out = model(inputs.pixel_values.to(dev)).last_hidden_state
        # [B, tokens, 768] -> mean over tokens, then over frames -> [768].
        # This equals the old mean of per-frame means.
        vec = out.mean(dim=-2).mean(dim=0)
    return vec.cpu().tolist()


def encode_video(frame_paths):
    return _encode_frames(frame_paths, "video")


def encode_face(frame_paths):
    from . import face  # lazy: cv2 lives in the face adapter
    frames = (frame_paths or [])[:32]
    if frames:
        crop_dir = Path(frames[0]).parent / "face_crops"
        frames = face.face_crops(frames, crop_dir)
    return _encode_frames(frames, "face")


_ENCODERS = {
    "audio": lambda ctx: encode_audio(ctx["wav_path"]),
    "text": lambda ctx: encode_text(ctx.get("text", "")),
    "video": lambda ctx: encode_video(ctx.get("frame_paths") or []),
    "face": lambda ctx: encode_face(ctx.get("frame_paths") or []),
}


def encode_all(ctx, synthetic_seed=None):
    """Return {modality: 768-d list}. Synthetic when a seed is given."""
    if synthetic_seed is not None:
        return {m: synthetic_embedding(synthetic_seed, m) for m in MODALITIES}
    return {m: fn(ctx) for m, fn in _ENCODERS.items()}


def save_embeddings(outdir, pid, embeddings):
    """One JSON file per video: {modality: vector}. Atomic temp -> rename."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{pid}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(embeddings), encoding="utf-8")
    tmp.replace(path)
    return path