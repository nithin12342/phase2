"""Face-region crops for the face encoder (a proxy until OpenFace is wired).

Purpose: give the face modality its own visual signal. The video encoder uses
full frames; the face encoder uses these crops, so the two embeddings differ.
A cv2 Haar cascade finds the largest face; with no detection, a center crop
biases toward the talking-head region typical of D-Vlog. cv2 is already a
pipeline dependency and is imported lazily (Colab only).
"""
from pathlib import Path

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        import cv2  # lazy: Colab only
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def _center_crop(img):
    h, w = img.shape[:2]
    ch, cw = int(h * 0.6), int(w * 0.6)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def face_crops(frame_paths, out_dir):
    """Write one face crop per input frame; return the crop paths in order.

    Frames that fail to decode are skipped. The crop dir lives under the local
    frames dir, so the pipeline's existing frame cleanup removes it too.
    """
    import cv2  # lazy: Colab only
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cas = _cascade()
    crops = []
    for i, p in enumerate(frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cas.detectMultiScale(gray, 1.1, 5)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(0.2 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            crop = img[y0:y1, x0:x1]
        else:
            crop = _center_crop(img)
        out = out_dir / f"face_{i:05d}.jpg"
        cv2.imwrite(str(out), crop)
        crops.append(out)
    return crops