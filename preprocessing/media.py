"""Media acquisition: YouTube download (yt-dlp) and decode (ffmpeg).

Heavy tools are imported lazily and run on Colab. The local sample mode does
not call these functions. Respect the YouTube terms of service and the D-Vlog
license before any bulk download.
"""
import shutil
import subprocess
from pathlib import Path

from . import config


def video_url(key):
    return f"https://www.youtube.com/watch?v={key}"


def _has_audio(path):
    """True when the file carries an audio stream (ffprobe at the boundary)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return "audio" in r.stdout


def download_video(key, raw_dir, proxy=None):
    """Download one video to raw_dir/<key>.mp4. Atomic + idempotent.

    proxy, when given, routes yt-dlp through a forward proxy (e.g. a tunnel to
    a trusted residential IP) so YouTube sees that IP instead of the Colab IP.

    A failed fetch never destroys a good cache: yt-dlp writes to a .part
    stem, and only a complete file with audio replaces the final path.

    The options come from yt-dlp's own CLI parser, not a hand-built dict.
    The parser fills the defaults that pass the YouTube player bot check;
    a minimal dict misses them and fails with LOGIN_REQUIRED.

    The web client forces SABR streaming (no format URLs). The mweb client
    still serves progressive format 18 (360p, video+audio), which downloads
    without a gvs PO token. Adaptive mweb streams 403, so prefer format 18.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{key}.mp4"
    if out.exists() and _has_audio(out):
        return out
    import yt_dlp  # lazy: Colab only
    part = raw_dir / f"{key}.part"
    for stale in raw_dir.glob(f"{key}.part*"):
        stale.unlink()
    argv = [
        "--cookies", str(config.YT_COOKIES),
        "--js-runtimes", f"node:{config.NODE_BIN}",
        "--remote-components", "ejs:github",
        "--extractor-args", "youtube:player_client=mweb",
        "--format", "18/b",
        "--merge-output-format", "mp4",
        "--no-warnings",
        video_url(key),
    ]
    if proxy:
        argv = ["--proxy", proxy] + argv
    parsed = yt_dlp.parse_options(argv)
    opts = next(x for x in parsed if isinstance(x, dict) and "format" in x)
    opts["outtmpl"] = str(part) + ".%(ext)s"
    opts["quiet"] = True
    opts["noprogress"] = True
    opts["sleep_interval"] = 3
    opts["max_sleep_interval"] = 7
    opts["sleep_interval_requests"] = 1
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video_url(key)])
    produced = sorted(raw_dir.glob(f"{key}.part.*"))
    if not produced:
        raise FileNotFoundError(f"download produced no file for {key}")
    final = produced[0]
    if not _has_audio(final):
        final.unlink()
        raise RuntimeError(f"download for {key} has no audio stream")
    final.replace(out)
    return out


def extract_media(video_path, frames_dir, fps=None, size=64, max_frames=32):
    """One ffmpeg decode -> WAV + JPG frames + raw grayscale plane stream.

    A single pass over the input feeds three outputs, replacing three separate
    decodes. The JPG output caps at max_frames (the GPU reads only those); the
    raw-gray output runs to end of input (the motion kernel needs every frame).
    Returns (wav_path, frame_paths, raw_gray_path). All are local intermediates,
    regenerable from the cached video, so they need not persist on Drive.
    """
    fps = fps or config.TARGET_FPS
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH")
    wav = frames_dir / (Path(video_path).stem + ".wav")
    gray = frames_dir / "motion.gray"
    pattern = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(config.AUDIO_SR), str(wav),
        "-vf", f"fps={fps}", "-frames:v", str(max_frames), pattern,
        "-vf", f"fps={fps},scale={size}:{size}",
        "-pix_fmt", "gray", "-f", "rawvideo", str(gray),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    return wav, frames, gray