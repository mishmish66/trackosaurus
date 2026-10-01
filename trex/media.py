"""Encoding logged arrays as PNG and MP4."""

import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _uint8(arr: ArrayLike) -> NDArray[np.uint8]:
    """Floats are scaled from [0, 1]."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float64) * (255.0 if a.dtype.kind == "f" else 1.0), 0, 255).astype(np.uint8)
    return a


def encode_png(arr: ArrayLike) -> bytes:
    """HW, HWC or CHW with C in 1-4."""
    a = _uint8(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    h, w = a.shape[:2]
    c = 1 if a.ndim == 2 else a.shape[2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), np.ascontiguousarray(a).reshape(h, w * c)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, {1: 0, 2: 4, 3: 2, 4: 6}[c], 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b"")


def encode_mp4(frames: ArrayLike, fps: float) -> bytes:
    """THW or THWC with C in 1 or 3; needs ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("log_video with an array needs ffmpeg on PATH")
    a = _uint8(frames)
    if a.ndim == 3:
        a = a[..., None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    _, h, w, _ = a.shape
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", f.name],
            input=np.ascontiguousarray(a[..., :3]).tobytes(), check=True,
        )
        return Path(f.name).read_bytes()


def sniff_image(b: bytes | bytearray) -> str:
    if b.startswith(b"\x89PNG"):
        return "png"
    if b.startswith(b"\xff\xd8"):
        return "jpg"
    if b.startswith(b"GIF8"):
        return "gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b.lstrip().startswith((b"<svg", b"<?xml")):
        return "svg"
    raise ValueError("unrecognized image bytes")
