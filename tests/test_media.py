import shutil
import struct
import zlib

import numpy as np
import numpy.typing as npt
import pytest

from trex.media import encode_mp4, encode_png, sniff_image

CHANNELS = {0: 1, 4: 2, 2: 3, 6: 4}  # PNG color type -> channels


def decode_png(b: bytes) -> tuple[int, npt.NDArray[np.uint8]]:
    """(color type, HxWxC uint8 pixels) of an unfiltered 8-bit PNG, checking every chunk's CRC."""
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    parts, i = dict[bytes, bytes](), 8
    while i < len(b):
        n = int.from_bytes(b[i:i + 4], "big")
        tag, data = b[i + 4:i + 8], b[i + 8:i + 8 + n]
        assert zlib.crc32(tag + data) == int.from_bytes(b[i + 8 + n:i + 12 + n], "big")
        parts[tag], i = data, i + 12 + n
    assert b"IEND" in parts
    w, h, depth, ctype = struct.unpack(">IIBB", parts[b"IHDR"][:10])
    c = CHANNELS[ctype]
    raw = np.frombuffer(zlib.decompress(parts[b"IDAT"]), np.uint8).reshape(h, 1 + w * c)
    assert depth == 8 and not raw[:, 0].any()
    return ctype, raw[:, 1:].reshape(h, w, c)


def pixels(shape: tuple[int, ...], seed: int = 0) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize("shape,ctype", [((5, 7), 0), ((5, 7, 1), 0), ((5, 7, 2), 4), ((5, 7, 3), 2), ((5, 7, 4), 6)])
def test_png_round_trips_hw_and_hwc_images(shape: tuple[int, ...], ctype: int) -> None:
    a = pixels(shape)
    got_type, got = decode_png(encode_png(a))
    assert got_type == ctype and np.array_equal(got, a.reshape(5, 7, -1))


@pytest.mark.parametrize("c", [1, 3, 4])
def test_png_accepts_channels_first_images(c: int) -> None:
    a = pixels((c, 5, 7))
    assert np.array_equal(decode_png(encode_png(a))[1], a.transpose(1, 2, 0))


def test_png_scales_floats_from_unit_range_and_clips_out_of_range_values() -> None:
    _, got = decode_png(encode_png(np.array([[0.0, 0.5, 1.0, -1.0, 2.0]])))
    assert got[0, :, 0].tolist() == [0, 127, 255, 0, 255]
    _, got = decode_png(encode_png(np.array([[-5, 7, 300]])))
    assert got[0, :, 0].tolist() == [0, 7, 255]


@pytest.mark.parametrize("data,ext", [
    (b"\x89PNG\r\n\x1a\n...", "png"), (b"\xff\xd8\xff\xe0", "jpg"), (b"GIF89a", "gif"),
    (b"RIFF\0\0\0\0WEBPVP8 ", "webp"), (b"  <svg xmlns='x'/>", "svg"), (b"<?xml version='1.0'?><svg/>", "svg"),
])
def test_image_formats_are_recognized_by_their_bytes(data: bytes, ext: str) -> None:
    assert sniff_image(data) == ext


def test_unrecognized_image_bytes_are_rejected() -> None:
    with pytest.raises(ValueError):
        sniff_image(b"hello")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("shape", [(4, 9, 7), (4, 9, 7, 1), (4, 9, 7, 3)])
def test_frame_arrays_encode_to_mp4_including_gray_and_odd_sizes(shape: tuple[int, ...]) -> None:
    mp4 = encode_mp4(pixels(shape), fps=10)
    assert mp4[4:8] == b"ftyp" and b"moov" in mp4


def test_mp4_without_ffmpeg_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(cmd: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", which)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        encode_mp4(pixels((2, 4, 4)), fps=10)
