"""Envelope pyramid tiles. Level L has buckets [b * 2**L, (b + 1) * 2**L) of steps; tile i holds
buckets [i * TILE, (i + 1) * TILE). Per non-empty bucket: min, max and mean of its finite values,
their mean runtime (tmean), mean step as an offset in bucket widths (soff), and count.

Encoding, little-endian:
    b"TKT2", i32 level, i64 index, u32 count, u32 0, u16 bucket[count] padded to 8,
    f32 min[count], max[count], mean[count], tmean[count], soff[count], u32 n[count]
"""

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

TILE: Final = 256
MIN_LEVEL: Final = -20
MAX_LEVEL: Final = 62
MAGIC: Final = b"TKT2"

type Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Tile:
    """A decoded tile; arrays view its bytes."""

    level: int
    index: int
    bucket: npt.NDArray[np.uint16]
    min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]
    mean: npt.NDArray[np.float32]
    tmean: npt.NDArray[np.float32]
    soff: npt.NDArray[np.float32]
    n: npt.NDArray[np.uint32]

    @property
    def mean_step(self) -> Floats:
        """Per bucket."""
        return (self.index * TILE + self.bucket.astype(np.float64) + self.soff.astype(np.float64)) * 2.0 ** self.level


def level_for(span: float) -> int:
    """Smallest level whose tiles are at least `span` steps wide."""
    if not span > 0:
        return 0
    return max(MIN_LEVEL, min(MAX_LEVEL, math.ceil(math.log2(span / TILE))))


def top_tiles(step_lo: float, step_hi: float) -> tuple[int, list[int]]:
    """(level, indices) of the finest level covering [step_lo, step_hi] in at most two tiles."""
    level = level_for(step_hi - step_lo)
    while len(covering(level, step_lo, step_hi)) > 2 and level < MAX_LEVEL:
        level += 1
    return level, list(covering(level, step_lo, step_hi))


def tile_range(level: int, index: int) -> tuple[float, float]:
    """[lo, hi) steps."""
    w = 2.0 ** level * TILE
    return index * w, (index + 1) * w


def build(steps: Floats, values: Floats, times: Floats, level: int, index: int) -> bytes:
    """Tile (level, index) of the points; non-finite values are left out."""
    lo, hi = tile_range(level, index)
    keep = (steps >= lo) & (steps < hi) & np.isfinite(values)
    s, v, t = steps[keep], values[keep], times[keep]
    if not s.size:
        e = np.empty(0)
        return _encode(level, index, np.empty(0, np.uint16), e, e, e, e, e, np.empty(0, np.int64))
    b = np.floor((s - lo) / 2.0 ** level).astype(np.int64)
    np.clip(b, 0, TILE - 1, out=b)
    order = np.argsort(b, kind="stable")
    b, s, v, t = b[order], s[order], v[order], t[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    n = np.diff(np.r_[starts, b.size])
    bucket = b[starts]
    soff = np.add.reduceat(s - lo, starts) / n / 2.0 ** level - bucket
    return _encode(level, index, bucket.astype(np.uint16), np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts),
                   np.add.reduceat(v, starts) / n, np.add.reduceat(t, starts) / n, soff, n)


def coarsen(blobs: Sequence[bytes], up: int) -> list[bytes]:
    """The tiles `up` levels above the given ones of a level."""
    ts = [decode(b) for b in blobs]
    if up <= 0 or not ts:
        return list(blobs)
    level = ts[0].level + up
    if level > MAX_LEVEL:
        raise ValueError(f"tile level {level} out of range")
    a = np.concatenate([t.index * TILE + t.bucket.astype(np.int64) for t in ts]) >> up
    if not a.size:
        e = np.empty(0)
        return [_encode(level, ts[0].index >> up, np.empty(0, np.uint16), e, e, e, e, e, np.empty(0, np.int64))]
    order = np.argsort(a, kind="stable")
    a = a[order]

    def joined(field: str) -> Floats:
        return np.concatenate([getattr(t, field) for t in ts]).astype(np.float64)[order]

    n = joined("n")
    step = np.concatenate([t.mean_step for t in ts])[order]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    nn = np.add.reduceat(n, starts)
    bucket = a[starts]
    mn, mx = np.minimum.reduceat(joined("min"), starts), np.maximum.reduceat(joined("max"), starts)
    mean = np.add.reduceat(joined("mean") * n, starts) / nn
    tmean = np.add.reduceat(joined("tmean") * n, starts) / nn
    soff = np.add.reduceat(step * n, starts) / nn / 2.0 ** level - bucket
    out: list[bytes] = []
    for index in np.unique(bucket // TILE):
        sel = bucket // TILE == index
        out.append(_encode(level, int(index), (bucket[sel] - index * TILE).astype(np.uint16), mn[sel], mx[sel], mean[sel],
                           tmean[sel], soff[sel], nn[sel]))
    return out


def _encode(level: int, index: int, bucket: npt.NDArray[np.uint16], mn: Floats, mx: Floats, mean: Floats, tmean: Floats,
            soff: Floats, n: npt.NDArray[np.int64] | Floats) -> bytes:
    pos = bucket.astype("<u2").tobytes()
    return b"".join([
        MAGIC, struct.pack("<iqII", level, index, bucket.size, 0), pos, b"\0" * (-len(pos) % 8),
        *(x.astype("<f4").tobytes() for x in (mn, mx, mean, tmean, soff)), n.astype("<u4").tobytes(),
    ])


def decode(blob: bytes) -> Tile:

    if blob[:4] != MAGIC:
        raise ValueError("bad tile magic")
    level, index, count, _ = struct.unpack_from("<iqII", blob, 4)
    off = 24 + -(-2 * count // 8) * 8
    if off + 24 * count != len(blob):
        raise ValueError("tile length mismatch")
    mn, mx, mean, tmean, soff = (np.frombuffer(blob, np.float32, count, off + 4 * k * count) for k in range(5))
    return Tile(level, index, np.frombuffer(blob, np.uint16, count, 24), mn, mx, mean, tmean, soff,
                np.frombuffer(blob, np.uint32, count, off + 20 * count))


def covering(level: int, step_lo: float, step_hi: float) -> range:
    """Indices of the tiles of `level` covering [step_lo, step_hi]."""
    w = 2.0 ** level * TILE
    return range(math.floor(step_lo / w), math.floor(step_hi / w) + 1)
