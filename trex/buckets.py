"""Bucket arrays: the one form metric data takes between run files and charts.

At level L, bucket b holds steps [b * 2**L, (b + 1) * 2**L). A bucket array holds some runs' buckets of one metric at
one level, each run's in step order. A bucket keeps the mean of its finite values (of its infinities when it has none:
infinite, NaN with both signs), the mean step of those values as an offset into the bucket, their mean runtime and
their count; NaN is no value. Requests and caches deal in blocks: block i of a level is its buckets
[i * BLOCK, (i + 1) * BLOCK).

Every view is made of three operations: `bucketize` (rows to buckets), `merge` (buckets to coarser ones, weighted by
their counts, as `bucketize` would make them from the rows) and `cut` (a range of buckets, or some of the runs).

Encoding, little-endian:
    b"TKB1", i32 level, i64 base (a block), u32 runs, u32 count, u32 path bytes, u32 0,
    the run paths in UTF-8 joined by NUL, padded to 8,
    u32 first[runs + 1] (run i's buckets are [first[i], first[i + 1])), padded to 8,
    u32 seq[runs] (the rows of each run its buckets hold), padded to 8,
    u16 offset[count] (bucket - base * BLOCK), padded to 8,
    u16 soff[count] (the mean step's offset in bucket widths, times SOFF_SCALE, rounded down), padded to 8,
    f32 mean[count], f32 tmean[count], u32 n[count]
"""

import math
import struct
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

BLOCK: Final = 256
MIN_LEVEL: Final = -20
MAX_LEVEL: Final = 62
MAGIC: Final = b"TKB1"
SOFF_SCALE: Final = 65536
HEAD: Final = struct.Struct("<4siqIIII")

type Ints = npt.NDArray[np.int64]
type Floats = npt.NDArray[np.float64]


class Buckets(NamedTuple):
    """Buckets at one level, in run then bucket order: run (an index into the runs they belong to), bucket (from step
    0), mean, mean step offset (times SOFF_SCALE, rounded down), mean runtime and count."""

    run: npt.NDArray[np.int32]
    bucket: Ints
    mean: npt.NDArray[np.float32]
    soff: npt.NDArray[np.uint16]
    tmean: npt.NDArray[np.float32]
    n: npt.NDArray[np.uint32]

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in self)

    def step(self, level: int) -> Floats:
        """Each bucket's mean step."""
        return (self.bucket + (self.soff + 0.5) / SOFF_SCALE) * 2.0 ** level


class BucketArray(NamedTuple):
    """A decoded bucket array: its level, its runs' paths, the rows of each its buckets hold, and the buckets."""

    level: int
    paths: list[str]
    seq: npt.NDArray[np.uint32]
    buckets: Buckets


class Stack(NamedTuple):
    """Buckets of many runs, each at a level of its own: the runs' paths, rows held and levels, and the buckets (each
    at its run's level)."""

    paths: list[str]
    seq: npt.NDArray[np.uint32]
    level: npt.NDArray[np.int8]
    buckets: Buckets

    def at(self, level: int, runs: npt.NDArray[np.int32] | None = None) -> Buckets:
        """The buckets of the runs `runs` (all when None) merged to `level`; ValueError when one is finer."""
        b = self.buckets
        if runs is not None:
            take = np.zeros(len(self.paths), bool)
            take[runs] = True
            b = select(b, take[b.run])
        return merge(b, level - self.level[b.run].astype(np.int64))


def empty() -> Buckets:
    return Buckets(np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.uint16),
                   np.empty(0, np.float32), np.empty(0, np.uint32))


def select(b: Buckets, keep: npt.NDArray[np.bool_]) -> Buckets:
    return take(b, np.flatnonzero(keep))


def take(b: Buckets, i: npt.NDArray[np.integer]) -> Buckets:
    """The buckets at positions `i` of `b`."""
    return Buckets(b.run[i], b.bucket[i], b.mean[i], b.soff[i], b.tmean[i], b.n[i])


# ---- levels and blocks ----


def level_for(span: float) -> int:
    """The finest level whose blocks are at least `span` steps wide."""
    if not span > 0:
        return 0
    mantissa, exp = math.frexp(span / BLOCK)
    return max(MIN_LEVEL, min(MAX_LEVEL, exp - 1 if mantissa == 0.5 else exp))


def blocks(level: int, lo: float, hi: float) -> range:
    """The blocks of `level` covering steps [lo, hi]."""
    w = 2.0 ** level * BLOCK
    return range(math.floor(lo / w), math.floor(hi / w) + 1)


def block_range(level: int, index: int) -> tuple[float, float]:
    """[lo, hi) steps of block `index` of `level`."""
    w = 2.0 ** level * BLOCK
    return index * w, (index + 1) * w


# ---- the operations ----


def bucketize(steps: Floats, values: Floats, times: Floats, level: int) -> Buckets:
    """One run's rows (in any order) as buckets of `level`."""
    keep = ~np.isnan(values)
    s, v, t = steps[keep], values[keep], times[keep]
    if not s.size:
        return empty()
    w = 2.0 ** level
    b = np.floor(s / w).astype(np.int64)
    order = np.argsort(b, kind="stable")
    b, s, v, t = b[order], s[order], v[order], t[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    use = _finite_first(np.isfinite(v), starts)
    n = np.add.reduceat(use.astype(np.int64), starts)
    bucket = b[starts]
    with np.errstate(invalid="ignore"):  # inf and -inf in one bucket make its mean NaN
        mean = np.add.reduceat(np.where(use, v, 0.0), starts) / n
    pos = np.add.reduceat(np.where(use, s / w, 0.0), starts) / n
    return Buckets(np.zeros(bucket.size, np.int32), bucket, mean.astype(np.float32), _quantized(pos - bucket),
                   (np.add.reduceat(np.where(use, t, 0.0), starts) / n).astype(np.float32), n.astype(np.uint32))


def merge(b: Buckets, up: int | Ints) -> Buckets:
    """`b` made `up` levels coarser (per bucket when an array, the same within a run): the buckets falling in one are
    merged as `bucketize` merges rows, each weighted by its count; ValueError when `up` is negative."""
    shift = np.asarray(up, np.int64)
    if np.any(shift < 0):
        raise ValueError("a merge cannot make buckets finer")
    if not b.run.size:
        return b
    scale = np.ldexp(1.0, -shift)
    nb = b.bucket >> shift
    pos = (b.bucket + (b.soff + 0.5) / SOFF_SCALE) * scale
    starts = np.flatnonzero(np.r_[True, (b.run[1:] != b.run[:-1]) | (nb[1:] != nb[:-1])])
    means = b.mean.astype(np.float64)
    w = np.where(_finite_first(np.isfinite(means), starts), b.n.astype(np.float64), 0.0)
    nn = np.add.reduceat(w, starts)
    bucket = nb[starts]
    with np.errstate(invalid="ignore"):  # inf and -inf in one bucket make its mean NaN
        mean = np.add.reduceat(np.where(w > 0, means * w, 0.0), starts) / nn
    return Buckets(b.run[starts], bucket, mean.astype(np.float32), _quantized(np.add.reduceat(pos * w, starts) / nn - bucket),
                   (np.add.reduceat(b.tmean * w, starts) / nn).astype(np.float32), nn.astype(np.uint32))


def cut(b: Buckets, lo: int, hi: int, take: npt.NDArray[np.bool_] | None = None) -> Buckets:
    """The buckets of `b` in [lo, hi) of the runs `take` marks (all when None)."""
    keep = (b.bucket >= lo) & (b.bucket < hi)
    return select(b, keep if take is None else keep & take[b.run])


def union(parts: Sequence[Buckets]) -> Buckets:
    """The buckets of parts holding different runs, in run order."""
    if len(parts) == 1:
        return parts[0]
    if not parts:
        return empty()
    run = np.concatenate([p.run for p in parts])
    order = np.argsort(run, kind="stable")
    return Buckets(*(np.concatenate(field)[order] for field in zip(*parts, strict=True)))


def _finite_first(finite: npt.NDArray[np.bool_], starts: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
    """Which items count toward their group's mean: its finite ones, or all of them when none is finite; groups are
    the runs of items from each of `starts`."""
    any_finite = np.maximum.reduceat(finite.astype(np.int8), starts) > 0
    return finite | np.repeat(~any_finite, np.diff(np.r_[starts, finite.size]))


def _quantized(frac: Floats) -> npt.NDArray[np.uint16]:
    """Offsets in [0, 1) bucket widths as soff: rounded down, so they stay inside their bucket."""
    return np.clip(np.floor(frac * SOFF_SCALE), 0, SOFF_SCALE - 1).astype(np.uint16)


# ---- encoding ----


def _pad(n: int) -> int:
    return -(-n // 8) * 8


def encode(level: int, base: int, paths: Sequence[str], seq: Sequence[int] | npt.NDArray[np.integer], b: Buckets) -> bytes:
    """Bucket array of runs `paths` (holding rows `seq`) at `level`, of buckets `b` (runs as indices into `paths`),
    all in [base * BLOCK, base * BLOCK + 65536)."""
    names = "\0".join(paths).encode()
    offset = b.bucket - base * BLOCK
    if offset.size and (offset.min() < 0 or offset.max() >= 65536):
        raise ValueError("buckets outside the array's range")
    first = np.searchsorted(b.run, np.arange(len(paths) + 1)).astype("<u4")
    parts = [HEAD.pack(MAGIC, level, base, len(paths), b.run.size, len(names), 0), names, b"\0" * (-len(names) % 8)]
    for a in (first, np.asarray(seq, "<u4"), offset.astype("<u2"), b.soff.astype("<u2")):
        parts += [a.tobytes(), b"\0" * (-a.nbytes % 8)]
    return b"".join([*parts, b.mean.astype("<f4").tobytes(), b.tmean.astype("<f4").tobytes(), b.n.astype("<u4").tobytes()])


def decode(blob: bytes) -> BucketArray:
    """The bucket array `blob` encodes; ValueError unless it is one."""
    if blob[:4] != MAGIC:
        raise ValueError("bad bucket array magic")
    _, level, base, runs, count, nbytes, _ = HEAD.unpack_from(blob)
    paths = blob[32:32 + nbytes].decode().split("\0") if runs else []
    at = 32 + _pad(nbytes)
    first = np.frombuffer(blob, np.uint32, runs + 1, at)
    at += _pad(4 * (runs + 1))
    seq = np.frombuffer(blob, np.uint32, runs, at)
    at += _pad(4 * runs)
    offset = np.frombuffer(blob, np.uint16, count, at)
    soff = np.frombuffer(blob, np.uint16, count, at + _pad(2 * count))
    at += 2 * _pad(2 * count)
    if at + 12 * count != len(blob) or len(paths) != runs or (runs and first[-1] != count):
        raise ValueError("bucket array length mismatch")
    run = np.repeat(np.arange(runs, dtype=np.int32), np.diff(first.astype(np.int64)))
    return BucketArray(level, paths, seq, Buckets(run, base * BLOCK + offset.astype(np.int64), np.frombuffer(blob, np.float32, count, at),
                                             soff, np.frombuffer(blob, np.float32, count, at + 4 * count),
                                             np.frombuffer(blob, np.uint32, count, at + 8 * count)))


def stack(paths: Sequence[str], blobs: Sequence[bytes]) -> Stack:
    """The Stack of runs `paths` from each one's one-run bucket array, read with numpy gathers over all their bytes at
    once (headers too); ValueError unless each is one."""
    if not blobs:
        return Stack(list(paths), np.empty(0, np.uint32), np.empty(0, np.int8), empty())
    joined = b"".join(blobs)
    lens = np.fromiter(map(len, blobs), np.int64, len(blobs))
    at = np.cumsum(lens) - lens  # each array's start, in bytes (arrays are 8-byte multiples)
    u32 = np.frombuffer(joined, "<u4")
    w = at // 4
    if np.any(u32[w] != int.from_bytes(MAGIC, "little")) or np.any(u32[w + 4] != 1) or np.any(u32[w + 6] != 0):
        raise ValueError("not one-run bucket arrays")
    counts = u32[w + 5].astype(np.int64)
    if np.any(lens != 48 + 2 * (-(-2 * counts // 8) * 8) + 12 * counts):
        raise ValueError("bucket array length mismatch")
    i32 = np.frombuffer(joined, "<i4")
    level, base = i32[w + 1], i32[w + 3].astype(np.int64) * 2 ** 32 + u32[w + 2]
    k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)  # position within its array
    c, start = np.repeat(counts, counts), np.repeat(at, counts)
    offs = start + 48  # each bucket's array's offsets, in bytes
    f32at = (offs + 2 * (-(-2 * c // 8) * 8)) // 4 + k  # its mean, in words
    u16 = np.frombuffer(joined, "<u2")
    f32 = np.frombuffer(joined, "<f4")
    b = Buckets(np.repeat(np.arange(len(blobs), dtype=np.int32), counts), np.repeat(base, counts) * BLOCK + u16[offs // 2 + k],
                f32[f32at], u16[(offs + -(-2 * c // 8) * 8) // 2 + k], f32[f32at + c], u32[f32at + 2 * c])
    return Stack(list(paths), u32[w + 10].copy(), level.astype(np.int8), b)


# ---- one run ----


def kept(steps: Floats, values: Floats, times: Floats, seq: int) -> bytes:
    """A run's rows `seq` (steps, values, runtimes) as the bucket array it keeps: at the finest level whose blocks are
    as wide as its steps (`level_for`), which then lie in at most two of them."""
    if not steps.size:
        return encode(0, 0, [""], [seq], empty())
    level = level_for(float(steps.max() - steps.min()))
    b = bucketize(steps, values, times, level)
    return encode(level, int(b.bucket[0]) // BLOCK if b.run.size else 0, [""], [seq], b)


def built(steps: Floats, values: Floats, times: Floats, seq: int, level: int, index: int) -> bytes:
    """Block `index` of `level` of a run's rows `seq` (steps, values, runtimes), as a one-run bucket array."""
    lo, hi = block_range(level, index)
    keep = (steps >= lo) & (steps < hi)
    b = cut(bucketize(steps[keep], values[keep], times[keep], level), index * BLOCK, (index + 1) * BLOCK)
    return encode(level, index, [""], [seq], b)
