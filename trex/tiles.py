"""Envelope pyramid tiles. Level L has buckets [b * 2**L, (b + 1) * 2**L) of steps; tile i holds
buckets [i * TILE, (i + 1) * TILE). Per non-empty bucket: min, max and mean of its finite values,
their mean runtime (tmean), mean step as an offset in bucket widths (soff), and count.

Encoding, little-endian:
    b"TKT2", i32 level, i64 index, u32 count, u32 0, u16 bucket[count] padded to 8,
    f32 min[count], max[count], mean[count], tmean[count], soff[count], u32 n[count]

A slab is one step range (level, index) of many runs at once, with each bucket's mean, mean step and count only:
    b"TKS2", i32 level, i64 index, u32 runs, u32 count, u32 path bytes, u32 0, the run paths in UTF-8 joined by NUL
    padded to 8, u32 first[runs + 1] (run i's buckets are [first[i], first[i + 1])) padded to 8,
    u16 bucket[count] padded to 8, u16 soff[count] (the mean step's offset in bucket widths, times 65536, rounded
    down) padded to 8, f32 mean[count], u32 n[count]
"""

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

TILE: Final = 256
MIN_LEVEL: Final = -20
MAX_LEVEL: Final = 62
MAGIC: Final = b"TKT2"
SLAB_MAGIC: Final = b"TKS2"
SOFF_SCALE: Final = 65536  # a slab's step offsets, in fractions of a bucket

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
    mantissa, exp = math.frexp(span / TILE)
    return max(MIN_LEVEL, min(MAX_LEVEL, exp - 1 if mantissa == 0.5 else exp))


def top_tiles(step_lo: float, step_hi: float) -> tuple[int, list[int]]:
    """(level, indices) of the finest level covering [step_lo, step_hi] in at most two tiles."""
    level = level_for(step_hi - step_lo)
    return level, list(covering(level, step_lo, step_hi))


def tile_range(level: int, index: int) -> tuple[float, float]:
    """[lo, hi) steps."""
    w = 2.0 ** level * TILE
    return index * w, (index + 1) * w


def build(steps: Floats, values: Floats, times: Floats, level: int, index: int) -> bytes:
    """Tile (level, index) of the points; NaN values are left out. A bucket's mean, mean step, mean runtime and count
    are of its finite values, or of its infinite ones when it has no finite value (the mean then infinite, NaN with
    both signs); its min and max are of all of them."""
    lo, hi = tile_range(level, index)
    keep = (steps >= lo) & (steps < hi) & ~np.isnan(values)
    s, v, t = steps[keep], values[keep], times[keep]
    if not s.size:
        e = np.empty(0)
        return _encode(level, index, np.empty(0, np.uint16), e, e, e, e, e, np.empty(0, np.int64))
    b = np.floor((s - lo) / 2.0 ** level).astype(np.int64)
    np.clip(b, 0, TILE - 1, out=b)
    order = np.argsort(b, kind="stable")
    b, s, v, t = b[order], s[order], v[order], t[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    bucket = b[starts]
    use = _finite_first(np.isfinite(v), starts)
    n = np.add.reduceat(use.astype(np.int64), starts)
    soff = np.add.reduceat(np.where(use, s - lo, 0.0), starts) / n / 2.0 ** level - bucket
    with np.errstate(invalid="ignore"):  # inf and -inf in one bucket make its mean NaN
        mean = np.add.reduceat(np.where(use, v, 0.0), starts) / n
    return _encode(level, index, bucket.astype(np.uint16), np.minimum.reduceat(v, starts), np.maximum.reduceat(v, starts),
                   mean, np.add.reduceat(np.where(use, t, 0.0), starts) / n, soff, n)


def _finite_first(finite: npt.NDArray[np.bool_], starts: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
    """Which items count toward their group's mean: its finite ones, or all of them when none is finite; groups are
    the runs of items from each of `starts`."""
    any_finite = np.maximum.reduceat(finite.astype(np.int8), starts) > 0
    return finite | np.repeat(~any_finite, np.diff(np.r_[starts, finite.size]))


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

    step = np.concatenate([t.mean_step for t in ts])[order]
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    means = joined("mean")
    n = np.where(_finite_first(np.isfinite(means), starts), joined("n"), 0.0)  # weights: as in `build`
    nn = np.add.reduceat(n, starts)
    bucket = a[starts]
    mn, mx = np.minimum.reduceat(joined("min"), starts), np.maximum.reduceat(joined("max"), starts)
    with np.errstate(invalid="ignore"):  # inf and -inf in one bucket make its mean NaN
        mean = np.add.reduceat(np.where(n > 0, means * n, 0.0), starts) / nn
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
    """The tile `blob` encodes; ValueError unless it is one."""
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


@dataclass(frozen=True, slots=True)
class Slab:
    """A decoded slab; arrays view its bytes."""

    level: int
    index: int
    paths: list[str]
    first: npt.NDArray[np.uint32]
    bucket: npt.NDArray[np.uint16]
    mean: npt.NDArray[np.float32]
    soff: Floats  # within half a 65536th of the bucket's mean step offset
    n: npt.NDArray[np.uint32]


type Ints = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Stack:
    """Every bucket of many runs' tiles, decoded once, in run then step order: its run (index into `paths`), index from
    step 0 at its tile's level, that level, mean, mean step offset and count."""

    paths: list[str]
    run: npt.NDArray[np.int32]
    a: Ints
    level: npt.NDArray[np.int8]
    mean: npt.NDArray[np.float32]
    soff: npt.NDArray[np.float32]
    n: npt.NDArray[np.uint32]

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.run, self.a, self.level, self.mean, self.soff, self.n))


def stack(runs: Sequence[tuple[str, Sequence[bytes]]]) -> Stack:
    """The Stack of runs given as [(path, tiles)] (each run's tiles in step order), read with numpy gathers over all
    their bytes at once, headers too."""
    paths = [p for p, _ in runs]
    blobs = [b for _, bs in runs for b in bs]
    if blobs:
        joined = b"".join(blobs)
        lens = np.fromiter(map(len, blobs), np.int64, len(blobs))
        at = (np.cumsum(lens) - lens) // 4  # each tile's start, in 4-byte words (tiles are 8-byte multiples)
        u32, i32 = np.frombuffer(joined, "<u4"), np.frombuffer(joined, "<i4")
        if np.any(u32[at] != int.from_bytes(MAGIC, "little")):
            raise ValueError("bad tile magic")
        run = np.repeat(np.arange(len(runs), dtype=np.int32), [len(bs) for _, bs in runs])
        lv, ix, counts = i32[at + 1], i32[at + 3].astype(np.int64) * 2 ** 32 + u32[at + 2], u32[at + 4].astype(np.int64)
        k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)  # position within its tile
        tile, c = np.repeat(at, counts), np.repeat(counts, counts)
        fields = tile + 6 + -(-2 * c // 8) * 2 + k  # a bucket's first f32 field, in words
        bucket = np.frombuffer(joined, "<u2")[2 * tile + 12 + k]
        return Stack(paths, np.repeat(run, counts), np.repeat(ix, counts) * TILE + bucket, np.repeat(lv, counts).astype(np.int8),
                     np.frombuffer(joined, "<f4")[fields + 2 * c], np.frombuffer(joined, "<f4")[fields + 4 * c], u32[fields + 5 * c])
    f = np.empty(0, np.float32)
    return Stack(paths, np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.int8), f, f, np.empty(0, np.uint32))


def stack_parts(st: Stack, pieces: int) -> list[Stack]:
    """`st` cut into about `pieces` stacks of whole runs (views; runs keep their indices into `st.paths`)."""
    cuts = np.searchsorted(st.run, np.linspace(0, len(st.paths), pieces + 1)[1:-1].astype(np.int32))
    bounds = [0, *cuts.tolist(), st.run.size]
    return [Stack(st.paths, st.run[a:b], st.a[a:b], st.level[a:b], st.mean[a:b], st.soff[a:b], st.n[a:b])
            for a, b in zip(bounds, bounds[1:]) if b > a]


class Buckets(NamedTuple):
    """Buckets of many runs at one level, as a slab holds them: run, bucket index (from step 0, or from a slab's start),
    mean, mean step offset (times SOFF_SCALE, rounded down) and count; in run then step order."""

    run: npt.NDArray[np.int32]
    bucket: Ints
    mean: npt.NDArray[np.float32]
    soff: npt.NDArray[np.uint16]
    n: npt.NDArray[np.uint32]


def slab_parts(st: Stack, level: int, index: int, runs: npt.NDArray[np.int32] | None = None) -> Buckets:
    """The buckets of slab (level, index) that the runs `runs` of `st` (all when None; their indices in `st.paths`)
    make: `cut` of `level_parts`."""
    return cut(level_parts(st, level, runs, index), index)


def level_parts(st: Stack, level: int, runs: npt.NDArray[np.int32] | None = None, index: int | None = None) -> Buckets:
    """The buckets the runs `runs` of `st` (all when None; their indices in `st.paths`) make at `level` (from step 0),
    in slab `index`'s step range only unless None, merged as `coarsen` merges buckets."""
    if runs is None:
        mine = np.ones(st.run.size, bool)
    else:
        take = np.zeros(len(st.paths), bool)
        take[runs] = True
        mine = take[st.run]
    if np.any(st.level[mine] > level):
        raise ValueError(f"a level {int(st.level[mine].max())} tile cannot make a level {level} slab")
    b = st.a >> np.maximum(level - st.level.astype(np.int64), 0)
    sel = np.flatnonzero(mine if index is None else mine & (b >= index * TILE) & (b < (index + 1) * TILE))
    run, b = st.run[sel], b[sel]
    if not sel.size:
        return Buckets(run, b, np.empty(0, np.float32), np.empty(0, np.uint16), np.empty(0, np.uint32))
    means, counts = st.mean[sel].astype(np.float64), st.n[sel].astype(np.float64)
    step = (st.a[sel] + st.soff[sel].astype(np.float64)) * 2.0 ** st.level[sel].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, (run[1:] != run[:-1]) | (b[1:] != b[:-1])])
    n = np.where(_finite_first(np.isfinite(means), starts), counts, 0.0)
    nn = np.add.reduceat(n, starts)
    bucket = b[starts]
    with np.errstate(invalid="ignore"):  # inf and -inf in one bucket make its mean NaN
        mean = np.add.reduceat(np.where(n > 0, means * n, 0.0), starts) / nn
    soff = np.add.reduceat(step * n, starts) / nn / 2.0 ** level - bucket
    return Buckets(run[starts], bucket, mean.astype(np.float32), np.clip(np.floor(soff * SOFF_SCALE), 0, SOFF_SCALE - 1).astype(np.uint16),
                   nn.astype(np.uint32))  # an offset rounded down stays inside its bucket


def cut(part: Buckets, index: int, take: npt.NDArray[np.bool_] | None = None) -> Buckets:
    """The buckets of `part` (from step 0, as level_parts) in slab `index`'s step range, of the runs `take` marks (all
    when None), counted from the slab's start."""
    keep = (part.bucket >= index * TILE) & (part.bucket < (index + 1) * TILE)
    sel = np.flatnonzero(keep if take is None else keep & take[part.run])
    return Buckets(part.run[sel], part.bucket[sel] - index * TILE, part.mean[sel], part.soff[sel], part.n[sel])


def join_buckets(parts: Sequence[Buckets], ordered: bool = False) -> Buckets:
    """The buckets of parts holding disjoint runs, in run order (as they come when `ordered`: each part's runs after
    the last part's)."""
    if len(parts) == 1:
        return parts[0]
    if not parts:
        return Buckets(np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.float32), np.empty(0, np.uint16), np.empty(0, np.uint32))
    run = np.concatenate([p.run for p in parts])
    order = slice(None) if ordered else np.argsort(run, kind="stable")
    return Buckets(run[order], np.concatenate([p.bucket for p in parts])[order], np.concatenate([p.mean for p in parts])[order],
                   np.concatenate([p.soff for p in parts])[order], np.concatenate([p.n for p in parts])[order])


def slab(runs: Sequence[tuple[str, Sequence[bytes]]], level: int, index: int) -> bytes:
    """Slab (level, index) of runs given as [(path, tiles)], each run's tiles of one level at or below `level`: every
    run's buckets of that step range, merged as `coarsen` merges them."""
    return encode_slab(level, index, [p for p, _ in runs], [slab_parts(stack(runs), level, index)])


def encode_slab(level: int, index: int, paths: list[str], parts: Sequence[Buckets]) -> bytes:
    """Slab (level, index) of runs `paths` from parts holding disjoint runs (their buckets from the slab's start, runs
    as indices into `paths`)."""
    b = join_buckets(parts)
    names = "\0".join(paths).encode()
    pos, offs = b.bucket.astype("<u2").tobytes(), b.soff.astype("<u2").tobytes()
    firsts = np.searchsorted(b.run, np.arange(len(paths) + 1)).astype("<u4").tobytes()
    return b"".join([
        SLAB_MAGIC, struct.pack("<iqIIII", level, index, len(paths), b.run.size, len(names), 0), names, b"\0" * (-len(names) % 8),
        firsts, b"\0" * (-len(firsts) % 8), pos, b"\0" * (-len(pos) % 8), offs, b"\0" * (-len(offs) % 8),
        b.mean.astype("<f4").tobytes(), b.n.astype("<u4").tobytes(),
    ])


def decode_slab(blob: bytes) -> Slab:
    """The slab `blob` encodes; ValueError unless it is one."""
    if blob[:4] != SLAB_MAGIC:
        raise ValueError("bad slab magic")
    level, index, runs, count, nbytes, _ = struct.unpack_from("<iqIIII", blob, 4)
    paths = blob[32:32 + nbytes].decode().split("\0") if runs else []
    off = 32 + -(-nbytes // 8) * 8
    first = np.frombuffer(blob, np.uint32, runs + 1, off)
    off += -(-4 * (runs + 1) // 8) * 8
    bucket = np.frombuffer(blob, np.uint16, count, off)
    off += -(-2 * count // 8) * 8
    soff = (np.frombuffer(blob, np.uint16, count, off).astype(np.float64) + 0.5) / SOFF_SCALE
    off += -(-2 * count // 8) * 8
    if off + 8 * count != len(blob):
        raise ValueError("slab length mismatch")
    return Slab(level, index, paths, first, bucket, np.frombuffer(blob, np.float32, count, off), soff,
                np.frombuffer(blob, np.uint32, count, off + 4 * count))
