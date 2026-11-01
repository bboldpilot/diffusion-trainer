"""K4-repair (SN56_SCALE_REPAIR=1, default off): blind detection + undo of the validator's weight_scaling.

Validator prep (G.O.D trainer/model_prep/augmentation.py) multiplies a random 25-75% of the eligible tensors
("weight" in the name, no "norm"/"embed"; named_parameters order) by ONE s in U(0.5, 1.5):
w' = RNE(fp32(w) * fp32(s)) in the stored dtype. Re-rounding a scaled bf16/f16 grid leaves mantissa codes with
no preimage; the empty-code set depends only on (s, mantissa), so each tensor's mantissa histogram identifies
both the scaled set and s ("lattice" detector). Measured on the two real cases, CPU, from the augmented weights
only (offline replay): a Qwen2.5-1.5B task, bf16 76/76 tensors, P=R=1, s_hat 1.1483388
(true 1.1482946, rel 3.9e-5); a bloomz-560m task f16 39/39 incl. ln_f, s_hat rel err 2.4e-7; both clean bases
-> nothing detected. The s vs s/2 alias is resolved by the log-RMS offset of detected tensors vs the per-family
layer trend of the untouched ones.

At load (before the LR search) the detected weights are restored: W <- RNE(fp32(W) / s_hat) in the model dtype,
one rounding away from the clean value. Nothing else changes; |log s_hat| < 0.00995 (|s-1| < ~1%) is left alone,
as is fp32 storage (no lattice) and any doubt (no detection). Not used on KL tasks: their KL reference is the
augmented base, which a repair would pay for (levers-1028-text.md K3).

The detector functions below are copied VERBATIM (ast segments; only the `K.` module prefix dropped) from the
offline detection scripts (k4common.py, k4_detect.py); the sampling plan (64 chunks of 512
elements per matrix, whole tensor up to 16384) is the one measured there. Reads the local safetensors directly
(header offsets + seek), so the stored dtype is what is inspected even when the model loads in another dtype.
"""

import json
import math
import os
import struct
import zlib
from pathlib import Path

import numpy as np

S_LO, S_HI = 0.5, 1.5        # upstream intensity support for weight_scaling
U_TOL = 1e-4                 # a tensor is "consistent with s" if its unreachable fraction <= U_TOL
MIN_E0 = 20.0                # informative only if >= 20 values would be expected on forbidden codes untouched
N_CHUNKS, CHUNK = 64, 512    # 32768 elements per matrix
WHOLE_MAX = 16384            # tensors up to this many elements are read whole
MIN_ABS_LOG_S = 0.00995      # |s - 1| below ~1%: not worth touching
_FCACHE = {}

# ---- verbatim from the offline k4_detect scripts (do not edit; re-copy) ----


def plan(name, shape):
    """Deterministic sample plan: list of (start_elem, n_elem). Same plan for aug and clean of a tensor."""
    N = int(np.prod(shape))
    if N <= WHOLE_MAX:
        return [(0, N)]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    K = N_CHUNKS
    if len(shape) == 2 and shape[1] >= CHUNK:
        rows, cols = shape
        r = ((np.arange(K) + rng.random(K)) * rows / K).astype(np.int64)      # stratified rows
        c = rng.integers(0, cols - CHUNK + 1, K)                               # random column offset
        st = r * cols + c
    else:
        st = ((np.arange(K) + rng.random(K)) * (N - CHUNK) / K).astype(np.int64)
    return [(int(s), CHUNK) for s in sorted(st)]


def decode(u16, dtype):
    if dtype == 'BF16':
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return u16.view(np.float16).astype(np.float32)


def to_fmt_bits(x32, dtype):
    """Round float32 -> bf16/f16 with round-to-nearest-even (what torch .to(bfloat16/float16) does); uint16 bits."""
    x32 = np.asarray(x32, dtype=np.float32)
    if dtype == 'BF16':
        u = x32.view(np.uint32).astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return x32.astype(np.float16).view(np.uint16)


def family(name):
    """(family, layer) for per-layer matrices, else (name, -1)."""
    parts = name.split('.')
    if 'layers' in parts or 'h' in parts:
        i = parts.index('layers') if 'layers' in parts else parts.index('h')
        return parts[-2], int(parts[i + 1])
    return name, -1


def nbits(dt):
    return 7 if dt == 'BF16' else 10


def mant_hist(u16, dt):
    """Mantissa-code histogram of normal values (zeros/subnormals/near-subnormals/inf excluded)."""
    if dt == 'BF16':
        e, m, ok_lo, ok_hi = (u16 >> 7) & 0xFF, u16 & 0x7F, 2, 0xFD
    else:
        e, m, ok_lo, ok_hi = (u16 >> 10) & 0x1F, u16 & 0x3FF, 2, 0x1D
    ok = (e >= ok_lo) & (e <= ok_hi)      # e >= 2 keeps every preimage v/s (s <= 1.5) normal for f16
    return np.bincount(m[ok].astype(np.int64), minlength=1 << nbits(dt)).astype(np.float64)


def reach(S, dt):
    """R[i, m] = mantissa code m (value 1 + m/2^p) has a preimage x on the grid with RNE(fp32(x)*fp32(S[i])) == it."""
    L = 1 << nbits(dt)
    S = np.asarray(S, np.float32)
    y = (1 + np.arange(L) / L).astype(np.float32)
    yb = to_fmt_bits(y, dt)
    x0 = to_fmt_bits((y[None, :].astype(np.float64) / S[:, None]).astype(np.float32), dt).astype(np.int32)
    R = np.zeros((len(S), L), bool)
    for k in range(-3, 4):  # preimage is within +-2 grid steps of RNE(y/s); +-3 for safety
        xb = (x0 + k).astype(np.uint16)
        R |= to_fmt_bits(decode(xb, dt) * S[:, None], dt) == yb[None, :]
    return R


def forbidden(S, dt, cache=False):
    """Packed (~reach) for a grid S; the coarse grid is cached (it does not depend on the data)."""
    key = (dt, len(S), float(S[0]), float(S[-1]))
    if key in _FCACHE:
        return _FCACHE[key]
    F = np.concatenate([np.packbits(~reach(S[a:a + 20000], dt), axis=1) for a in range(0, len(S), 20000)])
    if cache:
        _FCACHE[key] = F
    return F


def unreach(S, H, dt, Hs=None, chunk=4000, cache=False):
    """U[i, t] = fraction of tensor t's values on codes unreachable under S[i];
    E[i, t] = the same fraction expected if t were untouched (smoothed histogram Hs); nf[i] = #unreachable codes."""
    L = H.shape[1]
    n = H.sum(1)
    Hs = H if Hs is None else Hs
    Fp = forbidden(np.asarray(S, np.float32), dt, cache)
    U, E = np.empty((len(S), len(H))), np.empty((len(S), len(H)))
    nf = np.empty(len(S), np.int64)
    Ht, Hst = H.T.astype(np.float32), Hs.T.astype(np.float32)
    for a in range(0, len(S), chunk):
        F = np.unpackbits(Fp[a:a + chunk], axis=1, count=L).astype(np.float32)
        U[a:a + chunk] = (F @ Ht) / n
        E[a:a + chunk] = (F @ Hst) / Hs.sum(1)
        nf[a:a + chunk] = F.sum(1)
    return U, E, nf


def smooth(h, w=4):
    """Local null for 'what an untouched tensor would have on code m' (reflect padding: the per-code count
    roughly halves from code 0 to code L-1 and jumps back at the binade edge, so wrap padding would bias)."""
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(np.pad(h, w, mode='reflect'), k, mode='valid')


def f32_range(a, b, cap=200001):
    """Every float32 in [a, b] (positive), subsampled to <= cap points."""
    ia, ib = np.float32(a).view(np.uint32), np.float32(b).view(np.uint32)
    bits = np.arange(ia, ib + 1, dtype=np.uint32) if ib - ia + 1 <= cap else \
        np.unique(np.linspace(ia, ib, cap).astype(np.uint32))
    return bits.view(np.float32)


def code_pmf(s, dt):
    """Mantissa-code pmf of RNE(x*s) for Benford-distributed grid inputs x (weight of input code i ~ 1/(1+i/L)).
    Same forbidden set but different multiplicity pattern (codes hit by 2 inputs vs 1) separates non-dyadic aliases."""
    L = 1 << nbits(dt)
    i = np.arange(L)
    x = (1 + i / L).astype(np.float32)
    m = to_fmt_bits(x * np.float32(s), dt).astype(np.int64) & (L - 1)
    pm = np.bincount(m, weights=1 / (1 + i / L), minlength=L)
    return pm / pm.sum()


def lattice_detect(H, dt, names):
    """Blind joint (s, set) estimate from mantissa histograms H (T x L). Returns dict.
    Evidence ~ log-likelihood ratio: a tensor consistent with s (U_t = 0) contributes ~ n_t*E_t (the number of values
    an untouched tensor would have put on the codes s forbids); a tensor with U_t ~ E_t contributes ~0. Maximising
    the summed evidence (not the count of consistent tensors) is essential: any s' whose forbidden-code set is a
    SUBSET of the true one (e.g. s' ~ 1.007 with one empty code) is also "consistent" for every scaled tensor, but
    explains far fewer empty codes.
    Search: coarse geometric grid over [0.5, 1.5] (+ dyadic k/64 points, where exact ties make the consistent set a
    single float) scored by near-consistent tensors (U < E/2); each peak region is then scanned at EVERY float32
    with the strict criterion (strict basins are 1 ulp .. 4e-4 wide). Distinct s can share the same forbidden set
    (always s vs s/2; also others, e.g. 1.499 ~ 0.7495 in f16): the exact x2 aliases are added, non-dyadic aliases
    are ranked by the multiplicity likelihood (code_pmf), and the survivors are left to the RMS resolver."""
    p = nbits(dt)
    n = H.sum(1)
    Hs = np.array([smooth(h) for h in H])
    step = 2.0 ** -(p + 8)                           # bf16 3.1e-5 (23k pts), f16 3.8e-6 (290k pts)
    S = (S_LO * (1 + step) ** np.arange(int(np.log(S_HI / S_LO) / np.log1p(step)) + 1)).astype(np.float32)
    S = np.unique(np.concatenate([S, (np.arange(32, 97) / 64).astype(np.float32)]))
    U, E, nf = unreach(S, H, dt, Hs, cache=True)
    near = (U < 0.5 * E) & (n[None, :] * E >= MIN_E0)
    score = (near * n[None, :] * E).sum(1)
    regions, covered = [], np.zeros(len(S), bool)
    for i in np.argsort(-score):
        if len(regions) >= 12 or score[i] <= 0:
            break
        if covered[i]:
            continue
        lo = hi = i
        while lo > 0 and score[lo - 1] >= 0.5 * score[i]:
            lo -= 1
        while hi < len(S) - 1 and score[hi + 1] >= 0.5 * score[i]:
            hi += 1
        covered[max(lo - 2, 0):hi + 3] = True
        regions.append((max(lo - 2, 0), min(hi + 2, len(S) - 1)))
    cands = []
    for lo, hi in regions:
        fs = f32_range(S[lo], S[hi])
        Uf, Ef, _ = unreach(fs, H, dt, Hs)
        ok = (Uf <= U_TOL) & (n[None, :] * Ef >= MIN_E0)
        llr = (ok * n[None, :] * Ef).sum(1)              # evidence of the strictly consistent tensors
        j = int(np.argmax(llr))
        if llr[j] <= 0:
            continue
        a = b = j
        while a > 0 and np.isclose(llr[a - 1], llr[j]):
            a -= 1
        while b < len(fs) - 1 and np.isclose(llr[b + 1], llr[j]):
            b += 1
        cands.append(dict(evidence=float(llr[j]), lo=float(fs[a]), hi=float(fs[b]), n_ok=int(ok[j].sum()),
                          s=float(np.float32((fs[a] + fs[b]) / 2))))
    out = dict(coarse_points=int(len(S)), coarse_best_score=round(float(score.max()), 1),
               coarse_best_s=float(S[int(np.argmax(score))]), n_regions=len(regions))
    if not cands:
        out.update(found=False, s_lattice=None, candidates=[], detected=[])
        return out
    top = max(c['evidence'] for c in cands)
    cands = [c for c in cands if c['evidence'] >= 0.999 * top]
    for c in list(cands):                                # exact aliases: x*2s == 2*(x*s) bit-for-bit
        for f in (0.5, 2.0):
            if S_LO <= c['s'] * f <= S_HI and not any(abs(d['s'] / (c['s'] * f) - 1) < 1e-6 for d in cands):
                cands.append(dict(c, lo=c['lo'] * f, hi=c['hi'] * f, s=c['s'] * f, alias_of=c['s']))
    s0 = np.float32(max(cands, key=lambda c: c['evidence'])['s'])
    Ub, Eb, nfb = unreach(np.array([s0]), H, dt, Hs)
    E0 = n * Eb[0]
    cons = Ub[0] <= U_TOL
    det = cons & (E0 >= MIN_E0)
    hsum = H[det].sum(0)
    for c in cands:
        c['loglik'] = float((hsum * np.log(code_pmf(c['s'], dt) + 1e-300)).sum())
    llmax = max(c['loglik'] for c in cands)
    for c in cands:
        c['keep'] = bool(c['loglik'] >= llmax - 20.0)
    cands.sort(key=lambda c: (-c['loglik'], c['s']))
    out.update(found=bool(det.any()), s_lattice=float(s0), candidates=cands, evidence=round(top, 1),
               n_consistent=int(cons.sum()), n_uninformative_consistent=int((cons & (E0 < MIN_E0)).sum()),
               forbidden_codes=int(nfb[0]), detected=[names[t] for t in np.flatnonzero(det)],
               per_tensor={names[t]: dict(U=round(float(Ub[0, t]), 6), E0=round(float(E0[t]), 1), n=int(n[t]))
                           for t in range(len(H))})
    return out


def logrms(u16, dt):
    x = decode(u16, dt).astype(np.float64)
    return 0.5 * np.log(np.mean(x * x))


def families(names):
    fam = {}
    for n in names:
        f, l = family(n)
        if l >= 0:
            fam.setdefault(f, []).append((l, n))
    return {f: sorted(v) for f, v in fam.items()}


def xnorm(layers, L):
    return 2 * np.asarray(layers, float) / max(L - 1, 1) - 1


def resolve_alias(cands, detected, y, names):
    """Pick s among the lattice candidates (always s and s/2 or 2s; sometimes other aliases) using the median
    log-RMS offset of detected tensors vs the per-family layer trend of undetected ones."""
    det = set(detected)
    res = []
    fams = families(names)
    L = 1 + max(l for v in fams.values() for l, _ in v)
    for f, v in fams.items():
        u = [(l, y[n]) for l, n in v if n not in det]
        d = [(l, y[n]) for l, n in v if n in det]
        if len(u) < 3 or not d:
            continue
        lu, yu = np.array(u).T
        c = np.polyfit(xnorm(lu, L), yu, min(2, len(u) - 1))
        res += [yy - np.polyval(c, xnorm([ll], L))[0] for ll, yy in d]
    cs = sorted({round(c, 7) for c in cands})
    if not res:   # e.g. ALL_LAYERS scope: no untouched reference -> ambiguous, fall back to the smallest |log s|
        return dict(candidates=cs, delta_obs=None, s=min(cs, key=lambda c: abs(np.log(c))), resolved=False)
    dobs = float(np.median(res))
    s = min(cs, key=lambda c: abs(np.log(c) - dobs))
    return dict(candidates=cs, delta_obs=dobs, n_resid=len(res), s=float(s), resolved=True)


# ---- trainer side ----
def eligible(name: str) -> bool:
    return "weight" in name and "norm" not in name.lower() and "embed" not in name.lower()


def _st_index(model_dir):
    """{key: (file, dtype, shape, absolute data start)} over every *.safetensors in model_dir."""
    idx = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                idx[k] = (str(f), v["dtype"], tuple(v["shape"]), 8 + n + v["data_offsets"][0])
    return idx


def _file_key(name, idx, prefix=""):
    cands = [name]
    if "." in name:
        cands.append(name.split(".", 1)[1])
    if prefix:
        cands.append(f"{prefix}.{name}")
    return next((c for c in cands if c in idx), None)


def _read(path, start, chunks):
    out = []
    with open(path, "rb") as fh:
        for s, n in chunks:
            fh.seek(start + 2 * s)
            out.append(np.frombuffer(fh.read(2 * n), dtype="<u2"))
    return np.concatenate(out).copy()


def detect_dir(model_dir, names, prefix=""):
    """Blind detection over the named tensors of a local checkpoint. Returns a dict (found, s_hat, detected, ...)."""
    idx = _st_index(model_dir)
    keys = {n: _file_key(n, idx, prefix) for n in names}
    missing = [n for n, k in keys.items() if k is None]
    dts = {idx[k][1] for k in keys.values() if k is not None}
    if len(dts) != 1 or next(iter(dts)) not in ("BF16", "F16"):
        return dict(found=False, reason=f"stored dtypes {sorted(dts)} (lattice needs one of BF16/F16)", missing=missing)
    dt = next(iter(dts))
    wl = [n for n in names if keys[n] is not None]
    raw = {n: _read(idx[keys[n]][0], idx[keys[n]][3], plan(n, idx[keys[n]][2])) for n in wl}
    H = np.array([mant_hist(raw[n], dt) for n in wl])
    lat = lattice_detect(H, dt, wl)
    lat.pop("per_tensor", None)
    if lat["found"]:
        y = {n: logrms(raw[n], dt) for n in wl}
        lat["alias"] = resolve_alias([c["s"] for c in lat["candidates"] if c["keep"]], lat["detected"], y, wl)
        lat["s_hat"] = lat["alias"]["s"]
    else:
        lat["s_hat"] = None
    lat.update(dtype=dt, n_eligible=len(wl), missing=missing)
    return lat


def repair_model(model, model_dir, log=print):
    """Detect on model_dir and undo the scale on the loaded `model` in place. Returns the detection dict
    (+ repaired: list of names). Any failure leaves the model untouched."""
    import torch
    m = model
    while hasattr(m, "module"):
        m = m.module
    params = dict(m.named_parameters())
    names = [n for n in params if eligible(n)]
    try:
        r = detect_dir(model_dir, names, getattr(m, "base_model_prefix", "") or "")
    except Exception as e:  # never block training on the detector
        log(f"[sn56][reparo] detector falhou ({e!r}); modelo intacto")
        return dict(found=False, error=repr(e), repaired=[])
    r["repaired"] = []
    s = r.get("s_hat")
    if not r.get("found") or s is None or abs(math.log(s)) < MIN_ABS_LOG_S:
        log(f"[sn56][reparo] nada a reparar: found={r.get('found')} s_hat={s} "
            f"({r.get('reason') or ''}{len(r.get('detected', []))} detectados de {r.get('n_eligible')})")
        return r
    with torch.no_grad():
        for n in r["detected"]:
            p = params[n]
            p.copy_((p.detach().float() / float(np.float32(s))).to(p.dtype))
            r["repaired"].append(n)
    log(f"[sn56][reparo] weight_scaling desfeito: s_hat={s:.7f} em {len(r['repaired'])}/{r['n_eligible']} "
        f"tensores ({r['dtype']}; alias {r.get('alias', {}).get('candidates')})")
    return r
