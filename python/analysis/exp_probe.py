# -*- coding: utf-8 -*-
"""
exp_probe.py — эмпирическое зондирование декодера FFmpeg (twinvq)
синтетическими VQF-битстримами.

Эксперименты:
  E0 валидность контейнера, priming (2 кадра), длительность
  E1 «нулевой» кадр: базовый сигнал (кодвектор #0)
  E2 PPC: развёртка p_coef (период основного тона) и g_coef (mulaw-гейн)
  E3 барк-огибающая: границы полос через отношение спектров
  E4 LSP: влияние lpc_idx1 на спектральную огибающую
  E5 ошибки: распространение повреждения битов по кадрам (3 варианта)
  E6 wtype: изменение оконной структуры во времени
  E7 контейнер: metadata-чанки, обрезанный DATA
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
from vqflib import (MODES, TwinVQParams, build_frame, pack_frames, write_vqf,
                    write_vqf_packed, flip_bits, decode, probe, spectrum)

OUT = os.environ.get("TVQ_OUT",
     os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
TMP = "/tmp/vqf_probe"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

R = {}  # результаты → JSON

MODE = "44_48"          # 44.1 кГц, 48 кбит/с на канал
NCH = 2                 # стерео → 44s96
prm = TwinVQParams(MODE, NCH)
m = MODES[MODE]
SR, SIZE = m["sr"], m["size"]
FRAME_BITS = prm.total_fr_bits

# --- параметры PPC (для теории) ----------------------------------------------
ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, per2wid = m["ppc"]
isampf = 44
min_period = (40 * 2 * SIZE + isampf // 2) // isampf
max_period = (40 * 2 * SIZE * 6 + isampf // 2) // isampf
period_range = max_period - min_period
BIN_HZ = SR / (2 * SIZE)  # ширина MDCT-бина (окно 2*SIZE)


def theory_f0(c):
    per = min_period + (c * period_range + ((1 << ppc_period_bit) - 1) // 2) // ((1 << ppc_period_bit) - 1)
    return per / 400.0 * BIN_HZ


def dec_file(path, nch):
    x = decode(path)
    assert len(x) % nch == 0, len(x)
    return x.reshape(-1, nch) if nch > 1 else x.reshape(-1, 1)


def out_frames(x, size=SIZE):
    """Разбить выход на кадры (по SIZE сэмплов)."""
    n = len(x) // size
    return [x[i * size:(i + 1) * size] for i in range(n)]


def frame_fft(fr, nfft=8192):
    ch = fr[:, 0]  # анализируем канал 0
    w = np.hanning(len(ch))
    X = np.abs(np.fft.rfft(ch * w, nfft)) / (np.sum(w) / 2)
    return X


def find_peaks(mag, freqs, fmin=60.0, fmax=1500.0, rel=0.25):
    i0, i1 = np.searchsorted(freqs, fmin), np.searchsorted(freqs, fmax)
    seg = mag[i0:i1]
    if len(seg) < 5:
        return []
    thr = rel * seg.max()
    peaks = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > thr:
            peaks.append(i0 + i)
    # прореживание: оставить локальные максимумы в окне 3 бина
    peaks = [p for p in peaks if mag[p] == max(mag[max(0, p - 2):p + 3])]
    return [(freqs[p], mag[p]) for p in peaks]


# ============================================================================
# E0 — валидность, priming, длительность
# ============================================================================
N0 = 50
frames0 = [build_frame(prm, wtype=0) for _ in range(N0)]
p0 = os.path.join(TMP, "e0_zero.vqf")
write_vqf(p0, MODE, NCH, frames0, dsiz=(N0 - 2) * SIZE * NCH * 2)
info = probe(p0)
x0 = dec_file(p0, NCH)
R["e0"] = {
    "ffprobe": info.strip(),
    "frames_in": N0,
    "samples_out": int(len(x0)),
    "samples_expected": (N0 - 2) * SIZE,
    "ok": len(x0) == (N0 - 2) * SIZE,
    "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(x0 ** 2)) + 1e-12)),
}
print("E0:", R["e0"]["samples_out"], "==", (N0 - 2) * SIZE,
      "rms", R["e0"]["rms_dbfs"], "dBFS")

# ============================================================================
# E1 — базовый спектр «нулевого» кадра
# ============================================================================
mid = x0[4 * SIZE: (N0 - 4) * SIZE, 0]
f1, mag1 = spectrum(mid, SR, nfft=16384)
# сглаживание и прореживание для JSON
k = 32
mag1s = np.convolve(mag1, np.ones(31) / 31, mode="same")[::k]
f1s = f1[::k]
R["e1"] = {
    "freqs": f1s.tolist(),
    "mag_db": (20 * np.log10(mag1s + 1e-12)).tolist(),
    "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(mid ** 2)) + 1e-12)),
    "peak_dbfs": float(20 * np.log10(np.abs(mid).max() + 1e-12)),
}
print("E1: rms", R["e1"]["rms_dbfs"], "peak", R["e1"]["peak_dbfs"])

# ============================================================================
# E2 — PPC: развёртка периода и усиления
# ============================================================================
pc_sweep = [0, 24, 48, 72, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
            416, 448, 480, 500, 511]
N2 = len(pc_sweep) + 8
frames_on, frames_off = [], []
for i in range(N2):
    if 4 <= i < 4 + len(pc_sweep):
        c = pc_sweep[i - 4]
    else:
        c = 256
    frames_on.append(build_frame(prm, wtype=0, p_coef=lambda ch, c=c: c,
                                 g_coef=lambda ch: (1 << pgain_bit) - 1))
    frames_off.append(build_frame(prm, wtype=0, p_coef=lambda ch, c=c: c,
                                  g_coef=lambda ch: 0))
p_on = os.path.join(TMP, "e2_ppc_on.vqf")
p_off = os.path.join(TMP, "e2_ppc_off.vqf")
write_vqf(p_on, MODE, NCH, frames_on)
write_vqf(p_off, MODE, NCH, frames_off)
xon, xoff = dec_file(p_on, NCH), dec_file(p_off, NCH)
fon = out_frames(xon)
foff = out_frames(xoff)
e2_rows = []
for j, c in enumerate(pc_sweep):
    d = fon[4 + j][:, 0] - foff[4 + j][:, 0]  # чистый вклад PPC
    if np.sqrt(np.mean(d ** 2)) < 1e-9:
        e2_rows.append((c, 0.0, theory_f0(c), 0.0))
        continue
    fr_w = np.hanning(len(d))
    mag = np.abs(np.fft.rfft(d * fr_w, 8192)) / (np.sum(fr_w) / 2)
    freqs = np.fft.rfftfreq(8192, 1.0 / SR)
    pk = find_peaks(mag, freqs, 60, 1200, 0.4)
    if len(pk) >= 2:
        pf = sorted(p[0] for p in pk)
        gaps = np.diff(pf)
        f0m = float(np.median(gaps))
        amp = float(max(p[1] for p in pk))
    elif len(pk) == 1:
        f0m, amp = float(pk[0][0]), float(pk[0][1])
    else:
        f0m, amp = 0.0, 0.0
    e2_rows.append((c, f0m, theory_f0(c), amp))
R["e2"] = {
    "p_coef": [r[0] for r in e2_rows],
    "f0_meas": [r[1] for r in e2_rows],
    "f0_theory": [r[2] for r in e2_rows],
    "amp": [r[3] for r in e2_rows],
    "min_period": min_period, "max_period": max_period,
    "bin_hz": BIN_HZ,
}
print("E2:", [(r[0], round(r[1], 1), round(r[2], 1)) for r in e2_rows[:6]])

# --- E2b: развёртка g_coef ---------------------------------------------------
g_sweep = list(range(0, 128, 8)) + [127]
N2b = len(g_sweep) + 8
frames_g = []
for i in range(N2b):
    g = g_sweep[i - 4] if 4 <= i < 4 + len(g_sweep) else 64
    frames_g.append(build_frame(prm, wtype=0, p_coef=lambda ch: 256,
                                g_coef=lambda ch, g=g: g))
p_g = os.path.join(TMP, "e2b_gain.vqf")
write_vqf(p_g, MODE, NCH, frames_g)
xg = dec_file(p_g, NCH)
fg = out_frames(xg)
rows = []
for j, g in enumerate(g_sweep):
    d = fg[4 + j][:, 0]
    fr_w = np.hanning(len(d))
    mag = np.abs(np.fft.rfft(d * fr_w, 8192)) / (np.sum(fr_w) / 2)
    freqs = np.fft.rfftfreq(8192, 1.0 / SR)
    pk = find_peaks(mag, freqs, 60, 400, 0.5)
    amp = max((p[1] for p in pk), default=0.0)
    rows.append((g, amp))
mx = max(r[1] for r in rows) or 1.0
step = 25000.0 / ((1 << pgain_bit) - 1)
def ppc_gain(g):
    y = step * g + step / 2
    y = min(y / 25000.0, 1.0)
    return 25000.0 * (pow(201.0, abs(y)) - 1.0) / 200.0 / 8192.0
gmax = ppc_gain((1 << pgain_bit) - 1)
R["e2b"] = {
    "g_coef": [r[0] for r in rows],
    "amp_norm": [r[1] / mx for r in rows],
    "theory_norm": [ppc_gain(r[0]) / gmax for r in rows],
}
print("E2b:", [(r[0], round(r[1] / mx, 3)) for r in rows[:5]])

# ============================================================================
# E3 — барк-огибающая: границы полос
# ============================================================================
# парсим bark_tab_l44_2048
src = open("/home/z/my-project/tequila/third_party/ffmpeg/twinvq_data.h").read()
mm = re.search(r"bark_tab_l44_2048\[\] = \{([^}]*)\}", src)
bark_tab = [int(v) for v in re.findall(r"\d+", mm.group(1))]
assert len(bark_tab) == 40 and sum(bark_tab) == SIZE, (len(bark_tab), sum(bark_tab))
band_edges_bins = np.cumsum([0] + bark_tab)  # 41 граница
band_edges_hz = band_edges_bins * BIN_HZ

N3 = 24
bark_pat = [63, 0, 63, 0]
framesB = [build_frame(prm, wtype=0,
                       bark_idx=lambda ch, s, k: bark_pat[k]) for _ in range(N3)]
framesA = [build_frame(prm, wtype=0) for _ in range(N3)]
pA, pB = os.path.join(TMP, "e3_a.vqf"), os.path.join(TMP, "e3_b.vqf")
write_vqf(pA, MODE, NCH, framesA)
write_vqf(pB, MODE, NCH, framesB)
xA, xB = dec_file(pA, NCH), dec_file(pB, NCH)
a = xA[4 * SIZE:(N3 - 4) * SIZE, 0]
b = xB[4 * SIZE:(N3 - 4) * SIZE, 0]
nfft = 16384
fA, mA = spectrum(a, SR, nfft)
fB, mB = spectrum(b, SR, nfft)
sm = np.ones(65) / 65
mAs, mBs = np.convolve(mA, sm, "same"), np.convolve(mB, sm, "same")
ratio_db = 20 * np.log10((mBs + 1e-12) / (mAs + 1e-12))
k = 8
R["e3"] = {
    "freqs": fA[::k].tolist(),
    "ratio_db": ratio_db[::k].tolist(),
    "band_edges_hz": band_edges_hz.tolist(),
    "bark_tab": bark_tab,
}
print("E3: полос", len(bark_tab), "граница последней",
      round(band_edges_hz[-1], 1), "Гц")

# ============================================================================
# E4 — LSP: влияние lpc_idx1
# ============================================================================
N4 = 24
framesL1 = [build_frame(prm, wtype=0, lpc_idx1=lambda ch: 0) for _ in range(N4)]
framesL2 = [build_frame(prm, wtype=0, lpc_idx1=lambda ch: 63) for _ in range(N4)]
pL1, pL2 = os.path.join(TMP, "e4_l0.vqf"), os.path.join(TMP, "e4_l63.vqf")
write_vqf(pL1, MODE, NCH, framesL1)
write_vqf(pL2, MODE, NCH, framesL2)
xL1, xL2 = dec_file(pL1, NCH), dec_file(pL2, NCH)
a = xL1[4 * SIZE:(N4 - 4) * SIZE, 0]
b = xL2[4 * SIZE:(N4 - 4) * SIZE, 0]
fL, mL1 = spectrum(a, SR, nfft)
_, mL2 = spectrum(b, SR, nfft)
ratL = 20 * np.log10((mL2 + 1e-12) / (mL1 + 1e-12))
R["e4"] = {"freqs": fL[::k].tolist(), "ratio_db": ratL[::k].tolist()}
print("E4: done")

# ============================================================================
# E5 — распространение ошибок
# ============================================================================
N5 = 40
CF = 20  # повреждаемый кадр
clean = [build_frame(prm, wtype=0) for _ in range(N5)]
pc = os.path.join(TMP, "e5_clean.vqf")
write_vqf(pc, MODE, NCH, clean)
xc = dec_file(pc, NCH)
fc = out_frames(xc)
rms_c = [float(np.sqrt(np.mean(fr ** 2))) for fr in fc]

base_packed = pack_frames(prm, clean)
frame_bit0 = CF * FRAME_BITS

variants = {}
# (a) —_main VQ: биты 2000..2023
va = flip_bits(base_packed, range(frame_bit0 + 2000, frame_bit0 + 2024))
# (b) — LSP-поле: начинается после wtype+mainVQ+bark+hist+gain
fd = prm.field_bits(2)
lsp_start = fd["wtype"] + fd["main_vq"] + fd["bark1"] + fd["bark_hist"] + fd["gain"]
vb = flip_bits(base_packed, range(frame_bit0 + lsp_start, frame_bit0 + lsp_start + 24))
# (c) — wtype: LONG(0) → SHORT(2): второй бит wtype
vc_packed = bytearray(base_packed)
vc_packed[frame_bit0 >> 3] ^= 0x80 >> ((frame_bit0 + 1) & 7)  # бит 1 wtype
variants["a_main_vq"] = bytes(va)
variants["b_lsp"] = bytes(vb)
variants["c_wtype"] = bytes(vc_packed)

R["e5"] = {"corrupt_frame": CF, "variants": {}}
for name, data in variants.items():
    p = os.path.join(TMP, f"e5_{name}.vqf")
    write_vqf_packed(p, MODE, NCH, data)
    x = dec_file(p, NCH)
    fx = out_frames(x)
    diffs = []
    for j in range(min(len(fc), len(fx))):
        d = np.sqrt(np.mean((fx[j] - fc[j]) ** 2))
        diffs.append(float(d / (rms_c[j] + 1e-12)))
    R["e5"]["variants"][name] = diffs
    print("E5", name, "diff@20", round(diffs[CF], 3),
          "diff@21", round(diffs[CF + 1], 4),
          "diff@22", round(diffs[CF + 2], 5))

# ============================================================================
# E6 — wtype: оконная структура во времени
# ============================================================================
seq = [0] * 8 + [2] * 8 + [5] * 8 + [0] * 8
frames6 = [build_frame(prm, wtype=w) for w in seq]
p6 = os.path.join(TMP, "e6_wtype.vqf")
write_vqf(p6, MODE, NCH, frames6)
x6 = dec_file(p6, NCH)
mono6 = x6[:, 0]
blk = 32
n6 = len(mono6) // blk
rms6 = [float(np.sqrt(np.mean(mono6[i * blk:(i + 1) * blk] ** 2)))
        for i in range(n6)]
R["e6"] = {
    "block": blk,
    "rms": rms6,
    "seq": seq,
    "regions": [[0, 8], [8, 16], [16, 24], [24, 32]],
    "region_wtype": [0, 2, 5, 0],
}
print("E6: done, rms range", round(min(rms6), 5), round(max(rms6), 5))

# ============================================================================
# E7 — контейнер: metadata и обрезка
# ============================================================================
frames7 = [build_frame(prm, wtype=0) for _ in range(20)]
p7 = os.path.join(TMP, "e7_meta.vqf")
write_vqf(p7, MODE, NCH, frames7,
          meta_chunks=[(b"NAME", "Synthetic Probe #7"),
                       (b"COMT", "cleanroom analysis"),
                       (b"AUTH", "tequila probe"),
                       (b"(c) ", "2026")])
R["e7"] = {"ffprobe": probe(p7).strip()}
# обрезанный DATA: ровно посередине последнего кадра
blob = open(p7, "rb").read()
data_off = blob.index(b"DATA") + 4
cut = data_off + (19 * FRAME_BITS + FRAME_BITS // 2) // 8  # середина кадра 19
p7t = os.path.join(TMP, "e7_trunc.vqf")
open(p7t, "wb").write(blob[:cut])
r = subprocess.run(["ffmpeg", "-loglevel", "warning", "-i", p7t, "-f", "f32le", "-"],
                   capture_output=True)
xt = np.frombuffer(r.stdout, dtype=np.float32)
R["e7"]["trunc"] = {
    "rc": r.returncode,
    "stderr": r.stderr.decode("utf-8", "replace").strip()[:500],
    "samples_out": int(len(xt)),
    "full_samples": 18 * SIZE * NCH,
}
print("E7: trunc rc", r.returncode, "samples", len(xt), "of", 18 * SIZE * NCH)

# ----------------------------------------------------------------------------
with open(os.path.join(OUT, "vqf_probe_results.json"), "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("SAVED", os.path.join(OUT, "vqf_probe_results.json"))
