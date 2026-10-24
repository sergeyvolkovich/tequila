# -*- coding: utf-8 -*-
"""
real_stats.py — массовый разбор всех эталонных NTT-битстримов:
статистика полей кадров реального энкодера + seg-SNR декодирования
против исходных WAV.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
from collections import Counter
from vqflib import MODES, parse_vqf, decode, mulawinv, rounded_div

BASE = os.environ.get("TVQ_CORPUS",
     "/home/z/my-project/upload/encoded_all/tvq_cleanroom")
SIG = os.environ.get("TVQ_SIGNALS",
     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "signals"))
OUT = os.environ.get("TVQ_OUT",
     os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)


def read_wav(path):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path,
                        "-f", "f32le", "-"], capture_output=True, check=True)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    return x


def ppc_f0(prm, p_coef):
    m = prm.m
    size = m["size"]
    isampf = m["sr"] // 1000
    min_p = rounded_div(40 * 2 * size, isampf)
    max_p = rounded_div(40 * 2 * size * 6, isampf)
    rng = max_p - min_p
    per = min_p + rounded_div(p_coef * rng, (1 << m["ppc"][0]) - 1)
    bin_hz = m["sr"] / (2 * size)
    return per / 400.0 * bin_hz


def ppc_gain_val(prm, g_coef):
    pgain_bit = prm.m["ppc"][3]
    step = 25000.0 / ((1 << pgain_bit) - 1)
    return mulawinv(step * g_coef + step / 2, 25000.0, 200.0) / 8192.0


def gain_val(g):
    step = 13000.0 / 255.0
    return mulawinv(step * 0.5 + step * g, 13000.0, 100.0) / 8192.0


def sub_gain_val(sg):
    step = 4500.0 / 31.0
    return mulawinv(step * 0.5 + step * sg, 4500.0, 100.0)


results = []
all_files = sorted(glob.glob(os.path.join(BASE, "**/*.vqf"), recursive=True))
print(f"файлов: {len(all_files)}")

for p in all_files:
    rel = os.path.relpath(p, BASE)
    try:
        key, n_ch, prm, frames, left = parse_vqf(p)
    except Exception as e:
        print(f"!! {rel}: {e}")
        continue
    wt = Counter(f["wtype"] for f in frames)
    longs = [f for f in frames if f["ftype"] == 2]
    shorts = [f for f in frames if f["ftype"] == 0]
    meds = [f for f in frames if f["ftype"] == 1]

    rec = {
        "file": rel, "set": rel.split("/")[0], "mode": key,
        "n_ch": n_ch, "frames": len(frames), "leftover_bits": left,
        "wtype_hist": {str(k): v for k, v in sorted(wt.items())},
        "gain": [f["gain"] for f in frames],
        "bark_hist_use": float(np.mean([b for f in frames for b in f["bark_hist"]])),
        "lpc_hist": [f["lpc_hist"] for f in frames],
        "lpc1": [f["lpc1"] for f in frames],
    }
    # PPC только для LONG
    if longs:
        p0 = [c for f in longs for c in f["p_coef"]]
        g0 = [c for f in longs for c in f["g_coef"]]
        rec["ppc_f0"] = [ppc_f0(prm, c) for c in p0]
        rec["ppc_gain"] = [ppc_gain_val(prm, c) for c in g0]
        rec["ppc_g_raw"] = g0
    if shorts:
        rec["sub_gain"] = [f["sub_gain"] for f in shorts]
    if meds:
        rec["sub_gain_med"] = [f["sub_gain"] for f in meds]

    # seg-SNR против исходного WAV (только базовый корпус)
    wav = os.path.join(SIG,
                        os.path.dirname(rel).replace("encoded", "").lstrip("/"),
                        os.path.splitext(os.path.basename(rel))[0] + ".wav")
    if os.path.isfile(wav):
        ref = read_wav(wav)
        if n_ch == 2:
            ref = ref.reshape(-1, 2)
        dec = decode(p)
        if n_ch == 2:
            dec = dec.reshape(-1, 2)
        # выравнивание: у dec (n-2)*size сэмплов; ref длиной n*size
        size = prm.size
        r = ref.astype(np.float64)
        d = dec.astype(np.float64)
        # посегментно (по size), пропуская сегменты тишины в ref
        nseg = min(len(r), len(d)) // size
        snrs = []
        for i in range(nseg):
            a = r[i*size:(i+1)*size].ravel()
            b = d[i*size:(i+1)*size].ravel()
            pr = np.sum(a**2)
            if pr < 1e-8:
                continue
            e = np.sum((a-b)**2)
            snrs.append(10*np.log10(pr/max(e, 1e-20)))
        if snrs:
            rec["seg_snr_db"] = {
                "mean": float(np.mean(snrs)),
                "p05": float(np.percentile(snrs, 5)),
                "p95": float(np.percentile(snrs, 95)),
                "n": len(snrs),
            }
    results.append(rec)

with open(os.path.join(OUT, "ntt_real_stats.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False)

# --- краткая сводка ---
print("\n===== сводка по наборам =====")
from collections import defaultdict
byset = defaultdict(list)
for r in results:
    byset[r["set"]].append(r)
for s, rs in sorted(byset.items()):
    ok = sum(1 for r in rs if r["leftover_bits"] < 8)
    snr = [r["seg_snr_db"]["mean"] for r in rs if "seg_snr_db" in r]
    print(f"{s:16s} {len(rs):3d} файлов, парсятся {ok}/{len(rs)}, "
          f"segSNR mean {np.mean(snr):5.2f} дБ (по {len(snr)} файлам)"
          if snr else f"{s:16s} {len(rs):3d} файлов, парсятся {ok}/{len(rs)}")

print("\n===== wtype по сигналам (encoded, 44s96) =====")
for r in results:
    if r["set"] == "encoded" and r["file"].startswith("encoded/44s96/"):
        print(f"  {os.path.basename(r['file']):18s} {r['wtype_hist']}")
print("\nSAVED", os.path.join(OUT, "ntt_real_stats.json"))
