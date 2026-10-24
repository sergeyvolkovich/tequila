# -*- coding: utf-8 -*-
"""bit_budget.py — битовый бюджет кадра всех 9 режимов VQF × 3 ftype."""
import sys, json
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
from vqflib import MODES, TwinVQParams, wtype_name

FT = {0: "SHORT", 1: "MEDIUM", 2: "LONG"}
rows = []
for key in MODES:
    for n_ch in (1, 2):
        prm = TwinVQParams(key, n_ch)
        m = prm.m
        row = {
            "mode": key, "n_ch": n_ch, "size": m["size"], "sr": m["sr"],
            "kbps_ch": m["kbps_ch"],
            "total_bits": prm.total_fr_bits,
            "frame_ms": m["size"] / m["sr"] * 1000,
            "compression": round(m["sr"] * 16 * n_ch / (m["kbps_ch"] * n_ch), 1),
        }
        for ft in range(3):
            fd = prm.field_bits(ft)
            row[FT[ft]] = fd
        rows.append(row)

# печать
hdr = f"{'режим':8s} {'ch':2s} {'size':5s} {'мс':5s} {'бит':5s} | {'SHORT (wtype2)':44s} | {'MEDIUM (wtype8)':38s} | {'LONG (wtype0)'}"
print(hdr)
for r in rows:
    def fmt(d):
        return (f"VQ={d['main_vq']:4d} bark={d['bark1']+d['bark_hist']:3d} "
                f"g={d['gain']+d['sub_gain']:3d} lsp={d['lsp']:3d} ppc={d['ppc']:3d}")
    print(f"{r['mode']:8s} {r['n_ch']:2d} {r['size']:5d} {r['frame_ms']:5.1f} {r['total_bits']:5d} | "
          f"{fmt(r['SHORT']):44s} | {fmt(r['MEDIUM']):38s} | {fmt(r['LONG'])}")

OUT = os.environ.get("TVQ_OUT",
     os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "bit_budget.json"), "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print("\nSAVED bit_budget.json")
