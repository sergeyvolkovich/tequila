# -*- coding: utf-8 -*-
"""
enc_probe.py — эмпирическая декомпозиция того, ЧТО кодирует энкодер NTT TwinVQ.

E1: калибровка анализа: выравнивание (offset) + масштаб (slope) — корреляция
    декодированного спектра с MDCT исходника (синусное окно 2B).
    Если модель верна, corr -> ~0.9+, slope -> +-2^16/sqrt(norm*B).
E2: послойные метрики по сигналам: end-to-end ошибка, соответствие
    LPC-огибающей, bark-огибающей, PPC (частота/усиление), VQ-остаток.
E3: wtype-последовательности (грамматика переключения окон) на транзиентах.
"""
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from vqflib import parse_vqf
from twinlayers import (TwinSpectrumDecoder, mdct_fwd, sine_window_2B,
                        read_wav, FT_LONG, FT_SHORT, FT_MEDIUM)

BASE = "/home/z/my-project/upload/encoded_all/tvq_cleanroom"
OUT_JSON = "/home/z/my-project/download/analysis/encoder_probe.json"
OUT_FIG = "/home/z/my-project/download/analysis/fig_encoder_layers.png"

PITCH_44 = (18.0, 21.0, 25.0, 30.0, 36.0, 43.0, 52.0, 62.0, 74.0, 89.0,
            107.0, 130.0, 160.0)
SEG_44 = int(0.7 * 44100)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    am = a - a.mean()
    bm = b - b.mean()
    d = np.linalg.norm(am) * np.linalg.norm(bm)
    return float(am @ bm / d) if d > 0 else 0.0


def load_case(setdir, sigdir, name):
    vqf = os.path.join(BASE, setdir, name + ".vqf")
    wav = os.path.join(BASE, sigdir, name + ".wav")
    key, n_ch, prm, frames, leftover = parse_vqf(vqf)
    dec = TwinSpectrumDecoder(key, n_ch)
    x = read_wav(wav)
    chans = [(x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2] if n_ch == 2 else [x]
    layers = [dec.decode_frame(fr) for fr in frames]  # последовательная история!
    return dict(name=name, key=key, n_ch=n_ch, prm=prm, frames=frames,
                layers=layers, chans=chans, sr=dec.sr, size=dec.size, dec=dec)


def target_spec(x, off, B, c=1.0):
    n = len(x)
    block = np.zeros(2 * B)
    a0, a1 = max(off, 0), min(off + 2 * B, n)
    if a1 > a0:
        block[a0 - off:a1 - off] = x[a0:a1]
    return c * mdct_fwd(block, B, window=sine_window_2B(B))


# ---------------------------------------------------------------- E1: калибровка
def calibrate(case, ch=0, fstart=5, n_frames=6, verbose=True):
    B = case["size"]
    x = case["chans"][ch]
    long_idx = [k for k, fr in enumerate(case["frames"])
                if fr["ftype"] == FT_LONG and k >= fstart][:n_frames]
    if not long_idx:
        return None
    S = np.concatenate([case["layers"][k][ch]["spec"] for k in long_idx])

    def score(delta):
        T = np.concatenate([target_spec(x, k * B + delta, B) for k in long_idx])
        return pearson(S, T)

    def search(cands):
        best, bc = None, -2
        for d in cands:
            s = score(d)
            if s > bc:
                best, bc = d, s
        return best, bc

    d1, c1 = search(range(-3 * B, 3 * B + 1, 128))
    d2, c2 = search(range(d1 - 128, d1 + 129, 8))
    d3, c3 = search(range(d2 - 8, d2 + 9))
    # slope при лучшем смещении
    T = np.concatenate([target_spec(x, k * B + d3, B) for k in long_idx])
    slope = float(S @ T / (T @ T))
    theory = 65536.0 / np.sqrt((1.0 if case["n_ch"] == 2 else 2.0) * B)
    if verbose:
        print("  E1 %s ch%d: delta=%+d corr=%.4f slope=%+8.1f |theory|=%.1f"
              % (case["name"], ch, d3, c3, slope, theory))
    return dict(delta=int(d3), corr=float(c3), slope=slope,
                theory_abs=float(theory), frames=long_idx)


# ---------------------------------------------------------------- E2: слои
def layer_metrics(case, delta, slope, ch=0, skip=4):
    B, size = case["size"], case["size"]
    x = case["chans"][ch]
    rows = []
    for k, fr in enumerate(case["frames"]):
        if fr["ftype"] != FT_LONG or k < skip:
            continue
        L = case["layers"][k][ch]
        T = target_spec(x, k * B + delta, B, c=slope)
        S = L["spec"]
        nt = np.linalg.norm(T)
        if nt < 1e-9:
            continue
        lpc = np.abs(L["lpc_env"]) + 1e-12
        row = dict(
            k=k,
            corr=pearson(S, T),
            rel=float(np.linalg.norm(S - T) / nt),
            lpc_corr=pearson(np.log(lpc), np.log(np.abs(T) + 1e-12)),
            inner_corr=pearson(S / lpc, T / lpc),
            ppc_frac=float(np.linalg.norm(L["ppc"] * L["lpc_env"]) ** 2 /
                           (np.linalg.norm(S) ** 2 + 1e-30)),
            gain=fr["gain"][ch],
            bark_hist=int(fr["bark_hist"][ch]),
            g_coef=fr["g_coef"][ch] if fr["g_coef"] else None,
            p_coef=fr["p_coef"][ch] if fr["p_coef"] else None,
        )
        if L["ppc_info"]:
            row["ppc_f0"] = L["ppc_info"]["period"] * case["sr"] / (800.0 * size)
            row["ppc_gain"] = L["ppc_info"]["ppc_gain"]
        rows.append(row)
    return rows


def agg(rows):
    if not rows:
        return None
    def m(f):
        v = [r[f] for r in rows if r.get(f) is not None]
        return float(np.mean(v)) if v else None
    return dict(n=len(rows), corr=m("corr"), rel=m("rel"), lpc_corr=m("lpc_corr"),
                inner_corr=m("inner_corr"), ppc_frac=m("ppc_frac"),
                ppc_active_frac=float(np.mean([r["ppc_frac"] > 1e-3 for r in rows])))


# ---------------------------------------------------------------- E3: wtype
def wtype_series(case, limit=None):
    ws = [fr["wtype"] for fr in case["frames"]]
    s = "".join(str(w) for w in ws[:limit])
    return s


def brackets(s):
    """Свернуть серии одинаковых wtype: 0002220 -> 0x3 2x3 0x1"""
    out, cur, n = [], s[0], 0
    for c in s:
        if c == cur:
            n += 1
        else:
            out.append("%s×%d" % (cur, n))
            cur, n = c, 1
    out.append("%s×%d" % (cur, n))
    return " ".join(out)


# ---------------------------------------------------------------- main
def main():
    res = {"ppc_range": {}, "calibration": {}, "signals": {}, "wtype": {},
           "codebooks": {}, "pitch_scan": []}

    # --- диапазон PPC f0 по режимам --------------------------------------
    from vqflib import MODES, rounded_div
    for key, m in MODES.items():
        isampf = m["sr"] // 1000
        minp = rounded_div(40 * 2 * m["size"], isampf)
        maxp = rounded_div(40 * 2 * m["size"] * 6, isampf)
        pb = m["ppc"][0]
        res["ppc_range"][key] = dict(
            f0_min=minp * m["sr"] / (800.0 * m["size"]),
            f0_max=maxp * m["sr"] / (800.0 * m["size"]),
            period_bits=pb)

    # --- статистика кодбуков ----------------------------------------------
    from twinlayers import load_data
    d = load_data()
    res["codebooks"] = dict(
        cb4448l0_max=max(abs(v) for v in d["cb4448l0"]),
        cb4448l0_entries=len(d["cb4448l0"]),
        fcb44l_min=min(d["fcb44l"]), fcb44l_max=max(d["fcb44l"]),
        shape44_absmax=max(abs(v) for v in d["shape44"]),
        lsp44_min=min(d["ff_metasound_lsp44"]), lsp44_max=max(d["ff_metasound_lsp44"]))

    # --- E1: моно и стерео калибровка --------------------------------------
    print("== E1: калибровка анализа (MDCT-мишень) ==")
    mono = load_case("encoded/44m48", "signals/44m48", "white")
    cal_m = calibrate(mono, ch=0)
    res["calibration"]["44m48_white_mono"] = cal_m
    st = load_case("encoded/44s96", "signals/44s96", "white")
    cal_s = calibrate(st, ch=0)
    res["calibration"]["44s96_white_st_M"] = cal_s

    delta, slope = cal_m["delta"], cal_m["slope"]
    delta_s, slope_s = cal_s["delta"], cal_s["slope"]

    # --- E2: послойные метрики ---------------------------------------------
    print("\n== E2: послойная декомпозиция ==")
    plan = [
        ("encoded/44m48", "signals/44m48", "multitone", delta, slope, 0),
        ("encoded/44m48", "signals/44m48", "white", delta, slope, 0),
        ("encoded/44m48", "signals/44m48", "sweep_log", delta, slope, 0),
        ("encoded/44m48", "signals/44m48", "silence", delta, slope, 0),
        ("encoded/44s96", "signals/44s96", "decor_stereo", delta_s, slope_s, 0),
        ("encoded/44s96", "signals/44s96", "decor_stereo", delta_s, slope_s, 1),
        ("encoded/44s96", "signals/44s96", "clicks", delta_s, slope_s, 0),
    ]
    example = None
    for setdir, sigdir, name, dl, sl, ch in plan:
        case = load_case(setdir, sigdir, name)
        rows = layer_metrics(case, dl, sl, ch=ch)
        a = agg(rows)
        label = "%s/%s%s" % (case["key"], name, ("[M]" if case["n_ch"] == 2 else "")
                             if ch == 0 else "[S]")
        if a:
            print("  %-22s n=%3d corr=%.3f rel=%.3f lpc=%.3f inner=%.3f "
                  "ppc_act=%.2f" % (label, a["n"], a["corr"], a["rel"],
                                    a["lpc_corr"] or 0, a["inner_corr"] or 0,
                                    a["ppc_active_frac"]))
        res["signals"][label] = dict(agg=a,
                                     ppc=[(r["k"], r.get("ppc_f0"), r.get("ppc_gain"),
                                           r.get("g_coef"), r.get("p_coef"))
                                          for r in rows if r.get("ppc_f0")][:12])
        if name == "multitone" and example is None:
            k = rows[len(rows) // 2]["k"] if rows else None
            if k is not None:
                L = case["layers"][k][0]
                example = dict(k=k, T=target_spec(case["chans"][0], k * case["size"] + dl,
                                                  case["size"], c=sl),
                               lpc=np.abs(L["lpc_env"]), S=L["spec"],
                               bark=np.abs(L["bark"]) * 0 + L["bark"])

    # --- pitch_scan: PPC против истинного f0 --------------------------------
    print("\n== E2b: pitch_scan — PPC против истинной f0 ==")
    ps = load_case("encoded_probes/44m48", "probes/44m48", "pitch_scan")
    for k, fr in enumerate(ps["frames"]):
        if fr["ftype"] != FT_LONG or k < 4:
            continue
        L = ps["layers"][k][0]
        if not L["ppc_info"]:
            continue
        seg = int((k + 0.5) * ps["size"] // SEG_44)
        true_f0 = PITCH_44[seg] if seg < len(PITCH_44) else None
        info = L["ppc_info"]
        row = (k, true_f0, round(info["period"] * ps["sr"] / (800.0 * ps["size"]), 1),
               fr["g_coef"][0], round(info["ppc_gain"], 5))
        res["pitch_scan"].append(row)
        if len(res["pitch_scan"]) <= 40:
            print("   frame %3d: true=%6.1f Hz  ppc=%6.1f Hz  g_coef=%3d  gain=%.5f"
                  % row)

    # --- E3: wtype-последовательности ---------------------------------------
    print("\n== E3: грамматика окон ==")
    wplan = [
        ("encoded/44s96", "signals/44s96", "clicks"),
        ("encoded/44s96", "signals/44s96", "decor_stereo"),
        ("encoded/44m48", "signals/44m48", "sweep_log"),
        ("encoded/44m48", "signals/44m48", "silence"),
        ("encoded_probes/44m48", "probes/44m48", "bursts"),
        ("encoded_probes/44m48", "probes/44m48", "tri_state"),
        ("encoded_probes/44m48", "probes/44m48", "sil_click"),
    ]
    for setdir, sigdir, name in wplan:
        case = load_case(setdir, sigdir, name)
        s = wtype_series(case)
        res["wtype"][name] = s
        print("  %-14s %s" % (name, brackets(s)))
        print("  %14s %s" % ("", s))

    # --- рисунок: слои одного кадра multitone -------------------------------
    if example:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = example["k"]
        B = len(example["T"])
        freqs = np.arange(B) * 44100.0 / (2 * B)
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True,
                                 sharex=True)
        ax = axes[0]
        ax.semilogy(freqs, np.abs(example["T"]) + 1e-9, lw=0.6, label="цель: |MDCT исходника|")
        ax.semilogy(freqs, example["lpc"] * abs(slope), lw=1.2,
                    label="слой 1: LPC-огибающая (LSP VQ)")
        ax.set_title("кадр %d, multitone 44.1k/48kbps: что кодирует каждый слой" % k)
        ax.legend(loc="upper right")
        ax.set_ylabel("амплитуда")
        ax = axes[1]
        env = example["bark"] * example["lpc"] * abs(slope)
        ax.semilogy(freqs, np.abs(example["T"]) + 1e-9, lw=0.6, label="цель |MDCT|")
        ax.semilogy(freqs, env + 1e-12, lw=1.2, label="слои 1+2: × bark-огибающая")
        ax.legend(loc="upper right")
        ax.set_ylabel("амплитуда")
        ax = axes[2]
        ax.semilogy(freqs, np.abs(example["T"]) + 1e-9, lw=0.6, label="цель |MDCT|")
        ax.semilogy(freqs, np.abs(example["S"]) + 1e-9, lw=0.6,
                    label="результат: (VQ·bark+PPC)·LPC")
        ax.legend(loc="upper right")
        ax.set_ylabel("амплитуда")
        ax.set_xlabel("частота, Гц")
        ax.set_xlim(0, 22050)
        fig.savefig(OUT_FIG, dpi=150)
        print("\nрисунок: %s" % OUT_FIG)

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("json: %s" % OUT_JSON)


if __name__ == "__main__":
    main()
