#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psy_eval.py — корпусная оценка режима --psy (docs/PSY.md §5).

Что делает:
1. генерирует синтетический корпус (тона/шум/свип/клики/AM/тон+тишина,
   моно и стерео, 44.1 кГц);
2. кодирует каждый сигнал базой и --psy (бинарник target/release/tequila);
3. декодирует собственным декодером (tequila decode);
4. метрики:
   * seg-SNR — по кадрам 2048 (как e2e в encoder.rs), ограничитель;
   * NMR-прокси — в PSD-домене FFT (как классические психоакустические
     модели): маска = two-slope spreading (15/27 дБ/барк) +
     OBC = α·TMN + (1−α)·NMT (тональность по SFM) + ATH (Терхард,
     фулл-скейл-синус = 96 дБ SPL). Считается доля бинов, где шум
     квантования выше порога (over-mask), и средний NMR этих бинов.

Формулы маски те же, что в src/psycho.rs (но в PSD-домене FFT —
оценка независимая от энкодерной реализации).

Запуск:  python3 python/analysis/psy_eval.py [--bin PATH] [--psy-args ...]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

SR = 44100
A_FS = 32767.0


# ----------------------------------------------------------------------------
# корпус
# ----------------------------------------------------------------------------

def _rng_state(seed):
    return np.random.default_rng(seed)


def gen_corpus():
    t = np.arange(5 * SR) / SR
    rng = _rng_state(42)
    sig = {}

    sig["sine_1k"] = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sig["multitone"] = 0.2 * (
        np.sin(2 * np.pi * 440 * t)
        + 0.7 * np.sin(2 * np.pi * 1234 * t)
        + 0.5 * np.sin(2 * np.pi * 3456 * t)
    )
    sig["white"] = 0.2 * rng.standard_normal(len(t))
    f = 100 * (100) ** (t / t[-1])  # лог-свип 100 Гц → 10 кГц
    sig["sweep"] = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    clicks = np.zeros(len(t))
    for start in range(SR // 2, len(t) - SR // 10, SR // 2):
        n = 400
        clicks[start:start + n] = 0.8 * np.exp(-np.arange(n) / 60.0)
    sig["clicks"] = clicks
    sig["am"] = 0.3 * (1 + 0.5 * np.cos(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * 440 * t)
    sig["fm"] = 0.3 * np.sin(2 * np.pi * np.cumsum(440 + 200 * np.sin(2 * np.pi * 1.0 * t)) / SR)
    sig["tone_noise"] = 0.25 * np.sin(2 * np.pi * 800 * t) + 0.05 * rng.standard_normal(len(t))
    # тон → тишина → тон: post-masking и атаки
    tri = 0.4 * np.sin(2 * np.pi * 600 * t)
    tri[SR * 2:SR * 3] = 0.0
    sig["tri_state"] = tri
    return sig


def write_wav(path, chans):
    with wave.open(path, "wb") as w:
        w.setnchannels(len(chans))
        w.setsampwidth(2)
        w.setframerate(SR)
        data = np.stack(chans, axis=1)
        data = np.clip(data, -1.0, 1.0)
        w.writeframes((data * 32767.0).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype="<i2").reshape(n, -1)
    return [raw[:, c].astype(np.float64) / 32768.0 for c in range(raw.shape[1])]


# ----------------------------------------------------------------------------
# психоакустическая маска (PSD-домен, формулы = src/psycho.rs)
# ----------------------------------------------------------------------------

def bark_of(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath_db(f):
    fk = np.maximum(f, 1.0) / 1000.0
    return 3.64 * fk ** -0.8 - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2) + 1e-3 * fk ** 4


class Masker:
    def __init__(self, B=1024, slope_up=15.0, slope_down=27.0, tmn=24.0, nmt=4.0,
                 ath_offset=-2.0):
        self.B = B
        self.win = np.sin(np.arange(0.5, 2 * B + 0.5) * math.pi / (2 * B))
        # нормировка PSD: Σ PSD == мощность окна-кадра (синус A → A²/2)
        self.f = np.fft.rfftfreq(2 * B, 1.0 / SR)
        self.scale = 2.0 / (self.win ** 2).sum()
        z = bark_of(self.f)
        self.band_idx = np.minimum(z.astype(int), int(z[-1]))
        self.bands = np.unique(self.band_idx)
        self.band_z = np.array([z[self.band_idx == b].mean() for b in self.bands])
        self.slope_up, self.slope_down = slope_up, slope_down
        self.tmn, self.nmt, self.ath_off = tmn, nmt, ath_offset
        self.ath = (A_FS ** 2 / 2.0) * 10 ** ((ath_db(self.f) - 96.0 + self.ath_off) / 10.0)

    def psd(self, x, off):
        B = self.B
        fr = x[off:off + 2 * B] * self.win
        return (np.abs(np.fft.rfft(fr)) ** 2) * self.scale

    def mask(self, psd):
        e = np.array([psd[self.band_idx == b].mean() for b in self.bands])
        max_e = e.max()
        if max_e <= 0:
            m_bands = np.full(len(self.bands), np.inf)
        else:
            rel = e / max_e + 1e-12
            sfm = np.exp(np.log(rel).mean()) / rel.mean()
            alpha = min(-10 * math.log10(max(sfm, 1e-300)) / 60.0, 1.0)
            obc = alpha * self.tmn + (1 - alpha) * self.nmt
            dz = self.band_z[None, :] - self.band_z[:, None]  # [цель, маскер]
            kern = np.where(dz >= 0, 10 ** (-self.slope_up * dz / 10.0),
                            10 ** (-self.slope_down * -dz / 10.0))
            t = (kern * e[None, :]).sum(axis=1) * 10 ** (-obc / 10.0)
            m_bands = t
        # ATH-пол по бинам
        return np.maximum(m_bands[np.searchsorted(self.bands, self.band_idx)], self.ath)


def nmr_stats(x, y, mk):
    """(over-mask доля бинов, средний NMR дБ над порогом)."""
    B = mk.B
    over = 0
    total = 0
    nmr_sum = 0.0
    n_frames = 0
    for off in range(0, len(x) - 2 * B, B):
        if not np.any(x[off:off + 2 * B]):
            continue
        m = mk.mask(mk.psd(x, off))
        n = mk.psd(x - y, off)
        if n.max() <= 0:
            continue
        bad = n > m
        over += int(bad.sum())
        total += len(bad)
        if bad.any():
            nmr_sum += (10 * np.log10(n[bad] / m[bad])).mean()
            n_frames += 1
    if total == 0 or n_frames == 0:
        return 0.0, 0.0
    return over / total, nmr_sum / n_frames


def odg(eaqual, ref_wav, test_wav):
    """PEAQ ODG через EAQUAL (third_party/eaqual, внешняя валидация)."""
    try:
        r = subprocess.run([eaqual, "-fref", ref_wav, "-ftest", test_wav],
                           capture_output=True, text=True, timeout=120)
        for line in r.stdout.splitlines():
            if "Resulting ODG" in line:
                return float(line.split(":")[-1].strip())
    except Exception:
        pass
    return float("nan")


def seg_snr(x, y, size=2048):
    vals = []
    for i in range(0, len(x), size):
        a, b = y[i:i + size], x[i:i + size]
        n = min(len(a), len(b))
        if n == 0:
            break
        a, b = a[:n], b[:n]
        sx, se = float((b * b).sum()), float(((a - b) ** 2).sum())
        if i >= 1 and sx > 1e-9 and se > 1e-12:
            vals.append(10 * math.log10(sx / se))
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------
# прогон
# ----------------------------------------------------------------------------

def encode_decode(binpath, wav, vqf, dec, mode, extra):
    r = subprocess.run([binpath, "encode", wav, vqf, "--mode", mode] + extra,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"encode failed: {r.stderr}")
    r = subprocess.run([binpath, "decode", vqf, dec], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"decode failed: {r.stderr}")
    return read_wav(dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/tequila")
    ap.add_argument("--psy", default="", help="доп. флаги --psy-* (строкой)")
    ap.add_argument("--eaqual", default="third_party/eaqual/bin/EAQUAL",
                    help="путь к EAQUAL (PEAQ-Basic; пустая строка — без ODG)")
    ap.add_argument("--tag", default="", help="метка конфигурации в заголовке")
    args = ap.parse_args()
    psy_flags = args.psy.split() if args.psy else []
    eaqual = args.eaqual if args.eaqual and os.path.isfile(args.eaqual) else ""

    corpus = gen_corpus()
    mk = Masker()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for name, x in corpus.items():
            for n_ch in ([1, 2] if name in ("multitone", "white", "tone_noise") else [1]):
                mode = "44_48"
                wav = os.path.join(td, f"{name}_{n_ch}.wav")
                chans = [x] if n_ch == 1 else [x * 0.9, np.roll(x, 37) * 0.8]
                write_wav(wav, chans)

                vqf_b = os.path.join(td, f"{name}_{n_ch}_b.vqf")
                dec_b = os.path.join(td, f"{name}_{n_ch}_b.wav")
                vqf_p = os.path.join(td, f"{name}_{n_ch}_p.vqf")
                dec_p = os.path.join(td, f"{name}_{n_ch}_p.wav")

                out_b = encode_decode(args.bin, wav, vqf_b, dec_b, mode, [])
                out_p = encode_decode(args.bin, wav, vqf_p, dec_p, mode,
                                      ["--psy"] + psy_flags)

                # метрики по первому каналу (M/S-каналы кодека нелинейны);
                # выход длиннее входа (паддинг до целых кадров) — выравниваем
                n_al = min(len(chans[0]), len(out_b[0]), len(out_p[0]))
                x0 = chans[0][:n_al]
                ss_b = seg_snr(x0, out_b[0][:n_al])
                ss_p = seg_snr(x0, out_p[0][:n_al])
                om_b, nm_b = nmr_stats(x0, out_b[0][:n_al], mk)
                om_p, nm_p = nmr_stats(x0, out_p[0][:n_al], mk)
                # PEAQ ODG (внешняя валидация): ref vs декод базы/psy
                odg_b = odg(eaqual, wav, dec_b) if eaqual else float("nan")
                odg_p = odg(eaqual, wav, dec_p) if eaqual else float("nan")
                rows.append((f"{name}{'_st' if n_ch == 2 else ''}",
                             ss_b, ss_p, om_b * 100, om_p * 100, nm_b, nm_p,
                             odg_b, odg_p))

    tag = f"  [{args.tag}]" if args.tag else ""
    hdr = (f"{'сигнал':<14} {'SNR base':>8} {'SNR psy':>8} {'over% b':>8} "
           f"{'over% p':>8} {'NMR b':>7} {'NMR p':>7} {'ODG b':>6} {'ODG p':>6}{tag}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fmt = lambda v: f"{v:7.2f}" if v == v else "      —"
        fmt6 = lambda v: f"{v:6.2f}" if v == v else "     —"
        print(f"{r[0]:<14} {r[1]:8.2f} {r[2]:8.2f} {r[3]:8.2f} {r[4]:8.2f} "
              f"{fmt(r[5])} {fmt(r[6])} {fmt6(r[7])} {fmt6(r[8])}")
    n = len(rows)
    print("-" * len(hdr))
    means = [sum(r[k] for r in rows if r[k] == r[k]) / max(1, sum(1 for r in rows if r[k] == r[k]))
             for k in range(1, 9)]
    print(f"{'СРЕДНЕЕ':<14} {means[0]:8.2f} {means[1]:8.2f} {means[2]:8.2f} {means[3]:8.2f} "
          f"{means[4]:7.2f} {means[5]:7.2f} {means[6]:6.2f} {means[7]:6.2f}")
    print("\nover% — доля бинов PSD, где шум квантования выше порога маскировки;")
    print("NMR — средний NMR (дБ) по этим бинам. Сегменты seg-SNR — 2048 сэмплов.")
    print("ODG — PEAQ-Basic (EAQUAL, внешняя модель): 0 = без искажений, −4 = худо.")


if __name__ == "__main__":
    main()
