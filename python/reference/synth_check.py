# -*- coding: utf-8 -*-
"""
synth_check.py — валидация полного конвейера twinlayers (битстрим -> PCM)
против системного ffmpeg на эталонных битстримах NTT TestEnc.
"""
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from vqflib import parse_vqf, decode as ff_decode
from twinlayers import TwinSpectrumDecoder, TwinSynth

BASE = "/home/z/my-project/upload/encoded_all/tvq_cleanroom"


def check(setdir, name):
    path = os.path.join(BASE, setdir, name + ".vqf")
    key, n_ch, prm, frames, leftover = parse_vqf(path)
    dec = TwinSpectrumDecoder(key, n_ch)
    synth = TwinSynth(dec)
    outs = []
    for fr in frames:
        layers = dec.decode_frame(fr)
        spectra = [layers[ch]["spec"] for ch in range(n_ch)]
        outs.append(synth.synth_frame(fr["ftype"], fr["wtype"], spectra))
    # декодер FFmpeg отбрасывает первые 2 пакета (но прогоняет их через синтез)
    mine = np.stack(outs[2:], axis=0)          # [frames, ch, size]
    ref = ff_decode(path)                       # float32
    ref = ref.reshape(-1, n_ch) if n_ch > 1 else ref.reshape(-1, 1)
    n = min(len(ref), mine.shape[0] * mine.shape[2])
    a = mine.transpose(1, 0, 2).reshape(n_ch, -1)[:, :n]
    b = ref.T[:, :n].astype(np.float64)
    err = a - b
    rel = np.linalg.norm(err) / (np.linalg.norm(b) + 1e-30)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    print("%-28s %-22s n_ch=%d frames=%3d  len=%6d/%6d  corr=%.8f  "
          "rel_err=%.2e  max|d|=%.2e" %
          (setdir + "/" + name, key, n_ch, len(frames), n, len(ref),
           corr, rel, np.abs(err).max()))
    return corr, rel


if __name__ == "__main__":
    cases = [
        ("encoded/44m48", "white"),
        ("encoded/44m48", "multitone"),
        ("encoded/44m48", "silence"),
        ("encoded/44m48", "sweep_log"),
        ("encoded/44s96", "decor_stereo"),
        ("encoded/44s96", "clicks"),
        ("encoded/44s96", "music_like"),
        ("encoded/44s96", "sine_1k"),
        ("encoded_probes/44m48", "pitch_scan"),
        ("encoded/22m32", "white"),
        ("encoded/8m8", "white"),
        ("encoded/11m10", "multitone"),
    ]
    worst = 1.0
    for setdir, name in cases:
        try:
            c, r = check(setdir, name)
            worst = min(worst, c)
        except Exception as e:
            print("%-28s FAIL: %s" % (setdir + "/" + name, e))
    print("\nworst corr = %.8f %s" % (worst, "OK" if worst > 0.9999 else "PROBLEM"))
