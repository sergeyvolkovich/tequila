# -*- coding: utf-8 -*-
"""
twinlayers — точный порт декодирующего тракта TwinVQ (twinvq.c/twinvqdec.c)
на numpy, ДО уровня MDCT-спектра: permutation -> dequant(cb0+cb1) ->
bark env (с межкадровой историей) -> PPC (very_broken_op) -> LSP env
(включая интерполяционные quirk'и eval_lpcenv_or_interp).

Назначение: послойная декомпозиция того, что кодирует энкодер NTT,
на реальных битстримах. Спектр восстанавливается ровно так, как его
увидит IMDCT декодера FFmpeg.

Зависит от scripts/vqflib.py (парсинг кадров, init_bitstream_params).
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqflib import (MODES, TwinVQParams, mulawinv, rounded_div,
                    GAIN_BITS, SUB_GAIN_BITS, AMP_MAX, SUB_AMP_MAX,
                    MULAW_MU, PGAIN_MU)

FT_SHORT, FT_MEDIUM, FT_LONG, FT_PPC = 0, 1, 2, 3

DATA_H = "/home/z/my-project/tequila/third_party/ffmpeg/twinvq_data.h"
LSP_H = "/home/z/my-project/tequila/third_party/ffmpeg/metasound_twinvq_data.h"

N_LSP = {"08_08": 12, "11_08": 16, "11_10": 16, "16_16": 16, "22_20": 16,
         "22_24": 16, "22_32": 16, "44_40": 20, "44_48": 20}
CB_SUFFIX = {"08_08": "0808", "11_08": "1108", "11_10": "1110", "16_16": "1616",
             "22_20": "2220", "22_24": "2224", "22_32": "2232",
             "44_40": "4440", "44_48": "4448"}
FCB = {"08_08": "fcb08", "11_08": "fcb11", "11_10": "fcb11", "16_16": "fcb16",
       "22_20": "fcb22", "22_24": "fcb22", "22_32": "fcb22",
       "44_40": "fcb44", "44_48": "fcb44"}
# суффиксы имён полей: у большинства режимов fcb<xx><s|l|m>, у 22_х — fcb22<s|m|l>_{1,2}
FCB_SFX = {"22_20": ("s_1", "m_1", "l_1"), "22_24": ("s_1", "m_1", "l_1"),
           "22_32": ("s_2", "m_2", "l_2")}
SHAPE = {"08_08": "shape08", "11_08": "shape11", "11_10": "shape11",
         "16_16": "shape16", "22_20": "shape22_1", "22_24": "shape22_1",
         "22_32": "shape22_2", "44_40": "shape44", "44_48": "shape44"}
LSP_CB = {"08_08": "ff_metasound_lsp8", "11_08": "ff_metasound_lsp11",
          "11_10": "ff_metasound_lsp11", "16_16": "ff_metasound_lsp16",
          "22_20": "ff_metasound_lsp22", "22_24": "ff_metasound_lsp22",
          "22_32": "lsp22_2", "44_40": "ff_metasound_lsp44",
          "44_48": "ff_metasound_lsp44"}
BARK_TAB = {
    "08_08": {0: "bark_tab_s08_64", 1: "bark_tab_m08_256", 2: "bark_tab_l08_512"},
    "11_08": {0: "bark_tab_s11_64", 1: "bark_tab_m11_256", 2: "bark_tab_l11_512"},
    "11_10": {0: "bark_tab_s11_64", 1: "bark_tab_m11_256", 2: "bark_tab_l11_512"},
    "16_16": {0: "bark_tab_s16_128", 1: "bark_tab_m16_512", 2: "bark_tab_l16_1024"},
    "22_20": {0: "bark_tab_s22_128", 1: "bark_tab_m22_512", 2: "bark_tab_l22_1024"},
    "22_24": {0: "bark_tab_s22_128", 1: "bark_tab_m22_512", 2: "bark_tab_l22_1024"},
    "22_32": {0: "bark_tab_s22_128", 1: "bark_tab_m22_256", 2: "bark_tab_l22_512"},
    "44_40": {0: "bark_tab_s44_128", 1: "bark_tab_m44_512", 2: "bark_tab_l44_2048"},
    "44_48": {0: "bark_tab_s44_128", 1: "bark_tab_m44_512", 2: "bark_tab_l44_2048"},
}


# ----------------------------------------------------------------------------
# Парсинг C-заголовков с данными
# ----------------------------------------------------------------------------
def _ints(s):
    return [int(t) for t in re.findall(r"-?\d+", s)]


def _floats(s):
    return [float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


_DATA_CACHE = None


def load_data():
    """Все таблицы из twinvq_data.h + metasound_twinvq_data.h -> dict."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    src = open(DATA_H).read()
    arr = {}
    # простые 1D uint16-таблицы (bark_tab_*)
    for m in re.finditer(r"static const uint16_t (\w+)\[\]\s*=\s*\{([^}]*)\}", src):
        arr[m.group(1)] = _ints(m.group(2))
    # гигантский struct tab = { .field = {...}, ... }  (int16-поля)
    body = src.split("} tab = {", 1)[1].split("\n};", 1)[0]
    float_fields = {"lsp22_2"}
    for m in re.finditer(r"\.(\w+)\s*=\s*\{([^}]*)\}", body):
        name, vals = m.group(1), m.group(2)
        arr[name] = _floats(vals) if name in float_fields else _ints(vals)
    # 2D uint8-таблицы tab7..tab12 (хранятся плоско, как &tabX[0][0] в C)
    for m in re.finditer(r"static const uint8_t (tab\d+)\[\]\[(\d+)\]\s*=\s*\{(.*?)\n\};",
                         src, re.S):
        name = m.group(1)
        rows = re.findall(r"\{([^}]*)\}", m.group(3))
        arr[name] = [v for r in rows for v in _ints(r)]
    # tabs[] для very_broken_op: {size, &tabX[0][0]}
    seg = src.split("} tabs[] = {", 1)[1].split("\n};", 1)[0]
    tabs = []
    for mm in re.finditer(r"\{\s*(\d+),\s*&?(\w+)?", seg):
        size, ref = int(mm.group(1)), mm.group(2)
        tabs.append((size, ref))
    arr["_tabs"] = tabs
    # LSP-кодбуки (float)
    lsrc = open(LSP_H).read()
    for m in re.finditer(r"const float (ff_metasound_lsp\d+)\[\]\s*=\s*\{(.*?)\n\};",
                         lsrc, re.S):
        arr[m.group(1)] = _floats(m.group(2))
    _DATA_CACHE = arr
    return arr


# ----------------------------------------------------------------------------
# Перестановки interleave (construct_perm_table)
# ----------------------------------------------------------------------------
def _permutate_in_line(n_div, num_blocks, block_size, line_len, ftype):
    tab = [0] * (num_blocks * block_size)
    for i in range(line_len[0]):
        if (num_blocks == 1 or
                (ftype == FT_LONG and n_div % num_blocks) or
                (ftype != FT_LONG and n_div & 1) or
                i == line_len[1]):
            shift = 0
        elif ftype == FT_LONG:
            shift = i
        else:
            shift = i * i
        j = 0
        while j < n_div and (j + n_div * i < block_size * num_blocks):
            tab[i * n_div + j] = i * n_div + (j + shift) % n_div
            j += 1
    return tab


def _transpose_perm(n_div, line_len, length_div, src):
    out = []
    for i in range(n_div):
        for j in range(line_len[1 if i >= length_div else 0]):
            out.append(src[j * n_div + i])
    return out


def _linear_perm(n_blocks, size, src):
    block_size = size // n_blocks
    return [block_size * (v % n_blocks) + v // n_blocks for v in src]


# ----------------------------------------------------------------------------
# very_broken_op (twinvqdec.c) — «сломанное» округление NTT
# ----------------------------------------------------------------------------
def _av_log2(x):
    return x.bit_length() - 1 if x > 0 else 0  # av_log2(0)==0 в FFmpeg


class BrokenOp:
    def __init__(self, tabs):
        self.tabs = tabs  # [(size, refname)]

    def __call__(self, a, b):
        x = a * b + 200
        if (x % 400) or (b % 5):
            return x // 400
        x //= 400
        size, ref = self.tabs[b // 5]
        if size == 0 or ref is None:
            return x
        rtab = self.tabs[b // 5]
        # rtab хранится как (size, имя); значение ищем в _data
        return x  # заменяется в TwinDecoder (нужен доступ к таблицам)


# ----------------------------------------------------------------------------
# Декодер спектра
# ----------------------------------------------------------------------------
class TwinSpectrumDecoder:
    """Восстанавливает MDCT-спектр кадра из полей, распарсенных vqflib."""

    def __init__(self, mode_key, n_ch):
        self.key = mode_key
        self.n_ch = n_ch
        self.m = MODES[mode_key]
        self.prm = TwinVQParams(mode_key, n_ch)
        self.size = self.m["size"]
        self.sr = self.m["sr"]
        self.kbps_ch = self.m["kbps_ch"]
        d = load_data()
        self.d = d
        suf = CB_SUFFIX[mode_key]
        self.cb = {}   # ftype -> (cb0, cb1, cb_len) как float ndarray [n, cb_len]
        for ft, sfx in ((0, "s"), (1, "m"), (2, "l")):
            cblr = self.m["fmode"][ft]["cblr"]
            self.cb[ft] = (np.array(d["cb%s%s0" % (suf, sfx)], dtype=np.float64).reshape(-1, cblr),
                           np.array(d["cb%s%s1" % (suf, sfx)], dtype=np.float64).reshape(-1, cblr))
        # PPC shape: cb0 = shape[..], cb1 = shape + cb_len_p*64
        ppc_shape_len = self.prm.ppc[2]
        cb_len_p = (self.prm.n_div[3] + ppc_shape_len * n_ch - 1) // self.prm.n_div[3]
        shp = np.array(d[SHAPE[mode_key]], dtype=np.float64)
        self.ppc_cb = (shp[:cb_len_p * 64].reshape(64, cb_len_p),
                       shp[cb_len_p * 64:2 * cb_len_p * 64].reshape(64, cb_len_p))
        self.cb_len_p = cb_len_p
        # bark
        self.bark_tabs = {ft: d[BARK_TAB[mode_key][ft]] for ft in (0, 1, 2)}
        sfx = FCB_SFX.get(mode_key, ("s", "m", "l"))
        self.bark_cb = {ft: np.array(d[FCB[mode_key] + s], dtype=np.float64)
                        for ft, s in zip((0, 1, 2), sfx)}
        # lsp
        n_lsp = N_LSP[mode_key]
        lsp_all = np.array(d[LSP_CB[mode_key]], dtype=np.float64)
        b0, b1, b2 = self.m["lsp"][0], self.m["lsp"][1], self.m["lsp"][2]
        self.n_lsp = n_lsp
        self.lsp_cb = (lsp_all[: (1 << b1) * n_lsp].reshape(-1, n_lsp),
                       lsp_all[(1 << b1) * n_lsp: (1 << b1) * n_lsp + (1 << b2) * n_lsp].reshape(-1, n_lsp),
                       lsp_all[(1 << b1) * n_lsp + (1 << b2) * n_lsp:
                               (1 << b1) * n_lsp + (1 << b2) * n_lsp + (1 << b0) * n_lsp].reshape(-1, n_lsp))
        # перестановки
        self.perm = {}
        for ft in (0, 1, 2, 3):
            self.perm[ft] = np.array(self._construct_perm(ft), dtype=np.int64)
        # очень сломанная операция
        self._broken_tabs = []
        for size, ref in d["_tabs"]:
            if size and ref:
                self._broken_tabs.append((size, d[ref]))
            else:
                self._broken_tabs.append((0, None))
        # cos-таблицы (init_mdct_win)
        self.cos_tabs = {}
        for ft in (0, 1, 2):
            sub = self.m["fmode"][ft]["sub"]
            m = 4 * self.size // sub
            freq = 2 * np.pi / m
            t = np.zeros(m // 4)
            for j in range(m // 8 + 1):
                t[j] = np.cos((2 * j + 1) * freq)
            for j in range(1, m // 8):
                t[m // 4 - j] = t[j]
            self.cos_tabs[ft] = t
        # состояние (истории)
        self.lsp_hist = [np.zeros(n_lsp, dtype=np.float32) for _ in range(n_ch)]
        self.bark_hist = [[np.full(40, 0.1) for _ in range(n_ch)] for _ in range(3)]

    # -- перестановки --------------------------------------------------------
    def _construct_perm(self, ftype):
        prm = self.prm
        if ftype == FT_PPC:
            size = self.n_ch
            block_size = prm.ppc[2]
        else:
            size = self.n_ch * self.m["fmode"][ftype]["sub"]
            block_size = self.size // self.m["fmode"][ftype]["sub"]
        tmp = _permutate_in_line(prm.n_div[ftype], size, block_size,
                                 prm.length[ftype], ftype)
        t = _transpose_perm(prm.n_div[ftype], prm.length[ftype],
                            prm.length_change[ftype], tmp)
        return _linear_perm(size, size * block_size, t)

    # -- dequant -------------------------------------------------------------
    def dequant(self, cb_bits, ftype, cb0, cb1):
        prm = self.prm
        if ftype == FT_PPC:
            out = np.zeros(self.n_ch * prm.ppc[2])
            cb_len = self.cb_len_p
        else:
            out = np.zeros(self.n_ch * self.size)
            cb_len = self.m["fmode"][ftype]["cblr"]
        pos = 0
        idx = 0
        for i in range(prm.n_div[ftype]):
            length = prm.length[ftype][1 if i >= prm.length_change[ftype] else 0]
            second = 1 if i >= prm.bms_change[ftype] else 0
            b0 = prm.bms[ftype][(0, second)]
            b1 = prm.bms[ftype][(1, second)]
            tmp0 = cb_bits[idx]; idx += 1
            tmp1 = cb_bits[idx]; idx += 1
            sign0 = -1.0 if (b0 == 7 and (tmp0 & 0x40)) else 1.0
            sign1 = -1.0 if (b1 == 7 and (tmp1 & 0x40)) else 1.0
            if b0 == 7:
                tmp0 &= 0x3F
            if b1 == 7:
                tmp1 &= 0x3F
            v = sign0 * cb0[tmp0, :length] + sign1 * cb1[tmp1, :length]
            out[self.perm[ftype][pos:pos + length]] = v
            pos += length
        return out

    # -- gain ----------------------------------------------------------------
    def dec_gain(self, ftype, fr):
        sub = self.m["fmode"][ftype]["sub"]
        step = AMP_MAX / ((1 << GAIN_BITS) - 1)
        sub_step = SUB_AMP_MAX / ((1 << SUB_GAIN_BITS) - 1)
        if ftype == FT_LONG:
            return [(1.0 / (1 << 13)) *
                    mulawinv(step * 0.5 + step * fr["gain"][ch], AMP_MAX, MULAW_MU)
                    for ch in range(self.n_ch)]
        out = []
        for ch in range(self.n_ch):
            val = (1.0 / (1 << 23)) * \
                mulawinv(step * 0.5 + step * fr["gain"][ch], AMP_MAX, MULAW_MU)
            for j in range(sub):
                out.append(val * mulawinv(sub_step * 0.5 + sub_step * fr["sub_gain"][ch * sub + j],
                                          SUB_AMP_MAX, MULAW_MU))
        return out

    # -- bark env --------------------------------------------------------------
    def dec_bark_env(self, ftype, ch, sub_j, fr, gain):
        fm = self.m["fmode"][ftype]
        hist = self.bark_hist[ftype][ch]
        val = {0: 0.4, 1: 0.35, 2: 0.28}[ftype]
        ncoef = fm["ncoef"]
        fw_cb_len = fm["env"] // ncoef
        bark_tab = self.bark_tabs[ftype]
        bark_cb = self.bark_cb[ftype]
        in_coef = fr["bark1"][ch * fm["sub"] + sub_j]  # [ncoef] индексов
        use_hist = fr["bark_hist"][ch * fm["sub"] + sub_j]
        out = []
        idx = 0
        for i in range(fw_cb_len):
            for j in range(ncoef):
                tmp2 = bark_cb[fw_cb_len * in_coef[j] + i] * (1.0 / 4096)
                if use_hist:
                    st = (1.0 - val) * tmp2 + val * hist[idx] + 1.0
                else:
                    st = tmp2 + 1.0
                hist[idx] = tmp2
                if st < -1.0:
                    st = 1.0
                out.extend([st * gain] * bark_tab[idx])
                idx += 1
        return np.array(out)

    # -- PPC -------------------------------------------------------------------
    def very_broken_op(self, a, b):
        x = a * b + 200
        if (x % 400) or (b % 5):
            return x // 400
        x //= 400
        size, rtab = self._broken_tabs[b // 5]
        if size == 0 or rtab is None:
            return x
        return x - rtab[size * _av_log2(2 * (x - 1) // size) + (x - 1) % size]

    def decode_ppc(self, ch, fr, shape):
        prm = self.prm
        ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, per2wid = prm.ppc
        isampf = self.sr // 1000
        ibps = self.kbps_ch
        min_period = rounded_div(40 * 2 * self.size, isampf)
        max_period = rounded_div(40 * 2 * self.size * 6, isampf)
        period_range = max_period - min_period
        pgain_step = 25000.0 / ((1 << pgain_bit) - 1)
        ppc_gain = (1.0 / 8192) * mulawinv(
            pgain_step * fr["g_coef"][ch] + pgain_step / 2, 25000.0, PGAIN_MU)
        period = min_period + rounded_div(fr["p_coef"][ch] * period_range,
                                          (1 << ppc_period_bit) - 1)
        if isampf == 22 and ibps == 32:
            width = rounded_div((period + 800) * per2wid, 400 * self.size)
        else:
            width = period * per2wid // (400 * self.size)
        speech = np.zeros(self.size)
        # add_peak
        si = 0
        shape = np.asarray(shape, dtype=np.float64)
        for i in range(width // 2):
            speech[i] += ppc_gain * shape[si]
            si += 1
        i = 1
        while i < rounded_div(ppc_shape_len, width):
            center = self.very_broken_op(period, i)
            for j in range(-(width // 2), (width + 1) // 2):  # C: -w/2 (trunc)
                p = j + center
                if 0 <= p < self.size:
                    speech[p] += ppc_gain * shape[si]
                si += 1
            i += 1
        center = self.very_broken_op(period, i)
        for j in range(-(width // 2), (width + 1) // 2):
            if si >= ppc_shape_len:
                break
            p = j + center
            if 0 <= p < self.size:
                speech[p] += ppc_gain * shape[si]
            si += 1
        return speech, dict(period=period, width=width, ppc_gain=ppc_gain,
                            p_coef=fr["p_coef"][ch], g_coef=fr["g_coef"][ch])

    # -- LSP -------------------------------------------------------------------
    @staticmethod
    def _rearrange_lsp(lsp, min_dist):
        min_dist2 = min_dist * 0.5
        for i in range(1, len(lsp)):
            if lsp[i] - lsp[i - 1] < min_dist:
                avg = (lsp[i] + lsp[i - 1]) * 0.5
                lsp[i - 1] = avg - min_dist2
                lsp[i] = avg + min_dist2

    @staticmethod
    def _sort_nearly_sorted(vals):
        for i in range(len(vals) - 1):
            j = i
            while j >= 0 and vals[j] > vals[j + 1]:
                vals[j], vals[j + 1] = vals[j + 1], vals[j]
                j -= 1

    def decode_lsp(self, ch, fr):
        m = self.m
        lsp_split = m["lsp"][3]
        funny = [-2, -2 if lsp_split == 4 else 1, -2 if lsp_split == 4 else 1, 0]
        cb, cb2, cb3 = self.lsp_cb
        lsp = np.zeros(self.n_lsp, dtype=np.float32)
        j = 0
        for i in range(lsp_split):
            chunk_end = ((i + 1) * self.n_lsp + funny[i]) // lsp_split
            idx2 = fr["lpc2"][ch][i]
            while j < chunk_end:
                lsp[j] = np.float32(cb[fr["lpc1"][ch], j] + cb2[idx2, j])
                j += 1
        self._rearrange_lsp(lsp, np.float32(0.0001))
        hist = self.lsp_hist[ch]
        ma = cb3[fr["lpc_hist"][ch]].astype(np.float32)
        newlsp = (lsp * (np.float32(1.0) - ma) + hist * ma).astype(np.float32)
        self.lsp_hist[ch] = lsp.copy()
        lsp = newlsp
        self._rearrange_lsp(lsp, np.float32(0.0001))
        self._rearrange_lsp(lsp, np.float32(0.000095))
        self._sort_nearly_sorted(lsp)
        return lsp

    # -- LPC env -----------------------------------------------------------------
    @staticmethod
    def _eval_lpc_spectrum(lsp, cos_val, order):
        # float32 — как в FFmpeg: у полюсов 0.5/(p+q) разница float64/float32
        # амплифицируется на порядки
        p = np.float32(0.5)
        q = np.float32(0.5)
        two_cos_w = np.float32(2.0) * np.float32(cos_val)
        for j in range(0, order - 1, 4):
            q *= np.float32(lsp[j]) - two_cos_w
            p *= np.float32(lsp[j + 1]) - two_cos_w
            q *= np.float32(lsp[j + 2]) - two_cos_w
            p *= np.float32(lsp[j + 3]) - two_cos_w
        p = np.float32(p * p * (np.float32(2.0) - two_cos_w))
        q = np.float32(q * q * (np.float32(2.0) + two_cos_w))
        return np.float32(np.float32(0.5) / (p + q))

    @classmethod
    def _eval_lpc_spectrum_vec(cls, lsp, cos_vals, order):
        return np.array([cls._eval_lpc_spectrum(lsp, c, order) for c in cos_vals])

    def _interp(self, out, i0, v1, v2, n):
        # interpolate(out+i0, v1, v2, n): step=(v1-v2)/(n+1); out[i0+t]=v2+step*(t+1)
        step = (v1 - v2) / (n + 1)
        vv = v2
        for t in range(n):
            vv += step
            out[i0 + t] = vv

    def _eval_or_interp(self, ftype, out, o0, in_lsp, size, step, part):
        cos_tab = self.cos_tabs[ftype]
        order = self.n_lsp

        def get_cos(idx):
            return -cos_tab[size - idx - 1] if part else cos_tab[idx]

        for i in range(0, size, step):
            out[o0 + i] = self._eval_lpc_spectrum(in_lsp, get_cos(i), order)
        for i in range(step, size - 2 * step + 1, step):
            if (out[o0 + i + step] + out[o0 + i - step] > 1.95 * out[o0 + i]
                    or out[o0 + i + step] >= out[o0 + i - step]):
                self._interp(out, o0 + i - step + 1, out[o0 + i], out[o0 + i - step], step - 1)
            else:
                out[o0 + i - step // 2] = self._eval_lpc_spectrum(
                    in_lsp, get_cos(i - step // 2), order)
                self._interp(out, o0 + i - step + 1, out[o0 + i - step // 2],
                             out[o0 + i - step], step // 2 - 1)
                self._interp(out, o0 + i - step // 2 + 1, out[o0 + i],
                             out[o0 + i - step // 2], step // 2 - 1)
        self._interp(out, o0 + size - 2 * step + 1, out[o0 + size - step],
                     out[o0 + size - 2 * step], step - 1)

    def dec_lpc_env(self, ftype, lsp):
        size = self.size // self.m["fmode"][ftype]["sub"]
        # C: lsp[i] = 2*cos(lsp[i]) — cos в double, присваивание во float
        lsp2 = (2.0 * np.cos(lsp.astype(np.float64))).astype(np.float32)
        lpc = np.zeros(size, dtype=np.float32)
        if ftype == FT_LONG:
            self._eval_or_interp(ftype, lpc, 0, lsp2, size // 2, 8, 0)
            self._eval_or_interp(ftype, lpc, size // 2, lsp2, size // 2, 16, 1)
            self._interp(lpc, size // 2 - 8 + 1, lpc[size // 2], lpc[size // 2 - 8], 8)
            lpc[size - 2 * 8 + 1:size] = lpc[size - 2 * 8]
        elif ftype == FT_MEDIUM:
            self._eval_or_interp(ftype, lpc, 0, lsp2, size // 2, 2, 0)
            self._eval_or_interp(ftype, lpc, size // 2, lsp2, size // 2, 4, 1)
            self._interp(lpc, size // 2 - 2 + 1, lpc[size // 2], lpc[size // 2 - 2], 2)
            lpc[size - 2 * 2 + 1:size] = lpc[size - 2 * 2]
        else:  # SHORT
            size_s = self.size // self.m["fmode"][FT_SHORT]["sub"]
            cos_tab = self.cos_tabs[FT_SHORT]
            for i in range(size_s // 2):
                lpc[i] = self._eval_lpc_spectrum(lsp2, cos_tab[i], self.n_lsp)
                lpc[size_s - i - 1] = self._eval_lpc_spectrum(lsp2, -cos_tab[i], self.n_lsp)
        return lpc

    # -- полный кадр ---------------------------------------------------------------
    def decode_frame(self, fr):
        """Спектр [n_ch, size] + послойные компоненты."""
        ftype = fr["ftype"]
        fm = self.m["fmode"][ftype]
        sub = fm["sub"]
        block_size = self.size // sub
        # main VQ: пары (main0[i], main1[i]) чередуются как в read_cb_data
        cb_bits = []
        for i in range(self.prm.n_div[ftype]):
            cb_bits.append(fr["main0"][i])
            cb_bits.append(fr["main1"][i])
        spec = self.dequant(cb_bits, ftype, *self.cb[ftype])
        gains = self.dec_gain(ftype, fr)
        ppc_shape = None
        if ftype == FT_LONG:
            pcb = []
            for i in range(self.prm.n_div[3]):
                pcb.append(fr["ppc0"][i])
                pcb.append(fr["ppc1"][i])
            ppc_shape = self.dequant(pcb, FT_PPC, *self.ppc_cb)
        layers = []
        for ch in range(self.n_ch):
            chunk = spec[ch * self.size:(ch + 1) * self.size].copy()
            vq = chunk.copy()
            bark = np.zeros(self.size)
            for j in range(sub):
                env = self.dec_bark_env(ftype, ch, j, fr, gains[sub * ch + j] if ftype != FT_LONG else gains[ch])
                bs = block_size * j
                bark[bs:bs + len(env)] = env
                chunk[bs:bs + block_size] *= env  # env покрывает ровно block_size
            after_bark = chunk.copy()
            ppc = np.zeros(self.size)
            ppc_info = None
            if ftype == FT_LONG:
                ppc, ppc_info = self.decode_ppc(ch, fr,
                                                ppc_shape[ch * self.prm.ppc[2]:(ch + 1) * self.prm.ppc[2]])
                chunk += ppc
            lsp = self.decode_lsp(ch, fr)
            lpc_env = self.dec_lpc_env(ftype, lsp)
            # lpc_env имеет длину block_size и применяется к КАЖДОМУ подкадру
            for j in range(sub):
                bs = block_size * j
                chunk[bs:bs + block_size] *= lpc_env
            layers.append(dict(vq=vq, bark=bark, after_bark=after_bark,
                               ppc=ppc, ppc_info=ppc_info, lsp=lsp,
                               lpc_env=lpc_env, spec=chunk))
        return layers


# ----------------------------------------------------------------------------
# Полный синтез во временную область (порт imdct_and_window + imdct_output)
# ----------------------------------------------------------------------------
_IMDCT_CACHE = {}
_SINE_CACHE = {}


def imdct_compact_matrix(B, norm):
    """Компактный IMDCT (tx_template.c, без AV_TX_FULL_IMDCT):
    y[i]      = s·Σ X[k]·cos((2k+1)(2B−2i−1)π/(4B))   i < B/2
    y[i+B/2]  = −s·Σ X[k]·cos((2k+1)(3B+2i+1)π/(4B))
    s = −sqrt(norm/B)/2^15.
    """
    key = (B, norm)
    if key in _IMDCT_CACHE:
        return _IMDCT_CACHE[key]
    s = -np.sqrt(norm / B) / 32768.0
    k = np.arange(B, dtype=np.float64)
    i = np.arange(B // 2, dtype=np.float64)
    C1 = np.cos((2 * k[None, :] + 1) * (2 * B - 2 * i[:, None] - 1) * np.pi / (4 * B))
    C2 = np.cos((2 * k[None, :] + 1) * (3 * B + 2 * i[:, None] + 1) * np.pi / (4 * B))
    M = np.empty((B, B))
    M[:B // 2] = s * C1
    M[B // 2:] = -s * C2
    _IMDCT_CACHE[key] = M
    return M


def sine_table(wsize):
    """ff_sine_windows[log2(wsize)]: W[m] = sin((m+0.5)π/(2·wsize)), m < wsize."""
    if wsize not in _SINE_CACHE:
        m = np.arange(wsize, dtype=np.float64)
        _SINE_CACHE[wsize] = np.sin((m + 0.5) * np.pi / (2.0 * wsize))
    return _SINE_CACHE[wsize]


class TwinSynth:
    """IMDCT + окна + TDAC-цепочка. Выдаёт PCM кадрами по size сэмплов."""

    WTYPE_TO_WSIZE = [0, 0, 2, 2, 2, 1, 0, 1, 1]

    def __init__(self, dec):
        self.dec = dec
        self.size = dec.size
        self.n_ch = dec.n_ch
        m = dec.m
        self.types_sizes = [
            dec.size // m["fmode"][2]["sub"],       # LONG
            dec.size // m["fmode"][1]["sub"],       # MEDIUM
            dec.size // (m["fmode"][0]["sub"] * 2)  # SHORT-окно
        ]
        self.norm = 1.0 if dec.n_ch == 2 else 2.0
        self.curr = [np.zeros(2 * dec.size) for _ in range(dec.n_ch)]
        self.prev = [np.zeros(2 * dec.size) for _ in range(dec.n_ch)]
        self.last_pos = 0

    def _fmul_window(self, D, doff, sbuf, soff, S1, W, L):
        # vector_fmul_window: D[t]=P0[t]·W[2L−1−t]−S1[L−1−t]·W[t];
        #                     D[2L−1−t]=P0[t]·W[t]+S1[L−1−t]·W[2L−1−t]
        t = np.arange(L)
        s0 = sbuf[soff:soff + L]
        s1 = S1[L - 1 - t]
        w1 = W[t]
        w2 = W[2 * L - 1 - t]
        D[doff + t] = s0 * w2 - s1 * w1
        D[doff + 2 * L - 1 - t] = s0 * w1 + s1 * w2

    def _imdct_and_window(self, ftype, wtype, X_ch, prev_ch, out_ch, old_pos):
        sub = self.dec.m["fmode"][ftype]["sub"]
        B = self.size // sub
        ts = self.types_sizes
        first_wsize = ts[self.WTYPE_TO_WSIZE[wtype]]
        buf = np.zeros(sub * B)
        # ВАЖНО (FFmpeg): база prev вычисляется по СТАРОМУ last_block_pos,
        # одному для обоих каналов — указатель захвачен ДО цикла каналов
        prev_src = (prev_ch, old_pos + (self.size - B) // 2)
        out2 = 0
        for j in range(sub):
            sub_wtype = 8 if ftype == FT_MEDIUM else wtype
            if j == 0 and wtype == 4:
                sub_wtype = 4
            elif j == sub - 1 and wtype == 7:
                sub_wtype = 7
            wsize = ts[self.WTYPE_TO_WSIZE[sub_wtype]]
            y = imdct_compact_matrix(B, self.norm) @ X_ch[j * B:(j + 1) * B]
            buf[j * B:(j + 1) * B] = y
            sbuf, soff = prev_src
            self._fmul_window(out_ch, out2, sbuf, soff + (B - wsize) // 2,
                              y, sine_table(wsize), wsize // 2)
            out2 += wsize
            n = B - wsize // 2
            out_ch[out2:out2 + n] = y[wsize // 2:B]
            out2 += (B - wsize) // 2 if ftype == FT_MEDIUM else B - wsize
            prev_src = (buf, j * B + B // 2)
        return (self.size + first_wsize) // 2

    def synth_frame(self, ftype, wtype, spectra):
        """spectra: [n_ch][size]. Возвращает [n_ch][size] PCM (кадр)."""
        old_pos = self.last_pos
        for ch in range(self.n_ch):
            self.last_pos = self._imdct_and_window(ftype, wtype, spectra[ch],
                                                   self.prev[ch], self.curr[ch],
                                                   old_pos)
        size2 = self.last_pos
        size1 = self.size - size2
        out = []
        for ch in range(self.n_ch):
            o = np.zeros(self.size)
            o[:size1] = self.prev[ch][old_pos:old_pos + size1]
            o[size1:] = self.curr[ch][:size2]
            out.append(o)
        if self.n_ch == 2:
            out = [out[0] + out[1], out[0] - out[1]]  # бабочка L/R
        self.curr, self.prev = self.prev, self.curr  # FFSWAP
        return out


# ----------------------------------------------------------------------------
# Прямой MDCT (naive, конвенция FFmpeg TX) + чтение WAV
# ----------------------------------------------------------------------------
_MDCT_CACHE = {}


def mdct_matrix(B):
    if B in _MDCT_CACHE:
        return _MDCT_CACHE[B]
    j = np.arange(2 * B, dtype=np.float64)
    i = np.arange(B, dtype=np.float64)
    C = np.cos((2 * j[None, :] + 1 + B) * (2 * i[:, None] + 1) * np.pi / (4 * B))
    _MDCT_CACHE[B] = C
    return C


def mdct_fwd(x, B, window=None):
    """X[i] = sum_j x[j]*w[j]*cos((2j+1+B)(2i+1)pi/(4B)), i<B."""
    x = np.asarray(x, dtype=np.float64)
    assert len(x) == 2 * B
    if window is not None:
        x = x * window
    return mdct_matrix(B) @ x


def sine_window_2B(B):
    j = np.arange(2 * B, dtype=np.float64)
    return np.sin((j + 0.5) * np.pi / (2 * B))


def read_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    assert sw == 2, "ожидается 16-битный PCM"
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch)
    return x
