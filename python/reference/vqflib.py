# -*- coding: utf-8 -*-
"""
vqflib — синтетический writer VQF/TwinVQ битстримов + декодер-зонд.

Реализует (cleanroom, по спецификации из FFmpeg twinvqdec.c/twinvq.c/vqf.c):
  * контейнер VQF (TWIN97012000, чанки COMM/DSIZ/NAME/COMT/DATA)
  * побитовую упаковку кадров (MSB-first, frame_bit_len бит на кадр)
  * порядок полей кадра twinvq_read_bitstream
  * init_bitstream_params (n_div, bits_main_spec, length, ...)

Используется для эмпирического зондирования декодера FFmpeg.
"""
import io
import struct
import subprocess
import numpy as np

# ----------------------------------------------------------------------------
# Таблицы режимов (портированы из twinvqdec.c, поля названы как в twinvq.h)
# fmode: (sub, bark_env_size, bark_n_coef, bark_n_bit, cb_len_read)
#        bark_tab имя оставлено справочно
# далее: size, n_lsp, lsp_bit0, lsp_bit1, lsp_bit2, lsp_split,
#        ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, peak_per2wid
# ----------------------------------------------------------------------------
MODES = {
    "08_08": dict(
        sr=8000, kbps_ch=8,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=5, cblr=18),
               1: dict(sub=2,  env=20, ncoef=2, nbit=5, cblr=16),
               2: dict(sub=1,  env=30, ncoef=3, nbit=6, cblr=17)},
        size=512, lsp=(1, 5, 3, 3), ppc=(8, 28, 20, 6, 40)),
    "11_08": dict(
        sr=11025, kbps_ch=8,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=5, cblr=29),
               1: dict(sub=2,  env=20, ncoef=2, nbit=5, cblr=24),
               2: dict(sub=1,  env=30, ncoef=3, nbit=6, cblr=27)},
        size=512, lsp=(1, 6, 4, 3), ppc=(9, 36, 30, 7, 90)),
    "11_10": dict(
        sr=11025, kbps_ch=10,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=5, cblr=21),
               1: dict(sub=2,  env=20, ncoef=2, nbit=5, cblr=18),
               2: dict(sub=1,  env=30, ncoef=3, nbit=6, cblr=20)},
        size=512, lsp=(1, 6, 4, 3), ppc=(9, 36, 30, 7, 90)),
    "16_16": dict(
        sr=16000, kbps_ch=16,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=5, cblr=16),
               1: dict(sub=2,  env=20, ncoef=2, nbit=5, cblr=15),
               2: dict(sub=1,  env=30, ncoef=3, nbit=6, cblr=16)},
        size=1024, lsp=(1, 6, 4, 3), ppc=(9, 56, 60, 7, 180)),
    "22_20": dict(
        sr=22050, kbps_ch=20,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=6, cblr=18),
               1: dict(sub=2,  env=20, ncoef=2, nbit=6, cblr=17),
               2: dict(sub=1,  env=32, ncoef=4, nbit=6, cblr=18)},
        size=1024, lsp=(1, 6, 4, 3), ppc=(9, 56, 36, 7, 144)),
    "22_24": dict(
        sr=22050, kbps_ch=24,
        fmode={0: dict(sub=8,  env=10, ncoef=1, nbit=6, cblr=15),
               1: dict(sub=2,  env=20, ncoef=2, nbit=6, cblr=14),
               2: dict(sub=1,  env=32, ncoef=4, nbit=6, cblr=15)},
        size=1024, lsp=(1, 6, 4, 3), ppc=(9, 56, 36, 7, 144)),
    "22_32": dict(
        sr=22050, kbps_ch=32,
        fmode={0: dict(sub=4,  env=10, ncoef=1, nbit=6, cblr=11),
               1: dict(sub=2,  env=20, ncoef=2, nbit=6, cblr=11),
               2: dict(sub=1,  env=32, ncoef=4, nbit=6, cblr=12)},
        size=512, lsp=(1, 6, 4, 4), ppc=(9, 56, 36, 7, 72)),
    "44_40": dict(
        sr=44100, kbps_ch=40,
        fmode={0: dict(sub=16, env=10, ncoef=1, nbit=6, cblr=18),
               1: dict(sub=4,  env=20, ncoef=2, nbit=6, cblr=17),
               2: dict(sub=1,  env=40, ncoef=4, nbit=6, cblr=17)},
        size=2048, lsp=(1, 6, 4, 4), ppc=(9, 84, 54, 7, 432)),
    "44_48": dict(
        sr=44100, kbps_ch=48,
        fmode={0: dict(sub=16, env=10, ncoef=1, nbit=6, cblr=15),
               1: dict(sub=4,  env=20, ncoef=2, nbit=6, cblr=14),
               2: dict(sub=1,  env=40, ncoef=4, nbit=6, cblr=14)},
        size=2048, lsp=(1, 6, 4, 4), ppc=(9, 84, 54, 7, 432)),
}

WTYPE_TO_FTYPE = [2, 2, 0, 2, 1, 2, 2, 1, 1]  # ff_twinvq_wtype_to_ftype_table
GAIN_BITS, SUB_GAIN_BITS, WTYPE_BITS = 8, 5, 4
AMP_MAX, SUB_AMP_MAX, MULAW_MU = 13000.0, 4500.0, 100.0
PGAIN_MU = 200.0


def mulawinv(y, clip, mu):
    y = float(np.clip(y / clip, -1.0, 1.0))
    if y == 0:
        return 0.0
    s = 1.0 if y > 0 else -1.0
    return clip * s * (np.exp(np.log(1 + mu) * abs(y)) - 1.0) / mu


def rounded_div(a, b):
    return (a + (b >> 1)) // b if b > 0 else 0  # ROUNDED_DIV из FFmpeg


class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def put(self, value, nbits):
        if nbits == 0:
            return
        value &= (1 << nbits) - 1
        self.acc = (self.acc << nbits) | value
        self.nbits += nbits
        while self.nbits >= 8:
            self.nbits -= 8
            self.buf.append((self.acc >> self.nbits) & 0xFF)
        self.acc &= (1 << self.nbits) - 1

    def bitlen(self):
        return len(self.buf) * 8 + self.nbits

    def bytes(self, pad_to_bits=None):
        """Финализировать (MSB-first поток)."""
        out = bytearray(self.buf)
        acc, nb = self.acc, self.nbits
        if pad_to_bits is not None:
            while len(out) * 8 + nb < pad_to_bits:
                if nb < 8:
                    acc = acc << 1  # добиваем нулями
                    nb += 1
                    if nb == 8:
                        out.append(acc & 0xFF)
                        acc, nb = 0, 0
                else:
                    out.append(0)
            # хвостовые биты текущего аккумулятора
            if nb:
                out.append((acc << (8 - nb)) & 0xFF)
            return bytes(out)
        if nb:
            out.append((acc << (8 - nb)) & 0xFF)
        return bytes(out)


class TwinVQParams:
    """init_bitstream_params для конкретного (mode, n_ch)."""

    def __init__(self, mode_key, n_ch):
        m = MODES[mode_key]
        self.m = m
        self.key = mode_key
        self.n_ch = n_ch
        self.sr = m["sr"]
        self.size = m["size"]
        self.total_fr_bits = m["kbps_ch"] * 1000 * n_ch * m["size"] // m["sr"]
        lsp_bit0, lsp_bit1, lsp_bit2, lsp_split = m["lsp"]
        ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, per2wid = m["ppc"]
        self.lsp = m["lsp"]
        self.ppc = m["ppc"]
        lsp_bits = n_ch * (lsp_bit0 + lsp_bit1 + lsp_split * lsp_bit2)
        ppc_bits = n_ch * (pgain_bit + ppc_shape_bit + ppc_period_bit)
        bse = {i: n_ch * (m["fmode"][i]["ncoef"] * m["fmode"][i]["nbit"] + 1)
               for i in range(3)}
        self.bse = bse
        no_main = {2: bse[2] + lsp_bits + ppc_bits + WTYPE_BITS + n_ch * GAIN_BITS}
        for i in range(2):
            no_main[i] = (lsp_bits + n_ch * GAIN_BITS + WTYPE_BITS +
                          m["fmode"][i]["sub"] * (bse[i] + n_ch * SUB_GAIN_BITS))
        self.no_main = no_main
        self.lsp_bits, self.ppc_bits = lsp_bits, ppc_bits

        self.n_div, self.bms, self.bms_change = {}, {}, {}
        self.length, self.length_change = {}, {}
        for i in range(4):
            if i == 3:
                bit_size = n_ch * ppc_shape_bit
                vect_size = n_ch * ppc_shape_len
            else:
                bit_size = self.total_fr_bits - no_main[i]
                vect_size = n_ch * m["size"]
            nd = (bit_size + 13) // 14
            self.n_div[i] = nd
            ru = (bit_size + nd - 1) // nd
            rd = bit_size // nd
            nrd = ru * nd - bit_size
            nru = nd - nrd
            self.bms[i] = {(0, 0): (ru + 1) // 2, (1, 0): ru // 2,
                           (0, 1): (rd + 1) // 2, (1, 1): rd // 2}
            self.bms_change[i] = nru
            vru = (vect_size + nd - 1) // nd
            vrd = vect_size // nd
            vnrd = vru * nd - vect_size
            self.length[i] = {0: vru, 1: vrd}
            self.length_change[i] = nd - vnrd
            # контроль: суммарные биты div'ов == bit_size
            total = (ru * nru + rd * nrd)
            assert total == bit_size, (mode_key, n_ch, i, total, bit_size)

    def field_bits(self, ftype):
        """Сколько бит занимает каждое поле кадра данного ftype (для аудита)."""
        m, n_ch = self.m, self.n_ch
        sub = m["fmode"][ftype]["sub"]
        d = {"wtype": WTYPE_BITS}
        d["main_vq"] = sum(
            self.bms[ftype][(0, i >= self.bms_change[ftype])] +
            self.bms[ftype][(1, i >= self.bms_change[ftype])]
            for i in range(self.n_div[ftype]))
        d["bark1"] = n_ch * sub * m["fmode"][ftype]["ncoef"] * m["fmode"][ftype]["nbit"]
        d["bark_hist"] = n_ch * sub
        if ftype == 2:
            d["gain"] = n_ch * GAIN_BITS
            d["sub_gain"] = 0
            d["ppc"] = 0
        else:
            d["gain"] = n_ch * GAIN_BITS
            d["sub_gain"] = n_ch * sub * SUB_GAIN_BITS
            d["ppc"] = (0 if ftype != 2 else 0)
        d["lsp"] = self.lsp_bits
        if ftype == 2:
            d["ppc"] = self.ppc_bits + sum(
                self.bms[3][(0, i >= self.bms_change[3])] +
                self.bms[3][(1, i >= self.bms_change[3])]
                for i in range(self.n_div[3]))
            # ppc_bits уже содержит shape_bit+period+gain
            d["ppc"] = self.ppc_bits
        tot = sum(d.values())
        assert tot == self.total_fr_bits, (self.key, n_ch, ftype, tot, self.total_fr_bits)
        return d


def build_frame(prm, wtype=0, main_cb=None, ppc_cb=None,
                bark_idx=None, bark_hist=None, gain=None, sub_gain=None,
                lpc_hist_idx=None, lpc_idx1=None, lpc_idx2=None,
                p_coef=None, g_coef=None):
    """Собрать один кадр (ровно total_fr_bits бит).

    Все параметры могут быть скаляром (broadcast на все вхождения) или
    вложенными структурами правильной формы. По умолчанию — нули.
    """
    m, n_ch = prm.m, prm.n_ch
    ftype = WTYPE_TO_FTYPE[wtype]
    fm = m["fmode"][ftype]
    sub = fm["sub"]
    lsp_bit0, lsp_bit1, lsp_bit2, lsp_split = m["lsp"]
    ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, per2wid = m["ppc"]

    def grid_ch_sub_k(fn):
        return [[fn(ch, s, k) for k in range(fm["ncoef"])] for s in range(sub)
                for ch in range(n_ch)]

    bw = BitWriter()
    bw.put(wtype, WTYPE_BITS)
    # --- main VQ (read_cb_data) ---
    for i in range(prm.n_div[ftype]):
        second = i >= prm.bms_change[ftype]
        b0 = prm.bms[ftype][(0, second)]
        b1 = prm.bms[ftype][(1, second)]
        v0 = main_cb(ch=None, i=i, half=0) if main_cb else 0
        v1 = main_cb(ch=None, i=i, half=1) if main_cb else 0
        bw.put(v0, b0)
        bw.put(v1, b1)
    # --- bark1 ---
    for ch in range(n_ch):
        for s in range(sub):
            for k in range(fm["ncoef"]):
                v = bark_idx(ch, s, k) if bark_idx else 0
                bw.put(v, fm["nbit"])
    # --- bark_use_hist ---
    for ch in range(n_ch):
        for s in range(sub):
            v = bark_hist(ch, s) if bark_hist else 0
            bw.put(v, 1)
    # --- gain / sub_gain ---
    if ftype == 2:  # LONG
        for ch in range(n_ch):
            bw.put(gain(ch) if gain else 0, GAIN_BITS)
    else:
        for ch in range(n_ch):
            bw.put(gain(ch) if gain else 0, GAIN_BITS)
            for s in range(sub):
                bw.put(sub_gain(ch, s) if sub_gain else 0, SUB_GAIN_BITS)
    # --- LSP ---
    for ch in range(n_ch):
        bw.put(lpc_hist_idx(ch) if lpc_hist_idx else 0, lsp_bit0)
        bw.put(lpc_idx1(ch) if lpc_idx1 else 0, lsp_bit1)
        for j in range(lsp_split):
            v = lpc_idx2(ch, j) if lpc_idx2 else 0
            bw.put(v, lsp_bit2)
    # --- PPC (только LONG) ---
    if ftype == 2:
        for i in range(prm.n_div[3]):
            second = i >= prm.bms_change[3]
            b0 = prm.bms[3][(0, second)]
            b1 = prm.bms[3][(1, second)]
            v0 = ppc_cb(i=i, half=0) if ppc_cb else 0
            v1 = ppc_cb(i=i, half=1) if ppc_cb else 0
            bw.put(v0, b0)
            bw.put(v1, b1)
        for ch in range(n_ch):
            bw.put(p_coef(ch) if p_coef else 0, ppc_period_bit)
            bw.put(g_coef(ch) if g_coef else 0, pgain_bit)
    assert bw.bitlen() == prm.total_fr_bits, (bw.bitlen(), prm.total_fr_bits)
    return bw.bytes(pad_to_bits=prm.total_fr_bits)


def pack_frames(prm, frames):
    """Плотный битовый поток: кадр i начинается с бита i*total_fr_bits."""
    n = prm.total_fr_bits
    total_bits = n * len(frames)
    pad_bits = (-total_bits) % 8
    stream = BitWriter()
    for f in frames:
        # f = ceil(n/8) байт: последние len(f)*8-n бит — паддинг, их сбрасываем
        pad = len(f) * 8 - n
        v = int.from_bytes(f, "big")
        if pad:
            v >>= pad
        stream.put(v, n)
    if pad_bits:
        stream.put(0, pad_bits)
    data = stream.bytes() + b"\x00\x00"  # +2 запасных байта
    return data


class BitReader:
    """MSB-first побитовый читатель (зеркало BitWriter)."""

    def __init__(self, data):
        self.data = data
        self.pos = 0  # битовая позиция

    def get(self, nbits):
        v = 0
        for _ in range(nbits):
            byte = self.data[self.pos >> 3] if (self.pos >> 3) < len(self.data) else 0
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def remaining(self):
        return len(self.data) * 8 - self.pos


def mode_from_comm(path):
    """Определить (mode_key, n_ch) из COMM-чанка .vqf."""
    d = open(path, "rb").read()
    assert d[:4] == b"TWIN", "not a VQF"
    hs = struct.unpack(">I", d[12:16])[0]
    off = 16
    end = 16 + hs
    n_ch = br = rf = None
    while off < min(end + 4, len(d)):
        tag = d[off:off + 4]
        if tag == b"DATA":
            data_off = off + 4  # у DATA нет поля длины — сразу битстрим
            break
        ln = struct.unpack(">I", d[off + 4:off + 8])[0]
        if tag == b"COMM":
            n_ch = struct.unpack(">I", d[off + 8:off + 12])[0] + 1
            br = struct.unpack(">I", d[off + 12:off + 16])[0]
            rf = struct.unpack(">I", d[off + 16:off + 20])[0]
        off += 8 + ln
    else:
        data_off = None
    kbps_ch = br // n_ch
    sr = {44: 44100, 22: 22050, 11: 11025}.get(rf, rf * 1000)
    key = f"{sr//1000 if sr % 1000 == 0 else sr}_{kbps_ch}"
    # коррекция имён: 11025 → 11
    for k in MODES:
        if MODES[k]["sr"] == sr and MODES[k]["kbps_ch"] == kbps_ch:
            key = k
            break
    return key, n_ch, data_off, d[data_off:] if data_off else b""


def parse_frame(prm, br):
    """Разобрать один кадр BitReader → dict полей (зеркало build_frame)."""
    m, n_ch = prm.m, prm.n_ch
    wtype = br.get(WTYPE_BITS)
    assert wtype <= 8, f"invalid wtype {wtype} at bit {br.pos-4}"
    ftype = WTYPE_TO_FTYPE[wtype]
    fm = m["fmode"][ftype]
    sub = fm["sub"]
    lsp_bit0, lsp_bit1, lsp_bit2, lsp_split = m["lsp"]
    ppc_period_bit, ppc_shape_bit, ppc_shape_len, pgain_bit, per2wid = m["ppc"]

    fr = dict(wtype=wtype, ftype=ftype, main0=[], main1=[],
              bark1=[], bark_hist=[], gain=[], sub_gain=[],
              lpc_hist=[], lpc1=[], lpc2=[], ppc0=[], ppc1=[],
              p_coef=[], g_coef=[])
    for i in range(prm.n_div[ftype]):
        second = i >= prm.bms_change[ftype]
        fr["main0"].append(br.get(prm.bms[ftype][(0, second)]))
        fr["main1"].append(br.get(prm.bms[ftype][(1, second)]))
    for ch in range(n_ch):
        for s in range(sub):
            fr["bark1"].append([br.get(fm["nbit"]) for _ in range(fm["ncoef"])])
    for ch in range(n_ch):
        for s in range(sub):
            fr["bark_hist"].append(br.get(1))
    for ch in range(n_ch):
        fr["gain"].append(br.get(GAIN_BITS))
        if ftype != 2:
            for s in range(sub):
                fr["sub_gain"].append(br.get(SUB_GAIN_BITS))
    for ch in range(n_ch):
        fr["lpc_hist"].append(br.get(lsp_bit0))
        fr["lpc1"].append(br.get(lsp_bit1))
        fr["lpc2"].append([br.get(lsp_bit2) for _ in range(lsp_split)])
    if ftype == 2:
        for i in range(prm.n_div[3]):
            second = i >= prm.bms_change[3]
            fr["ppc0"].append(br.get(prm.bms[3][(0, second)]))
            fr["ppc1"].append(br.get(prm.bms[3][(1, second)]))
        for ch in range(n_ch):
            fr["p_coef"].append(br.get(ppc_period_bit))
            fr["g_coef"].append(br.get(pgain_bit))
    return fr


def parse_vqf(path, max_frames=None):
    """Полный разбор .vqf → (mode_key, n_ch, prm, [frames], leftover_bits)."""
    key, n_ch, data_off, data = mode_from_comm(path)
    prm = TwinVQParams(key, n_ch)
    br = BitReader(data)
    frames = []
    while br.remaining() >= prm.total_fr_bits:
        if max_frames and len(frames) >= max_frames:
            break
        frames.append(parse_frame(prm, br))
    return key, n_ch, prm, frames, br.remaining()


def wtype_name(w):
    return {0: "L0", 1: "L1", 2: "S2", 3: "L3", 4: "M4",
            5: "L5", 6: "L6", 7: "M7", 8: "M8"}[w]


def write_vqf_packed(path, mode_key, n_ch, data, dsiz=None, meta_chunks=None):
    """Записать .vqf с уже упакованным (pack_frames) битстримом DATA.

    Заголовок — байт-в-байт канон официального энкодера NTT: COMM строго
    16 байт (12 полей + 4 нулевых), DSIZ не пишется (параметр оставлен
    для совместимости вызовов и игнорируется). Официальный плеер
    ожидает поля на фиксированных офсетах — байт 36 обязан быть 0x00.
    """
    m = MODES[mode_key]
    bitrate_total = m["kbps_ch"] * n_ch
    rate_flag = {44100: 44, 22050: 22, 11025: 11}.get(m["sr"], m["sr"] // 1000)
    chunks = bytearray()
    comm = struct.pack(">III", n_ch - 1, bitrate_total, rate_flag) + b"\x00" * 4
    chunks += b"COMM" + struct.pack(">I", len(comm)) + comm
    for tag, text in (meta_chunks or []):
        b = text.encode("utf-8")
        chunks += tag + struct.pack(">I", len(b)) + b
    blob = bytearray()
    blob += b"TWIN" + b"97012000" + struct.pack(">I", len(chunks))
    blob += chunks + b"DATA" + data
    with open(path, "wb") as f:
        f.write(bytes(blob))


def flip_bits(buf, positions):
    """Инвертировать биты в bytes-like по абсолютным позициям бит (MSB-first)."""
    b = bytearray(buf)
    for p in positions:
        b[p >> 3] ^= 0x80 >> (p & 7)
    return bytes(b)


def write_vqf(path, mode_key, n_ch, frames, dsiz=None, meta_chunks=None):
    m = MODES[mode_key]
    prm = TwinVQParams(mode_key, n_ch)
    bitrate_total = m["kbps_ch"] * n_ch
    rate_flag = {44100: 44, 22050: 22, 11025: 11}.get(m["sr"], m["sr"] // 1000)

    chunks = bytearray()
    comm = struct.pack(">III", n_ch - 1, bitrate_total, rate_flag) + b"\x00" * 4
    chunks += b"COMM" + struct.pack(">I", len(comm)) + comm
    for tag, text in (meta_chunks or []):
        b = text.encode("utf-8")
        chunks += tag + struct.pack(">I", len(b)) + b
    header_size = len(chunks)

    data = pack_frames(prm, frames)
    blob = bytearray()
    blob += b"TWIN"
    blob += b"97012000"
    blob += struct.pack(">I", header_size)
    blob += chunks
    blob += b"DATA"
    blob += data
    with open(path, "wb") as f:
        f.write(bytes(blob))
    return prm


def decode(path, ff="ffmpeg"):
    """Декодировать .vqf в float32 numpy [samples, channels]."""
    cmd = [ff, "-loglevel", "error", "-i", path, "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise RuntimeError(raw.stderr.decode("utf-8", "replace"))
    x = np.frombuffer(raw.stdout, dtype=np.float32)
    return x


def probe(path, ff="ffprobe"):
    cmd = [ff, "-v", "error", "-show_format", "-show_streams",
           "-of", "default=noprint_wrappers=1", path]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return r.stdout + r.stderr


def spectrum(x, sr, nfft=8192):
    """|FFT| односторонний + частоты."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w, nfft)
    mag = np.abs(X) / (np.sum(w) / 2.0)
    return np.fft.rfftfreq(nfft, 1.0 / sr), mag
