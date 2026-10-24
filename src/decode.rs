//! Декодерный тракт TwinVQ: послойное восстановление спектра + синтез PCM.
//!
//! Точный порт бит-валидированного python-оракула (`python/reference/
//! twinlayers.py`, corr = 1.0 против ffmpeg на 12 файлах NTT):
//!
//! ```text
//! спектр = (VQ_dequant · bark_env · gain + PPC) · lpc_env
//! PCM    = TDAC-синтез (компактный IMDCT + sine-окна, wtype-грамматика)
//! ```
//!
//! Числовая дисциплина (важно для замкнутого поиска энкодера):
//! * LSP/MA и eval_lpc_spectrum — **f32** (как в FFmpeg; полюса 0.5/(p+q)
//!   амплифицируют численные расхождения на порядки);
//! * MDCT/IMDCT/bark/PPC/gain — f64 (как в оракуле);
//! * перестановки interleave — `dsp::construct_perm`.
//!
//! Слои доступны по отдельности (`FrameLayers`) — энкодер использует их
//! как точную реплику декодера в analysis-by-synthesis.

use crate::codec::StreamInfo;
use crate::dsp::{construct_perm, imdct_compact, sine_window};
use crate::frame::FrameData;
use crate::modes::{FrameType, WTYPE_TO_WSIZE_IDX};
use crate::vendor::cb_data::{
    BROKEN_OP_TABS, TAB10, TAB11, TAB12, TAB7, TAB8, TAB9,
};

pub const AMP_MAX: f64 = 13000.0;
pub const SUB_AMP_MAX: f64 = 4500.0;
pub const MULAW_MU: f64 = 100.0;
pub const PGAIN_MU: f64 = 200.0;
pub const GAIN_BITS: usize = 8;
pub const SUB_GAIN_BITS: usize = 5;

/// μ-law декодер декодера (twinvq.h `twinvq_mulawinv`).
pub fn mulawinv(y: f64, clip: f64, mu: f64) -> f64 {
    let y = (y / clip).clamp(-1.0, 1.0);
    if y == 0.0 {
        return 0.0;
    }
    let s = if y > 0.0 { 1.0 } else { -1.0 };
    clip * s * (((1.0 + mu).ln() * y.abs()).exp() - 1.0) / mu
}

/// ROUNDED_DIV FFmpeg: (a + b/2) / b.
pub fn rounded_div(a: i64, b: i64) -> i64 {
    if b > 0 {
        (a + b / 2) / b
    } else {
        0
    }
}

fn av_log2(v: i64) -> i64 {
    if v <= 0 {
        0
    } else {
        63 - v.leading_zeros() as i64
    }
}

fn flat2d<const W: usize, const N: usize>(t: &[[u8; W]; N], idx: usize) -> i64 {
    let (r, c) = (idx / W, idx % W);
    if r < N {
        t[r][c] as i64
    } else {
        0
    }
}

// very_broken_op (twinvqdec.c): «сломанное» округление NTT — реплицируется
// как есть (замена на честное округление увеличивает расхождение с
// оригинальным энкодером Yamaha/NTT).
pub fn very_broken_op(a: i64, b: i64) -> i64 {
    let x = a * b + 200;
    if x % 400 != 0 || b % 5 != 0 {
        return x / 400;
    }
    let x = x / 400;
    let (size, tab) = BROKEN_OP_TABS[(b / 5) as usize];
    if size == 0 || tab == 0 {
        return x;
    }
    let sz = size as i64;
    let idx = (sz * av_log2(2 * (x - 1) / sz) + (x - 1) % sz) as usize;
    let val = match tab {
        7 => flat2d(&TAB7, idx),
        8 => flat2d(&TAB8, idx),
        9 => flat2d(&TAB9, idx),
        10 => flat2d(&TAB10, idx),
        11 => flat2d(&TAB11, idx),
        12 => flat2d(&TAB12, idx),
        _ => 0,
    };
    x - val
}

// ----------------------------------------------------------------------------
// Слои спектра (state-free, кроме историй)
// ----------------------------------------------------------------------------

/// dec_gain: gain (LONG, на канал) или gain·sub_gain (на подкадр).
pub fn dec_gains(info: &StreamInfo, ftype: FrameType, fr: &FrameData) -> Vec<f64> {
    let sub = info.mode.fmode[ftype.idx()].sub;
    let step = AMP_MAX / ((1 << GAIN_BITS) - 1) as f64;
    let sub_step = SUB_AMP_MAX / ((1 << SUB_GAIN_BITS) - 1) as f64;
    let n_ch = info.n_ch;
    if ftype == FrameType::Long {
        (0..n_ch)
            .map(|ch| {
                (1.0 / (1 << 13) as f64)
                    * mulawinv(
                        step * 0.5 + step * fr.gain_bits[ch] as f64,
                        AMP_MAX,
                        MULAW_MU,
                    )
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(n_ch * sub);
        for ch in 0..n_ch {
            let val = (1.0 / (1 << 23) as f64)
                * mulawinv(
                    step * 0.5 + step * fr.gain_bits[ch] as f64,
                    AMP_MAX,
                    MULAW_MU,
                );
            for j in 0..sub {
                out.push(
                    val * mulawinv(
                        sub_step * 0.5
                            + sub_step * fr.sub_gain_bits[ch * sub + j] as f64,
                        SUB_AMP_MAX,
                        MULAW_MU,
                    ),
                );
            }
        }
        out
    }
}

/// Деквантование main VQ / PPC shape (dequant): пары кодбуков → плоский
/// спектр (n_ch·size) / PPC-форма (n_ch·shape_len) с перестановкой.
fn dequant_generic(
    info: &StreamInfo,
    ftype: FrameType,
    cb_bits: &[u8],
    cb0: &[i16],
    cb1: &[i16],
    cb_len: usize,
    perm: &[u16],
    out_len: usize,
) -> Vec<f64> {
    let d = info.layout.div(ftype);
    let mut out = vec![0.0f64; out_len];
    let mut pos = 0usize;
    for i in 0..d.n_div {
        let length = d.length_at(i);
        let (b0, b1) = d.bits_at(i);
        let mut tmp0 = cb_bits[2 * i] as i32;
        let mut tmp1 = cb_bits[2 * i + 1] as i32;
        let mut sign0 = 1.0f64;
        let mut sign1 = 1.0f64;
        if b0 == 7 {
            if tmp0 & 0x40 != 0 {
                sign0 = -1.0;
            }
            tmp0 &= 0x3F;
        }
        if b1 == 7 {
            if tmp1 & 0x40 != 0 {
                sign1 = -1.0;
            }
            tmp1 &= 0x3F;
        }
        let t0 = &cb0[tmp0 as usize * cb_len..][..cb_len];
        let t1 = &cb1[tmp1 as usize * cb_len..][..cb_len];
        for j in 0..length {
            let dst = perm[pos + j] as usize;
            out[dst] = sign0 * t0[j] as f64 + sign1 * t1[j] as f64;
        }
        pos += length;
    }
    out
}

/// Main VQ → плоский спектр (канал-мажор, подкадры подряд).
pub fn dequant_main(info: &StreamInfo, ftype: FrameType, fr: &FrameData, perm: &[u16]) -> Vec<f64> {
    let fm = &info.mode.fmode[ftype.idx()];
    dequant_generic(
        info,
        ftype,
        &fr.main_coeffs,
        fm.cb0,
        fm.cb1,
        fm.cb_len_read,
        perm,
        info.n_ch * info.mode.size,
    )
}

/// PPC shape cb → форма (n_ch·shape_len).
pub fn dequant_ppc_shape(info: &StreamInfo, fr: &FrameData, perm: &[u16]) -> Vec<f64> {
    let m = info.mode;
    let n_div = info.layout.ppc.n_div;
    let cb_len_p = (n_div + m.ppc_shape_len * info.n_ch - 1) / n_div;
    let cb = m.ppc_shape_cb;
    let (cb0, cb1) = cb.split_at(cb_len_p * 64);
    dequant_generic(
        info,
        FrameType::Ppc,
        &fr.ppc_coeffs,
        cb0,
        cb1,
        cb_len_p,
        perm,
        info.n_ch * m.ppc_shape_len,
    )
}

/// Bark-огибающая одного подкадра (dec_bark_env): длиной block_size,
///.piecewise-constant по bark-полосам; обновляет историю.
#[allow(clippy::too_many_arguments)]
pub fn dec_bark_env(
    info: &StreamInfo,
    ftype: FrameType,
    ch: usize,
    sub_j: usize,
    fr: &FrameData,
    gain: f64,
    hist: &mut [f64],
    out: &mut [f64],
) {
    let fm = &info.mode.fmode[ftype.idx()];
    let val = match ftype {
        FrameType::Short => 0.4,
        FrameType::Medium => 0.35,
        _ => 0.28,
    };
    let ncoef = fm.bark_n_coef;
    let fw_cb_len = fm.bark_env_size / ncoef;
    let in_coef = &fr.bark1[(ch * fm.sub + sub_j) * ncoef..][..ncoef];
    let use_hist = fr.bark_use_hist[ch * fm.sub + sub_j] != 0;
    let bark_tab = &fm.bark_tab[..fm.bark_env_size];
    let mut idx = 0usize;
    let mut wpos = 0usize;
    for i in 0..fw_cb_len {
        for j in 0..ncoef {
            let tmp2 = fm.bark_cb[fw_cb_len * in_coef[j] as usize + i] as f64 * (1.0 / 4096.0);
            let mut st = if use_hist {
                (1.0 - val) * tmp2 + val * hist[idx] + 1.0
            } else {
                tmp2 + 1.0
            };
            hist[idx] = tmp2;
            if st < -1.0 {
                st = 1.0;
            }
            let v = st * gain;
            let width = bark_tab[idx] as usize;
            out[wpos..wpos + width].fill(v);
            wpos += width;
            idx += 1;
        }
    }
    debug_assert_eq!(wpos, info.mode.size / fm.sub);
}

/// Параметры PPC-гребёнки кадра/канала.
#[derive(Clone, Copy, Debug)]
pub struct PpcInfo {
    pub period: i64,
    pub width: i64,
    pub ppc_gain: f64,
    pub p_coef: u32,
    pub g_coef: u32,
}

/// PPC (LONG): гармоническая гребёнка добавляется в плоский спектр канала.
pub fn decode_ppc(
    info: &StreamInfo,
    ch: usize,
    fr: &FrameData,
    shape: &[f64],
    speech: &mut [f64],
) -> PpcInfo {
    let m = info.mode;
    let size = m.size;
    let isampf = (info.sr / 1000) as i64;
    let ibps = (info.kbps_total / info.n_ch) as i64;
    let min_period = rounded_div(40 * 2 * size as i64, isampf);
    let max_period = rounded_div(40 * 2 * size as i64 * 6, isampf);
    let period_range = max_period - min_period;
    let pgain_step = 25000.0 / ((1u64 << m.pgain_bit) - 1) as f64;
    let ppc_gain = (1.0 / 8192.0)
        * mulawinv(
            pgain_step * fr.g_coef[ch] as f64 + pgain_step / 2.0,
            25000.0,
            PGAIN_MU,
        );
    let period = min_period
        + rounded_div(
            fr.p_coef[ch] as i64 * period_range,
            ((1u64 << m.ppc_period_bit) - 1) as i64,
        );
    let width = if isampf == 22 && ibps == 32 {
        rounded_div((period + 800) * m.peak_per2wid as i64, 400 * size as i64)
    } else {
        period * m.peak_per2wid as i64 / (400 * size as i64)
    };

    // add_peak
    let len = m.ppc_shape_len;
    if width <= 0 {
        return PpcInfo { period, width, ppc_gain, p_coef: fr.p_coef[ch], g_coef: fr.g_coef[ch] };
    }
    let mut si = 0usize;
    for i in 0..(width / 2) as usize {
        if i < speech.len() && si < shape.len() {
            speech[i] += ppc_gain * shape[si];
        }
        si += 1;
    }
    let half = width / 2;
    let full = (width + 1) / 2;
    let mut i = 1i64;
    while i < rounded_div(len as i64, width) {
        let center = very_broken_op(period, i);
        for j in -half..full {
            let p = j + center;
            if p >= 0 && (p as usize) < size && si < shape.len() {
                speech[p as usize] += ppc_gain * shape[si];
            }
            si += 1;
        }
        i += 1;
    }
    let center = very_broken_op(period, i);
    for j in -half..full {
        if si >= len {
            break;
        }
        let p = j + center;
        if p >= 0 && (p as usize) < size {
            speech[p as usize] += ppc_gain * shape[si];
        }
        si += 1;
    }
    PpcInfo { period, width, ppc_gain, p_coef: fr.p_coef[ch], g_coef: fr.g_coef[ch] }
}

// ----------------------------------------------------------------------------
// LSP → LPC-огибающая (f32-дисциплина!)
// ----------------------------------------------------------------------------

fn rearrange_lsp(lsp: &mut [f32], min_dist: f32) {
    let min_dist2 = min_dist * 0.5;
    for i in 1..lsp.len() {
        if lsp[i] - lsp[i - 1] < min_dist {
            let avg = (lsp[i] + lsp[i - 1]) * 0.5;
            lsp[i - 1] = avg - min_dist2;
            lsp[i] = avg + min_dist2;
        }
    }
}

fn sort_nearly_sorted(v: &mut [f32]) {
    // вставка с локальными обменами (порт ff_sort_nearly_sorted_floats)
    if v.len() < 2 {
        return;
    }
    for i in 0..v.len() - 1 {
        let mut j = i as isize;
        while j >= 0 && v[j as usize] > v[(j + 1) as usize] {
            v.swap(j as usize, (j + 1) as usize);
            j -= 1;
        }
    }
}

/// decode_lsp: кодбуки + MA-предсказание; обновляет lsp_hist канала.
pub fn decode_lsp(
    info: &StreamInfo,
    ch: usize,
    fr: &FrameData,
    lsp_hist: &mut [f32],
) -> Vec<f32> {
    let m = info.mode;
    let n_lsp = m.n_lsp;
    let cb = m.lspcodebook;
    let cb2_base = (1 << m.lsp_bit1) * n_lsp;
    let cb3_base = cb2_base + (1 << m.lsp_bit2) * n_lsp;
    let lsp_split = m.lsp_split;
    let funny: [i32; 4] = [
        -2,
        if lsp_split == 4 { -2 } else { 1 },
        if lsp_split == 4 { -2 } else { 1 },
        0,
    ];

    let mut lsp = vec![0.0f32; n_lsp];
    let mut j = 0usize;
    for i in 0..lsp_split {
        let chunk_end = (((i + 1) * n_lsp) as i32 + funny[i]) as usize / lsp_split;
        let idx2 = fr.lpc_idx2[ch * lsp_split + i] as usize;
        while j < chunk_end {
            // сумма в f64 → f32 (дисциплина python-оракула)
            lsp[j] = (cb[fr.lpc_idx1[ch] as usize * n_lsp + j] as f64
                + cb[cb2_base + idx2 * n_lsp + j] as f64) as f32;
            j += 1;
        }
    }
    rearrange_lsp(&mut lsp, 0.0001);
    let ma = &cb[cb3_base + fr.lpc_hist_idx[ch] as usize * n_lsp..][..n_lsp];
    for i in 0..n_lsp {
        let tmp1 = 1.0f32 - ma[i];
        let tmp2 = lsp_hist[i] * ma[i];
        lsp_hist[i] = lsp[i];
        lsp[i] = lsp[i] * tmp1 + tmp2;
    }
    rearrange_lsp(&mut lsp, 0.0001);
    rearrange_lsp(&mut lsp, 0.000095);
    sort_nearly_sorted(&mut lsp);
    lsp
}

/// eval_lpc_spectrum — ОДНА точка огибающей (f32!). `cos_val` — f64 из
/// cos-таблицы, приводится к f32 как в оракуле.
fn eval_lpc_spectrum(lsp: &[f32], cos_val: f64) -> f32 {
    let mut p = 0.5f32;
    let mut q = 0.5f32;
    let two_cos_w = 2.0f32 * (cos_val as f32);
    let order = lsp.len();
    let mut j = 0usize;
    while j + 1 < order {
        q *= lsp[j] - two_cos_w;
        p *= lsp[j + 1] - two_cos_w;
        q *= lsp[j + 2] - two_cos_w;
        p *= lsp[j + 3] - two_cos_w;
        j += 4;
    }
    p = p * p * (2.0f32 - two_cos_w);
    q = q * q * (2.0f32 + two_cos_w);
    0.5f32 / (p + q)
}

/// interpolate(out[i0..], v1, v2, n): линейная интерполяция n значений.
fn interp(out: &mut [f32], i0: usize, v1: f32, v2: f32, n: usize) {
    let step = (v1 - v2) / (n + 1) as f32;
    let mut vv = v2;
    for t in 0..n {
        vv += step;
        out[i0 + t] = vv;
    }
}

/// eval_lpcenv_or_interp: «siiiibiiii»-схема FFmpeg (quirk-ветвления
/// воспроизводятся точно; сравнения в f32).
fn eval_or_interp(
    cos_tab: &[f64],
    out: &mut [f32],
    o0: usize,
    lsp: &[f32],
    size: usize,
    step: usize,
    part: bool,
) {
    let get_cos = |idx: usize| -> f64 {
        if part {
            -cos_tab[size - idx - 1]
        } else {
            cos_tab[idx]
        }
    };
    let mut i = 0usize;
    while i < size {
        out[o0 + i] = eval_lpc_spectrum(lsp, get_cos(i));
        i += step;
    }
    let lim = size.saturating_sub(2 * step);
    let mut i = step;
    while i <= lim {
        if out[o0 + i + step] + out[o0 + i - step] > 1.95f32 * out[o0 + i]
            || out[o0 + i + step] >= out[o0 + i - step]
        {
            interp(out, o0 + i - step + 1, out[o0 + i], out[o0 + i - step], step - 1);
        } else {
            out[o0 + i - step / 2] = eval_lpc_spectrum(lsp, get_cos(i - step / 2));
            interp(
                out,
                o0 + i - step + 1,
                out[o0 + i - step / 2],
                out[o0 + i - step],
                step / 2 - 1,
            );
            interp(
                out,
                o0 + i - step / 2 + 1,
                out[o0 + i],
                out[o0 + i - step / 2],
                step / 2 - 1,
            );
        }
        i += step;
    }
    interp(
        out,
        o0 + size - 2 * step + 1,
        out[o0 + size - step],
        out[o0 + size - 2 * step],
        step - 1,
    );
}

/// cos-таблица ftype (init_mdct_win): cos((2j+1)·2π/m), зеркальная.
pub fn cos_tab(info: &StreamInfo, ftype: FrameType) -> Vec<f64> {
    let m = info.mode;
    let mm = 4 * m.size / m.fmode[ftype.idx()].sub;
    let freq = 2.0 * std::f64::consts::PI / mm as f64;
    let mut t = vec![0.0f64; mm / 4];
    for j in 0..=mm / 8 {
        t[j] = ((2 * j + 1) as f64 * freq).cos();
    }
    for j in 1..mm / 8 {
        t[mm / 4 - j] = t[j];
    }
    t
}

/// dec_lpc_env: LSP → LPC-огибающая длиной block_size (f32).
pub fn dec_lpc_env(info: &StreamInfo, ftype: FrameType, lsp: &[f32]) -> Vec<f32> {
    let m = info.mode;
    let size = m.size / m.fmode[ftype.idx()].sub;
    let lsp2: Vec<f32> = lsp.iter().map(|&w| (2.0 * (w as f64).cos()) as f32).collect();
    let mut lpc = vec![0.0f32; size];
    match ftype {
        FrameType::Long => {
            let ct = cos_tab(info, ftype);
            eval_or_interp(&ct, &mut lpc, 0, &lsp2, size / 2, 8, false);
            eval_or_interp(&ct, &mut lpc, size / 2, &lsp2, size / 2, 16, true);
            let (v1, v2) = (lpc[size / 2], lpc[size / 2 - 8]);
            interp(&mut lpc, size / 2 - 8 + 1, v1, v2, 8);
            let v = lpc[size - 2 * 8];
            lpc[size - 2 * 8 + 1..].fill(v);
        }
        FrameType::Medium => {
            let ct = cos_tab(info, ftype);
            eval_or_interp(&ct, &mut lpc, 0, &lsp2, size / 2, 2, false);
            eval_or_interp(&ct, &mut lpc, size / 2, &lsp2, size / 2, 4, true);
            let (v1, v2) = (lpc[size / 2], lpc[size / 2 - 2]);
            interp(&mut lpc, size / 2 - 2 + 1, v1, v2, 2);
            let v = lpc[size - 2 * 2];
            lpc[size - 2 * 2 + 1..].fill(v);
        }
        _ => {
            // SHORT: прямое вычисление обеих половин
            let ct = cos_tab(info, ftype);
            for i in 0..size / 2 {
                lpc[i] = eval_lpc_spectrum(&lsp2, ct[i]);
                lpc[size - i - 1] = eval_lpc_spectrum(&lsp2, -ct[i]);
            }
        }
    }
    lpc
}

// ----------------------------------------------------------------------------
// Послойный кадр
// ----------------------------------------------------------------------------

/// Слои декодирования одного кадра (на канал).
pub struct FrameLayers {
    /// Спектр после каждого слоя (длина size).
    pub vq: Vec<Vec<f64>>,
    pub bark: Vec<Vec<f64>>,
    pub after_bark: Vec<Vec<f64>>,
    pub ppc: Vec<Vec<f64>>,
    pub ppc_info: Vec<Option<PpcInfo>>,
    pub lsp: Vec<Vec<f32>>,
    pub lpc_env: Vec<Vec<f32>>,
    /// Итоговый спектр (вход IMDCT).
    pub spec: Vec<Vec<f64>>,
}

// ----------------------------------------------------------------------------
// Полный декодер (спектр + синтез) с состояниями
// ----------------------------------------------------------------------------

/// Декодер с историями (LSP/bark/синтез) — по кадрам.
pub struct Decoder {
    pub info: StreamInfo,
    perm: [Vec<u16>; 4],
    lsp_hist: Vec<Vec<f32>>,
    bark_hist: Vec<Vec<f64>>, // [ftype·2 + ch][40]
    curr: Vec<Vec<f64>>,      // [ch][2·size]
    prev: Vec<Vec<f64>>,      // [ch][2·size]
    last_pos: usize,
}

impl Decoder {
    pub fn new(info: StreamInfo) -> Self {
        let perm: [Vec<u16>; 4] = FrameType::ALL
            .iter()
            .map(|&ft| construct_perm(info.mode, &info.layout, ft, info.n_ch))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Decoder {
            lsp_hist: vec![vec![0.0f32; info.mode.n_lsp]; info.n_ch],
            bark_hist: vec![vec![0.1f64; 40]; 3 * 2],
            curr: vec![vec![0.0; 2 * info.mode.size]; info.n_ch],
            prev: vec![vec![0.0; 2 * info.mode.size]; info.n_ch],
            last_pos: 0,
            perm,
            info,
        }
    }

    /// Послойное декодирование кадра (обновляет LSP/bark-истории).
    pub fn decode_frame_layers(&mut self, fr: &FrameData) -> FrameLayers {
        let info = &self.info;
        let m = info.mode;
        let ftype = fr.ftype();
        let fm = &m.fmode[ftype.idx()];
        let sub = fm.sub;
        let block_size = m.size / sub;
        let n_ch = info.n_ch;
        let size = m.size;

        let spec_flat = dequant_main(info, ftype, fr, &self.perm[ftype.idx()]);
        let gains = dec_gains(info, ftype, fr);
        let ppc_shape = if ftype == FrameType::Long {
            dequant_ppc_shape(info, fr, &self.perm[FrameType::Ppc.idx()])
        } else {
            Vec::new()
        };

        let mut layers = FrameLayers {
            vq: Vec::with_capacity(n_ch),
            bark: Vec::with_capacity(n_ch),
            after_bark: Vec::with_capacity(n_ch),
            ppc: Vec::with_capacity(n_ch),
            ppc_info: Vec::with_capacity(n_ch),
            lsp: Vec::with_capacity(n_ch),
            lpc_env: Vec::with_capacity(n_ch),
            spec: Vec::with_capacity(n_ch),
        };

        for ch in 0..n_ch {
            let mut chunk = spec_flat[ch * size..(ch + 1) * size].to_vec();
            layers.vq.push(chunk.clone());
            let mut bark = vec![0.0; size];
            let mut env = vec![0.0f64; block_size];
            for j in 0..sub {
                let g = if ftype == FrameType::Long {
                    gains[ch]
                } else {
                    gains[sub * ch + j]
                };
                dec_bark_env(
                    info,
                    ftype,
                    ch,
                    j,
                    fr,
                    g,
                    &mut self.bark_hist[ftype.idx() * 2 + ch],
                    &mut env,
                );
                let bs = block_size * j;
                for k in 0..block_size {
                    chunk[bs + k] *= env[k];
                }
                bark[bs..bs + block_size].copy_from_slice(&env);
            }
            layers.bark.push(bark);
            layers.after_bark.push(chunk.clone());

            let mut ppc = vec![0.0; size];
            let mut ppc_info = None;
            if ftype == FrameType::Long {
                let sh = &ppc_shape[ch * m.ppc_shape_len..(ch + 1) * m.ppc_shape_len];
                ppc_info = Some(decode_ppc(info, ch, fr, sh, &mut ppc));
                for (a, b) in chunk.iter_mut().zip(ppc.iter()) {
                    *a += b;
                }
            }
            layers.ppc.push(ppc);
            layers.ppc_info.push(ppc_info);

            let lsp = decode_lsp(info, ch, fr, &mut self.lsp_hist[ch]);
            let lpc_env = dec_lpc_env(info, ftype, &lsp);
            for j in 0..sub {
                let bs = block_size * j;
                for k in 0..block_size {
                    chunk[bs + k] *= lpc_env[k] as f64;
                }
            }
            layers.lsp.push(lsp);
            layers.lpc_env.push(lpc_env);
            layers.spec.push(chunk);
        }
        layers
    }

    /// TDAC-синтез спектров (imdct_and_window + imdct_output).
    /// pub(crate): используется PR-тестом энкодера (analysis_targets).
    pub(crate) fn synth_frame(&mut self, ftype: FrameType, wtype: u8, spectra: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let size = self.info.mode.size;
        let old_pos = self.last_pos;
        for ch in 0..self.info.n_ch {
            let new_pos = imdct_and_window(
                &self.info,
                ftype,
                wtype,
                &spectra[ch],
                &self.prev[ch],
                &mut self.curr[ch],
                old_pos,
            );
            self.last_pos = new_pos;
        }
        let size2 = self.last_pos;
        let size1 = size - size2;
        let mut out = vec![vec![0.0f64; size]; self.info.n_ch];
        for ch in 0..self.info.n_ch {
            out[ch][..size1].copy_from_slice(&self.prev[ch][old_pos..old_pos + size1]);
            out[ch][size1..].copy_from_slice(&self.curr[ch][..size2]);
        }
        if self.info.n_ch == 2 {
            for i in 0..size {
                let l = out[0][i];
                let r = out[1][i];
                out[0][i] = l + r;
                out[1][i] = l - r;
            }
        }
        std::mem::swap(&mut self.curr, &mut self.prev);
        out
    }

    /// Полный декод кадра → PCM [ch][size].
    pub fn decode_frame(&mut self, fr: &FrameData) -> Vec<Vec<f64>> {
        let layers = self.decode_frame_layers(fr);
        self.synth_frame(fr.ftype(), fr.wtype, &layers.spec)
    }
}

/// imdct_and_window одного канала (порт twinvq.c + quirk python-оракула:
/// база prev считается по СТАРОМУ last_block_pos, общему для каналов).
#[allow(clippy::too_many_arguments)]
fn imdct_and_window(
    info: &StreamInfo,
    ftype: FrameType,
    wtype: u8,
    x_ch: &[f64],
    prev_ch: &[f64],
    curr_ch: &mut [f64],
    old_pos: usize,
) -> usize {
    let m = info.mode;
    let sub = m.fmode[ftype.idx()].sub;
    let b = m.size / sub;
    let size = m.size;
    let norm = info.norm();
    let ts = [
        size / m.fmode[FrameType::Long.idx()].sub,
        size / m.fmode[FrameType::Medium.idx()].sub,
        size / (m.fmode[FrameType::Short.idx()].sub * 2),
    ];
    let first_wsize = ts[WTYPE_TO_WSIZE_IDX[wtype as usize]];
    // хвост предыдущего подкадра (для первого — из prev-буфера кадра);
    // копия снимает алиасинг buf/prev при последовательных подкадрах
    let mut prev_tail: Vec<f64> = prev_ch[old_pos + (size - b) / 2..][..b / 2].to_vec();
    let mut out2 = 0usize;

    for j in 0..sub {
        let mut sub_wtype = if ftype == FrameType::Medium { 8u8 } else { wtype };
        if j == 0 && wtype == 4 {
            sub_wtype = 4;
        } else if j == sub - 1 && wtype == 7 {
            sub_wtype = 7;
        }
        let wsize = ts[WTYPE_TO_WSIZE_IDX[sub_wtype as usize]];
        let y = imdct_compact(&x_ch[b * j..b * (j + 1)], b, norm);

        // vector_fmul_window: TDAC-бабочка (src0 = prev_tail, src1 = y)
        let w = sine_window(wsize);
        let l = wsize / 2;
        let base = (b - wsize) / 2;
        for t in 0..l {
            let s0 = prev_tail[base + t];
            let s1 = y[l - 1 - t];
            let w1 = w[t];
            let w2 = w[2 * l - 1 - t];
            curr_ch[out2 + t] = s0 * w2 - s1 * w1;
            curr_ch[out2 + 2 * l - 1 - t] = s0 * w1 + s1 * w2;
        }
        out2 += wsize;
        let n = b - wsize / 2;
        curr_ch[out2..out2 + n].copy_from_slice(&y[wsize / 2..b]);
        out2 += if ftype == FrameType::Medium { (b - wsize) / 2 } else { b - wsize };
        prev_tail.copy_from_slice(&y[b / 2..]);
    }
    (size + first_wsize) / 2
}

/// Декодировать поток кадров в планарный PCM (как ffmpeg: первые 2 кадра
/// прогоняются через синтез, но в выход не попадают).
pub fn decode_stream(info: &StreamInfo, frames: &[FrameData]) -> Vec<Vec<f64>> {
    let mut dec = Decoder::new(*info);
    let n_ch = info.n_ch;
    let mut planar = vec![Vec::new(); n_ch];
    for (i, fr) in frames.iter().enumerate() {
        let pcm = dec.decode_frame(fr);
        // ffmpeg отбрасывает 2 первых пакета (синтез всё равно выполняется)
        if i >= 2 {
            for (p, f) in planar.iter_mut().zip(pcm.into_iter()) {
                p.extend_from_slice(&f);
            }
        }
    }
    planar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::parse_vqf;

    fn fixtures() -> Option<(std::path::PathBuf, Vec<(String, usize, usize)>)> {
        let dir = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden"));
        let manifest = std::fs::read_to_string(dir.join("manifest.txt")).ok()?;
        let mut out = Vec::new();
        for line in manifest.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() < 6 {
                continue;
            }
            out.push((t[0].to_string(), t[2].parse().unwrap(), t[3].parse().unwrap()));
        }
        if out.is_empty() {
            None
        } else {
            Some((dir, out))
        }
    }

    /// ГЛАВНЫЙ golden-тест: полный декод (спектр + синтез) 6 фикстур NTT
    /// против PCM python-оракула (twinlayers, corr=1.0 vs ffmpeg).
    #[test]
    fn golden_pcm_matches_python_oracle() {
        let Some((dir, fx)) = fixtures() else {
            eprintln!("golden-фикстуры не найдены — пропуск");
            return;
        };
        assert!(fx.len() >= 5);
        for (stem, n_ch, n_frames) in &fx {
            let blob = std::fs::read(dir.join(format!("{stem}.vqf"))).unwrap();
            let golden = std::fs::read(dir.join(format!("{stem}.pcm.f32"))).unwrap();
            let parsed = parse_vqf(&blob).unwrap();
            let info = &parsed.info;
            assert_eq!(info.n_ch, *n_ch, "{stem}");
            let pcm = decode_stream(info, &parsed.frames);
            let size = info.mode.size;
            let expect_frames = n_frames.saturating_sub(2);
            let expect_samples = expect_frames * size;
            assert_eq!(pcm[0].len(), expect_samples, "{stem}: длина PCM");
            assert_eq!(golden.len(), expect_samples * n_ch * 4, "{stem}: golden размер");

            // метрики: corr и rel_err (пороги шире python-vs-ffmpeg из-за
            // libm-различий в f32-точках LSP)
            for ch in 0..*n_ch {
                let g: Vec<f32> = golden[ch * expect_samples * 4..(ch + 1) * expect_samples * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let a = &pcm[ch];
                assert_eq!(a.len(), g.len(), "{stem}[{ch}]: длины");
                let n = g.len();
                let mut sa = 0.0f64;
                let mut sb = 0.0f64;
                let mut saa = 0.0f64;
                let mut sbb = 0.0f64;
                let mut sab = 0.0f64;
                for i in 0..n {
                    let x = a[i];
                    let y = g[i] as f64;
                    sa += x;
                    sb += y;
                    saa += x * x;
                    sbb += y * y;
                    sab += x * y;
                }
                let nn = n as f64;
                let cov = sab - sa * sb / nn;
                let va = saa - sa * sa / nn;
                let vb = sbb - sb * sb / nn;
                let corr = cov / (va.sqrt() * vb.sqrt());
                let err: f64 = (0..n).map(|i| (a[i] - g[i] as f64).powi(2)).sum::<f64>().sqrt();
                let rel = err / (sbb.sqrt() + 1e-30);
                eprintln!("{stem}[{ch}]: corr={corr:.8} rel_err={rel:.2e}");
                assert!(corr > 0.99999, "{stem}[{ch}]: corr={corr}");
                assert!(rel < 1e-3, "{stem}[{ch}]: rel_err={rel}");
            }
        }
    }

    /// very_broken_op: граничные случаи из twinvqdec.c.
    #[test]
    fn broken_op_basics() {
        // обычный случай: остаток ≠ 0 → просто деление с +200
        assert_eq!(very_broken_op(3, 7), (3 * 7 + 200) / 400);
        assert_eq!(very_broken_op(10, 10), (100 + 200) / 400); // 300/400 = 0
        // вход в таблицу: x%400==0 и b%5==0
        // a·b+200 = 400 → x=1: rtab[size·av_log2(0) + 0] = rtab[0]
        let v = very_broken_op(10, 20); // 200+200=400 → x=1, b/5=4 → (5,8)
        assert_eq!(v, 1 - (TAB8[0][0] as i64));
    }

    /// mulawinv: знак нуля (FFSIGN(0)=0).
    #[test]
    fn mulawinv_zero() {
        assert_eq!(mulawinv(0.0, 13000.0, 100.0), 0.0);
        let v = mulawinv(6500.0, 13000.0, 100.0);
        let expect = 13000.0 * ((101.0f64.ln() * 0.5).exp() - 1.0) / 100.0;
        assert!((v - expect).abs() < 1e-9, "{v} vs {expect}");
    }

    /// PPC-диапазон: p_coef=0/макс (44_48: f0 ≈ 100/600 Гц).
    #[test]
    fn ppc_period_grid() {
        let info = StreamInfo::new(44, 48, 1).unwrap();
        let mut fr = FrameData::new(&info, 0);
        fr.p_coef[0] = 0;
        let mut speech = vec![0.0; info.mode.size];
        let shape = vec![1.0; info.mode.ppc_shape_len];
        let p0 = decode_ppc(&info, 0, &fr, &shape, &mut speech);
        fr.p_coef[0] = (1 << info.mode.ppc_period_bit) - 1;
        let p1 = decode_ppc(&info, 0, &fr, &shape, &mut speech);
        let f0 = |p: i64| p as f64 * info.sr as f64 / (800.0 * info.mode.size as f64);
        assert!((99.9..=100.4).contains(&f0(p0.period)), "f0_min={}", f0(p0.period));
        assert!((599.9..=601.5).contains(&f0(p1.period)), "f0_max={}", f0(p1.period));
    }
}
