//! DSP-примитивы TwinVQ: перестановки interleave, окна, MDCT/IMDCT.
//!
//! Порты (бит-точный python-оракул `python/reference/twinlayers.py`,
//! corr = 1.0 против ffmpeg):
//! * `construct_perm_table` — permutate_in_line → transpose → linear;
//! * компактный IMDCT (tx_template.c, без AV_TX_FULL_IMDCT) + sine-окна;
//! * прямой MDCT (анализ, калибровка E1: синусное окно 2B, off=(k−2)·size,
//!   c = 2^16/√(norm·B) — docs/ARCHITECTURE.md §4a).
//!
//! Окна и твиддлы кэшируются по размеру блока (OnceLock). MDCT/IMDCT —
//! O(B·log B) через комплексный FFT размера 4B (крейт rustfft — SIMD
//! внутри); все три преобразования (fwd / compact-IMDCT / компактный
//! анализ) сводятся к одному ядру с таблицей W4c[t] = exp(−i·t·π/(4B))
//! (модуль `fft_mdct`). Наивные O(B²) реализации сохранены как эталон
//! для тестов (`naive`). Точность против оракула: rel ~1e-12
//! (тесты `fft_mdct_matches_naive`, `*_matches_python_golden`).

use crate::layout::FrameLayout;
use crate::modes::{FrameType, ModeTab};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ----------------------------------------------------------------------------
// Перестановки interleave (construct_perm_table)
// ----------------------------------------------------------------------------

fn permutate_in_line(
    tab: &mut [u16],
    n_div: usize,
    num_blocks: usize,
    block_size: usize,
    line_len: [usize; 2],
    ftype: FrameType,
) {
    for i in 0..line_len[0] {
        let shift = if num_blocks == 1
            || (ftype == FrameType::Long && n_div % num_blocks != 0)
            || (ftype != FrameType::Long && n_div & 1 != 0)
            || i == line_len[1]
        {
            0
        } else if ftype == FrameType::Long {
            i
        } else {
            i * i
        };
        let mut j = 0usize;
        while j < n_div && j + n_div * i < block_size * num_blocks {
            tab[i * n_div + j] = (i * n_div + (j + shift) % n_div) as u16;
            j += 1;
        }
    }
}

fn transpose_perm(
    out: &mut Vec<u16>,
    src: &[u16],
    n_div: usize,
    line_len: [usize; 2],
    length_div: usize,
) {
    out.clear();
    out.reserve(n_div * line_len[0]);
    for i in 0..n_div {
        let ll = line_len[usize::from(i >= length_div)];
        for j in 0..ll {
            out.push(src[j * n_div + i]);
        }
    }
}

fn linear_perm(out: &mut [u16], src: &[u16], n_blocks: usize, size: usize) {
    let block_size = size / n_blocks;
    for i in 0..size {
        let v = src[i] as usize;
        out[i] = (block_size * (v % n_blocks) + v / n_blocks) as u16;
    }
}

/// Перестановка interleave для ftype (tctx->permut[ftype]).
///
/// `S[t] = flat[permut[t]]`: VQ-последовательность → плоский спектр
/// (канал-мажор, подкадры подряд).
pub fn construct_perm(
    m: &ModeTab,
    layout: &FrameLayout,
    ftype: FrameType,
    n_ch: usize,
) -> Vec<u16> {
    let (size, block_size) = if ftype == FrameType::Ppc {
        (n_ch, m.ppc_shape_len)
    } else {
        let sub = m.fmode[ftype.idx()].sub;
        (n_ch * sub, m.size / sub)
    };
    let d = &layout.div(ftype);
    let mut tmp = vec![0u16; size * block_size];
    permutate_in_line(&mut tmp, d.n_div, size, block_size, d.length, ftype);

    let mut t = Vec::new();
    transpose_perm(&mut t, &tmp, d.n_div, d.length, d.length_change);

    let mut out = vec![0u16; size * block_size];
    linear_perm(&mut out, &t, size, size * block_size);
    out
}

// ----------------------------------------------------------------------------
// Окна и MDCT
// ----------------------------------------------------------------------------

fn table_cache<T>() -> Mutex<HashMap<usize, Arc<T>>> {
    Mutex::new(HashMap::new())
}

/// Sine-окно синтеза длины `wsize` (ff_sine_windows): sin((m+½)π/(2w)).
pub fn sine_window(wsize: usize) -> Arc<Vec<f64>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<f64>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(table_cache);
    if let Some(t) = cache.lock().unwrap().get(&wsize) {
        return t.clone();
    }
    let t: Vec<f64> = (0..wsize)
        .map(|m| ((m as f64 + 0.5) * std::f64::consts::PI / (2.0 * wsize as f64)).sin())
        .collect();
    let t = Arc::new(t);
    cache.lock().unwrap().insert(wsize, t.clone());
    t
}

/// Синусное окно анализа длины 2B (Princen–Bradley, 50% перекрытие).
pub fn sine_window_2b(b: usize) -> Arc<Vec<f64>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<f64>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(table_cache);
    if let Some(t) = cache.lock().unwrap().get(&b) {
        return t.clone();
    }
    let t: Vec<f64> = (0..2 * b)
        .map(|j| ((j as f64 + 0.5) * std::f64::consts::PI / (2.0 * b as f64)).sin())
        .collect();
    let t = Arc::new(t);
    cache.lock().unwrap().insert(b, t.clone());
    t
}

// ----------------------------------------------------------------------------
// MDCT/IMDCT через FFT: единое ядро cos_analysis / cos_synthesis
// ----------------------------------------------------------------------------

/// FFT-инфраструктура: планы rustfft (кэш по размеру), твиддл-таблица
/// W4c и thread-local scratch (без аллокаций в установившемся режиме).
///
/// Математика ядра (B — размер спектра, N = 4B):
/// ```text
/// анализ:  out[k] = Re Σ_j v[j]·e^{i(2k+1)t_jπ/(4B)}
///          Z[t_j mod N] = v[j]·(−1)^{⌊t_j/N⌋}·W4c[t_j mod N];  out[k] = Re FFT(Z)[k]
/// синтез:  out[j] = s_j·Σ_k x[k]·cos((2k+1)t_jπ/(4B))
///          Z[k] = x[k] (k < B);  F = FFT(Z);
///          out[j] = s_j·(W4c.re·F.re − W4c.im·F.im)[t_j mod N]·(−1)^{⌊t_j/N⌋}
/// ```
/// (вещественность v/x позволяет обходить сопряжения без лишних проходов;
/// перенос t_j через N меняет знак — множитель (−1)^q.)
mod fft_mdct {
    use rustfft::num_complex::Complex;
    use rustfft::{Fft, FftPlanner};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    /// FFT-план (forward) размера n — кэш по размеру.
    fn plan(n: usize) -> Arc<dyn Fft<f64>> {
        static PLANS: OnceLock<Mutex<HashMap<usize, Arc<dyn Fft<f64>>>>> = OnceLock::new();
        let plans = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(p) = plans.lock().unwrap().get(&n) {
            return p.clone();
        }
        let mut planner = FftPlanner::<f64>::new();
        let p = planner.plan_fft_forward(n);
        plans.lock().unwrap().insert(n, p.clone());
        p
    }

    /// Твиддл-таблица W4c[t] = exp(−i·t·π/(4B)), t = 0..4B (кэш по B).
    fn w4c(b: usize) -> Arc<Vec<Complex<f64>>> {
        static TABS: OnceLock<Mutex<HashMap<usize, Arc<Vec<Complex<f64>>>>>> = OnceLock::new();
        let tabs = TABS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(t) = tabs.lock().unwrap().get(&b) {
            return t.clone();
        }
        let n = 4 * b;
        let t: Vec<Complex<f64>> = (0..n)
            .map(|k| {
                let th = -(k as f64) * std::f64::consts::PI / (n as f64);
                Complex::new(th.cos(), th.sin())
            })
            .collect();
        let t = Arc::new(t);
        tabs.lock().unwrap().insert(b, t.clone());
        t
    }

    /// FFT in-place; длина `z` обязана равняться размеру плана.
    fn fft_inplace(z: &mut [Complex<f64>]) {
        let n = z.len();
        let p = plan(n);
        thread_local! {
            static SCRATCH: RefCell<HashMap<usize, Vec<Complex<f64>>>> =
                RefCell::new(HashMap::new());
        }
        SCRATCH.with(|cell| {
            let mut m = cell.borrow_mut();
            let v = m.entry(n).or_default();
            let need = p.get_inplace_scratch_len();
            if v.len() < need {
                v.resize(need, Complex::new(0.0, 0.0));
            }
            p.process_with_scratch(z, &mut v[..need]);
        });
    }

    /// Ядро «анализ»: `out[k] = Re Σ_j v[j]·exp(i·(2k+1)·t_j·π/(4B))`, k < n_out.
    pub(super) fn cos_analysis(v: &[f64], t: &[i64], b: usize, n_out: usize) -> Vec<f64> {
        debug_assert_eq!(v.len(), t.len());
        let n = 4 * b;
        let w = w4c(b);
        let mut z = vec![Complex::new(0.0, 0.0); n];
        for (j, &tj) in t.iter().enumerate() {
            let q = tj.div_euclid(n as i64);
            let tm = tj.rem_euclid(n as i64) as usize;
            let s = if q & 1 != 0 { -v[j] } else { v[j] };
            z[tm] = Complex::new(s * w[tm].re, s * w[tm].im);
        }
        fft_inplace(&mut z);
        z[..n_out].iter().map(|c| c.re).collect()
    }

    /// Ядро «синтез» (транспонированное к анализу):
    /// `out[j] = s_j·Σ_k x[k]·cos((2k+1)·t_j·π/(4B))`.
    pub(super) fn cos_synthesis(x: &[f64], t: &[i64], s: &[f64], b: usize) -> Vec<f64> {
        debug_assert_eq!(t.len(), s.len());
        let n = 4 * b;
        let w = w4c(b);
        let mut z = vec![Complex::new(0.0, 0.0); n];
        for k in 0..b {
            z[k].re = x[k];
        }
        fft_inplace(&mut z);
        t.iter()
            .zip(s.iter())
            .map(|(&tj, &sj)| {
                let q = tj.div_euclid(n as i64);
                let tm = tj.rem_euclid(n as i64) as usize;
                let sgn = if q & 1 != 0 { -sj } else { sj };
                sgn * (w[tm].re * z[tm].re - w[tm].im * z[tm].im)
            })
            .collect()
    }
}

/// Геометрия компактных форм (общая для анализа и синтеза — анализ
/// строился как точный адъюнкт синтеза):
/// m < b/2: t = 2b−2m−1, σ = +1; m ≥ b/2: t = 2m+2b+1, σ = −1.
fn compact_t_sigma(b: usize) -> (Vec<i64>, Vec<f64>) {
    let half = b / 2;
    let mut t = Vec::with_capacity(b);
    let mut s = Vec::with_capacity(b);
    for m in 0..half {
        t.push((2 * b - 2 * m - 1) as i64);
        s.push(1.0);
    }
    for m in half..b {
        t.push((2 * m + 2 * b + 1) as i64);
        s.push(-1.0);
    }
    (t, s)
}

/// Прямой MDCT блока `x` (2B сэмплов, окно применяется внутри):
/// `X[i] = Σ_{j<2B} x[j]·cos((2j+1+B)(2i+1)π/(4B))`, i < B.
///
/// FFT-путь: t_j = 2j+1+B < 5B — перенос через 4B (знак (−1)^q) учтён
/// ядром; один комплексный FFT размера 4B.
pub fn mdct_fwd(x: &[f64], b: usize) -> Vec<f64> {
    assert_eq!(x.len(), 2 * b);
    let win = sine_window_2b(b);
    let v: Vec<f64> = x.iter().zip(win.iter()).map(|(a, w)| a * w).collect();
    let t: Vec<i64> = (0..2 * b).map(|j| (2 * j + 1 + b) as i64).collect();
    fft_mdct::cos_analysis(&v, &t, b, b)
}

/// Компактный IMDCT (tx_template.c, len-выход; масштаб декодера):
/// ```text
/// y[i]     =  s·Σ X[k]·cos((2k+1)(2B−2i−1)π/(4B))   i < B/2
/// y[i+B/2] = −s·Σ X[k]·cos((2k+1)(3B+2i+1)π/(4B))
/// s = −sqrt(norm/B)/2^15
/// ```
pub fn imdct_compact(x: &[f64], b: usize, norm: f64) -> Vec<f64> {
    assert_eq!(x.len(), b);
    let s = -(norm / b as f64).sqrt() / 32768.0;
    let (t, sg) = compact_t_sigma(b);
    // множители s·σ_m — в ядро синтеза
    let sf: Vec<f64> = sg.iter().map(|&v| s * v).collect();
    fft_mdct::cos_synthesis(x, &t, &sf, b)
}

/// Анализный MDCT — точное сопряжение (транспонирование) компактного IMDCT:
/// ```text
/// X[k] = Σ_{i<B/2} y[i]·cos((2k+1)(2B−2i−1)π/(4B))
///      − Σ_{i<B/2} y[i+B/2]·cos((2k+1)(3B+2i+1)π/(4B))
/// ```
/// Вместе с бабочкой/плоской частью/P-частью (encoder::analysis_targets)
/// образует PR-пару синтезу декодера (PR-SNR ~247 дБ, масштаб 1.0).
pub fn mdct_analysis_compact(y: &[f64], b: usize) -> Vec<f64> {
    assert_eq!(y.len(), b);
    let (t, sg) = compact_t_sigma(b);
    // σ_m сворачивается во вход (анализ = T(σ⊙y), синтез = s·σ⊙Tᵀx)
    let v: Vec<f64> = y.iter().zip(sg.iter()).map(|(&a, &s)| s * a).collect();
    fft_mdct::cos_analysis(&v, &t, b, b)
}

/// Наивные O(B²) реализации — эталон для тестов эквивалентности FFT
/// (формулы — исходные, сверены с python-оракулом).
#[cfg(test)]
pub(crate) mod naive {
    /// Таблица прямого MDCT (B×2B, строка = коэффициент i).
    pub fn mdct_table(b: usize) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        let mut t = vec![0.0f64; b * 2 * b];
        for i in 0..b {
            let row = &mut t[i * 2 * b..(i + 1) * 2 * b];
            for (j, r) in row.iter_mut().enumerate() {
                *r = ((2 * j + 1 + b) as f64 * (2 * i + 1) as f64 * pi / (4.0 * b as f64)).cos();
            }
        }
        t
    }

    pub fn mdct_fwd(x: &[f64], b: usize, win: &[f64], tab: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; b];
        for (i, o) in out.iter_mut().enumerate() {
            let row = &tab[i * 2 * b..(i + 1) * 2 * b];
            let mut acc = 0.0;
            for ((r, w), v) in row.iter().zip(win.iter()).zip(x.iter()) {
                acc += r * w * v;
            }
            *o = acc;
        }
        out
    }

    pub fn imdct_compact(x: &[f64], b: usize, norm: f64) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        let s = -(norm / b as f64).sqrt() / 32768.0;
        let half = b / 2;
        let mut y = vec![0.0f64; b];
        for i in 0..half {
            let mut a1 = 0.0;
            let mut a2 = 0.0;
            for (k, xv) in x.iter().enumerate() {
                let c1 = ((2 * k + 1) as f64 * (2 * b - 2 * i - 1) as f64 * pi / (4.0 * b as f64))
                    .cos();
                let c2 = ((2 * k + 1) as f64 * (3 * b + 2 * i + 1) as f64 * pi / (4.0 * b as f64))
                    .cos();
                a1 += xv * c1;
                a2 += xv * c2;
            }
            y[i] = s * a1;
            y[i + half] = -s * a2;
        }
        y
    }

    pub fn mdct_analysis_compact(y: &[f64], b: usize) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        let half = b / 2;
        let mut x = vec![0.0f64; b];
        for (k, xk) in x.iter_mut().enumerate() {
            let mut a1 = 0.0;
            let mut a2 = 0.0;
            for i in 0..half {
                let c1 = ((2 * k + 1) as f64 * (2 * b - 2 * i - 1) as f64 * pi / (4.0 * b as f64))
                    .cos();
                let c2 = ((2 * k + 1) as f64 * (3 * b + 2 * i + 1) as f64 * pi / (4.0 * b as f64))
                    .cos();
                a1 += y[i] * c1;
                a2 += y[i + half] * c2;
            }
            *xk = a1 - a2;
        }
        x
    }
}

// ----------------------------------------------------------------------------
// Тесты
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::StreamInfo;

    /// Перестановка — биекция [0, vect_size) и согласованность размеров.
    #[test]
    fn permutation_is_bijection() {
        for (isampf, ibps, n_ch) in [
            (8u32, 8u32, 1usize),
            (22, 32, 1),
            (44, 48, 1),
            (44, 48, 2),
        ] {
            let si = StreamInfo::new(isampf, ibps, n_ch).unwrap();
            for ftype in FrameType::ALL {
                let perm = construct_perm(si.mode, &si.layout, ftype, si.n_ch);
                let vect_size = if ftype == FrameType::Ppc {
                    n_ch * si.mode.ppc_shape_len
                } else {
                    n_ch * si.mode.size
                };
                assert_eq!(perm.len(), vect_size, "{isampf}/{ibps} {ftype:?}");
                let mut seen = vec![false; vect_size];
                for &p in &perm {
                    assert!((p as usize) < vect_size);
                    assert!(!seen[p as usize], "дубль {p}");
                    seen[p as usize] = true;
                }
            }
        }
    }

    /// Spot: перестановка PPC 44_48 моно (num_blocks=1 → transpose 6×9).
    #[test]
    fn ppc_perm_spot() {
        let si = StreamInfo::new(44, 48, 1).unwrap();
        let perm = construct_perm(si.mode, &si.layout, FrameType::Ppc, 1);
        assert_eq!(perm.len(), 54);
        // permutate — тождество (num_blocks=1), linear(1) — тождество;
        // остаётся transpose: столбец-мажор 6×9
        let expect: Vec<u16> = (0..54).map(|v| ((v % 9) * 6 + v / 9) as u16).collect();
        assert_eq!(perm, expect);
    }

    /// Сверка компактного IMDCT с golden (python twinlayers
    /// .imdct_compact_matrix, scripts/make_golden.py → imdct_golden.json).
    #[test]
    fn imdct_matches_python_golden() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        let Ok(raw) = std::fs::read_to_string(format!("{dir}/imdct_golden.json")) else {
            eprintln!("imdct_golden.json не найден — пропуск");
            return;
        };
        // {"B":512,"norm":2.0,"x":[...],"y":[...]}
        let b = parse_json_num(&raw, "B").unwrap_or(512.0) as usize;
        let norm = parse_json_num(&raw, "norm").unwrap_or(2.0);
        let (xs, ys) = parse_two_arrays(&raw);
        assert_eq!(xs.len(), b, "размер спектра");
        assert_eq!(ys.len(), b, "размер выхода");
        let y = imdct_compact(&xs, b, norm);
        let mut max_rel = 0.0f64;
        for (g, e) in y.iter().zip(ys.iter()) {
            let rel = (g - e).abs() / (e.abs() + 1e-6);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-9, "IMDCT расходится: rel={max_rel}");
        eprintln!("imdct vs python golden: max_rel={max_rel:.2e}");
    }

    /// FFT-ядро против наивного эталона: случайные данные, все три
    /// преобразования, B из реального диапазона sub-блоков режимов.
    #[test]
    fn fft_mdct_matches_naive() {
        let mut rng = 0x5EED_1234_ABCD_0001u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            2.0 * ((rng >> 32) as f64 / u32::MAX as f64) - 1.0
        };
        let maxrel = |a: &[f64], c: &[f64]| -> f64 {
            let mx = c.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
            a.iter()
                .zip(c.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f64, f64::max)
                / mx
        };
        for &b in &[64usize, 256, 512, 2048] {
            let y: Vec<f64> = (0..b).map(|_| next()).collect();
            let r = mdct_analysis_compact(&y, b);
            let n = naive::mdct_analysis_compact(&y, b);
            let e1 = maxrel(&r, &n);
            assert!(e1 < 1e-10, "analysis b={b}: {e1:.2e}");

            let x: Vec<f64> = (0..b).map(|_| next()).collect();
            let r = imdct_compact(&x, b, 2.0);
            let n = naive::imdct_compact(&x, b, 2.0);
            let e2 = maxrel(&r, &n);
            assert!(e2 < 1e-10, "imdct b={b}: {e2:.2e}");

            let xx: Vec<f64> = (0..2 * b).map(|_| next()).collect();
            let r = mdct_fwd(&xx, b);
            let win = sine_window_2b(b);
            let n = naive::mdct_fwd(&xx, b, &win, &naive::mdct_table(b));
            let e3 = maxrel(&r, &n);
            assert!(e3 < 1e-10, "fwd b={b}: {e3:.2e}");
            eprintln!("b={b}: max_rel анализ={e1:.2e} imdct={e2:.2e} fwd={e3:.2e}");
        }
    }

    fn parse_two_arrays(raw: &str) -> (Vec<f64>, Vec<f64>) {
        let xblob = raw.split("\"x\":").nth(1).unwrap_or("");
        let xend = xblob.find(']').unwrap_or(0);
        let xs: Vec<f64> = xblob[..xend]
            .split(',')
            .filter_map(|t| t.trim().trim_start_matches('[').trim().parse().ok())
            .collect();
        let yblob = raw.split("\"y\":").nth(1).unwrap_or("");
        let yend = yblob.find(']').unwrap_or(0);
        let ys: Vec<f64> = yblob[..yend]
            .split(',')
            .filter_map(|t| t.trim().trim_start_matches('[').trim().parse().ok())
            .collect();
        (xs, ys)
    }

    /// Сверка значений forward MDCT с golden (python twinlayers.mdct_fwd,
    /// scripts/make_golden.py → tests/golden/mdct_golden.json).
    #[test]
    fn mdct_matches_python_golden() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        let Ok(raw) = std::fs::read_to_string(format!("{dir}/mdct_golden.json")) else {
            eprintln!("mdct_golden.json не найден — пропуск");
            return;
        };
        // формат: {"mode": "44_48", "B": 2048, "size": 2048, "norm": 2.0,
        //          "ncoef": 64, "frames": [[...64], ...]}
        let b = parse_json_num(&raw, "B").unwrap_or(2048.0) as usize;
        let norm = parse_json_num(&raw, "norm").unwrap_or(2.0);
        let c = 2.0f64.powi(16) / (norm * b as f64).sqrt();
        // массив массивов
        let frames_blob = raw.split("\"frames\":").nth(1).unwrap_or("");
        let mut idx_frame = 0usize;
        let mut checked = 0usize;
        let mut max_rel = 0.0f64;
        let wav = std::env::var("TVQ_SIGNALS")
            .map(|p| format!("{p}/44m48/white.wav"))
            .unwrap_or_else(|_| {
                "/home/z/my-project/upload/encoded_all/tvq_cleanroom/signals/44m48/white.wav"
                    .to_string()
            });
        let Ok(pcm) = crate::wav::read_wav(&wav) else {
            eprintln!("сигнал {wav} недоступен — пропуск");
            return;
        };
        let x = &pcm[0];
        // значения всех кадров подряд (токены, не парсящиеся как f64, — мусор)
        let all: Vec<f64> = frames_blob
            .split([',', ']', '[', '\n', ' ', '\t'])
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        for expect in all.chunks_exact(64) {
            let k = 2 + idx_frame;
            let off = (k - 2) * b;
            let mut seg = vec![0.0f64; 2 * b];
            let n = (x.len().saturating_sub(off)).min(2 * b);
            seg[..n].copy_from_slice(&x[off..off + n]);
            let got = mdct_fwd(&seg, b);
            for (i, e) in expect.iter().enumerate() {
                let g = got[i] * c;
                let rel = (g - e).abs() / (e.abs() + 1e-9);
                max_rel = max_rel.max(rel);
                checked += 1;
            }
            idx_frame += 1;
        }
        assert!(checked >= 200, "слишком мало сверки: {checked}");
        assert!(max_rel < 1e-9, "MDCT расходится с python: rel={max_rel}");
        eprintln!("mdct vs python golden: {checked} значений, max_rel={max_rel:.2e}");
    }

    fn parse_json_num(s: &str, name: &str) -> Option<f64> {
        let pat = format!("\"{name}\":");
        let p = s.find(&pat)?;
        let r = s[p + pat.len()..].trim_start();
        let end = r.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))?;
        r[..end].parse().ok()
    }
}
