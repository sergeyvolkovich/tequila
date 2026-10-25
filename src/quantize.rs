//! Квантователи слоёв энкодера (инверсии декодерских операций).
//!
//! Все искатели минимизируют фактическую ошибку реконструкции в домене
//! соответствующего слоя и используют ТОЧНУЮ арифметику декодера
//! (`decode::*`) для оценки кандидатов:
//!
//! * `quant_lsp` — split-VQ + MA-предсказание (сдвиг цели через cb3);
//! * `quant_bark` — раздельный поиск по чётностям полос + use_hist;
//! * `quant_gain*` — перебор mulaw-гридов (LONG / gain×sub_gain);
//! * `quant_ppc` — период (ACF) → жадный поиск (g_coef × shape VQ);
//! * `quant_vq` — конъюгатный VQ per-дивизия (знаки при 7 битах).

use crate::codec::StreamInfo;
use crate::decode::{mulawinv, AMP_MAX, MULAW_MU, SUB_AMP_MAX};
use crate::layout::DivLayout;
use crate::modes::FrameType;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Грид gain LONG: g(b) = mulawinv(step/2 + step·b)/2^13, b = 0..255.
/// Режимонезависимый — статический кэш (256 exp-вызовов один раз).
fn gain_long_grid() -> &'static [f64] {
    static G: OnceLock<Vec<f64>> = OnceLock::new();
    G.get_or_init(|| {
        let step = AMP_MAX / 255.0;
        (0..256)
            .map(|b| {
                (1.0 / (1 << 13) as f64)
                    * mulawinv(step * 0.5 + step * b as f64, AMP_MAX, MULAW_MU)
            })
            .collect()
    })
}

/// Базовое значение gain не-LONG: val(b) = mulawinv(...)/2^23.
fn gain_val(b: u8) -> f64 {
    let step = AMP_MAX / 255.0;
    (1.0 / (1 << 23) as f64) * mulawinv(step * 0.5 + step * b as f64, AMP_MAX, MULAW_MU)
}

/// Грид sub_gain: s(b) = mulawinv(sub_step/2 + sub_step·b, 4500, 100).
fn sub_gain_grid() -> &'static [f64] {
    static G: OnceLock<Vec<f64>> = OnceLock::new();
    G.get_or_init(|| {
        let step = SUB_AMP_MAX / 31.0;
        (0..32)
            .map(|b| mulawinv(step * 0.5 + step * b as f64, SUB_AMP_MAX, MULAW_MU))
            .collect()
    })
}

/// Индекс ближайшего элемента монотонно возрастающего грида; при равных
/// расстояниях — меньший индекс (семантика линейного перебора).
fn nearest_index(grid: &[f64], t: f64) -> usize {
    let hi = grid.partition_point(|&g| g < t); // первый ≥ t
    let lo = hi.saturating_sub(1);
    let mut best = lo;
    let mut be = (grid[lo] - t).abs();
    if hi < grid.len() {
        let e = (grid[hi] - t).abs();
        if e < be {
            best = hi;
            be = e;
        }
    }
    let _ = be;
    best
}

/// Квантование gain LONG: ближайший грид к целевому значению.
pub fn quant_gain_long(target: f64) -> u8 {
    nearest_index(gain_long_grid(), target) as u8
}

/// Квантование gain + sub_gain (не-LONG): минимизирует Σ_j (val·s_j − t_j)².
pub fn quant_gain_sub(targets: &[f64]) -> (u8, Vec<u8>) {
    let sg = sub_gain_grid();
    let mut best = (0u8, vec![0u8; targets.len()]);
    let mut best_err = f64::INFINITY;
    for b in 0..256u32 {
        let val = gain_val(b as u8);
        if val <= 0.0 {
            continue;
        }
        let mut err = 0.0;
        let mut subs = Vec::with_capacity(targets.len());
        for &t in targets {
            let want = t / val;
            let sb = nearest_index(sg, want) as u8;
            let d = val * sg[sb as usize] - t;
            err += d * d;
            subs.push(sb);
        }
        if err < best_err {
            best_err = err;
            best = (b as u8, subs);
        }
    }
    best
}

/// Декодированное значение gain LONG по битам (для согласования масштабов).
pub fn gain_long_value(bits: u8) -> f64 {
    gain_long_grid()[bits as usize]
}

// ----------------------------------------------------------------------------
// LSP split-VQ + MA
// ----------------------------------------------------------------------------

/// Результат LSP-квантования.
pub struct LspIndices {
    pub hist_idx: u8,
    pub idx1: u8,
    pub idx2: Vec<u8>,
}

/// Квантование LSP: split-VQ с MA-предсказанием.
///
/// Поиск: для каждого hist_idx (lsp_bit0) и idx1 (lsp_bit1) — независимый
/// перебор idx2 по сплитам (чанки коэффициентов не пересекаются), цель
/// сдвигается через MA-вектор cb3: target' = (target − ma·hist)/(1 − ma).
///
/// `weights` — перцептивные веса коэффициентов (режим `--psy`, PSY.md §7):
/// слышимость у частоты k-го LSP; None — равные веса (как базовый режим).
pub fn quant_lsp(
    info: &StreamInfo,
    target: &[f64],
    hist: &[f32],
    weights: Option<&[f64]>,
) -> LspIndices {
    let m = info.mode;
    let n = m.n_lsp;
    let cb = m.lspcodebook;
    let cb2_base = (1 << m.lsp_bit1) * n;
    let cb3_base = cb2_base + (1 << m.lsp_bit2) * n;
    let split = m.lsp_split;
    let funny: [i32; 4] = [
        -2,
        if split == 4 { -2 } else { 1 },
        if split == 4 { -2 } else { 1 },
        0,
    ];
    // границы чанков (как в decode_lsp)
    let mut chunk_bounds = Vec::with_capacity(split);
    let mut j0 = 0usize;
    for i in 0..split {
        let end = (((i + 1) * n) as i32 + funny[i]) as usize / split;
        chunk_bounds.push((j0, end));
        j0 = end;
    }

    let mut best = LspIndices { hist_idx: 0, idx1: 0, idx2: vec![0; split] };
    let mut best_err = f64::INFINITY;
    for hist_idx in 0..(1u32 << m.lsp_bit0) {
        let ma = &cb[cb3_base + hist_idx as usize * n..][..n];
        // сдвиг цели через MA (в f64; квантование грубое — дисциплина не критична)
        let mut shifted = vec![0.0f64; n];
        let mut ok = true;
        for k in 0..n {
            let denom = 1.0 - ma[k] as f64;
            if denom.abs() < 0.05 {
                ok = false;
                break;
            }
            shifted[k] = (target[k] - ma[k] as f64 * hist[k] as f64) / denom;
        }
        if !ok {
            continue;
        }
        for idx1 in 0..(1u32 << m.lsp_bit1) {
            let base = &cb[idx1 as usize * n..][..n];
            let mut err = 0.0;
            let mut idx2 = vec![0u8; split];
            for (si, &(lo, hi)) in chunk_bounds.iter().enumerate() {
                let mut b_err = f64::INFINITY;
                let mut b_idx = 0u8;
                for cand in 0..(1u32 << m.lsp_bit2) {
                    let cb2 = &cb[cb2_base + cand as usize * n..][..n];
                    let mut e = 0.0;
                    for k in lo..hi {
                        let d = (base[k] as f64 + cb2[k] as f64) - shifted[k];
                        let wk = weights.map_or(1.0, |w| w[k]);
                        e += wk * d * d;
                    }
                    if e < b_err {
                        b_err = e;
                        b_idx = cand as u8;
                    }
                }
                err += b_err;
                idx2[si] = b_idx;
            }
            // штраф за немонотонность cb-суммы: декодер сортирует lsp
            // (sort_nearly_sorted) — при пересечениях чёт/нечёт позиции
            // перемешиваются и огибающая получает полюса
            let mut sum = vec![0.0f64; n];
            let mut j0 = 0usize;
            for (si, &(lo, hi)) in chunk_bounds.iter().enumerate() {
                let cb2 = &cb[cb2_base + idx2[si] as usize * n..][..n];
                for k in lo..hi {
                    sum[k] = base[k] as f64 + cb2[k] as f64;
                }
                j0 = hi;
            }
            let _ = j0;
            for k in 1..n {
                if sum[k] < sum[k - 1] {
                    err += 100.0 * (sum[k - 1] - sum[k]).powi(2);
                }
            }
            if err < best_err {
                best_err = err;
                best = LspIndices { hist_idx: hist_idx as u8, idx1: idx1 as u8, idx2 };
            }
        }
    }
    best
}

// ----------------------------------------------------------------------------
// Bark-огибающая
// ----------------------------------------------------------------------------

/// Результат bark-квантования одного подкадра.
pub struct BarkDecision {
    pub indices: Vec<u8>,
    pub use_hist: bool,
    /// Декодированные st-значения по полосам (без gain).
    pub st: Vec<f64>,
}

/// Квантование bark-огибающей подкадра.
///
/// `bands` — целевые уровни по bark-полосам (RMS |W| в полосе, длина
/// bark_env_size); полосы независимы по чётности j (каждый индекс in[j]
/// управляет полосами своей чётности), масштаб μ итеративно уточняется.
///
/// `weights` — перцептивные веса полос (режим `--psy`, PSY.md §3):
/// ошибка st в полосе взвешивается слышимостью `(env_rms/m_rms)²`;
/// `None` — равные веса (обычный режим, битстрим не меняется).
pub fn quant_bark(
    info: &StreamInfo,
    ftype: FrameType,
    bands: &[f64],
    hist: &mut [f64],
    weights: Option<&[f64]>,
) -> BarkDecision {
    let fm = &info.mode.fmode[ftype.idx()];
    let ncoef = fm.bark_n_coef;
    let fw_cb_len = fm.bark_env_size / ncoef;
    let val = match ftype {
        FrameType::Short => 0.4,
        FrameType::Medium => 0.35,
        _ => 0.28,
    };

    // st-значения кандидата: полоса (i, j) = bark_cb[fw_cb_len·in[j] + i]/4096
    let st_of = |inj: u8, i: usize| -> f64 {
        fm.bark_cb[fw_cb_len * inj as usize + i] as f64 / 4096.0
    };
    let st_with_hist = |inj: u8, i: usize, idx: usize| -> f64 {
        let tmp2 = st_of(inj, i);
        let mut st = (1.0 - val) * tmp2 + val * hist[idx] + 1.0;
        if st < -1.0 {
            st = 1.0;
        }
        st
    };

    // начальный масштаб μ: цель/средний st
    let mean_target: f64 = bands.iter().sum::<f64>() / bands.len() as f64;
    let mut mu = if mean_target > 1e-9 { mean_target } else { 1.0 };

    let mut indices = vec![0u8; ncoef];
    let mut use_hist = false;
    for _iter in 0..3 {
        // перебор use_hist × индексов при фиксированном μ
        let mut best_total = f64::INFINITY;
        let mut best_indices = vec![0u8; ncoef];
        let mut best_hist = false;
        for uh in [false, true] {
            let mut total = 0.0;
            let mut idxs = vec![0u8; ncoef];
            for j in 0..ncoef {
                let mut b_err = f64::INFINITY;
                let mut b_idx = 0u8;
                for cand in 0..(1u32 << fm.bark_n_bit) {
                    let mut e = 0.0;
                    let mut idx = j;
                    for i in 0..fw_cb_len {
                        let st = if uh {
                            st_with_hist(cand as u8, i, idx)
                        } else {
                            st_of(cand as u8, i) + 1.0
                        };
                        let d = st * mu - bands[idx];
                        let w = weights.map_or(1.0, |w| w[idx]);
                        e += w * d * d;
                        idx += ncoef;
                    }
                    if e < b_err {
                        b_err = e;
                        b_idx = cand as u8;
                    }
                }
                idxs[j] = b_idx;
                total += b_err;
            }
            if total < best_total {
                best_total = total;
                best_indices = idxs;
                best_hist = uh;
            }
        }
        indices = best_indices;
        use_hist = best_hist;
        // пересчёт μ (взвешенный МНК по st·μ ≈ bands)
        let mut num = 0.0;
        let mut den = 0.0;
        let mut idx = 0usize;
        for i in 0..fw_cb_len {
            for j in 0..ncoef {
                let st = if use_hist {
                    st_with_hist(indices[j], i, idx)
                } else {
                    st_of(indices[j], i) + 1.0
                };
                let w = weights.map_or(1.0, |w| w[idx]);
                num += w * bands[idx] * st;
                den += w * st * st;
                idx += 1;
            }
        }
        if den > 1e-12 {
            mu = num / den;
        }
    }

    // финальные st (для последующего деления) + обновление MA-истории
    // (decode::dec_bark_env пишет tmp2 в hist — энкодер обязан делать то же)
    let mut st = vec![0.0f64; fm.bark_env_size];
    let mut idx = 0usize;
    for i in 0..fw_cb_len {
        for j in 0..ncoef {
            let tmp2 = st_of(indices[j], i);
            st[idx] = if use_hist {
                let mut v = (1.0 - val) * tmp2 + val * hist[idx] + 1.0;
                if v < -1.0 {
                    v = 1.0;
                }
                v
            } else {
                tmp2 + 1.0
            };
            hist[idx] = tmp2;
            idx += 1;
        }
    }
    BarkDecision { indices, use_hist, st }
}

// ----------------------------------------------------------------------------
// Конъюгатный VQ
// ----------------------------------------------------------------------------

/// RMS масштаб пары кодбуков: σ = sqrt(mean(cb0²) + mean(cb1²)).
pub fn cb_pair_sigma(cb0: &[i16], cb1: &[i16]) -> f64 {
    let s0: f64 = cb0.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let s1: f64 = cb1.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    ((s0 + s1) / (cb0.len() + cb1.len()) as f64).sqrt()
}

/// Кэш пары кодбуков конъюгатного VQ: f64-копии книг, нормы строк и
/// кросс-члены a·b — всё, что не зависит от цели квантования.
///
/// Без кэша кросс-члены пересчитывались на каждую дивизию каждого кадра
/// (n0·n1·len MAC ≈ 17 млн на кадр 44_48 стерео). Записи ленивы, ключ
/// (b0, b1, len): биты дивизий меняются после `change`, длины — после
/// `length_change` (≤ 4 комбинаций на пару книг).
pub struct CbPairCache {
    cb0: Vec<f64>,
    cb1: Vec<f64>,
    cb_len: usize,
    entries: HashMap<(usize, usize, usize), CbEntry>,
}

struct CbEntry {
    nrm0: Vec<f64>,
    nrm1: Vec<f64>,
    /// cross[i·n1 + j] = Σ_k cb0[i·cb_len+k]·cb1[j·cb_len+k].
    cross: Vec<f64>,
    /// cmax[i] = max_j |cross[i][j]| — для отсева строк поиска.
    cmax: Vec<f64>,
}

impl CbPairCache {
    /// Пара кодбуков (cb0/cb1) с размером входа cb_len.
    pub fn new(cb0: &[i16], cb1: &[i16], cb_len: usize) -> Self {
        CbPairCache {
            cb0: cb0.iter().map(|&v| v as f64).collect(),
            cb1: cb1.iter().map(|&v| v as f64).collect(),
            cb_len,
            entries: HashMap::new(),
        }
    }

    /// Чистый расчёт записи кэша (нормы строк, кросс-члены a·b, cmax) —
    /// не мутирует состояние: пригоден для локальных кэшей без прогрева
    /// и для многопоточного чтения после [`Self::prewarm`].
    fn compute_entry(&self, b0: usize, b1: usize, len: usize) -> CbEntry {
        let cb_len = self.cb_len;
            let n0 = if b0 == 7 { self.cb0.len() / cb_len } else { 1 << b0 };
            let n1 = if b1 == 7 { self.cb1.len() / cb_len } else { 1 << b1 };
            let mut nrm0 = vec![0.0f64; n0];
            for (i, q) in nrm0.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..len {
                    let v = self.cb0[i * cb_len + k];
                    acc += v * v;
                }
                *q = acc;
            }
            let mut nrm1 = vec![0.0f64; n1];
            for (j, q) in nrm1.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..len {
                    let v = self.cb1[j * cb_len + k];
                    acc += v * v;
                }
                *q = acc;
            }
            let mut cross = vec![0.0f64; n0 * n1];
            let mut cmax = vec![0.0f64; n0];
            for i in 0..n0 {
                let mut rowmax = 0.0f64;
                for j in 0..n1 {
                    let mut c = 0.0;
                    for k in 0..len {
                        c += self.cb0[i * cb_len + k] * self.cb1[j * cb_len + k];
                    }
                    cross[i * n1 + j] = c;
                    rowmax = rowmax.max(c.abs());
                }
                cmax[i] = rowmax;
            }
            CbEntry { nrm0, nrm1, cross, cmax }
    }

    fn entry(&mut self, b0: usize, b1: usize, len: usize) -> &CbEntry {
        if !self.entries.contains_key(&(b0, b1, len)) {
            let e = self.compute_entry(b0, b1, len);
            self.entries.insert((b0, b1, len), e);
        }
        self.entries.get(&(b0, b1, len)).unwrap()
    }

    /// Прогреть кэш кросс-членов для всех дивизий слоя: после этого
    /// [`Self::search`] работает через `&self` (без мутаций) — обязательное
    /// условие многопоточного кодирования и SIMD-ядер.
    pub fn prewarm(&mut self, d: &DivLayout) {
        for i in 0..d.n_div {
            let len = d.length_at(i);
            let (b0, b1) = d.bits_at(i);
            self.entry(b0, b1, len);
        }
    }

    /// Квантование одной дивизии: пара (i0, i1) ± знаки (совместный
    /// перебор 4 комбинаций ТОЛЬКО кодируемых при 7 битах; при 6 битах
    /// декодер всегда читает +1).
    ///
    /// Оптимизация: нормы/кроссы — из кэша; столбцовые члены и знаки
    /// факторуются; строки отсеваются нижней оценкой
    /// `vv + nrm0[i] − 2|dot0[i]| + min_j(nrm1[j] − 2|dot1[j]|) − 2·cmax[i]`
    /// (валидна для всех j и знаков — оптимальная пара никогда не
    /// отсекается); минимизация по j — векторизуемый min-reduce.
    /// Реассоциация FP может менять тай-брейки в вырожденных случаях
    /// (значения ошибок одинаковы — на качество не влияет).
    pub fn search(&self, v: &[f64], b0: usize, b1: usize) -> (u8, u8) {
        let len = v.len();
        let cb_len = self.cb_len;
        let n0 = if b0 == 7 { self.cb0.len() / cb_len } else { 1usize << b0 };
        let n1 = if b1 == 7 { self.cb1.len() / cb_len } else { 1usize << b1 };
        // скалярные произведения с целью: SIMD-ядра с лейнами по строкам
        // книг (порядок суммирования по k — как в .sum() скаляра)
        let mut dot0 = vec![0.0f64; n0];
        crate::simd::dots(&self.cb0, cb_len, 0, n0, v, &mut dot0);
        let mut dot1 = vec![0.0f64; n1];
        crate::simd::dots(&self.cb1, cb_len, 0, n1, v, &mut dot1);
        let vv: f64 = v.iter().map(|x| x * x).sum();
        // энкодер прогревает кэш (prewarm) — обычный путь; локальные кэши
        // без прогрева считают запись на лету без мутации состояния
        let fallback;
        let e: &CbEntry = match self.entries.get(&(b0, b1, len)) {
            Some(e) => e,
            None => {
                fallback = self.compute_entry(b0, b1, len);
                &fallback
            }
        };
        // столбцовые члены для s1 = ±1 и минимум для отсева строк
        let mut col_p = vec![0.0f64; n1];
        let mut col_m = vec![0.0f64; n1];
        for j in 0..n1 {
            col_p[j] = e.nrm1[j] - 2.0 * dot1[j];
            col_m[j] = e.nrm1[j] + 2.0 * dot1[j];
        }
        let min_neg1 = crate::simd::min_transform(&e.nrm1, &dot1);
        const COMBOS_44: [(f64, f64); 4] = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        const COMBOS_S0: [(f64, f64); 2] = [(1.0, 1.0), (-1.0, 1.0)];
        const COMBOS_S1: [(f64, f64); 2] = [(1.0, 1.0), (1.0, -1.0)];
        const COMBOS_1: [(f64, f64); 1] = [(1.0, 1.0)];
        let combos: &[(f64, f64)] = match (b0 == 7, b1 == 7) {
            (true, true) => &COMBOS_44,
            (true, false) => &COMBOS_S0,
            (false, true) => &COMBOS_S1,
            (false, false) => &COMBOS_1,
        };
        let mut best_err = f64::INFINITY;
        let mut best = (0usize, 0usize, 1.0f64, 1.0f64);
        for i in 0..n0 {
            // отсев строки: нижняя оценка ошибки по всем j и знакам
            let lb = vv + e.nrm0[i] - 2.0 * dot0[i].abs() + min_neg1 - 2.0 * e.cmax[i];
            if lb >= best_err {
                continue;
            }
            let row = &e.cross[i * n1..(i + 1) * n1];
            for &(s0, s1) in combos {
                let rc = vv + e.nrm0[i] - 2.0 * s0 * dot0[i];
                let col: &[f64] = if s1 > 0.0 { &col_p } else { &col_m };
                let k = 2.0 * s0 * s1;
                // SIMD-скан: значение минимума — всегда, первый индекс —
                // только при улучшении (семантика fold+position скаляра)
                let be = crate::simd::scan_min_val(col, row, rc, k);
                if be < best_err {
                    let jj = crate::simd::scan_min_idx(col, row, rc, k, be);
                    best_err = be;
                    best = (i, jj, s0, s1);
                }
            }
        }
        let encode = |i: usize, s: f64, bits: usize| -> u8 {
            let mut code = i as u8;
            if bits == 7 && s < 0.0 {
                code |= 0x40;
            }
            code
        };
        (encode(best.0, best.2, b0), encode(best.1, best.3, b1))
    }

    /// Взвешенный NN-поиск: min Σ_t w_t·(v_t − s0·a_it − s1·b_jt)².
    ///
    /// Это оптимум в домене выхода кадра: ошибка реконструкции бина t
    /// умножается на масштаб слоя scale_t = env·st·g, поэтому правильная
    /// метрика поиска — Σ (ΔV_t · scale_t)², а не Σ ΔV_t². На кадрах с
    /// контрастным bark-профилем (транзиенты: энергия в 1–2 подкадрах из
    /// 16, st «пустых» полос ≈ 0) невзвешенный поиск тратит бюджет
    /// ошибки на бины, которые декодер всё равно обнуляет, и искажает
    /// совпадение на значимых бинах (замер clicks: eff 0.40 → 0.025).
    ///
    /// Веса нормируются на max; бины с w < 1e-4·max исключаются из всех
    /// сумм (вклад в целевую < 1e-4·wmax·V² на бин). Отсев строк —
    /// Коши–Буняковского: |cross_ij| ≤ √(n0w_i·n1w_j).
    ///
    /// Кросс-члены зависят от весов (т.е. от кадра) — кэш `entries`
    /// здесь не используется, только f64-копии кодбуков.
    pub fn search_w(&self, v: &[f64], w: &[f64], b0: usize, b1: usize) -> (u8, u8) {
        let len = v.len();
        debug_assert_eq!(w.len(), len);
        let cb_len = self.cb_len;
        let n0 = if b0 == 7 { self.cb0.len() / cb_len } else { 1usize << b0 };
        let n1 = if b1 == 7 { self.cb1.len() / cb_len } else { 1usize << b1 };

        let wmax = w.iter().cloned().fold(0.0f64, f64::max);
        debug_assert!(wmax > 0.0, "search_w вызван для полностью «убитой» дивизии");

        // активное множество (индекс, нормированный вес)
        let thr = 1e-4 * wmax;
        let mut act: Vec<usize> = Vec::with_capacity(len);
        let mut wa: Vec<f64> = Vec::with_capacity(len);
        for (t, &x) in w.iter().enumerate() {
            if x >= thr {
                act.push(t);
                wa.push(x / wmax);
            }
        }
        let k = act.len();

        // vv, wva (= wa·v — одно округление, как в (wa·v)·a скаляра),
        // взвешенные точки и нормы строк обеих книг — SIMD по строкам
        let mut wva = vec![0.0f64; k];
        let mut vv = 0.0f64;
        for ki in 0..k {
            let t = act[ki];
            wva[ki] = wa[ki] * v[t];
            vv += wva[ki] * v[t];
        }
        let mut d0 = vec![0.0f64; n0];
        let mut n0w = vec![0.0f64; n0];
        crate::simd::dots_w(&self.cb0, cb_len, 0, n0, &act, &wva, &wa, &mut d0, &mut n0w);
        let mut d1 = vec![0.0f64; n1];
        let mut n1w = vec![0.0f64; n1];
        crate::simd::dots_w(&self.cb1, cb_len, 0, n1, &act, &wva, &wa, &mut d1, &mut n1w);
        let n1w_max = n1w.iter().cloned().fold(0.0f64, f64::max);
        // min_j (n1w[j] − 2|d1[j]|) — для отсева строк
        let min_neg1 = crate::simd::min_transform(&n1w, &d1);

        const COMBOS_44: [(f64, f64); 4] = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        const COMBOS_S0: [(f64, f64); 2] = [(1.0, 1.0), (-1.0, 1.0)];
        const COMBOS_S1: [(f64, f64); 2] = [(1.0, 1.0), (1.0, -1.0)];
        const COMBOS_1: [(f64, f64); 1] = [(1.0, 1.0)];
        let combos: &[(f64, f64)] = match (b0 == 7, b1 == 7) {
            (true, true) => &COMBOS_44,
            (true, false) => &COMBOS_S0,
            (false, true) => &COMBOS_S1,
            (false, false) => &COMBOS_1,
        };

        // столбцовые члены для s1 = ±1
        let mut col_p = vec![0.0f64; n1];
        let mut col_m = vec![0.0f64; n1];
        for j in 0..n1 {
            col_p[j] = n1w[j] - 2.0 * d1[j];
            col_m[j] = n1w[j] + 2.0 * d1[j];
        }

        // транспонированная активная подматрица cb1: bt[ki·n1 + j] =
        // cb1[j·cb_len + act[ki]] — внутренний цикл кроссов становится
        // непрерывным FMA (авто-векторизуется)
        let mut bt = vec![0.0f64; k * n1];
        for ki in 0..k {
            let t = act[ki];
            let dst = &mut bt[ki * n1..(ki + 1) * n1];
            for (j, d) in dst.iter_mut().enumerate() {
                *d = self.cb1[j * cb_len + t];
            }
        }

        let mut cross = vec![0.0f64; n1]; // строка кроссов текущего i
        let mut best_err = f64::INFINITY;
        let mut best = (0usize, 0usize, 1.0f64, 1.0f64);
        for i in 0..n0 {
            // отсев строки (Коши–Буняковский)
            let lb = vv + n0w[i] - 2.0 * d0[i].abs() + min_neg1
                - 2.0 * (n0w[i] * n1w_max).sqrt();
            if lb >= best_err {
                continue;
            }
            // cross[j] = Σ_k wa_k·a_ik·b_jk
            let row = &self.cb0[i * cb_len..];
            for cj in cross.iter_mut() {
                *cj = 0.0;
            }
            for ki in 0..k {
                let aik = wa[ki] * row[act[ki]];
                if aik == 0.0 {
                    continue;
                }
                let brow = &bt[ki * n1..(ki + 1) * n1];
                crate::simd::cross_accum(&mut cross, aik, brow);
            }
            for &(s0, s1) in combos {
                let rc = vv + n0w[i] - 2.0 * s0 * d0[i];
                let col: &[f64] = if s1 > 0.0 { &col_p } else { &col_m };
                let kk = 2.0 * s0 * s1;
                let be = crate::simd::scan_min_val(col, &cross, rc, kk);
                if be < best_err {
                    let jj = crate::simd::scan_min_idx(col, &cross, rc, kk, be);
                    best_err = be;
                    best = (i, jj, s0, s1);
                }
            }
        }
        let encode = |i: usize, s: f64, bits: usize| -> u8 {
            let mut code = i as u8;
            if bits == 7 && s < 0.0 {
                code |= 0x40;
            }
            code
        };
        (encode(best.0, best.2, b0), encode(best.1, best.3, b1))
    }
}

/// Квантование main VQ целого кадра: `target_flat` — плоский спектр в
/// **VQ-порядке** (после перестановки), длина n_ch·size.
///
/// Возвращает interleaved-пары индексов (порядок read_cb_data).
pub fn quant_vq_main(
    info: &StreamInfo,
    ftype: FrameType,
    target_flat: &[f64],
    perm: &[u16],
) -> Vec<u8> {
    let fm = &info.mode.fmode[ftype.idx()];
    let mut cache = CbPairCache::new(fm.cb0, fm.cb1, fm.cb_len_read);
    cache.prewarm(info.layout.div(ftype));
    quant_vq_main_cached(&cache, info.layout.div(ftype), target_flat, perm)
}

/// Основной VQ кадра по кэшированной паре книг (энкодер держит кэш
/// между кадрами — кросс-члены не пересчитываются).
pub fn quant_vq_main_cached(
    cache: &CbPairCache,
    d: &DivLayout,
    target_flat: &[f64],
    perm: &[u16],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(d.n_div * 2);
    let mut pos = 0usize;
    let mut v: Vec<f64> = Vec::new();
    for i in 0..d.n_div {
        let length = d.length_at(i);
        let (b0, b1) = d.bits_at(i);
        // VQ-порядок: S[t] = flat[perm[t]]
        v.clear();
        v.resize(length, 0.0);
        for (t, vt) in v.iter_mut().enumerate() {
            *vt = target_flat[perm[pos + t] as usize];
        }
        let (c0, c1) = cache.search(&v, b0, b1);
        out.push(c0);
        out.push(c1);
        pos += length;
    }
    out
}

/// Main VQ с весами бинов: минимизирует ошибку в домене выхода кадра
/// `Σ (ΔV·scale)²`, см. [`CbPairCache::search_w`].
///
/// Маршрутизация по дивизиям: если веса внутри дивизии почти равномерны
/// (max ≤ 8·min > 0), взвешенная и невзвешенная метрики эквивалентны с
/// точностью до константы — используется быстрый кэшированный поиск
/// (битстрим стационарных кадров не меняется). Контрастные дивизии
/// (тишина/транзиент, сток-спектры) идут точным взвешенным поиском.
///
/// Дивизии независимы (даны flat_vq/w_vq/perm) → параллелятся пулом:
/// каждая пишет только свою пару кодов; порядок выборки — детерминирован
/// слотами, битстрим не зависит от числа потоков.
pub fn quant_vq_main_w(
    pool: &crate::parallel::Pool,
    cache: &CbPairCache,
    d: &DivLayout,
    target_flat: &[f64],
    w_flat: &[f64],
    perm: &[u16],
) -> Vec<u8> {
    // кумулятивные позиции дивизий в VQ-порядке
    let mut pos_of = Vec::with_capacity(d.n_div);
    let mut acc = 0usize;
    for i in 0..d.n_div {
        pos_of.push(acc);
        acc += d.length_at(i);
    }
    let mut pairs = vec![[0u8, 0u8]; d.n_div];
    pool.map_into(&mut pairs, |i| {
        let length = d.length_at(i);
        let (b0, b1) = d.bits_at(i);
        let pos = pos_of[i];
        let mut v = vec![0.0f64; length];
        let mut wv = vec![0.0f64; length];
        let mut wmax = 0.0f64;
        let mut wmin = f64::INFINITY;
        for t in 0..length {
            let p = perm[pos + t] as usize;
            v[t] = target_flat[p];
            let x = w_flat[p];
            wv[t] = x;
            if x > wmax {
                wmax = x;
            }
            if x > 0.0 && x < wmin {
                wmin = x;
            }
        }
        let uniform = wmin.is_finite() && wmax <= 8.0 * wmin;
        let (c0, c1) = if uniform {
            cache.search(&v, b0, b1)
        } else if wmax > 0.0 {
            cache.search_w(&v, &wv, b0, b1)
        } else {
            // весь блок «убит» масштабом — коды не влияют на выход;
            // детерминированный выбор: невзвешенный поиск
            cache.search(&v, b0, b1)
        };
        [c0, c1]
    });
    let mut out = Vec::with_capacity(d.n_div * 2);
    for p in pairs {
        out.push(p[0]);
        out.push(p[1]);
    }
    out
}

/// Квантование PPC shape: цель — форма пиков (длина shape_len на канал,
/// VQ-порядок PPC-перестановки). Возвращает interleaved-пары.
pub fn quant_vq_ppc(
    info: &StreamInfo,
    target_flat: &[f64],
    perm: &[u16],
) -> Vec<u8> {
    let m = info.mode;
    let n_div = info.layout.ppc.n_div;
    let cb_len_p = (n_div + m.ppc_shape_len * info.n_ch - 1) / n_div;
    let (cb0, cb1) = m.ppc_shape_cb.split_at(cb_len_p * 64);
    let mut cache = CbPairCache::new(cb0, cb1, cb_len_p);
    cache.prewarm(&info.layout.ppc);
    quant_vq_ppc_cached(&cache, &info.layout.ppc, target_flat, perm)
}

/// PPC shape по кэшированной паре книг (энкодер держит кэш).
pub fn quant_vq_ppc_cached(
    cache: &CbPairCache,
    d: &DivLayout,
    target_flat: &[f64],
    perm: &[u16],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(d.n_div * 2);
    let mut pos = 0usize;
    let mut v: Vec<f64> = Vec::new();
    for i in 0..d.n_div {
        let length = d.length_at(i);
        let (b0, b1) = d.bits_at(i);
        v.clear();
        v.resize(length, 0.0);
        for (t, vt) in v.iter_mut().enumerate() {
            *vt = target_flat[perm[pos + t] as usize];
        }
        let (c0, c1) = cache.search(&v, b0, b1);
        out.push(c0);
        out.push(c1);
        pos += length;
    }
    out
}

// ----------------------------------------------------------------------------
// PPC: период/гейн
// ----------------------------------------------------------------------------

/// Сетка периодов PPC (по p_coef).
pub fn ppc_period_grid(info: &StreamInfo) -> Vec<i64> {
    let m = info.mode;
    let isampf = (info.sr / 1000) as i64;
    let min_p = crate::decode::rounded_div(40 * 2 * m.size as i64, isampf);
    let max_p = crate::decode::rounded_div(40 * 2 * m.size as i64 * 6, isampf);
    let range = max_p - min_p;
    (0..(1u32 << m.ppc_period_bit))
        .map(|p| {
            min_p
                + crate::decode::rounded_div(
                    p as i64 * range,
                    ((1u64 << m.ppc_period_bit) - 1) as i64,
                )
        })
        .collect()
}

/// Грид ppc_gain по g_coef.
pub fn ppc_gain_grid(info: &StreamInfo) -> Vec<f64> {
    let m = info.mode;
    let step = 25000.0 / ((1u64 << m.pgain_bit) - 1) as f64;
    (0..(1u32 << m.pgain_bit))
        .map(|g| {
            (1.0 / 8192.0)
                * mulawinv(step * g as f64 + step / 2.0, 25000.0, crate::decode::PGAIN_MU)
        })
        .collect()
}

/// Геометрия пиков add_peak для данного периода (позиции в спектре).
///
/// Возвращает (позиции бинов, индексы потребления shape si) — так, чтобы
/// PPC_synth[bin] += ppc_gain·shape[si].
pub fn ppc_peak_positions(info: &StreamInfo, period: i64) -> (Vec<i64>, Vec<usize>) {
    let m = info.mode;
    let size = m.size as i64;
    let isampf = (info.sr / 1000) as i64;
    let ibps = (info.kbps_total / info.n_ch) as i64;
    let width = if isampf == 22 && ibps == 32 {
        crate::decode::rounded_div((period + 800) * m.peak_per2wid as i64, 400 * size)
    } else {
        period * m.peak_per2wid as i64 / (400 * size)
    };
    let len = m.ppc_shape_len as i64;
    let mut bins = Vec::new();
    let mut sis = Vec::new();
    if width <= 0 {
        return (bins, sis);
    }
    let half = width / 2;
    let full = (width + 1) / 2;
    let mut si = 0usize;
    for i in 0..(width / 2) {
        if (i as usize) < m.ppc_shape_len {
            bins.push(i);
            sis.push(si);
        }
        si += 1;
    }
    let mut i = 1i64;
    while i < crate::decode::rounded_div(len, width) {
        let center = crate::decode::very_broken_op(period, i);
        for j in -half..full {
            let p = j + center;
            if p >= 0 && p < size {
                bins.push(p);
                sis.push(si);
            }
            si += 1;
        }
        i += 1;
    }
    let center = crate::decode::very_broken_op(period, i);
    for j in -half..full {
        if si >= m.ppc_shape_len {
            break;
        }
        let p = j + center;
        if p >= 0 && p < size {
            bins.push(p);
            sis.push(si);
        }
        si += 1;
    }
    (bins, sis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{dec_gains, dequant_main, Decoder};
    use crate::frame::FrameData;

    fn si_44_48() -> StreamInfo {
        StreamInfo::new(44, 48, 1).unwrap()
    }

    /// Round-trip gain: quant → decode → ближайший грид.
    #[test]
    fn gain_roundtrip() {
        let grid = gain_long_grid();
        for &t in &[0.01, 0.05, 0.2, 0.5, 1.0, 1.5] {
            let b = quant_gain_long(t);
            let g = grid[b as usize];
            // ближайший элемент грида к t
            let nearest = grid
                .iter()
                .min_by(|a, c| (**a - t).abs().partial_cmp(&(**c - t)).unwrap())
                .unwrap();
            assert!((g - nearest).abs() < 1e-12, "t={t}");
        }
    }

    /// gain×sub_gain round-trip через dec_gains.
    #[test]
    fn gain_sub_roundtrip_via_decoder() {
        let si = StreamInfo::new(44, 48, 1).unwrap();
        let targets = [0.02, 0.05, 0.1, 0.03];
        let (gain, subs) = quant_gain_sub(&targets);
        let mut fr = FrameData::new(&si, 2);
        fr.gain_bits[0] = gain;
        for (j, s) in subs.iter().enumerate() {
            fr.sub_gain_bits[j] = *s;
        }
        // dec_gains для SHORT 44_48 возвращает sub значений
        let g = dec_gains(&si, FrameType::Short, &fr);
        assert_eq!(g.len(), 16);
        for j in 0..4 {
            // сетка 256×32 произведений: точность ~10% внизу диапазона
            assert!(
                (g[j] - targets[j]).abs() < 0.1 * targets[j] + 1e-4,
                "g[{j}]={} target={}",
                g[j],
                targets[j]
            );
        }
    }

    /// LSP: quant → decode_lsp — MA-цепочка сходится к цели за несколько
    /// кадров (на первом кадре с hist=0 формат ограничивает достижимый
    /// диапазон: lsp = sum·(1−ma), суммы кодбуков ≤ ~3.1).
    #[test]
    fn lsp_quant_roundtrip() {
        let si = si_44_48();
        let info = &si;
        let m = info.mode;
        // цель — плавно растущие углы
        let target: Vec<f64> = (0..m.n_lsp)
            .map(|i| std::f64::consts::PI * (i as f64 + 0.5) / m.n_lsp as f64 * 0.9)
            .collect();
        let mut hist = vec![0.0f32; m.n_lsp];
        let mut last_err = f64::INFINITY;
        let mut last_dev = f64::INFINITY;
        for _step in 0..6 {
            let q = quant_lsp(info, &target, &hist, None);
            let mut fr = FrameData::new(info, 0);
            fr.lpc_hist_idx[0] = q.hist_idx;
            fr.lpc_idx1[0] = q.idx1;
            for (j, v) in q.idx2.iter().enumerate() {
                fr.lpc_idx2[j] = *v;
            }
            let lsp = crate::decode::decode_lsp(info, 0, &fr, &mut hist);
            let mut err = 0.0;
            let mut max_dev = 0.0f64;
            for k in 0..m.n_lsp {
                let d = (lsp[k] as f64 - target[k]).abs();
                err += d * d;
                max_dev = max_dev.max(d);
            }
            last_err = err;
            last_dev = max_dev;
        }
        // точность ограничена сеткой кодбука idx1 (шаг ~ (π/64)); после
        // адаптации MA каждый LSP — в пределах ~3 шагов сетки
        let step = std::f64::consts::PI / 64.0;
        assert!(last_dev < 3.0 * step, "max_dev={last_dev}");
        assert!(last_err < m.n_lsp as f64 * (1.5 * step).powi(2), "err={last_err}");
    }

    /// Bark: quant → dec_bark_env согласован (st воспроизводится декодером).
    #[test]
    fn bark_quant_consistent() {
        let si = StreamInfo::new(44, 48, 1).unwrap();
        let info = &si;
        let ftype = FrameType::Long;
        let fm = &info.mode.fmode[ftype.idx()];
        // случайная цель в полосах
        let mut rng = 0xBEEFu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            0.5 + ((rng >> 32) as f64 / u32::MAX as f64)
        };
        let bands: Vec<f64> = (0..fm.bark_env_size).map(|_| next()).collect();
        let mut hist = vec![0.1f64; 40];
        let d = quant_bark(info, ftype, &bands, &mut hist, None);
        assert_eq!(d.st.len(), fm.bark_env_size);

        // прогон через декодер с use_hist: st должны совпасть
        let mut fr = FrameData::new(info, 0);
        for k in 0..fm.bark_n_coef {
            fr.bark1[k] = d.indices[k];
        }
        fr.bark_use_hist[0] = u8::from(d.use_hist);
        let mut hist2 = vec![0.1f64; 40];
        let mut env = vec![0.0f64; info.mode.size];
        crate::decode::dec_bark_env(info, ftype, 0, 0, &fr, 1.0, &mut hist2, &mut env);
        // сравниваем piecewise-constant env с st
        let bark_tab = &fm.bark_tab[..fm.bark_env_size];
        let mut idx = 0usize;
        let mut pos = 0usize;
        for _i in 0..fm.bark_env_size / fm.bark_n_coef {
            for _ in 0..fm.bark_n_coef {
                let w = bark_tab[idx] as usize;
                for t in 0..w {
                    assert!((env[pos + t] - d.st[idx]).abs() < 1e-12);
                }
                pos += w;
                idx += 1;
            }
        }
    }

    /// VQ: quant → dequant — ошибка не хуже случайного кода.
    #[test]
    fn vq_quant_beats_random() {
        let si = si_44_48();
        let info = &si;
        let ftype = FrameType::Long;
        let fm = &info.mode.fmode[ftype.idx()];
        let d = info.layout.div(ftype);
        let mut rng = 0xABCDEF01u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            2.0 * ((rng >> 32) as f64 / u32::MAX as f64) - 1.0
        };
        // цель с масштабом кодбуков (равномерный шум ±sigma/2 — в диапазоне)
        let sigma = cb_pair_sigma(fm.cb0, fm.cb1);
        let mut target = vec![0.0f64; info.mode.size];
        for v in target.iter_mut() {
            *v = next() * sigma * 0.5; // next() ∈ (−1, 1]
        }
        // VQ-порядок = identity-подмена: применяю perm для честности
        let perm = crate::dsp::construct_perm(info.mode, &info.layout, ftype, 1);
        let mut flat = vec![0.0f64; info.mode.size];
        for (t, p) in perm.iter().enumerate() {
            flat[*p as usize] = target[t];
        }
        let codes = quant_vq_main(info, ftype, &flat, &perm);
        assert_eq!(codes.len(), d.n_div * 2);

        let mut fr = FrameData::new(info, 0);
        fr.main_coeffs = codes;
        let dec = Decoder::new(*info);
        let back = dequant_main(info, ftype, &fr, &crate::dsp::construct_perm(info.mode, &info.layout, ftype, 1));
        // ошибка квантования
        let mut err_q = 0.0;
        for k in 0..target.len() {
            err_q += (back[k] - target[k]).powi(2);
        }
        // случайные коды — с учётом битовой ширины каждой дивизии
        let mut fr_r = FrameData::new(info, 0);
        for i in 0..d.n_div {
            let (b0, b1) = d.bits_at(i);
            fr_r.main_coeffs[2 * i] = (next().abs() as u32 % (1 << b0)) as u8;
            fr_r.main_coeffs[2 * i + 1] = (next().abs() as u32 % (1 << b1)) as u8;
        }
        let back_r = dequant_main(info, ftype, &fr_r, &crate::dsp::construct_perm(info.mode, &info.layout, ftype, 1));
        let mut err_r = 0.0;
        for k in 0..target.len() {
            err_r += (back_r[k] - target[k]).powi(2);
        }
        assert!(err_q < 0.5 * err_r, "err_q={err_q} err_r={err_r}");
        let _ = dec;
    }

    /// PPC-гриды монотонны и в диапазоне.
    #[test]
    fn ppc_grids() {
        let si = si_44_48();
        let periods = ppc_period_grid(&si);
        assert_eq!(periods.len(), 512);
        assert!(periods.windows(2).all(|w| w[0] < w[1]));
        let gains = ppc_gain_grid(&si);
        assert_eq!(gains.len(), 128);
        assert!(gains.windows(2).all(|w| w[0] < w[1]));
        // позиции пиков ненулевые для середины грида
        let (bins, sis) = ppc_peak_positions(&si, periods[256]);
        assert!(!bins.is_empty());
        assert_eq!(bins.len(), sis.len());
        assert!(sis.last().unwrap() < &si.mode.ppc_shape_len);
    }

    /// Совместный перебор знаков + кодируемость: quant_vq_div обязан
    /// достигать глобального минимума по ВСЕМ кодируемым комбинациям
    /// (brute-force по правилам dequant_generic: знак существует только
    /// при 7-битном индексе, при 6 битах декодер всегда читает +1).
    ///
    /// Регрессия: раньше знаки выбирались независимо по знаку dot —
    /// при анти-коррелированных книгах это теряло до 4×|v|² (оптимум
    /// (+,+) при dot(v,a) < 0), а при 6-битных сторонах «выбирался»
    /// не существующий в потоке знак (реализация распадалась).
    #[test]
    fn vq_div_signs_joint_and_encodable() {
        let cb_len = 4usize;
        let mut rng = 0xFEED_FACE_CAFE_BABEu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        // анти-коррелированные книги: строка i книги 1 ≈ −c·строки книги 0
        let mut cb0 = vec![0i16; 64 * cb_len];
        let mut cb1 = vec![0i16; 64 * cb_len];
        for i in 0..64 {
            for k in 0..cb_len {
                let a = (100.0 * next()).round() as i16;
                cb0[i * cb_len + k] = a;
                cb1[i * cb_len + k] = if i == 0 && k == 0 { -3 } else { a };
            }
        }
        // цель = сумма строк (a·b сильно отрицателен ⇒ dot(v,a) может
        // быть < 0 при оптимальных (+,+))
        for &(b0, b1) in &[(7usize, 7usize), (7, 6), (6, 7), (6, 6)] {
            for t in 0..8 {
                let i0 = (t * 7) % 64;
                let j0 = (t * 13 + 5) % 64;
                let v: Vec<f64> = (0..cb_len)
                    .map(|k| {
                        cb0[i0 * cb_len + k] as f64 + cb1[j0 * cb_len + k] as f64
                            + 5.0 * next()
                    })
                    .collect();
                let (c0, c1) = CbPairCache::new(&cb0, &cb1, cb_len).search(&v, b0, b1);

                // кодируемость: при 6 битах знакового бита быть не может
                if b0 == 6 {
                    assert_eq!(c0 & 0x40, 0, "b0=6: знак не кодируется");
                }
                if b1 == 6 {
                    assert_eq!(c1 & 0x40, 0, "b1=6: знак не кодируется");
                }
                // индекс в диапазоне
                assert_eq!(c0 & 0x3F, c0 & ((1 << b0) - 1) as u8 & 0x3F);
                assert_eq!(c1 & 0x3F, c1 & ((1 << b1) - 1) as u8 & 0x3F);

                // brute-force минимум по всем кодируемым парам
                let deq = |code: u8, cb: &[i16], bits: usize| -> Vec<f64> {
                    let (sign, idx) = if bits == 7 && code & 0x40 != 0 {
                        (-1.0, (code & 0x3F) as usize)
                    } else {
                        (1.0, (code & ((1 << bits) - 1) as u8) as usize)
                    };
                    cb[idx * cb_len..][..v.len()]
                        .iter()
                        .map(|x| sign * *x as f64)
                        .collect()
                };
                let realized = |c0: u8, c1: u8| -> f64 {
                    let a = deq(c0, &cb0, b0);
                    let b = deq(c1, &cb1, b1);
                    v.iter()
                        .zip(a.iter().zip(b.iter()))
                        .map(|(x, (p, q))| (x - p - q).powi(2))
                        .sum()
                };
                let mut bf = f64::INFINITY;
                for i in 0..(1usize << b0) {
                    for j in 0..(1usize << b1) {
                        for &s0 in &[0u8, if b0 == 7 { 0x40 } else { 0 }] {
                            for &s1 in &[0u8, if b1 == 7 { 0x40 } else { 0 }] {
                                bf = bf.min(realized(i as u8 | s0, j as u8 | s1));
                            }
                        }
                    }
                }
                let got = realized(c0, c1);
                assert!(
                    got <= bf + 1e-6,
                    "b=({b0},{b1}) t={t}: получено {got:.4} > минимума {bf:.4}"
                );
            }
        }
    }
}
