//! SIMD-ядра поиска VQ: AVX2 с рантайм-диспетчеризацией (SSE2 — базис
//! x86-64, автоВЕКТОРИЗАЦИЯ scalar-путём на остальных архитектурах).
//!
//! # Дисциплина битовой точности
//!
//! Битстрим обязан совпадать с скалярным эталоном байт-в-байт, поэтому ядра
//! спроектированы так, чтобы **каждая FP-операция повторяла скалярный
//! порядок округлений**:
//!
//! * умножение и сложение — РАЗДЕЛЬНЫМИ инструкциями (без FMA-слияния:
//!   FMA округляет один раз, скаляр `a*b + c` — два);
//! * накопление сумм идёт строго в исходном порядке по k (индексу цели),
//!   векторные лейны распределяются по НЕЗАВИСИМЫМ строкам кодбуков
//!   (каждая строка — своя сумма, свой порядок);
//! * горизонтальные редукции — только для min (точная операция без
//!   округлений) и поиска первого индекса (сравнение на равенство битово
//!   детерминировано);
//! * выражения сохраняют ассоциативность скаляра: `rc + col[j] + k*r`
//!   считается как `(rc + col[j]) + (k*r)` — ровно как в замыкании
//!   скалярного скана.
//!
//! Проверка: корпус 36 файлов SHA256 не меняется (bench_corpus.py
//! --compare), см. также тест `simd_kernels_match_scalar`.

use std::sync::OnceLock;

/// Уровень SIMD-поддержки CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    /// AVX2 (4×f64): основные ядра.
    Avx2,
    /// Базис x86-64: скалярные циклы (LLVM автовекторизует SSE2 там, где
    /// реассоциация не требуется).
    Base,
}

/// Однократное определение возможностей CPU.
pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                Level::Avx2
            } else {
                Level::Base
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Level::Base
        }
    })
}

// ----------------------------------------------------------------------------
// Ядро 1: скалярные произведения строк кодбуков с целью (невзвешенный поиск)
// ----------------------------------------------------------------------------

/// `out[i] = Σ_k v[k]·cb[i·cb_len + k]` для строк `i0..i0+nrows`; порядок
/// суммирования по k — строго последовательный (как `.sum()` скаляра).
pub fn dots(cb: &[f64], cb_len: usize, i0: usize, nrows: usize, v: &[f64], out: &mut [f64]) {
    debug_assert!(v.len() <= cb_len);
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        unsafe { dots_avx2(cb, cb_len, i0, nrows, v, out) };
        return;
    }
    dots_scalar(cb, cb_len, i0, nrows, v, out);
}

fn dots_scalar(cb: &[f64], cb_len: usize, i0: usize, nrows: usize, v: &[f64], out: &mut [f64]) {
    for di in 0..nrows {
        let i = i0 + di;
        let mut acc = 0.0f64;
        for (k, &vk) in v.iter().enumerate() {
            acc += vk * cb[i * cb_len + k];
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dots_avx2(cb: &[f64], cb_len: usize, i0: usize, nrows: usize, v: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let mut di = 0usize;
    while di + 4 <= nrows {
        let base = cb.as_ptr().add((i0 + di) * cb_len);
        let p0 = base;
        let p1 = base.add(cb_len);
        let p2 = base.add(2 * cb_len);
        let p3 = base.add(3 * cb_len);
        let mut acc = _mm256_setzero_pd();
        for (k, &vk) in v.iter().enumerate() {
            let vv = _mm256_set1_pd(vk);
            let a = _mm256_set_pd(
                *p3.add(k), *p2.add(k), *p1.add(k), *p0.add(k),
            );
            acc = _mm256_add_pd(acc, _mm256_mul_pd(vv, a));
        }
        _mm256_storeu_pd(out.as_mut_ptr().add(i0 + di), acc);
        di += 4;
    }
    dots_scalar(cb, cb_len, i0 + di, nrows - di, v, out);
}

// ----------------------------------------------------------------------------
// Ядро 2: взвешенные точки и нормы строк (взвешенный поиск search_w)
// ----------------------------------------------------------------------------

/// Для строк `i0..i0+nrows` книги `cb` по активному множеству `act`:
/// `dots[i] = Σ_k wva[k]·cb[i·cb_len+act[k]]`,
/// `nrms[i] = Σ_k (wa[k]·a)·a` — с порядком округлений скалярного цикла
/// `dot += wa·v·a; nrm += wa·a·a` при предвычисленном `wva[k] = wa[k]·v[act[k]]`.
pub fn dots_w(
    cb: &[f64],
    cb_len: usize,
    i0: usize,
    nrows: usize,
    act: &[usize],
    wva: &[f64],
    wa: &[f64],
    dots_out: &mut [f64],
    nrms_out: &mut [f64],
) {
    debug_assert_eq!(act.len(), wva.len());
    debug_assert_eq!(act.len(), wa.len());
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        unsafe { dots_w_avx2(cb, cb_len, i0, nrows, act, wva, wa, dots_out, nrms_out) };
        return;
    }
    dots_w_scalar(cb, cb_len, i0, nrows, act, wva, wa, dots_out, nrms_out);
}

fn dots_w_scalar(
    cb: &[f64],
    cb_len: usize,
    i0: usize,
    nrows: usize,
    act: &[usize],
    wva: &[f64],
    wa: &[f64],
    dots_out: &mut [f64],
    nrms_out: &mut [f64],
) {
    for di in 0..nrows {
        let i = i0 + di;
        let row = &cb[i * cb_len..];
        let mut dot = 0.0f64;
        let mut nrm = 0.0f64;
        for (k, &t) in act.iter().enumerate() {
            let a = row[t];
            dot += wva[k] * a;
            nrm += (wa[k] * a) * a;
        }
        dots_out[i] = dot;
        nrms_out[i] = nrm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dots_w_avx2(
    cb: &[f64],
    cb_len: usize,
    i0: usize,
    nrows: usize,
    act: &[usize],
    wva: &[f64],
    wa: &[f64],
    dots_out: &mut [f64],
    nrms_out: &mut [f64],
) {
    use std::arch::x86_64::*;
    let mut di = 0usize;
    while di + 4 <= nrows {
        let base = cb.as_ptr().add((i0 + di) * cb_len);
        let p = [base, base.add(cb_len), base.add(2 * cb_len), base.add(3 * cb_len)];
        let mut accd = _mm256_setzero_pd();
        let mut accn = _mm256_setzero_pd();
        for (k, &t) in act.iter().enumerate() {
            let a = _mm256_set_pd(
                *p[3].add(t), *p[2].add(t), *p[1].add(t), *p[0].add(t),
            );
            accd = _mm256_add_pd(accd, _mm256_mul_pd(_mm256_set1_pd(wva[k]), a));
            let wa_a = _mm256_mul_pd(_mm256_set1_pd(wa[k]), a);
            accn = _mm256_add_pd(accn, _mm256_mul_pd(wa_a, a));
        }
        let o = i0 + di;
        _mm256_storeu_pd(dots_out.as_mut_ptr().add(o), accd);
        _mm256_storeu_pd(nrms_out.as_mut_ptr().add(o), accn);
        di += 4;
    }
    dots_w_scalar(cb, cb_len, i0 + di, nrows - di, act, wva, wa, dots_out, nrms_out);
}

// ----------------------------------------------------------------------------
// Ядро 3: кросс-аккумуляция (взвешенный поиск)
// ----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cross_accum_avx2(cross: &mut [f64], aik: f64, brow: &[f64]) {
    use std::arch::x86_64::*;
    let va = _mm256_set1_pd(aik);
    let mut j = 0usize;
    let n4 = cross.len() & !3;
    while j < n4 {
        let c = _mm256_loadu_pd(cross.as_ptr().add(j));
        let b = _mm256_loadu_pd(brow.as_ptr().add(j));
        _mm256_storeu_pd(
            cross.as_mut_ptr().add(j),
            _mm256_add_pd(c, _mm256_mul_pd(va, b)),
        );
        j += 4;
    }
    while j < cross.len() {
        cross[j] += aik * brow[j];
        j += 1;
    }
}

/// `cross[j] += aik·brow[j]` — поэлементно, порядок по j в лейнах
/// (каждый j — независимое накопление, вызовы строго по возрастанию k).
#[inline]
pub fn cross_accum(cross: &mut [f64], aik: f64, brow: &[f64]) {
    debug_assert_eq!(cross.len(), brow.len());
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        // тело — в target_feature-функции: интринсики обязаны инлайниться
        // (в обычной fn они компилируются в ВЫЗОВЫ с vzeroupper — ~20× медленнее)
        unsafe { cross_accum_avx2(cross, aik, brow) };
        return;
    }
    for (c, &b) in cross.iter_mut().zip(brow.iter()) {
        *c += aik * b;
    }
}

// ----------------------------------------------------------------------------
// Ядро 4: скан ошибки err[j] = (rc + col[j]) + (k·row[j]) → (min, первый индекс)
// ----------------------------------------------------------------------------

/// Минимум ошибки по j — семантика `fold(f64::min)` скалярного поиска
/// (min — точная операция, реассоциации не боится).
pub fn scan_min_val(col: &[f64], row: &[f64], rc: f64, k: f64) -> f64 {
    debug_assert_eq!(col.len(), row.len());
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        return unsafe { scan_min_val_avx2(col, row, rc, k) };
    }
    scan_min_val_scalar(col, row, rc, k)
}

/// ПЕРВЫЙ индекс j с err[j] ≤ m — семантика `position(|e| e <= be)`;
/// вызывается только при улучшении минимума (как в скалярном поиске).
pub fn scan_min_idx(col: &[f64], row: &[f64], rc: f64, k: f64, m: f64) -> usize {
    debug_assert_eq!(col.len(), row.len());
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        return unsafe { scan_min_idx_avx2(col, row, rc, k, m) };
    }
    scan_min_idx_scalar(col, row, rc, k, m)
}

fn scan_min_val_scalar(col: &[f64], row: &[f64], rc: f64, k: f64) -> f64 {
    let mut be = f64::INFINITY;
    for j in 0..col.len() {
        let e = rc + col[j] + k * row[j];
        if e < be {
            be = e;
        }
    }
    be
}

fn scan_min_idx_scalar(col: &[f64], row: &[f64], rc: f64, k: f64, m: f64) -> usize {
    for j in 0..col.len() {
        if rc + col[j] + k * row[j] <= m {
            return j;
        }
    }
    0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_min_val_avx2(col: &[f64], row: &[f64], rc: f64, k: f64) -> f64 {
    use std::arch::x86_64::*;
    let vrc = _mm256_set1_pd(rc);
    let vk = _mm256_set1_pd(k);
    let mut vmin = _mm256_set1_pd(f64::INFINITY);
    let n = col.len();
    let mut j = 0usize;
    let n4 = n & !3;
    while j < n4 {
        let c = _mm256_loadu_pd(col.as_ptr().add(j));
        let r = _mm256_loadu_pd(row.as_ptr().add(j));
        let e = _mm256_add_pd(_mm256_add_pd(vrc, c), _mm256_mul_pd(vk, r));
        vmin = _mm256_min_pd(vmin, e);
        j += 4;
    }
    let mut m = horiz_min(vmin);
    while j < n {
        let e = rc + col[j] + k * row[j];
        if e < m {
            m = e;
        }
        j += 1;
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_min_idx_avx2(col: &[f64], row: &[f64], rc: f64, k: f64, m: f64) -> usize {
    use std::arch::x86_64::*;
    let vrc = _mm256_set1_pd(rc);
    let vk = _mm256_set1_pd(k);
    let vm = _mm256_set1_pd(m);
    let n = col.len();
    let mut j = 0usize;
    let n4 = n & !3;
    while j < n4 {
        let c = _mm256_loadu_pd(col.as_ptr().add(j));
        let r = _mm256_loadu_pd(row.as_ptr().add(j));
        let e = _mm256_add_pd(_mm256_add_pd(vrc, c), _mm256_mul_pd(vk, r));
        let mask = _mm256_movemask_pd(_mm256_cmp_pd(e, vm, _CMP_LE_OQ));
        if mask != 0 {
            return j + mask.trailing_zeros() as usize;
        }
        j += 4;
    }
    while j < n {
        if rc + col[j] + k * row[j] <= m {
            return j;
        }
        j += 1;
    }
    0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horiz_min(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let l2 = _mm_min_pd(lo, hi);
    let s = _mm_min_sd(l2, _mm_unpackhi_pd(l2, l2));
    _mm_cvtsd_f64(s)
}

// ----------------------------------------------------------------------------
// Ядро 5: min_j (nrm[j] − 2·|d[j]|) — нижняя оценка для отсева строк
// ----------------------------------------------------------------------------

/// `min_j (nrm[j] − 2·|d[j]|)`: |d| — побитовый AND с маской знака,
/// умножение на 2 и вычитание — теми же раундами, что в скаляре.
pub fn min_transform(nrm: &[f64], d: &[f64]) -> f64 {
    debug_assert_eq!(nrm.len(), d.len());
    #[cfg(target_arch = "x86_64")]
    if level() == Level::Avx2 {
        return unsafe { min_transform_avx2(nrm, d) };
    }
    let mut m = f64::INFINITY;
    for j in 0..nrm.len() {
        let e = nrm[j] - 2.0 * d[j].abs();
        if e < m {
            m = e;
        }
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn min_transform_avx2(nrm: &[f64], d: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    let vtwo = _mm256_set1_pd(2.0);
    let mut vmin = _mm256_set1_pd(f64::INFINITY);
    let n = nrm.len();
    let mut j = 0usize;
    let n4 = n & !3;
    while j < n4 {
        let nv = _mm256_loadu_pd(nrm.as_ptr().add(j));
        let dv = _mm256_and_pd(_mm256_loadu_pd(d.as_ptr().add(j)), abs_mask);
        let e = _mm256_sub_pd(nv, _mm256_mul_pd(vtwo, dv));
        vmin = _mm256_min_pd(vmin, e);
        j += 4;
    }
    let mut m = horiz_min(vmin);
    while j < n {
        let e = nrm[j] - 2.0 * d[j].abs();
        if e < m {
            m = e;
        }
        j += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: &mut u64) -> f64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    /// Все ядра обязаны побитово совпадать с наивными скалярными циклами
    /// (порядок округлений — часть спецификации битовой точности).
    #[test]
    fn simd_kernels_match_scalar() {
        let mut rng = 0x5EED_1234_ABCD_EF01u64;
        for &(cb_len, nrows, vlen) in &[(8usize, 64usize, 8usize), (24, 64, 20), (6, 32, 5), (16, 128, 16)] {
            let cb: Vec<f64> = (0..nrows * cb_len).map(|_| (100.0 * xorshift(&mut rng)).round()).collect();
            let v: Vec<f64> = (0..vlen).map(|_| 10.0 * xorshift(&mut rng)).collect();
            // ядро 1: точки
            let mut got = vec![f64::NAN; nrows];
            let mut want = vec![f64::NAN; nrows];
            dots(&cb, cb_len, 0, nrows, &v, &mut got);
            for i in 0..nrows {
                want[i] = v.iter().zip(cb[i * cb_len..].iter()).map(|(&a, &b)| a * b).sum();
            }
            assert_eq!(got, want, "dots cb_len={cb_len}");

            // ядро 2: взвешенные точки/нормы по активному множеству
            let act: Vec<usize> = (0..vlen).step_by(2).collect();
            let wa: Vec<f64> = (0..act.len()).map(|i| 0.5 + i as f64).collect();
            let wva: Vec<f64> = act
                .iter()
                .zip(wa.iter())
                .map(|(&t, &w)| w * v[t])
                .collect();
            let mut gd = vec![f64::NAN; nrows];
            let mut gn = vec![f64::NAN; nrows];
            let mut wd = vec![f64::NAN; nrows];
            let mut wn = vec![f64::NAN; nrows];
            dots_w(&cb, cb_len, 0, nrows, &act, &wva, &wa, &mut gd, &mut gn);
            for i in 0..nrows {
                let mut dot = 0.0;
                let mut nrm = 0.0;
                for (k, &t) in act.iter().enumerate() {
                    let a = cb[i * cb_len + t];
                    dot += wa[k] * v[t] * a;
                    nrm += wa[k] * a * a;
                }
                wd[i] = dot;
                wn[i] = nrm;
            }
            assert_eq!(gd, wd, "dots_w dots cb_len={cb_len}");
            assert_eq!(gn, wn, "dots_w nrms cb_len={cb_len}");

            // ядро 3: кросс-аккумуляция
            let mut cross = vec![0.0f64; nrows];
            let brow: Vec<f64> = (0..nrows).map(|_| xorshift(&mut rng)).collect();
            for (k, &t) in act.iter().enumerate() {
                let aik = wa[k] * cb[t % nrows * cb_len + t % cb_len];
                cross_accum(&mut cross, aik, &brow);
            }
            let mut cross_ref = vec![0.0f64; nrows];
            for (k, &t) in act.iter().enumerate() {
                let aik = wa[k] * cb[t % nrows * cb_len + t % cb_len];
                for (c, &b) in cross_ref.iter_mut().zip(brow.iter()) {
                    *c += aik * b;
                }
            }
            assert_eq!(cross, cross_ref, "cross_accum cb_len={cb_len}");

            // ядро 4: скан минимума + первый индекс
            let col: Vec<f64> = (0..nrows).map(|_| 100.0 * xorshift(&mut rng)).collect();
            for &(rc, kk) in &[(10.0, 2.0), (-30.0, -2.0), (0.0, 0.0)] {
                let m = scan_min_val(&col, &cross, rc, kk);
                // эталон: fold min + position
                let be = col
                    .iter()
                    .zip(cross.iter())
                    .map(|(&c, &r)| rc + c + kk * r)
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(m.to_bits(), be.to_bits(), "scan_min val cb_len={cb_len}");
                let jj = scan_min_idx(&col, &cross, rc, kk, be);
                let jj_ref = col
                    .iter()
                    .zip(cross.iter())
                    .position(|(&c, &r)| rc + c + kk * r <= be)
                    .unwrap_or(0);
                assert_eq!(jj, jj_ref, "scan_min idx cb_len={cb_len}");
            }

            // ядро 5: min_j (nrm − 2|d|)
            let d: Vec<f64> = (0..nrows).map(|_| 10.0 * xorshift(&mut rng)).collect();
            let m = min_transform(&cross, &d);
            let m_ref = (0..nrows)
                .map(|j| cross[j] - 2.0 * d[j].abs())
                .fold(f64::INFINITY, f64::min);
            assert_eq!(m.to_bits(), m_ref.to_bits(), "min_transform cb_len={cb_len}");
        }
    }
}

