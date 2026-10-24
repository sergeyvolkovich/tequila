//! Анализ: LPC по спектральной цели, LPC→LSP-корни, оценка периода основного тона.
//!
//! Ключевое тождество (выведено из eval_lpc_spectrum FFmpeg, численно
//! проверяется тестом `lsp_identity`): огибающая декодера
//! **`env(θ) = 0.5/(p+q) = 0.5 / |A(e^{jθ})|²`**,
//! где A(z) — LPC-полином порядка n = n_lsp («+»-конвенция
//! A = 1 + a₁z⁻¹ + …), LSP-углы — корни полиномов
//! ```text
//! P(z) = A(z) + z^{-(n+1)}·A(1/z)   (корни z=−1 + e^{±jω}, ω → чётные lsp)
//! Q(z) = A(z) − z^{-(n+1)}·A(1/z)   (корни z=+1 + e^{±jω}, ω → нечётные lsp)
//! ```
//!
//! Стратегия энкодера: остаток VQ оптимален, когда flat = X/env равномерен
//! по RMS ⇒ env ≈ сглаженная |X| (амплитудная огибающая мишени) ⇒
//! |A|² ≈ |X| ⇒ автокорреляция строится по амплитудному спектру:
//! `r[m] = (2/N)·Σ_k S[k]·cos(π(k+½)m/N)`, S = |X| (сглаженная).

use crate::codec::StreamInfo;
use crate::modes::FrameType;

/// Автокорреляция из спектральной «плотности» S[k] ≥ 0 (N точек на [0, π)).
///
/// r[m] = (2/N)·Σ_k S[k]·cos(π(k+½)·m/N), m = 0..order.
pub fn autocorr_from_spectrum(s: &[f64], order: usize) -> Vec<f64> {
    let n = s.len();
    let mut r = vec![0.0f64; order + 1];
    for (m, rm) in r.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, sk) in s.iter().enumerate() {
            let th = std::f64::consts::PI * (k as f64 + 0.5) / n as f64;
            acc += sk * (th * m as f64).cos();
        }
        *rm = 2.0 * acc / n as f64;
    }
    r
}

/// Левинсон-Дурбин: r[0..=order] → LPC a[0..=order].
///
/// Конвенция: предиктор x̂[n] = Σ_{j≥1} a_j·x[n−j], т.е. полином
/// A(z) = 1 − a₁z⁻¹ − a₂z⁻² − … (a[0] = 1, знаки a[j] = AR-коэффициенты).
/// Для LSP-разложения («+»-конвенция) вызывающий должен инвертировать
/// знаки a[1..]. Возвращает None при вырожденной r (тишина).
pub fn levinson(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r[0] <= 1e-12 {
        return None;
    }
    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = if i < r.len() { r[i] } else { 0.0 };
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / err;
        let mut a_new = a.clone();
        a_new[i] = k;
        for j in 1..i {
            a_new[j] = a[j] - k * a[i - j];
        }
        a = a_new;
        err *= 1.0 - k * k;
        if err <= 0.0 || !err.is_finite() {
            return None;
        }
    }
    Some(a)
}

/// Собрать LPC-полином из вещественных полюсов (для тестов):
/// A(z) = Π(1 − m_i z⁻¹) в «+»-конвенции (a[0] = 1).
pub fn lpc_from_real_poles(mag: &[f64]) -> Vec<f64> {
    let mut a = vec![1.0f64];
    for &m in mag {
        let mut next = vec![0.0; a.len() + 1];
        for (k, ak) in a.iter().enumerate() {
            next[k] += ak;
            next[k + 1] -= ak * m;
        }
        a = next;
    }
    a
}

/// Корни P/Q на единичной окружности → LSP-углы (отсортированы, длина n).
///
/// Вещественные формы на z = e^{jθ}: P̃(θ) = Re{e^{j(n+1)θ/2}·P(e^{jθ})},
/// Q̃(θ) = −Im{...·Q(e^{jθ})}; нули ищутся сканом сетки + бисекция.
pub fn lpc_to_lsp(a: &[f64]) -> Vec<f64> {
    let n = a.len() - 1; // порядок
    debug_assert_eq!(n % 2, 0);
    let ext = |sign: f64| -> Vec<f64> {
        // c_k = a_k ± a_{n+1-k}, k = 0..=n+1 (a_j = 0 вне [0..n])
        (0..=n + 1)
            .map(|k| {
                let ak = if k <= n { a[k] } else { 0.0 };
                let am = if n + 1 - k <= n { a[n + 1 - k] } else { 0.0 };
                ak + sign * am
            })
            .collect::<Vec<_>>()
    };
    let p = ext(1.0); // симметричный: корень z=−1
    let q = ext(-1.0); // антисимметричный: корень z=+1

    let eval_real = |c: &[f64], th: f64, use_sin: bool| -> f64 {
        // e^{j(n+1)θ/2}·Σ c_k e^{−jkθ}; вещественная часть (P) или −мнимая (Q)
        let half = (n + 1) as f64 / 2.0;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, ck) in c.iter().enumerate() {
            let ph = (half - k as f64) * th;
            re += ck * ph.cos();
            im += ck * ph.sin();
        }
        if use_sin {
            -im
        } else {
            re
        }
    };

    let find_roots = |c: &[f64], use_sin: bool| -> Vec<f64> {
        // Парные коэффициенты: экспоненты (half−k) симметричны ±m (m —
        // полуцелые), поэтому
        //   re(θ) = Σ_m (c_k + c_{n+1−k})·cos(mθ),
        //   im(θ) = Σ_m (c_k − c_{n+1−k})·sin(mθ),   k < (n+2)/2.
        // Скан сетки — рекуррентными вращениями (npair·6 флопов на точку
        // вместо 44 sincos). Дрейф вращений гасится гибридно: ресинхронизация
        // каждые 8 шагов + ТОЧНЫЙ пересчёт точек с |v| < 1e-9·Σ|d| (знаки
        // в окрестностях корней всегда достоверны); бисекция — по точной
        // eval_real, бракет дополнительно проверяется точными концами.
        let npair = c.len() / 2;
        let half = (n + 1) as f64 / 2.0;
        let mut dc = vec![0.0f64; npair];
        let mut ds = vec![0.0f64; npair];
        for k in 0..npair {
            let k2 = c.len() - 1 - k;
            dc[k] = c[k] + c[k2];
            ds[k] = c[k] - c[k2];
        }
        let dsum: f64 = (if use_sin { &ds } else { &dc })
            .iter()
            .map(|v| v.abs())
            .sum::<f64>()
            .max(1e-30);
        let thresh = 1e-9 * dsum;
        let grid = 4096usize;
        let delta = std::f64::consts::PI / grid as f64;
        // шаг вращения для каждого m
        let rot: Vec<(f64, f64)> = (0..npair)
            .map(|k| {
                let m = half - k as f64;
                ((m * delta).cos(), (m * delta).sin())
            })
            .collect();
        // текущие (cos mθ, sin mθ); старт — прямое вычисление в θ = 1e-7
        let th0 = 1e-7f64;
        let mut cs: Vec<(f64, f64)> = (0..npair)
            .map(|k| {
                let m = half - k as f64;
                ((m * th0).cos(), (m * th0).sin())
            })
            .collect();
        let value = |cs: &[(f64, f64)]| -> f64 {
            if use_sin {
                let mut acc = 0.0f64;
                for k in 0..npair {
                    acc += ds[k] * cs[k].1;
                }
                -acc
            } else {
                let mut acc = 0.0f64;
                for k in 0..npair {
                    acc += dc[k] * cs[k].0;
                }
                acc
            }
        };
        let mut roots = Vec::with_capacity(n / 2 + 2);
        let mut prev_th = th0;
        let mut prev_v = eval_real(c, prev_th, use_sin);
        for g in 1..=grid {
            let mut th = std::f64::consts::PI * g as f64 / grid as f64;
            if g == grid {
                th -= 1e-9;
            }
            if g == 1 || g % 8 == 0 {
                // ресинхронизация (и неравномерный первый шаг) — прямые sincos
                for k in 0..npair {
                    let m = half - k as f64;
                    cs[k] = ((m * th).cos(), (m * th).sin());
                }
            } else {
                for k in 0..npair {
                    let (c0, s0) = cs[k];
                    let (rc, rs) = rot[k];
                    cs[k] = (c0 * rc - s0 * rs, s0 * rc + c0 * rs);
                }
            }
            let mut v = value(&cs);
            if v.abs() < thresh {
                // близко к нулю: знак решает только точная формула
                v = eval_real(c, th, use_sin);
            }
            if prev_v == 0.0 || prev_v * v < 0.0 {
                let vlo = eval_real(c, prev_th, use_sin);
                let vhi = eval_real(c, th, use_sin);
                if vlo * vhi > 0.0 {
                    // бракет от дрейфа — точные концы одного знака
                    prev_th = th;
                    prev_v = v;
                    continue;
                }
                let (mut lo, mut hi) = (prev_th, th);
                for _ in 0..80 {
                    let mid = 0.5 * (lo + hi);
                    let vm = eval_real(c, mid, use_sin);
                    if vm * vlo <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            prev_th = th;
            prev_v = v;
        }
        roots
    };

    let mut rp = find_roots(&p, false);
    let mut rq = find_roots(&q, true);
    // отбрасываем тривиальные корни (z=∓1 ↔ θ≈0/π — их вклад уже учтён
    // факторами (2∓2c) в формуле огибающей) и возможные дубли у границ
    let eps = 0.005;
    rp.retain(|t| *t > eps && *t < std::f64::consts::PI - eps);
    rq.retain(|t| *t > eps && *t < std::f64::consts::PI - eps);
    rp.sort_by(|x, y| x.partial_cmp(y).unwrap());
    rq.sort_by(|x, y| x.partial_cmp(y).unwrap());
    while rp.len() > n / 2 {
        // лишний — ближайший к границе
        if rp[0] < std::f64::consts::PI - *rp.last().unwrap() {
            rp.remove(0);
        } else {
            rp.pop();
        }
    }
    while rq.len() > n / 2 {
        if rq[0] < std::f64::consts::PI - *rq.last().unwrap() {
            rq.remove(0);
        } else {
            rq.pop();
        }
    }
    let mut all: Vec<f64> = rp.into_iter().chain(rq).collect();
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // недостающие корни вставляем в самые широкие «дыры» (стабильнее,
    // чем добивка в конец: сохраняет плотность распределения)
    while all.len() < n {
        let mut best_gap = 0.0f64;
        let mut best_idx = all.len(); // вставка в конец по умолчанию
        let mut prev = 0.0f64;
        for (i, v) in all.iter().enumerate() {
            if *v - prev > best_gap {
                best_gap = *v - prev;
                best_idx = i;
            }
            prev = *v;
        }
        if std::f64::consts::PI - prev > best_gap {
            // хвостовой интервал учтён ниже
            best_idx = all.len();
        }
        let mid = match best_idx {
            0 => all.first().map(|v| v / 2.0).unwrap_or(std::f64::consts::PI / 2.0),
            i if i >= all.len() => (all.last().unwrap() + std::f64::consts::PI) / 2.0,
            i => (all[i - 1] + all[i]) / 2.0,
        };
        all.insert(best_idx, mid);
    }
    all
}

/// Оценка периода основного тона по автокорреляции сигнала.
///
/// Возвращает (период в сэмплах, скор периодичности 0..1).
pub fn acf_pitch(x: &[f64], sr: u32, fmin: f64, fmax: f64) -> (usize, f64) {
    let n = x.len();
    if n < 4 {
        return (0, 0.0);
    }
    let mut e0 = 0.0f64;
    for v in x {
        e0 += v * v;
    }
    if e0 < 1e-12 {
        return (0, 0.0);
    }
    let lag_min = ((sr as f64 / fmax).floor() as usize).max(2);
    let lag_max = ((sr as f64 / fmin).ceil() as usize).min(n / 2);
    if lag_max <= lag_min {
        return (0, 0.0);
    }
    let mut best = (0usize, 0.0f64);
    for lag in lag_min..=lag_max {
        let mut acc = 0.0f64;
        for i in 0..n - lag {
            acc += x[i] * x[i + lag];
        }
        let score = acc / e0;
        if score > best.1 {
            best = (lag, score);
        }
    }
    best
}

/// Целевая «спектральная плотность» для LPC-анализа кадра: сглаженная |X|
/// (амплитудная огибающая ⇒ 1/|A|² ≈ |X| ⇒ env ≈ |X| — плоский VQ-остаток).
///
/// `spectra` — ПОДПИСАННЫЕ X подкадров (все на одной частотной сетке,
/// длина B): abs берётся здесь; усредняются |X| (декодер применяет одну
/// env ко всем подкадрам). Без abs сглаживание гасит знакопеременный
/// спектр в шум — огибающая схлопывается и перестаёт нести глубокую
/// форму (bark-слой имеет лишь ~10 дБ динамики: st ≥ 1.1).
pub fn lpc_target_spectrum(spectra: &[&[f64]], smooth: usize) -> Vec<f64> {
    let n = spectra[0].len();
    let mut s = vec![0.0f64; n];
    for sp in spectra {
        for k in 0..n {
            s[k] += sp[k].abs();
        }
    }
    let m = spectra.len() as f64;
    for v in s.iter_mut() {
        *v /= m;
    }
    // сглаживание скользящим средним (нечётное окно)
    let w = smooth | 1;
    let half = w / 2;
    let mut out = vec![0.0f64; n];
    let mut acc = 0.0;
    for k in 0..n {
        acc += s[k];
        if k >= w {
            acc -= s[k - w];
        }
        let cnt = (k + 1).min(w);
        if k >= half {
            out[k - half] = acc / cnt as f64;
        }
    }
    // хвост
    let mut acc2 = 0.0;
    let mut cnt2 = 0;
    for k in n.saturating_sub(w)..n {
        acc2 += s[k];
        cnt2 += 1;
    }
    let tail = acc2 / cnt2.max(1) as f64;
    for k in n - half..n {
        out[k] = tail;
    }
    out
}

/// Полный LSP-анализ канала: спектры подкадров → целевые LSP-углы.
pub fn analyze_lsp(info: &StreamInfo, ftype: FrameType, spectra: &[&[f64]]) -> Vec<f64> {
    let n_lsp = info.mode.n_lsp;
    let b = info.mode.size / info.mode.fmode[ftype.idx()].sub;
    let s = lpc_target_spectrum(spectra, (b / 64).max(3));
    // LPC по амплитудному спектру (см. модульный комментарий);
    // levinson — «−»-конвенция, lpc_to_lsp — «+» → инверсия знаков
    let r = autocorr_from_spectrum(&s, n_lsp);
    match levinson(&r, n_lsp) {
        Some(a) => {
            // расширение полос полюсов (bandwidth expansion, γ = 0.994):
            // отодвигает полюса от единичной окружности — страхует от
            // вырожденных lpc_env (полюса 0.5/(p+q) амплифицируют
            // численные расхождения на порядки, ARCHITECTURE §9)
            let gamma = 0.994f64;
            let a_plus: Vec<f64> = std::iter::once(1.0)
                .chain(
                    a[1..]
                        .iter()
                        .enumerate()
                        .map(|(j, v)| -v * gamma.powi(j as i32 + 1)),
                )
                .collect();
            lpc_to_lsp(&a_plus)
        }
        None => {
            // тишина/вырожденность: равномерные LSP (плоская env)
            (0..n_lsp)
                .map(|i| std::f64::consts::PI * (i as f64 + 0.5) / n_lsp as f64)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::dec_lpc_env;

    /// Тождество env_ffmpeg = 0.5/|A|²: численная проверка на
    /// случайном устойчивом A (корни P/Q → lsp → формула FFmpeg).
    #[test]
    fn lsp_identity() {
        // устойчивый A: корни внутри круга радиуса 0.9
        let orders = [12usize, 16, 20];
        for &order in &orders {
            let mut rng = 0x243F_6A88_85A3_08D3u64;
            let mut mag = vec![0.0f64; order];
            for m in mag.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                *m = 0.55 + 0.3 * ((rng >> 32) as f64 / u32::MAX as f64);
            }
            // A(z) = Π(1 − m_i z⁻¹) — устойчив; эталон |A| считаем по
            // корням (прямой полином у θ≈0 страдает катастрофическим
            // погашением: члены ~1e5, результат ~1e−11)
            let a = lpc_from_real_poles(&mag);
            let lsp = lpc_to_lsp(&a);
            assert_eq!(lsp.len(), order, "число LSP");
            // монотонность
            for i in 1..lsp.len() {
                assert!(lsp[i] > lsp[i - 1], "LSP не монотонны: {:?}", lsp);
            }

            // env(θ) через ffmpeg-формулу (f32 в decode; здесь сверяем в f64)
            let n = order;
            let eval_env = |c: f64| -> f64 {
                let q: Vec<f64> = lsp.iter().map(|w| 2.0 * w.cos()).collect();
                let tc = 2.0 * c;
                let mut p = 0.5f64;
                let mut qq = 0.5f64;
                let mut j = 0;
                while j + 1 < n {
                    qq *= q[j] - tc;
                    p *= q[j + 1] - tc;
                    qq *= q[j + 2] - tc;
                    p *= q[j + 3] - tc;
                    j += 4;
                }
                p *= p * (2.0 - tc);
                qq *= qq * (2.0 + tc);
                0.5 / (p + qq)
            };
            // стабильный |A| через корни: Π|1 − m_i·e^{jθ}|
            let eval_a_root = |th: f64| -> f64 {
                let mut v = 1.0f64;
                for &m in &mag {
                    v *= (1.0 + m * m - 2.0 * m * th.cos()).sqrt();
                }
                v
            };
            for g in 1..64 {
                let th = std::f64::consts::PI * g as f64 / 64.0;
                let env = eval_env(th.cos());
                let ia = eval_a_root(th);
                let expect = 0.5 / (ia * ia);
                let rel = (env - expect).abs() / expect;
                // у краёв θ точность корней ограничена погашением в P̃/Q̃
                // (члены ~1e5, значения ~1e-11 на вырожденных полиномнах
                // высоких порядков); шаг LSP-квантования (~1e-2) грубее
                // этих ошибок на порядки
                let tol = if th < std::f64::consts::PI / 8.0
                    || th > 7.0 * std::f64::consts::PI / 8.0
                {
                    0.05
                } else {
                    1e-3
                };
                assert!(rel < tol, "θ={th}: env={env} expect={expect} rel={rel}");
            }
        }
    }

    /// Левинсон: A корректно обращает AR-процесс (ошибка предсказания < 1).
    #[test]
    fn levinson_on_ar_process() {
        // AR(4) с известными коэффициентами: генерируем процесс, строим r, Левинсон
        let n = 20000;
        let mut x = vec![0.0f64; n];
        let mut rng = 0x1319_8A2E_0370_7344u64;
        let mut noise = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for i in 4..n {
            x[i] = noise()
                + 0.9 * x[i - 1]
                - 0.4 * x[i - 2]
                + 0.2 * x[i - 3]
                - 0.1 * x[i - 4];
        }
        let order = 4;
        let mut r = vec![0.0f64; order + 1];
        for m in 0..=order {
            let mut acc = 0.0;
            for i in order..n {
                acc += x[i] * x[i - m];
            }
            r[m] = acc / (n - order) as f64;
        }
        let a = levinson(&r, order).unwrap();
        // конвенция: a[k] = AR-коэффициенты генератора
        let expect = [1.0, 0.9, -0.4, 0.2, -0.1];
        for k in 0..=order {
            assert!(
                (a[k] - expect[k]).abs() < 0.05,
                "a[{k}]={:.4} vs {:.4}",
                a[k],
                expect[k]
            );
        }
        // «+»-конвенция для LSP: знаки инвертируются
        let a_plus: Vec<f64> = std::iter::once(1.0).chain(a[1..].iter().map(|v| -v)).collect();
        let lsp = lpc_to_lsp(&a_plus);
        assert_eq!(lsp.len(), order);
        for i in 1..lsp.len() {
            assert!(lsp[i] > lsp[i - 1]);
        }
    }

    /// ACF-оценка периода: синус 250 Гц @ 8 кГц → lag 32.
    #[test]
    fn acf_pitch_sine() {
        let sr = 8000;
        let x: Vec<f64> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * 250.0 * i as f64 / sr as f64).sin())
            .collect();
        let (lag, score) = acf_pitch(&x, sr, 100.0, 600.0);
        assert_eq!(lag, 32, "lag={lag}");
        assert!(score > 0.9, "score={score}");
        // белый шум → низкий скор
        let mut rng = 0xDEAD_BEEF_CAFE_1234u64;
        let w: Vec<f64> = (0..2000)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let (_, wscore) = acf_pitch(&w, sr, 100.0, 600.0);
        assert!(wscore < 0.3, "wscore={wscore}");
    }

    /// Полоса-ограниченный спектр (знакопеременный!): env обязана нести
    /// глубокую форму — провал в стоп-зоне ≥ 40 дБ (реально ~70 дБ).
    /// Регрессия: без abs в lpc_target_spectrum сглаживание гасило
    /// знакопеременный X, env схлопывалась до ~25 дБ динамики — «пустые»
    /// зоны звучали плоским шипением на 40–65 дБ выше цели.
    #[test]
    fn lpc_envelope_covers_deep_spectral_shape() {
        let info = crate::codec::StreamInfo::new(44, 48, 1).unwrap();
        let b = 2048usize;
        let mut rng = 0xC0FF_EE01_2345_6789u64;
        let mut noise = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        // полоса [0, 200) бин: сильные знакопеременные компоненты ~1e3;
        // выше — пол в 1e-1 (диапазон входа 80 дБ)
        let mut spec = vec![0.0f64; b];
        for (k, v) in spec.iter_mut().enumerate() {
            *v = if k < 200 {
                1e3 * (std::f64::consts::PI * k as f64 * 0.37).cos()
            } else {
                1e-1 * noise()
            };
        }
        let lsp = analyze_lsp(&info, FrameType::Long, &[&spec]);
        let env = dec_lpc_env(
            &info,
            FrameType::Long,
            &lsp.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        );
        let pass: f64 = env[..200].iter().sum::<f32>() as f64 / 200.0;
        let stop: f64 = env[400..].iter().sum::<f32>() as f64 / (env.len() - 400) as f64;
        let ratio_db = 10.0 * (stop / pass).log10();
        eprintln!("LPC-огибающая: полоса={pass:.3}, стоп={stop:.6}, провал={ratio_db:.1} дБ");
        assert!(
            ratio_db < -40.0,
            "env не несёт форму спектра: провал всего {ratio_db:.1} дБ (нужно ≤ −40)"
        );
    }

    /// analyze_lsp: для белого шума — LSP заполняют [0, π) монотонно.
    #[test]
    fn analyze_lsp_white_noise() {
        let info = crate::codec::StreamInfo::new(44, 48, 1).unwrap();
        let mut rng = 0x5DEE_CE66_D00D_FEEFu64;
        let spec: Vec<f64> = (0..2048)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                100.0 + 50.0 * ((rng >> 32) as f64 / u32::MAX as f64)
            })
            .collect();
        let lsp = analyze_lsp(&info, FrameType::Long, &[&spec]);
        assert_eq!(lsp.len(), 20);
        for i in 1..lsp.len() {
            assert!(lsp[i] > lsp[i - 1]);
        }
        assert!(lsp[0] > 0.0 && *lsp.last().unwrap() < std::f64::consts::PI);
        // env от этих LSP не вырождена (полюса конечны)
        let env = dec_lpc_env(&info, FrameType::Long, &lsp.iter().map(|v| *v as f32).collect::<Vec<_>>());
        assert!(env.iter().all(|v| v.is_finite() && *v > 0.0));
    }
}
