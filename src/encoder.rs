//! Энкодер TwinVQ: полный конвейер кадра (ARCHITECTURE.md §9).
//!
//! ```text
//! 1. мишень X_j — ТОЧНЫЙ АДЪЮНКТ синтеза декодера (imdct_and_window):
//!    Ō-буфер «желаемых значений» curr (линейная часть + фолд через
//!    бабочку кадра k+1), по подкадрам: бабочка / плоская часть /
//!    P-часть следующей бабочки; X_j = c·mdct_analysis_compact(y_j).
//!    PR-SNR ~247 дБ при масштабе 1.0 (тест analysis_synthesis_pr).
//! 2. LPC-анализ по |X| (плоский VQ-остаток) → LSP → split-VQ+MA
//!    → точная реплика decode_lsp → lpc_env
//! 3. flat = X / lpc_env
//! 4. (LONG) PPC: ACF f0 → период; (g_coef × shape) жадный VQ;
//!    W = flat − PPC_synth (точная реплика decode_ppc)
//! 5. bark: RMS(W) по полосам → VQ (use_hist, MA) → st
//! 6. gain/sub_gain: mulaw-гриды; σ-цель = κ·σ(книг)
//! 7. R = W/(st·g) → interleave → конъюгатный VQ
//! 8. FrameData → serialize
//! ```
//!
//! wtype-грамматика NTT (E3): атака → [L1, S2×n, L3], стационар → L0,
//! тишина → L0; старт ненулевого потока → [1, 2, 3, 0…].

use crate::analysis::analyze_lsp;
use crate::codec::StreamInfo;
use crate::decode::{
    dec_gains, dec_lpc_env, decode_lsp, decode_ppc, dequant_ppc_shape,
};
use crate::dsp::{construct_perm, mdct_analysis_compact, sine_window};
use crate::frame::FrameData;
use crate::modes::{FrameType, WTYPE_TO_WSIZE_IDX};
use crate::psycho::{PsyModel, PsyParams, PsyState};
use crate::quantize::{
    cb_pair_sigma, ppc_gain_grid, ppc_peak_positions, ppc_period_grid, quant_bark,
    quant_gain_long, quant_gain_sub, quant_lsp, quant_vq_main_w,
    quant_vq_ppc_cached, BarkDecision, CbPairCache,
};

/// Порог периодичности ACF для включения PPC (не используется спектральным
/// поиском — оставлен для документации исходной эвристики).
pub const PPC_PERIODICITY_MIN: f64 = 0.35;

/// Масштаб цели PPC-формы: цель не выше KAPPA·σ_shape — верхняя граница
/// естественной величины пары кодбуков (|s0·a + s1·b| ≈ √2·σ).
pub const PPC_SHAPE_KAPPA: f64 = 1.5;

/// Минимальная доля энергии кадра, которую гребёнка обязана чисто изъять
/// (покрытие пиков минус их ошибка квантования), чтобы PPC включился.
/// NTT-эталоны держат PPC на ~50% LONG-кадров почти на любом сигнале.
pub const PPC_MIN_REMOVAL_FRAC: f64 = 0.02;

/// Опции энкодера.
#[derive(Clone, Copy)]
pub struct EncoderOpts {
    /// Использовать PPC-гребёнку (LONG-кадры).
    pub ppc_enabled: bool,
    /// Множитель «сладкой зоны» VQ: целевой RMS остатка = κ·σ(книг).
    pub kappa: f64,
    /// Порог атаки для wtype-грамматики (отношение энергий кадров).
    pub attack_ratio: f64,
    /// Порог тишины (мощность на сэмпл).
    pub silence_power: f64,
    /// Число потоков кодирования: 0 — авто (available_parallelism
    /// либо TEQUILA_THREADS), 1 — строго однопоточно, N — пул из
    /// N−1 воркеров + поток-инициатор. Битстрим от числа потоков НЕ
    /// зависит: параллелятся только независимые индексы с записью в
    /// свои слоты (parallel::Pool::map_into), FP-порядок внутри
    /// индекса — скалярный.
    pub threads: usize,
    /// Психоакустическая модель (режим `--psy`, docs/PSY.md): перцептивные
    /// веса ошибки VQ/bark внутри фиксированного бюджета кадра.
    /// `None` — обычный режим (битстрим байт-в-байт как без модели).
    pub psy: Option<PsyParams>,
}

impl Default for EncoderOpts {
    fn default() -> Self {
        EncoderOpts {
            ppc_enabled: true,
            kappa: 1.0,
            attack_ratio: 4.0,
            silence_power: 1e-11,
            threads: 0,
            psy: None,
        }
    }
}

/// Квантованный кадр + диагностика.
pub struct FrameEncoding {
    pub frame: FrameData,
    /// Мишень MDCT (для анализа/тестов), [ch][size].
    pub target: Vec<Vec<f64>>,
}

/// Энкодер с состояниями квантователей (зеркально декодеру).
pub struct Encoder {
    pub info: StreamInfo,
    pub opts: EncoderOpts,
    lsp_hist: Vec<Vec<f32>>,
    bark_hist: Vec<Vec<f64>>,
    perm: [Vec<u16>; 4],
    sigma: [f64; 3],
    /// Кэши пар кодбуков VQ: [Short, Medium, Long] + PPC — кросс-члены
    /// и нормы книг не пересчитываются между кадрами (было ~17 млн
    /// MAC/кадр в 44_48 стерео).
    vq: [CbPairCache; 3],
    vq_ppc: CbPairCache,
    /// Режимозависимые гриды PPC и σ PPC-книг (кэш на весь поток).
    ppc_periods: Vec<i64>,
    ppc_gains: Vec<f64>,
    ppc_sigma: f64,
    /// Геометрия пиков add_peak по индексу грида периодов (зависит только
    /// от режима; рассчитывается один раз на поток — раньше 512 раз
    /// на каждый канал каждого LONG-кадра).
    ppc_peaks: Vec<(Vec<i64>, Vec<usize>)>,
    /// Пул воркеров для многопоточного кодирования батчей.
    pool: crate::parallel::Pool,
    /// Психоакустическая модель + состояние post-masking на канал
    /// (режим `--psy`). Коммит состояний — в порядке кадров (Phase B
    /// последовательна), поэтому временная маскировка корректна.
    psy: Option<(PsyModel, Vec<PsyState>)>,
    /// Инструментация стадий (сек, накопительно; вывод по TEQUILA_PROFILE=1).
    prof: [f64; 6],
}

impl Encoder {
    pub fn new(info: StreamInfo, opts: EncoderOpts) -> Self {
        let threads = match opts.threads {
            0 => std::env::var("TEQUILA_THREADS")
                .ok()
                .and_then(|t| t.parse().ok())
                .filter(|&t| t > 0)
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                }),
            t => t,
        };
        let pool = crate::parallel::Pool::new(threads.saturating_sub(1));
        let sigma = [
            cb_pair_sigma(info.mode.fmode[0].cb0, info.mode.fmode[0].cb1),
            cb_pair_sigma(info.mode.fmode[1].cb0, info.mode.fmode[1].cb1),
            cb_pair_sigma(info.mode.fmode[2].cb0, info.mode.fmode[2].cb1),
        ];
        let ppc_n_div = info.layout.ppc.n_div;
        let cb_len_p = (ppc_n_div + info.mode.ppc_shape_len * info.n_ch - 1) / ppc_n_div;
        let (ppc_cb0, ppc_cb1) = info.mode.ppc_shape_cb.split_at(cb_len_p * 64);
        let psy = opts.psy.map(|p| {
            let model = PsyModel::new(&info, p);
            let states = (0..info.n_ch).map(|_| model.new_state()).collect();
            (model, states)
        });
        Encoder {
            lsp_hist: vec![vec![0.0f32; info.mode.n_lsp]; info.n_ch],
            bark_hist: vec![vec![0.1f64; 40]; 6],
            perm: FrameType::ALL
                .iter()
                .map(|&ft| construct_perm(info.mode, &info.layout, ft, info.n_ch))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            vq: std::array::from_fn(|i| {
                let fm = &info.mode.fmode[i];
                let mut c = CbPairCache::new(fm.cb0, fm.cb1, fm.cb_len_read);
                c.prewarm(info.layout.div(FrameType::ALL[i]));
                c
            }),
            vq_ppc: {
                let mut c = CbPairCache::new(ppc_cb0, ppc_cb1, cb_len_p);
                c.prewarm(&info.layout.ppc);
                c
            },
            ppc_periods: ppc_period_grid(&info),
            ppc_gains: ppc_gain_grid(&info),
            ppc_sigma: cb_pair_sigma(ppc_cb0, ppc_cb1),
            ppc_peaks: ppc_period_grid(&info)
                .into_iter()
                .map(|p| ppc_peak_positions(&info, p))
                .collect(),
            prof: [0.0; 6],
            pool,
            info,
            opts,
            sigma,
            psy,
        }
    }

    /// Пул кодирования (для батчевых фаз в encode_stream).
    pub fn pool(&self) -> &crate::parallel::Pool {
        &self.pool
    }

    /// Закодировать кадр с данным wtype по ЗАРАНЕЕ ВЫЧИСЛЕННОЙ мишени
    /// MDCT (Phase A — параллельный батчевый префетч в encode_stream).
    ///
    /// `x` — мишень [ch][size] (адъюнкт синтеза, analysis_targets).
    /// `next` — (wtype, мишень) СЛЕДУЮЩЕГО кадра: нужен только режиму
    /// `--psy` для pre-masking (lookahead); None — в конце потока.
    /// Внутри: LSP-анализ по каналам — параллельно; период PPC и
    /// грубый скан гейна — параллельно по гриду; дивизии main VQ —
    /// параллельно; MA-истории LSP/bark — строго последовательно.
    pub fn encode_frame(
        &mut self,
        wtype: u8,
        x: Vec<Vec<f64>>,
        next: Option<(u8, &Vec<Vec<f64>>)>,
    ) -> FrameEncoding {
        // локальная копия (StreamInfo: Copy) — не держит заимствование
        // self: ниже нужны &self.pool / &self.vq* параллельно с &mut hist
        let info = self.info;
        let m = info.mode;
        let ftype = info.ftype_of_wtype(wtype);
        let fm = &m.fmode[ftype.idx()];
        let sub = fm.sub;
        let b = m.size / sub;
        let size = m.size;
        let n_ch = info.n_ch;
        let is_long = ftype == FrameType::Long;

        // 1. мишень MDCT — уже вычислена (Phase A, параллельно)

        // 1.5. психоакустическая маска (--psy, PSY.md §2): порог
        //      маскировки на бин МИШЕНИ, кадром с pre-masking lookahead
        //      (следующий кадр) и commit post-masking. Считается ДО LSP:
        //      LSP-веса тоже используют маску.
        let psy_params = self.psy.as_ref().map(|(model, _)| model.params());
        let psy: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = self.psy.as_mut().map(
            |(model, states)| {
                let beta2 = 2.0 * model.params().beta;
                let mut mm = Vec::with_capacity(n_ch);
                let mut wf = Vec::with_capacity(n_ch);
                for ch in 0..n_ch {
                    let specs: Vec<&[f64]> =
                        (0..sub).map(|j| &x[ch][j * b..(j + 1) * b]).collect();
                    let next_specs = next.map(|(nw, nx)| {
                        let ft_n = info.ftype_of_wtype(nw);
                        let sub_n = m.fmode[ft_n.idx()].sub;
                        let b_n = size / sub_n;
                        let v: Vec<&[f64]> = (0..sub_n)
                            .map(|j| &nx[ch][j * b_n..(j + 1) * b_n])
                            .collect();
                        (ft_n, v)
                    });
                    let rs = model.mask_frame(ftype, &specs, next_specs, &mut states[ch]);
                    let mut mc = vec![0.0f64; size];
                    for (j, r) in rs.iter().enumerate() {
                        mc[j * b..(j + 1) * b].copy_from_slice(&r.m);
                    }
                    let wc = mc.iter().map(|&v| v.powf(-beta2)).collect();
                    mm.push(mc);
                    wf.push(wc);
                }
                (mm, wf)
            },
        );

        // 2. LSP: анализ по каналам — независимые задачи (map_into);
        //    квантование + MA-обновление — последовательно
        let _tp = std::time::Instant::now();
        let mut frame = FrameData::new(&info, wtype);
        let mut lsp_targets = vec![Vec::<f64>::new(); n_ch];
        {
            let xref = &x;
            self.pool.map_into(&mut lsp_targets, |ch| {
                let spectra: Vec<&[f64]> =
                    (0..sub).map(|j| &xref[ch][j * b..(j + 1) * b]).collect();
                analyze_lsp(&info, ftype, &spectra)
            });
        }
        let mut envs: Vec<Vec<f32>> = Vec::with_capacity(n_ch);
        for ch in 0..n_ch {
            // перцептивные веса LSP (--psy, PSY.md §7): слышимость у
            // частоты k-го LSP — уровень |X|/порог в его окрестности,
            // нормировка на среднее 1 (переключает выбор кандидатов
            // split-VQ в пользу слышимых участков огибающей)
            let lsp_beta = psy_params.map(|p| p.lsp_beta).unwrap_or(0.0);
            let lsp_w: Option<Vec<f64>> = if lsp_beta > 0.0 {
                psy.as_ref().map(|(mm, _)| {
                    let n_lsp = m.n_lsp;
                    let e = 2.0 * lsp_beta;
                    let mut w = vec![0.0f64; n_lsp];
                    for k in 0..n_lsp {
                        let w_k = lsp_targets[ch][k].clamp(1e-4, std::f64::consts::PI - 1e-4);
                        let bin = ((w_k / std::f64::consts::PI) as f64 * b as f64) as usize;
                        let lo = bin.saturating_sub(1);
                        let hi = (bin + 2).clamp(1, b);
                        let mut sl = 0.0f64;
                        let mut sm = 0.0f64;
                        let mut n = 0.0f64;
                        for j in 0..sub {
                            for i in lo..hi {
                                sl += x[ch][j * b + i] * x[ch][j * b + i];
                                sm += mm[ch][j * b + i] * mm[ch][j * b + i];
                                n += 1.0;
                            }
                        }
                        let lvl = (sl / n).sqrt().max(1e-30);
                        let mask = (sm / n).sqrt().max(1e-30);
                        w[k] = (lvl / mask).powf(e);
                    }
                    let mean: f64 = w.iter().sum::<f64>() / n_lsp as f64;
                    if mean > 0.0 {
                        for v in w.iter_mut() {
                            *v /= mean;
                        }
                    }
                    w
                })
            } else {
                None
            };
            let q = quant_lsp(&info, &lsp_targets[ch], &self.lsp_hist[ch], lsp_w.as_deref());
            frame.lpc_hist_idx[ch] = q.hist_idx;
            frame.lpc_idx1[ch] = q.idx1;
            for (j, v) in q.idx2.iter().enumerate() {
                frame.lpc_idx2[ch * m.lsp_split + j] = *v;
            }
            // точная реплика декодера (обновляет MA-историю)
            let lsp_q = decode_lsp(&info, ch, &frame, &mut self.lsp_hist[ch]);
            envs.push(dec_lpc_env(&info, ftype, &lsp_q));
        }

        // 3. flat = X/env
        self.prof[1] += _tp.elapsed().as_secs_f64();
        let mut flat = vec![vec![0.0f64; size]; n_ch];
        for ch in 0..n_ch {
            for j in 0..sub {
                for i in 0..b {
                    let e = envs[ch][i].max(1e-7) as f64;
                    flat[ch][j * b + i] = x[ch][j * b + i] / e;
                }
            }
        }

        // 4. PPC (LONG): поиск (p, g), кодирование формы — СОВМЕСТНОЕ
        //    (интерлив-вектор n_ch·shape_len одним VQ).
        //    Стерео — ОБЩАЯ гребёнка (R17): период один на пару каналов
        //    (покрытие в L-домене c0+c1 — M/S сохраняет общий компонент
        //    в обоих каналах), гейны поканальные, приёмка ПАРЫ в
        //    L-домене — как NTT (corr гребёнок 0.78, ~83% энергии L).
        //    Моно — без изменений (поканальный поиск/приёмка).
        let mut w = flat.clone();
        let _tp = std::time::Instant::now();
        if is_long && self.opts.ppc_enabled {
            let mut sel: Vec<Option<PpcSel>> = Vec::with_capacity(n_ch);
            if n_ch == 2 {
                sel = self.ppc_search_stereo(&flat);
            } else {
                sel.push(self.ppc_search_ch(0, &flat[0]));
            }
            // итеративная приёмка: кодируем форму выживших каналов вместе,
            // канал, чья гребёнка не изымает энергию на СОВМЕСТНОЙ
            // реконструкции, отключаем и перекодируем
            let shape_len = info.mode.ppc_shape_len;
            loop {
                let mut target_flat = vec![0.0f64; n_ch * shape_len];
                for (ch, s) in sel.iter().enumerate() {
                    if let Some(s) = s {
                        for &(si, v) in &s.entries {
                            target_flat[ch * shape_len + si] = v;
                        }
                    }
                }
                let codes = quant_vq_ppc_cached(
                    &self.vq_ppc,
                    &info.layout.ppc,
                    &target_flat,
                    &self.perm[FrameType::Ppc.idx()],
                );
                // реконструкция формы и спектров каналов
                let mut fr_q = FrameData::new(&info, 0);
                fr_q.ppc_coeffs = codes.clone();
                let shape =
                    dequant_ppc_shape(&info, &fr_q, &self.perm[FrameType::Ppc.idx()]);
                let mut specs = vec![vec![0.0f64; size]; n_ch];
                let mut changed = false;
                for ch in 0..n_ch {
                    let Some(s) = &sel[ch] else {
                        continue;
                    };
                    let mut fr2 = FrameData::new(&info, 0);
                    fr2.p_coef[ch] = s.p_coef;
                    fr2.g_coef[ch] = s.g_coef;
                    fr2.ppc_coeffs = codes.clone();
                    decode_ppc(&info, ch, &fr2, &shape, &mut specs[ch]);
                }
                // приёмка на совместной реконструкции:
                //  стерео — ПАРА в L-домене (общая гребёнка живёт или
                //  умирает целиком; L-домен обязан чисто изымать ≥ ε
                //  энергии, R-домен — страховка от сильного вреда);
                //  моно — поканальный критерий (как раньше)
                let dbg = std::env::var("TEQUILA_PPC_DEBUG").is_ok();
                if n_ch == 2 && sel.iter().any(|s| s.is_some()) {
                    let bins: &[i64] = sel[0]
                        .as_ref()
                        .or_else(|| sel[1].as_ref())
                        .map(|s| s.bins.as_slice())
                        .unwrap_or(&[]);
                    let mut base_l = 0.0f64;
                    let mut err_l = 0.0f64;
                    let mut base_r = 0.0f64;
                    let mut err_r = 0.0f64;
                    let mut energy_l = 0.0f64;
                    let mut energy_r = 0.0f64;
                    for t in 0..size {
                        let fl = flat[0][t] + flat[1][t];
                        let fr = flat[0][t] - flat[1][t];
                        energy_l += fl * fl;
                        energy_r += fr * fr;
                    }
                    for &bp in bins {
                        let b = bp as usize;
                        if b >= size {
                            continue;
                        }
                        let fl = flat[0][b] + flat[1][b];
                        let fr = flat[0][b] - flat[1][b];
                        let sl = specs[0][b] + specs[1][b];
                        let sr = specs[0][b] - specs[1][b];
                        base_l += fl * fl;
                        err_l += (fl - sl) * (fl - sl);
                        base_r += fr * fr;
                        err_r += (fr - sr) * (fr - sr);
                    }
                    let keep_l = base_l - err_l >= PPC_MIN_REMOVAL_FRAC * energy_l;
                    let keep_r = base_r - err_r >= -PPC_MIN_REMOVAL_FRAC * energy_r;
                    if !(keep_l && keep_r) {
                        if dbg {
                            eprintln!(
                                "ppc k={} stereo KILL: L base={:.3e} err={:.3e} E={:.3e} \
                                 R base={:.3e} err={:.3e} E={:.3e}",
                                self.prof[5] as usize, base_l, err_l, energy_l,
                                base_r, err_r, energy_r
                            );
                        }
                        sel[0] = None;
                        sel[1] = None;
                        changed = true;
                    }
                } else {
                    for ch in 0..n_ch {
                        let Some(s) = &sel[ch] else {
                            continue;
                        };
                        // приёмка на совместной реконструкции
                        let base: f64 =
                            s.bins.iter().map(|&bp| flat[ch][bp as usize].powi(2)).sum();
                        let mut err = 0.0;
                        for &bp in &s.bins {
                            let d = flat[ch][bp as usize] - specs[ch][bp as usize];
                            err += d * d;
                        }
                        if base - err < PPC_MIN_REMOVAL_FRAC * s.energy {
                            if dbg {
                                eprintln!(
                                    "ppc k={} ch={}: KILL base={:.3e} err={:.3e} energy={:.3e} \
                                     p={} g={}",
                                    self.prof[5] as usize, ch, base, err, s.energy,
                                    s.p_coef, s.g_coef
                                );
                            }
                            sel[ch] = None;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    // зафиксировать: коды + p/g + вычитание из W
                    frame.ppc_coeffs = codes;
                    for ch in 0..n_ch {
                        if let Some(s) = &sel[ch] {
                            frame.p_coef[ch] = s.p_coef;
                            frame.g_coef[ch] = s.g_coef;
                            for (a, b2) in w[ch].iter_mut().zip(specs[ch].iter()) {
                                *a -= b2;
                            }
                        }
                    }
                    break;
                }
                if sel.iter().all(|s| s.is_none()) {
                    frame.ppc_coeffs = vec![0; info.layout.ppc.n_div * 2];
                    break;
                }
            }
        }

        // 5-6. bark + gain per subframe
        self.prof[2] += _tp.elapsed().as_secs_f64();
        let sigma_target = self.opts.kappa * self.sigma[ftype.idx()];
        let mut bark_dec: Vec<Vec<BarkDecision>> = Vec::with_capacity(n_ch);
        let mut g_des: Vec<f64> = Vec::with_capacity(n_ch * sub);
        for ch in 0..n_ch {
            let mut ch_dec = Vec::with_capacity(sub);
            for j in 0..sub {
                // целевые уровни полос: RMS |W| в полосе
                let mut bands = vec![0.0f64; fm.bark_env_size];
                {
                    let mut idx = 0usize;
                    let mut pos = 0usize;
                    let bark_tab = &fm.bark_tab[..fm.bark_env_size];
                    for band in bands.iter_mut() {
                        let wdt = bark_tab[idx] as usize;
                        let mut s = 0.0;
                        for t in 0..wdt {
                            let v = w[ch][j * b + pos + t];
                            s += v * v;
                        }
                        *band = (s / wdt as f64).sqrt();
                        pos += wdt;
                        idx += 1;
                    }
                }
                let hist = &mut self.bark_hist[ftype.idx() * 2 + ch];
                // перцептивные веса полос (--psy, PSY.md §3): слышимость
                // ошибки st ∝ (env_rms/порог)^{2·β_bark}; β_bark=0 —
                // равные веса (bark-слой как в базовом режиме).
                // Нормировка на 1 — argmin не меняет, численность бережёт.
                let bark_beta = psy_params.map(|p| p.bark_beta).unwrap_or(0.0);
                let bark_w: Option<Vec<f64>> = if bark_beta > 0.0 {
                    psy.as_ref().map(|(mm, _)| {
                        let e = 2.0 * bark_beta;
                        let mut w = Vec::with_capacity(fm.bark_env_size);
                        let mut wmax = 0.0f64;
                        let mut pos = 0usize;
                        for &w_dt in fm.bark_tab[..fm.bark_env_size].iter() {
                            let n = w_dt as usize;
                            let mut se = 0.0f64;
                            let mut sm2 = 0.0f64;
                            for t in 0..n {
                                let ev = envs[ch][pos + t] as f64;
                                se += ev * ev;
                                let mv = mm[ch][j * b + pos + t];
                                sm2 += mv * mv;
                            }
                            let m_eff = (sm2 / n as f64).sqrt().max(1e-30);
                            let v = ((se / n as f64).sqrt() / m_eff).powf(e);
                            wmax = wmax.max(v);
                            w.push(v);
                            pos += n;
                        }
                        for v in w.iter_mut() {
                            *v /= wmax;
                        }
                        w
                    })
                } else {
                    None
                };
                let dec = quant_bark(&info, ftype, &bands, hist, bark_w.as_deref());
                // масштаб: RMS(W/st) подкадра → σ-цель. Режим --psy с
                // gain_weighted (PSY.md §7, адаптивный κ): RMS считается
                // с перцептивными весами a=(env·st/m)^{2β} — сладкая зона
                // кодбуков выравнивается по СЛЫШИМОЙ части остатка
                // (замаскированные «горбы» от sloppy-st не уводят gain)
                let gain_w = psy_params.map(|p| p.gain_weighted).unwrap_or(false);
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                {
                    let beta_g = psy_params.map(|p| p.beta).unwrap_or(0.0);
                    let mut pos = 0usize;
                    for &w_dt in fm.bark_tab[..fm.bark_env_size].iter() {
                        let band = pos_to_band(&fm.bark_tab[..fm.bark_env_size], pos);
                        let stv = dec.st[band].max(1e-4);
                        for t in 0..w_dt as usize {
                            let bin = j * b + pos + t;
                            let v = w[ch][bin] / stv;
                            let a = if gain_w {
                                match &psy {
                                    Some((mm, _)) => {
                                        let sc = envs[ch][pos + t] as f64 * stv;
                                        let mv = mm[ch][bin].max(1e-30);
                                        ((sc / mv).abs()).powf(2.0 * beta_g)
                                    }
                                    None => 1.0,
                                }
                            } else {
                                1.0
                            };
                            num += a * v * v;
                            den += a;
                        }
                        pos += w_dt as usize;
                    }
                }
                let rms = (num / den.max(1.0)).sqrt();
                g_des.push(rms / sigma_target);
                ch_dec.push(dec);
            }
            bark_dec.push(ch_dec);
        }

        // заполнение bark-полей + gain
        for ch in 0..n_ch {
            for j in 0..sub {
                let d = &bark_dec[ch][j];
                for k2 in 0..fm.bark_n_coef {
                    frame.bark1[(ch * sub + j) * fm.bark_n_coef + k2] = d.indices[k2];
                }
                frame.bark_use_hist[ch * sub + j] = u8::from(d.use_hist);
            }
        }
        if is_long {
            for ch in 0..n_ch {
                frame.gain_bits[ch] = quant_gain_long(g_des[ch]);
            }
        } else {
            // каналы подряд: gain затем sub_gain каждого
            let mut targets = Vec::with_capacity(n_ch * sub);
            for ch in 0..n_ch {
                targets.extend_from_slice(&g_des[ch * sub..(ch + 1) * sub]);
            }
            let (gain, subs) = quant_gain_sub(&targets);
            for ch in 0..n_ch {
                frame.gain_bits[ch] = gain;
                for j in 0..sub {
                    frame.sub_gain_bits[ch * sub + j] = subs[ch * sub + j];
                }
            }
        }

        // 7. VQ-цель: R = W/(st·g_actual); веса = (env·st·g)^2 — метрика
        //    в домене выхода (см. CbPairCache::search_w)
        self.prof[3] += _tp.elapsed().as_secs_f64();
        let gains_actual = dec_gains(&info, ftype, &frame);
        let _tp2 = std::time::Instant::now();
        let mut flat_vq = vec![0.0f64; n_ch * size];
        let mut w_vq = vec![0.0f64; n_ch * size];
        for ch in 0..n_ch {
            for j in 0..sub {
                let g = if is_long {
                    gains_actual[ch]
                } else {
                    gains_actual[sub * ch + j]
                };
                let st = &bark_dec[ch][j].st;
                let mut pos = 0usize;
                let bark_tab = &fm.bark_tab[..fm.bark_env_size];
                for (band_idx, &w_dt) in bark_tab.iter().enumerate() {
                    let env_v = st[band_idx].max(1e-4) * g.max(1e-12);
                    for t in 0..w_dt as usize {
                        let bin = j * b + pos + t;
                        flat_vq[ch * size + bin] = w[ch][bin] / env_v;
                        // истинный масштаб выхода бина: env·st·g (без клампов);
                        // в режиме --psy — домен NMR: делим на порог^{2β}
                        // (PSY.md §3), маска > 0 всегда (пол ATH)
                        let sc = envs[ch][pos + t] as f64 * st[band_idx] as f64 * g;
                        w_vq[ch * size + bin] = match &psy {
                            Some((_, wf)) => sc * sc * wf[ch][bin],
                            None => sc * sc,
                        };
                    }
                    pos += w_dt as usize;
                }
            }
        }
        frame.main_coeffs = quant_vq_main_w(
            &self.pool,
            &self.vq[ftype.idx()],
            info.layout.div(ftype),
            &flat_vq,
            &w_vq,
            &self.perm[ftype.idx()],
        );
        self.prof[4] += _tp2.elapsed().as_secs_f64();
        self.prof[5] += 1.0; // кадров

        FrameEncoding { frame, target: x }
    }

    /// Поиск PPC одного канала: (p_coef, g_coef) + целевые значения
    /// формы на пиках (в слоте канала).
    ///
    /// Стратегия NTT (E2b + послойный анализ sine_1k): гребёнка подбирается
    /// так, чтобы её ГАРМОНИКИ накрывали энергию спектра — тон выше 600 Гц
    /// кодируется как k-я гармоника более низкой f0 (sine_1k → f0≈495 Гц).
    /// Период ищется максимизацией покрытой энергии по гриду p_coef;
    /// окончательная приёмка (изъятие ≥ ε энергии) делается в encode_frame
    /// на СОВМЕСТНОЙ реконструкции обоих каналов.
    fn ppc_search_ch(&self, ch: usize, flat: &[f64]) -> Option<PpcSel> {
        let info = self.info;
        let m = info.mode;
        let size = m.size;

        // энергия мишени: гребёнка должна накрывать спектр flat
        let energy: f64 = flat.iter().map(|v| v * v).sum();
        if energy < 1e-8 {
            return None;
        }
        let dbg = std::env::var("TEQUILA_PPC_DEBUG").is_ok();

        // спектральный поиск периода: максимум покрытой гармониками энергии
        // p = 0 — ЗАРЕЗЕРВИРОВАН декодером как «PPC выкл» (decode_ppc
        // игнорирует гребёнку при p_coef=0) — в переборе не участвует;
        // выбор p=0 означал бы бесполезно сожжённые PPC-биты (multitone:
        // лучший период попадал ровно в индекс 0 → гребёнка терялась)
        let periods = &self.ppc_periods;
        // покрытие энергии по гриду периодов: индексы независимы —
        // map_into (детерминизм: выбор периода — последовательный
        // argmax с строгим >, как в исходном цикле)
        let mut covers = vec![0.0f64; periods.len()];
        self.pool.map_into(&mut covers, |p| {
            let (bins, _sis) = &self.ppc_peaks[p];
            if bins.is_empty() {
                return 0.0;
            }
            cover_of_bins(bins, size, |i| flat[i].abs())
        });
        let mut p_best = 0usize;
        let mut best_cover = 0.0f64;
        for (p, &c) in covers.iter().enumerate() {
            if c > best_cover {
                best_cover = c;
                p_best = p;
            }
        }
        let (bins, sis) = self.ppc_peaks[p_best].clone();
        // ранний отсев только по абсолютной тишине
        if best_cover < 1e-6 * energy {
            if dbg {
                eprintln!("ppc ch={}: NOCOVER cover={:.3e} energy={:.3e}", ch, best_cover, energy);
            }
            return None;
        }

        // амплитуда пиков, скан гейна и цели формы
        let (amp, best_g, entries) = self.ppc_gain_scan(ch, flat, p_best, &bins, &sis);
        if amp < 1e-6 {
            return None;
        }
        if dbg {
            let ppc_gain = self.ppc_gains[best_g as usize];
            let c = (PPC_SHAPE_KAPPA * self.ppc_sigma * ppc_gain / amp).min(1.0);
            eprintln!(
                "ppc ch={}: FOUND p={} g={} cover/energy={:.4} amp={:.1} c={:.2} nbins={}",
                ch, p_best, best_g, best_cover / energy, amp, c, bins.len()
            );
        }
        Some(PpcSel {
            p_coef: p_best as u32,
            g_coef: best_g,
            bins,
            entries,
            energy,
        })
    }

    /// Поиск ОБЩЕЙ гребёнки стерео-пары (R17): период ОДИН на оба канала —
    /// покрытие ищется в L-домене (c0+c1: M/S сохраняет общий компонент
    /// в обоих каналах, decor_stereo = тон в L + шум в R → тон виден в
    /// c0 и c1 одновременно), гейны — поканальные, форма кодируется
    /// совместно в encode_frame. Приёмка пары — в L-домене.
    ///
    /// Возвращает sel обоих каналов (пара гейтится только NOCOVER в L);
    /// «пустой» канал получает форму ~0 (его гребёнка ≈ 0 — не вредит).
    fn ppc_search_stereo(&self, flat: &[Vec<f64>]) -> Vec<Option<PpcSel>> {
        let info = self.info;
        let m = info.mode;
        let size = m.size;

        // энергия L-домена
        let mut energy_l = 0.0f64;
        for t in 0..size {
            let vl = flat[0][t] + flat[1][t];
            energy_l += vl * vl;
        }
        if energy_l < 1e-8 {
            return vec![None, None];
        }
        let dbg = std::env::var("TEQUILA_PPC_DEBUG").is_ok();

        // общий период: максимум покрытия в L-домене
        let periods = &self.ppc_periods;
        let mut covers = vec![0.0f64; periods.len()];
        self.pool.map_into(&mut covers, |p| {
            let (bins, _sis) = &self.ppc_peaks[p];
            if bins.is_empty() {
                return 0.0;
            }
            cover_of_bins(bins, size, |i| (flat[0][i] + flat[1][i]).abs())
        });
        let mut p_best = 0usize;
        let mut best_cover = 0.0f64;
        for (p, &c) in covers.iter().enumerate() {
            if c > best_cover {
                best_cover = c;
                p_best = p;
            }
        }
        if best_cover < 1e-6 * energy_l {
            if dbg {
                eprintln!(
                    "ppc stereo: NOCOVER L cover={:.3e} energy={:.3e}",
                    best_cover, energy_l
                );
            }
            return vec![None, None];
        }
        // экспериментальный режим: полный перебор периодов по точной
        // L-оценке (VQ+decode, g=127) — поиск потолка качества выбора p
        if std::env::var("TEQUILA_PPC_FULLSCAN").is_ok() {
            let mut p_best_fs = p_best;
            let mut best_err_l = f64::INFINITY;
            for p in 1..periods.len() {
                let (b2, s2) = &self.ppc_peaks[p];
                if b2.is_empty() {
                    continue;
                }
                let err_l = self.ppc_eval_l(flat, p, b2, s2);
                if err_l < best_err_l {
                    best_err_l = err_l;
                    p_best_fs = p;
                }
            }
            if dbg {
                eprintln!(
                    "ppc stereo FULLSCAN: p {} (покрытие) -> {} (err_l={:.3e})",
                    p_best, p_best_fs, best_err_l
                );
            }
            p_best = p_best_fs;
        }
        let (bins, sis) = self.ppc_peaks[p_best].clone();

        // поканальные гейны и цели формы с ОБЩИМ периодом.
        // ГЕЙТ КАНАЛА (как NOCOVER поканального поиска): если общий
        // период не покрывает сигнал канала (тон в L, а канал — шум,
        // music_like), гребёнка канала аппроксимировала бы ШУМ
        // (entries = flat[ch]·c/g ≠ 0) и портила W[ch]; такой канал
        // идёт без гребёнки (sel = None), как в поканальном поиске.
        let mut sel = Vec::with_capacity(2);
        for ch in 0..2 {
            let energy_ch: f64 = flat[ch].iter().map(|&v| v * v).sum();
            let cover_ch = cover_of_bins(&bins, size, |i| flat[ch][i].abs());
            if cover_ch < 1e-6 * energy_ch {
                if dbg {
                    eprintln!(
                        "ppc stereo ch={}: NOCOVER cover={:.3e} energy={:.3e} (без гребёнки)",
                        ch, cover_ch, energy_ch
                    );
                }
                sel.push(None);
                continue;
            }
            let (amp, g, entries) = self.ppc_gain_scan(ch, &flat[ch], p_best, &bins, &sis);
            if amp < 1e-6 {
                sel.push(None);
                continue;
            }
            if dbg {
                let ppc_gain = self.ppc_gains[g as usize];
                let c = (PPC_SHAPE_KAPPA * self.ppc_sigma * ppc_gain / amp.max(1e-300)).min(1.0);
                eprintln!(
                    "ppc stereo ch={}: p={} g={} amp={:.1} c={:.2} coverL/E={:.4} nbins={}",
                    ch,
                    p_best,
                    g,
                    amp,
                    c,
                    best_cover / energy_l,
                    bins.len()
                );
            }
            sel.push(Some(PpcSel {
                p_coef: p_best as u32,
                g_coef: g,
                bins: bins.clone(),
                entries,
                // приёмка стерео идёт в L-домене и это поле не читает
                // (критерий использует energy_l, посчитанный в encode_frame)
                energy: energy_l,
            }));
        }
        sel
    }

    /// Быстрая L-оценка качества периода для ОБЩЕЙ гребёнки: оба канала
    /// кодируются с общим гейном-эвристикой (насыщение 127 — типичная
    /// рабочая точка и у NTT), ошибка меряется в L-домене на пиках.
    /// Используется экспериментальным полным перебором (TEQUILA_PPC_FULLSCAN)
    /// для поиска потолка качества выбора периода.
    fn ppc_eval_l(&self, flat: &[Vec<f64>], p: usize, bins: &[i64], sis: &[usize]) -> f64 {
        let info = self.info;
        let m = info.mode;
        let size = m.size;
        let g_hat = self.ppc_gains.len() - 1; // насыщение грида
        let ppc_gain = self.ppc_gains[g_hat];
        // амплитуда пиков L-домена
        let mut amp = 0.0f64;
        for &bp in bins {
            let v = flat[0][bp as usize] + flat[1][bp as usize];
            amp += v * v;
        }
        amp = (amp / bins.len().max(1) as f64).sqrt();
        let c = (PPC_SHAPE_KAPPA * self.ppc_sigma * ppc_gain / amp.max(1e-300)).min(1.0);
        // цели: каждый канал несёт половину L-компоненты
        let mut target_flat = vec![0.0f64; info.n_ch * m.ppc_shape_len];
        for ((&bpos, &si), t) in bins.iter().zip(sis.iter()).zip(target_flat.chunks_mut(m.ppc_shape_len)) {
            let v = (flat[0][bpos as usize] + flat[1][bpos as usize]) * 0.5 * c / ppc_gain;
            t[si] = v;
        }
        let codes = quant_vq_ppc_cached(
            &self.vq_ppc,
            &info.layout.ppc,
            &target_flat,
            &self.perm[FrameType::Ppc.idx()],
        );
        let mut fr = FrameData::new(&info, 0);
        fr.ppc_coeffs = codes;
        let shape = dequant_ppc_shape(&info, &fr, &self.perm[FrameType::Ppc.idx()]);
        let mut spec_l = vec![0.0f64; size];
        for ch in 0..2 {
            let mut fr2 = FrameData::new(&info, 0);
            fr2.p_coef[ch] = p as u32;
            fr2.g_coef[ch] = g_hat as u32;
            decode_ppc(&info, ch, &fr2, &shape, &mut spec_l);
        }
        let mut err = 0.0f64;
        for &bp in bins {
            let b = bp as usize;
            let fl = flat[0][b] + flat[1][b];
            let d = fl - spec_l[b];
            err += d * d;
        }
        err
    }

    /// Скан гейна PPC для ФИКСИРОВАННОГО периода: полный скан му-лог
    /// грида (σ-эвристика amp/σ часто вне диапазона грида — насыщение
    /// в 127 давало перелёт/недолёт гребёнки; KILL на decor c1).
    /// Грид ~логарифмический: грубый проход каждые 4 индекса
    /// (параллельно) + уточнение ±3 вокруг лучшего; оценка — фактическая
    /// ошибка на пиках при поканальном кодировании (приближение;
    /// финальное — совместное, в encode_frame).
    ///
    /// Возвращает (амплитуда пиков, g_coef, целевые значения формы).
    fn ppc_gain_scan(
        &self,
        ch: usize,
        flat: &[f64],
        p_best: usize,
        bins: &[i64],
        sis: &[usize],
    ) -> (f64, u32, Vec<(usize, f64)>) {
        let info = self.info;
        let m = info.mode;
        let size = m.size;

        // амплитуда пиков
        let mut amp = 0.0;
        for &bpos in bins {
            let v = flat[bpos as usize];
            amp += v * v;
        }
        amp = (amp / bins.len().max(1) as f64).sqrt();

        // сигма PPC-книг и грид гейна — кэш энкодера
        let shape_sigma = self.ppc_sigma;
        let g_grid = &self.ppc_gains;

        let evaluate_g = |g_cand: usize| -> f64 {
            let ppc_gain = g_grid[g_cand];
            // масштаб цели: не выше KAPPA_PPC·σ_shape. Полная амплитуда
            // пиков (amp/gain часто >> 2σ) паре кодбуков недостижима:
            // поиск берёт максимальные по норме строки с чужим направлением,
            // гребёнка «перелетает» и приёмка её убивает (decor c1).
            // Частичная гребёнка α·flat всё равно изымает (2α−α²) энергии
            // пиков, а направление кодируется точно.
            let c = (PPC_SHAPE_KAPPA * shape_sigma * ppc_gain / amp.max(1e-300)).min(1.0);
            let mut target_flat = vec![0.0f64; info.n_ch * m.ppc_shape_len];
            let base = ch * m.ppc_shape_len;
            for (bpos, si) in bins.iter().zip(sis.iter()) {
                target_flat[base + si] = flat[*bpos as usize] * c / ppc_gain;
            }
            let codes = quant_vq_ppc_cached(
                &self.vq_ppc,
                &info.layout.ppc,
                &target_flat,
                &self.perm[FrameType::Ppc.idx()],
            );
            let mut fr = FrameData::new(&info, 0);
            fr.ppc_coeffs = codes;
            let shape = dequant_ppc_shape(&info, &fr, &self.perm[FrameType::Ppc.idx()]);
            let sh = &shape[base..base + m.ppc_shape_len];
            let mut fr2 = FrameData::new(&info, 0);
            fr2.p_coef[ch] = p_best as u32;
            fr2.g_coef[ch] = g_cand as u32;
            let mut spec = vec![0.0f64; size];
            decode_ppc(&info, ch, &fr2, sh, &mut spec);
            let mut err = 0.0;
            for &bpos in bins {
                let d = flat[bpos as usize] - spec[bpos as usize];
                err += d * d;
            }
            err
        };
        // грубый проход (каждый 4-й индекс) — параллельно: кандидаты
        // независимы (VQ-поиск по общему кэшу &self, запись — свои слоты);
        // выбор — последовательный строгий < (семантика исходного цикла)
        let coarse: Vec<usize> = (0..g_grid.len()).step_by(4).collect();
        let mut coarse_errs = vec![f64::INFINITY; coarse.len()];
        self.pool.map_into(&mut coarse_errs, |ci| evaluate_g(coarse[ci]));
        let mut best_g = 0usize;
        let mut best_err = f64::INFINITY;
        for (ci, &e) in coarse_errs.iter().enumerate() {
            if e < best_err {
                best_err = e;
                best_g = coarse[ci];
            }
        }
        let lo = best_g.saturating_sub(3);
        let hi = (best_g + 4).min(g_grid.len());
        for g_cand in lo..hi {
            if g_cand == best_g || g_cand % 4 == 0 {
                continue; // уже оценён грубым проходом
            }
            let err = evaluate_g(g_cand);
            if err < best_err {
                best_err = err;
                best_g = g_cand;
            }
        }
        // целевые значения формы для совместного кодирования
        let ppc_gain = g_grid[best_g];
        let c = (PPC_SHAPE_KAPPA * shape_sigma * ppc_gain / amp.max(1e-300)).min(1.0);
        let entries: Vec<(usize, f64)> = bins
            .iter()
            .zip(sis.iter())
            .map(|(&bpos, &si)| (si, flat[bpos as usize] * c / ppc_gain))
            .collect();
        (amp, best_g as u32, entries)
    }
}

/// Покрытая гармониками энергия для данного набора пиков: max |get| в
/// окне ±1 бин вокруг центра пика (как в исходном поканальном поиске).
fn cover_of_bins(bins: &[i64], size: usize, get: impl Fn(usize) -> f64) -> f64 {
    let mut cover = 0.0f64;
    for &bpos in bins {
        let mut mx = 0.0f64;
        for d in [-1i64, 0, 1] {
            let q = bpos + d;
            if q >= 0 && (q as usize) < size {
                mx = mx.max(get(q as usize));
            }
        }
        cover += mx * mx;
    }
    cover
}

/// Выбор PPC одного канала (до совместного кодирования формы).
struct PpcSel {
    p_coef: u32,
    g_coef: u32,
    /// Позиции пиков гребёнки (бины спектра).
    bins: Vec<i64>,
    /// Целевые значения формы (si, flat[bin]/ppc_gain).
    entries: Vec<(usize, f64)>,
    /// Полная энергия flat канала (для критерия приёмки).
    energy: f64,
}

/// Позиция сэмпла → индекс полосы (bark_tab).
fn pos_to_band(bark_tab: &[u16], pos: usize) -> usize {
    let mut acc = 0usize;
    for (i, w) in bark_tab.iter().enumerate() {
        acc += *w as usize;
        if pos < acc {
            return i;
        }
    }
    bark_tab.len() - 1
}

// ----------------------------------------------------------------------------
// Мишень MDCT: точный адъюнкт синтеза декодера
// ----------------------------------------------------------------------------

/// Окно синтеза wtype → wsize (`types_sizes` + `wtype_to_wsize` FFmpeg).
fn synth_wsize(info: &StreamInfo, wtype: u8) -> usize {
    let m = info.mode;
    let ts = [
        m.size / m.fmode[FrameType::Long.idx()].sub,
        m.size / m.fmode[FrameType::Medium.idx()].sub,
        m.size / (m.fmode[FrameType::Short.idx()].sub * 2),
    ];
    ts[WTYPE_TO_WSIZE_IDX[wtype as usize]]
}

/// Сэмпл входа по позиции потока (за границами — нули).
fn sample_at(input: &[f64], pos: isize) -> f64 {
    if pos >= 0 && (pos as usize) < input.len() {
        input[pos as usize]
    } else {
        0.0
    }
}

/// Мишень MDCT кадра k — точный адъюнкт синтеза декодера
/// (imdct_and_window + imdct_output, схема Эдлера).
///
/// Геометрия (выведена из синтеза):
/// * `p = (size + first_wsize)/2` — last_block_pos после кадра k;
/// * `curr_k[c]` → позиция потока `(k−1)·size − p + c` (линейно) для
///   `c < p + size1_{k+1}`; хвост `[p + size1_{k+1}, p + size/2)` сворачивается
///   бабочкой кадра k+1 в поток `(k−1)·size + size1_{k+1} + [0, w_{k+1})`
///   (фолд: веса w2/w1 — подъядро P-пути);
/// * на подкадр j: (i) y[ws/2−1−t] — адъюнкт бабочки j;
///   (ii) y[n] = Ō[out2+ws+n−ws/2] — плоская часть (выживает до
///   перезаписи бабочкой j+1; для последнего подкадра — весь хвост через
///   фолд-Ō); (iii) y[b−ws_{j+1}/2+t] — P-часть бабочки j+1;
/// * X_j = c·mdct_analysis_compact(y_j), c = 2^16/√(norm·B) (калибровка E1).
///
/// PR: анализ∘синтез = тождество (тест analysis_synthesis_pr, ~247 дБ).
pub fn analysis_targets(
    info: &StreamInfo,
    k: usize,
    wtype: u8,
    wtype_next: u8,
    input: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let m = info.mode;
    let size = m.size;
    let ftype = info.ftype_of_wtype(wtype);
    let fm = &m.fmode[ftype.idx()];
    let sub = fm.sub;
    let b = size / sub;
    let norm = info.norm();
    let n_ch = info.n_ch;

    let p = (size + synth_wsize(info, wtype)) / 2;
    let w_next = synth_wsize(info, wtype_next);
    let size1_next = (size - w_next) / 2; // = size − p_{k+1}
    let base = (k as isize - 1) * size as isize - p as isize; // поток curr_k[0]

    // Ō: желаемое значение на позициях curr_k[c]; за пределами
    // [0, p + size/2) данные не доходят до выхода → 0
    let olen = p + size / 2;
    let win_n = sine_window(w_next);
    let mut o = vec![vec![0.0f64; olen]; n_ch];
    for ch in 0..n_ch {
        let inp = &input[ch];
        for c in 0..olen {
            if c < p + size1_next {
                o[ch][c] = sample_at(inp, base + c as isize);
            } else {
                let t = c - p - size1_next; // < w_next/2
                let s1 = (k as isize - 1) * size as isize + size1_next as isize + t as isize;
                let s2 = s1 + (w_next - 1 - 2 * t) as isize;
                o[ch][c] = win_n[w_next - 1 - t] * sample_at(inp, s1)
                    + win_n[t] * sample_at(inp, s2);
            }
        }
    }
    let oget = |oc: &[f64], i: usize| -> f64 { oc.get(i).copied().unwrap_or(0.0) };

    // суб_wtype логика — как в imdct_and_window
    let sub_wtype_of = |j: usize| -> u8 {
        if ftype == FrameType::Medium {
            if j == 0 && wtype == 4 {
                4
            } else if j == sub - 1 && wtype == 7 {
                7
            } else {
                8
            }
        } else {
            wtype
        }
    };

    let c_scale = 2.0f64.powi(16) / (norm * b as f64).sqrt();
    let mut x = vec![vec![0.0f64; size]; n_ch];
    for ch in 0..n_ch {
        let oc = &o[ch];
        let mut out2 = 0usize;
        for j in 0..sub {
            let ws = synth_wsize(info, sub_wtype_of(j));
            let win = sine_window(ws);
            let l = ws / 2;

            // (i) адъюнкт бабочки j
            let mut y = vec![0.0f64; b];
            for t in 0..l {
                y[l - 1 - t] = -win[t] * oget(oc, out2 + t)
                    + win[ws - 1 - t] * oget(oc, out2 + ws - 1 - t);
            }

            // (ii) плоская часть (выживает до перезаписи бабочкой j+1)
            let extra = if ftype == FrameType::Medium {
                (b - ws) / 2
            } else {
                b - ws
            };
            let flat_end = if j + 1 < sub { ws / 2 + extra } else { b };
            for n in ws / 2..flat_end.min(b) {
                y[n] = oget(oc, out2 + ws + n - ws / 2);
            }

            // (iii) P-часть бабочки j+1 (внутри кадра)
            if j + 1 < sub {
                let ws2 = synth_wsize(info, sub_wtype_of(j + 1));
                let win2 = sine_window(ws2);
                let out2_next = out2 + ws + extra;
                for t in 0..ws2 / 2 {
                    let n = b + t - ws2 / 2;
                    if n < b {
                        y[n] = win2[ws2 - 1 - t] * oget(oc, out2_next + t)
                            + win2[t] * oget(oc, out2_next + ws2 - 1 - t);
                    }
                }
            }

            let coeffs = mdct_analysis_compact(&y, b);
            // σ = −1: адъюнкт даёт −c·mdct_fwd (проверено тестом
            // analysis_long_matches_mdct_fwd, масштаб ровно −1)
            for (i, xv) in coeffs.iter().enumerate() {
                x[ch][j * b + i] = -c_scale * xv;
            }
            out2 += ws + extra;
        }
    }
    x
}

// ----------------------------------------------------------------------------
// wtype-грамматика
// ----------------------------------------------------------------------------

/// Энергия кадра k (мощность на сэмпл, максимум по каналам).
/// Кадр k покрывает вход [c_k, c_k + size), c_k = (k−2)·size — как его
/// окно анализа (адъюнкт): рассинхрон детектора и мишени недопустим.
fn frame_power(info: &StreamInfo, input: &[Vec<f64>], k: usize) -> f64 {
    let size = info.mode.size;
    let start = ((k as isize - 2) * size as isize).max(0) as usize;
    let mut best = 0.0f64;
    for ch in input {
        let mut s = 0.0;
        let mut n = 0usize;
        for i in start..(start + size).min(ch.len()) {
            s += ch[i] * ch[i];
            n += 1;
        }
        best = best.max(s / n.max(1) as f64);
    }
    best
}

/// Транзиентность кадра: отношение пиковой мощности подкадра (16-е) к
/// средней. Клик-«палочка» → десятки/сотни; плавная AM/тон → 1..3.
fn frame_transient(info: &StreamInfo, input: &[Vec<f64>], k: usize) -> f64 {
    let size = info.mode.size;
    let sub = 16usize;
    let ssub = size / sub;
    let start = ((k as isize - 2) * size as isize).max(0) as usize;
    let mut best = 0.0f64;
    for ch in input {
        let mut powers = [0.0f64; 16];
        for (i, p) in powers.iter_mut().enumerate() {
            let lo = start + i * ssub;
            let hi = (lo + ssub).min(ch.len());
            let mut s = 0.0;
            for t in lo..hi {
                s += ch[t] * ch[t];
            }
            *p = s / ssub as f64;
        }
        let mean: f64 = powers.iter().sum::<f64>() / sub as f64;
        let peak = powers.iter().cloned().fold(0.0f64, f64::max);
        if mean > 1e-12 {
            best = best.max(peak / mean);
        }
    }
    best
}

/// Решение wtype-последовательности (грамматика NTT, E3).
///
/// ```text
/// старт ненулевого сигнала:  [1, 2, 3, 0, …]
/// транзиент (T > THR):      [1(k), 2×n, 3, 0…], серия держится, пока
///                           в пределах HANG кадров ожидается новый
///                           транзиент (плотные клики → сплошные S2)
/// стационар/тишина:         L0
/// ```
///
/// Инварианты: S2 только после L1/S2; перед S2 не может быть L0/L3.
/// Затирание L3 единицей новой слипшейся скобки давало запрещённый
/// переход S→L1 (длинное окно после короткого) — при слипании новая
/// скобка начинается S2 сразу после L3 (L1 нужна только после L0).
pub fn decide_wtypes(info: &StreamInfo, input: &[Vec<f64>], n_frames: usize, opts: &EncoderOpts) -> Vec<u8> {
    let mut wt = vec![0u8; n_frames];
    let e: Vec<f64> = (0..n_frames + 2)
        .map(|k| frame_power(info, input, k))
        .collect();
    let tr: Vec<f64> = (0..n_frames + 2)
        .map(|k| frame_transient(info, input, k))
        .collect();
    let silence = opts.silence_power;
    const TRANSIENT: f64 = 8.0;
    const HANG: usize = 4;

    // старт: ненулевое начало → скобка [1, 2, 3]
    let mut k = 0usize;
    if n_frames > 0 && e[0] > 8.0 * silence {
        wt[0] = 1;
        if n_frames > 1 {
            wt[1] = 2;
        }
        if n_frames > 2 {
            wt[2] = 3;
        }
        k = 3;
    }

    while k < n_frames {
        // атака = резкий ВСПЛЕСК за окно из 3 кадров (не «выход из провала»:
        // плавная AM с глубокими провалами даёт e/prev до 30+ на каждом
        // периоде — NTT держит такие сигналы LONG) ИЛИ острый транзиент
        // внутри кадра (клик: peak/mean по 16 подокнам >> 1).
        let prev_ref = (1..=3)
            .filter(|&d| k >= d)
            .map(|d| e[k - d])
            .fold(silence, f64::max);
        let attack = (tr[k] > TRANSIENT || e[k] > opts.attack_ratio * prev_ref)
            && e[k] > 8.0 * silence;
        if !attack {
            k += 1;
            continue;
        }
        // скобка атаки: L1 на кадре ДО транзиента («приготовиться»),
        // S2 — на кадре транзиента и далее. L1 ставится только на L0:
        // после S2/L3 (слипшиеся скобки) сразу S2 — иначе запрещённый S→L1.
        let start = k.saturating_sub(1);
        if wt[start] == 0 {
            wt[start] = 1;
        }
        // первый S2 — кадр атаки k (безусловно), далее — по hangover-условию
        let mut next = k;
        let mut n = 1usize;
        if next < n_frames {
            wt[next] = 2;
        }
        next += 1;
        loop {
            if next >= n_frames || n >= 26 {
                break;
            }
            // продолжаем при собственном транзиенте/энергии кадра или
            // ожидаемом транзиенте в пределах HANG
            let upcoming = (next..(next + HANG).min(e.len()))
                .any(|j| tr[j] > TRANSIENT && e[j] > 8.0 * silence);
            let active = e[next] > 8.0 * silence && tr[next] > 2.0;
            if !(upcoming || active) {
                break;
            }
            wt[next] = 2;
            n += 1;
            next += 1;
        }
        let l3 = next;
        if l3 < n_frames {
            wt[l3] = 3;
        }
        // следующая атака ищется сразу после L3 (скобки могут слипаться)
        k = l3 + 1;
    }
    wt
}

/// Полный цикл кодирования: вход (планарные каналы ±1.0) → кадры.
///
/// Стерео-вход автоматически переводится в c0=(L+R)/2, c1=(L−R)/2.
pub fn encode_stream(
    info: &StreamInfo,
    input_ch: &[Vec<f64>],
    opts: EncoderOpts,
) -> (Vec<FrameData>, Vec<u8>) {
    let size = info.mode.size;
    let n = input_ch.first().map(|c| c.len()).unwrap_or(0);
    let n_frames = n.div_ceil(size) + 2;

    // каналы кодирования
    let chans: Vec<Vec<f64>> = if info.n_ch == 2 {
        let l = input_ch.first().cloned().unwrap_or_default();
        let r = input_ch.get(1).cloned().unwrap_or_else(|| l.clone());
        let mut c0 = vec![0.0f64; l.len()];
        let mut c1 = vec![0.0f64; l.len()];
        for i in 0..l.len() {
            c0[i] = (l[i] + r[i]) * 0.5;
            c1[i] = (l[i] - r[i]) * 0.5;
        }
        vec![c0, c1]
    } else {
        vec![input_ch.first().cloned().unwrap_or_default()]
    };

    // паддинг: окна последних кадров выходят за n_frames·size
    let pad_len = n_frames * size + 2 * size;
    let mut padded = vec![vec![0.0f64; pad_len]; info.n_ch];
    for (p, c) in padded.iter_mut().zip(chans.iter()) {
        let ncopy = c.len().min(pad_len);
        p[..ncopy].copy_from_slice(&c[..ncopy]);
    }

    let wtypes = decide_wtypes(info, &padded, n_frames, &opts);
    let mut enc = Encoder::new(*info, opts);
    let dump_path = std::env::var("TEQUILA_DUMP_TARGETS").ok();
    let mut dump: Vec<Vec<Vec<f64>>> = if dump_path.is_some() {
        Vec::with_capacity(n_frames)
    } else {
        Vec::new()
    };
    let mut frames = Vec::with_capacity(n_frames);

    // Батчевое кодирование: Phase A — мишени MDCT батча параллельно
    // (analysis_targets — чистая функция входа и wtypes), Phase B —
    // последовательное кодирование кадров батча (MA-истории LSP/bark
    // связывают кадры; внутри кадра параллелятся независимые поиски).
    // Батч ограничивает память мишеней (~BATCH·n_ch·size·8 байт);
    // считается на +1 кадр больше — lookahead pre-masking (--psy).
    const BATCH: usize = 32;
    let mut k0 = 0usize;
    while k0 < n_frames {
        let k1 = (k0 + BATCH).min(n_frames);
        let k_end = (k1 + 1).min(n_frames); // +1 кадр lookahead
        let mut targets: Vec<Vec<Vec<f64>>> =
            (0..k_end - k0).map(|_| Vec::new()).collect();
        {
            let _tp = std::time::Instant::now();
            let wtypes_ref = &wtypes;
            let padded_ref = &padded;
            let info_c = *info;
            let k_base = k0;
            enc.pool().map_into(&mut targets, |bi| {
                let k = k_base + bi;
                analysis_targets(
                    &info_c,
                    k,
                    wtypes_ref[k],
                    wtypes_ref.get(k + 1).copied().unwrap_or(0),
                    padded_ref,
                )
            });
            enc.prof[0] += _tp.elapsed().as_secs_f64();
        }
        for bi in 0..(k1 - k0) {
            let k = k0 + bi;
            let w = wtypes[k];
            let x = std::mem::take(&mut targets[bi]);
            let next = if k + 1 < n_frames {
                Some((wtypes[k + 1], &targets[bi + 1]))
            } else {
                None
            };
            let fe = enc.encode_frame(w, x, next);
            if dump_path.is_some() {
                dump.push(fe.target.clone());
            }
            frames.push(fe.frame);
        }
        k0 = k1;
    }
    if let Some(path) = dump_path {
        // формат: [u32 n_frames][u32 n_ch][u32 size][u32 pad]
        //         затем n_frames × n_ch × size значений f64 LE
        use std::io::Write;
        let mut buf = Vec::with_capacity(16 + dump.len() * info.n_ch * size * 8);
        buf.extend_from_slice(&(dump.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(info.n_ch as u32).to_le_bytes());
        buf.extend_from_slice(&(size as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for fr in &dump {
            for ch in fr {
                for v in ch {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = f.write_all(&buf);
        }
    }
    if std::env::var("TEQUILA_PROFILE").is_ok() {
        eprintln!(
            "профиль: mdct={:.3}s lsp+flat={:.3}s ppc={:.3}s bark+gain={:.3}s vq={:.3}s ({} кадров)",
            enc.prof[0], enc.prof[1], enc.prof[2], enc.prof[3], enc.prof[4],
            enc.prof[5] as usize
        );
    }
    (frames, wtypes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{parse_vqf, write_vqf};
    use crate::decode::decode_stream;

    fn make_signal(freqs: &[f64], sr: u32, dur_s: f64, seed: u64) -> Vec<f64> {
        let n = (sr as f64 * dur_s) as usize;
        let mut rng = seed;
        let mut x = vec![0.0f64; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f64 / sr as f64;
            let mut s = 0.0;
            for (fi, f) in freqs.iter().enumerate() {
                s += (2.0 * std::f64::consts::PI * f * t + fi as f64).sin();
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            s += 0.02 * (2.0 * ((rng >> 32) as f64 / u32::MAX as f64) - 1.0);
            *v = s * 0.2;
        }
        x
    }

    /// E2E: encode → parse → decode → seg-SNR против входа.
    #[test]
    fn e2e_encode_decode_sanity() {
        let sr = 44100u32;
        let info = StreamInfo::new(44, 48, 1).unwrap();
        // многочастотный сигнал 1 c
        let x = make_signal(&[440.0, 1234.0, 3456.0], sr, 1.0, 42);
        let (frames, wtypes) = encode_stream(&info, &[x.clone()], EncoderOpts::default());
        assert!(!frames.is_empty());
        assert!(wtypes[0] == 1 && wtypes[1] == 2 && wtypes[2] == 3, "стартовая скобка");

        // битстрим-райтер + парсер round-trip
        let blob = write_vqf(&info, &frames, &[]);
        let parsed = parse_vqf(&blob).unwrap();
        assert_eq!(parsed.frames, frames);

        // собственный декод: PCM vs вход
        let pcm = decode_stream(&parsed.info, &parsed.frames);
        let n_out = pcm[0].len();
        assert!(n_out >= x.len(), "выход короче входа: {n_out} < {}", x.len());
        // seg-SNR по кадрам size
        let size = info.mode.size;
        let mut seg_snr = Vec::new();
        for (seg_i, (a, b)) in pcm[0]
            .chunks(size)
            .zip(x.chunks(size))
            .enumerate()
        {
            if seg_i < 1 {
                continue; // краевой кадр TDAC
            }
            let mut se = 0.0;
            let mut sx = 0.0;
            for (p, q) in a.iter().zip(b.iter()) {
                se += (p - q) * (p - q);
                sx += q * q;
            }
            if sx > 1e-9 && se > 1e-12 {
                seg_snr.push(10.0 * (sx / se).log10());
            }
        }
        let avg = seg_snr.iter().sum::<f64>() / seg_snr.len().max(1) as f64;
        eprintln!("e2e 44_48 multitone: {} сегментов, средний seg-SNR = {avg:.2} дБ", seg_snr.len());
        assert!(avg > 4.0, "seg-SNR слишком низкий: {avg}");
    }

    /// E2E режима `--psy`: битстрим валиден (round-trip контейнера),
    /// декодируется, seg-SNR не разваливается; битстрим отличается от
    /// обычного режима (веса реально влияют на выбор кодов VQ).
    #[test]
    fn psy_mode_roundtrip_and_effect() {
        use crate::psycho::PsyParams;
        let sr = 44100u32;
        let info = StreamInfo::new(44, 48, 1).unwrap();
        let x = make_signal(&[440.0, 1234.0, 3456.0], sr, 1.0, 42);

        let (frames_psy, _) = {
            let mut o = EncoderOpts::default();
            o.psy = Some(PsyParams::default());
            encode_stream(&info, &[x.clone()], o)
        };
        // контейнер round-trip + собственный декод
        let blob = write_vqf(&info, &frames_psy, &[]);
        let parsed = parse_vqf(&blob).unwrap();
        assert_eq!(parsed.frames, frames_psy);
        let pcm = decode_stream(&parsed.info, &parsed.frames);

        let size = info.mode.size;
        let mut seg_snr = Vec::new();
        for (seg_i, (a, b)) in pcm[0].chunks(size).zip(x.chunks(size)).enumerate() {
            if seg_i < 1 {
                continue;
            }
            let mut se = 0.0;
            let mut sx = 0.0;
            for (p, q) in a.iter().zip(b.iter()) {
                se += (p - q) * (p - q);
                sx += q * q;
            }
            if sx > 1e-9 && se > 1e-12 {
                seg_snr.push(10.0 * (sx / se).log10());
            }
        }
        let avg = seg_snr.iter().sum::<f64>() / seg_snr.len().max(1) as f64;
        eprintln!("psy e2e 44_48 multitone: средний seg-SNR = {avg:.2} дБ");
        assert!(avg > 3.0, "psy seg-SNR слишком низкий: {avg}");

        // сравнение с базой: psy-битстрим отличается (веса меняют выбор VQ),
        // но формат кадра идентичен (та же раскладка полей/бит)
        let (frames_base, _) = encode_stream(&info, &[x], EncoderOpts::default());
        assert_eq!(frames_psy.len(), frames_base.len());
        let differs = frames_psy
            .iter()
            .zip(frames_base.iter())
            .any(|(a, b)| a.main_coeffs != b.main_coeffs || a.bark1 != b.bark1);
        assert!(differs, "psy не повлиял на битстрим — интеграция молчит");
    }

    /// `--psy` выключен → битстрим байт-в-байт как до появления модели
    /// (psy=None не должен менять ни одну FP-опрацию обычного пути).
    #[test]
    fn psy_disabled_matches_default() {
        let info = StreamInfo::new(44, 48, 2).unwrap();
        let l = make_signal(&[440.0, 1234.0], 44100, 0.4, 11);
        let r = make_signal(&[997.0, 2500.0], 44100, 0.4, 23);
        let mut o = EncoderOpts::default();
        assert!(o.psy.is_none());
        let (f1, _) = encode_stream(&info, &[l.clone(), r.clone()], o);
        o.psy = None; // явно
        let (f2, _) = encode_stream(&info, &[l, r], o);
        assert_eq!(f1, f2);
    }

    /// Многопоточность не меняет битстрим: потоки делят только
    /// независимые индексы (map_into), FP-порядок внутри индекса
    /// скалярный → кадры побитово совпадают при любом --threads.
    #[test]
    fn threads_do_not_change_bitstream() {
        let sr = 44100u32;
        let info = StreamInfo::new(44, 48, 2).unwrap();
        let l = make_signal(&[440.0, 1234.0], sr, 0.4, 11);
        let r = make_signal(&[997.0, 2500.0], sr, 0.4, 23);
        let mut o1 = EncoderOpts::default();
        o1.threads = 1;
        let mut o4 = EncoderOpts::default();
        o4.threads = 4;
        let (f1, w1) = encode_stream(&info, &[l.clone(), r.clone()], o1);
        let (f2, w2) = encode_stream(&info, &[l, r], o4);
        assert!(!f1.is_empty());
        assert_eq!(w1, w2, "wtype-решения разошлись");
        assert_eq!(f1, f2, "кадры разошлись: многопоточность не детерминирована");
    }

    /// wtype-грамматика: тишина → все L0; клик → скобка [L1, S2×n, L3].
    ///
    /// Детектор выровнен по покрытию кадра: кадр k покрывает вход
    /// [c_k, c_k+size), c_k = (k−2)·size — клик в выходном блоке 5
    /// ловится скобкой на кадрах 6..8 (раньше рассинхрон ~1.5 кадра).
    #[test]
    fn wtypes_grammar() {
        let info = StreamInfo::new(44, 48, 1).unwrap();
        let opts = EncoderOpts::default();
        // тишина
        let sil = vec![0.0f64; info.mode.size * 16];
        let wt = decide_wtypes(&info, &[sil.clone()], 16, &opts);
        assert!(wt.iter().all(|&w| w == 0), "{wt:?}");
        // клик в выходном блоке 5 (сэмплы [5·size, 5·size+200))
        let mut x = vec![0.0f64; info.mode.size * 16];
        for v in x[5 * info.mode.size..5 * info.mode.size + 200].iter_mut() {
            *v = 0.8;
        }
        let wt = decide_wtypes(&info, &[x], 16, &opts);
        // кадр 7 покрывает [5·size, 6·size) → S2 на 7, L1 на 6
        assert_eq!(wt[6], 1, "L1 перед атакой: {wt:?}");
        assert_eq!(wt[7], 2, "S2 на кадре клика: {wt:?}");
        assert!(wt[8] == 2 || wt[8] == 3, "S2/L3 после клика: {wt:?}");
        assert!(wt[..6].iter().all(|&w| w == 0), "до скобки — L0: {wt:?}");
        // плотные клики: два клика подряд → скобки слипаются без S→L1
        let mut x = vec![0.0f64; info.mode.size * 24];
        for blk in [5usize, 7] {
            for v in x[blk * info.mode.size..blk * info.mode.size + 300].iter_mut() {
                *v = 0.8;
            }
        }
        let wt = decide_wtypes(&info, &[x], 24, &opts);
        for (i, &w) in wt.iter().enumerate() {
            if i > 0 && w == 1 {
                assert!(
                    wt[i - 1] == 0 || wt[i - 1] == 5 || wt[i - 1] == 6,
                    "L1 в позиции {i} после {:?} — запрещённый переход: {wt:?}",
                    wt[i - 1]
                );
            }
        }
        eprintln!("плотные клики → wtypes: {wt:?}");
        assert!(wt.iter().any(|&w| w == 2), "есть S2-серия");
    }

    /// Эквивалентность адъюнкта старой калибровке: для LONG-кадров
    /// analysis_targets должен совпадать с c·mdct_fwd (E1) — это
    /// сохраняет масштаб мишени для всех слоёв (LSP/bark/gain/VQ).
    #[test]
    fn analysis_long_matches_mdct_fwd() {
        let info = StreamInfo::new(44, 48, 1).unwrap();
        let size = info.mode.size;
        let b = size;
        let mut rng = 0x1234_ABCD_5678_EF90u64;
        let mut x = vec![0.0f64; 6 * size];
        for v in x.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = 0.5 * ((rng >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
        }
        for k in [2usize, 3, 4] {
            let x_new = analysis_targets(&info, k, 0, 0, &[x.clone()]);
            // старая формула: c·mdct_fwd(seg @ off=(k−2)·size, окно 2b)
            let off = (k - 2) * b;
            let c = 2.0f64.powi(16) / (info.norm() * b as f64).sqrt();
            let seg = &x[off..off + 2 * b];
            let x_old = crate::dsp::mdct_fwd(seg, b);
            let mut max_rel = 0.0f64;
            let mut dot = 0.0f64;
            let mut nrm_new = 0.0f64;
            for i in 0..b {
                let nv = x_new[0][i];
                let ov = c * x_old[i];
                dot += nv * ov;
                nrm_new += nv * nv;
                let rel = (nv - ov).abs() / (ov.abs() + 1e-9);
                max_rel = max_rel.max(rel);
            }
            eprintln!("k={k}: max_rel={max_rel:.2e}, масштаб={:.9}", dot / nrm_new);
            assert!(
                max_rel < 1e-8,
                "адъюнкт не совпал с c·mdct_fwd для LONG: k={k}, max_rel={max_rel:.2e}"
            );
        }
    }

    /// PR без квантования: мишени (адъюнкт) → синтез декодера → вход.
    ///
    /// На всех допустимых последовательностях окон (L0-стационар,
    /// стартовая скобка, атака в середине, плотные клики, L3→S2 слипание)
    /// восстановление точное: SNR ~247 дБ при масштабе ровно 1.0.
    #[test]
    fn analysis_synthesis_pr() {
        let info = StreamInfo::new(44, 48, 1).unwrap();
        let size = info.mode.size;
        let n_frames = 14usize;
        let mut rng = 0x0F1E_2D3C_4B5A_6978u64;
        let mut x = vec![0.0f64; n_frames * size];
        for (i, v) in x.iter_mut().enumerate() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let t = i as f64 / 44100.0;
            *v = 0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 3210.0 * t).sin()
                + 0.05 * (((rng >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0);
        }
        // клик в выходном блоке 7
        for v in x[7 * size..7 * size + 400].iter_mut() {
            *v += 0.7;
        }

        let seqs: Vec<(&str, Vec<u8>)> = vec![
            ("все L0", vec![0; n_frames]),
            ("стартовая скобка", {
                let mut w = vec![0u8; n_frames];
                w[0] = 1;
                w[1] = 2;
                w[2] = 3;
                w
            }),
            ("атака в середине", {
                let mut w = vec![0u8; n_frames];
                w[6] = 1;
                w[7] = 2;
                w[8] = 2;
                w[9] = 3;
                w
            }),
            ("плотные клики (слипание)", {
                let mut w = vec![0u8; n_frames];
                w[5] = 1;
                for i in 6..11 {
                    w[i] = 2;
                }
                w[11] = 3;
                w
            }),
            ("S2-серия в конце", {
                let mut w = vec![0u8; n_frames];
                w[0] = 1;
                w[1] = 2;
                w[2] = 2;
                w[3] = 3;
                w[10] = 1;
                w[11] = 2;
                w[12] = 2;
                w[13] = 2;
                w
            }),
        ];

        for (name, wtypes) in seqs {
            let mut dec = crate::decode::Decoder::new(info);
            let mut out = Vec::<f64>::new();
            for k in 0..n_frames {
                let w = wtypes[k];
                let w_next = wtypes.get(k + 1).copied().unwrap_or(0);
                let targets = analysis_targets(&info, k, w, w_next, &[x.clone()]);
                let pcm = dec.synth_frame(info.ftype_of_wtype(w), w, &targets);
                if k >= 2 {
                    out.extend_from_slice(&pcm[0]);
                }
            }
            // сравнение с входом (выходной блок j ↔ вход [j·size, (j+1)·size))
            let n_cmp = out.len().min(x.len());
            let mut se = 0.0f64;
            let mut sx = 0.0f64;
            let mut dot = 0.0f64;
            let mut nrm_o = 0.0f64;
            for i in 0..n_cmp {
                se += (out[i] - x[i]) * (out[i] - x[i]);
                sx += x[i] * x[i];
                dot += out[i] * x[i];
                nrm_o += out[i] * out[i];
            }
            let snr = 10.0 * (sx / se.max(1e-300)).log10();
            let scale = (dot / nrm_o).sqrt();
            eprintln!("PR [{name}]: SNR = {snr:.1} дБ, масштаб = {scale:.9}");
            assert!(snr > 200.0, "PR нарушен [{name}]: SNR = {snr:.1} дБ");
            assert!(
                (scale - 1.0).abs() < 1e-9,
                "масштаб не 1.0 [{name}]: {scale:.12}"
            );
        }
    }
}
