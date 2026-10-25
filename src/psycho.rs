//! Психоакустическая модель режима `--psy` (дизайн — docs/PSY.md).
//!
//! Порог маскировки считается на мишени MDCT (амплитуды в масштабе
//! `c = 2^16/√(norm·b)`), поканально, на каждый подкадр. Модель живёт
//! только на стороне энкодера (f64) и не вмешивается в замкнутый
//! синтез-контур — битстрим и декодер не меняются.
//!
//! Слои модели (PSY.md §2):
//! * полосы ~1 барк (шкала Цвиккера, слитые до ≥1 бина);
//! * two-slope spreading по баркам + маскирующий офсет
//!   OBC = α·TMN + (1−α)·NMT, тональность α — по spectral flatness
//!   (ISO/IED 11172-3 Model 1, нормировка SFM_dB/−60);
//! * ATH с SPL-калибровкой: полномасштабная синусоида = 96 дБ SPL,
//!   опорная амплитуда `fullscale_ref` (тест fullscale_sine_peak);
//! * post-masking: каноническая сетка слотов ½ барка, порог спадает
//!   не быстрее `post_mask_db` дБ/кадр.
//!
//! CBR-интеграция (PSY.md §1, §3): бюджет кадра фиксирован, модель
//! управляет только весами ошибки — main VQ `w = (env·st·g)²·m^{−2β}`
//! и bark-полосы `(env_rms/m_rms)²`. PPC-гейты и wtype-грамматика
//! не трогаются (биты PPC/окон в CBR — sunk cost).

use crate::codec::StreamInfo;
use crate::modes::FrameType;

/// Константы модели: литературные дефолты (Herre & Dick 2019; ISO Model 1),
/// процедура корпусной калибровки — PSY.md §5.
#[derive(Clone, Copy, Debug)]
pub struct PsyParams {
    /// Spreading вверх по частоте (маскировка верхних полос), дБ/барк.
    pub slope_up_db: f64,
    /// Spreading вниз по частоте, дБ/барк (крутая сторона).
    pub slope_down_db: f64,
    /// Tone-masks-noise: насколько шум должен быть ниже тона-маскера, дБ.
    pub tmn_db: f64,
    /// Noise-masks-tone: офсет для шумового маскера, дБ.
    pub nmt_db: f64,
    /// Смещение ATH, дБ (≤ 0 — запас консерватизма).
    pub ath_offset_db: f64,
    /// Спад post-masking, дБ/кадр (~46 мс).
    pub post_mask_db: f64,
    /// Окно pre-masking, мс (порог «видит» будущие подкадры; 0 = выкл).
    /// Pre-masking слабее и короче post: базовое ослабление на dt=0
    /// `pre_mask_db_base`, спад `pre_mask_db_per_ms` за мс.
    pub pre_mask_ms: f64,
    pub pre_mask_db_base: f64,
    pub pre_mask_db_per_ms: f64,
    /// Степень перцептивности весов VQ: w = (env·st·g)²·m^{−2·beta}.
    /// 1.0 — полное деление на порог (слишком большой диапазон для
    /// фиксированных кодбуков: жертвуются слышимые полосы), мягкие
    /// значения 0.1–0.2 дают лучший перцептивный результат
    /// (корпусная калибровка, PSY.md §5).
    pub beta: f64,
    /// Степень перцептивности весов bark-полос: w = (env_rms/m)^{2·bark_beta}.
    /// 0.0 — bark-квантование как в базовом режиме (равные веса);
    /// 0.1 — калиброванный дефолт (см. PSY.md §5).
    pub bark_beta: f64,
    /// Степень перцептивности весов LSP-коэффициентов:
    /// w_k = (уровень|X|/m у частоты LSP_k)^{2·lsp_beta}; 0 = выкл.
    pub lsp_beta: f64,
    /// Адаптивный κ: g_des по перцептивно-взвешенному RMS остатка
    /// (сладкая зона кодбуков выравнивается по слышимой части; PSY.md §7).
    pub gain_weighted: bool,
}

impl Default for PsyParams {
    fn default() -> Self {
        // Калибровка по корпусу + PEAQ (PSY.md §5): β=0.10 доминирует
        // по over-mask/ODG/SNR; lsp_beta=0 — веса LSP на корпусе вредят
        // (over% ×5) и выключены; gain_weighted — сильный выигрыш
        // (SNR +4 дБ, over% −55%); pre-masking нейтрален на стационарных,
        // включён для транзиентов.
        PsyParams {
            slope_up_db: 15.0,
            slope_down_db: 27.0,
            tmn_db: 24.0,
            nmt_db: 4.0,
            ath_offset_db: -2.0,
            post_mask_db: 10.0,
            pre_mask_ms: 20.0,
            pre_mask_db_base: 6.0,
            pre_mask_db_per_ms: 1.0,
            beta: 0.10,
            bark_beta: 0.1,
            lsp_beta: 0.0,
            gain_weighted: true,
        }
    }
}

/// sqrt по полосам: энергия-порог → амплитуда-порог.
fn sqrt_bands(t: &[f64]) -> Vec<f64> {
    t.iter().map(|&v| v.sqrt()).collect()
}

/// bark-шкала Цвиккера.
pub fn bark_of(f_hz: f64) -> f64 {
    13.0 * (0.000_76 * f_hz).atan() + 3.5 * ((f_hz / 7500.0).powi(2)).atan()
}

/// ATH Терхарда, дБ SPL (f в Гц; валиден от ~20 Гц до ~20 кГц).
fn ath_db(f_hz: f64) -> f64 {
    let f = f_hz / 1000.0;
    3.64 * f.powf(-0.8) - 6.5 * (-0.6 * (f - 3.3).powi(2)).exp() + 1e-3 * f.powi(4)
}

/// Пик мишени MDCT от полномасштабной синусоиды (A=32767), выровненной
/// с базисом MDCT (косинус с фазой +B). Численно (b ∈ {128, 512, 2048}):
/// peak = A·(2¹⁷/π)·√(b/norm), на √2 больше косинуса в центре бина без
/// фазового выравнивания (проверяется тестом `fullscale_sine_peak`).
pub fn fullscale_ref(b: usize, norm: f64) -> f64 {
    32767.0 * (131_072.0 / std::f64::consts::PI) * (b as f64 / norm).sqrt()
}

/// Тональность по spectral flatness (ISO 11172-3 Model 1):
/// α = min(−10·log10(SFM)/60, 1), SFM = gm/am энергий полос.
/// Белый шум → α≈0 (широкая маскировка), синус → α≈1 (узкая).
fn tonality(band_energy: &[f64]) -> f64 {
    let n = band_energy.len() as f64;
    let max_e = band_energy.iter().cloned().fold(0.0f64, f64::max);
    if max_e <= 0.0 || n < 2.0 {
        return 0.0;
    }
    // нормировка на max +Clip пол −120 дБ: лог всегда конечен
    let sum_ln: f64 = band_energy
        .iter()
        .map(|&e| (e / max_e + 1e-12).ln())
        .sum();
    let am = band_energy.iter().sum::<f64>() / (max_e * n); // = 1
    let gm = (sum_ln / n).exp() / max_e;
    let sfm_db = 10.0 * (gm / am).log10();
    (sfm_db / -60.0).min(1.0).max(0.0)
}

/// Таблицы полос одного ftype (блок `b` бинов на подкадр).
struct FtypeBands {
    /// Размер блока подкадра, бинов (для длительностей pre-masking).
    b: usize,
    /// Границы полос в бинах (len = n_bands + 1), полосы сплошные ≥1 бина.
    edges: Vec<usize>,
    /// Барки-центры полос (для spreading-ядра).
    center_bark: Vec<f64>,
    /// ATH-амплитуда на бин полосы (с учётом ath_offset), та же шкала,
    /// что и спектр.
    ath_amp: Vec<f64>,
    /// Слоты post-masking [lo, hi) канонической сетки ½ барка.
    slot_range: Vec<(usize, usize)>,
}

/// Модель на поток: таблицы полос трёх ftype + общая слот-сетка.
pub struct PsyModel {
    params: PsyParams,
    ft: [FtypeBands; 3],
    /// Число слотов канонической сетки (½ барка до bark(nyquist)).
    n_slots: usize,
    /// Частота дискретизации (длительности подкадров для pre-masking).
    sr: u32,
}

/// Состояние post-masking на канал: порог-амплитуда на слоте после
/// последнего закодированного кадра.
pub struct PsyState {
    /// Читается энкодером (`mask`) и пишется только `commit`.
    pub slots: Vec<f64>,
}

/// Результат маскировки подкадра.
pub struct MaskResult {
    /// Порог-амплитуда на бин (для весов VQ/bark).
    pub m: Vec<f64>,
    /// Порог-амплитуда на полосу (для commit после кадра).
    pub band_amp: Vec<f64>,
}

impl PsyModel {
    pub fn new(info: &StreamInfo, params: PsyParams) -> Self {
        let nyq = info.sr as f64 * 0.5;
        let n_slots = ((bark_of(nyq) * 2.0).ceil() as usize).max(1);
        let ft = [FrameType::Short, FrameType::Medium, FrameType::Long].map(|ft| {
            let b = info.mode.size / info.mode.fmode[ft.idx()].sub;
            build_bands(b, info.sr, info.norm(), &params, n_slots)
        });
        PsyModel { params, ft, n_slots, sr: info.sr }
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn params(&self) -> PsyParams {
        self.params
    }

    pub fn new_state(&self) -> PsyState {
        PsyState { slots: vec![0.0; self.n_slots] }
    }

    /// Статический порог одного подкадра (без temporalmasking):
    /// spreading + OBC + ATH, энергия на бин по полосам.
    fn static_thresholds(&self, fb: &FtypeBands, spec: &[f64]) -> Vec<f64> {
        let n_bands = fb.center_bark.len();
        debug_assert_eq!(fb.edges.len(), n_bands + 1);

        let mut e = vec![0.0f64; n_bands];
        let mut max_e = 0.0f64;
        for p in 0..n_bands {
            let mut s = 0.0;
            for &v in &spec[fb.edges[p]..fb.edges[p + 1]] {
                s += v * v;
            }
            e[p] = s / (fb.edges[p + 1] - fb.edges[p]) as f64;
            max_e = max_e.max(e[p]);
        }
        if max_e <= 0.0 {
            // тишина: порог = ATH-пол
            return fb.ath_amp.iter().map(|&a| a * a).collect();
        }

        let alpha = tonality(&e);
        let obc = alpha * self.params.tmn_db + (1.0 - alpha) * self.params.nmt_db;
        let up = 10f64.powf(-self.params.slope_up_db / 10.0);
        let down = 10f64.powf(-self.params.slope_down_db / 10.0);
        let off = 10f64.powf(-obc / 10.0);

        let mut t = vec![0.0f64; n_bands];
        for p in 0..n_bands {
            let zp = fb.center_bark[p];
            let mut s = 0.0;
            for q in 0..n_bands {
                if e[q] <= 0.0 {
                    continue;
                }
                let d = zp - fb.center_bark[q];
                let k = if d >= 0.0 { up.powf(d) } else { down.powf(-d) };
                s += e[q] * k;
            }
            t[p] = (s * off).max(fb.ath_amp[p] * fb.ath_amp[p]);
        }
        t
    }

    /// Амплитуды полос → слоты канонической сетки (max по покрываемым).
    fn amps_to_slots(&self, fb: &FtypeBands, amp: &[f64]) -> Vec<f64> {
        let mut slots = vec![0.0f64; self.n_slots];
        for (p, &(lo, hi)) in fb.slot_range.iter().enumerate() {
            let a = amp[p];
            if a <= 0.0 {
                continue;
            }
            for s in lo..hi.min(self.n_slots) {
                if slots[s] < a {
                    slots[s] = a;
                }
            }
        }
        slots
    }

    /// Слоты → амплитуды полос (max по слотам диапазона полосы).
    fn slots_to_amps(&self, fb: &FtypeBands, slots: &[f64]) -> Vec<f64> {
        (0..fb.center_bark.len())
            .map(|p| {
                let (lo, hi) = fb.slot_range[p];
                let mut a = 0.0f64;
                for s in lo..hi.min(slots.len()) {
                    a = a.max(slots[s]);
                }
                a
            })
            .collect()
    }

    /// Разворот порогов полос на бины.
    fn expand(&self, fb: &FtypeBands, band_amp: &[f64], len: usize) -> Vec<f64> {
        let mut m = vec![0.0f64; len];
        for p in 0..band_amp.len() {
            let amp = band_amp[p];
            for v in &mut m[fb.edges[p]..fb.edges[p + 1]] {
                *v = amp;
            }
        }
        m
    }

    /// Порог маскировки подкадра (post-masking, без pre): `spec` —
    /// мишень MDCT (b амплитуд), `slots` — слоты состояния канала.
    pub fn mask(&self, ftype: FrameType, spec: &[f64], slots: &[f64]) -> MaskResult {
        let fb = &self.ft[ftype.idx()];
        let t = self.static_thresholds(fb, spec);
        let decay = 10f64.powf(-self.params.post_mask_db / 10.0);
        let band_amp: Vec<f64> = (0..t.len())
            .map(|p| {
                let mut te = t[p];
                let (lo, hi) = fb.slot_range[p];
                let mut prev = 0.0f64;
                for s in lo..hi.min(slots.len()) {
                    prev = prev.max(slots[s]);
                }
                if prev > 0.0 {
                    te = te.max(prev * prev * decay);
                }
                te.sqrt()
            })
            .collect();
        let m = self.expand(fb, &band_amp, spec.len());
        MaskResult { m, band_amp }
    }

    /// Порог маскировки ЦЕЛОГО КАДРА с lookahead (PSY.md §2.4):
    /// post-masking из состояния + pre-masking из будущих подкадров
    /// (хвост собственного кадра + начало следующего кадра `next`).
    ///
    /// `specs` — подкадры текущего кадра (канала), `next` — подкадры
    /// следующего кадра с его ftype (None в конце потока). Возвращает
    /// MaskResult по подкадрам; commit post-состояния — здесь же
    /// (pre-вклад НЕ пишется в состояние: будущий кадр сам поднимет
    /// свой порог, иначе двойной учёт).
    pub fn mask_frame(
        &self,
        ftype: FrameType,
        specs: &[&[f64]],
        next: Option<(FrameType, Vec<&[f64]>)>,
        state: &mut PsyState,
    ) -> Vec<MaskResult> {
        let fb = &self.ft[ftype.idx()];
        let sub = specs.len();
        let t_now: Vec<Vec<f64>> =
            specs.iter().map(|s| self.static_thresholds(fb, s)).collect();
        let decay = 10f64.powf(-self.params.post_mask_db / 10.0);

        // post-цепочка (пишется в состояние): max(статика, prev·спад)
        let prev_amps = self.slots_to_amps(fb, &state.slots);
        let t_post: Vec<Vec<f64>> = (0..sub)
            .map(|j| {
                (0..t_now[j].len())
                    .map(|p| t_now[j][p].max(prev_amps[p] * prev_amps[p] * decay))
                    .collect()
            })
            .collect();

        // источники pre-masking: (амплитуды полос через слоты, время
        // источника от начала кадра, мс) — потребитель считает dt сам
        let sub_ms = 1000.0 * fb.b as f64 / self.sr as f64;
        let pre_win = self.params.pre_mask_ms;
        let mut pre_src: Vec<(Vec<f64>, f64)> = Vec::new();
        if pre_win > 0.0 && sub > 1 {
            for q in 1..sub {
                let amp = self.slots_to_amps(
                    fb,
                    &self.amps_to_slots(fb, &sqrt_bands(&t_now[q])),
                );
                pre_src.push((amp, q as f64 * sub_ms));
            }
        }
        if pre_win > 0.0 {
            if let Some((ft_next, nspecs)) = next {
                let fb_next = &self.ft[ft_next.idx()];
                let sub_next_ms = 1000.0 * fb_next.b as f64 / self.sr as f64;
                let n_next =
                    nspecs.len().min((pre_win / sub_next_ms).ceil() as usize);
                for s in 0..n_next {
                    let t = self.static_thresholds(fb_next, nspecs[s]);
                    let amp =
                        self.slots_to_amps(fb, &self.amps_to_slots(fb_next, &sqrt_bands(&t)));
                    pre_src.push((amp, sub as f64 * sub_ms + s as f64 * sub_next_ms));
                }
            }
        }

        let base = 10f64.powf(-self.params.pre_mask_db_base / 10.0);
        let per_ms = self.params.pre_mask_db_per_ms;

        let mut out = Vec::with_capacity(sub);
        for j in 0..sub {
            let j_ms = j as f64 * sub_ms;
            let mut band_amp: Vec<f64> = Vec::with_capacity(t_post[j].len());
            for p in 0..t_post[j].len() {
                let mut te = t_post[j][p];
                if !pre_src.is_empty() {
                    let mut pre = 0.0f64;
                    for (amp, src_ms) in &pre_src {
                        let dt = src_ms - j_ms;
                        if dt <= 0.0 || dt > pre_win {
                            continue;
                        }
                        let a = amp[p] * base * 10f64.powf(-per_ms * dt / 10.0);
                        if a > pre {
                            pre = a;
                        }
                    }
                    if pre > 0.0 {
                        te = te.max(pre * pre);
                    }
                }
                band_amp.push(te.sqrt());
            }
            let m = self.expand(fb, &band_amp, specs[j].len());
            out.push(MaskResult { m, band_amp });
        }

        // commit: post-цепочка (max по подкадрам) в слоты;
        // pre-вклад не пишется — будущий кадр сам поднимет свой порог
        let mut tmp = vec![0.0f64; self.n_slots];
        for j in 0..sub {
            let slots_j = self.amps_to_slots(fb, &sqrt_bands(&t_post[j]));
            for s in 0..self.n_slots {
                if tmp[s] < slots_j[s] {
                    tmp[s] = slots_j[s];
                }
            }
        }
        state.slots.copy_from_slice(&tmp);
        out
    }


    /// Зафиксировать кадр: `band_amp` — максимум порогов полос по
    /// подкадрам кадра (перезапись слотов, decay применяется при чтении).
    pub fn commit(&self, ftype: FrameType, band_amp: &[f64], state: &mut PsyState) {
        let fb = &self.ft[ftype.idx()];
        let mut tmp = vec![0.0f64; self.n_slots];
        for (p, &(lo, hi)) in fb.slot_range.iter().enumerate() {
            let amp = band_amp[p];
            for s in lo..hi.min(self.n_slots) {
                tmp[s] = tmp[s].max(amp);
            }
        }
        state.slots.copy_from_slice(&tmp);
    }
}

/// Полосы ~1 барк: z_i = bark(f_i), полоса закрывается при смене floor(z);
/// крайние полосы могут быть уже/шире барка (бины не делятся).
fn build_bands(b: usize, sr: u32, norm: f64, params: &PsyParams, n_slots: usize) -> FtypeBands {
    let bin_hz = sr as f64 / (2.0 * b as f64);
    let z: Vec<f64> = (0..b).map(|i| bark_of((i as f64 + 0.5) * bin_hz)).collect();
    let f: Vec<f64> = (0..b).map(|i| (i as f64 + 0.5) * bin_hz).collect();

    let mut edges = vec![0usize];
    for i in 1..b {
        if z[i].floor() > z[edges.last().copied().unwrap_or(0)].floor() {
            edges.push(i);
        }
    }
    edges.push(b);

    let n = edges.len() - 1;
    let center_bark: Vec<f64> = (0..n)
        .map(|p| {
            let (lo, hi) = (edges[p], edges[p + 1]);
            z[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect();

    let fs_ref = fullscale_ref(b, norm);
    let ath_amp: Vec<f64> = (0..n)
        .map(|p| {
            let (lo, hi) = (edges[p], edges[p + 1]);
            let fc = f[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            fs_ref * 10f64.powf((ath_db(fc) - 96.0 + params.ath_offset_db) / 10.0)
        })
        .collect();

    let slot_range: Vec<(usize, usize)> = (0..n)
        .map(|p| {
            let (lo, hi) = (edges[p], edges[p + 1]);
            let z0 = z[lo] * 2.0;
            let z1 = z[hi - 1] * 2.0 + 1.0;
            ((z0.floor() as usize).min(n_slots), (z1.ceil() as usize).min(n_slots).max(1))
        })
        .collect();

    FtypeBands { b, edges, center_bark, ath_amp, slot_range }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_44_mono(params: PsyParams) -> PsyModel {
        let info = StreamInfo::new(44, 48, 1).unwrap();
        PsyModel::new(&info, params)
    }

    /// Полоса моно-спектра LONG 44_48 по частоте (бина 10.77 Гц).
    fn band_of_freq(model: &PsyModel, f_hz: f64) -> usize {
        let fb = &model.ft[FrameType::Long.idx()];
        let b = fb.edges[fb.edges.len() - 1] - 1; // size
        let bin = (f_hz / (44100.0 / (2.0 * b as f64))) as usize;
        (0..fb.edges.len() - 1)
            .find(|&p| bin >= fb.edges[p] && bin < fb.edges[p + 1])
            .unwrap()
    }

    #[test]
    fn fullscale_sine_peak() {
        // калибровка SPL: c·mdct_fwd(синус A=32767 в центре бина)
        // имеет пик ≈ fullscale_ref(b, norm)
        for &(b, norm) in &[(2048usize, 2.0f64), (512, 1.0), (128, 2.0)] {
            let k = b / 4;
            let pi = std::f64::consts::PI;
            let x: Vec<f64> = (0..2 * b)
                .map(|j| {
                    // косинус, выровненный с базисом MDCT (внутри +B)
                    32767.0
                        * (((2 * j + 1 + b) as f64) * ((2 * k + 1) as f64) * pi
                            / (4.0 * b as f64))
                            .cos()
                })
                .collect();
            let mdct = crate::dsp::mdct_fwd(&x, b);
            let c = 65536.0 / (norm * b as f64).sqrt();
            let peak = mdct.iter().map(|v| c * v).fold(0.0f64, f64::max);
            let ref_ = fullscale_ref(b, norm);
            assert!(
                (peak - ref_).abs() / ref_ < 0.01,
                "b={b}: peak={peak:.4e} ref={ref_:.4e}"
            );
        }
    }

    #[test]
    fn mask_below_masker_and_positive() {
        let model = model_44_mono(PsyParams::default());
        let mut state = model.new_state();
        let b = 2048usize;
        // энергия только в полосе ~1 кГц (амплитуда FS)
        let mut spec = vec![0.0f64; b];
        let bin = (1000.0 / (44100.0 / (2.0 * b as f64))) as usize;
        spec[bin] = fullscale_ref(b, 2.0);
        let r = model.mask(FrameType::Long, &spec, &state.slots);
        let p = band_of_freq(&model, 1000.0);
        let band_e = fullscale_ref(b, 2.0).powi(2);
        // порог ниже энергии маскера (OBC ≥ NMT=4 дБ)
        assert!(r.band_amp[p].powi(2) < band_e * 10f64.powf(-4.0 / 10.0));
        assert!(r.band_amp[p] > 0.0);
        // и весь ненулевой спектр выше ATH
        for p in 0..r.band_amp.len() {
            assert!(r.band_amp[p] > 0.0);
        }
        model.commit(FrameType::Long, &r.band_amp, &mut state);
    }

    #[test]
    fn spreading_asymmetric() {
        // маскер на ~500 Гц: порог на 3 барка ВЫШЕ маскера больше,
        // чем на 3 барка ниже (up-склон положе down-склона)
        let model = model_44_mono(PsyParams::default());
        let state = model.new_state();
        let b = 2048usize;
        let mut spec = vec![0.0f64; b];
        let bin = (500.0 / (44100.0 / (2.0 * b as f64))) as usize;
        spec[bin] = fullscale_ref(b, 2.0);
        let r = model.mask(FrameType::Long, &spec, &state.slots);
        let pm = band_of_freq(&model, 500.0);
        let up = (pm + 3).min(r.band_amp.len() - 1);
        let dn = pm.saturating_sub(3);
        assert!(
            r.band_amp[up] > r.band_amp[dn],
            "up({})={:.3e} dn({})={:.3e}",
            up, r.band_amp[up], dn, r.band_amp[dn]
        );
    }

    #[test]
    fn tonality_bounds() {
        assert!(tonality(&[1.0; 25]) < 0.05); // шум → α≈0
        let mut sine = vec![1e-9; 25];
        sine[7] = 1.0;
        assert!(tonality(&sine) > 0.95); // тон → α≈1
        assert_eq!(tonality(&[0.0; 25]), 0.0); // тишина
    }

    #[test]
    fn ath_floor_on_silence() {
        let model = model_44_mono(PsyParams::default());
        let state = model.new_state();
        let r = model.mask(FrameType::Long, &vec![0.0; 2048], &state.slots);
        // на тишине порог = ATH (min-пол даже при нулевом spreading)
        for (p, &amp) in r.band_amp.iter().enumerate() {
            let fb = &model.ft[FrameType::Long.idx()];
            assert!((amp - fb.ath_amp[p]).abs() < 1e-6 * amp.max(1.0));
        }
        // ATH имеет минимум в районе 2–5 кГц и растёт к НЧ
        let lo = r.band_amp[band_of_freq(&model, 100.0)];
        let mid = r.band_amp[band_of_freq(&model, 3000.0)];
        assert!(lo > 100.0 * mid, "ATH(100Гц)={lo:.3e} vs ATH(3кГц)={mid:.3e}");
    }

    #[test]
    fn post_masking_release() {
        let model = model_44_mono(PsyParams::default());
        let mut state = model.new_state();
        let b = 2048usize;
        // громкий кадр → commit
        let mut spec = vec![0.0f64; b];
        let bin = (1000.0 / (44100.0 / (2.0 * b as f64))) as usize;
        spec[bin] = fullscale_ref(b, 2.0);
        let r = model.mask(FrameType::Long, &spec, &state.slots);
        model.commit(FrameType::Long, &r.band_amp, &mut state);
        let p = band_of_freq(&model, 1000.0);
        let loud = r.band_amp[p];

        // тихий кадр: порог = max(ATH, prev·10^(−post/10)) — маскировка
        // «держится» один кадр и спадает ровно на post_mask_db
        let r2 = model.mask(FrameType::Long, &vec![0.0; b], &state.slots);
        let expected = loud * 10f64.powf(-PsyParams::default().post_mask_db / 20.0);
        assert!(
            (r2.band_amp[p] - expected).abs() < 1e-9 * expected,
            "{:.6e} vs {:.6e}",
            r2.band_amp[p],
            expected
        );
    }

    #[test]
    fn short_frame_bands_cover_all_bins() {
        // SHORT 44_48: b=128, полосы грубее, но покрывают все бины ≥1
        let model = model_44_mono(PsyParams::default());
        let fb = &model.ft[FrameType::Short.idx()];
        assert_eq!(fb.edges[0], 0);
        assert_eq!(*fb.edges.last().unwrap(), 128);
        for w in fb.edges.windows(2) {
            assert!(w[1] > w[0]);
        }
        let zeros = vec![0.0; model.n_slots()];
        let r = model.mask(FrameType::Short, &vec![0.0; 128], &zeros);
        assert_eq!(r.m.len(), 128);
    }

    #[test]
    fn pre_masking_lookahead() {
        // SHORT-кадр: тишина, на подкадре 8 — громкий тон. Pre-masking
        // поднимает порог подкадров ПЕРЕД атакой (в окне pre_mask_ms),
        // дальние подкадры — нет; при pre_mask_ms=0 эффекта нет.
        let mk = |pre_ms| {
            let p = PsyParams { pre_mask_ms: pre_ms, ..PsyParams::default() };
            model_44_mono(p)
        };
        let b = 128usize;
        let fs = fullscale_ref(b, 2.0);
        let mk_specs = || {
            (0..16usize)
                .map(|j| {
                    let mut s = vec![0.0f64; b];
                    if j == 8 {
                        s[10] = fs; // ~3.4 кГц
                    }
                    s
                })
                .collect::<Vec<Vec<f64>>>()
        };
        let specs_owned = mk_specs();
        let specs: Vec<&[f64]> = specs_owned.iter().map(|v| v.as_slice()).collect();

        let mut st = mk(20.0).new_state();
        let rs = mk(20.0).mask_frame(FrameType::Short, &specs, None, &mut st);
        // полоса тона: выше у подкадра 7 (2.9 мс до атаки), чем у 0 (23 мс)
        let fb = &mk(20.0).ft[FrameType::Short.idx()];
        let bin = (3430.0 / (44100.0 / (2.0 * b as f64))) as usize;
        let band = (0..fb.edges.len() - 1)
            .find(|&p| bin >= fb.edges[p] && bin < fb.edges[p + 1])
            .unwrap();
        assert!(
            rs[7].band_amp[band] > rs[0].band_amp[band] * 3.0,
            "pre: j=7 {:.3e} vs j=0 {:.3e}",
            rs[7].band_amp[band],
            rs[0].band_amp[band]
        );
        // за окном (j=0, 23 мс до атаки) — ровно ATH
        let ath = fb.ath_amp[band];
        assert!((rs[0].band_amp[band] - ath).abs() < 1e-6 * ath);

        // выключено — все тихие подкадры одинаковые
        let mut st0 = mk(0.0).new_state();
        let rs0 = mk(0.0).mask_frame(FrameType::Short, &specs, None, &mut st0);
        assert!((rs0[7].band_amp[band] - rs0[0].band_amp[band]).abs() < 1e-9);

        // commit post-состояния: громкий подкадр поднял слоты
        let loud_amp = rs[8].band_amp[band];
        assert!(st.slots.iter().cloned().fold(0.0f64, f64::max) > 0.9 * loud_amp);
    }
}
