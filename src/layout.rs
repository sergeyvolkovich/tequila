//! Битовый бюджет кадра — порт `init_bitstream_params` (FFmpeg twinvq.c).
//!
//! Формулы выверены против бит-точного python-оракула
//! (`python/reference/vqflib.py`, валидирован на 218 эталонных потоках NTT)
//! и проверяются тестами против `python/analysis/results/bit_budget.json`
//! (все 18 комбинаций режим × каналы; docs/ARCHITECTURE.md §7).
//!
//! Примечание:quirk `bsize_no_main_cb[1..2] += 2` из FFmpeg относится
//! только к MetaSound-кодеку и здесь не используется.

use crate::modes::{FrameType, ModeTab};

const WTYPE_BITS: usize = 4; // TWINVQ_WINDOW_TYPE_BITS
const GAIN_BITS: usize = 8; // TWINVQ_GAIN_BITS
const SUB_GAIN_BITS: usize = 5; // TWINVQ_SUB_GAIN_BITS

/// Раскладка последовательности векторов по «дивизиям» (n_div векторов,
/// кодируемых парами кодбуков cb0/cb1 — main VQ или PPC shape cb).
#[derive(Clone, Copy, Debug)]
pub struct DivLayout {
    /// Всего бит на слой (для main VQ = total − прочие поля кадра).
    pub bit_size: usize,
    /// Всего коэффициентов (n_ch × size; для PPC — n_ch × shape_len).
    pub vect_size: usize,
    /// Число дивизий.
    pub n_div: usize,
    /// Бит индекса книги cb (0/1) для дивизии j: `bits[cb][j >= change]`.
    ///
    /// Первые `change` дивизий используют округлённые вверх суммы битов
    /// (по 2 книги на дивизию), остальные — вниз.
    pub bits: [[usize; 2]; 2],
    /// bits_main_spec_change.
    pub change: usize,
    /// Размер вектора дивизии j: `length[j >= length_change]`.
    pub length: [usize; 2],
    /// length_change.
    pub length_change: usize,
}

impl DivLayout {
    /// Биты индексов (cb0, cb1) для дивизии `j`.
    pub fn bits_at(&self, j: usize) -> (usize, usize) {
        let k = usize::from(j >= self.change);
        (self.bits[0][k], self.bits[1][k])
    }

    /// Размер вектора дивизии `j`.
    pub fn length_at(&self, j: usize) -> usize {
        self.length[usize::from(j >= self.length_change)]
    }
}

/// Битовый бюджет кадра для комбинации (режим, число каналов).
#[derive(Clone, Copy, Debug)]
pub struct FrameLayout {
    /// total_fr_bits = kbps_total · 1000 · size / sr (усечение).
    pub total_bits: usize,
    /// LSP-поля кадра (все каналы).
    pub lsp_bits: usize,
    /// PPC-поля кадра: shape cb + период + гейн (все каналы, LONG).
    pub ppc_bits: usize,
    /// bark1 + hist бит **на подкадр** (все каналы) для каждого ftype.
    pub bse_bits: [usize; 3],
    /// Раскладка main VQ: [Short, Medium, Long].
    pub ftype: [DivLayout; 3],
    /// Раскладка PPC shape cb.
    pub ppc: DivLayout,
}

impl FrameLayout {
    /// Раскладка по ftype (Short/Medium/Long → ftype[], Ppc → ppc).
    pub fn div(&self, ftype: FrameType) -> &DivLayout {
        if ftype == FrameType::Ppc {
            &self.ppc
        } else {
            &self.ftype[ftype.idx()]
        }
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Ядро init_bitstream_params: раскладка бит/длин по дивизиям.
fn div_layout(bit_size: usize, vect_size: usize) -> DivLayout {
    let n_div = (bit_size + 13) / 14;

    // Биты: первые nru дивизий — rounded_up, остальные — rounded_down.
    let r_up = ceil_div(bit_size, n_div);
    let r_dn = bit_size / n_div;
    let nrd = r_up.saturating_mul(n_div).saturating_sub(bit_size); // дивизий «вниз»
    let change = n_div - nrd; // bits_main_spec_change = num_rounded_up

    // Длины векторов — аналогично.
    let l_up = ceil_div(vect_size, n_div);
    let l_dn = vect_size / n_div;
    let vnrd = l_up.saturating_mul(n_div).saturating_sub(vect_size);
    let length_change = n_div - vnrd;

    DivLayout {
        bit_size,
        vect_size,
        n_div,
        // bits_main_spec[cb][j >= change]: cb0 — ceil, cb1 — floor
        bits: [[ceil_div(r_up, 2), ceil_div(r_dn, 2)], [r_up / 2, r_dn / 2]],
        change,
        length: [l_up, l_dn],
        length_change,
    }
}

/// Вычисляет битовый бюджет кадра.
///
/// * `m` — таблица режима (`find_mode`);
/// * `n_ch` — число каналов (1/2);
/// * `kbps_total` — **суммарный** битрейт, кбит/с (для 44s96 = 96);
/// * `sr` — частота дискретизации, Гц (8000/11025/16000/22050/44100).
pub fn frame_layout(m: &ModeTab, n_ch: usize, kbps_total: usize, sr: usize) -> FrameLayout {
    let total_bits = kbps_total * 1000 * m.size / sr;

    let lsp_bits = n_ch * (m.lsp_bit0 + m.lsp_bit1 + m.lsp_split * m.lsp_bit2);
    let ppc_bits = n_ch * (m.pgain_bit + m.ppc_shape_bit + m.ppc_period_bit);

    let mut bse_bits = [0usize; 3];
    let mut bsize_no_main = [0usize; 3];
    for i in 0..3 {
        let fm = &m.fmode[i];
        // +1 — бит истории (bark_use_hist)
        bse_bits[i] = n_ch * (fm.bark_n_coef * fm.bark_n_bit + 1);
        bsize_no_main[i] = if i == FrameType::Long.idx() {
            bse_bits[i] + lsp_bits + ppc_bits + WTYPE_BITS + n_ch * GAIN_BITS
        } else {
            lsp_bits + n_ch * GAIN_BITS + WTYPE_BITS
                + fm.sub * (bse_bits[i] + n_ch * SUB_GAIN_BITS)
        };
    }

    let mut ftype = [DivLayout {
        bit_size: 0, vect_size: 0, n_div: 0,
        bits: [[0; 2]; 2], change: 0, length: [0; 2], length_change: 0,
    }; 3];
    for i in 0..3 {
        ftype[i] = div_layout(total_bits - bsize_no_main[i], n_ch * m.size);
    }
    let ppc = div_layout(n_ch * m.ppc_shape_bit, n_ch * m.ppc_shape_len);

    FrameLayout { total_bits, lsp_bits, ppc_bits, bse_bits, ftype, ppc }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::find_mode;

    /// Оракул: python/analysis/results/bit_budget.json (бит-точный порт,
    /// сверен с 218 эталонными потоками NTT).
    ///
    /// (isampf, ibps, n_ch, sr, total, [main SHORT, MEDIUM, LONG],
    ///  ppc LONG, lsp, bse LONG = bark1+hist)
    const BUDGET: &[(i32, i32, usize, usize, usize, [usize; 3], usize, usize, usize)] = &[
        (8, 8, 1, 8000, 512, [397, 453, 424], 42, 15, 19),
        (8, 8, 2, 8000, 1024, [798, 910, 852], 84, 30, 38),
        (11, 8, 1, 11025, 371, [252, 308, 269], 52, 19, 19),
        (11, 8, 2, 11025, 743, [509, 621, 543], 104, 38, 38),
        (11, 10, 1, 11025, 464, [345, 401, 362], 52, 19, 19),
        (11, 10, 2, 11025, 928, [694, 806, 728], 104, 38, 38),
        (16, 16, 1, 16000, 1024, [905, 961, 902], 72, 19, 19),
        (16, 16, 2, 16000, 2048, [1814, 1926, 1808], 144, 38, 38),
        (22, 20, 1, 22050, 928, [801, 861, 800], 72, 19, 25),
        (22, 20, 2, 22050, 1857, [1607, 1727, 1605], 144, 38, 50),
        (22, 24, 1, 22050, 1114, [987, 1047, 986], 72, 19, 25),
        (22, 24, 2, 22050, 2229, [1979, 2099, 1977], 144, 38, 50),
        (22, 32, 1, 22050, 743, [660, 672, 611], 72, 23, 25),
        (22, 32, 2, 22050, 1486, [1324, 1348, 1226], 144, 46, 50),
        (44, 40, 1, 44100, 1857, [1630, 1750, 1697], 100, 23, 25),
        (44, 40, 2, 44100, 3715, [3265, 3505, 3399], 200, 46, 50),
        (44, 48, 1, 44100, 2229, [2002, 2122, 2069], 100, 23, 25),
        (44, 48, 2, 44100, 4458, [4008, 4248, 4142], 200, 46, 50),
    ];

    /// Самосогласованность дивизий: суммы бит и длин по всем j.
    fn check_div(d: &DivLayout, ctx: &str) {
        let mut bits_sum = 0;
        let mut len_sum = 0;
        for j in 0..d.n_div {
            let (b0, b1) = d.bits_at(j);
            bits_sum += b0 + b1;
            len_sum += d.length_at(j);
        }
        assert_eq!(bits_sum, d.bit_size, "{ctx}: биты дивизий");
        assert_eq!(len_sum, d.vect_size, "{ctx}: длины дивизий");
        // длины не могут «переворачиваться»
        assert!(d.length[0] >= d.length[1], "{ctx}");
        assert!(d.bits[0][0] + d.bits[1][0] >= d.bits[0][1] + d.bits[1][1], "{ctx}");
    }

    #[test]
    fn budget_matches_python_oracle() {
        for &(isampf, ibps, n_ch, sr, total, mains, ppc, lsp, bse_long) in BUDGET {
            let ctx = format!("{isampf}/{ibps} x{n_ch}");
            let m = find_mode(isampf, ibps).unwrap();
            let l = frame_layout(m, n_ch, ibps as usize * n_ch, sr);

            assert_eq!(l.total_bits, total, "{ctx}: total");
            for (i, &main) in mains.iter().enumerate() {
                assert_eq!(l.ftype[i].bit_size, main, "{ctx}: main ftype {i}");
            }
            assert_eq!(l.ppc_bits, ppc, "{ctx}: ppc");
            assert_eq!(l.lsp_bits, lsp, "{ctx}: lsp");
            assert_eq!(l.bse_bits[FrameType::Long.idx()], bse_long, "{ctx}: bse LONG");

            for (i, d) in l.ftype.iter().enumerate() {
                check_div(d, &format!("{ctx} ftype {i}"));
                // кодбук покрывает максимальный вектор дивизии
                assert!(m.fmode[i].cb_len_read >= d.length[0],
                        "{ctx} ftype {i}: cb_len_read {} < length[0] {}",
                        m.fmode[i].cb_len_read, d.length[0]);
            }
            check_div(&l.ppc, &format!("{ctx} ppc"));
        }
    }

    /// Полная раскладка одного кадра 44.1 кГц / 48 кбит/с / моно
    /// (ручной пересчёт формул FFmpeg; см. docs/ARCHITECTURE.md §7a).
    #[test]
    fn div_layout_44_48_long_hand_check() {
        let m = find_mode(44, 48).unwrap();
        let l = frame_layout(m, 1, 48, 44100);

        // LONG: main 2069 бит, 2048 коэффициентов
        let d = &l.ftype[FrameType::Long.idx()];
        assert_eq!(d.bit_size, 2069);
        assert_eq!(d.vect_size, 2048);
        assert_eq!(d.n_div, 148);
        // 145 дивизий по 14 бит (7+7), 3 — по 13 (7+6): 145*14+3*13 = 2069
        assert_eq!(d.change, 145);
        assert_eq!(d.bits, [[7, 7], [7, 6]]);
        // 124 вектора по 14, 24 — по 13: 124*14+24*13 = 2048
        assert_eq!(d.length, [14, 13]);
        assert_eq!(d.length_change, 124);
        assert_eq!(d.bits_at(0), (7, 7));
        assert_eq!(d.bits_at(144), (7, 7));
        assert_eq!(d.bits_at(145), (7, 6));
        assert_eq!(d.length_at(0), 14);
        assert_eq!(d.length_at(123), 14);
        assert_eq!(d.length_at(124), 13);

        // PPC shape cb: 84 бита, 54 значения
        let p = &l.ppc;
        assert_eq!(p.bit_size, 84);
        assert_eq!(p.vect_size, 54);
        assert_eq!(p.n_div, 6);
        assert_eq!(p.length, [9, 9]);

        // Компоненты LONG-кадра: wtype+bark+gain+ppc+lsp+main = total
        let no_main = l.bse_bits[FrameType::Long.idx()] + l.lsp_bits + l.ppc_bits
            + WTYPE_BITS + 1 * GAIN_BITS;
        assert_eq!(no_main + d.bit_size, l.total_bits); // 160 + 2069 = 2229
        assert_eq!(l.total_bits, 2229);
    }

    /// PPC-гребёнка: диапазон f0 (FFmpeg twinvqdec.c decode_ppc:
    /// min/max_period = ROUNDED_DIV(40·2·size·{1,6}, isampf),
    /// f0 = period·sr/(800·size)). Эмпирика ENCODE_PIPELINE.md §4.5:
    /// f0 ∈ [100, 600] Гц одинаково для всех режимов
    /// (оракул: python/analysis/results/encoder_probe.json ppc_range).
    #[test]
    fn ppc_period_ranges() {
        for &(isampf, ibps, _n_ch, sr, ..) in BUDGET.iter().step_by(2) {
            let m = find_mode(isampf, ibps).unwrap();
            let rounded_div = |a: usize, b: usize| (a + b / 2) / b;
            let min_p = rounded_div(40 * 2 * m.size, isampf as usize);
            let max_p = rounded_div(40 * 2 * m.size * 6, isampf as usize);
            let f0 = |p: usize| p as f64 * sr as f64 / (800.0 * m.size as f64);
            let f0_min = f0(min_p);
            let f0_max = f0(max_p);
            assert!((100.0..=100.3).contains(&f0_min), "{isampf}/{ibps}: f0_min={f0_min}");
            assert!((599.9..=601.5).contains(&f0_max), "{isampf}/{ibps}: f0_max={f0_max}");
        }
    }
}
