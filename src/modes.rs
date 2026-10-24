//! Режимы TwinVQ: типы кадров, поиск режима, wtype-маппинги.
//!
//! Логика — cleanroom (по поведению FFmpeg twinvqdec.c/twinvq.c);
//! значения таблиц — vendored-данные формата NTT (`src/vendor/`).

/// Тип кадра (FFmpeg `TwinVQFrameType`).
///
/// `Short`/`Medium`/`Long` — типы кадров (индексируют `ModeTab::fmode`);
/// `Ppc` — служебный «ftype» для PPC-кодбука (кадром не является).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameType {
    Short = 0,
    Medium = 1,
    Long = 2,
    Ppc = 3,
}

impl FrameType {
    pub const ALL: [FrameType; 4] =
        [FrameType::Short, FrameType::Medium, FrameType::Long, FrameType::Ppc];

    /// Индекс в `ModeTab::fmode` (для `Ppc` — индекс раскладки PPC cb).
    pub const fn idx(self) -> usize {
        self as usize
    }
}

/// Параметры одного ftype (FFmpeg `TwinVQFrameMode`).
#[derive(Clone, Copy)]
pub struct FrameMode {
    /// Число подкадров (SHORT ∈ {4,8,16}, MEDIUM ∈ {2,4}, LONG = 1).
    pub sub: usize,
    /// Ширины bark-полос (сумма первых `bark_env_size` == size/sub).
    pub bark_tab: &'static [u16],
    /// Число значений bark-огибающей.
    pub bark_env_size: usize,
    /// Кодбук bark-огибающей (2^bark_n_bit входов).
    pub bark_cb: &'static [i16],
    /// Число коэффициентов кодбука, читаемых на подкадр.
    pub bark_n_coef: usize,
    /// Бит на коэффициент bark-кодбука.
    pub bark_n_bit: usize,
    /// Основной кодбук A (conjugate VQ).
    pub cb0: &'static [i16],
    /// Основной кодбук B (conjugate VQ).
    pub cb1: &'static [i16],
    /// Размерность вектора кодбука (входов: len/cb_len_read).
    pub cb_len_read: usize,
}

/// Таблица режима (FFmpeg `TwinVQModeTab`).
pub struct ModeTab {
    /// [Short, Medium, Long].
    pub fmode: [FrameMode; 3],
    /// Сэмплов на кадр (на канал).
    pub size: usize,
    /// Число LSP-коэффициентов.
    pub n_lsp: usize,
    pub lspcodebook: &'static [f32],
    pub lsp_bit0: usize,
    pub lsp_bit1: usize,
    pub lsp_bit2: usize,
    pub lsp_split: usize,
    pub ppc_shape_cb: &'static [i16],
    pub ppc_period_bit: usize,
    pub ppc_shape_bit: usize,
    pub ppc_shape_len: usize,
    pub pgain_bit: usize,
    pub peak_per2wid: usize,
}

/// Число допустимых wtype (0..=8).
pub const NUM_WTYPES: usize = 9;

/// wtype → ftype (FFmpeg `ff_twinvq_wtype_to_ftype_table`).
pub const WTYPE_TO_FTYPE: [FrameType; NUM_WTYPES] = [
    FrameType::Long,
    FrameType::Long,
    FrameType::Short,
    FrameType::Long,
    FrameType::Medium,
    FrameType::Long,
    FrameType::Long,
    FrameType::Medium,
    FrameType::Medium,
];

/// wtype → индекс в `types_sizes[]` (FFmpeg `wtype_to_wsize`).
pub const WTYPE_TO_WSIZE_IDX: [usize; NUM_WTYPES] = [0, 0, 2, 2, 2, 1, 0, 1, 1];

/// Поиск режима: `isampf` = кГц (8/11/16/22/44), `ibps` = кбит/с **на канал**.
///
/// Стерео-режимы VQF (22s40/22s48/44s96) не имеют собственных таблиц —
/// это те же режимы с n_ch=2 (например, 44s96 → 44_48 × 2 канала).
/// Возвращает `None` для комбинаций, отсутствующих в VQF-профиле FFmpeg.
pub fn find_mode(isampf: i32, ibps: i32) -> Option<&'static ModeTab> {
    use crate::vendor;
    Some(match (isampf, ibps) {
        (8, 8) => &vendor::MODE_08_08,
        (11, 8) => &vendor::MODE_11_08,
        (11, 10) => &vendor::MODE_11_10,
        (16, 16) => &vendor::MODE_16_16,
        (22, 20) => &vendor::MODE_22_20,
        (22, 24) => &vendor::MODE_22_24,
        (22, 32) => &vendor::MODE_22_32,
        (44, 40) => &vendor::MODE_44_40,
        (44, 48) => &vendor::MODE_44_48,
        _ => return None,
    })
}

/// Тип окна: wtype → ftype. Паникует при wtype > 8 (невалидный wtype —
/// ошибка вызывающего; декодер FFmpeg такие кадры отвергает).
pub fn wtype_to_ftype(wtype: usize) -> FrameType {
    WTYPE_TO_FTYPE[wtype]
}

/// Индекс ширины окна синтеза: wtype → 0/1/2 (`types_sizes`).
pub fn wtype_to_wsize_idx(wtype: usize) -> usize {
    WTYPE_TO_WSIZE_IDX[wtype]
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [(i32, i32); 9] = [
        (8, 8), (11, 8), (11, 10), (16, 16), (22, 20),
        (22, 24), (22, 32), (44, 40), (44, 48),
    ];

    #[test]
    fn all_nine_modes_found() {
        for &(s, b) in &ALL {
            assert!(find_mode(s, b).is_some(), "режим {s}/{b} не найден");
        }
        // 96 кбит/с на канал и 8 кГц/16 кбит/с в VQF-профиле отсутствуют
        assert!(find_mode(44, 96).is_none());
        assert!(find_mode(8, 16).is_none());
    }

    #[test]
    fn mode_invariants() {
        for &(s, b) in &ALL {
            let m = find_mode(s, b).unwrap();
            assert_eq!(m.fmode[FrameType::Long.idx()].sub, 1, "{s}/{b}");

            let bs = m.size / m.fmode[FrameType::Short.idx()].sub;
            let bm = m.size / m.fmode[FrameType::Medium.idx()].sub;
            let bl = m.size / m.fmode[FrameType::Long.idx()].sub;
            assert_eq!(bl, m.size);
            assert!(bs < bm && bm <= bl, "{s}/{b}: {bs} < {bm} <= {bl}");

            for ft in [FrameType::Short, FrameType::Medium, FrameType::Long] {
                let fm = &m.fmode[ft.idx()];
                let bsize = m.size / fm.sub;
                // bark-огибающая покрывает блок целиком
                let sum: u32 = fm.bark_tab[..fm.bark_env_size].iter().map(|&v| v as u32).sum();
                assert_eq!(sum as usize, bsize, "{s}/{b} ftype {ft:?}");
                // bark_env_size делится на bark_n_coef (fw_cb_len)
                assert_eq!(fm.bark_env_size % fm.bark_n_coef, 0);
                // bark-кодбук: 2^bark_n_bit входов размерности env/ncoef
                let dim = fm.bark_env_size / fm.bark_n_coef;
                assert_eq!(fm.bark_cb.len(), (1usize << fm.bark_n_bit) * dim,
                           "{s}/{b} ftype {ft:?}");
                // основной кодбук: целое число входов размерности cb_len_read
                assert_eq!(fm.cb0.len() % fm.cb_len_read, 0);
                assert_eq!(fm.cb1.len() % fm.cb_len_read, 0);
                assert_eq!(fm.cb0.len(), fm.cb1.len());
            }

            // LSP-книга согласована с разбиением на подвекторы
            assert_eq!(m.lspcodebook.len() % m.n_lsp, 0, "{s}/{b}");
        }
    }

    #[test]
    fn wtype_tables() {
        assert_eq!(wtype_to_ftype(0), FrameType::Long);
        assert_eq!(wtype_to_ftype(1), FrameType::Long);
        assert_eq!(wtype_to_ftype(2), FrameType::Short);
        assert_eq!(wtype_to_ftype(3), FrameType::Long);
        assert_eq!(wtype_to_ftype(4), FrameType::Medium);
        assert_eq!(wtype_to_ftype(5), FrameType::Long);
        assert_eq!(wtype_to_ftype(6), FrameType::Long);
        assert_eq!(wtype_to_ftype(7), FrameType::Medium);
        assert_eq!(wtype_to_ftype(8), FrameType::Medium);
        assert_eq!(wtype_to_wsize_idx(0), 0);
        assert_eq!(wtype_to_wsize_idx(2), 2);
        assert_eq!(wtype_to_wsize_idx(5), 1);
        assert_eq!(wtype_to_wsize_idx(8), 1);
    }

    /// Spot-значения против twinvq_data.h (первые 2 значения LONG/SHORT
    /// основных кодбуков каждого режима) — защита от перепутанных ссылок.
    #[test]
    fn spot_codebook_values() {
        // (isampf, ibps, LONG cb0[0..2], SHORT cb0[0..2])
        let spots: &[(i32, i32, [i16; 2], [i16; 2])] = &[
            (8, 8,    [96, -12592],   [-7488, -1327]),
            (11, 8,   [2354, 8016],   [-32768, -828]),
            (11, 10,  [-14944, -14950], [-32746, 360]),
            (16, 16,  [-185, -20290], [5604, 1491]),
            (22, 20,  [-12528, 350],  [-15119, 7508]),
            (22, 24,  [-12451, 389],  [-27522, 2628]),
            (22, 32,  [-9947, -673],  [-26218, 1606]),
            (44, 40,  [-14497, -1982], [-12085, 8192]),
            (44, 48,  [-15402, -5156], [-10720, 9997]),
        ];
        for &(s, b, l, sh) in spots {
            let m = find_mode(s, b).unwrap();
            let long = &m.fmode[FrameType::Long.idx()];
            let short = &m.fmode[FrameType::Short.idx()];
            assert_eq!(&long.cb0[..2], &l, "{s}/{b} LONG");
            assert_eq!(&short.cb0[..2], &sh, "{s}/{b} SHORT");
        }
    }

    /// Spot-значения LSP-книг против metasound_twinvq_data.h / lsp22_2.
    #[test]
    fn spot_lsp_values() {
        let m8 = find_mode(8, 8).unwrap();
        for (i, &e) in [0.2702f32, 0.5096, 0.6437].iter().enumerate() {
            assert!((m8.lspcodebook[i] - e).abs() < 1e-6);
        }
        let m44 = find_mode(44, 48).unwrap();
        for (i, &e) in [0.0927f32, 0.2291, 0.4059, 0.5779].iter().enumerate() {
            assert!((m44.lspcodebook[i] - e).abs() < 1e-6);
        }
        let m2232 = find_mode(22, 32).unwrap();
        for (i, &e) in [0.0712f32, 0.1830, 0.4167].iter().enumerate() {
            assert!((m2232.lspcodebook[i] - e).abs() < 1e-6);
        }
    }

    /// very_broken_op: таблица выбора (twinvq_data.h tabs[]).
    #[test]
    fn broken_op_tabs() {
        let t = &crate::vendor::cb_data::BROKEN_OP_TABS;
        assert_eq!(t.len(), 13);
        assert_eq!(t[0], (0, 0)); // NULL
        assert_eq!(t[3], (15, 12));
        assert_eq!(t[5], (25, 10));
        assert_eq!(t[7], (35, 7));
        assert_eq!(t[9], (45, 9));
        assert_eq!(t[11], (55, 11));
        assert_eq!(t[12], (15, 12));
    }
}
