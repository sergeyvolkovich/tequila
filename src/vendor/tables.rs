//! Таблицы 9 режимов VQF — перенос `mode_*` из FFmpeg twinvqdec.c.
//!
//! Логика сопоставления — cleanroom; значения таблиц — данные формата
//! TwinVQ (NTT), LGPL-2.1-or-later (см. заголовок модуля `vendor`).
//!
//! Сверка с оригиналом: cargo test (spot-значения кодбуков, инварианты
//! bark-таблиц, битовый бюджет против python/analysis/results/bit_budget.json).

use super::cb_data::*;
use super::lsp_data::*;
use crate::modes::{FrameMode, ModeTab};

pub static MODE_08_08: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S08_64,  bark_env_size: 10, bark_cb: &FCB08S, bark_n_coef: 1, bark_n_bit: 5, cb0: &CB0808S0, cb1: &CB0808S1, cb_len_read: 18 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M08_256, bark_env_size: 20, bark_cb: &FCB08M, bark_n_coef: 2, bark_n_bit: 5, cb0: &CB0808M0, cb1: &CB0808M1, cb_len_read: 16 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L08_512, bark_env_size: 30, bark_cb: &FCB08L, bark_n_coef: 3, bark_n_bit: 6, cb0: &CB0808L0, cb1: &CB0808L1, cb_len_read: 17 },
    ],
    size: 512,
    n_lsp: 12,
    lspcodebook: &FF_METASOUND_LSP8,
    lsp_bit0: 1, lsp_bit1: 5, lsp_bit2: 3, lsp_split: 3,
    ppc_shape_cb: &SHAPE08,
    ppc_period_bit: 8, ppc_shape_bit: 28, ppc_shape_len: 20, pgain_bit: 6, peak_per2wid: 40,
};

pub static MODE_11_08: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S11_64,  bark_env_size: 10, bark_cb: &FCB11S, bark_n_coef: 1, bark_n_bit: 5, cb0: &CB1108S0, cb1: &CB1108S1, cb_len_read: 29 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M11_256, bark_env_size: 20, bark_cb: &FCB11M, bark_n_coef: 2, bark_n_bit: 5, cb0: &CB1108M0, cb1: &CB1108M1, cb_len_read: 24 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L11_512, bark_env_size: 30, bark_cb: &FCB11L, bark_n_coef: 3, bark_n_bit: 6, cb0: &CB1108L0, cb1: &CB1108L1, cb_len_read: 27 },
    ],
    size: 512,
    n_lsp: 16,
    lspcodebook: &FF_METASOUND_LSP11,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 3,
    ppc_shape_cb: &SHAPE11,
    ppc_period_bit: 9, ppc_shape_bit: 36, ppc_shape_len: 30, pgain_bit: 7, peak_per2wid: 90,
};

pub static MODE_11_10: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S11_64,  bark_env_size: 10, bark_cb: &FCB11S, bark_n_coef: 1, bark_n_bit: 5, cb0: &CB1110S0, cb1: &CB1110S1, cb_len_read: 21 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M11_256, bark_env_size: 20, bark_cb: &FCB11M, bark_n_coef: 2, bark_n_bit: 5, cb0: &CB1110M0, cb1: &CB1110M1, cb_len_read: 18 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L11_512, bark_env_size: 30, bark_cb: &FCB11L, bark_n_coef: 3, bark_n_bit: 6, cb0: &CB1110L0, cb1: &CB1110L1, cb_len_read: 20 },
    ],
    size: 512,
    n_lsp: 16,
    lspcodebook: &FF_METASOUND_LSP11,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 3,
    ppc_shape_cb: &SHAPE11,
    ppc_period_bit: 9, ppc_shape_bit: 36, ppc_shape_len: 30, pgain_bit: 7, peak_per2wid: 90,
};

pub static MODE_16_16: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S16_128,  bark_env_size: 10, bark_cb: &FCB16S, bark_n_coef: 1, bark_n_bit: 5, cb0: &CB1616S0, cb1: &CB1616S1, cb_len_read: 16 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M16_512,  bark_env_size: 20, bark_cb: &FCB16M, bark_n_coef: 2, bark_n_bit: 5, cb0: &CB1616M0, cb1: &CB1616M1, cb_len_read: 15 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L16_1024, bark_env_size: 30, bark_cb: &FCB16L, bark_n_coef: 3, bark_n_bit: 6, cb0: &CB1616L0, cb1: &CB1616L1, cb_len_read: 16 },
    ],
    size: 1024,
    n_lsp: 16,
    lspcodebook: &FF_METASOUND_LSP16,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 3,
    ppc_shape_cb: &SHAPE16,
    ppc_period_bit: 9, ppc_shape_bit: 56, ppc_shape_len: 60, pgain_bit: 7, peak_per2wid: 180,
};

pub static MODE_22_20: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S22_128,  bark_env_size: 10, bark_cb: &FCB22S_1, bark_n_coef: 1, bark_n_bit: 6, cb0: &CB2220S0, cb1: &CB2220S1, cb_len_read: 18 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M22_512,  bark_env_size: 20, bark_cb: &FCB22M_1, bark_n_coef: 2, bark_n_bit: 6, cb0: &CB2220M0, cb1: &CB2220M1, cb_len_read: 17 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L22_1024, bark_env_size: 32, bark_cb: &FCB22L_1, bark_n_coef: 4, bark_n_bit: 6, cb0: &CB2220L0, cb1: &CB2220L1, cb_len_read: 18 },
    ],
    size: 1024,
    n_lsp: 16,
    lspcodebook: &FF_METASOUND_LSP22,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 3,
    ppc_shape_cb: &SHAPE22_1,
    ppc_period_bit: 9, ppc_shape_bit: 56, ppc_shape_len: 36, pgain_bit: 7, peak_per2wid: 144,
};

pub static MODE_22_24: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 8, bark_tab: &BARK_TAB_S22_128,  bark_env_size: 10, bark_cb: &FCB22S_1, bark_n_coef: 1, bark_n_bit: 6, cb0: &CB2224S0, cb1: &CB2224S1, cb_len_read: 15 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M22_512,  bark_env_size: 20, bark_cb: &FCB22M_1, bark_n_coef: 2, bark_n_bit: 6, cb0: &CB2224M0, cb1: &CB2224M1, cb_len_read: 14 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L22_1024, bark_env_size: 32, bark_cb: &FCB22L_1, bark_n_coef: 4, bark_n_bit: 6, cb0: &CB2224L0, cb1: &CB2224L1, cb_len_read: 15 },
    ],
    size: 1024,
    n_lsp: 16,
    lspcodebook: &FF_METASOUND_LSP22,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 3,
    ppc_shape_cb: &SHAPE22_1,
    ppc_period_bit: 9, ppc_shape_bit: 56, ppc_shape_len: 36, pgain_bit: 7, peak_per2wid: 144,
};

pub static MODE_22_32: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 4, bark_tab: &BARK_TAB_S22_128, bark_env_size: 10, bark_cb: &FCB22S_2, bark_n_coef: 1, bark_n_bit: 6, cb0: &CB2232S0, cb1: &CB2232S1, cb_len_read: 11 },
        FrameMode { sub: 2, bark_tab: &BARK_TAB_M22_256, bark_env_size: 20, bark_cb: &FCB22M_2, bark_n_coef: 2, bark_n_bit: 6, cb0: &CB2232M0, cb1: &CB2232M1, cb_len_read: 11 },
        FrameMode { sub: 1, bark_tab: &BARK_TAB_L22_512, bark_env_size: 32, bark_cb: &FCB22L_2, bark_n_coef: 4, bark_n_bit: 6, cb0: &CB2232L0, cb1: &CB2232L1, cb_len_read: 12 },
    ],
    size: 512,
    n_lsp: 16,
    lspcodebook: &LSP22_2,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 4,
    ppc_shape_cb: &SHAPE22_2,
    ppc_period_bit: 9, ppc_shape_bit: 56, ppc_shape_len: 36, pgain_bit: 7, peak_per2wid: 72,
};

pub static MODE_44_40: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 16, bark_tab: &BARK_TAB_S44_128,  bark_env_size: 10, bark_cb: &FCB44S, bark_n_coef: 1, bark_n_bit: 6, cb0: &CB4440S0, cb1: &CB4440S1, cb_len_read: 18 },
        FrameMode { sub: 4,  bark_tab: &BARK_TAB_M44_512,  bark_env_size: 20, bark_cb: &FCB44M, bark_n_coef: 2, bark_n_bit: 6, cb0: &CB4440M0, cb1: &CB4440M1, cb_len_read: 17 },
        FrameMode { sub: 1,  bark_tab: &BARK_TAB_L44_2048, bark_env_size: 40, bark_cb: &FCB44L, bark_n_coef: 4, bark_n_bit: 6, cb0: &CB4440L0, cb1: &CB4440L1, cb_len_read: 17 },
    ],
    size: 2048,
    n_lsp: 20,
    lspcodebook: &FF_METASOUND_LSP44,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 4,
    ppc_shape_cb: &SHAPE44,
    ppc_period_bit: 9, ppc_shape_bit: 84, ppc_shape_len: 54, pgain_bit: 7, peak_per2wid: 432,
};

pub static MODE_44_48: ModeTab = ModeTab {
    fmode: [
        FrameMode { sub: 16, bark_tab: &BARK_TAB_S44_128,  bark_env_size: 10, bark_cb: &FCB44S, bark_n_coef: 1, bark_n_bit: 6, cb0: &CB4448S0, cb1: &CB4448S1, cb_len_read: 15 },
        FrameMode { sub: 4,  bark_tab: &BARK_TAB_M44_512,  bark_env_size: 20, bark_cb: &FCB44M, bark_n_coef: 2, bark_n_bit: 6, cb0: &CB4448M0, cb1: &CB4448M1, cb_len_read: 14 },
        FrameMode { sub: 1,  bark_tab: &BARK_TAB_L44_2048, bark_env_size: 40, bark_cb: &FCB44L, bark_n_coef: 4, bark_n_bit: 6, cb0: &CB4448L0, cb1: &CB4448L1, cb_len_read: 14 },
    ],
    size: 2048,
    n_lsp: 20,
    lspcodebook: &FF_METASOUND_LSP44,
    lsp_bit0: 1, lsp_bit1: 6, lsp_bit2: 4, lsp_split: 4,
    ppc_shape_cb: &SHAPE44,
    ppc_period_bit: 9, ppc_shape_bit: 84, ppc_shape_len: 54, pgain_bit: 7, peak_per2wid: 432,
};
