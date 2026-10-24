//! Vendored-данные формата TwinVQ из FFmpeg — LGPL-2.1-or-later.
//!
//! `src/vendor/generated/*.rs` генерируются `tools/gen_vendor_rs.py` из
//! `third_party/ffmpeg/` (twinvq_data.h, metasound_twinvq_data.h) и
//! коммитятся в репозиторий — сборка не требует python.
//!
//! Таблицы режимов (`tables.rs`) — перенос `mode_*` из FFmpeg twinvqdec.c.
//! Комбинация с этими данными наследует LGPL-2.1-or-later (см. LICENSE).

#[rustfmt::skip]
#[path = "generated/twinvq_cb_data.rs"]
pub mod cb_data;

#[rustfmt::skip]
#[path = "generated/metasound_lsp_data.rs"]
pub mod lsp_data;

mod tables;

pub use tables::{
    MODE_08_08, MODE_11_08, MODE_11_10, MODE_16_16, MODE_22_20, MODE_22_24,
    MODE_22_32, MODE_44_40, MODE_44_48,
};
