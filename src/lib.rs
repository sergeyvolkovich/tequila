//! tequila — cleanroom-энкодер TwinVQ (VQF, NTT), битстрим-совместимый
//! с FFmpeg (`libavformat/vqf.c` + `libavcodec/twinvqdec.c`).
//!
//! - Спецификация формата: `docs/ARCHITECTURE.md`
//! - Модель эталонного энкодера NTT: `docs/ENCODE_PIPELINE.md`
//! - Журнал шагов: `docs/PROGRESS.md`
//!
//! Собственный код — MIT; vendored-данные кодбуков (`src/vendor/`) —
//! LGPL-2.1-or-later, портированы из FFmpeg (см. LICENSE).

pub mod analysis;
pub mod encoder;
pub mod bitstream;
pub mod codec;
pub mod container;
pub mod decode;
pub mod dsp;
pub mod frame;
pub mod layout;
pub mod modes;
pub mod parallel;
pub mod psycho;
pub mod quantize;
pub mod simd;
pub mod vendor;
pub mod wav;
