//! Параметры потока VQF: режим + каналы + битовый бюджет кадра.
//!
//! Связывает таблицу режима (`modes::find_mode`), размерности контейнера
//! (COMM: channels/bitrate/rate_flag) и битовую раскладку кадра
//! (`layout::frame_layout`). Один `StreamInfo` полностью описывает поток,
//! достаточен для парсера, райтера и энкодера.

use crate::layout::{frame_layout, FrameLayout};
use crate::modes::{find_mode, FrameType, ModeTab};

/// Скорректированная частота из rate_flag COMM (FFmpeg `twinvq_decode_init`).
pub fn rate_flag_to_sr(rf: u32) -> u32 {
    match rf {
        44 => 44100,
        22 => 22050,
        11 => 11025,
        _ => rf * 1000,
    }
}

/// sr → isampf (кГц, целочисленно — как в декодере).
pub fn sr_to_isampf(sr: u32) -> u32 {
    match sr {
        44100 => 44,
        22050 => 22,
        11025 => 11,
        _ => sr / 1000,
    }
}

/// Параметры потока VQF (режим × число каналов).
#[derive(Clone, Copy)]
pub struct StreamInfo {
    pub mode: &'static ModeTab,
    /// Число каналов (1/2).
    pub n_ch: usize,
    /// Частота дискретизации, Гц.
    pub sr: u32,
    /// Суммарный битрейт, кбит/с (на все каналы).
    pub kbps_total: usize,
    /// Битовый бюджет кадра (init_bitstream_params).
    pub layout: FrameLayout,
}

impl StreamInfo {
    /// Из (isampf кГц, кбит/с на канал, каналы) — способ энкодера.
    pub fn new(isampf: u32, ibps: u32, n_ch: usize) -> Option<Self> {
        let mode = find_mode(isampf as i32, ibps as i32)?;
        let sr = rate_flag_to_sr(isampf);
        let kbps_total = ibps as usize * n_ch;
        let layout = frame_layout(mode, n_ch, kbps_total, sr as usize);
        Some(StreamInfo { mode, n_ch, sr, kbps_total, layout })
    }

    /// Из полей COMM (channels-1 уже исправлено, bitrate кбит/с, rate_flag).
    pub fn from_comm(n_ch: usize, kbps_total: u32, rate_flag: u32) -> Option<Self> {
        let sr = rate_flag_to_sr(rate_flag);
        let isampf = sr_to_isampf(sr);
        let ibps = kbps_total / n_ch as u32;
        Self::new(isampf, ibps, n_ch)
    }

    /// Ключ режима («08_08», «44_48»).
    pub fn mode_key(&self) -> String {
        let isampf = sr_to_isampf(self.sr);
        format!("{:02}_{:02}", isampf, self.kbps_total / self.n_ch)
    }

    /// Норма IMDCT: моно 2.0, стерео 1.0 (FFmpeg init_mdct_win).
    pub fn norm(&self) -> f64 {
        if self.n_ch == 1 { 2.0 } else { 1.0 }
    }

    /// Битовый бюджет кадра.
    pub fn total_bits(&self) -> usize {
        self.layout.total_bits
    }

    /// Число байт DATA на кадр (block_align демиксера).
    pub fn frame_bytes(&self) -> usize {
        (self.layout.total_bits + 7) / 8
    }

    /// ftype кадра по wtype.
    pub fn ftype_of_wtype(&self, wtype: u8) -> FrameType {
        crate::modes::wtype_to_ftype(wtype as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_info_from_comm_roundtrip() {
        // 44s96: COMM = (ch-1=1, bitrate=96, rate=44) → режим 44_48, стерео
        let si = StreamInfo::from_comm(2, 96, 44).unwrap();
        assert_eq!(si.mode_key(), "44_48");
        assert_eq!(si.n_ch, 2);
        assert_eq!(si.sr, 44100);
        assert_eq!(si.kbps_total, 96);
        assert_eq!(si.total_bits(), 4458);
        assert_eq!(si.norm(), 1.0);
        assert_eq!(si.frame_bytes(), 558);

        // 8m8
        let si = StreamInfo::from_comm(1, 8, 8).unwrap();
        assert_eq!(si.mode_key(), "08_08");
        assert_eq!(si.total_bits(), 512);
        assert_eq!(si.norm(), 2.0);

        // 22m32: rate_flag 22 → 22050 Гц
        let si = StreamInfo::from_comm(1, 32, 22).unwrap();
        assert_eq!(si.mode_key(), "22_32");
        assert_eq!(si.sr, 22050);

        // Несуществующая комбинация
        assert!(StreamInfo::from_comm(1, 96, 44).is_none());
        assert!(StreamInfo::from_comm(2, 8, 8).is_none()); // 8_08×2 нет в VQF
    }

    #[test]
    fn rate_flag_variants() {
        assert_eq!(rate_flag_to_sr(44), 44100);
        assert_eq!(rate_flag_to_sr(22), 22050);
        assert_eq!(rate_flag_to_sr(11), 11025);
        assert_eq!(rate_flag_to_sr(8), 8000);
        assert_eq!(rate_flag_to_sr(16), 16000);
        assert_eq!(sr_to_isampf(44100), 44);
        assert_eq!(sr_to_isampf(11025), 11);
        assert_eq!(sr_to_isampf(8000), 8);
    }
}
