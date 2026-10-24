//! Кадр TwinVQ: все поля + парсинг/сериализация.
//!
//! Точный порт порядка полей `twinvq_read_bitstream` (twinvqdec.c) /
//! `build_frame` (python-оракул vqflib.py, сверен на 218 потоках NTT):
//!
//! ```text
//! wtype(4)
//! main VQ: n_div[ftype] × (cb0_bits, cb1_bits)      // биты по дивизиям
//! bark1[ch][sub][k]                                  // bark_n_bit каждый
//! bark_use_hist[ch][sub] (1 бит)
//! LONG:    gain[ch](8)
//! не-LONG: для каждого ch: gain(8), затем sub_gain[sub](5)
//! LSP: для каждого ch: lpc_hist(lsp_bit0), lpc_idx1(lsp_bit1),
//!      lpc_idx2[lsp_split](lsp_bit2)
//! LONG: PPC cb (n_div[3] × пары бит), затем для каждого ch:
//!      p_coef(ppc_period_bit), g_coef(pgain_bit)
//! ```
//!
//! Кадры в DATA упакованы вплотную на битовом уровне; после полей кадра
//! позиция добивается нулями до `total_bits`.

use crate::bitstream::{BitReader, BitWriter};
use crate::codec::StreamInfo;
use crate::modes::FrameType;

pub const WTYPE_BITS: usize = 4; // TWINVQ_WINDOW_TYPE_BITS
pub const GAIN_BITS: usize = 8; // TWINVQ_GAIN_BITS
pub const SUB_GAIN_BITS: usize = 5; // TWINVQ_SUB_GAIN_BITS

/// Ошибка парсинга кадра.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameError {
    /// Неверный wtype (> 8) — кадр отвергается декодером FFmpeg.
    InvalidWindowType(u32),
    /// Битовый поток закончился раньше полей кадра.
    UnexpectedEof,
}

/// Все поля одного кадра (эквивалент `TwinVQFrameData` FFmpeg).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameData {
    pub wtype: u8,
    /// Индексы main VQ, пары подряд: `[cb0_0, cb1_0, cb0_1, cb1_1, …]`
    /// (порядок `read_cb_data`).
    pub main_coeffs: Vec<u8>,
    /// Индексы PPC shape cb (только LONG), пары подряд.
    pub ppc_coeffs: Vec<u8>,
    /// bark1[ch][sub][k] — ch-major: `idx = (ch·sub + s)·ncoef + k`.
    pub bark1: Vec<u8>,
    /// bark_use_hist[ch·sub + s].
    pub bark_use_hist: Vec<u8>,
    /// gain_bits[ch].
    pub gain_bits: Vec<u8>,
    /// sub_gain_bits[ch·sub + s] (не-LONG).
    pub sub_gain_bits: Vec<u8>,
    /// lpc_hist_idx[ch] (MA-вектор кодбука lsp_bit0 бит).
    pub lpc_hist_idx: Vec<u8>,
    /// lpc_idx1[ch].
    pub lpc_idx1: Vec<u8>,
    /// lpc_idx2[ch·lsp_split + j].
    pub lpc_idx2: Vec<u8>,
    /// p_coef[ch] (LONG).
    pub p_coef: Vec<u32>,
    /// g_coef[ch] (LONG).
    pub g_coef: Vec<u32>,
}

impl FrameData {
    /// Выделить векторы под поля кадра данного wtype (нули).
    pub fn new(info: &StreamInfo, wtype: u8) -> Self {
        let m = info.mode;
        let ftype = info.ftype_of_wtype(wtype);
        let fm = &m.fmode[ftype.idx()];
        let sub = fm.sub;
        let n_ch = info.n_ch;
        let is_long = ftype == FrameType::Long;
        FrameData {
            wtype,
            main_coeffs: vec![0; info.layout.ftype[ftype.idx()].n_div * 2],
            ppc_coeffs: if is_long { vec![0; info.layout.ppc.n_div * 2] } else { Vec::new() },
            bark1: vec![0; n_ch * sub * fm.bark_n_coef],
            bark_use_hist: vec![0; n_ch * sub],
            gain_bits: vec![0; n_ch],
            sub_gain_bits: if is_long { Vec::new() } else { vec![0; n_ch * sub] },
            lpc_hist_idx: vec![0; n_ch],
            lpc_idx1: vec![0; n_ch],
            lpc_idx2: vec![0; n_ch * m.lsp_split],
            p_coef: if is_long { vec![0; n_ch] } else { Vec::new() },
            g_coef: if is_long { vec![0; n_ch] } else { Vec::new() },
        }
    }

    /// ftype кадра.
    pub fn ftype(&self) -> FrameType {
        FrameType::from_wtype(self.wtype)
    }

    /// Разобрать один кадр из потока (зеркало `serialize`).
    pub fn parse(br: &mut BitReader, info: &StreamInfo) -> Result<Self, FrameError> {
        let m = info.mode;
        let n_ch = info.n_ch;
        let wtype = br.read(WTYPE_BITS);
        if wtype > 8 {
            return Err(FrameError::InvalidWindowType(wtype));
        }
        let wtype = wtype as u8;
        let ftype = info.ftype_of_wtype(wtype);
        let fm = &m.fmode[ftype.idx()];
        let sub = fm.sub;
        let is_long = ftype == FrameType::Long;

        let mut fr = FrameData::new(info, wtype);

        // main VQ (read_cb_data)
        let d = &info.layout.ftype[ftype.idx()];
        for i in 0..d.n_div {
            let (b0, b1) = d.bits_at(i);
            fr.main_coeffs[2 * i] = br.read(b0) as u8;
            fr.main_coeffs[2 * i + 1] = br.read(b1) as u8;
        }

        // bark1: ch → sub → k
        for ch in 0..n_ch {
            for s in 0..sub {
                for k in 0..fm.bark_n_coef {
                    fr.bark1[(ch * sub + s) * fm.bark_n_coef + k] =
                        br.read(fm.bark_n_bit) as u8;
                }
            }
        }

        // bark_use_hist: ch → sub
        for ch in 0..n_ch {
            for s in 0..sub {
                fr.bark_use_hist[ch * sub + s] = br.read(1) as u8;
            }
        }

        // gain / sub_gain: внешний цикл по каналам, внутри gain → все sub_gain
        if is_long {
            for ch in 0..n_ch {
                fr.gain_bits[ch] = br.read(GAIN_BITS) as u8;
            }
        } else {
            for ch in 0..n_ch {
                fr.gain_bits[ch] = br.read(GAIN_BITS) as u8;
                for s in 0..sub {
                    fr.sub_gain_bits[ch * sub + s] = br.read(SUB_GAIN_BITS) as u8;
                }
            }
        }

        // LSP: ch → (hist, idx1, idx2[split])
        for ch in 0..n_ch {
            fr.lpc_hist_idx[ch] = br.read(m.lsp_bit0) as u8;
            fr.lpc_idx1[ch] = br.read(m.lsp_bit1) as u8;
            for j in 0..m.lsp_split {
                fr.lpc_idx2[ch * m.lsp_split + j] = br.read(m.lsp_bit2) as u8;
            }
        }

        // PPC (LONG): shape cb, затем p_coef/g_coef всех каналов
        if is_long {
            let p = &info.layout.ppc;
            for i in 0..p.n_div {
                let (b0, b1) = p.bits_at(i);
                fr.ppc_coeffs[2 * i] = br.read(b0) as u8;
                fr.ppc_coeffs[2 * i + 1] = br.read(b1) as u8;
            }
            for ch in 0..n_ch {
                fr.p_coef[ch] = br.read(m.ppc_period_bit);
                fr.g_coef[ch] = br.read(m.pgain_bit);
            }
        }

        Ok(fr)
    }

    /// Записать поля кадра (без паддинга — вызывающий добивает до total_bits).
    pub fn serialize(&self, w: &mut BitWriter, info: &StreamInfo) {
        let m = info.mode;
        let n_ch = info.n_ch;
        let ftype = info.ftype_of_wtype(self.wtype);
        let fm = &m.fmode[ftype.idx()];
        let sub = fm.sub;
        let is_long = ftype == FrameType::Long;

        w.write(self.wtype as u32, WTYPE_BITS);

        let d = &info.layout.ftype[ftype.idx()];
        for i in 0..d.n_div {
            let (b0, b1) = d.bits_at(i);
            w.write(self.main_coeffs[2 * i] as u32, b0);
            w.write(self.main_coeffs[2 * i + 1] as u32, b1);
        }

        for ch in 0..n_ch {
            for s in 0..sub {
                for k in 0..fm.bark_n_coef {
                    w.write(
                        self.bark1[(ch * sub + s) * fm.bark_n_coef + k] as u32,
                        fm.bark_n_bit,
                    );
                }
            }
        }

        for ch in 0..n_ch {
            for s in 0..sub {
                w.write(self.bark_use_hist[ch * sub + s] as u32, 1);
            }
        }

        if is_long {
            for ch in 0..n_ch {
                w.write(self.gain_bits[ch] as u32, GAIN_BITS);
            }
        } else {
            for ch in 0..n_ch {
                w.write(self.gain_bits[ch] as u32, GAIN_BITS);
                for s in 0..sub {
                    w.write(self.sub_gain_bits[ch * sub + s] as u32, SUB_GAIN_BITS);
                }
            }
        }

        for ch in 0..n_ch {
            w.write(self.lpc_hist_idx[ch] as u32, m.lsp_bit0);
            w.write(self.lpc_idx1[ch] as u32, m.lsp_bit1);
            for j in 0..m.lsp_split {
                w.write(self.lpc_idx2[ch * m.lsp_split + j] as u32, m.lsp_bit2);
            }
        }

        if is_long {
            let p = &info.layout.ppc;
            for i in 0..p.n_div {
                let (b0, b1) = p.bits_at(i);
                w.write(self.ppc_coeffs[2 * i] as u32, b0);
                w.write(self.ppc_coeffs[2 * i + 1] as u32, b1);
            }
            for ch in 0..n_ch {
                w.write(self.p_coef[ch], m.ppc_period_bit);
                w.write(self.g_coef[ch], m.pgain_bit);
            }
        }
    }
}

impl FrameType {
    /// ftype по wtype кадра.
    pub fn from_wtype(wtype: u8) -> FrameType {
        crate::modes::wtype_to_ftype(wtype as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_44_48_mono() -> StreamInfo {
        StreamInfo::new(44, 48, 1).unwrap()
    }

    /// Простой round-trip: поля → биты → поля (моно, LONG, 44_48).
    #[test]
    fn roundtrip_long_mono() {
        let si = si_44_48_mono();
        let mut fr = FrameData::new(&si, 0);
        // значения main VQ обязаны влезать в свою битовую ширину
        // (44_48 LONG: 145 дивизий по 7+7 бит, 3 последних — 7+6)
        let d = &si.layout.ftype[FrameType::Long.idx()];
        for i in 0..d.n_div {
            let (b0, b1) = d.bits_at(i);
            fr.main_coeffs[2 * i] = ((i * 7 + 3) & ((1 << b0) - 1)) as u8;
            fr.main_coeffs[2 * i + 1] = ((i * 11 + 5) & ((1 << b1) - 1)) as u8;
        }
        fr.bark1.iter_mut().enumerate().for_each(|(i, v)| {
            *v = ((i * 5 + 1) % 64) as u8;
        });
        fr.bark_use_hist[0] = 1;
        fr.gain_bits[0] = 137;
        fr.lpc_hist_idx[0] = 1;
        fr.lpc_idx1[0] = 42;
        fr.lpc_idx2.iter_mut().enumerate().for_each(|(i, v)| {
            *v = ((i + 9) % 16) as u8;
        });
        fr.ppc_coeffs.iter_mut().enumerate().for_each(|(i, v)| {
            *v = ((i * 3) % 128) as u8;
        });
        fr.p_coef[0] = 300;
        fr.g_coef[0] = 100;

        let mut w = BitWriter::new();
        fr.serialize(&mut w, &si);
        // LONG 44_48 моно: поля занимают весь бюджет кадра без паддинга
        assert_eq!(w.bit_pos(), si.total_bits());
        w.align_zero(si.total_bits());
        assert_eq!(w.bit_pos(), si.total_bits());

        let mut r = BitReader::new(w.bytes(), si.total_bits());
        let back = FrameData::parse(&mut r, &si).unwrap();
        assert_eq!(back, fr);
        assert_eq!(r.bit_pos(), si.total_bits());
    }

    /// Round-trip SHORT-кадра 44_48 (sub_gain присутствует, PPC нет).
    #[test]
    fn roundtrip_short_mono() {
        let si = si_44_48_mono();
        let mut fr = FrameData::new(&si, 2);
        assert_eq!(fr.sub_gain_bits.len(), 16); // sub=16
        assert!(fr.ppc_coeffs.is_empty());
        let d = &si.layout.ftype[FrameType::Short.idx()];
        assert_eq!(fr.main_coeffs.len(), d.n_div * 2);
        for (i, v) in fr.main_coeffs.iter_mut().enumerate() {
            *v = ((i * 11 + 5) % 64) as u8;
        }
        for (i, v) in fr.sub_gain_bits.iter_mut().enumerate() {
            *v = ((i * 3) % 32) as u8;
        }
        fr.gain_bits[0] = 200;

        let mut w = BitWriter::new();
        fr.serialize(&mut w, &si);
        w.align_zero(si.total_bits());
        let mut r = BitReader::new(w.bytes(), si.total_bits());
        let back = FrameData::parse(&mut r, &si).unwrap();
        assert_eq!(back, fr);
    }

    /// Стерео 44s96: порядок gain (все каналы) / sub_gain внутри канала.
    #[test]
    fn roundtrip_short_stereo() {
        let si = StreamInfo::new(44, 48, 2).unwrap();
        let mut fr = FrameData::new(&si, 2);
        assert_eq!(fr.gain_bits.len(), 2);
        assert_eq!(fr.sub_gain_bits.len(), 32);
        assert_eq!(fr.bark1.len(), 2 * 16 * 1); // ncoef=1
        fr.gain_bits[0] = 10;
        fr.gain_bits[1] = 20;
        fr.sub_gain_bits[0] = 1;
        fr.sub_gain_bits[16] = 2; // первый sub_gain канала 1

        let mut w = BitWriter::new();
        fr.serialize(&mut w, &si);
        w.align_zero(si.total_bits());
        let mut r = BitReader::new(w.bytes(), si.total_bits());
        let back = FrameData::parse(&mut r, &si).unwrap();
        assert_eq!(back, fr);
    }

    #[test]
    fn invalid_wtype_rejected() {
        let si = si_44_48_mono();
        let mut w = BitWriter::new();
        w.write(9, WTYPE_BITS);
        w.align_zero(si.total_bits());
        let mut r = BitReader::new(w.bytes(), si.total_bits());
        assert_eq!(
            FrameData::parse(&mut r, &si),
            Err(FrameError::InvalidWindowType(9))
        );
    }

    /// Нейтральный кадр (все нули, wtype=0) должен занимать ровно total_bits.
    #[test]
    fn neutral_frame_fits_budget() {
        for (isampf, ibps, n_ch) in [
            (8u32, 8u32, 1usize),
            (11, 10, 1),
            (16, 16, 2),
            (22, 20, 1),
            (22, 32, 2),
            (44, 40, 1),
            (44, 48, 2),
        ] {
            let si = StreamInfo::new(isampf, ibps, n_ch).unwrap();
            for wtype in [0u8, 2, 4] {
                let fr = FrameData::new(&si, wtype);
                let mut w = BitWriter::new();
                fr.serialize(&mut w, &si);
                assert_eq!(
                    w.bit_pos(),
                    si.total_bits(),
                    "{} x{} wtype {}",
                    si.mode_key(),
                    n_ch,
                    wtype
                );
            }
        }
    }
}
