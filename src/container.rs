//! Контейнер VQF: заголовок TWIN97012000, чанки, DATA (twinvq demuxer FFmpeg).
//!
//! ```text
//! "TWIN" | "97012000" | BE32 header_size
//! | чанки: tag(4) + BE32 len + payload          (COMM, NAME, …)
//! | "DATA" | битстрим кадров (длины не имеет!)
//! ```
//!
//! * `header_size` = Σ(8 + chunk_len) по чанкам до DATA; DATA-payload
//!   начинается с байта `16 + header_size + 4`;
//! * COMM — 16 байт (layout NTT): BE32 channels−1, BE32 bitrate (общий,
//!   кбит/с), BE32 rate_flag + 4 нулевых байта паддинга. Канон обязателен:
//!   официальный плеер и `file` ожидают поля на фиксированных офсетах
//!   (байт 36 должен быть 0x00);
//! * DSIZ официальным энкодером не пишется — наш writer тоже не пишет
//!   (парсер при этом понимает его в чужих файлах);
//! * кадры упакованы вплотную на битовом уровне; в конце DATA — паддинг
//!   нулями до байта + 2 запасных байта (чтобы демиксер не спотыкался).

use crate::bitstream::BitWriter;
use crate::codec::StreamInfo;
use crate::frame::{FrameData, FrameError};

/// Ошибка контейнера.
#[derive(Clone, Debug)]
pub enum ContainerError {
    NotVqf,
    Truncated,
    MissingComm,
    MissingData,
    UnknownMode(String),
    Frame(FrameError),
}

impl std::fmt::Display for ContainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerError::NotVqf => write!(f, "не VQF (нет магии TWIN)"),
            ContainerError::Truncated => write!(f, "файл обрезан"),
            ContainerError::MissingComm => write!(f, "нет чанка COMM"),
            ContainerError::MissingData => write!(f, "нет чанка DATA"),
            ContainerError::UnknownMode(s) => write!(f, "неподдерживаемый режим {s}"),
            ContainerError::Frame(e) => write!(f, "кадр: {e:?}"),
        }
    }
}

impl From<FrameError> for ContainerError {
    fn from(e: FrameError) -> Self {
        ContainerError::Frame(e)
    }
}

/// Разобранный заголовок VQF.
#[derive(Clone, Debug)]
pub struct VqfHeader {
    pub n_ch: usize,
    pub kbps_total: u32,
    pub rate_flag: u32,
    /// Размер исходных PCM, байт (чанк DSIZ, если был).
    pub dsiz: Option<u32>,
    /// Байтовое смещение битстрима кадров (payload DATA).
    pub data_offset: usize,
    /// Прочие чанки (tag, bytes) в исходном порядке.
    pub meta_chunks: Vec<([u8; 4], Vec<u8>)>,
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Пройти чанки заголовка до DATA (данные после — битстрим кадров).
pub fn parse_header(d: &[u8]) -> Result<(VqfHeader, StreamInfo), ContainerError> {
    if d.len() < 16 || &d[0..4] != b"TWIN" || &d[4..12] != b"97012000" {
        return Err(ContainerError::NotVqf);
    }
    let header_size = be32(&d[12..16]) as usize;
    let mut off = 16usize;
    let end = 16usize.saturating_add(header_size);
    let mut comm: Option<(usize, u32, u32)> = None; // (n_ch, kbps, rate_flag)
    let mut dsiz = None;
    let mut meta_chunks = Vec::new();
    let mut data_offset = None;

    while off + 4 <= d.len() {
        let tag = [d[off], d[off + 1], d[off + 2], d[off + 3]];
        if tag == *b"DATA" {
            data_offset = Some(off + 4);
            break;
        }
        if off + 8 > d.len() {
            return Err(ContainerError::Truncated);
        }
        let len = be32(&d[off + 4..off + 8]) as usize;
        if off + 8 + len > d.len() {
            return Err(ContainerError::Truncated);
        }
        let payload = &d[off + 8..off + 8 + len];
        match &tag {
            b"COMM" => {
                if payload.len() >= 12 {
                    comm = Some((
                        be32(&payload[0..4]) as usize + 1,
                        be32(&payload[4..8]),
                        be32(&payload[8..12]),
                    ));
                }
            }
            b"DSIZ" => {
                if payload.len() >= 4 {
                    dsiz = Some(be32(&payload[0..4]));
                }
            }
            _ => meta_chunks.push((tag, payload.to_vec())),
        }
        off += 8 + len;
        if off > end + 4 {
            // header_size меньше фактического — продолжаем до DATA всё равно
        }
    }

    let data_offset =
        data_offset.ok_or(ContainerError::MissingData)?;
    let (n_ch, kbps_total, rate_flag) = comm.ok_or(ContainerError::MissingComm)?;
    let info = StreamInfo::from_comm(n_ch, kbps_total, rate_flag)
        .ok_or_else(|| ContainerError::UnknownMode(format!("{rate_flag}kHz {kbps_total}kbps x{n_ch}")))?;
    Ok((
        VqfHeader { n_ch, kbps_total, rate_flag, dsiz, data_offset, meta_chunks },
        info,
    ))
}

/// Полный разбор VQF: заголовок + все кадры.
pub struct VqfFile {
    pub header: VqfHeader,
    pub info: StreamInfo,
    pub frames: Vec<FrameData>,
    /// Бит после последнего кадра до конца данных (паддинг NTT 0..4).
    pub leftover_bits: usize,
}

/// Разобрать .vqf из байтов.
pub fn parse_vqf(d: &[u8]) -> Result<VqfFile, ContainerError> {
    let (header, info) = parse_header(d)?;
    let data = &d[header.data_offset..];
    let nbits = data.len() * 8;
    let total = info.total_bits();
    let mut br = crate::bitstream::BitReader::new(data, nbits);
    let mut frames = Vec::new();
    while br.remaining() >= total {
        frames.push(FrameData::parse(&mut br, &info)?);
    }
    let leftover_bits = br.remaining();
    Ok(VqfFile { header, info, frames, leftover_bits })
}

/// Упаковать кадры в DATA-битстрим: вплотную, паддинг до байта + 2 запасных
/// байта (`pack_frames` python-оракула; поведение демиксера FFmpeg).
pub fn pack_frames(info: &StreamInfo, frames: &[FrameData]) -> Vec<u8> {
    let total = info.total_bits();
    let mut w = BitWriter::new();
    for fr in frames {
        let start = w.bit_pos();
        fr.serialize(&mut w, info);
        debug_assert_eq!(w.bit_pos() - start, total, "кадр не уложился в бюджет");
        let _ = start;
    }
    // паддинг до границы байта: (8 − total·n % 8) % 8
    let pad = (8 - (total * frames.len()) % 8) % 8;
    w.write(0, pad);
    let mut data = w.take_bytes();
    data.extend_from_slice(&[0, 0]); // +2 запасных байта
    data
}

/// Метаданные для записи (текстовые чанки).
pub type MetaChunk = ([u8; 4], String);

/// Канонические текстовые чанки VQF ↔ ключи метаданных FFmpeg
/// (`vqf_metadata_conv`, libavformat/vqf.c) — все 18 из таблицы FFmpeg.
///
/// Порядок записи: первые четыре — как у псевдотегов test3.vqf
/// (NAME, COMT, AUTH, "(c) "; NAME первым пишет и NTT TestEnc — имя
/// сигнала), далее остальные в алфавитном порядке chunk-id.
pub const TAG_TABLE: &[(&[u8; 4], &str)] = &[
    (b"NAME", "title"),
    (b"COMT", "comment"),
    (b"AUTH", "author"),
    (b"(c) ", "copyright"),
    (b"ARNG", "arranger"),
    (b"BAND", "band"),
    (b"CDCT", "conductor"),
    (b"FILE", "filename"),
    (b"GENR", "genre"),
    (b"LABL", "publisher"),
    (b"MUSC", "composer"),
    (b"NOTE", "note"),
    (b"PROD", "producer"),
    (b"PRSN", "personnel"),
    (b"REMX", "remixer"),
    (b"SING", "singer"),
    (b"TRCK", "track"),
    (b"WORD", "words"),
];

/// Chunk-id, зарезервированные контейнером (не текстовые меты).
const RESERVED_CHUNKS: [&[u8; 4]; 4] = [b"COMM", b"DSIZ", b"DATA", b"TWIN"];

/// Ключ FFmpeg по chunk-id ("NAME" → "title"); None для нестандартных чанков.
pub fn tag_key(chunk: &[u8; 4]) -> Option<&'static str> {
    TAG_TABLE.iter().find(|(id, _)| *id == chunk).map(|(_, k)| *k)
}

/// Chunk-id по дружественному имени: принимает ключ FFmpeg ("title"),
/// имя флага CLI ("--title") или сам 4-байтовый id ("NAME").
/// Служебные чанки (COMM/DSIZ/DATA/TWIN) не находит; нестандартные
/// пользовательские id передаются напрямую через MetaChunk/--tag.
pub fn tag_chunk_id(name: &str) -> Option<&'static [u8; 4]> {
    let key = name.trim().trim_start_matches("--");
    TAG_TABLE
        .iter()
        .find(|(id, k)| key.eq_ignore_ascii_case(k) || key.as_bytes() == *id)
        .map(|(id, _)| *id)
}

/// Валиден ли chunk-id пользовательского тега: ровно 4 байта, не служебный.
pub fn is_valid_tag_chunk(id: &[u8]) -> bool {
    if id.len() != 4 {
        return false;
    }
    let arr: &[u8; 4] = id.try_into().unwrap();
    !RESERVED_CHUNKS.contains(&arr)
}

/// Упорядочить мета-чанки для записи: канонический порядок TAG_TABLE
/// (NAME первым), нестандартные id — в конце в порядке поступления.
/// Повторяющиеся id схлопываются: побеждает последнее значение
/// (последний флаг CLI в командной строке перекрывает ранние).
pub fn order_meta_chunks(chunks: Vec<MetaChunk>) -> Vec<MetaChunk> {
    let mut uniq: Vec<MetaChunk> = Vec::with_capacity(chunks.len());
    for (id, val) in chunks {
        match uniq.iter_mut().find(|(i, _)| *i == id) {
            Some(slot) => slot.1 = val,
            None => uniq.push((id, val)),
        }
    }
    let mut out: Vec<MetaChunk> = Vec::with_capacity(uniq.len());
    for (id, _) in TAG_TABLE {
        if let Some(pos) = uniq.iter().position(|(i, _)| i == *id) {
            out.push(uniq.remove(pos));
        }
    }
    out.extend(uniq); // нестандартные — как заданы
    out
}

/// Собрать полный .vqf из кадров.
///
/// Заголовок повторяет байт-в-байт layout официального энкодера NTT
/// (test2.vqf, корпус 218 эталонов): COMM строго 16 байт, без DSIZ,
/// DATA без поля длины.
pub fn write_vqf(
    info: &StreamInfo,
    frames: &[FrameData],
    meta_chunks: &[MetaChunk],
) -> Vec<u8> {
    let mut chunks: Vec<u8> = Vec::new();
    // COMM: 12 байт полей + 4 нулевых байта — как NTT (итоговый header_size
    // без меты = 24; поля на канонических офсетах 24/28/32, байт 36 = 0x00)
    chunks.extend_from_slice(b"COMM");
    chunks.extend_from_slice(&16u32.to_be_bytes());
    let rate_flag = crate::codec::sr_to_isampf(info.sr);
    chunks.extend_from_slice(&((info.n_ch - 1) as u32).to_be_bytes());
    chunks.extend_from_slice(&(info.kbps_total as u32).to_be_bytes());
    chunks.extend_from_slice(&rate_flag.to_be_bytes());
    chunks.extend_from_slice(&[0u8; 4]);
    for (tag, text) in meta_chunks {
        let b = text.as_bytes();
        chunks.extend_from_slice(tag);
        chunks.extend_from_slice(&(b.len() as u32).to_be_bytes());
        chunks.extend_from_slice(b);
    }

    let mut out = Vec::with_capacity(16 + chunks.len() + 4 + frames.len() * info.frame_bytes());
    out.extend_from_slice(b"TWIN");
    out.extend_from_slice(b"97012000");
    out.extend_from_slice(&(chunks.len() as u32).to_be_bytes());
    out.extend_from_slice(&chunks);
    out.extend_from_slice(b"DATA");
    out.extend_from_slice(&pack_frames(info, frames));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Строка манифеста golden-фикстур: stem mode n_ch n_frames size wtypes.
    struct Fixture {
        stem: String,
        mode: String,
        n_ch: usize,
        n_frames: usize,
        wtypes: Vec<u8>,
    }

    fn golden_fixtures() -> Option<Vec<Fixture>> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        let manifest = std::fs::read_to_string(format!("{dir}/manifest.txt")).ok()?;
        let mut out = Vec::new();
        for line in manifest.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() < 6 {
                continue;
            }
            out.push(Fixture {
                stem: t[0].to_string(),
                mode: t[1].to_string(),
                n_ch: t[2].parse().unwrap(),
                n_frames: t[3].parse().unwrap(),
                wtypes: t[5].split(',').map(|w| w.parse().unwrap()).collect(),
            });
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Golden-фикстуры NTT (tests/golden, готовятся scripts/make_golden.py
    /// из корпуса 218 эталонов): parse → serialize → биты побайтово идентичны.
    #[test]
    fn golden_roundtrip_bits() {
        let Some(fixtures) = golden_fixtures() else {
            eprintln!("golden-фикстуры не найдены — пропуск (scripts/make_golden.py)");
            return;
        };
        assert!(fixtures.len() >= 5, "ожидается >= 5 фикстур");
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        for fx in &fixtures {
            let blob = std::fs::read(format!("{dir}/{}.vqf", fx.stem)).expect("фикстура");
            let parsed = parse_vqf(&blob).expect("parse");
            assert_eq!(parsed.frames.len(), fx.n_frames, "{}: кадров", fx.stem);
            assert_eq!(parsed.info.mode_key(), fx.mode, "{}: режим", fx.stem);
            assert_eq!(parsed.info.n_ch, fx.n_ch, "{}: каналы", fx.stem);
            // фикстуры усечены по байтовой границе → хвост 0..7 бит (нулевые)
            assert!(parsed.leftover_bits < 8, "{}: leftover {}", fx.stem, parsed.leftover_bits);
            for (i, fr) in parsed.frames.iter().enumerate() {
                assert_eq!(fr.wtype, fx.wtypes[i], "{}[{}] wtype", fx.stem, i);
            }

            // serialize → те же биты кадров (область DATA)
            let data = &blob[parsed.header.data_offset..];
            let total = parsed.info.total_bits();
            let mut w = BitWriter::new();
            for fr in &parsed.frames {
                fr.serialize(&mut w, &parsed.info);
            }
            assert_eq!(w.bit_pos(), total * parsed.frames.len(), "{}", fx.stem);
            let rewritten = w.bytes();
            let nbits = total * parsed.frames.len();
            for bit in 0..nbits {
                let orig = (data[bit / 8] >> (7 - bit % 8)) & 1;
                let mine = (rewritten[bit / 8] >> (7 - bit % 8)) & 1;
                assert_eq!(orig, mine, "{}: бит {}", fx.stem, bit);
            }
        }
    }

    /// Полный файл: write_vqf → parse_vqf → те же поля (round-trip своей
    /// записи, включая контейнер).
    #[test]
    fn file_write_parse_roundtrip() {
        let si = StreamInfo::new(22, 32, 1).unwrap();
        let mut frames = Vec::new();
        for k in 0..5 {
            let mut fr = FrameData::new(&si, if k == 1 { 2 } else { 0 });
            fr.main_coeffs.iter_mut().enumerate().for_each(|(i, v)| {
                *v = ((i + k * 13) % 32) as u8;
            });
            fr.gain_bits[0] = (k * 40) as u8;
            frames.push(fr);
        }
        let blob = write_vqf(&si, &frames, &[]);
        let parsed = parse_vqf(&blob).unwrap();
        assert_eq!(parsed.info.mode_key(), "22_32");
        assert_eq!(parsed.frames, frames);
        // 2 запасных байта в конце
        assert_eq!(&blob[blob.len() - 2..], &[0, 0]);
    }

    /// Заголовок байт-в-байт как у официального энкодера NTT
    /// (эталон 44s96 stereo: header_size=24, COMM=16 c нулевым паддингом,
    /// байт 36 = 0x00, DATA сразу за чанком).
    #[test]
    fn header_matches_ntt_canonical_layout() {
        // 44s96 stereo: копия префикса test2.vqf / clicks.vqf NTT TestEnc
        let ntt_prefix_44s96: [u8; 44] = [
            0x54, 0x57, 0x49, 0x4e, 0x39, 0x37, 0x30, 0x31, // TWIN9701
            0x32, 0x30, 0x30, 0x30, 0x00, 0x00, 0x00, 0x18, // 2000, hs=24
            0x43, 0x4f, 0x4d, 0x4d, 0x00, 0x00, 0x00, 0x10, // COMM len=16
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x60, // stereo, 96kbps
            0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, // 44kHz, pad
            0x44, 0x41, 0x54, 0x41, // DATA
        ];
        let si = StreamInfo::new(44, 48, 2).unwrap();
        let fr = FrameData::new(&si, 0);
        let blob = write_vqf(&si, std::slice::from_ref(&fr), &[]);
        assert_eq!(&blob[..44], &ntt_prefix_44s96[..], "префикс 44s96 отличается от NTT");

        // инварианты для всех режимов: hs=24, COMM=16, байт 36 = 0
        for &(isampf, ibps, n_ch) in
            &[(8u32, 8u32, 1usize), (11, 10, 1), (16, 16, 1), (22, 20, 1), (22, 24, 1), (22, 32, 1), (44, 40, 1), (44, 48, 1), (44, 48, 2)]
        {
            let si = StreamInfo::new(isampf, ibps, n_ch).unwrap();
            let fr = FrameData::new(&si, 0);
            let blob = write_vqf(&si, std::slice::from_ref(&fr), &[]);
            assert_eq!(be32(&blob[12..16]), 24, "{isampf}/{ibps}: header_size");
            assert_eq!(&blob[16..20], b"COMM");
            assert_eq!(be32(&blob[20..24]), 16, "{isampf}/{ibps}: COMM len");
            assert_eq!(be32(&blob[24..28]), (n_ch - 1) as u32);
            assert_eq!(be32(&blob[28..32]), si.kbps_total as u32);
            assert_eq!(be32(&blob[32..36]), isampf, "{isampf}/{ibps}: rate_flag");
            assert_eq!(&blob[36..40], &[0, 0, 0, 0], "{isampf}/{ibps}: паддинг COMM");
            assert_eq!(&blob[40..44], b"DATA");
            assert_eq!(parsed_data_start(&blob), 44, "{isampf}/{ibps}: DATA сразу за COMM");
        }
    }

    /// Позиция битстрима (после тега DATA) через парсер заголовка.
    fn parsed_data_start(blob: &[u8]) -> usize {
        parse_header(blob).unwrap().0.data_offset
    }

    /// Канонический порядок тегов + дедупликация (последний побеждает).
    #[test]
    fn tags_order_and_dedupe() {
        let ordered = order_meta_chunks(vec![
            (*b"AUTH", "a".into()),
            (*b"ZZZZ", "custom".into()),
            (*b"NAME", "first".into()),
            (*b"NAME", "second".into()),
            (*b"(c) ", "©".into()),
        ]);
        let ids: Vec<[u8; 4]> = ordered.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![*b"NAME", *b"AUTH", *b"(c) ", *b"ZZZZ"]);
        assert_eq!(ordered[0].1, "second", "повтор NAME: побеждает последнее");
        assert_eq!(ordered[2].1, "©", "не-ASCII в теге проходит как UTF-8");
        assert_eq!(ordered[3].1, "custom", "нестандартный id — в конце как задан");
    }

    /// Поиск chunk-id по имени: все 18 ключей FFmpeg + флаги + прямые id;
    /// служебные чанки недоступны.
    #[test]
    fn tag_id_lookup() {
        assert_eq!(tag_chunk_id("title"), Some(&*b"NAME"));
        assert_eq!(tag_chunk_id("--copyright"), Some(&*b"(c) "));
        assert_eq!(tag_chunk_id("NAME"), Some(&*b"NAME"));
        assert_eq!(tag_key(b"COMT"), Some("comment"));
        assert_eq!(tag_key(b"FILE"), Some("filename"));
        assert_eq!(tag_chunk_id("COMM"), None, "служебный чанк — не тег");
        assert_eq!(tag_chunk_id("DATA"), None);
        assert_eq!(TAG_TABLE.len(), 18, "все теги из vqf_metadata_conv");
        assert!(!is_valid_tag_chunk(b"DATA"));
        assert!(!is_valid_tag_chunk(b"TWI"));
        assert!(is_valid_tag_chunk(b"TRK1"));
    }

    /// Теги: round-trip + бюджет header_size = 24 + Σ(8+len), DATA сразу
    /// за последним тегом; нечётные длины допустимы, выравнивания нет
    /// (test3.vqf: 26/19/9/21 байт).
    #[test]
    fn tags_roundtrip_and_budget() {
        let si = StreamInfo::new(44, 48, 1).unwrap();
        let fr = FrameData::new(&si, 0);
        let tags = vec![
            (*b"NAME", "T".repeat(19)),
            (*b"COMT", "C".repeat(9)),
            (*b"AUTH", "A".repeat(21)),
        ];
        let blob = write_vqf(&si, std::slice::from_ref(&fr), &tags);
        let hs = 24 + (8 + 19) + (8 + 9) + (8 + 21);
        assert_eq!(be32(&blob[12..16]) as usize, hs);
        assert_eq!(parsed_data_start(&blob), 16 + hs + 4);
        let (h, _) = parse_header(&blob).unwrap();
        let want: Vec<([u8; 4], Vec<u8>)> = tags
            .iter()
            .map(|(i, v)| (*i, v.as_bytes().to_vec()))
            .collect();
        assert_eq!(h.meta_chunks, want, "меты пережили round-trip без изменений");
        // кадры за метами не сместились
        assert_eq!(parse_vqf(&blob).unwrap().frames.len(), 1);
    }

    /// Заголовок с тегами байт-в-байт как test3.vqf (валидация Task 6):
    /// NAME/COMT/AUTH/(c) после COMM, header_size=131, DATA-тег @147.
    /// Файл test3.vqf принимают FFmpeg/tequila/канон NTT — воспроизводим
    /// его префикс точно.
    #[test]
    fn header_with_tags_matches_test3() {
        let si = StreamInfo::new(44, 48, 2).unwrap();
        let fr = FrameData::new(&si, 0);
        let tags = order_meta_chunks(vec![
            (*b"NAME", "You Shot A Hole In My Soul".into()),
            (*b"COMT", "Best music from 90s".into()),
            (*b"AUTH", "C.C.Catch".into()),
            (*b"(c) ", "SonyMusics Production".into()),
        ]);
        let blob = write_vqf(&si, std::slice::from_ref(&fr), &tags);
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(b"TWIN97012000");
        expect.extend_from_slice(&131u32.to_be_bytes());
        expect.extend_from_slice(
            b"COMM\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00\x60\x00\x00\x00\x2c\x00\x00\x00\x00",
        );
        expect.extend_from_slice(b"NAME\x00\x00\x00\x1aYou Shot A Hole In My Soul");
        expect.extend_from_slice(b"COMT\x00\x00\x00\x13Best music from 90s");
        expect.extend_from_slice(b"AUTH\x00\x00\x00\x09C.C.Catch");
        expect.extend_from_slice(b"(c) \x00\x00\x00\x15SonyMusics Production");
        expect.extend_from_slice(b"DATA");
        assert_eq!(&blob[..expect.len()], &expect[..], "префикс test3.vqf не воспроизведён");
        assert_eq!(parsed_data_start(&blob), 151);
        // порядок сохранён парсером
        let (h, _) = parse_header(&blob).unwrap();
        let ids: Vec<[u8; 4]> = h.meta_chunks.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![*b"NAME", *b"COMT", *b"AUTH", *b"(c) "]);
    }

    /// Демиксерная семантика: после последнего кадра остаются только
    /// паддинг-биты (у эталонов NTT — 0..4 бита, у своей записи ≤ 7 + 16).
    #[test]
    fn parse_past_end_is_zeros() {
        let si = StreamInfo::new(44, 48, 1).unwrap();
        let fr = FrameData::new(&si, 0);
        let blob = write_vqf(&si, std::slice::from_ref(&fr), &[]);
        let parsed = parse_vqf(&blob).unwrap();
        assert_eq!(parsed.frames.len(), 1);
        assert!(parsed.leftover_bits < si.total_bits());
    }

    /// Полный корпус NTT (218 .vqf): parse → serialize → биты идентичны.
    /// Запускается при TVQ_CORPUS=<корпус> (как python/analysis):
    /// `TVQ_CORPUS=… cargo test corpus_roundtrip -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn corpus_roundtrip_all() {
        let base = std::env::var("TVQ_CORPUS").expect("TVQ_CORPUS не задан");
        let mut files = Vec::new();
        collect_vqf(std::path::Path::new(&base), &mut files);
        assert!(!files.is_empty(), "в {base} нет .vqf");
        let mut total = 0usize;
        let mut frames_all = 0usize;
        let mut wtypes = [0usize; 9];
        for f in &files {
            let blob = std::fs::read(f).expect("read");
            let parsed = parse_vqf(&blob).expect("parse");
            let data = &blob[parsed.header.data_offset..];
            let tb = parsed.info.total_bits();
            let mut w = BitWriter::new();
            for fr in &parsed.frames {
                fr.serialize(&mut w, &parsed.info);
                wtypes[fr.wtype as usize] += 1;
            }
            assert_eq!(w.bit_pos(), tb * parsed.frames.len(), "{}", f.display());
            let rw = w.bytes();
            for bit in 0..tb * parsed.frames.len() {
                assert_eq!(
                    (data[bit / 8] >> (7 - bit % 8)) & 1,
                    (rw[bit / 8] >> (7 - bit % 8)) & 1,
                    "{}: бит {}",
                    f.display(),
                    bit
                );
            }
            // у эталонов NTT хвост — малый паддинг (наблюдаются 0..6 бит)
            assert!(parsed.leftover_bits < 8, "{}: хвост {}", f.display(), parsed.leftover_bits);
            total += 1;
            frames_all += parsed.frames.len();
        }
        eprintln!(
            "корпус: {total} файлов, {frames_all} кадров, wtype-гистограмма {wtypes:?}"
        );
    }

    fn collect_vqf(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    collect_vqf(&p, out);
                } else if p.extension().map(|x| x == "vqf").unwrap_or(false) {
                    out.push(p);
                }
            }
        }
    }
}
