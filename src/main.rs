//! tequila CLI: encode / decode / info.
//!
//! ```text
//! tequila encode <in.wav> <out.vqf> [--mode 44_48] [--no-ppc] [--kappa 1.0]
//!                 [--title S] [--author S] [--comment S] [--copyright S]
//!                 [--genre S] … (ещё 14 тегов) [--tag CHUNK=S]
//! tequila decode <in.vqf> <out.wav>
//! tequila info <in.vqf>
//! ```

use std::process::ExitCode;
use tequila::codec::StreamInfo;
use tequila::container::{
    is_valid_tag_chunk, order_meta_chunks, parse_vqf, tag_key, write_vqf, MetaChunk,
};
use tequila::decode::decode_stream;
use tequila::encoder::{encode_stream, EncoderOpts};

/// CLI-флаги мета-тегов → chunk-id. Все 18 канонических чанков
/// (vqf_metadata_conv FFmpeg): title→NAME, comment→COMT, author→AUTH,
/// copyright→“(c) ”, а также arranger…words.
const TAG_FLAGS: &[(&str, &[u8; 4])] = &[
    ("--title", b"NAME"),
    ("--comment", b"COMT"),
    ("--author", b"AUTH"),
    ("--copyright", b"(c) "),
    ("--arranger", b"ARNG"),
    ("--band", b"BAND"),
    ("--conductor", b"CDCT"),
    ("--filename", b"FILE"),
    ("--genre", b"GENR"),
    ("--publisher", b"LABL"),
    ("--composer", b"MUSC"),
    ("--note", b"NOTE"),
    ("--producer", b"PROD"),
    ("--personnel", b"PRSN"),
    ("--remixer", b"REMX"),
    ("--singer", b"SING"),
    ("--track", b"TRCK"),
    ("--words", b"WORD"),
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        return ExitCode::from(2);
    }
    let cmd = args[1].as_str();
    let rest = &args[2..];
    match cmd {
        "encode" => cmd_encode(rest),
        "decode" => cmd_decode(rest),
        "info" => cmd_info(rest),
        "bench" => cmd_bench(rest),
        "-h" | "--help" | "help" => {
            usage();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("неизвестная команда: {cmd}");
            usage();
            ExitCode::from(2)
        }
    }
}

fn usage() {
    eprintln!("tequila — cleanroom TwinVQ (VQF) энкодер\n\
               \n\
               USAGE:\n\
               \x20 tequila encode <in.wav> <out.vqf> [--mode 44_48] [--no-ppc] [--kappa F]\n\
               \x20                 [--threads N] [--psy [--psy-beta F] [--psy-tmn F] …]\n\
               \x20                 [--title S] [--author S] [--comment S] [--copyright S]\n\
               \x20                 [--genre S] [--track S] … (18 тегов) [--tag CHUNK=S]\n\
               \x20 tequila decode <in.vqf> <out.wav>\n\
               \x20 tequila info <in.vqf>\n\
               \x20 tequila bench [in.wav] [--mode 44_48] — 3 прогона, лучший, RT-фактор\n\
               \n\
               Режимы: 08_08 11_08 11_10 16_16 22_20 22_24 22_32 44_40 44_48\n\
               (auto: максимальный битрейт для частоты входа)\n\
               \n\
               --psy — психоакустическая модель (docs/PSY.md): перцептивные веса\n\
               \x20        ошибки квантования + адаптивный κ + pre/post-masking,\n\
               \x20        битрейт/битстрим-формат не меняются.\n\
               \x20        Константы: --psy-beta (0.10), --psy-bark-beta (0.1),\n\
               \x20        --psy-lsp-beta (0 = выкл), --psy-no-gain-w (адаптивный κ\n\
               \x20        выключить), --psy-pre-mask (20 мс), --psy-slope-up (15),\n\
               \x20        --psy-slope-down (27), --psy-tmn (24), --psy-nmt (4),\n\
               \x20        --psy-ath-offset (-2), --psy-post-mask (10 дБ/кадр)\n\
               \n\
               Теги (пишутся после COMM, порядок NAME,COMT,AUTH,(c),…):\n\
               \x20 --title --comment --author --copyright --arranger --band\n\
               \x20 --conductor --filename --genre --publisher --composer --note\n\
               \x20 --producer --personnel --remixer --singer --track --words\n\
               \x20 --tag CHUNK=S — произвольный чанк (4 байта, не COMM/DSIZ/DATA)");
}

fn cmd_encode(args: &[String]) -> ExitCode {
    // одиночный проход: позиционные аргументы не должны захватывать
    // значения флагов (например, текст --title)
    let mut pos: Vec<&String> = Vec::new();
    let mut mode: Option<String> = None;
    let mut opts = EncoderOpts::default();
    let mut meta: Vec<MetaChunk> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if let Some((_, id)) = TAG_FLAGS.iter().find(|(f, _)| *f == a) {
            let Some(v) = args.get(i + 1) else {
                eprintln!("ошибка: {a} требует значение");
                return ExitCode::from(2);
            };
            meta.push((**id, v.clone()));
            i += 2;
            continue;
        }
        match a {
            "--mode" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: --mode требует значение");
                    return ExitCode::from(2);
                };
                mode = Some(v.clone());
                i += 2;
            }
            "--no-ppc" => {
                opts.ppc_enabled = false;
                i += 1;
            }
            "--kappa" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: --kappa требует значение");
                    return ExitCode::from(2);
                };
                opts.kappa = v.parse().unwrap_or(1.0);
                i += 2;
            }
            "--threads" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: --threads требует значение (0=авто, 1=однопоточно)");
                    return ExitCode::from(2);
                };
                opts.threads = v.parse().unwrap_or(0);
                i += 2;
            }
            "--psy" => {
                // психоакустическая модель (docs/PSY.md): перцептивные
                // веса ошибки VQ/bark, бюджет кадра не меняется
                if opts.psy.is_none() {
                    opts.psy = Some(Default::default());
                }
                i += 1;
            }
            "--psy-beta" | "--psy-bark-beta" | "--psy-lsp-beta" | "--psy-slope-up"
            | "--psy-slope-down" | "--psy-tmn" | "--psy-nmt" | "--psy-ath-offset"
            | "--psy-post-mask" | "--psy-pre-mask" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: {a} требует числовое значение");
                    return ExitCode::from(2);
                };
                let val: f64 = match v.parse() {
                    Ok(x) => x,
                    Err(_) => {
                        eprintln!("ошибка: {a} требует число, получено «{v}»");
                        return ExitCode::from(2);
                    }
                };
                let p = opts.psy.get_or_insert_with(Default::default);
                match a {
                    "--psy-beta" => p.beta = val,
                    "--psy-bark-beta" => p.bark_beta = val,
                    "--psy-lsp-beta" => p.lsp_beta = val,
                    "--psy-slope-up" => p.slope_up_db = val,
                    "--psy-slope-down" => p.slope_down_db = val,
                    "--psy-tmn" => p.tmn_db = val,
                    "--psy-nmt" => p.nmt_db = val,
                    "--psy-ath-offset" => p.ath_offset_db = val,
                    "--psy-post-mask" => p.post_mask_db = val,
                    _ => p.pre_mask_ms = val,
                }
                i += 2;
            }
            "--psy-no-gain-w" => {
                // абляция адаптивного κ (PSY.md §7): 0/1
                if let Some(p) = opts.psy.as_mut() {
                    p.gain_weighted = false;
                }
                i += 1;
            }
            "--tag" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: --tag требует значение CHUNK=value");
                    return ExitCode::from(2);
                };
                match parse_custom_tag(v) {
                    Some(chunk) => meta.push(chunk),
                    None => {
                        eprintln!("ошибка: --tag ожидает CHUNK=value (CHUNK — ровно 4 байта, не COMM/DSIZ/DATA)");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            other if other.starts_with("--") => {
                eprintln!("неизвестный флаг: {other}");
                return ExitCode::from(2);
            }
            _ => {
                pos.push(&args[i]);
                i += 1;
            }
        }
    }
    if pos.len() < 2 {
        usage();
        return ExitCode::from(2);
    }
    let in_path = pos[0];
    let out_path = pos[1];

    if let Some(p) = opts.psy {
        if !(p.beta > 0.0 && p.beta <= 4.0) {
            eprintln!("ошибка: --psy-beta должен быть в (0, 4]");
            return ExitCode::from(2);
        }
    }

    let chans = match tequila::wav::read_wav(in_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ошибка: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_ch = chans.len();
    let n = chans.first().map(|c| c.len()).unwrap_or(0);
    if n == 0 {
        eprintln!("ошибка: пустой вход");
        return ExitCode::FAILURE;
    }

    // режим: из WAV нет частоты в нашем минималистичном ридере — берём из
    // имени/опции; по умолчанию 44_48
    let isampf_ibps: (u32, u32) = match mode.as_deref() {
        Some(s) => match parse_mode(s) {
            Some(v) => v,
            None => {
                eprintln!("неизвестный режим {s}");
                return ExitCode::from(2);
            }
        },
        None => (44, 48),
    };
    let info = match StreamInfo::new(isampf_ibps.0, isampf_ibps.1, n_ch) {
        Some(i) => i,
        None => {
            eprintln!("режим {}x{n_ch} каналов не поддерживается", isampf_ibps.0);
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "кодирование: {} сэмплов x {n_ch} кан. → {} ({} бит/кадр, {} кадров, threads={})",
        n,
        info.mode_key(),
        info.total_bits(),
        n.div_ceil(info.mode.size) + 2,
        if opts.threads == 0 {
            std::env::var("TEQUILA_THREADS")
                .ok()
                .and_then(|t| t.parse::<usize>().ok())
                .filter(|&t| t > 0)
                .map(|t: usize| t.to_string())
                .unwrap_or_else(|| "авто".to_string())
        } else {
            opts.threads.to_string()
        }
    );
    let t0 = std::time::Instant::now();
    let (frames, wtypes) = encode_stream(&info, &chans, opts);
    eprintln!("готово за {:.2} с (wtype-гистограмма ниже)", t0.elapsed().as_secs_f64());
    let mut hist = [0usize; 9];
    for &w in &wtypes {
        hist[w as usize] += 1;
    }
    eprintln!("wtypes: {hist:?}");

    // заголовок — байт-в-байт канон NTT (COMM=16, без DSIZ): официальный
    // плеер требует поля на фиксированных офсетах; теги — после COMM,
    // в каноническом порядке (NAME первым, повторы схлопнуты)
    let n_tags = meta.len();
    let blob = write_vqf(&info, &frames, &order_meta_chunks(meta));
    if let Err(e) = std::fs::write(out_path, &blob) {
        eprintln!("ошибка записи: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("записан {out_path} ({} байт){}", blob.len(),
        if n_tags == 0 { String::new() } else { format!(", {n_tags} тегов") });
    ExitCode::SUCCESS
}

/// bench: 3 прогона кодирования (лучший), RT-фактор. Без файла —
/// встроенный многочастотный стереосигнал 5 с @ 44.1 кГц.
fn cmd_bench(args: &[String]) -> ExitCode {
    let mut mode = String::from("44_48");
    let mut file: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("ошибка: --mode требует значение");
                    return ExitCode::from(2);
                };
                mode = v.clone();
                i += 2;
            }
            other if other.starts_with("--") => {
                eprintln!("неизвестный флаг: {other}");
                return ExitCode::from(2);
            }
            _ => {
                file = Some(&args[i]);
                i += 1;
            }
        }
    }
    let Some((isampf, ibps)) = parse_mode(&mode) else {
        eprintln!("неизвестный режим {mode}");
        return ExitCode::from(2);
    };

    let (chans, sr) = match file {
        Some(p) => match tequila::wav::read_wav(p) {
            Ok(c) => (c, 0),
            Err(e) => {
                eprintln!("ошибка: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => {
            // встроенный сигнал: multitone + шум, стерео 5 с
            let sr = 44100u32;
            let n = 5 * sr as usize;
            let mut l = vec![0.0f64; n];
            let mut r = vec![0.0f64; n];
            let mut rng = 0xBADC0DE_5EED_1234u64;
            for t in 0..n {
                let ts = t as f64 / sr as f64;
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let noise = 0.02 * (2.0 * ((rng >> 32) as f64 / u32::MAX as f64) - 1.0);
                l[t] = 0.25
                    * ((2.0 * std::f64::consts::PI * 440.0 * ts).sin()
                        + 0.6 * (2.0 * std::f64::consts::PI * 1234.0 * ts).sin()
                        + 0.4 * (2.0 * std::f64::consts::PI * 3456.0 * ts).sin())
                    + noise;
                r[t] = 0.8 * l[t] + 0.2 * noise;
            }
            (vec![l, r], sr)
        }
    };
    let n_ch = chans.len();
    let n = chans.first().map(|c| c.len()).unwrap_or(0);
    if n == 0 {
        eprintln!("ошибка: пустой вход");
        return ExitCode::FAILURE;
    }
    let Some(info) = StreamInfo::new(isampf, ibps, n_ch) else {
        eprintln!("режим {}x{n_ch} не поддерживается", isampf);
        return ExitCode::FAILURE;
    };
    let dur = if sr > 0 {
        n as f64 / sr as f64
    } else {
        n as f64 / info.sr as f64
    };
    eprintln!(
        "bench: {} с × {n_ch} кан., режим {} ({} бит/кадр, {} кадров)",
        dur,
        info.mode_key(),
        info.total_bits(),
        n.div_ceil(info.mode.size) + 2
    );
    let mut best = f64::INFINITY;
    for run in 1..=3 {
        let t0 = std::time::Instant::now();
        let (frames, _) = encode_stream(&info, &chans, EncoderOpts::default());
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        eprintln!("  прогон {run}: {dt:.3} с ({} кадров)", frames.len());
    }
    eprintln!(
        "итог: лучший {best:.3} с → {:.2}× realtime",
        dur / best
    );
    ExitCode::SUCCESS
}

/// --tag CHUNK=value → MetaChunk (CHUNK — ровно 4 байта, не служебный).
fn parse_custom_tag(kv: &str) -> Option<MetaChunk> {
    let (id, val) = kv.split_once('=')?;
    let b = id.as_bytes();
    if !is_valid_tag_chunk(b) {
        return None;
    }
    Some(([b[0], b[1], b[2], b[3]], val.to_string()))
}

fn parse_mode(s: &str) -> Option<(u32, u32)> {
    let t: Vec<&str> = s.split(['_', '-']).collect();
    if t.len() == 2 {
        let a: u32 = t[0].parse().ok()?;
        let b: u32 = t[1].parse().ok()?;
        Some((a, b))
    } else {
        None
    }
}

fn cmd_decode(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        usage();
        return ExitCode::from(2);
    }
    let blob = match std::fs::read(&args[0]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ошибка: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parsed = match parse_vqf(&blob) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ошибка: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pcm = decode_stream(&parsed.info, &parsed.frames);
    eprintln!(
        "декодировано {} кадров → {} сэмплов/канал ({})",
        parsed.frames.len(),
        pcm.first().map(|c| c.len()).unwrap_or(0),
        parsed.info.mode_key()
    );
    if let Err(e) = tequila::wav::write_wav(&args[1], &pcm, parsed.info.sr) {
        eprintln!("ошибка записи: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn cmd_info(args: &[String]) -> ExitCode {
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    let blob = match std::fs::read(&args[0]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ошибка: {e}");
            return ExitCode::FAILURE;
        }
    };
    let parsed = match parse_vqf(&blob) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ошибка: {e}");
            return ExitCode::FAILURE;
        }
    };
    let h = &parsed.header;
    println!("формат:      VQF TwinVQ (TWIN97012000)");
    println!("режим:       {}", parsed.info.mode_key());
    println!("каналы:      {}", h.n_ch);
    println!("частота:     {} Гц (rate_flag {})", parsed.info.sr, h.rate_flag);
    println!("битрейт:     {} кбит/с ({} на канал)", h.kbps_total, h.kbps_total / h.n_ch as u32);
    println!("размер кадра: {} сэмплов", parsed.info.mode.size);
    println!("бит/кадр:    {}", parsed.info.total_bits());
    println!("кадров:      {} (выход {} сэмплов)", parsed.frames.len(),
        parsed.frames.len().saturating_sub(2) * parsed.info.mode.size);
    println!("DSIZ:         {:?}", h.dsiz);
    println!("хвост DATA:   {} бит", parsed.leftover_bits);
    if !h.meta_chunks.is_empty() {
        println!("теги:");
        for (id, val) in &h.meta_chunks {
            let key = tag_key(id).unwrap_or("custom");
            println!(
                "  {} ({}) = {:?}",
                String::from_utf8_lossy(id),
                key,
                String::from_utf8_lossy(val)
            );
        }
    }
    let mut hist = [0usize; 9];
    for fr in &parsed.frames {
        hist[fr.wtype as usize] += 1;
    }
    println!("wtype-гистограмма: {hist:?}");
    ExitCode::SUCCESS
}
