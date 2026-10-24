//! Минимальный WAV I/O: чтение/запись PCM16 (моно/стерео).
//!
//! Возвращает планарные данные `[канал][сэмпл]` как f64 (±1.0).
//! Поддерживается ровно тот формат, который генерирует корпус сигналов
//! и `ffmpeg -c:a pcm_s16le`.

/// Прочитать WAV → планарные каналы (f64, деление на 32768).
pub fn read_wav(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let d = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if d.len() < 44 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return Err(format!("{path}: не RIFF/WAVE"));
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16, u16)> = None; // (audio_fmt, ch, sr, bits, block)
    let mut data: Option<(usize, usize)> = None; // (off, len)
    while off + 8 <= d.len() {
        let tag = &d[off..off + 4];
        let len = u32::from_le_bytes([d[off + 4], d[off + 5], d[off + 6], d[off + 7]]) as usize;
        let p = off + 8;
        if p + len > d.len() {
            return Err(format!("{path}: чанк обрезан"));
        }
        match tag {
            b"fmt " => {
                if len < 16 {
                    return Err(format!("{path}: fmt короткий"));
                }
                let rd = |o: usize| u16::from_le_bytes([d[p + o], d[p + o + 1]]);
                let rd32 = |o: usize| {
                    u32::from_le_bytes([d[p + o], d[p + o + 1], d[p + o + 2], d[p + o + 3]])
                };
                fmt = Some((rd(0), rd(2), rd32(4), rd(14), rd(12)));
            }
            b"data" => data = Some((p, len)),
            _ => {}
        }
        off = p + len + (len & 1); // чанки выровнены на 2
    }
    let (audio_fmt, n_ch, _sr, bits, _block) =
        fmt.ok_or_else(|| format!("{path}: нет fmt "))?;
    if audio_fmt != 1 && audio_fmt != 0xFFFE {
        return Err(format!("{path}: не PCM (fmt {audio_fmt})"));
    }
    if bits != 16 {
        return Err(format!("{path}: не 16 бит ({bits})"));
    }
    let n_ch = n_ch as usize;
    if n_ch < 1 || n_ch > 2 {
        return Err(format!("{path}: каналов {n_ch}"));
    }
    let (p, len) = data.ok_or_else(|| format!("{path}: нет data"))?;
    let n_samples = len / (2 * n_ch);
    let mut out = vec![vec![0.0f64; n_samples]; n_ch];
    for i in 0..n_samples {
        for (ch, o) in out.iter_mut().enumerate() {
            let b0 = d[p + (i * n_ch + ch) * 2] as i16;
            let b1 = d[p + (i * n_ch + ch) * 2 + 1] as i16;
            let v = ((b1 as i32) << 8 | (b0 as i32 & 0xFF)) as i16;
            o[i] = v as f64 / 32768.0;
        }
    }
    Ok(out)
}

/// Записать планарные каналы в WAV PCM16 (клиппинг ±1).
pub fn write_wav(path: &str, chans: &[Vec<f64>], sr: u32) -> Result<(), String> {
    let n_ch = chans.len();
    let n = chans.first().map(|c| c.len()).unwrap_or(0);
    if chans.iter().any(|c| c.len() != n) {
        return Err("каналы разной длины".into());
    }
    let data_len = n * n_ch * 2;
    let mut d = Vec::with_capacity(44 + data_len);
    let wr16 = |d: &mut Vec<u8>, v: u16| d.extend_from_slice(&v.to_le_bytes());
    let wr32 = |d: &mut Vec<u8>, v: u32| d.extend_from_slice(&v.to_le_bytes());
    d.extend_from_slice(b"RIFF");
    wr32(&mut d, 36 + data_len as u32);
    d.extend_from_slice(b"WAVE");
    d.extend_from_slice(b"fmt ");
    wr32(&mut d, 16);
    wr16(&mut d, 1); // PCM
    wr16(&mut d, n_ch as u16);
    wr32(&mut d, sr);
    wr32(&mut d, sr * n_ch as u32 * 2);
    wr16(&mut d, (n_ch * 2) as u16);
    wr16(&mut d, 16);
    d.extend_from_slice(b"data");
    wr32(&mut d, data_len as u32);
    for i in 0..n {
        for c in chans {
            let v = (c[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            d.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, d).map_err(|e| format!("write {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip() {
        let ch0: Vec<f64> = (0..1000).map(|i| ((i as f64) * 0.01).sin() * 0.5).collect();
        let ch1: Vec<f64> = (0..1000).map(|i| -((i as f64) * 0.013).cos() * 0.3).collect();
        let path = std::env::temp_dir().join("tequila_wav_test.wav");
        let path = path.to_str().unwrap();
        write_wav(path, &[ch0.clone(), ch1.clone()], 44100).unwrap();
        let back = read_wav(path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].len(), 1000);
        for (a, b) in back[0].iter().zip(ch0.iter()) {
            assert!((a - b).abs() < 1.5 / 32768.0);
        }
        for (a, b) in back[1].iter().zip(ch1.iter()) {
            assert!((a - b).abs() < 1.5 / 32768.0);
        }
        std::fs::remove_file(path).ok();
    }
}
