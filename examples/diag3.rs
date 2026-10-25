use tequila::codec::StreamInfo;
use tequila::encoder::{analysis_targets, Encoder, EncoderOpts};
use tequila::decode::Decoder;

fn run(forced: u8, click_at: usize, n: usize) -> (usize, f64) {
    let info = StreamInfo::new(44, 48, 1).unwrap();
    let size = info.mode.size;
    let mut x = vec![0.0f64; n];
    x[click_at] = 1.0;
    let mut padded = x.clone();
    padded.resize(n + 6 * size, 0.0);
    let n_frames = n.div_ceil(size) + 2;
    let opts = EncoderOpts { ppc_enabled: false, ..Default::default() };
    let mut enc = Encoder::new(info, opts);
    let mut frames = vec![];
    for k in 0..n_frames {
        let w = if k < 3 { [1u8, 2, 3][k] } else { forced };
        let x = analysis_targets(&info, k, w, forced, &[padded.clone()]);
        let fe = enc.encode_frame(w, x, None);
        frames.push(fe.frame);
    }
    let mut dec = Decoder::new(info);
    let mut best = (0usize, 0.0f64);
    for k in 0..n_frames {
        let pcm = dec.decode_frame(&frames[k]);
        for (i, v) in pcm[0].iter().enumerate() {
            if v.abs() > best.1 {
                best = (k * size + i, v.abs()); // позиция в конкатенации outs
            }
        }
    }
    best
}

fn main() {
    let size = 2048;
    let n = 24 * size;
    let click = 10 * size + 500;
    for forced in [0u8, 2] {
        let (pos, amp) = run(forced, click, n);
        println!("forced wtype={forced} (no-ppc): пик на {pos} (кадр {:.2}), амплитуда {amp:.3}; вход на {click} (кадр {:.2})",
            pos as f64 / size as f64, click as f64 / size as f64);
    }
}
