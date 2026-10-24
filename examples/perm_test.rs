fn main() {
    let info = tequila::codec::StreamInfo::new(44, 48, 2).unwrap();
    let perm = tequila::dsp::construct_perm(info.mode, &info.layout, tequila::modes::FrameType::Long, 2);
    let size = info.mode.size;
    println!("n={} size={}", perm.len(), size);
    // первые 20 позиций VQ-порядка: куда идут t=0..19
    for t in 0..20 {
        let p = perm[t] as usize;
        println!("t={t:2} -> bin {p:5} (ch {}, бин {})", p / size, p % size);
    }
    // проверка гипотазы 2m/2m+1: соседние позиции VQ = один бин двух каналов?
    let mut ok = 0; let mut bad = 0;
    for t in (0..perm.len()-1).step_by(2) {
        let a = perm[t] as usize; let b = perm[t+1] as usize;
        if a % size == b % size && a / size != b / size { ok += 1 } else { bad += 1 }
    }
    println!("пары (2m,2m+1) с одинаковым бином разных каналов: ok={ok} bad={bad}");
}
