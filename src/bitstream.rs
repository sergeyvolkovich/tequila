//! Побитовый I/O MSB-first — семантика FFmpeg `get_bits` / NTT `put_strm`.
//!
//! Кадры VQF упакованы **на битовом уровне вплотную**: кадр `i` начинается
//! с бита `i * frame_bit_len`, байтовой выравнивки нет (docs/ARCHITECTURE.md
//! §1). Поэтому весь DATA-чанк пишется одним `BitWriter`, а после каждого
//! кадра позиция добивается нулями до границы кадра (`align_zero`).
//!
//! Чтение за концом потока возвращает нули — как `get_bits` FFmpeg
//! (демиксер VQF может обращаться к последнему неполному байту).

/// Запись битов MSB-first в непрерывный поток.
#[derive(Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { buf: Vec::new(), bit_pos: 0 }
    }

    /// Пишет младшие `nbits` бит `value` (старшие биты `value` игнорируются).
    ///
    /// `nbits` не должен превышать 32.
    pub fn write(&mut self, value: u32, nbits: usize) {
        debug_assert!(nbits <= 32);
        if nbits == 0 {
            return;
        }
        for i in (0..nbits).rev() {
            let bit = (value >> i) & 1;
            if self.bit_pos % 8 == 0 {
                self.buf.push(0);
            }
            if bit != 0 {
                let last = self.buf.len() - 1;
                self.buf[last] |= 1u8 << (7 - self.bit_pos % 8);
            }
            self.bit_pos += 1;
        }
    }

    /// Запись знакового значения (два-дополнение, младшие `nbits` бит).
    pub fn write_signed(&mut self, value: i32, nbits: usize) {
        self.write(value as u32, nbits);
    }

    /// Добивает поток нулями до битовой позиции `nbits_total`
    /// (битовый бюджет кадра; паддинг кадра).
    pub fn align_zero(&mut self, nbits_total: usize) {
        while self.bit_pos < nbits_total {
            self.write(0, 1);
        }
        debug_assert_eq!(self.bit_pos, nbits_total);
    }

    /// Дописывает байты (по 8 бит каждый, MSB-first).
    pub fn append_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write(b as u32, 8);
        }
    }

    /// Сколько бит записано всего.
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn take_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Чтение битов MSB-first; за концом потока — нули (семантика FFmpeg).
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    nbits: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// `nbits` — длина потока в битах (может не быть кратной 8).
    pub fn new(data: &'a [u8], nbits: usize) -> Self {
        BitReader { data, nbits, bit_pos: 0 }
    }

    /// Читает `nbits` (≤ 32); за концом — нули, позиция всё равно растёт.
    pub fn read(&mut self, nbits: usize) -> u32 {
        debug_assert!(nbits <= 32);
        let mut v: u32 = 0;
        for _ in 0..nbits {
            v <<= 1;
            if self.bit_pos < self.nbits {
                let byte = self.data[self.bit_pos / 8];
                v |= u32::from((byte >> (7 - self.bit_pos % 8)) & 1);
            }
            self.bit_pos += 1;
        }
        v
    }

    /// Заглядывание вперёд без сдвига позиции.
    pub fn peek(&self, nbits: usize) -> u32 {
        let mut r = self.clone();
        r.read(nbits)
    }

    pub fn skip(&mut self, nbits: usize) {
        self.bit_pos += nbits;
    }

    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// Сколько бит осталось до конца потока (0, если вышли за границу).
    pub fn remaining(&self) -> usize {
        self.nbits.saturating_sub(self.bit_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_pattern() {
        let mut w = BitWriter::new();
        w.write(0b101, 3);
        w.write(0b0110, 4);
        w.write(0b11111, 5);
        assert_eq!(w.bit_pos(), 12);
        let b = w.bytes();
        assert_eq!(b.len(), 2);
        // 101 0110 1 | 1111 ....
        assert_eq!(b[0], (0b101 << 5) | (0b0110 << 1) | 0b1);
        assert_eq!(b[1], 0b1111_0000);
    }

    #[test]
    fn value_masked_to_nbits() {
        let mut w = BitWriter::new();
        w.write(u32::MAX, 3); // пишутся только 3 младших бита
        assert_eq!(w.bytes(), &[0b1110_0000]);
    }

    #[test]
    fn zero_bits_write_is_noop() {
        let mut w = BitWriter::new();
        w.write(0xDEAD_BEEF, 0);
        assert_eq!(w.bit_pos(), 0);
        assert!(w.bytes().is_empty());
    }

    /// Детерминированный xorshift64 — воспроизводимые «случайные» значения.
    struct XorShift(u64);
    impl XorShift {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
    }

    #[test]
    fn roundtrip_various_widths() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut w = BitWriter::new();
        let mut expect = Vec::new();
        for nbits in 1..=17usize {
            let v = if nbits < 32 {
                rng.next_u32() & ((1u32 << nbits) - 1)
            } else {
                rng.next_u32()
            };
            w.write(v, nbits);
            expect.push((v, nbits));
        }
        let nbits_total = expect.iter().map(|e| e.1).sum::<usize>();
        let mut r = BitReader::new(w.bytes(), nbits_total);
        for (v, nbits) in expect {
            assert_eq!(r.read(nbits), v, "width {nbits}");
        }
        assert_eq!(r.bit_pos(), nbits_total);
    }

    #[test]
    fn signed_two_complement() {
        let mut w = BitWriter::new();
        w.write_signed(-3, 6);
        // -3 в дополнительном коде: 111101
        let mut r = BitReader::new(w.bytes(), 6);
        assert_eq!(r.read(6), 0b111101);
    }

    #[test]
    fn signed_roundtrip() {
        for v in [-1i32, -3, -32, -33, 0, 31, 32] {
            let mut w = BitWriter::new();
            w.write_signed(v, 7);
            let mut r = BitReader::new(w.bytes(), 7);
            let raw = r.read(7);
            assert_eq!(raw, (v as u32) & 0x7F, "v={v}");
        }
    }

    #[test]
    fn past_end_yields_zeros() {
        let data = [0b1010_1010u8];
        let mut r = BitReader::new(&data, 8);
        assert_eq!(r.read(8), 0xAA);
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.read(4), 0); // за концом — нули
        assert_eq!(r.read(8), 0);
        assert_eq!(r.bit_pos(), 20);
    }

    #[test]
    fn frames_packed_bit_back_to_back() {
        // Два 6-битных кадра: 101101 | 011001 → байты 10110101 1001....
        let mut w = BitWriter::new();
        w.write(0b101101, 6);
        w.align_zero(6);
        w.write(0b011001, 6);
        w.align_zero(12);
        assert_eq!(w.bytes(), &[0b1011_0101, 0b1001_0000]);

        let mut r = BitReader::new(w.bytes(), 12);
        assert_eq!(r.read(6), 0b101101);
        assert_eq!(r.read(6), 0b011001);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn frame_padding_non_byte_aligned() {
        // Кадр полезной длины 5, бюджет кадра 6: паддинг одним нулевым битом.
        let mut w = BitWriter::new();
        w.write(0b10111, 5);
        w.align_zero(6);
        w.write(0b000001, 6); // следующий кадр начинается с бита 6
        assert_eq!(w.bytes(), &[0b1011_1000, 0b0001_0000]);

        let mut r = BitReader::new(w.bytes(), 12);
        assert_eq!(r.read(6), 0b101110);
        assert_eq!(r.read(6), 0b000001);
    }

    #[test]
    fn peek_and_skip() {
        let data = [0b1100_1010u8];
        let mut r = BitReader::new(&data, 8);
        assert_eq!(r.peek(4), 0b1100);
        assert_eq!(r.bit_pos(), 0);
        r.skip(4);
        assert_eq!(r.peek(4), 0b1010);
        assert_eq!(r.read(4), 0b1010);
        assert_eq!(r.bit_pos(), 8);
    }

    #[test]
    fn append_bytes_roundtrip() {
        let mut w = BitWriter::new();
        w.write(0b1, 1);
        w.append_bytes(&[0xDE, 0xAD]);
        assert_eq!(w.bit_pos(), 17);
        let mut r = BitReader::new(w.bytes(), 17);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read(8), 0xDE);
        assert_eq!(r.read(8), 0xAD);
    }
}
