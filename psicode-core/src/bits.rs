//! Упаковка полей в 80-битный payload поверх u128.
//! Порядок: big-endian по битам — первое записанное поле занимает старшие биты.

pub const PAYLOAD_BITS: usize = 80;

pub struct BitWriter {
    acc: u128,
    used: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self { acc: 0, used: 0 }
    }

    pub fn write(&mut self, value: u32, bits: usize) {
        debug_assert!(bits >= 1 && bits <= PAYLOAD_BITS);
        debug_assert!(self.used + bits <= PAYLOAD_BITS, "payload overflow");
        debug_assert!(
            bits >= 32 || (value as u64) < (1u64 << bits),
            "value {value} does not fit in {bits} bits"
        );
        self.acc |= (value as u128) << (PAYLOAD_BITS - self.used - bits);
        self.used += bits;
    }

    pub fn finish(self) -> u128 {
        debug_assert_eq!(self.used, PAYLOAD_BITS, "payload underfilled: {} bits", self.used);
        self.acc
    }
}

pub struct BitReader {
    acc: u128,
    pos: usize,
}

impl BitReader {
    pub fn new(acc: u128) -> Self {
        Self { acc, pos: 0 }
    }

    pub fn read(&mut self, bits: usize) -> u32 {
        debug_assert!(self.pos + bits <= PAYLOAD_BITS);
        let shift = PAYLOAD_BITS - self.pos - bits;
        let mask: u128 = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
        self.pos += bits;
        ((self.acc >> shift) & mask) as u32
    }
}

/// u128 (80 бит) <-> 16 пятибитных символов, старшие биты — первый символ.
pub fn payload_to_symbols(p: u128) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, s) in out.iter_mut().enumerate() {
        let shift = PAYLOAD_BITS - 5 * (i + 1);
        *s = ((p >> shift) & 0x1F) as u8;
    }
    out
}

pub fn symbols_to_payload(symbols: &[u8; 16]) -> u128 {
    let mut p: u128 = 0;
    for &s in symbols.iter() {
        debug_assert!(s < 32);
        p = (p << 5) | s as u128;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_roundtrip() {
        let p: u128 = 0x0123_4567_89AB_CDEF_0123_4567_89AB_u128 & ((1u128 << 80) - 1);
        assert_eq!(symbols_to_payload(&payload_to_symbols(p)), p);
    }

    #[test]
    fn writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        w.write(0xA, 4);
        w.write(63, 6);
        w.write(0, 2);
        w.write(0x1FFFF, 17);
        w.write(1, 1);
        w.write(0, 50);
        let p = w.finish();
        let mut r = BitReader::new(p);
        assert_eq!(r.read(4), 0xA);
        assert_eq!(r.read(6), 63);
        assert_eq!(r.read(2), 0);
        assert_eq!(r.read(17), 0x1FFFF);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read(50), 0);
    }
}
