use psicode_core::{base32, CalibProfile, ChromaMode};

fn main() {
    let p = CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 3,
        chroma_mode: ChromaMode::Chroma2,
        gamma_g_q: 28,
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15,
        black_level_q: 2,
        noise_sigma_q: 12,
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3,
        crosstalk_gb_q: 4,
        quiet_zone: 1,
        fec_overhead: 2,
    };
    let code = p.encode_string().unwrap();
    println!("Код на экране телефона : {code}");

    // человек вводит с четырьмя опечатками, в нижнем регистре, с пробелами
    let mut chars: Vec<char> = code.chars().collect();
    for i in [2usize, 9, 20, 33] {
        let v = base32::char_to_symbol(chars[i]).unwrap();
        chars[i] = base32::symbol_to_char((v + 16) % 32);
    }
    let typed: String = chars
        .iter()
        .map(|&c| if c == '-' { ' ' } else { c.to_ascii_lowercase() })
        .collect();
    println!("Человек ввёл           : {typed}");

    let (q, fixed) = CalibProfile::decode_string(&typed).unwrap();
    println!("Исправлено опечаток    : {fixed}");
    println!("Профиль восстановлен   : {}", q == p);
    println!("  ячейка {} px, {:.0} кадр/с, {} бит люмы, гамма G {:.3}, белый {}%",
        q.cell_size_px, q.effective_fps(), q.luma_bits, q.gamma_g(), q.white_level_pct());
}
