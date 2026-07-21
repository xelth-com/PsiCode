//! psicode-core: общее ядро оптического канала «монитор -> камера».
//!
//! Сейчас реализован калибровочный профиль: 160-битный код (32 символа,
//! 8 групп по 4) для ручного обратного канала (телефон показывает, человек
//! вводит на передатчике).
//!
//! Слои:
//! - [`gf32`]    — арифметика GF(32)
//! - [`rs`]      — два перемежённых Рид-Соломона RS(16,8): любые 4 опечатки,
//!   до 8 при равномерном распределении
//! - [`base32`]  — eck Base32 с терпимым парсером
//! - [`bits`]    — битовая упаковка 80-битного payload поверх u128
//! - [`profile`] — [`profile::CalibProfile`]: поля, CRC-8, encode/decode строки
//!
//! Crate по умолчанию `no_std` (в тестовой сборке — обычный std, чтобы тесты
//! пользовались std как обычно). Аллокатор нужен только для `String`/`Vec`
//! (`extern crate alloc`). Фича `std` (включена по умолчанию) добавляет
//! `impl std::error::Error` для типов ошибок. Для кодирования есть путь без
//! аллокаций: [`profile::CalibProfile::encode_into`] и
//! [`base32::parse_into`] / [`base32::format_grouped_into`].

#![cfg_attr(not(test), no_std)]

extern crate alloc;

// Фича `std` линкует std, чтобы работали impl std::error::Error (в no_std-сборке
// без неё крейт остаётся чисто core + alloc).
#[cfg(feature = "std")]
extern crate std;

pub mod base32;
pub mod bits;
pub mod gf32;
pub mod profile;
pub mod rs;

pub use profile::{CalibProfile, ChromaMode, ProfileError};
