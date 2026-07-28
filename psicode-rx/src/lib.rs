//! psicode-rx: приёмное ядро Android (§8).
//!
//! Оборачивает проверенный приёмный тракт psicode-core (детекция ЗЧ-рамки §3.2 ->
//! демодуляция §5 с самокалибровкой тона §3.4 -> разбор L3 §6.2 -> XOR-фонтан
//! §9.2) для кадров Camera2 `YUV_420_888`, за интерфейсом JNI.
//!
//! Слои:
//! - [`yuv`]      — разбор YUV_420_888 и BT.601 limited-range -> RGB
//! - [`session`]  — [`session::RxSession`]: чистый Rust, десктоп-тестируемый
//! - `jni_glue`   — статические native-методы `com.xelth.psicode.PsiCodeCore`
//!
//! `RxSession` полностью работает без JNI (десктоп-тесты гоняют весь тракт по
//! rlib-пути), а `jni_glue` — тонкая безопасная (`catch_unwind`) обёртка.

pub mod session;
pub mod yuv;

// Компилируется всегда: символы JNI объявляются и в rlib, и в cdylib (JVM
// подставляется в рантайме; на десктопе методы просто не вызываются).
mod jni_glue;

pub use session::{
    tx_chromatic_profile, tx_default_profile, FrameStatus, IsiMode, KernelPool, RxSession,
    RxState,
};
