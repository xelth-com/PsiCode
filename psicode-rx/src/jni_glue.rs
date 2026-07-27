//! JNI-мост для класса `com.xelth.psicode.PsiCodeCore` (§8).
//!
//! Четыре статических native-метода: `rxInit` создаёт сессию и отдаёт непрозрачный
//! указатель (`jlong`), `rxProcessFrame` обрабатывает кадр Camera2 и возвращает
//! JSON-статус, `rxTakeResult` забирает собранный объект (или null), `rxFree`
//! освобождает сессию. КАЖДЫЙ метод обёрнут в `catch_unwind` и возвращает
//! безопасное значение — паника Rust НИКОГДА не должна ронять приложение.
//!
//! Модуль компилируется всегда (и в cdylib для Android, и в rlib для десктоп-
//! тестов): сам по себе он лишь объявляет символы, JVM подставляется в рантайме.

use std::panic::{catch_unwind, AssertUnwindSafe};

use jni::objects::{JByteArray, JClass};
use jni::sys::{jbyteArray, jint, jlong, jstring};
use jni::JNIEnv;

use crate::session::RxSession;

/// Безопасный JSON-статус на случай паники/битого хэндла (совпадает по схеме с
/// [`crate::session::FrameStatus::to_json`]).
const SAFE_STATUS: &str = "{\"detected\":false,\"score\":0.0000,\"rotation\":0,\
\"px_per_cell\":0.00,\"stripes_ok\":0,\"symbols_new\":0,\"k\":0,\"symbols_have\":0,\
\"done\":false,\"crc_ok\":false}";

/// `PsiCodeCore.rxInit(profileCellPx: Int): Long` — создать сессию, вернуть
/// непрозрачный указатель `Box<RxSession>`. При панике/ошибке -> 0 (невалидный).
#[no_mangle]
pub extern "system" fn Java_com_xelth_psicode_PsiCodeCore_rxInit<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
    profile_cell_px: jint,
    chromatic: jint,
) -> jlong {
    catch_unwind(|| {
        let cell = if (8..256).contains(&profile_cell_px) {
            profile_cell_px as u8
        } else {
            12
        };
        Box::into_raw(Box::new(RxSession::with_cell_mode(cell, chromatic != 0))) as jlong
    })
    .unwrap_or(0)
}

/// `PsiCodeCore.rxProcessFrame(handle, y, u, v, width, height, yRowStride,
/// uvRowStride, uvPixelStride): String` — обработать кадр, вернуть JSON-статус.
/// При любой ошибке/панике -> [`SAFE_STATUS`]. Никогда не роняет приложение.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_xelth_psicode_PsiCodeCore_rxProcessFrame<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    y: JByteArray<'local>,
    u: JByteArray<'local>,
    v: JByteArray<'local>,
    width: jint,
    height: jint,
    y_row_stride: jint,
    uv_row_stride: jint,
    uv_pixel_stride: jint,
) -> jstring {
    let json = catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            return None;
        }
        // SAFETY: хэндл выдан rxInit как Box::into_raw и живёт до rxFree.
        let session = unsafe { &mut *(handle as *mut RxSession) };
        let yb = env.convert_byte_array(&y).ok()?;
        let ub = env.convert_byte_array(&u).ok()?;
        let vb = env.convert_byte_array(&v).ok()?;
        let status = session.process_frame_yuv(
            &yb,
            &ub,
            &vb,
            width.max(0) as usize,
            height.max(0) as usize,
            y_row_stride.max(0) as usize,
            uv_row_stride.max(0) as usize,
            uv_pixel_stride.max(0) as usize,
        );
        Some(status.to_json())
    }))
    .ok()
    .flatten()
    .unwrap_or_else(|| SAFE_STATUS.to_string());

    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// `PsiCodeCore.rxTakeResult(handle): ByteArray?` — забрать собранный объект
/// ровно один раз; null, если ещё не готов (или ошибка/паника).
#[no_mangle]
pub extern "system" fn Java_com_xelth_psicode_PsiCodeCore_rxTakeResult<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jbyteArray {
    let bytes = catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            return None;
        }
        // SAFETY: см. rxProcessFrame.
        let session = unsafe { &mut *(handle as *mut RxSession) };
        session.take_result()
    }))
    .ok()
    .flatten();

    match bytes {
        Some(b) => match env.byte_array_from_slice(&b) {
            Ok(arr) => arr.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// `PsiCodeCore.rxFree(handle)` — освободить сессию. Безопасно на 0/повторный вызов
/// (повторный вызов на живой хэндл — двойное освобождение, ответственность вызова).
#[no_mangle]
pub extern "system" fn Java_com_xelth_psicode_PsiCodeCore_rxFree<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: хэндл выдан rxInit; вызывающий гарантирует единственный free.
        unsafe {
            drop(Box::from_raw(handle as *mut RxSession));
        }
    }));
}
