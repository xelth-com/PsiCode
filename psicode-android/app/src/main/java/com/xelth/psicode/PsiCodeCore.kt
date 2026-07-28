package com.xelth.psicode

/**
 * Тонкая обёртка над Rust-ядром psicode_rx (JNI).
 *
 * Загрузка .so обёрнута в try/catch: если библиотека не собрана (cargo ndk ещё
 * не прогонялся), приложение НЕ падает — [available] == false, а UI показывает
 * баннер вместо краша. Все native-вызовы разрешены только при available == true.
 *
 * Сигнатуры native-методов заморожены — Rust-сторона реализует ровно их.
 */
object PsiCodeCore {

    /** true, если libpsicode_rx.so успешно загружена. */
    val available: Boolean

    init {
        available = try {
            System.loadLibrary("psicode_rx")
            true
        } catch (e: UnsatisfiedLinkError) {
            // .so нет в APK (jniLibs/arm64-v8a пуст) — работаем в "no core" режиме.
            false
        } catch (e: Throwable) {
            false
        }
    }

    /** Создать приёмник под профиль (cell_size_px из §7). Возвращает нативный handle. */
    /** @param chromatic 1 = §5.1-CL (постоянная яркость, 2 бит/клетку), 0 = моно 1 бит. */
    external fun rxInit(profileCellPx: Int, chromatic: Int): Long

    /**
     * То же плюс явный выбор редакции ЗЧ-рамки (§3.2):
     * 0 = v0 (тихая зона есть), 1 = v1 (полосы, зоны нет), 2 = v1 в цветности.
     * Рамка НЕ определяется автоматически — приёмник ищет ту, что ему назвали.
     */
    external fun rxInitBorder(profileCellPx: Int, chromatic: Int, border: Int): Long

    /** То же плюс выравниватель межсимвольной помехи: 0 = выкл, 1 = вкл. */
    external fun rxInitBorderIsi(profileCellPx: Int, chromatic: Int, border: Int, isi: Int): Long

    /**
     * Скормить один YUV_420_888 кадр. Плоскости передаются как есть (с row padding);
     * strides из [android.media.Image.Plane]. Возвращает JSON-статус (см. MainActivity).
     */
    external fun rxProcessFrame(
        handle: Long,
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int
    ): String // JSON

    /** Забрать собранный payload (или null, если ещё не готов). Вызывать при done && crc_ok. */
    external fun rxTakeResult(handle: Long): ByteArray?

    /** Освободить нативный handle. */
    external fun rxFree(handle: Long)
}
