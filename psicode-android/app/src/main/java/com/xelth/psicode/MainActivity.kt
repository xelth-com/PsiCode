package com.xelth.psicode

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * PsiCode receiver shell (SPEC §8).
 * Camera2 с блокировкой 3A -> YUV_420_888 кадры -> JNI в Rust-ядро (psicode_rx).
 * Никакой обработки изображения в Kotlin — только транспорт кадров и отрисовка телеметрии.
 */
class MainActivity : AppCompatActivity() {

    // --- Views ---
    private lateinit var preview: SurfaceView
    private lateinit var statusView: TextView
    private lateinit var bannerView: TextView
    private lateinit var progressBar: ProgressBar

    // --- Camera2 ---
    private lateinit var cameraManager: CameraManager
    private var cameraId: String? = null
    private var characteristics: CameraCharacteristics? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var captureSize: Size = Size(1920, 1080)
    private lateinit var previewBuilder: CaptureRequest.Builder

    private var camThread: HandlerThread? = null
    private var camHandler: Handler? = null
    private var procThread: HandlerThread? = null
    private var procHandler: Handler? = null

    private var surfaceReady = false
    private var permissionGranted = false

    // --- 3A lock FSM ---
    // WAITING_AF оставлен для совместимости, но НЕ используется: contrast-AF (CONTINUOUS_PICTURE)
    // на time-varying коде сходится случайно — payload мерцает каждые 100мс, метрика контраста
    // осциллирует. Фокус ставим decoder-guided свипом (FOCUS_SWEEP), см. startSweep().
    private enum class Af3A { WAITING_CONVERGE, WAITING_AF, FOCUS_SWEEP, LOCKED }
    @Volatile private var afState = Af3A.WAITING_CONVERGE   // читается из proc-потока (handleStatus)
    private var framesInState = 0
    private var hasAf = false
    private var afOffSupported = false
    private var lastFocusDistance: Float? = null
    // сошедшиеся значения экспозиции, снятые ДО локирования (item 1: экспозиционный пол)
    private var convergedExposureNs: Long? = null
    private var convergedIso: Int? = null

    // --- возможности сенсора (ручная экспозиция против бандинга) ---
    private var manualSensorSupported = false
    private var isoMin = 0
    private var isoMax = 0
    private var expMinNs = 0L
    private var expMaxNs = Long.MAX_VALUE

    // --- здоровье захвата после лока (item 2: AF-recovery) ---
    @Volatile private var lastDetectMs = 0L      // последний кадр с detected==true (база = момент лока)
    @Volatile private var lastRecoveryMs = 0L    // троттлинг попыток восстановления
    @Volatile private var noDetectActive = false // идёт ли серия без детекта
    private var recoveryCount = 0                // только cam-поток

    // --- decoder-guided focus sweep (заменяет contrast-AF на time-varying коде) ---
    @Volatile private var sweepActive = false
    @Volatile private var sweepObserveFrom = Long.MAX_VALUE  // до этого времени шаг «устаканивается»
    @Volatile private var curStepBestScore = 0.0             // лучший score текущего шага (пишет proc)
    @Volatile private var sweepStepResults = 0               // обработанных кадров на текущем шаге
    private var sweepIndex = 0                               // только cam-поток
    private var sweepBestScore = 0.0                         // только cam-поток
    private var sweepBestD = -1.0f                           // лучшая дистанция (диоптрии)
    private var sweepSteps = floatArrayOf()                  // текущая таблица шагов
    private var sweepFine = false                            // фаза: false=грубая, true=тонкая
    private var sweepGen = 0                                 // поколение шага (failsafe-таймер)

    // --- Rust core ---
    private var handle: Long = 0L
    private val coreReady get() = PsiCodeCore.available && handle != 0L

    // --- Frame feed (single proc thread, drop-while-busy via acquireLatest + time throttle) ---
    private var lastProcMs = 0L
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)
    private var dumpCount = 0 // отладка: дамп кадров в filesDir
    private var procCount = 0 // счётчик обработанных кадров (дампим после сходимости 3A)
    @Volatile private var saved = false

    private val permLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            permissionGranted = granted
            if (granted) maybeOpenCamera()
            else showBanner(getString(R.string.perm_needed), Color.parseColor("#B00020"))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        preview = findViewById(R.id.preview)
        statusView = findViewById(R.id.status)
        bannerView = findViewById(R.id.banner)
        progressBar = findViewById(R.id.progress)

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        selectCamera()

        // Инициализация Rust-ядра (или "no core" баннер).
        if (PsiCodeCore.available) {
            handle = try {
                PsiCodeCore.rxInit(PROFILE_CELL_PX)
            } catch (t: Throwable) {
                0L
            }
            if (handle == 0L) showBanner("rxInit failed", Color.parseColor("#B00020"))
        } else {
            showBanner(getString(R.string.no_core), Color.parseColor("#8A6D00"))
        }

        preview.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                holder.setFixedSize(captureSize.width, captureSize.height)
                surfaceReady = true
                maybeOpenCamera()
            }
            override fun surfaceChanged(h: SurfaceHolder, f: Int, w: Int, ht: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) { surfaceReady = false }
        })

        permissionGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        if (!permissionGranted) permLauncher.launch(Manifest.permission.CAMERA)
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThreads()
        if (permissionGranted && surfaceReady && cameraDevice == null) maybeOpenCamera()
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThreads()
        super.onPause()
    }

    override fun onDestroy() {
        if (handle != 0L) {
            try { PsiCodeCore.rxFree(handle) } catch (_: Throwable) {}
            handle = 0L
        }
        super.onDestroy()
    }

    // --- Camera setup ---

    private fun selectCamera() {
        for (id in cameraManager.cameraIdList) {
            val ch = cameraManager.getCameraCharacteristics(id)
            if (ch.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) {
                cameraId = id
                characteristics = ch
                captureSize = chooseSize(ch)
                configureAfCaps(ch)
                configureSensorCaps(ch)
                return
            }
        }
    }

    /** Ближайший к 1920x1080 поддерживаемый YUV_420_888 размер (по разнице площади). */
    private fun chooseSize(ch: CameraCharacteristics): Size {
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = map?.getOutputSizes(ImageFormat.YUV_420_888) ?: return Size(1920, 1080)
        val targetArea = 1920L * 1080L
        return sizes.minByOrNull { abs(it.width.toLong() * it.height - targetArea) } ?: Size(1920, 1080)
    }

    private fun configureAfCaps(ch: CameraCharacteristics) {
        val afModes = ch.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
        hasAf = afModes?.any { it != CameraMetadata.CONTROL_AF_MODE_OFF } == true
        val minFocus = ch.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
        afOffSupported = (afModes?.contains(CameraMetadata.CONTROL_AF_MODE_OFF) == true) &&
                minFocus != null && minFocus > 0f
    }

    /** Ручная экспозиция (MANUAL_SENSOR) + диапазоны ISO/выдержки для экспозиционного пола (item 1). */
    private fun configureSensorCaps(ch: CameraCharacteristics) {
        val caps = ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        manualSensorSupported =
            caps?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
        ch.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.let {
            isoMin = it.lower; isoMax = it.upper
        }
        ch.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let {
            expMinNs = it.lower; expMaxNs = it.upper
        }
    }

    private fun maybeOpenCamera() {
        if (permissionGranted && surfaceReady && cameraDevice == null) openCamera()
    }

    private fun openCamera() {
        val id = cameraId ?: run {
            showBanner("нет задней камеры", Color.parseColor("#B00020")); return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) return

        imageReader = ImageReader.newInstance(
            captureSize.width, captureSize.height, ImageFormat.YUV_420_888, 3
        ).apply { setOnImageAvailableListener(onImage, procHandler) }

        try {
            cameraManager.openCamera(id, deviceCallback, camHandler)
        } catch (t: Throwable) {
            showBanner("openCamera: ${t.message}", Color.parseColor("#B00020"))
        }
    }

    private val deviceCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(device: CameraDevice) {
            cameraDevice = device
            createSession()
        }
        override fun onDisconnected(device: CameraDevice) { device.close(); cameraDevice = null }
        override fun onError(device: CameraDevice, error: Int) {
            device.close(); cameraDevice = null
            runOnUiThread { showBanner("camera error $error", Color.parseColor("#B00020")) }
        }
    }

    @Suppress("DEPRECATION")
    private fun createSession() {
        val device = cameraDevice ?: return
        val reader = imageReader ?: return
        val surfaces = listOf(preview.holder.surface, reader.surface)
        previewBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(preview.holder.surface)
            addTarget(reader.surface)
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            // ISP-фильтры ВЫКЛ: темпоральное шумоподавление (3DNR) усредняет
            // сменяющиеся кадры передачи в кашу (живой замер 2026-07-26 —
            // страйпы совпадали с далёкими seq); статичная рамка при этом
            // оставалась чёткой. Отключаем всё темпорально-сглаживающее.
            set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
            set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
            set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF
            )
            if (hasAf) set(
                CaptureRequest.CONTROL_AF_MODE,
                CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            )
        }
        afState = Af3A.WAITING_CONVERGE
        framesInState = 0

        device.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                try {
                    session.setRepeatingRequest(previewBuilder.build(), captureCallback, camHandler)
                } catch (t: Throwable) {
                    runOnUiThread { showBanner("repeat: ${t.message}", Color.parseColor("#B00020")) }
                }
            }
            override fun onConfigureFailed(session: CameraCaptureSession) {
                runOnUiThread { showBanner("session config failed", Color.parseColor("#B00020")) }
            }
        }, camHandler)
    }

    // --- 3A convergence -> lock (AE/AWB lock + AF trigger + focus hold, SPEC §8) ---

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            when (afState) {
                Af3A.WAITING_CONVERGE -> {
                    framesInState++
                    val ae = result.get(CaptureResult.CONTROL_AE_STATE)
                    val awb = result.get(CaptureResult.CONTROL_AWB_STATE)
                    val aeOk = ae == null ||
                            ae == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                            ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED ||
                            ae == CaptureResult.CONTROL_AE_STATE_LOCKED
                    val awbOk = awb == null ||
                            awb == CaptureResult.CONTROL_AWB_STATE_CONVERGED ||
                            awb == CaptureResult.CONTROL_AWB_STATE_LOCKED
                    if ((aeOk && awbOk) || framesInState > CONVERGE_TIMEOUT_FRAMES) {
                        // снимаем сошедшуюся экспозицию/ISO ДО фиксации (item 1)
                        convergedExposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                        convergedIso = result.get(CaptureResult.SENSOR_SENSITIVITY)
                        if (afOffSupported) {
                            // Экспозицию пинуем + AWB lock ДО свипа -> кадры консистентны.
                            applyExposurePinAndAwbLock(result)
                            startSweep()               // decoder-guided focus sweep (вместо contrast-AF)
                        } else {
                            lockAll(result)            // fixed-focus/legacy fallback
                        }
                    }
                }
                Af3A.WAITING_AF -> {
                    // НЕ ДОСТИЖИМО: путь contrast-AF отключён на time-varying коде (см. enum-коммент).
                    framesInState++
                    val af = result.get(CaptureResult.CONTROL_AF_STATE)
                    val afDone = af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                            af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED
                    if (afDone || framesInState > CONVERGE_TIMEOUT_FRAMES) {
                        lastFocusDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
                        lockAll(result)
                    }
                }
                Af3A.FOCUS_SWEEP -> { /* свип управляется таймерами camHandler, не результатами 3A */ }
                Af3A.LOCKED -> { /* 3A заморожено, дальнейших действий нет */ }
            }
        }
    }

    // НЕ ИСПОЛЬЗУЕТСЯ: contrast-AF бесполезен на мерцающем payload (§ time-varying codes).
    // Оставлен для истории/справки; фокус ставится decoder-guided свипом (startSweep).
    @Suppress("unused")
    private fun triggerAf() {
        val session = captureSession ?: return
        // Одиночный AF-триггер; репитинг остаётся CONTINUOUS_PICTURE до фиксации.
        previewBuilder.set(
            CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START
        )
        try { session.capture(previewBuilder.build(), captureCallback, camHandler) } catch (_: Throwable) {}
        previewBuilder.set(
            CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE
        )
        afState = Af3A.WAITING_AF
        framesInState = 0
    }

    /**
     * item 1 (edit координатора): пин экспозиции + AWB lock. Мутирует previewBuilder, НЕ строит
     * запрос и НЕ трогает фокус/afState. Короче ~16.7мс — rolling-shutter бэнд подсветки/refresh;
     * длиннее ~20мс — выдержка блендит смену кадра передачи (hold 100мс) в серую кашу. Пин t≈16.7мс
     * минимизирует оба; яркость держим ISO (сохраняем t*ISO).
     */
    private fun applyExposurePinAndAwbLock(result: TotalCaptureResult) {
        previewBuilder.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        val tConv = convergedExposureNs ?: result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
        val isoConv = convergedIso ?: result.get(CaptureResult.SENSOR_SENSITIVITY)
        var chosenExp = tConv
        var chosenIso = isoConv
        if (manualSensorSupported && isoMax > 0 &&
            tConv != null && isoConv != null &&
            (tConv < EXPOSURE_FLOOR_NS || tConv > EXPOSURE_PIN_MAX_NS)) {
            val tNew = EXPOSURE_FLOOR_NS.coerceIn(expMinNs, expMaxNs)
            val isoNew = (isoConv.toLong() * tConv / tNew).toInt().coerceIn(isoMin, isoMax)
            previewBuilder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
            previewBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, tNew)
            previewBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, isoNew)
            chosenExp = tNew
            chosenIso = isoNew
        } else {
            // Выдержка уже в окне или ручной сенсор недоступен -> обычная фиксация AE.
            previewBuilder.set(CaptureRequest.CONTROL_AE_LOCK, true)
        }
        android.util.Log.d(
            "PsiCodeRX",
            "PIN exp=${chosenExp}ns iso=$chosenIso manualExp=${chosenExp != tConv} " +
            "(converged exp=${tConv}ns iso=$isoConv)"
        )
    }

    /** Fixed-focus/legacy fallback (нет afOffSupported): пин экспозиции + AF AUTO-hold, сразу LOCKED. */
    private fun lockAll(result: TotalCaptureResult) {
        val session = captureSession ?: return
        if (lastFocusDistance == null) lastFocusDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
        applyExposurePinAndAwbLock(result)
        if (hasAf) previewBuilder.set(
            CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO
        )
        try {
            session.setRepeatingRequest(previewBuilder.build(), null, camHandler)
        } catch (_: Throwable) {}
        afState = Af3A.LOCKED
        val now = SystemClock.elapsedRealtime()
        lastDetectMs = now
        noDetectActive = false
        android.util.Log.d("PsiCodeRX", "LOCK (fallback, no manual focus) focus=$lastFocusDistance")
        runOnUiThread { if (!coreReady && PsiCodeCore.available) { /* keep rxInit banner */ } }
    }

    // ===== decoder-guided focus sweep (SPEC-worthy: не доверяем contrast-AF на time-varying коде) =====
    // Логика: пинуем экспозицию+AWB (кадры консистентны), прогоняем LENS_FOCUS_DISTANCE по таблице
    // диоптрий; на каждом шаге даём линзе устаканиться, затем берём max(score) по кадрам с detected==true
    // (score из JSON ядра — оно и есть «фокус-метрика», иммунная к мерцанию). Лучшая дистанция -> LOCKED.
    // Всё продвижение шагов — на cam-потоке (camHandler.postDelayed); агрегация score — на proc-потоке.

    private fun startSweep() {
        afState = Af3A.FOCUS_SWEEP
        sweepIndex = 0
        sweepBestScore = 0.0
        sweepBestD = -1.0f
        sweepSteps = SWEEP_DIOPTERS
        sweepFine = false
        curStepBestScore = 0.0
        sweepStepResults = 0
        sweepActive = true
        android.util.Log.d("PsiCodeRX", "FOCUS-SWEEP start (${sweepSteps.size} steps)")
        beginStep(0)
    }

    /** Установить шаг i: линза, окно наблюдения, failsafe-таймер (cam-поток). */
    private fun beginStep(i: Int) {
        sweepIndex = i
        applyFocusStep(sweepSteps[i])
        curStepBestScore = 0.0
        sweepStepResults = 0
        sweepObserveFrom = SystemClock.elapsedRealtime() + SWEEP_SETTLE_MS
        sweepGen++
        val gen = sweepGen
        // failsafe: если proc-поток не поставляет результатов (ядро занято/умерло),
        // шаг всё равно закроется по таймеру.
        camHandler?.postDelayed({ if (sweepActive && sweepGen == gen) advanceSweep() }, SWEEP_STEP_TIMEOUT_MS)
    }

    /** Продвижение свипа (cam-поток): закрыть текущий шаг, перейти к следующему либо финиш. */
    private fun advanceSweep() {
        if (!sweepActive) return
        val i = sweepIndex
        sweepObserveFrom = Long.MAX_VALUE          // мгновенно останавливаем запись proc-потоком
        val s = curStepBestScore                   // лучший score этого шага
        if (s > sweepBestScore) { sweepBestScore = s; sweepBestD = sweepSteps[i] }
        android.util.Log.d("PsiCodeRX", "  sweep[${if (sweepFine) "fine " else ""}$i] d=${sweepSteps[i]} score=$s")
        val next = i + 1
        if (next >= sweepSteps.size) { finishSweep(); return }
        beginStep(next)
    }

    /** Ранний выход свипа (cam-поток): текущая дистанция даёт РЕЗКОЕ кольцо. */
    private fun earlySweepLock() {
        if (!sweepActive) return
        val d = sweepSteps[sweepIndex]
        sweepActive = false
        sweepObserveFrom = Long.MAX_VALUE
        lastFocusDistance = d
        afState = Af3A.LOCKED
        lastDetectMs = SystemClock.elapsedRealtime()
        noDetectActive = false
        android.util.Log.d("PsiCodeRX", "FOCUS-SWEEP early lock: d=$d score=$curStepBestScore")
    }

    private fun finishSweep() {
        if (!sweepFine && sweepBestD >= 0f) {
            // грубая фаза нашла окрестность — тонкий проход вокруг победителя:
            // на 11 см DOF миллиметровая, шаг 1-2 диоптрии слишком груб для payload.
            val base = sweepBestD
            sweepFine = true
            sweepSteps = SWEEP_FINE_OFFSETS.map { (base + it).coerceIn(0f, 10f) }.toFloatArray()
            android.util.Log.d(
                "PsiCodeRX",
                "FOCUS-SWEEP fine phase around d=$base (${sweepSteps.joinToString()})"
            )
            beginStep(0)
            return
        }
        sweepActive = false
        sweepObserveFrom = Long.MAX_VALUE
        if (sweepBestD >= 0f) {
            val d = sweepBestD
            lastFocusDistance = d
            applyFocusStep(d)                       // выставляем лучшую дистанцию (репитинг без колбэка)
            afState = Af3A.LOCKED
            val now = SystemClock.elapsedRealtime()
            lastDetectMs = now
            noDetectActive = false
            android.util.Log.d("PsiCodeRX", "FOCUS-SWEEP done: best=$d score=$sweepBestScore -> LOCKED")
        } else {
            android.util.Log.d("PsiCodeRX", "FOCUS-SWEEP nothing detected; retry in ${SWEEP_RETRY_MS}ms")
            camHandler?.postDelayed({ startSweep() }, SWEEP_RETRY_MS)
        }
    }

    /** Выставить дистанцию фокуса (диоптрии) и перезапустить репитинг. Экспозиция/AWB уже в builder. */
    private fun applyFocusStep(diopter: Float) {
        val session = captureSession ?: return
        previewBuilder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
        previewBuilder.set(CaptureRequest.LENS_FOCUS_DISTANCE, diopter)
        try { session.setRepeatingRequest(previewBuilder.build(), null, camHandler) } catch (_: Throwable) {}
    }

    /** item 2 (обновлено): восстановление = ПЕРЕЗАПУСК свипа (не contrast-AF). Экспозиция-пин держится. */
    private fun startRecoverySweep() {
        recoveryCount++
        android.util.Log.d("PsiCodeRX", "RECOVERY #$recoveryCount: re-run focus sweep")
        startSweep()
    }

    // --- Frame feed ---

    private val onImage = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        try {
            val now = SystemClock.elapsedRealtime()
            // Троттл ~8 fps + пропуск кадров, пока ядро не готово.
            if (!coreReady || now - lastProcMs < PROC_INTERVAL_MS) return@OnImageAvailableListener
            lastProcMs = now

            val planes = image.planes
            val yb: ByteBuffer = planes[0].buffer
            val ub: ByteBuffer = planes[1].buffer
            val vb: ByteBuffer = planes[2].buffer
            val yn = yb.remaining(); val un = ub.remaining(); val vn = vb.remaining()
            if (yBytes.size != yn) yBytes = ByteArray(yn)
            if (uBytes.size != un) uBytes = ByteArray(un)
            if (vBytes.size != vn) vBytes = ByteArray(vn)
            yb.get(yBytes, 0, yn); ub.get(uBytes, 0, un); vb.get(vBytes, 0, vn)

            procCount++
            val json = PsiCodeCore.rxProcessFrame(
                handle, yBytes, uBytes, vBytes,
                image.width, image.height,
                planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride
            )
            // отладка: дамп первых двух кадров С ХОРОШО ОТТРЕКАННОЙ геометрией
            // (score >= 0.99 => фокус и лок уже устоялись), пуллятся через run-as
            if (dumpCount < 2 && json.contains("\"score\":0.9")) {
                try {
                    val i = dumpCount
                    java.io.File(filesDir, "dump$i.meta").writeText(
                        "${image.width} ${image.height} ${planes[0].rowStride} " +
                        "${planes[1].rowStride} ${planes[1].pixelStride} " +
                        "${planes[2].rowStride} ${planes[2].pixelStride} $yn $un $vn")
                    java.io.File(filesDir, "dump$i.y").writeBytes(yBytes.copyOf(yn))
                    java.io.File(filesDir, "dump$i.u").writeBytes(uBytes.copyOf(un))
                    java.io.File(filesDir, "dump$i.v").writeBytes(vBytes.copyOf(vn))
                } catch (_: Throwable) {}
                dumpCount++
            }
            android.util.Log.d(
                "PsiCodeRX",
                "${image.width}x${image.height} yS=${planes[0].rowStride} " +
                "uvS=${planes[1].rowStride}/${planes[1].pixelStride} " +
                "${SystemClock.elapsedRealtime() - now}ms $json"
            )
            handleStatus(json)
        } catch (t: Throwable) {
            runOnUiThread { showBanner("frame: ${t.message}", Color.parseColor("#B00020")) }
        } finally {
            image.close()
        }
    }

    private fun handleStatus(json: String) {
        val o = try { JSONObject(json) } catch (_: Throwable) { return }
        val detected = o.optBoolean("detected")
        val score = o.optDouble("score", 0.0)
        val rotation = o.optInt("rotation")
        val pxCell = o.optDouble("px_per_cell", 0.0)
        val stripesOk = o.optInt("stripes_ok")
        val symbolsNew = o.optInt("symbols_new")
        val k = o.optInt("k")
        val symbolsHave = o.optInt("symbols_have")
        val done = o.optBoolean("done")
        val crcOk = o.optBoolean("crc_ok")

        // --- focus sweep: шаг закрывается по ЧИСЛУ ОБРАБОТАННЫХ КАДРОВ, не по
        // таймеру — неудачный acquire занимает ~2.3 с, и настенные окна шагов
        // закрывались раньше первого результата. Кадр относится к шагу, если
        // его ОБРАБОТКА НАЧАЛАСЬ после установки линзы (lastProcMs — момент
        // старта обработки, тот же proc-поток). detected со score выше порога —
        // ранний выход: фокус найден, свип дальше не нужен. ---
        if (sweepActive && lastProcMs >= sweepObserveFrom) {
            if (detected && score > curStepBestScore) curStepBestScore = score
            if (detected && score > SWEEP_EARLY_SCORE) {
                camHandler?.post { earlySweepLock() }
            } else {
                sweepStepResults++
                if (sweepStepResults >= SWEEP_RESULTS_PER_STEP) {
                    sweepStepResults = 0
                    camHandler?.post { advanceSweep() }
                }
            }
        }

        // --- item 2: мониторинг здоровья захвата после лока -> recovery = re-sweep ---
        if (afState == Af3A.LOCKED) {
            val now = SystemClock.elapsedRealtime()
            if (detected) {
                if (noDetectActive) {
                    noDetectActive = false
                    android.util.Log.d("PsiCodeRX", "detect RECOVERED — streak reset")
                }
                lastDetectMs = now
            } else {
                noDetectActive = true
                if (now - lastDetectMs >= NODETECT_RECOVER_MS &&
                    now - lastRecoveryMs >= RECOVERY_COOLDOWN_MS) {
                    lastRecoveryMs = now
                    lastDetectMs = now   // рестарт отсчёта до следующего кулдауна
                    camHandler?.post { startRecoverySweep() }
                }
            }
        }

        val line1 = "det=%s score=%.3f rot=%d px/cell=%.1f"
            .format(if (detected) "Y" else "-", score, rotation, pxCell)
        val line2 = "K=%d have=%d (+%d) stripes=%d".format(k, symbolsHave, symbolsNew, stripesOk)
        val pct = if (k > 0) (symbolsHave * 100 / k).coerceIn(0, 100) else 0

        if (done && crcOk && !saved) {
            saved = true
            saveResult() // на proc-потоке (фоновый) — файловый IO допустим
        }

        runOnUiThread {
            statusView.text = "$line1\n$line2"
            progressBar.progress = pct
            if (done && crcOk) {
                statusView.setBackgroundColor(Color.parseColor("#1B5E20"))
                showBanner("DONE — CRC OK", Color.parseColor("#1B5E20"))
            } else if (done) {
                statusView.setBackgroundColor(Color.parseColor("#8A6D00"))
            } else {
                statusView.setBackgroundColor(Color.TRANSPARENT)
            }
        }
    }

    private fun saveResult() {
        val bytes = try { PsiCodeCore.rxTakeResult(handle) } catch (_: Throwable) { null } ?: return
        val name = "psicode_received_${System.currentTimeMillis() / 1000}.bin"
        val path = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveViaMediaStore(name, bytes)
            } else {
                @Suppress("DEPRECATION")
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val f = File(dir, name)
                FileOutputStream(f).use { it.write(bytes) }
                f.absolutePath
            }
        } catch (t: Throwable) {
            "save failed: ${t.message}"
        }
        runOnUiThread {
            Toast.makeText(this, "$path (${bytes.size} B)", Toast.LENGTH_LONG).show()
        }
    }

    private fun saveViaMediaStore(name: String, bytes: ByteArray): String {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return "insert failed"
        resolver.openOutputStream(uri)?.use { it.write(bytes) }
        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return "Downloads/$name"
    }

    // --- helpers ---

    private fun showBanner(text: String, color: Int) {
        bannerView.text = text
        bannerView.setBackgroundColor(color)
        bannerView.visibility = TextView.VISIBLE
    }

    private fun startBackgroundThreads() {
        if (camThread == null) {
            camThread = HandlerThread("psicode-cam").also { it.start() }
            camHandler = Handler(camThread!!.looper)
        }
        if (procThread == null) {
            procThread = HandlerThread("psicode-proc").also { it.start() }
            procHandler = Handler(procThread!!.looper)
        }
    }

    private fun stopBackgroundThreads() {
        camThread?.quitSafely(); camThread = null; camHandler = null
        procThread?.quitSafely(); procThread = null; procHandler = null
    }

    private fun closeCamera() {
        try { captureSession?.close() } catch (_: Throwable) {}
        captureSession = null
        try { cameraDevice?.close() } catch (_: Throwable) {}
        cameraDevice = null
        try { imageReader?.close() } catch (_: Throwable) {}
        imageReader = null
    }

    companion object {
        // cell_size_px из эталонного профиля §7.4 (16 px). Rust rxInit подстраивается сам.
        private const val PROFILE_CELL_PX = 16
        private const val PROC_INTERVAL_MS = 120L    // ~8 fps
        private const val CONVERGE_TIMEOUT_FRAMES = 45
        private const val EXPOSURE_FLOOR_NS = 16_700_000L   // пин: 1 кадр 60Гц (против бандинга)
        private const val EXPOSURE_PIN_MAX_NS = 20_000_000L // длиннее -> бленд смены кадров tx
        private const val NODETECT_RECOVER_MS = 6_000L     // серия без детекта до re-sweep
        private const val RECOVERY_COOLDOWN_MS = 10_000L   // не чаще раза в 10с
        // decoder-guided focus sweep: диоптрии от бесконечности (0) до ~11см (9); log-разрежённо.
        // Вероятные дистанции первыми (экран ~25–45 см): ранний выход обычно
        // срабатывает на первых шагах и свип не гоняет весь стол.
        private val SWEEP_DIOPTERS = floatArrayOf(3.2f, 2.4f, 4.2f, 1.7f, 5.5f, 1.0f, 7.0f, 0.0f, 9.0f)
        private const val SWEEP_SETTLE_MS = 400L          // устаканивание линзы
        private const val SWEEP_RESULTS_PER_STEP = 2      // кадров-результатов на шаг (не таймер!)
        private const val SWEEP_EARLY_SCORE = 0.90        // ранний лок ТОЛЬКО на резком кольце:
                                                          // score 0.6-0.7 детектирует, но payload
                                                          // при таком фокусе уже нечитаем (DOF на
                                                          // 11 см — миллиметры)
        private val SWEEP_FINE_OFFSETS = floatArrayOf(-0.6f, -0.3f, 0.3f, 0.6f) // тонкий проход
        private const val SWEEP_STEP_TIMEOUT_MS = 15_000L // failsafe, если результаты не идут
        private const val SWEEP_RETRY_MS = 5_000L   // пауза перед повтором, если никто не задетектил
    }
}
