package com.murdoc820.hyperplay

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.content.edit
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.murdoc820.hyperplay.audio.ConsoleUiSoundEngine
import com.murdoc820.hyperplay.engine.GameLauncher
import com.murdoc820.hyperplay.engine.HyperPlayNative
import com.murdoc820.hyperplay.engine.PortableExeScanner
import com.murdoc820.hyperplay.input.AndroidRumbleOutput
import com.murdoc820.hyperplay.input.ControllerManager
import com.murdoc820.hyperplay.input.EvdevXInputSink
import com.murdoc820.hyperplay.input.InputMode
import com.murdoc820.hyperplay.input.NoInputSink
import com.murdoc820.hyperplay.input.VirtualTouchOverlayView
import com.murdoc820.hyperplay.diag.Diagnostics
import com.murdoc820.hyperplay.input.XInputPadBridge
import com.murdoc820.hyperplay.input.XInputPadNative
import com.murdoc820.hyperplay.model.ExeArch
import com.murdoc820.hyperplay.model.PortableGame
import com.murdoc820.hyperplay.session.FpsCapOption
import com.murdoc820.hyperplay.session.FrameGenLogic
import com.murdoc820.hyperplay.session.GameSessionLogic
import com.murdoc820.hyperplay.session.PerfMonitor
import com.murdoc820.hyperplay.session.DisplayPolicy
import com.murdoc820.hyperplay.session.ThermalDecision
import com.murdoc820.hyperplay.session.ThermalGovernor
import com.murdoc820.hyperplay.session.UpscalerMode
import com.murdoc820.hyperplay.session.readBatteryTempC
import com.murdoc820.hyperplay.session.PerfOverlayMode
import com.murdoc820.hyperplay.session.QuickMenuItem
import com.murdoc820.hyperplay.session.GameProcess
import com.murdoc820.hyperplay.session.QuickResume
import com.murdoc820.hyperplay.session.QuickResumeAction
import com.murdoc820.hyperplay.session.QuickResumeLogic
import com.murdoc820.hyperplay.session.SessionOverlay
import com.murdoc820.hyperplay.session.SessionPhase
import com.murdoc820.hyperplay.session.SessionUiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Sesión de juego a pantalla completa: SurfaceView (DirectVulkanBridge), mando táctil
 * auto-ocultable, overlay de carga y Quick Menu (Atrás o L3+R3 / Select+Start).
 *
 * Lanza el `.exe` con [GameLauncher] (motor directo ZVDS, dentro del proceso de la app) y termina
 * cuando el punto de entrada PE vuelve. Al destruirse para el traductor y libera audio y superficie.
 */
class GameSessionActivity : ComponentActivity() {

    private val state = SessionUiState()
    private lateinit var prefs: SharedPreferences
    private lateinit var sounds: ConsoleUiSoundEngine
    private lateinit var controller: ControllerManager
    private lateinit var touchOverlay: VirtualTouchOverlayView

    // ---- Puente XInput (mandos → DirectXInputBridge, con rumble) -------------------------------
    // Mandos físicos: los lee padBridge (slot por mando); el ControllerManager solo detecta combos
    // y el auto-ocultado del mando táctil (sin salida: NoInputSink).
    // Mando táctil: su propio ControllerManager → EvdevXInputSink (slot 0 del puente).
    private lateinit var padBridge: XInputPadBridge
    private lateinit var touchController: ControllerManager
    private val padTouchSink by lazy { EvdevXInputSink { padBridge.setTouchState(it) } }
    private val touchControllerListener = object : ControllerManager.Listener {
        override fun onQuickMenuRequested() = toggleMenu()
        override fun onInputModeChanged(mode: InputMode) = Unit
        override fun onControllersChanged(names: List<String>) = Unit
    }

    private var exePath: String = ""
    private var launcher: GameLauncher? = null
    private var logFile: File? = null
    private var startedAt = 0L
    private var pollJob: Job? = null
    private var displayHz = 60f
    private var gameSurface: android.view.Surface? = null
    private val thermalGovernor = ThermalGovernor()
    /** Último tope de FPS enviado al motor. */
    private var appliedGameFps = Int.MIN_VALUE
    /** Refresco de panel pedido ahora mismo (cambia con el tope: 72 FPS → 144 Hz, resto → 120 Hz). */
    private var requestedPanelHz = 0

    // Navegación del menú con hat / stick (flancos).
    private var lastMenuAxisX = 0
    private var lastMenuDpadKeyAt = 0L
    private var lastMenuAxisY = 0

    private val controllerListener = object : ControllerManager.Listener {
        override fun onQuickMenuRequested() = toggleMenu()
        override fun onInputModeChanged(mode: InputMode) = updateOverlayVisibility()
        override fun onControllersChanged(names: List<String>) {
            Log.i(TAG, "mandos: $names")
        }
    }

    private val surfaceCallback = object : SurfaceHolder.Callback {
        override fun surfaceCreated(holder: SurfaceHolder) {
            gameSurface = holder.surface
            applySurfaceFrameRate(holder.surface)
            val ok = HyperPlayNative.nativeDirectAttachSurface(holder.surface)
            HyperPlayNative.nativeSetPaused(false)  // con superficie el juego vuelve a correr
            Log.i(TAG, "surfaceCreated: attach=$ok")
        }

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            // El swapchain se recrea solo (VK_ERROR_OUT_OF_DATE_KHR / SUBOPTIMAL).
            Log.i(TAG, "surfaceChanged ${width}x$height")
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            gameSurface = null
            // Sin superficie (Inicio, otra app, pantalla apagada): el juego se congela en su
            // siguiente Present en vez de seguir quemando CPU/GPU y batería sin verse.
            HyperPlayNative.nativeSetPaused(true)
            HyperPlayNative.nativeDirectDetachSurface()
            Log.i(TAG, "surfaceDestroyed")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Solo builds depurables: pruebas por ADB con el teléfono bloqueado (PIN) sin desbloquearlo.
        // No da acceso a nada más: al cerrar la sesión vuelve la pantalla de bloqueo.
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        window.attributes = window.attributes.apply {
            layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        WindowCompat.setDecorFitsSystemWindows(window, false)
        hideSystemBars()

        prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        // Modo eficiente: tope de FPS (60 por defecto), protección térmica y escala de render.
        state.fpsCap = FpsCapOption.fromKey(prefs.getString(FpsCapOption.PREFS_KEY, null))
        state.thermalProtection = prefs.getBoolean(ThermalGovernor.PREFS_KEY, ThermalGovernor.DEFAULT_ENABLED)
        state.upscaler = UpscalerMode.fromKey(prefs.getString(UpscalerMode.PREFS_KEY, null))
        HyperPlayNative.nativeSetUpscaler(state.upscaler.nativeId)
        HyperPlayNative.nativeSetSgsrSharpness(SGSR_SHARPNESS)
        state.frameGen = prefs.getBoolean(FrameGenLogic.PREFS_KEY, FrameGenLogic.DEFAULT_ENABLED)
        HyperPlayNative.nativeSetFrameGen(state.frameGen)
        state.thermal = thermalGovernor.update(null, state.thermalProtection, state.fpsCap)
        appliedGameFps = FrameGenLogic.gameFps(state.thermal, state.frameGen)
        HyperPlayNative.nativeSetFpsCap(appliedGameFps)

        applyDisplayModeForTarget(FrameGenLogic.panelHz(state.thermal, state.frameGen))
        displayHz = display?.refreshRate ?: 60f
        // Resolución nativa de este teléfono, siempre: la resolución y el aspecto los elige el juego
        // en sus propias opciones (HyperPlay ya no tiene escala en % en el Quick Menu).
        HyperPlayNative.nativeSetRenderScale(DisplayPolicy.RENDER_SCALE_PERCENT)
        windowManager.maximumWindowMetrics.bounds.let { b ->
            HyperPlayNative.nativeSetDisplayMode(b.width(), b.height(), Math.round(displayHz))
        }
        sounds = ConsoleUiSoundEngine(this)
        // Por defecto vibra el teléfono si el mando no tiene motores (mandos tipo GameSir enganchados
        // al móvil, la mayoría de mandos Bluetooth HID): se puede apagar en el Quick Menu.
        state.phoneRumble = prefs.getBoolean(PREFS_PHONE_RUMBLE, true)
        padBridge = XInputPadBridge(this, XInputPadNative, AndroidRumbleOutput(this) { state.phoneRumble })
        padBridge.phoneRumbleFallback = state.phoneRumble
        controller = ControllerManager(this, NoInputSink, controllerListener)
        touchController = ControllerManager(null, padTouchSink, touchControllerListener)

        exePath = intent.getStringExtra(EXTRA_EXE_PATH).orEmpty()
        state.title = File(exePath).parentFile?.name ?: exePath
        // Sin deformar nunca: panorámica real si el juego se puede parchear, si no su aspecto.
        HyperPlayNative.nativeSetScreenFit(DisplayPolicy.SCREEN_FIT_NATIVE_ID)
        state.art = runCatching { com.murdoc820.hyperplay.art.GameArtRepository.cached(this, exePath) }.getOrNull()
        state.perfMode = PerfOverlayMode.fromKey(prefs.getString(PerfOverlayMode.PREFS_KEY, null))
        startPerfMonitor()
        startThermalGovernor()

        val surface = SurfaceView(this).apply {
            setZOrderOnTop(false)
            holder.addCallback(surfaceCallback)
        }
        touchOverlay = VirtualTouchOverlayView(this, touchController)
        val compose = ComposeView(this).apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed)
            setContent {
                SessionOverlay(
                    state = state,
                    onMenuItem = ::onMenuItem,
                    onConfirmKill = ::onConfirmKill,
                    onCloseMenu = { closeMenu() },
                    onExit = { finish() },
                    onMenuStep = { item, step -> state.menuIndex = item.ordinal; changeFocusedValue(step) },
                )
            }
        }
        val root = FrameLayout(this).apply {
            setBackgroundColor(android.graphics.Color.BLACK)
            // Sin vistas enfocables: si no, tras tocar la pantalla ViewRootImpl se come la primera
            // tecla de cruceta para salir del modo táctil y el Quick Menu "pierde" pulsaciones.
            // Todo el teclado/mando se gestiona en dispatchKeyEvent.
            descendantFocusability = ViewGroup.FOCUS_BLOCK_DESCENDANTS
            val match = ViewGroup.LayoutParams.MATCH_PARENT
            addView(surface, FrameLayout.LayoutParams(match, match))
            addView(touchOverlay, FrameLayout.LayoutParams(match, match))
            addView(compose, FrameLayout.LayoutParams(match, match))
        }
        setContentView(root)
        updateOverlayVisibility()

        startSession()
    }

    override fun onResume() {
        super.onResume()
        hideSystemBars()
        controller.start()
        padBridge.resumeOutput()
    }

    override fun onPause() {
        controller.stop()  // también suelta botones y centra ejes
        touchOverlay.releaseAll()
        touchController.releaseAll()
        padBridge.suspendOutput()  // mandos neutros y rumble parado
        super.onPause()
    }

    // ---- Medición del arranque ------------------------------------------------------------------
    // "Juego dibujando" = primera muestra con >= 20 frames presentados/s. Solo para el log.
    private var startupLogged = false

    private fun logStartupMilestone(fps: Float) {
        if (startupLogged || fps < STARTUP_FPS) return
        startupLogged = true
        Log.i(TAG, "arranque: juego dibujando a ${fps.toInt()} FPS tras ${SystemClock.elapsedRealtime() - startedAt} ms")
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemBars()
    }

    override fun onDestroy() {
        pollJob?.cancel()
        val running = launcher?.isRunning() == true && state.phase != SessionPhase.ENDED
        val held = launcher
        if (running && !state.killing && held != null) {
            // Quick Resume: el juego queda congelado en memoria (el servicio mantiene vivo el
            // proceso). Volver a abrir el mismo .exe lo reanuda al instante.
            HyperPlayNative.nativeSetPaused(true)
            QuickResume.hold(QuickResume.Held(exePath, held, logFile, startedAt))
            GameProcess.publishHeld(this, exePath)  // el launcher (otro proceso) lo ve
            Log.i(TAG, "Quick Resume: juego suspendido en memoria")
            // En pausa no cuenta como sesión abierta: si el sistema mata el proceso en segundo
            // plano no es un fallo. Al reanudar se vuelve a abrir.
            endDiagSession(0, null)
        } else {
            launcher?.forceKill()  // para el traductor y el audio directo
            QuickResumeService.stop(this)
            endDiagSession(0, null)
        }
        padBridge.close()  // para el rumble y deja el mando a cero
        HyperPlayNative.nativeDirectDetachSurface()
        sounds.release()
        Log.i(TAG, "sesión cerrada")
        super.onDestroy()
        if (!(running && !state.killing && held != null)) {
            // Juego cerrado (o forzado): sus hilos no se pueden parar uno a uno; el proceso
            // ":game" termina para que el siguiente juego empiece limpio (GameProcess).
            val app = applicationContext
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({ GameProcess.exitGameProcess(app) }, 300)
        }
    }

    // ------------------------------------------------------------------ sesión

    /** Sesión abierta en el registro de errores (para no cerrarla dos veces). */
    private var diagSessionOpen = false

    private fun endDiagSession(exitCode: Int, error: String?) {
        if (!diagSessionOpen) {
            if (error != null) Diagnostics.sessionEnded(exePath, logFile, exitCode, error)
            return
        }
        diagSessionOpen = false
        Diagnostics.sessionEnded(exePath, logFile, exitCode, error)
    }

    private fun startSession() {
        startedAt = SystemClock.elapsedRealtime()
        val action = QuickResumeLogic.action(QuickResume.heldExe, QuickResume.isHolding(), exePath)
        if (action == QuickResumeAction.RESUME) {
            QuickResume.take(exePath)?.let { GameProcess.publishHeld(this, null); resumeHeld(it); return }
        }
        QuickResumeService.start(this, state.title, exePath)
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    if (action == QuickResumeAction.KILL_AND_LAUNCH) {
                        QuickResume.killHeld()  // nativeDirectStop: para el audio y luego despausa
                        Log.i(TAG, "Quick Resume: cerrado el juego suspendido anterior")
                    }
                    launchGame()
                }
            }
            result.onFailure { e ->
                Log.e(TAG, "no se pudo lanzar $exePath", e)
                state.errorMessage = e.message ?: e.toString()
                state.logTail = GameSessionLogic.readLogTail(logFile)
                state.phase = SessionPhase.ENDED
                endDiagSession(-1, state.errorMessage)
                QuickResumeService.stop(this@GameSessionActivity)
                sounds.error()
            }
            result.onSuccess { pollJob = launch { pollSession() } }
        }
    }

    /** Quick Resume: retoma el juego congelado en memoria (se reanuda al crear la superficie). */
    private fun resumeHeld(held: QuickResume.Held) {
        launcher = held.launcher
        logFile = held.logFile
        Diagnostics.sessionStarted(exePath, logFile)
        diagSessionOpen = true
        QuickResumeService.start(this, state.title, exePath)
        if (padBridge.open()) padBridge.start()
        state.phase = SessionPhase.RUNNING
        updateOverlayVisibility()
        Log.i(TAG, "Quick Resume: reanudando juego en memoria tras ${SystemClock.elapsedRealtime() - startedAt} ms")
        pollJob = lifecycleScope.launch { pollSession() }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val requested = intent.getStringExtra(EXTRA_EXE_PATH).orEmpty()
        if (requested.isEmpty() || QuickResumeLogic.sameExe(requested, exePath)) return
        // Otro juego con esta sesión abierta (singleTask): se cierra este (su proceso termina) y el
        // launcher abre el nuevo en un proceso limpio.
        Log.i(TAG, "otro juego pedido con la sesión abierta: cerrando $exePath")
        state.killing = true
        lifecycleScope.launch {
            withContext(Dispatchers.IO) { launcher?.forceKill() }
            launcher = null
            startActivity(Intent(this@GameSessionActivity, MainActivity::class.java)
                .putExtra(MainActivity.EXTRA_LAUNCH_EXE, requested)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP))
            finish()
        }
    }

    /** Bloqueante (E/S y JNI): en Dispatchers.IO. */
    private fun launchGame() {
        require(exePath.isNotEmpty()) { "Falta el extra $EXTRA_EXE_PATH con la ruta del .exe" }
        // Ruta canónica: /sdcard es un enlace simbólico.
        val exe = File(exePath).let { runCatching { it.canonicalFile }.getOrDefault(it) }
        require(exe.isFile) { "No se encuentra el ejecutable: $exePath" }
        require(exe.canRead()) { "Sin permiso de lectura: $exePath (concede 'Acceso a todos los archivos')" }
        val folder = exe.parentFile ?: error("El ejecutable no tiene carpeta: $exePath")
        val pe = HyperPlayNative.inspectPe(exe) ?: error("No es un ejecutable PE válido: ${exe.name}")
        require(pe.arch == ExeArch.X86 || pe.arch == ExeArch.X64) { "Arquitectura no soportada: ${pe.arch.badge}" }
        val game = PortableGame(
            title = state.title,
            exePath = exe.absolutePath,
            mountFolder = folder.absolutePath,
            pe = pe,
            engineHints = PortableExeScanner.detectEngineHints(folder, exe),
        )
        Log.i(TAG, "juego: ${game.title} ${pe.arch.badge} LAA=${pe.largeAddressAware} hints=${game.engineHints}")

        // Estado XInput a cero antes de lanzar el juego.
        if (padBridge.open()) {
            runOnUiThread { if (!isDestroyed) padBridge.start() }
        }

        val hz = GameSessionLogic.refreshHzFor(displayHz)
        val gameLauncher = GameLauncher(this)
        launcher = gameLauncher
        val log = gameLauncher.launch(game, hz).getOrThrow()
        logFile = log
        Diagnostics.sessionStarted(exePath, log)
        diagSessionOpen = true

        Log.i(TAG, "lanzado @$hz Hz log=${log.absolutePath}")
    }

    /** Bucle de la UI (2 veces por segundo): carga, FPS y fin de sesión. */
    private suspend fun pollSession() {
        state.logPath = logFile?.absolutePath.orEmpty()
        var lastFrames = HyperPlayNative.nativeDirectFramesPresented()
        var lastAt = SystemClock.elapsedRealtime()
        while (currentCoroutineContext().isActive) {
            val now = SystemClock.elapsedRealtime()
            val frames = HyperPlayNative.nativeDirectFramesPresented()
            state.elapsedMillis = now - startedAt
            state.presentFps = HyperPlayNative.nativeDirectPresentFps()
            state.gameFps = HyperPlayNative.nativeGameFps()
            state.receivedFps = GameSessionLogic.ratePerSecond(lastFrames, frames, now - lastAt)
            logStartupMilestone(state.receivedFps)
            lastFrames = frames
            lastAt = now

            if (state.phase == SessionPhase.LOADING) {
                val text = withContext(Dispatchers.IO) { GameSessionLogic.readLogText(logFile) }
                state.logTail = GameSessionLogic.tailLines(text)
                state.stage = GameSessionLogic.loadingStage(text, state.stage)
                // Fuera la pantalla de carga cuando el juego dibuja de verdad (no al crear el
                // dispositivo: Unity presenta antes muchos segundos de frames negros).
                val draws = runCatching { HyperPlayNative.nativeDirectDrawCount() }.getOrDefault(0L)
                if (GameSessionLogic.contentStarted(
                        HyperPlayNative.nativeDirectGetTelemetry(), draws, state.elapsedMillis)) {
                    state.phase = SessionPhase.RUNNING
                    Log.i(TAG, "gráficos activos tras ${GameSessionLogic.formatElapsed(state.elapsedMillis)}")
                    updateOverlayVisibility()
                }
            }
            if (launcher?.isRunning() != true) {
                state.exitCode = launcher?.exitCode() ?: -1
                state.logTail = withContext(Dispatchers.IO) { GameSessionLogic.readLogTail(logFile) }
                state.phase = SessionPhase.ENDED
                state.menuOpen = false
                padBridge.suspendOutput()  // el juego salió: sin rumble colgado
                QuickResumeService.stop(this@GameSessionActivity)
                updateOverlayVisibility()
                Log.i(TAG, "sesión terminada: código ${state.exitCode}")
                // Forzar cierre (killing) no es un fallo del juego.
                endDiagSession(if (state.killing) 0 else state.exitCode, state.errorMessage)
                return
            }
            delay(POLL_MS)
        }
    }

    // ------------------------------------------------------------------ Quick Menu

    private fun toggleMenu() = if (state.menuOpen) closeMenu() else openMenu()

    private fun openMenu() {
        if (state.phase == SessionPhase.ENDED || state.menuOpen) return
        controller.releaseAll()
        touchOverlay.releaseAll()
        touchController.releaseAll()
        padBridge.releaseAll()  // el juego no debe ver el L3+R3 / Select+Start del combo pulsado
        state.menuIndex = 0
        state.confirmKill = false
        state.notice = null
        state.menuOpen = true
        lastMenuAxisX = 0
        lastMenuAxisY = 0
        updateOverlayVisibility()
        sounds.confirm()
    }

    private fun closeMenu() {
        if (!state.menuOpen || state.killing) return
        state.menuOpen = false
        state.confirmKill = false
        updateOverlayVisibility()
        sounds.back()
    }

    private fun onMenuItem(item: QuickMenuItem) {
        state.menuIndex = item.ordinal
        when (item) {
            QuickMenuItem.RESUME -> closeMenu()
            QuickMenuItem.FPS_CAP -> changeFpsCap(1)
            QuickMenuItem.FRAME_GEN -> toggleFrameGen()
            QuickMenuItem.THERMAL -> toggleThermalProtection()
            QuickMenuItem.UPSCALER -> changeUpscaler(1)
            QuickMenuItem.FSR_UPSCALE -> toggleFsrUpscale()
            QuickMenuItem.OVERLAY -> changePerfMode(1)
            QuickMenuItem.PHONE_RUMBLE -> togglePhoneRumble()
            QuickMenuItem.QUICK_EXIT -> {
                sounds.back()
                finish()  // onDestroy lo deja congelado en memoria (Quick Resume)
            }
            QuickMenuItem.FORCE_KILL -> {
                state.confirmKill = true
                state.confirmIndex = 0
                sounds.confirm()
            }
        }
    }

    private fun onConfirmKill(kill: Boolean) {
        if (state.killing) return
        if (!kill) {
            state.confirmKill = false
            sounds.back()
            return
        }
        state.killing = true
        sounds.confirm()
        lifecycleScope.launch {
            withContext(Dispatchers.IO) { launcher?.forceKill() }
            Log.i(TAG, "Force Kill desde el Quick Menu")
            finish()
        }
    }

    private fun toggleFsrUpscale() {
        state.fsrUpscale = !state.fsrUpscale
        HyperPlayNative.nativeDirectSetFsrEnabled(state.fsrUpscale)
        sounds.confirm()
    }

    private fun togglePhoneRumble() {
        val next = !state.phoneRumble
        state.phoneRumble = next
        padBridge.phoneRumbleFallback = next
        prefs.edit { putBoolean(PREFS_PHONE_RUMBLE, next) }
        sounds.navigate()
    }

    // ------------------------------------------------------------------ HUD de rendimiento

    private fun changePerfMode(step: Int) {
        val next = PerfOverlayMode.cycle(state.perfMode, step)
        state.perfMode = next
        prefs.edit { putString(PerfOverlayMode.PREFS_KEY, next.key) }
        sounds.navigate()
    }

    // ---- Ritmo de pantalla (frame pacing) ------------------------------------------------------
    // El panel debe ir a un múltiplo entero de TARGET_FPS (60 FPS → 60 Hz; nunca 144/90, que
    // repartirían los frames 2-3-2… = judder): se pide el modo de pantalla adecuado a la ventana y
    // además la frecuencia de la superficie.

    /**
     * Panel a un múltiplo entero del tope: 60/120 FPS → 120 Hz (60 = 2 vsyncs exactos), 72 → 144 Hz.
     * Cambiar el modo del panel no recrea la superficie ni el swapchain.
     */
    private fun applyDisplayModeForTarget(panelHz: Int) {
        if (panelHz == requestedPanelHz) return
        val current = display ?: return
        val size = current.mode
        val options = current.supportedModes
            .filter { it.physicalWidth == size.physicalWidth && it.physicalHeight == size.physicalHeight }
            .map { GameSessionLogic.DisplayModeOption(it.modeId, it.refreshRate) }
        val pick = GameSessionLogic.pickDisplayMode(options, panelHz) ?: return
        requestedPanelHz = panelHz
        window.attributes = window.attributes.apply { preferredDisplayModeId = pick.id }
        gameSurface?.let(::applySurfaceFrameRate)
        Log.i(TAG, "modo de pantalla pedido: ${pick.refreshHz} Hz (id ${pick.id}) para $panelHz Hz")
    }

    private fun applySurfaceFrameRate(surface: android.view.Surface) {
        runCatching {
            surface.setFrameRate(
                (if (requestedPanelHz > 0) requestedPanelHz else GameSessionLogic.TARGET_FPS).toFloat(),
                android.view.Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE,
                android.view.Surface.CHANGE_FRAME_RATE_ALWAYS,
            )
        }.onFailure { Log.w(TAG, "setFrameRate falló: ${it.message}") }
    }

    // ---- Modo eficiente y protección térmica -----------------------------------------------------

    /** Aplica al motor la decisión vigente (tope en caliente, anclaje Prime) y el refresco del panel. */
    private fun applyThermalDecision(d: ThermalDecision) {
        val prev = state.thermal
        state.thermal = d
        // Con HyperFG el juego va a la mitad del panel y el motor intercala un sintético por frame.
        val gameFps = FrameGenLogic.gameFps(d, state.frameGen)
        if (gameFps != appliedGameFps) {
            HyperPlayNative.nativeSetFpsCap(gameFps)
            appliedGameFps = gameFps
        }
        if (d.relaxPrime != prev.relaxPrime) HyperPlayNative.nativeSetThermalRelax(d.relaxPrime)
        applyDisplayModeForTarget(FrameGenLogic.panelHz(d, state.frameGen))
        if (d.throttled != prev.throttled) {
            Log.i(TAG, "protección térmica ${if (d.throttled) "ACTIVA" else "desactivada"}: tope ${d.effectiveFps}")
        }
    }

    /** Batería cada 2 s (aunque el HUD esté apagado): el gobernador decide el tope y el anclaje. */
    private fun startThermalGovernor() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                while (currentCoroutineContext().isActive) {
                    val c = withContext(Dispatchers.IO) { readBatteryTempC(applicationContext) }
                    applyThermalDecision(thermalGovernor.update(c, state.thermalProtection, state.fpsCap))
                    delay(THERMAL_PERIOD_MS)
                }
            }
        }
    }

    private fun changeFpsCap(step: Int) {
        if (!FrameGenLogic.fpsCapSelectable(state.frameGen)) {
            // Con HyperFG el ritmo lo decide la generación (60 -> 120): el tope queda bloqueado.
            state.notice = getString(R.string.notice_fps_locked)
            sounds.error()
            return
        }
        val next = FpsCapOption.cycle(state.fpsCap, step)
        state.fpsCap = next
        prefs.edit { putString(FpsCapOption.PREFS_KEY, next.key) }
        applyThermalDecision(thermalGovernor.update(null, state.thermalProtection, next))
        sounds.navigate()
    }

    /** HyperFG en caliente: sin recrear el swapchain (los recursos se crean o liberan en el motor). */
    private fun toggleFrameGen() {
        val next = !state.frameGen
        state.frameGen = next
        prefs.edit { putBoolean(FrameGenLogic.PREFS_KEY, next) }
        HyperPlayNative.nativeSetFrameGen(next)
        applyThermalDecision(state.thermal)  // tope del juego (60, o 30 con calor) y panel a 120 Hz
        sounds.navigate()
    }

    private fun toggleThermalProtection() {
        val next = !state.thermalProtection
        state.thermalProtection = next
        prefs.edit { putBoolean(ThermalGovernor.PREFS_KEY, next) }
        applyThermalDecision(thermalGovernor.update(state.perf.batteryTempC, next, state.fpsCap))
        sounds.navigate()
    }

    private fun changeUpscaler(step: Int) {
        val next = UpscalerMode.cycle(state.upscaler, step)
        state.upscaler = next
        prefs.edit { putString(UpscalerMode.PREFS_KEY, next.key) }
        HyperPlayNative.nativeSetUpscaler(next.nativeId)  // en caliente: el pase elige por frame
        sounds.navigate()
    }

    /** Muestreo (4 Hz FPS, 2 Hz /proc y /sys) solo con la sesión en marcha, el HUD visible y la app en primer plano. */
    private fun startPerfMonitor() {
        val monitor = PerfMonitor(applicationContext)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                monitor.run(
                    enabled = { state.phase == SessionPhase.RUNNING && state.perfMode != PerfOverlayMode.OFF },
                    publish = { state.perf = it },
                )
            }
        }
    }

    private fun activateFocused() {
        if (state.confirmKill) onConfirmKill(state.confirmIndex == 1)
        else onMenuItem(QuickMenuItem.entries[state.menuIndex])
    }

    private fun moveMenuFocus(delta: Int) {
        if (state.confirmKill) {
            state.confirmIndex = GameSessionLogic.moveFocus(state.confirmIndex, delta, 2)
        } else {
            state.menuIndex = GameSessionLogic.moveFocus(state.menuIndex, delta, QuickMenuItem.entries.size)
        }
        sounds.navigate()
    }

    private fun changeFocusedValue(step: Int) {
        if (state.confirmKill) return
        when (QuickMenuItem.entries[state.menuIndex]) {
            QuickMenuItem.FPS_CAP -> changeFpsCap(step)
            QuickMenuItem.FRAME_GEN -> toggleFrameGen()
            QuickMenuItem.THERMAL -> toggleThermalProtection()
            QuickMenuItem.UPSCALER -> changeUpscaler(step)
            QuickMenuItem.FSR_UPSCALE -> toggleFsrUpscale()
            QuickMenuItem.OVERLAY -> changePerfMode(step)
            QuickMenuItem.PHONE_RUMBLE -> togglePhoneRumble()
            else -> Unit
        }
    }

    private fun backInMenu() {
        if (state.confirmKill) {
            if (!state.killing) {
                state.confirmKill = false
                sounds.back()
            }
        } else {
            closeMenu()
        }
    }

    // ------------------------------------------------------------------ entrada

    // Falso positivo conocido de lint: androidx.core.app.ComponentActivity marca su override de
    // Activity.dispatchKeyEvent (API pública del framework) como @RestrictTo.
    @SuppressLint("RestrictedApi")
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val code = event.keyCode
        if (code in PASS_THROUGH_KEYS) return super.dispatchKeyEvent(event)

        if (state.phase == SessionPhase.ENDED) {
            if (event.action == KeyEvent.ACTION_UP && code in EXIT_KEYS) finish()
            return true
        }
        if (state.menuOpen) {
            handleMenuKey(event)
            return true
        }
        if (code == KeyEvent.KEYCODE_BACK) {
            // Se abre al soltar: así el ACTION_UP no llega al menú ya abierto.
            if (event.action == KeyEvent.ACTION_UP && !event.isCanceled) openMenu()
            return true
        }
        // El puente XInput toma el botón (por mando); el ControllerManager lo ve igual para los
        // combos del Quick Menu y el auto-ocultado, sin salida propia (NoInputSink).
        val toXInput = padBridge.onKeyEvent(event)
        if (controller.onKeyEvent(event) || toXInput) return true
        return super.dispatchKeyEvent(event)
    }

    private fun handleMenuKey(event: KeyEvent) {
        val code = event.keyCode
        if (event.action == KeyEvent.ACTION_DOWN) {
            if (code in DPAD_KEYS) lastMenuDpadKeyAt = android.os.SystemClock.uptimeMillis()
            when (code) {
                KeyEvent.KEYCODE_DPAD_UP -> moveMenuFocus(-1)
                KeyEvent.KEYCODE_DPAD_DOWN -> moveMenuFocus(1)
                KeyEvent.KEYCODE_DPAD_LEFT -> menuHorizontal(-1)
                KeyEvent.KEYCODE_DPAD_RIGHT -> menuHorizontal(1)
                // LB/RB: saltar entre la columna de acciones y la de ajustes.
                KeyEvent.KEYCODE_BUTTON_L1 -> if (event.repeatCount == 0) jumpMenuColumn(toSettings = false)
                KeyEvent.KEYCODE_BUTTON_R1 -> if (event.repeatCount == 0) jumpMenuColumn(toSettings = true)
            }
            return
        }
        if (event.action != KeyEvent.ACTION_UP || event.isCanceled) return
        // Activar / volver al soltar, para que el ACTION_UP no se cuele al juego tras cerrar.
        when (code) {
            KeyEvent.KEYCODE_BUTTON_A, KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_DPAD_CENTER,
            KeyEvent.KEYCODE_NUMPAD_ENTER -> activateFocused()
            KeyEvent.KEYCODE_BUTTON_B, KeyEvent.KEYCODE_BACK, KeyEvent.KEYCODE_ESCAPE,
            KeyEvent.KEYCODE_BUTTON_START -> backInMenu()
        }
    }

    override fun dispatchGenericMotionEvent(event: MotionEvent): Boolean {
        if (state.menuOpen) {
            if (event.isFromSource(InputDevice.SOURCE_JOYSTICK) && event.actionMasked == MotionEvent.ACTION_MOVE) {
                handleMenuAxes(event)
                return true
            }
            return super.dispatchGenericMotionEvent(event)
        }
        if (state.phase != SessionPhase.ENDED) {
            val toXInput = padBridge.onGenericMotionEvent(event)
            if (controller.onGenericMotionEvent(event) || toXInput) return true
        }
        return super.dispatchGenericMotionEvent(event)
    }

    /** Hat o stick izquierdo como cruceta del menú (solo en el flanco de entrada). */
    /**
     * Cruceta (como ejes HAT) y stick izquierdo en el Quick Menu. Una sola dirección a la vez:
     * - La cruceta de muchos mandos llega como teclas Y como ejes HAT: si acaba de llegar una tecla
     *   de cruceta, el HAT se ignora (antes cada pulsación contaba dos veces).
     * - Stick: solo el eje dominante y con histéresis (entra a 0,6, sale a 0,35). Antes, al empujar a
     *   la derecha para cambiar un valor, la desviación vertical bajaba el foco a la fila de abajo.
     */
    private fun handleMenuAxes(event: MotionEvent) {
        val now = android.os.SystemClock.uptimeMillis()
        val hatFresh = now - lastMenuDpadKeyAt > MENU_DPAD_DEDUP_MS
        val hx = if (hatFresh) event.getAxisValue(MotionEvent.AXIS_HAT_X) else 0f
        val hy = if (hatFresh) event.getAxisValue(MotionEvent.AXIS_HAT_Y) else 0f
        val sx = event.getAxisValue(MotionEvent.AXIS_X)
        val sy = event.getAxisValue(MotionEvent.AXIS_Y)
        var x = 0
        var y = 0
        when {
            kotlin.math.abs(hx) >= 0.5f || kotlin.math.abs(hy) >= 0.5f -> {
                if (kotlin.math.abs(hx) >= kotlin.math.abs(hy)) x = if (hx > 0) 1 else -1 else y = if (hy > 0) 1 else -1
            }
            else -> {
                val ax = kotlin.math.abs(sx)
                val ay = kotlin.math.abs(sy)
                val engaged = lastMenuAxisX != 0 || lastMenuAxisY != 0
                val threshold = if (engaged) MENU_STICK_RELEASE else MENU_STICK_THRESHOLD
                if (maxOf(ax, ay) >= threshold) {
                    // Mantener el eje ya activo mientras siga por encima del umbral de salida.
                    val useX = when {
                        lastMenuAxisX != 0 && ax >= MENU_STICK_RELEASE -> true
                        lastMenuAxisY != 0 && ay >= MENU_STICK_RELEASE -> false
                        else -> ax >= ay
                    }
                    if (useX) x = if (sx > 0) 1 else -1 else y = if (sy > 0) 1 else -1
                }
            }
        }
        if (y != 0 && y != lastMenuAxisY) moveMenuFocus(y)
        if (x != 0 && x != lastMenuAxisX) menuHorizontal(x)
        lastMenuAxisY = y
        lastMenuAxisX = x
    }

    /**
     * Izquierda/derecha: en un ajuste cambian su valor; en la columna de acciones, derecha salta a
     * los ajustes (no hace nada inesperado).
     */
    private fun menuHorizontal(step: Int) {
        if (state.confirmKill) {
            state.confirmIndex = GameSessionLogic.moveFocus(state.confirmIndex, step, 2)
            sounds.navigate()
            return
        }
        val item = QuickMenuItem.entries[state.menuIndex]
        if (item.isAction) {
            if (step > 0) jumpMenuColumn(toSettings = true)
            return
        }
        changeFocusedValue(step)
    }

    private fun jumpMenuColumn(toSettings: Boolean) {
        if (state.confirmKill) return
        val target = if (toSettings) QuickMenuItem.entries.first { !it.isAction } else QuickMenuItem.RESUME
        if (state.menuIndex == target.ordinal) return
        state.menuIndex = target.ordinal
        sounds.navigate()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        // Tocar la pantalla vuelve a mostrar el mando táctil (si estaba oculto por usar un mando).
        if (event.actionMasked == MotionEvent.ACTION_DOWN && state.phase == SessionPhase.RUNNING) controller.onTouch()
        return super.dispatchTouchEvent(event)
    }

    private fun updateOverlayVisibility() {
        if (!::touchOverlay.isInitialized) return
        val show = state.phase == SessionPhase.RUNNING && !state.menuOpen && controller.mode == InputMode.TOUCH
        touchOverlay.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun hideSystemBars() {
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
    }

    companion object {
        private const val TAG = "HyperPlaySession"

        /**
         * Ruta absoluta del `.exe` (String). Para pruebas por ADB (build debug):
         * `adb shell "am start -n com.murdoc820.hyperplay/.GameSessionActivity --es com.murdoc820.hyperplay.extra.EXE_PATH '/sdcard/PC games/Juego/juego.exe'"`
         */
        const val EXTRA_EXE_PATH = "com.murdoc820.hyperplay.extra.EXE_PATH"

        private const val PREFS = "game_session"
        /** Vibración en el teléfono si el mando no vibra (por defecto no: solo el mando). */
        private const val PREFS_PHONE_RUMBLE = "pad_phone_rumble"
        private const val STARTUP_FPS = 20f
        private const val POLL_MS = 500L
        /** Gobernador térmico: la batería cambia despacio; 2 s basta y apenas cuesta. */
        private const val THERMAL_PERIOD_MS = 2000L
        /** Nitidez de SGSR v1: 0,30 = intensidad original, equilibrada para arte 2D (sin halos). */
        private const val SGSR_SHARPNESS = 0.30f
        private const val MENU_STICK_THRESHOLD = 0.6f
        /** Histéresis del stick: el eje activo se suelta por debajo de este valor. */
        private const val MENU_STICK_RELEASE = 0.35f
        /** Tras una tecla de cruceta, los ejes HAT del mismo mando se ignoran este tiempo. */
        private const val MENU_DPAD_DEDUP_MS = 400L
        private val DPAD_KEYS = setOf(
            KeyEvent.KEYCODE_DPAD_UP, KeyEvent.KEYCODE_DPAD_DOWN,
            KeyEvent.KEYCODE_DPAD_LEFT, KeyEvent.KEYCODE_DPAD_RIGHT,
        )

        private val PASS_THROUGH_KEYS = setOf(
            KeyEvent.KEYCODE_VOLUME_UP, KeyEvent.KEYCODE_VOLUME_DOWN, KeyEvent.KEYCODE_VOLUME_MUTE,
            KeyEvent.KEYCODE_POWER,
        )
        private val EXIT_KEYS = setOf(
            KeyEvent.KEYCODE_BUTTON_A, KeyEvent.KEYCODE_BUTTON_B, KeyEvent.KEYCODE_ENTER,
            KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_BACK, KeyEvent.KEYCODE_ESCAPE,
        )

        fun start(context: Context, exePath: String) {
            val intent = Intent(context, GameSessionActivity::class.java).putExtra(EXTRA_EXE_PATH, exePath)
            if (context !is Activity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        }
    }
}
