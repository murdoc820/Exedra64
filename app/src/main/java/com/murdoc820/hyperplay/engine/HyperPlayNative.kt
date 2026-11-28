package com.murdoc820.hyperplay.engine

import android.view.Surface
import com.murdoc820.hyperplay.model.GraphicsApi
import com.murdoc820.hyperplay.model.PeInfo
import java.io.File

/** Puente JNI hacia `libhyperplay.so` (motor directo ZVDS, ver `cpp/jni_bridge.cpp`). */
object HyperPlayNative {
    // Índices del IntArray que devuelve nativeInspectPe (ver jni_bridge.cpp).
    private const val PE_STATUS = 0
    private const val PE_MACHINE = 1
    private const val PE_CHARACTERISTICS = 2
    private const val PE_SUBSYSTEM = 3
    private const val PE_IS_64 = 4
    private const val PE_LAA = 5
    private const val PE_APIS = 6
    private const val PE_FIELDS = 7
    private const val PE_STATUS_OK = 0

    init {
        System.loadLibrary("hyperplay")
    }

    @JvmStatic
    private external fun nativeInspectPe(path: String): IntArray?

    // ---- Scheduler Oryon ----
    @JvmStatic external fun nativePinGameThread(pid: Int): Boolean
    @JvmStatic external fun nativePinWorkerThread(pid: Int): Boolean
    @JvmStatic external fun nativeSetupAdpf(tid: Int, targetNs: Long): Boolean
    @JvmStatic external fun nativeCoreTopology(): String

    // ---- Modo eficiente / protección térmica ----
    /** Tope de FPS en caliente: >0 FPS, 0 = sin límite (`debug.hyperplay.fpscap` tiene prioridad). */
    @JvmStatic external fun nativeSetFpsCap(fps: Int)
    @JvmStatic external fun nativeGetEffectiveFpsCap(): Int
    /** true = relaja el anclaje del hilo principal a Prime (reparte el calor). */
    @JvmStatic external fun nativeSetThermalRelax(relax: Boolean)
    /** Escala de render (10-100 %); se aplica en el siguiente nativeSetDisplayMode (inicio de sesión). */
    @JvmStatic external fun nativeSetRenderScale(percent: Int)
    /** Reescalado al presentar: 0 auto (SGSR v1 si escala < 100 %), 1 SGSR siempre, 2 bilineal. */
    @JvmStatic external fun nativeSetUpscaler(mode: Int)
    /** Juego .NET: ejecuta [gameExe] con mono.exe de [runtimeDir] (ver cpp/loader/ManagedHost.h). */
    @JvmStatic external fun nativeDirectLoadManaged(gameExe: String, runtimeDir: String): Boolean
    /** Ajuste a la pantalla: 0 original, 1 estirar, 2 ampliar (ver gfx/PresentOptions.h). */
    @JvmStatic external fun nativeSetScreenFit(mode: Int)
    /** Nitidez de SGSR v1 (0..1; 0,30 = intensidad original). */
    @JvmStatic external fun nativeSetSgsrSharpness(sharpness: Float)
    /** HyperFG 2x (apagado por defecto); `debug.hyperplay.fg` tiene prioridad. */
    @JvmStatic external fun nativeSetFrameGen(on: Boolean)
    /** Quick Resume / segundo plano: congela el juego en su siguiente Present y el audio. */
    @JvmStatic external fun nativeSetPaused(paused: Boolean)
    @JvmStatic external fun nativeIsFrameGenActive(): Boolean
    /** [generados, omitidos por frame lento, frames reales presentados, cancelados]. */
    @JvmStatic external fun nativeGetFrameGenStats(): LongArray?

    // ---- Pure Bionic Direct-Silicon Engine (ZVDS) ----
    @JvmStatic external fun nativeDirectLoadPe(exePath: String): Boolean
    /** Informe detallado del último fallo de nativeDirectLoadPe (IAT, imports pendientes). */
    @JvmStatic external fun nativeDirectGetLastLoadReport(): String
    @JvmStatic external fun nativeDirectLaunch(): Int
    @JvmStatic external fun nativeDirectStop()
    @JvmStatic external fun nativeDirectPrecompileAot(exePath: String, savesDir: String): LongArray?
    @JvmStatic external fun nativeDirectSetFsrEnabled(enabled: Boolean)
    @JvmStatic external fun nativeDirectIsFsrEnabled(): Boolean
    @JvmStatic external fun nativeDirectGetTelemetry(): LongArray?
    /** Draws D3D11 acumulados (0 mientras el juego solo limpia la pantalla al cargar). */
    @JvmStatic external fun nativeDirectDrawCount(): Long
    @JvmStatic external fun nativeDirectGetImports(exePath: String): Array<String>?
    /** Estado XInput del mando 0 (lo lee `XInputGetState` vía DirectXInputBridge). */
    @JvmStatic external fun nativeDirectSetGamepadState(buttons: Int, lt: Int, rt: Int, lx: Int, ly: Int, rx: Int, ry: Int)
    /** Rumble pedido por el juego en el mando 0: `(izquierdo << 16) | derecho`. */
    @JvmStatic external fun nativeDirectGetRumble(): Int
    @JvmStatic external fun nativeDirectInitVulkan(customDriver: String): Boolean
    /** Resolución nativa (cualquier orientación) y Hz de la pantalla: el escritorio que verá el juego. */
    @JvmStatic external fun nativeSetDisplayMode(width: Int, height: Int, hz: Int)
    @JvmStatic external fun nativeDirectAttachSurface(surface: Surface): Boolean
    @JvmStatic external fun nativeDirectDetachSurface()
    @JvmStatic external fun nativeDirectPresentTestFrame(): Boolean
    @JvmStatic external fun nativeDirectPresentFps(): Float
    /** FPS reales del juego (Present), sin repeticiones ni sintéticos. */
    @JvmStatic external fun nativeGameFps(): Float
    /** Frames presentados por DirectVulkanBridge desde que arrancó el proceso. */
    @JvmStatic external fun nativeDirectFramesPresented(): Long
    @JvmStatic external fun nativeDirectStartAudio(rate: Int, channels: Int): Boolean
    @JvmStatic external fun nativeDirectStopAudio()

    /** Lee las cabeceras PE del `.exe`; null si no es un PE válido. */
    fun inspectPe(exe: File): PeInfo? {
        val v = nativeInspectPe(exe.absolutePath) ?: return null
        if (v.size < PE_FIELDS || v[PE_STATUS] != PE_STATUS_OK) return null
        return PeInfo(
            machine = v[PE_MACHINE],
            characteristics = v[PE_CHARACTERISTICS],
            subsystem = v[PE_SUBSYSTEM],
            is64Bit = v[PE_IS_64] != 0,
            largeAddressAware = v[PE_LAA] != 0,
            graphicsApis = GraphicsApi.fromMask(v[PE_APIS]),
        )
    }
}
