package com.murdoc820.hyperplay.session

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import androidx.core.content.ContextCompat
import com.murdoc820.hyperplay.engine.HyperPlayNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import androidx.annotation.StringRes
import com.murdoc820.hyperplay.R

/** Modo del HUD de rendimiento (Quick Menu → "Overlay"). Se guarda globalmente en SharedPreferences. */
enum class PerfOverlayMode(val key: String, @StringRes val label: Int) {
    OFF("off", R.string.state_off),
    COMPACT("compact", R.string.state_on),
    /** Ya no se ofrece (se veía mal en el centro); un "full" guardado pasa a [COMPACT]. */
    FULL("full", R.string.overlay_full);

    companion object {
        val DEFAULT = COMPACT
        const val PREFS_KEY = "perf_overlay_mode"
        /** Opciones del menú: solo activar o desactivar el HUD compacto. */
        val SELECTABLE = listOf(OFF, COMPACT)

        fun fromKey(key: String?): PerfOverlayMode =
            entries.firstOrNull { it.key == key }?.let { if (it == FULL) COMPACT else it } ?: DEFAULT

        fun cycle(current: PerfOverlayMode, step: Int = 1): PerfOverlayMode {
            val i = SELECTABLE.indexOf(current).coerceAtLeast(0)
            return SELECTABLE[(i + step).mod(SELECTABLE.size)]
        }
    }
}

/** Semáforo de una métrica; [NONE] = no disponible. */
enum class PerfLevel { GOOD, WARN, BAD, NONE }

/** Frecuencia actual de un clúster de CPU (kHz, como en cpufreq). */
data class ClusterFreq(val cpus: List<Int>, val curKhz: Long, val maxKhz: Long) {
    /** Carga de frecuencia 0..1 (cur / max). */
    val load: Float get() = if (maxKhz <= 0) 0f else (curKhz.toFloat() / maxKhz).coerceIn(0f, 1f)
}

/**
 * Lectura del HUD. null = métrica no disponible (se muestra "—", nunca un número inventado).
 * [frameTimesMs] son los últimos [PerfMonitorLogic.HISTORY_SIZE] tiempos de frame (más antiguo primero).
 */
data class PerfSnapshot(
    val fps: Float? = null,
    val frameTimeMs: Float? = null,
    val presentFps: Float? = null,
    /** CPU usada por los procesos de la app (el juego corre dentro de ella), % del total de núcleos. */
    val cpuPercent: Float? = null,
    val gpuPercent: Float? = null,
    val cpuTempC: Float? = null,
    val gpuTempC: Float? = null,
    val batteryTempC: Float? = null,
    /** Consumo instantáneo de la batería (W, siempre positivo); null si el sensor no responde. */
    val batteryPowerW: Float? = null,
    /** Cargando: el consumo mostrado es lo que entra (el HUD lo marca con ⚡). */
    val charging: Boolean = false,
    /** PowerManager.getThermalHeadroom: 0 = frío, 1 = empieza el throttling. Solo si no hay °C de CPU. */
    val thermalHeadroom: Float? = null,
    val clusters: List<ClusterFreq> = emptyList(),
    val frameTimesMs: List<Float> = emptyList(),
) {
    companion object {
        val EMPTY = PerfSnapshot()
    }
}

/** Lógica pura del monitor (parseo y agregación), probada en la JVM. */
object PerfMonitorLogic {
    const val HISTORY_SIZE = 120
    /** Lecturas de `gpubusy` idénticas seguidas a partir de las cuales la GPU se da por en reposo. */
    const val GPU_IDLE_REPEATS = 3

    private const val MIN_VALID_C = -30f
    private const val MAX_VALID_C = 150f

    /** utime + stime (campos 14 y 15) de `/proc/<pid>/stat`; null si el formato no cuadra. */
    fun parseStatTicks(content: String): Long? {
        val rparen = content.lastIndexOf(')')
        if (rparen < 0) return null
        // Tras ")": state(3) ppid(4) ... utime(14) stime(15) → índices 11 y 12 desde el campo 3.
        val fields = content.substring(rparen + 1).trim().split(' ')
        if (fields.size < 13) return null
        val utime = fields[11].toLongOrNull() ?: return null
        val stime = fields[12].toLongOrNull() ?: return null
        if (utime < 0 || stime < 0) return null
        return utime + stime
    }

    /**
     * % de CPU (0..100 sobre TODOS los núcleos) entre dos muestras pid → ticks. Solo cuenta los pids
     * presentes en ambas (los que nacen o mueren en medio se ignoran). null sin datos suficientes.
     */
    fun cpuPercent(prev: Map<Int, Long>, cur: Map<Int, Long>, elapsedMs: Long, clkTck: Long, cores: Int): Float? {
        if (prev.isEmpty() || cur.isEmpty() || elapsedMs <= 0 || clkTck <= 0 || cores <= 0) return null
        var delta = 0L
        for ((pid, ticks) in cur) {
            val before = prev[pid] ?: continue
            if (ticks >= before) delta += ticks - before
        }
        val capacity = clkTck.toDouble() * elapsedMs / 1000.0 * cores
        return (delta / capacity * 100.0).toFloat().coerceIn(0f, 100f)
    }

    /** `gpubusy` de KGSL: "  busy total" → par; null si no se entiende. */
    fun parseGpuBusy(content: String): Pair<Long, Long>? {
        val parts = content.trim().split(Regex("\\s+"))
        if (parts.size < 2) return null
        val busy = parts[0].toLongOrNull() ?: return null
        val total = parts[1].toLongOrNull() ?: return null
        if (busy < 0 || total < 0) return null
        return busy to total
    }

    /** % ocupación de la GPU en la última ventana de KGSL; null si total = 0. */
    fun gpuPercent(busy: Long, total: Long): Float? {
        if (total <= 0) return null
        return (busy.toDouble() / total * 100.0).toFloat().coerceIn(0f, 100f)
    }

    /** `temp` de thermal_zone (m°C; algunos sensores dan °C) → °C; null si es un valor centinela. */
    fun parseThermalTemp(content: String): Float? {
        val raw = content.trim().toLongOrNull() ?: return null
        val c = if (raw > 1000 || raw < -1000) raw / 1000f else raw.toFloat()
        return if (c in MIN_VALID_C..MAX_VALID_C) c else null
    }

    /** Zonas térmicas de núcleos de CPU: `cpu-*` (preferidas); si no hay, `cpuss-*`, `cpu*`. */
    fun selectCpuZones(types: Map<String, String>): List<String> {
        val cores = types.filterValues { it.startsWith("cpu-") }.keys
        if (cores.isNotEmpty()) return cores.sorted()
        val ss = types.filterValues { it.startsWith("cpuss") }.keys
        if (ss.isNotEmpty()) return ss.sorted()
        return types.filterValues { it.startsWith("cpu") && !it.contains("therm") }.keys.sorted()
    }

    /** Zonas térmicas de la GPU (`gpuss-*`, `gpu`, `gpu-*`). */
    fun selectGpuZones(types: Map<String, String>): List<String> =
        types.filterValues { it.startsWith("gpuss") || it == "gpu" || it.startsWith("gpu-") || it.startsWith("gpu_") }
            .keys.sorted()

    /** Máximo de las lecturas válidas; null si ninguna lo es. */
    fun maxValid(temps: List<Float?>): Float? = temps.filterNotNull().maxOrNull()

    /** "0-5", "0 1 2 3", "0-1,4" → lista de CPUs. */
    fun parseCpuList(content: String): List<Int> {
        val out = sortedSetOf<Int>()
        for (token in content.trim().split(Regex("[,\\s]+"))) {
            if (token.isEmpty()) continue
            val dash = token.indexOf('-')
            if (dash > 0) {
                val a = token.substring(0, dash).toIntOrNull() ?: continue
                val b = token.substring(dash + 1).toIntOrNull() ?: continue
                if (a <= b) for (i in a..b) out.add(i)
            } else {
                token.toIntOrNull()?.let { out.add(it) }
            }
        }
        return out.toList()
    }

    /** Agrupa CPUs por clúster: (cpu → related_cpus) → listas únicas ordenadas por la primera CPU. */
    fun groupClusters(related: Map<Int, List<Int>>): List<List<Int>> =
        related.map { (cpu, rel) -> rel.ifEmpty { listOf(cpu) }.sorted() }.distinct().sortedBy { it.first() }

    /** Tiempo medio de frame (ms) en la ventana; null si no llegó ningún frame. */
    fun frameTimeMs(frames: Long, elapsedMs: Long): Float? =
        if (frames <= 0 || elapsedMs <= 0) null else elapsedMs.toFloat() / frames

    /** Frames por segundo en la ventana; null si el contador retrocedió o la ventana es vacía. */
    fun fps(prevCount: Long, count: Long, elapsedMs: Long): Float? =
        if (elapsedMs <= 0 || count < prevCount) null else (count - prevCount) * 1000f / elapsedMs

    /** Décimas de °C del intent de batería → °C; null si falta (Int.MIN_VALUE) o es absurdo. */
    fun batteryTempC(tenths: Int): Float? {
        if (tenths == Int.MIN_VALUE) return null
        val c = tenths / 10f
        return if (c in MIN_VALID_C..MAX_VALID_C) c else null
    }

    /**
     * FPS por jugabilidad, no por el tope: verde desde 55 FPS, ámbar entre 30 y 55, rojo solo por debajo de 30
     * (por debajo de ahí un juego deja de considerarse jugable). Un juego a 90 con tope 120 va bien: verde.
     */
    fun fpsLevel(fps: Float?): PerfLevel = when {
        fps == null -> PerfLevel.NONE
        fps >= GOOD_FPS -> PerfLevel.GOOD
        fps >= PLAYABLE_FPS -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /**
     * Con un tope bajo (30, 40...) se mide contra él: verde hasta 3 FPS por debajo del tope, rojo por debajo del
     * 80 % del tope. Sin tope (0) o con tope alto, la escala fija de [fpsLevel].
     */
    fun fpsLevel(fps: Float?, targetFps: Int): PerfLevel {
        if (fps == null) return PerfLevel.NONE
        if (targetFps <= 0) return fpsLevel(fps)
        val good = minOf(GOOD_FPS, targetFps - 3f)
        val playable = minOf(PLAYABLE_FPS, targetFps * 0.8f)
        return when {
            fps >= good -> PerfLevel.GOOD
            fps >= playable -> PerfLevel.WARN
            else -> PerfLevel.BAD
        }
    }

    /** Desde aquí el juego va fluido (verde). */
    const val GOOD_FPS = 55f
    /** Por debajo de esto un juego deja de ser jugable (rojo). */
    const val PLAYABLE_FPS = 30f
    /** Temperatura del SoC (CPU/GPU) del Snapdragon 8 Elite: < 65 °C verde, ≤ 80 °C ámbar, > 80 °C rojo. */
    fun tempLevel(c: Float?): PerfLevel = when {
        c == null -> PerfLevel.NONE
        c < 65f -> PerfLevel.GOOD
        c <= 80f -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /** Batería (objetivo < 40 °C): < 38 °C verde, < 41,5 °C ámbar, ≥ 41,5 °C rojo. */
    fun batteryTempLevel(c: Float?): PerfLevel = when {
        c == null -> PerfLevel.NONE
        c < 38f -> PerfLevel.GOOD
        c < ThermalGovernor.HOT_ON_C -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /** Uso de CPU/GPU: < 70 % verde, 70–90 % amarillo, > 90 % rojo. */
    fun usageLevel(pct: Float?): PerfLevel = when {
        pct == null -> PerfLevel.NONE
        pct < 70f -> PerfLevel.GOOD
        pct <= 90f -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /** Consumo en partida de un teléfono: < 4,5 W verde, ≤ 7 W ámbar, > 7 W rojo (calienta). */
    fun powerLevel(w: Float?): PerfLevel = when {
        w == null -> PerfLevel.NONE
        w < 4.5f -> PerfLevel.GOOD
        w <= 7f -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /**
     * Potencia (W) de BATTERY_PROPERTY_CURRENT_NOW (µA según la API; algunos fabricantes dan mA) y
     * EXTRA_VOLTAGE (mV). Valor absoluto: el signo depende del fabricante. null si falta algo.
     */
    fun batteryPowerW(current: Long, voltageMv: Int): Float? {
        if (current == Long.MIN_VALUE || current == Int.MIN_VALUE.toLong() || voltageMv <= 0) return null
        val abs = kotlin.math.abs(current)
        if (abs == 0L) return null
        // < 20 000 en µA serían < 20 mA, imposible con la pantalla encendida: viene en mA.
        val amps = if (abs < 20_000L) abs / 1_000f else abs / 1_000_000f
        val w = amps * voltageMv / 1000f
        return if (w in 0.01f..40f) w else null
    }

    /** Margen térmico (0 frío … 1 throttling). */
    fun headroomLevel(h: Float?): PerfLevel = when {
        h == null -> PerfLevel.NONE
        h < 0.7f -> PerfLevel.GOOD
        h < 0.9f -> PerfLevel.WARN
        else -> PerfLevel.BAD
    }

    /** Número con [decimals] decimales o "—". Siempre con punto decimal (independiente del locale). */
    fun formatOrDash(value: Float?, decimals: Int = 0): String {
        if (value == null || value.isNaN() || value.isInfinite()) return "—"
        return String.format(java.util.Locale.ROOT, "%.${decimals}f", value)
    }

    /** GHz con un decimal ("3.5"); "—" si no hay dato. */
    fun formatGhz(khz: Long): String = if (khz <= 0) "—" else String.format(java.util.Locale.ROOT, "%.1f", khz / 1_000_000f)

    /** MHz enteros ("1363"); "—" si no hay dato. */
    fun formatMhz(khz: Long): String = if (khz <= 0) "—" else (khz / 1000).toString()

    /**
     * FPS mínimos del historial (peor ventana de 250 ms de los últimos ~30 s) = 1000 / mayor
     * tiempo de frame. No es un "1 % low" real: el muestreo promedia cada ventana. null sin datos.
     */
    fun minFps(frameTimesMs: List<Float>): Float? {
        val worst = frameTimesMs.filter { it > 0f && it.isFinite() }.maxOrNull() ?: return null
        return 1000f / worst
    }
}

/** Intent fijo ACTION_BATTERY_CHANGED (sin registrar receptor: es "sticky"). */
fun batteryIntent(context: Context): Intent? = runCatching {
    ContextCompat.registerReceiver(context, null, IntentFilter(Intent.ACTION_BATTERY_CHANGED), ContextCompat.RECEIVER_NOT_EXPORTED)
}.getOrNull()

/** Temperatura de la batería (°C) o null. La usan el HUD y el gobernador térmico. */
fun readBatteryTempC(context: Context): Float? =
    PerfMonitorLogic.batteryTempC(batteryIntent(context)?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE)

/** Historial circular de tiempos de frame (para el sparkline). */
class FrameTimeHistory(private val capacity: Int = PerfMonitorLogic.HISTORY_SIZE) {
    private val values = ArrayDeque<Float>(capacity)

    fun add(ms: Float) {
        if (values.size == capacity) values.removeFirst()
        values.addLast(ms)
    }

    fun snapshot(): List<Float> = values.toList()

    val size: Int get() = values.size
}

/**
 * Detecta la GPU en reposo: KGSL deja de actualizar `gpubusy` cuando no hay trabajo, así que el
 * mismo par repetido [PerfMonitorLogic.GPU_IDLE_REPEATS] veces seguidas = 0 %.
 */
class GpuBusyTracker {
    private var last: Pair<Long, Long>? = null
    private var repeats = 0

    fun update(sample: Pair<Long, Long>?): Float? {
        if (sample == null) {
            last = null
            repeats = 0
            return null
        }
        repeats = if (sample == last) repeats + 1 else 0
        last = sample
        if (repeats >= PerfMonitorLogic.GPU_IDLE_REPEATS) return 0f
        return PerfMonitorLogic.gpuPercent(sample.first, sample.second)
    }
}

/**
 * Lecturas de /proc y /sys (raíces inyectables para los tests). Todo lo que falla devuelve null o
 * vacío: el HUD muestra "—".
 *
 * Fuentes (RedMagic 10 Pro, Android 16, comprobadas con run-as; ver informe):
 * - CPU: `/proc/<pid>/stat` de los procesos visibles. /proc está montado con hidepid, así que una
 *   app solo ve los de su UID (con el motor directo, el propio proceso de la app).
 *   `/proc/stat` (CPU global) está bloqueado.
 * - Frecuencias: `/sys/devices/system/cpu/cpuN/cpufreq/{scaling_cur_freq,cpuinfo_max_freq,related_cpus}`.
 * - GPU: `/sys/class/kgsl/kgsl-3d0/gpubusy` (etiqueta vendor_sysfs_kgsl_gpubusy). gpu_busy_percentage,
 *   gpuclk y devfreq están denegados.
 * - Temperaturas: `/sys/class/thermal/thermal_zone*` (tipos cpu-X-Y-Z, cpuss-*, gpuss-*).
 */
class PerfSources(
    private val procRoot: File = File("/proc"),
    private val cpuRoot: File = File("/sys/devices/system/cpu"),
    private val thermalRoot: File = File("/sys/class/thermal"),
    private val gpuBusyFile: File = File("/sys/class/kgsl/kgsl-3d0/gpubusy"),
) {
    private var cpuZones: List<File>? = null
    private var gpuZones: List<File>? = null
    private var clusterLayout: List<Pair<List<Int>, Long>>? = null

    private fun read(file: File): String? = try {
        file.readText()
    } catch (_: Exception) {
        null
    }

    /** pid → utime+stime de todos los procesos legibles. */
    fun readProcessTicks(): Map<Int, Long> {
        val out = HashMap<Int, Long>()
        val names = procRoot.list() ?: return out
        for (name in names) {
            val pid = name.toIntOrNull() ?: continue
            val ticks = read(File(procRoot, "$name/stat"))?.let(PerfMonitorLogic::parseStatTicks) ?: continue
            out[pid] = ticks
        }
        return out
    }

    fun readGpuBusy(): Pair<Long, Long>? = read(gpuBusyFile)?.let(PerfMonitorLogic::parseGpuBusy)

    private fun discoverZones() {
        val types = HashMap<String, String>()
        thermalRoot.list()?.filter { it.startsWith("thermal_zone") }?.forEach { zone ->
            read(File(thermalRoot, "$zone/type"))?.trim()?.let { types[zone] = it }
        }
        // Solo zonas cuya temp se lea ahora y dé un valor válido (descarta sensores ausentes).
        fun usable(zones: List<String>) = zones.map { File(thermalRoot, "$it/temp") }
            .filter { f -> read(f)?.let(PerfMonitorLogic::parseThermalTemp) != null }
        cpuZones = usable(PerfMonitorLogic.selectCpuZones(types))
        gpuZones = usable(PerfMonitorLogic.selectGpuZones(types))
    }

    private fun maxTemp(zones: List<File>): Float? =
        PerfMonitorLogic.maxValid(zones.map { f -> read(f)?.let(PerfMonitorLogic::parseThermalTemp) })

    fun readCpuTemp(): Float? {
        if (cpuZones == null) discoverZones()
        return maxTemp(cpuZones.orEmpty())
    }

    fun readGpuTemp(): Float? {
        if (gpuZones == null) discoverZones()
        return maxTemp(gpuZones.orEmpty())
    }

    fun readClusters(): List<ClusterFreq> {
        val layout = clusterLayout ?: discoverClusters().also { clusterLayout = it }
        return layout.mapNotNull { (cpus, maxKhz) ->
            // Primera CPU en línea del clúster (una apagada no tiene scaling_cur_freq legible).
            val cur = cpus.firstNotNullOfOrNull { cpu ->
                read(File(cpuRoot, "cpu$cpu/cpufreq/scaling_cur_freq"))?.trim()?.toLongOrNull()
            } ?: return@mapNotNull null
            ClusterFreq(cpus, cur, maxKhz)
        }
    }

    private fun discoverClusters(): List<Pair<List<Int>, Long>> {
        val cpus = read(File(cpuRoot, "possible"))?.let(PerfMonitorLogic::parseCpuList).orEmpty()
        val related = HashMap<Int, List<Int>>()
        for (cpu in cpus) {
            val rel = read(File(cpuRoot, "cpu$cpu/cpufreq/related_cpus"))?.let(PerfMonitorLogic::parseCpuList) ?: continue
            related[cpu] = rel
        }
        return PerfMonitorLogic.groupClusters(related).mapNotNull { group ->
            val max = read(File(cpuRoot, "cpu${group.first()}/cpufreq/cpuinfo_max_freq"))?.trim()?.toLongOrNull()
                ?: return@mapNotNull null
            group to max
        }
    }
}

/**
 * Muestreador del HUD: FPS a 4 Hz (contadores nativos del puente) y /proc + /sys a 2 Hz, en
 * hilos de fondo. [run] no vuelve hasta que se cancela su corrutina; entrega cada lectura con
 * [publish] en el contexto del llamador (lanzarlo desde el hilo principal para tocar estado Compose).
 * Mientras [enabled] devuelva false no lee nada.
 */
class PerfMonitor(
    private val context: Context,
    private val sources: PerfSources = PerfSources(),
    private val frameCounter: () -> Long = HyperPlayNative::nativeDirectFramesPresented,
    // FPS reales del juego; si aún no hay muestra, cae a las presentaciones del swapchain.
    private val presentFps: () -> Float = {
        HyperPlayNative.nativeGameFps().takeIf { it > 0f } ?: HyperPlayNative.nativeDirectPresentFps()
    },
) {
    private val clkTck = runCatching { Os.sysconf(OsConstants._SC_CLK_TCK) }.getOrDefault(100L).takeIf { it > 0 } ?: 100L
    private val cores = runCatching { Os.sysconf(OsConstants._SC_NPROCESSORS_CONF).toInt() }.getOrDefault(0)
        .takeIf { it > 0 } ?: Runtime.getRuntime().availableProcessors()
    private val history = FrameTimeHistory()
    private val gpuTracker = GpuBusyTracker()

    suspend fun run(enabled: () -> Boolean, publish: (PerfSnapshot) -> Unit) {
        var snap = PerfSnapshot.EMPTY
        var lastFrames = -1L
        var lastFramesAt = 0L
        var lastTicks: Map<Int, Long> = emptyMap()
        var lastTicksAt = 0L
        var tick = 0L
        while (currentCoroutineContext().isActive) {
            if (!enabled()) {
                lastFrames = -1L
                lastTicks = emptyMap()
                delay(FPS_PERIOD_MS)
                continue
            }
            val now = SystemClock.elapsedRealtime()
            val frames = frameCounter()
            var fps: Float? = snap.fps
            var frameMs: Float? = snap.frameTimeMs
            if (lastFrames >= 0) {
                val dt = now - lastFramesAt
                fps = PerfMonitorLogic.fps(lastFrames, frames, dt)
                frameMs = PerfMonitorLogic.frameTimeMs(frames - lastFrames, dt)
                frameMs?.let(history::add)
            }
            lastFrames = frames
            lastFramesAt = now
            snap = snap.copy(
                fps = fps,
                frameTimeMs = frameMs,
                presentFps = presentFps().takeIf { it > 0f },
                frameTimesMs = history.snapshot(),
            )

            if (tick % SYS_EVERY_TICKS == 0L) {
                val withBattery = tick % BATTERY_EVERY_TICKS == 0L
                val prev = snap
                snap = withContext(Dispatchers.IO) {
                    val ticks = sources.readProcessTicks()
                    val readAt = SystemClock.elapsedRealtime()
                    val cpu = PerfMonitorLogic.cpuPercent(lastTicks, ticks, readAt - lastTicksAt, clkTck, cores)
                    lastTicks = ticks
                    lastTicksAt = readAt
                    val cpuTemp = sources.readCpuTemp()
                    prev.copy(
                        cpuPercent = cpu ?: prev.cpuPercent,
                        gpuPercent = gpuTracker.update(sources.readGpuBusy()),
                        cpuTempC = cpuTemp,
                        gpuTempC = sources.readGpuTemp(),
                        clusters = sources.readClusters(),
                        batteryTempC = if (withBattery) readBatteryTempC(context) else prev.batteryTempC,
                        batteryPowerW = readBatteryPower().first ?: prev.batteryPowerW,
                        charging = readBatteryPower().second,
                        thermalHeadroom = when {
                            cpuTemp != null -> null
                            withBattery -> readHeadroom()
                            else -> prev.thermalHeadroom
                        },
                    )
                }
            }
            publish(snap)
            tick++
            delay(FPS_PERIOD_MS)
        }
    }

    private var powerCache: Pair<Float?, Boolean> = null to false
    private var powerCacheAt = 0L

    /** Potencia de la batería (2 Hz, cacheada dentro de la misma lectura) y si está cargando. */
    private fun readBatteryPower(): Pair<Float?, Boolean> {
        val now = SystemClock.elapsedRealtime()
        if (now - powerCacheAt < 200L) return powerCache
        powerCacheAt = now
        val bm = context.getSystemService(BatteryManager::class.java)
        val current = runCatching { bm?.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) }.getOrNull()
            ?: Long.MIN_VALUE
        val intent = batteryIntent(context)
        val mv = intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        val status = intent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
        powerCache = PerfMonitorLogic.batteryPowerW(current, mv) to charging
        return powerCache
    }

    /** Respaldo si no hay °C de CPU legibles. La API limita a ~1 llamada/s (si no, NaN). */
    private fun readHeadroom(): Float? {
        val pm = context.getSystemService(PowerManager::class.java) ?: return null
        val h = runCatching { pm.getThermalHeadroom(HEADROOM_FORECAST_S) }.getOrNull() ?: return null
        return h.takeIf { !it.isNaN() && it >= 0f }
    }

    companion object {
        /** 4 Hz para FPS / tiempo de frame. */
        const val FPS_PERIOD_MS = 250L
        /** /proc y /sys cada 2 ticks = 2 Hz. */
        private const val SYS_EVERY_TICKS = 2L
        /** Batería y margen térmico cada 8 ticks = 0,5 Hz (cambian despacio; la API térmica limita). */
        private const val BATTERY_EVERY_TICKS = 8L
        private const val HEADROOM_FORECAST_S = 0
    }
}
