#include "FramePacing.h"

#include <android/log.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <cstdlib>
#include <ctime>

#include "Win32DisplayInfo.h"
#include "cpu/GuestProfiler.h"
#include "cpu/OryonSiliconScheduler.h"

#define LOG_TAG "HyperPlay.Pacing"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace hyperplay::pacing {
namespace {

std::atomic<int> g_requestedCap{-1};
std::mutex g_pauseMutex;
std::condition_variable g_pauseCv;
std::atomic<bool> g_paused{false};
std::atomic<pid_t> g_mainTid{0};
std::atomic<pid_t> g_renderTid{0};
std::atomic<pid_t> g_workerTid{0};
std::atomic<bool> g_relax{false};

OryonSiliconScheduler& scheduler() {
    static OryonSiliconScheduler s;
    return s;
}

int64_t nowNs() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

// `debug.hyperplay.fpscap` (leído una vez): >= 0 fija el tope para pruebas A/B; -1 = sin fijar.
int propCap() {
    static const int s_prop = [] {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.fpscap", v) > 0) return std::atoi(v);
        return -1;
    }();
    return s_prop;
}

bool adpfEnabled() {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return !(__system_property_get("debug.hyperplay.adpf", v) > 0 && v[0] == '0');
    }();
    return s_on;
}

// Anclaje estricto del hilo principal a los 2 Prime: DESACTIVADO por defecto. Medido en Hollow
// Knight (Cruce Olvidado, tope 72): 62,6 fps anclado frente a 66,8 fps libre; con todos los
// núcleos el hilo principal rinde más. `setprop debug.hyperplay.primepin 1` lo activa (y entonces
// la protección térmica lo relaja al calentarse).
bool primePinEnabled() {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.primepin", v) > 0 && v[0] == '1';
    }();
    return s_on;
}

bool cpuPin32Enabled() {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.cpupin32", v) > 0 && v[0] == '1';
    }();
    return s_on;
}

// Objetivo que se pide a ADPF: el 80 % del periodo del tope. Con el periodo exacto el reloj se
// quedaba justo y los frames que se pasaban unas décimas caían al vsync siguiente (60 -> 57 fps).
// `setprop debug.hyperplay.adpfpct <50-100>` lo ajusta para pruebas A/B.
int64_t adpfMarginPct() {
    static const int64_t s_pct = [] {
        char v[PROP_VALUE_MAX] = {};
        const int p = __system_property_get("debug.hyperplay.adpfpct", v) > 0 ? std::atoi(v) : 0;
        return static_cast<int64_t>(p >= 50 && p <= 100 ? p : 80);
    }();
    return s_pct;
}

// FPS reales del juego: una muestra por llamada a paceFrame(), ventana de 500 ms.
std::atomic<float> g_gameFps{0.0f};
std::atomic<int64_t> g_lastFrameNs{0};

void countFrame(int64_t t) {
    static int64_t s_winStart = 0;
    static int s_winFrames = 0;
    if (s_winStart == 0) s_winStart = t;
    ++s_winFrames;
    if (t - s_winStart >= 500000000LL) {
        g_gameFps.store(static_cast<float>(s_winFrames) * 1e9f / static_cast<float>(t - s_winStart),
                        std::memory_order_relaxed);
        s_winStart = t;
        s_winFrames = 0;
    }
    g_lastFrameNs.store(t, std::memory_order_relaxed);
}

}  // namespace

float gameFps() noexcept {
    if (nowNs() - g_lastFrameNs.load(std::memory_order_relaxed) > 1000000000LL) return 0.0f;
    return g_gameFps.load(std::memory_order_relaxed);
}

void setFpsCap(int fps) noexcept {
    const int old = g_requestedCap.exchange(fps);
    if (old != fps) LOGI("Tope de FPS pedido: %d (en vigor %d)", fps, effectiveCap());
}

int effectiveCap() noexcept {
    const int p = propCap();
    if (p >= 0) return p;
    const int req = g_requestedCap.load(std::memory_order_relaxed);
    if (req >= 0) return req;
    // Por defecto (sin app que lo fije): la mitad del refresco, 2 vsyncs exactos.
    const int hz = static_cast<int>(::hyperplay::win32::refreshHz());
    return hz >= 90 ? hz / 2 : 0;
}

void setMainThread(pid_t tid) noexcept {
    g_mainTid.store(tid);
    ::hyperplay::prof::setMainThread(tid);
    if (cpuPin32Enabled()) {
        static_cast<void>(scheduler().pinToPrimeCores(tid));
    } else if (primePinEnabled()) {
        static_cast<void>(scheduler().setPrimeRelaxed(tid, g_relax.load()));
    }
}

void setRenderThread(pid_t tid) noexcept {
    if (tid <= 0) return;
    const pid_t prev = g_renderTid.exchange(tid);
    if (prev != tid && cpuPin32Enabled()) {
        static_cast<void>(scheduler().pinToPrimeCores(tid));
    }
}

void setRenderWorkerThread(pid_t tid) noexcept {
    if (tid <= 0) return;
    const pid_t prev = g_workerTid.exchange(tid);
    if (prev != tid && cpuPin32Enabled()) {
        static_cast<void>(scheduler().pinToPerfCores(tid));
    }
}

pid_t mainThread() noexcept { return g_mainTid.load(); }

bool thermalRelaxed() noexcept { return g_relax.load(std::memory_order_relaxed); }

void setThermalRelax(bool relax) noexcept {
    if (g_relax.exchange(relax) == relax) return;
    const pid_t tid = g_mainTid.load();
    if (tid > 0 && primePinEnabled()) static_cast<void>(scheduler().setPrimeRelaxed(tid, relax));
}

// Limitador: espera absoluta con clock_nanosleep (sin deriva acumulada); un frame tardío
// reancla el reloj en vez de intentar "recuperar". Idéntico al limitador anterior; el periodo
// se recalcula en cada frame para poder cambiar el tope en caliente sin tocar el swapchain.
void setPaused(bool paused) noexcept {
    {
        std::lock_guard<std::mutex> lock(g_pauseMutex);
        if (g_paused.exchange(paused) == paused) return;
    }
    g_pauseCv.notify_all();
    LOGI("Juego %s", paused ? "EN PAUSA (Quick Resume)" : "reanudado");
}

bool paused() noexcept { return g_paused.load(std::memory_order_relaxed); }

void paceFrame() noexcept {
    static int64_t s_next = 0;
    static int64_t s_lastWake = 0;
    // Diagnóstico (`debug.hyperplay.framelog 1`, leído una vez): numera cada Present en logcat
    // para ordenar por frames los eventos del juego (p. ej. el trazado de Mono).
    static const bool s_frameLog = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.framelog", v) > 0 && v[0] == '1';
    }();
    if (s_frameLog) {
        static uint64_t s_frameNo = 0;
        LOGI("[Frame] %llu", static_cast<unsigned long long>(++s_frameNo));
    }
    if (g_paused.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(g_pauseMutex);
        g_pauseCv.wait(lock, [] { return !g_paused.load(std::memory_order_acquire); });
        // Al volver: sin "recuperar" frames ni informar a ADPF de la pausa como trabajo.
        s_next = 0;
        s_lastWake = 0;
    }
    const int cap = effectiveCap();
    const int64_t now = nowNs();

    // ADPF: trabajo real del frame = desde que terminó la espera anterior hasta este Present.
    const int hz = static_cast<int>(::hyperplay::win32::refreshHz());
    const int64_t target = 1000000000LL / (cap > 0 ? cap : (hz > 0 ? hz : 60)) * adpfMarginPct() / 100;
    if (s_lastWake != 0 && adpfEnabled()) {
        const pid_t rtid = g_renderTid.load();
        const pid_t wtid = g_workerTid.load();
        const pid_t tids[3] = {g_mainTid.load(), rtid > 0 ? rtid : static_cast<pid_t>(gettid()), wtid};
        const size_t n = wtid > 0 ? 3 : 2;
        if (scheduler().ensureFrameSession(tids, n, target)) {
            scheduler().reportFrameWork(now - s_lastWake, target);
        }
    }

    if (cap <= 0) {
        s_next = 0;
        s_lastWake = nowNs();
        countFrame(s_lastWake);
        return;
    }
    const int64_t period = 1000000000LL / cap;
    int64_t next = s_next;
    if (next == 0 || now - next > period) {
        next = now;  // primer frame o frame tardío: se reancla, no se intenta "recuperar"
    } else if (next > now) {
        const timespec t{static_cast<time_t>(next / 1000000000LL), static_cast<long>(next % 1000000000LL)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr) == EINTR) {
        }
    }
    s_next = next + period;
    s_lastWake = nowNs();
    countFrame(s_lastWake);
}

}  // namespace hyperplay::pacing
