#include "OryonSiliconScheduler.h"

#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define LOG_TAG "HyperPlay.Scheduler"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace {

// Por encima de 3.9 GHz solo están los núcleos Prime de Oryon/SM8750.
constexpr int64_t kPrimeFreqThresholdKhz = 3900000;

int64_t readCpuMaxFreq(int cpu) noexcept {
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[32];
    const ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char* end = nullptr;
    const long long value = std::strtoll(buf, &end, 10);
    if (end == buf || value <= 0) return -1;
    return static_cast<int64_t>(value);
}

pid_t resolveTid(pid_t tid) noexcept {
    return tid > 0 ? tid : static_cast<pid_t>(gettid());
}

bool applyMask(pid_t tid, const cpu_set_t& mask, const char* label) noexcept {
    const pid_t target = resolveTid(tid);
    if (target <= 0) {
        LOGW("Afinidad %s: tid inválido (%d)", label, static_cast<int>(tid));
        return false;
    }
    // Diagnóstico: setprop debug.hyperplay.onecpu N fuerza todo a la CPU N (los hilos nuevos la heredan).
    // Sirve para distinguir carreras / orden de memoria de fallos deterministas.
    static const int s_oneCpu = [] {
        char b[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.onecpu", b) > 0 ? std::atoi(b) : -1;
    }();
    if (s_oneCpu >= 0 && s_oneCpu < CPU_SETSIZE) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(s_oneCpu, &one);
        return sched_setaffinity(target, sizeof(one), &one) == 0;
    }
    if (sched_setaffinity(target, sizeof(mask), &mask) != 0) {
        LOGW("sched_setaffinity(%s) falló para tid %d: errno=%d (%s)", label,
             static_cast<int>(target), errno, std::strerror(errno));
        return false;
    }
    return true;
}

struct AdpfApi {
    void* handle = nullptr;
    void* manager = nullptr;
    void* (*createSession)(void*, const int32_t*, size_t, int64_t) = nullptr;
    int (*closeSession)(void*) = nullptr;
    int (*reportActual)(void*, int64_t) = nullptr;   // APerformanceHint_reportActualWorkDuration
    int (*updateTarget)(void*, int64_t) = nullptr;   // APerformanceHint_updateTargetWorkDuration
};

AdpfApi& adpfApi() noexcept {
    static AdpfApi api = [] {
        AdpfApi out;
        out.handle = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
        if (out.handle == nullptr) {
            const char* err = dlerror();
            LOGW("ADPF: dlopen(libandroid.so) falló: %s", err != nullptr ? err : "sin detalle");
            return out;
        }
        auto getManager =
            reinterpret_cast<void* (*)()>(dlsym(out.handle, "APerformanceHint_getManager"));
        // Nombre real del NDK (API 33): APerformanceHint_createSession(manager, tids, n, objetivo).
        // Antes se buscaba "APerformanceHintManager_createSession", que no existe: ADPF nunca llegó
        // a activarse.
        out.createSession = reinterpret_cast<void* (*)(void*, const int32_t*, size_t, int64_t)>(
            dlsym(out.handle, "APerformanceHint_createSession"));
        out.closeSession =
            reinterpret_cast<int (*)(void*)>(dlsym(out.handle, "APerformanceHint_closeSession"));
        out.reportActual = reinterpret_cast<int (*)(void*, int64_t)>(
            dlsym(out.handle, "APerformanceHint_reportActualWorkDuration"));
        out.updateTarget = reinterpret_cast<int (*)(void*, int64_t)>(
            dlsym(out.handle, "APerformanceHint_updateTargetWorkDuration"));
        if (getManager == nullptr || out.createSession == nullptr) {
            const char* err = dlerror();
            LOGW("ADPF: símbolos no disponibles en libandroid.so: %s",
                 err != nullptr ? err : "sin detalle");
            dlclose(out.handle);
            out = AdpfApi{};
            return out;
        }
        out.manager = getManager();
        if (out.manager == nullptr) {
            LOGW("ADPF: APerformanceHint_getManager devolvió null (dispositivo sin soporte)");
        }
        // El handle se conserva: cerrarlo descargaría la librería mientras la sesión sigue viva.
        return out;
    }();
    return api;
}

}  // namespace

OryonSiliconScheduler::OryonSiliconScheduler() noexcept
    : topology_(detect()) {}

OryonSiliconScheduler::~OryonSiliconScheduler() noexcept {
    closeAdpf();
    closeFrameSession();
}

// ---- Sesión ADPF por frame (modo eficiente) -------------------------------------------------
// Android ajusta la frecuencia de los núcleos de estos hilos a lo justo para cumplir el
// presupuesto del tope de FPS: si el juego va sobrado, baja relojes (menos calor) en vez de
// mantener los Prime al máximo; si se acerca al límite, sube.
bool OryonSiliconScheduler::ensureFrameSession(const pid_t* tids, size_t count, int64_t targetNs) noexcept {
    std::lock_guard<std::mutex> lock(frameMutex_);
    int32_t ids[8];
    size_t n = 0;
    for (size_t i = 0; i < count && n < 8; ++i) {
        if (tids[i] > 0) ids[n++] = static_cast<int32_t>(tids[i]);
    }
    if (frameSession_ != nullptr) {
        if (n <= frameTidCount_) return true;
        AdpfApi& api = adpfApi();
        if (api.closeSession != nullptr) api.closeSession(frameSession_);
        frameSession_ = nullptr;
        frameTidCount_ = 0;
    }
    if (frameSessionFailed_ || n == 0 || targetNs <= 0) return false;
    AdpfApi& api = adpfApi();
    if (api.manager == nullptr || api.createSession == nullptr || api.reportActual == nullptr) {
        frameSessionFailed_ = true;
        LOGW("ADPF por frame no disponible");
        return false;
    }
    frameSession_ = api.createSession(api.manager, ids, n, targetNs);
    if (frameSession_ == nullptr) {
        frameSessionFailed_ = true;
        LOGW("ADPF por frame: createSession falló (%zu hilos, objetivo %lld ns)", n, static_cast<long long>(targetNs));
        return false;
    }
    frameTidCount_ = n;
    frameTargetNs_ = targetNs;
    LOGI("ADPF por frame activo: %zu hilos, objetivo %.2f ms", n, static_cast<double>(targetNs) / 1e6);
    return true;
}

void OryonSiliconScheduler::reportFrameWork(int64_t actualNs, int64_t targetNs) noexcept {
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (frameSession_ == nullptr || actualNs <= 0) return;
    AdpfApi& api = adpfApi();
    if (targetNs > 0 && targetNs != frameTargetNs_ && api.updateTarget != nullptr) {
        api.updateTarget(frameSession_, targetNs);
        frameTargetNs_ = targetNs;
    }
    api.reportActual(frameSession_, actualNs);
}

void OryonSiliconScheduler::closeFrameSession() noexcept {
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (frameSession_ == nullptr) return;
    AdpfApi& api = adpfApi();
    if (api.closeSession != nullptr) api.closeSession(frameSession_);
    frameSession_ = nullptr;
}

// Anclaje del hilo principal del juego: estricto a Prime o relajado a todos los núcleos (para
// repartir el calor). Reversible en cualquier momento; no toca el resto de hilos.
bool OryonSiliconScheduler::setPrimeRelaxed(pid_t tid, bool relaxed) noexcept {
    if (tid <= 0) return false;
    const bool ok = relaxed ? pinToAllCores(tid) : pinToPrimeCores(tid);
    LOGI("Hilo principal %d: anclaje %s (%s)", static_cast<int>(tid), relaxed ? "relajado (todos los núcleos)" : "Prime",
         ok ? "ok" : "falló");
    return ok;
}

CoreTopology OryonSiliconScheduler::detect() noexcept {
    CoreTopology t;
    long nprocs = sysconf(_SC_NPROCESSORS_CONF);
    if (nprocs <= 0) nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs <= 0) return t;
    t.cpuCount = std::min(static_cast<int>(nprocs), static_cast<int>(CPU_SETSIZE));

    struct CoreFreq {
        int cpu = 0;
        int64_t khz = -1;
    };
    std::vector<CoreFreq> cores;
    cores.reserve(static_cast<size_t>(t.cpuCount));
    for (int cpu = 0; cpu < t.cpuCount; ++cpu) {
        cores.push_back(CoreFreq{cpu, readCpuMaxFreq(cpu)});
    }

    // Orden descendente por frecuencia máxima; las lecturas inválidas van al final y desempatan
    // por índice, que en ARM big.LITTLE identifica los núcleos de mayor capacidad.
    std::stable_sort(cores.begin(), cores.end(), [](const CoreFreq& a, const CoreFreq& b) {
        if (a.khz != b.khz) return a.khz > b.khz;
        return a.cpu > b.cpu;
    });

    for (const CoreFreq& core : cores) {
        if (t.primeCount < 2) {
            t.primeCores[t.primeCount++] = core.cpu;
        } else if (t.perfCount < 6) {
            t.perfCores[t.perfCount++] = core.cpu;
        }
    }

    for (int i = 0; i < t.primeCount; ++i) {
        t.primeMaxKhz = std::max(t.primeMaxKhz, cores[static_cast<size_t>(i)].khz);
    }
    for (int i = t.primeCount; i < t.primeCount + t.perfCount; ++i) {
        t.perfMaxKhz = std::max(t.perfMaxKhz, cores[static_cast<size_t>(i)].khz);
    }

    t.isOryon = t.primeCount == 2 && t.perfCount >= 4 &&
                t.primeMaxKhz >= kPrimeFreqThresholdKhz && t.perfMaxKhz < t.primeMaxKhz;
    return t;
}

bool OryonSiliconScheduler::pinToPrimeCores(pid_t tid) noexcept {
    if (topology_.primeCount == 0) {
        LOGW("Afinidad Prime: topología sin núcleos detectados");
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < topology_.primeCount; ++i) {
        const int cpu = topology_.primeCores[i];
        if (cpu < 0 || cpu >= topology_.cpuCount || cpu >= CPU_SETSIZE) continue;
        CPU_SET(cpu, &set);
    }
    if (CPU_COUNT(&set) == 0) {
        LOGW("Afinidad Prime: ninguna CPU válida en la máscara");
        return false;
    }
    if (!applyMask(tid, set, "prime")) return false;
    if (topology_.primeCount == 1) {
        LOGI("Hilo %d fijado a Prime (cpu%d @ %.2fGHz)", static_cast<int>(resolveTid(tid)),
             topology_.primeCores[0], static_cast<double>(topology_.primeMaxKhz) / 1000000.0);
    } else {
        LOGI("Hilo %d fijado a Prime (cpu%d, cpu%d @ %.2fGHz)", static_cast<int>(resolveTid(tid)),
             topology_.primeCores[0], topology_.primeCores[1],
             static_cast<double>(topology_.primeMaxKhz) / 1000000.0);
    }
    return true;
}

bool OryonSiliconScheduler::pinToPerfCores(pid_t tid) noexcept {
    if (topology_.perfCount == 0) {
        LOGW("Afinidad Performance: topología sin núcleos detectados");
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < topology_.perfCount; ++i) {
        const int cpu = topology_.perfCores[i];
        if (cpu < 0 || cpu >= topology_.cpuCount || cpu >= CPU_SETSIZE) continue;
        CPU_SET(cpu, &set);
    }
    if (CPU_COUNT(&set) == 0) {
        LOGW("Afinidad Performance: ninguna CPU válida en la máscara");
        return false;
    }
    return applyMask(tid, set, "perf");
}

bool OryonSiliconScheduler::pinToAllCores(pid_t tid) noexcept {
    if (topology_.cpuCount <= 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu = 0; cpu < topology_.cpuCount && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &set);
    return applyMask(tid, set, "all");
}

bool OryonSiliconScheduler::setupAdpf(pid_t tid, int64_t targetFrameNs) noexcept {
    if (adpfSession_ != nullptr) return true;
    if (targetFrameNs <= 0) {
        LOGW("ADPF: objetivo de frame inválido (%lld ns)", static_cast<long long>(targetFrameNs));
        return false;
    }
    AdpfApi& api = adpfApi();
    if (api.manager == nullptr || api.createSession == nullptr) {
        LOGW("ADPF no disponible en este dispositivo");
        return false;
    }

    const int32_t targetTid = static_cast<int32_t>(resolveTid(tid));
    adpfSession_ = api.createSession(api.manager, &targetTid, 1, targetFrameNs);
    if (adpfSession_ == nullptr) {
        LOGW("ADPF: createSession falló para tid %d (objetivo %lld ns)", targetTid,
             static_cast<long long>(targetFrameNs));
        return false;
    }
    LOGI("ADPF activo para tid %d con objetivo de %lld ns", targetTid,
         static_cast<long long>(targetFrameNs));
    return true;
}

void OryonSiliconScheduler::closeAdpf() noexcept {
    if (adpfSession_ == nullptr) return;
    AdpfApi& api = adpfApi();
    if (api.closeSession != nullptr) {
        const int rc = api.closeSession(adpfSession_);
        if (rc != 0) LOGW("ADPF: closeSession devolvió %d", rc);
    }
    adpfSession_ = nullptr;
}

void OryonSiliconScheduler::setThreadPriority(pid_t tid, int schedPolicy, int priority) noexcept {
    struct sched_param param;
    std::memset(&param, 0, sizeof(param));
    param.sched_priority = priority;
    const pid_t target = resolveTid(tid);
    if (sched_setscheduler(target, schedPolicy, &param) != 0) {
        LOGW("sched_setscheduler(tid %d, policy %d, prio %d) falló: errno=%d (%s)",
             static_cast<int>(target), schedPolicy, priority, errno, std::strerror(errno));
    }
}

std::string OryonSiliconScheduler::topologyInfo() noexcept {
    const CoreTopology t = detect();
    if (t.cpuCount <= 0) return std::string("Topología de CPU no disponible");

    char prime[48] = "n/d";
    char perf[64] = "n/d";
    if (t.primeCount == 2) {
        std::snprintf(prime, sizeof(prime), "cpu%d, cpu%d @ %.2fGHz", t.primeCores[0],
                      t.primeCores[1], static_cast<double>(t.primeMaxKhz) / 1000000.0);
    } else if (t.primeCount == 1) {
        std::snprintf(prime, sizeof(prime), "cpu%d @ %.2fGHz", t.primeCores[0],
                      static_cast<double>(t.primeMaxKhz) / 1000000.0);
    }
    if (t.perfCount > 0) {
        std::snprintf(perf, sizeof(perf), "cpu%d-cpu%d @ %.2fGHz", t.perfCores[0],
                      t.perfCores[t.perfCount - 1], static_cast<double>(t.perfMaxKhz) / 1000000.0);
    }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (%d núcleos): Prime [%s] | Perf [%s]",
                  t.isOryon ? "Oryon" : "ARM", t.cpuCount, prime, perf);
    return std::string(buf);
}

}  // namespace hyperplay
