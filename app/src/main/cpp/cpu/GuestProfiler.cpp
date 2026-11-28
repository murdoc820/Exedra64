#include "GuestProfiler.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <sys/system_properties.h>
#define PLOG(...) profLog(__VA_ARGS__)
#else
#define PLOG(...) (std::fprintf(stderr, __VA_ARGS__), std::fputc('\n', stderr))
#endif

namespace hyperplay::prof {
namespace {

std::string g_reportPath;  // <juego>/_HyperPlaySaves/profile.txt
std::FILE* g_reportFile = nullptr;

// Escribe en logcat y en el archivo de reporte (logcat pierde líneas con mucho tráfico).
__attribute__((format(printf, 1, 2))) void profLog(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
#if defined(__ANDROID__)
    __android_log_write(ANDROID_LOG_INFO, "HyperPlay.Prof", buf);
#endif
    if (g_reportFile == nullptr && !g_reportPath.empty()) g_reportFile = std::fopen(g_reportPath.c_str(), "w");
    if (g_reportFile != nullptr) {
        std::fputs(buf, g_reportFile);
        std::fputc('\n', g_reportFile);
        std::fflush(g_reportFile);
    }
}

constexpr size_t kMaxSamples = 1u << 21;  // ~35 min de CPU a 1 kHz
std::atomic<uint64_t>* g_samples = nullptr;
std::atomic<uint32_t>* g_tids = nullptr;  // hilo de cada muestra
std::atomic<size_t> g_sampleCount{0};
std::atomic<bool> g_sampling{false};

struct BlockRange {
    uintptr_t hostStart;
    uint32_t hostSize;
    uintptr_t guestRip;
};
std::mutex g_blocksMutex;
std::vector<BlockRange>* g_blocks = nullptr;

int requestedSeconds() {
#if defined(__ANDROID__)
    char v[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.hyperplay.prof", v) > 0) return std::atoi(v);
#endif
    return 0;
}

std::atomic<uint64_t> g_overruns{0};
std::atomic<pid_t> g_mainTid{0};  // hilo principal del juego (lo fija loader/FramePacing)

void onSigprof(int, siginfo_t* si, void* uc) {
    if (!g_sampling.load(std::memory_order_relaxed)) return;
    if (si != nullptr && si->si_overrun > 0) g_overruns.fetch_add(static_cast<uint64_t>(si->si_overrun), std::memory_order_relaxed);
    uint64_t pc = 0;
#if defined(__aarch64__)
    pc = static_cast<ucontext_t*>(uc)->uc_mcontext.pc;
#else
    (void)uc;
#endif
    const size_t i = g_sampleCount.fetch_add(1, std::memory_order_relaxed);
    if (i < kMaxSamples) {
        g_samples[i].store(pc, std::memory_order_relaxed);
        g_tids[i].store(static_cast<uint32_t>(gettid()), std::memory_order_relaxed);  // async-signal-safe
    }
}

void dumpReport(int seconds) {
    g_sampling.store(false);
    timespec cpu{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
    PLOG("CPU del proceso: %.1f s en %d s de reloj (%.2f núcleos de media); expiraciones fusionadas=%llu",
         static_cast<double>(cpu.tv_sec) + cpu.tv_nsec / 1e9, seconds,
         (static_cast<double>(cpu.tv_sec) + cpu.tv_nsec / 1e9) / seconds,
         static_cast<unsigned long long>(g_overruns.load()));
    const size_t n = std::min(g_sampleCount.load(), kMaxSamples);
    std::vector<BlockRange> blocks;
    {
        std::lock_guard<std::mutex> lock(g_blocksMutex);
        blocks = *g_blocks;
    }
    std::sort(blocks.begin(), blocks.end(),
              [](const BlockRange& a, const BlockRange& b) { return a.hostStart < b.hostStart; });

    std::unordered_map<uintptr_t, uint64_t> guestHits;  // por RIP de inicio de bloque
    std::unordered_map<uintptr_t, uint64_t> hostHits;   // PCs fuera de bloques JIT
    uint64_t inJit = 0, outJit = 0;
    for (size_t i = 0; i < n; ++i) {
        const uintptr_t pc = static_cast<uintptr_t>(g_samples[i].load(std::memory_order_relaxed));
        auto it = std::upper_bound(blocks.begin(), blocks.end(), pc,
                                   [](uintptr_t v, const BlockRange& b) { return v < b.hostStart; });
        if (it != blocks.begin()) {
            --it;
            if (pc < it->hostStart + it->hostSize) {
                ++guestHits[it->guestRip];
                ++inJit;
                continue;
            }
        }
        ++hostHits[pc];
        ++outJit;
    }
    // Nativos: agrupar por símbolo (dladdr fuera del manejador de señal).
    std::map<std::string, uint64_t> nativeBySym;
    for (const auto& [pc, c] : hostHits) {
        Dl_info info{};
        std::string key;
        if (pc != 0 && dladdr(reinterpret_cast<void*>(pc), &info) != 0) {
            const char* lib = info.dli_fname != nullptr ? std::strrchr(info.dli_fname, '/') : nullptr;
            key = std::string(lib != nullptr ? lib + 1 : (info.dli_fname ? info.dli_fname : "?")) + "!" +
                  (info.dli_sname != nullptr ? info.dli_sname : "?");
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "pc=0x%llx", static_cast<unsigned long long>(pc & ~0xFFFull));
            key = buf;
        }
        nativeBySym[key] += c;
    }
    PLOG("=== Perfil %d s: %zu muestras (1 ms de CPU c/u), JIT=%llu (%.1f%%) nativo=%llu, bloques registrados=%zu ===",
         seconds, n, static_cast<unsigned long long>(inJit), n ? 100.0 * static_cast<double>(inJit) / n : 0.0,
         static_cast<unsigned long long>(outJit), blocks.size());
    std::vector<std::pair<uint64_t, uintptr_t>> topG;
    topG.reserve(guestHits.size());
    for (const auto& [rip, c] : guestHits) topG.emplace_back(c, rip);
    std::sort(topG.rbegin(), topG.rend());
    for (size_t i = 0; i < topG.size() && i < 120; ++i) {
        PLOG("G %6llu %5.2f%% rip=0x%llx", static_cast<unsigned long long>(topG[i].first),
             100.0 * static_cast<double>(topG[i].first) / static_cast<double>(n),
             static_cast<unsigned long long>(topG[i].second));
    }
    std::vector<std::pair<uint64_t, std::string>> topN;
    for (const auto& [sym, c] : nativeBySym) topN.emplace_back(c, sym);
    std::sort(topN.rbegin(), topN.rend());
    for (size_t i = 0; i < topN.size() && i < 60; ++i) {
        PLOG("N %6llu %5.2f%% %s", static_cast<unsigned long long>(topN[i].first),
             100.0 * static_cast<double>(topN[i].first) / static_cast<double>(n), topN[i].second.c_str());
    }
    // Por hilo: el principal del juego suele ser el cuello de botella (FPS) aunque haya otros
    // hilos ocupados (mezclador de audio, workers).
    std::unordered_map<uint32_t, uint64_t> byTid;
    for (size_t i = 0; i < n; ++i) ++byTid[g_tids[i].load(std::memory_order_relaxed)];
    std::vector<std::pair<uint64_t, uint32_t>> topT;
    for (const auto& [tid, c] : byTid) topT.emplace_back(c, tid);
    std::sort(topT.rbegin(), topT.rend());
    for (size_t i = 0; i < topT.size() && i < 8; ++i) {
        PLOG("T %6llu %5.2f%% tid=%u", static_cast<unsigned long long>(topT[i].first),
             100.0 * static_cast<double>(topT[i].first) / static_cast<double>(n), topT[i].second);
    }
    for (size_t t = 0; t < topT.size() && t < 4; ++t) {
        const uint32_t tid = topT[t].second;
        std::unordered_map<uintptr_t, uint64_t> g, h;
        for (size_t i = 0; i < n; ++i) {
            if (g_tids[i].load(std::memory_order_relaxed) != tid) continue;
            const uintptr_t pc = static_cast<uintptr_t>(g_samples[i].load(std::memory_order_relaxed));
            auto it = std::upper_bound(blocks.begin(), blocks.end(), pc,
                                       [](uintptr_t v, const BlockRange& b) { return v < b.hostStart; });
            if (it != blocks.begin() && pc < (it - 1)->hostStart + (it - 1)->hostSize) ++g[(it - 1)->guestRip];
            else ++h[pc];
        }
        std::vector<std::pair<uint64_t, uintptr_t>> tg(g.size());
        size_t k = 0;
        for (const auto& [rip, c] : g) tg[k++] = {c, rip};
        std::sort(tg.rbegin(), tg.rend());
        std::map<std::string, uint64_t> hs;
        for (const auto& [pc, c] : h) {
            Dl_info info{};
            const bool found = pc != 0 && dladdr(reinterpret_cast<void*>(pc), &info) != 0;
            if (found && info.dli_sname != nullptr) {
                hs[info.dli_sname] += c;
            } else if (found && info.dli_fbase != nullptr) {
                // Sin símbolo exportado (funciones internas): desplazamiento en la librería, para
                // `llvm-symbolizer --obj=libhyperplay.so 0x...` con la versión sin recortar.
                char key[96];
                const char* lib = info.dli_fname ? std::strrchr(info.dli_fname, '/') : nullptr;
                std::snprintf(key, sizeof(key), "%s+0x%llx", lib ? lib + 1 : "?",
                              static_cast<unsigned long long>((pc - reinterpret_cast<uintptr_t>(info.dli_fbase)) & ~uintptr_t{0x3F}));
                hs[key] += c;
            } else {
                hs["?"] += c;
            }
        }
        std::vector<std::pair<uint64_t, std::string>> th;
        for (const auto& [s, c] : hs) th.emplace_back(c, s);
        std::sort(th.rbegin(), th.rend());
        PLOG("=== hilo %u: %llu muestras ===", tid, static_cast<unsigned long long>(topT[t].first));
        for (size_t i = 0; i < tg.size() && i < 60; ++i) {
            PLOG("tG %6llu %5.2f%% rip=0x%llx", static_cast<unsigned long long>(tg[i].first),
                 100.0 * static_cast<double>(tg[i].first) / static_cast<double>(topT[t].first),
                 static_cast<unsigned long long>(tg[i].second));
        }
        for (size_t i = 0; i < th.size() && i < 20; ++i) {
            PLOG("tN %6llu %5.2f%% %s", static_cast<unsigned long long>(th[i].first),
                 100.0 * static_cast<double>(th[i].first) / static_cast<double>(topT[t].first), th[i].second.c_str());
        }
    }
    PLOG("=== fin del perfil ===");
}

}  // namespace

void setMainThread(pid_t tid) noexcept { g_mainTid.store(tid); }

bool enabled() noexcept {
    static const bool on = requestedSeconds() > 0;
    return on;
}

void startIfRequested(const std::string& savesDir) {
    static std::once_flag once;
    std::call_once(once, [&savesDir] {
        if (!savesDir.empty()) g_reportPath = savesDir + "/profile.txt";
        const int seconds = requestedSeconds();
        PLOG("startIfRequested: debug.hyperplay.prof=%d", seconds);
        g_samples = new std::atomic<uint64_t>[kMaxSamples];
        g_tids = new std::atomic<uint32_t>[kMaxSamples];
        {
            std::lock_guard<std::mutex> lock(g_blocksMutex);
            if (g_blocks == nullptr) {
                g_blocks = new std::vector<BlockRange>();
                g_blocks->reserve(1u << 20);
            }
        }
        struct sigaction sa {};
        sa.sa_sigaction = &onSigprof;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPROF, &sa, nullptr);
        timer_t timer{};
        sigevent sev{};
        sev.sigev_notify = SIGEV_SIGNAL;
        sev.sigev_signo = SIGPROF;
        if (timer_create(CLOCK_PROCESS_CPUTIME_ID, &sev, &timer) != 0) {
            PLOG("timer_create falló: %s", std::strerror(errno));
            return;
        }
        const auto run = [timer](int secs, int delay) {
            if (delay > 0) std::this_thread::sleep_for(std::chrono::seconds(delay));
            g_sampleCount.store(0);
            g_overruns.store(0);
            // `debug.hyperplay.proftid main|<tid>`: muestrea solo ese hilo con su propio reloj de
            // CPU y la señal dirigida a él. Con el timer del proceso las expiraciones se fusionan
            // (6 000 muestras para 180 s de CPU) y el hilo principal queda mal representado.
            timer_t active = timer;
            timer_t threadTimer{};
            bool perThread = false;
            char tidText[PROP_VALUE_MAX] = {};
            if (__system_property_get("debug.hyperplay.proftid", tidText) > 0) {
                const pid_t tid = std::strcmp(tidText, "main") == 0 ? g_mainTid.load()
                                                                     : static_cast<pid_t>(std::atoi(tidText));
                if (tid > 0) {
                    sigevent tsev{};
                    tsev.sigev_notify = SIGEV_THREAD_ID;
                    tsev.sigev_signo = SIGPROF;
                    tsev._sigev_un._tid = tid;
                    const clockid_t clk = static_cast<clockid_t>((~static_cast<unsigned>(tid) << 3) | 6u);  // CPU del hilo
                    if (timer_create(clk, &tsev, &threadTimer) == 0) {
                        active = threadTimer;
                        perThread = true;
                        PLOG("Perfil por hilo: tid=%d", tid);
                    } else {
                        PLOG("timer por hilo falló (tid=%d): %s", tid, std::strerror(errno));
                    }
                }
            }
            itimerspec its{};
            its.it_interval.tv_nsec = 1000000;  // 1 ms de CPU
            its.it_value.tv_nsec = 1000000;
            g_sampling.store(true);
            timer_settime(active, 0, &its, nullptr);
            PLOG("Perfilador activo durante %d s tras %d s (SIGPROF cada 1 ms de CPU)", secs, delay);
            std::this_thread::sleep_for(std::chrono::seconds(secs));
            itimerspec off{};
            timer_settime(active, 0, &off, nullptr);
            if (perThread) timer_delete(threadTimer);
            dumpReport(secs);
        };
        // debug.hyperplay.profdelay <s>: espera antes de muestrear (p. ej. para medir ya en partida).
        char delayText[PROP_VALUE_MAX] = {};
        const int delay = __system_property_get("debug.hyperplay.profdelay", delayText) > 0 ? std::atoi(delayText) : 0;
        // En caliente: `setprop debug.hyperplay.profnow <s>` (cada valor nuevo lanza un perfil).
        std::thread([run, seconds, delay] {
            if (seconds > 0) run(seconds, delay);
            char last[PROP_VALUE_MAX] = {};
            __system_property_get("debug.hyperplay.profnow", last);
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                char now[PROP_VALUE_MAX] = {};
                __system_property_get("debug.hyperplay.profnow", now);
                if (std::strcmp(now, last) == 0) continue;
                std::strcpy(last, now);
                const int secs = std::atoi(now);
                if (secs > 0) run(std::min(secs, 120), 0);
            }
        }).detach();
    });
}

void registerBlock(uintptr_t hostStart, size_t hostSize, uintptr_t guestRip) {
    // Siempre: el perfil en caliente (profnow) necesita poder atribuir muestras a bloques.
    std::lock_guard<std::mutex> lock(g_blocksMutex);
    if (g_blocks == nullptr) {  // bloques traducidos antes de startIfRequested
        g_blocks = new std::vector<BlockRange>();
        g_blocks->reserve(1u << 20);
    }
    g_blocks->push_back({hostStart, static_cast<uint32_t>(hostSize), guestRip});
}

}  // namespace hyperplay::prof
