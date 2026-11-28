#include "NativeDllInterceptor.h"
#include "../cpu/GuestProfiler.h"
#include "../gfx/D3D11GpuRenderer.h"
#include "Win32DisplayInfo.h"
#include "WidescreenPatch.h"
#include "ManagedHost.h"
#include "VulkanPassthrough.h"
#include "gfx/PresentOptions.h"
#include "FramePacing.h"
#include "Win32KnownFolders.h"
#include "Win32PathUtils.h"
#include "Win32Registry.h"
#include "WinSafeHeap.h"
#include "D3D11ResourceModel.h"
#include "D3D11Sampler.h"
#include "DxbcShader.h"
#include "DxbcVm.h"
#include "SoftRaster.h"
#include "Win64ThunkAbi.h"
#include "Win32Arity.h"
#include "LowMem.h"
#include "Win64Context.h"
#include "Win32Context.h"
#include "D3D9Probe32.h"
#include "DirectInput32.h"
#include "DirectSound32.h"
#include "SteamEmu32.h"
#include "GuestSeh.h"
#include "gfx/GuestOpenGl.h"
#include "cpu/OryonEmitterBackend.h"
#include "cpu/OryonSiliconScheduler.h"
#include "gfx/DirectVulkanBridge.h"
#include "input/DirectXInputBridge.h"
#include "audio/DirectAAudioEngine.h"
#include "audio/FmodStudioHost.h"
#include "media/MediaFoundationBridge.h"

#include <android/log.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <unwind.h>
#include <fnmatch.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cwchar>
#include <cwctype>
#include <sys/statvfs.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

#define LOG_TAG "HyperPlay.DllInterceptor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace {

LoadedPeImage* s_activeImage = nullptr;
std::atomic<uint64_t> s_kernel32Calls{0};
std::atomic<uint64_t> s_ntdllCalls{0};
std::atomic<uint64_t> s_d3dCalls{0};
std::atomic<uint64_t> s_vulkanCalls{0};
std::atomic<uint64_t> s_xinputCalls{0};
std::atomic<uint64_t> s_xaudioCalls{0};
std::atomic<uint64_t> s_steamCalls{0};
std::atomic<uint64_t> s_otherCalls{0};

uintptr_t s_unityMemoryManagerAllocate = 0;
uintptr_t s_unityMemoryManagerReallocate = 0;
uintptr_t s_unityMemoryManagerDeallocate = 0;

std::mutex s_trampolineMutex;
std::unordered_set<uintptr_t> s_registeredTrampolines;
std::unordered_map<uintptr_t, std::string> s_trampolineNames;
std::atomic<uint64_t> s_trampolineGen{1};
std::atomic<uintptr_t> s_trampolineMin{UINTPTR_MAX};
std::atomic<uintptr_t> s_trampolineMax{0};

DirectVulkanBridge& directVulkanBridge() {
    return DirectVulkanBridge::shared();  // misma instancia que la superficie de Kotlin
}

void registerTrampolineAddress(void* ptr, const std::string& name = "") {
    if (ptr == nullptr) return;
    std::lock_guard<std::mutex> lock(s_trampolineMutex);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (s_registeredTrampolines.insert(addr).second) {
        uintptr_t mn = s_trampolineMin.load(std::memory_order_relaxed);
        while (addr < mn && !s_trampolineMin.compare_exchange_weak(mn, addr, std::memory_order_relaxed)) {}
        uintptr_t mx = s_trampolineMax.load(std::memory_order_relaxed);
        while (addr > mx && !s_trampolineMax.compare_exchange_weak(mx, addr, std::memory_order_relaxed)) {}
        s_trampolineGen.fetch_add(1, std::memory_order_release);
    }
    if (!name.empty()) {
        s_trampolineNames[addr] = name;
    }
}

std::string findTrampolineName(uintptr_t addr);
}  // namespace (se reabre abajo)

namespace w32arity {
namespace {
std::mutex& arityMutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<const void*, Arity>& arityMap() {
    static std::unordered_map<const void*, Arity> m;
    return m;
}
}  // namespace
void record(const void* fn, const Arity& a) {
    std::lock_guard<std::mutex> lk(arityMutex());
    arityMap()[fn] = a;
}
bool lookup(const void* fn, Arity& out) {
    std::lock_guard<std::mutex> lk(arityMutex());
    const auto it = arityMap().find(fn);
    if (it == arityMap().end()) return false;
    out = it->second;
    return true;
}
}  // namespace w32arity

// --- Alias bajo 4 GB de funciones nativas (invitados de 32 bits) ---------------------------------
// Una IAT de 32 bits solo guarda punteros de 4 bytes, y las funciones de libhyperplay.so viven por
// encima de 4 GB. Cada función recibe una dirección propia en una región baja (16 bytes por entrada,
// rellena con INT3: nunca se ejecuta, el traductor la reconoce como trampolín antes de traducirla) y
// dispatchThunk la traduce de vuelta a la función real.
namespace w32alias {
namespace {
std::mutex s_mtx;
std::unordered_map<uintptr_t, uintptr_t> s_hostToLow;
std::unordered_map<uintptr_t, uintptr_t> s_lowToHost;
uint8_t* s_region = nullptr;
size_t s_used = 0;
constexpr size_t kRegionSize = 1u << 20;  // 65536 entradas
constexpr size_t kEntry = 16;
alignas(16) OryonEmitterBackend::Fast32AliasEntry s_fast32Table[kRegionSize / kEntry]{};

uint8_t* region() {
    if (s_region != nullptr) return s_region;
    // Por debajo de 2 GB y lejos del arena del invitado (que empieza abajo): se prueba de arriba abajo.
    for (uintptr_t hint = 0x7F000000u; hint >= 0x40000000u; hint -= 0x01000000u) {
        void* p = mmap(reinterpret_cast<void*>(hint), kRegionSize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) continue;
        if (reinterpret_cast<uintptr_t>(p) + kRegionSize > 0xFFFFFFFFull) {
            munmap(p, kRegionSize);
            continue;
        }
        std::memset(p, 0xCC, kRegionSize);  // INT3
        mprotect(p, kRegionSize, PROT_READ);
        s_region = static_cast<uint8_t*>(p);
        OryonEmitterBackend::fast32AliasBase.store(reinterpret_cast<uintptr_t>(p));  // CALL indirectos rápidos
        OryonEmitterBackend::fast32Table.store(s_fast32Table);
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "Alias de 32 bits en %p (%zu entradas)", p, kRegionSize / kEntry);
        break;
    }
    return s_region;
}
}  // namespace

uintptr_t hostFor(uintptr_t addr) {
    std::lock_guard<std::mutex> lk(s_mtx);
    const auto it = s_lowToHost.find(addr);
    return it == s_lowToHost.end() ? addr : it->second;
}

bool isAlias(uintptr_t addr) {
    return s_region != nullptr && addr >= reinterpret_cast<uintptr_t>(s_region) &&
           addr < reinterpret_cast<uintptr_t>(s_region) + kRegionSize;
}

// Dirección < 4 GB para 'host' (la misma si ya lo está); 0 si no queda espacio.
uintptr_t lowFor(void* host, const std::string& name) {
    const auto h = reinterpret_cast<uintptr_t>(host);
    if (h <= 0xFFFFFFFFull) return h;
    uintptr_t low = 0;
    {
        std::lock_guard<std::mutex> lk(s_mtx);
        if (const auto it = s_hostToLow.find(h); it != s_hostToLow.end()) return it->second;
        uint8_t* r = region();
        if (r == nullptr || s_used + kEntry > kRegionSize) return 0;
        const size_t idx = s_used / kEntry;
        low = reinterpret_cast<uintptr_t>(r + s_used);
        s_used += kEntry;
        s_hostToLow[h] = low;
        s_lowToHost[low] = h;
        // Rellenar entrada directa para emitFastIndirect32 en llamadas D3D9 / Win32 no bloqueantes (stdcall <= 8 args).
        bool fastBlocking = false;
        const bool isFastD3d9 = (name.rfind("d3d9.dll!", 0) == 0 &&
                                 name != "d3d9.dll!IDirect3DDevice9#17" &&
                                 name != "d3d9.dll!IDirect3DSwapChain9#3");
        const bool isFastWin = OryonEmitterBackend::lookupFastNative(h, fastBlocking) && !fastBlocking;
        if (isFastD3d9 || isFastWin) {
            w32arity::Arity ar{};
            if (w32arity::lookup(host, ar) && !ar.variadic && ar.wideMask == 0 && ar.slots <= 8) {
                s_fast32Table[idx] = {h, ar.slots, 0, {}};
            }
        }
    }
    registerTrampolineAddress(reinterpret_cast<void*>(low), name.empty() ? findTrampolineName(h) : name);
    return low;
}
}  // namespace w32alias

// Llamada a código del invitado (definida en sehimpl).
namespace sehimpl {
uint64_t callGuest(uint64_t fn, const uint64_t* args, size_t argc);
}  // namespace sehimpl

// SEH de 32 bits (definido junto a RaiseException).
namespace seh32 {
int32_t hp_NtContinue32(const uint8_t* ctx, int32_t testAlert);
uint32_t hp_RtlUnwind32(uint32_t target, uint32_t targetIp, uint8_t* rec, uint32_t retVal);
}  // namespace seh32

namespace {
std::string findTrampolineName(uintptr_t addr) {
    std::lock_guard<std::mutex> lock(s_trampolineMutex);
    auto it = s_trampolineNames.find(addr);
    if (it != s_trampolineNames.end()) return it->second;
    return {};
}

struct WinEventObj {
    bool manualReset{false};
    std::atomic<bool> signaled{false};
    std::mutex mtx;
    std::condition_variable cv;
};
static std::mutex s_eventMutex;
static std::unordered_map<uintptr_t, std::shared_ptr<WinEventObj>> s_events;
static std::atomic<uintptr_t> s_nextEventHandle{0x1000};

struct WinCondVarObj {
    std::mutex mtx;
    std::condition_variable cv;
};
static std::mutex s_cvMutex;
static std::unordered_map<void*, std::shared_ptr<WinCondVarObj>> s_condVars;

static std::shared_ptr<WinCondVarObj> getOrCreateCv(void* cvPtr) {
    if (!cvPtr) return nullptr;
    std::lock_guard<std::mutex> lk(s_cvMutex);
    auto it = s_condVars.find(cvPtr);
    if (it != s_condVars.end()) return it->second;
    auto cv = std::make_shared<WinCondVarObj>();
    s_condVars[cvPtr] = cv;
    return cv;
}

// WaitOnAddress/WakeByAddress reales: el valor se lee de la memoria huésped y
// el despertar se sincroniza mediante un condvar por dirección. No es un stub.
struct WinWaitAddressObj {
    std::mutex mtx;
    std::condition_variable cv;
    uint64_t wakeGeneration{0};
};
static std::mutex s_waitAddressMutex;
static std::unordered_map<uintptr_t, std::shared_ptr<WinWaitAddressObj>> s_waitAddresses;

static std::shared_ptr<WinWaitAddressObj> getOrCreateWaitAddress(const void* address) {
    if (address == nullptr) return nullptr;
    const uintptr_t key = reinterpret_cast<uintptr_t>(address);
    std::lock_guard<std::mutex> lk(s_waitAddressMutex);
    auto it = s_waitAddresses.find(key);
    if (it != s_waitAddresses.end()) return it->second;
    auto obj = std::make_shared<WinWaitAddressObj>();
    s_waitAddresses[key] = obj;
    return obj;
}

static std::shared_ptr<WinWaitAddressObj> findWaitAddress(const void* address) {
    if (address == nullptr) return nullptr;
    std::lock_guard<std::mutex> lk(s_waitAddressMutex);
    auto it = s_waitAddresses.find(reinterpret_cast<uintptr_t>(address));
    return (it != s_waitAddresses.end()) ? it->second : nullptr;
}

static bool isGuestRangeReadable(const void* ptr, size_t size) {
    if (ptr == nullptr || size == 0) return false;
    const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (pageSize == 0) return false;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t first = addr & ~(static_cast<uintptr_t>(pageSize) - 1);
    const size_t span = static_cast<size_t>((addr + size - 1 - first) / pageSize) + 1;
    std::vector<unsigned char> resident(span);
    if (mincore(reinterpret_cast<void*>(first), span * pageSize, resident.data()) != 0) return false;
    for (unsigned char page : resident) {
        if ((page & 1) == 0) return false;
    }
    return true;
}

[[maybe_unused]] static bool readGuestScalar(const void* ptr, size_t size, uint64_t& out) {
    if (!isGuestRangeReadable(ptr, size)) return false;
    switch (size) {
        case 1: out = *static_cast<const uint8_t*>(ptr); return true;
        case 2: { uint16_t v = 0; std::memcpy(&v, ptr, sizeof(v)); out = v; return true; }
        case 4: { uint32_t v = 0; std::memcpy(&v, ptr, sizeof(v)); out = v; return true; }
        case 8: { uint64_t v = 0; std::memcpy(&v, ptr, sizeof(v)); out = v; return true; }
        default: return false;
    }
}

static uint64_t loadGuestScalarAtomic(const void* ptr, size_t size) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (static_cast<uintptr_t>(size) - 1)) == 0) {
        switch (size) {
            case 1: return __atomic_load_n(static_cast<const uint8_t*>(ptr), __ATOMIC_SEQ_CST);
            case 2: return __atomic_load_n(static_cast<const uint16_t*>(ptr), __ATOMIC_SEQ_CST);
            case 4: return __atomic_load_n(static_cast<const uint32_t*>(ptr), __ATOMIC_SEQ_CST);
            case 8: return __atomic_load_n(static_cast<const uint64_t*>(ptr), __ATOMIC_SEQ_CST);
            default: break;
        }
    }
    uint64_t value = 0;
    std::memcpy(&value, ptr, size);
    return value;
}

struct WinFileMappingObj {
    int fd{-1};
    uint64_t maxSize{0};
    uint32_t flProtect{0};
    std::string name;
};
static std::mutex s_mappingMutex;
static std::unordered_map<uintptr_t, std::shared_ptr<WinFileMappingObj>> s_fileMappings;
static std::unordered_map<std::string, std::shared_ptr<WinFileMappingObj>> s_namedMappings;
static std::atomic<uintptr_t> s_nextMappingHandle{0xB000};

struct WinMapViewObj {
    void* addr{nullptr};
    size_t size{0};
};
static std::mutex s_viewMutex;
static std::unordered_map<uintptr_t, WinMapViewObj> s_mapViews;

struct WinSemaphoreObj {
    std::mutex mtx;
    std::condition_variable cv;
    int32_t count{0};
    int32_t maxCount{0x7FFFFFFF};
    // Mutex (CreateMutex): tiene dueño y es recursivo para él, como en Windows. Antes era un semáforo binario y
    // un hilo que volvía a pedir su propio mutex se bloqueaba (Bink en DmC al empezar los créditos).
    bool isMutex = false;
    int ownerTid = 0;
    int32_t recursion = 0;
    bool available(int tid) const { return count > 0 || (isMutex && ownerTid == tid); }
    void take(int tid) {
        if (!isMutex) { --count; return; }
        if (ownerTid == tid) { ++recursion; return; }
        ownerTid = tid;
        recursion = 1;
        count = 0;
    }
};
static std::mutex s_semMutex;
static std::unordered_map<uintptr_t, std::shared_ptr<WinSemaphoreObj>> s_semaphores;
static std::atomic<uintptr_t> s_nextSemHandle{0x5000};

struct WinThreadObj {
    pthread_t pt{};
    std::atomic<int> tid{0};  // gettid() del hilo (lo publica guestThreadTrampoline)
    std::atomic<bool> finished{false};
    uint32_t exitCode{0};
    std::mutex mtx;
    std::condition_variable cv;
};
static std::mutex s_threadMutex;
static std::unordered_map<uintptr_t, std::shared_ptr<WinThreadObj>> s_threads;
static std::atomic<uintptr_t> s_nextThreadHandle{0x9000};

struct GuestThreadArgs {
    void* lpStartAddress{nullptr};
    void* lpParameter{nullptr};
    size_t stackSize{0};
    std::shared_ptr<WinThreadObj> obj;
};

static void* guestThreadTrampoline(void* arg) {
    auto* info = static_cast<GuestThreadArgs*>(arg);
    {
        std::lock_guard<std::mutex> lk(info->obj->mtx);
        info->obj->tid.store(static_cast<int>(gettid()));
        info->obj->cv.notify_all();
    }
    auto active = OryonEmitterBackend::activeBackend();
    if (active != nullptr) {
        auto threadBackend = std::make_unique<OryonEmitterBackend>();
        EmitterConfig cfg;
        cfg.is32Bit = active->is32Bit();
        if (info->stackSize > 0) cfg.stackSize = info->stackSize;
        threadBackend->configure(cfg);
        for (const auto& r : active->codeRanges()) {
            threadBackend->addCodeRange(r.base, r.size);
        }
        // El user del thunk es el backend del propio hilo: dispatchThunk valida los argumentos
        // de pila (arg4..arg11) contra la pila virtual de ESTE hilo, no la del hilo principal.
        threadBackend->setNativeThunkHandler(active->nativeThunkHandler(), threadBackend.get());
        threadBackend->setAotCache(active->aotCache());
        threadBackend->allocateTeb();
        threadBackend->cloneTlsFrom(*active);

        OryonSiliconScheduler scheduler;
        static_cast<void>(scheduler.pinToPerfCores(0));

        LOGI("Oryon: Iniciando hilo invitado en %p (param=%p)", info->lpStartAddress, info->lpParameter);
        int64_t res = threadBackend->executeThread(info->lpStartAddress, info->lpParameter);
        LOGI("Oryon: Hilo invitado en %p finalizó con código %lld", info->lpStartAddress, static_cast<long long>(res));
        info->obj->exitCode = static_cast<uint32_t>(res);
    } else {
        LOGE("Oryon: CreateThread sin activeBackend disponible!");
        info->obj->exitCode = static_cast<uint32_t>(-1);
    }
    {
        std::lock_guard<std::mutex> lk(info->obj->mtx);
        info->obj->finished.store(true);
        info->obj->cv.notify_all();
    }
    delete info;
    return nullptr;
}

// Backend que despacha el thunk en curso en este hilo (para llamar callbacks invitados).
thread_local OryonEmitterBackend* t_dispatchBackend = nullptr;

// SEH: durante un thunk, contexto del llamador (RSP ya sin la dirección de retorno) y la propia
// dirección de retorno; RtlCaptureContext/RaiseException parten de ahí. Un thunk que "no vuelve"
// (RtlRestoreContext, RtlUnwindEx) deja en t_ctxOverride el contexto con el que seguir.
thread_local CpuContext* t_thunkCtx = nullptr;
thread_local uint64_t t_thunkRet = 0;
thread_local bool t_ctxOverride = false;
thread_local CpuContext t_ctxOverrideValue{};

// kernel32!ExitThread: termina el hilo invitado actual al volver del thunk (dispatchThunk pone
// RIP=0). Antes era un stub que volvía y el código seguía tras la llamada "sin retorno": Mono
// (mono_threads_platform_exit = detach + ExitThread) liberaba dos veces y abortaba (refcount.h).
thread_local bool t_exitThreadRequested = false;
thread_local uint32_t t_exitThreadCode = 0;
static void hp_ExitThread(uint32_t code) {
    t_exitThreadRequested = true;
    t_exitThreadCode = code;
}
// true mientras se resuelven los imports de una DLL local (no del exe): ver resolveNativeSymbol.
thread_local bool t_resolvingLocalDllImports = false;
static void* resolveSteamOffline(const char* name);

// VirtualAlloc/VirtualFree: Windows libera con MEM_RELEASE sin tamaño; se recuerda el tamaño real.
std::mutex s_allocMutex;
std::unordered_map<uintptr_t, size_t> s_allocations;

// WinSafeHeap (loader/WinSafeHeap.h): clases de tamaño sobre arenas de 64 MB, aislado de Scudo.
static void* safeHeapAlloc(size_t userBytes) {
    void* p = winheap::processHeap().alloc(userBytes);
    if (p == nullptr) LOGE("WinSafeHeap: sin memoria para %zu bytes", userBytes);
    return p;
}

static bool safeHeapFree(void* ptr) {
    if (winheap::processHeap().free(ptr)) return true;
    LOGW("WinSafeHeap: free de %p ajeno o ya liberado (ignorado)", ptr);
    return false;
}

static size_t safeHeapSize(const void* ptr) {
    return winheap::processHeap().size(ptr);
}

static bool isSafeHeapPointer(const void* ptr) {
    return winheap::processHeap().owns(ptr);
}

size_t s_pageSize() {
    const long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? static_cast<size_t>(ps) : 4096;
}

// Proteccion Win32 por pagina: VirtualProtect debe devolver la proteccion anterior real y
// VirtualFree debe limpiar las entradas para no devolver valores de una asignacion ya liberada.
std::mutex s_protMutex;
std::unordered_map<uintptr_t, uint32_t> s_pageProt;

// Hook SMC: protección POSIX de una página de VirtualAlloc/VirtualProtect, o -1.
int smcWritableProt(uintptr_t page) {
    std::lock_guard<std::mutex> lock(s_protMutex);
    const auto it = s_pageProt.find(page);
    if (it == s_pageProt.end()) return -1;
    switch (it->second & 0xFFu) {
        case 0x04: case 0x08: return PROT_READ | PROT_WRITE;                 // READWRITE / WRITECOPY
        case 0x40: case 0x80: return PROT_READ | PROT_WRITE | PROT_EXEC;     // EXECUTE_READWRITE / WRITECOPY
        default: return -1;                                                  // no escribible
    }
}

struct SmcHookInstaller {
    SmcHookInstaller() { OryonEmitterBackend::setSmcWritableHook(&smcWritableProt); }
} s_smcHookInstaller;

uint32_t queryPageProt(uintptr_t addr) {
    const size_t ps = s_pageSize();
    const uintptr_t page = addr & ~(static_cast<uintptr_t>(ps) - 1);
    std::lock_guard<std::mutex> lock(s_protMutex);
    const auto it = s_pageProt.find(page);
    return it != s_pageProt.end() ? it->second : 0x04u;  // PAGE_READWRITE
}

void recordPageProt(uintptr_t addr, size_t len, uint32_t prot) {
    if (len == 0) return;
    const size_t ps = s_pageSize();
    const uintptr_t mask = static_cast<uintptr_t>(ps) - 1;
    const uintptr_t first = addr & ~mask;
    if (len > std::numeric_limits<uintptr_t>::max() - addr) {
        len = std::numeric_limits<uintptr_t>::max() - addr;
    }
    uintptr_t end = addr + len;
    if (end > std::numeric_limits<uintptr_t>::max() - mask) {
        end = std::numeric_limits<uintptr_t>::max();
    } else {
        end = (end + mask) & ~mask;
    }
    std::lock_guard<std::mutex> lock(s_protMutex);
    for (uintptr_t page = first; page < end; page += ps) s_pageProt[page] = prot;
}

void erasePageProt(uintptr_t addr, size_t len) {
    if (len == 0) return;
    const size_t ps = s_pageSize();
    const uintptr_t mask = static_cast<uintptr_t>(ps) - 1;
    const uintptr_t first = addr & ~mask;
    if (len > std::numeric_limits<uintptr_t>::max() - addr) {
        len = std::numeric_limits<uintptr_t>::max() - addr;
    }
    uintptr_t end = addr + len;
    if (end > std::numeric_limits<uintptr_t>::max() - mask) {
        end = std::numeric_limits<uintptr_t>::max();
    } else {
        end = (end + mask) & ~mask;
    }
    std::lock_guard<std::mutex> lock(s_protMutex);
    for (uintptr_t page = first; page < end; page += ps) s_pageProt.erase(page);
}

// Protección de la sección que contiene `rva` (la IAT suele estar en una sección de solo lectura).
int protForRva(const LoadedPeImage& image, uint32_t rva) {
    const uint64_t target = rva;
    const uintptr_t base = reinterpret_cast<uintptr_t>(image.baseAddress);
    for (const auto& sec : image.sections) {
        const uintptr_t secAddr = reinterpret_cast<uintptr_t>(sec.virtualAddress);
        const uint64_t start = secAddr >= base ? static_cast<uint64_t>(secAddr - base) : 0;
        if (target >= start && target < start + sec.size) {
            int prot = PROT_READ;
            if ((sec.characteristics & 0x80000000u) != 0) prot |= PROT_WRITE;
            if ((sec.characteristics & 0x20000000u) != 0) prot |= PROT_EXEC;
            return prot;
        }
    }
    return PROT_READ;
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string utf16ToUtf8(const char16_t* wstr) {
    if (!wstr) return "";
    std::string out;
    while (*wstr) {
        char16_t wc = *wstr++;
        if (wc < 0x80) {
            out.push_back(static_cast<char>(wc));
        } else if (wc < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (wc >> 6)));
            out.push_back(static_cast<char>(0x80 | (wc & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (wc >> 12)));
            out.push_back(static_cast<char>(0x80 | ((wc >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (wc & 0x3F)));
        }
    }
    return out;
}

std::string wcharToUtf8(const wchar_t* wstr) {
    // En Win64 huésped, wchar_t es UTF-16LE (2 bytes), no 4 bytes como en Linux/NDK.
    return utf16ToUtf8(reinterpret_cast<const char16_t*>(wstr));
}

// Operaciones de archivo Win32 (MoveFileEx/DeleteFile/CopyFile/ReplaceFile/RemoveDirectory).
// Antes eran stubs que devolvían 0: Hollow Knight guarda en "user1.dat.new" y lo renombra con
// MoveFileExW(REPLACE_EXISTING); el renombrado fallaba y las partidas no persistían.
namespace winfile {
static std::string hostPath(const void* w, bool wide) {
    const std::string raw = wide ? utf16ToUtf8(static_cast<const char16_t*>(w)) : std::string(static_cast<const char*>(w));
    return s_activeImage != nullptr ? BionicPeLoader::resolveVirtualPath(*s_activeImage, raw) : raw;
}
static uint32_t winError(int e) {
    switch (e) {
        case ENOENT: return 2;      // ERROR_FILE_NOT_FOUND
        case ENOTDIR: return 3;     // ERROR_PATH_NOT_FOUND
        case EACCES: case EPERM: case EROFS: return 5;  // ERROR_ACCESS_DENIED
        case EEXIST: return 183;    // ERROR_ALREADY_EXISTS
        case ENOTEMPTY: return 145; // ERROR_DIR_NOT_EMPTY
        case ENOSPC: return 112;    // ERROR_DISK_FULL
        case EBUSY: return 32;      // ERROR_SHARING_VIOLATION
        default: return 31;         // ERROR_GEN_FAILURE
    }
}
static int32_t fail(int e) {
    NativeDllInterceptor::win_SetLastError(winError(e));
    return 0;
}
static bool exists(const std::string& p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0;
}
static bool copyFile(const std::string& from, const std::string& to, bool failIfExists) {
    const int in = open(from.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) return false;
    const int out = open(to.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | (failIfExists ? O_EXCL : O_TRUNC), 0660);
    if (out < 0) {
        const int e = errno;
        close(in);
        errno = e;
        return false;
    }
    std::vector<char> buf(65536);  // en el heap: la pila host del juego puede ir muy justa
    bool ok = true;
    for (;;) {
        const ssize_t n = read(in, buf.data(), buf.size());
        if (n == 0) break;
        if (n < 0 || write(out, buf.data(), static_cast<size_t>(n)) != n) {
            ok = false;
            break;
        }
    }
    const int e = errno;
    close(in);
    if (close(out) != 0) ok = false;
    errno = e;
    return ok;
}
static int32_t move(const std::string& from, const std::string& to, uint32_t flags) {
    constexpr uint32_t kReplace = 0x1, kCopyAllowed = 0x2;
    if (!exists(from)) return fail(ENOENT);
    if (!(flags & kReplace) && exists(to)) return fail(EEXIST);
    if (rename(from.c_str(), to.c_str()) == 0) return 1;
    if (errno == EXDEV && (flags & kCopyAllowed)) {
        if (!copyFile(from, to, false)) return fail(errno);
        unlink(from.c_str());
        return 1;
    }
    return fail(errno);
}
// Rutas fuera de la carpeta del ejecutable (instalación alcanzada con ..\): nunca se borran,
// mueven ni reemplazan. Devuelve true (y ERROR_ACCESS_DENIED) si hay que negar la operación.
static bool deniedEscaped(const std::string& real, const char* op) {
    if (!BionicPeLoader::lastPathEscapedGameDir()) return false;
    LOGW("%s denegado fuera de la carpeta del juego: %s", op, real.c_str());
    NativeDllInterceptor::win_SetLastError(5);
    return true;
}
static int32_t MoveFileExW(const void* a, const void* b, uint32_t flags) {
    if (a == nullptr) return fail(ENOENT);
    if (b == nullptr) {  // destino nulo: borrar (DELAY_UNTIL_REBOOT no aplica)
        const std::string del = hostPath(a, true);
        if (deniedEscaped(del, "MoveFileExW")) return 0;
        return unlink(del.c_str()) == 0 ? 1 : fail(errno);
    }
    const std::string from = hostPath(a, true);
    if (deniedEscaped(from, "MoveFileExW")) return 0;
    const std::string to = hostPath(b, true);
    if (deniedEscaped(to, "MoveFileExW")) return 0;
    const int32_t r = move(from, to, flags);
    LOGI("MoveFileExW('%s' -> '%s', 0x%x) -> %d", from.c_str(), to.c_str(), flags, r);
    return r;
}
static int32_t MoveFileExA(const void* a, const void* b, uint32_t flags) {
    if (a == nullptr || b == nullptr) return fail(ENOENT);
    const std::string from = hostPath(a, false);
    if (deniedEscaped(from, "MoveFileExA")) return 0;
    const std::string to = hostPath(b, false);
    if (deniedEscaped(to, "MoveFileExA")) return 0;
    return move(from, to, flags);
}
static int32_t MoveFileW(const void* a, const void* b) { return MoveFileExW(a, b, 0x2); }
static int32_t MoveFileA(const void* a, const void* b) { return MoveFileExA(a, b, 0x2); }
static int32_t DeleteFileW(const void* a) {
    if (a == nullptr) return fail(ENOENT);
    const std::string p = hostPath(a, true);
    if (deniedEscaped(p, "DeleteFileW")) return 0;
    return unlink(p.c_str()) == 0 ? 1 : fail(errno);
}
static int32_t DeleteFileA(const void* a) {
    if (a == nullptr) return fail(ENOENT);
    const std::string p = hostPath(a, false);
    if (deniedEscaped(p, "DeleteFileA")) return 0;
    return unlink(p.c_str()) == 0 ? 1 : fail(errno);
}
static int32_t RemoveDirectoryW(const void* a) {
    if (a == nullptr) return fail(ENOENT);
    const std::string p = hostPath(a, true);
    if (deniedEscaped(p, "RemoveDirectoryW")) return 0;
    return rmdir(p.c_str()) == 0 ? 1 : fail(errno);
}
static int32_t RemoveDirectoryA(const void* a) {
    if (a == nullptr) return fail(ENOENT);
    const std::string p = hostPath(a, false);
    if (deniedEscaped(p, "RemoveDirectoryA")) return 0;
    return rmdir(p.c_str()) == 0 ? 1 : fail(errno);
}
static int32_t CopyFileW(const void* a, const void* b, int32_t failIfExists) {
    if (a == nullptr || b == nullptr) return fail(ENOENT);
    return copyFile(hostPath(a, true), hostPath(b, true), failIfExists != 0) ? 1 : fail(errno);
}
static int32_t CopyFileA(const void* a, const void* b, int32_t failIfExists) {
    if (a == nullptr || b == nullptr) return fail(ENOENT);
    return copyFile(hostPath(a, false), hostPath(b, false), failIfExists != 0) ? 1 : fail(errno);
}
// CopyFileExW(src, dst, progress, data, cancel, flags): COPY_FILE_FAIL_IF_EXISTS = 1.
static int32_t CopyFileExW(const void* a, const void* b, void*, void*, void*, uint32_t flags) {
    return CopyFileW(a, b, (flags & 1u) ? 1 : 0);
}
// ReplaceFileW(reemplazado, reemplazo, copia, flags, ...): copia de seguridad opcional y
// después el reemplazo ocupa el nombre del reemplazado.
static int32_t ReplaceFileW(const void* replaced, const void* replacement, const void* backup, uint32_t,
                            void*, void*) {
    if (replaced == nullptr || replacement == nullptr) return fail(ENOENT);
    const std::string dst = hostPath(replaced, true);
    if (deniedEscaped(dst, "ReplaceFileW")) return 0;
    const std::string src = hostPath(replacement, true);
    if (deniedEscaped(src, "ReplaceFileW")) return 0;
    if (!exists(src)) {
        NativeDllInterceptor::win_SetLastError(1177);  // ERROR_UNABLE_TO_MOVE_REPLACEMENT
        return 0;
    }
    if (backup != nullptr && exists(dst)) {
        const std::string bak = hostPath(backup, true);
        if (rename(dst.c_str(), bak.c_str()) != 0 && !copyFile(dst, bak, false)) return fail(errno);
    }
    const int32_t r = rename(src.c_str(), dst.c_str()) == 0 ? 1 : fail(errno);
    LOGI("ReplaceFileW('%s' <- '%s') -> %d", dst.c_str(), src.c_str(), r);
    return r;
}

}  // namespace winfile
int fdFromHandle(void* handle);
namespace winfile {
// ntdll!NtQueryDirectoryFile: lo usa el enumerador de archivos de Mono/.NET (Directory.GetFiles,
// File.Replace con copias). Sin él, el guardado de Hollow Knight lanzaba EntryPointNotFoundException
// después de escribir user<N>.dat.new y nunca lo pasaba a user<N>.dat (la partida no persistía).
// El handle es un fd de CreateFileW sobre la carpeta; la enumeración vive en un DIR* por fd
// (validado por dispositivo+inodo, por si el fd se reutiliza tras CloseHandle).
namespace ntdir {
constexpr uint32_t kStatusSuccess = 0, kStatusNoMoreFiles = 0x80000006u, kStatusBufferOverflow = 0x80000005u,
                   kStatusInvalidHandle = 0xC0000008u, kStatusInvalidInfoClass = 0xC0000003u,
                   kStatusNoSuchFile = 0xC000000Fu, kStatusInvalidParameter = 0xC000000Du;
struct Enum {
    DIR* dir = nullptr;
    dev_t dev = 0;
    ino_t ino = 0;
    std::string pattern;  // filtro de la primera llamada (UTF-8, minúsculas)
    bool havePending = false;
    std::string pending;  // entrada leída que no cupo en la llamada anterior
};
std::mutex s_mutex;
std::unordered_map<int, Enum> s_enums;

uint64_t fileTime(const timespec& t) {
    return (static_cast<uint64_t>(t.tv_sec) + 11644473600ull) * 10000000ull + static_cast<uint64_t>(t.tv_nsec) / 100u;
}

bool wildMatch(const char* p, const char* s) {  // '*' y '?' (sin distinguir mayúsculas)
    if (*p == 0) return *s == 0;
    if (*p == '*') return wildMatch(p + 1, s) || (*s != 0 && wildMatch(p, s + 1));
    if (*s == 0) return false;
    if (*p == '?' || std::tolower(static_cast<unsigned char>(*p)) == std::tolower(static_cast<unsigned char>(*s)))
        return wildMatch(p + 1, s + 1);
    return false;
}

// Escribe una entrada; devuelve los bytes ocupados (alineados a 8) o 0 si no cabe.
size_t writeEntry(uint8_t* out, size_t room, uint32_t infoClass, int dirFd, const std::string& name) {
    size_t nameOff = 0;
    switch (infoClass) {
        case 1: nameOff = 64; break;   // FileDirectoryInformation
        case 2: nameOff = 68; break;   // FileFullDirectoryInformation
        case 3: nameOff = 94; break;   // FileBothDirectoryInformation
        case 37: nameOff = 104; break; // FileIdBothDirectoryInformation
        case 38: nameOff = 80; break;  // FileIdFullDirectoryInformation
        case 12: nameOff = 12; break;  // FileNamesInformation
        default: return 0;
    }
    std::u16string wname;
    for (size_t i = 0; i < name.size();) {  // UTF-8 -> UTF-16 (BMP + pares sustitutos)
        uint32_t c = static_cast<uint8_t>(name[i]);
        size_t len = 1;
        if (c >= 0xF0 && i + 3 < name.size() + 1) { c &= 0x07; len = 4; }
        else if (c >= 0xE0) { c &= 0x0F; len = 3; }
        else if (c >= 0xC0) { c &= 0x1F; len = 2; }
        for (size_t k = 1; k < len && i + k < name.size(); ++k) c = (c << 6) | (static_cast<uint8_t>(name[i + k]) & 0x3F);
        i += len;
        if (c >= 0x10000) {
            c -= 0x10000;
            wname.push_back(static_cast<char16_t>(0xD800 + (c >> 10)));
            wname.push_back(static_cast<char16_t>(0xDC00 + (c & 0x3FF)));
        } else {
            wname.push_back(static_cast<char16_t>(c));
        }
    }
    const size_t nameBytes = wname.size() * 2;
    const size_t total = (nameOff + nameBytes + 7) & ~size_t{7};
    if (nameOff + nameBytes > room) return 0;
    std::memset(out, 0, std::min(total, room));
    struct stat st{};
    const bool haveStat = fstatat(dirFd, name.c_str(), &st, 0) == 0;
    const bool isDir = haveStat && S_ISDIR(st.st_mode);
    const uint32_t attrs = isDir ? 0x10u : 0x80u;  // DIRECTORY / NORMAL
    const auto put32 = [&](size_t off, uint32_t v) { std::memcpy(out + off, &v, 4); };
    const auto put64 = [&](size_t off, uint64_t v) { std::memcpy(out + off, &v, 8); };
    if (infoClass == 12) {
        put32(8, static_cast<uint32_t>(nameBytes));
    } else {
        if (haveStat) {
            put64(8, fileTime(st.st_mtim));   // CreationTime (Android no da la de creación)
            put64(16, fileTime(st.st_atim));
            put64(24, fileTime(st.st_mtim));
            put64(32, fileTime(st.st_ctim));
            put64(40, isDir ? 0 : static_cast<uint64_t>(st.st_size));
            put64(48, isDir ? 0 : (static_cast<uint64_t>(st.st_size) + 4095) & ~uint64_t{4095});
        }
        put32(56, attrs);
        put32(60, static_cast<uint32_t>(nameBytes));
        if (infoClass == 37) put64(96, haveStat ? static_cast<uint64_t>(st.st_ino) : 0);  // FileId
        if (infoClass == 38) put64(72, haveStat ? static_cast<uint64_t>(st.st_ino) : 0);
    }
    std::memcpy(out + nameOff, wname.data(), nameBytes);
    return std::min(total, room);
}
}  // namespace ntdir

static uint32_t NtQueryDirectoryFile(void* handle, void*, void*, void*, void* ioStatus, void* info, uint32_t length,
                                     uint32_t infoClass, uint8_t singleEntry, void* fileName, uint8_t restart) {
    using namespace ntdir;
    const int fd = fdFromHandle(handle);
    const auto finish = [&](uint32_t status, uint64_t written) {
        if (ioStatus != nullptr) {
            auto* io = static_cast<uint64_t*>(ioStatus);
            io[0] = status;
            io[1] = written;
        }
        return status;
    };
    if (fd < 0) return finish(kStatusInvalidHandle, 0);
    if (info == nullptr || length == 0) return finish(kStatusInvalidParameter, 0);
    if (infoClass != 1 && infoClass != 2 && infoClass != 3 && infoClass != 12 && infoClass != 37 && infoClass != 38)
        return finish(kStatusInvalidInfoClass, 0);
    struct stat fst{};
    if (fstat(fd, &fst) != 0 || !S_ISDIR(fst.st_mode)) return finish(kStatusInvalidHandle, 0);

    std::lock_guard<std::mutex> lock(s_mutex);
    Enum& e = s_enums[fd];
    if (e.dir == nullptr || e.dev != fst.st_dev || e.ino != fst.st_ino) {
        if (e.dir != nullptr) closedir(e.dir);
        e = Enum{};
        const int dupFd = dup(fd);
        e.dir = dupFd >= 0 ? fdopendir(dupFd) : nullptr;
        if (e.dir == nullptr) {
            if (dupFd >= 0) close(dupFd);
            s_enums.erase(fd);
            return finish(kStatusInvalidHandle, 0);
        }
        e.dev = fst.st_dev;
        e.ino = fst.st_ino;
        restart = 1;
    }
    if (restart) {
        rewinddir(e.dir);
        e.havePending = false;
        e.pattern.clear();
        if (fileName != nullptr) {  // UNICODE_STRING {u16 Length, u16 Max, pad, PWSTR}
            const auto* us = static_cast<const uint8_t*>(fileName);
            uint16_t lenBytes = 0;
            std::memcpy(&lenBytes, us, 2);
            const char16_t* buf = nullptr;
            std::memcpy(&buf, us + 8, 8);
            if (buf != nullptr) {
                for (uint16_t i = 0; i < lenBytes / 2; ++i) e.pattern.push_back(buf[i] < 0x80 ? static_cast<char>(buf[i]) : '?');
            }
            if (e.pattern == "*" || e.pattern == "*.*") e.pattern.clear();
        }
    }
    auto* out = static_cast<uint8_t*>(info);
    size_t used = 0;
    uint8_t* lastEntry = nullptr;
    int count = 0;
    const int dirFd = dirfd(e.dir);
    for (;;) {
        std::string name;
        if (e.havePending) {
            name = e.pending;
            e.havePending = false;
        } else {
            dirent* d = readdir(e.dir);
            if (d == nullptr) break;
            name = d->d_name;
            if (!e.pattern.empty() && name != "." && name != ".." && !wildMatch(e.pattern.c_str(), name.c_str())) continue;
        }
        const size_t n = writeEntry(out + used, length - used, infoClass, dirFd, name);
        if (n == 0) {  // no cabe: se guarda para la siguiente llamada
            e.pending = name;
            e.havePending = true;
            if (count == 0) return finish(kStatusBufferOverflow, 0);
            break;
        }
        if (lastEntry != nullptr) {
            const uint32_t next = static_cast<uint32_t>((out + used) - lastEntry);
            std::memcpy(lastEntry, &next, 4);
        }
        lastEntry = out + used;
        used += n;
        ++count;
        if (singleEntry) break;
    }
    if (count == 0) return finish(restart && fileName != nullptr ? kStatusNoSuchFile : kStatusNoMoreFiles, 0);
    return finish(kStatusSuccess, used);
}
}  // namespace winfile

size_t guestWStringLength(std::string_view utf8) {
    size_t count = 0;
    for (size_t i = 0; i < utf8.size();) {
        uint32_t cp = 0;
        const auto c = static_cast<uint8_t>(utf8[i]);
        if (c < 0x80) { cp = c; i += 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < utf8.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(utf8[i + 1]) & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < utf8.size()) {
            cp = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(utf8[i + 1]) & 0x3F) << 6) |
                 (static_cast<uint8_t>(utf8[i + 2]) & 0x3F);
            i += 3;
        } else {
            cp = static_cast<uint8_t>(utf8[i++]);
        }
        count += (cp < 0x10000) ? 1 : 2;
    }
    return count;
}

size_t writeGuestWString(wchar_t* dest, size_t destCap, std::string_view utf8) {
    if (!dest || destCap == 0) return 0;
    auto* out = reinterpret_cast<uint16_t*>(dest);
    size_t written = 0;
    for (size_t i = 0; i < utf8.size() && written + 1 < destCap;) {
        uint32_t cp = 0;
        const auto c = static_cast<uint8_t>(utf8[i]);
        if (c < 0x80) { cp = c; i += 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < utf8.size()) {
            cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(utf8[i + 1]) & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < utf8.size()) {
            cp = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(utf8[i + 1]) & 0x3F) << 6) |
                 (static_cast<uint8_t>(utf8[i + 2]) & 0x3F);
            i += 3;
        } else {
            cp = static_cast<uint8_t>(utf8[i++]);
        }
        if (cp < 0x10000) {
            out[written++] = static_cast<uint16_t>(cp);
        } else if (written + 2 < destCap) {
            cp -= 0x10000;
            out[written++] = static_cast<uint16_t>(0xD800 + (cp >> 10));
            out[written++] = static_cast<uint16_t>(0xDC00 + (cp & 0x3FF));
        } else {
            break;
        }
    }
    out[written] = 0;
    return written;
}

int fdFromHandle(void* handle) {
    const uintptr_t val = reinterpret_cast<uintptr_t>(handle);
    if (val == 0 || val == reinterpret_cast<uintptr_t>(MAP_FAILED)) return -1;
    if (val == 0x10f5) return STDOUT_FILENO; // STD_OUTPUT_HANDLE
    if (val == 0x10f4) return STDERR_FILENO; // STD_ERROR_HANDLE
    if (val == 0x10f6) return STDIN_FILENO;  // STD_INPUT_HANDLE
    return static_cast<int>(val - 1);
}

void* handleFromFd(int fd) {
    if (fd < 0) return reinterpret_cast<void*>(~static_cast<uintptr_t>(0)); // INVALID_HANDLE_VALUE
    return reinterpret_cast<void*>(static_cast<uintptr_t>(fd + 1));
}

alignas(16) static char s_cmdA[512] = {};
alignas(16) static char16_t s_cmdW[512] = {};
alignas(16) static char16_t s_exeNameW[512] = {};

// ---------------------------------------------------------------------------------------------
// advapi32 Reg*: registro en memoria persistido en <Juego>/_HyperPlaySaves/registry.hpreg
// (PlayerPrefs de Unity viven en HKCU\Software\<Compañía>\<Producto>).
// ---------------------------------------------------------------------------------------------
win32reg::Registry& registry() {
    static win32reg::Registry s_reg;
    static std::once_flag s_once;
    std::call_once(s_once, [] {
        if (s_activeImage != nullptr && !s_activeImage->savesDir.empty()) {
            s_reg.setBackingFile(s_activeImage->savesDir + "/registry.hpreg");
        }
    });
    return s_reg;
}

// Datos de cadena de las variantes A: ANSI(UTF-8) <-> UTF-16 con NUL.
bool regIsString(uint32_t type) {
    return type == win32reg::kRegSz || type == win32reg::kRegExpandSz || type == win32reg::kRegMultiSz;
}

// HKEY* del invitado: 4 bytes en 32 bits (lowmem::storePtr).
template <class S>
uint32_t regOpenOut(uint64_t hKey, const S& sub, bool create, uint64_t* out, uint32_t* disp = nullptr) {
    uint64_t h = 0;
    const uint32_t r = registry().open(hKey, sub, create, out != nullptr ? &h : nullptr, disp);
    if (out != nullptr) lowmem::storePtr(out, h);
    return r;
}

int32_t win_RegOpenKeyExW(uint64_t hKey, const void* subKey, uint32_t, uint32_t, uint64_t* out) {
    const uint32_t r = regOpenOut(hKey, win32reg::fromGuestW(subKey), false, out);
    static std::atomic<uint32_t> s_logs{0};
    if (s_logs.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOGI("RegOpenKeyExW(0x%llx, '%s') -> %u", static_cast<unsigned long long>(hKey),
             win32reg::toUtf8(win32reg::fromGuestW(subKey)).c_str(), r);
    }
    return static_cast<int32_t>(r);
}

int32_t win_RegOpenKeyExA(uint64_t hKey, const char* subKey, uint32_t, uint32_t, uint64_t* out) {
    return static_cast<int32_t>(regOpenOut(hKey, win32reg::fromUtf8(subKey), false, out));
}

int32_t win_RegOpenKeyW(uint64_t hKey, const void* subKey, uint64_t* out) {
    return static_cast<int32_t>(regOpenOut(hKey, win32reg::fromGuestW(subKey), false, out));
}

int32_t win_RegCreateKeyExW(uint64_t hKey, const void* subKey, uint32_t, const void*, uint32_t,
                            uint32_t, const void*, uint64_t* out, uint32_t* disposition) {
    const uint32_t r = regOpenOut(hKey, win32reg::fromGuestW(subKey), true, out, disposition);
    static std::atomic<uint32_t> s_logs{0};
    if (s_logs.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOGI("RegCreateKeyExW(0x%llx, '%s') -> %u", static_cast<unsigned long long>(hKey),
             win32reg::toUtf8(win32reg::fromGuestW(subKey)).c_str(), r);
    }
    return static_cast<int32_t>(r);
}

int32_t win_RegCreateKeyExA(uint64_t hKey, const char* subKey, uint32_t, const char*, uint32_t,
                            uint32_t, const void*, uint64_t* out, uint32_t* disposition) {
    return static_cast<int32_t>(regOpenOut(hKey, win32reg::fromUtf8(subKey), true, out, disposition));
}

int32_t win_RegCreateKeyW(uint64_t hKey, const void* subKey, uint64_t* out) {
    return static_cast<int32_t>(regOpenOut(hKey, win32reg::fromGuestW(subKey), true, out));
}

int32_t win_RegCloseKey(uint64_t hKey) {
    return static_cast<int32_t>(registry().close(hKey));
}

int32_t win_RegFlushKey(uint64_t) {
    return 0;
}

int32_t win_RegSetValueExW(uint64_t hKey, const void* name, uint32_t, uint32_t type, const uint8_t* data,
                           uint32_t size) {
    return static_cast<int32_t>(registry().setValue(hKey, win32reg::fromGuestW(name), type, data, size));
}

int32_t win_RegSetValueExA(uint64_t hKey, const char* name, uint32_t, uint32_t type, const uint8_t* data,
                           uint32_t size) {
    if (regIsString(type) && data != nullptr) {
        std::string s(reinterpret_cast<const char*>(data), size);
        while (!s.empty() && s.back() == '\0') s.pop_back();
        std::u16string w = win32reg::fromUtf8(s.c_str());
        w.push_back(u'\0');
        return static_cast<int32_t>(registry().setValue(hKey, win32reg::fromUtf8(name), type,
                                                        reinterpret_cast<const uint8_t*>(w.data()),
                                                        w.size() * sizeof(char16_t)));
    }
    return static_cast<int32_t>(registry().setValue(hKey, win32reg::fromUtf8(name), type, data, size));
}

int32_t win_RegQueryValueExW(uint64_t hKey, const void* name, uint32_t*, uint32_t* type, uint8_t* data,
                             uint32_t* size) {
    return static_cast<int32_t>(registry().queryValue(hKey, win32reg::fromGuestW(name), type, data, size));
}

int32_t win_RegQueryValueExA(uint64_t hKey, const char* name, uint32_t*, uint32_t* type, uint8_t* data,
                             uint32_t* size) {
    uint32_t t = 0, need = 0;
    const std::u16string wname = win32reg::fromUtf8(name);
    uint32_t r = registry().queryValue(hKey, wname, &t, nullptr, &need);
    if (r != win32reg::kSuccess) return static_cast<int32_t>(r);
    if (type != nullptr) *type = t;
    if (!regIsString(t)) return static_cast<int32_t>(registry().queryValue(hKey, wname, nullptr, data, size));
    std::vector<uint8_t> raw(need);
    r = registry().queryValue(hKey, wname, nullptr, raw.data(), &need);
    if (r != win32reg::kSuccess) return static_cast<int32_t>(r);
    std::u16string w(raw.size() / sizeof(char16_t), u'\0');
    if (!w.empty()) std::memcpy(w.data(), raw.data(), w.size() * sizeof(char16_t));
    while (!w.empty() && w.back() == u'\0') w.pop_back();
    std::string a = win32reg::toUtf8(w);
    a.push_back('\0');
    const auto bytes = static_cast<uint32_t>(a.size());
    if (data != nullptr) {
        if (size == nullptr) return static_cast<int32_t>(win32reg::kInvalidParameter);
        if (*size < bytes) { *size = bytes; return static_cast<int32_t>(win32reg::kMoreData); }
        std::memcpy(data, a.data(), bytes);
    }
    if (size != nullptr) *size = bytes;
    return 0;
}

int32_t win_RegGetValueW(uint64_t hKey, const void* subKey, const void* value, uint32_t, uint32_t* type,
                         uint8_t* data, uint32_t* size) {
    uint64_t h = hKey;
    const std::u16string sub = win32reg::fromGuestW(subKey);
    if (!sub.empty()) {
        const uint32_t r = registry().open(hKey, sub, false, &h);
        if (r != win32reg::kSuccess) return static_cast<int32_t>(r);
    }
    const uint32_t r = registry().queryValue(h, win32reg::fromGuestW(value), type, data, size);
    if (h != hKey) registry().close(h);
    return static_cast<int32_t>(r);
}

int32_t win_RegDeleteValueW(uint64_t hKey, const void* name) {
    return static_cast<int32_t>(registry().deleteValue(hKey, win32reg::fromGuestW(name)));
}

int32_t win_RegDeleteValueA(uint64_t hKey, const char* name) {
    return static_cast<int32_t>(registry().deleteValue(hKey, win32reg::fromUtf8(name)));
}

int32_t win_RegDeleteKeyW(uint64_t hKey, const void* subKey) {
    return static_cast<int32_t>(registry().deleteKey(hKey, win32reg::fromGuestW(subKey)));
}

int32_t win_RegEnumValueW(uint64_t hKey, uint32_t index, char16_t* name, uint32_t* nameChars, uint32_t*,
                          uint32_t* type, uint8_t* data, uint32_t* size) {
    return static_cast<int32_t>(registry().enumValue(hKey, index, name, nameChars, type, data, size));
}

int32_t win_RegEnumKeyExW(uint64_t hKey, uint32_t index, char16_t* name, uint32_t* nameChars, uint32_t*,
                          void*, uint32_t*, void*) {
    return static_cast<int32_t>(registry().enumKey(hKey, index, name, nameChars));
}

int32_t win_RegQueryInfoKeyW(uint64_t hKey, void* cls, uint32_t* clsChars, uint32_t*, uint32_t* subKeys,
                             uint32_t* maxSubKeyLen, uint32_t* maxClassLen, uint32_t* values,
                             uint32_t* maxValueNameLen, uint32_t* maxValueLen, uint32_t* secDesc,
                             void* lastWrite) {
    if (cls != nullptr && clsChars != nullptr && *clsChars > 0) static_cast<char16_t*>(cls)[0] = 0;
    if (clsChars != nullptr) *clsChars = 0;
    if (maxClassLen != nullptr) *maxClassLen = 0;
    if (secDesc != nullptr) *secDesc = 0;
    if (lastWrite != nullptr) std::memset(lastWrite, 0, 8);
    return static_cast<int32_t>(registry().queryInfo(hKey, subKeys, maxSubKeyLen, values, maxValueNameLen,
                                                     maxValueLen));
}

// ---------------------------------------------------------------------------------------------
// Exportaciones de kernel32/ntdll que antes solo existían como stubs dinámicos. Con
// GetProcAddress devolviendo NULL para nombres desconocidos, hay que darlas de verdad: Mono
// (mini-windows.c) aserta si no encuentra ni RtlAddGrowableFunctionTable ni
// RtlInstallFunctionTableCallback. El desenrollado de pila lo hace el JIT, no Windows: las
// tablas de funciones solo se aceptan.
// ---------------------------------------------------------------------------------------------
// Desde el SEH propio (GuestSeh), las tablas se guardan de verdad: RtlLookupFunctionEntry las
// consulta para el código generado por los JIT (CoreCLR, Mono).
uint32_t win_RtlAddGrowableFunctionTable(void** dynamicTable, void* table, uint32_t count, uint32_t max,
                                         uint64_t rangeBase, uint64_t rangeEnd) {
    return seh::addGrowableTable(dynamicTable, static_cast<const seh::RuntimeFunction*>(table), count, max, rangeBase,
                                 rangeEnd);
}

void win_RtlGrowFunctionTable(void* table, uint32_t count) { seh::growTable(table, count); }

void win_RtlDeleteGrowableFunctionTable(void* table) { seh::deleteGrowableTable(table); }

int32_t win_RtlInstallFunctionTableCallback(uint64_t tableId, uint64_t base, uint32_t length, void* callback,
                                            void* context, const void*) {
    return seh::installCallback(tableId, base, length, reinterpret_cast<uint64_t>(callback),
                                reinterpret_cast<uint64_t>(context))
               ? 1
               : 0;
}

int32_t win_RtlAddFunctionTable(void* table, uint32_t count, uint64_t base) {
    return seh::addTable(static_cast<const seh::RuntimeFunction*>(table), count, base) ? 1 : 0;
}

int32_t win_RtlDeleteFunctionTable(void* table) { return seh::deleteTable(table) ? 1 : 0; }

// Definidas tras el SEH (sehimpl).
void win_RtlRestoreContext(void* context, void* record);
void win_RtlUnwind(void* targetFrame, void* targetIp, void* record, void* returnValue);
void win_RtlRaiseException(void* record);

int32_t win_AreFileApisANSI() {
    return 1;
}

// --- .NET 6 (CoreCLR, Stardew Valley) -----------------------------------------------------------
// Afinidad: todos los núcleos (8). El GC de CoreCLR falla al crearse si devuelve FALSE.
int32_t win_GetProcessAffinityMask(void*, uint64_t* process, uint64_t* system) {
    lowmem::storePtr(process, uint64_t{0xFF});  // DWORD_PTR: 4 bytes en 32 bits
    lowmem::storePtr(system, uint64_t{0xFF});
    return 1;
}

uint32_t win_GetCurrentProcessorNumber() {
    const int cpu = sched_getcpu();
    return cpu >= 0 ? static_cast<uint32_t>(cpu) : 0u;
}

uint32_t win_WerRegisterRuntimeExceptionModule(const char16_t*, void*) { return 0; }  // S_OK

// ntdll!RtlGetVersion: Windows 10 22H2 (como GetVersionEx).
uint32_t win_RtlGetVersion(uint32_t* info) {
    if (info == nullptr) return 0xC000000Du;
    info[1] = 10;     // dwMajorVersion
    info[2] = 0;      // dwMinorVersion
    info[3] = 19045;  // dwBuildNumber
    info[4] = 2;      // VER_PLATFORM_WIN32_NT
    return 0;
}

// VerSetConditionMask / VerifyVersionInfoW: HyperPlay es Windows 10 22H2 (10.0.19045).
// ULONGLONG por valor = int64_t: en 32 bits ocupa 2 ranuras de pila (w32arity::isWide).
uint64_t win_VerSetConditionMask(int64_t maskIn, uint32_t typeMask, uint8_t condition) {
    auto mask = static_cast<uint64_t>(maskIn);
    for (uint32_t i = 0; i < 8; ++i) {
        if ((typeMask & (1u << i)) != 0) mask |= static_cast<uint64_t>(condition & 7u) << (i * 3);
    }
    return mask;
}

int32_t win_VerifyVersionInfoW(const uint32_t* info, uint32_t typeMask, int64_t conditionsIn) {
    const auto conditions = static_cast<uint64_t>(conditionsIn);
    if (info == nullptr) return 0;
    // OSVERSIONINFOEXW: [1]=major [2]=minor [3]=build; wServicePackMajor en byte 276.
    const uint64_t have[3] = {10, 0, 19045};
    const uint32_t bits[3] = {0x2 /*MAJOR*/, 0x1 /*MINOR*/, 0x4 /*BUILD*/};
    const uint32_t shift[3] = {1, 0, 2};
    for (int i = 0; i < 3; ++i) {
        if ((typeMask & bits[i]) == 0) continue;
        const uint32_t cond = static_cast<uint32_t>((conditions >> (shift[i] * 3)) & 7u);
        const uint64_t want = info[i + 1];
        bool ok = true;
        switch (cond) {
            case 1: ok = have[i] == want; break;  // VER_EQUAL
            case 2: ok = have[i] > want; break;   // VER_GREATER
            case 3: ok = have[i] >= want; break;  // VER_GREATER_EQUAL
            case 4: ok = have[i] < want; break;   // VER_LESS
            case 5: ok = have[i] <= want; break;  // VER_LESS_EQUAL
            default: break;
        }
        // Mayor/menor se comparan juntos en Windows: con mayor estrictamente mayor, el menor no cuenta.
        if (!ok && i == 1 && (cond == 3 || cond == 2) && (typeMask & 0x2) != 0 && have[0] > info[1]) ok = true;
        if (!ok && i == 2 && (cond == 3 || cond == 2) && (have[0] > info[1] || have[1] > info[2])) ok = true;
        if (!ok) {
            NativeDllInterceptor::win_SetLastError(1150);  // ERROR_OLD_WIN_VERSION
            return 0;
        }
    }
    // Resto de campos de OSVERSIONINFOEXW (bytes): dwPlatformId @16, wServicePackMajor @276,
    // wServicePackMinor @278, wSuiteMask @280, wProductType @282. Somos estación de trabajo
    // (VER_NT_WORKSTATION = 1), plataforma NT (2), sin service pack. Antes se ignoraban y
    // "¿es servidor?" también devolvía TRUE (mscoree tomaba la ruta de servidor).
    const auto* b = reinterpret_cast<const uint8_t*>(info);
    auto rd16 = [b](size_t off) { uint16_t v; std::memcpy(&v, b + off, 2); return static_cast<uint64_t>(v); };
    struct Extra { uint32_t bit; uint32_t idx; uint64_t have; uint64_t want; };
    const Extra extra[] = {
        {0x08, 3, 2, info[4]},          // VER_PLATFORMID
        {0x10, 4, 0, rd16(278)},        // VER_SERVICEPACKMINOR
        {0x20, 5, 0, rd16(276)},        // VER_SERVICEPACKMAJOR
        {0x80, 7, 1, b[282]},           // VER_PRODUCT_TYPE
    };
    for (const Extra& e : extra) {
        if ((typeMask & e.bit) == 0) continue;
        const uint32_t cond = static_cast<uint32_t>((conditions >> (e.idx * 3)) & 7u);
        bool ok = true;
        switch (cond) {
            case 1: ok = e.have == e.want; break;
            case 2: ok = e.have > e.want; break;
            case 3: ok = e.have >= e.want; break;
            case 4: ok = e.have < e.want; break;
            case 5: ok = e.have <= e.want; break;
            default: break;
        }
        if (!ok) {
            NativeDllInterceptor::win_SetLastError(1150);
            return 0;
        }
    }
    if ((typeMask & 0x40) != 0) {  // VER_SUITENAME: VER_AND (6) / VER_OR (7) sobre wSuiteMask
        constexpr uint64_t kSuite = 0x100;  // VER_SUITE_SINGLEUSERTS (Windows de escritorio)
        const uint32_t cond = static_cast<uint32_t>((conditions >> (6 * 3)) & 7u);
        const uint64_t want = rd16(280);
        const bool ok = cond == 7 ? (kSuite & want) != 0 : (kSuite & want) == want;
        if (!ok) {
            NativeDllInterceptor::win_SetLastError(1150);
            return 0;
        }
    }
    return 1;
}

// Ruta real de un handle de archivo (fd).
std::string pathOfHandle(void* h) {
    const int fd = fdFromHandle(h);
    if (fd < 0) return {};
    char buf[PATH_MAX] = {};
    const ssize_t n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

// SetEndOfFile: el archivo termina en la posición actual (FileStream.SetLength, guardados).
int32_t win_SetEndOfFile(void* h) {
    const int fd = fdFromHandle(h);
    if (fd < 0) return 0;
    const off_t pos = lseek(fd, 0, SEEK_CUR);
    return pos >= 0 && ftruncate(fd, pos) == 0 ? 1 : 0;
}

// Bloqueos de rango: un solo proceso usa los archivos del juego; se aceptan.
int32_t win_LockFileEx(void*, uint32_t, uint32_t, uint32_t, uint32_t, void*) { return 1; }
int32_t win_UnlockFileEx(void*, uint32_t, uint32_t, uint32_t, void*) { return 1; }
int32_t win_SetFileTime(void*, const void*, const void*, const void*) { return 1; }
int32_t win_SetFileAttributesW(const char16_t*, uint32_t) { return 1; }

uint32_t win_GetLogicalDrives() { return (1u << 2) | (1u << 3); }  // C: y D:

int32_t win_GetDiskFreeSpaceW(const char16_t*, uint32_t* sectorsPerCluster, uint32_t* bytesPerSector,
                              uint32_t* freeClusters, uint32_t* totalClusters) {
    struct statvfs v{};
    const std::string dir = s_activeImage != nullptr ? s_activeImage->gameDir : std::string("/sdcard");
    if (statvfs(dir.c_str(), &v) != 0) return 0;
    const uint64_t cluster = 4096;
    const uint64_t freeBytes = static_cast<uint64_t>(v.f_bavail) * v.f_frsize;
    const uint64_t totalBytes = static_cast<uint64_t>(v.f_blocks) * v.f_frsize;
    if (sectorsPerCluster != nullptr) *sectorsPerCluster = 8;
    if (bytesPerSector != nullptr) *bytesPerSector = 512;
    if (freeClusters != nullptr) *freeClusters = static_cast<uint32_t>(std::min<uint64_t>(freeBytes / cluster, 0xFFFFFFFFu));
    if (totalClusters != nullptr) *totalClusters = static_cast<uint32_t>(std::min<uint64_t>(totalBytes / cluster, 0xFFFFFFFFu));
    return 1;
}

// SetFileInformationByHandle: tamaño (FileEndOfFileInfo), borrar (FileDispositionInfo) y
// renombrar (FileRenameInfo). El resto se acepta sin cambios.
// true si la ruta real no está dentro de la carpeta del ejecutable ni de _HyperPlaySaves.
static bool outsideGameDir(const std::string& p) {
    if (s_activeImage == nullptr || p.empty()) return false;
    auto under = [&](const std::string& d) { return !d.empty() && p.compare(0, d.size(), d) == 0 && (p.size() == d.size() || p[d.size()] == '/'); };
    return !under(s_activeImage->gameDir) && !under(s_activeImage->savesDir);
}
int32_t win_SetFileInformationByHandle(void* h, uint32_t infoClass, const void* info, uint32_t size) {
    const int fd = fdFromHandle(h);
    if (fd < 0 || info == nullptr) return 0;
    switch (infoClass) {
        case 6:  // FileEndOfFileInfo
        case 5: {  // FileAllocationInfo
            int64_t len = 0;
            std::memcpy(&len, info, 8);
            if (infoClass == 6) return ftruncate(fd, static_cast<off_t>(len)) == 0 ? 1 : 0;
            return 1;
        }
        case 4: {  // FileDispositionInfo: DeleteFile (BOOLEAN)
            if (*static_cast<const uint8_t*>(info) == 0) return 1;
            const std::string p = pathOfHandle(h);
            if (outsideGameDir(p)) { LOGW("FileDispositionInfo denegado fuera de la carpeta del juego: %s", p.c_str()); NativeDllInterceptor::win_SetLastError(5); return 0; }
            return !p.empty() && unlink(p.c_str()) == 0 ? 1 : 0;
        }
        case 3: {  // FileRenameInfo {BOOLEAN ReplaceIfExists; HANDLE Root; DWORD Len; WCHAR Name[]}
            if (size < 24 || s_activeImage == nullptr) return 0;
            const auto* b = static_cast<const uint8_t*>(info);
            uint32_t nameBytes = 0;
            std::memcpy(&nameBytes, b + 16, 4);
            std::string name;
            for (uint32_t i = 0; i + 1 < nameBytes + 1 && i / 2 < nameBytes / 2; i += 2) {
                char16_t c = 0;
                std::memcpy(&c, b + 20 + i, 2);
                if (c < 0x80) name.push_back(static_cast<char>(c));
                else if (c < 0x800) { name.push_back(static_cast<char>(0xC0 | (c >> 6))); name.push_back(static_cast<char>(0x80 | (c & 0x3F))); }
                else { name.push_back(static_cast<char>(0xE0 | (c >> 12))); name.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F))); name.push_back(static_cast<char>(0x80 | (c & 0x3F))); }
            }
            const std::string from = pathOfHandle(h);
            const std::string to = BionicPeLoader::resolveVirtualPath(*s_activeImage, name);
            struct stat st{};
            if (from.empty() || to.empty() || (b[0] == 0 && stat(to.c_str(), &st) == 0)) return 0;
            if (outsideGameDir(from) || outsideGameDir(to)) { LOGW("FileRenameInfo denegado fuera de la carpeta del juego: %s", from.c_str()); NativeDllInterceptor::win_SetLastError(5); return 0; }
            const bool ok = rename(from.c_str(), to.c_str()) == 0;
            __android_log_print(ANDROID_LOG_INFO, "HyperPlay.DllInterceptor", "SetFileInformationByHandle(rename) '%s' -> '%s' %s",
                                from.c_str(), to.c_str(), ok ? "OK" : strerror(errno));
            return ok ? 1 : 0;
        }
        default:
            return 1;
    }
}

// ntdll!NtQuerySystemInformation: SystemProcessInformation (5) con un único proceso, el juego
// (System.Diagnostics.Process de .NET: nombre, id, memoria). El resto: clase no soportada.
uint32_t win_NtQuerySystemInformation(uint32_t infoClass, void* buffer, uint32_t length, uint32_t* returnLength) {
    if (infoClass != 5) return 0xC0000003u;  // STATUS_INVALID_INFO_CLASS
    std::string exe = "game.exe";
    if (s_activeImage != nullptr) {
        exe = s_activeImage->filePath;
        if (const size_t slash = exe.find_last_of("/\\"); slash != std::string::npos) exe = exe.substr(slash + 1);
    }
    std::u16string name;
    for (char c : exe) name.push_back(static_cast<char16_t>(static_cast<uint8_t>(c)));
    const uint32_t nameBytes = static_cast<uint32_t>(name.size() * 2);
    const uint32_t need = 0x100 + nameBytes + 2;
    if (returnLength != nullptr) *returnLength = need;
    if (buffer == nullptr || length < need) return 0xC0000004u;  // STATUS_INFO_LENGTH_MISMATCH
    auto* p = static_cast<uint8_t*>(buffer);
    std::memset(p, 0, need);
    const uint16_t len = static_cast<uint16_t>(nameBytes), maxLen = static_cast<uint16_t>(nameBytes + 2);
    const uint64_t nameBuf = reinterpret_cast<uint64_t>(p + 0x100);
    std::memcpy(p + 0x38, &len, 2);
    std::memcpy(p + 0x3A, &maxLen, 2);
    std::memcpy(p + 0x40, &nameBuf, 8);
    std::memcpy(p + 0x100, name.data(), nameBytes);
    const int32_t basePriority = 8;
    std::memcpy(p + 0x48, &basePriority, 4);
    const uint64_t pid = static_cast<uint64_t>(getpid());
    std::memcpy(p + 0x50, &pid, 8);
    // Memoria: VmRSS de /proc/self/statm (páginas residentes) como conjunto de trabajo.
    uint64_t sizePages = 0, residentPages = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        unsigned long long a = 0, b = 0;
        if (std::fscanf(f, "%llu %llu", &a, &b) == 2) {
            sizePages = a;
            residentPages = b;
        }
        std::fclose(f);
    }
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    const uint64_t vsize = sizePages * page, ws = residentPages * page;
    std::memcpy(p + 0x70, &vsize, 8);  // PeakVirtualSize
    std::memcpy(p + 0x78, &vsize, 8);  // VirtualSize
    std::memcpy(p + 0x88, &ws, 8);     // PeakWorkingSetSize
    std::memcpy(p + 0x90, &ws, 8);     // WorkingSetSize
    std::memcpy(p + 0xB8, &ws, 8);     // PagefileUsage
    std::memcpy(p + 0xC0, &ws, 8);     // PeakPagefileUsage
    std::memcpy(p + 0xC8, &ws, 8);     // PrivatePageCount
    return 0;
}

// NUMA: un solo nodo (0). El GC de CoreCLR lo consulta al iniciar y falla si devuelve FALSE.
int32_t win_GetNumaHighestNodeNumber(uint32_t* highest) {
    if (highest != nullptr) *highest = 0;
    return 1;
}

// HeapCompact: el heap de HyperPlay no se compacta; se informa de un bloque libre grande.
size_t win_HeapCompact(void*, uint32_t) { return 0x100000; }

// HyperPlay ejecuta siempre código x64 nativo (no WOW64).
int32_t win_IsWow64Process(void*, int32_t* wow64) {
    if (wow64 != nullptr) *wow64 = 0;
    return 1;
}

void win_OutputDebugStringW(const char16_t* s) {
    if (s == nullptr) return;
    std::string out;
    for (size_t i = 0; s[i] != 0 && i < 1024; ++i) out.push_back(s[i] < 0x80 ? static_cast<char>(s[i]) : '?');
    __android_log_print(ANDROID_LOG_INFO, "HyperPlay.Guest", "%s", out.c_str());
}

// LCMapStringEx / LCMapStringW: mayúsculas/minúsculas (LCMAP_UPPERCASE/LOWERCASE) y clave de
// ordenación (LCMAP_SORTKEY) simple. .NET sin ICU (modo NLS) la usa para ToUpper/ToLower y
// GetSortKey.
int32_t lcMapString(uint32_t flags, const char16_t* src, int32_t cchSrc, void* dest, int32_t cchDest) {
    constexpr uint32_t kLower = 0x100, kUpper = 0x200, kSortKey = 0x400, kIgnoreCase = 0x1;
    if (src == nullptr) return 0;
    size_t n = 0;
    if (cchSrc < 0) {
        while (src[n] != 0) ++n;
        ++n;  // incluye el NUL
    } else {
        n = static_cast<size_t>(cchSrc);
    }
    auto mapChar = [&](char16_t c, bool upper) -> char16_t {
        if (c >= 0xD800 && c <= 0xDFFF) return c;  // sustitutos: sin cambio
        const wint_t m = upper ? towupper(static_cast<wint_t>(c)) : towlower(static_cast<wint_t>(c));
        return m <= 0xFFFF ? static_cast<char16_t>(m) : c;
    };
    if ((flags & kSortKey) != 0) {
        // Bytes: cada carácter (sin mayúsculas si se ignoran) en big-endian + 1, y un 0 final.
        size_t chars = n;
        if (cchSrc < 0 && chars > 0) --chars;
        const size_t need = chars * 2 + 1;
        if (cchDest == 0 || dest == nullptr) return static_cast<int32_t>(need);
        if (static_cast<size_t>(cchDest) < need) return 0;
        auto* out = static_cast<uint8_t*>(dest);
        for (size_t i = 0; i < chars; ++i) {
            char16_t c = (flags & kIgnoreCase) != 0 ? mapChar(src[i], false) : src[i];
            const uint32_t v = std::min<uint32_t>(static_cast<uint32_t>(c) + 1, 0xFFFF);
            out[i * 2] = static_cast<uint8_t>(v >> 8);
            out[i * 2 + 1] = static_cast<uint8_t>(v & 0xFF);
        }
        out[chars * 2] = 0;
        return static_cast<int32_t>(need);
    }
    if (cchDest == 0 || dest == nullptr) return static_cast<int32_t>(n);
    if (static_cast<size_t>(cchDest) < n) return 0;
    auto* out = static_cast<char16_t*>(dest);
    const bool upper = (flags & kUpper) != 0 && (flags & kLower) == 0;
    const bool lower = (flags & kLower) != 0 && (flags & kUpper) == 0;
    for (size_t i = 0; i < n; ++i) out[i] = upper ? mapChar(src[i], true) : lower ? mapChar(src[i], false) : src[i];
    return static_cast<int32_t>(n);
}

int32_t win_LCMapStringEx(const char16_t*, uint32_t flags, const char16_t* src, int32_t cchSrc, void* dest,
                          int32_t cchDest, void*, void*, intptr_t) {
    return lcMapString(flags, src, cchSrc, dest, cchDest);
}

// ETW (EventSource de .NET): sin sesiones de trazado; registrar da un handle válido y escribir
// no hace nada (ERROR_SUCCESS).
uint32_t win_EventRegister(const void*, void*, void*, uint64_t* handle) {
    static std::atomic<uint64_t> s_next{1};
    if (handle != nullptr) *handle = s_next.fetch_add(1);
    return 0;
}
// REGHANDLE es ULONGLONG por valor: int64_t (2 ranuras de pila en 32 bits).
uint32_t win_EventUnregister(int64_t) { return 0; }
uint32_t win_EventSetInformation(int64_t, uint32_t, void*, uint32_t) { return 0; }
uint32_t win_EventWriteTransfer(int64_t, const void*, const void*, const void*, uint32_t, void*) { return 0; }
uint32_t win_EventActivityIdControl(uint32_t, void* guid) {
    if (guid != nullptr) std::memset(guid, 0, 16);
    return 0;
}
uint8_t win_EventEnabled(int64_t, const void*) { return 0; }

// Registro de eventos de Windows: no hay visor de eventos; se acepta y no se guarda.
void* win_RegisterEventSourceW(const char16_t*, const char16_t*) { return reinterpret_cast<void*>(0xE7E70001); }
int32_t win_DeregisterEventSource(void*) { return 1; }
int32_t win_ReportEventW(void*, uint16_t, uint16_t, uint32_t, void*, uint16_t, uint32_t, const char16_t**, void*) {
    return 1;
}

int32_t win_SetThreadErrorMode(uint32_t, uint32_t* oldMode) {
    if (oldMode != nullptr) *oldMode = 0;
    return 1;
}

void win_GetSystemTimePreciseAsFileTime(uint64_t* ft) {
    if (ft == nullptr) return;
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    *ft = (static_cast<uint64_t>(ts.tv_sec) + 11644473600ULL) * 10000000ULL +
          static_cast<uint64_t>(ts.tv_nsec) / 100ULL;
}

int32_t win_AppPolicyGetThreadInitializationType(void*, int32_t* policy) {
    if (policy != nullptr) *policy = 0;  // AppPolicyThreadInitializationType_None
    return 0;
}

// CONDITION_VARIABLE_INIT es un puntero nulo; Sleep/WakeConditionVariable usan su dirección.
void win_InitializeConditionVariable(void** cv) {
    lowmem::storePtr(cv, uint64_t{0});
}

int32_t win_GetCurrentPackageId(uint32_t* len, void*) {
    if (len != nullptr) *len = 0;
    return 15700;  // APPMODEL_ERROR_NO_PACKAGE: aplicación de escritorio
}

// QueryDepthSList / InterlockedPushListSListEx (cabecera: +0 primer nodo, +8 profundidad).
uint16_t win_QueryDepthSList(const uint64_t* head) {
    if (head != nullptr && lowmem::enabled()) return reinterpret_cast<const uint16_t*>(head)[2];  // x86: Depth en +4
    return head != nullptr ? static_cast<uint16_t>(head[1] & 0xFFFFu) : 0;
}
std::mutex& slistLock(const void* head);  // definido junto a InterlockedPushEntrySList

// Inserta la cadena ya enlazada list..listEnd delante de la cabecera, de forma atómica.
void* win_InterlockedPushListSListEx(void* head, void* list, void* listEnd, uint32_t count) {
    if (head == nullptr || list == nullptr || listEnd == nullptr) return nullptr;
    if (lowmem::enabled()) {
        auto* h32 = static_cast<uint32_t*>(head);
        std::lock_guard<std::mutex> lock(slistLock(head));
        const uint32_t old = h32[0];
        std::memcpy(listEnd, &old, 4);
        h32[0] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(list));
        auto* ds = reinterpret_cast<uint16_t*>(h32 + 1);
        ds[0] = static_cast<uint16_t>(ds[0] + count);
        ++ds[1];
        return reinterpret_cast<void*>(static_cast<uintptr_t>(old));
    }
    auto* h = static_cast<uint64_t*>(head);
    std::lock_guard<std::mutex> lock(slistLock(head));
    const uint64_t old = h[0];
    *static_cast<uint64_t*>(listEnd) = old;
    h[0] = reinterpret_cast<uint64_t>(list);
    h[1] = (h[1] & ~0xFFFFull) | (((h[1] & 0xFFFFull) + count) & 0xFFFFull);
    return reinterpret_cast<void*>(old);
}

// FileTimeToLocalFileTime / LocalFileTimeToFileTime: FILETIME (100 ns desde 1601) ± desfase de la
// zona horaria local. Mono las usa al abrir ensamblados si GetFileInformationByHandle da fechas.
int64_t localOffset100ns() {
    const time_t now = ::time(nullptr);
    struct tm lt{};
    localtime_r(&now, &lt);
    return static_cast<int64_t>(lt.tm_gmtoff) * 10000000LL;
}
int32_t win_FileTimeToLocalFileTime(const uint64_t* in, uint64_t* out) {
    if (in == nullptr || out == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    uint64_t v = 0;
    std::memcpy(&v, in, 8);
    v = static_cast<uint64_t>(static_cast<int64_t>(v) + localOffset100ns());
    std::memcpy(out, &v, 8);
    return 1;
}
int32_t win_LocalFileTimeToFileTime(const uint64_t* in, uint64_t* out) {
    if (in == nullptr || out == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    uint64_t v = 0;
    std::memcpy(&v, in, 8);
    v = static_cast<uint64_t>(static_cast<int64_t>(v) - localOffset100ns());
    std::memcpy(out, &v, 8);
    return 1;
}

// SetCriticalSectionSpinCount: las secciones críticas son mutex nativos; devuelve la cuenta anterior (0).
uint32_t win_SetCriticalSectionSpinCount(void*, uint32_t) { return 0; }

// OpenProcess: solo existe el propio proceso (pseudo-handle -1); otros PID -> NULL.
void* win_OpenProcess(uint32_t, int32_t, uint32_t pid) {
    if (pid == static_cast<uint32_t>(getpid()) || pid == 0) return reinterpret_cast<void*>(~uintptr_t{0});
    NativeDllInterceptor::win_SetLastError(87);
    return nullptr;
}
uint32_t win_GetProcessId(void*) { return static_cast<uint32_t>(getpid()); }

// Avisos de memoria: nunca hay "memoria baja" (el GC de .NET solo los consulta).
void* win_CreateMemoryResourceNotification(uint32_t) { return reinterpret_cast<void*>(uintptr_t{0xCAFE0040u}); }
int32_t win_QueryMemoryResourceNotification(void*, int32_t* state) {
    if (state != nullptr) *state = 0;
    return 1;
}
// El proceso no está dentro de un Job object.
int32_t win_IsProcessInJob(void*, void*, int32_t* result) {
    if (result != nullptr) *result = 0;
    return 1;
}

// Idioma de interfaz: en-US (como el resto del entorno).
uint16_t win_GetUserDefaultUILanguage() { return 0x0409; }

// OpenEvent: no hay eventos con nombre de otros procesos -> NULL + ERROR_FILE_NOT_FOUND.
void* win_OpenEventW(uint32_t, int32_t, const void*) {
    NativeDllInterceptor::win_SetLastError(2);
    return nullptr;
}

// Seguridad: un solo usuario, sin ACL. Los descriptores se inicializan (revisión 1, vacíos) y
// las operaciones sobre ellos tienen éxito; el CLR falla con HRESULT_FROM_WIN32(GetLastError())
// si devuelven FALSE.
int32_t win_InitializeSecurityDescriptor(uint8_t* sd, uint32_t revision) {
    if (sd == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    std::memset(sd, 0, lowmem::enabled() ? 20 : 40);  // SECURITY_DESCRIPTOR
    sd[0] = static_cast<uint8_t>(revision);
    return 1;
}
int32_t win_SetSecurityDescriptorDacl(uint8_t* sd, int32_t present, void* dacl, int32_t defaulted) {
    if (sd == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    uint16_t control = 0;
    std::memcpy(&control, sd + 2, 2);
    control = static_cast<uint16_t>((control & ~0x000Cu) | (present ? 0x4u : 0u) | (defaulted ? 0x8u : 0u));
    std::memcpy(sd + 2, &control, 2);
    lowmem::storePtr(sd + (lowmem::enabled() ? 16 : 32), dacl);  // Dacl
    return 1;
}

int32_t win_GetSecurityDescriptorDacl(const uint8_t* sd, int32_t* present, void** dacl, int32_t* defaulted) {
    if (sd == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    uint16_t control = 0;
    std::memcpy(&control, sd + 2, 2);
    if (present != nullptr) *present = (control & 0x4u) != 0 ? 1 : 0;
    if (defaulted != nullptr) *defaulted = (control & 0x8u) != 0 ? 1 : 0;
    lowmem::storePtr(dacl, lowmem::loadPtr(sd + (lowmem::enabled() ? 16 : 32)));
    return 1;
}
int32_t win_IsValidSecurityDescriptor(const uint8_t* sd) { return sd != nullptr && sd[0] == 1 ? 1 : 0; }

// SDDL -> descriptor: uno vacío (revisión 1, sin DACL = acceso total) en el montón (LocalFree).
int32_t win_ConvertStringSecurityDescriptorToSecurityDescriptorW(const void*, uint32_t, void** out, uint32_t* size) {
    const uint32_t n = lowmem::enabled() ? 20u : 40u;
    auto* sd = static_cast<uint8_t*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 8, n));
    if (sd == nullptr || out == nullptr) { NativeDllInterceptor::win_SetLastError(8); return 0; }
    sd[0] = 1;
    lowmem::storePtr(out, sd);
    if (size != nullptr) *size = n;
    return 1;
}

// El hilo no suplanta a nadie: OpenThreadToken falla con ERROR_NO_TOKEN (lo que esperan los
// llamadores, que entonces usan el token del proceso).
int32_t win_OpenThreadToken(void*, uint32_t, int32_t, void** token) {
    lowmem::storePtr(token, uint64_t{0});
    NativeDllInterceptor::win_SetLastError(1008);
    return 0;
}

// GetTokenInformation: usuario S-1-5-21-0-0-0-1000, elevado, integridad alta. Estructuras con
// punteros del tamaño del invitado (SID_AND_ATTRIBUTES = {PSID, DWORD}) y el SID detrás.
int32_t win_GetTokenInformation(void*, uint32_t cls, uint8_t* buf, uint32_t len, uint32_t* retLen) {
    const uint32_t ps = static_cast<uint32_t>(lowmem::guestPtrSize());
    auto fail = [](uint32_t err) {
        NativeDllInterceptor::win_SetLastError(err);
        return 0;
    };
    if (cls == 1 || cls == 25) {  // TokenUser / TokenIntegrityLevel: SID_AND_ATTRIBUTES + SID
        const uint8_t userSid[] = {1, 5, 0, 0, 0, 0, 0, 5, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xE8, 3, 0, 0};
        const uint8_t highSid[] = {1, 1, 0, 0, 0, 0, 0, 16, 0x00, 0x30, 0, 0};
        const uint8_t* sid = cls == 1 ? userSid : highSid;
        const uint32_t sidLen = cls == 1 ? sizeof(userSid) : sizeof(highSid);
        const uint32_t hdr = ps * 2;  // PSID + DWORD Attributes (con relleno en 64 bits)
        const uint32_t need = hdr + sidLen;
        if (retLen != nullptr) *retLen = need;
        if (buf == nullptr || len < need) return fail(122);  // ERROR_INSUFFICIENT_BUFFER
        std::memset(buf, 0, hdr);
        lowmem::storePtr(buf, buf + hdr);
        const uint32_t attrs = cls == 25 ? 0x60u /*SE_GROUP_INTEGRITY|ENABLED*/ : 0u;
        std::memcpy(buf + ps, &attrs, 4);
        std::memcpy(buf + hdr, sid, sidLen);
        return 1;
    }
    if (cls == 18 || cls == 20) {  // TokenElevationType (1 = por defecto) / TokenElevation (1)
        if (retLen != nullptr) *retLen = 4;
        if (buf == nullptr || len < 4) return fail(122);
        const uint32_t v = 1;
        std::memcpy(buf, &v, 4);
        return 1;
    }
    if (retLen != nullptr) *retLen = 0;
    return fail(87);
}

// SID: {Revision=1, SubAuthorityCount, IdentifierAuthority[6], SubAuthority[n] (DWORD)}.
uint32_t sidLength(const uint8_t* sid) { return sid == nullptr ? 0u : 8u + 4u * sid[1]; }
int32_t win_AllocateAndInitializeSid(const uint8_t* authority, uint8_t count, uint32_t s0, uint32_t s1, uint32_t s2,
                                     uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6, uint32_t s7, void** out) {
    if (authority == nullptr || out == nullptr || count > 8) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    auto* sid = static_cast<uint8_t*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 8, 8 + 4 * 8));
    if (sid == nullptr) { NativeDllInterceptor::win_SetLastError(8); return 0; }
    sid[0] = 1;
    sid[1] = count;
    std::memcpy(sid + 2, authority, 6);
    const uint32_t sub[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    std::memcpy(sid + 8, sub, 4u * count);
    lowmem::storePtr(out, sid);
    return 1;
}
void* win_FreeSid(void* sid) {
    if (sid != nullptr) NativeDllInterceptor::win_HeapFree(NativeDllInterceptor::win_GetProcessHeap(), 0, sid);
    return nullptr;
}
uint32_t win_GetLengthSid(const uint8_t* sid) { return sidLength(sid); }
uint8_t* win_GetSidSubAuthorityCount(uint8_t* sid) { return sid != nullptr ? sid + 1 : nullptr; }

// POWRPROF!CallNtPowerInformation (stdcall, 5 args). ProcessorInformation (11) devuelve un
// PROCESSOR_POWER_INFORMATION de 24 bytes por CPU (8, como GetSystemInfo); el resto, ceros.
// MSVCR90 (cdecl): _encode_pointer/_decode_pointer sin cifrado (identidad) y _encoded_null.
uint32_t crt90IdentityPtr(uint32_t p) { return p; }
uint32_t crt90EncodedNull() { return 0; }

int32_t win_CallNtPowerInformation(int32_t level, const void*, uint32_t, void* out, uint32_t outLen) {
    if (out == nullptr || outLen == 0) return 0;
    if (level != 11) { std::memset(out, 0, outLen); return 0; }
    constexpr uint32_t kCpus = 8, kEntry = 24;
    if (outLen < kCpus * kEntry) return static_cast<int32_t>(0xC0000023u);  // STATUS_BUFFER_TOO_SMALL
    auto* b = static_cast<uint8_t*>(out);
    for (uint32_t i = 0; i < kCpus; ++i) {
        const uint32_t mhz = i >= 6 ? 4320u : 3530u;
        const uint32_t e[6] = {i, mhz, mhz, mhz, 0, 0};  // Number, MaxMhz, CurrentMhz, MhzLimit, Idle
        std::memcpy(b + i * kEntry, e, kEntry);
    }
    return 0;
}
uint32_t* win_GetSidSubAuthority(uint8_t* sid, uint32_t n) {
    return sid != nullptr ? reinterpret_cast<uint32_t*>(sid + 8 + 4 * n) : nullptr;
}
uint8_t* win_GetSidIdentifierAuthority(uint8_t* sid) { return sid != nullptr ? sid + 2 : nullptr; }
// "S-1-<autoridad>-<sub>..." en memoria del montón (el llamador la libera con LocalFree).
int32_t win_ConvertSidToStringSidW(const uint8_t* sid, void** out) {
    if (sid == nullptr || out == nullptr || sid[0] != 1) { NativeDllInterceptor::win_SetLastError(1337); return 0; }
    uint64_t auth = 0;
    for (int i = 2; i < 8; ++i) auth = (auth << 8) | sid[i];
    std::string s = "S-1-" + std::to_string(auth);
    for (uint32_t i = 0; i < sid[1]; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, sid + 8 + 4 * i, 4);
        s += "-" + std::to_string(v);
    }
    auto* w = static_cast<char16_t*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 0, (s.size() + 1) * 2));
    if (w == nullptr) { NativeDllInterceptor::win_SetLastError(8); return 0; }
    for (size_t i = 0; i <= s.size(); ++i) w[i] = static_cast<char16_t>(i < s.size() ? s[i] : 0);
    lowmem::storePtr(out, w);
    return 1;
}
int32_t win_IsValidSid(const uint8_t* sid) { return sid != nullptr && sid[0] == 1 && sid[1] <= 15 ? 1 : 0; }
int32_t win_EqualSid(const uint8_t* a, const uint8_t* b) {
    return a != nullptr && b != nullptr && sidLength(a) == sidLength(b) && std::memcmp(a, b, sidLength(a)) == 0 ? 1 : 0;
}
int32_t win_CopySid(uint32_t len, uint8_t* dst, const uint8_t* src) {
    const uint32_t n = sidLength(src);
    if (dst == nullptr || src == nullptr || len < n) { NativeDllInterceptor::win_SetLastError(122); return 0; }
    std::memcpy(dst, src, n);
    return 1;
}
// ACL vacía {Revision, Sbz1, AclSize, AceCount=0, Sbz2}; los ACE se aceptan sin guardarse.
int32_t win_InitializeAcl(uint8_t* acl, uint32_t size, uint32_t revision) {
    if (acl == nullptr || size < 8) { NativeDllInterceptor::win_SetLastError(122); return 0; }
    std::memset(acl, 0, 8);
    acl[0] = static_cast<uint8_t>(revision);
    const auto s = static_cast<uint16_t>(size);
    std::memcpy(acl + 2, &s, 2);
    return 1;
}
int32_t win_AddAccessAllowedAce(void*, uint32_t, uint32_t, void*) { return 1; }
int32_t win_AddAccessAllowedAceEx(void*, uint32_t, uint32_t, uint32_t, void*) { return 1; }
// Un solo usuario: pertenece a todos los grupos que se consulten (como un administrador local).
int32_t win_CheckTokenMembership(void*, void*, int32_t* isMember) {
    if (isMember != nullptr) *isMember = 1;
    return 1;
}

void* resolveSystemExtraExport(const std::string& funcName) {
    static const std::unordered_map<std::string, void*> kExports = {
        {"RtlAddGrowableFunctionTable", w32arity::hpFn(&win_RtlAddGrowableFunctionTable)},
        {"RtlGrowFunctionTable", w32arity::hpFn(&win_RtlGrowFunctionTable)},
        {"RtlDeleteGrowableFunctionTable", w32arity::hpFn(&win_RtlDeleteGrowableFunctionTable)},
        {"RtlInstallFunctionTableCallback", w32arity::hpFn(&win_RtlInstallFunctionTableCallback)},
        {"RtlAddFunctionTable", w32arity::hpFn(&win_RtlAddFunctionTable)},
        {"RtlDeleteFunctionTable", w32arity::hpFn(&win_RtlDeleteFunctionTable)},
        {"AreFileApisANSI", w32arity::hpFn(&win_AreFileApisANSI)},
        {"IsWow64Process", w32arity::hpFn(&win_IsWow64Process)},
        {"HeapCompact", w32arity::hpFn(&win_HeapCompact)},
        {"RtlRestoreContext", w32arity::hpFn(&win_RtlRestoreContext)},
        {"RtlUnwind", w32arity::hpFn(&win_RtlUnwind)},
        {"RtlRaiseException", w32arity::hpFn(&win_RtlRaiseException)},
        {"GetNumaHighestNodeNumber", w32arity::hpFn(&win_GetNumaHighestNodeNumber)},
        {"NtQuerySystemInformation", w32arity::hpFn(&win_NtQuerySystemInformation)},
        {"GetProcessAffinityMask", w32arity::hpFn(&win_GetProcessAffinityMask)},
        {"GetCurrentProcessorNumber", w32arity::hpFn(&win_GetCurrentProcessorNumber)},
        {"WerRegisterRuntimeExceptionModule", w32arity::hpFn(&win_WerRegisterRuntimeExceptionModule)},
        {"RtlGetVersion", w32arity::hpFn(&win_RtlGetVersion)},
        {"VerSetConditionMask", w32arity::hpFn(&win_VerSetConditionMask)},
        {"VerifyVersionInfoW", w32arity::hpFn(&win_VerifyVersionInfoW)},
        {"SetEndOfFile", w32arity::hpFn(&win_SetEndOfFile)},
        {"LockFileEx", w32arity::hpFn(&win_LockFileEx)},
        {"UnlockFileEx", w32arity::hpFn(&win_UnlockFileEx)},
        {"SetFileTime", w32arity::hpFn(&win_SetFileTime)},
        {"SetFileAttributesW", w32arity::hpFn(&win_SetFileAttributesW)},
        {"GetLogicalDrives", w32arity::hpFn(&win_GetLogicalDrives)},
        {"GetDiskFreeSpaceW", w32arity::hpFn(&win_GetDiskFreeSpaceW)},
        {"SetFileInformationByHandle", w32arity::hpFn(&win_SetFileInformationByHandle)},
        {"OutputDebugStringW", w32arity::hpFn(&win_OutputDebugStringW)},
        {"LCMapStringEx", w32arity::hpFn(&win_LCMapStringEx)},
        {"RegisterEventSourceW", w32arity::hpFn(&win_RegisterEventSourceW)},
        {"DeregisterEventSource", w32arity::hpFn(&win_DeregisterEventSource)},
        {"ReportEventW", w32arity::hpFn(&win_ReportEventW)},
        {"SetThreadErrorMode", w32arity::hpFn(&win_SetThreadErrorMode)},
        {"GetSystemTimePreciseAsFileTime", w32arity::hpFn(&win_GetSystemTimePreciseAsFileTime)},
        {"AppPolicyGetThreadInitializationType", w32arity::hpFn(&win_AppPolicyGetThreadInitializationType)},
        {"GetCurrentPackageId", w32arity::hpFn(&win_GetCurrentPackageId)},
        {"InitializeConditionVariable", w32arity::hpFn(&win_InitializeConditionVariable)},
        {"QueryDepthSList", w32arity::hpFn(&win_QueryDepthSList)},
        {"RtlQueryDepthSList", w32arity::hpFn(&win_QueryDepthSList)},
        {"InterlockedPushListSListEx", w32arity::hpFn(&win_InterlockedPushListSListEx)},
        {"RtlInterlockedPushListSListEx", w32arity::hpFn(&win_InterlockedPushListSListEx)},
        {"RtlInterlockedPushEntrySList", w32arity::hpFn(&NativeDllInterceptor::win_InterlockedPushEntrySList)},
        {"RtlInterlockedPopEntrySList", w32arity::hpFn(&NativeDllInterceptor::win_InterlockedPopEntrySList)},
        {"RtlInterlockedFlushSList", w32arity::hpFn(&NativeDllInterceptor::win_InterlockedFlushSList)},
        {"RtlInitializeSListHead", w32arity::hpFn(&NativeDllInterceptor::win_InitializeSListHead)},
        {"GetUserDefaultUILanguage", w32arity::hpFn(&win_GetUserDefaultUILanguage)},
        {"GetSystemDefaultUILanguage", w32arity::hpFn(&win_GetUserDefaultUILanguage)},
        {"OpenEventW", w32arity::hpFn(&win_OpenEventW)},
        {"OpenEventA", w32arity::hpFn(&win_OpenEventW)},
        {"OpenProcess", w32arity::hpFn(&win_OpenProcess)},
        {"SetCriticalSectionSpinCount", w32arity::hpFn(&win_SetCriticalSectionSpinCount)},
        {"FileTimeToLocalFileTime", w32arity::hpFn(&win_FileTimeToLocalFileTime)},
        {"LocalFileTimeToFileTime", w32arity::hpFn(&win_LocalFileTimeToFileTime)},
        {"GetProcessId", w32arity::hpFn(&win_GetProcessId)},
        {"CreateMemoryResourceNotification", w32arity::hpFn(&win_CreateMemoryResourceNotification)},
        {"QueryMemoryResourceNotification", w32arity::hpFn(&win_QueryMemoryResourceNotification)},
        {"IsProcessInJob", w32arity::hpFn(&win_IsProcessInJob)},
    };
    auto it = kExports.find(funcName);
    return it == kExports.end() ? nullptr : it->second;
}

void* resolveRegistryExport(const std::string& funcName) {
    static const std::unordered_map<std::string, void*> kExports = {
        {"RegOpenKeyExW", w32arity::hpFn(&win_RegOpenKeyExW)},
        {"InitializeSecurityDescriptor", w32arity::hpFn(&win_InitializeSecurityDescriptor)},
        {"SetSecurityDescriptorDacl", w32arity::hpFn(&win_SetSecurityDescriptorDacl)},
        {"AllocateAndInitializeSid", w32arity::hpFn(&win_AllocateAndInitializeSid)},
        {"GetSecurityDescriptorDacl", w32arity::hpFn(&win_GetSecurityDescriptorDacl)},
        {"IsValidSecurityDescriptor", w32arity::hpFn(&win_IsValidSecurityDescriptor)},
        {"OpenThreadToken", w32arity::hpFn(&win_OpenThreadToken)},
        {"ConvertStringSecurityDescriptorToSecurityDescriptorW",
         w32arity::hpFn(&win_ConvertStringSecurityDescriptorToSecurityDescriptorW)},
        {"GetTokenInformation", w32arity::hpFn(&win_GetTokenInformation)},
        {"FreeSid", w32arity::hpFn(&win_FreeSid)},
        {"GetLengthSid", w32arity::hpFn(&win_GetLengthSid)},
        {"GetSidSubAuthorityCount", w32arity::hpFn(&win_GetSidSubAuthorityCount)},
        {"GetSidSubAuthority", w32arity::hpFn(&win_GetSidSubAuthority)},
        {"GetSidIdentifierAuthority", w32arity::hpFn(&win_GetSidIdentifierAuthority)},
        {"ConvertSidToStringSidW", w32arity::hpFn(&win_ConvertSidToStringSidW)},
        {"IsValidSid", w32arity::hpFn(&win_IsValidSid)},
        {"EqualSid", w32arity::hpFn(&win_EqualSid)},
        {"CopySid", w32arity::hpFn(&win_CopySid)},
        {"InitializeAcl", w32arity::hpFn(&win_InitializeAcl)},
        {"AddAccessAllowedAce", w32arity::hpFn(&win_AddAccessAllowedAce)},
        {"AddAccessAllowedAceEx", w32arity::hpFn(&win_AddAccessAllowedAceEx)},
        {"CheckTokenMembership", w32arity::hpFn(&win_CheckTokenMembership)},
        {"RegisterEventSourceW", w32arity::hpFn(&win_RegisterEventSourceW)},
        {"DeregisterEventSource", w32arity::hpFn(&win_DeregisterEventSource)},
        {"ReportEventW", w32arity::hpFn(&win_ReportEventW)},
        {"EventRegister", w32arity::hpFn(&win_EventRegister)},
        {"EventUnregister", w32arity::hpFn(&win_EventUnregister)},
        {"EventSetInformation", w32arity::hpFn(&win_EventSetInformation)},
        {"EventWriteTransfer", w32arity::hpFn(&win_EventWriteTransfer)},
        {"EventActivityIdControl", w32arity::hpFn(&win_EventActivityIdControl)},
        {"EventEnabled", w32arity::hpFn(&win_EventEnabled)},
        {"RegOpenKeyExA", w32arity::hpFn(&win_RegOpenKeyExA)},
        {"RegOpenKeyW", w32arity::hpFn(&win_RegOpenKeyW)},
        {"RegCreateKeyExW", w32arity::hpFn(&win_RegCreateKeyExW)},
        {"RegCreateKeyExA", w32arity::hpFn(&win_RegCreateKeyExA)},
        {"RegCreateKeyW", w32arity::hpFn(&win_RegCreateKeyW)},
        {"RegCloseKey", w32arity::hpFn(&win_RegCloseKey)},
        {"RegFlushKey", w32arity::hpFn(&win_RegFlushKey)},
        {"RegSetValueExW", w32arity::hpFn(&win_RegSetValueExW)},
        {"RegSetValueExA", w32arity::hpFn(&win_RegSetValueExA)},
        {"RegQueryValueExW", w32arity::hpFn(&win_RegQueryValueExW)},
        {"RegQueryValueExA", w32arity::hpFn(&win_RegQueryValueExA)},
        {"RegGetValueW", w32arity::hpFn(&win_RegGetValueW)},
        {"RegDeleteValueW", w32arity::hpFn(&win_RegDeleteValueW)},
        {"RegDeleteValueA", w32arity::hpFn(&win_RegDeleteValueA)},
        {"RegDeleteKeyW", w32arity::hpFn(&win_RegDeleteKeyW)},
        {"RegEnumValueW", w32arity::hpFn(&win_RegEnumValueW)},
        {"RegEnumKeyExW", w32arity::hpFn(&win_RegEnumKeyExW)},
        {"RegQueryInfoKeyW", w32arity::hpFn(&win_RegQueryInfoKeyW)},
    };
    auto it = kExports.find(funcName);
    return it == kExports.end() ? nullptr : it->second;
}

}  // namespace

void NativeDllInterceptor::init(LoadedPeImage* image) {
    s_activeImage = image;
    // Unreal Engine 3 de 32 bits (<raíz>/Binaries/Win32/juego.exe y <raíz>/Engine/Config): D:\ pasa a ser la
    // raíz y el .exe se presenta en D:\Binaries\Win32\. Unreal colapsa "<base>\..\..\" y con la base en D:\
    // perdía la unidad ("DevilGame\CookedPCConsole\Core.upk" relativo). 64 bits: sin cambios.
    BionicPeLoader::setExeSubdir({});
    if (image != nullptr && image->is32Bit) OryonEmitterBackend::armWriteWatchFromProperty();
    if (image != nullptr && image->is32Bit &&
        access((image->gameDir + "/../../Engine/Config/BaseEngine.ini").c_str(), F_OK) == 0) {
        const size_t s1 = image->gameDir.find_last_of('/');
        const size_t s2 = s1 == std::string::npos || s1 == 0 ? std::string::npos : image->gameDir.find_last_of('/', s1 - 1);
        if (s2 != std::string::npos) {
            BionicPeLoader::setExeSubdir(image->gameDir.substr(s2 + 1));
            LOGI("Unreal Engine 3: .exe en %s", BionicPeLoader::exeWinDir().c_str());
        }
    }
    // Invitado de 32 bits: el montón Win32 y demás memoria que se le entrega, por debajo de 4 GB.
    if (image != nullptr && image->is32Bit) {
        lowmem::g_enabled.store(true);
        // Los HMODULE de las DLL de sistema son 0x7FFF00xx. El .NET Framework lee la cabecera PE de un
        // HMODULE (MZ/PE/SizeOfImage): una página de ceros hace que la lea, vea que no es una imagen y
        // siga, en vez de fallar. Solo 32 bits (en 64 bits nadie la ha necesitado).
        static std::once_flag s_fakeHmodPage;
        std::call_once(s_fakeHmodPage, [] {
            void* p = mmap(reinterpret_cast<void*>(0x7FFF0000u), 0x10000, PROT_READ,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p == MAP_FAILED || p != reinterpret_cast<void*>(0x7FFF0000u)) {
                if (p != MAP_FAILED) munmap(p, 0x10000);
                LOGW("No se pudo reservar la página de HMODULE de sistema en 0x7FFF0000");
            }
        });
    }
    gpu::presentopt::g_widePatched.store(false);  // juego nuevo: aún sin parche panorámico
    s_cmdA[0] = '\0';
    s_cmdW[0] = u'\0';
    s_exeNameW[0] = u'\0';
    s_unityMemoryManagerAllocate = 0;
    s_unityMemoryManagerReallocate = 0;
    s_unityMemoryManagerDeallocate = 0;
    {
        char tso[PROP_VALUE_MAX] = {};
        const bool off = __system_property_get("debug.hyperplay.tso", tso) > 0 && tso[0] == '0';
        OryonEmitterBackend::tsoMemory.store(!off);
        LOGI("Orden de memoria x86 (TSO LDAPR/STLR): %s", off ? "desactivado" : "activo");
        char xflags[PROP_VALUE_MAX] = {};
        const bool xflagsOn = __system_property_get("debug.hyperplay.xflags", xflags) > 0 && xflags[0] == '1';
        OryonEmitterBackend::crossBlockFlags.store(xflagsOn);
        LOGI("Flags muertos entre bloques: %s", xflagsOn ? "activo" : "desactivado");
        char chain[PROP_VALUE_MAX] = {};
        const bool noChain = __system_property_get("debug.hyperplay.chain", chain) > 0 && chain[0] == '0';
        OryonEmitterBackend::chainBlocks.store(!noChain);
        LOGI("Encadenado de bloques JIT: %s", noChain ? "desactivado" : "activo");
        char dead[PROP_VALUE_MAX] = {};
        const bool noDead = __system_property_get("debug.hyperplay.deadflags", dead) > 0 && dead[0] == '0';
        OryonEmitterBackend::deadFlags.store(!noDead);
        LOGI("Eliminación de flags muertos: %s", noDead ? "desactivada" : "activa");
        char fuse[PROP_VALUE_MAX] = {};
        const bool noFuse = __system_property_get("debug.hyperplay.fuse", fuse) > 0 && fuse[0] == '0';
        OryonEmitterBackend::fuseFlags.store(!noFuse);
        LOGI("Fusión de flags (B.cond/CSEL/CSET sobre NZCV): %s", noFuse ? "desactivada" : "activa");
        char ras[PROP_VALUE_MAX] = {};
        const bool noRas = __system_property_get("debug.hyperplay.ras", ras) > 0 && ras[0] == '0';
        OryonEmitterBackend::returnStack.store(!noRas);
        LOGI("Pila de retornos CALL/RET (BL + RET predicho): %s", noRas ? "desactivada" : "activa");
        char inl[PROP_VALUE_MAX] = {};
        const bool noInline = __system_property_get("debug.hyperplay.inlinedispatch", inl) > 0 && inl[0] == '0';
        OryonEmitterBackend::inlineDispatch.store(!noInline);
        LOGI("Búsqueda en línea de RET/saltos indirectos: %s", noInline ? "desactivada" : "activa");
        char fastn[PROP_VALUE_MAX] = {};
        const bool noFast = __system_property_get("debug.hyperplay.fastnative", fastn) > 0 && fastn[0] == '0';
        OryonEmitterBackend::fastNativeCalls.store(!noFast);
        LOGI("Llamadas nativas directas desde el JIT: %s", noFast ? "desactivadas" : "activas");
        // Lista rápida: wrappers de <= 4 argumentos enteros, sin callbacks al juego ni acceso al
        // contexto del despachador. Los que pueden esperar a otro hilo invitado van como bloqueantes.
        const auto fast = [](auto fn, bool blocking) {
            OryonEmitterBackend::registerFastNative(reinterpret_cast<uintptr_t>(fn), blocking);
        };
        fast(&win_TlsGetValue, false);
        fast(&win_TlsSetValue, false);
        fast(&win_FlsGetValue, false);
        fast(&win_FlsSetValue, false);
        fast(&win_GetCurrentThreadId, false);
        fast(&win_GetCurrentProcessId, false);
        fast(&win_GetCurrentProcess, false);
        fast(&win_SwitchToThread, true);
        fast(&win_Sleep, true);
        fast(&win_GetCurrentThread, false);
        fast(&win_GetLastError, false);
        fast(&win_SetLastError, false);
        fast(&win_QueryPerformanceCounter, false);
        fast(&win_QueryPerformanceFrequency, false);
        fast(&win_GetTickCount64, false);
        fast(&win_HeapAlloc, false);
        fast(&win_HeapFree, false);
        fast(&win_HeapReAlloc, false);
        fast(&win_HeapSize, false);
        fast(&win_LeaveCriticalSection, false);
        fast(&win_TryEnterCriticalSection, false);
        fast(&win_EnterCriticalSection, true);
        fast(&win_AcquireSRWLockExclusive, true);
        fast(&win_AcquireSRWLockShared, true);
        fast(&win_ReleaseSRWLockExclusive, false);
        fast(&win_ReleaseSRWLockShared, false);
        fast(&win_TryAcquireSRWLockExclusive, false);
        fast(&win_WakeByAddress, false);
        fast(&win_WakeByAddressAll, false);
        fast(&win_WaitOnAddress, true);
        fast(&win_SetEvent, false);
        fast(&win_ResetEvent, false);
        fast(&win_ReleaseSemaphore, false);
        fast(&win_InterlockedPushEntrySList, false);
        fast(&win_InterlockedPopEntrySList, false);
        fast(&win_GetKeyState, false);
        fast(&win_WaitForSingleObject, true);
        // 32 bits: la IAT guarda alias bajos; solo se aceptan stdcall con aridad conocida y sin argumentos de 64 bits.
        OryonEmitterBackend::fast32Host.store([](uintptr_t a) -> uintptr_t {
            return w32alias::isAlias(a) ? w32alias::hostFor(a) : 0;
        });
        OryonEmitterBackend::fast32Indirect.store([](CpuContext* ctx, uint32_t target, uint32_t nextRip) -> uint32_t {
            // CALL indirecto desde el bloque traducido: se apila el retorno y se atiende como cualquier thunk.
            thread_local OryonEmitterBackend* t_self = OryonEmitterBackend::forThread(static_cast<int>(gettid()));
            const uint32_t esp = static_cast<uint32_t>(ctx->gpr[4]) - 4;
            std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(esp)), &nextRip, 4);
            ctx->gpr[4] = esp;
            if (!NativeDllInterceptor::dispatchThunk(t_self, target, *ctx)) {
                ctx->rip = target;  // no es un thunk: llamada normal al código invitado
                return 1;
            }
            return ctx->rip == nextRip ? 0u : 1u;
        });
        OryonEmitterBackend::fast32Slots.store([](uintptr_t h) -> int {
            w32arity::Arity ar{};
            if (!w32arity::lookup(reinterpret_cast<const void*>(h), ar) || ar.variadic || ar.wideMask != 0) return -1;
            return ar.slots;
        });
    }
    if (image) {
        prof::startIfRequested(image->gameDir + "/_HyperPlaySaves");  // setprop debug.hyperplay.prof <s>
        NativeSteamStub::init(image->gameDir);
    }
    resetTelemetry();
    LOGI("NativeDllInterceptor inicializado para imagen en %p", image ? image->baseAddress : nullptr);
}

void NativeDllInterceptor::registerUnityMemoryHooks(uintptr_t unityPlayerBase) {
    if (unityPlayerBase == 0) return;
    // Desactivado por defecto: se creó cuando el asignador temporal de Unity (TLS) devolvía NULL
    // por el TLS roto de las DLL dinámicas (ya corregido). Sus RVAs son fijas de una versión de
    // UnityPlayer y Reallocate copia tamaños adivinados: riesgo de corrupción del heap.
    // `adb shell setprop debug.hyperplay.unitymemhook 1` lo reactiva para comparar.
    {
        char prop[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.unitymemhook", prop) <= 0 || prop[0] != '1') {
            LOGI("Hooks de Unity MemoryManager desactivados");
            return;
        }
    }

    // RVAs correspondientes a UnityPlayer.dll (versión de Hollow Knight 2020.2 / 2020.3 x64)
    s_unityMemoryManagerAllocate = unityPlayerBase + 0x2b3a60;
    s_unityMemoryManagerReallocate = unityPlayerBase + 0x2b3cf0;
    s_unityMemoryManagerDeallocate = unityPlayerBase + 0x2b40b0;

    registerTrampolineAddress(reinterpret_cast<void*>(s_unityMemoryManagerAllocate), "Unity!MemoryManager::Allocate");
    registerTrampolineAddress(reinterpret_cast<void*>(s_unityMemoryManagerReallocate), "Unity!MemoryManager::Reallocate");
    registerTrampolineAddress(reinterpret_cast<void*>(s_unityMemoryManagerDeallocate), "Unity!MemoryManager::Deallocate");

    LOGI("NativeDllInterceptor: Unity memory hooks registrados en Allocate=%p, Reallocate=%p, Deallocate=%p",
         reinterpret_cast<void*>(s_unityMemoryManagerAllocate),
         reinterpret_cast<void*>(s_unityMemoryManagerReallocate),
         reinterpret_cast<void*>(s_unityMemoryManagerDeallocate));
}

void NativeDllInterceptor::resetTelemetry() noexcept {
    s_kernel32Calls.store(0);
    s_ntdllCalls.store(0);
    s_d3dCalls.store(0);
    s_vulkanCalls.store(0);
    s_xinputCalls.store(0);
    s_xaudioCalls.store(0);
    s_steamCalls.store(0);
    s_otherCalls.store(0);
}

InterceptTelemetry NativeDllInterceptor::getTelemetry() noexcept {
    InterceptTelemetry t;
    t.kernel32Calls = s_kernel32Calls.load();
    t.ntdllCalls = s_ntdllCalls.load();
    t.d3dCalls = s_d3dCalls.load();
    t.vulkanCalls = s_vulkanCalls.load();
    t.xinputCalls = s_xinputCalls.load();
    t.xaudioCalls = s_xaudioCalls.load();
    t.steamCalls = s_steamCalls.load();
    t.otherCalls = s_otherCalls.load();
    return t;
}

namespace {
class DynamicStubManager {
public:
    static void* getOrCreateStub(const std::string& dllName, const std::string& funcName) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string key = toLower(dllName) + "!" + funcName;
        auto it = stubsByKey_.find(key);
        if (it != stubsByKey_.end()) return it->second;

        if (arena_ == nullptr || arenaOffset_ + 16 > arenaSize_) {
            const size_t sz = 65536;
            void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                p = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) return nullptr;
            }
            arena_ = static_cast<uint8_t*>(p);
            arenaSize_ = sz;
            arenaOffset_ = 0;
        }

        uint8_t* stubPtr = arena_ + arenaOffset_;
        arenaOffset_ += 16;

        const uint32_t retInst = 0xD65F03C0u; // AArch64 RET
        std::memcpy(stubPtr, &retInst, sizeof(retInst));

        const uintptr_t target = reinterpret_cast<uintptr_t>(stubPtr);
        stubsByKey_[key] = stubPtr;
        stubNamesByAddr_[target] = dllName + "!" + funcName;

        registerTrampolineAddress(stubPtr);
        return stubPtr;
    }

    static const std::string* findStubName(uintptr_t target) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stubNamesByAddr_.find(target);
        if (it != stubNamesByAddr_.end()) return &it->second;
        return nullptr;
    }

private:
    static inline std::mutex mutex_;
    static inline uint8_t* arena_ = nullptr;
    static inline size_t arenaSize_ = 0;
    static inline size_t arenaOffset_ = 0;
    static inline std::unordered_map<std::string, void*> stubsByKey_;
    static inline std::unordered_map<uintptr_t, std::string> stubNamesByAddr_;
};
}  // namespace

void* NativeDllInterceptor::getOrCreateDynamicStub(const std::string& dllName, const std::string& funcName) {
    return DynamicStubManager::getOrCreateStub(dllName, funcName);
}

bool NativeDllInterceptor::isRegisteredTrampoline(uintptr_t target) noexcept {
    if (target == 0) return false;
    if (target < s_trampolineMin.load(std::memory_order_relaxed) ||
        target > s_trampolineMax.load(std::memory_order_relaxed)) {
        return false;
    }
    // Se consulta en cada búsqueda lenta del despachador de todos los hilos: caché por hilo
    // invalidada cuando se registra un trampolín nuevo (s_trampolineGen).
    thread_local std::unordered_map<uintptr_t, bool> t_known;
    thread_local uint64_t t_gen = 0;
    const uint64_t gen = s_trampolineGen.load(std::memory_order_acquire);
    if (gen != t_gen) {
        t_known.clear();
        t_gen = gen;
    }
    if (auto it = t_known.find(target); it != t_known.end()) return it->second;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(s_trampolineMutex);
        found = s_registeredTrampolines.find(target) != s_registeredTrampolines.end();
    }
    if (t_known.size() > 200000) t_known.clear();
    t_known.emplace(target, found);
    return found;
}

static bool isVoidWin32Function(std::string_view name) {
    auto pos = name.rfind('!');
    std::string_view fn = (pos != std::string_view::npos) ? name.substr(pos + 1) : name;
    return (fn == "EnterCriticalSection" ||
            fn == "LeaveCriticalSection" ||
            fn == "DeleteCriticalSection" ||
            fn == "InitializeCriticalSection" ||
            fn == "Sleep" ||
            fn == "SetLastError" ||
            fn == "OutputDebugStringA" ||
            fn == "OutputDebugStringW" ||
            fn == "GetStartupInfoA" ||
            fn == "GetStartupInfoW" ||
            fn == "GetSystemTimeAsFileTime" ||
            fn == "GetSystemInfo" ||
            fn == "GetNativeSystemInfo" ||
            fn == "RtlCaptureContext" ||
            fn == "AcquireSRWLockExclusive" ||
            fn == "ReleaseSRWLockExclusive" ||
            fn == "AcquireSRWLockShared" ||
            fn == "ReleaseSRWLockShared" ||
            fn == "WakeAllConditionVariable" ||
            fn == "WakeConditionVariable" ||
            fn == "WakeByAddress" ||
            fn == "WakeByAddressSingle" ||
            fn == "WakeByAddressAll" ||
            fn == "InitializeConditionVariable" ||
            fn == "InitializeSListHead" ||
            fn == "InitializeSRWLock" ||
            fn == "GetCurrentProcessorNumberEx" ||
            fn == "FlushProcessWriteBuffers" ||
            fn == "PostQuitMessage");
}

uint32_t win_FormatMessageA(uint32_t dwFlags, const void* lpSource, uint32_t dwMessageId, uint32_t, char* lpBuffer,
                            uint32_t nSize, void*);

// Datos de un trampolín resueltos una sola vez (nunca se liberan: punteros estables).
struct ThunkInfo {
    std::string name;
    const std::string* stubName = nullptr;
    bool noisy = false;
    bool isVoid = false;
    mutable std::atomic<uint32_t> calls{0};
    // Llamada de 32 bits ya resuelta (se calcula una vez por función; antes, en cada llamada: nombre en minúsculas,
    // búsquedas de aridad y de stub... y el hilo de render de DmC hace >200 000 llamadas D3D9 por segundo).
    mutable std::atomic<bool> d32Ready{false};
    mutable uintptr_t d32Host = 0;
    mutable w32arity::Arity d32Ar{};
    mutable bool d32Known = false, d32Stdcall = false;
    mutable const std::string* d32Stub = nullptr;
    mutable uint32_t d32StubRet = 0;
    mutable std::atomic<uint64_t> total{0};      // llamadas totales (diagnóstico de rendimiento)
    mutable uint64_t lastReported = 0;
};

// Nombre de función "ruidosa" (millones de llamadas): no se registra en logcat.
static bool computeNoisy(const std::string& callName) {
    std::string_view fn = callName;
    const size_t bang = fn.rfind('!');
    if (bang != std::string_view::npos) fn.remove_prefix(bang + 1);
    static const std::unordered_set<std::string_view> kNoisy = {
        "WaitOnAddress", "HeapAlloc", "AcquireSRWLockExclusive", "ReleaseSRWLockExclusive",
        "AcquireSRWLockShared", "ReleaseSRWLockShared", "TryAcquireSRWLockExclusive",
        "TlsGetValue", "TlsSetValue", "FlsGetValue", "FlsSetValue", "GetLastError", "SetLastError",
        "QueryPerformanceCounter", "QueryPerformanceFrequency", "EnterCriticalSection",
        "LeaveCriticalSection", "TryEnterCriticalSection", "VirtualQuery", "SuspendThread",
        "ResumeThread", "GetThreadContext", "HeapFree", "HeapReAlloc", "HeapSize", "Sleep", "SleepEx",
        "XInputGetState", "timeGetTime", "GetTickCount64", "GetCurrentThreadId"};
    return kNoisy.count(fn) != 0;
}

// --- Vigilante de bloqueos: qué thunk ejecuta cada hilo y desde cuándo -------------------------
struct ThunkInFlight {
    std::atomic<const ThunkInfo*> info{nullptr};
    std::atomic<uint64_t> since{0};  // ms monotónicos
    std::atomic<uint64_t> ret{0};
    std::atomic<uint64_t> a0{0}, a1{0};
};
static std::mutex s_inFlightMutex;
static std::unordered_map<int, ThunkInFlight*> s_inFlight;

static uint64_t monoMs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000u + static_cast<uint64_t>(ts.tv_nsec) / 1000000u;
}

// Perfil de esperas en tiempo real (`setprop debug.hyperplay.waitprof <s>`): cada 1 ms anota en
// qué está cada hilo invitado: código JIT, un thunk (nombre) o una llamada nativa directa
// bloqueante. Muestra qué espera el hilo principal cuando no está saturado.
static std::atomic<bool> s_waitProfOn{false};
static std::mutex s_signalerMutex;
static std::unordered_map<uintptr_t, std::unordered_map<int, uint32_t>> s_signalers;
void noteSignaler(uintptr_t handle) {
    if (!s_waitProfOn.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(s_signalerMutex);
    ++s_signalers[handle][static_cast<int>(gettid())];
}

static void logSignalers() {
    std::lock_guard<std::mutex> lock(s_signalerMutex);
    std::vector<std::pair<uint32_t, uintptr_t>> v;
    for (auto& [h, m] : s_signalers) {
        uint32_t n = 0;
        for (auto& [t, c] : m) n += c;
        v.emplace_back(n, h);
    }
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 8; ++i) {
        std::string line;
        for (auto& [t, c] : s_signalers[v[i].second]) {
            char b[48];
            std::snprintf(b, sizeof(b), " tid%d=%u", t, c);
            line += b;
        }
        LOGI("[Señales] h=0x%llx total=%u:%s", static_cast<unsigned long long>(v[i].second), v[i].first, line.c_str());
    }
    s_signalers.clear();
}

// Captura de tirones (`setprop debug.hyperplay.stallprof 1`): un muestreador cada 2 ms guarda
// en un anillo qué hace cada hilo; cuando un frame pasa de 100 ms, Present pide el desglose de
// esa ventana y se registra "[Tirón]" con lo que hizo cada hilo mientras el frame no salía.
static std::atomic<uint32_t> s_suspendCalls{0};
static std::atomic<uint64_t> s_readFileBytes{0};
namespace stallprof {
struct Sample {
    uint32_t ms;
    int32_t tid;
    uint32_t key;
};
constexpr size_t kRing = 1u << 16;
static Sample s_ring[kRing];
static std::atomic<uint64_t> s_head{0};
static std::atomic<bool> s_on{false};
static std::mutex s_reqMutex;
static std::condition_variable s_reqCv;
static std::vector<std::pair<uint32_t, uint32_t>> s_requests;  // ventanas [desde, hasta] ms
static std::vector<std::string> s_keys;
static std::unordered_map<std::string, uint32_t> s_keyIdx;

static uint32_t intern(const std::string& k) {
    auto it = s_keyIdx.find(k);
    if (it != s_keyIdx.end()) return it->second;
    s_keys.push_back(k);
    return s_keyIdx[k] = static_cast<uint32_t>(s_keys.size() - 1);
}

void requestWindow(uint32_t fromMs, uint32_t toMs) {
    if (!s_on.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(s_reqMutex);
    s_requests.emplace_back(fromMs, toMs);
    s_reqCv.notify_one();
}
}  // namespace stallprof

static std::string classifyThread(int tid, ThunkInFlight* s, std::unordered_map<uintptr_t, std::string>& names,
                                  bool withRip) {
    const ThunkInfo* info = s->info.load(std::memory_order_acquire);
    if (info != nullptr) {
        std::string key = info->name;
        if (key.find("Wait") != std::string::npos || key.find("Sleep") != std::string::npos) {
            char b[64];
            std::snprintf(b, sizeof(b), "@0x%llx(h=0x%llx)", static_cast<unsigned long long>(s->ret.load()),
                          static_cast<unsigned long long>(s->a0.load()));
            key += b;
        }
        return key;
    }
    auto* b = OryonEmitterBackend::forThread(tid);
    if (b == nullptr) return "(?)";
    if (b->runState() == OryonEmitterBackend::kInNative) {
        const uintptr_t fn = static_cast<uintptr_t>(b->context().nativeFn);
        auto it = names.find(fn);
        if (it == names.end()) it = names.emplace(fn, "directa:" + findTrampolineName(fn)).first;
        return it->second;
    }
    if (b->runState() == OryonEmitterBackend::kParked) return "(suspendido)";
    if (!withRip) return "(JIT)";
    char r[48];
    std::snprintf(r, sizeof(r), "(JIT@0x%llx)", static_cast<unsigned long long>(b->context().rip & ~0xFFFull));
    return r;
}

static void stallSamplerLoop() {
    using namespace stallprof;
    std::unordered_map<uintptr_t, std::string> names;
    uint32_t lastSusp = s_suspendCalls.load();
    uint64_t lastRead = s_readFileBytes.load();
    for (;;) {
        const uint32_t now = static_cast<uint32_t>(monoMs());
        {
            std::lock_guard<std::mutex> lock(s_inFlightMutex);
            for (const auto& [tid, s] : s_inFlight) {
                const uint32_t k = intern(classifyThread(tid, s, names, true));
                const uint64_t h = s_head.fetch_add(1, std::memory_order_relaxed);
                s_ring[h & (kRing - 1)] = Sample{now, tid, k};
            }
        }
        std::vector<std::pair<uint32_t, uint32_t>> reqs;
        {
            std::unique_lock<std::mutex> lock(s_reqMutex);
            s_reqCv.wait_for(lock, std::chrono::milliseconds(2), [] { return !s_requests.empty(); });
            reqs.swap(s_requests);
        }
        for (auto [from, to] : reqs) {
            const uint32_t susp = s_suspendCalls.load();
            const uint64_t rd = s_readFileBytes.load();
            std::unordered_map<int, std::unordered_map<uint32_t, uint32_t>> hist;
            const uint64_t head = s_head.load();
            const uint64_t lo = head > kRing ? head - kRing : 0;
            for (uint64_t i = head; i-- > lo;) {
                const Sample& sm = s_ring[i & (kRing - 1)];
                if (sm.ms < from) break;
                if (sm.ms <= to) ++hist[sm.tid][sm.key];
            }
            LOGW("[Tirón] frame de %u ms | SuspendThread +%u | ReadFile +%llu KB", to - from, susp - lastSusp,
                 static_cast<unsigned long long>((rd - lastRead) / 1024));
            lastSusp = susp;
            lastRead = rd;
            // Hilos con menos espera primero (los que trabajaban durante el tirón).
            std::vector<std::pair<uint32_t, int>> order;
            for (auto& [tid, m] : hist) {
                uint32_t busy = 0;
                for (auto& [k, c] : m) {
                    const std::string& n = s_keys[k];
                    if (n.find("Wait") == std::string::npos && n.find("Sleep") == std::string::npos &&
                        n.find("GetQueuedCompletion") == std::string::npos)
                        busy += c;
                }
                order.emplace_back(busy, tid);
            }
            std::sort(order.rbegin(), order.rend());
            for (size_t i = 0; i < order.size() && i < 5; ++i) {
                auto& m = hist[order[i].second];
                uint32_t total = 0;
                for (auto& [k, c] : m) total += c;
                std::vector<std::pair<uint32_t, uint32_t>> v;
                for (auto& [k, c] : m) v.emplace_back(c, k);
                std::sort(v.rbegin(), v.rend());
                std::string line;
                for (size_t j = 0; j < v.size() && j < 5; ++j) {
                    char b[200];
                    std::snprintf(b, sizeof(b), " %s=%.0f%%", s_keys[v[j].second].c_str(),
                                  100.0 * v[j].first / std::max(1u, total));
                    line += b;
                }
                LOGW("[Tirón]   tid=%d:%s", order[i].second, line.c_str());
            }
        }
        if (s_keys.size() > 50000) {  // evita crecer sin límite con RIPs distintos
            s_keys.clear();
            s_keyIdx.clear();
            s_head.store(0);
        }
    }
}

static void runWaitProfile(int seconds) {
    s_waitProfOn.store(true);
    std::unordered_map<int, std::unordered_map<std::string, uint32_t>> hist;
    std::unordered_map<uintptr_t, std::string> names;
    const uint64_t end = monoMs() + static_cast<uint64_t>(seconds) * 1000u;
    while (monoMs() < end) {
        {
            std::lock_guard<std::mutex> lock(s_inFlightMutex);
            for (const auto& [tid, s] : s_inFlight) {
                const ThunkInfo* info = s->info.load(std::memory_order_acquire);
                std::string key;
                if (info != nullptr) {
                    key = info->name;
                    if (key.find("Wait") != std::string::npos) {
                        char b[64];
                        std::snprintf(b, sizeof(b), "@0x%llx(h=0x%llx)", static_cast<unsigned long long>(s->ret.load()),
                                      static_cast<unsigned long long>(s->a0.load()));
                        key += b;
                    }
                } else if (auto* b = OryonEmitterBackend::forThread(tid); b != nullptr) {
                    if (b->runState() == OryonEmitterBackend::kInNative) {
                        const uintptr_t fn = static_cast<uintptr_t>(b->context().nativeFn);
                        auto it = names.find(fn);
                        if (it == names.end()) it = names.emplace(fn, "directa:" + findTrampolineName(fn)).first;
                        key = it->second;
                    } else if (b->runState() == OryonEmitterBackend::kParked) {
                        key = "(suspendido)";
                    } else {
                        key = "(JIT)";
                    }
                } else {
                    key = "(?)";
                }
                ++hist[tid][key];
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    s_waitProfOn.store(false);
    logSignalers();
    std::vector<std::pair<uint32_t, int>> order;
    for (auto& [tid, m] : hist) {
        uint32_t busy = 0;
        for (auto& [k, c] : m) if (k == "(JIT)") busy = c;
        order.emplace_back(busy, tid);
    }
    std::sort(order.rbegin(), order.rend());
    for (size_t i = 0; i < order.size() && i < 5; ++i) {
        auto& m = hist[order[i].second];
        uint32_t total = 0;
        for (auto& [k, c] : m) total += c;
        std::vector<std::pair<uint32_t, std::string>> v;
        for (auto& [k, c] : m) v.emplace_back(c, k);
        std::sort(v.rbegin(), v.rend());
        std::string line;
        for (size_t j = 0; j < v.size() && j < 6; ++j) {
            char b[160];
            std::snprintf(b, sizeof(b), " %s=%.0f%%", v[j].second.c_str(), 100.0 * v[j].first / std::max(1u, total));
            line += b;
        }
        LOGI("[Esperas] tid=%d:%s", order[i].second, line.c_str());
    }
}

static ThunkInFlight& myInFlight() {
    thread_local ThunkInFlight* t_slot = [] {
        auto* s = new ThunkInFlight();  // vive mientras viva el proceso (el vigilante lo lee)
        std::lock_guard<std::mutex> lock(s_inFlightMutex);
        s_inFlight[static_cast<int>(gettid())] = s;
        return s;
    }();
    return *t_slot;
}

static void logTopThunks(uint64_t elapsedMs);

namespace hoststack {
void dump(int tid);
}

static void startStallWatchdog() {
    static std::once_flag s_once;
    std::call_once(s_once, [] {
        std::thread([] {
            char last[PROP_VALUE_MAX] = {};
            __system_property_get("debug.hyperplay.waitprof", last);
            // Volcado de pilas: solo ante un valor NUEVO tras arrancar (uno viejo que quedó puesto
            // no debe interrumpir cada arranque; el desenrollado sobre código JIT es delicado).
            static char s_lastDump[PROP_VALUE_MAX] = {};
            __system_property_get("debug.hyperplay.dumpstacks", s_lastDump);
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                char now[PROP_VALUE_MAX] = {};
                __system_property_get("debug.hyperplay.waitprof", now);
                if (std::strcmp(now, last) != 0) {
                    std::memcpy(last, now, sizeof(last));
                    const int s = std::atoi(now);
                    if (s > 0 && s <= 120) runWaitProfile(s);
                }
                // setprop debug.hyperplay.dumpstacks <n> (cada valor nuevo): estado, RIP, pila
                // invitada y pila nativa de todos los hilos del juego (cuelgues sin thunk largo).
                char dp[PROP_VALUE_MAX] = {};
                __system_property_get("debug.hyperplay.dumpstacks", dp);
                if (dp[0] != 0 && std::strcmp(dp, s_lastDump) != 0) {
                    std::memcpy(s_lastDump, dp, sizeof(s_lastDump));
                    std::vector<int> tids;
                    {
                        std::lock_guard<std::mutex> lock(s_inFlightMutex);
                        for (const auto& [tid, s] : s_inFlight) tids.push_back(tid);
                    }
                    for (int tid : tids) {
                        OryonEmitterBackend* b = OryonEmitterBackend::forThread(tid);
                        if (b == nullptr) continue;
                        LOGW("[Volcado] tid=%d estado=%d RIP=0x%llx", tid, b->runState(),
                             static_cast<unsigned long long>(b->context().rip));
                        char tag[40];
                        std::snprintf(tag, sizeof(tag), "[Volcado] tid=%d", tid);
                        b->logGuestStackChain(tag);
                        hoststack::dump(tid);
                    }
                }
                char sp[PROP_VALUE_MAX] = {};
                __system_property_get("debug.hyperplay.stallprof", sp);
                if (sp[0] == '1' && !stallprof::s_on.exchange(true)) {
                    LOGI("[Tirón] captura de tirones activa");
                    std::thread(stallSamplerLoop).detach();
                }
            }
        }).detach();
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(15));
                logTopThunks(15000);
                const uint64_t now = monoMs();
                std::lock_guard<std::mutex> lock(s_inFlightMutex);
                for (const auto& [tid, s] : s_inFlight) {
                    const ThunkInfo* info = s->info.load(std::memory_order_acquire);
                    const uint64_t since = s->since.load(std::memory_order_acquire);
                    if (info == nullptr) {
                        // Fuera de thunks: estado del despachador (0 ejecutando, 2 aparcado por
                        // SuspendThread) y RIP actual; detecta hilos suspendidos sin reanudar.
                        if (auto* b = OryonEmitterBackend::forThread(tid); b != nullptr && now - since > 5000) {
                            LOGW("[Bloqueo] tid=%d fuera de thunk %llu ms: estado=%d suspensiones=%u RIP=0x%llx",
                                 tid, static_cast<unsigned long long>(now - since), b->runState(), b->suspendCount(),
                                 static_cast<unsigned long long>(b->context().rip));
                        }
                        continue;
                    }
                    if (since == 0 || now - since < 5000) continue;
                    LOGW("[Bloqueo] tid=%d lleva %llu ms en %s(0x%llx, 0x%llx) desde RIP=0x%llx", tid,
                         static_cast<unsigned long long>(now - since), info->name.c_str(),
                         static_cast<unsigned long long>(s->a0.load()), static_cast<unsigned long long>(s->a1.load()),
                         static_cast<unsigned long long>(s->ret.load()));
                    // Una vez por hilo: cadena de retornos del huésped para ver qué espera.
                    static std::unordered_set<int> s_stackDumped;
                    if (now - since > 20000 && s_stackDumped.insert(tid).second) {
                        if (auto* b = OryonEmitterBackend::forThread(tid)) {
                            char tag[48];
                            std::snprintf(tag, sizeof(tag), "[Bloqueo] tid=%d", tid);
                            b->logGuestStackChain(tag);
                        }
                    }
                }
            }
        }).detach();
    });
}

static std::mutex s_infoMutex;
static std::unordered_map<uintptr_t, std::unique_ptr<ThunkInfo>> s_infos;

// Muestreo (1/64) de desde dónde se llama a cada thunk: qué patrón de llamada no resuelve el JIT.
static std::mutex s_callerMutex;
static std::unordered_map<std::string, std::unordered_map<uint64_t, uint32_t>> s_callers;
static void sampleThunkCaller(const std::string& name, uintptr_t target, uint64_t ret) {
    thread_local uint32_t t_n = 0;
    if ((++t_n & 63) != 0) return;
    char key[200];
    std::snprintf(key, sizeof(key), "%s@0x%llx", name.c_str(), static_cast<unsigned long long>(target));
    std::lock_guard<std::mutex> lock(s_callerMutex);
    ++s_callers[key][ret];
}
static void logThunkCallers() {
    std::lock_guard<std::mutex> lock(s_callerMutex);
    for (auto& [name, m] : s_callers) {
        std::vector<std::pair<uint32_t, uint64_t>> v;
        for (auto& [r, c] : m) v.emplace_back(c, r);
        std::sort(v.rbegin(), v.rend());
        std::string line;
        for (size_t i = 0; i < v.size() && i < 6; ++i) {
            char b[48];
            std::snprintf(b, sizeof(b), " 0x%llx(%u)", static_cast<unsigned long long>(v[i].second), v[i].first);
            line += b;
        }
        if (v.size() > 0 && v[0].first >= 50) LOGI("[Llamantes] %s:%s", name.c_str(), line.c_str());
    }
    s_callers.clear();
}

// Funciones nativas más llamadas desde el último informe (llamadas/s): dónde sale el JIT.
static void logTopThunks(uint64_t elapsedMs) {
    if (elapsedMs == 0) return;
    std::vector<std::pair<uint64_t, const ThunkInfo*>> top;
    {
        std::lock_guard<std::mutex> lock(s_infoMutex);
        for (auto& [addr, info] : s_infos) {
            const uint64_t t = info->total.load(std::memory_order_relaxed);
            const uint64_t d = t - info->lastReported;
            info->lastReported = t;
            if (d > 0) top.emplace_back(d, info.get());
        }
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string line;
    static const size_t s_topN = [] {  // setprop debug.hyperplay.thunktop N (por defecto 12)
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.thunktop", v) > 0 ? static_cast<size_t>(std::atoi(v)) : size_t{12};
    }();
    for (size_t i = 0; i < top.size() && i < s_topN; ++i) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), " %s=%llu/s", top[i].second->name.c_str(),
                      static_cast<unsigned long long>(top[i].first * 1000 / elapsedMs));
        line += buf;
    }
    if (!line.empty()) LOGI("[Thunks/s]%s", line.c_str());
    // Entrada y ventana aparte (no suelen entrar en el top 12): diagnostica mandos/ratón que no responden.
    std::string input;
    for (const auto& [n, info] : top) {
        const std::string& nm = info->name;
        if (nm.find("XInput") == std::string::npos && nm.find("PeekMessage") == std::string::npos &&
            nm.find("Cursor") == std::string::npos && nm.find("KeyState") == std::string::npos) continue;
        char buf[160];
        std::snprintf(buf, sizeof(buf), " %s=%llu/s", nm.c_str(), static_cast<unsigned long long>(n * 1000 / elapsedMs));
        input += buf;
    }
    if (!input.empty()) LOGI("[Entrada/s]%s", input.c_str());
    logThunkCallers();
}

static const ThunkInfo* thunkInfoFor(uintptr_t target) {
    thread_local std::unordered_map<uintptr_t, const ThunkInfo*> t_cache;
    if (auto it = t_cache.find(target); it != t_cache.end()) return it->second;
    const ThunkInfo* result = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_infoMutex);
        auto& slot = s_infos[target];
        if (!slot) {
            slot = std::make_unique<ThunkInfo>();
            slot->name = findTrampolineName(target);
            slot->stubName = DynamicStubManager::findStubName(target);
            if (slot->name.empty() && slot->stubName != nullptr) slot->name = *slot->stubName;
            slot->noisy = computeNoisy(slot->name);
            slot->isVoid = isVoidWin32Function(slot->name);
        }
        result = slot.get();
    }
    t_cache.emplace(target, result);
    return result;
}

// Unity llama a MemoryManager millones de veces: solo las 300 primeras van a logcat (si no,
// expulsan del búfer todos los demás diagnósticos).
static bool unityHookLogBudget() {
    static std::atomic<uint32_t> s_count{0};
    return s_count.fetch_add(1, std::memory_order_relaxed) < 300;
}

// Aridad stdcall (palabras de 32 bits; DWORD64 = 2) y retorno de los stubs de 32 bits de funciones de sistema
// que no implementamos. Un stub "cdecl" en una función stdcall deja la pila del llamador descuadrada.
static bool stub32Arity(const std::string& fn, uint32_t& slots, uint32_t& ret) {
    struct E { uint8_t slots; uint32_t ret; };
    constexpr uint32_t kInvalid = 0xFFFFFFFFu, kFail = 0x80004005u;
    static const std::unordered_map<std::string, E> s_table = {
        // KERNEL32
        {"SetConsoleTitleW", {1, 1}}, {"SetConsoleTitleA", {1, 1}}, {"PulseEvent", {1, 1}}, {"TerminateThread", {2, 1}},
        {"CreateProcessW", {10, 0}}, {"CreateProcessA", {10, 0}}, {"SetDllDirectoryW", {1, 1}}, {"SetDllDirectoryA", {1, 1}},
        {"ReadDirectoryChangesW", {8, 0}}, {"GetTempFileNameW", {4, 0}}, {"GetTempFileNameA", {4, 0}},
        {"SetProcessAffinityMask", {2, 1}}, {"FindFirstFileA", {2, kInvalid}}, {"FindNextFileA", {2, 0}},
        {"SetConsoleTextAttribute", {2, 1}}, {"AllocConsole", {0, 0}}, {"FreeConsole", {0, 1}},
        {"SetConsoleScreenBufferSize", {2, 1}}, {"CompareStringA", {6, 2}}, {"SetPriorityClass", {2, 1}},
        {"GetPriorityClass", {1, 0x20}}, {"GetFileTime", {4, 0}}, {"RtlCaptureStackBackTrace", {4, 0}},
        {"GetFullPathNameA", {4, 0}}, {"GlobalSize", {1, 0}},
        // GDI32
        {"GetTextMetricsW", {2, 0}}, {"CreateSolidBrush", {1, 0}}, {"GetGlyphOutlineW", {7, kInvalid}}, {"BitBlt", {9, 0}},
        {"GetDIBits", {7, 0}}, {"SetTextColor", {2, 0}}, {"TextOutW", {5, 0}}, {"GetTextExtentPoint32W", {4, 0}},
        {"SetBkMode", {2, 0}}, {"CreateFontW", {14, 0}}, {"GetStockObject", {1, 0}}, {"CreateFontIndirectW", {1, 0}},
        {"GetGlyphIndicesW", {5, kInvalid}}, {"SetBkColor", {2, kInvalid}}, {"GetPixel", {3, kInvalid}},
        {"CreateCompatibleBitmap", {3, 0}},
        // ole32
        {"StringFromCLSID", {2, kFail}}, {"CoSetProxyBlanket", {8, 0}}, {"OleGetClipboard", {1, kFail}},
        {"ReleaseStgMedium", {1, 0}},
        // dbghelp (solo informes de fallo)
        {"SymGetSymFromAddr64", {5, 0}}, {"SymGetLineFromAddr64", {5, 0}}, {"SymGetModuleInfo64", {4, 0}},
        {"SymSetSearchPath", {2, 0}}, {"SymGetModuleBase64", {3, 0}}, {"SymLoadModule64", {7, 0}}, {"SymSetOptions", {1, 0}},
        {"SymFunctionTableAccess64", {3, 0}}, {"StackWalk64", {9, 0}}, {"SymInitialize", {3, 0}},
        {"MiniDumpWriteDump", {7, 0}}, {"SymGetOptions", {0, 0}},
        // CRYPT32, SHELL32, ADVAPI32, IPHLPAPI
        {"CryptProtectData", {7, 0}}, {"CryptUnprotectData", {7, 0}}, {"Shell_NotifyIconW", {2, 0}},
        {"RegEnumKeyExA", {8, 259}}, {"RegConnectRegistryA", {3, 2}}, {"RegQueryInfoKeyA", {12, 2}},
        {"RegEnumValueA", {8, 259}}, {"GetNamedSecurityInfoA", {8, 2}}, {"LookupAccountSidA", {7, 0}},
        {"GetUserNameW", {2, 0}}, {"GetUserNameA", {2, 0}}, {"GetAdaptersInfo", {2, 232}},  // ERROR_NO_DATA
        // USER32
        {"ShowOwnedPopups", {2, 1}}, {"SendInput", {3, 0}}, {"GetDlgItem", {2, 0}}, {"UnregisterHotKey", {2, 1}},
        {"RegisterClipboardFormatA", {1, 0xC000}}, {"BringWindowToTop", {1, 1}}, {"EnableWindow", {2, 0}},
        {"RegisterHotKey", {4, 1}}, {"GetTopWindow", {1, 0}}, {"FindWindowExA", {4, 0}}, {"MapWindowPoints", {4, 0}},
        {"SetRectEmpty", {1, 1}}, {"EndPaint", {2, 1}}, {"DrawStateW", {10, 1}}, {"BeginPaint", {2, 0}},
        {"GetWindowThreadProcessId", {2, 0}}, {"GetWindow", {2, 0}}, {"IsWindow", {1, 1}}, {"FillRect", {3, 1}},
        {"GetCapture", {0, 0}}, {"DrawFocusRect", {2, 1}}, {"InflateRect", {3, 1}}, {"ChangeDisplaySettingsW", {2, 0}},
        {"GetKeyboardLayoutNameA", {1, 0}}, {"ActivateKeyboardLayout", {2, 0}}, {"GetKeyboardLayoutList", {2, 0}},
        {"GetWindowInfo", {2, 0}}, {"ToUnicodeEx", {7, 0}}, {"GetKeyboardState", {1, 0}},
        // d3dx9_43 (WINAPI): sin compilador HLSL ni utilidades de malla
        {"D3DXDebugMute", {1, 0}}, {"D3DXPreprocessShader", {6, kFail}}, {"D3DXCompileShader", {10, kFail}},
        {"D3DXCreateMesh", {6, kFail}}, {"D3DXCreateBuffer", {2, kFail}}, {"D3DXDisassembleShader", {4, kFail}},
        {"D3DXCleanMesh", {7, kFail}},
    };
    const auto it = s_table.find(fn);
    if (it == s_table.end()) return false;
    slots = it->second.slots;
    ret = it->second.ret;
    return true;
}

bool NativeDllInterceptor::dispatchThunk(void* user, uintptr_t target, CpuContext& ctx) {
    if (!isRegisteredTrampoline(target)) {
        return false;
    }

    // --- Invitado de 32 bits (WoW64 propio): argumentos en la pila y limpieza stdcall ---------------
    if (ctx.mode32 != 0) {
        const uint32_t esp = static_cast<uint32_t>(ctx.gpr[4]);
        if (esp == 0) return false;
        uint32_t ret32 = 0;
        std::memcpy(&ret32, reinterpret_cast<const void*>(static_cast<uintptr_t>(esp)), 4);
        const ThunkInfo* info = thunkInfoFor(target);
        info->total.fetch_add(1, std::memory_order_relaxed);
        const std::string& name = info->name;
        sampleThunkCaller(name, target, ret32);  // informe [Llamantes] (1 de cada 64)
        const size_t bang = name.rfind('!');
        if (!info->d32Ready.load(std::memory_order_acquire)) {
            const uintptr_t h = w32alias::hostFor(target);
            const std::string dllName = toLower(bang == std::string::npos ? std::string() : name.substr(0, bang));
            // CRT y variádicas: cdecl (limpia el llamador). El resto de la API de Windows: stdcall.
            const bool crt = dllName.find("msvcr") != std::string::npos || dllName.find("msvcp") != std::string::npos ||
                             dllName.find("ucrtbase") != std::string::npos || dllName.find("api-ms-win-crt") != std::string::npos ||
                             dllName == "msvcrt.dll" || dllName.find("steam_api") != std::string::npos;
            w32arity::Arity a{};
            const bool k = w32arity::lookup(reinterpret_cast<const void*>(h), a);
            const std::string* sn = info->stubName != nullptr ? info->stubName : DynamicStubManager::findStubName(h);
            bool sc = k && sn == nullptr && !a.variadic && !crt;
            uint32_t ss = 0, sr = 0;
            if (sn != nullptr && !crt && stub32Arity(bang == std::string::npos ? name : name.substr(bang + 1), ss, sr)) {
                sc = true;
                a.slots = static_cast<uint8_t>(ss);
            }
            info->d32Host = h;
            info->d32Ar = a;
            info->d32Known = k;
            info->d32Stdcall = sc;
            info->d32Stub = sn;
            info->d32StubRet = sr;
            info->d32Ready.store(true, std::memory_order_release);
        }
        const uintptr_t host = info->d32Host;
        w32arity::Arity ar = info->d32Ar;
        const bool known = info->d32Known;
        const std::string* stubName = info->d32Stub;
        if ((!known || stubName != nullptr) && info->calls.load(std::memory_order_relaxed) == 0) {
            static std::mutex s_m;
            static std::unordered_set<std::string> s_warned;
            std::lock_guard<std::mutex> lk(s_m);
            if (s_warned.insert(name).second)
                LOGW("[32 bits] %s: aridad desconocida%s (se trata como cdecl)", name.c_str(),
                     stubName != nullptr ? " (stub)" : "");
        }
        win64abi::ThunkArgs args{};
        uint32_t slot = 0;
        const size_t nargs = known ? ar.args : 8;
        for (size_t i = 0; i < nargs && i < args.size(); ++i) {
            uint32_t lo = 0, hi = 0;
            std::memcpy(&lo, reinterpret_cast<const void*>(static_cast<uintptr_t>(esp) + 4 + 4 * slot), 4);
            ++slot;
            if (known && (ar.wideMask & (1u << i)) != 0) {
                std::memcpy(&hi, reinterpret_cast<const void*>(static_cast<uintptr_t>(esp) + 4 + 4 * slot), 4);
                ++slot;
            }
            args[i] = (static_cast<uint64_t>(hi) << 32) | lo;
        }
        // Stubs de la API de sistema sin implementar: aridad stdcall de la tabla (ver stub32Arity) y retorno de
        // "fallo limpio"; ya resueltos en d32*.
        const bool stdcall = info->d32Stdcall;
        const uint32_t stubRet = info->d32StubRet;
        // Llamadas registradas por función (setprop debug.hyperplay.thunklog N; por defecto 4).
        static const uint32_t s_logMax = [] {
            char v[PROP_VALUE_MAX] = {};
            return __system_property_get("debug.hyperplay.thunklog", v) > 0 ? static_cast<uint32_t>(std::atoi(v)) : 4u;
        }();
        // Además, todas las llamadas desde un rango de código (setprop debug.hyperplay.thunkrange lo,hi en hex; máx. 40000).
        static const std::pair<uint32_t, uint32_t> s_range = [] {
            char v[PROP_VALUE_MAX] = {};
            if (__system_property_get("debug.hyperplay.thunkrange", v) <= 0) return std::pair<uint32_t, uint32_t>{0, 0};
            char* e = nullptr;
            const uint32_t lo = static_cast<uint32_t>(std::strtoul(v, &e, 16));
            const uint32_t hi = (e != nullptr && *e == ',') ? static_cast<uint32_t>(std::strtoul(e + 1, nullptr, 16)) : 0;
            return std::pair<uint32_t, uint32_t>{lo, hi};
        }();
        static std::atomic<int> s_rangeLogs{0};
        const bool inRange = ret32 >= s_range.first && ret32 < s_range.second && s_rangeLogs.fetch_add(1) < 40000;
        if (info->calls.fetch_add(1, std::memory_order_relaxed) < s_logMax || inRange) {
            LOGI("[NativeThunk32] %s(0x%llx, 0x%llx, 0x%llx, 0x%llx) desde EIP=0x%x (%s, %u slots)", name.c_str(),
                 static_cast<unsigned long long>(args[0]), static_cast<unsigned long long>(args[1]),
                 static_cast<unsigned long long>(args[2]), static_cast<unsigned long long>(args[3]), ret32,
                 stdcall ? "stdcall" : "cdecl", known ? ar.slots : 0u);
        }
        uint64_t ret = 0;
        if (stubName != nullptr) {
            ret = stubRet;
        } else {
            t_dispatchBackend = static_cast<OryonEmitterBackend*>(user);
            CpuContext* const prevThunkCtx = t_thunkCtx;
            const uint64_t prevThunkRet = t_thunkRet;
            t_thunkCtx = &ctx;
            t_thunkRet = ret32;
            // Mismo vigilante que en 64 bits: llamadas nativas largas y volcado de pilas (dumpstacks).
            startStallWatchdog();
            ThunkInFlight& inFlight = myInFlight();
            inFlight.ret.store(ret32, std::memory_order_relaxed);
            inFlight.a0.store(args[0], std::memory_order_relaxed);
            inFlight.a1.store(args[1], std::memory_order_relaxed);
            inFlight.since.store(monoMs(), std::memory_order_relaxed);
            inFlight.info.store(info, std::memory_order_release);
            ret = win64abi::invokeThunk(host, args);
            inFlight.info.store(nullptr, std::memory_order_release);
            t_thunkCtx = prevThunkCtx;
            t_thunkRet = prevThunkRet;
            if (t_ctxOverride) {
                t_ctxOverride = false;
                ctx = t_ctxOverrideValue;
                return true;
            }
        }
        if (info->calls.load(std::memory_order_relaxed) <= s_logMax || inRange) {
            LOGI("[NativeThunk32] %s -> 0x%llx%s", name.c_str(), static_cast<unsigned long long>(ret),
                 ret > 0xFFFFFFFFull && !info->isVoid ? " (¡>4GB: truncado!)" : "");
        }
        if (!info->isVoid) {
            ctx.gpr[0] = ret & 0xFFFFFFFFull;          // EAX
            ctx.gpr[2] = (ret >> 32) & 0xFFFFFFFFull;  // EDX (retornos de 64 bits)
        }
        ctx.gpr[4] = static_cast<uint64_t>(esp) + 4 + (stdcall ? 4u * ar.slots : 0u);
        ctx.rip = ret32;
        if (t_exitThreadRequested) {
            // ExitThread no vuelve (como en 64 bits): RIP=0 termina el hilo con el código en EAX. Antes el hilo
            // seguía tras la llamada, caía en el relleno int3 y quien esperaba su handle no despertaba (PhysX en DmC).
            t_exitThreadRequested = false;
            ctx.gpr[0] = t_exitThreadCode;
            ctx.rip = 0;
        }
        return true;
    }

    // --- Hooks internos de Unity MemoryManager (evitan asignaciones nulas para label=1 kMemTempAlloc) ---
    if (target == s_unityMemoryManagerAllocate) {
        const uint16_t label = static_cast<uint16_t>(ctx.gpr[9]);
        if (label == 1) { // kMemTempAlloc
            const uintptr_t oldRsp = ctx.gpr[4];
            if (oldRsp == 0) return false;

            uint64_t returnAddress = 0;
            std::memcpy(&returnAddress, reinterpret_cast<const void*>(oldRsp), sizeof(returnAddress));
            ctx.gpr[4] += 8;

            const size_t size = static_cast<size_t>(ctx.gpr[2]);
            void* ptr = safeHeapAlloc(size);
            if (unityHookLogBudget()) LOGI("[UnityHook] MemoryManager::Allocate(size=%zu, align=%llu, label=%u) -> %p",
                 size, static_cast<unsigned long long>(ctx.gpr[8]), label, ptr);

            ctx.gpr[0] = reinterpret_cast<uint64_t>(ptr);
            ctx.rip = returnAddress;
            return true;
        }
        return false;
    }

    if (target == s_unityMemoryManagerReallocate) {
        const uintptr_t oldRsp = ctx.gpr[4];
        if (oldRsp == 0) return false;

        auto* oryon = static_cast<OryonEmitterBackend*>(user);
        uint64_t labelVal = 0;
        if (oryon != nullptr && oryon->isValidStackAddress(oldRsp + 0x28, 8)) {
            std::memcpy(&labelVal, reinterpret_cast<const void*>(oldRsp + 0x28), sizeof(labelVal));
        }
        const uint16_t label = static_cast<uint16_t>(labelVal);
        void* oldPtr = reinterpret_cast<void*>(ctx.gpr[2]);
        const size_t newSize = static_cast<size_t>(ctx.gpr[8]);

        const bool isOurPtr = isSafeHeapPointer(oldPtr);
        if (label == 1 || isOurPtr) {
            uint64_t returnAddress = 0;
            std::memcpy(&returnAddress, reinterpret_cast<const void*>(oldRsp), sizeof(returnAddress));
            ctx.gpr[4] += 8;

            void* newPtr = safeHeapAlloc(newSize);
            if (oldPtr != nullptr && newPtr != nullptr) {
                size_t copySize = 0;
                if (isOurPtr) {
                    copySize = safeHeapSize(oldPtr);
                } else {
                    auto* hdr = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(oldPtr) - 8);
                    uint32_t possibleSize = (*hdr) >> 1;
                    if (possibleSize > 0 && possibleSize < 100 * 1024 * 1024) {
                        copySize = possibleSize;
                    } else {
                        copySize = newSize;
                    }
                }
                if (copySize > newSize) copySize = newSize;
                if (copySize > 0) {
                    std::memcpy(newPtr, oldPtr, copySize);
                }
                if (isOurPtr) {
                    safeHeapFree(oldPtr);
                }
            }

            if (unityHookLogBudget()) LOGI("[UnityHook] MemoryManager::Reallocate(ptr=%p, newSize=%zu, label=%u) -> %p",
                 oldPtr, newSize, label, newPtr);

            ctx.gpr[0] = reinterpret_cast<uint64_t>(newPtr);
            ctx.rip = returnAddress;
            return true;
        }
        return false;
    }

    if (target == s_unityMemoryManagerDeallocate) {
        void* ptr = reinterpret_cast<void*>(ctx.gpr[2]);
        if (isSafeHeapPointer(ptr)) {
            const uintptr_t oldRsp = ctx.gpr[4];
            if (oldRsp == 0) return false;

            uint64_t returnAddress = 0;
            std::memcpy(&returnAddress, reinterpret_cast<const void*>(oldRsp), sizeof(returnAddress));
            ctx.gpr[4] += 8;

            safeHeapFree(ptr);
            if (unityHookLogBudget()) LOGI("[UnityHook] MemoryManager::Deallocate(ptr=%p) -> freed", ptr);
            ctx.gpr[0] = 1;
            ctx.rip = returnAddress;
            return true;
        }
        return false;
    }

    const uintptr_t oldRsp = ctx.gpr[4];
    if (oldRsp == 0) {
        LOGW("dispatchThunk: RSP nulo en target %p", reinterpret_cast<void*>(target));
        return false;
    }

    // 1. Lee la dirección de retorno desde [RSP] y haz pop (RSP += 8)
    uint64_t returnAddress = 0;
    std::memcpy(&returnAddress, reinterpret_cast<const void*>(oldRsp), sizeof(returnAddress));
    ctx.gpr[4] += 8;

    // Nombre, stub y "ruidoso" se resuelven una vez por dirección (caché por hilo, sin locks):
    // antes cada thunk tomaba 2-3 mutex globales y hacía ~20 búsquedas de cadena.
    const ThunkInfo* info = thunkInfoFor(target);
    info->total.fetch_add(1, std::memory_order_relaxed);
    const std::string& callName = info->name;
    sampleThunkCaller(callName, target, returnAddress);
    const std::string* stubName = info->stubName;
    if (auto* b = static_cast<OryonEmitterBackend*>(user); b != nullptr && !callName.empty()) {
        b->recordThunkCall(callName, ctx.gpr[1], ctx.gpr[2]);
    }

    // ¿Es un stub dinámico de diagnóstico?
    if (stubName != nullptr) {
        // advapi32!Reg* devuelve LSTATUS: 0 sería ERROR_SUCCESS con un HKEY/valor sin rellenar.
        // Sin registro emulado, la respuesta honesta es ERROR_FILE_NOT_FOUND (2).
        const size_t bangPos = stubName->rfind('!');
        const bool isRegistry = bangPos != std::string::npos &&
                                stubName->compare(bangPos + 1, 3, "Reg") == 0 &&
                                toLower(*stubName).find("advapi32") != std::string::npos;
        // Familias COM/WinRT/Media Foundation/DirectSound/XAudio2 devuelven HRESULT: 0 sería
        // S_OK con los punteros de salida sin rellenar (Unity: CoCreateFreeThreadedMarshaler ->
        // deref de NULL). La respuesta honesta de una función no implementada es E_NOTIMPL.
        bool isHresultFamily = false;
        if (bangPos != std::string::npos) {
            const std::string fn = stubName->substr(bangPos + 1);
            const std::string dllLower = toLower(stubName->substr(0, bangPos));
            const bool comDll = dllLower.find("ole32") != std::string::npos || dllLower.find("combase") != std::string::npos ||
                                dllLower.find("api-ms-win-core-com") != std::string::npos ||
                                dllLower.find("winrt") != std::string::npos;
            const bool comFn = (fn.rfind("Co", 0) == 0 && fn.rfind("CoTaskMem", 0) != 0 && fn != "CoGetCurrentProcess" &&
                                fn != "CoUninitialize" && fn != "CoFreeUnusedLibraries") ||
                               (fn.rfind("Ro", 0) == 0 && fn != "RoUninitialize");
            isHresultFamily = (comDll && comFn) || fn.rfind("MFCreate", 0) == 0 || fn == "MFStartup" ||
                              fn.rfind("DirectSoundCreate", 0) == 0 || fn.rfind("DirectSoundCaptureCreate", 0) == 0 ||
                              fn == "XAudio2Create";
        }
        const uint64_t stubRet = isRegistry ? 2 : (isHresultFamily ? 0x80004001u : 0);
        static std::atomic<uint64_t> s_stubLogCount{0};
        const uint64_t stubN = s_stubLogCount.fetch_add(1, std::memory_order_relaxed);
        if (stubN < 3000 || (stubN % 10000) == 0)
        LOGI("[NativeStub] Called %s (RCX=0x%llx, RDX=0x%llx, R8=0x%llx, R9=0x%llx) -> returning %llu",
             stubName->c_str(),
             static_cast<unsigned long long>(ctx.gpr[1]),
             static_cast<unsigned long long>(ctx.gpr[2]),
             static_cast<unsigned long long>(ctx.gpr[8]),
             static_cast<unsigned long long>(ctx.gpr[9]),
             static_cast<unsigned long long>(stubRet));
        ctx.gpr[0] = stubRet;
        ctx.rip = returnAddress;
        return true;
    }

    // 2. Extrae hasta 12 argumentos según Win64: RCX, RDX, R8, R9 y [RSP+0x20..0x58]
    // tras el shadow space (cada lectura de pila validada con isValidStackAddress).
    const win64abi::ThunkArgs args =
        win64abi::readThunkArgs(ctx, static_cast<const OryonEmitterBackend*>(user));
    // Backend del hilo actual para trampolines que invocan callbacks invitados.
    t_dispatchBackend = static_cast<OryonEmitterBackend*>(user);

    // 3. Invoca el wrapper C++ host registrado
    const bool isNoisy = info->noisy;
    // Volumen acotado: las primeras 30.000 llamadas completas y después 1 de cada 1.000
    // (logd descarta líneas de una app que satura el búfer y se perderían los diagnósticos).
    // Además, pasado el tope, las primeras 4 llamadas de cada función siguen saliendo:
    // así se ve qué API nueva aparece justo antes de un cierre o fallo tardío.
    static std::atomic<uint64_t> s_thunkLogCount{0};
    // Registro de llamadas: por defecto solo las 4 primeras de cada función (antes las 30 000
    // primeras de todas + 1 de cada 1000: ~60 000 líneas por sesión, CPU y calor en logd).
    // `setprop debug.hyperplay.thunklog <N>` registra además las N primeras llamadas globales.
    static const uint64_t s_thunkLogFirst = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.thunklog", v) > 0 ? std::strtoull(v, nullptr, 10) : 0ull;
    }();
    const bool logThisCall = !callName.empty() && !isNoisy && [info] {
        if (s_thunkLogFirst != 0 && s_thunkLogCount.fetch_add(1, std::memory_order_relaxed) < s_thunkLogFirst) {
            return true;
        }
        return info->calls.fetch_add(1, std::memory_order_relaxed) < 4;
    }();
    if (logThisCall) {
        LOGI("[NativeThunk] %s(0x%llx, 0x%llx, 0x%llx, 0x%llx) from RIP=0x%llx",
             callName.c_str(),
             static_cast<unsigned long long>(args[0]),
             static_cast<unsigned long long>(args[1]),
             static_cast<unsigned long long>(args[2]),
             static_cast<unsigned long long>(args[3]),
             static_cast<unsigned long long>(returnAddress));
    }

    startStallWatchdog();
    ThunkInFlight& inFlight = myInFlight();
    inFlight.ret.store(returnAddress, std::memory_order_relaxed);
    inFlight.a0.store(args[0], std::memory_order_relaxed);
    inFlight.a1.store(args[1], std::memory_order_relaxed);
    inFlight.since.store(monoMs(), std::memory_order_relaxed);
    inFlight.info.store(info, std::memory_order_release);
    CpuContext* const prevThunkCtx = t_thunkCtx;
    const uint64_t prevThunkRet = t_thunkRet;
    t_thunkCtx = &ctx;
    t_thunkRet = returnAddress;
    const uint64_t ret = win64abi::invokeThunk(target, args);
    t_thunkCtx = prevThunkCtx;
    t_thunkRet = prevThunkRet;
    inFlight.info.store(nullptr, std::memory_order_release);
    if (t_ctxOverride) {
        // RtlRestoreContext / RtlUnwindEx / RaiseException resuelta: se sigue en otro contexto.
        t_ctxOverride = false;
        ctx = t_ctxOverrideValue;
        return true;
    }
    inFlight.since.store(monoMs(), std::memory_order_relaxed);  // último thunk terminado

    if (logThisCall) {
        LOGI("[NativeThunk] %s -> RAX=0x%llx",
             callName.c_str(),
             static_cast<unsigned long long>(ret));
    }

    // 4. Guarda el retorno en RAX (ctx.regs[0]) si no es una función void y actualiza ctx.rip = returnAddress
    if (!info->isVoid) {
        ctx.gpr[0] = ret;
    }
    ctx.rip = returnAddress;
    if (t_exitThreadRequested) {
        // ExitThread no vuelve: RIP=0 termina executeThread con el código en RAX.
        t_exitThreadRequested = false;
        ctx.gpr[0] = t_exitThreadCode;
        ctx.rip = 0;
    }
    return true;
}


// Nombre de un import por ordinal en las DLL de sistema que se importan así (clr.dll de .NET
// Framework importa OLEAUT32 solo por ordinal). Vacío si el ordinal no se conoce.
static std::string importName(const std::string& dll, const ImportedFunction& fn) {
    if (!fn.name.empty() || fn.ordinal == 0) return fn.name;
    std::string d = dll;
    for (char& c : d) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (d == "oleaut32.dll" || d == "oleaut32") {
        static const std::unordered_map<uint16_t, const char*> kOleAut = {
            {2, "SysAllocString"}, {4, "SysAllocStringLen"}, {6, "SysFreeString"}, {7, "SysStringLen"},
            {8, "VariantInit"}, {9, "VariantClear"}, {10, "VariantCopy"}, {12, "VariantChangeType"},
            {16, "SafeArrayDestroy"}, {17, "SafeArrayGetDim"}, {18, "SafeArrayGetElemsize"},
            {19, "SafeArrayGetUBound"}, {20, "SafeArrayGetLBound"}, {23, "SafeArrayAccessData"},
            {24, "SafeArrayUnaccessData"}, {25, "SafeArrayGetElement"}, {26, "SafeArrayPutElement"},
            {37, "SafeArrayAllocData"}, {41, "SafeArrayAllocDescriptorEx"}, {44, "SafeArraySetRecordInfo"},
            {45, "SafeArrayGetRecordInfo"}, {77, "SafeArrayGetVartype"}, {147, "VariantChangeTypeEx"},
            {149, "SysStringByteLen"}, {150, "SysAllocStringByteLen"}, {162, "LoadRegTypeLib"},
            {164, "QueryPathOfRegTypeLib"}, {179, "VarDecMul"}, {180, "CreateTypeLib2"}, {183, "LoadTypeLibEx"},
            {187, "VarDecFix"}, {188, "VarDecInt"}, {193, "VarDecFromR4"}, {194, "VarDecFromR8"},
            {196, "VarDecFromCy"}, {200, "GetErrorInfo"}, {201, "SetErrorInfo"}, {202, "CreateErrorInfo"},
            {203, "VarDecRound"}, {204, "VarDecCmp"}, {216, "VarR4FromDec"}, {220, "VarR8FromDec"},
            {228, "VarCyFromDec"}, {323, "GetRecordInfoFromTypeInfo"}, {411, "SafeArrayCreateVector"},
        };
        if (const auto it = kOleAut.find(fn.ordinal); it != kOleAut.end()) return it->second;
    }
    if (d == "wsock32.dll" || d == "wsock32" || d == "ws2_32.dll" || d == "ws2_32") {
        static const std::unordered_map<uint16_t, const char*> kWsock = {
            {1, "accept"}, {2, "bind"}, {3, "closesocket"}, {4, "connect"}, {5, "getpeername"}, {6, "getsockname"},
            {7, "getsockopt"}, {8, "htonl"}, {9, "htons"}, {10, "inet_addr"}, {11, "inet_ntoa"}, {12, "ioctlsocket"},
            {13, "listen"}, {14, "ntohl"}, {15, "ntohs"}, {16, "recv"}, {17, "recvfrom"}, {18, "select"}, {19, "send"},
            {20, "sendto"}, {21, "setsockopt"}, {22, "shutdown"}, {23, "socket"}, {51, "gethostbyaddr"},
            {52, "gethostbyname"}, {57, "gethostname"}, {101, "WSAAsyncSelect"}, {111, "WSAGetLastError"},
            {112, "WSASetLastError"}, {115, "WSAStartup"}, {116, "WSACleanup"}, {151, "__WSAFDIsSet"},
        };
        if (const auto it = kWsock.find(fn.ordinal); it != kWsock.end()) return it->second;
    }
    if (d.rfind("xinput", 0) == 0) {
        static const std::unordered_map<uint16_t, const char*> kXInput = {
            {2, "XInputGetState"}, {3, "XInputSetState"}, {4, "XInputGetCapabilities"}, {5, "XInputEnable"},
            {6, "XInputGetDSoundAudioDeviceGuids"}, {7, "XInputGetBatteryInformation"}, {8, "XInputGetKeystroke"},
            {100, "XInputGetStateEx"},
        };
        if (const auto it = kXInput.find(fn.ordinal); it != kXInput.end()) return it->second;
    }
    return fn.name;
}

size_t NativeDllInterceptor::hijackIat(LoadedPeImage& image) {
    return hijackIatChecked(image).hijacked;
}

IatHijackStats NativeDllInterceptor::hijackIatChecked(LoadedPeImage& image) {
    IatHijackStats stats;
    init(&image);
    if (image.baseAddress == nullptr || image.totalMappedSize == 0) return stats;

    const size_t pageSize = s_pageSize();
    auto* base = static_cast<uint8_t*>(image.baseAddress);
    const size_t ptrSize = image.is32Bit ? 4 : 8;

    for (const auto& mod : image.peInfo.imports) {
        for (const auto& fn : mod.functions) {
            if (fn.iatRva == 0) continue;
            ++stats.total;
            // Límite del slot dentro de la imagen mapeada: un RVA corrupto no debe escribir fuera.
            if (static_cast<uint64_t>(fn.iatRva) + ptrSize > image.totalMappedSize) {
                ++stats.outOfImage;
                if (stats.unresolvedImports.size() < 512) {
                    stats.unresolvedImports.push_back(mod.name + "!" + fn.name + " (fuera de imagen)");
                }
                continue;
            }
            void* nativeFunc = resolveNativeSymbol(mod.name, importName(mod.name, fn));
            if (nativeFunc == nullptr && image.is32Bit) {
                // 32 bits (en desarrollo): stub de diagnóstico como en las DLL locales. El log dice cuáles
                // se llegan a llamar ("[32 bits] ... aridad desconocida (stub)") para implementarlas.
                nativeFunc = DynamicStubManager::getOrCreateStub(mod.name, fn.name);
                if (nativeFunc != nullptr) LOGW("IAT32 stub: %s!%s", mod.name.c_str(), fn.name.c_str());
            }
            if (nativeFunc == nullptr) {
                ++stats.unresolved;
                if (stats.unresolvedImports.size() < 512) {
                    stats.unresolvedImports.push_back(mod.name + "!" + fn.name);
                }
                continue;  // se deja el slot intacto; el llamador aborta la ejecución
            }
            if (image.is32Bit && reinterpret_cast<uintptr_t>(nativeFunc) > 0xFFFFFFFFull) {
                // IAT de 32 bits: alias bajo 4 GB que dispatchThunk traduce a la función real.
                const uintptr_t low = w32alias::lowFor(nativeFunc, mod.name + "!" + fn.name);
                if (low == 0) {
                    ++stats.unresolved;
                    if (stats.unresolvedImports.size() < 512) {
                        stats.unresolvedImports.push_back(mod.name + "!" + fn.name + " (sin alias <4GB)");
                    }
                    continue;
                }
                nativeFunc = reinterpret_cast<void*>(low);
            }

            uint8_t* target = base + fn.iatRva;
            const uintptr_t pageStart = reinterpret_cast<uintptr_t>(target) & ~(pageSize - 1);
            const int prot = protForRva(image, fn.iatRva);
            if ((prot & PROT_WRITE) == 0) {
                if (mprotect(reinterpret_cast<void*>(pageStart), pageSize, prot | PROT_WRITE) != 0) {
                    LOGW("mprotect(RW) de la IAT en RVA 0x%x falló: %s", fn.iatRva, std::strerror(errno));
                    ++stats.unresolved;
                    if (stats.unresolvedImports.size() < 512) {
                        stats.unresolvedImports.push_back(mod.name + "!" + fn.name + " (mprotect)");
                    }
                    continue;
                }
            }

            const uintptr_t value = reinterpret_cast<uintptr_t>(nativeFunc);
            if (image.is32Bit) {
                const uint32_t slot = static_cast<uint32_t>(value);
                std::memcpy(target, &slot, sizeof(slot));
            } else {
                const uint64_t slot = static_cast<uint64_t>(value);
                std::memcpy(target, &slot, sizeof(slot));
            }

            if ((prot & PROT_WRITE) == 0) {
                // Se restaura la protección original: W^X se mantiene (solo la página se abre un instante).
                if (mprotect(reinterpret_cast<void*>(pageStart), pageSize, prot) != 0) {
                    LOGW("No se pudo restaurar la protección de la IAT en RVA 0x%x", fn.iatRva);
                }
            }
            ++stats.hijacked;
        }
    }

    LOGI("IAT: %zu entradas, %zu resueltas, %zu sin resolver, %zu fuera de imagen", stats.total, stats.hijacked,
         stats.unresolved, stats.outOfImage);
    // 32 bits (en desarrollo): la lista completa de lo que falta, una línea por import.
    if (image.is32Bit) {
        for (const auto& u : stats.unresolvedImports) LOGW("IAT32 sin resolver: %s", u.c_str());
    }
    return stats;
}

namespace {
void* hostMemmove(void* d, const void* s, size_t n) { return std::memmove(d, s, n); }

// memcpy/memmove estáticos del CRT de MSVC x64 (dentro de coreclr.dll y otras DLL nativas):
//   mov rax, rcx ; lea r10, [rip -> ImageBase] ; cmp r8, 0Fh ; ja ...
// La entrada se cambia por `mov rax, imm64 ; jmp rax` hacia un memmove nativo registrado como
// trampolín (mismo contrato: devuelve el destino, admite solapamiento). El memcpy traducido dejaba
// a veces nombres de tipo sin su 0 final en CoreCLR (Stardew Valley: TypeLoad de tipos válidos).
// `setprop debug.hyperplay.hostmemcpy 0` lo desactiva.
void patchStaticMsvcMemcpy(LoadedPeImage& image) {
    if (image.is32Bit) return;
    char v[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.hyperplay.hostmemcpy", v) > 0 && v[0] == '0') return;
    static const bool s_reg = [] {
        registerTrampolineAddress(w32arity::hpFn(&hostMemmove), "hostcrt!memmove");
        return true;
    }();
    (void)s_reg;
    auto* base = static_cast<uint8_t*>(image.baseAddress);
    const size_t ps = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    int patched = 0;
    for (const auto& sec : image.sections) {
        if ((sec.characteristics & 0x20000000u) == 0) continue;  // IMAGE_SCN_MEM_EXECUTE
        auto* p = static_cast<uint8_t*>(sec.virtualAddress);
        if (p == nullptr || sec.size < 32) continue;
        for (size_t i = 0; i + 20 <= sec.size; ++i) {
            const uint8_t* q = p + i;
            if (q[0] != 0x48 || q[1] != 0x8B || q[2] != 0xC1 || q[3] != 0x4C || q[4] != 0x8D || q[5] != 0x15) continue;
            if (q[10] != 0x49 || q[11] != 0x83 || q[12] != 0xF8 || q[13] != 0x0F || q[14] != 0x0F || q[15] != 0x87) continue;
            int32_t disp = 0;
            std::memcpy(&disp, q + 6, 4);
            if (q + 10 + disp != base) continue;  // el lea apunta a la base de la imagen
            const uintptr_t page = reinterpret_cast<uintptr_t>(q) & ~(ps - 1);
            const size_t len = ((reinterpret_cast<uintptr_t>(q) + 12 - page) + ps - 1) & ~(ps - 1);
            if (mprotect(reinterpret_cast<void*>(page), len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) continue;
            uint8_t stub[12] = {0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xE0};
            const auto target = reinterpret_cast<uint64_t>(&hostMemmove);
            std::memcpy(stub + 2, &target, 8);
            std::memcpy(p + i, stub, sizeof(stub));
            mprotect(reinterpret_cast<void*>(page), len, PROT_READ | PROT_EXEC);
            ++patched;
            i += 12;
        }
    }
    if (patched > 0) LOGI("memcpy estático del CRT -> memmove nativo: %d sitio(s)", patched);
}
}  // namespace

IatHijackStats NativeDllInterceptor::hijackLocalDllIat(LoadedPeImage& image) {
    IatHijackStats stats;
    if (image.baseAddress == nullptr || image.totalMappedSize == 0) return stats;
    patchStaticMsvcMemcpy(image);
    struct LocalImportsScope {
        bool prev = t_resolvingLocalDllImports;
        LocalImportsScope() { t_resolvingLocalDllImports = true; }
        ~LocalImportsScope() { t_resolvingLocalDllImports = prev; }
    } localScope;

    const size_t pageSize = s_pageSize();
    auto* base = static_cast<uint8_t*>(image.baseAddress);
    const size_t ptrSize = image.is32Bit ? 4 : 8;

    for (const auto& mod : image.peInfo.imports) {
        for (const auto& fn : mod.functions) {
            if (fn.iatRva == 0) continue;
            ++stats.total;
            if (static_cast<uint64_t>(fn.iatRva) + ptrSize > image.totalMappedSize) {
                ++stats.outOfImage;
                continue;
            }
            void* nativeFunc = resolveNativeSymbol(mod.name, importName(mod.name, fn));
            if (nativeFunc == nullptr) {
                // Asigna un dynamic diagnostic stub por slot para imports secundarios
                nativeFunc = DynamicStubManager::getOrCreateStub(mod.name, fn.name);
            }
            if (nativeFunc == nullptr) {
                ++stats.unresolved;
                stats.unresolvedImports.push_back(mod.name + "!" + fn.name);
                continue;
            }

            uint8_t* target = base + fn.iatRva;
            const uintptr_t pageStart = reinterpret_cast<uintptr_t>(target) & ~(pageSize - 1);
            const int prot = protForRva(image, fn.iatRva);
            if ((prot & PROT_WRITE) == 0) {
                if (mprotect(reinterpret_cast<void*>(pageStart), pageSize, prot | PROT_WRITE) != 0) {
                    LOGW("mprotect(RW) de IAT local en RVA 0x%x falló: %s", fn.iatRva, std::strerror(errno));
                    ++stats.unresolved;
                    continue;
                }
            }

            uintptr_t value = reinterpret_cast<uintptr_t>(nativeFunc);
            if (image.is32Bit) {
                // Antes se truncaba a 32 bits en silencio (salto a basura): alias bajo 4 GB.
                if (value > 0xFFFFFFFFull) value = w32alias::lowFor(nativeFunc, mod.name + "!" + importName(mod.name, fn));
                const uint32_t slot = static_cast<uint32_t>(value);
                std::memcpy(target, &slot, sizeof(slot));
            } else {
                const uint64_t slot = static_cast<uint64_t>(value);
                std::memcpy(target, &slot, sizeof(slot));
            }

            if ((prot & PROT_WRITE) == 0) {
                mprotect(reinterpret_cast<void*>(pageStart), pageSize, prot);
            }
            ++stats.hijacked;
        }
    }

    LOGI("Local DLL IAT (%s): %zu entradas, %zu resueltas (con dynamic stubs), %zu sin resolver",
         image.gameDir.c_str(), stats.total, stats.hijacked, stats.unresolved);
    return stats;
}

// --- ntdll NtCreateFile / NtQueryDirectoryFile (System.IO de Mono: Directory.GetFiles) ---------
// Mono 6 (corefx) enumera directorios con NtCreateFile (abrir carpeta, ruta NT "\??\D:\..." o
// relativa a RootDirectory) + NtQueryDirectoryFile (FILE_FULL_DIR_INFORMATION). Celeste lo usa
// para listar sus fuentes. El handle es el de un fd de directorio (CloseHandle lo cierra).
struct DirListing {
    std::vector<std::string> names;
    size_t next = 0;
    dev_t dev = 0;
    ino_t ino = 0;  // carpeta a la que pertenece (el número de fd se reutiliza)
};
static std::mutex s_dirMutex;
static std::unordered_map<int, DirListing> s_dirListings;

static void forgetDirListing(int fd) {
    std::lock_guard<std::mutex> lock(s_dirMutex);
    s_dirListings.erase(fd);
}

static std::string unicodeStringToUtf8(const void* us) {
    if (us == nullptr) return {};
    uint16_t lenBytes = 0;
    std::memcpy(&lenBytes, us, 2);
    const char16_t* buf = nullptr;
    std::memcpy(&buf, static_cast<const uint8_t*>(us) + 8, sizeof(buf));
    if (buf == nullptr) return {};
    std::u16string s(buf, lenBytes / 2);
    return utf16ToUtf8(s.c_str());
}

static uint32_t hp_NtCreateFile(void** fileHandle, uint32_t desiredAccess, const void* objAttr, void* ioStatus,
                                void*, uint32_t, uint32_t, uint32_t disposition, uint32_t createOptions, void*, uint32_t) {
    if (fileHandle == nullptr || objAttr == nullptr || s_activeImage == nullptr) return 0xC000000Du;  // INVALID_PARAMETER
    void* root = nullptr;
    const void* name = nullptr;
    std::memcpy(&root, static_cast<const uint8_t*>(objAttr) + 8, sizeof(root));
    std::memcpy(&name, static_cast<const uint8_t*>(objAttr) + 16, sizeof(name));
    std::string path = unicodeStringToUtf8(name);
    if (path.rfind("\\??\\", 0) == 0) path = path.substr(4);
    std::string real;
    int fd = -1;
    const bool wantDir = (createOptions & 0x1u) != 0;  // FILE_DIRECTORY_FILE
    const int flags = O_RDONLY | O_CLOEXEC | (wantDir ? O_DIRECTORY : 0);
    if (root != nullptr) {
        std::string rel = path;
        for (char& c : rel) if (c == '\\') c = '/';
        fd = openat(fdFromHandle(root), rel.c_str(), flags);
        real = rel;
    } else {
        real = BionicPeLoader::resolveVirtualPath(*s_activeImage, path);
        fd = open(real.c_str(), flags);
    }
    const uint32_t status = fd >= 0 ? 0u : (errno == ENOENT ? 0xC0000034u : (errno == ENOTDIR ? 0xC0000103u : 0xC0000022u));
    if (ioStatus != nullptr) {
        const uint64_t st = status, info = fd >= 0 ? 1u /*FILE_OPENED*/ : 0u;
        std::memcpy(ioStatus, &st, 8);
        std::memcpy(static_cast<uint8_t*>(ioStatus) + 8, &info, 8);
    }
    LOGI("NtCreateFile('%s' dir=%d acc=0x%x disp=%u) -> fd=%d status=0x%x", real.c_str(), wantDir ? 1 : 0,
         desiredAccess, disposition, fd, status);
    if (fd < 0) return status;
    forgetDirListing(fd);  // apertura nueva: nada del listado de un fd anterior con el mismo número
    *fileHandle = handleFromFd(fd);
    return 0;
}

static uint32_t hp_NtQueryDirectoryFile(void* fileHandle, void*, void*, void*, void* ioStatus, void* info,
                                        uint32_t length, uint32_t infoClass, uint8_t single, void* fileName,
                                        uint8_t restart) {
    (void)fileName;
    const int fd = fdFromHandle(fileHandle);
    auto setIo = [&](uint32_t status, uint64_t bytes) {
        if (ioStatus == nullptr) return;
        const uint64_t st = status;
        std::memcpy(ioStatus, &st, 8);
        std::memcpy(static_cast<uint8_t*>(ioStatus) + 8, &bytes, 8);
    };
    if (fd < 0 || info == nullptr) { setIo(0xC0000008u, 0); return 0xC0000008u; }  // INVALID_HANDLE
    if (infoClass != 2 && infoClass != 1 && infoClass != 3) {  // Full / Directory / Both
        LOGW("NtQueryDirectoryFile: clase %u no soportada", infoClass);
        setIo(0xC0000003u, 0);
        return 0xC0000003u;  // INVALID_INFO_CLASS
    }
    std::lock_guard<std::mutex> lock(s_dirMutex);
    auto [it, fresh] = s_dirListings.try_emplace(fd);
    DirListing& dl = it->second;
    // Si el fd se cerró por otra vía (NtClose, close) y ahora es otra carpeta, el listado viejo no
    // vale: antes se devolvía agotado (vacío) y Celeste fallaba con "Missing Language Files".
    struct stat dst{};
    const bool haveStat = fstat(fd, &dst) == 0;
    const bool otherDir = haveStat && (dst.st_dev != dl.dev || dst.st_ino != dl.ino);
    if (fresh || restart || otherDir) {
        dl = DirListing{};
        if (haveStat) {
            dl.dev = dst.st_dev;
            dl.ino = dst.st_ino;
        }
        dl.names = {".", ".."};
        const int dupFd = dup(fd);
        if (DIR* d = dupFd >= 0 ? fdopendir(dupFd) : nullptr) {
            rewinddir(d);
            while (struct dirent* e = readdir(d)) {
                if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0) dl.names.emplace_back(e->d_name);
            }
            closedir(d);
        } else if (dupFd >= 0) {
            close(dupFd);
        }
    }
    auto* out = static_cast<uint8_t*>(info);
    size_t used = 0;
    uint8_t* prev = nullptr;
    // FILE_FULL_DIR_INFORMATION: nombre en 68; FILE_DIRECTORY_INFORMATION en 64;
    // FILE_BOTH_DIR_INFORMATION en 94 (con ShortName de 24 bytes antes).
    const size_t nameOff = infoClass == 2 ? 68 : (infoClass == 1 ? 64 : 94);
    while (dl.next < dl.names.size()) {
        const std::string& n = dl.names[dl.next];
        std::u16string wn;
        for (unsigned char c : n) wn.push_back(static_cast<char16_t>(c));  // nombres ASCII/UTF-8 simples
        const size_t entry = (nameOff + wn.size() * 2 + 7) & ~static_cast<size_t>(7);
        if (used + nameOff + wn.size() * 2 > length) break;
        uint8_t* e = out + used;
        std::memset(e, 0, std::min<size_t>(entry, length - used));
        struct stat st{};
        fstatat(fd, n.c_str(), &st, 0);
        auto ft = [](time_t t) { return (static_cast<uint64_t>(t) + 11644473600ULL) * 10000000ULL; };
        const uint64_t ctime = ft(st.st_ctime), atime = ft(st.st_atime), mtime = ft(st.st_mtime);
        const uint64_t size = static_cast<uint64_t>(st.st_size);
        const uint32_t attrs = S_ISDIR(st.st_mode) ? 0x10u : 0x80u;  // DIRECTORY / NORMAL
        const uint32_t nameLen = static_cast<uint32_t>(wn.size() * 2);
        std::memcpy(e + 8, &ctime, 8);
        std::memcpy(e + 16, &atime, 8);
        std::memcpy(e + 24, &mtime, 8);
        std::memcpy(e + 32, &mtime, 8);
        std::memcpy(e + 40, &size, 8);
        std::memcpy(e + 48, &size, 8);
        std::memcpy(e + 56, &attrs, 4);
        std::memcpy(e + 60, &nameLen, 4);
        std::memcpy(e + nameOff, wn.data(), nameLen);
        if (prev != nullptr) {
            const uint32_t off = static_cast<uint32_t>(e - prev);
            std::memcpy(prev, &off, 4);
        }
        prev = e;
        used += entry;
        ++dl.next;
        if (single) break;
    }
    if (prev == nullptr) {
        const uint32_t st = dl.next >= dl.names.size() ? 0x80000006u : 0x80000005u;  // NO_MORE_FILES / BUFFER_OVERFLOW
        setIo(st, 0);
        return st;
    }
    setIo(0, used);
    return 0;
}

static uint32_t hp_RtlNtStatusToDosError(uint32_t status) {
    switch (status) {
        case 0: return 0;
        case 0xC0000034u: return 2;    // OBJECT_NAME_NOT_FOUND -> FILE_NOT_FOUND
        case 0xC000003Au: return 3;    // OBJECT_PATH_NOT_FOUND -> PATH_NOT_FOUND
        case 0xC0000022u: return 5;    // ACCESS_DENIED
        case 0xC0000103u: return 267;  // NOT_A_DIRECTORY -> ERROR_DIRECTORY
        case 0x80000006u: return 18;   // NO_MORE_FILES
        case 0xC0000008u: return 6;    // INVALID_HANDLE
        default: return 317;           // ERROR_MR_MID_NOT_FOUND
    }
}

// shell32!ShellExecuteExW: no hay otras aplicaciones (Celeste abre su error_log.txt con el bloc de
// notas al fallar). Se responde éxito sin proceso para que eso no convierta un error en un cierre.
static int32_t hp_ShellExecuteExW(void* sei) {
    if (sei != nullptr) {
        void* none = nullptr;
        std::memcpy(static_cast<uint8_t*>(sei) + 104, &none, sizeof(none));  // hProcess (x64)
        const uintptr_t ok = 33;  // hInstApp > 32 = éxito
        std::memcpy(static_cast<uint8_t*>(sei) + 56, &ok, sizeof(ok));
    }
    LOGI("ShellExecuteExW: ignorado (sin aplicaciones externas)");
    return 1;
}

// --- user32 SetProp/GetProp/RemoveProp (SDL3 guarda su SDL_Window en la HWND) ------------------
// La clave es un ATOM (palabra baja, puntero < 0x10000) o una cadena.
static std::mutex s_propMutex;
static std::map<std::pair<uintptr_t, std::string>, uintptr_t> s_windowProps;

static std::string propKey(const void* name, bool wide) {
    const auto v = reinterpret_cast<uintptr_t>(name);
    if (v < 0x10000) return "#" + std::to_string(v);
    if (!wide) return std::string(static_cast<const char*>(name));
    std::string out;
    for (const auto* p = static_cast<const char16_t*>(name); *p; ++p) out.push_back(static_cast<char>(*p & 0x7F));
    return out;
}
static int32_t setProp(void* hwnd, const void* name, void* data, bool wide) {
    std::lock_guard<std::mutex> lock(s_propMutex);
    s_windowProps[{reinterpret_cast<uintptr_t>(hwnd), propKey(name, wide)}] = reinterpret_cast<uintptr_t>(data);
    return 1;
}
static void* getProp(void* hwnd, const void* name, bool wide) {
    std::lock_guard<std::mutex> lock(s_propMutex);
    auto it = s_windowProps.find({reinterpret_cast<uintptr_t>(hwnd), propKey(name, wide)});
    return it == s_windowProps.end() ? nullptr : reinterpret_cast<void*>(it->second);
}
static void* removeProp(void* hwnd, const void* name, bool wide) {
    std::lock_guard<std::mutex> lock(s_propMutex);
    auto it = s_windowProps.find({reinterpret_cast<uintptr_t>(hwnd), propKey(name, wide)});
    if (it == s_windowProps.end()) return nullptr;
    void* v = reinterpret_cast<void*>(it->second);
    s_windowProps.erase(it);
    return v;
}
// Interlocked* de kernel32 (solo existen como exports en 32 bits; en x64 son intrínsecos).
static int32_t hp_InterlockedIncrement(int32_t* p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
static int32_t hp_InterlockedDecrement(int32_t* p) { return __atomic_sub_fetch(p, 1, __ATOMIC_SEQ_CST); }
static int32_t hp_InterlockedExchange(int32_t* p, int32_t v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
static int32_t hp_InterlockedExchangeAdd(int32_t* p, int32_t v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
static int32_t hp_InterlockedCompareExchange(int32_t* p, int32_t exchange, int32_t comparand) {
    __atomic_compare_exchange_n(p, &comparand, exchange, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return comparand;  // valor inicial de *p
}
static int64_t hp_InterlockedCompareExchange64(int64_t* p, int64_t exchange, int64_t comparand) {
    __atomic_compare_exchange_n(p, &comparand, exchange, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return comparand;
}

// GetVersionExA: Windows 10 responde 6.2.9200 a los programas sin manifiesto (lo normal en juegos de
// 32 bits antiguos como DmC). OSVERSIONINFOA (148 bytes) u OSVERSIONINFOEXA (156).
static int32_t hp_GetVersionExA(uint8_t* info) {
    if (info == nullptr) return 0;
    uint32_t size = 0;
    std::memcpy(&size, info, 4);
    if (size != 148 && size != 156) return 0;
    const uint32_t v[4] = {6, 2, 9200, 2};  // major, minor, build, VER_PLATFORM_WIN32_NT
    std::memcpy(info + 4, v, sizeof(v));
    std::memset(info + 20, 0, 128);         // szCSDVersion
    if (size == 156) std::memset(info + 148, 0, 8);
    if (size == 156) info[154] = 1;         // wProductType = VER_NT_WORKSTATION
    return 1;
}

// GetVersionExW: igual que la A con cadenas UTF-16 (OSVERSIONINFOW 276 bytes, EXW 284).
static int32_t hp_GetVersionExW(uint8_t* info) {
    if (info == nullptr) return 0;
    uint32_t size = 0;
    std::memcpy(&size, info, 4);
    if (size != 276 && size != 284) return 0;
    const uint32_t v[4] = {6, 2, 9200, 2};
    std::memcpy(info + 4, v, sizeof(v));
    std::memset(info + 20, 0, 256);  // szCSDVersion[128] WCHAR
    if (size == 284) {
        std::memset(info + 276, 0, 8);
        info[282] = 1;  // wProductType = VER_NT_WORKSTATION
    }
    return 1;
}

// GetStartupInfoA: STARTUPINFOA mide 68 bytes en 32 bits y 104 en 64; sin datos de arranque.
static void hp_GetStartupInfoA(uint8_t* si) {
    if (si == nullptr) return;
    const uint32_t cb = lowmem::enabled() ? 68u : 104u;
    std::memset(si, 0, cb);
    std::memcpy(si, &cb, 4);
}

// GetThreadTimes: creación/salida a 0 y tiempo de CPU del hilo actual como usuario (msvcr100 lo usa
// en clock()). FILETIME en unidades de 100 ns.
// GlobalMemoryStatus: MEMORYSTATUS con campos SIZE_T (32 bytes en 32 bits, 56 en 64). Un juego
// de 32 bits con LARGE_ADDRESS_AWARE ve 4 GB virtuales (WOW64: 0xFFFE0000).
static void hp_GlobalMemoryStatus(void* buffer) {
    if (buffer == nullptr) return;
    auto* b = static_cast<uint8_t*>(buffer);
    const bool w32 = lowmem::enabled();
    const size_t ps = w32 ? 4 : 8;
    const uint32_t len = w32 ? 32u : 56u;
    const uint32_t load = 25;
    std::memcpy(b, &len, 4);
    std::memcpy(b + 4, &load, 4);
    // En 32 bits Windows recorta a 4 GB - 1 los valores que no caben en un SIZE_T.
    auto clip = [w32](uint64_t v) { return w32 && v > 0xFFFFFFFFull ? 0xFFFFFFFFull : v; };
    const uint64_t vals[6] = {
        clip(16ULL << 30), clip(12ULL << 30),       // TotalPhys, AvailPhys
        clip(24ULL << 30), clip(20ULL << 30),       // TotalPageFile, AvailPageFile
        w32 ? 0xFFFE0000ull : (1ULL << 40),         // TotalVirtual
        w32 ? 0xE0000000ull : (1ULL << 40),         // AvailVirtual
    };
    for (size_t i = 0; i < 6; ++i) lowmem::storePtr(b + 8 + i * ps, vals[i]);
}

// GetWindowsDirectory / GetSystemDirectory: devuelven la longitud sin el NUL si cabe; si no, el
// tamaño necesario con el NUL. Rutas virtuales: el cargador las lleva a redist/<arq>/netfx/Windows.
template <class C>
static uint32_t copyWinDir(C* buf, uint32_t size, std::string_view path) {
    const auto len = static_cast<uint32_t>(path.size());
    if (buf == nullptr || size <= len) return len + 1;
    for (uint32_t i = 0; i < len; ++i) buf[i] = static_cast<C>(path[i]);
    buf[len] = 0;
    return len;
}
static uint32_t hp_GetWindowsDirectoryA(char* buf, uint32_t size) { return copyWinDir(buf, size, "C:\\Windows"); }
static uint32_t hp_GetWindowsDirectoryW(char16_t* buf, uint32_t size) { return copyWinDir(buf, size, "C:\\Windows"); }
static uint32_t hp_GetSystemDirectoryA(char* buf, uint32_t size) {
    return copyWinDir(buf, size, "C:\\Windows\\System32");
}
static uint32_t hp_GetSystemDirectoryW(char16_t* buf, uint32_t size) {
    return copyWinDir(buf, size, "C:\\Windows\\System32");
}

// OLEAUT32 mínimo. BSTR: [longitud en bytes (DWORD)][UTF-16][NUL]; el puntero apunta a los datos.
// Memoria del montón del proceso (por debajo de 4 GB en invitados de 32 bits).
namespace oleaut {
constexpr int32_t kENotImpl = static_cast<int32_t>(0x80004001u);
char16_t* allocBstrBytes(const void* src, uint32_t bytes) {
    auto* p = static_cast<uint8_t*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 0, bytes + 6));
    if (p == nullptr) return nullptr;
    std::memcpy(p, &bytes, 4);
    if (src != nullptr) std::memcpy(p + 4, src, bytes);
    else std::memset(p + 4, 0, bytes);
    p[4 + bytes] = 0;
    p[5 + bytes] = 0;
    return reinterpret_cast<char16_t*>(p + 4);
}
char16_t* SysAllocStringLen(const char16_t* s, uint32_t len) { return allocBstrBytes(s, len * 2); }
char16_t* SysAllocString(const char16_t* s) {
    if (s == nullptr) return nullptr;
    uint32_t n = 0;
    while (s[n] != 0) ++n;
    return SysAllocStringLen(s, n);
}
char16_t* SysAllocStringByteLen(const char* s, uint32_t bytes) { return allocBstrBytes(s, bytes); }
uint32_t SysStringByteLen(const char16_t* b) {
    if (b == nullptr) return 0;
    uint32_t n = 0;
    std::memcpy(&n, reinterpret_cast<const uint8_t*>(b) - 4, 4);
    return n;
}
uint32_t SysStringLen(const char16_t* b) { return SysStringByteLen(b) / 2; }
void SysFreeString(char16_t* b) {
    if (b != nullptr) NativeDllInterceptor::win_HeapFree(NativeDllInterceptor::win_GetProcessHeap(), 0, reinterpret_cast<uint8_t*>(b) - 4);
}
// VARIANT: vt (WORD) en +0, valor en +8. Solo se liberan los BSTR (VT_BSTR = 8).
void VariantInit(uint8_t* v) {
    if (v != nullptr) std::memset(v, 0, 8);
}
int32_t VariantClear(uint8_t* v) {
    if (v == nullptr) return 0;
    uint16_t vt = 0;
    std::memcpy(&vt, v, 2);
    if (vt == 8) SysFreeString(reinterpret_cast<char16_t*>(static_cast<uintptr_t>(lowmem::loadPtr(v + 8))));
    std::memset(v, 0, 8);
    return 0;
}
int32_t VariantCopy(uint8_t* dst, const uint8_t* src) {
    if (dst == nullptr || src == nullptr) return static_cast<int32_t>(0x80070057u);
    VariantClear(dst);
    std::memcpy(dst, src, lowmem::enabled() ? 16 : 24);
    uint16_t vt = 0;
    std::memcpy(&vt, src, 2);
    if (vt == 8) {
        auto* b = reinterpret_cast<const char16_t*>(static_cast<uintptr_t>(lowmem::loadPtr(src + 8)));
        lowmem::storePtr(dst + 8, b != nullptr ? SysAllocStringLen(b, SysStringLen(b)) : nullptr);
    }
    return 0;
}
// Sin objetos IErrorInfo: "no hay información de error" (S_FALSE) y puntero nulo.
int32_t GetErrorInfo(uint32_t, void** pp) {
    lowmem::storePtr(pp, uint64_t{0});
    return 1;
}
int32_t SetErrorInfo(uint32_t, void*) { return 0; }
int32_t CreateErrorInfo(void** pp) {
    lowmem::storePtr(pp, uint64_t{0});
    return kENotImpl;
}
uint32_t Zero1(void*) { return 0; }
int32_t NotImpl1(void*) { return kENotImpl; }
int32_t NotImpl2(void*, void*) { return kENotImpl; }
int32_t NotImpl3(void*, void*, void*) { return kENotImpl; }
int32_t NotImpl4(void*, void*, void*, void*) { return kENotImpl; }
int32_t NotImpl5(void*, void*, void*, void*, void*) { return kENotImpl; }
void* Null3(void*, void*, void*) { return nullptr; }
}  // namespace oleaut

// UrlIsW(url, URLIS): URLIS_URL (0) = tiene esquema "xxx:" de más de una letra (C:\ no cuenta);
// URLIS_FILEURL (3) = empieza por "file:". El resto de consultas: FALSE.
static int32_t hp_UrlIsW(const char16_t* url, int32_t kind) {
    if (url == nullptr) return 0;
    if (kind == 3) {
        const char16_t* f = u"file:";
        for (int i = 0; i < 5; ++i) {
            char16_t c = url[i];
            if (c >= u'A' && c <= u'Z') c = static_cast<char16_t>(c + 32);
            if (c != f[i]) return 0;
        }
        return 1;
    }
    if (kind == 0) {
        int i = 0;
        while ((url[i] >= u'a' && url[i] <= u'z') || (url[i] >= u'A' && url[i] <= u'Z') ||
               (url[i] >= u'0' && url[i] <= u'9') || url[i] == u'+' || url[i] == u'-' || url[i] == u'.') ++i;
        return i > 1 && url[i] == u':' ? 1 : 0;
    }
    return 0;
}

static int32_t hp_PathIsURLW(const char16_t* p) { return hp_UrlIsW(p, 0); }

// StrToIntW/A: decimal con signo opcional; se para en el primer carácter no numérico.
template <class C>
static int32_t strToIntT(const C* s) {
    if (s == nullptr) return 0;
    while (*s == ' ' || *s == '\t') ++s;
    bool neg = false;
    if (*s == '-' || *s == '+') neg = *s++ == '-';
    int64_t v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
    return static_cast<int32_t>(neg ? -v : v);
}
static int32_t hp_StrToIntW(const char16_t* s) { return strToIntT(s); }
static int32_t hp_StrToIntA(const char* s) { return strToIntT(s); }

// PathRemoveBackslashW: quita una '\' final salvo en raíces ("C:\"); devuelve el puntero al final.
static char16_t* hp_PathRemoveBackslashW(char16_t* p) {
    if (p == nullptr) return nullptr;
    size_t n = 0;
    while (p[n] != 0) ++n;
    if (n > 0 && p[n - 1] == u'\\' && !(n == 3 && p[1] == u':')) p[--n] = 0;
    return p + (n > 0 ? n - 1 : 0);
}
// StrChrW: primera aparición del carácter (WCHAR en 32 bits como argumento de 4 bytes).
static char16_t* hp_StrChrW(char16_t* s, uint32_t ch) {
    if (s == nullptr) return nullptr;
    for (; *s != 0; ++s) if (*s == static_cast<char16_t>(ch)) return s;
    return nullptr;
}

// UrlCanonicalizeW / UrlUnescapeW (S_OK / E_POINTER). Rutas locales: se copian; %xx se decodifica
// al desescapar (URL_UNESCAPE_INPLACE = 0x100000 trabaja sobre la propia cadena).
static int32_t urlCopyOut(const std::u16string& s, char16_t* out, uint32_t* cch) {
    if (cch == nullptr) return static_cast<int32_t>(0x80070057u);
    if (out == nullptr || *cch <= s.size()) {
        *cch = static_cast<uint32_t>(s.size() + 1);
        return static_cast<int32_t>(0x80004003u);  // E_POINTER
    }
    std::memcpy(out, s.data(), s.size() * 2);
    out[s.size()] = 0;
    *cch = static_cast<uint32_t>(s.size());
    return 0;
}
static std::u16string u16of(const char16_t* p) {
    std::u16string s;
    if (p != nullptr) for (; *p != 0; ++p) s.push_back(*p);
    return s;
}
// Como Windows: una ruta DOS se convierte en URL de archivo ("D:\a b.exe" -> "file:///D:/a%20b.exe",
// "\\srv\c" -> "file://srv/c"); el .NET Framework la usa como codebase del ejecutable. Una URL
// (con esquema) se deja igual.
static int32_t hp_UrlCanonicalizeW(const char16_t* url, char16_t* out, uint32_t* cch, uint32_t flags) {
    std::u16string s = u16of(url);
    const bool drive = s.size() >= 2 && s[1] == u':' &&
                       ((s[0] >= u'A' && s[0] <= u'Z') || (s[0] >= u'a' && s[0] <= u'z'));
    const bool unc = s.size() >= 2 && s[0] == u'\\' && s[1] == u'\\';
    if (drive || unc) {
        std::u16string r = drive ? u"file:///" : u"file://";
        for (size_t i = unc ? 2 : 0; i < s.size(); ++i) {
            const char16_t c = s[i];
            if (c == u'\\') r.push_back(u'/');
            else if (c == u' ' && (flags & 0x02000000u) == 0) r += u"%20";  // salvo URL_DONT_ESCAPE_EXTRA_INFO
            else r.push_back(c);
        }
        s = r;
    }
    return urlCopyOut(s, out, cch);
}
static int32_t hp_UrlUnescapeW(char16_t* url, char16_t* out, uint32_t* cch, uint32_t flags) {
    const std::u16string in = u16of(url);
    std::u16string s;
    auto hex = [](char16_t c) -> int {
        if (c >= u'0' && c <= u'9') return c - u'0';
        if (c >= u'a' && c <= u'f') return c - u'a' + 10;
        if (c >= u'A' && c <= u'F') return c - u'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == u'%' && i + 2 < in.size() + 0 && hex(in[i + 1]) >= 0 && hex(in[i + 2]) >= 0) {
            s.push_back(static_cast<char16_t>(hex(in[i + 1]) * 16 + hex(in[i + 2])));
            i += 2;
        } else {
            s.push_back(in[i]);
        }
    }
    if ((flags & 0x100000u) != 0 && url != nullptr) {
        std::memcpy(url, s.data(), s.size() * 2);
        url[s.size()] = 0;
        return 0;
    }
    return urlCopyOut(s, out, cch);
}

// ---------------------------------------------------------------------------------------------
// Resto de importaciones de clr.dll (.NET Framework 4, 32 bits). Firmas exactas: en stdcall una
// aridad equivocada desbalancea la pila. Solo se resuelven para invitados de 32 bits.
// ---------------------------------------------------------------------------------------------
// Cola de mensajes por hilo para invitados de 32 bits (PostThreadMessage / GetMessage). Unreal 3 tiene
// un hilo que espera en GetMessage las órdenes de instalar/quitar su gancho de teclado.
namespace threadmsg {
struct Msg { uint32_t hwnd, message, wParam, lParam; };
std::mutex g_m;
std::condition_variable g_cv;
std::unordered_map<uint32_t, std::deque<Msg>> g_queues;
bool post(uint32_t tid, uint32_t msg, uint32_t w, uint32_t l) {
    {
        std::lock_guard<std::mutex> lk(g_m);
        g_queues[tid].push_back({0, msg, w, l});
    }
    g_cv.notify_all();
    return true;
}
// Bloquea hasta que haya un mensaje para este hilo; devuelve 0 con WM_QUIT.
int32_t get(uint32_t lpMsg) {
    const uint32_t tid = static_cast<uint32_t>(gettid());
    std::unique_lock<std::mutex> lk(g_m);
    g_cv.wait(lk, [&] { auto it = g_queues.find(tid); return it != g_queues.end() && !it->second.empty(); });
    const Msg m = g_queues[tid].front();
    g_queues[tid].pop_front();
    lk.unlock();
    if (lpMsg != 0) {
        const uint32_t out[7] = {m.hwnd, m.message, m.wParam, m.lParam, static_cast<uint32_t>(monoMs()), 0, 0};
        std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(lpMsg)), out, sizeof(out));
    }
    return m.message == 0x0012 ? 0 : 1;  // WM_QUIT
}
// ShowCursor: contador de visualización (empieza en 0: hay "ratón").
std::atomic<int32_t> g_cursorCount{0};
}  // namespace threadmsg

namespace w32more {
constexpr int32_t kENotImpl = static_cast<int32_t>(0x80004001u);
constexpr int32_t kEInvalidArg = static_cast<int32_t>(0x80070057u);
constexpr int32_t kClassNotReg = static_cast<int32_t>(0x80040154u);  // REGDB_E_CLASSNOTREG

size_t len16(const char16_t* s) {
    size_t n = 0;
    if (s != nullptr) while (s[n] != 0) ++n;
    return n;
}

// --- SHLWAPI ---
int32_t PathCreateFromUrlW(const char16_t* url, char16_t* path, uint32_t* cch, uint32_t) {
    std::u16string s = u16of(url);
    const std::u16string pre = u"file:///";
    if (s.size() < pre.size()) return kEInvalidArg;
    for (size_t i = 0; i < pre.size(); ++i) {
        char16_t c = s[i];
        if (c >= u'A' && c <= u'Z') c = static_cast<char16_t>(c + 32);
        if (c != pre[i]) return kEInvalidArg;
    }
    s = s.substr(pre.size());
    std::u16string r;
    auto hexv = [](char16_t c) -> int {
        if (c >= u'0' && c <= u'9') return c - u'0';
        if (c >= u'a' && c <= u'f') return c - u'a' + 10;
        if (c >= u'A' && c <= u'F') return c - u'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == u'%' && i + 2 < s.size() && hexv(s[i + 1]) >= 0 && hexv(s[i + 2]) >= 0) {
            r.push_back(static_cast<char16_t>(hexv(s[i + 1]) * 16 + hexv(s[i + 2])));
            i += 2;
        } else {
            r.push_back(s[i] == u'/' ? u'\\' : s[i]);
        }
    }
    return urlCopyOut(r, path, cch);
}
char16_t* PathAddBackslashW(char16_t* p) {
    if (p == nullptr) return nullptr;
    size_t n = len16(p);
    if (n == 0 || p[n - 1] != u'\\') { p[n++] = u'\\'; p[n] = 0; }
    return p + n;
}
char16_t* StrStrW(char16_t* s, const char16_t* f) {
    if (s == nullptr || f == nullptr) return nullptr;
    const size_t m = len16(f);
    for (; *s != 0; ++s) if (std::memcmp(s, f, m * 2) == 0) return s;
    return m == 0 ? s : nullptr;
}
char16_t* StrRChrW(char16_t* start, const char16_t* end, uint32_t ch) {
    if (start == nullptr) return nullptr;
    const char16_t* e = end != nullptr ? end : start + len16(start);
    char16_t* last = nullptr;
    for (char16_t* p = start; p < e && *p != 0; ++p) if (*p == static_cast<char16_t>(ch)) last = p;
    return last;
}
int32_t StrCmpNW(const char16_t* a, const char16_t* b, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        const char16_t x = a != nullptr ? a[i] : 0, y = b != nullptr ? b[i] : 0;
        if (x != y) return x < y ? -1 : 1;
        if (x == 0) return 0;
    }
    return 0;
}
void PathRemoveExtensionW(char16_t* p) {
    if (p == nullptr) return;
    for (size_t i = len16(p); i > 0; --i) {
        if (p[i - 1] == u'\\' || p[i - 1] == u'/') return;
        if (p[i - 1] == u'.') { p[i - 1] = 0; return; }
    }
}
int32_t PathIsUNCW(const char16_t* p) { return p != nullptr && p[0] == u'\\' && p[1] == u'\\' ? 1 : 0; }
int32_t UrlEscapeW(const char16_t* url, char16_t* out, uint32_t* cch, uint32_t) { return urlCopyOut(u16of(url), out, cch); }
int32_t UrlCombineW(const char16_t* base, const char16_t* rel, char16_t* out, uint32_t* cch, uint32_t) {
    std::u16string r = u16of(rel);
    if (hp_UrlIsW(rel, 0) || (r.size() > 1 && r[1] == u':') || (!r.empty() && (r[0] == u'\\' || r[0] == u'/')))
        return urlCopyOut(r, out, cch);
    std::u16string b = u16of(base);
    const size_t cut = b.find_last_of(u"/\\");
    b = cut == std::u16string::npos ? std::u16string() : b.substr(0, cut + 1);
    return urlCopyOut(b + r, out, cch);
}
int32_t UrlGetPartW(const char16_t* url, char16_t* out, uint32_t* cch, uint32_t part, uint32_t) {
    const std::u16string s = u16of(url);
    const size_t colon = s.find(u':');
    if (part == 1 && colon != std::u16string::npos && colon > 1) return urlCopyOut(s.substr(0, colon), out, cch);  // URL_PART_SCHEME
    return static_cast<int32_t>(0x80004005u);  // E_FAIL
}

// --- WinRT HSTRING: {flags (1 = referencia), longitud, puntero a los datos}. ---
struct HStr {
    uint32_t flags;
    uint32_t len;
    const char16_t* data;  // en 32 bits el puntero cabe en los 20 bytes de HSTRING_HEADER
};
const char16_t kEmpty16[1] = {0};
int32_t WindowsCreateString(const char16_t* src, uint32_t len, void** out) {
    if (out == nullptr) return kEInvalidArg;
    if (len == 0) { lowmem::storePtr(out, uint64_t{0}); return 0; }
    if (src == nullptr) return kEInvalidArg;
    auto* h = static_cast<uint8_t*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 0,
                                                                         sizeof(HStr) + (len + 1) * 2));
    if (h == nullptr) return static_cast<int32_t>(0x8007000Eu);
    auto* data = reinterpret_cast<char16_t*>(h + sizeof(HStr));
    std::memcpy(data, src, len * 2);
    data[len] = 0;
    *reinterpret_cast<HStr*>(h) = HStr{0, len, data};
    lowmem::storePtr(out, h);
    return 0;
}
int32_t WindowsCreateStringReference(const char16_t* src, uint32_t len, void* header, void** out) {
    if (out == nullptr || header == nullptr) return kEInvalidArg;
    if (len == 0) { lowmem::storePtr(out, uint64_t{0}); return 0; }
    *static_cast<HStr*>(header) = HStr{1, len, src};
    lowmem::storePtr(out, header);
    return 0;
}
int32_t WindowsDeleteString(void* h) {
    if (h != nullptr && (static_cast<HStr*>(h)->flags & 1u) == 0)
        NativeDllInterceptor::win_HeapFree(NativeDllInterceptor::win_GetProcessHeap(), 0, h);
    return 0;
}
uint32_t WindowsGetStringLen(void* h) { return h != nullptr ? static_cast<HStr*>(h)->len : 0u; }
int32_t WindowsIsStringEmpty(void* h) { return WindowsGetStringLen(h) == 0 ? 1 : 0; }
const char16_t* WindowsGetStringRawBuffer(void* h, uint32_t* len) {
    if (len != nullptr) *len = WindowsGetStringLen(h);
    return h != nullptr ? static_cast<HStr*>(h)->data : kEmpty16;
}
int32_t WindowsDuplicateString(void* h, void** out) {
    uint32_t n = 0;
    const char16_t* d = WindowsGetStringRawBuffer(h, &n);
    return WindowsCreateString(d, n, out);
}
int32_t WindowsCompareStringOrdinal(void* a, void* b, int32_t* result) {
    uint32_t na = 0, nb = 0;
    const char16_t* da = WindowsGetStringRawBuffer(a, &na);
    const char16_t* db = WindowsGetStringRawBuffer(b, &nb);
    int32_t r = 0;
    for (uint32_t i = 0; i < std::min(na, nb) && r == 0; ++i) if (da[i] != db[i]) r = da[i] < db[i] ? -1 : 1;
    if (r == 0 && na != nb) r = na < nb ? -1 : 1;
    if (result != nullptr) *result = r;
    return 0;
}
int32_t RoGetActivationFactory(void*, const void*, void** out) {
    lowmem::storePtr(out, uint64_t{0});
    return kClassNotReg;
}
// --- RoGetParameterizedTypeInstanceIID: IID de una instancia de interfaz genérica de WinRT. ---
// Firma de tipos (pinterface({piid};arg;...), {iid}, delegate({iid}), rc(...), struct(...),
// enum(...), tipos básicos) y UUID v5: SHA-1 del espacio de nombres
// {11f47ad5-7b73-42c0-abae-878b1e16adee} + firma UTF-8. Los nombres que no son básicos se
// resuelven con el IRoMetaDataLocator del llamador (Locate llama a un IRoSimpleMetaDataBuilder,
// que aquí es un objeto en memoria del invitado cuyos métodos son funciones nativas).
namespace roiid {
struct Res {
    bool param = false;
    uint32_t nargs = 0;
    std::string sig;
    bool ok = false;
};
thread_local std::vector<Res*> t_res;
thread_local uint32_t t_locator = 0;
uint32_t rd32g(uint32_t a) {
    uint32_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a)), 4);
    return v;
}
uint32_t s_builder = 0;  // objeto IRoSimpleMetaDataBuilder (vtable de alias nativos)

std::string guidStr(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint8_t g[16];
    std::memcpy(g, &a, 4);
    std::memcpy(g + 4, &b, 4);
    std::memcpy(g + 8, &c, 4);
    std::memcpy(g + 12, &d, 4);
    uint32_t d1 = 0;
    uint16_t d2 = 0, d3 = 0;
    std::memcpy(&d1, g, 4);
    std::memcpy(&d2, g + 4, 2);
    std::memcpy(&d3, g + 6, 2);
    char b2[48];
    std::snprintf(b2, sizeof(b2), "{%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}", d1, d2, d3, g[8], g[9], g[10],
                  g[11], g[12], g[13], g[14], g[15]);
    return b2;
}
std::string guidStrPtr(const uint8_t* g) {
    uint32_t w[4];
    std::memcpy(w, g, 16);
    return guidStr(w[0], w[1], w[2], w[3]);
}
std::string u8(const char16_t* p) { return p != nullptr ? utf16ToUtf8(p) : std::string(); }

const char* builtin(const std::string& n) {
    static const std::unordered_map<std::string, const char*> k = {
        {"Int8", "i1"}, {"UInt8", "u1"}, {"Int16", "i2"}, {"UInt16", "u2"}, {"Int32", "i4"}, {"UInt32", "u4"},
        {"Int64", "i8"}, {"UInt64", "u8"}, {"Single", "f4"}, {"Double", "f8"}, {"Boolean", "b1"}, {"Char16", "c2"},
        {"String", "string"}, {"Guid", "g16"}, {"Object", "cinterface(IInspectable)"},
    };
    const auto it = k.find(n);
    return it == k.end() ? nullptr : it->second;
}

Res* cur() { return t_res.empty() ? nullptr : t_res.back(); }

bool locate(const char16_t* name, Res& out) {
    if (t_locator == 0 || s_builder == 0 || name == nullptr) return false;
    const uint32_t vtbl = rd32g(t_locator);
    const uint32_t fn = rd32g(vtbl);  // IRoMetaDataLocator::Locate (stdcall: this, nombre, builder)
    t_res.push_back(&out);
    const uint64_t a[3] = {t_locator, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(name)), s_builder};
    const auto hr = static_cast<int32_t>(sehimpl::callGuest(fn, a, 3));
    t_res.pop_back();
    return hr >= 0 && out.ok;
}

std::string sigOfName(const char16_t* name);
// Elementos [idx..]: un tipo y, si es parametrizado, sus argumentos a continuación.
std::string parse(const char16_t* const* el, uint32_t count, uint32_t& idx, bool& ok) {
    if (idx >= count) { ok = false; return {}; }
    const char16_t* name = el[idx++];
    if (const char* b = builtin(u8(name))) return b;
    Res r;
    if (!locate(name, r)) { ok = false; return {}; }
    if (!r.param) return r.sig;
    std::string s = r.sig;
    for (uint32_t k = 0; k < r.nargs; ++k) s += ";" + parse(el, count, idx, ok);
    return s + ")";
}
std::string parseList(uint32_t count, const void* elems, bool& ok) {
    std::vector<const char16_t*> v(count);
    for (uint32_t i = 0; i < count; ++i)
        v[i] = reinterpret_cast<const char16_t*>(uintptr_t{rd32g(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(elems)) + 4 * i)});
    uint32_t idx = 0;
    return parse(v.data(), count, idx, ok);
}
std::string sigOfName(const char16_t* name) {
    bool ok = true;
    const char16_t* one[1] = {name};
    uint32_t idx = 0;
    return parse(one, 1, idx, ok);
}

// Métodos del builder (stdcall con this; un GUID por valor ocupa 4 ranuras).
int32_t SetWinRtInterface(void*, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    if (Res* r = cur()) { r->sig = guidStr(a, b, c, d); r->ok = true; }
    return 0;
}
int32_t SetDelegate(void*, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    if (Res* r = cur()) { r->sig = "delegate(" + guidStr(a, b, c, d) + ")"; r->ok = true; }
    return 0;
}
int32_t SetInterfaceGroupSimpleDefault(void*, const char16_t* name, const char16_t* defName, const uint8_t* defIid) {
    if (Res* r = cur()) {
        r->sig = "ig(" + u8(name) + ";" + (defIid != nullptr ? guidStrPtr(defIid) : sigOfName(defName)) + ")";
        r->ok = true;
    }
    return 0;
}
int32_t SetInterfaceGroupParameterizedDefault(void*, const char16_t* name, uint32_t n, const void* elems) {
    if (Res* r = cur()) {
        bool ok = true;
        const std::string d = parseList(n, elems, ok);
        r->sig = "ig(" + u8(name) + ";" + d + ")";
        r->ok = ok;
    }
    return 0;
}
int32_t SetRuntimeClassSimpleDefault(void*, const char16_t* name, const char16_t* defName, const uint8_t* defIid) {
    if (Res* r = cur()) {
        r->sig = "rc(" + u8(name) + ";" + (defIid != nullptr ? guidStrPtr(defIid) : sigOfName(defName)) + ")";
        r->ok = true;
    }
    return 0;
}
int32_t SetRuntimeClassParameterizedDefault(void*, const char16_t* name, uint32_t n, const void* elems) {
    if (Res* r = cur()) {
        bool ok = true;
        const std::string d = parseList(n, elems, ok);
        r->sig = "rc(" + u8(name) + ";" + d + ")";
        r->ok = ok;
    }
    return 0;
}
int32_t SetStruct(void*, const char16_t* name, uint32_t n, const void* fields) {
    if (Res* r = cur()) {
        std::string s = "struct(" + u8(name);
        for (uint32_t i = 0; i < n; ++i) {
            const auto* f = reinterpret_cast<const char16_t*>(
                uintptr_t{rd32g(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(fields)) + 4 * i)});
            s += ";" + sigOfName(f);
        }
        r->sig = s + ")";
        r->ok = true;
    }
    return 0;
}
int32_t SetEnum(void*, const char16_t* name, const char16_t* base) {
    if (Res* r = cur()) { r->sig = "enum(" + u8(name) + ";" + sigOfName(base) + ")"; r->ok = true; }
    return 0;
}
int32_t SetParameterized(void*, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t nargs) {
    if (Res* r = cur()) {
        r->param = true;
        r->nargs = nargs;
        r->sig = "pinterface(" + guidStr(a, b, c, d);
        r->ok = true;
    }
    return 0;
}

// SHA-1 (FIPS 180-1).
void sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::vector<uint8_t> m(data, data + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(len) * 8;
    for (int i = 7; i >= 0; --i) m.push_back(static_cast<uint8_t>(bits >> (i * 8)));
    auto rol = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t{m[off + 4 * i]} << 24) | (uint32_t{m[off + 4 * i + 1]} << 16) |
                   (uint32_t{m[off + 4 * i + 2]} << 8) | m[off + 4 * i + 3];
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6u; }
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<uint8_t>(h[i] >> (24 - 8 * j));
}

uint32_t builder() {
    static std::once_flag once;
    std::call_once(once, [] {
        void* mem = lowmem::mmapBelow4G(4096);
        if (mem == MAP_FAILED) return;
        auto* p = static_cast<uint8_t*>(mem);
        void* fns[10] = {
            w32arity::hpFn(&SetWinRtInterface), w32arity::hpFn(&SetDelegate),
            w32arity::hpFn(&SetInterfaceGroupSimpleDefault), w32arity::hpFn(&SetInterfaceGroupParameterizedDefault),
            w32arity::hpFn(&SetRuntimeClassSimpleDefault), w32arity::hpFn(&SetRuntimeClassParameterizedDefault),
            w32arity::hpFn(&SetStruct), w32arity::hpFn(&SetEnum), w32arity::hpFn(&SetParameterized),
            w32arity::hpFn(&SetParameterized)};  // SetParameterizedDelegate: misma firma "pinterface("
        const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
        for (int i = 0; i < 10; ++i) {
            const auto a = static_cast<uint32_t>(w32alias::lowFor(fns[i], "combase.dll!IRoSimpleMetaDataBuilder"));
            std::memcpy(p + 0x100 + 4 * i, &a, 4);
        }
        const uint32_t vt = base + 0x100;
        std::memcpy(p, &vt, 4);
        s_builder = base;
    });
    return s_builder;
}
}  // namespace roiid

int32_t RoGetParameterizedTypeInstanceIID(uint32_t count, void* names, void* locator, uint8_t* iid, void* extra) {
    lowmem::storePtr(extra, uint64_t{0});
    if (count == 0 || names == nullptr || locator == nullptr || iid == nullptr || roiid::builder() == 0) return kEInvalidArg;
    const uint32_t prevLoc = roiid::t_locator;
    roiid::t_locator = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(locator));
    bool ok = true;
    const std::string sig = roiid::parseList(count, names, ok);
    roiid::t_locator = prevLoc;
    if (!ok) {
        LOGW("RoGetParameterizedTypeInstanceIID: no se pudo resolver '%s'", sig.c_str());
        return static_cast<int32_t>(0x80004005u);
    }
    static const uint8_t kNs[16] = {0x11, 0xf4, 0x7a, 0xd5, 0x7b, 0x73, 0x42, 0xc0,
                                    0xab, 0xae, 0x87, 0x8b, 0x1e, 0x16, 0xad, 0xee};
    std::vector<uint8_t> buf(kNs, kNs + 16);
    buf.insert(buf.end(), sig.begin(), sig.end());
    uint8_t h[20];
    roiid::sha1(buf.data(), buf.size(), h);
    h[6] = static_cast<uint8_t>((h[6] & 0x0F) | 0x50);  // versión 5
    h[8] = static_cast<uint8_t>((h[8] & 0x3F) | 0x80);  // variante RFC 4122
    // Bytes big-endian -> GUID (Data1..3 en little-endian).
    const uint32_t d1 = (uint32_t{h[0]} << 24) | (uint32_t{h[1]} << 16) | (uint32_t{h[2]} << 8) | h[3];
    const uint16_t d2 = static_cast<uint16_t>((h[4] << 8) | h[5]), d3 = static_cast<uint16_t>((h[6] << 8) | h[7]);
    std::memcpy(iid, &d1, 4);
    std::memcpy(iid + 4, &d2, 2);
    std::memcpy(iid + 6, &d3, 2);
    std::memcpy(iid + 8, h + 8, 8);
    static std::atomic<int> s_logs{0};
    if (s_logs.fetch_add(1) < 20) LOGI("RoGetParameterizedTypeInstanceIID: %s -> %s", sig.c_str(), roiid::guidStrPtr(iid).c_str());
    return 0;
}
int32_t RoParseTypeName(void*, uint32_t* count, void** parts) {
    if (count != nullptr) *count = 0;
    lowmem::storePtr(parts, uint64_t{0});
    return kENotImpl;
}

// --- ole32: GUID en texto y COM sin servidores registrados. ---
int32_t parseGuid(const char16_t* s, uint8_t* g) {
    if (s == nullptr || g == nullptr) return kEInvalidArg;
    std::string a;
    for (; *s != 0; ++s) if (*s != u'{' && *s != u'}' && *s != u'-') a.push_back(static_cast<char>(*s));
    if (a.size() != 32) return static_cast<int32_t>(0x800401F3u);  // CO_E_CLASSSTRING
    auto hexv = [&](size_t at, size_t n) { return std::strtoull(a.substr(at, n).c_str(), nullptr, 16); };
    const uint32_t d1 = static_cast<uint32_t>(hexv(0, 8));
    const uint16_t d2 = static_cast<uint16_t>(hexv(8, 4)), d3 = static_cast<uint16_t>(hexv(12, 4));
    std::memcpy(g, &d1, 4);
    std::memcpy(g + 4, &d2, 2);
    std::memcpy(g + 6, &d3, 2);
    for (int i = 0; i < 8; ++i) g[8 + i] = static_cast<uint8_t>(hexv(16 + 2 * i, 2));
    return 0;
}
int32_t CLSIDFromString(const char16_t* s, uint8_t* g) { return parseGuid(s, g); }
int32_t CLSIDFromProgID(const char16_t*, uint8_t*) { return static_cast<int32_t>(0x800401F3u); }
int32_t StringFromGUID2(const uint8_t* g, char16_t* out, int32_t cch) {
    if (g == nullptr || out == nullptr || cch < 39) return 0;
    uint32_t d1 = 0;
    uint16_t d2 = 0, d3 = 0;
    std::memcpy(&d1, g, 4);
    std::memcpy(&d2, g + 4, 2);
    std::memcpy(&d3, g + 6, 2);
    char b[40];
    std::snprintf(b, sizeof(b), "{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}", d1, d2, d3, g[8], g[9], g[10],
                  g[11], g[12], g[13], g[14], g[15]);
    for (int i = 0; i < 39; ++i) out[i] = static_cast<char16_t>(b[i]);
    return 39;
}
int32_t CoCreateInstanceFromApp(const void*, void*, uint32_t, void*, uint32_t, void*) { return kClassNotReg; }
int32_t CoGetClassObject(const void*, uint32_t, void*, const void*, void** ppv) {
    lowmem::storePtr(ppv, uint64_t{0});
    return kClassNotReg;
}
int32_t CoUnmarshalInterface(void*, const void*, void** ppv) {
    lowmem::storePtr(ppv, uint64_t{0});
    return kENotImpl;
}
int32_t CoMarshalInterface(void*, const void*, void*, uint32_t, void*, uint32_t) { return kENotImpl; }
int32_t CoGetMarshalSizeMax(uint32_t* size, const void*, void*, uint32_t, void*, uint32_t) {
    if (size != nullptr) *size = 0;
    return kENotImpl;
}
int32_t CoReleaseMarshalData(void*) { return kENotImpl; }
int32_t CoGetObjectContext(const void*, void** ppv) {
    lowmem::storePtr(ppv, uint64_t{0});
    return static_cast<int32_t>(0x800401F0u);  // CO_E_NOTINITIALIZED
}
int32_t CoRegisterInitializeSpy(void*, uint64_t* cookie) {
    if (cookie != nullptr) *cookie = 1;
    return 0;
}
int32_t CoRevokeInitializeSpy(int64_t) { return 0; }  // ULARGE_INTEGER por valor: 2 ranuras
int32_t CoWaitForMultipleHandles(uint32_t flags, uint32_t ms, uint32_t count, void* const* handles, uint32_t* index) {
    const uint32_t r = NativeDllInterceptor::win_WaitForMultipleObjects(count, handles, (flags & 1u) != 0 ? 1 : 0, ms);
    if (r == 0x102u) return static_cast<int32_t>(0x80010115u);  // RPC_S_CALLPENDING (tiempo agotado)
    if (r == 0xFFFFFFFFu) return static_cast<int32_t>(0x80004005u);
    if (index != nullptr) *index = r;
    return 0;
}
int32_t CreateStreamOnHGlobal(void*, int32_t, void** out) {
    lowmem::storePtr(out, uint64_t{0});
    return kENotImpl;
}
int32_t CoCreateFreeThreadedMarshaler(void*, void** out) {
    lowmem::storePtr(out, uint64_t{0});
    return kENotImpl;
}
int32_t CoRegisterClassObject(const void*, void*, uint32_t, uint32_t, uint32_t* cookie) {
    if (cookie != nullptr) *cookie = 1;
    return 0;
}

// --- Firma Authenticode, unidades de red, sesiones y Restart Manager: no disponibles. ---
int32_t WinVerifyTrust(void*, const void*, void*) { return static_cast<int32_t>(0x800B0100u); }  // TRUST_E_NOSIGNATURE
uint32_t WNetGetConnectionW(const char16_t*, char16_t*, uint32_t*) { return 2250; }               // ERROR_NOT_CONNECTED
int32_t WTSEnumerateProcessesW(void*, uint32_t, uint32_t, void** info, uint32_t* count) {
    lowmem::storePtr(info, uint64_t{0});
    if (count != nullptr) *count = 0;
    NativeDllInterceptor::win_SetLastError(50);
    return 0;
}
void WTSFreeMemory(void*) {}
uint32_t RmStartSession(uint32_t* handle, uint32_t, char16_t*) {
    if (handle != nullptr) *handle = 0;
    return 5;  // ERROR_ACCESS_DENIED: el llamador no sigue con RmRegisterResources/RmGetList
}

// --- Winsock (WSOCK32/WS2_32) sin red: arranca, pero toda operación de red falla con
// WSAENETDOWN (los juegos offline lo toleran). Conversiones de bytes y direcciones reales. ---
constexpr int32_t kSocketError = -1;
constexpr uint32_t kWsaENetDown = 10050;
thread_local uint32_t t_wsaError = 0;
int32_t wsaFail() {
    t_wsaError = kWsaENetDown;
    return kSocketError;
}
int32_t WSAStartup(uint32_t ver, uint8_t* data) {
    if (data != nullptr) {
        std::memset(data, 0, 400);
        const auto v = static_cast<uint16_t>(ver);
        std::memcpy(data, &v, 2);      // wVersion
        const uint16_t hi = 0x0202;
        std::memcpy(data + 2, &hi, 2);  // wHighVersion
        std::memcpy(data + 4, "WinSock 2.0", 12);
        std::memcpy(data + 261, "Running", 8);
    }
    return 0;
}
int32_t WSACleanup() { return 0; }
int32_t WSAGetLastError() { return static_cast<int32_t>(t_wsaError); }
void WSASetLastError(int32_t e) { t_wsaError = static_cast<uint32_t>(e); }
uint32_t SockInvalid3(int32_t, int32_t, int32_t) {
    t_wsaError = kWsaENetDown;
    return 0xFFFFFFFFu;  // INVALID_SOCKET
}
uint32_t SockAccept(uint32_t, void*, int32_t*) {
    t_wsaError = kWsaENetDown;
    return 0xFFFFFFFFu;
}
int32_t SockFail1(uint32_t) { return wsaFail(); }
int32_t SockFail2(uint32_t, int32_t) { return wsaFail(); }
int32_t SockFail3(uint32_t, const void*, int32_t) { return wsaFail(); }
int32_t SockFail4(uint32_t, void*, int32_t, int32_t) { return wsaFail(); }
int32_t SockFail5(uint32_t, int32_t, int32_t, void*, int32_t) { return wsaFail(); }
int32_t SockFail6(uint32_t, void*, int32_t, int32_t, void*, void*) { return wsaFail(); }
int32_t SockSelect(int32_t, void*, void*, void*, const void*) { return wsaFail(); }
int32_t SockAsyncSelect(uint32_t, void*, uint32_t, int32_t) { return wsaFail(); }
int32_t WSAFDIsSet(uint32_t, void*) { return 0; }
int32_t CloseSocket(uint32_t) { return 0; }
uint32_t Htonl(uint32_t v) { return __builtin_bswap32(v); }
uint32_t Htons(uint32_t v) { return __builtin_bswap16(static_cast<uint16_t>(v)); }
uint32_t InetAddr(const char* s) {
    if (s == nullptr) return 0xFFFFFFFFu;
    uint32_t parts[4] = {0, 0, 0, 0};
    int n = 0;
    for (const char* p = s; *p != 0 && n < 4; ++p) {
        if (*p == '.') { ++n; continue; }
        if (*p < '0' || *p > '9') return 0xFFFFFFFFu;  // INADDR_NONE
        parts[n] = parts[n] * 10 + static_cast<uint32_t>(*p - '0');
    }
    if (n != 3) return 0xFFFFFFFFu;
    return parts[0] | (parts[1] << 8) | (parts[2] << 16) | (parts[3] << 24);  // orden de red
}
// inet_ntoa devuelve un búfer por hilo; en 32 bits tiene que estar por debajo de 4 GB.
char* InetNtoa(uint32_t addr) {
    static thread_local char* buf = nullptr;
    if (buf == nullptr) {
        buf = static_cast<char*>(NativeDllInterceptor::win_HeapAlloc(NativeDllInterceptor::win_GetProcessHeap(), 0, 16));
        if (buf == nullptr) return nullptr;
    }
    std::snprintf(buf, 16, "%u.%u.%u.%u", addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0xFF, addr >> 24);
    return buf;
}
void* GetHostByName(const char*) {
    t_wsaError = 11001;  // WSAHOST_NOT_FOUND
    return nullptr;
}
void* GetHostByAddr(const void*, int32_t, int32_t) {
    t_wsaError = 11001;
    return nullptr;
}
int32_t GetHostName(char* name, int32_t len) {
    static const char kName[] = "ANDROID";
    if (name == nullptr || len < static_cast<int32_t>(sizeof(kName))) return wsaFail();
    std::memcpy(name, kName, sizeof(kName));
    return 0;
}
int32_t GetNameInfo(const void*, int32_t, char*, uint32_t, char*, uint32_t, int32_t) { return 11003; }  // EAI_FAIL

// --- Recursos PE (FindResource/LoadResource/SizeofResource/LockResource). HRSRC = puntero a la
// IMAGE_RESOURCE_DATA_ENTRY; HGLOBAL = puntero a los datos (como Windows con imágenes cargadas). ---
bool resModule(void* h, uint8_t*& base, const PeInfo*& info) {
    if (s_activeImage == nullptr) return false;
    base = static_cast<uint8_t*>(h != nullptr ? h : s_activeImage->baseAddress);
    info = nullptr;
    if (base == s_activeImage->baseAddress) info = &s_activeImage->peInfo;
    else
        for (const auto& kv : s_activeImage->localDlls)
            if (kv.second.base == base) { info = &kv.second.peInfo; break; }
    return info != nullptr && info->resourceRva != 0 && info->resourceSize >= 16;
}
uint32_t rdu32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rdu16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
// Clave de búsqueda: id (MAKEINTRESOURCE o "#123") o nombre en mayúsculas.
template <class C>
bool resKey(const C* k, uint32_t& id, std::u16string& name) {
    const auto v = reinterpret_cast<uintptr_t>(k);
    if ((v >> 16) == 0) { id = static_cast<uint32_t>(v); return true; }
    if (k[0] == '#') {
        id = 0;
        for (const C* p = k + 1; *p >= '0' && *p <= '9'; ++p) id = id * 10 + static_cast<uint32_t>(*p - '0');
        return true;
    }
    name.clear();
    for (const C* p = k; *p != 0; ++p) {
        char16_t c = static_cast<char16_t>(*p);
        if (c >= u'a' && c <= u'z') c = static_cast<char16_t>(c - 32);
        name.push_back(c);
    }
    return true;
}
// Entrada del directorio en dirOff con esa clave (any: la primera). Devuelve el offset del hijo.
bool resFind(const uint8_t* root, uint32_t size, uint32_t dirOff, bool any, uint32_t id, const std::u16string& name,
             uint32_t& out) {
    if (dirOff + 16 > size) return false;
    const uint8_t* dir = root + dirOff;
    const uint32_t named = rdu16(dir + 12), ids = rdu16(dir + 14);
    for (uint32_t i = 0; i < named + ids; ++i) {
        const uint32_t eOff = dirOff + 16 + i * 8;
        if (eOff + 8 > size) return false;
        const uint32_t nm = rdu32(root + eOff);
        bool match = any;
        if (!match && name.empty()) match = (nm & 0x80000000u) == 0 && nm == id;
        if (!match && !name.empty() && (nm & 0x80000000u) != 0) {
            const uint32_t sOff = nm & 0x7FFFFFFFu;
            if (sOff + 2 <= size) {
                const uint16_t len = rdu16(root + sOff);
                if (len == name.size() && sOff + 2 + len * 2u <= size) {
                    match = true;
                    for (uint16_t j = 0; j < len && match; ++j) {
                        char16_t c = static_cast<char16_t>(rdu16(root + sOff + 2 + j * 2));
                        if (c >= u'a' && c <= u'z') c = static_cast<char16_t>(c - 32);
                        match = c == name[j];
                    }
                }
            }
        }
        if (match) { out = rdu32(root + eOff + 4); return true; }
    }
    return false;
}
template <class C>
void* findResourceT(void* h, const C* nameKey, const C* typeKey, int32_t lang) {
    uint8_t* base = nullptr;
    const PeInfo* info = nullptr;
    if (!resModule(h, base, info)) { NativeDllInterceptor::win_SetLastError(1812); return nullptr; }
    const uint8_t* root = base + info->resourceRva;
    const uint32_t size = info->resourceSize;
    uint32_t tid = 0, nid = 0, e = 0;
    std::u16string tname, nname;
    resKey(typeKey, tid, tname);
    resKey(nameKey, nid, nname);
    if (!resFind(root, size, 0, false, tid, tname, e) || (e & 0x80000000u) == 0 ||
        !resFind(root, size, e & 0x7FFFFFFFu, false, nid, nname, e) || (e & 0x80000000u) == 0) {
        NativeDllInterceptor::win_SetLastError(1814);  // ERROR_RESOURCE_NAME_NOT_FOUND
        return nullptr;
    }
    const uint32_t langDir = e & 0x7FFFFFFFu;
    if (!(lang >= 0 && resFind(root, size, langDir, false, static_cast<uint32_t>(lang), u"", e)) &&
        !resFind(root, size, langDir, true, 0, u"", e)) {
        NativeDllInterceptor::win_SetLastError(1815);
        return nullptr;
    }
    if ((e & 0x80000000u) != 0 || e + 16 > size) return nullptr;
    return const_cast<uint8_t*>(root + e);
}
void* FindResourceW(void* h, const char16_t* name, const char16_t* type) { return findResourceT(h, name, type, -1); }
void* FindResourceA(void* h, const char* name, const char* type) { return findResourceT(h, name, type, -1); }
void* FindResourceExW(void* h, const char16_t* type, const char16_t* name, uint32_t lang) {
    return findResourceT(h, name, type, lang == 0 ? -1 : static_cast<int32_t>(lang));
}
void* LoadResource(void* h, const uint8_t* res) {
    uint8_t* base = nullptr;
    const PeInfo* info = nullptr;
    if (res == nullptr || !resModule(h, base, info)) return nullptr;
    return base + rdu32(res);  // OffsetToData es una RVA
}
uint32_t SizeofResource(void*, const uint8_t* res) { return res != nullptr ? rdu32(res + 4) : 0u; }
void* LockResource(void* data) { return data; }
int32_t FreeResource(void*) { return 0; }
int32_t HeapSetInformation(void*, uint32_t, void*, uint32_t) { return 1; }

// --- D3DX9 (matemáticas) ---
// D3DXMatrixInverse(out, det*, m): inversa 4x4 por cofactores; NULL si no es invertible.
float* D3DXMatrixInverse(float* out, float* det, const float* m) {
    if (out == nullptr || m == nullptr) return nullptr;
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const float d = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det != nullptr) *det = d;
    if (d == 0.0f) return nullptr;
    const float r = 1.0f / d;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * r;
    return out;
}

// D3DXCheckVersion(D3D_SDK_VERSION, D3DX_SDK_VERSION) (WINAPI): TRUE = versiones compatibles.
uint32_t D3DXCheckVersion(uint32_t, uint32_t) { return 1; }

// ReadProcessMemory (stdcall, 5): solo el proceso actual; copia si la memoria es legible (el
// manejador de errores de Unreal lee así la pila). ReportFault (stdcall, 2): sin informe a Microsoft.
uint32_t ReadProcessMemory(uint32_t process, uint32_t src, uint32_t dst, uint32_t size, uint32_t readOut) {
    uint32_t done = 0;
    const void* s = reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
    if ((process == 0xFFFFFFFFu || process == 0) && src != 0 && dst != 0 && size != 0 && isGuestRangeReadable(s, size)) {
        std::memmove(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), s, size);
        done = size;
    }
    if (readOut != 0) std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(readOut)), &done, 4);
    if (done == 0) NativeDllInterceptor::win_SetLastError(299);  // ERROR_PARTIAL_COPY
    return done != 0 ? 1 : 0;
}
uint32_t ReportFault(uint32_t, uint32_t) { return 0; }

// WinINet sin red (stdcall, firmas exactas): abrir/conectar devuelve NULL con
// ERROR_INTERNET_CANNOT_CONNECT y el juego sigue sin conexión, como en un PC sin internet.
uint32_t InetFail(uint32_t r) { NativeDllInterceptor::win_SetLastError(12029); return r; }
uint32_t InetOpen5(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return InetFail(0); }
uint32_t InetConnect8(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return InetFail(0); }
uint32_t InetOpenUrl6(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return InetFail(0); }
uint32_t InetFalse5(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return InetFail(0); }
uint32_t InetFalse4(uint32_t, uint32_t, uint32_t, uint32_t) { return InetFail(0); }
uint32_t InetTrue4(uint32_t, uint32_t, uint32_t, uint32_t) { return 1; }
uint32_t InetClose1(uint32_t) { return 1; }
uint32_t InetCallback2(uint32_t, uint32_t) { return 0; }
// Consola (stdcall): sin ventana de consola; OutputDebugStringA va a logcat.
// PIX (d3d9.dll!D3DPERF_*, stdcall): sin herramienta de perfilado, como en un PC normal.
uint32_t PerfBeginEvent(uint32_t, uint32_t) { return 0; }
uint32_t PerfEndEvent() { return 0; }
void PerfMarker(uint32_t, uint32_t) {}
void PerfSetOptions(uint32_t) {}
uint32_t PerfZero0() { return 0; }
// PostThreadMessageW/A(idThread, msg, wParam, lParam): los mensajes a hilos no se entregan todavía.
uint32_t PostThreadMessage4(uint32_t tid, uint32_t msg, uint32_t w, uint32_t l) { return threadmsg::post(tid, msg, w, l) ? 1u : 0u; }
// GetCursorInfo(PCURSORINFO): cursor oculto en (0,0).
uint32_t GetCursorInfo1(uint32_t info) {
    if (info == 0) return 0;
    auto* p = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(info));
    p[1] = 0; p[2] = 0; p[3] = 0; p[4] = 0;  // flags, hCursor, ptScreenPos
    return 1;
}
uint32_t GetConsoleWindow0() { return 0; }
uint32_t GetConsoleScreenBufferInfo2(uint32_t, uint32_t) { NativeDllInterceptor::win_SetLastError(6); return 0; }
void OutputDebugStringA1(const char* s) {
    if (s != nullptr) __android_log_print(ANDROID_LOG_INFO, "HyperPlay.DllInterceptor", "[OutputDebugString] %.400s", s);
}

void* resolve(const std::string& dll, const std::string& fn) {
    static const bool s_fastReg = []() {
        const auto reg = [](auto f) {
            OryonEmitterBackend::registerFastNative(reinterpret_cast<uintptr_t>(f), false);
        };
        reg(&D3DXMatrixInverse);
        reg(&PerfBeginEvent);
        reg(&PerfEndEvent);
        reg(&PerfMarker);
        reg(&PerfSetOptions);
        reg(&PerfZero0);
        return true;
    }();
    (void)s_fastReg;
    static const std::unordered_map<std::string, void*> kMap = {
        {"D3DXMatrixInverse", w32arity::hpFn(&D3DXMatrixInverse)},
        {"D3DXCheckVersion", w32arity::hpFn(&D3DXCheckVersion)},
        {"ReadProcessMemory", w32arity::hpFn(&ReadProcessMemory)}, {"ReportFault", w32arity::hpFn(&ReportFault)},
        {"InternetOpenW", w32arity::hpFn(&InetOpen5)}, {"InternetOpenA", w32arity::hpFn(&InetOpen5)},
        {"InternetConnectW", w32arity::hpFn(&InetConnect8)}, {"InternetConnectA", w32arity::hpFn(&InetConnect8)},
        {"HttpOpenRequestW", w32arity::hpFn(&InetConnect8)}, {"HttpOpenRequestA", w32arity::hpFn(&InetConnect8)},
        {"InternetOpenUrlA", w32arity::hpFn(&InetOpenUrl6)}, {"InternetOpenUrlW", w32arity::hpFn(&InetOpenUrl6)},
        {"HttpSendRequestW", w32arity::hpFn(&InetFalse5)}, {"HttpSendRequestA", w32arity::hpFn(&InetFalse5)},
        {"HttpQueryInfoA", w32arity::hpFn(&InetFalse5)}, {"HttpQueryInfoW", w32arity::hpFn(&InetFalse5)},
        {"InternetReadFile", w32arity::hpFn(&InetFalse4)}, {"InternetQueryOptionW", w32arity::hpFn(&InetFalse4)},
        {"InternetCrackUrlW", w32arity::hpFn(&InetFalse4)}, {"InternetSetOptionW", w32arity::hpFn(&InetTrue4)},
        {"InternetCloseHandle", w32arity::hpFn(&InetClose1)},
        {"D3DPERF_BeginEvent", w32arity::hpFn(&PerfBeginEvent)}, {"D3DPERF_EndEvent", w32arity::hpFn(&PerfEndEvent)},
        {"D3DPERF_SetMarker", w32arity::hpFn(&PerfMarker)}, {"D3DPERF_SetRegion", w32arity::hpFn(&PerfMarker)},
        {"D3DPERF_SetOptions", w32arity::hpFn(&PerfSetOptions)}, {"D3DPERF_QueryRepeatFrame", w32arity::hpFn(&PerfZero0)},
        {"D3DPERF_GetStatus", w32arity::hpFn(&PerfZero0)},
        {"PostThreadMessageW", w32arity::hpFn(&PostThreadMessage4)}, {"PostThreadMessageA", w32arity::hpFn(&PostThreadMessage4)},
        {"GetCursorInfo", w32arity::hpFn(&GetCursorInfo1)},
        {"GetConsoleWindow", w32arity::hpFn(&GetConsoleWindow0)},
        {"GetConsoleScreenBufferInfo", w32arity::hpFn(&GetConsoleScreenBufferInfo2)},
        {"OutputDebugStringA", w32arity::hpFn(&OutputDebugStringA1)},
        {"InternetSetStatusCallbackW", w32arity::hpFn(&InetCallback2)},
        {"FindResourceW", w32arity::hpFn(&FindResourceW)}, {"FindResourceA", w32arity::hpFn(&FindResourceA)},
        {"FindResourceExW", w32arity::hpFn(&FindResourceExW)}, {"LoadResource", w32arity::hpFn(&LoadResource)},
        {"SizeofResource", w32arity::hpFn(&SizeofResource)}, {"LockResource", w32arity::hpFn(&LockResource)},
        {"FreeResource", w32arity::hpFn(&FreeResource)}, {"HeapSetInformation", w32arity::hpFn(&HeapSetInformation)},
        {"WSAStartup", w32arity::hpFn(&WSAStartup)}, {"WSACleanup", w32arity::hpFn(&WSACleanup)},
        {"WSAGetLastError", w32arity::hpFn(&WSAGetLastError)}, {"WSASetLastError", w32arity::hpFn(&WSASetLastError)},
        {"socket", w32arity::hpFn(&SockInvalid3)}, {"accept", w32arity::hpFn(&SockAccept)},
        {"closesocket", w32arity::hpFn(&CloseSocket)}, {"bind", w32arity::hpFn(&SockFail3)},
        {"connect", w32arity::hpFn(&SockFail3)}, {"getpeername", w32arity::hpFn(&SockFail3)},
        {"getsockname", w32arity::hpFn(&SockFail3)}, {"ioctlsocket", w32arity::hpFn(&SockFail3)},
        {"listen", w32arity::hpFn(&SockFail2)}, {"shutdown", w32arity::hpFn(&SockFail2)},
        {"recv", w32arity::hpFn(&SockFail4)}, {"send", w32arity::hpFn(&SockFail4)},
        {"getsockopt", w32arity::hpFn(&SockFail5)}, {"setsockopt", w32arity::hpFn(&SockFail5)},
        {"recvfrom", w32arity::hpFn(&SockFail6)}, {"sendto", w32arity::hpFn(&SockFail6)},
        {"select", w32arity::hpFn(&SockSelect)}, {"WSAAsyncSelect", w32arity::hpFn(&SockAsyncSelect)},
        {"__WSAFDIsSet", w32arity::hpFn(&WSAFDIsSet)},
        {"htonl", w32arity::hpFn(&Htonl)}, {"ntohl", w32arity::hpFn(&Htonl)},
        {"htons", w32arity::hpFn(&Htons)}, {"ntohs", w32arity::hpFn(&Htons)},
        {"inet_addr", w32arity::hpFn(&InetAddr)}, {"inet_ntoa", w32arity::hpFn(&InetNtoa)},
        {"gethostbyname", w32arity::hpFn(&GetHostByName)}, {"gethostbyaddr", w32arity::hpFn(&GetHostByAddr)},
        {"gethostname", w32arity::hpFn(&GetHostName)}, {"getnameinfo", w32arity::hpFn(&GetNameInfo)},
        {"PathCreateFromUrlW", w32arity::hpFn(&PathCreateFromUrlW)}, {"PathAddBackslashW", w32arity::hpFn(&PathAddBackslashW)},
        {"StrStrW", w32arity::hpFn(&StrStrW)}, {"StrRChrW", w32arity::hpFn(&StrRChrW)}, {"StrCmpNW", w32arity::hpFn(&StrCmpNW)},
        {"PathRemoveExtensionW", w32arity::hpFn(&PathRemoveExtensionW)}, {"PathIsUNCW", w32arity::hpFn(&PathIsUNCW)},
        {"UrlEscapeW", w32arity::hpFn(&UrlEscapeW)}, {"UrlCombineW", w32arity::hpFn(&UrlCombineW)},
        {"UrlGetPartW", w32arity::hpFn(&UrlGetPartW)},
        {"WindowsCreateString", w32arity::hpFn(&WindowsCreateString)},
        {"WindowsCreateStringReference", w32arity::hpFn(&WindowsCreateStringReference)},
        {"WindowsDeleteString", w32arity::hpFn(&WindowsDeleteString)}, {"WindowsGetStringLen", w32arity::hpFn(&WindowsGetStringLen)},
        {"WindowsIsStringEmpty", w32arity::hpFn(&WindowsIsStringEmpty)},
        {"WindowsGetStringRawBuffer", w32arity::hpFn(&WindowsGetStringRawBuffer)},
        {"WindowsDuplicateString", w32arity::hpFn(&WindowsDuplicateString)},
        {"WindowsCompareStringOrdinal", w32arity::hpFn(&WindowsCompareStringOrdinal)},
        {"RoGetActivationFactory", w32arity::hpFn(&RoGetActivationFactory)},
        {"RoGetParameterizedTypeInstanceIID", w32arity::hpFn(&RoGetParameterizedTypeInstanceIID)},
        {"RoParseTypeName", w32arity::hpFn(&RoParseTypeName)},
        {"CLSIDFromString", w32arity::hpFn(&CLSIDFromString)}, {"IIDFromString", w32arity::hpFn(&CLSIDFromString)},
        {"CLSIDFromProgID", w32arity::hpFn(&CLSIDFromProgID)}, {"CLSIDFromProgIDEx", w32arity::hpFn(&CLSIDFromProgID)},
        {"StringFromGUID2", w32arity::hpFn(&StringFromGUID2)},
        {"CoCreateInstanceFromApp", w32arity::hpFn(&CoCreateInstanceFromApp)},
        {"CoGetClassObject", w32arity::hpFn(&CoGetClassObject)}, {"CoUnmarshalInterface", w32arity::hpFn(&CoUnmarshalInterface)},
        {"CoMarshalInterface", w32arity::hpFn(&CoMarshalInterface)}, {"CoGetMarshalSizeMax", w32arity::hpFn(&CoGetMarshalSizeMax)},
        {"CoReleaseMarshalData", w32arity::hpFn(&CoReleaseMarshalData)}, {"CoGetObjectContext", w32arity::hpFn(&CoGetObjectContext)},
        {"CoRegisterInitializeSpy", w32arity::hpFn(&CoRegisterInitializeSpy)},
        {"CoRevokeInitializeSpy", w32arity::hpFn(&CoRevokeInitializeSpy)},
        {"CoWaitForMultipleHandles", w32arity::hpFn(&CoWaitForMultipleHandles)},
        {"CreateStreamOnHGlobal", w32arity::hpFn(&CreateStreamOnHGlobal)},
        {"CoCreateFreeThreadedMarshaler", w32arity::hpFn(&CoCreateFreeThreadedMarshaler)},
        {"CoRegisterClassObject", w32arity::hpFn(&CoRegisterClassObject)},
        {"WinVerifyTrust", w32arity::hpFn(&WinVerifyTrust)}, {"WNetGetConnectionW", w32arity::hpFn(&WNetGetConnectionW)},
        {"WTSEnumerateProcessesW", w32arity::hpFn(&WTSEnumerateProcessesW)}, {"WTSFreeMemory", w32arity::hpFn(&WTSFreeMemory)},
        {"RmStartSession", w32arity::hpFn(&RmStartSession)},
    };
    (void)dll;
    const auto it = kMap.find(fn);
    return it == kMap.end() ? nullptr : it->second;
}
}  // namespace w32more

static uint16_t hp_GetSystemDefaultLangID() { return 0x0409; }  // en-US
static uint16_t hp_GetUserDefaultLangID() { return 0x0409; }

// GetDateFormatW / GetTimeFormatW con formato por defecto de en-US ("MM/dd/yyyy", "hh:mm:ss tt").
// Si no se da SYSTEMTIME se usa la hora local. Devuelve caracteres escritos con el NUL (o los
// necesarios si cch = 0).
static int32_t formatDateTimeW(const uint16_t* st, char16_t* out, int32_t cch, bool time) {
    uint16_t y, mo, d, h, mi, s;
    if (st != nullptr) {
        y = st[0]; mo = st[1]; d = st[3]; h = st[4]; mi = st[5]; s = st[6];
    } else {
        const time_t now = ::time(nullptr);
        struct tm lt{};
        localtime_r(&now, &lt);
        y = static_cast<uint16_t>(lt.tm_year + 1900); mo = static_cast<uint16_t>(lt.tm_mon + 1);
        d = static_cast<uint16_t>(lt.tm_mday); h = static_cast<uint16_t>(lt.tm_hour);
        mi = static_cast<uint16_t>(lt.tm_min); s = static_cast<uint16_t>(lt.tm_sec);
    }
    char buf[32];
    if (time) {
        const int h12 = h % 12 == 0 ? 12 : h % 12;
        std::snprintf(buf, sizeof(buf), "%02d:%02u:%02u %s", h12, mi, s, h < 12 ? "AM" : "PM");
    } else {
        std::snprintf(buf, sizeof(buf), "%02u/%02u/%04u", mo, d, y);
    }
    const auto need = static_cast<int32_t>(std::strlen(buf)) + 1;
    if (cch == 0 || out == nullptr) return need;
    if (cch < need) { NativeDllInterceptor::win_SetLastError(122); return 0; }
    for (int32_t i = 0; i < need; ++i) out[i] = static_cast<char16_t>(buf[i]);
    return need;
}
static int32_t hp_GetDateFormatW(uint32_t, uint32_t, const uint16_t* st, const char16_t*, char16_t* out, int32_t cch) {
    return formatDateTimeW(st, out, cch, false);
}
static int32_t hp_GetTimeFormatW(uint32_t, uint32_t, const uint16_t* st, const char16_t*, char16_t* out, int32_t cch) {
    return formatDateTimeW(st, out, cch, true);
}

// GetLongPathName / GetShortPathName: no hay nombres 8.3; la ruta es la misma. Devuelve los
// caracteres copiados sin el NUL, o el tamaño necesario con el NUL si no cabe.
template <class C>
static uint32_t samePathT(const C* in, C* out, uint32_t cch) {
    if (in == nullptr) { NativeDllInterceptor::win_SetLastError(87); return 0; }
    uint32_t n = 0;
    while (in[n] != 0) ++n;
    if (out == nullptr || cch <= n) return n + 1;
    if (out != in) std::memmove(out, in, n * sizeof(C));
    out[n] = 0;
    return n;
}
static uint32_t hp_GetLongPathNameW(const char16_t* in, char16_t* out, uint32_t cch) { return samePathT(in, out, cch); }

// Contextos de activación (manifiestos SxS): no hay ninguno activo. ActivationContextBasicInformation
// (1) = {hActCtx = NULL, dwFlags = 0}; el resto de consultas falla con ERROR_INVALID_PARAMETER.
static int32_t hp_QueryActCtxW(uint32_t, void*, void*, uint32_t infoClass, void* buf, uint32_t cb, uint32_t* written) {
    const uint32_t need = static_cast<uint32_t>(lowmem::guestPtrSize()) + 4;
    if (infoClass == 1) {
        if (written != nullptr) *written = need;
        if (buf == nullptr || cb < need) { NativeDllInterceptor::win_SetLastError(122); return 0; }
        std::memset(buf, 0, need);
        return 1;
    }
    if (written != nullptr) *written = 0;
    NativeDllInterceptor::win_SetLastError(87);
    return 0;
}

// CoGetContextToken: COM sin inicializar en el hilo (CO_E_NOTINITIALIZED); el CLR lo tolera.
static int32_t hp_CoGetContextToken(void* token) {
    lowmem::storePtr(token, uint64_t{0});
    return static_cast<int32_t>(0x800401F0u);
}
static uint32_t hp_GetLongPathNameA(const char* in, char* out, uint32_t cch) { return samePathT(in, out, cch); }

// FindFirstFileW = FindFirstFileExW(FindExInfoStandard, FindExSearchNameMatch). WIN32_FIND_DATAW no
// lleva punteros: misma forma en 32 y 64 bits.
static void* hp_FindFirstFileW(const void* name, void* data) {
    return NativeDllInterceptor::win_FindFirstFileExW(name, 0, data, 0, nullptr, 0);
}

// lstr*: cadenas de kernel32 (UTF-16 de 2 bytes en W). NULL se trata como cadena vacía.
static int32_t hp_lstrlenW(const char16_t* s) {
    if (s == nullptr) return 0;
    int32_t n = 0;
    while (s[n] != 0) ++n;
    return n;
}
static int32_t hp_lstrlenA(const char* s) { return s == nullptr ? 0 : static_cast<int32_t>(std::strlen(s)); }
static char16_t* hp_lstrcpyW(char16_t* d, const char16_t* s) {
    if (d == nullptr || s == nullptr) return nullptr;
    int32_t i = 0;
    for (; s[i] != 0; ++i) d[i] = s[i];
    d[i] = 0;
    return d;
}
static char16_t* hp_lstrcpynW(char16_t* d, const char16_t* s, int32_t max) {
    if (d == nullptr || s == nullptr || max <= 0) return d;
    int32_t i = 0;
    for (; i < max - 1 && s[i] != 0; ++i) d[i] = s[i];
    d[i] = 0;
    return d;
}
static char16_t* hp_lstrcatW(char16_t* d, const char16_t* s) {
    if (d == nullptr || s == nullptr) return nullptr;
    hp_lstrcpyW(d + hp_lstrlenW(d), s);
    return d;
}
static int32_t lstrCompareW(const char16_t* a, const char16_t* b, bool ci) {
    if (a == nullptr) a = u"";
    if (b == nullptr) b = u"";
    auto fold = [ci](char16_t c) { return ci && c >= u'A' && c <= u'Z' ? static_cast<char16_t>(c + 32) : c; };
    for (;; ++a, ++b) {
        const char16_t x = fold(*a), y = fold(*b);
        if (x != y) return x < y ? -1 : 1;
        if (x == 0) return 0;
    }
}
static int32_t hp_lstrcmpW(const char16_t* a, const char16_t* b) { return lstrCompareW(a, b, false); }
static int32_t hp_lstrcmpiW(const char16_t* a, const char16_t* b) { return lstrCompareW(a, b, true); }

static int32_t hp_GetThreadTimes(void*, uint64_t* creation, uint64_t* exitTime, uint64_t* kernel, uint64_t* user) {
    struct timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    const uint64_t t100 = static_cast<uint64_t>(ts.tv_sec) * 10000000ull + static_cast<uint64_t>(ts.tv_nsec) / 100u;
    auto put = [](uint64_t* p, uint64_t v) { if (p) std::memcpy(p, &v, 8); };
    put(creation, 0);
    put(exitTime, 0);
    put(kernel, 0);
    put(user, t100);
    return 1;
}

// GetVersion: compilación << 16 | menor << 8 | mayor (6.2.9200, como GetVersionEx sin manifiesto).
static uint32_t hp_GetVersion() { return (9200u << 16) | (2u << 8) | 6u; }

// SetHandleCount (Win16, solo 32 bits): devuelve el número pedido.
static uint32_t hp_SetHandleCount(uint32_t n) { return n; }

static int32_t hp_GetClipCursor(void* lpRect);                        // definidas más abajo
static int32_t hp_GetThreadIOPendingFlag(void* hThread, int32_t* lpIOIsPending);

// La cola de mensajes (wndq, más abajo) espera a esto: SDL guarda sus datos de ventana con SetPropW y
// su WndProc ignora los mensajes que llegan antes.
// Cada SetPropW nuevo vuelve a armar la activación: las ventanas auxiliares de SDL ponen props antes que
// la principal (que es la única cuyos datos encuentra el WndProc).
static std::atomic<bool> s_wndPropSet{false};
static std::atomic<int> s_wndPropGen{0};
static int32_t hp_SetPropW(void* h, const void* n, void* d) {
    const int32_t r = setProp(h, n, d, true);
    s_wndPropSet.store(true);
    s_wndPropGen.fetch_add(1);
    return r;
}
static void* hp_GetPropW(void* h, const void* n) { return getProp(h, n, true); }
static void* hp_RemovePropW(void* h, const void* n) { return removeProp(h, n, true); }
static int32_t hp_SetPropA(void* h, const void* n, void* d) { return setProp(h, n, d, false); }
static void* hp_GetPropA(void* h, const void* n) { return getProp(h, n, false); }
static void* hp_RemovePropA(void* h, const void* n) { return removeProp(h, n, false); }

// --- Locale y PSAPI que pide Mono x64 (mono.exe / mono-2.0-sgen.dll) ------------------------
static std::string guestModulePath(void* hModule);
static int32_t win_GetProcessMemoryInfo(void*, void* counters, uint32_t cb);

// --- IMM32 (editor de métodos de entrada) para juegos de 32 bits: no hay IME. Funciones stdcall con aridad
// exacta (un stub sin aridad descuadra la pila del llamador). Unreal 3 las importa para el texto asiático.
namespace imm32 {
using u32 = uint32_t;
u32 GetConversionStatus(u32, u32 a, u32 b) {
    if (a != 0) *reinterpret_cast<u32*>(static_cast<uintptr_t>(a)) = 0;
    if (b != 0) *reinterpret_cast<u32*>(static_cast<uintptr_t>(b)) = 0;
    return 0;
}
u32 IsUIMessageW(u32, u32, u32, u32) { return 0; }
u32 GetDefaultIMEWnd(u32) { return 0; }
u32 ReleaseContext(u32, u32) { return 1; }
u32 AssociateContext(u32, u32) { return 0; }
u32 CreateContext() { return 0x1C0; }
u32 GetContext(u32) { return 0; }
u32 DestroyContext(u32) { return 1; }
u32 GetCompositionStringW(u32, u32, u32, u32) { return 0; }
u32 GetOpenStatus(u32) { return 0; }
u32 GetIMEFileNameA(u32, u32, u32) { return 0; }
u32 SetConversionStatus(u32, u32, u32) { return 1; }
u32 SetOpenStatus(u32, u32) { return 1; }
u32 NotifyIME(u32, u32, u32, u32) { return 0; }
u32 SetCandidateWindow(u32, u32) { return 1; }
u32 SetCompositionWindow(u32, u32) { return 1; }
u32 GetProperty(u32, u32) { return 0; }
u32 GetCandidateListW(u32, u32, u32, u32) { return 0; }
u32 GetVirtualKey(u32) { return 0; }
u32 AssociateContextEx(u32, u32, u32) { return 1; }
u32 SetCompositionStringW(u32, u32, u32, u32, u32, u32) { return 0; }

void* resolve(const std::string& f) {
    if (f.rfind("Imm", 0) != 0) return nullptr;
    const std::string n = f.substr(3);
    if (n == "GetConversionStatus") return w32arity::hpFn(&GetConversionStatus);
    if (n == "IsUIMessageW" || n == "IsUIMessageA") return w32arity::hpFn(&IsUIMessageW);
    if (n == "GetDefaultIMEWnd") return w32arity::hpFn(&GetDefaultIMEWnd);
    if (n == "ReleaseContext") return w32arity::hpFn(&ReleaseContext);
    if (n == "AssociateContext") return w32arity::hpFn(&AssociateContext);
    if (n == "CreateContext") return w32arity::hpFn(&CreateContext);
    if (n == "GetContext") return w32arity::hpFn(&GetContext);
    if (n == "DestroyContext") return w32arity::hpFn(&DestroyContext);
    if (n == "GetCompositionStringW" || n == "GetCompositionStringA") return w32arity::hpFn(&GetCompositionStringW);
    if (n == "GetOpenStatus") return w32arity::hpFn(&GetOpenStatus);
    if (n == "GetIMEFileNameA" || n == "GetIMEFileNameW") return w32arity::hpFn(&GetIMEFileNameA);
    if (n == "SetConversionStatus") return w32arity::hpFn(&SetConversionStatus);
    if (n == "SetOpenStatus") return w32arity::hpFn(&SetOpenStatus);
    if (n == "NotifyIME") return w32arity::hpFn(&NotifyIME);
    if (n == "SetCandidateWindow") return w32arity::hpFn(&SetCandidateWindow);
    if (n == "SetCompositionWindow") return w32arity::hpFn(&SetCompositionWindow);
    if (n == "GetProperty") return w32arity::hpFn(&GetProperty);
    if (n == "GetCandidateListW" || n == "GetCandidateListA") return w32arity::hpFn(&GetCandidateListW);
    if (n == "GetVirtualKey") return w32arity::hpFn(&GetVirtualKey);
    if (n == "AssociateContextEx") return w32arity::hpFn(&AssociateContextEx);
    if (n == "SetCompositionStringW" || n == "SetCompositionStringA") return w32arity::hpFn(&SetCompositionStringW);
    return nullptr;
}
}  // namespace imm32

// GetComputerNameW/A(lpBuffer, lpnSize) (stdcall). Nombre fijo; ERROR_BUFFER_OVERFLOW si no cabe.
static constexpr char kComputerName[] = "EXEDRA64";
static int32_t win_GetComputerNameW(uint16_t* buf, uint32_t* size) {
    constexpr uint32_t n = sizeof(kComputerName) - 1;
    if (size == nullptr) return 0;
    if (buf == nullptr || *size < n + 1) { *size = n + 1; return 0; }
    for (uint32_t i = 0; i <= n; ++i) buf[i] = static_cast<uint8_t>(kComputerName[i]);
    *size = n;
    return 1;
}
static int32_t win_GetComputerNameA(char* buf, uint32_t* size) {
    constexpr uint32_t n = sizeof(kComputerName) - 1;
    if (size == nullptr) return 0;
    if (buf == nullptr || *size < n + 1) { *size = n + 1; return 0; }
    std::memcpy(buf, kComputerName, n + 1);
    *size = n;
    return 1;
}
// SetThreadIdealProcessor(hThread, dwIdealProcessor) (stdcall): el planificador de Android decide; devuelve
// el "ideal" anterior (0).
static uint32_t win_SetThreadIdealProcessor(void*, uint32_t) { return 0; }
// SetThreadAffinityMask(hThread, mask) (stdcall): sin efecto (el planificador fija los núcleos); devuelve la máscara
// anterior, distinta de 0 = éxito.
static uintptr_t win_SetThreadAffinityMask(void*, uintptr_t) { return 0xFF; }

static uint32_t hp_GetThreadLocale() { return 0x0409; }  // en-US

// GetLocaleInfoA: lo mínimo que consulta Mono (página de códigos y nombres ISO), en-US.
static int32_t hp_GetLocaleInfoA(uint32_t, uint32_t lcType, char* data, int32_t size) {
    const char* v = "";
    switch (lcType & 0x0FFFFFFFu) {  // sin LOCALE_NOUSEROVERRIDE / LOCALE_RETURN_NUMBER
        case 0x1004: v = "1252"; break;     // LOCALE_IDEFAULTANSICODEPAGE
        case 0x000B: v = "437"; break;      // LOCALE_IDEFAULTCODEPAGE (OEM)
        case 0x0059: v = "en"; break;       // LOCALE_SISO639LANGNAME
        case 0x005A: v = "US"; break;       // LOCALE_SISO3166CTRYNAME
        case 0x0002: v = "English (United States)"; break;  // LOCALE_SLANGUAGE
        default: break;
    }
    const int32_t need = static_cast<int32_t>(std::strlen(v)) + 1;
    if (data == nullptr || size == 0) return need;
    if (size < need) { NativeDllInterceptor::win_SetLastError(122); return 0; }
    std::memcpy(data, v, static_cast<size_t>(need));
    return need;
}

static uint32_t hp_K32GetModuleBaseNameW(void*, void* hModule, char16_t* name, uint32_t size) {
    const std::string path = guestModulePath(hModule);
    if (path.empty() || name == nullptr || size == 0) { NativeDllInterceptor::win_SetLastError(126); return 0; }
    const size_t slash = path.find_last_of("\\/");
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(base.size()), size - 1);
    for (uint32_t i = 0; i < n; ++i) name[i] = static_cast<char16_t>(static_cast<unsigned char>(base[i]));
    name[n] = u'\0';
    return n;
}

// Juego .NET con el CRT de Microsoft en el runtime (bin/ucrtbase.dll): las importaciones del CRT
// van a esa DLL (traducida). Unity enlaza su CRT estático y nunca llega aquí.
static bool guestCrtDll(const std::string& lowerDll) {
    if (!managed::active()) return false;
    if (lowerDll.rfind("api-ms-win-crt-", 0) != 0 && lowerDll != "ucrtbase.dll" && lowerDll != "msvcrt.dll") return false;
    return access((managed::g_runtimeDir + "/bin/ucrtbase.dll").c_str(), F_OK) == 0;
}

// msvcrt.dll (FNA3D/FAudio/theorafile, compilados con MinGW): casi todo lo exporta ucrtbase.dll
// con el mismo nombre; estas seis no existen en el UCRT. Antes msvcrt.dll entero eran stubs que
// devolvían 0 (memcpy no copiaba nada y FNA3D se caía).
alignas(16) static uint8_t s_fakeIob[3 * 48] = {};
static void* hp_msvcrt_iob_func() { return s_fakeIob; }
static void hp_msvcrt_lock(int) {}
static void hp_msvcrt_unlock(int) {}
static void hp_msvcrt_amsg_exit(int code) {
    LOGE("msvcrt!_amsg_exit(%d)", code);
    NativeDllInterceptor::win_ExitProcess(255);
}
static int hp_msvcrt_fprintf(void*, const char* fmt) {
    if (fmt != nullptr) __android_log_print(ANDROID_LOG_INFO, "HyperPlay.GameOut", "[fprintf] %s", fmt);
    return 0;
}
static void* msvcrtOnlyExport(const std::string& fn) {
    if (fn == "__iob_func") return w32arity::hpFn(&hp_msvcrt_iob_func);
    if (fn == "_lock") return w32arity::hpFn(&hp_msvcrt_lock);
    if (fn == "_unlock") return w32arity::hpFn(&hp_msvcrt_unlock);
    if (fn == "_amsg_exit") return w32arity::hpFn(&hp_msvcrt_amsg_exit);
    if (fn == "fprintf" || fn == "vfprintf") return w32arity::hpFn(&hp_msvcrt_fprintf);
    return nullptr;
}

// Funciones "seguras" de cadenas del UCRT, nativas y con la semántica exacta de Windows (errno_t,
// _TRUNCATE, STRUNCATE, destino vaciado si no cabe). CoreCLR parte y une los nombres de tipo con
// ellas; las versiones SSE del ucrtbase.dll traducido dejaban a veces nombres sin terminar.
namespace hostcrt {
constexpr size_t kTruncate = static_cast<size_t>(-1);
constexpr int kEinval = 22, kErange = 34, kStruncate = 80;

template <typename C>
size_t lenN(const C* s, size_t max) {
    size_t n = 0;
    while (n < max && s[n] != 0) ++n;
    return n;
}
template <typename C>
int cpyS(C* dst, size_t size, const C* src) {
    if (dst == nullptr || size == 0) return kEinval;
    if (src == nullptr) { dst[0] = 0; return kEinval; }
    const size_t n = lenN(src, size);
    if (n >= size) { dst[0] = 0; return kErange; }
    std::memcpy(dst, src, n * sizeof(C));
    dst[n] = 0;
    return 0;
}
template <typename C>
int ncpyS(C* dst, size_t size, const C* src, size_t count) {
    if (count == 0 && dst == nullptr && size == 0) return 0;
    if (dst == nullptr || size == 0) return kEinval;
    if (count == 0) { dst[0] = 0; return 0; }
    if (src == nullptr) { dst[0] = 0; return kEinval; }
    if (count == kTruncate) {
        const size_t n = lenN(src, size);
        if (n == size) {
            std::memcpy(dst, src, (size - 1) * sizeof(C));
            dst[size - 1] = 0;
            return kStruncate;
        }
        std::memcpy(dst, src, n * sizeof(C));
        dst[n] = 0;
        return 0;
    }
    const size_t n = lenN(src, count);
    if (n >= size) { dst[0] = 0; return kErange; }
    std::memcpy(dst, src, n * sizeof(C));
    dst[n] = 0;
    return 0;
}
template <typename C>
int ncatS(C* dst, size_t size, const C* src, size_t count) {
    if (count == 0 && dst == nullptr && size == 0) return 0;
    if (dst == nullptr || size == 0) return kEinval;
    if (src == nullptr && count != 0) { dst[0] = 0; return kEinval; }
    const size_t dlen = lenN(dst, size);
    if (dlen == size) { dst[0] = 0; return kEinval; }
    const size_t room = size - dlen;  // incluye el terminador
    if (count == kTruncate) {
        const size_t n = lenN(src, room);
        if (n == room) {
            std::memcpy(dst + dlen, src, (room - 1) * sizeof(C));
            dst[size - 1] = 0;
            return kStruncate;
        }
        std::memcpy(dst + dlen, src, n * sizeof(C));
        dst[dlen + n] = 0;
        return 0;
    }
    const size_t n = count == 0 ? 0 : lenN(src, count);
    if (n >= room) { dst[0] = 0; return kErange; }
    std::memcpy(dst + dlen, src, n * sizeof(C));
    dst[dlen + n] = 0;
    return 0;
}

int strcpy_s(char* d, size_t z, const char* s) { return cpyS(d, z, s); }
int strncpy_s(char* d, size_t z, const char* s, size_t c) { return ncpyS(d, z, s, c); }
int strcat_s(char* d, size_t z, const char* s) { return ncatS(d, z, s, s == nullptr ? 0 : lenN(s, kTruncate - 1)); }
int strncat_s(char* d, size_t z, const char* s, size_t c) { return ncatS(d, z, s, c); }
int wcscpy_s(char16_t* d, size_t z, const char16_t* s) { return cpyS(d, z, s); }
int wcsncpy_s(char16_t* d, size_t z, const char16_t* s, size_t c) { return ncpyS(d, z, s, c); }
int wcscat_s(char16_t* d, size_t z, const char16_t* s) { return ncatS(d, z, s, s == nullptr ? 0 : lenN(s, kTruncate - 1)); }
int wcsncat_s(char16_t* d, size_t z, const char16_t* s, size_t c) { return ncatS(d, z, s, c); }
size_t strnlen(const char* s, size_t m) { return s == nullptr ? 0 : lenN(s, m); }
size_t wcsnlen(const char16_t* s, size_t m) { return s == nullptr ? 0 : lenN(s, m); }

void* resolve(const std::string& fn) {
    if (fn == "strcpy_s") return w32arity::hpFn(&strcpy_s);
    if (fn == "strncpy_s") return w32arity::hpFn(&strncpy_s);
    if (fn == "strcat_s") return w32arity::hpFn(&strcat_s);
    if (fn == "strncat_s") return w32arity::hpFn(&strncat_s);
    if (fn == "wcscpy_s") return w32arity::hpFn(&wcscpy_s);
    if (fn == "wcsncpy_s") return w32arity::hpFn(&wcsncpy_s);
    if (fn == "wcscat_s") return w32arity::hpFn(&wcscat_s);
    if (fn == "wcsncat_s") return w32arity::hpFn(&wcsncat_s);
    if (fn == "strnlen") return w32arity::hpFn(&strnlen);
    if (fn == "wcsnlen") return w32arity::hpFn(&wcsnlen);
    return nullptr;
}
}  // namespace hostcrt

static void* resolvePeExport(void* base, const PeInfo& pe, const char* lpProcName, int depth);
static void ensureGuestGl();



// Funciones de mensajes de ventana con la firma de 32 bits (stdcall, WPARAM/LPARAM de 4 bytes).
namespace msg32 {
uint32_t Post4(uint32_t, uint32_t, uint32_t, uint32_t) { return 1; }
uint32_t Send4(uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
uint32_t CallProc5(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
uint32_t SendTimeout7(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t result) {
    if (result != 0) { const uint32_t z = 0; std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(result)), &z, 4); }
    return 1;
}
void* resolve(const std::string& fn) {
    if (fn == "PostMessageW" || fn == "PostMessageA" || fn == "SendNotifyMessageW" || fn == "SendNotifyMessageA")
        return w32arity::hpFn(&Post4);
    if (fn == "SendMessageW" || fn == "SendMessageA" || fn == "DefWindowProcW" || fn == "DefWindowProcA")
        return w32arity::hpFn(&Send4);
    if (fn == "CallWindowProcW" || fn == "CallWindowProcA") return w32arity::hpFn(&CallProc5);
    if (fn == "SendMessageTimeoutW" || fn == "SendMessageTimeoutA") return w32arity::hpFn(&SendTimeout7);
    return nullptr;
}
}  // namespace msg32
namespace lowmem {
void traceMap(void* p, size_t len, int prot, void* caller) {
    if (g_traceMaps.fetch_sub(1) <= 0) return;
    Dl_info di{};
    const char* sym = (dladdr(caller, &di) != 0 && di.dli_sname != nullptr) ? di.dli_sname : "?";
    const uintptr_t off = di.dli_fbase != nullptr ? reinterpret_cast<uintptr_t>(caller) - reinterpret_cast<uintptr_t>(di.dli_fbase) : 0;
    __android_log_print(ANDROID_LOG_INFO, "HyperPlay.DllInterceptor", "[Mapas] 0x%lx..0x%lx prot=%d desde %s+0x%lx",
                        static_cast<unsigned long>(reinterpret_cast<uintptr_t>(p)),
                        static_cast<unsigned long>(reinterpret_cast<uintptr_t>(p) + len), prot, sym, static_cast<unsigned long>(off));
}
}  // namespace lowmem
namespace guestcwd {  // definido más abajo
uint32_t SetCurrentDirectoryA(const char* p);
uint32_t GetCurrentDirectoryA(uint32_t len, char* buf);
}  // namespace guestcwd

void* NativeDllInterceptor::resolveNativeSymbol(const std::string& dllName, const std::string& funcName) {
    const std::string dll = toLower(dllName);
    void* result = nullptr;

    // 1. Steam API. Nunca se carga la steam_api(64).dll real del juego (sin cliente Steam falla;
    // la de Cuphead es un emulador que saltaba a una IAT sin resolver). Para el exe se usa
    // NativeSteamStub (Steam simulado); para DLL locales como CSteamworks.dll (Steamworks.NET) el
    // modo sin Steam, coherente con el P/Invoke. Lo no implementado devuelve 0 (stub dinámico).
    if (dll.find("steam_api") != std::string::npos) {
        // Juegos de 32 bits: Steam propio de 32 bits (objetos < 4 GB, métodos thiscall). El de 64
        // bits de abajo no cambia.
        if (s_activeImage != nullptr && s_activeImage->is32Bit) {
            if (void* p32 = steamemu32::resolve(funcName, s_activeImage->savesDir)) {
                s_steamCalls.fetch_add(1);
                registerTrampolineAddress(p32, dll + "!" + funcName);
                return p32;
            }
        }
        void* p = t_resolvingLocalDllImports ? resolveSteamOffline(funcName.c_str())
                                             : NativeSteamStub::resolveSymbol(funcName);
        if (p == nullptr) p = DynamicStubManager::getOrCreateStub(dllName, funcName);
        if (p) {
            s_steamCalls.fetch_add(1);
            registerTrampolineAddress(p, dll + "!" + funcName);
            return p;
        }
    }

    // 2. XInput & DirectInput
    if (!result && (dll.find("xinput") != std::string::npos || dll == "dinput8.dll")) {
        s_xinputCalls.fetch_add(1);
        if (funcName == "XInputGetState" || funcName == "XInputGetStateEx") {
            result = w32arity::hpFn(&DirectXInputBridge::XInputGetState);
        }
        else if (funcName == "XInputSetState") result = w32arity::hpFn(&DirectXInputBridge::XInputSetState);
        else if (funcName == "XInputGetCapabilities") result = w32arity::hpFn(&DirectXInputBridge::XInputGetCapabilities);
        else if (funcName == "XInputEnable") result = w32arity::hpFn(&DirectXInputBridge::XInputEnable);
        else if (funcName == "XInputGetBatteryInformation") {
            result = w32arity::hpFn(&DirectXInputBridge::XInputGetBatteryInformation);
        }
        else if (funcName == "XInputGetKeystroke") result = w32arity::hpFn(&DirectXInputBridge::XInputGetKeystroke);
    }

    if (!result && dll == "msvcrt.dll" && guestCrtDll(dll)) result = msvcrtOnlyExport(funcName);
    if (!result && guestCrtDll(dll)) {
        char v[PROP_VALUE_MAX] = {};
        const bool guestOnly = __system_property_get("debug.hyperplay.guestcrtstr", v) > 0 && v[0] == '1';
        if (!guestOnly) result = hostcrt::resolve(funcName);
    }

    // 3. Kernel32 & Ntdll & ApiSet (el CRT de un juego .NET lo pone ucrtbase.dll, paso 6)
    if (!result && !guestCrtDll(dll) &&
        (dll == "kernel32.dll" || dll == "ntdll.dll" || dll == "kernelbase.dll" || dll == "ucrtbase.dll" ||
         dll == "psapi.dll" || dll.rfind("api-ms-win-", 0) == 0)) {
        if (dll == "kernel32.dll") s_kernel32Calls.fetch_add(1);
        else s_ntdllCalls.fetch_add(1);

        // Memoria
        if (funcName == "VirtualAlloc") result = w32arity::hpFn(&win_VirtualAlloc);
        else if (funcName == "VirtualFree") result = w32arity::hpFn(&win_VirtualFree);
        else if (funcName == "VirtualProtect") result = w32arity::hpFn(&win_VirtualProtect);
        else if (funcName == "GetProcessHeap") result = w32arity::hpFn(&win_GetProcessHeap);
        else if (funcName == "HeapAlloc") result = w32arity::hpFn(&win_HeapAlloc);
        else if (funcName == "HeapFree") result = w32arity::hpFn(&win_HeapFree);
        else if (funcName == "HeapSize") result = w32arity::hpFn(&win_HeapSize);
        else if (funcName == "HeapReAlloc") result = w32arity::hpFn(&win_HeapReAlloc);
        else if (funcName == "VirtualQuery") result = w32arity::hpFn(&win_VirtualQuery);
        else if (funcName == "HeapCreate") result = w32arity::hpFn(&win_HeapCreate);
        else if (funcName == "HeapDestroy") result = w32arity::hpFn(&win_HeapDestroy);

        // Archivos y E/S
        else if (funcName == "CreateFileW") result = w32arity::hpFn(&win_CreateFileW);
        else if (funcName == "CreateFileA") result = w32arity::hpFn(&win_CreateFileA);
        else if (funcName == "ReadFile") result = w32arity::hpFn(&win_ReadFile);
        else if (funcName == "WriteFile") result = w32arity::hpFn(&win_WriteFile);
        else if (funcName == "CloseHandle") result = w32arity::hpFn(&win_CloseHandle);
        else if (funcName == "GetFileSize") result = w32arity::hpFn(&win_GetFileSize);
        else if (funcName == "GetFileSizeEx") result = w32arity::hpFn(&win_GetFileSizeEx);
        else if (funcName == "SetFilePointer") result = w32arity::hpFn(&win_SetFilePointer);
        else if (funcName == "SetFilePointerEx") result = w32arity::hpFn(&win_SetFilePointerEx);
        else if (funcName == "FlushFileBuffers") result = w32arity::hpFn(&win_FlushFileBuffers);
        else if (funcName == "CreateFileMappingW") result = w32arity::hpFn(&win_CreateFileMappingW);
        else if (funcName == "CreateFileMappingA") result = w32arity::hpFn(&win_CreateFileMappingA);
        else if (funcName == "MapViewOfFile") result = w32arity::hpFn(&win_MapViewOfFile);
        else if (funcName == "MapViewOfFileEx") result = w32arity::hpFn(&win_MapViewOfFileEx);
        else if (funcName == "UnmapViewOfFile") result = w32arity::hpFn(&win_UnmapViewOfFile);
        else if (funcName == "FlushViewOfFile") result = w32arity::hpFn(&win_FlushViewOfFile);
        else if (funcName == "OpenFileMappingW") result = w32arity::hpFn(&win_OpenFileMappingW);
        else if (funcName == "OpenFileMappingA") result = w32arity::hpFn(&win_OpenFileMappingA);
        else if (funcName == "FindFirstFileExW") result = w32arity::hpFn(&win_FindFirstFileExW);
        else if (funcName == "FindFirstFileW") result = w32arity::hpFn(&hp_FindFirstFileW);
        else if (funcName == "QueryActCtxW") result = w32arity::hpFn(&hp_QueryActCtxW);
        else if (funcName == "GetLongPathNameW" || funcName == "GetShortPathNameW")
            result = w32arity::hpFn(&hp_GetLongPathNameW);
        else if (funcName == "GetLongPathNameA" || funcName == "GetShortPathNameA")
            result = w32arity::hpFn(&hp_GetLongPathNameA);
        else if (funcName == "GetSystemDefaultLangID") result = w32arity::hpFn(&hp_GetSystemDefaultLangID);
        else if (funcName == "GetUserDefaultLangID") result = w32arity::hpFn(&hp_GetUserDefaultLangID);
        else if (funcName == "GetDateFormatW") result = w32arity::hpFn(&hp_GetDateFormatW);
        else if (funcName == "GetTimeFormatW") result = w32arity::hpFn(&hp_GetTimeFormatW);
        else if (funcName == "lstrlenW") result = w32arity::hpFn(&hp_lstrlenW);
        else if (funcName == "lstrlenA" || funcName == "lstrlen") result = w32arity::hpFn(&hp_lstrlenA);
        else if (funcName == "lstrcpyW") result = w32arity::hpFn(&hp_lstrcpyW);
        else if (funcName == "lstrcpynW") result = w32arity::hpFn(&hp_lstrcpynW);
        else if (funcName == "lstrcatW") result = w32arity::hpFn(&hp_lstrcatW);
        else if (funcName == "lstrcmpW") result = w32arity::hpFn(&hp_lstrcmpW);
        else if (funcName == "lstrcmpiW") result = w32arity::hpFn(&hp_lstrcmpiW);
        else if (funcName == "FindNextFileW") result = w32arity::hpFn(&win_FindNextFileW);
        else if (funcName == "FindClose") result = w32arity::hpFn(&win_FindClose);
        else if (funcName == "GetCurrentDirectoryW") result = w32arity::hpFn(&win_GetCurrentDirectoryW);
        else if (funcName == "SetCurrentDirectoryW") result = w32arity::hpFn(&win_SetCurrentDirectoryW);
        else if (funcName == "GetDriveTypeW") result = w32arity::hpFn(&win_GetDriveTypeW);
        else if (funcName == "GetLogicalDriveStringsW") result = w32arity::hpFn(&win_GetLogicalDriveStringsW);
        else if (funcName == "GetDiskFreeSpaceExW") result = w32arity::hpFn(&win_GetDiskFreeSpaceExW);
        else if (funcName == "GetVolumeInformationW") result = w32arity::hpFn(&win_GetVolumeInformationW);
        else if (funcName == "GetFullPathNameW") result = w32arity::hpFn(&win_GetFullPathNameW);
        else if (funcName == "GetFileInformationByHandle") result = w32arity::hpFn(&win_GetFileInformationByHandle);
        else if (funcName == "GetFileInformationByHandleEx") result = w32arity::hpFn(&win_GetFileInformationByHandleEx);

        // Procesos, Hilos y Consola
        else if (funcName == "GetCurrentProcessId") result = w32arity::hpFn(&win_GetCurrentProcessId);
        else if (funcName == "GetCurrentProcess") result = w32arity::hpFn(&win_GetCurrentProcess);
        else if (funcName == "GetStartupInfoW") result = w32arity::hpFn(&win_GetStartupInfoW);
        else if (funcName == "GetCommandLineA") result = w32arity::hpFn(&win_GetCommandLineA);
        else if (funcName == "GetCommandLineW") result = w32arity::hpFn(&win_GetCommandLineW);
        else if (funcName == "GetEnvironmentStringsW") result = w32arity::hpFn(&win_GetEnvironmentStringsW);
        else if (funcName == "FreeEnvironmentStringsW") result = w32arity::hpFn(&win_FreeEnvironmentStringsW);
        else if (funcName == "GetEnvironmentVariableW") result = w32arity::hpFn(&win_GetEnvironmentVariableW);
        else if (funcName == "GetEnvironmentVariableA") result = w32arity::hpFn(&win_GetEnvironmentVariableA);
        else if (funcName == "SetEnvironmentVariableW") result = w32arity::hpFn(&win_SetEnvironmentVariableW);
        else if (funcName == "GetStdHandle") result = w32arity::hpFn(&win_GetStdHandle);
        else if (funcName == "SetStdHandle") result = w32arity::hpFn(&win_SetStdHandle);
        else if (funcName == "WriteConsoleW") result = w32arity::hpFn(&win_WriteConsoleW);
        else if (funcName == "GetConsoleOutputCP") result = w32arity::hpFn(&win_GetConsoleOutputCP);
        else if (funcName == "GetConsoleCP") result = w32arity::hpFn(&win_GetConsoleCP);
        else if (funcName == "GetConsoleMode") result = w32arity::hpFn(&win_GetConsoleMode);
        else if (funcName == "GetFileType") result = w32arity::hpFn(&win_GetFileType);
        else if (funcName == "GetFileAttributesA") result = w32arity::hpFn(&win_GetFileAttributesA);
        else if (funcName == "GetFileAttributesW") result = w32arity::hpFn(&win_GetFileAttributesW);
        else if (funcName == "CreateDirectoryW") result = w32arity::hpFn(&win_CreateDirectoryW);
        else if (funcName == "MoveFileExW") result = w32arity::hpFn(&winfile::MoveFileExW);
        else if (funcName == "MoveFileExA") result = w32arity::hpFn(&winfile::MoveFileExA);
        else if (funcName == "MoveFileW") result = w32arity::hpFn(&winfile::MoveFileW);
        else if (funcName == "MoveFileA") result = w32arity::hpFn(&winfile::MoveFileA);
        else if (funcName == "DeleteFileW") result = w32arity::hpFn(&winfile::DeleteFileW);
        else if (funcName == "DeleteFileA") result = w32arity::hpFn(&winfile::DeleteFileA);
        else if (funcName == "RemoveDirectoryW") result = w32arity::hpFn(&winfile::RemoveDirectoryW);
        else if (funcName == "RemoveDirectoryA") result = w32arity::hpFn(&winfile::RemoveDirectoryA);
        else if (funcName == "CopyFileW") result = w32arity::hpFn(&winfile::CopyFileW);
        else if (funcName == "CopyFileA") result = w32arity::hpFn(&winfile::CopyFileA);
        else if (funcName == "CopyFileExW") result = w32arity::hpFn(&winfile::CopyFileExW);
        else if (funcName == "ReplaceFileW") result = w32arity::hpFn(&winfile::ReplaceFileW);
        else if (funcName == "NtQueryDirectoryFile") result = w32arity::hpFn(&winfile::NtQueryDirectoryFile);
        else if (funcName == "CreateDirectoryA") result = w32arity::hpFn(&win_CreateDirectoryA);
        else if (funcName == "GetFileAttributesExW") result = w32arity::hpFn(&win_GetFileAttributesExW);
        else if (funcName == "GetFileAttributesExA") result = w32arity::hpFn(&win_GetFileAttributesExA);
        else if (funcName == "GetModuleFileNameW") result = w32arity::hpFn(&win_GetModuleFileNameW);
        else if (funcName == "K32GetProcessMemoryInfo" || funcName == "GetProcessMemoryInfo")
            result = w32arity::hpFn(&win_GetProcessMemoryInfo);
        else if (funcName == "K32EnumProcessModules" || funcName == "EnumProcessModules" ||
                 funcName == "K32EnumProcessModulesEx" || funcName == "EnumProcessModulesEx")
            result = w32arity::hpFn(&win_EnumProcessModules);
        else if (funcName == "K32GetModuleInformation" || funcName == "GetModuleInformation")
            result = w32arity::hpFn(&win_GetModuleInformation);
        else if (funcName == "K32GetModuleFileNameExW" || funcName == "GetModuleFileNameExW")
            result = w32arity::hpFn(&win_GetModuleFileNameExW);
        else if (funcName == "ExitThread" || funcName == "RtlExitUserThread") result = w32arity::hpFn(&hp_ExitThread);
        else if (funcName == "NtCreateFile") result = w32arity::hpFn(&hp_NtCreateFile);
        else if (funcName == "NtQueryDirectoryFile") result = w32arity::hpFn(&hp_NtQueryDirectoryFile);
        else if (funcName == "RtlNtStatusToDosError") result = w32arity::hpFn(&hp_RtlNtStatusToDosError);
        else if (funcName == "K32GetModuleBaseNameW" || funcName == "GetModuleBaseNameW")
            result = w32arity::hpFn(&hp_K32GetModuleBaseNameW);
        else if (funcName == "GetThreadLocale") result = w32arity::hpFn(&hp_GetThreadLocale);
        else if (funcName == "GetLocaleInfoA") result = w32arity::hpFn(&hp_GetLocaleInfoA);
        else if (funcName == "GetModuleFileNameA") result = w32arity::hpFn(&win_GetModuleFileNameA);
        else if (funcName == "GetModuleHandleExW") result = w32arity::hpFn(&win_GetModuleHandleExW);
        else if (funcName == "GetTempPathW") result = w32arity::hpFn(&win_GetTempPathW);
        else if (funcName == "GetTempPathA") result = w32arity::hpFn(&win_GetTempPathA);
        else if (funcName == "SetConsoleCtrlHandler") result = w32arity::hpFn(&win_SetConsoleCtrlHandler);
        else if (funcName == "ExitProcess") result = w32arity::hpFn(&win_ExitProcess);
        else if (funcName == "TerminateProcess") result = w32arity::hpFn(&win_TerminateProcess);
        else if (funcName == "GetComputerNameW") result = w32arity::hpFn(&win_GetComputerNameW);
        else if (funcName == "GetComputerNameA") result = w32arity::hpFn(&win_GetComputerNameA);
        else if (funcName == "SetThreadIdealProcessor") result = w32arity::hpFn(&win_SetThreadIdealProcessor);
        else if (funcName == "SetThreadAffinityMask") result = w32arity::hpFn(&win_SetThreadAffinityMask);
        else if (funcName == "IsProcessorFeaturePresent") result = w32arity::hpFn(&win_IsProcessorFeaturePresent);
        else if (funcName == "IsDebuggerPresent") result = w32arity::hpFn(&win_IsDebuggerPresent);
        else if (funcName == "GetNativeSystemInfo") result = w32arity::hpFn(&win_GetSystemInfo);
        else if (funcName == "GlobalMemoryStatusEx") result = w32arity::hpFn(&win_GlobalMemoryStatusEx);
        else if (funcName == "GlobalMemoryStatus") result = w32arity::hpFn(&hp_GlobalMemoryStatus);
        else if (funcName == "GetWindowsDirectoryA" || funcName == "GetSystemWindowsDirectoryA")
            result = w32arity::hpFn(&hp_GetWindowsDirectoryA);
        else if (funcName == "GetWindowsDirectoryW" || funcName == "GetSystemWindowsDirectoryW")
            result = w32arity::hpFn(&hp_GetWindowsDirectoryW);
        else if (funcName == "GetSystemDirectoryA") result = w32arity::hpFn(&hp_GetSystemDirectoryA);
        else if (funcName == "GetSystemDirectoryW") result = w32arity::hpFn(&hp_GetSystemDirectoryW);
        else if (funcName == "CreatePipe") result = w32arity::hpFn(&win_CreatePipe);
        else if (funcName == "SetErrorMode") result = w32arity::hpFn(&win_SetErrorMode);
        else if (funcName == "SetHandleInformation") result = w32arity::hpFn(&win_SetHandleInformation);
        else if (funcName == "LocalFree" || funcName == "GlobalFree") result = w32arity::hpFn(&win_LocalFree);
        else if (funcName == "LocalAlloc" || funcName == "GlobalAlloc") result = w32arity::hpFn(&win_LocalAlloc);
        else if (funcName == "GlobalLock") result = w32arity::hpFn(&win_GlobalLock);
        else if (funcName == "GlobalUnlock") result = w32arity::hpFn(&win_GlobalUnlock);
        else if (funcName == "DuplicateHandle") result = w32arity::hpFn(&win_DuplicateHandle);
        else if (funcName == "InitializeProcThreadAttributeList") result = w32arity::hpFn(&win_InitializeProcThreadAttributeList);
        else if (funcName == "UpdateProcThreadAttribute") result = w32arity::hpFn(&win_UpdateProcThreadAttribute);
        else if (funcName == "DeleteProcThreadAttributeList") result = w32arity::hpFn(&win_DeleteProcThreadAttributeList);
        else if (funcName == "DisableThreadLibraryCalls") result = w32arity::hpFn(&win_DisableThreadLibraryCalls);
        else if (funcName == "GetCurrentProcessorNumberEx") result = w32arity::hpFn(&win_GetCurrentProcessorNumberEx);
        else if (funcName == "FlushProcessWriteBuffers") result = w32arity::hpFn(&win_FlushProcessWriteBuffers);
        else if (funcName == "FlushInstructionCache") result = w32arity::hpFn(&win_FlushInstructionCache);
        else if (funcName == "CreateThreadpoolWork") result = w32arity::hpFn(&win_CreateThreadpoolWork);
        else if (funcName == "SubmitThreadpoolWork") result = w32arity::hpFn(&win_SubmitThreadpoolWork);
        else if (funcName == "WaitForThreadpoolWorkCallbacks") result = w32arity::hpFn(&win_WaitForThreadpoolWorkCallbacks);
        else if (funcName == "CloseThreadpoolWork") result = w32arity::hpFn(&win_CloseThreadpoolWork);
        else if (funcName == "GetThreadIOPendingFlag") result = w32arity::hpFn(&hp_GetThreadIOPendingFlag);
        else if (funcName == "GetVersionExA") result = w32arity::hpFn(&hp_GetVersionExA);
        else if (funcName == "GetVersionExW") result = w32arity::hpFn(&hp_GetVersionExW);
        else if (funcName == "GetStartupInfoA") result = w32arity::hpFn(&hp_GetStartupInfoA);
        else if (funcName == "SetHandleCount") result = w32arity::hpFn(&hp_SetHandleCount);
        else if (funcName == "GetThreadTimes") result = w32arity::hpFn(&hp_GetThreadTimes);
        else if (funcName == "GetVersion") result = w32arity::hpFn(&hp_GetVersion);
        else if (funcName == "InterlockedIncrement" || funcName == "InterlockedDecrement" ||
                 funcName == "InterlockedExchange" || funcName == "InterlockedExchangeAdd" ||
                 funcName == "InterlockedCompareExchange") {
            static const bool s_fastInterlocked = []() {
                const auto reg = [](auto f) {
                    OryonEmitterBackend::registerFastNative(reinterpret_cast<uintptr_t>(f), false);
                };
                reg(&hp_InterlockedIncrement);
                reg(&hp_InterlockedDecrement);
                reg(&hp_InterlockedExchange);
                reg(&hp_InterlockedExchangeAdd);
                reg(&hp_InterlockedCompareExchange);
                return true;
            }();
            (void)s_fastInterlocked;
            if (funcName == "InterlockedIncrement") result = w32arity::hpFn(&hp_InterlockedIncrement);
            else if (funcName == "InterlockedDecrement") result = w32arity::hpFn(&hp_InterlockedDecrement);
            else if (funcName == "InterlockedExchange") result = w32arity::hpFn(&hp_InterlockedExchange);
            else if (funcName == "InterlockedExchangeAdd") result = w32arity::hpFn(&hp_InterlockedExchangeAdd);
            else result = w32arity::hpFn(&hp_InterlockedCompareExchange);
        }
        else if (funcName == "InterlockedCompareExchange64") result = w32arity::hpFn(&hp_InterlockedCompareExchange64);
        else if (funcName == "CreateIoCompletionPort") result = w32arity::hpFn(&win_CreateIoCompletionPort);
        else if (funcName == "GetQueuedCompletionStatus") result = w32arity::hpFn(&win_GetQueuedCompletionStatus);
        else if (funcName == "GetQueuedCompletionStatusEx") result = w32arity::hpFn(&win_GetQueuedCompletionStatusEx);
        else if (funcName == "PostQueuedCompletionStatus") result = w32arity::hpFn(&win_PostQueuedCompletionStatus);
        else if (funcName == "GetProcessTimes") result = w32arity::hpFn(&win_GetProcessTimes);
        else if (funcName == "GetSystemTimes") result = w32arity::hpFn(&win_GetSystemTimes);
        else if (funcName == "FormatMessageW") result = w32arity::hpFn(&win_FormatMessageW);
        else if (funcName == "FormatMessageA") result = w32arity::hpFn(&win_FormatMessageA);

        // Errores, Excepciones y Unwind
        else if (funcName == "GetLastError") result = w32arity::hpFn(&win_GetLastError);
        else if (funcName == "SetLastError") result = w32arity::hpFn(&win_SetLastError);
        else if (funcName == "SetUnhandledExceptionFilter") result = w32arity::hpFn(&win_SetUnhandledExceptionFilter);
        else if (funcName == "UnhandledExceptionFilter") result = w32arity::hpFn(&win_UnhandledExceptionFilter);
        else if (funcName == "RaiseException") result = w32arity::hpFn(&win_RaiseException);
        else if (funcName == "RtlCaptureContext") result = w32arity::hpFn(&win_RtlCaptureContext);
        else if (funcName == "RtlLookupFunctionEntry") result = w32arity::hpFn(&win_RtlLookupFunctionEntry);
        else if (funcName == "RtlVirtualUnwind") result = w32arity::hpFn(&win_RtlVirtualUnwind);
        else if (funcName == "RtlUnwindEx") result = w32arity::hpFn(&win_RtlUnwindEx);
        // SEH x86 (solo invitados de 32 bits; en x64 RtlUnwind/NtContinue siguen sin tocarse).
        else if (funcName == "RtlUnwind" && s_activeImage != nullptr && s_activeImage->is32Bit)
            result = w32arity::hpFn(&seh32::hp_RtlUnwind32);
        else if ((funcName == "NtContinue" || funcName == "ZwContinue") && s_activeImage != nullptr &&
                 s_activeImage->is32Bit)
            result = w32arity::hpFn(&seh32::hp_NtContinue32);
        else if (funcName == "RtlPcToFileHeader") result = w32arity::hpFn(&win_RtlPcToFileHeader);
        else if (funcName == "EncodePointer") result = w32arity::hpFn(&win_EncodePointer);
        else if (funcName == "DecodePointer") result = w32arity::hpFn(&win_DecodePointer);
        else if (funcName == "InitializeSListHead") result = w32arity::hpFn(&win_InitializeSListHead);
        else if (funcName == "AddVectoredExceptionHandler") result = w32arity::hpFn(&win_AddVectoredExceptionHandler);
        else if (funcName == "RemoveVectoredExceptionHandler") result = w32arity::hpFn(&win_RemoveVectoredExceptionHandler);
        else if (funcName == "SetThreadStackGuarantee") result = w32arity::hpFn(&win_SetThreadStackGuarantee);

        // Hilos, Sincronización, TLS y FLS
        else if (funcName == "CreateThread") result = w32arity::hpFn(&win_CreateThread);
        else if (funcName == "GetCurrentThreadId") result = w32arity::hpFn(&win_GetCurrentThreadId);
        else if (funcName == "Sleep") result = w32arity::hpFn(&win_Sleep);
        else if (funcName == "SleepEx") result = w32arity::hpFn(&win_SleepEx);
        else if (funcName == "InitializeCriticalSection") result = w32arity::hpFn(&win_InitializeCriticalSection);
        else if (funcName == "InitializeCriticalSectionAndSpinCount") result = w32arity::hpFn(&win_InitializeCriticalSectionAndSpinCount);
        else if (funcName == "InitializeCriticalSectionEx") result = w32arity::hpFn(&win_InitializeCriticalSectionEx);
        else if (funcName == "EnterCriticalSection") result = w32arity::hpFn(&win_EnterCriticalSection);
        else if (funcName == "TryEnterCriticalSection") result = w32arity::hpFn(&win_TryEnterCriticalSection);
        else if (funcName == "LeaveCriticalSection") result = w32arity::hpFn(&win_LeaveCriticalSection);
        else if (funcName == "DeleteCriticalSection") result = w32arity::hpFn(&win_DeleteCriticalSection);
        else if (funcName == "InitializeSRWLock") result = w32arity::hpFn(&win_InitializeSRWLock);
        else if (funcName == "AcquireSRWLockExclusive") result = w32arity::hpFn(&win_AcquireSRWLockExclusive);
        else if (funcName == "ReleaseSRWLockExclusive") result = w32arity::hpFn(&win_ReleaseSRWLockExclusive);
        else if (funcName == "TryAcquireSRWLockExclusive") result = w32arity::hpFn(&win_TryAcquireSRWLockExclusive);
        else if (funcName == "AcquireSRWLockShared") result = w32arity::hpFn(&win_AcquireSRWLockShared);
        else if (funcName == "ReleaseSRWLockShared") result = w32arity::hpFn(&win_ReleaseSRWLockShared);
        else if (funcName == "TryAcquireSRWLockShared") result = w32arity::hpFn(&win_TryAcquireSRWLockShared);
        else if (funcName == "SleepConditionVariableSRW") result = w32arity::hpFn(&win_SleepConditionVariableSRW);
        else if (funcName == "SleepConditionVariableCS") result = w32arity::hpFn(&win_SleepConditionVariableCS);
        else if (funcName == "WakeAllConditionVariable") result = w32arity::hpFn(&win_WakeAllConditionVariable);
        else if (funcName == "WakeConditionVariable") result = w32arity::hpFn(&win_WakeConditionVariable);
        else if (funcName == "WaitOnAddress") result = w32arity::hpFn(&win_WaitOnAddress);
        else if (funcName == "WaitOnAddressCheck") result = w32arity::hpFn(&win_WaitOnAddress);
        else if (funcName == "WakeByAddress") result = w32arity::hpFn(&win_WakeByAddress);
        else if (funcName == "WakeByAddressSingle") result = w32arity::hpFn(&win_WakeByAddress);
        else if (funcName == "WakeByAddressAll") result = w32arity::hpFn(&win_WakeByAddressAll);
        else if (funcName == "CreateEventA") result = w32arity::hpFn(&win_CreateEventA);
        else if (funcName == "CreateEventW") result = w32arity::hpFn(&win_CreateEventW);
        else if (funcName == "CreateEventExW") result = w32arity::hpFn(&win_CreateEventExW);
        else if (funcName == "CreateWaitableTimerW" || funcName == "CreateWaitableTimerA") {
            result = w32arity::hpFn(&win_CreateWaitableTimerW);
        }
        else if (funcName == "CreateWaitableTimerExW" || funcName == "CreateWaitableTimerExA") {
            result = w32arity::hpFn(&win_CreateWaitableTimerExW);
        }
        else if (funcName == "SetWaitableTimer" || funcName == "SetWaitableTimerEx") {
            result = w32arity::hpFn(&win_SetWaitableTimer);
        }
        else if (funcName == "CancelWaitableTimer") result = w32arity::hpFn(&win_CancelWaitableTimer);
        else if (funcName == "SetEvent") result = w32arity::hpFn(&win_SetEvent);
        else if (funcName == "ResetEvent") result = w32arity::hpFn(&win_ResetEvent);
        else if (funcName == "WaitForSingleObject") result = w32arity::hpFn(&win_WaitForSingleObject);
        else if (funcName == "WaitForSingleObjectEx") result = w32arity::hpFn(&win_WaitForSingleObjectEx);
        else if (funcName == "WaitForMultipleObjects") result = w32arity::hpFn(&win_WaitForMultipleObjects);
        else if (funcName == "WaitForMultipleObjectsEx") result = w32arity::hpFn(&win_WaitForMultipleObjectsEx);
        else if (funcName == "CreateSemaphoreA") result = w32arity::hpFn(&win_CreateSemaphoreA);
        else if (funcName == "CreateSemaphoreW") result = w32arity::hpFn(&win_CreateSemaphoreW);
        else if (funcName == "CreateSemaphoreExA") result = w32arity::hpFn(&win_CreateSemaphoreExA);
        else if (funcName == "CreateSemaphoreExW") result = w32arity::hpFn(&win_CreateSemaphoreExW);
        else if (funcName == "ReleaseSemaphore") result = w32arity::hpFn(&win_ReleaseSemaphore);
        else if (funcName == "OpenSemaphoreW") result = w32arity::hpFn(&win_OpenSemaphoreW);
        else if (funcName == "CreateMutexA") result = w32arity::hpFn(&win_CreateMutexA);
        else if (funcName == "CreateMutexW") result = w32arity::hpFn(&win_CreateMutexW);
        else if (funcName == "CreateMutexExA") result = w32arity::hpFn(&win_CreateMutexExA);
        else if (funcName == "CreateMutexExW") result = w32arity::hpFn(&win_CreateMutexExW);
        else if (funcName == "ReleaseMutex") result = w32arity::hpFn(&win_ReleaseMutex);
        else if (funcName == "OpenMutexW") result = w32arity::hpFn(&win_OpenMutexW);
        else if (funcName == "GetExitCodeThread") result = w32arity::hpFn(&win_GetExitCodeThread);
        else if (funcName == "GetExitCodeProcess") result = w32arity::hpFn(&win_GetExitCodeProcess);
        else if (funcName == "SwitchToThread") result = w32arity::hpFn(&win_SwitchToThread);
        else if (funcName == "GetCurrentThread") result = w32arity::hpFn(&win_GetCurrentThread);
        else if (funcName == "SetThreadPriority") result = w32arity::hpFn(&win_SetThreadPriority);
        else if (funcName == "GetThreadPriority") result = w32arity::hpFn(&win_GetThreadPriority);
        else if (funcName == "ResumeThread") result = w32arity::hpFn(&win_ResumeThread);
        else if (funcName == "SuspendThread") result = w32arity::hpFn(&win_SuspendThread);
        else if (funcName == "GetThreadContext") result = w32arity::hpFn(&win_GetThreadContext);
        else if (funcName == "SetThreadContext") result = w32arity::hpFn(&win_SetThreadContext);
        else if (funcName == "OpenThread") result = w32arity::hpFn(&win_OpenThread);
        else if (funcName == "GetThreadId") result = w32arity::hpFn(&win_GetThreadId);
        else if (funcName == "SetThreadDescription") result = w32arity::hpFn(&win_SetThreadDescription);
        else if (funcName == "InterlockedPushEntrySList") result = w32arity::hpFn(&win_InterlockedPushEntrySList);
        else if (funcName == "InterlockedPopEntrySList") result = w32arity::hpFn(&win_InterlockedPopEntrySList);
        else if (funcName == "InterlockedFlushSList") result = w32arity::hpFn(&win_InterlockedFlushSList);
        else if (funcName == "TlsAlloc") result = w32arity::hpFn(&win_TlsAlloc);
        else if (funcName == "InitOnceBeginInitialize") result = w32arity::hpFn(&win_InitOnceBeginInitialize);
        else if (funcName == "InitOnceComplete") result = w32arity::hpFn(&win_InitOnceComplete);
        else if (funcName == "TlsGetValue") result = w32arity::hpFn(&win_TlsGetValue);
        else if (funcName == "TlsSetValue") result = w32arity::hpFn(&win_TlsSetValue);
        else if (funcName == "TlsFree") result = w32arity::hpFn(&win_TlsFree);
        else if (funcName == "FlsAlloc") result = w32arity::hpFn(&win_FlsAlloc);
        else if (funcName == "FlsGetValue") result = w32arity::hpFn(&win_FlsGetValue);
        else if (funcName == "FlsSetValue") result = w32arity::hpFn(&win_FlsSetValue);
        else if (funcName == "FlsFree") result = w32arity::hpFn(&win_FlsFree);

        // Tiempo & Info del sistema
        else if (funcName == "GetTickCount") result = w32arity::hpFn(&win_GetTickCount);
        else if (funcName == "GetTickCount64") result = w32arity::hpFn(&win_GetTickCount64);
        else if (funcName == "GetSystemTimeAsFileTime") result = w32arity::hpFn(&win_GetSystemTimeAsFileTime);
        else if (funcName == "GetLocalTime") result = w32arity::hpFn(&win_GetLocalTime);
        else if (funcName == "SystemTimeToFileTime") result = w32arity::hpFn(&win_SystemTimeToFileTime);
        else if (funcName == "QueryPerformanceCounter") result = w32arity::hpFn(&win_QueryPerformanceCounter);
        else if (funcName == "QueryPerformanceFrequency") result = w32arity::hpFn(&win_QueryPerformanceFrequency);
        else if (funcName == "GetSystemInfo") result = w32arity::hpFn(&win_GetSystemInfo);
        else if (funcName == "GetLogicalProcessorInformation") result = w32arity::hpFn(&win_GetLogicalProcessorInformation);
        else if (funcName == "GetLogicalProcessorInformationEx") result = w32arity::hpFn(&win_GetLogicalProcessorInformationEx);

        // Módulos
        else if (funcName == "GetModuleHandleW") result = w32arity::hpFn(&win_GetModuleHandleW);
        else if (funcName == "GetModuleHandleA") result = w32arity::hpFn(&win_GetModuleHandleA);
        else if (funcName == "GetProcAddress") result = w32arity::hpFn(&win_GetProcAddress);
        else if (funcName == "LoadLibraryW") result = w32arity::hpFn(&win_LoadLibraryW);
        else if (funcName == "LoadLibraryA") result = w32arity::hpFn(&win_LoadLibraryA);
        else if (funcName == "LoadLibraryExW") result = w32arity::hpFn(&win_LoadLibraryExW);
        else if (funcName == "FreeLibrary") result = w32arity::hpFn(&win_FreeLibrary);

        // Strings y Locale
        else if (funcName == "MultiByteToWideChar") result = w32arity::hpFn(&win_MultiByteToWideChar);
        else if (funcName == "WideCharToMultiByte") result = w32arity::hpFn(&win_WideCharToMultiByte);
        else if (funcName == "IsValidCodePage") result = w32arity::hpFn(&win_IsValidCodePage);
        else if (funcName == "GetACP") result = w32arity::hpFn(&win_GetACP);
        else if (funcName == "GetOEMCP") result = w32arity::hpFn(&win_GetOEMCP);
        else if (funcName == "GetCPInfo") result = w32arity::hpFn(&win_GetCPInfo);
        else if (funcName == "GetStringTypeW") result = w32arity::hpFn(&win_GetStringTypeW);
        else if (funcName == "LCMapStringW") result = w32arity::hpFn(&win_LCMapStringW);
        else if (funcName == "GetUserDefaultLCID") result = w32arity::hpFn(&win_GetUserDefaultLCID);
        else if (funcName == "GetUserDefaultLocaleName") result = w32arity::hpFn(&win_GetUserDefaultLocaleName);
        else if (funcName == "LoadStringW") result = w32arity::hpFn(&win_LoadStringW);
        else if (funcName == "EnumSystemLocalesW") result = w32arity::hpFn(&win_EnumSystemLocalesW);
        else if (funcName == "IsValidLocale") result = w32arity::hpFn(&win_IsValidLocale);
        else if (funcName == "CompareStringW") result = w32arity::hpFn(&win_CompareStringW);
        else if (funcName == "GetLocaleInfoW") result = w32arity::hpFn(&win_GetLocaleInfoW);
        else if (funcName == "GetLocaleInfoEx") result = w32arity::hpFn(&win_GetLocaleInfoEx);
        else if (funcName == "GetTimeZoneInformation") result = w32arity::hpFn(&win_GetTimeZoneInformation);
        else if (funcName == "SystemTimeToTzSpecificLocalTime") result = w32arity::hpFn(&win_SystemTimeToTzSpecificLocalTime);
        else if (funcName == "TzSpecificLocalTimeToSystemTime") result = w32arity::hpFn(&win_TzSpecificLocalTimeToSystemTime);
        else if (funcName == "FileTimeToSystemTime") result = w32arity::hpFn(&win_FileTimeToSystemTime);
    }

    // 4. WINMM (temporizadores multimedia de Unity/FMOD)
    if (!result && (dll == "winmm.dll" || dll == "winmm")) {
        s_otherCalls.fetch_add(1);
        if (funcName == "timeGetTime") result = w32arity::hpFn(&win_timeGetTime);
        else if (funcName == "timeBeginPeriod") result = w32arity::hpFn(&win_timeBeginPeriod);
        else if (funcName == "timeEndPeriod") result = w32arity::hpFn(&win_timeEndPeriod);
        else if (funcName == "timeGetDevCaps") result = w32arity::hpFn(&win_timeGetDevCaps);
    }

    // 4. User32 (Windows & Messages)
    // 32 bits: WPARAM/LPARAM miden 4 bytes. Las versiones de 64 bits (uint64_t/int64_t) sacaban 2 huecos
    // de pila por cada uno (PostMessageW de Unreal 3 dejaba ESP 4 bytes desplazado: fallo /GS).
    if (!result && s_activeImage != nullptr && s_activeImage->is32Bit && dll == "user32.dll") result = msg32::resolve(funcName);    if (!result && dll == "user32.dll") {
        s_otherCalls.fetch_add(1);
        if (funcName == "CreateWindowExW") result = w32arity::hpFn(&win_CreateWindowExW);
        else if (funcName == "LoadStringW") result = w32arity::hpFn(&win_LoadStringW);
        else if (funcName == "CreateWindowExA") result = w32arity::hpFn(&win_CreateWindowExA);
        else if (funcName == "ShowWindow") result = w32arity::hpFn(&win_ShowWindow);
        else if (funcName == "SetPropW") result = w32arity::hpFn(&hp_SetPropW);
        else if (funcName == "GetPropW") result = w32arity::hpFn(&hp_GetPropW);
        else if (funcName == "RemovePropW") result = w32arity::hpFn(&hp_RemovePropW);
        else if (funcName == "SetPropA") result = w32arity::hpFn(&hp_SetPropA);
        else if (funcName == "GetPropA") result = w32arity::hpFn(&hp_GetPropA);
        else if (funcName == "RemovePropA") result = w32arity::hpFn(&hp_RemovePropA);
        else if (funcName == "PeekMessageW") result = w32arity::hpFn(&win_PeekMessageW);
        else if (funcName == "DispatchMessageW") result = w32arity::hpFn(&win_DispatchMessageW);
        else if (funcName == "GetForegroundWindow") result = w32arity::hpFn(&win_GetForegroundWindow);
        else if (funcName == "GetActiveWindow") result = w32arity::hpFn(&win_GetActiveWindow);
        else if (funcName == "GetDesktopWindow") result = w32arity::hpFn(&win_GetDesktopWindow);
        else if (funcName == "GetShellWindow") result = w32arity::hpFn(&win_GetShellWindow);
        else if (funcName == "SetForegroundWindow") result = w32arity::hpFn(&win_SetForegroundWindow);
        else if (funcName == "SetActiveWindow") result = w32arity::hpFn(&win_SetActiveWindow);
        else if (funcName == "GetFocus") result = w32arity::hpFn(&win_GetFocus);
        else if (funcName == "SetFocus") result = w32arity::hpFn(&win_SetFocus);
        else if (funcName == "GetClientRect") result = w32arity::hpFn(&win_GetClientRect);
        else if (funcName == "GetWindowRect") result = w32arity::hpFn(&win_GetWindowRect);
        else if (funcName == "AdjustWindowRect") result = w32arity::hpFn(&win_AdjustWindowRect);
        else if (funcName == "AdjustWindowRectEx") result = w32arity::hpFn(&win_AdjustWindowRectEx);
        else if (funcName == "GetSystemMetrics") result = w32arity::hpFn(&win_GetSystemMetrics);
        else if (funcName == "SetWindowPos") result = w32arity::hpFn(&win_SetWindowPos);
        else if (funcName == "SetWindowTextW") result = w32arity::hpFn(&win_SetWindowTextW);
        else if (funcName == "SetWindowTextA") result = w32arity::hpFn(&win_SetWindowTextA);
        else if (funcName == "GetDC") result = w32arity::hpFn(&win_GetDC);
        else if (funcName == "ReleaseDC") result = w32arity::hpFn(&win_ReleaseDC);
        else if (funcName == "ValidateRect") result = w32arity::hpFn(&win_ValidateRect);
        else if (funcName == "InvalidateRect") result = w32arity::hpFn(&win_InvalidateRect);
        else if (funcName == "UpdateWindow") result = w32arity::hpFn(&win_UpdateWindow);
        else if (funcName == "SetCursor") result = w32arity::hpFn(&win_SetCursor);
        else if (funcName == "ShowCursor") result = w32arity::hpFn(&win_ShowCursor);
        else if (funcName == "DefWindowProcW") result = w32arity::hpFn(&win_DefWindowProcW);
        else if (funcName == "DefWindowProcA") result = w32arity::hpFn(&win_DefWindowProcA);
        else if (funcName == "PostMessageW") result = w32arity::hpFn(&win_PostMessageW);
        else if (funcName == "PostQuitMessage") result = w32arity::hpFn(&win_PostQuitMessage);
        else if (funcName == "RegisterWindowMessageW") result = w32arity::hpFn(&win_RegisterWindowMessageW);
        else if (funcName == "RegisterWindowMessageA") result = w32arity::hpFn(&win_RegisterWindowMessageA);
        else if (funcName == "MonitorFromWindow") result = w32arity::hpFn(&win_MonitorFromWindow);
        else if (funcName == "MonitorFromPoint") result = w32arity::hpFn(&win_MonitorFromPoint);
        else if (funcName == "MonitorFromRect") result = w32arity::hpFn(&win_MonitorFromRect);
        else if (funcName == "GetMonitorInfoW") result = w32arity::hpFn(&win_GetMonitorInfoW);
        else if (funcName == "GetMonitorInfoA") result = w32arity::hpFn(&win_GetMonitorInfoA);
        else if (funcName == "EnumDisplayMonitors") result = w32arity::hpFn(&win_EnumDisplayMonitors);
        else if (funcName == "EnumDisplaySettingsW") result = w32arity::hpFn(&win_EnumDisplaySettingsW);
        else if (funcName == "EnumDisplaySettingsA") result = w32arity::hpFn(&win_EnumDisplaySettingsA);
        else if (funcName == "GetWindowLongW") result = w32arity::hpFn(&win_GetWindowLongW);
        else if (funcName == "GetWindowLongA") result = w32arity::hpFn(&win_GetWindowLongA);
        else if (funcName == "GetWindowLongPtrW") result = w32arity::hpFn(&win_GetWindowLongPtrW);
        else if (funcName == "GetWindowLongPtrA") result = w32arity::hpFn(&win_GetWindowLongPtrA);
        else if (funcName == "SetWindowLongW") result = w32arity::hpFn(&win_SetWindowLongW);
        else if (funcName == "SetWindowLongA") result = w32arity::hpFn(&win_SetWindowLongA);
        else if (funcName == "SetWindowLongPtrW") result = w32arity::hpFn(&win_SetWindowLongPtrW);
        else if (funcName == "SetWindowLongPtrA") result = w32arity::hpFn(&win_SetWindowLongPtrA);
        else if (funcName == "RegisterClassW") result = w32arity::hpFn(&win_RegisterClassW);
        else if (funcName == "RegisterClassExW") result = w32arity::hpFn(&win_RegisterClassExW);
        else if (funcName == "UnregisterClassW") result = w32arity::hpFn(&win_UnregisterClassW);
        else if (funcName == "DestroyWindow") result = w32arity::hpFn(&win_DestroyWindow);
        else if (funcName == "SetTimer") result = w32arity::hpFn(&win_SetTimer);
        else if (funcName == "KillTimer") result = w32arity::hpFn(&win_KillTimer);
        else if (funcName == "SystemParametersInfoW") result = w32arity::hpFn(&win_SystemParametersInfoW);
        else if (funcName == "GetCursorPos") result = w32arity::hpFn(&win_GetCursorPos);
        else if (funcName == "SetCursorPos") result = w32arity::hpFn(&win_SetCursorPos);
        else if (funcName == "ClipCursor") result = w32arity::hpFn(&win_ClipCursor);
        else if (funcName == "GetClipCursor") result = w32arity::hpFn(&hp_GetClipCursor);
        else if (funcName == "ClientToScreen") result = w32arity::hpFn(&win_ClientToScreen);
        else if (funcName == "ScreenToClient") result = w32arity::hpFn(&win_ScreenToClient);
        else if (funcName == "AllowSetForegroundWindow") result = w32arity::hpFn(&win_AllowSetForegroundWindow);

        // Desktop & Window Station
        else if (funcName == "GetThreadDesktop") result = w32arity::hpFn(&win_GetThreadDesktop);
        else if (funcName == "GetProcessWindowStation") result = w32arity::hpFn(&win_GetProcessWindowStation);
        else if (funcName == "SetThreadDesktop") result = w32arity::hpFn(&win_SetThreadDesktop);
        else if (funcName == "SetProcessWindowStation") result = w32arity::hpFn(&win_SetProcessWindowStation);
        else if (funcName == "GetUserObjectInformationA") result = w32arity::hpFn(&win_GetUserObjectInformationA);
        else if (funcName == "GetUserObjectInformationW") result = w32arity::hpFn(&win_GetUserObjectInformationW);

        // Message Wait & Pump
        else if (funcName == "MsgWaitForMultipleObjects") result = w32arity::hpFn(&win_MsgWaitForMultipleObjects);
        else if (funcName == "MsgWaitForMultipleObjectsEx") result = w32arity::hpFn(&win_MsgWaitForMultipleObjectsEx);
        else if (funcName == "TranslateMessage") result = w32arity::hpFn(&win_TranslateMessage);
        else if (funcName == "GetMessageW") result = w32arity::hpFn(&win_GetMessageW);
        else if (funcName == "GetMessageA") result = w32arity::hpFn(&win_GetMessageA);
        else if (funcName == "PeekMessageA") result = w32arity::hpFn(&win_PeekMessageA);
        else if (funcName == "DispatchMessageA") result = w32arity::hpFn(&win_DispatchMessageA);
        else if (funcName == "SendMessageW") result = w32arity::hpFn(&win_SendMessageW);
        else if (funcName == "SendMessageA") result = w32arity::hpFn(&win_SendMessageA);
        else if (funcName == "PostMessageA") result = w32arity::hpFn(&win_PostMessageA);
        else if (funcName == "SendMessageTimeoutA") result = w32arity::hpFn(&win_SendMessageTimeoutA);

        // Displays & Monitors
        else if (funcName == "EnumDisplayDevicesA") result = w32arity::hpFn(&win_EnumDisplayDevicesA);
        else if (funcName == "EnumDisplayDevicesW") result = w32arity::hpFn(&win_EnumDisplayDevicesW);
        else if (funcName == "EnumWindows") result = w32arity::hpFn(&win_EnumWindows);

        // Raw Input & Keyboard & Mouse
        else if (funcName == "GetRawInputDeviceList") result = w32arity::hpFn(&win_GetRawInputDeviceList);
        else if (funcName == "RegisterRawInputDevices") result = w32arity::hpFn(&win_RegisterRawInputDevices);
        else if (funcName == "GetRawInputDeviceInfoW") result = w32arity::hpFn(&win_GetRawInputDeviceInfoW);
        else if (funcName == "GetRawInputData") result = w32arity::hpFn(&win_GetRawInputData);
        else if (funcName == "GetRawInputBuffer") result = w32arity::hpFn(&win_GetRawInputBuffer);
        else if (funcName == "GetKeyState") result = w32arity::hpFn(&win_GetKeyState);
        else if (funcName == "GetAsyncKeyState") result = w32arity::hpFn(&win_GetAsyncKeyState);
        else if (funcName == "MapVirtualKeyW") result = w32arity::hpFn(&win_MapVirtualKeyW);
        else if (funcName == "MapVirtualKeyA") result = w32arity::hpFn(&win_MapVirtualKeyA);
        else if (funcName == "MapVirtualKeyExA") result = w32arity::hpFn(&win_MapVirtualKeyExA);
        else if (funcName == "GetKeyNameTextW") result = w32arity::hpFn(&win_GetKeyNameTextW);
        else if (funcName == "ToUnicode") result = w32arity::hpFn(&win_ToUnicode);
        else if (funcName == "GetKeyboardLayout") result = w32arity::hpFn(&win_GetKeyboardLayout);
        else if (funcName == "GetKeyboardLayoutNameW") result = w32arity::hpFn(&win_GetKeyboardLayoutNameW);

        // Mouse & Cursor & Capture
        else if (funcName == "SetCapture") result = w32arity::hpFn(&win_SetCapture);
        else if (funcName == "ReleaseCapture") result = w32arity::hpFn(&win_ReleaseCapture);
        else if (funcName == "LoadCursorA") result = w32arity::hpFn(&win_LoadCursorA);
        else if (funcName == "LoadCursorW") result = w32arity::hpFn(&win_LoadCursorW);
        else if (funcName == "DestroyCursor") result = w32arity::hpFn(&win_DestroyCursor);
        else if (funcName == "LoadIconA") result = w32arity::hpFn(&win_LoadIconA);
        else if (funcName == "LoadIconW") result = w32arity::hpFn(&win_LoadIconW);
        else if (funcName == "DestroyIcon") result = w32arity::hpFn(&win_DestroyIcon);
        else if (funcName == "LoadImageW") result = w32arity::hpFn(&win_LoadImageW);
        else if (funcName == "LoadImageA") result = w32arity::hpFn(&win_LoadImageA);
        else if (funcName == "TrackMouseEvent") result = w32arity::hpFn(&win_TrackMouseEvent);
        else if (funcName == "EnableMouseInPointer") result = w32arity::hpFn(&win_EnableMouseInPointer);
        else if (funcName == "IsMouseInPointerEnabled") result = w32arity::hpFn(&win_IsMouseInPointerEnabled);

        // Window State & Geometry
        else if (funcName == "GetWindowPlacement") result = w32arity::hpFn(&win_GetWindowPlacement);
        else if (funcName == "SetWindowPlacement") result = w32arity::hpFn(&win_SetWindowPlacement);
        else if (funcName == "AdjustWindowRectExForDpi") result = w32arity::hpFn(&win_AdjustWindowRectExForDpi);
        else if (funcName == "IsWindowVisible") result = w32arity::hpFn(&win_IsWindowVisible);
        else if (funcName == "IsIconic") result = w32arity::hpFn(&win_IsIconic);
        else if (funcName == "MoveWindow") result = w32arity::hpFn(&win_MoveWindow);
        else if (funcName == "PtInRect") result = w32arity::hpFn(&win_PtInRect);
        else if (funcName == "OffsetRect") result = w32arity::hpFn(&win_OffsetRect);
        else if (funcName == "CopyRect") result = w32arity::hpFn(&win_CopyRect);

        // Misc User32
        else if (funcName == "GetMessageExtraInfo") result = w32arity::hpFn(&win_GetMessageExtraInfo);
        else if (funcName == "GetDoubleClickTime") result = w32arity::hpFn(&win_GetDoubleClickTime);
        else if (funcName == "GetCaretBlinkTime") result = w32arity::hpFn(&win_GetCaretBlinkTime);
        else if (funcName == "OpenClipboard") result = w32arity::hpFn(&win_OpenClipboard);
        else if (funcName == "CloseClipboard") result = w32arity::hpFn(&win_CloseClipboard);
        else if (funcName == "EmptyClipboard") result = w32arity::hpFn(&win_EmptyClipboard);
        else if (funcName == "GetClipboardData") result = w32arity::hpFn(&win_GetClipboardData);
        else if (funcName == "SetClipboardData") result = w32arity::hpFn(&win_SetClipboardData);
        else if (funcName == "IsClipboardFormatAvailable") result = w32arity::hpFn(&win_IsClipboardFormatAvailable);
        else if (funcName == "MessageBoxA") result = w32arity::hpFn(&win_MessageBoxA);
        else if (funcName == "MessageBoxW") result = w32arity::hpFn(&win_MessageBoxW);
        else if (funcName == "DialogBoxParamW") result = w32arity::hpFn(&win_DialogBoxParamW);
        else if (funcName == "EndDialog") result = w32arity::hpFn(&win_EndDialog);
        else if (funcName == "SetDlgItemTextW") result = w32arity::hpFn(&win_SetDlgItemTextW);
        else if (funcName == "SetDlgItemTextA") result = w32arity::hpFn(&win_SetDlgItemTextA);
        else if (funcName == "SendDlgItemMessageW") result = w32arity::hpFn(&win_SendDlgItemMessageW);
        else if (funcName == "RegisterDeviceNotificationW") result = w32arity::hpFn(&win_RegisterDeviceNotificationW);
        else if (funcName == "UnregisterDeviceNotification") result = w32arity::hpFn(&win_UnregisterDeviceNotification);
        else if (funcName == "SetWindowsHookExW" || funcName == "SetWindowsHookExA" || funcName == "SetWindowsHookEx") {
            result = w32arity::hpFn(&win_SetWindowsHookExW);
        }
        else if (funcName == "UnhookWindowsHookEx") result = w32arity::hpFn(&win_UnhookWindowsHookEx);
        else if (funcName == "CallNextHookEx") result = w32arity::hpFn(&win_CallNextHookEx);
        else if (funcName == "RegisterTouchWindow") result = w32arity::hpFn(&win_RegisterTouchWindow);
        else if (funcName == "GetDisplayConfigBufferSizes") result = w32arity::hpFn(&win_GetDisplayConfigBufferSizes);
        else if (funcName == "QueryDisplayConfig") result = w32arity::hpFn(&win_QueryDisplayConfig);
        else if (funcName == "DisplayConfigGetDeviceInfo") result = w32arity::hpFn(&win_DisplayConfigGetDeviceInfo);
    }

    // 4b. GDI32
    if (!result && dll == "gdi32.dll") {
        s_otherCalls.fetch_add(1);
        ensureGuestGl();
        if (void* p = guestgl::gdiProcAddr(funcName.c_str())) result = p;  // formatos de píxel, DC, mapas de bits
        else if (funcName == "ChoosePixelFormat") result = w32arity::hpFn(&win_ChoosePixelFormat);
        else if (funcName == "SetPixelFormat") result = w32arity::hpFn(&win_SetPixelFormat);
        else if (funcName == "GetDeviceCaps") result = w32arity::hpFn(&win_GetDeviceCaps);
        else if (funcName == "SwapBuffers") result = w32arity::hpFn(&win_SwapBuffers);
    }

    // 4b. Shell32 / ShFolder
    if (!result && (dll == "shell32.dll" || dll == "shfolder.dll")) {
        s_otherCalls.fetch_add(1);
        if (funcName == "CommandLineToArgvW") result = w32arity::hpFn(&win_CommandLineToArgvW);
        else if (funcName == "ShellExecuteExW") result = w32arity::hpFn(&hp_ShellExecuteExW);
        else if (funcName == "DllGetVersion") result = w32arity::hpFn(&win_DllGetVersion);
        else if (funcName == "SHGetKnownFolderPath") result = w32arity::hpFn(&win_SHGetKnownFolderPath);
        else if (funcName == "SHGetFolderPathW") result = w32arity::hpFn(&win_SHGetFolderPathW);
        else if (funcName == "SHGetFolderPathA") result = w32arity::hpFn(&win_SHGetFolderPathA);
        else if (funcName == "SHGetSpecialFolderPathW") result = w32arity::hpFn(&win_SHGetSpecialFolderPathW);
        else if (funcName == "SHGetSpecialFolderPathA") result = w32arity::hpFn(&win_SHGetSpecialFolderPathA);
        else if (funcName == "ShellExecuteW") result = w32arity::hpFn(&win_ShellExecuteW);
        else if (funcName == "ShellExecuteA") result = w32arity::hpFn(&win_ShellExecuteA);
    }

    // 4a'. Resto de importaciones de clr.dll (solo 32 bits; los juegos de 64 bits siguen igual).
    if (!result && s_activeImage != nullptr && s_activeImage->is32Bit) result = w32more::resolve(dll, funcName);
    if (!result && s_activeImage != nullptr && s_activeImage->is32Bit && dll == "kernel32.dll") {
        if (funcName == "SetCurrentDirectoryA") result = w32arity::hpFn(&guestcwd::SetCurrentDirectoryA);
        else if (funcName == "GetCurrentDirectoryA") result = w32arity::hpFn(&guestcwd::GetCurrentDirectoryA);
    }

    // 4b. OLEAUT32: BSTR, VARIANT y IErrorInfo; el resto E_NOTIMPL con su aridad exacta (en 32 bits
    // stdcall: una aridad equivocada desbalancea la pila del invitado).
    if (!result && dll == "oleaut32.dll" && s_activeImage != nullptr && s_activeImage->is32Bit) {
        using namespace oleaut;
        if (funcName == "SysAllocString") result = w32arity::hpFn(&SysAllocString);
        else if (funcName == "SysAllocStringLen") result = w32arity::hpFn(&SysAllocStringLen);
        else if (funcName == "SysAllocStringByteLen") result = w32arity::hpFn(&SysAllocStringByteLen);
        else if (funcName == "SysFreeString") result = w32arity::hpFn(&SysFreeString);
        else if (funcName == "SysStringLen") result = w32arity::hpFn(&SysStringLen);
        else if (funcName == "SysStringByteLen") result = w32arity::hpFn(&SysStringByteLen);
        else if (funcName == "VariantInit") result = w32arity::hpFn(&VariantInit);
        else if (funcName == "VariantClear") result = w32arity::hpFn(&VariantClear);
        else if (funcName == "VariantCopy") result = w32arity::hpFn(&VariantCopy);
        else if (funcName == "GetErrorInfo") result = w32arity::hpFn(&GetErrorInfo);
        else if (funcName == "SetErrorInfo") result = w32arity::hpFn(&SetErrorInfo);
        else if (funcName == "CreateErrorInfo") result = w32arity::hpFn(&CreateErrorInfo);
        else if (funcName == "SafeArrayGetDim" || funcName == "SafeArrayGetElemsize")
            result = w32arity::hpFn(&Zero1);
        else if (funcName == "SafeArrayDestroy" || funcName == "SafeArrayUnaccessData" ||
                 funcName == "SafeArrayAllocData")
            result = w32arity::hpFn(&NotImpl1);
        else if (funcName == "VariantCopyInd" || funcName == "SafeArrayAccessData" || funcName == "SafeArraySetRecordInfo" ||
                 funcName == "SafeArrayGetRecordInfo" || funcName == "SafeArrayGetVartype" || funcName == "VarDecFix" ||
                 funcName == "VarDecInt" || funcName == "VarDecFromR4" || funcName == "VarDecCmp" ||
                 funcName == "VarR4FromDec" || funcName == "VarR8FromDec" || funcName == "VarCyFromDec" ||
                 funcName == "GetRecordInfoFromTypeInfo")
            result = w32arity::hpFn(&NotImpl2);
        else if (funcName == "SafeArrayGetUBound" || funcName == "SafeArrayGetLBound" ||
                 funcName == "SafeArrayGetElement" || funcName == "SafeArrayPutElement" ||
                 funcName == "SafeArrayAllocDescriptorEx" || funcName == "VarDecMul" || funcName == "CreateTypeLib2" ||
                 funcName == "LoadTypeLibEx" || funcName == "VarDecFromR8" || funcName == "VarDecFromCy" ||
                 funcName == "VarDecRound")
            result = w32arity::hpFn(&NotImpl3);
        else if (funcName == "SafeArrayCreateVector") result = w32arity::hpFn(&Null3);
        else if (funcName == "VariantChangeType") result = w32arity::hpFn(&NotImpl4);
        else if (funcName == "VariantChangeTypeEx" || funcName == "LoadRegTypeLib" || funcName == "QueryPathOfRegTypeLib")
            result = w32arity::hpFn(&NotImpl5);
    }

    // 4b'. SHLWAPI (también expuesto vía api-ms-win-core-path / shcore)
    if (!result && (dll == "shlwapi.dll" || dll == "shcore.dll" ||
                    dll.rfind("api-ms-win-core-shlwapi", 0) == 0)) {
        if (funcName == "PathCanonicalizeW") result = w32arity::hpFn(&win_PathCanonicalizeW);
        else if (funcName == "UrlIsW") result = w32arity::hpFn(&hp_UrlIsW);
        else if (funcName == "StrCmpW") result = w32arity::hpFn(&hp_lstrcmpW);
        else if (funcName == "PathIsURLW") result = w32arity::hpFn(&hp_PathIsURLW);
        else if (funcName == "StrToIntW") result = w32arity::hpFn(&hp_StrToIntW);
        else if (funcName == "StrToIntA") result = w32arity::hpFn(&hp_StrToIntA);
        else if (funcName == "PathRemoveBackslashW") result = w32arity::hpFn(&hp_PathRemoveBackslashW);
        else if (funcName == "StrChrW") result = w32arity::hpFn(&hp_StrChrW);
        else if (funcName == "UrlCanonicalizeW") result = w32arity::hpFn(&hp_UrlCanonicalizeW);
        else if (funcName == "UrlUnescapeW") result = w32arity::hpFn(&hp_UrlUnescapeW);
        else if (funcName == "StrCmpIW") result = w32arity::hpFn(&hp_lstrcmpiW);
        else if (funcName == "PathCanonicalizeA") result = w32arity::hpFn(&win_PathCanonicalizeA);
        else if (funcName == "PathCombineW") result = w32arity::hpFn(&win_PathCombineW);
        else if (funcName == "PathCombineA") result = w32arity::hpFn(&win_PathCombineA);
        else if (funcName == "PathAppendW") result = w32arity::hpFn(&win_PathAppendW);
        else if (funcName == "PathAppendA") result = w32arity::hpFn(&win_PathAppendA);
        else if (funcName == "PathRemoveFileSpecW") result = w32arity::hpFn(&win_PathRemoveFileSpecW);
        else if (funcName == "PathRemoveFileSpecA") result = w32arity::hpFn(&win_PathRemoveFileSpecA);
        else if (funcName == "PathFindFileNameW") result = w32arity::hpFn(&win_PathFindFileNameW);
        else if (funcName == "PathFindFileNameA") result = w32arity::hpFn(&win_PathFindFileNameA);
        else if (funcName == "PathFindExtensionW") result = w32arity::hpFn(&win_PathFindExtensionW);
        else if (funcName == "PathFindExtensionA") result = w32arity::hpFn(&win_PathFindExtensionA);
        else if (funcName == "PathFileExistsW") result = w32arity::hpFn(&win_PathFileExistsW);
        else if (funcName == "PathFileExistsA") result = w32arity::hpFn(&win_PathFileExistsA);
        else if (funcName == "PathIsRelativeW") result = w32arity::hpFn(&win_PathIsRelativeW);
        else if (funcName == "PathIsRelativeA") result = w32arity::hpFn(&win_PathIsRelativeA);
    }

    // 4c. WinRT / Ole32 / ComBase / ApiSet
    if (!result && (dll.find("winrt") != std::string::npos || dll == "ole32.dll" || dll == "combase.dll" || dll.find("com-l1") != std::string::npos)) {
        s_otherCalls.fetch_add(1);
        if (funcName == "RoInitialize") result = w32arity::hpFn(&win_RoInitialize);
        else if (funcName == "CoInitialize") result = w32arity::hpFn(&win_CoInitialize);
        else if (funcName == "CoInitializeEx") result = w32arity::hpFn(&win_CoInitializeEx);
        else if (funcName == "CoGetContextToken") result = w32arity::hpFn(&hp_CoGetContextToken);
        else if (funcName == "CoUninitialize") result = w32arity::hpFn(&win_CoUninitialize);
        else if (funcName == "CoCreateInstance") result = w32arity::hpFn(&win_CoCreateInstance);
        else if (funcName == "CoTaskMemAlloc") result = w32arity::hpFn(&win_CoTaskMemAlloc);
        else if (funcName == "CoTaskMemFree") result = w32arity::hpFn(&win_CoTaskMemFree);
        else if (funcName == "CoTaskMemRealloc") result = w32arity::hpFn(&win_CoTaskMemRealloc);
        else if (funcName == "CoCreateGuid") result = w32arity::hpFn(&win_CoCreateGuid);
        else if (funcName == "PropVariantClear") result = w32arity::hpFn(&win_PropVariantClear);
    }

    // 4d. BCrypt
    if (!result && dll.find("bcrypt") != std::string::npos) {
        s_otherCalls.fetch_add(1);
        if (funcName == "BCryptGenRandom") result = w32arity::hpFn(&win_BCryptGenRandom);
        else if (funcName == "BCryptOpenAlgorithmProvider") result = w32arity::hpFn(&win_BCryptOpenAlgorithmProvider);
        else if (funcName == "BCryptCloseAlgorithmProvider") result = w32arity::hpFn(&win_BCryptCloseAlgorithmProvider);
        else if (funcName == "SystemFunction036") result = w32arity::hpFn(&win_SystemFunction036);
        else if (funcName == "RtlGenRandom") result = w32arity::hpFn(&win_SystemFunction036);
    }

    // 4e. Advapi32 Crypto & Security
    if (!result && (dll.find("advapi32") != std::string::npos || dll.find("crypt") != std::string::npos)) {
        s_otherCalls.fetch_add(1);
        if (funcName == "CryptAcquireContextW") result = w32arity::hpFn(&win_CryptAcquireContextW);
        else if (funcName == "CryptAcquireContextA") result = w32arity::hpFn(&win_CryptAcquireContextA);
        else if (funcName == "CryptReleaseContext") result = w32arity::hpFn(&win_CryptReleaseContext);
        else if (funcName == "CryptGenRandom") result = w32arity::hpFn(&win_CryptGenRandom);
        else if (funcName == "SystemFunction036") result = w32arity::hpFn(&win_SystemFunction036);
        else if (funcName == "RtlGenRandom") result = w32arity::hpFn(&win_SystemFunction036);
        else if (funcName == "OpenProcessToken") result = w32arity::hpFn(&win_OpenProcessToken);
        else if (funcName.compare(0, 3, "Reg") == 0 || funcName.compare(0, 5, "Event") == 0 ||
                 funcName == "DeregisterEventSource" || funcName == "ReportEventW" ||
                 funcName == "InitializeSecurityDescriptor" || funcName == "SetSecurityDescriptorDacl" ||
                 funcName.find("Sid") != std::string::npos || funcName.find("Acl") != std::string::npos ||
                 funcName.compare(0, 9, "AddAccess") == 0 || funcName == "CheckTokenMembership" ||
                 funcName == "IsValidSecurityDescriptor" || funcName == "GetSecurityDescriptorDacl" ||
                 funcName == "OpenThreadToken" || funcName == "GetTokenInformation" ||
                 funcName == "ConvertStringSecurityDescriptorToSecurityDescriptorW")
            result = resolveRegistryExport(funcName);
    }
    if (!result && (dll.find("registry-l") != std::string::npos || dll.find("kernelbase") != std::string::npos)) {
        result = resolveRegistryExport(funcName);
    }
    if (!result && dll.find("powrprof") != std::string::npos && funcName == "CallNtPowerInformation") {
        result = w32arity::hpFn(&win_CallNtPowerInformation);
    }
    if (!result && (dll.find("kernel32") != std::string::npos || dll.find("ntdll") != std::string::npos ||
                    dll.find("kernelbase") != std::string::npos || dll.find("api-ms-win") != std::string::npos)) {
        result = resolveSystemExtraExport(funcName);
    }

    // 4f. VERSION.dll (metadatos de versión del PE)
    if (!result && (dll == "version.dll" || dll.find("version-l1") != std::string::npos)) {
        s_otherCalls.fetch_add(1);
        if (funcName == "GetFileVersionInfoSizeW") result = w32arity::hpFn(&win_GetFileVersionInfoSizeW);
        else if (funcName == "GetFileVersionInfoSizeA") result = w32arity::hpFn(&win_GetFileVersionInfoSizeA);
        else if (funcName == "GetFileVersionInfoSizeExW") result = w32arity::hpFn(&win_GetFileVersionInfoSizeW);
        else if (funcName == "GetFileVersionInfoSizeExA") result = w32arity::hpFn(&win_GetFileVersionInfoSizeA);
        else if (funcName == "GetFileVersionInfoW") result = w32arity::hpFn(&win_GetFileVersionInfoW);
        else if (funcName == "GetFileVersionInfoA") result = w32arity::hpFn(&win_GetFileVersionInfoA);
        else if (funcName == "VerQueryValueW") result = w32arity::hpFn(&win_VerQueryValueW);
        else if (funcName == "VerQueryValueA") result = w32arity::hpFn(&win_VerQueryValueA);
    }

    // 5. Direct3D / DirectX Graphics
    if (!result && (dll == "d3d11.dll" || dll == "dxgi.dll")) {
        s_d3dCalls.fetch_add(1);
        if (funcName == "D3D11CreateDeviceAndSwapChain") result = w32arity::hpFn(&win_D3D11CreateDeviceAndSwapChain);
        else if (funcName == "D3D11CreateDevice") result = w32arity::hpFn(&win_D3D11CreateDevice);
        else if (funcName == "CreateDXGIFactory") result = w32arity::hpFn(&win_CreateDXGIFactory);
        else if (funcName == "CreateDXGIFactory1") result = w32arity::hpFn(&win_CreateDXGIFactory1);
        else if (funcName == "CreateDXGIFactory2") result = w32arity::hpFn(&win_CreateDXGIFactory2);
    }
    if (!result && dll == "dsound.dll" && s_activeImage != nullptr && s_activeImage->is32Bit) result = dsound32::resolve(funcName);
    if (!result && dll == "dinput8.dll" && s_activeImage != nullptr && s_activeImage->is32Bit) result = dinput32::resolve(funcName);
    if (!result && dll == "imm32.dll" && s_activeImage != nullptr && s_activeImage->is32Bit) result = imm32::resolve(funcName);
    if (!result && dll == "d3d9.dll") {
        s_d3dCalls.fetch_add(1);
        if (s_activeImage != nullptr && s_activeImage->is32Bit) {
            // 32 bits (Unreal Engine 3): IDirect3D9 de solo consulta; renderiza con Direct3D 11.
            result = d3d9probe32::resolve(funcName, {static_cast<uint32_t>(win32::screenWidth()),
                                                     static_cast<uint32_t>(win32::screenHeight()),
                                                     static_cast<uint32_t>(win32::kPrimaryMonitor)});
        }
        if (!result && funcName == "Direct3DCreate9") result = w32arity::hpFn(&win_Direct3DCreate9);
    }
    if (!result && dll == "opengl32.dll") {
        ensureGuestGl();
        result = guestgl::glProcAddr(funcName.c_str());
    }
    // 32 bits: WPARAM/LPARAM miden 4 bytes. Las versiones de 64 bits (uint64_t/int64_t) sacaban 2 huecos
    // de pila por cada uno (PostMessageW de Unreal 3 dejaba ESP 4 bytes desplazado: fallo /GS).
    if (!result && s_activeImage != nullptr && s_activeImage->is32Bit && dll == "user32.dll") result = msg32::resolve(funcName);    if (!result && dll == "user32.dll") {
        ensureGuestGl();
        result = guestgl::user32ProcAddr(funcName.c_str());
    }
    if (!result && (dll == "vulkan-1.dll" || dll == "vulkan")) {
        s_vulkanCalls.fetch_add(1);
        // Vulkan del teléfono, casi tal cual (VulkanPassthrough.h). Antes se devolvía el
        // vkGetInstanceProcAddr nativo y los punteros que entregaba no eran trampolines.
        static std::once_flag s_vkInit;
        std::call_once(s_vkInit, [] {
            vkpass::init([](void* fn, const char* name) { registerTrampolineAddress(fn, name); });
        });
        result = vkpass::guestProcAddr(funcName.c_str());
    }

    // 6. Local DLLs (e.g. UnityPlayer.dll -> UnityMain2). Juego .NET: también las del runtime
    //    (mono-2.0-sgen.dll junto a mono.exe, FNA y shims en fna/) y el CRT de Microsoft
    //    (api-ms-win-crt-* -> ucrtbase.dll del runtime, traducido como cualquier DLL).
    bool isGuestExport = false;
    if (!result && s_activeImage != nullptr) {
        std::string localName = guestCrtDll(dll) ? std::string("ucrtbase.dll") : dllName;
        // 32 bits: msvcrt.dll (CRT del sistema, p. ej. dbghelp.dll de DmC) -> msvcr100.dll de los runtimes
        // comunes: mismo CRT de Microsoft con las mismas funciones (antes eran stubs: malloc devolvía 0).
        if (s_activeImage->is32Bit && dll == "msvcrt.dll") localName = "MSVCR100.dll";
        struct stat st{};
        // 32 bits: MSVCR90.dll (VC++ 2008, p. ej. PhysX de DmC) sin el runtime 2008 instalado -> msvcr100.dll:
        // la API C es la misma. Las tres funciones que 2010 quitó (_encode/_decode_pointer, _encoded_null)
        // se resuelven aquí; sin esto malloc era un stub que devolvía 0.
        if (s_activeImage->is32Bit && dll == "msvcr90.dll" &&
            stat((s_activeImage->gameDir + "/" + localName).c_str(), &st) != 0 &&
            stat("/sdcard/HyperPlay/redist/x86/msvcr90.dll", &st) != 0) {
            if (funcName == "_encode_pointer" || funcName == "_decode_pointer") return w32arity::hpFn(&crt90IdentityPtr);
            if (funcName == "_encoded_null") return w32arity::hpFn(&crt90EncodedNull);
            localName = "MSVCR100.dll";
        }
        bool existsOnDisk = stat((s_activeImage->gameDir + "/" + localName).c_str(), &st) == 0;
        // Runtimes redistribuibles comunes (BionicPeLoader::loadLocalDll también los busca ahí).
        if (!existsOnDisk) {
            existsOnDisk = stat((std::string("/sdcard/HyperPlay/redist/") + (s_activeImage->is32Bit ? "x86/" : "x64/") +
                                 toLower(localName)).c_str(), &st) == 0 ||
                           stat((std::string("/sdcard/HyperPlay/redist/") + (s_activeImage->is32Bit ? "x86" : "x64") +
                                 "/netfx/Windows/System32/" + toLower(localName)).c_str(), &st) == 0;
        }
        if (!existsOnDisk && managed::active()) {
            const std::string user = managed::userOverride(toLower(localName));
            existsOnDisk = stat((managed::g_runtimeDir + "/bin/" + localName).c_str(), &st) == 0 ||
                           stat((managed::g_runtimeDir + "/fna/" + localName).c_str(), &st) == 0 ||
                           (!user.empty() && stat(user.c_str(), &st) == 0);
        }
        const bool alreadyLoaded = (s_activeImage->localDlls.find(localName) != s_activeImage->localDlls.end());
        if (existsOnDisk || alreadyLoaded) {
            void* dllBase = nullptr;
            if (alreadyLoaded) {
                dllBase = s_activeImage->localDlls[localName].base;
            } else {
                static BionicPeLoader s_localLoader;
                dllBase = s_localLoader.loadLocalDll(*s_activeImage, localName);
            }
            if (dllBase != nullptr) {
                auto it = s_activeImage->localDlls.find(localName);
                if (it != s_activeImage->localDlls.end()) {
                    // Sigue los exports reenviados (api-ms-win-crt-* -> ucrtbase.dll del juego).
                    if (void* exp = resolvePeExport(dllBase, it->second.peInfo, funcName.c_str(), 0)) {
                        result = exp;
                        isGuestExport = true;
                    }
                }
            }
        }
    }

    if (!result && (dll == "unityplayer.dll" || dll.find("unity") != std::string::npos) && funcName == "UnityMain2") {
        static auto stub_UnityMain2 = [](uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) -> uint64_t {
            LOGI("UnityMain2 stub invocado: arrancando engine nativo DirectVulkanBridge (120Hz)");
            directVulkanBridge().init();
            directVulkanBridge().presentTestFrame();
            return 0;
        };
        result = reinterpret_cast<void*>(+stub_UnityMain2);
    }

    if (result != nullptr && !isGuestExport) {
        registerTrampolineAddress(result, dllName + "!" + funcName);
    }
    return result;
}

// ---- Implementaciones de trampolines Win32 -> Android Bionic POSIX ----

static int winProtToPosix(uint32_t flProtect) {
    const uint32_t pageProt = flProtect & 0xFFu;
    switch (pageProt) {
        case 0x01: return PROT_NONE;                                 // PAGE_NOACCESS
        case 0x02: case 0x08: return PROT_READ;                      // PAGE_READONLY / WRITECOPY
        case 0x04: return PROT_READ | PROT_WRITE;                    // PAGE_READWRITE
        case 0x10: return PROT_EXEC;                                 // PAGE_EXECUTE
        case 0x20: case 0x80: return PROT_READ | PROT_EXEC;          // PAGE_EXECUTE_READ / WRITECOPY
        case 0x40: return PROT_READ | PROT_WRITE | PROT_EXEC;        // PAGE_EXECUTE_READWRITE
        default:   return PROT_READ | PROT_WRITE;
    }
}

void* NativeDllInterceptor::win_VirtualAlloc(void* lpAddress, size_t dwSize, uint32_t flAllocationType, uint32_t flProtect) {
    s_kernel32Calls.fetch_add(1);
    if (dwSize == 0) return nullptr;
    const size_t pageSize = s_pageSize();

    constexpr uint32_t kMemReset   = 0x00080000u;

    int prot = winProtToPosix(flProtect);

    // Caso 1: Dirección específica solicitada (lpAddress != nullptr)
    if (lpAddress != nullptr) {
        const uintptr_t reqAddr = reinterpret_cast<uintptr_t>(lpAddress);
        const uintptr_t pageAddr = reqAddr & ~(static_cast<uintptr_t>(pageSize) - 1);
        const size_t offset = reqAddr - pageAddr;
        const size_t allocSize = (dwSize + offset + pageSize - 1) & ~(pageSize - 1);
        if (allocSize < dwSize) return nullptr; // desbordamiento

        // Si es MEM_RESET: descartar páginas físicas manteniendo la reserva
        if ((flAllocationType & kMemReset) != 0) {
            madvise(reinterpret_cast<void*>(pageAddr), allocSize, MADV_DONTNEED);
            return lpAddress;
        }

        // Si es MEM_COMMIT: verificar si cae en una reserva previa o mapa existente
        bool isReserved = false;
        {
            std::lock_guard<std::mutex> lock(s_allocMutex);
            for (const auto& [base, resSize] : s_allocations) {
                if (pageAddr >= base && (pageAddr + allocSize) <= (base + resSize)) {
                    isReserved = true;
                    break;
                }
            }
        }

        // Solo se cambia la protección de memoria que el propio huésped reservó. Antes un
        // MEM_COMMIT a una dirección arbitraria hacía mprotect sobre memoria del HOST (p. ej. el
        // texto de liblog) y el juego la sobrescribía -> SIGILL en hilos de Android.
        constexpr uint32_t kMemReserve = 0x00002000u;
        if (!isReserved && (flAllocationType & kMemReserve) == 0) {
            LOGW("win_VirtualAlloc: MEM_COMMIT en %p sin reserva previa -> ERROR_INVALID_ADDRESS", lpAddress);
            win_SetLastError(487);  // ERROR_INVALID_ADDRESS
            return nullptr;
        }
        // MEM_RESERVE sobre un rango que YA pertenece a otra reserva: Windows falla. Antes se
        // "confirmaba" dentro de la reserva ajena (p. ej. el heap de Boehm) y el gestor de código
        // de Mono, que pide direcciones cercanas a su código, escribía JIT dentro del heap del GC.
        if ((flAllocationType & kMemReserve) != 0) {
            bool overlaps = false;
            {
                std::lock_guard<std::mutex> lock(s_allocMutex);
                for (const auto& [base, resSize] : s_allocations) {
                    if (pageAddr < base + resSize && pageAddr + allocSize > base) {
                        overlaps = true;
                        break;
                    }
                }
            }
            if (overlaps) {
                win_SetLastError(487);
                return nullptr;
            }
            isReserved = false;  // reserva nueva en dirección fija (MAP_FIXED_NOREPLACE abajo)
        }
        if (isReserved) {
            if (mprotect(reinterpret_cast<void*>(pageAddr), allocSize, prot) == 0) {
                recordPageProt(pageAddr, allocSize, flProtect == 0 ? 0x04u : (flProtect & 0xFFu));
                if ((flProtect & 0x70) != 0 || (flProtect & 0x10) != 0) {
                    if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
                        OryonEmitterBackend::addExecutableRangeAll(pageAddr, allocSize);
                    }
                }
                LOGI("win_VirtualAlloc: commit en rango existente lpAddress=%p (base=%p, size=0x%zx, prot=0x%x)",
                     lpAddress, reinterpret_cast<void*>(pageAddr), allocSize, flProtect);
                return lpAddress;
            }
            if ((prot & PROT_EXEC) != 0 && (prot & PROT_WRITE) != 0) {
                if (mprotect(reinterpret_cast<void*>(pageAddr), allocSize, PROT_READ | PROT_WRITE) == 0) {
                    recordPageProt(pageAddr, allocSize, 0x04u);
                    if ((flProtect & 0x70) != 0 || (flProtect & 0x10) != 0) {
                        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
                            OryonEmitterBackend::addExecutableRangeAll(pageAddr, allocSize);
                        }
                    }
                    return lpAddress;
                }
            }
        }

        // MEM_RESERVE a dirección fija: nunca se pisa un mapeo existente (MAP_FIXED_NOREPLACE).
        // Si el rango está ocupado, Windows falla con ERROR_INVALID_ADDRESS y el runtime
        // (Mono/Unity) reintenta en otra dirección o con NULL.
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
        void* ptr = mmap(reinterpret_cast<void*>(pageAddr), allocSize, prot,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (ptr == MAP_FAILED && errno != EEXIST && (prot & PROT_EXEC) != 0) {
            ptr = mmap(reinterpret_cast<void*>(pageAddr), allocSize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        }
        if (ptr != MAP_FAILED && ptr != reinterpret_cast<void*>(pageAddr)) {
            // Kernel sin NOREPLACE (lo trata como pista): no es la dirección pedida.
            munmap(ptr, allocSize);
            ptr = MAP_FAILED;
        }
        if (ptr == MAP_FAILED) {
            LOGW("win_VirtualAlloc: rango %p+0x%zx ocupado -> ERROR_INVALID_ADDRESS", lpAddress, allocSize);
            win_SetLastError(487);
            return nullptr;
        }
        if (ptr != MAP_FAILED) {
            {
                std::lock_guard<std::mutex> lock(s_allocMutex);
                s_allocations[reinterpret_cast<uintptr_t>(ptr)] = allocSize;
            }
            recordPageProt(reinterpret_cast<uintptr_t>(ptr), allocSize,
                           flProtect == 0 ? 0x04u : (flProtect & 0xFFu));
            if ((flProtect & 0x70) != 0 || (flProtect & 0x10) != 0) {
                if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
                    OryonEmitterBackend::addExecutableRangeAll(reinterpret_cast<uintptr_t>(ptr), allocSize);
                }
            }
            return lpAddress;
        }
        return nullptr;
    }

    // Caso 2: Reserva o asignación nueva (lpAddress == nullptr)
    const size_t allocSize = (dwSize + pageSize - 1) & ~(pageSize - 1);
    if (allocSize < dwSize) return nullptr;

    // Windows devuelve siempre direcciones alineadas a la granularidad de asignación (64 KB) y
    // hay runtimes que lo dan por hecho (CoreCLR calcula `ptr & ~0xFFFF` para su tabla de bloques
    // y escribía antes del bloque). Se reserva de más y se recorta hasta la frontera de 64 KB.
    constexpr size_t kGranularity = 0x10000;
    void* ptr = MAP_FAILED;
    if (lowmem::enabled()) {
        // Invitado de 32 bits: la dirección tiene que caber en 32 bits (antes se truncaba). Los
        // huecos de mmapBelow4G ya están alineados a 64 KB.
        ptr = lowmem::mmapBelow4G(allocSize, prot);
        if (ptr == MAP_FAILED && (prot & PROT_EXEC) != 0) ptr = lowmem::mmapBelow4G(allocSize, PROT_READ | PROT_WRITE);
        if (ptr == MAP_FAILED) {
            win_SetLastError(8);  // ERROR_NOT_ENOUGH_MEMORY
            return nullptr;
        }
    } else {
        ptr = mmap(nullptr, allocSize + kGranularity, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED && (prot & PROT_EXEC) != 0) {
            ptr = mmap(nullptr, allocSize + kGranularity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
    }
    if (ptr == MAP_FAILED) return nullptr;
    if (!lowmem::enabled()) {
        const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t aligned = (raw + kGranularity - 1) & ~(kGranularity - 1);
        if (aligned > raw) munmap(ptr, aligned - raw);
        const uintptr_t tail = raw + allocSize + kGranularity - (aligned + allocSize);
        if (tail > 0) munmap(reinterpret_cast<void*>(aligned + allocSize), tail);
        ptr = reinterpret_cast<void*>(aligned);
    }

    {
        std::lock_guard<std::mutex> lock(s_allocMutex);
        s_allocations[reinterpret_cast<uintptr_t>(ptr)] = allocSize;
    }
    recordPageProt(reinterpret_cast<uintptr_t>(ptr), allocSize,
                   flProtect == 0 ? 0x04u : (flProtect & 0xFFu));
    if ((flProtect & 0x70) != 0 || (flProtect & 0x10) != 0) {
        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
            OryonEmitterBackend::addExecutableRangeAll(reinterpret_cast<uintptr_t>(ptr), allocSize);
        }
    }
    LOGI("win_VirtualAlloc: nueva reserva ptr=%p, size=0x%zx, type=0x%x, prot=0x%x",
         ptr, allocSize, flAllocationType, flProtect);
    return ptr;
}

int32_t NativeDllInterceptor::win_VirtualFree(void* lpAddress, size_t dwSize, uint32_t dwFreeType) {
    s_kernel32Calls.fetch_add(1);
    if (!lpAddress) return 0;

    constexpr uint32_t kMemDecommit = 0x00004000u;
    constexpr uint32_t kMemRelease  = 0x00008000u;

    const size_t pageSize = s_pageSize();

    // MEM_DECOMMIT: descarta páginas físicas y pone PROT_NONE, manteniendo la reserva virtual
    if ((dwFreeType & kMemDecommit) != 0) {
        if (dwSize == 0) return 0;
        const uintptr_t reqAddr = reinterpret_cast<uintptr_t>(lpAddress);
        const uintptr_t pageAddr = reqAddr & ~(static_cast<uintptr_t>(pageSize) - 1);
        const size_t offset = reqAddr - pageAddr;
        const size_t decommitSize = (dwSize + offset + pageSize - 1) & ~(pageSize - 1);
        mprotect(reinterpret_cast<void*>(pageAddr), decommitSize, PROT_NONE);
        madvise(reinterpret_cast<void*>(pageAddr), decommitSize, MADV_DONTNEED);
        recordPageProt(pageAddr, decommitSize, 0x01u /* PAGE_NOACCESS */);
        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
            OryonEmitterBackend::smcForgetRange(pageAddr, decommitSize); OryonEmitterBackend::removeExecutableRangeAll(pageAddr, decommitSize);
        }
        return 1;
    }

    // MEM_RELEASE: libera la región de memoria virtual completa
    if ((dwFreeType & kMemRelease) != 0) {
        size_t size = 0;
        void* unmapBase = lpAddress;
        {
            std::lock_guard<std::mutex> lock(s_allocMutex);
            auto it = s_allocations.find(reinterpret_cast<uintptr_t>(lpAddress));
            if (it != s_allocations.end()) {
                size = it->second;
                s_allocations.erase(it);
            } else {
                uintptr_t addr = reinterpret_cast<uintptr_t>(lpAddress);
                for (auto iter = s_allocations.begin(); iter != s_allocations.end(); ++iter) {
                    if (addr >= iter->first && addr < iter->first + iter->second) {
                        unmapBase = reinterpret_cast<void*>(iter->first);
                        size = iter->second;
                        s_allocations.erase(iter);
                        break;
                    }
                }
            }
        }
        if (size == 0) size = dwSize;
        if (size == 0) {
            // Región que no registró VirtualAlloc (p. ej. ya liberada en parte o de otra API):
            // Windows liberaría la reserva entera; aquí no se conoce su tamaño. Mono aborta con
            // cualquier fallo de VirtualFree, así que se responde éxito (se pierde la región).
            static std::atomic<int> s_unknownFrees{0};
            if (s_unknownFrees.fetch_add(1) < 20) LOGW("VirtualFree(MEM_RELEASE) de %p no registrado: se ignora", lpAddress);
            return 1;
        }
        OryonEmitterBackend::noteUnmap(reinterpret_cast<uintptr_t>(unmapBase), size, "VirtualFree(MEM_RELEASE)", t_thunkRet);
        if (munmap(unmapBase, size) != 0) return 0;
        erasePageProt(reinterpret_cast<uintptr_t>(unmapBase), size);
        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
            OryonEmitterBackend::smcForgetRange(reinterpret_cast<uintptr_t>(unmapBase), size); OryonEmitterBackend::removeExecutableRangeAll(reinterpret_cast<uintptr_t>(unmapBase), size);
        }
        return 1;
    }

    return 0;
}

int32_t NativeDllInterceptor::win_VirtualProtect(void* lpAddress, size_t dwSize, uint32_t flNewProtect,
                                                 uint32_t* lpflOldProtect) {
    s_kernel32Calls.fetch_add(1);
    if (lpAddress == nullptr || dwSize == 0) return 0;
    const uint32_t oldProt = queryPageProt(reinterpret_cast<uintptr_t>(lpAddress));
    if (lpflOldProtect != nullptr) *lpflOldProtect = oldProt;
    {
        // Windows cambia las páginas completas que tocan [lpAddress, lpAddress+dwSize); mprotect
        // exige dirección alineada (CoreCLR protege 0xb0 bytes en mitad de una página).
        const uintptr_t page = static_cast<uintptr_t>(getpagesize());
        const uintptr_t a = reinterpret_cast<uintptr_t>(lpAddress);
        const uintptr_t start = a & ~(page - 1);
        const uintptr_t end = (a + dwSize + page - 1) & ~(page - 1);
        lpAddress = reinterpret_cast<void*>(start);
        dwSize = end - start;
    }

    const uint32_t pageProt = flNewProtect & 0xFFu;
    const bool executable = (flNewProtect & 0x70u) != 0 || (flNewProtect & 0x10u) != 0;
    int prot = winProtToPosix(flNewProtect);
    if (mprotect(lpAddress, dwSize, prot) == 0) {
        recordPageProt(reinterpret_cast<uintptr_t>(lpAddress), dwSize, pageProt);
        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
            // El invitado puede haber escrito bytes x86 nuevos en esta página: las traducciones
            // anteriores ya no representan el código actual. Se descartan antes de re-traducir.
            OryonEmitterBackend::smcForgetRange(reinterpret_cast<uintptr_t>(lpAddress), dwSize); OryonEmitterBackend::invalidateCodeRangeAll(reinterpret_cast<uintptr_t>(lpAddress), dwSize);
            if (executable) {
                OryonEmitterBackend::addExecutableRangeAll(reinterpret_cast<uintptr_t>(lpAddress), dwSize);
            }
        }
        LOGI("win_VirtualProtect: addr=%p size=0x%zx prot=0x%x (old=0x%x)", lpAddress, dwSize,
             pageProt, oldProt);
        return 1;
    }
    if ((prot & PROT_EXEC) != 0 && (prot & PROT_WRITE) != 0) {
        // Android puede denegar RWX; se deja en RX para no romper la ejecución (los datos deben
        // haberse escrito antes de pedir PAGE_EXECUTE_READWRITE).
        if (mprotect(lpAddress, dwSize, PROT_READ | PROT_EXEC) == 0) {
            recordPageProt(reinterpret_cast<uintptr_t>(lpAddress), dwSize, 0x20u);
            if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
                OryonEmitterBackend::smcForgetRange(reinterpret_cast<uintptr_t>(lpAddress), dwSize); OryonEmitterBackend::invalidateCodeRangeAll(reinterpret_cast<uintptr_t>(lpAddress), dwSize);
                OryonEmitterBackend::addExecutableRangeAll(reinterpret_cast<uintptr_t>(lpAddress), dwSize);
            }
            LOGI("win_VirtualProtect: RWX degradado a RX en addr=%p size=0x%zx", lpAddress, dwSize);
            return 1;
        }
    }
    win_SetLastError(87); // ERROR_INVALID_PARAMETER
    return 0;
}

// Unity: <Juego>_Data/boot.config con el GC incremental activado (`gc-max-time-slice=3`) si el
// juego no lo trae. El GC de Mono (Boehm) para el mundo para marcar; traducido tarda 8-10 veces
// más que en un PC (pausas de 350-450 ms en Hollow Knight). El modo incremental, soportado por
// Unity 2019.1+ y pensado para activarse por ajuste, lo reparte en tramos de 3 ms por frame.
// Copia en _HyperPlaySaves (los archivos del juego no se tocan).
// DESACTIVADO por defecto (`setprop debug.hyperplay.incgc 1` para probar): con Hollow Knight
// (Unity 6000.0.61f1) Mono se cae al iniciar al analizar las opciones del GC (strchr de '=' que
// devuelve NULL en mono-2.0-bdwgc+0x26341d). Pendiente de investigar el formato que espera.
static std::string unityBootConfigOverlay(const std::string& realPath) {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.incgc", v) > 0 && v[0] == '1';
    }();
    if (!s_on || s_activeImage == nullptr || s_activeImage->savesDir.empty()) return realPath;
    const std::string suffix = "_Data/boot.config";
    if (realPath.size() < suffix.size() ||
        strcasecmp(realPath.c_str() + realPath.size() - suffix.size(), suffix.c_str()) != 0) {
        return realPath;
    }
    std::string content;
    {
        const int in = open(realPath.c_str(), O_RDONLY | O_CLOEXEC);
        if (in < 0) return realPath;
        char buf[4096];
        ssize_t n;
        while ((n = read(in, buf, sizeof(buf))) > 0) content.append(buf, static_cast<size_t>(n));
        close(in);
    }
    if (content.find("gc-max-time-slice") != std::string::npos) return realPath;  // ya lo decide el juego
    if (!content.empty() && content.back() != '\n') content.push_back('\n');
    content += "gc-max-time-slice=3\n";
    const std::string out = s_activeImage->savesDir + "/boot.config";
    const int fd = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0660);
    if (fd < 0) return realPath;
    const bool ok = write(fd, content.data(), content.size()) == static_cast<ssize_t>(content.size());
    close(fd);
    if (!ok) return realPath;
    LOGI("Unity boot.config: GC incremental activado (gc-max-time-slice=3) -> %s", out.c_str());
    return out;
}

// Panorámica (Pantalla = Panorámica, ver WidescreenPatch.h): al abrir para leer el ensamblado de
// Cuphead se entrega una copia parcheada en memoria (memfd), con la proporción real de la
// pantalla en lugar de 16:9. Los archivos del juego no se tocan. @return fd o -1 (sin parche).
static int widescreenOverlayFd(const std::string& realPath) {
    if (gpu::presentopt::g_fit.load() != gpu::presentopt::kFitWide) return -1;
    if (!widescreen::isCupheadAssembly(realPath.c_str())) return -1;
    const int in = open(realPath.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) return -1;
    // Lectura directa al vector (nada grande en la pila: esta función se integra en CreateFileW,
    // que se llama con pilas host muy profundas; un búfer de 64 KB aquí la desbordaba).
    std::vector<uint8_t> bytes;
    {
        struct stat st{};
        if (fstat(in, &st) != 0 || st.st_size <= 0) {
            close(in);
            return -1;
        }
        bytes.resize(static_cast<size_t>(st.st_size));
        size_t got = 0;
        while (got < bytes.size()) {
            const ssize_t n = read(in, bytes.data() + got, bytes.size() - got);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            got += static_cast<size_t>(n);
        }
        close(in);
        if (got != bytes.size()) return -1;
    }
    const float aspect = static_cast<float>(win32::screenWidth()) / static_cast<float>(win32::screenHeight());
    if (widescreen::patchCupheadUpdateRect(bytes.data(), bytes.size(), aspect) == 0) {
        LOGW("Panorámica: firma de UpdateRect no encontrada en %s (sin parche)", realPath.c_str());
        return -1;
    }
    const int fd = static_cast<int>(syscall(__NR_memfd_create, "hp_widescreen", 1u /*MFD_CLOEXEC*/));
    if (fd < 0) return -1;
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t w = write(fd, bytes.data() + done, bytes.size() - done);
        if (w <= 0) { close(fd); return -1; }
        done += static_cast<size_t>(w);
    }
    lseek(fd, 0, SEEK_SET);
    gpu::presentopt::g_widePatched.store(true);
    LOGI("Panorámica: Cuphead UpdateRect 16:9 -> %.4f (%dx%d), %zu bytes en memfd",
         aspect, win32::screenWidth(), win32::screenHeight(), bytes.size());
    return fd;
}

// Diagnóstico (`setprop debug.hyperplay.codedump <texto>`): la primera vez que la consola del juego
// imprime una línea con <texto> (p. ej. el "ENTER:c Clase:Metodo" del trazado de Mono), recorre la
// pila del hilo invitado y vuelca el código x86 alrededor de las direcciones de retorno que caen en
// memoria ejecutable anónima (código JIT de Mono) a <saves>/codedump_<n>.bin, con la dirección base
// en el log. Se desensambla en el PC con llvm-objdump.
static void dumpGuestCallerCodeIfRequested(std::string_view text) {
    static std::atomic<bool> s_done{false};
    static const std::string s_needle = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.codedump", v) > 0 ? std::string(v) : std::string();
    }();
    if (s_needle.empty() || s_done.load() || text.find(s_needle) == std::string_view::npos) return;
    OryonEmitterBackend* backend = t_dispatchBackend;
    if (backend == nullptr || s_activeImage == nullptr || s_done.exchange(true)) return;
    struct Range { uintptr_t lo, hi; };
    std::vector<Range> exec;
    if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), maps)) {
            unsigned long long lo = 0, hi = 0;
            char perms[8] = {};
            if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) != 3) continue;
            if (perms[0] != 'r' || perms[1] != 'w') continue;  // el código JIT de Mono es RWX anónimo
            if (std::strchr(line, '/') != nullptr) continue;
            exec.push_back({static_cast<uintptr_t>(lo), static_cast<uintptr_t>(hi)});
        }
        std::fclose(maps);
    }
    auto isProt = [](uintptr_t a) {
        const uint32_t prot = queryPageProt(a) & 0xFFu;
        return prot == 0x10 || prot == 0x20 || prot == 0x40;  // PAGE_EXECUTE / _READ / _READWRITE
    };
    const auto* sp = reinterpret_cast<const uint64_t*>(backend->context().gpr[4]);
    int dumped = 0;
    for (int i = 0; i < 8192 && dumped < 6; ++i) {
        const uintptr_t v = static_cast<uintptr_t>(sp[i]);
        if (v < 0x10000 || (v >> 48) != 0 || !isProt(v) || !isProt(v - 0x800) || !isProt(v + 0xFFF)) continue;
        bool seen = false;
        for (int k = 0; k < i; ++k) seen |= static_cast<uintptr_t>(sp[k]) == v;
        if (seen) continue;
        const uintptr_t base = v - 0x800;
        std::vector<uint8_t> buf(0x1800);
        iovec local{buf.data(), buf.size()};
        iovec remote{reinterpret_cast<void*>(base), buf.size()};
        if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != static_cast<ssize_t>(buf.size())) continue;
        const std::string out = s_activeImage->savesDir + "/codedump_" + std::to_string(dumped) + ".bin";
        const int fd = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0660);
        if (fd < 0) continue;
        const bool ok = write(fd, buf.data(), buf.size()) == static_cast<ssize_t>(buf.size());
        close(fd);
        LOGI("[CodeDump] pila[%d]=0x%llx -> %s (base 0x%llx, %s)", i, static_cast<unsigned long long>(v),
             out.c_str(), static_cast<unsigned long long>(base), ok ? "ok" : "fallo");
        ++dumped;
    }
    if (dumped == 0) {
        LOGW("[CodeDump] sin direcciones de retorno en código JIT (RSP=%p, %zu regiones rw anónimas)",
             static_cast<const void*>(sp), exec.size());
        int shown = 0;
        for (int i = 0; i < 8192 && shown < 40; ++i) {
            const uintptr_t v = static_cast<uintptr_t>(sp[i]);
            if (v < 0x10000 || (v >> 48) != 0) continue;
            const uint32_t prot = queryPageProt(v) & 0xFFu;
            if (prot == 0x04) continue;
            LOGW("[CodeDump]   pila[%d]=0x%llx prot=0x%x", i, static_cast<unsigned long long>(v), prot);
            ++shown;
        }
    }
}

void* NativeDllInterceptor::win_CreateFileW(const wchar_t* lpFileName, uint32_t dwDesiredAccess, uint32_t,
                                            void*, uint32_t dwCreationDisposition, uint32_t, void*) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName || !s_activeImage) return handleFromFd(-1);

    const std::string rawPath = wcharToUtf8(lpFileName);
    if (rawPath.empty()) {
        win_SetLastError(3);  // ERROR_PATH_NOT_FOUND
        return handleFromFd(-1);
    }
    std::string realPath = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    if ((dwDesiredAccess & 0x40000000u) == 0) realPath = unityBootConfigOverlay(realPath);

    int flags = 0;
    if ((dwDesiredAccess & 0xC0000000) == 0xC0000000) flags = O_RDWR;
    else if (dwDesiredAccess & 0x40000000) flags = O_WRONLY;
    else flags = O_RDONLY;

    switch (dwCreationDisposition) {
        case 1: flags |= O_CREAT | O_EXCL; break;  // CREATE_NEW
        case 2: flags |= O_CREAT | O_TRUNC; break; // CREATE_ALWAYS
        case 4: flags |= O_CREAT; break;           // OPEN_ALWAYS
        case 5: flags |= O_TRUNC; break;           // TRUNCATE_EXISTING
        default: break;                            // OPEN_EXISTING
    }

    int fd = (flags & O_ACCMODE) == O_RDONLY && (flags & (O_CREAT | O_TRUNC)) == 0
                 ? widescreenOverlayFd(realPath) : -1;
    if (fd < 0) fd = open(realPath.c_str(), flags, 0644);
    win_SetLastError((fd >= 0) ? 0 : 2); // ERROR_FILE_NOT_FOUND (2)
    LOGI("win_CreateFileW: raw='%s' real='%s' access=0x%x disp=%u -> fd=%d (err=%d)",
         rawPath.c_str(), realPath.c_str(), dwDesiredAccess, dwCreationDisposition, fd, (fd >= 0) ? 0 : 2);
    return handleFromFd(fd);
}

void* NativeDllInterceptor::win_CreateFileA(const char* lpFileName, uint32_t dwDesiredAccess, uint32_t dwShareMode,
                                            void* lpSecurityAttributes, uint32_t dwCreationDisposition,
                                            uint32_t dwFlagsAndAttributes, void* hTemplateFile) {
    if (!lpFileName) return handleFromFd(-1);
    // La variante W lee WCHAR de 16 bits: el búfer intermedio debe ser char16_t, no wchar_t.
    std::vector<char16_t> wbuf(guestWStringLength(lpFileName) + 1);
    writeGuestWString(reinterpret_cast<wchar_t*>(wbuf.data()), wbuf.size(), lpFileName);
    return win_CreateFileW(reinterpret_cast<const wchar_t*>(wbuf.data()), dwDesiredAccess, dwShareMode, lpSecurityAttributes,
                           dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile);
}

int32_t NativeDllInterceptor::win_ReadFile(void* hFile, void* lpBuffer, uint32_t nNumberOfBytesToRead,
                                           uint32_t* lpNumberOfBytesRead, void* lpOverlapped) {
    s_kernel32Calls.fetch_add(1);
    const int fd = fdFromHandle(hFile);
    if (fd < 0 || !lpBuffer) return 0;
    // 32 bits: con OVERLAPPED se lee en su desplazamiento (Offset/OffsetHigh en +8/+12), como Windows en
    // un handle síncrono; antes se leía de la posición actual (Unreal 3: "contains unrecognizable data").
    // Se completa al momento: Internal=0, InternalHigh=bytes y se señala hEvent (+16).
    if (lpOverlapped != nullptr && s_activeImage != nullptr && s_activeImage->is32Bit) {
        auto* ov = static_cast<uint8_t*>(lpOverlapped);
        uint32_t lo = 0, hi = 0, ev = 0;
        std::memcpy(&lo, ov + 8, 4);
        std::memcpy(&hi, ov + 12, 4);
        std::memcpy(&ev, ov + 16, 4);
        const off_t pos = static_cast<off_t>((static_cast<uint64_t>(hi) << 32) | lo);
        const ssize_t n = pread(fd, lpBuffer, nNumberOfBytesToRead, pos);
        if (n < 0) return 0;
        if (lseek(fd, pos + n, SEEK_SET) < 0) { /* el puntero de archivo es orientativo */ }
        const uint32_t zero = 0, got = static_cast<uint32_t>(n);
        std::memcpy(ov, &zero, 4);
        std::memcpy(ov + 4, &got, 4);
        if (lpNumberOfBytesRead) *lpNumberOfBytesRead = got;
        if ((ev & ~1u) != 0) win_SetEvent(reinterpret_cast<void*>(static_cast<uintptr_t>(ev & ~1u)));  // bit 0: sin puerto
        s_readFileBytes.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
        static std::atomic<int> s_logged{0};
        if (s_logged.fetch_add(1) < 3) LOGI("ReadFile[32] OVERLAPPED: %u bytes en 0x%llx -> %zd", nNumberOfBytesToRead,
                                            static_cast<unsigned long long>(pos), n);
        return 1;
    }
    {
        // Diagnóstico (setprop debug.hyperplay.tracetrad <tamaño hex>): una lectura de ese
        // tamaño activa el registro de las siguientes 1500 traducciones del JIT.
        static const uint32_t s_traceSize = [] {
            char v[PROP_VALUE_MAX] = {};
            return __system_property_get("debug.hyperplay.tracetrad", v) > 0
                       ? static_cast<uint32_t>(std::strtoul(v, nullptr, 16))
                       : 0u;
        }();
        static std::atomic<bool> s_traceArmed{false};
        if (s_traceSize != 0 && nNumberOfBytesToRead == s_traceSize) {
            static std::atomic<int> s_stackDumps{0};
            if (t_dispatchBackend != nullptr && s_stackDumps.fetch_add(1) < 4) {
                t_dispatchBackend->logGuestStackChain("[ReadFile]");
            }
            if (!s_traceArmed.exchange(true)) {
                LOGE("[TraceTrad] activado por ReadFile de 0x%x bytes", nNumberOfBytesToRead);
                OryonEmitterBackend::traceTranslations.store(1500);
            }
        }
    }

    // Diagnóstico (setprop debug.hyperplay.readlog N): las N primeras lecturas con fd, posición y bytes.
    static const int s_readLog = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.readlog", v) > 0 ? std::atoi(v) : 0;
    }();
    static std::atomic<int> s_readLogged{0};
    const off_t posBefore = s_readLog > 0 ? lseek(fd, 0, SEEK_CUR) : 0;
    const ssize_t n = read(fd, lpBuffer, nNumberOfBytesToRead);
    if (s_readLog > 0 && s_readLogged.fetch_add(1) < s_readLog) {
        uint32_t head = 0;
        if (n >= 4) std::memcpy(&head, lpBuffer, 4);
        LOGI("[ReadLog] fd=%d pos=0x%llx pedido=%u leído=%zd buf=%p primeros=%08x ov=%p", fd,
             static_cast<unsigned long long>(posBefore), nNumberOfBytesToRead, n, lpBuffer, head, lpOverlapped);
    }
    if (n < 0) return 0;
    s_readFileBytes.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
    if (lpNumberOfBytesRead) *lpNumberOfBytesRead = static_cast<uint32_t>(n);
    return 1;
}

int32_t NativeDllInterceptor::win_WriteFile(void* hFile, const void* lpBuffer, uint32_t nNumberOfBytesToWrite,
                                            uint32_t* lpNumberOfBytesWritten, void*) {
    s_kernel32Calls.fetch_add(1);
    const int fd = fdFromHandle(hFile);
    if (fd < 0 || !lpBuffer) return 0;

    if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
        LOGI("[Guest Console] %.*s", static_cast<int>(nNumberOfBytesToWrite), static_cast<const char*>(lpBuffer));
        dumpGuestCallerCodeIfRequested(std::string_view(static_cast<const char*>(lpBuffer), nNumberOfBytesToWrite));
        // Diagnóstico (setprop debug.hyperplay.memscan <texto>): cuando la consola imprime una
        // línea con "right version", busca <texto> en la memoria rw anónima y vuelca el contexto.
        static std::atomic<int> s_memscanDone{0};
        char scanNeedle[PROP_VALUE_MAX] = {};
        if (s_memscanDone.load() < 2 && __system_property_get("debug.hyperplay.memscan", scanNeedle) > 1 &&
            std::string_view(static_cast<const char*>(lpBuffer), nNumberOfBytesToWrite).find("right version") !=
                std::string_view::npos) {
            s_memscanDone.fetch_add(1);
            size_t needleLen = std::strlen(scanNeedle);
            if (std::strncmp(scanNeedle, "hex:", 4) == 0) {  // patrón binario: "hex:0102ab..."
                size_t out = 0;
                for (size_t k = 4; k + 1 < needleLen; k += 2) {
                    char pair[3] = {scanNeedle[k], scanNeedle[k + 1], 0};
                    scanNeedle[out++] = static_cast<char>(std::strtoul(pair, nullptr, 16));
                }
                needleLen = out;
            }
            if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
                char line[512];
                int hits = 0;
                while (hits < 24 && std::fgets(line, sizeof(line), maps)) {
                    unsigned long long lo = 0, hi = 0;
                    char perms[8] = {};
                    if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) != 3) continue;
                    if (perms[0] != 'r' || perms[1] != 'w' || hi - lo > (512ull << 20)) continue;
                    if (std::strchr(line, '/') != nullptr && std::strstr(line, "PC games") == nullptr) continue;
                    const auto* p = reinterpret_cast<const uint8_t*>(lo);
                    const size_t n = static_cast<size_t>(hi - lo);
                    for (size_t i = 0; i + needleLen + 48 <= n && hits < 24; ++i) {
                        if (p[i] != static_cast<uint8_t>(scanNeedle[0]) || std::memcmp(p + i, scanNeedle, needleLen) != 0)
                            continue;
                        if (p + i == reinterpret_cast<const uint8_t*>(scanNeedle)) continue;
                        const size_t from = i >= 24 ? i - 24 : 0;
                        char hex[3 * 80 + 1] = {};
                        for (size_t k = 0; k < 72 && from + k < n; ++k)
                            std::snprintf(hex + 3 * k, 4, "%02x ", p[from + k]);
                        LOGI("[MemScan] %p (-%zu): %s", static_cast<const void*>(p + from), i - from, hex);
                        ++hits;
                        i += needleLen;
                    }
                }
                std::fclose(maps);
            }
        }
    }

    if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
        // Consola del juego (Mono, el CRT, Celeste "Console.WriteLine") -> logcat, por líneas.
        static std::mutex s_outMutex;
        static std::string s_line[3];
        std::lock_guard<std::mutex> lock(s_outMutex);
        std::string& line = s_line[fd];
        const auto* p = static_cast<const char*>(lpBuffer);
        for (uint32_t i = 0; i < nNumberOfBytesToWrite; ++i) {
            if (p[i] == '\n' || line.size() > 900) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                __android_log_print(ANDROID_LOG_INFO, "HyperPlay.GameOut", "%s %s", fd == STDOUT_FILENO ? "[out]" : "[err]",
                                    line.c_str());
                line.clear();
                if (p[i] == '\n') continue;
            }
            if (p[i] != '\0' && p[i] != '\r') line.push_back(p[i]);  // sin NUL (UTF-16) ni '\r'
        }
        if (lpNumberOfBytesWritten) *lpNumberOfBytesWritten = nNumberOfBytesToWrite;
        return 1;
    }
    const ssize_t n = write(fd, lpBuffer, nNumberOfBytesToWrite);
    if (n < 0) return 0;
    if (lpNumberOfBytesWritten) *lpNumberOfBytesWritten = static_cast<uint32_t>(n);
    return 1;
}

int32_t NativeDllInterceptor::win_CloseHandle(void* hObject) {
    s_kernel32Calls.fetch_add(1);
    if (hObject == nullptr || hObject == reinterpret_cast<void*>(~static_cast<uintptr_t>(0)) ||
        hObject == reinterpret_cast<void*>(~static_cast<uintptr_t>(1))) {
        return 1;
    }
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hObject);
    if ((handle & ~static_cast<uintptr_t>(0xFFFFFFu)) == 0x7E000000u) return 1;  // handle de hilo etiquetado
    {
        std::lock_guard<std::mutex> lk(s_eventMutex);
        if (s_events.erase(handle) > 0) return 1;
    }
    {
        std::lock_guard<std::mutex> lk(s_semMutex);
        if (s_semaphores.erase(handle) > 0) return 1;
    }
    {
        std::lock_guard<std::mutex> lk(s_threadMutex);
        if (s_threads.erase(handle) > 0) return 1;
    }
    {
        std::lock_guard<std::mutex> lk(s_mappingMutex);
        auto it = s_fileMappings.find(handle);
        if (it != s_fileMappings.end()) {
            if (it->second->fd >= 0) close(it->second->fd);
            if (!it->second->name.empty()) s_namedMappings.erase(it->second->name);
            s_fileMappings.erase(it);
            return 1;
        }
    }
    const int fd = fdFromHandle(hObject);
    if (fd >= 0) {
        forgetDirListing(fd);
        return (close(fd) == 0) ? 1 : 0;
    }
    return 1;
}

uint32_t NativeDllInterceptor::win_GetFileSize(void* hFile, uint32_t* lpFileSizeHigh) {
    s_kernel32Calls.fetch_add(1);
    const int fd = fdFromHandle(hFile);
    if (fd < 0) return 0xFFFFFFFF;

    struct stat st;
    if (fstat(fd, &st) != 0) return 0xFFFFFFFF;
    if (lpFileSizeHigh) *lpFileSizeHigh = static_cast<uint32_t>(st.st_size >> 32);
    return static_cast<uint32_t>(st.st_size & 0xFFFFFFFF);
}

int32_t NativeDllInterceptor::win_GetFileSizeEx(void* hFile, int64_t* lpFileSize) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileSize) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }
    const int fd = fdFromHandle(hFile);
    if (fd < 0) {
        win_SetLastError(6); // ERROR_INVALID_HANDLE
        return 0;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        win_SetLastError(6); // ERROR_INVALID_HANDLE
        return 0;
    }
    *lpFileSize = static_cast<int64_t>(st.st_size);
    LOGI("win_GetFileSizeEx: fd=%d -> size=%lld (OK)", fd, static_cast<long long>(st.st_size));
    win_SetLastError(0);
    return 1;
}

uint32_t NativeDllInterceptor::win_SetFilePointer(void* hFile, int32_t lDistanceToMove,
                                                  int32_t* lpDistanceToMoveHigh, uint32_t dwMoveMethod) {
    s_kernel32Calls.fetch_add(1);
    const int fd = fdFromHandle(hFile);
    if (fd < 0) return 0xFFFFFFFF;

    int64_t offset = lDistanceToMove;
    if (lpDistanceToMoveHigh) {
        offset |= (static_cast<int64_t>(*lpDistanceToMoveHigh) << 32);
    }

    int whence = SEEK_SET;
    if (dwMoveMethod == 1) whence = SEEK_CUR;
    else if (dwMoveMethod == 2) whence = SEEK_END;

    const off_t res = lseek(fd, static_cast<off_t>(offset), whence);
    if (res == static_cast<off_t>(-1)) return 0xFFFFFFFF;

    if (lpDistanceToMoveHigh) *lpDistanceToMoveHigh = static_cast<int32_t>(res >> 32);
    return static_cast<uint32_t>(res & 0xFFFFFFFF);
}

void* NativeDllInterceptor::win_CreateFileMappingW(void* hFile, void*, uint32_t flProtect,
                                                   uint32_t dwMaxSizeHigh, uint32_t dwMaxSizeLow, const wchar_t* lpName) {
    s_kernel32Calls.fetch_add(1);
    std::string name;
    if (lpName) name = wcharToUtf8(lpName);

    uint64_t maxSize = (static_cast<uint64_t>(dwMaxSizeHigh) << 32) | dwMaxSizeLow;
    int dupFd = -1;
    const uintptr_t hVal = reinterpret_cast<uintptr_t>(hFile);
    const bool isAnon = (hFile == nullptr || hVal == ~static_cast<uintptr_t>(0) || hVal == 0xFFFFFFFFu);

    if (!name.empty()) {
        std::lock_guard<std::mutex> lk(s_mappingMutex);
        auto it = s_namedMappings.find(name);
        if (it != s_namedMappings.end()) {
            win_SetLastError(183); // ERROR_ALREADY_EXISTS
            const uintptr_t handle = s_nextMappingHandle.fetch_add(4);
            s_fileMappings[handle] = it->second;
            return reinterpret_cast<void*>(handle);
        }
    }

    if (!isAnon) {
        const int inFd = fdFromHandle(hFile);
        if (inFd < 0) {
            win_SetLastError(6); // ERROR_INVALID_HANDLE
            return nullptr;
        }
        dupFd = dup(inFd);
        if (dupFd < 0) {
            win_SetLastError(6);
            return nullptr;
        }
        if (maxSize == 0) {
            struct stat st{};
            if (fstat(dupFd, &st) == 0) {
                maxSize = static_cast<uint64_t>(st.st_size);
            }
            if (maxSize == 0) {
                close(dupFd);
                win_SetLastError(1006); // ERROR_FILE_INVALID
                return nullptr;
            }
        }
    } else {
        if (maxSize == 0) {
            win_SetLastError(87); // ERROR_INVALID_PARAMETER
            return nullptr;
        }
    }

    auto mapping = std::make_shared<WinFileMappingObj>();
    mapping->fd = dupFd;
    mapping->maxSize = maxSize;
    mapping->flProtect = flProtect;
    mapping->name = name;

    const uintptr_t handle = s_nextMappingHandle.fetch_add(4);
    {
        std::lock_guard<std::mutex> lk(s_mappingMutex);
        s_fileMappings[handle] = mapping;
        if (!name.empty()) s_namedMappings[name] = mapping;
    }
    LOGI("win_CreateFileMappingW: handle=%p name='%s' size=%llu (OK)",
         reinterpret_cast<void*>(handle), name.c_str(), static_cast<unsigned long long>(maxSize));
    win_SetLastError(0);
    return reinterpret_cast<void*>(handle);
}

void* NativeDllInterceptor::win_CreateFileMappingA(void* hFile, void* lpAttributes, uint32_t flProtect,
                                                   uint32_t dwMaxSizeHigh, uint32_t dwMaxSizeLow, const char* lpName) {
    if (lpName) {
        std::u16string wname;
        for (const char* p = lpName; *p; ++p) wname.push_back(static_cast<char16_t>(static_cast<uint8_t>(*p)));
        return win_CreateFileMappingW(hFile, lpAttributes, flProtect, dwMaxSizeHigh, dwMaxSizeLow,
                                      reinterpret_cast<const wchar_t*>(wname.c_str()));
    }
    return win_CreateFileMappingW(hFile, lpAttributes, flProtect, dwMaxSizeHigh, dwMaxSizeLow, nullptr);
}

void* NativeDllInterceptor::win_MapViewOfFile(void* hFileMappingObject, uint32_t dwDesiredAccess,
                                              uint32_t dwFileOffsetHigh, uint32_t dwFileOffsetLow, size_t dwNumberOfBytesToMap) {
    return win_MapViewOfFileEx(hFileMappingObject, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap, nullptr);
}

void* NativeDllInterceptor::win_MapViewOfFileEx(void* hFileMappingObject, uint32_t dwDesiredAccess,
                                                uint32_t dwFileOffsetHigh, uint32_t dwFileOffsetLow,
                                                size_t dwNumberOfBytesToMap, void* lpBaseAddress) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hFileMappingObject);
    std::shared_ptr<WinFileMappingObj> mapping;
    {
        std::lock_guard<std::mutex> lk(s_mappingMutex);
        auto it = s_fileMappings.find(handle);
        if (it != s_fileMappings.end()) mapping = it->second;
    }
    if (!mapping) {
        win_SetLastError(6); // ERROR_INVALID_HANDLE
        return nullptr;
    }

    const uint64_t offset = (static_cast<uint64_t>(dwFileOffsetHigh) << 32) | dwFileOffsetLow;
    size_t bytesToMap = dwNumberOfBytesToMap;
    if (bytesToMap == 0) {
        bytesToMap = (offset < mapping->maxSize) ? static_cast<size_t>(mapping->maxSize - offset) : 0;
    }
    if (bytesToMap == 0) {
        win_SetLastError(87);
        return nullptr;
    }

    // SEC_IMAGE (0x1000000): vista con la disposición de imagen PE (cabeceras y secciones en su
    // RVA), como el cargador de Windows. CoreCLR (.NET 6) abre así los ensamblados IL.
    if ((mapping->flProtect & 0x1000000u) != 0 && mapping->fd >= 0) {
        uint8_t hdr[4096] = {};
        if (pread(mapping->fd, hdr, sizeof(hdr), 0) < 0x200 || hdr[0] != 'M' || hdr[1] != 'Z') {
            win_SetLastError(193);  // ERROR_BAD_EXE_FORMAT
            return nullptr;
        }
        uint32_t lfanew = 0;
        std::memcpy(&lfanew, hdr + 0x3C, 4);
        if (lfanew > sizeof(hdr) - 0x108 || std::memcmp(hdr + lfanew, "PE\0\0", 4) != 0) {
            win_SetLastError(193);
            return nullptr;
        }
        const uint8_t* fh = hdr + lfanew + 4;
        uint16_t nSect = 0, optSize = 0;
        std::memcpy(&nSect, fh + 2, 2);
        std::memcpy(&optSize, fh + 16, 2);
        const uint8_t* opt = fh + 20;
        uint32_t sizeOfImage = 0, sizeOfHeaders = 0;
        std::memcpy(&sizeOfImage, opt + 56, 4);
        std::memcpy(&sizeOfHeaders, opt + 60, 4);
        const size_t page = static_cast<size_t>(getpagesize());
        const size_t imgSize = (static_cast<size_t>(sizeOfImage) + page - 1) & ~(page - 1);
        const size_t sectOff = lfanew + 4 + 20 + optSize;
        if (imgSize == 0 || sectOff + nSect * 40u > sizeof(hdr)) {
            win_SetLastError(193);
            return nullptr;
        }
        int iflags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (lpBaseAddress != nullptr) iflags |= MAP_FIXED_NOREPLACE;
        void* img = lpBaseAddress == nullptr && lowmem::enabled()
                        ? lowmem::mmapBelow4G(imgSize)  // invitado de 32 bits: vista < 4 GB
                        : ::mmap(lpBaseAddress, imgSize, PROT_READ | PROT_WRITE, iflags, -1, 0);
        if (img == MAP_FAILED || (lpBaseAddress != nullptr && img != lpBaseAddress)) {
            if (img != MAP_FAILED) ::munmap(img, imgSize);
            win_SetLastError(8);
            return nullptr;
        }
        auto* base = static_cast<uint8_t*>(img);
        pread(mapping->fd, base, std::min<size_t>(sizeOfHeaders, imgSize), 0);
        for (uint16_t i = 0; i < nSect; ++i) {
            const uint8_t* sh = hdr + sectOff + i * 40u;
            uint32_t va = 0, rawSize = 0, rawPtr = 0;
            std::memcpy(&va, sh + 12, 4);
            std::memcpy(&rawSize, sh + 16, 4);
            std::memcpy(&rawPtr, sh + 20, 4);
            if (va >= imgSize || rawSize == 0) continue;
            pread(mapping->fd, base + va, std::min<size_t>(rawSize, imgSize - va), rawPtr);
        }
        {
            std::lock_guard<std::mutex> lk(s_viewMutex);
            s_mapViews[reinterpret_cast<uintptr_t>(img)] = {img, imgSize};
        }
        LOGI("win_MapViewOfFileEx(SEC_IMAGE): handle=%p -> addr=%p imagen=0x%zx, %u secciones", hFileMappingObject,
             img, imgSize, nSect);
        win_SetLastError(0);
        return img;
    }

    int prot = PROT_READ;
    // FILE_MAP_WRITE = 2, FILE_MAP_ALL_ACCESS = 0xF001F, FILE_MAP_COPY = 1
    if ((dwDesiredAccess & 2) != 0 || (dwDesiredAccess & 0xF0000) != 0 || dwDesiredAccess == 1) {
        prot |= PROT_WRITE;
    }

    int flags = (dwDesiredAccess == 1) ? MAP_PRIVATE : MAP_SHARED;
    if (mapping->fd < 0) {
        flags = MAP_PRIVATE | MAP_ANONYMOUS;
    }
    if (lpBaseAddress != nullptr) {
        flags |= MAP_FIXED_NOREPLACE;  // nunca pisar memoria del host; ocupado -> fallo Win32
    }

    void* mapped = MAP_FAILED;
    if (lpBaseAddress == nullptr && lowmem::enabled()) {
        // Invitado de 32 bits: el puntero devuelto debe caber en 32 bits. Se reserva un hueco bajo y
        // la vista se coloca encima con MAP_FIXED (el hueco es nuestro).
        void* hole = lowmem::mmapBelow4G(bytesToMap, PROT_NONE);
        if (hole != MAP_FAILED) {
            mapped = ::mmap(hole, bytesToMap, prot, flags | MAP_FIXED, mapping->fd, static_cast<off_t>(offset));
            if (mapped == MAP_FAILED) ::munmap(hole, bytesToMap);
        }
    } else {
        mapped = ::mmap(lpBaseAddress, bytesToMap, prot, flags, mapping->fd, static_cast<off_t>(offset));
    }
    if (mapped != MAP_FAILED && lpBaseAddress != nullptr && mapped != lpBaseAddress) {
        ::munmap(mapped, bytesToMap);
        mapped = MAP_FAILED;
    }
    if (mapped == MAP_FAILED) {
        LOGE("win_MapViewOfFileEx: mmap falló (errno=%d %s)", errno, std::strerror(errno));
        win_SetLastError(8); // ERROR_NOT_ENOUGH_MEMORY
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lk(s_viewMutex);
        s_mapViews[reinterpret_cast<uintptr_t>(mapped)] = { mapped, bytesToMap };
    }
    LOGI("win_MapViewOfFileEx: handle=%p -> addr=%p size=%zu (OK)", hFileMappingObject, mapped, bytesToMap);
    win_SetLastError(0);
    return mapped;
}

int32_t NativeDllInterceptor::win_UnmapViewOfFile(const void* lpBaseAddress) {
    s_kernel32Calls.fetch_add(1);
    if (!lpBaseAddress) return 0;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(lpBaseAddress);
    size_t sizeToUnmap = 0;
    {
        std::lock_guard<std::mutex> lk(s_viewMutex);
        auto it = s_mapViews.find(addr);
        if (it != s_mapViews.end()) {
            sizeToUnmap = it->second.size;
            s_mapViews.erase(it);
        }
    }
    if (sizeToUnmap > 0) {
        OryonEmitterBackend::noteUnmap(addr, sizeToUnmap, "UnmapViewOfFile", t_thunkRet);
        ::munmap(const_cast<void*>(lpBaseAddress), sizeToUnmap);
        return 1;
    }
    return 1;
}

int32_t NativeDllInterceptor::win_FlushViewOfFile(const void* lpBaseAddress, size_t dwNumberOfBytesToFlush) {
    s_kernel32Calls.fetch_add(1);
    if (!lpBaseAddress) return 0;
    const size_t sz = (dwNumberOfBytesToFlush == 0) ? 4096 : dwNumberOfBytesToFlush;
    ::msync(const_cast<void*>(lpBaseAddress), sz, MS_SYNC);
    return 1;
}

void* NativeDllInterceptor::win_OpenFileMappingW(uint32_t, int32_t, const wchar_t* lpName) {
    s_kernel32Calls.fetch_add(1);
    if (!lpName) {
        win_SetLastError(87);
        return nullptr;
    }
    const std::string name = wcharToUtf8(lpName);
    std::lock_guard<std::mutex> lk(s_mappingMutex);
    auto it = s_namedMappings.find(name);
    if (it != s_namedMappings.end()) {
        const uintptr_t handle = s_nextMappingHandle.fetch_add(4);
        s_fileMappings[handle] = it->second;
        win_SetLastError(0);
        return reinterpret_cast<void*>(handle);
    }
    win_SetLastError(2); // ERROR_FILE_NOT_FOUND
    return nullptr;
}

void* NativeDllInterceptor::win_OpenFileMappingA(uint32_t access, int32_t inherit, const char* lpName) {
    if (lpName) {
        std::u16string wname;
        for (const char* p = lpName; *p; ++p) wname.push_back(static_cast<char16_t>(static_cast<uint8_t>(*p)));
        return win_OpenFileMappingW(access, inherit, reinterpret_cast<const wchar_t*>(wname.c_str()));
    }
    return nullptr;
}

void* NativeDllInterceptor::win_CreateThread(void*, size_t stackSize, void* lpStartAddress,
                                             void* lpParameter, uint32_t, uint32_t* lpThreadId) {
    s_kernel32Calls.fetch_add(1);
    if (lpStartAddress == nullptr) return nullptr;

    auto threadObj = std::make_shared<WinThreadObj>();
    const uintptr_t handle = s_nextThreadHandle.fetch_add(4);

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // Pila host (no la del juego, que el JIT reserva aparte): al menos 8 MB, porque el despachador
    // anida thunks y callbacks; el tamaño que pide el juego es para su propia pila x86.
    constexpr size_t kMinHostStack = 8u * 1024 * 1024;
    pthread_attr_setstacksize(&attr, std::max(stackSize, kMinHostStack));

    bool isHostFn = false;
    {
        std::lock_guard<std::mutex> lock(s_trampolineMutex);
        isHostFn = (s_registeredTrampolines.find(reinterpret_cast<uintptr_t>(lpStartAddress)) != s_registeredTrampolines.end());
    }

    if (isHostFn) {
        using PthreadFn = void* (*)(void*);
        PthreadFn fn = reinterpret_cast<PthreadFn>(lpStartAddress);
        if (pthread_create(&thread, &attr, fn, lpParameter) != 0) {
            pthread_attr_destroy(&attr);
            return nullptr;
        }
        threadObj->pt = thread;
    } else {
        auto* args = new GuestThreadArgs{lpStartAddress, lpParameter, stackSize, threadObj};
        if (pthread_create(&thread, &attr, &guestThreadTrampoline, args) != 0) {
            delete args;
            pthread_attr_destroy(&attr);
            return nullptr;
        }
        threadObj->pt = thread;
    }
    pthread_attr_destroy(&attr);

    {
        std::lock_guard<std::mutex> lk(s_threadMutex);
        s_threads[handle] = threadObj;
    }

    // ThreadId de Windows = TID del kernel (igual que GetCurrentThreadId); antes se devolvía el
    // pthread_t truncado y no coincidía con el que el propio hilo ve.
    uint32_t tid = static_cast<uint32_t>(thread);
    if (!isHostFn) {
        std::unique_lock<std::mutex> lk(threadObj->mtx);
        threadObj->cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return threadObj->tid.load() != 0; });
        if (threadObj->tid.load() != 0) tid = static_cast<uint32_t>(threadObj->tid.load());
    }
    if (lpThreadId) *lpThreadId = tid;
    return reinterpret_cast<void*>(handle);
}

// --- Handles de hilo reales y suspensión (GC de Mono/Boehm: stop-the-world) ---------------

static constexpr uintptr_t kThreadHandleTag = 0x7E000000u;
static constexpr uintptr_t kPseudoCurrentThread = ~static_cast<uintptr_t>(1);  // GetCurrentThread()

static void* makeThreadHandle(int tid) {
    return reinterpret_cast<void*>(kThreadHandleTag | (static_cast<uintptr_t>(tid) & 0xFFFFFFu));
}

static bool isTaggedThreadHandle(uintptr_t h) {
    return (h & ~static_cast<uintptr_t>(0xFFFFFFu)) == kThreadHandleTag;
}

// Handle de hilo (pseudo, etiquetado o de CreateThread) -> TID del kernel; 0 si no es un hilo.
static int threadTidFromHandle(void* hThread) {
    const auto h = reinterpret_cast<uintptr_t>(hThread);
    if (h == kPseudoCurrentThread) return static_cast<int>(gettid());
    if (isTaggedThreadHandle(h)) return static_cast<int>(h & 0xFFFFFFu);
    std::lock_guard<std::mutex> lk(s_threadMutex);
    auto it = s_threads.find(h);
    return it != s_threads.end() ? it->second->tid.load() : 0;
}

void* NativeDllInterceptor::win_OpenThread(uint32_t dwDesiredAccess, int32_t, uint32_t dwThreadId) {
    s_kernel32Calls.fetch_add(1);
    (void)dwDesiredAccess;
    if (dwThreadId == 0 || OryonEmitterBackend::forThread(static_cast<int>(dwThreadId)) == nullptr) {
        win_SetLastError(87);
        return nullptr;
    }
    return makeThreadHandle(static_cast<int>(dwThreadId));
}

uint32_t NativeDllInterceptor::win_GetThreadId(void* hThread) {
    s_kernel32Calls.fetch_add(1);
    const int tid = threadTidFromHandle(hThread);
    if (tid == 0) win_SetLastError(6);  // ERROR_INVALID_HANDLE
    return static_cast<uint32_t>(tid);
}

// CONTEXT x64 (layout en Win64Context.h).
static void copyGuestContextOut(const CpuContext& c, uint8_t* out) { win64ctx::toContext(c, out, 0); }
static void copyGuestContextIn(CpuContext& c, const uint8_t* in) { win64ctx::fromContext(c, in, false); }

// --- Excepciones del huésped: VEH y filtro de nivel superior ------------------------------
static std::mutex s_vehMutex;
static std::vector<std::pair<uintptr_t, uintptr_t>> s_vehHandlers;  // {id, handler} en orden de llamada
static std::atomic<uintptr_t> s_nextVehId{0xCAFE9001};
static std::atomic<uintptr_t> s_unhandledFilter{0};
static std::atomic<uint32_t> s_guestExceptionsLogged{0};

// =============================================================================================
// SEH x64 del huésped: VEH -> manejadores de marco (.pdata, como RtlDispatchException) ->
// filtro de excepciones no manejadas. RtlUnwindEx llama a los manejadores de terminación y
// reanuda en el marco destino; RtlRestoreContext consolida los catch de C++ y los longjmp.
//
// Los manejadores (código invitado) se ejecutan anidados (callGuestNested), con dirección de
// retorno 0. Para que un desenrollado desde dentro de un manejador llegue a los marcos de fuera,
// al encontrar ese centinela se continúa con el "contexto padre" (el del punto donde se lanzó la
// excepción, o el marco destino de un catch). Al reanudar en un marco exterior, las llamadas
// anidadas más profundas se cortan (abortNestedAbove) y el nivel del lanzamiento aplica el
// contexto final.
// =============================================================================================
namespace sehimpl {
using Ctx = std::array<uint8_t, win64ctx::kContextSize>;
struct Level {
    int depth;
    uint64_t rsp;
};
thread_local std::vector<Level> t_levels;
thread_local std::vector<Ctx> t_parents;
thread_local bool t_pending = false;
thread_local int t_pendingDepth = 0;
alignas(16) thread_local Ctx t_pendingCtx{};

struct DispatcherContext {
    uint64_t controlPc, imageBase, functionEntry, establisherFrame, targetIp, contextRecord, languageHandler,
        handlerData, historyTable;
    uint32_t scopeIndex, fill0;
};

enum class Result { kContinue, kUnwound, kAborted, kUnhandled };

// `setprop debug.hyperplay.sehlog 1`: cada marco recorrido por el despacho y el desenrollado.
bool sehLog() {
    static const bool on = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.sehlog", v) > 0 && v[0] == '1';
    }();
    return on;
}

OryonEmitterBackend* backendHere() {
    if (t_dispatchBackend != nullptr) return t_dispatchBackend;
    return OryonEmitterBackend::forThread(static_cast<int>(gettid()));
}

uint64_t callGuest(uint64_t fn, const uint64_t* args, size_t argc) {
    OryonEmitterBackend* b = backendHere();
    return b != nullptr ? static_cast<uint64_t>(b->callGuestNested(fn, args, argc)) : 0;
}

bool findModule(uint64_t pc, uint64_t* base, uint64_t* size) {
    if (s_activeImage == nullptr) return false;
    const auto mainBase = reinterpret_cast<uint64_t>(s_activeImage->baseAddress);
    if (pc >= mainBase && pc < mainBase + s_activeImage->totalMappedSize) {
        *base = mainBase;
        *size = s_activeImage->totalMappedSize;
        return true;
    }
    for (const auto& [name, dll] : s_activeImage->localDlls) {
        const auto b = reinterpret_cast<uint64_t>(dll.base);
        if (b != 0 && pc >= b && pc < b + dll.size) {
            *base = b;
            *size = dll.size;
            return true;
        }
    }
    return false;
}

void initOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        seh::setModuleFinder(&findModule);
        seh::setGuestCaller(&callGuest);
    });
}

// Contexto del llamador del thunk en curso (RIP = dirección de retorno, RSP tras el RET).
void callerContext(uint8_t* out) {
    std::memset(out, 0, win64ctx::kContextSize);
    CpuContext c = t_thunkCtx != nullptr ? *t_thunkCtx : CpuContext{};
    if (t_thunkCtx == nullptr) {
        if (OryonEmitterBackend* b = backendHere(); b != nullptr) c = b->context();
    } else {
        c.rip = t_thunkRet;
    }
    win64ctx::toContext(c, out, win64ctx::kContextAll);
}

// Reanudar en `ctx` desde un thunk: si el destino está por encima de un lanzamiento en curso,
// se corta hasta ese nivel; si no, el thunk sigue directamente ahí.
void resumeFromThunk(const uint8_t* ctx) {
    const uint64_t target = seh::ctxRsp(ctx);
    for (const Level& l : t_levels) {
        if (target >= l.rsp) {
            std::memcpy(t_pendingCtx.data(), ctx, win64ctx::kContextSize);
            t_pending = true;
            t_pendingDepth = l.depth;
            if (OryonEmitterBackend* b = backendHere(); b != nullptr) b->abortNestedAbove(l.depth);
            return;
        }
    }
    CpuContext c = t_thunkCtx != nullptr ? *t_thunkCtx : CpuContext{};
    win64ctx::fromContext(c, ctx, true);
    t_ctxOverrideValue = c;
    t_ctxOverride = true;
}

// Llamada a código invitado desde el SEH: si el invitado la abandona saltando a un marco
// exterior (CoreCLR: ClrRestoreNonvolatileContext tras ejecutar un catch), ese contexto se
// aplica como un RtlRestoreContext (pendiente para el nivel que corresponda).
int64_t sehCall(OryonEmitterBackend& b, uint64_t fn, const uint64_t* args, size_t argc) {
    b.watchNextEscape();
    const int64_t r = b.callGuestNested(fn, args, argc);
    CpuContext esc{};
    if (b.takeEscape(esc)) {
        alignas(16) Ctx ctx;
        std::memset(ctx.data(), 0, ctx.size());
        win64ctx::toContext(esc, ctx.data(), win64ctx::kContextAll);
        if (sehLog()) {
            LOGI("[SEH] el invitado salió de la llamada a 0x%llx hacia RIP=0x%llx RSP=0x%llx",
                 (unsigned long long)fn, (unsigned long long)esc.rip, (unsigned long long)esc.gpr[4]);
        }
        resumeFromThunk(ctx.data());
    }
    return r;
}

// Tras una llamada a código invitado: ¿hay un desenrollado pendiente?
bool pendingFor(int depth, uint8_t* ctxOut, Result& r) {
    if (!t_pending) return false;
    if (t_pendingDepth == depth) {
        std::memcpy(ctxOut, t_pendingCtx.data(), win64ctx::kContextSize);
        t_pending = false;
        if (OryonEmitterBackend* b = backendHere(); b != nullptr) b->abortNestedAbove(0x7FFFFFFF);
        r = Result::kUnwound;
    } else {
        r = Result::kAborted;
    }
    return true;
}

uint32_t& recFlags(uint8_t* rec) { return *reinterpret_cast<uint32_t*>(rec + 4); }

// RtlRestoreContext: consolidación (catch de C++) y longjmp, después reanudar.
void restoreContext(uint8_t* ctx, uint8_t* rec) {
    if (rec != nullptr) {
        uint32_t code = 0, nparams = 0;
        std::memcpy(&code, rec, 4);
        std::memcpy(&nparams, rec + 0x18, 4);
        uint64_t info0 = 0;
        std::memcpy(&info0, rec + 0x20, 8);
        if (code == seh::kStatusUnwindConsolidate && nparams >= 1 && info0 != 0) {
            // El callback (p. ej. __CxxCallCatchBlock) ejecuta el catch y devuelve dónde seguir.
            Ctx parent;
            std::memcpy(parent.data(), ctx, win64ctx::kContextSize);
            t_parents.push_back(parent);
            const uint64_t args[1] = {reinterpret_cast<uint64_t>(rec)};
            OryonEmitterBackend* cb = backendHere();
            const uint64_t next = cb != nullptr ? static_cast<uint64_t>(sehCall(*cb, info0, args, 1)) : 0;
            t_parents.pop_back();
            if (t_pending) return;  // el catch lanzó y otro marco lo capturó más arriba
            seh::setCtxRip(ctx, next);
        } else if (code == seh::kStatusLongjump && nparams >= 1 && info0 != 0) {
            // _JUMP_BUFFER x64: Frame, Rbx, Rsp, Rbp, Rsi, Rdi, R12-R15, Rip, MxCsr, FpCsr, Spare, Xmm6-15.
            const auto* jb = reinterpret_cast<const uint8_t*>(info0);
            const int regs[9] = {3, 4, 5, 6, 7, 12, 13, 14, 15};
            for (int i = 0; i < 9; ++i) std::memcpy(ctx + 0x78 + regs[i] * 8, jb + 8 + i * 8, 8);
            std::memcpy(ctx + 0xF8, jb + 80, 8);
            std::memcpy(ctx + 0x1A0 + 6 * 16, jb + 96, 16 * 10);
        }
    }
    resumeFromThunk(ctx);
}

// RtlDispatchException sobre los marcos (sin los VEH).
Result dispatchFrames(OryonEmitterBackend& b, int depth, uint8_t* rec, uint8_t* origCtx) {
    alignas(16) Ctx cur;
    std::memcpy(cur.data(), origCtx, win64ctx::kContextSize);
    size_t parentIdx = t_parents.size();
    for (int guard = 0; guard < 4096; ++guard) {
        if (seh::ctxRip(cur.data()) == 0) {
            if (parentIdx == 0) break;
            --parentIdx;
            cur = t_parents[parentIdx];
            continue;
        }
        alignas(16) Ctx frameCtx = cur;  // contexto del marco antes de desenrollarlo
        seh::FrameInfo fi;
        seh::unwindStep(seh::kUnwFlagEHandler, cur.data(), fi);
        if (sehLog()) {
            LOGI("[SEH] despacho: pc=0x%llx base=0x%llx fn=%p marco=0x%llx manejador=0x%llx -> rip=0x%llx rsp=0x%llx",
                 (unsigned long long)fi.controlPc, (unsigned long long)fi.imageBase, (const void*)fi.function,
                 (unsigned long long)fi.establisherFrame, (unsigned long long)fi.handler,
                 (unsigned long long)seh::ctxRip(cur.data()), (unsigned long long)seh::ctxRsp(cur.data()));
        }
        if (fi.handler != 0) {
            DispatcherContext dc{fi.controlPc, fi.imageBase, reinterpret_cast<uint64_t>(fi.function),
                                 fi.establisherFrame, 0, reinterpret_cast<uint64_t>(cur.data()), fi.handler,
                                 fi.handlerData, 0, 0, 0};
            Ctx parent;
            std::memcpy(parent.data(), origCtx, win64ctx::kContextSize);
            t_parents.push_back(parent);
            const uint64_t args[4] = {reinterpret_cast<uint64_t>(rec), fi.establisherFrame,
                                      reinterpret_cast<uint64_t>(origCtx), reinterpret_cast<uint64_t>(&dc)};
            const auto disp = static_cast<int32_t>(sehCall(b, fi.handler, args, 4));
            t_parents.pop_back();
            if (sehLog()) LOGI("[SEH] manejador 0x%llx -> %d", (unsigned long long)fi.handler, disp);
            Result r;
            if (pendingFor(depth, origCtx, r)) return r;
            if (disp == seh::kDispContinueExecution) return Result::kContinue;
            if (disp == seh::kDispNested) recFlags(rec) |= seh::kEhNestedCall;
            (void)frameCtx;
        }
        if (seh::ctxRsp(cur.data()) == 0) break;
    }
    return Result::kUnhandled;
}

// Lanzamiento completo: VEH -> marcos -> filtro.
Result raise(OryonEmitterBackend& b, uint8_t* rec, uint8_t* ctx) {
    initOnce();
    const int depth = b.nestDepth();
    t_levels.push_back({depth, seh::ctxRsp(ctx)});
    struct Pop {
        ~Pop() { t_levels.pop_back(); }
    } pop;
    uint64_t ptrs[2] = {reinterpret_cast<uint64_t>(rec), reinterpret_cast<uint64_t>(ctx)};
    const uint64_t args[1] = {reinterpret_cast<uint64_t>(ptrs)};
    std::vector<std::pair<uintptr_t, uintptr_t>> handlers;
    {
        std::lock_guard<std::mutex> lk(s_vehMutex);
        handlers = s_vehHandlers;
    }
    for (const auto& [id, handler] : handlers) {
        const auto verdict = static_cast<int32_t>(sehCall(b, handler, args, 1));
        Result r;
        if (pendingFor(depth, ctx, r)) return r;
        if (verdict == win64ctx::kExceptionContinueExecution) return Result::kContinue;
    }
    const Result fr = dispatchFrames(b, depth, rec, ctx);
    if (fr != Result::kUnhandled) return fr;
    if (const uintptr_t filter = s_unhandledFilter.load(); filter != 0) {
        const auto verdict = static_cast<int32_t>(sehCall(b, filter, args, 1));
        Result r;
        if (pendingFor(depth, ctx, r)) return r;
        if (verdict == win64ctx::kExceptionContinueExecution) return Result::kContinue;
    }
    return Result::kUnhandled;
}

// RtlUnwindEx (desde un thunk).
void unwindEx(uint64_t targetFrame, uint64_t targetIp, uint8_t* recIn, uint64_t retVal, uint8_t* ctxOut) {
    initOnce();
    alignas(16) uint8_t localRec[win64ctx::kExceptionRecordSize] = {};
    uint8_t* rec = recIn;
    if (rec == nullptr) {
        rec = localRec;
        const uint32_t code = seh::kStatusUnwind;
        std::memcpy(rec, &code, 4);
        const uint64_t addr = t_thunkRet;
        std::memcpy(rec + 0x10, &addr, 8);
    }
    recFlags(rec) |= seh::kEhUnwinding | (targetFrame != 0 ? 0u : seh::kEhExitUnwind);
    alignas(16) Ctx context;
    callerContext(context.data());
    alignas(16) Ctx next = context;
    size_t parentIdx = t_parents.size();
    OryonEmitterBackend* b = backendHere();
    bool reached = false;
    for (int guard = 0; guard < 4096 && b != nullptr; ++guard) {
        if (seh::ctxRip(next.data()) == 0) {
            if (parentIdx == 0) break;
            --parentIdx;
            next = t_parents[parentIdx];
            context = next;
            continue;
        }
        seh::FrameInfo fi;
        seh::unwindStep(seh::kUnwFlagUHandler, next.data(), fi);
        if (sehLog()) {
            LOGI("[SEH] desenrollado: pc=0x%llx marco=0x%llx manejador=0x%llx destino=0x%llx",
                 (unsigned long long)fi.controlPc, (unsigned long long)fi.establisherFrame,
                 (unsigned long long)fi.handler, (unsigned long long)targetFrame);
        }
        if (targetFrame != 0 && fi.establisherFrame > targetFrame) {
            __android_log_print(ANDROID_LOG_WARN, "HyperPlay.Seh",
                                "RtlUnwindEx: marco 0x%llx pasado del destino 0x%llx (PC 0x%llx)",
                                static_cast<unsigned long long>(fi.establisherFrame),
                                static_cast<unsigned long long>(targetFrame),
                                static_cast<unsigned long long>(fi.controlPc));
            break;
        }
        if (fi.handler != 0) {
            if (fi.establisherFrame == targetFrame) recFlags(rec) |= seh::kEhTargetUnwind;
            DispatcherContext dc{fi.controlPc, fi.imageBase, reinterpret_cast<uint64_t>(fi.function),
                                 fi.establisherFrame, targetIp, reinterpret_cast<uint64_t>(context.data()), fi.handler,
                                 fi.handlerData, 0, 0, 0};
            const uint64_t args[4] = {reinterpret_cast<uint64_t>(rec), fi.establisherFrame,
                                      reinterpret_cast<uint64_t>(context.data()), reinterpret_cast<uint64_t>(&dc)};
            const auto disp = static_cast<int32_t>(sehCall(*b, fi.handler, args, 4));
            if (t_pending) return;  // un manejador de terminación resolvió otra excepción más arriba
            if (disp == seh::kDispContinueSearch) recFlags(rec) &= ~seh::kEhCollidedUnwind;
            else if (disp == seh::kDispCollided) recFlags(rec) |= seh::kEhCollidedUnwind;
        }
        if (fi.establisherFrame == targetFrame) {
            reached = true;
            break;
        }
        context = next;
    }
    if (!reached && targetFrame != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "HyperPlay.Seh", "RtlUnwindEx: no se encontró el marco destino 0x%llx",
                            static_cast<unsigned long long>(targetFrame));
        return;
    }
    seh::setCtxRax(context.data(), retVal);
    seh::setCtxRip(context.data(), targetIp);
    if (ctxOut != nullptr) std::memcpy(ctxOut, context.data(), win64ctx::kContextSize);
    restoreContext(context.data(), rec);
}
}  // namespace sehimpl

namespace {  // declaradas en el bloque anónimo de las exportaciones extra
void win_RtlRestoreContext(void* context, void* record) {
    if (context == nullptr) return;
    alignas(16) sehimpl::Ctx ctx;
    std::memcpy(ctx.data(), context, win64ctx::kContextSize);
    sehimpl::restoreContext(ctx.data(), static_cast<uint8_t*>(record));
}

void win_RtlUnwind(void* targetFrame, void* targetIp, void* record, void* returnValue) {
    sehimpl::unwindEx(reinterpret_cast<uint64_t>(targetFrame), reinterpret_cast<uint64_t>(targetIp),
                      static_cast<uint8_t*>(record), reinterpret_cast<uint64_t>(returnValue), nullptr);
}

void win_RtlRaiseException(void* record) {
    if (record == nullptr) return;
    const auto* rec = static_cast<const uint8_t*>(record);
    uint32_t code = 0, flags = 0, n = 0;
    std::memcpy(&code, rec, 4);
    std::memcpy(&flags, rec + 4, 4);
    std::memcpy(&n, rec + 0x18, 4);
    NativeDllInterceptor::win_RaiseException(code, flags, std::min<uint32_t>(n, 15), rec + 0x20);
}
}  // namespace

int32_t NativeDllInterceptor::dispatchGuestException(OryonEmitterBackend& backend, uint64_t exceptionPointers) {
    std::vector<std::pair<uintptr_t, uintptr_t>> handlers;
    {
        std::lock_guard<std::mutex> lk(s_vehMutex);
        handlers = s_vehHandlers;
    }
    const auto* ptrs = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(exceptionPointers));
    const auto* rec = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[0]));
    uint32_t code = 0;
    uint64_t address = 0, info1 = 0;
    std::memcpy(&code, rec, 4);
    std::memcpy(&address, rec + 0x10, 8);
    std::memcpy(&info1, rec + 0x28, 8);
    const bool logIt = s_guestExceptionsLogged.fetch_add(1) < 40;
    if (logIt) {
        const auto* ctxRec = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[1]));
        uint64_t g[16];
        std::memcpy(g, ctxRec + 0x78, sizeof(g));
        LOGI("[GuestException] RAX=%llx RCX=%llx RDX=%llx RBX=%llx RSP=%llx RBP=%llx RSI=%llx RDI=%llx",
             (unsigned long long)g[0], (unsigned long long)g[1], (unsigned long long)g[2], (unsigned long long)g[3],
             (unsigned long long)g[4], (unsigned long long)g[5], (unsigned long long)g[6], (unsigned long long)g[7]);
        LOGI("[GuestException] R8=%llx R9=%llx R10=%llx R11=%llx R12=%llx R13=%llx R14=%llx R15=%llx",
             (unsigned long long)g[8], (unsigned long long)g[9], (unsigned long long)g[10], (unsigned long long)g[11],
             (unsigned long long)g[12], (unsigned long long)g[13], (unsigned long long)g[14], (unsigned long long)g[15]);
        if (isGuestRangeReadable(reinterpret_cast<const void*>(address), 32)) {
            char hex[32 * 3 + 1] = {};
            const auto* b = reinterpret_cast<const uint8_t*>(address);
            for (int i = 0; i < 32; ++i) std::snprintf(hex + i * 3, 4, "%02x ", b[i]);
            LOGI("[GuestException] bytes@RIP: %s", hex);
        }
        if (isGuestRangeReadable(reinterpret_cast<const void*>(address - 32), 32)) {
            char hex[32 * 3 + 1] = {};
            const auto* b = reinterpret_cast<const uint8_t*>(address - 32);
            for (int i = 0; i < 32; ++i) std::snprintf(hex + i * 3, 4, "%02x ", b[i]);
            LOGI("[GuestException] bytes@RIP-32: %s", hex);
        }
        // Lecturas de diagnóstico que no pueden fallar (process_vm_readv sobre el propio proceso).
        auto peek = [](uint64_t addr, void* out, size_t n) {
            struct iovec l{out, n}, r{reinterpret_cast<void*>(addr), n};
            return addr > 0x10000 && process_vm_readv(getpid(), &l, 1, &r, 1, 0) == static_cast<ssize_t>(n);
        };
        // Llamador: dirección de retorno en [RSP] y el código anterior a ella.
        uint64_t ret = 0;
        uint8_t callerCode[48] = {};
        if (peek(g[4], &ret, 8) && peek(ret - 48, callerCode, 48)) {
            char hex[48 * 3 + 1] = {};
            for (int i = 0; i < 48; ++i) std::snprintf(hex + i * 3, 4, "%02x ", callerCode[i]);
            LOGI("[GuestException] [RSP]=0x%llx código antes: %s", (unsigned long long)ret, hex);
        } else {
            LOGI("[GuestException] [RSP]=0x%llx (código ilegible)", (unsigned long long)ret);
        }
        // Memoria apuntada por RCX y RDX (objetos/estructuras del llamador).
        for (int r : {1, 2}) {
            uint64_t q[8] = {};
            if (peek(g[r] - 16, q, sizeof(q))) {
                LOGI("[GuestException] [%s-16..+48]: %llx %llx | %llx %llx %llx %llx %llx %llx", r == 1 ? "RCX" : "RDX",
                     (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)q[2],
                     (unsigned long long)q[3], (unsigned long long)q[4], (unsigned long long)q[5],
                     (unsigned long long)q[6], (unsigned long long)q[7]);
            }
        }
    }
    (void)handlers;
    auto* recBytes = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ptrs[0]));
    auto* ctxBytes = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ptrs[1]));
    const sehimpl::Result r = sehimpl::raise(backend, recBytes, ctxBytes);
    if (r != sehimpl::Result::kUnhandled) {
        if (logIt) {
            LOGI("[GuestException] 0x%08x en RIP=0x%llx (addr=0x%llx) -> %s", code, static_cast<unsigned long long>(address),
                 static_cast<unsigned long long>(info1),
                 r == sehimpl::Result::kContinue ? "continúa" : (r == sehimpl::Result::kUnwound ? "desenrollada" : "cortada"));
        }
        return win64ctx::kExceptionContinueExecution;  // deliverPendingFault aplica el CONTEXT
    }
    LOGE("[GuestException] 0x%08x en RIP=0x%llx (addr=0x%llx) sin manejar", code,
         static_cast<unsigned long long>(address), static_cast<unsigned long long>(info1));
    return win64ctx::kExceptionContinueSearch;
}

uint32_t NativeDllInterceptor::win_GetCurrentThreadId() {
    s_kernel32Calls.fetch_add(1);
    // Windows expone el TID del kernel; gettid() es el equivalente real, pthread_self() es un
    // puntero opaco y rompe comparaciones de identidad de hilo dentro del binario PE.
    return static_cast<uint32_t>(gettid());
}

void NativeDllInterceptor::win_Sleep(uint32_t dwMilliseconds) {
    s_kernel32Calls.fetch_add(1);
    if (dwMilliseconds == 0) {
        sched_yield();
        return;
    }
    struct timespec ts;
    ts.tv_sec = dwMilliseconds / 1000;
    ts.tv_nsec = (dwMilliseconds % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

// SleepEx: sin APC encoladas nunca se despierta antes (devolvería WAIT_IO_COMPLETION = 0xC0).
uint32_t NativeDllInterceptor::win_SleepEx(uint32_t dwMilliseconds, int32_t /*bAlertable*/) {
    if (dwMilliseconds == 0xFFFFFFFFu) dwMilliseconds = 24u * 3600u * 1000u;
    win_Sleep(dwMilliseconds);
    return 0;
}

// CRITICAL_SECTION: en Win64 mide 40 bytes, lo mismo que pthread_mutex_t en arm64, y el mutex vive
// dentro. En 32 bits mide 24: el mutex desbordaba 16 bytes sobre lo que sigue (msvcr100 tiene una
// tabla de locks contiguos y justo detrás sus punteros a FlsAlloc...: R6016 en DmC). Con un invitado
// de 32 bits el mutex va aparte (memoria baja) y el CRITICAL_SECTION solo guarda su dirección de 4 bytes
// en el primer campo (DebugInfo); el resto de la estructura no se toca.
namespace {
pthread_mutex_t* newLowMutex() {
    static std::mutex s_m;
    static uint8_t* s_pool = nullptr;
    static size_t s_left = 0;
    std::lock_guard<std::mutex> lk(s_m);
    constexpr size_t kSlot = (sizeof(pthread_mutex_t) + 15) & ~size_t{15};
    if (s_left < kSlot) {
        void* p = lowmem::mmapBelow4G(1u << 20);
        if (p == MAP_FAILED) return nullptr;
        s_pool = static_cast<uint8_t*>(p);
        s_left = 1u << 20;
    }
    auto* m = reinterpret_cast<pthread_mutex_t*>(s_pool);
    s_pool += kSlot;
    s_left -= kSlot;
    return m;
}

pthread_mutex_t* csMutex(void* cs) {
    if (!lowmem::enabled()) return reinterpret_cast<pthread_mutex_t*>(cs);
    uint32_t addr = 0;
    std::memcpy(&addr, cs, 4);
    return reinterpret_cast<pthread_mutex_t*>(static_cast<uintptr_t>(addr));
}
}  // namespace

void NativeDllInterceptor::win_InitializeCriticalSection(void* lpCriticalSection) {
    s_kernel32Calls.fetch_add(1);
    if (!lpCriticalSection) return;
    pthread_mutex_t* mtx = reinterpret_cast<pthread_mutex_t*>(lpCriticalSection);
    if (lowmem::enabled()) {
        mtx = newLowMutex();
        if (mtx == nullptr) return;
        const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mtx));
        std::memset(lpCriticalSection, 0, 24);
        std::memcpy(lpCriticalSection, &addr, 4);
        static_cast<int32_t*>(lpCriticalSection)[1] = -1;  // LockCount: libre
    }
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(mtx, &attr);
    pthread_mutexattr_destroy(&attr);
}

void NativeDllInterceptor::win_EnterCriticalSection(void* lpCriticalSection) {
    s_kernel32Calls.fetch_add(1);
    if (!lpCriticalSection) return;
    if (pthread_mutex_t* m = csMutex(lpCriticalSection)) {
        if (!lowmem::enabled()) { pthread_mutex_lock(m); return; }
        // 32 bits: CRITICAL_SECTION x86 = {DebugInfo, LockCount, RecursionCount, OwningThread, ...}.
        // Se mantienen RecursionCount/OwningThread como Windows (código y diagnóstico los leen).
        auto* f = static_cast<int32_t*>(lpCriticalSection);
        if (pthread_mutex_trylock(m) != 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 10;
            if (pthread_mutex_timedlock(m, &ts) != 0) {
                LOGW("EnterCriticalSection(%p): más de 10 s esperando; dueño tid=%d recursión=%d", lpCriticalSection,
                     __atomic_load_n(&f[3], __ATOMIC_RELAXED), __atomic_load_n(&f[2], __ATOMIC_RELAXED));
                pthread_mutex_lock(m);
            }
        }
        if (f[2]++ == 0) __atomic_store_n(&f[3], static_cast<int32_t>(gettid()), __ATOMIC_RELAXED);
        return;
    }
    static std::atomic<int> s_nulls{0};
    if (s_nulls.fetch_add(1) < 5) LOGW("EnterCriticalSection(%p): sin mutex (CS no inicializada)", lpCriticalSection);
}

void NativeDllInterceptor::win_LeaveCriticalSection(void* lpCriticalSection) {
    s_kernel32Calls.fetch_add(1);
    if (!lpCriticalSection) return;
    if (pthread_mutex_t* m = csMutex(lpCriticalSection)) {
        if (lowmem::enabled()) {
            auto* f = static_cast<int32_t*>(lpCriticalSection);
            if (f[2] > 0 && --f[2] == 0) __atomic_store_n(&f[3], 0, __ATOMIC_RELAXED);
        }
        pthread_mutex_unlock(m);
    }
}

void NativeDllInterceptor::win_DeleteCriticalSection(void* lpCriticalSection) {
    s_kernel32Calls.fetch_add(1);
    if (lpCriticalSection) {
        if (pthread_mutex_t* m = csMutex(lpCriticalSection)) pthread_mutex_destroy(m);
    }
}

uint32_t NativeDllInterceptor::win_GetTickCount() {
    s_kernel32Calls.fetch_add(1);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint32_t>(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

int32_t NativeDllInterceptor::win_QueryPerformanceCounter(int64_t* lpPerformanceCount) {
    s_kernel32Calls.fetch_add(1);
    if (!lpPerformanceCount) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *lpPerformanceCount = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    return 1;
}

int32_t NativeDllInterceptor::win_QueryPerformanceFrequency(int64_t* lpFrequency) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFrequency) return 0;
    *lpFrequency = 1000000000LL; // 1 GHz (nanosegundos)
    return 1;
}

uint32_t NativeDllInterceptor::win_timeGetTime() {
    s_otherCalls.fetch_add(1);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const auto ms = static_cast<uint32_t>(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
    static std::atomic<uint32_t> s_logged{0};
    if (s_logged.load(std::memory_order_relaxed) < 3) {
        const uint32_t n = s_logged.fetch_add(1, std::memory_order_relaxed);
        if (n < 3) LOGI("timeGetTime #%u -> %u ms (CLOCK_MONOTONIC)", n + 1, ms);
    }
    return ms;
}

uint32_t NativeDllInterceptor::win_timeBeginPeriod(uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0; // TIMERR_NOERROR
}

uint32_t NativeDllInterceptor::win_timeEndPeriod(uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0; // TIMERR_NOERROR
}

uint32_t NativeDllInterceptor::win_timeGetDevCaps(void* ptc, uint32_t cbtc) {
    s_otherCalls.fetch_add(1);
    if (ptc == nullptr || cbtc < 8) return 11; // MMSYSERR_INVALPARAM
    auto* caps = static_cast<uint32_t*>(ptc);
    caps[0] = 1;        // TIMECAPS.wPeriodMin
    caps[1] = 1000000;  // TIMECAPS.wPeriodMax
    return 0;           // TIMERR_NOERROR
}

static void* const kHModKernel32 = reinterpret_cast<void*>(0x7FFF0001);
static void* const kHModUser32   = reinterpret_cast<void*>(0x7FFF0002);
static void* const kHModVulkan   = reinterpret_cast<void*>(0x7FFF0003);
static void* const kHModNtdll    = reinterpret_cast<void*>(0x7FFF0004);
static void* const kHModAdvapi32 = reinterpret_cast<void*>(0x7FFF0005);
static void* const kHModShell32  = reinterpret_cast<void*>(0x7FFF0006);
static void* const kHModOle32    = reinterpret_cast<void*>(0x7FFF0007);
static void* const kHModWs2_32   = reinterpret_cast<void*>(0x7FFF0008);
static void* const kHModD3D11    = reinterpret_cast<void*>(0x7FFF0009);
static void* const kHModDxgi     = reinterpret_cast<void*>(0x7FFF000A);
static void* const kHModXInput   = reinterpret_cast<void*>(0x7FFF000B);
static void* const kHModWinmm    = reinterpret_cast<void*>(0x7FFF000C);
static void* const kHModBcrypt   = reinterpret_cast<void*>(0x7FFF000D);
static void* const kHModCryptbase = reinterpret_cast<void*>(0x7FFF000E);
static void* const kHModShlwapi  = reinterpret_cast<void*>(0x7FFF000F);
static void* const kHModSteamApi = reinterpret_cast<void*>(0x7FFF0010);
static void* const kHModMfplat   = reinterpret_cast<void*>(0x7FFF0011);  // mfplat/mfreadwrite/mf
static void* const kHModOpenGl   = reinterpret_cast<void*>(0x7FFF0012);  // opengl32 (EGL/GLES)
static void* const kHModGdi32    = reinterpret_cast<void*>(0x7FFF0013);

// opengl32.dll de HyperPlay (gfx/GuestOpenGl.h): registro de trampolines y lectura de XMM.
static void ensureGuestGl() {
    static std::once_flag once;
    std::call_once(once, [] {
        guestgl::init([](void* fn, const char* name) { registerTrampolineAddress(fn, name); },
                      [](int i) -> const uint8_t* {
                          if (t_dispatchBackend == nullptr || i < 0 || i > 15) return nullptr;
                          return t_dispatchBackend->context().simd[i];
                      });
    });
}

// Media Foundation sobre NDK Media: los métodos COM del juego (IMFByteStream,
// IMFSourceReaderCallback) se invocan con el backend del hilo invitado que llama.
static void ensureMediaFoundation() {
    hyperplay::mf::init(
        [](void* fn, const char* name) { registerTrampolineAddress(fn, name); },
        [](uintptr_t fn, const uint64_t* args, size_t n) -> int64_t {
            if (t_dispatchBackend == nullptr) {
                LOGE("Media Foundation: llamada al huésped sin hilo invitado");
                return static_cast<int64_t>(0x80004005u);
            }
            return t_dispatchBackend->callGuestNested(fn, args, n);
        });
}

// steam_api(64).dll cargada por LoadLibrary (P/Invoke de Steamworks.NET). Las DLL de Steam que
// trae el juego (o emuladores como el de la carpeta steam_settings) no funcionan sin cliente; el
// juego gestionado recibe un Steam "no inicializado", ruta que los juegos ya contemplan (Hollow
// Knight: "Steam failed to initialize" y sigue sin logros). Las importaciones nativas (IAT)
// siguen usando NativeSteamStub, que sí simula un Steam iniciado.
namespace {  // mismo espacio anónimo que la declaración adelantada de resolveSteamOffline
namespace steamoffline {
int32_t Init() {
    LOGI("SteamAPI_Init (P/Invoke) -> false: modo sin Steam");
    return 0;
}
int32_t InitFlat(char* msg) {
    if (msg != nullptr) std::strcpy(msg, "Steam no disponible en HyperPlay");
    return 1;  // k_ESteamAPIInitResult_FailedGeneric
}
int32_t InternalInit(const char*, char* msg) { return InitFlat(msg); }
int32_t ReturnFalse() { return 0; }
void NoOp() {}
}  // namespace steamoffline

static void* resolveSteamOffline(const char* name) {
    const std::string n = name;
    if (n == "SteamAPI_Init") return w32arity::hpFn(&steamoffline::Init);
    if (n == "SteamAPI_InitFlat") return w32arity::hpFn(&steamoffline::InitFlat);
    if (n == "SteamInternal_SteamAPI_Init") return w32arity::hpFn(&steamoffline::InternalInit);
    if (n == "SteamAPI_RestartAppIfNecessary" || n == "SteamAPI_IsSteamRunning" ||
        n == "SteamAPI_GetHSteamPipe" || n == "SteamAPI_GetHSteamUser" ||
        n == "SteamGameServer_GetHSteamPipe" || n == "SteamGameServer_GetHSteamUser") {
        return w32arity::hpFn(&steamoffline::ReturnFalse);
    }
    if (n == "SteamAPI_Shutdown" || n == "SteamAPI_RunCallbacks" || n == "SteamAPI_ReleaseCurrentThreadMemory" ||
        n == "SteamAPI_ManualDispatch_Init" || n == "SteamAPI_ManualDispatch_RunFrame" ||
        n == "SteamAPI_RegisterCallback" || n == "SteamAPI_UnregisterCallback" ||
        n == "SteamAPI_RegisterCallResult" || n == "SteamAPI_UnregisterCallResult" ||
        n == "SteamClient_SetWarningMessageHook" || n == "SteamAPI_ISteamClient_SetWarningMessageHook") {
        return w32arity::hpFn(&steamoffline::NoOp);
    }
    return nullptr;
}
}  // namespace

static void* resolveModuleHandle(std::string name) {
    if (name.empty()) {
        return s_activeImage ? s_activeImage->baseAddress : reinterpret_cast<void*>(0x140000000ULL);
    }
    const std::string rawName = name;
    std::string baseName = name;
    const size_t slash = baseName.find_last_of("/\\");
    if (slash != std::string::npos) baseName = baseName.substr(slash + 1);
    std::string lowerBase = baseName;
    for (char& c : lowerBase) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    // .NET Framework: mscoree.dll cargado es una copia de mscoreei.dll; este se busca a sí mismo por
    // su nombre real (recursos de LoadString, rutas).
    if ((lowerBase == "mscoreei.dll" || lowerBase == "mscoreei") && s_activeImage != nullptr) {
        for (const auto& kv : s_activeImage->localDlls) {
            std::string n = kv.first;
            for (char& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (n.size() >= 11 && n.compare(n.size() - 11, 11, "mscoree.dll") == 0) return kv.second.base;
        }
    }

    if (lowerBase == "kernel32" || lowerBase == "kernel32.dll" || lowerBase == "kernelbase" || lowerBase == "kernelbase.dll" ||
        lowerBase == "psapi" || lowerBase == "psapi.dll") return kHModKernel32;
    if (lowerBase == "user32" || lowerBase == "user32.dll") return kHModUser32;
    if (lowerBase == "vulkan-1" || lowerBase == "vulkan-1.dll" || lowerBase == "vulkan") return kHModVulkan;
    if (lowerBase == "opengl32" || lowerBase == "opengl32.dll") return kHModOpenGl;
    if (lowerBase == "gdi32" || lowerBase == "gdi32.dll") return kHModGdi32;
    if (lowerBase == "ntdll" || lowerBase == "ntdll.dll") return kHModNtdll;
    if (lowerBase == "advapi32" || lowerBase == "advapi32.dll") return kHModAdvapi32;
    if (lowerBase == "shell32" || lowerBase == "shell32.dll" || lowerBase == "shfolder" || lowerBase == "shfolder.dll") return kHModShell32;
    if (lowerBase == "ole32" || lowerBase == "ole32.dll" || lowerBase == "combase" || lowerBase == "combase.dll") return kHModOle32;
    if (lowerBase == "ws2_32" || lowerBase == "ws2_32.dll") return kHModWs2_32;
    if (lowerBase == "d3d11" || lowerBase == "d3d11.dll") return kHModD3D11;
    if (lowerBase == "dxgi" || lowerBase == "dxgi.dll") return kHModDxgi;
    if (lowerBase == "winmm" || lowerBase == "winmm.dll") return kHModWinmm;
    if (lowerBase == "bcrypt" || lowerBase == "bcrypt.dll") return kHModBcrypt;
    if (lowerBase == "cryptbase" || lowerBase == "cryptbase.dll") return kHModCryptbase;
    if (lowerBase == "shlwapi" || lowerBase == "shlwapi.dll") return kHModShlwapi;
    if (lowerBase == "steam_api64" || lowerBase == "steam_api64.dll" || lowerBase == "steam_api" ||
        lowerBase == "steam_api.dll") return kHModSteamApi;
    if (lowerBase == "mfplat" || lowerBase == "mfplat.dll" || lowerBase == "mfreadwrite" ||
        lowerBase == "mfreadwrite.dll" || lowerBase == "mf" || lowerBase == "mf.dll") {
        ensureMediaFoundation();
        return kHModMfplat;
    }
    if (lowerBase.rfind("xinput", 0) == 0 || lowerBase == "dinput8.dll" || lowerBase == "dinput8") return kHModXInput;
    if (lowerBase.rfind("api-ms-win-core-", 0) == 0 ||
        lowerBase.rfind("api-ms-win-crt-", 0) == 0 ||
        lowerBase.rfind("api-ms-win-", 0) == 0 ||
        lowerBase.rfind("ext-ms-win-", 0) == 0) {
        return kHModKernel32;
    }

    if (s_activeImage != nullptr) {
        auto it = s_activeImage->localDlls.find(rawName);
        if (it != s_activeImage->localDlls.end()) return it->second.base;
        it = s_activeImage->localDlls.find(baseName);
        if (it != s_activeImage->localDlls.end()) return it->second.base;
        it = s_activeImage->localDlls.find(lowerBase);
        if (it != s_activeImage->localDlls.end()) return it->second.base;

        BionicPeLoader loader;
        // Intenta primero con la ruta completa relativa provista (p.ej. MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll)
        void* dllBase = loader.loadLocalDll(*s_activeImage, rawName);
        if (dllBase != nullptr) return dllBase;

        // Fallback al nombre base si era diferente
        if (baseName != rawName) {
            dllBase = loader.loadLocalDll(*s_activeImage, baseName);
            if (dllBase != nullptr) return dllBase;
        }
    }

    NativeDllInterceptor::win_SetLastError(126); // ERROR_MOD_NOT_FOUND (126)
    return nullptr;
}

void* NativeDllInterceptor::win_GetModuleHandleW(const wchar_t* lpModuleName) {
    s_kernel32Calls.fetch_add(1);
    if (lpModuleName == nullptr) {
        return s_activeImage ? s_activeImage->baseAddress : reinterpret_cast<void*>(0x140000000ULL);
    }
    return resolveModuleHandle(wcharToUtf8(lpModuleName));
}

void* NativeDllInterceptor::win_GetModuleHandleA(const char* lpModuleName) {
    s_kernel32Calls.fetch_add(1);
    if (lpModuleName == nullptr) {
        return s_activeImage ? s_activeImage->baseAddress : reinterpret_cast<void*>(0x140000000ULL);
    }
    return resolveModuleHandle(lpModuleName);
}

// Export de una imagen PE mapeada: por nombre u ordinal (lpProcName < 0x10000), resolviendo
// exports reenviados ("DLL.Func" / "DLL.#ord") de forma recursiva.
static void* resolvePeExport(void* base, const PeInfo& pe, const char* lpProcName, int depth) {
    const auto procValue = reinterpret_cast<uintptr_t>(lpProcName);
    uint32_t rva = 0;
    if ((procValue >> 16) == 0) {
        for (const auto& e : pe.exports) {
            if (e.ordinal == procValue) {
                rva = e.rva;
                break;
            }
        }
    } else {
        rva = pe.findExportRva(lpProcName);
    }
    if (rva == 0) return nullptr;
    auto* addr = static_cast<uint8_t*>(base) + rva;
    if (rva < pe.exportRva || rva >= pe.exportRva + pe.exportSize) return addr;
    // Forwarder: cadena "MODULO.Funcion" dentro del directorio de exports.
    // Tope por hilo: GetProcAddress del destino puede volver a buscar en las DLL locales y dar con
    // el mismo forwarder (kernel32.X sin implementar reenviado desde api-ms-win-core-*).
    static thread_local int t_forwardDepth = 0;
    if (depth >= 4 || t_forwardDepth >= 4) return nullptr;
    struct DepthGuard {
        DepthGuard() { ++t_forwardDepth; }
        ~DepthGuard() { --t_forwardDepth; }
    } guard;
    const std::string fwd(reinterpret_cast<const char*>(addr));
    const size_t dot = fwd.find('.');
    if (dot == std::string::npos) return nullptr;
    const std::string dll = fwd.substr(0, dot) + ".dll";
    const std::string func = fwd.substr(dot + 1);
    LOGI("GetProcAddress: export reenviado '%s'", fwd.c_str());
    // api-ms-win-crt-string-* -> ucrtbase: las funciones seguras de cadenas van nativas (hostcrt).
    if (toLower(dll) == "ucrtbase.dll") {
        char v[PROP_VALUE_MAX] = {};
        const bool guestOnly = __system_property_get("debug.hyperplay.guestcrtstr", v) > 0 && v[0] == '1';
        if (void* host = guestOnly ? nullptr : hostcrt::resolve(func); host != nullptr) {
            registerTrampolineAddress(host, "ucrtbase.dll!" + func);
            return host;
        }
    }
    void* mod = NativeDllInterceptor::win_LoadLibraryA(dll.c_str());
    if (mod == nullptr) return nullptr;
    if (!func.empty() && func[0] == '#') {
        const auto ord = static_cast<uintptr_t>(std::strtoul(func.c_str() + 1, nullptr, 10));
        return NativeDllInterceptor::win_GetProcAddress(mod, reinterpret_cast<const char*>(ord));
    }
    return NativeDllInterceptor::win_GetProcAddress(mod, func.c_str());
}

// Diagnóstico (`setprop debug.hyperplay.monotrace "M:Clase:Metodo,..."`, más `monotrace2` para
// alargarlo): antes de `mono_jit_init_version` activa el trazado de llamadas de Mono
// (`mono_jit_set_trace_options`), que imprime ENTER/LEAVE de esos métodos por la consola del juego.
static void enableMonoTraceIfRequested(void* base, const PeInfo& pe) {
    static std::atomic<bool> s_done{false};
    char a[PROP_VALUE_MAX] = {}, b[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.hyperplay.monotrace", a) <= 0 || s_done.exchange(true)) return;
    __system_property_get("debug.hyperplay.monotrace2", b);
    static std::string s_spec;
    s_spec = a;
    if (b[0] != '\0') s_spec += std::string(",") + b;
    void* fn = resolvePeExport(base, pe, "mono_jit_set_trace_options", 0);
    OryonEmitterBackend* backend = t_dispatchBackend;
    if (fn == nullptr || backend == nullptr) {
        LOGW("monotrace: no disponible (fn=%p, backend=%p)", fn, static_cast<void*>(backend));
        return;
    }
    const uint64_t args[1] = {reinterpret_cast<uint64_t>(s_spec.c_str())};
    const int64_t ok = backend->callGuestNested(reinterpret_cast<uintptr_t>(fn), args, 1);
    LOGI("monotrace: mono_jit_set_trace_options(\"%s\") -> %lld", s_spec.c_str(), static_cast<long long>(ok));
}

// ¿Es el fmodstudio.dll mudo que trae el runtime de HyperPlay (y no un FMOD real del usuario)?
static bool isHyperPlayFmodShim(const std::string& hostPath) {
    if (!managed::active() || hostPath.compare(0, managed::g_runtimeDir.size(), managed::g_runtimeDir) != 0) {
        return false;
    }
    std::string base = hostPath.substr(hostPath.find_last_of('/') + 1);
    for (char& c : base) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (base != "fmodstudio.dll" && base != "fmodstudio64.dll") return false;
    // setprop debug.hyperplay.fmodnative 0: usa el shim mudo (diagnóstico).
    char prop[PROP_VALUE_MAX] = {};
    __system_property_get("debug.hyperplay.fmodnative", prop);
    if (prop[0] == '0') return false;
    static std::once_flag once;
    std::call_once(once, [] {
        fmodhost::Hooks hooks;
        hooks.resolvePath = [](const std::string& winPath) {
            return s_activeImage ? BionicPeLoader::resolveVirtualPath(*s_activeImage, winPath) : winPath;
        };
        hooks.xmm = [](int i) -> const uint8_t* {
            if (t_dispatchBackend == nullptr || i < 0 || i > 15) return nullptr;
            return t_dispatchBackend->context().simd[i];
        };
        hooks.gameDir = managed::g_gameDir;
        fmodhost::init(hooks);
        LOGI("FMOD nativo de HyperPlay activo para %s", hooks.gameDir.c_str());
    });
    return true;
}

namespace {
std::string lowerAscii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::atomic<uint64_t> s_alIsExtensionPresentReal{0};
// ALboolean alIsExtensionPresent(const ALchar*): AL_SOFT_MSADPCM -> falso; lo demás, el de OpenAL Soft.
uint8_t hp_alIsExtensionPresent(const char* name) {
    if (name != nullptr && lowerAscii(name) == "al_soft_msadpcm") return 0;
    const uint64_t real = s_alIsExtensionPresentReal.load();
    if (real == 0) return 0;
    const uint64_t args[1] = {reinterpret_cast<uint64_t>(name)};
    return static_cast<uint8_t>(sehimpl::callGuest(real, args, 1));
}
}  // namespace

void* NativeDllInterceptor::win_GetProcAddress(void* hModule, const char* lpProcName) {
    void* p = nullptr;
    // .NET Framework: nuestro mscoree.dll es una copia de mscoreei.dll (el shim de System32 no se
    // distribuye). mscoreei pide al shim su ordinal 142, que en el shim real es GetVersionExW (salto
    // directo a la IAT); con la copia sería su propio _CorExeMain.
    if (reinterpret_cast<uintptr_t>(lpProcName) == 142 && s_activeImage != nullptr) {
        for (const auto& kv : s_activeImage->localDlls) {
            const std::string n = toLower(kv.first);
            if (kv.second.base == hModule && n.size() >= 11 && n.compare(n.size() - 11, 11, "mscoree.dll") == 0) {
                p = w32arity::hpFn(&hp_GetVersionExW);
                LOGI("GetProcAddress(mscoree, #142) -> GetVersionExW (shim)");
                break;
            }
        }
    }
    if (p == nullptr) p = win_GetProcAddressCore(hModule, lpProcName);
    // Invitado de 32 bits: un puntero nativo (>4 GB) no cabe en sus variables; se da el alias bajo.
    if (p != nullptr && s_activeImage != nullptr && s_activeImage->is32Bit &&
        reinterpret_cast<uintptr_t>(p) > 0xFFFFFFFFull) {
        const bool byOrdinal = (reinterpret_cast<uintptr_t>(lpProcName) >> 16) == 0;
        const uintptr_t low = w32alias::lowFor(p, byOrdinal ? std::string() : std::string(lpProcName));
        return low != 0 ? reinterpret_cast<void*>(low) : nullptr;
    }
    return p;
}

void* NativeDllInterceptor::win_GetProcAddressCore(void* hModule, const char* lpProcName) {
    s_kernel32Calls.fetch_add(1);
    if (lpProcName == nullptr || hModule == nullptr) {
        win_SetLastError(127); // ERROR_PROC_NOT_FOUND
        return nullptr;
    }
    const bool byOrdinal = (reinterpret_cast<uintptr_t>(lpProcName) >> 16) == 0;

    // 1. Si hModule es el exe principal:
    if (s_activeImage && hModule == s_activeImage->baseAddress) {
        if (void* p = resolvePeExport(hModule, s_activeImage->peInfo, lpProcName, 0)) return p;
        if (byOrdinal) {
            win_SetLastError(127);
            return nullptr;
        }
    }

    // 2. Comprobar si hModule coincide con alguna DLL local mapeada
    if (s_activeImage != nullptr) {
        for (const auto& kv : s_activeImage->localDlls) {
            if (kv.second.base == hModule) {
                if (!byOrdinal && std::strcmp(lpProcName, "mono_jit_init_version") == 0) {
                    enableMonoTraceIfRequested(hModule, kv.second.peInfo);
                }
                // fmodstudio.dll del runtime (shim mudo): FMOD Studio nativo de HyperPlay. Con el
                // FMOD propio del usuario (/sdcard/HyperPlay/fmod) se usa el suyo tal cual.
                if (!byOrdinal && isHyperPlayFmodShim(kv.second.hostPath)) {
                    if (void* p = fmodhost::resolveExport(lpProcName)) {
                        registerTrampolineAddress(p, std::string("fmodstudio.dll!") + lpProcName);
                        return p;
                    }
                    static std::mutex s_missMu;
                    static std::unordered_set<std::string> s_missing;
                    std::lock_guard<std::mutex> lock(s_missMu);
                    if (s_missing.insert(lpProcName).second) {
                        LOGW("FMOD nativo: %s sin implementar (se usa el shim)", lpProcName);
                    }
                }
                void* p = resolvePeExport(hModule, kv.second.peInfo, lpProcName, 0);
                // OpenAL Soft: setprop debug.hyperplay.alnoadpcm 1 oculta AL_SOFT_MSADPCM, así MonoGame
                // decodifica el ADPCM de los efectos él mismo (diagnóstico del audio distorsionado).
                if (p != nullptr && !byOrdinal && std::strcmp(lpProcName, "alIsExtensionPresent") == 0 &&
                    lowerAscii(kv.second.hostPath).find("soft_oal") != std::string::npos) {
                    char v[PROP_VALUE_MAX] = {};
                    if (__system_property_get("debug.hyperplay.alnoadpcm", v) > 0 && v[0] == '1') {
                        s_alIsExtensionPresentReal = reinterpret_cast<uint64_t>(p);
                        void* w = w32arity::hpFn(&hp_alIsExtensionPresent);
                        registerTrampolineAddress(w, "soft_oal.dll!alIsExtensionPresent");
                        LOGI("OpenAL: alIsExtensionPresent envuelto (sin AL_SOFT_MSADPCM)");
                        return w;
                    }
                }
                if (p) return p;
                if (byOrdinal) {
                    win_SetLastError(127);
                    return nullptr;
                }
                break;
            }
        }
    }
    if (byOrdinal) {  // ordinal sobre módulo de sistema sintético: no hay tabla de ordinales
        // XInput: el ordinal 100 es XInputGetStateEx (incluye el botón Guide); mismo estado.
        if (hModule == kHModXInput && reinterpret_cast<uintptr_t>(lpProcName) == 100) {
            void* p = w32arity::hpFn(&DirectXInputBridge::XInputGetState);
            registerTrampolineAddress(p, "xinput1_4.dll!XInputGetStateEx");
            return p;
        }
        LOGW("GetProcAddress(%p, #%u): ordinal en módulo sin tabla", hModule,
             static_cast<unsigned>(reinterpret_cast<uintptr_t>(lpProcName)));
        win_SetLastError(127);
        return nullptr;
    }

    if (hModule == kHModOpenGl) {
        ensureGuestGl();
        return guestgl::glProcAddr(lpProcName);  // NULL si no existe, como Windows
    }
    if (hModule == kHModGdi32) {
        if (void* p = resolveNativeSymbol("gdi32.dll", lpProcName)) return p;
        return nullptr;
    }
    if (hModule == kHModMfplat) {
        ensureMediaFoundation();
        if (void* p = hyperplay::mf::resolveExport(lpProcName)) return p;
        win_SetLastError(127);
        return nullptr;
    }

    if (hModule == kHModSteamApi) {
        if (s_activeImage != nullptr && s_activeImage->is32Bit && (reinterpret_cast<uintptr_t>(lpProcName) >> 16) != 0) {
            if (void* p32 = steamemu32::resolve(lpProcName, s_activeImage->savesDir)) {
                registerTrampolineAddress(p32, std::string("steam_api.dll!") + lpProcName);
                return p32;
            }
        }
        if (void* p = resolveSteamOffline(lpProcName)) {
            registerTrampolineAddress(p, std::string("steam_api64.dll!") + lpProcName);
            return p;
        }
        LOGW("GetProcAddress(steam_api64.dll!%s) -> NULL (modo sin Steam)", lpProcName);
        win_SetLastError(127);
        return nullptr;
    }

    // 3. Mapear hModule a biblioteca del sistema si corresponde
    std::string targetDll = "kernel32.dll";
    if (hModule == kHModKernel32) targetDll = "kernel32.dll";
    else if (hModule == kHModUser32) targetDll = "user32.dll";
    else if (hModule == kHModVulkan) targetDll = "vulkan-1.dll";
    else if (hModule == kHModNtdll) targetDll = "ntdll.dll";
    else if (hModule == kHModAdvapi32) targetDll = "advapi32.dll";
    else if (hModule == kHModShell32) targetDll = "shell32.dll";
    else if (hModule == kHModOle32) targetDll = "ole32.dll";
    else if (hModule == kHModWs2_32) targetDll = "ws2_32.dll";
    else if (hModule == kHModD3D11) targetDll = "d3d11.dll";
    else if (hModule == kHModDxgi) targetDll = "dxgi.dll";
    else if (hModule == kHModWinmm) targetDll = "winmm.dll";
    else if (hModule == kHModBcrypt) targetDll = "bcrypt.dll";
    else if (hModule == kHModCryptbase) targetDll = "cryptbase.dll";
    else if (hModule == kHModShlwapi) targetDll = "shlwapi.dll";
    else if (hModule == kHModXInput) targetDll = "xinput1_3.dll";

    void* sym = resolveNativeSymbol(targetDll, lpProcName);
    if (sym != nullptr) return sym;

    // 4. Si no se resolvió en la biblioteca indicada, buscar en kernel32 o DLLs locales
    if (targetDll != "kernel32.dll") {
        sym = resolveNativeSymbol("kernel32.dll", lpProcName);
        if (sym != nullptr) return sym;
    }
    // Seguridad (SID/ACL/descriptores): en Windows moderno vive en kernelbase y se pide a menudo
    // por api-ms-win-security-* (que aquí es el handle de kernel32).
    if (targetDll != "advapi32.dll") {
        sym = resolveNativeSymbol("advapi32.dll", lpProcName);
        if (sym != nullptr) return sym;
    }
    if (s_activeImage != nullptr) {
        for (const auto& kv : s_activeImage->localDlls) {
            if (kv.second.peInfo.findExportRva(lpProcName) > 0) {
                if (void* exp = resolvePeExport(kv.second.base, kv.second.peInfo, lpProcName, 0)) return exp;
            }
        }
    }

    // 5. Nombre desconocido: NULL + ERROR_PROC_NOT_FOUND, como Windows. Un stub aquí engaña a
    // quien sondea APIs opcionales: Mono P/Invoke prueba primero "GetFileAttributesExWW" /
    // "FindFirstFileExWW" y, con un stub, File.Exists devolvía true para cualquier ruta (Hollow
    // Knight veía Steam+GOG+GameCore "empaquetados" y llamaba a Application.Quit); Unity también
    // sondea UnityPluginLoad en cada DLL. Las importaciones de la IAT siguen usando stubs.
    // Una vez por nombre base (Mono prueba muchas variantes decoradas "_Nombre@N").
    static std::mutex s_missMutex;
    static std::unordered_set<std::string> s_missSeen;
    std::string missKey = targetDll + "!" + lpProcName;
    if (const size_t at = missKey.find('@'); at != std::string::npos) missKey.resize(at);
    if (const size_t bang = missKey.find("!_"); bang != std::string::npos) missKey.erase(bang + 1, 1);
    bool firstMiss = false;
    {
        std::lock_guard<std::mutex> lock(s_missMutex);
        firstMiss = s_missSeen.size() < 2000 && s_missSeen.insert(missKey).second;
    }
    if (firstMiss) {
        LOGW("GetProcAddress(%s!%s) -> NULL (no implementado)", targetDll.c_str(), lpProcName);
    }
    win_SetLastError(127);
    return nullptr;
}

void* NativeDllInterceptor::win_LoadLibraryW(const wchar_t* lpLibFileName) {
    s_kernel32Calls.fetch_add(1);
    if (lpLibFileName == nullptr) return nullptr;
    return resolveModuleHandle(wcharToUtf8(lpLibFileName));
}

void* NativeDllInterceptor::win_LoadLibraryA(const char* lpLibFileName) {
    s_kernel32Calls.fetch_add(1);
    if (lpLibFileName == nullptr) return nullptr;
    return resolveModuleHandle(lpLibFileName);
}

void NativeDllInterceptor::win_GetSystemInfo(void* lpSystemInfo) {
    s_kernel32Calls.fetch_add(1);
    if (lpSystemInfo == nullptr) return;
    struct SystemInfoLayout {
        uint16_t wProcessorArchitecture;
        uint16_t wReserved;
        uint32_t dwPageSize;
        void* lpMinimumApplicationAddress;
        void* lpMaximumApplicationAddress;
        uintptr_t dwActiveProcessorMask;
        uint32_t dwNumberOfProcessors;
        uint32_t dwProcessorType;
        uint32_t dwAllocationGranularity;
        uint16_t wProcessorLevel;
        uint16_t wProcessorRevision;
    };
    if (s_activeImage && s_activeImage->is32Bit) {
        // SYSTEM_INFO de 32 bits (36 bytes, punteros de 4): el de x64 (48) desbordaba el del invitado.
        struct SystemInfo32 {
            uint16_t wProcessorArchitecture, wReserved;
            uint32_t dwPageSize, lpMinimumApplicationAddress, lpMaximumApplicationAddress, dwActiveProcessorMask;
            uint32_t dwNumberOfProcessors, dwProcessorType, dwAllocationGranularity;
            uint16_t wProcessorLevel, wProcessorRevision;
        };
        static_assert(sizeof(SystemInfo32) == 36, "SYSTEM_INFO de 32 bits");
        SystemInfo32 s{};
        s.wProcessorArchitecture = 0;  // PROCESSOR_ARCHITECTURE_INTEL
        s.dwPageSize = 4096;
        s.lpMinimumApplicationAddress = 0x10000;
        // Proceso de 32 bits en Windows x64 con LARGE_ADDRESS_AWARE: 4 GB; si no, 2 GB.
        s.lpMaximumApplicationAddress = s_activeImage->peInfo.largeAddressAware ? 0xFFFEFFFFu : 0x7FFEFFFFu;
        s.dwActiveProcessorMask = 0xFF;
        s.dwNumberOfProcessors = 8;
        s.dwProcessorType = 586;  // PROCESSOR_INTEL_PENTIUM
        s.dwAllocationGranularity = 65536;
        s.wProcessorLevel = 6;
        std::memcpy(lpSystemInfo, &s, sizeof(s));
        return;
    }
    auto* si = reinterpret_cast<SystemInfoLayout*>(lpSystemInfo);
    si->wProcessorArchitecture = (s_activeImage && s_activeImage->is32Bit) ? 0 : 9;
    si->wReserved = 0;
    si->dwPageSize = 4096;
    si->lpMinimumApplicationAddress = reinterpret_cast<void*>(0x10000);
    si->lpMaximumApplicationAddress = reinterpret_cast<void*>(0x7FFFFFFEFFFFULL);
    si->dwActiveProcessorMask = 0xFF; // 8 núcleos Snapdragon 8 Elite
    si->dwNumberOfProcessors = 8;
    si->dwProcessorType = 8664;
    si->dwAllocationGranularity = 65536;
    si->wProcessorLevel = 6;
    si->wProcessorRevision = 0;
}

uint64_t NativeDllInterceptor::win_GetTickCount64() {
    s_kernel32Calls.fetch_add(1);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000ULL + static_cast<uint64_t>(ts.tv_nsec / 1000000ULL);
}

void* NativeDllInterceptor::win_HeapAlloc(void* hHeap, uint32_t dwFlags, size_t dwBytes) {
    s_kernel32Calls.fetch_add(1);
    (void)dwFlags;
    // Sin log por llamada: Mono/Unity hacen decenas de miles por segundo.
    void* p = safeHeapAlloc(dwBytes);
    // Montón ejecutable (HeapCreate con HEAP_CREATE_ENABLE_EXECUTE; el CLR de .NET Framework pone ahí
    // sus stubs): sus bloques se registran como código del invitado, por páginas.
    if (p != nullptr && reinterpret_cast<uintptr_t>(hHeap) == 0xCAFE0003u) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{0xFFF};
        const uintptr_t end = (reinterpret_cast<uintptr_t>(p) + dwBytes + 0xFFF) & ~uintptr_t{0xFFF};
        OryonEmitterBackend::addExecutableRangeAll(start, end - start);
    }
    return p;
}

int32_t NativeDllInterceptor::win_HeapFree(void*, uint32_t, void* lpMem) {
    s_kernel32Calls.fetch_add(1);
    safeHeapFree(lpMem);
    return 1;
}

namespace {
struct InitOnceState {
    bool completed = false;
    void* context = nullptr;
};
std::mutex s_initOnceMutex;
std::unordered_map<uintptr_t, InitOnceState> s_initOnceStates;

constexpr size_t kMaxWinTlsSlots = 1088;
std::mutex s_tlsAllocMutex;
std::bitset<kMaxWinTlsSlots> s_tlsAllocatedBits;
thread_local void* t_winTlsValues[kMaxWinTlsSlots] = {nullptr};
// Generación de cada índice: TlsFree la incrementa. En Windows TlsFree pone el valor a 0 en TODOS los hilos;
// aquí cada hilo recuerda con qué generación guardó su valor y lo ignora si el índice se liberó después (el
// índice reutilizado devolvía el objeto viejo de otro módulo: PhysX en DmC al cargar un nivel).
std::atomic<uint32_t> s_tlsGen[kMaxWinTlsSlots];
thread_local uint32_t t_tlsGen[kMaxWinTlsSlots] = {0};
thread_local void* t_guestTeb = nullptr;
thread_local bool t_isGuest64 = true;
thread_local void* t_unityTempAllocBuffer = nullptr;

// Provisión proactiva para Unity ThreadTempAllocators (típicamente TLS index 1)
void ensureUnityTempAllocatorProvisioned(void* lpTlsValue, uint32_t dwTlsIndex, bool isGuest64) {
    if (lpTlsValue == nullptr || !isGuest64) return;
    const uintptr_t val = reinterpret_cast<uintptr_t>(lpTlsValue);
    if (val < 0x10000u) return; // No es un puntero válido (p.ej. código de error o entero)
    if (dwTlsIndex != 1) return; // Solo Unity ThreadTempAllocators en TLS slot 1
    // Solo en juegos Unity: en cualquier otro (CoreCLR de Stardew Valley) el slot 1 guarda otra
    // estructura y escribir en +0x100..+0x130 la corrompía (objetos .NET con datos basura).
    static const bool s_isUnity = [] {
        if (s_activeImage == nullptr) return false;
        for (const auto& kv : s_activeImage->localDlls) {
            if (toLower(kv.first).find("unityplayer.dll") != std::string::npos) return true;
        }
        return false;
    }();
    if (!s_isUnity) return;

    auto* obj = static_cast<uint8_t*>(lpTlsValue);
    void* currentBase = *reinterpret_cast<void**>(obj + 0x108);
    uint64_t currentReserve = *reinterpret_cast<uint64_t*>(obj + 0x118);
    if (currentBase == nullptr || currentReserve == 0) {
        constexpr size_t kTempBufferSize = 32 * 1024 * 1024; // 32 MB arena
        if (t_unityTempAllocBuffer == nullptr) {
            t_unityTempAllocBuffer = mmap(nullptr, kTempBufferSize, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
        if (t_unityTempAllocBuffer != MAP_FAILED && t_unityTempAllocBuffer != nullptr) {
            *reinterpret_cast<void**>(obj + 0x108) = t_unityTempAllocBuffer;
            *reinterpret_cast<uint64_t*>(obj + 0x110) = kTempBufferSize;
            *reinterpret_cast<uint64_t*>(obj + 0x118) = kTempBufferSize;
            *reinterpret_cast<void**>(obj + 0x100) = nullptr;
            *reinterpret_cast<uint64_t*>(obj + 0x130) = 0;
            LOGI("NativeDllInterceptor: Provisionado buffer temporal de %zu MB en %p para Unity ThreadTempAllocators (TLS %u)",
                 kTempBufferSize / (1024 * 1024), t_unityTempAllocBuffer, dwTlsIndex);
        } else {
            LOGE("NativeDllInterceptor: Fallo al reservar buffer temporal para ThreadTempAllocators: %s",
                 std::strerror(errno));
        }
    }
}
}  // namespace

void NativeDllInterceptor::setGuestTeb(void* teb, bool is64Bit) noexcept {
    t_guestTeb = teb;
    t_isGuest64 = is64Bit;
    if (teb != nullptr) {
        auto* t = static_cast<uint8_t*>(teb);
        const uint64_t pid = static_cast<uint64_t>(getpid());
        const uint64_t tid = static_cast<uint64_t>(gettid());
        if (is64Bit) {
            std::memcpy(t + 0x40, &pid, 8);
            std::memcpy(t + 0x48, &tid, 8);
        } else {
            const uint32_t pid32 = static_cast<uint32_t>(pid);
            const uint32_t tid32 = static_cast<uint32_t>(tid);
            std::memcpy(t + 0x20, &pid32, 4);
            std::memcpy(t + 0x24, &tid32, 4);
        }
    }
}

namespace {
// TlsExpansionSlots de un TEB de 32 bits (+0xF94): puntero de 4 bytes a 1024 entradas de 4 bytes en memoria
// baja. Antes se guardaba un puntero de 8 bytes de calloc (pisaba +0xF98 y no cabía en 32 bits).
uint32_t* tlsExp32(uint8_t* teb, bool create) {
    uint32_t p = 0;
    std::memcpy(&p, teb + 0x0F94, 4);
    if (p == 0 && create) {
        void* m = lowmem::mmapBelow4G(4096);
        if (m == MAP_FAILED) return nullptr;
        p = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(m));
        std::memcpy(teb + 0x0F94, &p, 4);
    }
    return reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(p));
}
}  // namespace

uint32_t NativeDllInterceptor::win_TlsAlloc() {
    s_kernel32Calls.fetch_add(1);
    std::lock_guard<std::mutex> lock(s_tlsAllocMutex);
    for (size_t i = 0; i < kMaxWinTlsSlots; ++i) {
        if (!s_tlsAllocatedBits.test(i)) {
            s_tlsAllocatedBits.set(i);
            LOGI("win_TlsAlloc() -> index %zu", i);
            return static_cast<uint32_t>(i);
        }
    }
    LOGE("win_TlsAlloc() out of TLS slots (max %zu)", kMaxWinTlsSlots);
    return 0xFFFFFFFF;
}

int32_t NativeDllInterceptor::win_InitOnceBeginInitialize(void* lpInitOnce, uint32_t dwFlags,
                                                          int32_t* fPending, void** lpContext) {
    s_kernel32Calls.fetch_add(1);
    if (lpInitOnce == nullptr || fPending == nullptr) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }

    constexpr uint32_t kInitOnceCheckOnly = 0x1u;
    std::lock_guard<std::mutex> lock(s_initOnceMutex);
    InitOnceState& state = s_initOnceStates[reinterpret_cast<uintptr_t>(lpInitOnce)];

    if ((dwFlags & kInitOnceCheckOnly) != 0u) {
        *fPending = state.completed ? 0 : 1;
        lowmem::storePtr(lpContext, state.context);
        return state.completed ? 1 : 0;
    }
    if (!state.completed) {
        *fPending = 1;
        lowmem::storePtr(lpContext, uint64_t{0});
        return 1;
    }
    *fPending = 0;
    lowmem::storePtr(lpContext, state.context);
    return 1;
}

int32_t NativeDllInterceptor::win_InitOnceComplete(void* lpInitOnce, uint32_t dwFlags, void* lpContext) {
    s_kernel32Calls.fetch_add(1);
    if (lpInitOnce == nullptr) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }

    constexpr uint32_t kInitOnceInitFailed = 0x4u;
    std::lock_guard<std::mutex> lock(s_initOnceMutex);
    InitOnceState& state = s_initOnceStates[reinterpret_cast<uintptr_t>(lpInitOnce)];
    if (!state.completed) {
        state.completed = true;
        state.context = (dwFlags & kInitOnceInitFailed) != 0u ? nullptr : lpContext;
    }
    return 1;
}

int32_t NativeDllInterceptor::win_TlsFree(uint32_t dwTlsIndex) {
    s_kernel32Calls.fetch_add(1);
    if (dwTlsIndex < kMaxWinTlsSlots) {
        std::lock_guard<std::mutex> lock(s_tlsAllocMutex);
        s_tlsAllocatedBits.reset(dwTlsIndex);
        s_tlsGen[dwTlsIndex].fetch_add(1);
        t_winTlsValues[dwTlsIndex] = nullptr;
        if (t_guestTeb != nullptr) {
            auto* teb = static_cast<uint8_t*>(t_guestTeb);
            if (t_isGuest64) {
                if (dwTlsIndex < 64) {
                    *reinterpret_cast<void**>(teb + 0x1480 + dwTlsIndex * 8) = nullptr;
                } else {
                    void** exp = *reinterpret_cast<void***>(teb + 0x1780);
                    if (exp != nullptr) exp[dwTlsIndex - 64] = nullptr;
                }
            } else {
                if (dwTlsIndex < 64) {
                    *reinterpret_cast<uint32_t*>(teb + 0x0E10 + dwTlsIndex * 4) = 0;
                } else {
                    uint32_t* exp32 = tlsExp32(teb, false);
                    if (exp32 != nullptr) exp32[dwTlsIndex - 64] = 0;
                }
            }
        }
        LOGI("win_TlsFree(%u)", dwTlsIndex);
        return 1;
    }
    return 0;
}

void* NativeDllInterceptor::win_TlsGetValue(uint32_t dwTlsIndex) {
    s_kernel32Calls.fetch_add(1);
    if (dwTlsIndex >= kMaxWinTlsSlots) {
        return nullptr;
    }
    void* result = nullptr;
    if (t_tlsGen[dwTlsIndex] != s_tlsGen[dwTlsIndex].load(std::memory_order_relaxed)) {
        win_TlsSetValue(dwTlsIndex, nullptr);  // liberado y reasignado desde que este hilo lo escribió
    }
    if (t_guestTeb != nullptr) {
        auto* teb = static_cast<uint8_t*>(t_guestTeb);
        if (t_isGuest64) {
            if (dwTlsIndex < 64) {
                result = *reinterpret_cast<void**>(teb + 0x1480 + dwTlsIndex * 8);
            } else {
                void** exp = *reinterpret_cast<void***>(teb + 0x1780);
                if (exp != nullptr) {
                    result = exp[dwTlsIndex - 64];
                }
            }
        } else {
            if (dwTlsIndex < 64) {
                uint32_t v32 = *reinterpret_cast<uint32_t*>(teb + 0x0E10 + dwTlsIndex * 4);
                if (v32 != 0) result = reinterpret_cast<void*>(static_cast<uintptr_t>(v32));
            } else {
                uint32_t* exp32 = tlsExp32(teb, false);
                if (exp32 != nullptr && exp32[dwTlsIndex - 64] != 0) {
                    result = reinterpret_cast<void*>(static_cast<uintptr_t>(exp32[dwTlsIndex - 64]));
                }
            }
        }
    }
    if (result == nullptr) {
        result = t_winTlsValues[dwTlsIndex];
    }
    ensureUnityTempAllocatorProvisioned(result, dwTlsIndex, t_isGuest64);
    return result;
}

int32_t NativeDllInterceptor::win_TlsSetValue(uint32_t dwTlsIndex, void* lpTlsValue) {
    s_kernel32Calls.fetch_add(1);
    if (dwTlsIndex >= kMaxWinTlsSlots) {
        return 0;
    }
    ensureUnityTempAllocatorProvisioned(lpTlsValue, dwTlsIndex, t_isGuest64);
    t_winTlsValues[dwTlsIndex] = lpTlsValue;
    t_tlsGen[dwTlsIndex] = s_tlsGen[dwTlsIndex].load(std::memory_order_relaxed);
    if (t_guestTeb != nullptr) {
        auto* teb = static_cast<uint8_t*>(t_guestTeb);
        if (t_isGuest64) {
            if (dwTlsIndex < 64) {
                *reinterpret_cast<void**>(teb + 0x1480 + dwTlsIndex * 8) = lpTlsValue;
            } else {
                void** exp = *reinterpret_cast<void***>(teb + 0x1780);
                if (exp == nullptr) {
                    exp = static_cast<void**>(std::calloc(1024, sizeof(void*)));
                    *reinterpret_cast<void***>(teb + 0x1780) = exp;
                }
                if (exp != nullptr) {
                    exp[dwTlsIndex - 64] = lpTlsValue;
                }
            }
        } else {
            const uint32_t val32 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lpTlsValue));
            if (dwTlsIndex < 64) {
                *reinterpret_cast<uint32_t*>(teb + 0x0E10 + dwTlsIndex * 4) = val32;
            } else {
                uint32_t* exp32 = tlsExp32(teb, true);
                if (exp32 != nullptr) {
                    exp32[dwTlsIndex - 64] = val32;
                }
            }
        }
    }
    static std::atomic<uint32_t> s_tlsSetLogs{0};
    if (s_tlsSetLogs.fetch_add(1, std::memory_order_relaxed) < 200) {
        LOGI("win_TlsSetValue(%u, %p)", dwTlsIndex, lpTlsValue);
    }
    return 1;
}

static void* createMainWindow(const char* api, uint32_t dwExStyle, uint32_t dwStyle,
                              int32_t x, int32_t y, int32_t nWidth, int32_t nHeight,
                              void* hWndParent) {
    s_otherCalls.fetch_add(1);
    const int32_t w = win32::normalizeWindowExtent(nWidth, win32::screenWidth());
    const int32_t h = win32::normalizeWindowExtent(nHeight, win32::screenHeight());
    LOGI("%s(exStyle=0x%x, style=0x%x, pos=%d,%d, size=%dx%d (pedido %dx%d), parent=%p) -> 0xCAFE1000",
         api, dwExStyle, dwStyle, win32::normalizeWindowOrigin(x), win32::normalizeWindowOrigin(y),
         w, h, nWidth, nHeight, hWndParent);
    return reinterpret_cast<void*>(0xCAFE1000);
}

// --- Cola de mensajes mínima de la ventana principal ---------------------------------------------
// Solo lleva lo que los juegos SDL2 (MonoGame/FNA) necesitan para saberse activos: al primer ShowWindow
// se encolan WM_ACTIVATEAPP, WM_ACTIVATE y WM_SETFOCUS, y PeekMessageW/DispatchMessageW los entregan al
// WndProc del invitado (el de su clase o el que puso con SetWindowLongPtrW(GWLP_WNDPROC)). Sin esto SDL
// nunca da el foco a la ventana y MonoGame deja Game.IsActive en false: Stardew ignora el mando.
namespace wndq {
struct Msg { uint32_t message; uint64_t wParam; int64_t lParam; };
std::mutex s_mtx;
std::deque<Msg> s_queue;
std::unordered_map<std::u16string, uint64_t> s_classProcs;  // nombre de clase (minúsculas) -> WndProc
std::atomic<uint64_t> s_wndProc{0};
std::atomic<int64_t> s_userData{0};
std::atomic<bool> s_activated{false};
std::atomic<int> s_activatedGen{-1};  // generación de props para la que ya se encoló la activación

std::u16string classKey(const void* name) {
    const auto v = reinterpret_cast<uintptr_t>(name);
    if (v == 0) return {};
    if (v <= 0xFFFF) return u"#" + std::u16string(1, static_cast<char16_t>(v));  // átomo
    std::u16string s;
    for (const auto* p = static_cast<const char16_t*>(name); *p != 0 && s.size() < 256; ++p)
        s.push_back(*p < 0x80 ? static_cast<char16_t>(std::tolower(*p)) : *p);
    return s;
}

void registerClass(const void* wndClass) {
    if (wndClass == nullptr) return;
    // Los desplazamientos son los de x64. En 32 bits (punteros de 4 bytes) se leía basura como puntero
    // al nombre (SIGSEGV en DmC); la cola de mensajes con WndProc aún es solo para invitados de 64 bits.
    if (s_activeImage != nullptr && s_activeImage->is32Bit) return;
    const auto* b = static_cast<const uint8_t*>(wndClass);
    uint64_t proc = 0;
    const void* name = nullptr;
    std::memcpy(&proc, b + 8, 8);   // lpfnWndProc (mismo desplazamiento en WNDCLASSW y WNDCLASSEXW)
    std::memcpy(&name, b + 64, 8);  // lpszClassName
    const std::u16string key = classKey(name);
    if (proc == 0 || key.empty()) return;
    std::lock_guard<std::mutex> lk(s_mtx);
    s_classProcs[key] = proc;
}

void bindWindow(const void* className) {
    const std::u16string key = classKey(className);
    std::lock_guard<std::mutex> lk(s_mtx);
    const auto it = s_classProcs.find(key);
    if (it != s_classProcs.end()) s_wndProc.store(it->second);
}

void activate() {
    const int gen = s_wndPropGen.load();
    if (s_activatedGen.exchange(gen) == gen) return;
    std::lock_guard<std::mutex> lk(s_mtx);
    s_queue.push_back({0x001C, 1, 0});  // WM_ACTIVATEAPP(TRUE)
    s_queue.push_back({0x0006, 1, 0});  // WM_ACTIVATE(WA_ACTIVE)
    s_queue.push_back({0x0007, 0, 0});  // WM_SETFOCUS
    s_activated.store(true);
}
}  // namespace wndq

void* NativeDllInterceptor::win_CreateWindowExW(uint32_t dwExStyle, const wchar_t* lpClassName, const wchar_t*,
                                               uint32_t dwStyle, int32_t x, int32_t y, int32_t nWidth, int32_t nHeight,
                                               void* hWndParent, void*, void*, void*) {
    // Todas las ventanas son 0xCAFE1000: solo la principal (sin padre, ni HWND_MESSAGE, ni WS_DISABLED)
    // fija el WndProc. Las auxiliares de SDL (IME 10x10 deshabilitada, ventana de mensajes) no.
    if (hWndParent == nullptr && (dwStyle & 0x08000000u) == 0) wndq::bindWindow(lpClassName);
    return createMainWindow("CreateWindowExW", dwExStyle, dwStyle, x, y, nWidth, nHeight, hWndParent);
}

void* NativeDllInterceptor::win_CreateWindowExA(uint32_t dwExStyle, const char*, const char*,
                                               uint32_t dwStyle, int32_t x, int32_t y, int32_t nWidth, int32_t nHeight,
                                               void* hWndParent, void*, void*, void*) {
    return createMainWindow("CreateWindowExA", dwExStyle, dwStyle, x, y, nWidth, nHeight, hWndParent);
}

int32_t NativeDllInterceptor::win_ShowWindow(void*, int32_t nCmdShow) {
    s_otherCalls.fetch_add(1);
    // Solo con un WndProc conocido: los juegos sin clase registrada siguen como antes.
    // Mostrar la ventana (MonoGame lo hace al entrar en su bucle) vuelve a activarla, como en Windows.
    if (nCmdShow != 0 && wndq::s_wndProc.load() != 0 && s_wndPropSet.load()) {
        s_wndPropGen.fetch_add(1);
        wndq::activate();
    }
    return 1;
}

int32_t NativeDllInterceptor::win_PeekMessageW(void* lpMsg, void*, uint32_t wMsgFilterMin, uint32_t wMsgFilterMax,
                                               uint32_t wRemoveMsg) {
    s_otherCalls.fetch_add(1);
    if (lpMsg == nullptr) return 0;
    // La ventana principal ya existe (WndProc conocido) y el juego bombea mensajes: está "en primer plano".
    if (wndq::s_wndProc.load() != 0 && s_wndPropSet.load()) wndq::activate();
    wndq::Msg m{};
    {
        std::lock_guard<std::mutex> lk(wndq::s_mtx);
        if (wndq::s_queue.empty()) return 0;
        m = wndq::s_queue.front();
        if ((wMsgFilterMin != 0 || wMsgFilterMax != 0) && (m.message < wMsgFilterMin || m.message > wMsgFilterMax))
            return 0;
        if (wRemoveMsg & 1u) wndq::s_queue.pop_front();  // PM_REMOVE
    }
    auto* b = static_cast<uint8_t*>(lpMsg);
    const uint64_t hwnd = 0xCAFE1000;
    std::memset(b, 0, 48);  // MSG x64: hwnd, message, wParam, lParam, time, pt
    std::memcpy(b, &hwnd, 8);
    std::memcpy(b + 8, &m.message, 4);
    std::memcpy(b + 16, &m.wParam, 8);
    std::memcpy(b + 24, &m.lParam, 8);
    const uint32_t t = static_cast<uint32_t>(win_GetTickCount());
    std::memcpy(b + 32, &t, 4);
    return 1;
}

int32_t NativeDllInterceptor::win_DispatchMessageW(const void* lpMsg) {
    s_otherCalls.fetch_add(1);
    const uint64_t proc = wndq::s_wndProc.load();
    if (lpMsg == nullptr || proc == 0) return 0;
    const auto* b = static_cast<const uint8_t*>(lpMsg);
    uint64_t args[4] = {};
    uint32_t message = 0;
    std::memcpy(&args[0], b, 8);
    std::memcpy(&message, b + 8, 4);
    args[1] = message;
    std::memcpy(&args[2], b + 16, 8);
    std::memcpy(&args[3], b + 24, 8);
    // Solo los primeros: SDL/FNA despachan miles de mensajes por segundo (log de 400 MB en Celeste).
    static std::atomic<uint32_t> s_dispatchLogs{0};
    if (s_dispatchLogs.fetch_add(1, std::memory_order_relaxed) < 50) {
        LOGI("DispatchMessageW: msg=0x%x -> WndProc %p", message, reinterpret_cast<void*>(proc));
    }
    return static_cast<int32_t>(sehimpl::callGuest(proc, args, 4));
}

void* NativeDllInterceptor::win_GetForegroundWindow() {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0xCAFE1000);
}

void* NativeDllInterceptor::win_GetActiveWindow() {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0xCAFE1000);
}

void* NativeDllInterceptor::win_GetDesktopWindow() {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10000);
}

void* NativeDllInterceptor::win_GetShellWindow() {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10000);
}

int32_t NativeDllInterceptor::win_SetForegroundWindow(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_SetActiveWindow(void*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(1);
}

void* NativeDllInterceptor::win_GetFocus() {
    s_otherCalls.fetch_add(1);
    // 32 bits: la cola de mensajes propia (msg32) no pasa por wndq; la ventana del juego siempre tiene el foco
    // (Unreal 3 solo lee teclado/mando si GetFocus() es su ventana).
    if (lowmem::enabled()) return reinterpret_cast<void*>(0xCAFE1000);
    return reinterpret_cast<void*>(wndq::s_activated.load() ? 0xCAFE1000 : 1);
}

void* NativeDllInterceptor::win_SetFocus(void*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(1);
}

int32_t NativeDllInterceptor::win_GetClientRect(void*, void* lpRect) {
    s_otherCalls.fetch_add(1);
    win32::fillScreenRect(lpRect);
    return 1;
}

int32_t NativeDllInterceptor::win_GetWindowRect(void*, void* lpRect) {
    s_otherCalls.fetch_add(1);
    win32::fillWindowRect(lpRect);
    return 1;
}

int32_t NativeDllInterceptor::win_AdjustWindowRect(void*, uint32_t, int32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_AdjustWindowRectEx(void*, uint32_t, int32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_GetSystemMetrics(int32_t nIndex) {
    s_otherCalls.fetch_add(1);
    return win32::systemMetric(nIndex);
}

int32_t NativeDllInterceptor::win_SetWindowPos(void*, void*, int32_t, int32_t, int32_t, int32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetWindowTextW(void*, const wchar_t*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetWindowTextA(void*, const char*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_GetDC(void*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(win32::kPrimaryDeviceContext);
}

int32_t NativeDllInterceptor::win_ReleaseDC(void*, void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_ValidateRect(void*, const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_InvalidateRect(void*, const void*, int32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_UpdateWindow(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_SetCursor(void* hCursor) {
    s_otherCalls.fetch_add(1);
    return hCursor;
}

int32_t NativeDllInterceptor::win_ShowCursor(int32_t bShow) {
    s_otherCalls.fetch_add(1);
    // 32 bits: contador real (Unreal 3 hace while (ShowCursor(FALSE) >= 0) para ocultarlo).
    if (s_activeImage != nullptr && s_activeImage->is32Bit) return bShow ? ++threadmsg::g_cursorCount : --threadmsg::g_cursorCount;
    return 0;
}

int64_t NativeDllInterceptor::win_DefWindowProcW(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int64_t NativeDllInterceptor::win_DefWindowProcA(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_PostMessageW(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void NativeDllInterceptor::win_PostQuitMessage(int32_t) {
    s_otherCalls.fetch_add(1);
}

uint32_t NativeDllInterceptor::win_RegisterWindowMessageW(const void*) {
    s_otherCalls.fetch_add(1);
    return 0xC001;
}

uint32_t NativeDllInterceptor::win_RegisterWindowMessageA(const char*) {
    s_otherCalls.fetch_add(1);
    return 0xC001;
}

void* NativeDllInterceptor::win_MonitorFromWindow(void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(win32::kPrimaryMonitor);
}

void* NativeDllInterceptor::win_MonitorFromPoint(int64_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(win32::kPrimaryMonitor);
}

void* NativeDllInterceptor::win_MonitorFromRect(const void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(win32::kPrimaryMonitor);
}

int32_t NativeDllInterceptor::win_GetMonitorInfoW(void*, void* lpmi) {
    s_otherCalls.fetch_add(1);
    win32::fillMonitorInfo(lpmi);
    return 1;
}

int32_t NativeDllInterceptor::win_GetMonitorInfoA(void* hMonitor, void* lpmi) {
    return win_GetMonitorInfoW(hMonitor, lpmi);
}

int32_t NativeDllInterceptor::win_EnumDisplayMonitors(void* hdc, const void* lprcClip, void* lpfnEnum,
                                                     int64_t dwData) {
    s_otherCalls.fetch_add(1);
    if (lpfnEnum == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    OryonEmitterBackend* backend = t_dispatchBackend;
    if (backend == nullptr) {
        LOGE("EnumDisplayMonitors: sin backend invitado para invocar el callback %p", lpfnEnum);
        win_SetLastError(87);
        return 0;
    }
    // MONITORENUMPROC(HMONITOR, HDC, LPRECT, LPARAM): un único monitor primario 2688x1216.
    static int32_t s_monitorRect[4] = {0, 0, win32::screenWidth(), win32::screenHeight()};
    const uint64_t cbArgs[4] = {
        static_cast<uint64_t>(win32::kPrimaryMonitor),
        reinterpret_cast<uint64_t>(hdc),
        reinterpret_cast<uint64_t>(s_monitorRect),
        static_cast<uint64_t>(dwData),
    };
    const int64_t cbRet = backend->callGuestNested(reinterpret_cast<uintptr_t>(lpfnEnum), cbArgs, 4);
    LOGI("EnumDisplayMonitors(hdc=%p, clip=%p, cb=%p) -> callback(HMONITOR=0x%llx) = %lld",
         hdc, lprcClip, lpfnEnum, static_cast<unsigned long long>(win32::kPrimaryMonitor),
         static_cast<long long>(cbRet));
    return cbRet < 0 ? 0 : 1;
}

int32_t NativeDllInterceptor::win_EnumDisplaySettingsW(const void*, uint32_t iModeNum, void* lpDevMode) {
    s_otherCalls.fetch_add(1);
    // Lista de resoluciones del "monitor" (win32::displayModes): las opciones de vídeo del juego
    // ofrecen la nativa y resoluciones menores.
    const bool ok = win32::fillDevMode(lpDevMode, iModeNum);
    if (!ok) LOGI("Win32: EnumDisplaySettings terminó la lista en el modo %u", iModeNum);
    return ok ? 1 : 0;
}

int32_t NativeDllInterceptor::win_EnumDisplaySettingsA(const void* dev, uint32_t mode, void* dm) {
    return win_EnumDisplaySettingsW(dev, mode, dm);
}

int32_t NativeDllInterceptor::win_GetWindowLongW(void*, int32_t nIndex) {
    s_otherCalls.fetch_add(1);
    if (nIndex == -16) return 0x10000000 | 0x00C00000; // WS_VISIBLE | WS_CAPTION
    return 0;
}

int32_t NativeDllInterceptor::win_GetWindowLongA(void* hWnd, int32_t nIndex) {
    return win_GetWindowLongW(hWnd, nIndex);
}

int64_t NativeDllInterceptor::win_GetWindowLongPtrW(void*, int32_t nIndex) {
    s_otherCalls.fetch_add(1);
    if (nIndex == -16) return 0x10000000 | 0x00C00000;
    if (nIndex == -4) return static_cast<int64_t>(wndq::s_wndProc.load());   // GWLP_WNDPROC
    if (nIndex == -21) return wndq::s_userData.load();                       // GWLP_USERDATA
    return 0;
}

int64_t NativeDllInterceptor::win_GetWindowLongPtrA(void* hWnd, int32_t nIndex) {
    return win_GetWindowLongPtrW(hWnd, nIndex);
}

int32_t NativeDllInterceptor::win_SetWindowLongW(void*, int32_t, int32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_SetWindowLongA(void*, int32_t, int32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int64_t NativeDllInterceptor::win_SetWindowLongPtrW(void*, int32_t nIndex, int64_t value) {
    s_otherCalls.fetch_add(1);
    if (nIndex == -4) return static_cast<int64_t>(wndq::s_wndProc.exchange(static_cast<uint64_t>(value)));
    if (nIndex == -21) return wndq::s_userData.exchange(value);
    return 0;
}

int64_t NativeDllInterceptor::win_SetWindowLongPtrA(void*, int32_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

uint16_t NativeDllInterceptor::win_RegisterClassW(const void* wc) {
    s_otherCalls.fetch_add(1);
    wndq::registerClass(wc);
    return 0x100;
}

uint16_t NativeDllInterceptor::win_RegisterClassExW(const void* wc) {
    s_otherCalls.fetch_add(1);
    wndq::registerClass(wc);
    return 0x100;
}

int32_t NativeDllInterceptor::win_UnregisterClassW(const void*, void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_DestroyWindow(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

uintptr_t NativeDllInterceptor::win_SetTimer(void*, uintptr_t nIDEvent, uint32_t, void*) {
    s_otherCalls.fetch_add(1);
    return nIDEvent != 0 ? nIDEvent : 1;
}

int32_t NativeDllInterceptor::win_KillTimer(void*, uintptr_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SystemParametersInfoW(uint32_t, uint32_t, void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// Cursor del sistema: SetCursorPos lo mueve y GetCursorPos lo devuelve. Stardew (MonoGame) maneja el
// cursor con el mando moviéndolo con Mouse.SetPosition -> SDL_WarpMouse -> SetCursorPos y lo vuelve a
// leer cada frame; con una posición fija el cursor no se movía y A pulsaba siempre en el mismo sitio.
namespace {
std::atomic<int64_t> s_cursorPos{(static_cast<int64_t>(640) << 32) | 360u};
}

int32_t NativeDllInterceptor::win_GetCursorPos(void* lpPoint) {
    s_otherCalls.fetch_add(1);
    if (lpPoint) {
        const int64_t v = s_cursorPos.load(std::memory_order_relaxed);
        auto* pt = static_cast<int32_t*>(lpPoint);
        pt[0] = static_cast<int32_t>(v >> 32);
        pt[1] = static_cast<int32_t>(static_cast<uint32_t>(v));
    }
    return 1;
}

int32_t NativeDllInterceptor::win_SetCursorPos(int32_t x, int32_t y) {
    s_otherCalls.fetch_add(1);
    x = std::clamp(x, 0, win32::screenWidth() - 1);
    y = std::clamp(y, 0, win32::screenHeight() - 1);
    s_cursorPos.store((static_cast<int64_t>(x) << 32) | static_cast<uint32_t>(y), std::memory_order_relaxed);
    return 1;
}

// Sin recorte del cursor: el rectángulo es la pantalla entera (SDL lo consulta en cada frame).
static int32_t hp_GetClipCursor(void* lpRect) {
    if (lpRect == nullptr) return 0;
    const int32_t r[4] = {0, 0, win32::screenWidth(), win32::screenHeight()};
    std::memcpy(lpRect, r, sizeof(r));
    return 1;
}

// El pool de hilos de .NET pregunta si un hilo tiene E/S pendiente antes de retirarlo: aquí nunca.
static int32_t hp_GetThreadIOPendingFlag(void*, int32_t* lpIOIsPending) {
    if (lpIOIsPending == nullptr) return 0;
    *lpIOIsPending = 0;
    return 1;
}

int32_t NativeDllInterceptor::win_ClipCursor(const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_ClientToScreen(void*, void* lpPoint) {
    s_otherCalls.fetch_add(1);
    win32::translatePoint(lpPoint, 0, 0);
    return 1;
}

int32_t NativeDllInterceptor::win_ScreenToClient(void*, void* lpPoint) {
    s_otherCalls.fetch_add(1);
    win32::translatePoint(lpPoint, 0, 0);
    return 1;
}

int32_t NativeDllInterceptor::win_AllowSetForegroundWindow(uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// -----------------------------------------------------------------------------
// Desktop & Window Station Emulation
// -----------------------------------------------------------------------------
static constexpr uintptr_t kHDeskDefault = 0xCAFE0030;  // distinto de kPrimaryMonitor
static constexpr uintptr_t kHWinsta0     = 0xCAFE0030;  // distinto de kPrimaryDeviceContext

void* NativeDllInterceptor::win_GetThreadDesktop(uint32_t dwThreadId) {
    s_otherCalls.fetch_add(1);
    LOGI("GetThreadDesktop(threadId=%u) -> %p", dwThreadId, reinterpret_cast<void*>(kHDeskDefault));
    return reinterpret_cast<void*>(kHDeskDefault);
}

void* NativeDllInterceptor::win_GetProcessWindowStation() {
    s_otherCalls.fetch_add(1);
    LOGI("GetProcessWindowStation() -> %p", reinterpret_cast<void*>(kHWinsta0));
    return reinterpret_cast<void*>(kHWinsta0);
}

int32_t NativeDllInterceptor::win_SetThreadDesktop(void* /*hDesktop*/) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetProcessWindowStation(void* /*hWinSta*/) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_GetUserObjectInformationA(void* hObj, int32_t nIndex, void* pvInfo, uint32_t nLength, uint32_t* lpnLengthNeeded) {
    s_otherCalls.fetch_add(1);
    LOGI("GetUserObjectInformationA(hObj=%p, nIndex=%d, pvInfo=%p, nLength=%u)",
         hObj, nIndex, pvInfo, nLength);

    if (nIndex == 1) { // UOI_FLAGS
        const uint32_t needed = 12; // sizeof(USEROBJECTFLAGS)
        if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = needed;
        if (pvInfo == nullptr || nLength < needed) {
            win_SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
            return 0;
        }
        auto* flags = static_cast<uint32_t*>(pvInfo);
        flags[0] = 0; // fInherit
        flags[1] = 0; // fReserved
        flags[2] = 1; // dwFlags = WSF_VISIBLE
        return 1;
    }

    if (nIndex == 2) { // UOI_NAME
        const char* name = "Default";
        if (hObj == reinterpret_cast<void*>(kHWinsta0)) {
            name = "WinSta0";
        }
        const uint32_t needed = static_cast<uint32_t>(std::strlen(name) + 1);
        if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = needed;
        if (pvInfo == nullptr || nLength < needed) {
            win_SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
            return 0;
        }
        std::memcpy(pvInfo, name, needed);
        return 1;
    }

    if (nIndex == 3) { // UOI_TYPE
        const char* typeName = (hObj == reinterpret_cast<void*>(kHWinsta0)) ? "WindowStation" : "Desktop";
        const uint32_t needed = static_cast<uint32_t>(std::strlen(typeName) + 1);
        if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = needed;
        if (pvInfo == nullptr || nLength < needed) {
            win_SetLastError(122);
            return 0;
        }
        std::memcpy(pvInfo, typeName, needed);
        return 1;
    }

    if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = 0;
    return 1;
}

int32_t NativeDllInterceptor::win_GetUserObjectInformationW(void* hObj, int32_t nIndex, void* pvInfo, uint32_t nLength, uint32_t* lpnLengthNeeded) {
    s_otherCalls.fetch_add(1);
    LOGI("GetUserObjectInformationW(hObj=%p, nIndex=%d, pvInfo=%p, nLength=%u)",
         hObj, nIndex, pvInfo, nLength);

    if (nIndex == 1) { // UOI_FLAGS
        const uint32_t needed = 12;
        if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = needed;
        if (pvInfo == nullptr || nLength < needed) {
            win_SetLastError(122);
            return 0;
        }
        auto* flags = static_cast<uint32_t*>(pvInfo);
        flags[0] = 0;
        flags[1] = 0;
        flags[2] = 1;
        return 1;
    }

    if (nIndex == 2 || nIndex == 3) { // UOI_NAME / UOI_TYPE (WCHAR de 16 bits)
        const bool winsta = hObj == reinterpret_cast<void*>(kHWinsta0);
        const std::u16string text = nIndex == 2 ? (winsta ? u"WinSta0" : u"Default")
                                                : (winsta ? u"WindowStation" : u"Desktop");
        const uint32_t needed = static_cast<uint32_t>((text.size() + 1) * sizeof(char16_t));
        if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = needed;
        if (pvInfo == nullptr || nLength < needed) {
            win_SetLastError(122);
            return 0;
        }
        std::memcpy(pvInfo, text.c_str(), needed);
        return 1;
    }

    if (lpnLengthNeeded != nullptr) *lpnLengthNeeded = 0;
    return 1;
}

// -----------------------------------------------------------------------------
// Message Wait & Pump
// -----------------------------------------------------------------------------
uint32_t NativeDllInterceptor::win_MsgWaitForMultipleObjects(uint32_t nCount, void* const* pHandles, int32_t bWaitAll, uint32_t dwMilliseconds, uint32_t /*dwWakeMask*/) {
    s_otherCalls.fetch_add(1);
    if (nCount > 0 && pHandles != nullptr) {
        return win_WaitForMultipleObjects(nCount, pHandles, bWaitAll, dwMilliseconds);
    }
    if (dwMilliseconds == 0) return 0; // WAIT_OBJECT_0
    if (dwMilliseconds != 0xFFFFFFFF) {
        usleep(std::min(dwMilliseconds, 5u) * 1000);
        return 0x102; // WAIT_TIMEOUT
    }
    return 0;
}

uint32_t NativeDllInterceptor::win_MsgWaitForMultipleObjectsEx(uint32_t nCount, void* const* pHandles, uint32_t dwMilliseconds, uint32_t dwWakeMask, uint32_t /*dwFlags*/) {
    return win_MsgWaitForMultipleObjects(nCount, pHandles, 0, dwMilliseconds, dwWakeMask);
}

int32_t NativeDllInterceptor::win_TranslateMessage(const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_GetMessageW(void* lpMsg, void*, uint32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    // 32 bits: espera real en la cola del hilo (antes devolvía 1 al instante: bucle al 100 % de CPU).
    if (s_activeImage != nullptr && s_activeImage->is32Bit) return threadmsg::get(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lpMsg)));
    return 1;
}

int32_t NativeDllInterceptor::win_GetMessageA(void*, void*, uint32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_PeekMessageA(void*, void*, uint32_t, uint32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_DispatchMessageA(const void*) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int64_t NativeDllInterceptor::win_SendMessageW(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int64_t NativeDllInterceptor::win_SendMessageA(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_PostMessageA(void*, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int64_t NativeDllInterceptor::win_SendMessageTimeoutA(void*, uint32_t, uint64_t, int64_t, uint32_t, uint32_t, uintptr_t* lpdwResult) {
    s_otherCalls.fetch_add(1);
    lowmem::storePtr(lpdwResult, uint64_t{0});  // DWORD_PTR
    return 1;
}

// -----------------------------------------------------------------------------
// Displays & Monitors
// -----------------------------------------------------------------------------
// lpDevice == NULL enumera adaptadores; lpDevice = "\\.\DISPLAY1" enumera sus monitores.
int32_t NativeDllInterceptor::win_EnumDisplayDevicesA(const char* lpDevice, uint32_t iDevNum, void* lpDisplayDevice, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (iDevNum > 0 || lpDisplayDevice == nullptr) return 0;
    if (lpDevice != nullptr && lpDevice[0] != 0) win32::fillMonitorDeviceA(lpDisplayDevice);
    else win32::fillDisplayDeviceA(lpDisplayDevice);
    return 1;
}

int32_t NativeDllInterceptor::win_EnumDisplayDevicesW(const wchar_t* lpDevice, uint32_t iDevNum, void* lpDisplayDevice, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (iDevNum > 0 || lpDisplayDevice == nullptr) return 0;
    const auto* dev16 = reinterpret_cast<const char16_t*>(lpDevice);
    if (dev16 != nullptr && dev16[0] != 0) win32::fillMonitorDeviceW(lpDisplayDevice);
    else win32::fillDisplayDeviceW(lpDisplayDevice);
    return 1;
}

int32_t NativeDllInterceptor::win_EnumWindows(void*, int64_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// -----------------------------------------------------------------------------
// Raw Input & Keyboard & Mouse
// -----------------------------------------------------------------------------
uint32_t NativeDllInterceptor::win_GetRawInputDeviceList(void* pRawInputDeviceList, uint32_t* puiNumDevices, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (pRawInputDeviceList == nullptr) {
        if (puiNumDevices != nullptr) *puiNumDevices = 0;
        return 0;
    }
    return 0;
}

int32_t NativeDllInterceptor::win_RegisterRawInputDevices(const void*, uint32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

uint32_t NativeDllInterceptor::win_GetRawInputDeviceInfoW(void*, uint32_t, void*, uint32_t* pcbSize) {
    s_otherCalls.fetch_add(1);
    if (pcbSize != nullptr) *pcbSize = 0;
    return 0;
}

uint32_t NativeDllInterceptor::win_GetRawInputData(void*, uint32_t, void*, uint32_t* pcbSize, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (pcbSize != nullptr) *pcbSize = 0;
    return 0;
}

uint32_t NativeDllInterceptor::win_GetRawInputBuffer(void*, uint32_t* pcbSize, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (pcbSize != nullptr) *pcbSize = 0;
    return 0;
}

int16_t NativeDllInterceptor::win_GetKeyState(int32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int16_t NativeDllInterceptor::win_GetAsyncKeyState(int32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

uint32_t NativeDllInterceptor::win_MapVirtualKeyW(uint32_t uCode, uint32_t) {
    s_otherCalls.fetch_add(1);
    return uCode;
}

uint32_t NativeDllInterceptor::win_MapVirtualKeyA(uint32_t uCode, uint32_t) {
    s_otherCalls.fetch_add(1);
    return uCode;
}

uint32_t NativeDllInterceptor::win_MapVirtualKeyExA(uint32_t uCode, uint32_t, void*) {
    s_otherCalls.fetch_add(1);
    return uCode;
}

int32_t NativeDllInterceptor::win_GetKeyNameTextW(int32_t, wchar_t* lpString, int32_t cchSize) {
    s_otherCalls.fetch_add(1);
    if (lpString != nullptr && cchSize > 0) reinterpret_cast<char16_t*>(lpString)[0] = 0;
    return 0;
}

int32_t NativeDllInterceptor::win_ToUnicode(uint32_t, uint32_t, const uint8_t*, wchar_t* pwszBuff, int32_t cchBuff, uint32_t) {
    s_otherCalls.fetch_add(1);
    if (pwszBuff != nullptr && cchBuff > 0) reinterpret_cast<char16_t*>(pwszBuff)[0] = 0;
    return 0;
}

void* NativeDllInterceptor::win_GetKeyboardLayout(uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x04090409);
}

int32_t NativeDllInterceptor::win_GetKeyboardLayoutNameW(wchar_t* pwszKLID) {
    s_otherCalls.fetch_add(1);
    if (pwszKLID != nullptr) {
        writeGuestWString(pwszKLID, 9, "00000409");  // KL_NAMELENGTH = 9
        return 1;
    }
    return 0;
}

void* NativeDllInterceptor::win_SetCapture(void*) {
    s_otherCalls.fetch_add(1);
    return nullptr;
}

int32_t NativeDllInterceptor::win_ReleaseCapture() {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_LoadCursorA(void*, const char*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10001);
}

void* NativeDllInterceptor::win_LoadCursorW(void*, const wchar_t*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10001);
}

int32_t NativeDllInterceptor::win_DestroyCursor(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_LoadIconA(void*, const char*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10002);
}

void* NativeDllInterceptor::win_LoadIconW(void*, const wchar_t*) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10002);
}

int32_t NativeDllInterceptor::win_DestroyIcon(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_LoadImageW(void*, const wchar_t*, uint32_t, int32_t, int32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10003);
}

void* NativeDllInterceptor::win_LoadImageA(void*, const char*, uint32_t, int32_t, int32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0x10003);
}

int32_t NativeDllInterceptor::win_TrackMouseEvent(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_EnableMouseInPointer(int32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_IsMouseInPointerEnabled() {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_GetWindowPlacement(void*, void* lpwndpl) {
    s_otherCalls.fetch_add(1);
    if (lpwndpl != nullptr) {
        auto* data = static_cast<uint32_t*>(lpwndpl);
        data[0] = 44; // length = sizeof(WINDOWPLACEMENT)
        data[1] = 0;  // flags
        data[2] = 1;  // showCmd = SW_SHOWNORMAL
    }
    return 1;
}

int32_t NativeDllInterceptor::win_SetWindowPlacement(void*, const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_AdjustWindowRectExForDpi(void*, uint32_t, int32_t, uint32_t, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_IsWindowVisible(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_IsIconic(void*) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_MoveWindow(void*, int32_t, int32_t, int32_t, int32_t, int32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_PtInRect(const void* lprc, int32_t x, int32_t y) {
    s_otherCalls.fetch_add(1);
    if (lprc == nullptr) return 0;
    const auto* r = static_cast<const int32_t*>(lprc);
    return (x >= r[0] && x < r[2] && y >= r[1] && y < r[3]) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_OffsetRect(void* lprc, int32_t dx, int32_t dy) {
    s_otherCalls.fetch_add(1);
    if (lprc == nullptr) return 0;
    auto* r = static_cast<int32_t*>(lprc);
    r[0] += dx; r[2] += dx;
    r[1] += dy; r[3] += dy;
    return 1;
}

int32_t NativeDllInterceptor::win_CopyRect(void* lprcDst, const void* lprcSrc) {
    s_otherCalls.fetch_add(1);
    if (lprcDst == nullptr || lprcSrc == nullptr) return 0;
    std::memcpy(lprcDst, lprcSrc, 16);
    return 1;
}

int64_t NativeDllInterceptor::win_GetMessageExtraInfo() {
    s_otherCalls.fetch_add(1);
    return 0;
}

uint32_t NativeDllInterceptor::win_GetDoubleClickTime() {
    s_otherCalls.fetch_add(1);
    return 500;
}

uint32_t NativeDllInterceptor::win_GetCaretBlinkTime() {
    s_otherCalls.fetch_add(1);
    return 530;
}

int32_t NativeDllInterceptor::win_OpenClipboard(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_CloseClipboard() {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_EmptyClipboard() {
    s_otherCalls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_GetClipboardData(uint32_t) {
    s_otherCalls.fetch_add(1);
    return nullptr;
}

void* NativeDllInterceptor::win_SetClipboardData(uint32_t, void* hMem) {
    s_otherCalls.fetch_add(1);
    return hMem;
}

int32_t NativeDllInterceptor::win_IsClipboardFormatAvailable(uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_MessageBoxA(void*, const char* lpText, const char* lpCaption, uint32_t uType) {
    s_otherCalls.fetch_add(1);
    // Sin interfaz: se registra el texto (suele ser una aserción del runtime) y se acepta.
    static std::atomic<uint64_t> s_count{0};
    const uint64_t n = s_count.fetch_add(1);
    if (n < 20 || (n % 10000) == 0) {
        LOGW("MessageBoxA[%s] (0x%x) #%llu: %.400s", lpCaption != nullptr ? lpCaption : "", uType,
             static_cast<unsigned long long>(n + 1), lpText != nullptr ? lpText : "");
    }
    return 1; // IDOK
}

int32_t NativeDllInterceptor::win_MessageBoxW(void*, const wchar_t* lpText, const wchar_t* lpCaption, uint32_t uType) {
    s_otherCalls.fetch_add(1);
    LOGW("MessageBoxW[%s] (0x%x): %.400s", lpCaption != nullptr ? wcharToUtf8(lpCaption).c_str() : "", uType,
         lpText != nullptr ? wcharToUtf8(lpText).c_str() : "");
    return 1; // IDOK
}

intptr_t NativeDllInterceptor::win_DialogBoxParamW(void*, const wchar_t*, void*, void*, intptr_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_EndDialog(void*, intptr_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetDlgItemTextW(void*, int32_t, const wchar_t*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetDlgItemTextA(void*, int32_t, const char*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int64_t NativeDllInterceptor::win_SendDlgItemMessageW(void*, int32_t, uint32_t, uint64_t, int64_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

void* NativeDllInterceptor::win_RegisterDeviceNotificationW(void*, void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return reinterpret_cast<void*>(0xCAFE0050);
}

int32_t NativeDllInterceptor::win_UnregisterDeviceNotification(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// Rewired (Cuphead) instala un hook WH_GETMESSAGE/WH_CALLWNDPROC sobre la ventana de Unity para
// leer WM_INPUT; sin él su fuente de entrada no se inicializa y cae a la entrada de Unity. Los
// mandos le llegan por XInput (sondeo), así que basta un HHOOK válido.
void* NativeDllInterceptor::win_SetWindowsHookExW(int32_t idHook, void* lpfn, void*, uint32_t threadId) {
    s_otherCalls.fetch_add(1);
    static std::atomic<uintptr_t> s_nextHook{0xCAFE2000};
    LOGI("SetWindowsHookEx(id=%d, proc=%p, hilo=%u) -> hook simulado", idHook, lpfn, threadId);
    return reinterpret_cast<void*>(s_nextHook.fetch_add(0x10));
}

int32_t NativeDllInterceptor::win_UnhookWindowsHookEx(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

intptr_t NativeDllInterceptor::win_CallNextHookEx(void*, int32_t, uintptr_t, intptr_t) {
    s_otherCalls.fetch_add(1);
    return 0;
}

int32_t NativeDllInterceptor::win_RegisterTouchWindow(void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// -----------------------------------------------------------------------------
// GDI32
// -----------------------------------------------------------------------------
int32_t NativeDllInterceptor::win_ChoosePixelFormat(void*, const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_SetPixelFormat(void*, int32_t, const void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_GetDeviceCaps(void*, int32_t index) {
    s_otherCalls.fetch_add(1);
    return win32::deviceCaps(index);
}

int32_t NativeDllInterceptor::win_SwapBuffers(void*) {
    s_otherCalls.fetch_add(1);
    return 1;
}

// -----------------------------------------------------------------------------
// DisplayConfig
// -----------------------------------------------------------------------------
// CCD (Connecting and Configuring Displays): un path activo pantalla interna -> 2 modos
// (origen y destino). Antes QueryDisplayConfig devolvía éxito sin rellenar nada y Unity indexaba
// sus arrays con modeInfoIdx basura (AV en UnityPlayer+0x11d8030).
namespace {
constexpr uint32_t kCcdAdapterLow = 0x00001234u;
constexpr uint32_t kCcdTargetId = 0x1100u;

void put32(uint8_t* p, size_t off, uint32_t v) { std::memcpy(p + off, &v, 4); }
void put64(uint8_t* p, size_t off, uint64_t v) { std::memcpy(p + off, &v, 8); }
}  // namespace

int32_t NativeDllInterceptor::win_GetDisplayConfigBufferSizes(uint32_t, uint32_t* numPathArrayElements, uint32_t* numModeInfoArrayElements) {
    s_otherCalls.fetch_add(1);
    if (numPathArrayElements == nullptr || numModeInfoArrayElements == nullptr) return 87;
    *numPathArrayElements = 1;
    *numModeInfoArrayElements = 2;
    return 0; // ERROR_SUCCESS
}

int32_t NativeDllInterceptor::win_QueryDisplayConfig(uint32_t flags, uint32_t* numPaths, void* paths, uint32_t* numModes,
                                                     void* modes, void* currentTopologyId) {
    s_otherCalls.fetch_add(1);
    if (numPaths == nullptr || numModes == nullptr || paths == nullptr || modes == nullptr) return 87;
    if (*numPaths < 1 || *numModes < 2) return 122;  // ERROR_INSUFFICIENT_BUFFER
    const uint32_t w = static_cast<uint32_t>(win32::screenWidth());
    const uint32_t h = static_cast<uint32_t>(win32::screenHeight());
    const uint32_t hz = static_cast<uint32_t>(win32::refreshHz());

    // DISPLAYCONFIG_PATH_INFO (72 B)
    auto* p = static_cast<uint8_t*>(paths);
    std::memset(p, 0, 72);
    put32(p, 0, kCcdAdapterLow);     // sourceInfo.adapterId
    put32(p, 8, 0);                  // sourceInfo.id
    put32(p, 12, 0);                 // sourceInfo.modeInfoIdx -> modes[0]
    put32(p, 16, 1);                 // DISPLAYCONFIG_SOURCE_IN_USE
    put32(p, 20, kCcdAdapterLow);    // targetInfo.adapterId
    put32(p, 28, kCcdTargetId);      // targetInfo.id
    put32(p, 32, 1);                 // targetInfo.modeInfoIdx -> modes[1]
    put32(p, 36, 0x80000000u);       // DISPLAYCONFIG_OUTPUT_TECHNOLOGY_INTERNAL
    put32(p, 40, 1);                 // rotation IDENTITY
    put32(p, 44, 1);                 // scaling IDENTITY
    put32(p, 48, hz);                // refreshRate.Numerator
    put32(p, 52, 1);                 // refreshRate.Denominator
    put32(p, 56, 1);                 // scanLineOrdering PROGRESSIVE
    put32(p, 60, 1);                 // targetAvailable
    put32(p, 64, 1);                 // DISPLAYCONFIG_TARGET_IN_USE
    put32(p, 68, 1);                 // DISPLAYCONFIG_PATH_ACTIVE

    // DISPLAYCONFIG_MODE_INFO (64 B) x2
    auto* m = static_cast<uint8_t*>(modes);
    std::memset(m, 0, 128);
    put32(m, 0, 1);                  // DISPLAYCONFIG_MODE_INFO_TYPE_SOURCE
    put32(m, 4, 0);
    put32(m, 8, kCcdAdapterLow);
    put32(m, 16, w);
    put32(m, 20, h);
    put32(m, 24, 4);                 // DISPLAYCONFIG_PIXELFORMAT_32BPP
    uint8_t* t = m + 64;
    put32(t, 0, 2);                  // DISPLAYCONFIG_MODE_INFO_TYPE_TARGET
    put32(t, 4, kCcdTargetId);
    put32(t, 8, kCcdAdapterLow);
    put64(t, 16, static_cast<uint64_t>(w) * h * hz);  // pixelRate
    put32(t, 24, h * hz);            // hSyncFreq
    put32(t, 28, 1);
    put32(t, 32, hz);                // vSyncFreq
    put32(t, 36, 1);
    put32(t, 40, w);                 // activeSize
    put32(t, 44, h);
    put32(t, 48, w);                 // totalSize
    put32(t, 52, h);
    put32(t, 56, 255);               // videoStandard OTHER
    put32(t, 60, 1);                 // scanLineOrdering PROGRESSIVE
    *numPaths = 1;
    *numModes = 2;
    if ((flags & 0x4u) != 0 && currentTopologyId != nullptr) {  // QDC_DATABASE_CURRENT
        put32(static_cast<uint8_t*>(currentTopologyId), 0, 1);  // DISPLAYCONFIG_TOPOLOGY_INTERNAL
    }
    return 0;
}

int32_t NativeDllInterceptor::win_DisplayConfigGetDeviceInfo(void* request) {
    s_otherCalls.fetch_add(1);
    if (request == nullptr) return 87;
    auto* r = static_cast<uint8_t*>(request);
    uint32_t type = 0, size = 0;
    std::memcpy(&type, r, 4);
    std::memcpy(&size, r + 4, 4);
    if (size < 20) return 87;
    std::memset(r + 20, 0, size - 20);
    auto wstr = [&](size_t off, size_t chars, const char* text) {
        if (off + chars * 2 <= size) win32::copyAsciiToWchar16(reinterpret_cast<char16_t*>(r + off), chars, text);
    };
    switch (type) {
        case 1:  // GET_SOURCE_NAME: viewGdiDeviceName[32]
            wstr(20, 32, "\\\\.\\DISPLAY1");
            return 0;
        case 2:  // GET_TARGET_NAME
            if (size >= 36) {
                put32(r, 20, 1);            // flags: friendlyNameFromEdid
                put32(r, 24, 0x80000000u);  // outputTechnology INTERNAL
            }
            wstr(36, 64, "HyperPlay Display");
            wstr(164, 128, "\\\\?\\DISPLAY#HYP0001#1&0&UID0#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}");
            return 0;
        case 4:  // GET_ADAPTER_NAME: adapterDevicePath[128]
            wstr(20, 128, "\\\\?\\PCI#VEN_5143&DEV_0830#0&0#{5b45201d-f2f2-4f3b-85bb-30ff1f953599}");
            return 0;
        case 11:  // GET_SDR_WHITE_LEVEL (1000 = 80 nits)
            if (size >= 24) put32(r, 20, 1000);
            return 0;
        default:  // ADVANCED_COLOR_INFO etc.: sin HDR (todo a cero)
            return 0;
    }
}

// -----------------------------------------------------------------------------
// Proceso, Hilos y Consola
// -----------------------------------------------------------------------------
uint32_t NativeDllInterceptor::win_GetCurrentProcessId() {
    s_kernel32Calls.fetch_add(1);
    return static_cast<uint32_t>(getpid());
}

void* NativeDllInterceptor::win_GetCurrentProcess() {
    s_kernel32Calls.fetch_add(1);
    return reinterpret_cast<void*>(~static_cast<uintptr_t>(0));
}

void NativeDllInterceptor::win_GetStartupInfoW(void* lpStartupInfo) {
    s_kernel32Calls.fetch_add(1);
    if (!lpStartupInfo) return;
    struct WinStartupInfoW {
        uint32_t cb;
        wchar_t* lpReserved;
        wchar_t* lpDesktop;
        wchar_t* lpTitle;
        uint32_t dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        uint16_t wShowWindow, cbReserved2;
        uint8_t* lpReserved2;
        void* hStdInput, *hStdOutput, *hStdError;
    };
    // 32 bits: STARTUPINFOW mide 68 bytes (punteros de 4); escribir los 104 de x64 pisaba la pila del
    // llamador (mscoree.dll de .NET Framework en DmC).
    const uint32_t cb = lowmem::enabled() ? 68u : static_cast<uint32_t>(sizeof(WinStartupInfoW));
    std::memset(lpStartupInfo, 0, cb);
    std::memcpy(lpStartupInfo, &cb, 4);
}

static std::string getActiveExeName() {
    if (s_activeImage && !s_activeImage->filePath.empty()) {
        size_t slash = s_activeImage->filePath.find_last_of("/\\");
        return (slash != std::string::npos) ? s_activeImage->filePath.substr(slash + 1) : "game.exe";
    }
    return "game.exe";
}

// Línea de comandos del juego. Unity: arranca a la resolución nativa del teléfono en pantalla
// completa, sin depender de la resolución guardada en PlayerPrefs (que puede ser menor).
static std::string buildCommandLine() {
    if (managed::active()) {
        // Juego .NET: Mono ejecuta el .exe del juego (ver ManagedHost.h).
        const std::string full = "\"" + std::string(managed::kWinRoot) + "\\bin\\mono.exe\" \"D:\\" + managed::g_gameExe + "\"";
        LOGI("Línea de comandos (juego .NET): %s", full.c_str());
        return full;
    }
    std::string full = "\"" + BionicPeLoader::exeWinDir() + getActiveExeName() + "\"";
    if (s_activeImage != nullptr && access((s_activeImage->gameDir + "/UnityPlayer.dll").c_str(), F_OK) == 0) {
        char args[96];
        std::snprintf(args, sizeof(args), " -screen-width %d -screen-height %d -screen-fullscreen 1",
                      static_cast<int>(win32::screenWidth()), static_cast<int>(win32::screenHeight()));
        full += args;
    }
    // Unreal Engine 3: RHI elegido a mano para pruebas (setprop debug.hyperplay.ue3rhi d3d9|dx11|opengl).
    // Sin la propiedad, el del juego (DmC solo trae D3D9 activo: su comprobación de D3D11 devuelve 0 y no
    // incluye shaders de OpenGL).
    if (s_activeImage != nullptr && s_activeImage->is32Bit &&
        access((s_activeImage->gameDir + "/../../Engine/Config/BaseEngine.ini").c_str(), F_OK) == 0) {
        char rhi[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.ue3rhi", rhi) > 0) full += std::string(" -") + rhi;
    }
    LOGI("Línea de comandos del juego: %s", full.c_str());
    return full;
}

// Copia estable por debajo de 4 GB de datos nativos que se entregan a un invitado de 32 bits (línea
// de comandos, bloque de entorno...): un puntero de libhyperplay.so no cabe en sus 4 bytes. Sin
// invitado de 32 bits devuelve el mismo puntero. Se cachea por puntero de origen.
static const void* lowCopy(const void* src, size_t bytes) {
    if (!lowmem::enabled() || src == nullptr || reinterpret_cast<uintptr_t>(src) <= 0xFFFFFFFFull) return src;
    static std::mutex s_m;
    static std::unordered_map<const void*, void*> s_copies;
    static uint8_t* s_pool = nullptr;
    static size_t s_left = 0;
    std::lock_guard<std::mutex> lk(s_m);
    if (const auto it = s_copies.find(src); it != s_copies.end()) return it->second;
    const size_t need = (bytes + 15) & ~size_t{15};
    if (need > s_left) {
        const size_t chunk = std::max<size_t>(need, 1u << 20);
        void* p = lowmem::mmapBelow4G(chunk);
        if (p == MAP_FAILED) return src;
        s_pool = static_cast<uint8_t*>(p);
        s_left = chunk;
    }
    void* dst = s_pool;
    std::memcpy(dst, src, bytes);
    s_pool += need;
    s_left -= need;
    s_copies[src] = dst;
    return dst;
}

const char* NativeDllInterceptor::win_GetCommandLineA() {
    s_kernel32Calls.fetch_add(1);
    if (s_cmdA[0] == '\0') {
        std::string full = buildCommandLine();
        std::snprintf(s_cmdA, sizeof(s_cmdA), "%s", full.c_str());
    }
    return static_cast<const char*>(lowCopy(s_cmdA, std::strlen(s_cmdA) + 1));
}

const void* NativeDllInterceptor::win_GetCommandLineW() {
    s_kernel32Calls.fetch_add(1);
    if (s_cmdW[0] == u'\0') {
        std::string full = buildCommandLine();
        size_t i = 0;
        for (; i < full.size() && i < 500; ++i) {
            s_cmdW[i] = static_cast<char16_t>(full[i]);
        }
        s_cmdW[i] = u'\0';
    }
    size_t n = 0;
    while (s_cmdW[n] != u'\0') ++n;
    return lowCopy(s_cmdW, (n + 1) * sizeof(char16_t));
}

const void* NativeDllInterceptor::win_GetEnvironmentStringsW() {
    s_kernel32Calls.fetch_add(1);
    // CoreCLR lee su configuración (DOTNET_*) de este bloque, no con GetEnvironmentVariableW:
    // las variables que HyperPlay fija para .NET tienen que estar aquí también.
    static std::u16string s_block = [] {
        std::vector<std::string> vars = {"=::=::\\", "PATH=D:\\", "SYSTEMROOT=C:\\Windows", "TEMP=D:\\Temp",
                                         "DOTNET_EnableDiagnostics=0", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1",
                                         "DOTNET_EnableHWIntrinsic=0", "DOTNET_gcConcurrent=0"};
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.r2r", v) > 0 && v[0] == '0') vars.push_back("DOTNET_ReadyToRun=0");
        // Diagnóstico: EventPipe a archivo con los eventos de excepción del runtime (tipo y
        // mensaje de cada excepción .NET lanzada) en la carpeta del juego.
        if (__system_property_get("debug.hyperplay.dotnettrace", v) > 0 && v[0] == '1') {
            vars.erase(std::remove(vars.begin(), vars.end(), std::string("DOTNET_EnableDiagnostics=0")), vars.end());
            vars.push_back("DOTNET_EnableEventPipe=1");
            vars.push_back("DOTNET_EventPipeOutputPath=D:\\hp-dotnet.nettrace");
            vars.push_back("DOTNET_EventPipeConfig=Microsoft-Windows-DotNETRuntime:0x800C:5");
            vars.push_back("DOTNET_EventPipeRundown=0");
            vars.push_back("DOTNET_EventPipeOutputStreaming=1");
        }
        std::u16string b;
        for (const auto& s : vars) {
            for (char c : s) b.push_back(static_cast<char16_t>(static_cast<uint8_t>(c)));
            b.push_back(u'\0');
        }
        b.push_back(u'\0');
        return b;
    }();
    return lowCopy(s_block.data(), s_block.size() * sizeof(char16_t));
}

int32_t NativeDllInterceptor::win_FreeEnvironmentStringsW(void*) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

void* NativeDllInterceptor::win_GetStdHandle(uint32_t nStdHandle) {
    s_kernel32Calls.fetch_add(1);
    return reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000u | (nStdHandle & 0xFFu)));
}

int32_t NativeDllInterceptor::win_SetStdHandle(uint32_t, void*) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_WriteConsoleW(void*, const void*, uint32_t nNumberOfCharsToWrite, uint32_t* lpNumberOfCharsWritten, void*) {
    s_kernel32Calls.fetch_add(1);
    if (lpNumberOfCharsWritten) *lpNumberOfCharsWritten = nNumberOfCharsToWrite;
    return 1;
}

uint32_t NativeDllInterceptor::win_GetConsoleOutputCP() {
    s_kernel32Calls.fetch_add(1);
    return 65001; // CP_UTF8
}

uint32_t NativeDllInterceptor::win_GetConsoleCP() {
    s_kernel32Calls.fetch_add(1);
    return 65001; // CP_UTF8
}

int32_t NativeDllInterceptor::win_GetConsoleMode(void*, uint32_t* lpMode) {
    s_kernel32Calls.fetch_add(1);
    if (lpMode) *lpMode = 0x0007; // ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT
    return 1;
}

uint32_t NativeDllInterceptor::win_GetFileType(void* hFile) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t val = reinterpret_cast<uintptr_t>(hFile);
    if (val >= 0x1000u && val <= 0x10FFu) {
        LOGI("win_GetFileType: handle=%p (console stdhandle) -> FILE_TYPE_CHAR (2)", hFile);
        return 2; // FILE_TYPE_CHAR
    }
    const int fd = fdFromHandle(hFile);
    if (fd < 0) {
        LOGW("win_GetFileType: handle=%p (invalid fd=%d) -> FILE_TYPE_UNKNOWN (0)", hFile, fd);
        win_SetLastError(6); // ERROR_INVALID_HANDLE
        return 0; // FILE_TYPE_UNKNOWN
    }
    struct stat st{};
    if (fstat(fd, &st) == 0) {
        if (S_ISREG(st.st_mode)) {
            LOGI("win_GetFileType: handle=%p (fd=%d) -> FILE_TYPE_DISK (1)", hFile, fd);
            return 1; // FILE_TYPE_DISK
        }
        if (S_ISCHR(st.st_mode)) {
            LOGI("win_GetFileType: handle=%p (fd=%d) -> FILE_TYPE_CHAR (2)", hFile, fd);
            return 2; // FILE_TYPE_CHAR
        }
        if (S_ISFIFO(st.st_mode) || S_ISSOCK(st.st_mode)) {
            LOGI("win_GetFileType: handle=%p (fd=%d) -> FILE_TYPE_PIPE (3)", hFile, fd);
            return 3; // FILE_TYPE_PIPE
        }
    }
    LOGI("win_GetFileType: handle=%p (fd=%d fallback) -> FILE_TYPE_DISK (1)", hFile, fd);
    return 1; // FILE_TYPE_DISK
}

uint32_t NativeDllInterceptor::win_GetFileAttributesA(const char* lpFileName) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName) {
        win_SetLastError(2);
        return 0xFFFFFFFFu;
    }
    const std::string rawPath = lpFileName;
    std::string realPath = rawPath;
    if (s_activeImage) {
        realPath = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    }
    struct stat st{};
    if (stat(realPath.c_str(), &st) == 0 || stat(rawPath.c_str(), &st) == 0) {
        const uint32_t attrs = S_ISDIR(st.st_mode) ? 0x10u : 0x80u;
        LOGI("win_GetFileAttributesA: raw='%s' real='%s' -> 0x%x (FOUND)", rawPath.c_str(), realPath.c_str(), attrs);
        win_SetLastError(0);
        return attrs;
    }
    LOGI("win_GetFileAttributesA: raw='%s' real='%s' -> NOT FOUND", rawPath.c_str(), realPath.c_str());
    win_SetLastError(2); // ERROR_FILE_NOT_FOUND
    return 0xFFFFFFFFu;
}

uint32_t NativeDllInterceptor::win_GetFileAttributesW(const void* lpFileName) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName) {
        win_SetLastError(2);
        return 0xFFFFFFFFu;
    }
    const auto* u16 = static_cast<const char16_t*>(lpFileName);
    char hexDump[128] = {};
    size_t hexPos = 0;
    for (size_t i = 0; i < 24 && u16[i] != 0 && hexPos + 6 < sizeof(hexDump); ++i) {
        hexPos += std::snprintf(hexDump + hexPos, sizeof(hexDump) - hexPos, "%04x ", static_cast<uint16_t>(u16[i]));
    }
    const std::string rawPath = utf16ToUtf8(u16);
    std::string realPath = rawPath;
    if (s_activeImage) {
        realPath = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    }
    struct stat st{};
    if (stat(realPath.c_str(), &st) == 0 || stat(rawPath.c_str(), &st) == 0) {
        const uint32_t attrs = S_ISDIR(st.st_mode) ? 0x10u : 0x80u;
        LOGI("win_GetFileAttributesW: raw='%s' hex=[%s] real='%s' -> 0x%x (FOUND)", rawPath.c_str(), hexDump, realPath.c_str(), attrs);
        win_SetLastError(0);
        return attrs;
    }
    LOGI("win_GetFileAttributesW: raw='%s' hex=[%s] real='%s' -> NOT FOUND", rawPath.c_str(), hexDump, realPath.c_str());
    win_SetLastError(2); // ERROR_FILE_NOT_FOUND
    return 0xFFFFFFFFu;
}

int32_t NativeDllInterceptor::win_GetFileAttributesExW(const void* lpFileName, uint32_t fInfoLevelId, void* lpFileInformation) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName || !lpFileInformation) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }
    (void)fInfoLevelId; // GetFileExInfoStandard = 0
    const auto* u16 = static_cast<const char16_t*>(lpFileName);
    const std::string rawPath = utf16ToUtf8(u16);
    std::string realPath = rawPath;
    if (s_activeImage) {
        realPath = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    }
    struct stat st{};
    if (stat(realPath.c_str(), &st) != 0 && stat(rawPath.c_str(), &st) != 0) {
        LOGI("win_GetFileAttributesExW: raw='%s' real='%s' -> NOT FOUND", rawPath.c_str(), realPath.c_str());
        win_SetLastError(2); // ERROR_FILE_NOT_FOUND
        return 0;
    }

    struct Win32FileAttributeData {
        uint32_t dwFileAttributes;
        uint32_t ftCreationTimeLow;
        uint32_t ftCreationTimeHigh;
        uint32_t ftLastAccessTimeLow;
        uint32_t ftLastAccessTimeHigh;
        uint32_t ftLastWriteTimeLow;
        uint32_t ftLastWriteTimeHigh;
        uint32_t nFileSizeHigh;
        uint32_t nFileSizeLow;
    };
    auto* out = static_cast<Win32FileAttributeData*>(lpFileInformation);
    out->dwFileAttributes = S_ISDIR(st.st_mode) ? 0x10u : 0x80u;

    auto toFileTime = [](time_t sec, long nsec, uint32_t& low, uint32_t& high) {
        uint64_t ft = (static_cast<uint64_t>(sec) + 11644473600ULL) * 10000000ULL + (static_cast<uint64_t>(nsec) / 100ULL);
        low = static_cast<uint32_t>(ft & 0xFFFFFFFFULL);
        high = static_cast<uint32_t>((ft >> 32) & 0xFFFFFFFFULL);
    };

    toFileTime(st.st_ctime, 0, out->ftCreationTimeLow, out->ftCreationTimeHigh);
    toFileTime(st.st_atime, 0, out->ftLastAccessTimeLow, out->ftLastAccessTimeHigh);
    toFileTime(st.st_mtime, 0, out->ftLastWriteTimeLow, out->ftLastWriteTimeHigh);
    out->nFileSizeHigh = static_cast<uint32_t>((static_cast<uint64_t>(st.st_size) >> 32) & 0xFFFFFFFFULL);
    out->nFileSizeLow = static_cast<uint32_t>(static_cast<uint64_t>(st.st_size) & 0xFFFFFFFFULL);

    LOGI("win_GetFileAttributesExW: raw='%s' real='%s' -> size=%lld attr=0x%x (OK)",
         rawPath.c_str(), realPath.c_str(), static_cast<long long>(st.st_size), out->dwFileAttributes);
    win_SetLastError(0);
    return 1;
}

int32_t NativeDllInterceptor::win_GetFileAttributesExA(const char* lpFileName, uint32_t fInfoLevelId, void* lpFileInformation) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName || !lpFileInformation) {
        win_SetLastError(87);
        return 0;
    }
    (void)fInfoLevelId;
    const std::string rawPath = lpFileName;
    std::string realPath = rawPath;
    if (s_activeImage) {
        realPath = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    }
    struct stat st{};
    if (stat(realPath.c_str(), &st) != 0 && stat(rawPath.c_str(), &st) != 0) {
        LOGI("win_GetFileAttributesExA: raw='%s' real='%s' -> NOT FOUND", rawPath.c_str(), realPath.c_str());
        win_SetLastError(2);
        return 0;
    }

    struct Win32FileAttributeData {
        uint32_t dwFileAttributes;
        uint32_t ftCreationTimeLow;
        uint32_t ftCreationTimeHigh;
        uint32_t ftLastAccessTimeLow;
        uint32_t ftLastAccessTimeHigh;
        uint32_t ftLastWriteTimeLow;
        uint32_t ftLastWriteTimeHigh;
        uint32_t nFileSizeHigh;
        uint32_t nFileSizeLow;
    };
    auto* out = static_cast<Win32FileAttributeData*>(lpFileInformation);
    out->dwFileAttributes = S_ISDIR(st.st_mode) ? 0x10u : 0x80u;

    auto toFileTime = [](time_t sec, long nsec, uint32_t& low, uint32_t& high) {
        uint64_t ft = (static_cast<uint64_t>(sec) + 11644473600ULL) * 10000000ULL + (static_cast<uint64_t>(nsec) / 100ULL);
        low = static_cast<uint32_t>(ft & 0xFFFFFFFFULL);
        high = static_cast<uint32_t>((ft >> 32) & 0xFFFFFFFFULL);
    };

    toFileTime(st.st_ctime, 0, out->ftCreationTimeLow, out->ftCreationTimeHigh);
    toFileTime(st.st_atime, 0, out->ftLastAccessTimeLow, out->ftLastAccessTimeHigh);
    toFileTime(st.st_mtime, 0, out->ftLastWriteTimeLow, out->ftLastWriteTimeHigh);
    out->nFileSizeHigh = static_cast<uint32_t>((static_cast<uint64_t>(st.st_size) >> 32) & 0xFFFFFFFFULL);
    out->nFileSizeLow = static_cast<uint32_t>(static_cast<uint64_t>(st.st_size) & 0xFFFFFFFFULL);

    LOGI("win_GetFileAttributesExA: raw='%s' real='%s' -> size=%lld attr=0x%x (OK)",
         rawPath.c_str(), realPath.c_str(), static_cast<long long>(st.st_size), out->dwFileAttributes);
    win_SetLastError(0);
    return 1;
}

// Ruta Windows de un HMODULE: exe (NULL o base), DLL local mapeada o DLL de sistema sintética.
// Cadena vacía si el handle no es un módulo.
static std::string guestModulePath(void* hModule) {
    if (s_activeImage == nullptr) return {};
    if (hModule == nullptr || hModule == s_activeImage->baseAddress) {
        // Juego .NET: el proceso es mono.exe del runtime (Mono calcula su raíz desde aquí).
        if (std::string w = managed::toWinPath(s_activeImage->filePath); !w.empty()) return w;
        return BionicPeLoader::exeWinDir() + getActiveExeName();
    }
    for (const auto& kv : s_activeImage->localDlls) {
        if (kv.second.base != hModule) continue;
        // DLL de sistema instaladas por la app (/sdcard/HyperPlay/redist/<arq>/...): su ruta de
        // Windows real. mscoree/.NET calculan sus carpetas a partir de la suya.
        {
            const std::string& hp = kv.second.hostPath;
            for (const char* arch : {"/redist/x86/", "/redist/x64/"}) {
                const size_t r = hp.find(arch);
                if (r == std::string::npos) continue;
                std::string rest = hp.substr(r + std::strlen(arch));
                std::string win;
                if (rest == "netfx/Windows/System32/mscoree.dll") {
                    // Es la copia de mscoreei.dll: se presenta con su ruta real, donde busca su
                    // satélite de recursos (mscoreeis.dll) y el resto del Framework.
                    win = "C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319\\mscoreei.dll";
                } else if (rest.rfind("netfx/Windows/", 0) == 0) {
                    win = "C:\\Windows\\" + rest.substr(14);
                } else {
                    win = "C:\\Windows\\System32\\" + rest;
                }
                for (char& ch : win) if (ch == '/') ch = '\\';
                return win;
            }
        }
        if (std::string w = managed::toWinPath(kv.second.hostPath); !w.empty()) return w;
        std::string p = kv.first;
        for (char& ch : p) if (ch == '/') ch = '\\';
        if (p.size() >= 2 && p[1] == ':') return p;
        return BionicPeLoader::exeWinDir() + p;
    }
    const auto h = reinterpret_cast<uintptr_t>(hModule);
    if ((h & ~static_cast<uintptr_t>(0xFF)) == 0x7FFF0000u) {
        static const char* kSystem[] = {"", "kernel32.dll", "user32.dll", "vulkan-1.dll", "ntdll.dll", "advapi32.dll",
                                        "shell32.dll", "ole32.dll", "ws2_32.dll", "d3d11.dll", "dxgi.dll",
                                        "xinput1_3.dll", "winmm.dll", "bcrypt.dll", "cryptbase.dll", "shlwapi.dll",
                                        "steam_api64.dll", "mfplat.dll", "opengl32.dll", "gdi32.dll"};
        const size_t idx = h & 0xFF;
        if (idx > 0 && idx < sizeof(kSystem) / sizeof(kSystem[0])) return std::string("C:\\Windows\\System32\\") + kSystem[idx];
    }
    return {};
}

uint32_t NativeDllInterceptor::win_GetModuleFileNameW(void* hModule, void* lpFilename, uint32_t nSize) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFilename || nSize == 0) return 0;
    const std::string path = guestModulePath(hModule);
    if (path.empty()) {
        win_SetLastError(126);  // ERROR_MOD_NOT_FOUND
        static_cast<char16_t*>(lpFilename)[0] = 0;
        return 0;
    }
    const auto len = static_cast<uint32_t>(guestWStringLength(path));
    writeGuestWString(static_cast<wchar_t*>(lpFilename), nSize, path);
    if (len >= nSize) {
        win_SetLastError(122);  // ERROR_INSUFFICIENT_BUFFER: truncado, devuelve nSize
        return nSize;
    }
    LOGI("win_GetModuleFileNameW(%p) -> '%s'", hModule, path.c_str());
    return len;
}

// --- PSAPI: enumeración de módulos (Mono mono-dl busca símbolos en todos los módulos) ------
static std::vector<std::pair<void*, const PeInfo*>> realGuestModules() {
    std::vector<std::pair<void*, const PeInfo*>> mods;
    if (s_activeImage == nullptr) return mods;
    mods.emplace_back(s_activeImage->baseAddress, &s_activeImage->peInfo);
    for (const auto& kv : s_activeImage->localDlls) {
        bool dup = false;
        for (const auto& m : mods) dup = dup || m.first == kv.second.base;
        if (!dup && kv.second.base != nullptr) mods.emplace_back(kv.second.base, &kv.second.peInfo);
    }
    return mods;
}

int32_t NativeDllInterceptor::win_EnumProcessModules(void*, void** lphModule, uint32_t cb, uint32_t* lpcbNeeded) {
    s_kernel32Calls.fetch_add(1);
    const auto mods = realGuestModules();
    const size_t ps = lowmem::guestPtrSize();  // HMODULE: 4 bytes en 32 bits
    const auto needed = static_cast<uint32_t>(mods.size() * ps);
    if (lpcbNeeded != nullptr) *lpcbNeeded = needed;
    if (lphModule != nullptr) {
        const size_t fit = std::min<size_t>(mods.size(), cb / ps);
        auto* out = reinterpret_cast<uint8_t*>(lphModule);
        for (size_t i = 0; i < fit; ++i) lowmem::storePtr(out + i * ps, mods[i].first);
    }
    LOGI("EnumProcessModules -> %zu módulos (cb=%u)", mods.size(), cb);
    return 1;
}

// GetProcessMemoryInfo(hProcess, PROCESS_MEMORY_COUNTERS[_EX]*, cb) (stdcall). DWORD cb, DWORD PageFaultCount y
// después campos SIZE_T: 4 bytes en 32 bits (40/44 bytes), 8 en 64 (72/80). Unreal 3 lo llama cada fotograma
// para sus estadísticas; sin él caía en un stub sin aridad y descuadraba la pila (fallo de cookie GS en DmC).
static int32_t win_GetProcessMemoryInfo(void*, void* counters, uint32_t cb) {
    if (counters == nullptr || cb < 8) return 0;
    uint64_t rss = 0, hwm = 0, vm = 0;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[128];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "VmRSS: %llu", &kb) == 1) rss = kb * 1024;
            else if (std::sscanf(line, "VmHWM: %llu", &kb) == 1) hwm = kb * 1024;
            else if (std::sscanf(line, "VmData: %llu", &kb) == 1) vm = kb * 1024;
        }
        std::fclose(f);
    }
    const size_t ps = lowmem::guestPtrSize();
    const uint64_t cap = ps == 4 ? 0xFFFFFFFFull : ~0ull;
    const uint64_t vals[9] = {hwm, rss, 0, 0, 0, 0, vm, vm, vm};  // ...PagefileUsage, PeakPagefileUsage, PrivateUsage
    auto* out = static_cast<uint8_t*>(counters);
    std::memset(out, 0, cb);
    std::memcpy(out, &cb, 4);
    for (size_t i = 0; i < 9 && 8 + (i + 1) * ps <= cb; ++i) lowmem::storePtr(out + 8 + i * ps, std::min(vals[i], cap));
    return 1;
}

int32_t NativeDllInterceptor::win_GetModuleInformation(void*, void* hModule, void* lpmodinfo, uint32_t cb) {
    s_kernel32Calls.fetch_add(1);
    const bool w32 = lowmem::enabled();
    const uint32_t infoSize = w32 ? 12u : 24u;  // MODULEINFO: x86 = 12 B, x64 = 24 B
    if (lpmodinfo == nullptr || cb < infoSize) {
        win_SetLastError(122);
        return 0;
    }
    for (const auto& [base, pe] : realGuestModules()) {
        if (base != hModule) continue;
        auto* out = static_cast<uint8_t*>(lpmodinfo);
        std::memset(out, 0, infoSize);
        const uint32_t size = pe->imageSize;
        void* entry = pe->entryPointRva != 0 ? static_cast<uint8_t*>(base) + pe->entryPointRva : nullptr;
        lowmem::storePtr(out, base);                     // lpBaseOfDll
        std::memcpy(out + (w32 ? 4 : 8), &size, 4);      // SizeOfImage
        lowmem::storePtr(out + (w32 ? 8 : 16), entry);   // EntryPoint
        return 1;
    }
    win_SetLastError(6);  // ERROR_INVALID_HANDLE
    return 0;
}

uint32_t NativeDllInterceptor::win_GetModuleFileNameExW(void*, void* hModule, void* lpFilename, uint32_t nSize) {
    return win_GetModuleFileNameW(hModule, lpFilename, nSize);
}

uint32_t NativeDllInterceptor::win_GetModuleFileNameA(void* hModule, char* lpFilename, uint32_t nSize) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFilename || nSize == 0) return 0;
    const std::string full = guestModulePath(hModule);
    if (full.empty()) {
        win_SetLastError(126);
        lpFilename[0] = '\0';
        return 0;
    }
    const uint32_t len = static_cast<uint32_t>(full.size());
    const uint32_t copyLen = std::min(len, nSize - 1);
    std::memcpy(lpFilename, full.c_str(), copyLen);
    lpFilename[copyLen] = '\0';
    LOGI("win_GetModuleFileNameA -> %s", lpFilename);
    return copyLen;
}

// %TEMP% del perfil emulado. resolveVirtualPath lo aísla en <Game>/_HyperPlaySaves/users/...,
// y se crea en disco para que GetFileAttributes/CreateFile sobre él funcionen.
static const char* ensureGuestTempDir() {
    static constexpr char kTemp[] = "C:\\users\\default\\AppData\\Local\\Temp\\";
    if (s_activeImage != nullptr) {
        std::error_code ec;
        std::filesystem::create_directories(BionicPeLoader::resolveVirtualPath(*s_activeImage, kTemp), ec);
    }
    return kTemp;
}

uint32_t NativeDllInterceptor::win_GetTempPathW(uint32_t nBufferLength, void* lpBuffer) {
    s_kernel32Calls.fetch_add(1);
    const std::string temp = ensureGuestTempDir();
    const auto len = static_cast<uint32_t>(temp.size());
    if (lpBuffer == nullptr || nBufferLength <= len) return len + 1;
    // WCHAR de 16 bits (antes se copiaba un literal L"" de 32 bits: el juego leía solo "D").
    writeGuestWString(static_cast<wchar_t*>(lpBuffer), nBufferLength, temp);
    LOGI("win_GetTempPathW -> %s", temp.c_str());
    return len;
}

uint32_t NativeDllInterceptor::win_GetTempPathA(uint32_t nBufferLength, char* lpBuffer) {
    s_kernel32Calls.fetch_add(1);
    const std::string temp = ensureGuestTempDir();
    const auto len = static_cast<uint32_t>(temp.size());
    if (lpBuffer == nullptr || nBufferLength <= len) return len + 1;
    std::memcpy(lpBuffer, temp.c_str(), len + 1);
    LOGI("win_GetTempPathA -> %s", temp.c_str());
    return len;
}

int32_t NativeDllInterceptor::win_SetConsoleCtrlHandler(void*, int32_t) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_OpenProcessToken(void*, uint32_t, void** TokenHandle) {
    s_otherCalls.fetch_add(1);
    lowmem::storePtr(TokenHandle, uint64_t{0xCAFE0002ULL});
    return 1;
}

int32_t NativeDllInterceptor::win_GetModuleHandleExW(uint32_t dwFlags, const void* lpModuleName, void** phModule) {
    s_kernel32Calls.fetch_add(1);
    constexpr uint32_t kFromAddress = 0x4;  // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
    if (phModule && (dwFlags & kFromAddress) != 0) {
        // lpModuleName es una dirección dentro de un módulo, no una cadena.
        const auto addr = reinterpret_cast<uintptr_t>(lpModuleName);
        void* found = nullptr;
        if (s_activeImage != nullptr) {
            const auto exeBase = reinterpret_cast<uintptr_t>(s_activeImage->baseAddress);
            if (addr >= exeBase && addr < exeBase + s_activeImage->totalMappedSize) {
                found = s_activeImage->baseAddress;
            }
            for (const auto& kv : s_activeImage->localDlls) {
                const auto b = reinterpret_cast<uintptr_t>(kv.second.base);
                if (addr >= b && addr < b + kv.second.size) { found = kv.second.base; break; }
            }
        }
        lowmem::storePtr(phModule, found);  // HMODULE*: 4 bytes en 32 bits
        if (found == nullptr) { win_SetLastError(126); return 0; }
        return 1;
    }
    if (phModule) {
        void* h = win_GetModuleHandleW(reinterpret_cast<const wchar_t*>(lpModuleName));
        lowmem::storePtr(phModule, h);
        return h != nullptr ? 1 : 0;
    }
    return 0;
}

void NativeDllInterceptor::win_ExitProcess(uint32_t uExitCode) {
    s_kernel32Calls.fetch_add(1);
    LOGI("ExitProcess invocado con código %u", uExitCode);
    if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
        const uint64_t rsp = b->context().gpr[4];
        LOGI("=== Guest Stack Trace at ExitProcess (RSP=0x%llx) ===", static_cast<unsigned long long>(rsp));
        for (int i = 0; i < 40; ++i) {
            uint64_t val = 0;
            std::memcpy(&val, reinterpret_cast<const void*>(rsp + i * 8), 8);
            LOGI("  [RSP+0x%02x] = 0x%llx", i * 8, static_cast<unsigned long long>(val));
        }
        b->stop();
    }
}

int32_t NativeDllInterceptor::win_TerminateProcess(void* hProcess, uint32_t uExitCode) {
    s_kernel32Calls.fetch_add(1);
    LOGI("TerminateProcess invocado con código %u", uExitCode);
    // En 32 bits el pseudo-handle -1 llega como 0xFFFFFFFF (sin extender el signo).
    const bool self = hProcess == nullptr || hProcess == reinterpret_cast<void*>(static_cast<uintptr_t>(-1LL)) ||
                      (lowmem::enabled() && reinterpret_cast<uintptr_t>(hProcess) == 0xFFFFFFFFu);
    if (self) {
        if (auto* b = OryonEmitterBackend::activeBackend(); b != nullptr) {
            b->stop();
        }
        // El hilo que llama tampoco debe seguir: el CRT (__fastfail / invalid parameter) no
        // espera volver de TerminateProcess y se repetía hasta desbordar la pila.
        if (t_dispatchBackend != nullptr) t_dispatchBackend->stop();
    }
    return 1;
}

int32_t NativeDllInterceptor::win_IsProcessorFeaturePresent(uint32_t ProcessorFeature) {
    s_kernel32Calls.fetch_add(1);
    switch (ProcessorFeature) {
        case 2:  // PF_COMPARE_EXCHANGE_DOUBLE
        case 3:  // PF_MMX_INSTRUCTIONS_AVAILABLE
        case 6:  // PF_XMMI_INSTRUCTIONS_AVAILABLE (SSE)
        case 8:  // PF_RDTSC_INSTRUCTION_AVAILABLE
        case 10: // PF_XMMI64_INSTRUCTIONS_AVAILABLE (SSE2)
        case 13: // PF_SSE3_INSTRUCTIONS_AVAILABLE
        case 14: // PF_COMPARE_EXCHANGE_128
        case 36: // PF_SSSE3_INSTRUCTIONS_AVAILABLE
        case 37: // PF_SSE4_1_INSTRUCTIONS_AVAILABLE
        case 38: // PF_SSE4_2_INSTRUCTIONS_AVAILABLE
        case 39: // PF_AVX_INSTRUCTIONS_AVAILABLE
            return 1;
        default:
            return 0;
    }
}

int32_t NativeDllInterceptor::win_IsDebuggerPresent() {
    s_kernel32Calls.fetch_add(1);
    return 0;
}

// CommandLineToArgvW con las reglas de Windows: comillas agrupan, 2n '\' + '"' -> n '\' y abre o
// cierra comillas, 2n+1 '\' + '"' -> n '\' + '"' literal; el resto de '\' son literales. El
// resultado es un único bloque (punteros + cadenas) que se libera con LocalFree. Antes devolvía
// siempre {"game.exe"} y Mono (mono.exe Celeste.exe) no veía el programa.
void* NativeDllInterceptor::win_CommandLineToArgvW(const void* lpCmdLine, int32_t* pNumArgs) {
    s_otherCalls.fetch_add(1);
    const auto* cmd = static_cast<const char16_t*>(lpCmdLine);
    std::vector<std::u16string> args;
    if (cmd == nullptr || *cmd == u'\0') {
        args.emplace_back(u"game.exe");
    } else {
        size_t i = 0;
        // argv[0]: hasta el siguiente espacio, o entre comillas (sin reglas de '\').
        std::u16string a0;
        if (cmd[0] == u'"') {
            for (i = 1; cmd[i] != u'\0' && cmd[i] != u'"'; ++i) a0.push_back(cmd[i]);
            if (cmd[i] == u'"') ++i;
        } else {
            for (; cmd[i] != u'\0' && cmd[i] != u' ' && cmd[i] != u'\t'; ++i) a0.push_back(cmd[i]);
        }
        args.push_back(a0);
        for (;;) {
            while (cmd[i] == u' ' || cmd[i] == u'\t') ++i;
            if (cmd[i] == u'\0') break;
            std::u16string cur;
            bool quoted = false;
            while (cmd[i] != u'\0' && (quoted || (cmd[i] != u' ' && cmd[i] != u'\t'))) {
                size_t bs = 0;
                while (cmd[i] == u'\\') { ++bs; ++i; }
                if (cmd[i] == u'"') {
                    cur.append(bs / 2, u'\\');
                    if (bs % 2 == 1) { cur.push_back(u'"'); ++i; continue; }
                    if (quoted && cmd[i + 1] == u'"') { cur.push_back(u'"'); i += 2; continue; }
                    quoted = !quoted;
                    ++i;
                } else {
                    cur.append(bs, u'\\');
                    if (cmd[i] != u'\0' && (quoted || (cmd[i] != u' ' && cmd[i] != u'\t'))) cur.push_back(cmd[i++]);
                }
            }
            args.push_back(cur);
        }
    }
    size_t chars = 0;
    for (const auto& a : args) chars += a.size() + 1;
    const size_t ptrBytes = (args.size() + 1) * sizeof(char16_t*);
    auto* block = static_cast<uint8_t*>(win_LocalAlloc(0x40, ptrBytes + chars * sizeof(char16_t)));
    if (block == nullptr) return nullptr;
    auto** argv = reinterpret_cast<char16_t**>(block);
    auto* text = reinterpret_cast<char16_t*>(block + ptrBytes);
    for (size_t k = 0; k < args.size(); ++k) {
        argv[k] = text;
        std::memcpy(text, args[k].data(), args[k].size() * sizeof(char16_t));
        text += args[k].size();
        *text++ = u'\0';
    }
    argv[args.size()] = nullptr;
    if (pNumArgs) *pNumArgs = static_cast<int32_t>(args.size());
    return argv;
}

namespace {
std::string getKnownFolderPath(const void* rfid, int32_t csidl) {
    std::string path = (rfid != nullptr) ? win32::knownFolderFromGuid(rfid)
                                         : win32::knownFolderFromCsidl(csidl);

    if (s_activeImage != nullptr) {
        const std::string real = BionicPeLoader::resolveVirtualPath(*s_activeImage, path);
        std::error_code ec;
        std::filesystem::create_directories(real, ec);
    }
    return path;
}
} // namespace

int32_t NativeDllInterceptor::win_SHGetKnownFolderPath(const void* rfid, uint32_t, void*, void** ppszPath) {
    s_otherCalls.fetch_add(1);
    if (!ppszPath) return static_cast<int32_t>(0x80070057); // E_INVALIDARG
    lowmem::storePtr(ppszPath, uint64_t{0});

    const std::string path = getKnownFolderPath(rfid, -1);
    const size_t numChars = guestWStringLength(path) + 1;
    const size_t numBytes = numChars * sizeof(uint16_t);

    void* mem = win_HeapAlloc(win_GetProcessHeap(), 0, numBytes);
    if (!mem) return static_cast<int32_t>(0x8007000E); // E_OUTOFMEMORY
    writeGuestWString(static_cast<wchar_t*>(mem), numChars, path);
    lowmem::storePtr(ppszPath, mem);
    LOGI("win_SHGetKnownFolderPath: rfid=%p -> '%s' (ptr=%p, S_OK)", rfid, path.c_str(), mem);
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_SHGetFolderPathW(void*, int32_t csidl, void*, uint32_t, void* pszPath) {
    s_otherCalls.fetch_add(1);
    if (!pszPath) return static_cast<int32_t>(0x80070057); // E_INVALIDARG
    const std::string path = getKnownFolderPath(nullptr, csidl);
    writeGuestWString(static_cast<wchar_t*>(pszPath), 260, path);
    LOGI("win_SHGetFolderPathW: csidl=0x%x -> '%s' (S_OK)", csidl, path.c_str());
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_SHGetFolderPathA(void*, int32_t csidl, void*, uint32_t, char* pszPath) {
    s_otherCalls.fetch_add(1);
    if (!pszPath) return static_cast<int32_t>(0x80070057); // E_INVALIDARG
    const std::string path = getKnownFolderPath(nullptr, csidl);
    std::strncpy(pszPath, path.c_str(), 260);
    pszPath[259] = '\0';
    LOGI("win_SHGetFolderPathA: csidl=0x%x -> '%s' (S_OK)", csidl, pszPath);
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_SHGetSpecialFolderPathW(void* hwnd, void* pszPath, int32_t csidl, int32_t) {
    return (win_SHGetFolderPathW(hwnd, csidl, nullptr, 0, pszPath) == 0) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_SHGetSpecialFolderPathA(void* hwnd, char* pszPath, int32_t csidl, int32_t) {
    return (win_SHGetFolderPathA(hwnd, csidl, nullptr, 0, pszPath) == 0) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_ShellExecuteW(void*, const void*, const void*, const void*, const void*, int32_t) {
    s_otherCalls.fetch_add(1);
    return 42; // > 32 indica éxito en ShellExecute
}

int32_t NativeDllInterceptor::win_ShellExecuteA(void*, const char*, const char*, const char*, const char*, int32_t) {
    s_otherCalls.fetch_add(1);
    return 42;
}

int32_t NativeDllInterceptor::win_DllGetVersion(void* pdvi) {
    s_otherCalls.fetch_add(1);
    // E_POINTER / E_INVALIDARG como HRESULT de 32 bits.
    if (pdvi == nullptr || !isGuestRangeReadable(pdvi, sizeof(uint32_t))) {
        return static_cast<int32_t>(0x80004003u);
    }
    uint32_t cbSize = 0;
    std::memcpy(&cbSize, pdvi, sizeof(cbSize));
    constexpr uint32_t kMinInfoSize = 20; // DLLVERSIONINFO
    if (cbSize < kMinInfoSize || !isGuestRangeReadable(pdvi, cbSize)) {
        return static_cast<int32_t>(0x80070057u);
    }

    struct DllVersionInfo {
        uint32_t cbSize;
        uint32_t dwMajorVersion;
        uint32_t dwMinorVersion;
        uint32_t dwBuildNumber;
        uint32_t dwPlatformID;
    };
    struct DllVersionInfo2 {
        DllVersionInfo info1;
        uint32_t dwFlags;
        uint64_t ullVersion;
    };
    DllVersionInfo2 out{};
    out.info1.cbSize = kMinInfoSize;
    out.info1.dwMajorVersion = 6;
    out.info1.dwMinorVersion = 1;
    out.info1.dwBuildNumber = 7601;
    out.info1.dwPlatformID = 2; // VER_PLATFORM_WIN32_NT
    out.dwFlags = 0;
    out.ullVersion = (6ull << 48) | (1ull << 32) | (7601ull << 16);

    const size_t bytesToWrite = std::min<size_t>(cbSize, sizeof(out));
    std::memcpy(pdvi, &out, bytesToWrite);
    return 0; // S_OK
}

// --- CreateDirectory ------------------------------------------------------------------------

static int32_t createDirectoryUtf8(const std::string& rawPath) {
    if (s_activeImage == nullptr || rawPath.empty()) {
        NativeDllInterceptor::win_SetLastError(3);  // ERROR_PATH_NOT_FOUND
        return 0;
    }
    const std::string real = BionicPeLoader::resolveVirtualPath(*s_activeImage, rawPath);
    if (mkdir(real.c_str(), 0755) == 0) {
        LOGI("CreateDirectory('%s') -> '%s' (creado)", rawPath.c_str(), real.c_str());
        return 1;
    }
    const int err = errno;
    // ERROR_ALREADY_EXISTS (183) / ERROR_PATH_NOT_FOUND (3) / ERROR_ACCESS_DENIED (5)
    const uint32_t winErr = err == EEXIST ? 183u : (err == ENOENT || err == ENOTDIR) ? 3u : 5u;
    NativeDllInterceptor::win_SetLastError(winErr);
    LOGI("CreateDirectory('%s') -> '%s' falla errno=%d -> Win32 %u", rawPath.c_str(), real.c_str(), err, winErr);
    return 0;
}

int32_t NativeDllInterceptor::win_CreateDirectoryW(const void* lpPathName, void* /*lpSecurityAttributes*/) {
    s_kernel32Calls.fetch_add(1);
    if (lpPathName == nullptr) {
        win_SetLastError(3);
        return 0;
    }
    return createDirectoryUtf8(utf16ToUtf8(static_cast<const char16_t*>(lpPathName)));
}

int32_t NativeDllInterceptor::win_CreateDirectoryA(const char* lpPathName, void* /*lpSecurityAttributes*/) {
    s_kernel32Calls.fetch_add(1);
    if (lpPathName == nullptr) {
        win_SetLastError(3);
        return 0;
    }
    return createDirectoryUtf8(lpPathName);
}

// --- SHLWAPI ------------------------------------------------------------------------------

namespace {
inline char16_t* asW(void* p) { return static_cast<char16_t*>(p); }
inline const char16_t* asW(const void* p) { return static_cast<const char16_t*>(p); }
}  // namespace

int32_t NativeDllInterceptor::win_PathCanonicalizeW(void* pszBuf, const void* pszPath) {
    s_otherCalls.fetch_add(1);
    if (pszBuf == nullptr || pszPath == nullptr) {
        win_SetLastError(87);  // ERROR_INVALID_PARAMETER
        return 0;
    }
    const std::string in = utf16ToUtf8(asW(pszPath));
    if (!win32path::canonicalizeInto(asW(pszBuf), asW(pszPath))) {
        win_SetLastError(122);  // ERROR_INSUFFICIENT_BUFFER
        LOGW("PathCanonicalizeW('%s') -> FALSE (excede MAX_PATH)", in.c_str());
        return 0;
    }
    LOGI("PathCanonicalizeW('%s') -> '%s'", in.c_str(), utf16ToUtf8(asW(pszBuf)).c_str());
    return 1;
}

int32_t NativeDllInterceptor::win_PathCanonicalizeA(char* pszBuf, const char* pszPath) {
    s_otherCalls.fetch_add(1);
    if (pszBuf == nullptr || pszPath == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    if (!win32path::canonicalizeInto(pszBuf, pszPath)) {
        win_SetLastError(122);
        return 0;
    }
    LOGI("PathCanonicalizeA -> '%s'", pszBuf);
    return 1;
}

void* NativeDllInterceptor::win_PathCombineW(void* pszDest, const void* pszDir, const void* pszFile) {
    s_otherCalls.fetch_add(1);
    if (!win32path::combineInto(asW(pszDest), asW(pszDir), asW(pszFile))) {
        if (pszDest != nullptr) asW(pszDest)[0] = 0;
        return nullptr;
    }
    LOGI("PathCombineW -> '%s'", utf16ToUtf8(asW(pszDest)).c_str());
    return pszDest;
}

char* NativeDllInterceptor::win_PathCombineA(char* pszDest, const char* pszDir, const char* pszFile) {
    s_otherCalls.fetch_add(1);
    if (!win32path::combineInto(pszDest, pszDir, pszFile)) {
        if (pszDest != nullptr) pszDest[0] = 0;
        return nullptr;
    }
    return pszDest;
}

int32_t NativeDllInterceptor::win_PathAppendW(void* pszPath, const void* pszMore) {
    s_otherCalls.fetch_add(1);
    const bool ok = win32path::appendInto(asW(pszPath), asW(pszMore));
    if (ok) LOGI("PathAppendW -> '%s'", utf16ToUtf8(asW(pszPath)).c_str());
    return ok ? 1 : 0;
}

int32_t NativeDllInterceptor::win_PathAppendA(char* pszPath, const char* pszMore) {
    s_otherCalls.fetch_add(1);
    return win32path::appendInto(pszPath, pszMore) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_PathRemoveFileSpecW(void* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::removeFileSpec(asW(pszPath)) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_PathRemoveFileSpecA(char* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::removeFileSpec(pszPath) ? 1 : 0;
}

void* NativeDllInterceptor::win_PathFindFileNameW(void* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::findFileName(asW(pszPath));
}

char* NativeDllInterceptor::win_PathFindFileNameA(char* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::findFileName(pszPath);
}

void* NativeDllInterceptor::win_PathFindExtensionW(void* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::findExtension(asW(pszPath));
}

char* NativeDllInterceptor::win_PathFindExtensionA(char* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::findExtension(pszPath);
}

int32_t NativeDllInterceptor::win_PathFileExistsW(const void* pszPath) {
    if (pszPath == nullptr) return 0;
    return win_GetFileAttributesW(pszPath) != 0xFFFFFFFFu ? 1 : 0;  // INVALID_FILE_ATTRIBUTES
}

int32_t NativeDllInterceptor::win_PathFileExistsA(const char* pszPath) {
    if (pszPath == nullptr) return 0;
    return win_GetFileAttributesA(pszPath) != 0xFFFFFFFFu ? 1 : 0;
}

int32_t NativeDllInterceptor::win_PathIsRelativeW(const void* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::isRelative(asW(pszPath)) ? 1 : 0;
}

int32_t NativeDllInterceptor::win_PathIsRelativeA(const char* pszPath) {
    s_otherCalls.fetch_add(1);
    return win32path::isRelative(pszPath) ? 1 : 0;
}

namespace {
struct WinMemoryStatusEx {
    uint32_t dwLength;
    uint32_t dwMemoryLoad;
    uint64_t ullTotalPhys;
    uint64_t ullAvailPhys;
    uint64_t ullTotalPageFile;
    uint64_t ullAvailPageFile;
    uint64_t ullTotalVirtual;
    uint64_t ullAvailVirtual;
    uint64_t ullAvailExtendedVirtual;
};
} // namespace

int32_t NativeDllInterceptor::win_GlobalMemoryStatusEx(void* lpBuffer) {
    s_kernel32Calls.fetch_add(1);
    if (!lpBuffer) return 0;
    auto* ms = static_cast<WinMemoryStatusEx*>(lpBuffer);
    ms->dwLength = sizeof(WinMemoryStatusEx);
    ms->dwMemoryLoad = 25;
    ms->ullTotalPhys = 16ULL * 1024 * 1024 * 1024;
    ms->ullAvailPhys = 12ULL * 1024 * 1024 * 1024;
    ms->ullTotalPageFile = 24ULL * 1024 * 1024 * 1024;
    ms->ullAvailPageFile = 20ULL * 1024 * 1024 * 1024;
    // 32 bits con LARGE_ADDRESS_AWARE: 4 GB de espacio de direcciones (como WOW64).
    ms->ullTotalVirtual = lowmem::enabled() ? 0xFFFE0000ull : 1ULL << 40;
    ms->ullAvailVirtual = lowmem::enabled() ? 0xE0000000ull : 1ULL << 40;
    ms->ullAvailExtendedVirtual = 0;
    return 1;
}

int32_t NativeDllInterceptor::win_CreatePipe(void** phReadPipe, void** phWritePipe, void*, uint32_t) {
    s_kernel32Calls.fetch_add(1);
    int fds[2];
    if (pipe(fds) == 0) {
        lowmem::storePtr(phReadPipe, handleFromFd(fds[0]));
        lowmem::storePtr(phWritePipe, handleFromFd(fds[1]));
        return 1;
    }
    lowmem::storePtr(phReadPipe, handleFromFd(100));
    lowmem::storePtr(phWritePipe, handleFromFd(101));
    return 1;
}

uint32_t NativeDllInterceptor::win_SetErrorMode(uint32_t uMode) {
    s_kernel32Calls.fetch_add(1);
    return uMode;
}

int32_t NativeDllInterceptor::win_RoInitialize(uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_CoInitialize(void*) {
    s_otherCalls.fetch_add(1);
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_CoInitializeEx(void*, uint32_t) {
    s_otherCalls.fetch_add(1);
    return 0; // S_OK
}

void NativeDllInterceptor::win_CoUninitialize() {
    s_otherCalls.fetch_add(1);
}

namespace {
int32_t wasapiCreateEnumerator(uint32_t iidData1, void** ppv);
uint32_t wasapi32ToGuest(void* host);  // proxy de 32 bits de un objeto WASAPI (más abajo)
}  // namespace

int32_t NativeDllInterceptor::win_CoCreateInstance(const void* rclsid, void* pUnkOuter, uint32_t,
                                                   const void* riid, void** ppv) {
    s_otherCalls.fetch_add(1);
    if (ppv == nullptr) return static_cast<int32_t>(0x80004003);  // E_POINTER
    // Tamaño de puntero del invitado: en 32 bits escribir 8 bytes pisaba lo que sigue a ppv en la pila
    // (el EBP guardado de Wwise en DmC).
    lowmem::storePtr(ppv, uint64_t{0});
    if (rclsid == nullptr || riid == nullptr) return static_cast<int32_t>(0x80070057);  // E_INVALIDARG

    const auto* c = static_cast<const uint8_t*>(rclsid);
    const auto* i = static_cast<const uint8_t*>(riid);
    const uint32_t clsData1 = static_cast<uint32_t>(c[0]) | (static_cast<uint32_t>(c[1]) << 8) |
                              (static_cast<uint32_t>(c[2]) << 16) | (static_cast<uint32_t>(c[3]) << 24);
    const uint32_t iidData1 = static_cast<uint32_t>(i[0]) | (static_cast<uint32_t>(i[1]) << 8) |
                              (static_cast<uint32_t>(i[2]) << 16) | (static_cast<uint32_t>(i[3]) << 24);
    // MMDeviceEnumerator: el objeto WASAPI es de 64 bits (vive por encima de 4 GB); un invitado de 32 bits
    // lo truncaba y fallaba. Hasta tener audio de 32 bits: "clase no registrada" (el juego sigue sin audio).
    // DirectSound8 / DirectSound en 32 bits (Wwise cuando no hay XAudio2): IDirectSound8 sobre AAudio.
    if ((clsData1 == 0x3901CC3Fu || clsData1 == 0x47D4D946u) && s_activeImage != nullptr && s_activeImage->is32Bit) {
        if (ppv == nullptr) return static_cast<int32_t>(0x80004003);
        const uint32_t dev = dsound32::create();
        *reinterpret_cast<uint32_t*>(ppv) = dev;
        return dev != 0 ? 0 : static_cast<int32_t>(0x8007000E);
    }
    // MMDeviceEnumerator en 32 bits: el objeto real es de 64 bits; el invitado recibe su proxy de 32 bits
    // (vtable < 4 GB con las aridades exactas, ver wasapi32).
    if (clsData1 == 0xBCDE0395u && s_activeImage != nullptr && s_activeImage->is32Bit) {
        if (ppv == nullptr) return static_cast<int32_t>(0x80004003);  // E_POINTER
        void* host = nullptr;
        const int32_t r = wasapiCreateEnumerator(iidData1, &host);
        const uint32_t guest = r == 0 ? wasapi32ToGuest(host) : 0u;
        *reinterpret_cast<uint32_t*>(ppv) = guest;
        LOGI("CoCreateInstance(MMDeviceEnumerator) 32 bits -> proxy 0x%x (hr=0x%x)", guest, static_cast<uint32_t>(r));
        return guest != 0 ? r : static_cast<int32_t>(0x80040154);
    }
    if (clsData1 == 0xBCDE0395u) return wasapiCreateEnumerator(iidData1, ppv);  // MMDeviceEnumerator
    LOGW("CoCreateInstance: CLSID={%08X-...} IID={%08X-...} outer=%p no registrado",
         clsData1, iidData1, pUnkOuter);
    // Honestidad estricta: nunca S_OK con *ppv == NULL. Si el CLSID no está implementado,
    // el motor debe poder caer a su ruta alternativa en vez de desreferenciar NULL.
    return static_cast<int32_t>(0x80040154);  // REGDB_E_CLASSNOTREG
}

void* NativeDllInterceptor::win_CoTaskMemAlloc(size_t cb) {
    s_otherCalls.fetch_add(1);
    return win_HeapAlloc(win_GetProcessHeap(), 0, cb);
}

void NativeDllInterceptor::win_CoTaskMemFree(void* pv) {
    s_otherCalls.fetch_add(1);
    if (pv) {
        win_HeapFree(win_GetProcessHeap(), 0, pv);
    }
}

void* NativeDllInterceptor::win_CoTaskMemRealloc(void* pv, size_t cb) {
    s_otherCalls.fetch_add(1);
    if (!pv) return win_CoTaskMemAlloc(cb);
    if (cb == 0) {
        win_CoTaskMemFree(pv);
        return nullptr;
    }
    return win_HeapReAlloc(win_GetProcessHeap(), 0, pv, cb);
}

int32_t NativeDllInterceptor::win_CoCreateGuid(void* pguid) {
    s_otherCalls.fetch_add(1);
    if (!pguid) return static_cast<int32_t>(0x80070057); // E_INVALIDARG
    uint8_t* b = static_cast<uint8_t*>(pguid);
    for (int i = 0; i < 16; ++i) {
        b[i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
    b[6] = (b[6] & 0x0F) | 0x40; // UUID v4
    b[8] = (b[8] & 0x3F) | 0x80;
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_PropVariantClear(void* pvar) {
    s_otherCalls.fetch_add(1);
    if (pvar) std::memset(pvar, 0, 24);
    return 0; // S_OK
}

int32_t NativeDllInterceptor::win_SetHandleInformation(void* hObject, uint32_t dwMask, uint32_t dwFlags) {
    s_kernel32Calls.fetch_add(1);
    (void)hObject;
    (void)dwMask;
    (void)dwFlags;
    return 1; // TRUE
}

void* NativeDllInterceptor::win_LocalFree(void* hMem) {
    s_kernel32Calls.fetch_add(1);
    safeHeapFree(hMem);
    return nullptr;
}

void* NativeDllInterceptor::win_LocalAlloc(uint32_t uFlags, size_t uBytes) {
    s_kernel32Calls.fetch_add(1);
    (void)uFlags;
    return safeHeapAlloc(uBytes);
}

void* NativeDllInterceptor::win_GlobalLock(void* hMem) {
    s_kernel32Calls.fetch_add(1);
    return hMem;
}

int32_t NativeDllInterceptor::win_GlobalUnlock(void* hMem) {
    s_kernel32Calls.fetch_add(1);
    (void)hMem;
    return 0;
}

int32_t NativeDllInterceptor::win_DuplicateHandle(void* hSourceProcessHandle, void* hSourceHandle,
                                                 void* hTargetProcessHandle, void** lpTargetHandle,
                                                 uint32_t dwDesiredAccess, int32_t bInheritHandle,
                                                 uint32_t dwOptions) {
    s_kernel32Calls.fetch_add(1);
    (void)hSourceProcessHandle;
    (void)hTargetProcessHandle;
    (void)dwDesiredAccess;
    (void)bInheritHandle;
    (void)dwOptions;
    if (lpTargetHandle != nullptr) {
        // DuplicateHandle(GetCurrentThread()) produce un handle REAL del hilo llamador (así es
        // como Mono/Boehm guardan el hilo para suspenderlo); el pseudo-handle no sirve fuera.
        const auto src = reinterpret_cast<uintptr_t>(hSourceHandle);
        // En 32 bits el pseudo-handle -2 llega como 0xFFFFFFFE (sin extender el signo).
        if (src == kPseudoCurrentThread || (lowmem::enabled() && src == 0xFFFFFFFEu)) {
            lowmem::storePtr(lpTargetHandle, makeThreadHandle(static_cast<int>(gettid())));
        } else {
            lowmem::storePtr(lpTargetHandle, hSourceHandle);
        }
    }
    return 1;
}

int32_t NativeDllInterceptor::win_InitializeProcThreadAttributeList(void* lpAttributeList, uint32_t dwAttributeCount,
                                                                    uint32_t dwFlags, size_t* lpSize) {
    s_kernel32Calls.fetch_add(1);
    (void)dwFlags;
    if (lpSize == nullptr) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }
    const size_t needed = 64 + static_cast<size_t>(dwAttributeCount) * 32;
    if (lpAttributeList == nullptr || *lpSize < needed) {
        *lpSize = needed;
        win_SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
        return 0;
    }
    std::memset(lpAttributeList, 0, needed);
    return 1;
}

int32_t NativeDllInterceptor::win_UpdateProcThreadAttribute(void* lpAttributeList, uint32_t dwFlags,
                                                            uintptr_t Attribute, void* lpValue, size_t cbSize,
                                                            void* lpPreviousValue, size_t* lpReturnSize) {
    s_kernel32Calls.fetch_add(1);
    (void)lpAttributeList;
    (void)dwFlags;
    (void)Attribute;
    (void)lpValue;
    (void)cbSize;
    (void)lpPreviousValue;
    (void)lpReturnSize;
    return 1;
}

void NativeDllInterceptor::win_DeleteProcThreadAttributeList(void* lpAttributeList) {
    s_kernel32Calls.fetch_add(1);
    (void)lpAttributeList;
}

// -----------------------------------------------------------------------------
// Errores, Excepciones y Unwind
// -----------------------------------------------------------------------------
thread_local uint32_t t_winLastError = 0;

uint32_t NativeDllInterceptor::win_GetLastError() {
    if (t_guestTeb != nullptr) {
        auto* t = static_cast<uint8_t*>(t_guestTeb);
        if (t_isGuest64) {
            return *reinterpret_cast<const uint32_t*>(t + 0x68);
        } else {
            return *reinterpret_cast<const uint32_t*>(t + 0x34);
        }
    }
    return t_winLastError;
}

void NativeDllInterceptor::win_SetLastError(uint32_t dwErrCode) {
    t_winLastError = dwErrCode;
    if (t_guestTeb != nullptr) {
        auto* t = static_cast<uint8_t*>(t_guestTeb);
        if (t_isGuest64) {
            *reinterpret_cast<uint32_t*>(t + 0x68) = dwErrCode;
        } else {
            *reinterpret_cast<uint32_t*>(t + 0x34) = dwErrCode;
        }
    }
}

void* NativeDllInterceptor::win_SetUnhandledExceptionFilter(void* filter) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t previous = s_unhandledFilter.exchange(reinterpret_cast<uintptr_t>(filter));
    LOGI("SetUnhandledExceptionFilter(%p) -> anterior %p", filter, reinterpret_cast<void*>(previous));
    return reinterpret_cast<void*>(previous);
}

int32_t NativeDllInterceptor::win_UnhandledExceptionFilter(void* pointers) {
    s_kernel32Calls.fetch_add(1);
    // Registro del fallo (p. ej. __report_gsfailure del CRT): código, dirección y registros.
    if (pointers != nullptr && s_activeImage != nullptr && s_activeImage->is32Bit &&
        isGuestRangeReadable(pointers, 8)) {
        uint32_t ptrs[2] = {};
        std::memcpy(ptrs, pointers, 8);
        auto* rec = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[0]));
        auto* ctx = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[1]));
        uint32_t code = 0, addr = 0, eip = 0, esp = 0, ebp = 0;
        if (rec != nullptr && isGuestRangeReadable(rec, 16)) { std::memcpy(&code, rec, 4); std::memcpy(&addr, rec + 12, 4); }
        if (ctx != nullptr && isGuestRangeReadable(ctx, 0xCC)) {
            std::memcpy(&ebp, ctx + 0xB4, 4); std::memcpy(&eip, ctx + 0xB8, 4); std::memcpy(&esp, ctx + 0xC4, 4);
        }
        LOGW("UnhandledExceptionFilter[32]: código=0x%x dirección=0x%x EIP=0x%x ESP=0x%x EBP=0x%x",
             code, addr, eip, esp, ebp);
    }
    return 1; // EXCEPTION_EXECUTE_HANDLER
}

namespace {
/// Saca de un bloque de memoria las cadenas UTF-16 (>= 4 caracteres) y ASCII (>= 6).
void collectTexts(uint64_t base, size_t size, std::string& out) {
    if (base == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(base), size)) return;
    const auto* b = reinterpret_cast<const uint8_t*>(base);
    std::string run;
    for (size_t i = 0; i + 1 < size; i += 2) {  // UTF-16
        const uint16_t c = static_cast<uint16_t>(b[i] | (b[i + 1] << 8));
        if (c >= 0x20 && c < 0x7F) { run.push_back(static_cast<char>(c)); continue; }
        if (run.size() >= 4) out += " u\"" + run + "\"";
        run.clear();
    }
    if (run.size() >= 4) out += " u\"" + run + "\"";
    run.clear();
    for (size_t i = 0; i < size; ++i) {  // ASCII/UTF-8
        if (b[i] >= 0x20 && b[i] < 0x7F) { run.push_back(static_cast<char>(b[i])); continue; }
        if (run.size() >= 6) out += " \"" + run + "\"";
        run.clear();
    }
    if (run.size() >= 6) out += " \"" + run + "\"";
}

/// Textos del objeto y de los bloques a los que apuntan sus primeros campos (diagnóstico).
void logTextsNear(uint64_t obj) {
    if (obj == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(obj), 0x200)) return;
    std::string out;
    collectTexts(obj, 0x200, out);
    for (uint64_t off = 0; off < 0x200; off += 8) {
        uint64_t f = 0;
        std::memcpy(&f, reinterpret_cast<const void*>(obj + off), 8);
        if (f > 0x10000 && (f & 1) == 0 && f != obj) collectTexts(f, 0x100, out);
        if (out.size() > 900) break;
    }
    LOGW("[EE textos] %.1000s", out.c_str());
    char hex[16 * 20 + 1] = {};
    for (int i = 0; i < 16; ++i) {
        uint64_t q = 0;
        std::memcpy(&q, reinterpret_cast<const void*>(obj + static_cast<uint64_t>(i) * 8), 8);
        std::snprintf(hex + i * 20, 21, "%016llx    ", static_cast<unsigned long long>(q));
    }
    LOGW("[EE campos] %s", hex);
    // EETypeLoadException: tras InlineSString<1024> (~2 KB) van el ensamblado, el argumento y
    // m_resIDWhy (motivo). Se vuelcan los enteros de 32 bits pequeños de esa zona.
    if (isGuestRangeReadable(reinterpret_cast<const void*>(obj), 0xC00)) {
        std::string tail;
        for (uint64_t off = 0; off < 0xC00; off += 4) {
            uint32_t d = 0;
            std::memcpy(&d, reinterpret_cast<const void*>(obj + off), 4);
            if (d >= 0x1000 && d < 0x3000) tail += " +0x" + [](uint64_t v) { char b[16]; std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v)); return std::string(b); }(off) + "=0x" + [](uint32_t v) { char b[16]; std::snprintf(b, sizeof b, "%x", v); return std::string(b); }(d);
        }
        LOGW("[EE cola] %s", tail.c_str());
    }
}

}  // namespace

// =============================================================================
// SEH de 32 bits (x86): cadena de EXCEPTION_REGISTRATION_RECORD en fs:[0] ({Next, Handler},
// fin = 0xFFFFFFFF). Solo para invitados de 32 bits; los de 64 bits usan sehimpl (.pdata).
//
// Como en Windows, la fase de búsqueda corre como código del invitado (KiUserExceptionDispatcher):
// un catch de C++ salta a su continuación (_JumpToContinuation) y abandona el despachador sin
// dejar nada pendiente en la pila nativa. El desenrollado (RtlUnwind) es nativo: sus manejadores
// siempre vuelven.
// =============================================================================
namespace seh32 {

bool log() { return sehimpl::sehLog(); }

uint32_t tebOf(const CpuContext& c) { return static_cast<uint32_t>(c.fsBase); }
uint32_t rd32(uint32_t a) {
    uint32_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a)), 4);
    return v;
}
void wr32(uint32_t a, uint32_t v) { std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a)), &v, 4); }

// NtContinue(CONTEXT*, BOOLEAN): reanuda el hilo en ese contexto.
int32_t hp_NtContinue32(const uint8_t* ctx, int32_t) {
    if (ctx == nullptr || t_thunkCtx == nullptr) return static_cast<int32_t>(0xC000000Du);
    CpuContext c = *t_thunkCtx;
    win32ctx::fromContext(c, ctx);
    t_ctxOverrideValue = c;
    t_ctxOverride = true;
    return 0;
}

// Nombre del tipo de una excepción C++ de MSVC x86 (parámetros con punteros absolutos).
std::string cxxTypeName(const uint8_t* rec) {
    if (win32ctx::get32(rec, 0) != 0xE06D7363u || win32ctx::get32(rec, 0x10) < 3) return {};
    const uint32_t throwInfo = win32ctx::get32(rec, 0x14 + 8);
    if (throwInfo == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{throwInfo}), 16)) return {};
    const uint32_t cta = rd32(throwInfo + 12);
    if (cta == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{cta}), 8)) return {};
    const uint32_t ct = rd32(cta + 4);
    if (ct == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{ct}), 8)) return {};
    const uint32_t td = rd32(ct + 4);
    if (td == 0 || !isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{td} + 8), 64)) return {};
    return std::string(reinterpret_cast<const char*>(uintptr_t{td} + 8), strnlen(reinterpret_cast<const char*>(uintptr_t{td} + 8), 64));
}

// Ningún marco la manejó: filtro de SetUnhandledExceptionFilter si lo hay; si no (o si pide
// seguir buscando), se reanuda tras RaiseException como hacía el motor antes de tener SEH x86.
int32_t hp_Unhandled32(const uint8_t* rec, uint8_t* ctx) {
    const uint32_t code = rec != nullptr ? win32ctx::get32(rec, 0) : 0;
    const std::string type = rec != nullptr ? cxxTypeName(rec) : std::string();
    LOGE("[SEH32] excepción 0x%08x sin manejar en EIP=0x%x%s%s", code,
         ctx != nullptr ? win32ctx::get32(ctx, win32ctx::kCtxEip) : 0u, type.empty() ? "" : " tipo ", type.c_str());
    if (const uintptr_t filter = s_unhandledFilter.load(); filter != 0 && rec != nullptr && ctx != nullptr) {
        // EXCEPTION_POINTERS {rec, ctx} en la pila del invitado, por debajo del CONTEXT.
        const uint32_t ptrs = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ctx)) - 16;
        wr32(ptrs, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(rec)));
        wr32(ptrs + 4, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ctx)));
        const uint64_t arg = ptrs;
        const auto r = static_cast<int32_t>(sehimpl::callGuest(filter, &arg, 1));
        LOGE("[SEH32] UnhandledExceptionFilter -> %d", r);
        if (r == 1) {  // EXCEPTION_EXECUTE_HANDLER: el proceso termina
            NativeDllInterceptor::win_ExitProcess(code);
            return 0;
        }
    }
    // Fallo de hardware (acceso inválido, instrucción ilegal, división por cero, pila agotada) sin
    // manejar: Windows termina el proceso. Reanudar repetiría la misma instrucción para siempre.
    if (code == 0xC0000005u || code == 0xC000001Du || code == 0xC0000094u || code == 0xC00000FDu) {
        NativeDllInterceptor::win_ExitProcess(code);
        return 0;
    }
    return hp_NtContinue32(ctx, 0);
}

// Manejadores vectoriales (AddVectoredExceptionHandler), antes que los marcos, como en Windows.
// EXCEPTION_POINTERS {rec, ctx} en el hueco de 16 bytes entre el registro y el CONTEXT. Si uno
// devuelve EXCEPTION_CONTINUE_EXECUTION (-1) se reanuda en el CONTEXT (NtContinue); si no, 0.
int32_t hp_Vectored32(uint8_t* rec, uint8_t* ctx) {
    std::vector<std::pair<uintptr_t, uintptr_t>> handlers;
    {
        std::lock_guard<std::mutex> lk(s_vehMutex);
        handlers = s_vehHandlers;
    }
    if (handlers.empty() || rec == nullptr || ctx == nullptr) return 0;
    const auto recAddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(rec));
    const uint32_t ptrs = recAddr + static_cast<uint32_t>(win32ctx::kExceptionRecordSize);
    wr32(ptrs, recAddr);
    wr32(ptrs + 4, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ctx)));
    for (const auto& h : handlers) {
        const uint64_t arg = ptrs;
        const auto r = static_cast<int32_t>(sehimpl::callGuest(h.second, &arg, 1));
        if (log()) LOGI("[SEH32] VEH 0x%llx -> %d", static_cast<unsigned long long>(h.second), r);
        if (r == -1) return hp_NtContinue32(ctx, 0);
    }
    return 0;
}

// Código x86 del despachador (KiUserExceptionDispatcher + RtlDispatchException):
//   [esp] = EXCEPTION_RECORD*, [esp+4] = CONTEXT*
//   Vectored(rec, ctx); luego, para cada marco de fs:[0]: eax = Handler(rec, marco, ctx, &dc) (cdecl)
//     0 -> NtContinue(ctx)   1 -> siguiente marco   fin de la cadena -> Unhandled(rec, ctx)
uint32_t dispatcherStub() {
    static std::once_flag once;
    static uint32_t s_stub = 0;
    std::call_once(once, [] {
        void* mem = lowmem::mmapBelow4G(4096);
        if (mem == MAP_FAILED) return;
        auto* p = static_cast<uint8_t*>(mem);
        const auto base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
        const auto contAlias = static_cast<uint32_t>(w32alias::lowFor(w32arity::hpFn(&hp_NtContinue32), "ntdll.dll!NtContinue"));
        const auto unhAlias = static_cast<uint32_t>(w32alias::lowFor(w32arity::hpFn(&hp_Unhandled32), "ntdll.dll!SehUnhandled32"));
        const auto vehAlias = static_cast<uint32_t>(w32alias::lowFor(w32arity::hpFn(&hp_Vectored32), "ntdll.dll!SehVectored32"));
        const uint8_t code[] = {
            0x8B, 0x34, 0x24,                          // 00 mov esi, [esp]
            0x8B, 0x7C, 0x24, 0x04,                    // 03 mov edi, [esp+4]
            0x57,                                      // 07 push edi
            0x56,                                      // 08 push esi
            0xE8, 0, 0, 0, 0,                          // 09 call Vectored (stdcall; si maneja, no vuelve)
            0x64, 0x8B, 0x1D, 0x00, 0x00, 0x00, 0x00,  // 0E mov ebx, fs:[0]
            0x83, 0xFB, 0xFF,                          // 15 loop: cmp ebx, -1
            0x74, 0x22,                                // 18 je unhandled (3C)
            0x85, 0xDB,                                // 1A test ebx, ebx
            0x74, 0x1E,                                // 1C je unhandled (3C)
            0x6A, 0x00,                                // 1E push 0 (DispatcherContext)
            0x89, 0xE0,                                // 20 mov eax, esp
            0x50,                                      // 22 push eax
            0x57,                                      // 23 push edi
            0x53,                                      // 24 push ebx
            0x56,                                      // 25 push esi
            0xFF, 0x53, 0x04,                          // 26 call [ebx+4]
            0x83, 0xC4, 0x14,                          // 29 add esp, 0x14
            0x85, 0xC0,                                // 2C test eax, eax
            0x74, 0x04,                                // 2E je cont (34)
            0x8B, 0x1B,                                // 30 mov ebx, [ebx]
            0xEB, 0xE1,                                // 32 jmp loop (15)
            0x6A, 0x00,                                // 34 cont: push 0 (TestAlert)
            0x57,                                      // 36 push edi
            0xE8, 0, 0, 0, 0,                          // 37 call NtContinue
            0x57,                                      // 3C unhandled: push edi
            0x56,                                      // 3D push esi
            0xE8, 0, 0, 0, 0,                          // 3E call Unhandled
            0xCC,                                      // 43 (no vuelve: reanuda vía contexto)
        };
        std::memcpy(p, code, sizeof(code));
        auto rel = [&](size_t at, uint32_t target) {
            const uint32_t r = target - (base + static_cast<uint32_t>(at) + 5);
            std::memcpy(p + at + 1, &r, 4);
        };
        rel(0x09, vehAlias);
        rel(0x37, contAlias);
        rel(0x3E, unhAlias);
        mprotect(mem, 4096, PROT_READ);  // código del invitado: el JIT lo traduce, nadie lo escribe
        OryonEmitterBackend::addExecutableRangeAll(base, 4096);
        s_stub = base;
        LOGI("[SEH32] despachador x86 en 0x%x", base);
    });
    return s_stub;
}

// RaiseException de un invitado de 32 bits: EXCEPTION_RECORD y CONTEXT en su pila y salto al
// despachador. Devuelve false si no se pudo (se sigue como antes).
bool raise(uint32_t code, uint32_t flags, uint32_t n, const void* args) {
    if (t_thunkCtx == nullptr) return false;
    const uint32_t stub = dispatcherStub();
    if (stub == 0) return false;
    const CpuContext& c = *t_thunkCtx;
    const auto esp = static_cast<uint32_t>(c.gpr[4]);  // apunta a la dirección de retorno
    const uint32_t retEip = static_cast<uint32_t>(t_thunkRet);
    const uint32_t ctxAddr = (esp - 0x300u) & ~0xFu;
    const uint32_t recAddr = ctxAddr - 0x60u;
    const uint32_t newEsp = recAddr - 0x10u;
    auto* ctx = reinterpret_cast<uint8_t*>(uintptr_t{ctxAddr});
    auto* rec = reinterpret_cast<uint8_t*>(uintptr_t{recAddr});
    win32ctx::toContext(c, retEip, esp + 4 + 16, ctx);  // stdcall de 4 argumentos
    std::memset(rec, 0, win32ctx::kExceptionRecordSize);
    n = std::min<uint32_t>(n, 15);
    win32ctx::put32(rec, 0x00, code);
    win32ctx::put32(rec, 0x04, flags & win32ctx::kExceptionNoncontinuable);
    win32ctx::put32(rec, 0x0C, retEip);
    win32ctx::put32(rec, 0x10, n);
    if (n > 0 && args != nullptr) std::memcpy(rec + 0x14, args, n * 4);
    wr32(newEsp, recAddr);
    wr32(newEsp + 4, ctxAddr);
    if (log()) {
        const std::string type = cxxTypeName(rec);
        LOGI("[SEH32] RaiseException 0x%08x desde 0x%x%s%s; fs:[0]=0x%x", code, retEip, type.empty() ? "" : " tipo ",
             type.c_str(), rd32(tebOf(c)));
        // Objeto C++ lanzado: en las excepciones del CLR el HRESULT está en los primeros campos.
        uint32_t obj = n >= 2 ? win32ctx::get32(rec, 0x14 + 4) : 0u;
        // ".PAV..." = se lanza un puntero: el objeto real está un nivel más abajo.
        if (obj != 0 && type.rfind(".PAV", 0) == 0 && isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{obj}), 4))
            obj = rd32(obj);
        if (code == 0xE06D7363u && obj != 0 && isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{obj}), 64)) {
            std::string words;
            for (uint32_t k = 0; k < 64; k += 4) {
                char w[12];
                std::snprintf(w, sizeof(w), " %08x", rd32(obj + k));
                words += w;
            }
            LOGI("[SEH32]   objeto 0x%x:%s", obj, words.c_str());
            // throw L"texto" (Unreal: appErrorf/appThrowf): se muestra el texto lanzado.
            const uint32_t str = rd32(obj);
            if (type == ".PA_W" && str != 0 && isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{str}), 2)) {
                std::string text;
                for (uint32_t k = 0; k < 400; ++k) {
                    if (!isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{str + 2 * k}), 2)) break;
                    uint16_t ch = 0;
                    std::memcpy(&ch, reinterpret_cast<const void*>(uintptr_t{str + 2 * k}), 2);
                    if (ch == 0) break;
                    text.push_back(ch < 0x80 ? static_cast<char>(ch) : '?');
                }
                LOGW("[SEH32]   texto lanzado: \"%s\"", text.c_str());
            }
        }
    }
    CpuContext d = c;
    d.gpr[4] = newEsp;
    d.rip = stub;
    t_ctxOverrideValue = d;
    t_ctxOverride = true;
    return true;
}

// RtlUnwind(TargetFrame, TargetIp, ExceptionRecord, ReturnValue): llama a cada manejador por
// encima de TargetFrame con EXCEPTION_UNWINDING y lo saca de la cadena. Vuelve a quien llamó
// con EAX = ReturnValue (en x86 TargetIp no se usa para reanudar).
uint32_t hp_RtlUnwind32(uint32_t target, uint32_t, uint8_t* rec, uint32_t retVal) {
    if (t_thunkCtx == nullptr) return retVal;
    const CpuContext& c = *t_thunkCtx;
    const uint32_t teb = tebOf(c);
    const auto esp = static_cast<uint32_t>(c.gpr[4]);
    // Registro, CONTEXT y DispatcherContext en la pila del invitado, dentro del margen que deja
    // callGuestNested (2 KB) por debajo del ESP actual.
    const uint32_t ctxAddr = (esp - 0x400u) & ~0xFu;
    const uint32_t ownRec = ctxAddr - 0x60u;
    const uint32_t dcAddr = ownRec - 0x10u;
    auto* ctx = reinterpret_cast<uint8_t*>(uintptr_t{ctxAddr});
    win32ctx::toContext(c, static_cast<uint32_t>(t_thunkRet), esp + 4 + 16, ctx);
    if (rec == nullptr) {
        rec = reinterpret_cast<uint8_t*>(uintptr_t{ownRec});
        std::memset(rec, 0, win32ctx::kExceptionRecordSize);
        win32ctx::put32(rec, 0x00, 0xC0000027u);  // STATUS_UNWIND
        win32ctx::put32(rec, 0x0C, static_cast<uint32_t>(t_thunkRet));
    }
    uint32_t fl = win32ctx::get32(rec, 0x04) | win32ctx::kExceptionUnwinding;
    if (target == 0) fl |= win32ctx::kExceptionExitUnwind;
    win32ctx::put32(rec, 0x04, fl);
    const auto recPtr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(rec));
    uint32_t frame = rd32(teb);
    int guard = 0;
    while (frame != 0xFFFFFFFFu && frame != 0 && frame != target && guard++ < 4096) {
        const uint32_t handler = rd32(frame + 4);
        if (log()) LOGI("[SEH32] RtlUnwind: marco 0x%x manejador 0x%x (destino 0x%x)", frame, handler, target);
        wr32(dcAddr, 0);
        const uint64_t a[4] = {recPtr, frame, ctxAddr, dcAddr};
        sehimpl::callGuest(handler, a, 4);
        const uint32_t next = rd32(frame);
        wr32(teb, next);  // RtlpUnlinkHandler
        frame = next;
    }
    return retVal;
}

}  // namespace seh32

bool NativeDllInterceptor::prepareGuestException32(CpuContext& ctx, uint32_t code, uint32_t address, uint32_t access) {
    const uint32_t stub = seh32::dispatcherStub();
    if (stub == 0) return false;
    const auto eip = static_cast<uint32_t>(ctx.rip);
    const auto esp = static_cast<uint32_t>(ctx.gpr[4]);
    const uint32_t ctxAddr = (esp - 0x300u) & ~0xFu;
    const uint32_t recAddr = ctxAddr - 0x60u;
    const uint32_t newEsp = recAddr - 0x10u;
    if (!isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{newEsp}), esp - newEsp)) return false;
    auto* cx = reinterpret_cast<uint8_t*>(uintptr_t{ctxAddr});
    auto* rec = reinterpret_cast<uint8_t*>(uintptr_t{recAddr});
    win32ctx::toContext(ctx, eip, esp, cx);
    std::memset(rec, 0, win32ctx::kExceptionRecordSize);
    win32ctx::put32(rec, 0x00, code);
    win32ctx::put32(rec, 0x0C, eip);
    if (code == 0xC0000005u) {  // EXCEPTION_ACCESS_VIOLATION: {0 lectura / 1 escritura / 8 ejecución, dirección}
        win32ctx::put32(rec, 0x10, 2);
        win32ctx::put32(rec, 0x14, access);
        win32ctx::put32(rec, 0x18, address);
    }
    seh32::wr32(newEsp, recAddr);
    seh32::wr32(newEsp + 4, ctxAddr);
    static std::atomic<int> s_logs{0};
    if (s_logs.fetch_add(1) < 40) {
        LOGI("[SEH32] fallo hardware 0x%08x en EIP=0x%x dir=0x%x -> despachador", code, eip, address);
        // Pila en el fallo: direcciones de retorno candidatas (para saber quién llamó).
        std::string words;
        for (uint32_t k = 0; k < 24; ++k) {
            if (!isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{esp + 4 * k}), 4)) break;
            char w[12];
            std::snprintf(w, sizeof(w), " %08x", seh32::rd32(esp + 4 * k));
            words += w;
        }
        // Memoria en [ESI] (estructura con la que trabajaba el código que falló).
        {
            const uint32_t esiv = static_cast<uint32_t>(ctx.gpr[6]);
            std::string mem;
            for (uint32_t k = 0; k < 32 && isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{esiv + 4 * k}), 4); ++k) {
                char w[12];
                std::snprintf(w, sizeof(w), " %08x", seh32::rd32(esiv + 4 * k));
                mem += w;
            }
            LOGI("[SEH32]   [ESI]:%s", mem.c_str());
            // Nodos a los que apunta [ESI+4..] (árbol de dlmalloc): 24 palabras desde nodo-0x10.
            for (uint32_t k = 1; k < 33; ++k) {
                if (!isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{esiv + 4 * k}), 4)) break;
                const uint32_t node = seh32::rd32(esiv + 4 * k);
                if (node < 0x10000 || !isGuestRangeReadable(reinterpret_cast<const void*>(uintptr_t{node - 0x10}), 0x60)) continue;
                std::string nm;
                for (uint32_t q = 0; q < 24; ++q) {
                    char w[12];
                    std::snprintf(w, sizeof(w), " %08x", seh32::rd32(node - 0x10 + 4 * q));
                    nm += w;
                }
                LOGI("[SEH32]   nodo[%u]=0x%x desde -0x10:%s", k - 1, node, nm.c_str());
            }
        }
        // Mapeos de /proc/self/maps que contienen ESI/EAX (de dónde sale la memoria del fallo).
        static std::atomic<int> s_mapDumps{0};
        if (FILE* mf = s_mapDumps.fetch_add(1) < 3 ? std::fopen("/proc/self/maps", "r") : nullptr) {
            char line[512];
            const uint64_t want[2] = {ctx.gpr[6] & 0xFFFFFFFFull, ctx.gpr[0] & 0xFFFFFFFFull};
            while (std::fgets(line, sizeof(line), mf) != nullptr) {
                unsigned long long lo = 0, hi = 0;
                if (std::sscanf(line, "%llx-%llx", &lo, &hi) != 2) continue;
                for (uint64_t w : want) {
                    if (w >= lo && w < hi) { LOGI("[SEH32]   mapa de 0x%llx: %s", static_cast<unsigned long long>(w), line); break; }
                }
                // vecinos de ±1 MB para ver qué hay alrededor
                if (lo <= want[0] + 0x100000 && hi + 0x100000 >= want[0]) LOGI("[SEH32]   cerca: %s", line);
            }
            std::fclose(mf);
        }
        LOGI("[SEH32]   EAX=%x EBX=%x ECX=%x EDX=%x ESI=%x EDI=%x EBP=%x ESP=%x pila:%s",
             static_cast<uint32_t>(ctx.gpr[0]), static_cast<uint32_t>(ctx.gpr[3]), static_cast<uint32_t>(ctx.gpr[1]),
             static_cast<uint32_t>(ctx.gpr[2]), static_cast<uint32_t>(ctx.gpr[6]), static_cast<uint32_t>(ctx.gpr[7]),
             static_cast<uint32_t>(ctx.gpr[5]), esp, words.c_str());
    }
    ctx.gpr[4] = newEsp;
    ctx.rip = stub;
    return true;
}

void NativeDllInterceptor::win_RaiseException(uint32_t dwExceptionCode, uint32_t dwExceptionFlags,
                                             uint32_t nNumberOfArguments, const void* lpArguments) {
    s_kernel32Calls.fetch_add(1);
    // Invitado de 32 bits: SEH por la cadena fs:[0] (camino propio; el de 64 bits no cambia).
    if (s_activeImage != nullptr && s_activeImage->is32Bit) {
        if (seh32::raise(dwExceptionCode, dwExceptionFlags, nNumberOfArguments, lpArguments)) return;
        LOGE("[SEH32] RaiseException 0x%08x: sin contexto de thunk", dwExceptionCode);
        return;
    }
    static std::atomic<int> s_raiseLogs{0};
    if (s_raiseLogs.fetch_add(1) < 200) {
        LOGW("RaiseException invocado: código 0x%08x flags=0x%x argc=%u args=%p", dwExceptionCode, dwExceptionFlags,
             nNumberOfArguments, lpArguments);
    }
    // Excepción C++ de MSVC: tipo lanzado (ThrowInfo -> CatchableTypeArray -> TypeDescriptor).
    if (dwExceptionCode == 0xE06D7363u && nNumberOfArguments >= 4 && lpArguments != nullptr && s_raiseLogs.load() < 200) {
        const auto* p = static_cast<const uint64_t*>(lpArguments);
        const uint64_t throwInfo = p[2], imgBase = p[3];
        if (throwInfo != 0 && imgBase != 0 && isGuestRangeReadable(reinterpret_cast<const void*>(throwInfo), 16)) {
            int32_t ctaRva = 0;
            std::memcpy(&ctaRva, reinterpret_cast<const void*>(throwInfo + 12), 4);
            const uint64_t cta = imgBase + static_cast<uint32_t>(ctaRva);
            int32_t n = 0, ctRva = 0, tdRva = 0;
            std::memcpy(&n, reinterpret_cast<const void*>(cta), 4);
            std::memcpy(&ctRva, reinterpret_cast<const void*>(cta + 4), 4);
            std::memcpy(&tdRva, reinterpret_cast<const void*>(imgBase + static_cast<uint32_t>(ctRva) + 4), 4);
            const char* name = reinterpret_cast<const char*>(imgBase + static_cast<uint32_t>(tdRva) + 16);
            LOGW("RaiseException C++: tipo %.80s (%d tipos capturables)", name, n);
            // Excepciones del runtime de .NET (EETypeLoadException, EEFileLoadException...): el
            // nombre del tipo/ensamblado va en SString dentro del objeto; se registran los textos.
            if (std::strstr(name, "EE") != nullptr) {
                logTextsNear(p[1]);
                // Direcciones de retorno en la pila dentro del módulo que lanza (coreclr.dll):
                // RVAs para localizar en el binario quién lanzó y con qué motivo.
                if (OryonEmitterBackend* bk = sehimpl::backendHere(); bk != nullptr && t_thunkCtx != nullptr) {
                    const uint64_t rsp = t_thunkCtx->gpr[4];
                    std::string rvas;
                    int found = 0;
                    for (uint64_t a = rsp; a < rsp + 0x3000 && found < 24; a += 8) {
                        if (!isGuestRangeReadable(reinterpret_cast<const void*>(a), 8)) break;
                        uint64_t q = 0;
                        std::memcpy(&q, reinterpret_cast<const void*>(a), 8);
                        if (q > imgBase && q < imgBase + 0x600000) {
                            // Con el 5.º argumento de esa llamada ([rsp+0x20] del llamador = +0x28).
                            uint32_t arg5 = 0;
                            if (isGuestRangeReadable(reinterpret_cast<const void*>(a + 0x28), 4)) {
                                std::memcpy(&arg5, reinterpret_cast<const void*>(a + 0x28), 4);
                            }
                            char b[40];
                            std::snprintf(b, sizeof b, " %llx(a5=%x)", static_cast<unsigned long long>(q - imgBase), arg5);
                            rvas += b;
                            ++found;
                        }
                    }
                    LOGW("[EE pila] RVAs:%s", rvas.c_str());
                }
            }
        }
    }
    // Excepción .NET (CoreCLR, 'CCR'): args[0] es el objeto. Se registran los textos que lleva
    // (mensaje; en TypeLoadException el tipo y el ensamblado van en campos propios porque el
    // mensaje se compone al leerlo) y los de la excepción interna. Diagnóstico de Stardew Valley.
    if (dwExceptionCode == 0xE0434352u && nNumberOfArguments >= 1 && lpArguments != nullptr && s_raiseLogs.load() < 200) {
        const auto* a = static_cast<const uint64_t*>(lpArguments);
        LOGW("[.NET args] %llx %llx %llx %llx %llx", static_cast<unsigned long long>(a[0]),
             static_cast<unsigned long long>(nNumberOfArguments > 1 ? a[1] : 0),
             static_cast<unsigned long long>(nNumberOfArguments > 2 ? a[2] : 0),
             static_cast<unsigned long long>(nNumberOfArguments > 3 ? a[3] : 0),
             static_cast<unsigned long long>(nNumberOfArguments > 4 ? a[4] : 0));
    }
    // EXCEPTION_RECORD + CONTEXT del llamador, y despacho SEH completo (VEH, marcos, filtro).
    if (OryonEmitterBackend* b = sehimpl::backendHere(); b != nullptr && t_thunkCtx != nullptr) {
        alignas(16) uint8_t rec[win64ctx::kExceptionRecordSize] = {};
        alignas(16) sehimpl::Ctx ctx;
        sehimpl::callerContext(ctx.data());
        const uint32_t flags = dwExceptionFlags & seh::kEhNonContinuable;
        const uint32_t n = std::min<uint32_t>(nNumberOfArguments, 15);
        std::memcpy(rec, &dwExceptionCode, 4);
        std::memcpy(rec + 4, &flags, 4);
        const uint64_t addr = t_thunkRet;
        std::memcpy(rec + 0x10, &addr, 8);
        std::memcpy(rec + 0x18, &n, 4);
        if (n > 0 && lpArguments != nullptr) std::memcpy(rec + 0x20, lpArguments, n * 8);
        const sehimpl::Result r = sehimpl::raise(*b, rec, ctx.data());
        if (r == sehimpl::Result::kContinue || r == sehimpl::Result::kUnwound) {
            CpuContext c = *t_thunkCtx;
            win64ctx::fromContext(c, ctx.data(), true);
            t_ctxOverrideValue = c;
            t_ctxOverride = true;
            return;
        }
        if (r == sehimpl::Result::kAborted) return;  // las llamadas anidadas se cortan hasta el destino
        LOGE("RaiseException 0x%08x sin manejar (ningún VEH ni marco la captura)", dwExceptionCode);
    }

    // Mono/GLib guarda el mensaje ya formateado en un global justo antes de
    // llamar a RaiseException(STATUS_ASSERTION_FAILURE). Volcarlo permite
    // identificar la aserción real sin depender del stderr del CRT de Windows.
    if (dwExceptionCode == 0xE0000001u && s_activeImage != nullptr) {
        for (const auto& kv : s_activeImage->localDlls) {
            if (kv.first.find("mono-2.0-bdwgc.dll") == std::string::npos) continue;
            const uintptr_t base = reinterpret_cast<uintptr_t>(kv.second.base);
            if (base == 0) continue;
            const char* message = *reinterpret_cast<const char* const*>(base + 0x7402D0);
            if (message != nullptr) {
                LOGW("Mono assertion formateada: %s", message);
            } else {
                LOGW("Mono assertion formateada: <null> (global en %p)",
                     reinterpret_cast<const void*>(base + 0x7402D0));
            }
            break;
        }
    }
}

void NativeDllInterceptor::win_RtlCaptureContext(void* ContextRecord) {
    s_ntdllCalls.fetch_add(1);
    if (ContextRecord != nullptr) sehimpl::callerContext(static_cast<uint8_t*>(ContextRecord));
}

void* NativeDllInterceptor::win_RtlLookupFunctionEntry(uint64_t ControlPc, uint64_t* ImageBase, void* HistoryTable) {
    s_ntdllCalls.fetch_add(1);
    (void)HistoryTable;
    sehimpl::initOnce();
    return const_cast<seh::RuntimeFunction*>(seh::lookup(ControlPc, ImageBase));
}

void* NativeDllInterceptor::win_RtlVirtualUnwind(uint32_t HandlerType, uint64_t ImageBase, uint64_t ControlPc,
                                                 void* FunctionEntry, void* ContextRecord, void** HandlerData,
                                                 uint64_t* EstablisherFrame, void* ContextPointers) {
    s_ntdllCalls.fetch_add(1);
    if (FunctionEntry == nullptr || ContextRecord == nullptr) return nullptr;
    uint64_t data = 0, frame = 0;
    const uint64_t h = seh::virtualUnwind(HandlerType, ImageBase, ControlPc,
                                          static_cast<const seh::RuntimeFunction*>(FunctionEntry),
                                          static_cast<uint8_t*>(ContextRecord), &data, &frame,
                                          static_cast<uint64_t*>(ContextPointers));
    if (HandlerData != nullptr) *HandlerData = reinterpret_cast<void*>(data);
    if (EstablisherFrame != nullptr) *EstablisherFrame = frame;
    return reinterpret_cast<void*>(h);
}

void NativeDllInterceptor::win_RtlUnwindEx(void* TargetFrame, void* TargetIp, void* ExceptionRecord, void* ReturnValue,
                                           void* ContextRecord, void*) {
    s_ntdllCalls.fetch_add(1);
    sehimpl::unwindEx(reinterpret_cast<uint64_t>(TargetFrame), reinterpret_cast<uint64_t>(TargetIp),
                      static_cast<uint8_t*>(ExceptionRecord), reinterpret_cast<uint64_t>(ReturnValue),
                      static_cast<uint8_t*>(ContextRecord));
}

namespace {
// Para la sonda del traductor (debug.hyperplay.probe): RVAs relativas al módulo del cargador.
[[maybe_unused]] const bool s_probeResolverSet = [] {
    OryonEmitterBackend::setModuleBaseResolver([](uint64_t pc) -> uint64_t {
        sehimpl::initOnce();
        return seh::moduleBaseOf(pc);
    });
    return true;
}();
}  // namespace

void* NativeDllInterceptor::win_RtlPcToFileHeader(void* PcValue, void** BaseOfImage) {
    s_ntdllCalls.fetch_add(1);
    sehimpl::initOnce();
    void* base = reinterpret_cast<void*>(seh::moduleBaseOf(reinterpret_cast<uint64_t>(PcValue)));
    lowmem::storePtr(BaseOfImage, base);
    return base;
}

void* NativeDllInterceptor::win_EncodePointer(void* Ptr) {
    return Ptr;
}

void* NativeDllInterceptor::win_DecodePointer(void* Ptr) {
    return Ptr;
}

void NativeDllInterceptor::win_InitializeSListHead(void* ListHead) {
    s_kernel32Calls.fetch_add(1);
    // SLIST_HEADER: 16 bytes en x64, 8 en x86 (con 16 se pisaba la variable siguiente).
    if (ListHead) std::memset(ListHead, 0, lowmem::enabled() ? 8 : 16);
}

// -----------------------------------------------------------------------------
// Sincronización, TLS y FLS
// -----------------------------------------------------------------------------
int32_t NativeDllInterceptor::win_InitializeCriticalSectionAndSpinCount(void* lpCriticalSection, uint32_t) {
    win_InitializeCriticalSection(lpCriticalSection);
    return 1;
}

int32_t NativeDllInterceptor::win_InitializeCriticalSectionEx(void* lpCriticalSection, uint32_t, uint32_t) {
    win_InitializeCriticalSection(lpCriticalSection);
    return 1;
}

// SRWLOCK (puntero, SRWLOCK_INIT = 0): bit 63 = exclusivo, bits bajos = lectores compartidos.
// Antes el modo compartido era exclusivo: dos lectores se excluían y una adquisición compartida
// recursiva (permitida en Windows) bloqueaba al propio hilo.
// El SRWLOCK es un puntero: 8 bytes en x64 (bit 63 = exclusivo) y 4 en 32 bits (bit 31). Un atómico
// de 8 bytes sobre el de 32 bits pisaba el campo vecino (y en ARM falla si no está alineado a 8).
namespace {
void srwBackoff(unsigned& spins) {
    if (++spins < 64) {
        sched_yield();
    } else {
        usleep(spins < 1024 ? 50 : 500);  // espera larga: no quemar un núcleo
    }
}

template <class T>
constexpr T srwExclusive() { return T{1} << (sizeof(T) * 8 - 1); }

template <class T>
void srwAcquireExclusive(void* p) {
    auto* lock = reinterpret_cast<std::atomic<T>*>(p);
    unsigned spins = 0;
    T expected = 0;
    while (!lock->compare_exchange_weak(expected, srwExclusive<T>(), std::memory_order_acquire)) {
        expected = 0;
        srwBackoff(spins);
    }
}
template <class T>
void srwReleaseExclusive(void* p) { reinterpret_cast<std::atomic<T>*>(p)->store(0, std::memory_order_release); }
template <class T>
int32_t srwTryExclusive(void* p) {
    T expected = 0;
    return reinterpret_cast<std::atomic<T>*>(p)->compare_exchange_strong(expected, srwExclusive<T>(),
                                                                       std::memory_order_acquire) ? 1 : 0;
}
template <class T>
void srwAcquireShared(void* p) {
    auto* lock = reinterpret_cast<std::atomic<T>*>(p);
    unsigned spins = 0;
    T cur = lock->load(std::memory_order_relaxed);
    for (;;) {
        if ((cur & srwExclusive<T>()) == 0) {
            if (lock->compare_exchange_weak(cur, cur + 1, std::memory_order_acquire)) return;
            continue;  // cur actualizado por el CAS
        }
        srwBackoff(spins);
        cur = lock->load(std::memory_order_relaxed);
    }
}
template <class T>
void srwReleaseShared(void* p) {
    auto* lock = reinterpret_cast<std::atomic<T>*>(p);
    T cur = lock->load(std::memory_order_relaxed);
    while ((cur & ~srwExclusive<T>()) != 0 &&
           !lock->compare_exchange_weak(cur, cur - 1, std::memory_order_release)) {
    }
}
template <class T>
int32_t srwTryShared(void* p) {
    auto* lock = reinterpret_cast<std::atomic<T>*>(p);
    T cur = lock->load(std::memory_order_relaxed);
    while ((cur & srwExclusive<T>()) == 0) {
        if (lock->compare_exchange_weak(cur, cur + 1, std::memory_order_acquire)) return 1;
    }
    return 0;
}
}  // namespace

void NativeDllInterceptor::win_AcquireSRWLockExclusive(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return;
    lowmem::enabled() ? srwAcquireExclusive<uint32_t>(SRWLock) : srwAcquireExclusive<uint64_t>(SRWLock);
}

void NativeDllInterceptor::win_ReleaseSRWLockExclusive(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return;
    lowmem::enabled() ? srwReleaseExclusive<uint32_t>(SRWLock) : srwReleaseExclusive<uint64_t>(SRWLock);
}

int32_t NativeDllInterceptor::win_TryAcquireSRWLockExclusive(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return 0;
    return lowmem::enabled() ? srwTryExclusive<uint32_t>(SRWLock) : srwTryExclusive<uint64_t>(SRWLock);
}

void NativeDllInterceptor::win_AcquireSRWLockShared(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return;
    lowmem::enabled() ? srwAcquireShared<uint32_t>(SRWLock) : srwAcquireShared<uint64_t>(SRWLock);
}

void NativeDllInterceptor::win_ReleaseSRWLockShared(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return;
    lowmem::enabled() ? srwReleaseShared<uint32_t>(SRWLock) : srwReleaseShared<uint64_t>(SRWLock);
}

int32_t NativeDllInterceptor::win_TryAcquireSRWLockShared(void* SRWLock) {
    s_kernel32Calls.fetch_add(1);
    if (!SRWLock) return 0;
    return lowmem::enabled() ? srwTryShared<uint32_t>(SRWLock) : srwTryShared<uint64_t>(SRWLock);
}

int32_t NativeDllInterceptor::win_SleepConditionVariableSRW(void* ConditionVariable, void* SRWLock, uint32_t dwMilliseconds, uint32_t Flags) {
    s_kernel32Calls.fetch_add(1);
    if (!ConditionVariable || !SRWLock) return 0;
    auto cv = getOrCreateCv(ConditionVariable);
    const bool shared = (Flags & 1); // CONDITION_VARIABLE_LOCKMODE_SHARED
    std::unique_lock<std::mutex> cvLock(cv->mtx);
    if (shared) win_ReleaseSRWLockShared(SRWLock);
    else win_ReleaseSRWLockExclusive(SRWLock);

    bool ok = true;
    if (dwMilliseconds == 0xFFFFFFFF) {
        cv->cv.wait(cvLock);
    } else {
        ok = (cv->cv.wait_for(cvLock, std::chrono::milliseconds(dwMilliseconds)) != std::cv_status::timeout);
    }
    cvLock.unlock();

    if (shared) win_AcquireSRWLockShared(SRWLock);
    else win_AcquireSRWLockExclusive(SRWLock);
    // Como Windows: un timeout devuelve FALSE con ERROR_TIMEOUT (Mono aborta con cualquier otro).
    win_SetLastError(ok ? 0 : 1460);
    return ok ? 1 : 0;
}

int32_t NativeDllInterceptor::win_SleepConditionVariableCS(void* ConditionVariable, void* CriticalSection, uint32_t dwMilliseconds) {
    s_kernel32Calls.fetch_add(1);
    if (!ConditionVariable || !CriticalSection) return 0;
    auto cv = getOrCreateCv(ConditionVariable);
    std::unique_lock<std::mutex> cvLock(cv->mtx);
    win_LeaveCriticalSection(CriticalSection);

    bool ok = true;
    if (dwMilliseconds == 0xFFFFFFFF) {
        cv->cv.wait(cvLock);
    } else {
        ok = (cv->cv.wait_for(cvLock, std::chrono::milliseconds(dwMilliseconds)) != std::cv_status::timeout);
    }
    cvLock.unlock();

    win_EnterCriticalSection(CriticalSection);
    win_SetLastError(ok ? 0 : 1460);  // ERROR_TIMEOUT (ver SleepConditionVariableSRW)
    return ok ? 1 : 0;
}

void NativeDllInterceptor::win_WakeAllConditionVariable(void* ConditionVariable) {
    s_kernel32Calls.fetch_add(1);
    if (!ConditionVariable) return;
    std::shared_ptr<WinCondVarObj> cv;
    {
        std::lock_guard<std::mutex> lk(s_cvMutex);
        auto it = s_condVars.find(ConditionVariable);
        if (it != s_condVars.end()) cv = it->second;
    }
    // Con el mutex de la CV: un durmiente que ya soltó el SRW/CS pero aún no está en wait()
    // lo tiene tomado, así que la notificación no puede perderse en esa ventana.
    if (cv) {
        std::lock_guard<std::mutex> lk(cv->mtx);
        cv->cv.notify_all();
    }
}

void NativeDllInterceptor::win_WakeConditionVariable(void* ConditionVariable) {
    s_kernel32Calls.fetch_add(1);
    if (!ConditionVariable) return;
    std::shared_ptr<WinCondVarObj> cv;
    {
        std::lock_guard<std::mutex> lk(s_cvMutex);
        auto it = s_condVars.find(ConditionVariable);
        if (it != s_condVars.end()) cv = it->second;
    }
    if (cv) {
        std::lock_guard<std::mutex> lk(cv->mtx);
        cv->cv.notify_one();
    }
}

uint32_t NativeDllInterceptor::win_WaitOnAddress(const void* Address, const void* CompareAddress,
                                                 size_t AddressSize, uint32_t dwMilliseconds) {
    s_kernel32Calls.fetch_add(1);
    // Sin comprobar residencia con mincore (2 syscalls + una asignación por llamada en un camino
    // muy caliente: el sistema de jobs de Unity espera aquí cientos de veces por segundo). En
    // Windows una dirección inválida también fallaría; solo se validan puntero y tamaño.
    uint64_t expected = 0;
    const bool sizeOk = AddressSize == 1 || AddressSize == 2 || AddressSize == 4 || AddressSize == 8;
    if (sizeOk && CompareAddress != nullptr) expected = loadGuestScalarAtomic(CompareAddress, AddressSize);
    if (!sizeOk || Address == nullptr || CompareAddress == nullptr) {
        LOGW("WaitOnAddress: dirección o comparación inválida (Address=%p, Compare=%p, size=%zu)",
             Address, CompareAddress, AddressSize);
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }

    auto waitObj = getOrCreateWaitAddress(Address);
    if (!waitObj) {
        win_SetLastError(87);
        return 0;
    }

    std::unique_lock<std::mutex> lk(waitObj->mtx);
    // Windows despierta con cualquier WakeByAddress* aunque el valor haya vuelto al original
    // (ABA); esperar solo al cambio de valor podía dormir el hilo para siempre.
    const auto startGen = waitObj->wakeGeneration;
    const auto valueChanged = [&]() {
        return waitObj->wakeGeneration != startGen ||
               loadGuestScalarAtomic(Address, AddressSize) != expected;
    };
    if (valueChanged()) {
        win_SetLastError(0);
        return 1;
    }

    bool awakened = false;
    if (dwMilliseconds == 0xFFFFFFFFu) {
        waitObj->cv.wait(lk, valueChanged);
        awakened = valueChanged();
    } else {
        // 0xFFFFFFFE es efectivamente INFINITE; acotar evita desbordar steady_clock.
        constexpr uint64_t kMaxWaitMs = 24ull * 60 * 60 * 1000;
        const uint64_t waitMs = std::min<uint64_t>(dwMilliseconds, kMaxWaitMs);
        awakened = waitObj->cv.wait_for(lk, std::chrono::milliseconds(waitMs), valueChanged);
    }
    lk.unlock();

    if (!awakened) {
        win_SetLastError(1460); // ERROR_TIMEOUT
        return 0;
    }
    win_SetLastError(0);
    return 1;
}

static void wakeWaitAddress(const void* Address) {
    auto waitObj = findWaitAddress(Address);
    if (!waitObj) return;
    {
        std::lock_guard<std::mutex> lk(waitObj->mtx);
        ++waitObj->wakeGeneration;
    }
    waitObj->cv.notify_all();
}

void NativeDllInterceptor::win_WakeByAddress(const void* Address) {
    s_kernel32Calls.fetch_add(1);
    wakeWaitAddress(Address);
}

void NativeDllInterceptor::win_WakeByAddressAll(const void* Address) {
    s_kernel32Calls.fetch_add(1);
    wakeWaitAddress(Address);
}

void* NativeDllInterceptor::win_CreateEventW(void*, int32_t bManualReset, int32_t bInitialState, const wchar_t*) {
    s_kernel32Calls.fetch_add(1);
    auto ev = std::make_shared<WinEventObj>();
    ev->manualReset = (bManualReset != 0);
    ev->signaled.store(bInitialState != 0);
    const uintptr_t handle = s_nextEventHandle.fetch_add(4);
    std::lock_guard<std::mutex> lk(s_eventMutex);
    s_events[handle] = ev;
    return reinterpret_cast<void*>(handle);
}

// --- Waitable timers -------------------------------------------------------------------------
namespace {
struct WaitableTimer {
    bool active = false;
    std::chrono::steady_clock::time_point due;
    std::chrono::milliseconds period{0};
};
std::mutex s_timerMutex;
std::condition_variable s_timerCv;
std::unordered_map<uintptr_t, WaitableTimer> s_timers;

// Hilo de servicio único: señala cada timer al vencer y lo reprograma si es periódico.
void timerServiceLoop() {
    std::unique_lock<std::mutex> lock(s_timerMutex);
    for (;;) {
        auto next = std::chrono::steady_clock::time_point::max();
        const auto now = std::chrono::steady_clock::now();
        std::vector<uintptr_t> fire;
        for (auto& [h, t] : s_timers) {
            if (!t.active) continue;
            if (t.due <= now) {
                fire.push_back(h);
                if (t.period.count() > 0) {
                    while (t.due <= now) t.due += t.period;
                } else {
                    t.active = false;
                }
            }
            if (t.active) next = std::min(next, t.due);
        }
        if (!fire.empty()) {
            lock.unlock();
            for (uintptr_t h : fire) NativeDllInterceptor::win_SetEvent(reinterpret_cast<void*>(h));
            lock.lock();
            continue;
        }
        if (next == std::chrono::steady_clock::time_point::max()) s_timerCv.wait(lock);
        else s_timerCv.wait_until(lock, next);
    }
}
void ensureTimerService() {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(timerServiceLoop).detach(); });
}
}  // namespace

void* NativeDllInterceptor::win_CreateWaitableTimerW(void* attrs, int32_t manualReset, const void*) {
    void* h = win_CreateEventW(attrs, manualReset, 0, nullptr);
    {
        std::lock_guard<std::mutex> lock(s_timerMutex);
        s_timers[reinterpret_cast<uintptr_t>(h)] = WaitableTimer{};
    }
    ensureTimerService();
    return h;
}

void* NativeDllInterceptor::win_CreateWaitableTimerExW(void* attrs, const void* name, uint32_t flags, uint32_t) {
    return win_CreateWaitableTimerW(attrs, (flags & 1u) ? 1 : 0, name);  // CREATE_WAITABLE_TIMER_MANUAL_RESET
}

int32_t NativeDllInterceptor::win_SetWaitableTimer(void* hTimer, const int64_t* dueTime, int32_t periodMs, void*, void*,
                                                   int32_t) {
    if (hTimer == nullptr || dueTime == nullptr) return 0;
    const uintptr_t h = reinterpret_cast<uintptr_t>(hTimer);
    // dueTime en 100 ns: negativo = relativo; positivo = FILETIME absoluto (UTC desde 1601).
    int64_t relative100ns = *dueTime;
    if (relative100ns > 0) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        const int64_t nowFt = (static_cast<int64_t>(ts.tv_sec) + 11644473600LL) * 10000000LL + ts.tv_nsec / 100;
        relative100ns = -(relative100ns - nowFt);
        if (relative100ns > 0) relative100ns = 0;  // ya vencido
    }
    win_ResetEvent(hTimer);
    {
        std::lock_guard<std::mutex> lock(s_timerMutex);
        WaitableTimer& t = s_timers[h];
        t.active = true;
        t.due = std::chrono::steady_clock::now() + std::chrono::nanoseconds((-relative100ns) * 100);
        t.period = std::chrono::milliseconds(std::max(0, periodMs));
    }
    ensureTimerService();
    s_timerCv.notify_all();
    return 1;
}

int32_t NativeDllInterceptor::win_CancelWaitableTimer(void* hTimer) {
    std::lock_guard<std::mutex> lock(s_timerMutex);
    auto it = s_timers.find(reinterpret_cast<uintptr_t>(hTimer));
    if (it == s_timers.end()) return 0;
    it->second.active = false;
    return 1;
}

void* NativeDllInterceptor::win_CreateEventA(void* a, int32_t m, int32_t i, const char*) {
    return win_CreateEventW(a, m, i, nullptr);
}

void* NativeDllInterceptor::win_CreateEventExW(void* a, const wchar_t* name, uint32_t flags, uint32_t) {
    const int32_t manual = (flags & 1) ? 1 : 0;
    const int32_t initial = (flags & 2) ? 1 : 0;
    return win_CreateEventW(a, manual, initial, name);
}

int32_t NativeDllInterceptor::win_SetEvent(void* hEvent) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hEvent);
    noteSignaler(handle);
    std::shared_ptr<WinEventObj> ev;
    {
        std::lock_guard<std::mutex> lk(s_eventMutex);
        auto it = s_events.find(handle);
        if (it != s_events.end()) ev = it->second;
    }
    if (ev) {
        {
            std::lock_guard<std::mutex> lock(ev->mtx);
            ev->signaled.store(true);
        }
        if (ev->manualReset) {
            ev->cv.notify_all();
        } else {
            ev->cv.notify_one();
        }
        return 1;
    }
    return 0;
}

int32_t NativeDllInterceptor::win_ResetEvent(void* hEvent) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hEvent);
    std::shared_ptr<WinEventObj> ev;
    {
        std::lock_guard<std::mutex> lk(s_eventMutex);
        auto it = s_events.find(handle);
        if (it != s_events.end()) ev = it->second;
    }
    if (ev) {
        std::lock_guard<std::mutex> lock(ev->mtx);
        ev->signaled.store(false);
        return 1;
    }
    return 0;
}

void* NativeDllInterceptor::win_CreateSemaphoreW(void*, int32_t lInitialCount, int32_t lMaximumCount, const wchar_t*) {
    s_kernel32Calls.fetch_add(1);
    if (lMaximumCount <= 0 || lInitialCount < 0 || lInitialCount > lMaximumCount) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return nullptr;
    }
    auto sem = std::make_shared<WinSemaphoreObj>();
    sem->count = lInitialCount;
    sem->maxCount = lMaximumCount;
    const uintptr_t handle = s_nextSemHandle.fetch_add(4);
    {
        std::lock_guard<std::mutex> lk(s_semMutex);
        s_semaphores[handle] = sem;
    }
    return reinterpret_cast<void*>(handle);
}

void* NativeDllInterceptor::win_CreateSemaphoreA(void* a, int32_t i, int32_t m, const char*) {
    return win_CreateSemaphoreW(a, i, m, nullptr);
}

void* NativeDllInterceptor::win_CreateSemaphoreExW(void* a, int32_t i, int32_t m, const wchar_t* name, uint32_t, uint32_t) {
    return win_CreateSemaphoreW(a, i, m, name);
}

void* NativeDllInterceptor::win_CreateSemaphoreExA(void* a, int32_t i, int32_t m, const char*, uint32_t, uint32_t) {
    return win_CreateSemaphoreW(a, i, m, nullptr);
}

int32_t NativeDllInterceptor::win_ReleaseSemaphore(void* hSemaphore, int32_t lReleaseCount, int32_t* lpPreviousCount) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hSemaphore);
    noteSignaler(handle);
    std::shared_ptr<WinSemaphoreObj> sem;
    {
        std::lock_guard<std::mutex> lk(s_semMutex);
        auto it = s_semaphores.find(handle);
        if (it != s_semaphores.end()) sem = it->second;
    }
    if (!sem) {
        win_SetLastError(6); // ERROR_INVALID_HANDLE
        return 0;
    }
    std::unique_lock<std::mutex> lock(sem->mtx);
    if (lpPreviousCount != nullptr) {
        *lpPreviousCount = sem->count;
    }
    if (lReleaseCount <= 0 || sem->count + lReleaseCount > sem->maxCount) {
        win_SetLastError(298); // ERROR_TOO_MANY_POSTS
        return 0;
    }
    sem->count += lReleaseCount;
    if (lReleaseCount == 1) {
        sem->cv.notify_one();
    } else {
        sem->cv.notify_all();
    }
    return 1;
}

void* NativeDllInterceptor::win_OpenSemaphoreW(uint32_t, int32_t, const wchar_t*) {
    s_kernel32Calls.fetch_add(1);
    return nullptr;
}

void* NativeDllInterceptor::win_CreateMutexW(void* lpMutexAttributes, int32_t bInitialOwner, const wchar_t* lpName) {
    void* h = win_CreateSemaphoreW(lpMutexAttributes, bInitialOwner ? 0 : 1, 1, lpName);
    std::lock_guard<std::mutex> lk(s_semMutex);
    auto it = s_semaphores.find(reinterpret_cast<uintptr_t>(h));
    if (it != s_semaphores.end()) {
        it->second->isMutex = true;
        if (bInitialOwner) { it->second->ownerTid = static_cast<int>(gettid()); it->second->recursion = 1; }
    }
    return h;
}

void* NativeDllInterceptor::win_CreateMutexA(void* lpMutexAttributes, int32_t bInitialOwner, const char*) {
    return win_CreateMutexW(lpMutexAttributes, bInitialOwner, nullptr);
}

void* NativeDllInterceptor::win_CreateMutexExW(void* lpMutexAttributes, const wchar_t* lpName, uint32_t dwFlags, uint32_t) {
    const int32_t initialOwner = (dwFlags & 1) ? 1 : 0;
    return win_CreateMutexW(lpMutexAttributes, initialOwner, lpName);
}

void* NativeDllInterceptor::win_CreateMutexExA(void* lpMutexAttributes, const char*, uint32_t dwFlags, uint32_t) {
    const int32_t initialOwner = (dwFlags & 1) ? 1 : 0;
    return win_CreateMutexW(lpMutexAttributes, initialOwner, nullptr);
}

int32_t NativeDllInterceptor::win_ReleaseMutex(void* hMutex) {
    s_kernel32Calls.fetch_add(1);
    std::shared_ptr<WinSemaphoreObj> sem;
    {
        std::lock_guard<std::mutex> lk(s_semMutex);
        auto it = s_semaphores.find(reinterpret_cast<uintptr_t>(hMutex));
        if (it != s_semaphores.end()) sem = it->second;
    }
    if (!sem) { win_SetLastError(6); return 0; }
    if (!sem->isMutex) return win_ReleaseSemaphore(hMutex, 1, nullptr);
    std::unique_lock<std::mutex> lock(sem->mtx);
    if (sem->ownerTid != static_cast<int>(gettid())) { win_SetLastError(288); return 0; }  // ERROR_NOT_OWNER
    if (--sem->recursion > 0) return 1;
    sem->ownerTid = 0;
    sem->count = 1;
    sem->cv.notify_one();
    return 1;
}

void* NativeDllInterceptor::win_OpenMutexW(uint32_t, int32_t, const wchar_t*) {
    s_kernel32Calls.fetch_add(1);
    return nullptr;
}

uint32_t NativeDllInterceptor::win_WaitForSingleObject(void* hHandle, uint32_t dwMilliseconds) {
    s_kernel32Calls.fetch_add(1);
    if (hHandle == nullptr) return 0; // WAIT_OBJECT_0
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hHandle);

    // 1. Evento
    std::shared_ptr<WinEventObj> ev;
    {
        std::lock_guard<std::mutex> lk(s_eventMutex);
        auto it = s_events.find(handle);
        if (it != s_events.end()) ev = it->second;
    }
    if (ev) {
        std::unique_lock<std::mutex> lock(ev->mtx);
        if (ev->signaled.load()) {
            if (!ev->manualReset) ev->signaled.store(false);
            return 0; // WAIT_OBJECT_0
        }
        if (dwMilliseconds == 0) return 0x102; // WAIT_TIMEOUT
        auto pred = [&]() -> bool { return ev->signaled.load(); };
        bool ok = false;
        if (dwMilliseconds == 0xFFFFFFFF) {
            ev->cv.wait(lock, pred);
            ok = true;
        } else {
            ok = ev->cv.wait_for(lock, std::chrono::milliseconds(dwMilliseconds), pred);
        }
        if (ok && ev->signaled.load()) {
            if (!ev->manualReset) ev->signaled.store(false);
            return 0; // WAIT_OBJECT_0
        }
        return 0x102; // WAIT_TIMEOUT
    }

    // 2. Semáforo / Mutex
    std::shared_ptr<WinSemaphoreObj> sem;
    {
        std::lock_guard<std::mutex> lk(s_semMutex);
        auto it = s_semaphores.find(handle);
        if (it != s_semaphores.end()) sem = it->second;
    }
    if (sem) {
        std::unique_lock<std::mutex> lock(sem->mtx);
        const int tid = static_cast<int>(gettid());
        if (sem->available(tid)) {
            sem->take(tid);
            return 0; // WAIT_OBJECT_0
        }
        if (dwMilliseconds == 0) return 0x102; // WAIT_TIMEOUT
        if (dwMilliseconds == 0xFFFFFFFF) {
            sem->cv.wait(lock, [&]() { return sem->available(tid); });
            sem->take(tid);
            return 0; // WAIT_OBJECT_0
        }
        if (sem->cv.wait_for(lock, std::chrono::milliseconds(dwMilliseconds), [&]() { return sem->available(tid); })) {
            sem->take(tid);
            return 0; // WAIT_OBJECT_0
        }
        return 0x102; // WAIT_TIMEOUT
    }

    // 3. Hilo
    std::shared_ptr<WinThreadObj> thr;
    {
        std::lock_guard<std::mutex> lk(s_threadMutex);
        auto it = s_threads.find(handle);
        if (it != s_threads.end()) thr = it->second;
    }
    if (thr) {
        std::unique_lock<std::mutex> lock(thr->mtx);
        if (thr->finished.load()) return 0; // WAIT_OBJECT_0
        if (dwMilliseconds == 0) return 0x102; // WAIT_TIMEOUT
        if (dwMilliseconds == 0xFFFFFFFF) {
            thr->cv.wait(lock, [&]() { return thr->finished.load(); });
            return 0; // WAIT_OBJECT_0
        }
        if (thr->cv.wait_for(lock, std::chrono::milliseconds(dwMilliseconds), [&]() { return thr->finished.load(); })) {
            return 0; // WAIT_OBJECT_0
        }
        return 0x102; // WAIT_TIMEOUT
    }

    return 0; // WAIT_OBJECT_0
}

uint32_t NativeDllInterceptor::win_WaitForSingleObjectEx(void* hHandle, uint32_t dwMilliseconds, int32_t) {
    return win_WaitForSingleObject(hHandle, dwMilliseconds);
}

uint32_t NativeDllInterceptor::win_WaitForMultipleObjects(uint32_t nCount, void* const* lpHandles,
                                                         int32_t bWaitAll, uint32_t dwMilliseconds) {
    s_kernel32Calls.fetch_add(1);
    if (nCount == 0 || lpHandles == nullptr || nCount > 64) return 0xFFFFFFFF; // WAIT_FAILED
    // Invitado de 32 bits: el array es de HANDLE de 4 bytes.
    std::array<void*, 64> handles32{};
    if (lowmem::enabled()) {
        const auto* src = reinterpret_cast<const uint32_t*>(lpHandles);
        for (uint32_t i = 0; i < nCount; ++i) handles32[i] = reinterpret_cast<void*>(uintptr_t{src[i]});
        lpHandles = handles32.data();
    }

    auto tryAcquire = [](void* h) -> bool {
        if (!h) return false;
        const uintptr_t handle = reinterpret_cast<uintptr_t>(h);
        {
            std::lock_guard<std::mutex> lk(s_eventMutex);
            auto it = s_events.find(handle);
            if (it != s_events.end()) {
                if (it->second->signaled.load()) {
                    if (!it->second->manualReset) it->second->signaled.store(false);
                    return true;
                }
                return false;
            }
        }
        {
            std::lock_guard<std::mutex> lk(s_semMutex);
            auto it = s_semaphores.find(handle);
            if (it != s_semaphores.end()) {
                std::lock_guard<std::mutex> slock(it->second->mtx);
                const int tid = static_cast<int>(gettid());
                if (it->second->available(tid)) {
                    it->second->take(tid);
                    return true;
                }
                return false;
            }
        }
        {
            std::lock_guard<std::mutex> lk(s_threadMutex);
            auto it = s_threads.find(handle);
            if (it != s_threads.end()) {
                return it->second->finished.load();
            }
        }
        return false;
    };

    uint32_t waitedMs = 0;
    while (true) {
        if (!bWaitAll) {
            for (uint32_t i = 0; i < nCount; ++i) {
                if (tryAcquire(lpHandles[i])) {
                    return i; // WAIT_OBJECT_0 + i
                }
            }
        } else {
            bool allReady = true;
            for (uint32_t i = 0; i < nCount; ++i) {
                const uintptr_t handle = reinterpret_cast<uintptr_t>(lpHandles[i]);
                bool ready = false;
                {
                    std::lock_guard<std::mutex> lk(s_eventMutex);
                    auto it = s_events.find(handle);
                    if (it != s_events.end() && it->second->signaled.load()) ready = true;
                }
                if (!ready) {
                    std::lock_guard<std::mutex> lk(s_semMutex);
                    auto it = s_semaphores.find(handle);
                    if (it != s_semaphores.end() && it->second->available(static_cast<int>(gettid()))) ready = true;
                }
                if (!ready) {
                    std::lock_guard<std::mutex> lk(s_threadMutex);
                    auto it = s_threads.find(handle);
                    if (it != s_threads.end() && it->second->finished.load()) ready = true;
                }
                if (!ready) {
                    allReady = false;
                    break;
                }
            }
            if (allReady) {
                for (uint32_t i = 0; i < nCount; ++i) {
                    tryAcquire(lpHandles[i]);
                }
                return 0; // WAIT_OBJECT_0
            }
        }

        if (dwMilliseconds == 0) return 0x102; // WAIT_TIMEOUT
        if (dwMilliseconds != 0xFFFFFFFF && waitedMs >= dwMilliseconds) {
            return 0x102; // WAIT_TIMEOUT
        }

        usleep(500);
        waitedMs += 1;
    }
}

uint32_t NativeDllInterceptor::win_WaitForMultipleObjectsEx(uint32_t nCount, void* const* lpHandles,
                                                           int32_t bWaitAll, uint32_t dwMilliseconds, int32_t) {
    return win_WaitForMultipleObjects(nCount, lpHandles, bWaitAll, dwMilliseconds);
}

int32_t NativeDllInterceptor::win_GetExitCodeThread(void* hThread, uint32_t* lpExitCode) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hThread);
    std::shared_ptr<WinThreadObj> thr;
    {
        std::lock_guard<std::mutex> lk(s_threadMutex);
        auto it = s_threads.find(handle);
        if (it != s_threads.end()) thr = it->second;
    }
    if (thr != nullptr) {
        if (lpExitCode != nullptr) {
            *lpExitCode = thr->finished.load() ? thr->exitCode : 259; // STILL_ACTIVE = 259
        }
        return 1;
    }
    // Handle real de hilo (DuplicateHandle/OpenThread): vivo mientras ejecuta código invitado.
    if (const int tid = threadTidFromHandle(hThread); tid != 0) {
        uint32_t code = 0;
        if (tid == static_cast<int>(gettid()) || OryonEmitterBackend::forThread(tid) != nullptr) {
            code = 259;
        } else {
            std::lock_guard<std::mutex> lk(s_threadMutex);
            for (const auto& kv : s_threads) {
                if (kv.second->tid.load() == tid) {
                    code = kv.second->finished.load() ? kv.second->exitCode : 259;
                    break;
                }
            }
        }
        if (lpExitCode != nullptr) *lpExitCode = code;
        return 1;
    }
    win_SetLastError(6);  // ERROR_INVALID_HANDLE
    if (lpExitCode != nullptr) *lpExitCode = 0;
    return 0;
}

int32_t NativeDllInterceptor::win_GetExitCodeProcess(void*, uint32_t* lpExitCode) {
    s_kernel32Calls.fetch_add(1);
    if (lpExitCode != nullptr) *lpExitCode = 259; // STILL_ACTIVE
    return 1;
}

int32_t NativeDllInterceptor::win_GetLogicalProcessorInformation(void* Buffer, uint32_t* ReturnedLength) {
    s_kernel32Calls.fetch_add(1);
    if (ReturnedLength == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    const uint32_t reqSize = 8 * 32;
    if (Buffer == nullptr || *ReturnedLength < reqSize) {
        *ReturnedLength = reqSize;
        win_SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
        return 0;
    }
    uint8_t* p = static_cast<uint8_t*>(Buffer);
    std::memset(p, 0, reqSize);
    for (uint32_t i = 0; i < 8; ++i) {
        uint8_t* entry = p + i * 32;
        *reinterpret_cast<uint64_t*>(entry) = (1ULL << i); // ProcessorMask
        *reinterpret_cast<uint32_t*>(entry + 8) = 0;      // RelationProcessorCore
    }
    *ReturnedLength = reqSize;
    return 1;
}

int32_t NativeDllInterceptor::win_GetLogicalProcessorInformationEx(uint32_t RelationshipType, void* Buffer, uint32_t* ReturnedLength) {
    s_kernel32Calls.fetch_add(1);
    if (ReturnedLength == nullptr) {
        win_SetLastError(87);
        return 0;
    }

    uint32_t reqSize = 0;
    if (RelationshipType == 4) {
        reqSize = 80;
    } else if (RelationshipType == 0) {
        reqSize = 384;
    } else if (RelationshipType == 0xFFFF) {
        reqSize = 464;
    } else {
        reqSize = 80;
    }

    if (Buffer == nullptr || *ReturnedLength < reqSize) {
        *ReturnedLength = reqSize;
        win_SetLastError(122); // ERROR_INSUFFICIENT_BUFFER
        return 0;
    }

    uint8_t* p = static_cast<uint8_t*>(Buffer);
    std::memset(p, 0, reqSize);

    auto fillCores = [](uint8_t* dst) {
        for (uint32_t i = 0; i < 8; ++i) {
            uint8_t* entry = dst + i * 48;
            *reinterpret_cast<uint32_t*>(entry) = 0;  // RelationProcessorCore
            *reinterpret_cast<uint32_t*>(entry + 4) = 48; // Size
            entry[8] = 0; // Flags (0 = single thread per core)
            entry[9] = 0; // EfficiencyClass
            *reinterpret_cast<uint16_t*>(entry + 30) = 1; // GroupCount
            *reinterpret_cast<uint64_t*>(entry + 32) = (1ULL << i); // GroupMask[0].Mask
            *reinterpret_cast<uint16_t*>(entry + 40) = 0; // GroupMask[0].Group
        }
    };

    auto fillGroup = [](uint8_t* dst) {
        *reinterpret_cast<uint32_t*>(dst) = 4;  // RelationGroup
        *reinterpret_cast<uint32_t*>(dst + 4) = 80; // Size
        *reinterpret_cast<uint16_t*>(dst + 8) = 1;  // MaximumGroupCount
        *reinterpret_cast<uint16_t*>(dst + 10) = 1; // ActiveGroupCount
        dst[32] = 64; // MaximumProcessorCount
        dst[33] = 8;  // ActiveProcessorCount
        *reinterpret_cast<uint64_t*>(dst + 72) = 0xFF; // ActiveProcessorMask (8 cores)
    };

    if (RelationshipType == 4) {
        fillGroup(p);
    } else if (RelationshipType == 0) {
        fillCores(p);
    } else if (RelationshipType == 0xFFFF) {
        fillCores(p);
        fillGroup(p + 384);
    } else {
        fillGroup(p);
    }

    *ReturnedLength = reqSize;
    return 1;
}

int32_t NativeDllInterceptor::win_BCryptGenRandom(void*, uint8_t* pbBuffer, uint32_t cbBuffer, uint32_t) {
    if (pbBuffer != nullptr && cbBuffer > 0) {
        arc4random_buf(pbBuffer, cbBuffer);
    }
    return 0; // STATUS_SUCCESS
}

int32_t NativeDllInterceptor::win_BCryptOpenAlgorithmProvider(void** phAlgorithm, const wchar_t*, const wchar_t*, uint32_t) {
    lowmem::storePtr(phAlgorithm, uint64_t{0xBEEF1001});
    return 0; // STATUS_SUCCESS
}

int32_t NativeDllInterceptor::win_BCryptCloseAlgorithmProvider(void*, uint32_t) {
    return 0; // STATUS_SUCCESS
}

int32_t NativeDllInterceptor::win_CryptAcquireContextW(void** phProv, const wchar_t*, const wchar_t*, uint32_t, uint32_t) {
    lowmem::storePtr(phProv, uint64_t{0xCAFE2002});  // HCRYPTPROV = ULONG_PTR
    return 1;
}

int32_t NativeDllInterceptor::win_CryptAcquireContextA(void** phProv, const char*, const char*, uint32_t, uint32_t) {
    lowmem::storePtr(phProv, uint64_t{0xCAFE2002});
    return 1;
}

int32_t NativeDllInterceptor::win_CryptReleaseContext(void*, uint32_t) {
    return 1;
}

int32_t NativeDllInterceptor::win_CryptGenRandom(void*, uint32_t dwLen, uint8_t* pbBuffer) {
    if (pbBuffer != nullptr && dwLen > 0) {
        arc4random_buf(pbBuffer, dwLen);
    }
    return 1;
}

uint8_t NativeDllInterceptor::win_SystemFunction036(void* randomBuffer, uint32_t randomBufferLength) {
    if (randomBuffer == nullptr || randomBufferLength == 0) return 0;
    arc4random_buf(randomBuffer, randomBufferLength);
    return 1;
}

uint32_t NativeDllInterceptor::win_GetFileVersionInfoSizeW(const wchar_t*, uint32_t* handle) {
    if (handle != nullptr) *handle = 0;
    win_SetLastError(1813);  // ERROR_RESOURCE_TYPE_NOT_FOUND
    return 0;
}

uint32_t NativeDllInterceptor::win_GetFileVersionInfoSizeA(const char*, uint32_t* handle) {
    return win_GetFileVersionInfoSizeW(nullptr, handle);
}

int32_t NativeDllInterceptor::win_GetFileVersionInfoW(const wchar_t*, uint32_t, uint32_t, void*) {
    win_SetLastError(1813);
    return 0;
}

int32_t NativeDllInterceptor::win_GetFileVersionInfoA(const char*, uint32_t handle, uint32_t len,
                                                      void* data) {
    return win_GetFileVersionInfoW(nullptr, handle, len, data);
}

int32_t NativeDllInterceptor::win_VerQueryValueW(const void*, const wchar_t*, void** buffer, uint32_t* len) {
    lowmem::storePtr(buffer, uint64_t{0});
    if (len != nullptr) *len = 0;
    win_SetLastError(1813);
    return 0;
}

int32_t NativeDllInterceptor::win_VerQueryValueA(const void* block, const char* subBlock, void** buffer,
                                                 uint32_t* len) {
    (void)subBlock;
    return win_VerQueryValueW(block, nullptr, buffer, len);
}

int32_t NativeDllInterceptor::win_SwitchToThread() {
    s_kernel32Calls.fetch_add(1);
    sched_yield();
    return 1;
}

void* NativeDllInterceptor::win_GetCurrentThread() {
    s_kernel32Calls.fetch_add(1);
    return reinterpret_cast<void*>(~static_cast<uintptr_t>(1));
}

int32_t NativeDllInterceptor::win_SetThreadPriority(void*, int32_t) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_GetThreadPriority(void*) {
    s_kernel32Calls.fetch_add(1);
    return 0;
}

uint32_t NativeDllInterceptor::win_ResumeThread(void* hThread) {
    s_kernel32Calls.fetch_add(1);
    const int tid = threadTidFromHandle(hThread);
    OryonEmitterBackend* target = tid != 0 ? OryonEmitterBackend::forThread(tid) : nullptr;
    if (target == nullptr) {
        // Hilo creado con CREATE_SUSPENDED (se arranca de inmediato) o host: contador 0.
        if (tid == 0) win_SetLastError(6);
        return tid == 0 ? 0xFFFFFFFFu : 0;
    }
    return target->requestResume();
}

uint32_t NativeDllInterceptor::win_SuspendThread(void* hThread) {
    s_kernel32Calls.fetch_add(1);
    const int tid = threadTidFromHandle(hThread);
    OryonEmitterBackend* target = tid != 0 ? OryonEmitterBackend::forThread(tid) : nullptr;
    if (target == nullptr) {
        static std::atomic<uint32_t> s_failLogged{0};
        if (s_failLogged.fetch_add(1) < 8) {
            LOGW("SuspendThread(%p): tid=%d sin backend invitado (llamador tid=%d)", hThread, tid,
                 static_cast<int>(gettid()));
        }
        win_SetLastError(6);  // ERROR_INVALID_HANDLE
        return 0xFFFFFFFFu;
    }
    s_suspendCalls.fetch_add(1, std::memory_order_relaxed);
    if (tid == static_cast<int>(gettid())) {
        // Auto-suspensión: Windows bloquearía el hilo hasta un ResumeThread externo; ningún
        // runtime soportado lo usa y bloquear aquí colgaría el proceso.
        LOGW("SuspendThread sobre el hilo actual (tid=%d) ignorado", tid);
        return 0;
    }
    return target->requestSuspend();
}

// Pila nativa de otro hilo: se le envía una señal y su manejador recorre su propia pila.
namespace hoststack {
constexpr int kMax = 24;
std::atomic<int> s_count{-1};
uintptr_t s_pcs[kMax];

_Unwind_Reason_Code unwindStep(_Unwind_Context* uc, void* arg) {
    int& n = *static_cast<int*>(arg);
    if (n >= kMax) return _URC_END_OF_STACK;
    s_pcs[n++] = _Unwind_GetIP(uc);
    return _URC_NO_REASON;
}

void handler(int, siginfo_t*, void*) {
    int n = 0;
    _Unwind_Backtrace(unwindStep, &n);
    s_count.store(n, std::memory_order_release);
}

void dump(int tid) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lock(s_mutex);
    const int sig = SIGRTMAX - 3;
    static bool s_installed = false;
    if (!s_installed) {
        struct sigaction sa{};
        sa.sa_sigaction = handler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, nullptr);
        s_installed = true;
    }
    s_count.store(-1, std::memory_order_release);
    if (syscall(SYS_tgkill, getpid(), tid, sig) != 0) return;
    for (int i = 0; i < 200 && s_count.load(std::memory_order_acquire) < 0; ++i) usleep(1000);
    const int n = s_count.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        Dl_info di{};
        const bool ok = dladdr(reinterpret_cast<void*>(s_pcs[i]), &di) != 0;
        const char* lib = ok && di.dli_fname ? std::strrchr(di.dli_fname, '/') : nullptr;
        LOGW("[Punto seguro] tid=%d #%d %s %s+0x%llx", tid, i, lib ? lib + 1 : "?",
             ok && di.dli_sname ? di.dli_sname : "?",
             static_cast<unsigned long long>(s_pcs[i] - reinterpret_cast<uintptr_t>(ok && di.dli_saddr ? di.dli_saddr
                                                                                          : di.dli_fbase)));
    }
}
}  // namespace hoststack

// Diagnóstico (una vez por hilo): qué hace en el host un hilo que no llega al punto seguro
// (estado del kernel, syscall en curso y su PC, función nativa rápida en curso y pila invitada).
static void logStuckHostThread(int tid, OryonEmitterBackend* target) {
    static std::mutex s_mutex;
    static std::unordered_set<int> s_done;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_done.insert(tid).second) return;
    }
    auto readSmall = [](const std::string& path) {
        char buf[256] = {};
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return std::string("?");
        const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
        ::close(fd);
        std::string s(buf, n > 0 ? static_cast<size_t>(n) : 0);
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        return s;
    };
    const std::string base = "/proc/self/task/" + std::to_string(tid) + "/";
    const std::string stat = readSmall(base + "stat");
    const size_t close = stat.rfind(')');
    const char kstate = close != std::string::npos && close + 2 < stat.size() ? stat[close + 2] : '?';
    const std::string sc = readSmall(base + "syscall");
    // Formato: "nr a0 a1 a2 a3 a4 a5 sp pc" (o "running").
    uintptr_t pc = 0;
    if (const size_t sp = sc.rfind(' '); sp != std::string::npos) pc = std::strtoull(sc.c_str() + sp + 1, nullptr, 16);
    Dl_info di{};
    const char* lib = pc != 0 && dladdr(reinterpret_cast<void*>(pc), &di) && di.dli_fname ? di.dli_fname : "?";
    const char* sym = di.dli_sname ? di.dli_sname : "?";
    const uintptr_t fn = target->context().nativeFn;
    Dl_info fi{};
    const char* fnSym = fn != 0 && dladdr(reinterpret_cast<void*>(fn), &fi) && fi.dli_sname ? fi.dli_sname : "?";
    LOGW("[Punto seguro] tid=%d kernel=%c estado=%d syscall='%s' pc=%s!%s nativo=%s RIP=0x%llx", tid, kstate,
         target->runState(), sc.c_str(), lib, sym, fnSym, static_cast<unsigned long long>(target->context().rip));
    char tag[48];
    std::snprintf(tag, sizeof(tag), "[Punto seguro] tid=%d", tid);
    target->logGuestStackChain(tag);
    hoststack::dump(tid);
}

int32_t NativeDllInterceptor::win_GetThreadContext(void* hThread, void* lpContext) {
    s_kernel32Calls.fetch_add(1);
    if (lpContext == nullptr) {
        win_SetLastError(998);  // ERROR_NOACCESS
        return 0;
    }
    const int tid = threadTidFromHandle(hThread);
    OryonEmitterBackend* target = tid != 0 ? OryonEmitterBackend::forThread(tid) : nullptr;
    if (target == nullptr) {
        win_SetLastError(6);
        return 0;
    }
    if (tid != static_cast<int>(gettid()) && !target->waitUntilSuspended(2000)) {
        LOGW("GetThreadContext: el hilo %d no llegó a un punto seguro en 2 s", tid);
        logStuckHostThread(tid, target);
        win_SetLastError(31);  // ERROR_GEN_FAILURE
        return 0;
    }
    copyGuestContextOut(target->context(), static_cast<uint8_t*>(lpContext));
    return 1;
}

int32_t NativeDllInterceptor::win_SetThreadContext(void* hThread, const void* lpContext) {
    s_kernel32Calls.fetch_add(1);
    if (lpContext == nullptr) {
        win_SetLastError(998);
        return 0;
    }
    const int tid = threadTidFromHandle(hThread);
    OryonEmitterBackend* target = tid != 0 ? OryonEmitterBackend::forThread(tid) : nullptr;
    if (target == nullptr || tid == static_cast<int>(gettid()) ||
        target->runState() != OryonEmitterBackend::kParked) {
        // Solo es seguro reescribir el contexto de un hilo detenido entre bloques.
        win_SetLastError(target == nullptr ? 6 : 31);
        return 0;
    }
    copyGuestContextIn(target->context(), static_cast<const uint8_t*>(lpContext));
    return 1;
}

int32_t NativeDllInterceptor::win_SetThreadDescription(void*, const wchar_t*) {
    s_kernel32Calls.fetch_add(1);
    return 0; // S_OK
}

// SLIST_HEADER (16 B): +0 primer nodo, +8 profundidad. Windows evita el problema ABA con un
// contador de secuencia de 128 bits; el CAS de 64 bits de antes no (Pop leía next de un nodo que
// otro hilo podía sacar y volver a meter -> lista corrupta en el sistema de jobs de Unity). Se
// serializa por cabecera con locks repartidos.
namespace {
std::mutex s_slistLocks[64];
} // namespace
namespace {
std::mutex& slistLock(const void* head) {
    return s_slistLocks[(reinterpret_cast<uintptr_t>(head) >> 4) & 63u];
}
} // namespace

void* NativeDllInterceptor::win_InterlockedPushEntrySList(void* ListHead, void* ListEntry) {
    s_kernel32Calls.fetch_add(1);
    if (!ListHead || !ListEntry) return nullptr;
    if (lowmem::enabled()) {
        // x86: SLIST_HEADER = {Next (4), Depth (2), Sequence (2)}; SLIST_ENTRY.Next de 4 bytes. Con el
        // formato x64 se escribían 8 bytes en la entrada y 8 más tras la cabecera (montón corrupto en DmC).
        auto* h = static_cast<uint32_t*>(ListHead);
        std::lock_guard<std::mutex> lock(slistLock(ListHead));
        const uint32_t old = h[0];
        std::memcpy(ListEntry, &old, 4);
        h[0] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ListEntry));
        auto* ds = reinterpret_cast<uint16_t*>(h + 1);
        ++ds[0];
        ++ds[1];
        return reinterpret_cast<void*>(static_cast<uintptr_t>(old));
    }
    auto* h = static_cast<uintptr_t*>(ListHead);
    std::lock_guard<std::mutex> lock(slistLock(ListHead));
    const uintptr_t old = h[0];
    *static_cast<uintptr_t*>(ListEntry) = old;
    h[0] = reinterpret_cast<uintptr_t>(ListEntry);
    h[1] = (h[1] & ~0xFFFFull) | ((h[1] + 1) & 0xFFFFull);
    return reinterpret_cast<void*>(old);
}

void* NativeDllInterceptor::win_InterlockedPopEntrySList(void* ListHead) {
    s_kernel32Calls.fetch_add(1);
    if (!ListHead) return nullptr;
    if (lowmem::enabled()) {
        auto* h = static_cast<uint32_t*>(ListHead);
        std::lock_guard<std::mutex> lock(slistLock(ListHead));
        const uint32_t first = h[0];
        if (first == 0) return nullptr;
        uint32_t next = 0;
        std::memcpy(&next, reinterpret_cast<const void*>(static_cast<uintptr_t>(first)), 4);
        h[0] = next;
        auto* ds = reinterpret_cast<uint16_t*>(h + 1);
        if (ds[0] != 0) --ds[0];
        ++ds[1];
        return reinterpret_cast<void*>(static_cast<uintptr_t>(first));
    }
    auto* h = static_cast<uintptr_t*>(ListHead);
    std::lock_guard<std::mutex> lock(slistLock(ListHead));
    const uintptr_t first = h[0];
    if (first == 0) return nullptr;
    h[0] = *reinterpret_cast<const uintptr_t*>(first);
    if ((h[1] & 0xFFFFull) != 0) --h[1];
    return reinterpret_cast<void*>(first);
}

void* NativeDllInterceptor::win_InterlockedFlushSList(void* ListHead) {
    s_kernel32Calls.fetch_add(1);
    if (!ListHead) return nullptr;
    if (lowmem::enabled()) {
        auto* h = static_cast<uint32_t*>(ListHead);
        std::lock_guard<std::mutex> lock(slistLock(ListHead));
        const uint32_t first = h[0];
        h[0] = 0;
        reinterpret_cast<uint16_t*>(h + 1)[0] = 0;
        return reinterpret_cast<void*>(static_cast<uintptr_t>(first));
    }
    auto* h = static_cast<uintptr_t*>(ListHead);
    std::lock_guard<std::mutex> lock(slistLock(ListHead));
    const uintptr_t first = h[0];
    h[0] = 0;
    h[1] &= ~0xFFFFull;
    return reinterpret_cast<void*>(first);
}

namespace {
constexpr size_t kMaxFlsSlots = 128;
std::atomic<uint32_t> s_flsSlotCounter{1};
thread_local void* t_flsValues[kMaxFlsSlots] = {nullptr};
} // namespace

uint32_t NativeDllInterceptor::win_FlsAlloc(void*) {
    s_kernel32Calls.fetch_add(1);
    uint32_t slot = s_flsSlotCounter.fetch_add(1);
    return slot < kMaxFlsSlots ? slot : 0xFFFFFFFFu;
}

void* NativeDllInterceptor::win_FlsGetValue(uint32_t dwFlsIndex) {
    s_kernel32Calls.fetch_add(1);
    return dwFlsIndex < kMaxFlsSlots ? t_flsValues[dwFlsIndex] : nullptr;
}

int32_t NativeDllInterceptor::win_FlsSetValue(uint32_t dwFlsIndex, void* lpFlsData) {
    s_kernel32Calls.fetch_add(1);
    if (dwFlsIndex >= kMaxFlsSlots) return 0;
    t_flsValues[dwFlsIndex] = lpFlsData;
    return 1;
}

int32_t NativeDllInterceptor::win_FlsFree(uint32_t dwFlsIndex) {
    s_kernel32Calls.fetch_add(1);
    if (dwFlsIndex < kMaxFlsSlots) t_flsValues[dwFlsIndex] = nullptr;
    return 1;
}

// -----------------------------------------------------------------------------
// Memoria, Tiempo y Sistema de Archivos
// -----------------------------------------------------------------------------
void NativeDllInterceptor::win_GetSystemTimeAsFileTime(void* lpSystemTimeAsFileTime) {
    s_kernel32Calls.fetch_add(1);
    if (!lpSystemTimeAsFileTime) return;
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const uint64_t intervals = (static_cast<uint64_t>(ts.tv_sec) + 11644473600ULL) * 10000000ULL +
                               (static_cast<uint64_t>(ts.tv_nsec) / 100ULL);
    uint32_t parts[2];
    parts[0] = static_cast<uint32_t>(intervals & 0xFFFFFFFFu);
    parts[1] = static_cast<uint32_t>((intervals >> 32) & 0xFFFFFFFFu);
    std::memcpy(lpSystemTimeAsFileTime, parts, sizeof(parts));
}

void NativeDllInterceptor::win_GetLocalTime(void* lpSystemTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpSystemTime == nullptr) return;
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm_buf{};
    localtime_r(&ts.tv_sec, &tm_buf);
    struct SystemTimeLayout {
        uint16_t wYear;
        uint16_t wMonth;
        uint16_t wDayOfWeek;
        uint16_t wDay;
        uint16_t wHour;
        uint16_t wMinute;
        uint16_t wSecond;
        uint16_t wMilliseconds;
    };
    auto* st = static_cast<SystemTimeLayout*>(lpSystemTime);
    st->wYear = static_cast<uint16_t>(tm_buf.tm_year + 1900);
    st->wMonth = static_cast<uint16_t>(tm_buf.tm_mon + 1);
    st->wDayOfWeek = static_cast<uint16_t>(tm_buf.tm_wday);
    st->wDay = static_cast<uint16_t>(tm_buf.tm_mday);
    st->wHour = static_cast<uint16_t>(tm_buf.tm_hour);
    st->wMinute = static_cast<uint16_t>(tm_buf.tm_min);
    st->wSecond = static_cast<uint16_t>(tm_buf.tm_sec);
    st->wMilliseconds = static_cast<uint16_t>(ts.tv_nsec / 1000000);
}

int32_t NativeDllInterceptor::win_SystemTimeToFileTime(const void* lpSystemTime, void* lpFileTime) {
    s_kernel32Calls.fetch_add(1);
    (void)lpSystemTime;
    if (lpFileTime != nullptr) {
        win_GetSystemTimeAsFileTime(lpFileTime);
    }
    return 1;
}

void* NativeDllInterceptor::win_GetProcessHeap() {
    s_kernel32Calls.fetch_add(1);
    return reinterpret_cast<void*>(0xCAFE0001ULL);
}

size_t NativeDllInterceptor::win_HeapSize(void*, uint32_t, const void* lpMem) {
    s_kernel32Calls.fetch_add(1);
    return safeHeapSize(lpMem);
}

void* NativeDllInterceptor::win_HeapReAlloc(void* hHeap, uint32_t dwFlags, void* lpMem, size_t dwBytes) {
    s_kernel32Calls.fetch_add(1);
    if (lpMem != nullptr && !isSafeHeapPointer(lpMem)) {
        // Copiar desde un puntero ajeno sin conocer su tamaño leería fuera de rango.
        LOGW("HeapReAlloc: %p no pertenece al WinSafeHeap", lpMem);
        win_SetLastError(87);
        return nullptr;
    }
    void* newMem = win_HeapAlloc(hHeap, dwFlags, dwBytes);
    if (newMem && lpMem) {
        const size_t oldSize = win_HeapSize(hHeap, dwFlags, lpMem);
        std::memcpy(newMem, lpMem, std::min(oldSize, dwBytes));
        win_HeapFree(hHeap, dwFlags, lpMem);
    }
    return newMem;
}

// --- VirtualQuery real sobre /proc/self/maps ---------------------------------------------
// El GC de Boehm (Mono) recorre el espacio de direcciones con VirtualQuery y registra como
// raíces las regiones MEM_IMAGE escribibles (.data/.bss de las DLL). Antes todo se describía
// como "64 KB MEM_PRIVATE RWX": no se escaneaban las globales de Mono y el GC liberaba objetos
// vivos (cascada de NullReferenceException).
namespace {
struct HostMapping {
    uint64_t start = 0, end = 0;
    uint32_t protect = 0x01;  // PAGE_*
    bool file = false;
};
std::mutex s_mapsMutex;
std::vector<HostMapping> s_hostMaps;
uint64_t s_hostMapsStampMs = 0;

uint32_t permsToPageProtect(const char* p) {
    const bool r = p[0] == 'r', w = p[1] == 'w', x = p[2] == 'x';
    if (x) return w ? 0x40u : (r ? 0x20u : 0x10u);
    if (w) return 0x04u;
    if (r) return 0x02u;
    return 0x01u;
}

void refreshHostMapsLocked() {
    std::vector<HostMapping> maps;
    FILE* f = std::fopen("/proc/self/maps", "re");
    if (f == nullptr) return;
    char line[512];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        unsigned long long start = 0, end = 0, inode = 0;
        char perms[8] = {};
        char path[256] = {};
        if (std::sscanf(line, "%llx-%llx %7s %*s %*s %llu %255s", &start, &end, perms, &inode, path) < 4) continue;
        HostMapping m;
        m.start = start;
        m.end = end;
        m.protect = permsToPageProtect(perms);
        m.file = inode != 0 || path[0] == '/';
        if (!maps.empty() && maps.back().end == m.start && maps.back().protect == m.protect &&
            maps.back().file == m.file) {
            maps.back().end = m.end;  // misma clase contigua: una sola región como en Windows
        } else {
            maps.push_back(m);
        }
    }
    std::fclose(f);
    s_hostMaps.swap(maps);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    s_hostMapsStampMs = static_cast<uint64_t>(ts.tv_sec) * 1000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000000ull;
}

// Imagen PE del huésped que contiene addr: {base, fin}.
bool guestImageRange(uint64_t addr, uint64_t& base, uint64_t& end) {
    if (s_activeImage == nullptr) return false;
    const auto exe = reinterpret_cast<uint64_t>(s_activeImage->baseAddress);
    const uint64_t exeEnd = exe + std::max<uint64_t>(s_activeImage->totalMappedSize, s_activeImage->peInfo.imageSize);
    if (exe != 0 && addr >= exe && addr < exeEnd) {
        base = exe;
        end = exeEnd;
        return true;
    }
    for (const auto& kv : s_activeImage->localDlls) {
        const auto b = reinterpret_cast<uint64_t>(kv.second.base);
        const uint64_t e = b + std::max<uint64_t>(kv.second.size, kv.second.peInfo.imageSize);
        if (b != 0 && addr >= b && addr < e) {
            base = b;
            end = e;
            return true;
        }
    }
    return false;
}
}  // namespace

size_t NativeDllInterceptor::win_VirtualQuery(const void* lpAddress, void* lpBuffer, size_t dwLength) {
    // Invitado de 32 bits: MEMORY_BASIC_INFORMATION mide 28 bytes (campos de 4). Se calcula el de x64 en
    // un búfer temporal y se copia campo a campo (escribir los 48 desbordaba el del invitado).
    static thread_local bool t_inner = false;
    if (lowmem::enabled() && !t_inner) {
        if (lpBuffer == nullptr || dwLength < 28) {
            win_SetLastError(24);
            return 0;
        }
        uint8_t mbi64[48] = {};
        t_inner = true;
        const size_t r = win_VirtualQuery(lpAddress, mbi64, sizeof(mbi64));
        t_inner = false;
        if (r == 0) return 0;
        auto field = [&](size_t off) {
            uint64_t v = 0;
            std::memcpy(&v, mbi64 + off, 8);
            return static_cast<uint32_t>(std::min<uint64_t>(v, 0xFFFFFFFFull));
        };
        uint32_t mbi32[7];
        mbi32[0] = field(0x00);                                   // BaseAddress
        mbi32[1] = field(0x08);                                   // AllocationBase
        std::memcpy(&mbi32[2], mbi64 + 0x10, 4);                  // AllocationProtect
        mbi32[3] = field(0x18);                                   // RegionSize
        std::memcpy(&mbi32[4], mbi64 + 0x20, 4);                  // State
        std::memcpy(&mbi32[5], mbi64 + 0x24, 4);                  // Protect
        std::memcpy(&mbi32[6], mbi64 + 0x28, 4);                  // Type
        std::memcpy(lpBuffer, mbi32, sizeof(mbi32));
        return sizeof(mbi32);
    }
    s_kernel32Calls.fetch_add(1);
    constexpr size_t kMbiSize = 48;  // MEMORY_BASIC_INFORMATION (x64)
    constexpr uint64_t kMaxUser = 0x7FFFFFFEFFFFull;
    if (lpBuffer == nullptr || dwLength < kMbiSize) {
        win_SetLastError(24);  // ERROR_BAD_LENGTH
        return 0;
    }
    const uint64_t addr = reinterpret_cast<uint64_t>(lpAddress);
    if (addr > kMaxUser) {
        win_SetLastError(87);
        return 0;
    }
    const uint64_t page = addr & ~0xFFFull;

    // Pila invitada del hilo: en Windows es una sola reserva (AllocationBase = su fondo).
    // CoreCLR calcula el límite de pila así (Thread::GetStackLowerBound) y con un segmento
    // parcial de /proc/self/maps creía estar sin pila ("Stack overflow.").
    if (OryonEmitterBackend* b = t_dispatchBackend; b != nullptr && b->virtualStack() != nullptr) {
        const auto lo = reinterpret_cast<uint64_t>(b->virtualStack());
        const uint64_t hi = lo + b->virtualStackSize();
        static thread_local int t_stackLogs = 0;
        if (t_stackLogs < 3 && t_thunkCtx != nullptr && addr + 0x100000 > t_thunkCtx->gpr[4] &&
            addr < t_thunkCtx->gpr[4] + 0x100000) {
            ++t_stackLogs;
            LOGI("VirtualQuery(pila) addr=0x%llx pila=[0x%llx,0x%llx) RSP=0x%llx %s", (unsigned long long)addr,
                 (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)t_thunkCtx->gpr[4],
                 addr >= lo && addr < hi ? "dentro" : "FUERA");
        }
        if (addr >= lo && addr < hi) {
            auto* buf = static_cast<uint8_t*>(lpBuffer);
            std::memset(buf, 0, kMbiSize);
            const uint64_t regionSize = hi - page;
            const uint32_t commit = 0x1000, rw = 0x04, priv = 0x20000;
            std::memcpy(buf + 0x00, &page, 8);
            std::memcpy(buf + 0x08, &lo, 8);
            std::memcpy(buf + 0x10, &rw, 4);
            std::memcpy(buf + 0x18, &regionSize, 8);
            std::memcpy(buf + 0x20, &commit, 4);
            std::memcpy(buf + 0x24, &rw, 4);
            std::memcpy(buf + 0x28, &priv, 4);
            return kMbiSize;
        }
    }

    HostMapping hit{};
    bool mapped = false;
    uint64_t nextStart = kMaxUser + 1;
    {
        std::lock_guard<std::mutex> lock(s_mapsMutex);
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const uint64_t nowMs = static_cast<uint64_t>(ts.tv_sec) * 1000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000000ull;
        auto lookup = [&]() {
            mapped = false;
            nextStart = kMaxUser + 1;
            for (const HostMapping& m : s_hostMaps) {
                if (page >= m.start && page < m.end) {
                    hit = m;
                    mapped = true;
                    return;
                }
                if (m.start > page && m.start < nextStart) nextStart = m.start;
            }
        };
        if (s_hostMaps.empty() || nowMs - s_hostMapsStampMs > 50) refreshHostMapsLocked();
        lookup();
        if (!mapped && nowMs - s_hostMapsStampMs > 2) {  // pudo mapearse hace un instante
            refreshHostMapsLocked();
            lookup();
        }
    }
    if (!mapped) {
        // Reserva de VirtualAlloc más reciente que la copia de /proc/self/maps: la tabla propia
        // manda. Sin esto salía MEM_FREE con AllocationBase 0 y Mono (mono_vfree) hacía
        // VirtualFree(NULL) y abortaba (Celeste: mono-mmap-windows.c:132).
        std::lock_guard<std::mutex> lock(s_allocMutex);
        for (const auto& [b, sz] : s_allocations) {
            if (page >= b && page < b + sz) {
                hit.start = b;
                hit.end = b + sz;
                hit.protect = 0x04;  // PAGE_READWRITE
                hit.file = false;
                mapped = true;
                break;
            }
        }
    }

    uint64_t regionBase = page, regionEnd = 0, allocBase = 0;
    uint32_t state = 0x10000, protect = 0x01, type = 0;  // MEM_FREE / PAGE_NOACCESS
    if (mapped) {
        state = 0x1000;  // MEM_COMMIT
        protect = hit.protect;
        regionEnd = hit.end;
        allocBase = hit.start;
        type = hit.file ? 0x40000u : 0x20000u;  // MEM_MAPPED / MEM_PRIVATE
        uint64_t imgBase = 0, imgEnd = 0;
        if (guestImageRange(page, imgBase, imgEnd)) {
            type = 0x1000000u;  // MEM_IMAGE: imagen PE del huésped
            allocBase = imgBase;
            if (regionEnd > imgEnd) regionEnd = imgEnd;
        } else {
            std::lock_guard<std::mutex> lock(s_allocMutex);
            for (const auto& [b, sz] : s_allocations) {  // reservas de VirtualAlloc
                if (page >= b && page < b + sz) {
                    allocBase = b;
                    if (regionEnd > b + sz) regionEnd = b + sz;
                    break;
                }
            }
        }
    } else {
        regionEnd = nextStart;
    }

    auto* buf = static_cast<uint8_t*>(lpBuffer);
    std::memset(buf, 0, kMbiSize);
    const uint64_t regionSize = regionEnd > regionBase ? regionEnd - regionBase : 0x1000;
    std::memcpy(buf + 0x00, &regionBase, 8);  // BaseAddress
    std::memcpy(buf + 0x08, &allocBase, 8);   // AllocationBase
    std::memcpy(buf + 0x10, &protect, 4);     // AllocationProtect
    std::memcpy(buf + 0x18, &regionSize, 8);  // RegionSize
    std::memcpy(buf + 0x20, &state, 4);       // State
    if (state != 0x10000) std::memcpy(buf + 0x24, &protect, 4);  // Protect (0 si MEM_FREE)
    std::memcpy(buf + 0x28, &type, 4);        // Type
    return kMbiSize;
}

void* NativeDllInterceptor::win_HeapCreate(uint32_t flOptions, size_t /*dwInitialSize*/, size_t /*dwMaximumSize*/) {
    s_kernel32Calls.fetch_add(1);
    // 32 bits: HEAP_CREATE_ENABLE_EXECUTE (0x40000) -> handle propio; HeapAlloc registra sus bloques como código.
    if ((flOptions & 0x40000u) != 0 && s_activeImage != nullptr && s_activeImage->is32Bit)
        return reinterpret_cast<void*>(0xCAFE0003ULL);
    return reinterpret_cast<void*>(0xCAFE0002ULL);
}

int32_t NativeDllInterceptor::win_HeapDestroy(void* /*hHeap*/) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

static std::mutex s_envMutex;
static std::unordered_map<std::string, std::string> s_customEnvVars;

// Variables de un juego .NET (ManagedHost.h): Mono busca FNA y los reenvíos de XNA en fna/ del
// runtime; SDL GPU (FNA3D) usa Vulkan, que HyperPlay pasa al Vulkan del teléfono.
static bool managedEnvVar(const std::string& upper, std::string& val) {
    // SDL2/SDL3 de cualquier juego: mandos solo por XInput (el puente de HyperPlay). DirectInput
    // no está registrado (SDL2 da su CoCreateInstance fallido por fatal y SDL_Init falla) y
    // HIDAPI/RawInput buscan dispositivos USB que no existen.
    if (upper == "SDL_DIRECTINPUT_ENABLED" || upper == "SDL_JOYSTICK_HIDAPI" || upper == "SDL_JOYSTICK_RAWINPUT" ||
        upper == "SDL_JOYSTICK_WGI" || upper == "SDL_JOYSTICK_GAMEINPUT" || upper == "SDL_JOYSTICK_DIRECTINPUT") {
        val = "0";
        return true;
    }
    if (upper == "SDL_XINPUT_ENABLED") { val = "1"; return true; }
    if (!managed::active()) return false;
    if (upper == "MONO_PATH") {
        // fna/ (FNA + reenvíos XNA) y las bibliotecas de clases (System.dll...) sin copiar el GAC.
        const std::string root = managed::kWinRoot;
        val = root + "\\fna;" + root + "\\lib\\mono\\4.5;" + root + "\\lib\\mono\\4.5\\Facades";
        return true;
    }
    if (upper == "SDL_GPU_DRIVER" || upper == "SDL_HINT_GPU_DRIVER") { val = "vulkan"; return true; }
    // Mandos: solo XInput (el puente de HyperPlay). HIDAPI/RawInput/WGI buscan hid.dll, libusb y
    // dispositivos USB en cada sondeo (cientos de LoadLibrary fallidos por segundo).
    if (upper == "SDL_JOYSTICK_HIDAPI" || upper == "SDL_JOYSTICK_RAWINPUT" || upper == "SDL_JOYSTICK_WGI" ||
        upper == "SDL_JOYSTICK_GAMEINPUT" || upper == "SDL_JOYSTICK_DIRECTINPUT") { val = "0"; return true; }
    if (upper == "FNA3D_FORCE_DRIVER") { val = "SDLGPU"; return true; }
    return false;
}

uint32_t NativeDllInterceptor::win_GetEnvironmentVariableW(const wchar_t* lpName, wchar_t* lpBuffer, uint32_t nSize) {
    s_kernel32Calls.fetch_add(1);
    if (lpName == nullptr) {
        win_SetLastError(87); // ERROR_INVALID_PARAMETER
        return 0;
    }
    std::string name = wcharToUtf8(lpName);
    std::string upper = name;
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    LOGI("[EnvVar] GetEnvironmentVariableW('%s')", name.c_str());

    std::string val;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(s_envMutex);
        auto it = s_customEnvVars.find(upper);
        if (it != s_customEnvVars.end()) {
            val = it->second;
            found = true;
        }
    }

    // Diagnóstico de .NET Framework (setprop debug.hyperplay.clrloadlog 1): mscoreei escribe en la
    // carpeta del juego por qué elige o rechaza cada runtime.
    if (!found && upper == "COMPLUS_CLRLOADLOGDIR") {
        char lv[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.clrloadlog", lv) > 0 && lv[0] == '1') { val = "D:\\"; found = true; }
    }
    if (!found) found = managedEnvVar(upper, val);
    // Diagnóstico (setprop debug.hyperplay.dotnettrace 1): EventPipe lee aquí su configuración.
    if (!found && upper.rfind("DOTNET_", 0) == 0) {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.dotnettrace", v) > 0 && v[0] == '1') {
            if (upper == "DOTNET_ENABLEDIAGNOSTICS") { val = "1"; found = true; }
            else if (upper == "DOTNET_ENABLEEVENTPIPE") { val = "1"; found = true; }
            else if (upper == "DOTNET_EVENTPIPEOUTPUTPATH") { val = "D:\\hp-dotnet.nettrace"; found = true; }
            else if (upper == "DOTNET_EVENTPIPECONFIG") { val = "Microsoft-Windows-DotNETRuntime:0x800C:5"; found = true; }
            else if (upper == "DOTNET_EVENTPIPERUNDOWN") { val = "0"; found = true; }
            else if (upper == "DOTNET_EVENTPIPEOUTPUTSTREAMING") { val = "1"; found = true; }
        }
    }
    if (!found) {
        if (upper == "NUMBER_OF_PROCESSORS") { val = "8"; found = true; }
        // .NET (CoreCLR): sin depurador/diagnóstico (canales con named pipes que no hay en Android).
        else if (upper == "DOTNET_ENABLEDIAGNOSTICS") { val = "0"; found = true; }
        // Sin ICU ni NLS de Windows: cultura invariante (fechas y números como en inglés; los
        // textos del juego vienen de sus propios archivos de idioma).
        else if (upper == "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT") { val = "1"; found = true; }
        // RyuJIT: solo la base x64 (SSE2), sin AVX/BMI/FMA, que son lo menos probado del traductor.
        else if (upper == "DOTNET_ENABLEHWINTRINSIC") { val = "0"; found = true; }
        // GC sin hilo de fondo concurrente: su marcado concurrente depende del orden de memoria
        // fuerte de x86 (TSO), que el código traducido a ARM no garantiza en todos los casos.
        else if (upper == "DOTNET_GCCONCURRENT") { val = "0"; found = true; }
        // SDL2 (MonoGame/FNA): sin cola de mensajes real la ventana nunca recibe WM_ACTIVATE/WM_SETFOCUS,
        // SDL no le da el foco y descarta el mando. Con esta pista acepta el mando sin foco. HIDAPI y
        // RawInput fuera: el mando llega por XInput y así no busca hid.dll decenas de veces por segundo.
        else if (upper == "SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS") { val = "1"; found = true; }
        else if (upper == "SDL_JOYSTICK_HIDAPI" || upper == "SDL_JOYSTICK_RAWINPUT") { val = "0"; found = true; }
        // Experimento (setprop debug.hyperplay.r2r 0): sin código ReadyToRun, todo por el JIT.
        else if (upper == "DOTNET_READYTORUN") {
            char v[PROP_VALUE_MAX] = {};
            if (__system_property_get("debug.hyperplay.r2r", v) > 0 && v[0] == '0') { val = "0"; found = true; }
        }
        else if (upper == "PROCESSOR_ARCHITECTURE") { val = (s_activeImage && s_activeImage->is32Bit) ? "x86" : "AMD64"; found = true; }
        else if (upper == "PROCESSOR_IDENTIFIER") { val = "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD"; found = true; }
        else if (upper == "PROCESSOR_LEVEL") { val = "6"; found = true; }
        else if (upper == "PROCESSOR_REVISION") { val = "0000"; found = true; }
        else if (upper == "OS") { val = "Windows_NT"; found = true; }
        else if (upper == "SYSTEMROOT" || upper == "WINDIR") { val = "C:\\Windows"; found = true; }
        else if (upper == "SYSTEMDRIVE") { val = "C:"; found = true; }
        else if (upper == "HOMEDRIVE") { val = "C:"; found = true; }
        else if (upper == "HOMEPATH") { val = "\\Users\\HyperPlay"; found = true; }
        else if (upper == "USERPROFILE") { val = "C:\\Users\\HyperPlay"; found = true; }
        else if (upper == "APPDATA") { val = "C:\\Users\\HyperPlay\\AppData\\Roaming"; found = true; }
        else if (upper == "LOCALAPPDATA") { val = "C:\\Users\\HyperPlay\\AppData\\Local"; found = true; }
        else if (upper == "TEMP" || upper == "TMP") { val = "C:\\Temp"; found = true; }
        else if (upper == "PROGRAMFILES") { val = "C:\\Program Files"; found = true; }
        else if (upper == "COMMONPROGRAMFILES") { val = "C:\\Program Files\\Common Files"; found = true; }
    }

    if (!found) {
        win_SetLastError(203); // ERROR_ENVVAR_NOT_FOUND
        return 0;
    }

    const uint32_t needed = static_cast<uint32_t>(guestWStringLength(val));
    if (lpBuffer == nullptr || nSize <= needed) {
        return needed + 1; // Return required buffer size including null terminator
    }
    writeGuestWString(lpBuffer, nSize, val);
    return needed;
}

uint32_t NativeDllInterceptor::win_GetEnvironmentVariableA(const char* lpName, char* lpBuffer, uint32_t nSize) {
    s_kernel32Calls.fetch_add(1);
    if (lpName == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    LOGI("[EnvVar] GetEnvironmentVariableA('%s')", lpName);
    std::string upper = lpName;
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string val;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(s_envMutex);
        auto it = s_customEnvVars.find(upper);
        if (it != s_customEnvVars.end()) {
            val = it->second;
            found = true;
        }
    }

    // Diagnóstico de .NET Framework (setprop debug.hyperplay.clrloadlog 1): mscoreei escribe en la
    // carpeta del juego por qué elige o rechaza cada runtime.
    if (!found && upper == "COMPLUS_CLRLOADLOGDIR") {
        char lv[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.clrloadlog", lv) > 0 && lv[0] == '1') { val = "D:\\"; found = true; }
    }
    if (!found) found = managedEnvVar(upper, val);
    // Diagnóstico (setprop debug.hyperplay.dotnettrace 1): EventPipe lee aquí su configuración.
    if (!found && upper.rfind("DOTNET_", 0) == 0) {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.dotnettrace", v) > 0 && v[0] == '1') {
            if (upper == "DOTNET_ENABLEDIAGNOSTICS") { val = "1"; found = true; }
            else if (upper == "DOTNET_ENABLEEVENTPIPE") { val = "1"; found = true; }
            else if (upper == "DOTNET_EVENTPIPEOUTPUTPATH") { val = "D:\\hp-dotnet.nettrace"; found = true; }
            else if (upper == "DOTNET_EVENTPIPECONFIG") { val = "Microsoft-Windows-DotNETRuntime:0x800C:5"; found = true; }
            else if (upper == "DOTNET_EVENTPIPERUNDOWN") { val = "0"; found = true; }
            else if (upper == "DOTNET_EVENTPIPEOUTPUTSTREAMING") { val = "1"; found = true; }
        }
    }
    if (!found) {
        if (upper == "NUMBER_OF_PROCESSORS") { val = "8"; found = true; }
        // .NET (CoreCLR): sin depurador/diagnóstico (canales con named pipes que no hay en Android).
        else if (upper == "DOTNET_ENABLEDIAGNOSTICS") { val = "0"; found = true; }
        // Sin ICU ni NLS de Windows: cultura invariante (fechas y números como en inglés; los
        // textos del juego vienen de sus propios archivos de idioma).
        else if (upper == "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT") { val = "1"; found = true; }
        // RyuJIT: solo la base x64 (SSE2), sin AVX/BMI/FMA, que son lo menos probado del traductor.
        else if (upper == "DOTNET_ENABLEHWINTRINSIC") { val = "0"; found = true; }
        // GC sin hilo de fondo concurrente: su marcado concurrente depende del orden de memoria
        // fuerte de x86 (TSO), que el código traducido a ARM no garantiza en todos los casos.
        else if (upper == "DOTNET_GCCONCURRENT") { val = "0"; found = true; }
        // SDL2 (MonoGame/FNA): sin cola de mensajes real la ventana nunca recibe WM_ACTIVATE/WM_SETFOCUS,
        // SDL no le da el foco y descarta el mando. Con esta pista acepta el mando sin foco. HIDAPI y
        // RawInput fuera: el mando llega por XInput y así no busca hid.dll decenas de veces por segundo.
        else if (upper == "SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS") { val = "1"; found = true; }
        else if (upper == "SDL_JOYSTICK_HIDAPI" || upper == "SDL_JOYSTICK_RAWINPUT") { val = "0"; found = true; }
        // Experimento (setprop debug.hyperplay.r2r 0): sin código ReadyToRun, todo por el JIT.
        else if (upper == "DOTNET_READYTORUN") {
            char v[PROP_VALUE_MAX] = {};
            if (__system_property_get("debug.hyperplay.r2r", v) > 0 && v[0] == '0') { val = "0"; found = true; }
        }
        else if (upper == "PROCESSOR_ARCHITECTURE") { val = (s_activeImage && s_activeImage->is32Bit) ? "x86" : "AMD64"; found = true; }
        else if (upper == "OS") { val = "Windows_NT"; found = true; }
        else if (upper == "SYSTEMROOT" || upper == "WINDIR") { val = "C:\\Windows"; found = true; }
        else if (upper == "TEMP" || upper == "TMP") { val = "C:\\Temp"; found = true; }
        else if (upper == "USERPROFILE") { val = "C:\\Users\\HyperPlay"; found = true; }
        else if (upper == "APPDATA") { val = "C:\\Users\\HyperPlay\\AppData\\Roaming"; found = true; }
        else if (upper == "LOCALAPPDATA") { val = "C:\\Users\\HyperPlay\\AppData\\Local"; found = true; }
    }

    if (!found) {
        win_SetLastError(203);
        return 0;
    }

    const uint32_t len = static_cast<uint32_t>(val.size());
    if (lpBuffer == nullptr || nSize <= len) {
        return len + 1;
    }
    std::memcpy(lpBuffer, val.c_str(), len + 1);
    return len;
}

int32_t NativeDllInterceptor::win_SetEnvironmentVariableW(const wchar_t* lpName, const wchar_t* lpValue) {
    s_kernel32Calls.fetch_add(1);
    if (lpName == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    std::string name = wcharToUtf8(lpName);
    for (char& c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::lock_guard<std::mutex> lock(s_envMutex);
    if (lpValue == nullptr) {
        s_customEnvVars.erase(name);
    } else {
        s_customEnvVars[name] = wcharToUtf8(lpValue);
    }
    return 1;
}

// Directorio actual de los invitados de 32 bits (Unreal 3 hace SetCurrentDirectory y luego abre
// "DevilGame\CookedPCConsole\..." en relativo). Los de 64 bits siguen con D:\ fijo.
namespace guestcwd {
std::string current() {
    const std::string c = BionicPeLoader::guestCwd();
    std::string w = c.empty() ? BionicPeLoader::exeWinDir() : c;
    for (char& ch : w) if (ch == '/') ch = '\\';
    return w;
}
int32_t set(const std::string& path) {
    if (s_activeImage == nullptr || path.empty()) return 0;
    const std::string canon = BionicPeLoader::canonicalWinPath(path);
    const std::string host = BionicPeLoader::resolveVirtualPath(*s_activeImage, canon);
    struct stat st{};
    if (stat(host.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        NativeDllInterceptor::win_SetLastError(3);  // ERROR_PATH_NOT_FOUND
        LOGW("SetCurrentDirectory('%s') -> no existe (%s)", path.c_str(), host.c_str());
        return 0;
    }
    BionicPeLoader::setGuestCwd(canon);
    LOGI("SetCurrentDirectory('%s') -> %s (%s)", path.c_str(), canon.c_str(), host.c_str());
    return 1;
}
uint32_t SetCurrentDirectoryA(const char* p) { return p != nullptr ? static_cast<uint32_t>(set(p)) : 0u; }
uint32_t GetCurrentDirectoryA(uint32_t len, char* buf) {
    const std::string cur = current();
    if (buf == nullptr || len <= cur.size()) return static_cast<uint32_t>(cur.size() + 1);
    std::memcpy(buf, cur.c_str(), cur.size() + 1);
    return static_cast<uint32_t>(cur.size());
}
}  // namespace guestcwd

uint32_t NativeDllInterceptor::win_GetCurrentDirectoryW(uint32_t nBufferLength, wchar_t* lpBuffer) {
    s_kernel32Calls.fetch_add(1);
    const std::string cur = (s_activeImage != nullptr && s_activeImage->is32Bit) ? guestcwd::current() : "D:\\";
    const uint32_t len = static_cast<uint32_t>(cur.size());
    if (lpBuffer == nullptr || nBufferLength <= len) {
        return len + 1;
    }
    writeGuestWString(lpBuffer, nBufferLength, cur);
    return len;
}

int32_t NativeDllInterceptor::win_SetCurrentDirectoryW(const wchar_t* lpPathName) {
    s_kernel32Calls.fetch_add(1);
    if (s_activeImage != nullptr && s_activeImage->is32Bit && lpPathName != nullptr)
        return guestcwd::set(wcharToUtf8(lpPathName));
    return 1;
}

uint32_t NativeDllInterceptor::win_GetDriveTypeW(const wchar_t* /*lpRootPathName*/) {
    s_kernel32Calls.fetch_add(1);
    return 3; // DRIVE_FIXED
}

uint32_t NativeDllInterceptor::win_GetLogicalDriveStringsW(uint32_t nBufferLength, wchar_t* lpBuffer) {
    s_kernel32Calls.fetch_add(1);
    constexpr uint16_t drives[] = { 'C', ':', '\\', 0, 'D', ':', '\\', 0, 0 };
    constexpr uint32_t count = sizeof(drives) / sizeof(uint16_t);
    if (lpBuffer == nullptr || nBufferLength < count) {
        return count;
    }
    std::memcpy(lpBuffer, drives, sizeof(drives));
    return count - 1;
}

int32_t NativeDllInterceptor::win_GetDiskFreeSpaceExW(
    const wchar_t* /*lpDirectoryName*/, uint64_t* lpFreeBytesAvailableToCaller,
    uint64_t* lpTotalNumberOfBytes, uint64_t* lpTotalNumberOfFreeBytes) {
    s_kernel32Calls.fetch_add(1);
    constexpr uint64_t k100GB = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    if (lpFreeBytesAvailableToCaller) *lpFreeBytesAvailableToCaller = k100GB;
    if (lpTotalNumberOfBytes) *lpTotalNumberOfBytes = k100GB * 2;
    if (lpTotalNumberOfFreeBytes) *lpTotalNumberOfFreeBytes = k100GB;
    return 1;
}

int32_t NativeDllInterceptor::win_GetVolumeInformationW(
    const wchar_t* /*lpRootPathName*/, wchar_t* lpVolumeNameBuffer, uint32_t nVolumeNameSize,
    uint32_t* lpVolumeSerialNumber, uint32_t* lpMaximumComponentLength,
    uint32_t* lpFileSystemFlags, wchar_t* lpFileSystemNameBuffer, uint32_t nFileSystemNameSize) {
    s_kernel32Calls.fetch_add(1);
    if (lpVolumeNameBuffer && nVolumeNameSize > 0) {
        writeGuestWString(lpVolumeNameBuffer, nVolumeNameSize, "HyperPlay");
    }
    if (lpVolumeSerialNumber) *lpVolumeSerialNumber = 0x820CAFE;
    if (lpMaximumComponentLength) *lpMaximumComponentLength = 255;
    if (lpFileSystemFlags) *lpFileSystemFlags = 0x00000007; // CASE_PRESERVED | CASE_SENSITIVE | UNICODE_ON_DISK
    if (lpFileSystemNameBuffer && nFileSystemNameSize > 0) {
        writeGuestWString(lpFileSystemNameBuffer, nFileSystemNameSize, "NTFS");
    }
    return 1;
}

int32_t NativeDllInterceptor::win_DisableThreadLibraryCalls(void* /*hLibModule*/) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_TryEnterCriticalSection(void* lpCriticalSection) {
    s_kernel32Calls.fetch_add(1);
    // Nunca espera: si otro hilo la tiene, devuelve 0. Antes esperaba como EnterCriticalSection;
    // va en la lista rápida como "no bloqueante", así que el hilo dormía marcado como "ejecutando"
    // y el GC de Mono (que lo usa mientras suspende hilos) se quedaba esperando para siempre
    // (Silksong: pantalla negra al arrancar, 1 de cada ~5 veces).
    if (lpCriticalSection == nullptr) return 0;
    pthread_mutex_t* m = lpCriticalSection != nullptr ? csMutex(lpCriticalSection) : nullptr;
    if (m == nullptr) { static std::atomic<int> s_nulls{0}; if (s_nulls.fetch_add(1) < 5) LOGW("TryEnterCriticalSection(%p): sin mutex (CS no inicializada)", lpCriticalSection); }
    if (m == nullptr || pthread_mutex_trylock(m) != 0) return 0;
    if (lowmem::enabled()) {  // RecursionCount/OwningThread como en EnterCriticalSection
        auto* f = static_cast<int32_t*>(lpCriticalSection);
        if (f[2]++ == 0) __atomic_store_n(&f[3], static_cast<int32_t>(gettid()), __ATOMIC_RELAXED);
    }
    return 1;
}

void NativeDllInterceptor::win_InitializeSRWLock(void* srwLock) {
    s_kernel32Calls.fetch_add(1);
    if (srwLock != nullptr) {
        lowmem::storePtr(srwLock, uint64_t{0});  // SRWLOCK: 4 bytes en 32 bits
    }
}

void NativeDllInterceptor::win_GetCurrentProcessorNumberEx(void* procNumber) {
    s_kernel32Calls.fetch_add(1);
    if (procNumber != nullptr) {
        uint8_t* p = static_cast<uint8_t*>(procNumber);
        p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0;
    }
}

// FlushInstructionCache: el huésped avisa de que reescribió código (CoreCLR al parchear sus
// precode/stubs, otros JIT). Se descartan las traducciones del rango en todos los hilos y se
// devuelve TRUE (antes era un stub que devolvía FALSE y no invalidaba nada).
int32_t NativeDllInterceptor::win_FlushInstructionCache(void* /*process*/, const void* address, size_t size) {
    s_kernel32Calls.fetch_add(1);
    if (address != nullptr && size != 0) {
        OryonEmitterBackend::invalidateCodeRangeAll(reinterpret_cast<uintptr_t>(address), size);
    }
    return 1;
}

namespace {
// Trabajo del pool. 'stub' es código x86-64 del huésped (memoria ejecutable de VirtualAlloc):
//   xor ecx,ecx; mov rdx,ctx; mov r8,trabajo; mov rax,callback; jmp rax
// Sirve de punto de entrada de CreateThread: el callback recibe (instancia=NULL, ctx, trabajo) y su
// RET vuelve a la salida normal del hilo.
struct TpWork {
    uint8_t* stub = nullptr;
    std::mutex mtx;
    std::vector<void*> threads;  // hilos lanzados y aún sin esperar
};

void tpWaitAll(TpWork* w) {
    std::vector<void*> pending;
    {
        std::lock_guard<std::mutex> lk(w->mtx);
        pending.swap(w->threads);
    }
    for (void* h : pending) {
        NativeDllInterceptor::win_WaitForSingleObject(h, 0xFFFFFFFFu);
        NativeDllInterceptor::win_CloseHandle(h);
    }
}
}  // namespace

void* NativeDllInterceptor::win_CreateThreadpoolWork(void* pfnwk, void* pv, void* /*pcbe*/) {
    s_kernel32Calls.fetch_add(1);
    if (pfnwk == nullptr) {
        win_SetLastError(87);
        return nullptr;
    }
    auto* w = new TpWork();
    w->stub = static_cast<uint8_t*>(win_VirtualAlloc(nullptr, 64, 0x3000, 0x40));  // RWX
    if (w->stub == nullptr) {
        delete w;
        win_SetLastError(8);
        return nullptr;
    }
    uint8_t* p = w->stub;
    const auto put64 = [&p](uint64_t v) { std::memcpy(p, &v, 8); p += 8; };
    *p++ = 0x31; *p++ = 0xC9;                                                    // xor ecx, ecx
    *p++ = 0x48; *p++ = 0xBA; put64(reinterpret_cast<uint64_t>(pv));            // mov rdx, ctx
    *p++ = 0x49; *p++ = 0xB8; put64(reinterpret_cast<uint64_t>(w));             // mov r8, trabajo
    *p++ = 0x48; *p++ = 0xB8; put64(reinterpret_cast<uint64_t>(pfnwk));         // mov rax, callback
    *p++ = 0xFF; *p++ = 0xE0;                                                    // jmp rax
    OryonEmitterBackend::invalidateCodeRangeAll(reinterpret_cast<uintptr_t>(w->stub), 64);
    LOGI("CreateThreadpoolWork(callback=%p, ctx=%p) -> %p", pfnwk, pv, static_cast<void*>(w));
    return w;
}

void NativeDllInterceptor::win_SubmitThreadpoolWork(void* pwk) {
    s_kernel32Calls.fetch_add(1);
    auto* w = static_cast<TpWork*>(pwk);
    if (w == nullptr) return;
    void* h = win_CreateThread(nullptr, 0, w->stub, nullptr, 0, nullptr);
    if (h == nullptr) {
        LOGE("SubmitThreadpoolWork(%p): no se pudo crear el hilo", pwk);
        return;
    }
    std::lock_guard<std::mutex> lk(w->mtx);
    w->threads.push_back(h);
}

void NativeDllInterceptor::win_WaitForThreadpoolWorkCallbacks(void* pwk, int32_t /*fCancelPendingCallbacks*/) {
    s_kernel32Calls.fetch_add(1);
    // Cada envío arranca su hilo enseguida: no hay cola que cancelar, solo esperar.
    if (pwk != nullptr) tpWaitAll(static_cast<TpWork*>(pwk));
}

void NativeDllInterceptor::win_CloseThreadpoolWork(void* pwk) {
    s_kernel32Calls.fetch_add(1);
    auto* w = static_cast<TpWork*>(pwk);
    if (w == nullptr) return;
    // Windows NO espera: libera el objeto de forma asíncrona cuando acaban los callbacks pendientes
    // (OpenAL Soft cierra el trabajo mientras su hilo sigue vivo). Se sueltan los handles sin esperar y el
    // objeto (y su stub de 64 bytes) se queda: un callback aún en marcha recibió su puntero.
    std::vector<void*> pending;
    {
        std::lock_guard<std::mutex> lk(w->mtx);
        pending.swap(w->threads);
    }
    for (void* h : pending) win_CloseHandle(h);
}

namespace {
struct IocpPacket { uint32_t bytes; uint64_t key; void* overlapped; };
struct Iocp {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<IocpPacket> queue;
};
std::mutex s_iocpMtx;
std::unordered_map<uintptr_t, std::shared_ptr<Iocp>> s_iocps;
std::atomic<uintptr_t> s_nextIocp{0x7E000000};

std::shared_ptr<Iocp> findIocp(void* h) {
    std::lock_guard<std::mutex> lk(s_iocpMtx);
    const auto it = s_iocps.find(reinterpret_cast<uintptr_t>(h));
    return it == s_iocps.end() ? nullptr : it->second;
}

// Espera hasta 'ms' (INFINITE = 0xFFFFFFFF) a que haya al menos un paquete. true si lo hay.
bool iocpWait(Iocp& p, std::unique_lock<std::mutex>& lk, uint32_t ms) {
    if (ms == 0xFFFFFFFFu) {
        p.cv.wait(lk, [&] { return !p.queue.empty(); });
        return true;
    }
    return p.cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return !p.queue.empty(); });
}
}  // namespace

void* NativeDllInterceptor::win_CreateIoCompletionPort(void* fileHandle, void* existingPort, uint64_t, uint32_t) {
    s_kernel32Calls.fetch_add(1);
    if (reinterpret_cast<intptr_t>(fileHandle) != -1) {
        // Asociar un archivo: las E/S de HyperPlay terminan de forma síncrona, no generan paquetes.
        if (existingPort != nullptr) return existingPort;
    }
    const uintptr_t h = s_nextIocp.fetch_add(4);
    {
        std::lock_guard<std::mutex> lk(s_iocpMtx);
        s_iocps[h] = std::make_shared<Iocp>();
    }
    LOGI("CreateIoCompletionPort -> %p", reinterpret_cast<void*>(h));
    return reinterpret_cast<void*>(h);
}

int32_t NativeDllInterceptor::win_PostQueuedCompletionStatus(void* port, uint32_t bytes, uint64_t key, void* overlapped) {
    s_kernel32Calls.fetch_add(1);
    auto p = findIocp(port);
    if (!p) {
        win_SetLastError(6);  // ERROR_INVALID_HANDLE
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(p->mtx);
        p->queue.push_back({bytes, key, overlapped});
    }
    p->cv.notify_one();
    return 1;
}

int32_t NativeDllInterceptor::win_GetQueuedCompletionStatus(void* port, uint32_t* bytes, uint64_t* key, void** overlapped,
                                                            uint32_t ms) {
    s_kernel32Calls.fetch_add(1);
    lowmem::storePtr(overlapped, uint64_t{0});
    auto p = findIocp(port);
    if (!p) {
        win_SetLastError(6);
        return 0;
    }
    std::unique_lock<std::mutex> lk(p->mtx);
    if (!iocpWait(*p, lk, ms)) {
        win_SetLastError(258);  // WAIT_TIMEOUT
        return 0;
    }
    const IocpPacket pk = p->queue.front();
    p->queue.pop_front();
    if (bytes) *bytes = pk.bytes;
    lowmem::storePtr(key, pk.key);  // ULONG_PTR
    lowmem::storePtr(overlapped, pk.overlapped);
    return 1;
}

int32_t NativeDllInterceptor::win_GetQueuedCompletionStatusEx(void* port, void* entries, uint32_t count, uint32_t* removed,
                                                              uint32_t ms, int32_t) {
    s_kernel32Calls.fetch_add(1);
    if (removed) *removed = 0;
    auto p = findIocp(port);
    if (!p || entries == nullptr || count == 0) {
        win_SetLastError(p ? 87 : 6);
        return 0;
    }
    std::unique_lock<std::mutex> lk(p->mtx);
    if (!iocpWait(*p, lk, ms)) {
        win_SetLastError(258);
        return 0;
    }
    // OVERLAPPED_ENTRY x64: lpCompletionKey, lpOverlapped, Internal, dwNumberOfBytesTransferred (32 bytes).
    auto* e = static_cast<uint8_t*>(entries);
    uint32_t n = 0;
    while (n < count && !p->queue.empty()) {
        const IocpPacket pk = p->queue.front();
        p->queue.pop_front();
        uint8_t* row = e + static_cast<size_t>(n) * 32;
        std::memset(row, 0, 32);
        std::memcpy(row, &pk.key, 8);
        std::memcpy(row + 8, &pk.overlapped, 8);
        std::memcpy(row + 24, &pk.bytes, 4);
        ++n;
    }
    if (removed) *removed = n;
    return 1;
}

void NativeDllInterceptor::win_FlushProcessWriteBuffers() {
    s_kernel32Calls.fetch_add(1);
#if defined(__aarch64__)
    asm volatile("isb sy" ::: "memory");
#endif
}

void* NativeDllInterceptor::win_AddVectoredExceptionHandler(uint32_t first, void* handler) {
    s_kernel32Calls.fetch_add(1);
    if (handler == nullptr) return nullptr;
    const uintptr_t id = s_nextVehId.fetch_add(0x10);
    {
        std::lock_guard<std::mutex> lk(s_vehMutex);
        const std::pair<uintptr_t, uintptr_t> entry{id, reinterpret_cast<uintptr_t>(handler)};
        if (first != 0) s_vehHandlers.insert(s_vehHandlers.begin(), entry);
        else s_vehHandlers.push_back(entry);
    }
    LOGI("AddVectoredExceptionHandler(first=%u, handler=%p) -> %p", first, handler, reinterpret_cast<void*>(id));
    return reinterpret_cast<void*>(id);
}

uint32_t NativeDllInterceptor::win_RemoveVectoredExceptionHandler(void* handle) {
    s_kernel32Calls.fetch_add(1);
    std::lock_guard<std::mutex> lk(s_vehMutex);
    for (auto it = s_vehHandlers.begin(); it != s_vehHandlers.end(); ++it) {
        if (it->first == reinterpret_cast<uintptr_t>(handle)) {
            s_vehHandlers.erase(it);
            return 1;
        }
    }
    return 0;
}

int32_t NativeDllInterceptor::win_SetThreadStackGuarantee(uint32_t* stackSizeInBytes) {
    s_kernel32Calls.fetch_add(1);
    if (stackSizeInBytes != nullptr) {
        *stackSizeInBytes = 65536;
    }
    return 1;
}

uint32_t NativeDllInterceptor::win_GetUserDefaultLCID() {
    s_kernel32Calls.fetch_add(1);
    return 0x0409; // en-US
}

// Nombre de la configuración regional del usuario: "en-US", como GetUserDefaultLCID. CoreCLR lo
// usa para elegir el idioma de sus mensajes de error (antes era un stub que devolvía 0 y todas
// las excepciones del runtime salían sin mensaje).
int32_t NativeDllInterceptor::win_GetUserDefaultLocaleName(char16_t* lpLocaleName, int32_t cchLocaleName) {
    s_kernel32Calls.fetch_add(1);
    static constexpr char16_t kName[] = u"en-US";
    constexpr int32_t kLen = static_cast<int32_t>(sizeof(kName) / sizeof(kName[0]));  // con el 0 final
    if (lpLocaleName == nullptr || cchLocaleName < kLen) return 0;
    std::memcpy(lpLocaleName, kName, sizeof(kName));
    return kLen;
}

// LoadStringW: busca la cadena en los recursos RT_STRING del módulo (tablas de 16 cadenas
// UTF-16 con longitud delante). Con cchBufferMax == 0 devuelve un puntero de solo lectura a la
// cadena, como Windows. CoreCLR lee así sus mensajes de error de mscorrc.dll.
int32_t NativeDllInterceptor::win_LoadStringW(void* hInstance, uint32_t uID, char16_t* lpBuffer, int32_t cchBufferMax) {
    s_kernel32Calls.fetch_add(1);
    if (s_activeImage == nullptr || lpBuffer == nullptr) return 0;
    uID &= 0xFFFFu;  // como Windows: los id de recurso son de 16 bits (CoreCLR pasa HRESULT enteros)
    const PeInfo* info = nullptr;
    auto* base = static_cast<uint8_t*>(hInstance != nullptr ? hInstance : s_activeImage->baseAddress);
    if (base == s_activeImage->baseAddress) {
        info = &s_activeImage->peInfo;
    } else {
        for (const auto& [name, dll] : s_activeImage->localDlls) {
            if (dll.base == base) { info = &dll.peInfo; break; }
        }
    }
    if (info == nullptr || info->resourceRva == 0 || info->resourceSize < 16) return 0;
    const uint8_t* root = base + info->resourceRva;
    const uint32_t rsrcSize = info->resourceSize;
    const auto rd16 = [](const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; };
    const auto rd32 = [](const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; };
    // Entrada de un directorio con ese id (o la primera, con wantAny): offset relativo a root.
    const auto findEntry = [&](uint32_t dirOff, uint32_t id, bool wantAny, uint32_t& out) {
        if (dirOff + 16 > rsrcSize) return false;
        const uint8_t* dir = root + dirOff;
        const uint32_t count = static_cast<uint32_t>(rd16(dir + 12)) + rd16(dir + 14);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t eOff = dirOff + 16 + i * 8;
            if (eOff + 8 > rsrcSize) return false;
            const uint32_t name = rd32(root + eOff);
            if (!wantAny && ((name & 0x80000000u) != 0 || name != id)) continue;
            out = rd32(root + eOff + 4);
            return true;
        }
        return false;
    };
    uint32_t e = 0;
    if (!findEntry(0, 6 /* RT_STRING */, false, e) || (e & 0x80000000u) == 0) return 0;
    if (!findEntry(e & 0x7FFFFFFFu, (uID >> 4) + 1, false, e) || (e & 0x80000000u) == 0) return 0;
    if (!findEntry(e & 0x7FFFFFFFu, 0, true, e) || (e & 0x80000000u) != 0) return 0;  // primer idioma
    if (e + 16 > rsrcSize) return 0;
    const uint32_t dataRva = rd32(root + e);
    const uint32_t dataSize = rd32(root + e + 4);
    if (dataRva == 0 || static_cast<uint64_t>(dataRva) + dataSize > info->imageSize) return 0;
    const uint8_t* p = base + dataRva;
    const uint8_t* end = p + dataSize;
    for (uint32_t i = 0; i < (uID & 15u); ++i) {
        if (p + 2 > end) return 0;
        p += 2 + static_cast<size_t>(rd16(p)) * 2;
    }
    if (p + 2 > end) return 0;
    const int32_t len = rd16(p);
    const auto* str = reinterpret_cast<const char16_t*>(p + 2);
    if (len == 0 || p + 2 + static_cast<size_t>(len) * 2 > end) return 0;
    if (cchBufferMax == 0) {
        lowmem::storePtr(lpBuffer, str);  // puntero de solo lectura (4 bytes en 32 bits)
        return len;
    }
    const int32_t n = std::min(len, cchBufferMax - 1);
    std::memcpy(lpBuffer, str, static_cast<size_t>(n) * 2);
    lpBuffer[n] = 0;
    return n;
}

int32_t NativeDllInterceptor::win_EnumSystemLocalesW(void* /*lpLocaleEnumProc*/, uint32_t /*dwFlags*/) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_IsValidLocale(uint32_t /*locale*/, uint32_t /*dwFlags*/) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

int32_t NativeDllInterceptor::win_CompareStringW(
    uint32_t /*locale*/, uint32_t /*dwCmpFlags*/, const wchar_t* lpString1, int32_t cchCount1,
    const wchar_t* lpString2, int32_t cchCount2) {
    s_kernel32Calls.fetch_add(1);
    if (lpString1 == nullptr || lpString2 == nullptr) return 2; // CSTR_EQUAL
    const std::string s1 = wcharToUtf8(lpString1);
    const std::string s2 = wcharToUtf8(lpString2);
    (void)cchCount1; (void)cchCount2;
    if (s1 < s2) return 1; // CSTR_LESS_THAN
    if (s1 > s2) return 3; // CSTR_GREATER_THAN
    return 2; // CSTR_EQUAL
}

int32_t NativeDllInterceptor::win_GetLocaleInfoW(uint32_t /*locale*/, uint32_t lcType, wchar_t* lpLCData, int32_t cchData) {
    s_kernel32Calls.fetch_add(1);
    std::string result = "0409";
    const uint32_t type = lcType & 0xFFFF;
    if (type == 0x000E) result = "."; // LOCALE_SDECIMAL
    else if (type == 0x000F) result = ","; // LOCALE_STHOUSAND
    else if (type == 0x0002) result = "enu"; // LOCALE_SABBREVLANGNAME
    else if (type == 0x0001) result = "0409"; // LOCALE_ILANGUAGE
    else if (type == 0x0007) result = "en-US"; // LOCALE_SNAME
    const int32_t needed = static_cast<int32_t>(result.size()) + 1;
    if (lpLCData == nullptr || cchData <= 0) return needed;
    writeGuestWString(lpLCData, static_cast<size_t>(cchData), result);
    return std::min(cchData, needed);
}

int32_t NativeDllInterceptor::win_GetLocaleInfoEx(const wchar_t* /*lpLocaleName*/, uint32_t lcType, wchar_t* lpLCData, int32_t cchData) {
    return win_GetLocaleInfoW(0x0409, lcType, lpLCData, cchData);
}

uint32_t NativeDllInterceptor::win_GetTimeZoneInformation(void* lpTimeZoneInformation) {
    s_kernel32Calls.fetch_add(1);
    if (lpTimeZoneInformation != nullptr) {
        std::memset(lpTimeZoneInformation, 0, 172); // sizeof(TIME_ZONE_INFORMATION)
    }
    return 0; // TIME_ZONE_ID_UNKNOWN
}

int32_t NativeDllInterceptor::win_SystemTimeToTzSpecificLocalTime(const void*, const void* lpUniversalTime, void* lpLocalTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpUniversalTime && lpLocalTime) {
        std::memcpy(lpLocalTime, lpUniversalTime, 16); // sizeof(SYSTEMTIME)
    }
    return 1;
}

int32_t NativeDllInterceptor::win_TzSpecificLocalTimeToSystemTime(const void*, const void* lpLocalTime, void* lpUniversalTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpLocalTime && lpUniversalTime) {
        std::memcpy(lpUniversalTime, lpLocalTime, 16);
    }
    return 1;
}

int32_t NativeDllInterceptor::win_FileTimeToSystemTime(const void* lpFileTime, void* lpSystemTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpFileTime && lpSystemTime) {
        std::memset(lpSystemTime, 0, 16);
        uint16_t* st = static_cast<uint16_t*>(lpSystemTime);
        st[0] = 2026; // Year
        st[1] = 11;   // Month
        st[2] = 0;    // DayOfWeek
        st[3] = 16;   // Day
        st[4] = 12;   // Hour
        st[5] = 0;    // Minute
        st[6] = 0;    // Second
        st[7] = 0;    // Milliseconds
    }
    return 1;
}

uint32_t NativeDllInterceptor::win_GetFullPathNameW(const wchar_t* lpFileName, uint32_t nBufferLength, wchar_t* lpBuffer, wchar_t** lpFilePart) {
    s_kernel32Calls.fetch_add(1);
    if (lpFileName == nullptr) return 0;
    const std::string fn = wcharToUtf8(lpFileName);
    const uint32_t needed = static_cast<uint32_t>(guestWStringLength(fn));
    if (lpBuffer == nullptr || nBufferLength <= needed) {
        return needed + 1;
    }
    writeGuestWString(lpBuffer, nBufferLength, fn);
    if (lpFilePart != nullptr) {
        const auto* wbuf = reinterpret_cast<const uint16_t*>(lpBuffer);
        int lastSlash = -1;
        for (uint32_t i = 0; i < needed; ++i) {
            if (wbuf[i] == '/' || wbuf[i] == '\\') lastSlash = static_cast<int>(i);
        }
        // Aritmética en unidades WCHAR de 2 bytes (wchar_t de Bionic mide 4).
        auto* base16 = reinterpret_cast<char16_t*>(lpBuffer);
        lowmem::storePtr(lpFilePart, lastSlash >= 0 ? base16 + lastSlash + 1 : base16);
    }
    return needed;
}

int32_t NativeDllInterceptor::win_GetFileInformationByHandle(void* hFile, void* lpFileInformation) {
    s_kernel32Calls.fetch_add(1);
    if (lpFileInformation != nullptr) {
        std::memset(lpFileInformation, 0, 52); // sizeof(BY_HANDLE_FILE_INFORMATION)
        const int fd = fdFromHandle(hFile);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0) {
                uint32_t* info = static_cast<uint32_t*>(lpFileInformation);
                info[0] = S_ISDIR(st.st_mode) ? 0x10 /* FILE_ATTRIBUTE_DIRECTORY */ : 0x80 /* FILE_ATTRIBUTE_NORMAL */;
                // Mismas fechas que GetFileAttributesExW (segundos enteros): .NET compara ambas
                // para saber si el archivo cambió entre abrirlo y cargarlo.
                auto ft = [&](time_t t, int idx) {
                    const uint64_t v = (static_cast<uint64_t>(t) + 11644473600ULL) * 10000000ULL;
                    info[idx] = static_cast<uint32_t>(v);
                    info[idx + 1] = static_cast<uint32_t>(v >> 32);
                };
                ft(st.st_ctime, 1);   // ftCreationTime
                ft(st.st_atime, 3);   // ftLastAccessTime
                ft(st.st_mtime, 5);   // ftLastWriteTime
                info[7] = static_cast<uint32_t>(st.st_dev) | 1u;  // dwVolumeSerialNumber (distinto de 0)
                const uint64_t sz = static_cast<uint64_t>(st.st_size);
                info[8] = static_cast<uint32_t>(sz >> 32);
                info[9] = static_cast<uint32_t>(sz & 0xFFFFFFFF);
                info[10] = static_cast<uint32_t>(st.st_nlink > 0 ? st.st_nlink : 1);  // nNumberOfLinks
                const uint64_t ino = static_cast<uint64_t>(st.st_ino);
                info[11] = static_cast<uint32_t>(ino >> 32);  // nFileIndexHigh
                info[12] = static_cast<uint32_t>(ino);        // nFileIndexLow
            }
        }
    }
    return 1;
}

int32_t NativeDllInterceptor::win_GetFileInformationByHandleEx(void* /*hFile*/, int32_t, void* lpFileInformation, uint32_t dwBufferSize) {
    s_kernel32Calls.fetch_add(1);
    if (lpFileInformation && dwBufferSize > 0) {
        std::memset(lpFileInformation, 0, dwBufferSize);
    }
    return 1;
}

int32_t NativeDllInterceptor::win_GetProcessTimes(void* /*hProcess*/, void* lpCreationTime, void* lpExitTime, void* lpKernelTime, void* lpUserTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpCreationTime) std::memset(lpCreationTime, 0, 8);
    if (lpExitTime) std::memset(lpExitTime, 0, 8);
    if (lpKernelTime) std::memset(lpKernelTime, 0, 8);
    if (lpUserTime) std::memset(lpUserTime, 0, 8);
    return 1;
}

int32_t NativeDllInterceptor::win_GetSystemTimes(void* lpIdleTime, void* lpKernelTime, void* lpUserTime) {
    s_kernel32Calls.fetch_add(1);
    if (lpIdleTime) std::memset(lpIdleTime, 0, 8);
    if (lpKernelTime) std::memset(lpKernelTime, 0, 8);
    if (lpUserTime) std::memset(lpUserTime, 0, 8);
    return 1;
}

namespace {
// Texto de FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM) para los códigos Win32 habituales.
std::string systemMessageText(uint32_t code) {
    switch (code) {
        case 0: return "The operation completed successfully.";
        case 1: return "Incorrect function.";
        case 2: return "The system cannot find the file specified.";
        case 3: return "The system cannot find the path specified.";
        case 5: return "Access is denied.";
        case 6: return "The handle is invalid.";
        case 8: return "Not enough memory resources are available to process this command.";
        case 14: return "Not enough memory resources are available to complete this operation.";
        case 32: return "The process cannot access the file because it is being used by another process.";
        case 87: return "The parameter is incorrect.";
        case 122: return "The data area passed to a system call is too small.";
        case 126: return "The specified module could not be found.";
        case 127: return "The specified procedure could not be found.";
        case 183: return "Cannot create a file when that file already exists.";
        case 487: return "Attempt to access invalid address.";
        case 1167: return "The device is not connected.";
        default: {
            char buf[48];
            std::snprintf(buf, sizeof(buf), "Unknown error (0x%x).", code);
            return buf;
        }
    }
}

// FormatMessage A/W: FROM_SYSTEM (tabla), FROM_STRING (copia literal), ALLOCATE_BUFFER (LocalAlloc).
// Mono antiguo (Cuphead) aserta en mono-dl.c:125 (w32_dlerror) si FormatMessageW devuelve 0.
constexpr uint32_t kFmAllocate = 0x100, kFmFromString = 0x400, kFmFromSystem = 0x1000;

std::string formatMessageText(uint32_t flags, const void* source, uint32_t id, bool wide) {
    if ((flags & kFmFromString) != 0 && source != nullptr) {
        return wide ? utf16ToUtf8(static_cast<const char16_t*>(source)) : std::string(static_cast<const char*>(source));
    }
    if ((flags & kFmFromSystem) != 0) return systemMessageText(id) + "\r\n";
    return {};
}
}  // namespace

uint32_t NativeDllInterceptor::win_FormatMessageW(
    uint32_t dwFlags, const void* lpSource, uint32_t dwMessageId,
    uint32_t /*dwLanguageId*/, wchar_t* lpBuffer, uint32_t nSize, void* /*arguments*/) {
    s_kernel32Calls.fetch_add(1);
    if (lpBuffer == nullptr) {
        win_SetLastError(87);
        return 0;
    }
    const std::string text = formatMessageText(dwFlags, lpSource, dwMessageId, true);
    if (text.empty()) {
        win_SetLastError(317);  // ERROR_MR_MID_NOT_FOUND
        return 0;
    }
    std::u16string w;
    for (char c : text) w.push_back(static_cast<char16_t>(static_cast<unsigned char>(c)));
    char16_t* out = nullptr;
    if ((dwFlags & kFmAllocate) != 0) {
        const size_t chars = std::max<size_t>(w.size() + 1, nSize);
        out = static_cast<char16_t*>(win_LocalAlloc(0x40, chars * sizeof(char16_t)));
        if (out == nullptr) {
            win_SetLastError(8);
            return 0;
        }
        *reinterpret_cast<void**>(lpBuffer) = out;
    } else {
        if (nSize <= w.size()) {
            win_SetLastError(122);
            return 0;
        }
        out = reinterpret_cast<char16_t*>(lpBuffer);
    }
    std::memcpy(out, w.c_str(), (w.size() + 1) * sizeof(char16_t));
    win_SetLastError(0);
    return static_cast<uint32_t>(w.size());
}

uint32_t win_FormatMessageA(uint32_t dwFlags, const void* lpSource, uint32_t dwMessageId, uint32_t, char* lpBuffer,
                            uint32_t nSize, void*) {
    if (lpBuffer == nullptr) {
        NativeDllInterceptor::win_SetLastError(87);
        return 0;
    }
    const std::string text = formatMessageText(dwFlags, lpSource, dwMessageId, false);
    if (text.empty()) {
        NativeDllInterceptor::win_SetLastError(317);
        return 0;
    }
    char* out = nullptr;
    if ((dwFlags & kFmAllocate) != 0) {
        out = static_cast<char*>(NativeDllInterceptor::win_LocalAlloc(0x40, std::max<size_t>(text.size() + 1, nSize)));
        if (out == nullptr) return 0;
        *reinterpret_cast<void**>(lpBuffer) = out;
    } else {
        if (nSize <= text.size()) {
            NativeDllInterceptor::win_SetLastError(122);
            return 0;
        }
        out = lpBuffer;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return static_cast<uint32_t>(text.size());
}

int32_t NativeDllInterceptor::win_FlushFileBuffers(void* hFile) {
    s_kernel32Calls.fetch_add(1);
    // Las partidas guardadas llegan al disco antes de seguir (un cierre o la batería no las corta).
    const int fd = fdFromHandle(hFile);
    if (fd > STDERR_FILENO) fsync(fd);
    return 1;
}

int32_t NativeDllInterceptor::win_SetFilePointerEx(void* hFile, int64_t liDistanceToMove, int64_t* lpNewFilePointer, uint32_t dwMoveMethod) {
    s_kernel32Calls.fetch_add(1);
    const int fd = fdFromHandle(hFile);
    if (fd < 0) return 0;
    int whence = SEEK_SET;
    if (dwMoveMethod == 1) whence = SEEK_CUR;
    else if (dwMoveMethod == 2) whence = SEEK_END;
    const off_t res = lseek(fd, static_cast<off_t>(liDistanceToMove), whence);
    if (res == static_cast<off_t>(-1)) return 0;
    if (lpNewFilePointer) *lpNewFilePointer = static_cast<int64_t>(res);
    return 1;
}

namespace {
struct WinFindSearchState {
    DIR* dir = nullptr;
    std::string dirPath;
    std::string pattern;
};
std::mutex s_findMutex;
uintptr_t s_findHandleCounter = 0x8000;
std::unordered_map<uintptr_t, WinFindSearchState> s_findSearches;

struct WIN32_FIND_DATAW_STRUCT {
    uint32_t dwFileAttributes;
    uint32_t ftCreationTime_low, ftCreationTime_high;
    uint32_t ftLastAccessTime_low, ftLastAccessTime_high;
    uint32_t ftLastWriteTime_low, ftLastWriteTime_high;
    uint32_t nFileSizeHigh;
    uint32_t nFileSizeLow;
    uint32_t dwReserved0, dwReserved1;
    char16_t cFileName[260];
    char16_t cAlternateFileName[14];
};
} // namespace

void* NativeDllInterceptor::win_FindFirstFileExW(const void* lpFileName, int32_t, void* lpFindFileData, int32_t, void*, uint32_t) {
    s_kernel32Calls.fetch_add(1);
    if (!lpFileName || !lpFindFileData) return reinterpret_cast<void*>(~static_cast<uintptr_t>(0));
    std::string pathUtf8 = utf16ToUtf8(reinterpret_cast<const char16_t*>(lpFileName));
    std::string resolved = s_activeImage ? BionicPeLoader::resolveVirtualPath(*s_activeImage, pathUtf8) : pathUtf8;

    std::string dirPath = resolved;
    std::string pattern = "*";
    size_t slash = resolved.find_last_of('/');
    if (slash != std::string::npos) {
        dirPath = resolved.substr(0, slash);
        pattern = resolved.substr(slash + 1);
    }
    if (dirPath.empty()) dirPath = ".";

    DIR* d = opendir(dirPath.c_str());
    LOGI("win_FindFirstFileExW: raw='%s' dir='%s' pattern='%s' -> opendir=%s",
         pathUtf8.c_str(), dirPath.c_str(), pattern.c_str(), d ? "OK" : "FAILED");
    if (!d) return reinterpret_cast<void*>(~static_cast<uintptr_t>(0));

    uintptr_t handle = 0;
    {
        std::lock_guard<std::mutex> lock(s_findMutex);
        handle = ++s_findHandleCounter;
        s_findSearches[handle] = WinFindSearchState{d, dirPath, pattern};
    }

    if (win_FindNextFileW(reinterpret_cast<void*>(handle), lpFindFileData)) {
        return reinterpret_cast<void*>(handle);
    }

    win_FindClose(reinterpret_cast<void*>(handle));
    return reinterpret_cast<void*>(~static_cast<uintptr_t>(0));
}

int32_t NativeDllInterceptor::win_FindNextFileW(void* hFindFile, void* lpFindFileData) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hFindFile);
    DIR* d = nullptr;
    std::string dirPath;
    std::string pattern;
    {
        std::lock_guard<std::mutex> lock(s_findMutex);
        auto it = s_findSearches.find(handle);
        if (it == s_findSearches.end() || it->second.dir == nullptr) { win_SetLastError(6); return 0; }
        d = it->second.dir;
        dirPath = it->second.dirPath;
        pattern = it->second.pattern;
    }

    struct dirent* entry = nullptr;
    while ((entry = readdir(d)) != nullptr) {
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) continue;
        if (!pattern.empty() && pattern != "*" && pattern != "*.*") {
            if (fnmatch(pattern.c_str(), entry->d_name, FNM_CASEFOLD) != 0) {
                continue;
            }
        }
        break;
    }
    if (!entry) {
        win_SetLastError(18);  // ERROR_NO_MORE_FILES: fin normal (.NET distingue esto de un error)
        return 0;
    }

    auto* fd = static_cast<WIN32_FIND_DATAW_STRUCT*>(lpFindFileData);
    std::memset(fd, 0, sizeof(*fd));

    std::string fullItem = dirPath + "/" + entry->d_name;
    struct stat st{};
    if (stat(fullItem.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) fd->dwFileAttributes = 0x00000010; // FILE_ATTRIBUTE_DIRECTORY
        else fd->dwFileAttributes = 0x00000080; // FILE_ATTRIBUTE_NORMAL
        fd->nFileSizeLow = static_cast<uint32_t>(st.st_size & 0xFFFFFFFF);
        fd->nFileSizeHigh = static_cast<uint32_t>((st.st_size >> 32) & 0xFFFFFFFF);
    } else {
        fd->dwFileAttributes = 0x00000080;
    }

    size_t i = 0;
    for (; i < 259 && entry->d_name[i] != '\0'; ++i) {
        fd->cFileName[i] = static_cast<char16_t>(static_cast<uint8_t>(entry->d_name[i]));
    }
    fd->cFileName[i] = 0;
    return 1;
}

int32_t NativeDllInterceptor::win_FindClose(void* hFindFile) {
    s_kernel32Calls.fetch_add(1);
    const uintptr_t handle = reinterpret_cast<uintptr_t>(hFindFile);
    std::lock_guard<std::mutex> lock(s_findMutex);
    auto it = s_findSearches.find(handle);
    if (it != s_findSearches.end()) {
        if (it->second.dir) closedir(it->second.dir);
        s_findSearches.erase(it);
        return 1;
    }
    return 0;
}

void* NativeDllInterceptor::win_LoadLibraryExW(const void* lpLibFileName, void* hFile, uint32_t dwFlags) {
    s_kernel32Calls.fetch_add(1);
    (void)hFile;
    if (lpLibFileName == nullptr) return nullptr;
    const std::string name = wcharToUtf8(reinterpret_cast<const wchar_t*>(lpLibFileName));
    void* hMod = resolveModuleHandle(name);
    LOGI("win_LoadLibraryExW('%s', flags=0x%x) -> %p", name.c_str(), dwFlags, hMod);
    return hMod;
}

int32_t NativeDllInterceptor::win_FreeLibrary(void*) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

// -----------------------------------------------------------------------------
// Strings y Locale
// -----------------------------------------------------------------------------
// CP_UTF8 (65001, también ACP/OEMCP de este entorno) se convierte de verdad; el resto de
// páginas de código se trata como Latin-1. Con buffer insuficiente: 0 + ERROR_INSUFFICIENT_BUFFER.
int32_t NativeDllInterceptor::win_MultiByteToWideChar(uint32_t codePage, uint32_t, const char* lpMultiByteStr, int32_t cbMultiByte, void* lpWideCharStr, int32_t cchWideChar) {
    s_kernel32Calls.fetch_add(1);
    if (!lpMultiByteStr || cbMultiByte == 0 || cchWideChar < 0) {
        win_SetLastError(87);
        return 0;
    }
    const auto* in = reinterpret_cast<const uint8_t*>(lpMultiByteStr);
    const size_t inLen = cbMultiByte < 0 ? std::strlen(lpMultiByteStr) + 1 : static_cast<size_t>(cbMultiByte);
    const bool utf8 = codePage == 65001 || codePage == 0 || codePage == 1 || codePage == 3;
    std::u16string out;
    out.reserve(inLen);
    for (size_t i = 0; i < inLen;) {
        uint32_t cp = in[i];
        size_t n = 1;
        if (utf8 && cp >= 0x80) {
            if ((cp & 0xE0) == 0xC0) { n = 2; cp &= 0x1F; }
            else if ((cp & 0xF0) == 0xE0) { n = 3; cp &= 0x0F; }
            else if ((cp & 0xF8) == 0xF0) { n = 4; cp &= 0x07; }
            else { n = 0; }
            bool ok = n != 0 && i + n <= inLen;
            for (size_t k = 1; ok && k < n; ++k) {
                if ((in[i + k] & 0xC0) != 0x80) ok = false;
                else cp = (cp << 6) | (in[i + k] & 0x3F);
            }
            if (!ok) { cp = 0xFFFD; n = 1; }
        }
        i += n;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
    }
    if (cchWideChar == 0 || !lpWideCharStr) return static_cast<int32_t>(out.size());
    if (out.size() > static_cast<size_t>(cchWideChar)) {
        win_SetLastError(122);
        return 0;
    }
    std::memcpy(lpWideCharStr, out.data(), out.size() * sizeof(char16_t));
    return static_cast<int32_t>(out.size());
}

int32_t NativeDllInterceptor::win_WideCharToMultiByte(uint32_t codePage, uint32_t, const void* lpWideCharStr, int32_t cchWideChar, char* lpMultiByteStr, int32_t cbMultiByte, const char*, int32_t*) {
    s_kernel32Calls.fetch_add(1);
    if (!lpWideCharStr || cchWideChar == 0 || cbMultiByte < 0) {
        win_SetLastError(87);
        return 0;
    }
    auto* in = static_cast<const char16_t*>(lpWideCharStr);
    {   // Diagnóstico: setprop debug.hyperplay.w2mlog N -> registra los N textos siguientes (ASCII) con su llamador.
        // La propiedad se relee cada 4096 llamadas: se puede activar en mitad de la partida (setprop ... 50, luego '').
        static std::atomic<int> s_left{0};
        static std::atomic<uint32_t> s_calls{0};
        static std::atomic<int> s_lastProp{0};
        if ((s_calls.fetch_add(1, std::memory_order_relaxed) & 4095) == 0) {
            char v[PROP_VALUE_MAX] = {};
            const int p = __system_property_get("debug.hyperplay.w2mlog", v) > 0 ? std::atoi(v) : 0;
            if (p != s_lastProp.exchange(p)) s_left.store(p);
        }
        if (s_left.load(std::memory_order_relaxed) > 0 && s_left.fetch_sub(1) > 0) {
            std::string a;
            for (size_t k = 0; k < 200 && in[k] != 0 && (cchWideChar < 0 || k < static_cast<size_t>(cchWideChar)); ++k)
                a.push_back(in[k] < 0x80 ? static_cast<char>(in[k]) : '?');
            LOGI("[W2M] desde 0x%llx: %s", static_cast<unsigned long long>(t_thunkRet), a.c_str());
        }
    }
    size_t inLen = 0;
    if (cchWideChar < 0) {
        while (in[inLen] != 0) ++inLen;
        ++inLen;
    } else {
        inLen = static_cast<size_t>(cchWideChar);
    }
    const bool utf8 = codePage == 65001 || codePage == 0 || codePage == 1 || codePage == 3;
    std::string out;
    out.reserve(inLen);
    for (size_t i = 0; i < inLen; ++i) {
        uint32_t cp = in[i];
        if (!utf8) {
            out.push_back(cp < 0x100 ? static_cast<char>(cp) : '?');
            continue;
        }
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < inLen && in[i + 1] >= 0xDC00 && in[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (in[i + 1] - 0xDC00);
            ++i;
        } else if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;
        }
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    if (cbMultiByte == 0 || !lpMultiByteStr) return static_cast<int32_t>(out.size());
    if (out.size() > static_cast<size_t>(cbMultiByte)) {
        win_SetLastError(122);
        return 0;
    }
    std::memcpy(lpMultiByteStr, out.data(), out.size());
    return static_cast<int32_t>(out.size());
}

int32_t NativeDllInterceptor::win_IsValidCodePage(uint32_t) {
    s_kernel32Calls.fetch_add(1);
    return 1;
}

uint32_t NativeDllInterceptor::win_GetACP() {
    s_kernel32Calls.fetch_add(1);
    return 65001;
}

uint32_t NativeDllInterceptor::win_GetOEMCP() {
    s_kernel32Calls.fetch_add(1);
    return 65001;
}

int32_t NativeDllInterceptor::win_GetCPInfo(uint32_t codePage, void* lpCPInfo) {
    s_kernel32Calls.fetch_add(1);
    if (lpCPInfo == nullptr) return 0;
    // CPINFO {UINT MaxCharSize; BYTE DefaultChar[2]; BYTE LeadByte[12]} = 20 bytes. La ANSI del
    // sistema (CP_ACP/CP_THREAD_ACP/CP_OEMCP) es UTF-8 (GetACP = 65001): hasta 4 bytes por carácter.
    // .NET usa MaxCharSize para dimensionar los búferes ANSI; con 0 reservaba mal y fallaba.
    auto* info = static_cast<uint8_t*>(lpCPInfo);
    std::memset(info, 0, 20);
    const bool utf8 = codePage == 0 || codePage == 1 || codePage == 3 || codePage == 65001;
    const uint32_t maxCharSize = utf8 ? 4u : (codePage == 65000 ? 5u : 1u);
    std::memcpy(info, &maxCharSize, 4);
    info[4] = '?';
    return 1;
}

// Clasificación CT_CTYPE1 de un carácter UTF-16 (C1_UPPER 0x1, LOWER 0x2, DIGIT 0x4, SPACE 0x8,
// PUNCT 0x10, CNTRL 0x20, BLANK 0x40, XDIGIT 0x80, ALPHA 0x100, DEFINED 0x200).
static uint16_t ctype1(char16_t c) {
    uint16_t t = 0x200;
    const auto w = static_cast<wint_t>(c);
    if (c < 0x80) {
        if (c >= 'A' && c <= 'Z') t |= 0x101;
        if (c >= 'a' && c <= 'z') t |= 0x102;
        if (c >= '0' && c <= '9') t |= 0x84;
        if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) t |= 0x80;
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) t |= 0x08;
        if (c == ' ' || c == '\t') t |= 0x40;
        if (c < 0x20 || c == 0x7F) t |= 0x20;
        if (c > 0x20 && c < 0x7F && !std::isalnum(static_cast<int>(c))) t |= 0x10;
        return t;
    }
    if (iswupper(w)) t |= 0x101;
    if (iswlower(w)) t |= 0x102;
    if (iswalpha(w)) t |= 0x100;
    if (iswdigit(w)) t |= 0x04;
    if (iswspace(w)) t |= 0x08;
    if (iswpunct(w)) t |= 0x10;
    if (iswcntrl(w)) t |= 0x20;
    if (c == 0xA0) t |= 0x48;  // NBSP
    return t;
}

// Antes devolvía todo a 0: el CRT de Microsoft (ucrtbase) construye con esto su tabla de ctype y
// isspace(' ') era falso (Mono no separaba "Tipo, Ensamblado": Type.GetType fallaba).
int32_t NativeDllInterceptor::win_GetStringTypeW(uint32_t dwInfoType, const void* lpSrcStr, int32_t cchSrc, uint16_t* lpCharType) {
    s_kernel32Calls.fetch_add(1);
    if (lpCharType == nullptr || lpSrcStr == nullptr) { win_SetLastError(87); return 0; }
    const auto* src = static_cast<const char16_t*>(lpSrcStr);
    if (cchSrc < 0) { cchSrc = 0; while (src[cchSrc] != u'\0') ++cchSrc; ++cchSrc; }
    for (int32_t i = 0; i < cchSrc; ++i) {
        if (dwInfoType == 1) {          // CT_CTYPE1
            lpCharType[i] = ctype1(src[i]);
        } else if (dwInfoType == 4) {   // CT_CTYPE3: C3_ALPHA 0x8000 para letras
            lpCharType[i] = iswalpha(static_cast<wint_t>(src[i])) ? 0x8000 : 0;
        } else {                        // CT_CTYPE2 (dirección): C2_LEFTTORIGHT
            lpCharType[i] = 1;
        }
    }
    return 1;
}

int32_t NativeDllInterceptor::win_LCMapStringW(uint32_t, uint32_t dwMapFlags, const void* lpSrcStr, int32_t cchSrc, void* lpDestStr, int32_t cchDest) {
    s_kernel32Calls.fetch_add(1);
    return lcMapString(dwMapFlags, static_cast<const char16_t*>(lpSrcStr), cchSrc, lpDestStr, cchDest);
}

namespace {

// =============================================================================
// COM real para DXGI / D3D11 sobre DirectVulkanBridge.
//
// Cada objeto COM vive en memoria host y su primer campo es un puntero a una
// vtable de funciones host. El JIT invitado hace `call [vtable+slot]`, el
// dispatcher reconoce la dirección (registerTrampolineAddress) y ejecuta la
// función C++ con la ABI Win64. Nunca se devuelve S_OK con un puntero de salida
// nulo: los métodos desconocidos devuelven E_NOTIMPL / E_NOINTERFACE.
// =============================================================================

constexpr uint32_t kHrOk = 0x00000000u;
constexpr uint32_t kHrNotImpl = 0x80004001u;
constexpr uint32_t kHrNoInterface = 0x80004002u;
constexpr uint32_t kHrPointer = 0x80004003u;
constexpr uint32_t kHrOutOfMemory = 0x8007000Eu;
constexpr uint32_t kDxgiErrorNotFound = 0x887A0002u;
constexpr uint32_t kDxgiErrorUnsupported = 0x887A0004u;

enum ComInterface : int {
    kComFactory = 0,
    kComAdapter = 1,
    kComOutput = 2,
    kComSwapChain = 3,
    kComDevice = 4,
    kComContext = 5,
    kComDxgiDevice = 6,
    kComTexture = 7,
    kComRtv = 8,
    kComDsv = 9,
    kComSrv = 10,
    kComBuffer = 11,
    kComVertexShader = 12,
    kComPixelShader = 13,
    kComSampler = 14,
    kComGeneric = 15,
    // WASAPI / MMDevAPI (audio de FMOD, Wwise, XAudio2 sobre WASAPI) -> AAudio
    kComMMEnumerator = 16,
    kComMMDevice = 17,
    kComMMCollection = 18,
    kComPropertyStore = 19,
    kComAudioClient = 20,
    kComAudioRender = 21,
    kComAudioClock = 22,
    kComAudioVolume = 23,
    kComAudioSession = 24,
    kComMMEndpoint = 25,
};

struct ComObject {
    void** vtbl;
    std::atomic<uint32_t> refs{1};
    uint32_t kind{0};
    bool heap{false};  // objeto creado por Create* (no singleton)
};

// Los 12 argumentos Win64 llegan completos desde dispatchThunk (CopySubresourceRegion usa 9,
// CreateGeometryShaderWithStreamOutput 10).
struct ComArgs {
    uint64_t a[12];
};

uint64_t comDispatch(int iface, int slot, const ComArgs& args);

template <int Iface, int Slot>
uint64_t comSlot(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5,
                 uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9, uint64_t a10, uint64_t a11) {
    const ComArgs args{{a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11}};
    return comDispatch(Iface, Slot, args);
}

template <int Iface, size_t... Is>
std::array<void*, sizeof...(Is)> makeComVtable(std::index_sequence<Is...>) {
    return {{reinterpret_cast<void*>(&comSlot<Iface, static_cast<int>(Is)>)...}};
}

template <int Iface, size_t N>
struct ComVtable {
    static inline std::array<void*, N> slots = makeComVtable<Iface>(std::make_index_sequence<N>{});
};

ComObject s_comFactory;
ComObject s_comAdapter;
ComObject s_comOutput;
ComObject s_comSwapChain;
ComObject s_comDevice;
ComObject s_comContext;
ComObject s_comDxgiDevice;
ComObject s_comTexture;
ComObject s_comRtv;
ComObject s_comDsv;
ComObject s_comSrv;
ComObject s_comBuffer;
ComObject s_comVertexShader;
ComObject s_comPixelShader;
ComObject s_comSampler;

std::once_flag s_comOnce;

inline uint64_t hr(uint32_t value) noexcept { return static_cast<uint64_t>(value); }

// --- Objetos D3D11 reales (uno por Create*) ---------------------------------------------------

// Buffer / Texture1D / Texture2D / Texture3D con almacenamiento propio (D3D11ResourceModel.h).
struct D3DResourceObj {
    ComObject com;
    d3d11::Resource res;
    std::mutex mtx;
    // Se incrementa en cada escritura (Map/Update/Copy/Clear/draw): invalida cachés derivadas
    // (mips BC descomprimidos del rasterizador).
    std::atomic<uint64_t> generation{1};
};

// RTV / DSV / SRV / UAV: referencian su recurso y guardan el descriptor de vista.
struct D3DViewObj {
    ComObject com;
    D3DResourceObj* resource = nullptr;
    uint32_t viewKind = 0;  // kComRtv / kComDsv / kComSrv (también UAV)
    uint8_t desc[48] = {};
    bool hasDesc = false;
};

// Estados, samplers, queries, shaders e input layouts: descriptor copiado para GetDesc.
enum DescKind : uint32_t { kDescState = 0, kDescQuery = 1, kDescOpaque = 2 };
struct D3DDescObj {
    ComObject com;
    uint32_t descKind = kDescOpaque;
    uint32_t descSize = 0;
    uint8_t desc[272] = {};
    // Shaders: bytecode DXBC completo (rasterizador). Input layouts: D3D11_INPUT_ELEMENT_DESC
    // aplanados (nombre de semántica copiado).
    std::vector<uint8_t> bytecode;
    std::unique_ptr<dxbc::Program> program;  // decodificado al primer Draw
    struct InputElement {
        std::string semantic;
        uint32_t semanticIndex = 0, format = 0, slot = 0, offset = 0, perInstance = 0, stepRate = 0;
    };
    std::vector<InputElement> inputElements;
};

std::mutex s_comRegistryMutex;
std::unordered_set<const void*> s_comResources;
std::unordered_set<const void*> s_comViews;
std::unordered_set<const void*> s_comDescObjects;

D3DResourceObj* asResource(uint64_t p) {
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    const void* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(p));
    return s_comResources.count(ptr) != 0 ? reinterpret_cast<D3DResourceObj*>(static_cast<uintptr_t>(p)) : nullptr;
}

D3DViewObj* asView(uint64_t p) {
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    const void* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(p));
    return s_comViews.count(ptr) != 0 ? reinterpret_cast<D3DViewObj*>(static_cast<uintptr_t>(p)) : nullptr;
}

D3DDescObj* asDescObject(uint64_t p) {
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    const void* ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(p));
    return s_comDescObjects.count(ptr) != 0 ? reinterpret_cast<D3DDescObj*>(static_cast<uintptr_t>(p)) : nullptr;
}

// Swapchain: backbuffer real que el juego limpia/copia y Present sube a Vulkan.
struct SwapChainState {
    std::mutex mtx;
    uint32_t width = 0, height = 0, format = 28;
    D3DResourceObj* backbuffer = nullptr;
    uint64_t presents = 0;
    std::vector<uint8_t> rgba;
};
SwapChainState s_swapChain;

// Tamaños de vtable (slots) registrados en ensureComReady.
constexpr size_t kVtResource = 16;
constexpr size_t kVtView = 12;
constexpr size_t kVtSampler = 12;
constexpr size_t kVtShader = 8;
constexpr size_t kVtGeneric = 12;

void initHeapCom(ComObject& com, void** vtbl, uint32_t kind) {
    com.vtbl = vtbl;
    com.refs.store(1, std::memory_order_relaxed);
    com.kind = kind;
    com.heap = true;
}

D3DResourceObj* createResource(const d3d11::ResourceDesc& desc, const uint8_t* initData) {
    auto* obj = new D3DResourceObj();
    if (!obj->res.init(desc)) {
        delete obj;
        return nullptr;
    }
    const bool isBuffer = desc.dim == d3d11::Dimension::Buffer;
    initHeapCom(obj->com,
                isBuffer ? ComVtable<kComBuffer, kVtResource>::slots.data() : ComVtable<kComTexture, kVtResource>::slots.data(),
                isBuffer ? kComBuffer : kComTexture);
    if (initData != nullptr) obj->res.initialize(initData);
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    s_comResources.insert(obj);
    return obj;
}

D3DViewObj* createView(uint32_t viewKind, D3DResourceObj* res, const void* desc, size_t descSize) {
    auto* obj = new D3DViewObj();
    void** vtbl = viewKind == kComRtv   ? ComVtable<kComRtv, kVtView>::slots.data()
                  : viewKind == kComDsv ? ComVtable<kComDsv, kVtView>::slots.data()
                                        : ComVtable<kComSrv, kVtView>::slots.data();
    initHeapCom(obj->com, vtbl, viewKind);
    obj->viewKind = viewKind;
    obj->resource = res;
    if (res != nullptr) res->com.refs.fetch_add(1, std::memory_order_relaxed);
    if (desc != nullptr && descSize > 0) {
        std::memcpy(obj->desc, desc, std::min(descSize, sizeof(obj->desc)));
        obj->hasDesc = true;
    }
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    s_comViews.insert(obj);
    return obj;
}

// Diagnóstico del rasterizador: los 600 primeros shaders del juego a
// <Juego>/_HyperPlaySaves/shaderdump/<tipo>_<n>.dxbc (corpus para el decodificador DXBC).
void dumpShaderBytecode(const char* kind, const std::vector<uint8_t>& bc) {
    static std::atomic<uint32_t> s_count{0};
    const uint32_t n = s_count.fetch_add(1, std::memory_order_relaxed);
    if (n >= 600 || s_activeImage == nullptr || s_activeImage->savesDir.empty()) return;
    const std::string dir = s_activeImage->savesDir + "/shaderdump";
    mkdir(dir.c_str(), 0775);
    char name[64];
    std::snprintf(name, sizeof(name), "/%s_%03u.dxbc", kind, n);
    if (FILE* f = std::fopen((dir + name).c_str(), "wb")) {
        std::fwrite(bc.data(), 1, bc.size(), f);
        std::fclose(f);
    }
}

D3DDescObj* createDescObject(int iface, uint32_t descKind, const void* desc, size_t descSize) {
    auto* obj = new D3DDescObj();
    void** vtbl = iface == kComSampler        ? ComVtable<kComSampler, kVtSampler>::slots.data()
                  : iface == kComVertexShader ? ComVtable<kComVertexShader, kVtShader>::slots.data()
                  : iface == kComPixelShader  ? ComVtable<kComPixelShader, kVtShader>::slots.data()
                                              : ComVtable<kComGeneric, kVtGeneric>::slots.data();
    initHeapCom(obj->com, vtbl, static_cast<uint32_t>(iface));
    obj->descKind = descKind;
    if (desc != nullptr && descSize > 0) {
        obj->descSize = static_cast<uint32_t>(std::min(descSize, sizeof(obj->desc)));
        std::memcpy(obj->desc, desc, obj->descSize);
    }
    std::lock_guard<std::mutex> lock(s_comRegistryMutex);
    s_comDescObjects.insert(obj);
    return obj;
}

// Release a 0: la memoria del recurso se devuelve; la "cáscara" COM se conserva (un puntero
// colgante del juego no provoca un use-after-free nativo).
void onFinalRelease(ComObject* self) {
    if (!self->heap) return;
    if (D3DResourceObj* r = asResource(reinterpret_cast<uint64_t>(self)); r != nullptr) {
        gpu::Renderer::shared().release(r);
        std::lock_guard<std::mutex> lock(r->mtx);
        r->res.releaseStorage();
        return;
    }
    if (D3DViewObj* v = asView(reinterpret_cast<uint64_t>(self)); v != nullptr && v->resource != nullptr) {
        D3DResourceObj* res = v->resource;
        v->resource = nullptr;
        if (res->com.refs.fetch_sub(1, std::memory_order_acq_rel) == 1) onFinalRelease(&res->com);
    }
}

// Crea (o recrea en ResizeBuffers) el backbuffer RGBA del swapchain.
void configureSwapChain(uint32_t width, uint32_t height, uint32_t format) {
    std::lock_guard<std::mutex> lock(s_swapChain.mtx);
    s_swapChain.width = width != 0 ? width : static_cast<uint32_t>(win32::screenWidth());
    s_swapChain.height = height != 0 ? height : static_cast<uint32_t>(win32::screenHeight());
    if (format != 0) s_swapChain.format = format;
    d3d11::ResourceDesc desc;
    desc.dim = d3d11::Dimension::Texture2D;
    desc.width = s_swapChain.width;
    desc.height = s_swapChain.height;
    desc.format = s_swapChain.format;
    desc.bindFlags = 0x20 | 0x8;  // RENDER_TARGET | SHADER_RESOURCE
    D3DResourceObj* bb = createResource(desc, nullptr);
    if (s_swapChain.backbuffer != nullptr) {
        D3DResourceObj* old = s_swapChain.backbuffer;
        if (old->com.refs.fetch_sub(1, std::memory_order_acq_rel) == 1) onFinalRelease(&old->com);
    }
    s_swapChain.backbuffer = bb;
    LOGI("DXGI swapchain: backbuffer %ux%u formato %u -> %p", s_swapChain.width, s_swapChain.height,
         s_swapChain.format, static_cast<void*>(bb));
}

std::atomic<uint64_t> s_contextSlotCalls[128];

// Contadores del rasterizador por software (se imprimen en Present).
struct RasterCounters {
    std::atomic<uint64_t> draws{0}, drawn{0}, noShader{0}, noTarget{0}, rejected{0}, pixels{0}, triangles{0};
    std::atomic<uint64_t> nanos{0};
};
RasterCounters s_raster;

bool gpuPathActive();
gpu::TextureBinding gpuResource(D3DResourceObj* r);

// Diagnóstico de la ruta GPU: una escritura de CPU sobre un recurso con contenido de la GPU
// provoca su resubida (pisa lo dibujado). Se registran las primeras de cada tipo.
void noteGpuClobber(const char* what, D3DResourceObj* r) {
    if (r == nullptr || !gpuPathActive() || !gpu::Renderer::shared().isGpuWritten(r)) return;
    static std::mutex m;
    static std::map<std::string, int> seen;
    std::lock_guard<std::mutex> lock(m);
    if (seen[what]++ < 3) {
        const auto& d = r->res.desc();
        LOGW("GPU D3D11: %s sobre un recurso escrito por la GPU (%p %ux%u fmt %u)", what, static_cast<void*>(r), d.width,
             d.height, d.format);
    }
}

// IDXGISwapChain::Present: backbuffer -> RGBA8 -> DirectVulkanBridge::presentPixels.
// Monitor de ritmo: cada 2 s, FPS medios, peor frame y tirones (frames > 33 ms y > 100 ms).
// Se sigue en vivo con `adb logcat | grep Ritmo` (este ROM descarta la etiqueta "HyperPlay.FPS").
// Desglose del frame (para saber si el límite es el juego en CPU o el Present/GPU):
// juego = desde que terminó el Present anterior hasta que el juego llama a Present;
// present = lo que tarda nuestro Present (grabar, enviar, adquirir/esperar al swapchain).
struct FrameSplit {
    double gameMax = 0, presentMax = 0, gameSum = 0, presentSum = 0;
    uint32_t gameOver = 0, n = 0;  // gameOver: frames con juego > 15 ms
};
FrameSplit s_split;

void logFramePacing() {
    using clock = std::chrono::steady_clock;
    static clock::time_point s_last{}, s_window{};
    static uint32_t s_frames = 0, s_over18 = 0, s_over33 = 0, s_over100 = 0;
    static double s_worst = 0;
    const auto now = clock::now();
    if (s_last.time_since_epoch().count() != 0) {
        const double ms = std::chrono::duration<double, std::milli>(now - s_last).count();
        s_worst = std::max(s_worst, ms);
        if (ms > 18.0) ++s_over18;
        if (ms > 33.4) ++s_over33;
        if (ms > 100.0) ++s_over100;
        if (ms > 40.0) {  // captura de tirones (solo con debug.hyperplay.stallprof 1)
            const uint32_t nowMs = static_cast<uint32_t>(monoMs());
            stallprof::requestWindow(nowMs - static_cast<uint32_t>(ms), nowMs);
        }
    } else {
        s_window = now;
    }
    s_last = now;
    ++s_frames;
    const double span = std::chrono::duration<double, std::milli>(now - s_window).count();
    if (span >= 2000.0) {
        const double n = s_split.n > 0 ? s_split.n : 1;
        LOGI("[Ritmo] %.1f fps | peor %.1f ms | >18ms %u | >33ms %u | >100ms %u | juego med %.1f max %.1f >15ms %u | present med %.1f max %.1f",
             s_frames * 1000.0 / span, s_worst, s_over18, s_over33, s_over100, s_split.gameSum / n, s_split.gameMax,
             s_split.gameOver, s_split.presentSum / n, s_split.presentMax);
        s_split = FrameSplit{};
        s_window = now;
        s_frames = s_over18 = s_over33 = s_over100 = 0;
        s_worst = 0;
    }
}

uint32_t presentSwapChain() {
    s_d3dCalls.fetch_add(1);
    using sclock = std::chrono::steady_clock;
    static sclock::time_point s_presentEnd{};
    const auto tEnter = sclock::now();
    if (s_presentEnd.time_since_epoch().count() != 0) {
        const double g = std::chrono::duration<double, std::milli>(tEnter - s_presentEnd).count();
        if (g < 1000.0) {  // una pausa (Quick Resume) no es trabajo del juego
            s_split.gameSum += g;
            s_split.gameMax = std::max(s_split.gameMax, g);
            if (g > 15.0) ++s_split.gameOver;
        }
    }
    // Limitador preciso (clock_nanosleep absoluto) + tope en caliente + ADPF: loader/FramePacing.
    ::hyperplay::pacing::paceFrame();
    logFramePacing();
    const auto tAfterPace = sclock::now();
    struct SplitEnd {
        sclock::time_point start;
        ~SplitEnd() {
            s_presentEnd = sclock::now();
            const double p = std::chrono::duration<double, std::milli>(s_presentEnd - start).count();
            s_split.presentSum += p;
            s_split.presentMax = std::max(s_split.presentMax, p);
            ++s_split.n;
        }
    } splitEnd{tAfterPace};
    std::lock_guard<std::mutex> lock(s_swapChain.mtx);
    D3DResourceObj* bb = s_swapChain.backbuffer;
    ++s_swapChain.presents;
    bool ok = false;
    uint32_t w = 0, h = 0;
    if (bb != nullptr && gpuPathActive()) {
        // Ruta GPU: el backbuffer ya está en la GPU; se dibuja en el swapchain con la rotación.
        ok = gpu::Renderer::shared().present(gpuResource(bb));
        w = bb->res.desc().width;
        h = bb->res.desc().height;
        if (s_swapChain.presents <= 3 || (s_swapChain.presents % 600) == 0) {
            const gpu::Renderer::Stats st = gpu::Renderer::shared().stats();
            LOGI("GPU D3D11: draws=%llu fallidos=%llu pipelines=%llu subidas=%llu (%.1f MB) presents=%llu",
                 static_cast<unsigned long long>(st.draws), static_cast<unsigned long long>(st.failedDraws),
                 static_cast<unsigned long long>(st.pipelines), static_cast<unsigned long long>(st.uploads),
                 static_cast<double>(st.uploadBytes) / 1048576.0, static_cast<unsigned long long>(st.presents));
        }
    } else if (bb != nullptr) {
        bool converted = false;
        {
            std::lock_guard<std::mutex> resLock(bb->mtx);
            converted = bb->res.toRgba8(s_swapChain.rgba, w, h);
        }
        if (converted) ok = directVulkanBridge().presentPixels(s_swapChain.rgba.data(), w, h);
    }
    if (s_swapChain.presents <= 5 || (s_swapChain.presents % 600) == 0) {
        const uint8_t* px = s_swapChain.rgba.empty() ? nullptr : s_swapChain.rgba.data();
        LOGI("DXGI Present #%llu: %ux%u -> Vulkan %s (pixel0 RGBA=%02x%02x%02x%02x, frames=%llu)",
             static_cast<unsigned long long>(s_swapChain.presents), w, h, ok ? "OK" : "sin superficie/fallo",
             px ? px[0] : 0, px ? px[1] : 0, px ? px[2] : 0, px ? px[3] : 0,
             static_cast<unsigned long long>(directVulkanBridge().totalFramesPresented()));
    }
    if (s_swapChain.presents == 120 || s_swapChain.presents == 600) {
        std::string hist;
        for (int i = 0; i < 128; ++i) {
            const uint64_t n = s_contextSlotCalls[i].load(std::memory_order_relaxed);
            if (n != 0) hist += " " + std::to_string(i) + ":" + std::to_string(n);
        }
        LOGI("D3D11 DeviceContext llamadas por slot (hasta Present #%llu):%s",
             static_cast<unsigned long long>(s_swapChain.presents), hist.c_str());
    }
    if (s_swapChain.presents <= 10 || s_swapChain.presents % 120 == 0) {
        LOGI("Rasterizador: draws=%llu dibujados=%llu sinShader=%llu sinRT=%llu rechazados=%llu tri=%llu px=%llu (%.1f ms acumulados)",
             static_cast<unsigned long long>(s_raster.draws.load()), static_cast<unsigned long long>(s_raster.drawn.load()),
             static_cast<unsigned long long>(s_raster.noShader.load()), static_cast<unsigned long long>(s_raster.noTarget.load()),
             static_cast<unsigned long long>(s_raster.rejected.load()), static_cast<unsigned long long>(s_raster.triangles.load()),
             static_cast<unsigned long long>(s_raster.pixels.load()), static_cast<double>(s_raster.nanos.load()) / 1e6);
    }
    return kHrOk;  // DXGI no falla el Present por una superficie aún no vinculada
}

// D3D11_FORMAT_SUPPORT para CheckFormatSupport.
uint32_t formatSupport(uint32_t format) {
    const d3d11::FormatInfo fi = d3d11::formatInfo(format);
    if (format == 0) return 0;
    if (fi.depth || format == 19 || format == 44) return 0x20 | 0x80 | 0x10000;  // TEX2D|CUBE|DEPTH_STENCIL
    if (fi.block) return 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x1000;  // TEX2D/3D/CUBE|LOAD|SAMPLE|MIP
    return 0x1 | 0x2 | 0x4 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x1000 | 0x2000 | 0x4000 | 0x8000 |
           0x20000 | 0x40000 | 0x80000;
}

uint32_t storePtr(uint64_t address, void* value) noexcept {
    if (address == 0) return kHrPointer;
    *reinterpret_cast<void**>(static_cast<uintptr_t>(address)) = value;
    return kHrOk;
}

uint32_t storeU32(uint64_t address, uint32_t value) noexcept {
    if (address == 0) return kHrPointer;
    *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(address)) = value;
    return kHrOk;
}

uint32_t readGuidData1(uint64_t address) noexcept {
    if (address == 0) return 0;
    const auto* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address));
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void zeroMemory(uint64_t address, size_t size) noexcept {
    if (address == 0 || size == 0) return;
    std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(address)), 0, size);
}

void fillAdapterDesc(uint64_t address, bool desc1) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, desc1 ? 312 : 304);
    win32::copyAsciiToWchar16(reinterpret_cast<char16_t*>(b), 128, "Qualcomm Adreno 830");
    *reinterpret_cast<uint32_t*>(b + 256) = 0x5143u;  // VendorId Qualcomm
    *reinterpret_cast<uint32_t*>(b + 260) = 0x0830u;  // DeviceId Adreno 830
    *reinterpret_cast<uint64_t*>(b + 272) = 8ull * 1024ull * 1024ull * 1024ull;
    *reinterpret_cast<uint64_t*>(b + 280) = 0;
    *reinterpret_cast<uint64_t*>(b + 288) = 8ull * 1024ull * 1024ull * 1024ull;
    *reinterpret_cast<uint64_t*>(b + 296) = 0;  // AdapterLuid
    if (desc1) *reinterpret_cast<uint32_t*>(b + 304) = 0;
}

void fillOutputDesc(uint64_t address) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, 96);
    win32::copyAsciiToWchar16(reinterpret_cast<char16_t*>(b), 32, "\\\\.\\DISPLAY1");
    *reinterpret_cast<int32_t*>(b + 64) = 0;
    *reinterpret_cast<int32_t*>(b + 68) = 0;
    *reinterpret_cast<int32_t*>(b + 72) = win32::screenWidth();
    *reinterpret_cast<int32_t*>(b + 76) = win32::screenHeight();
    *reinterpret_cast<uint32_t*>(b + 80) = 1;  // AttachedToDesktop
    *reinterpret_cast<uint32_t*>(b + 84) = 0;  // DXGI_MODE_ROTATION_IDENTITY
    *reinterpret_cast<uint64_t*>(b + 88) = win32::kPrimaryMonitor;
}

void fillModeDesc(uint64_t address, uint32_t width, uint32_t height, uint32_t hz,
                  uint32_t format) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, 28);
    *reinterpret_cast<uint32_t*>(b + 0) = width;
    *reinterpret_cast<uint32_t*>(b + 4) = height;
    *reinterpret_cast<uint32_t*>(b + 8) = hz;
    *reinterpret_cast<uint32_t*>(b + 12) = 1;
    *reinterpret_cast<uint32_t*>(b + 16) = format;
}

void fillSwapChainDesc(uint64_t address) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, 72);
    fillModeDesc(address, win32::screenWidth(), win32::screenHeight(), win32::refreshHz(), 28);
    *reinterpret_cast<uint32_t*>(b + 28) = 1;  // SampleDesc.Count
    *reinterpret_cast<uint32_t*>(b + 32) = 0;
    *reinterpret_cast<uint32_t*>(b + 36) = 0x20;  // DXGI_USAGE_RENDER_TARGET_OUTPUT
    *reinterpret_cast<uint32_t*>(b + 40) = 2;
    *reinterpret_cast<uint64_t*>(b + 48) = 0xCAFE0050;  // OutputWindow
    *reinterpret_cast<uint32_t*>(b + 56) = 1;           // Windowed
    *reinterpret_cast<uint32_t*>(b + 60) = 2;           // DXGI_SWAP_EFFECT_FLIP_DISCARD
}

void fillSwapChainDesc1(uint64_t address) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, 48);
    *reinterpret_cast<uint32_t*>(b + 0) = win32::screenWidth();
    *reinterpret_cast<uint32_t*>(b + 4) = win32::screenHeight();
    *reinterpret_cast<uint32_t*>(b + 8) = 28;  // DXGI_FORMAT_R8G8B8A8_UNORM
    *reinterpret_cast<uint32_t*>(b + 12) = 0;
    *reinterpret_cast<uint32_t*>(b + 16) = 1;
    *reinterpret_cast<uint32_t*>(b + 20) = 0;
    *reinterpret_cast<uint32_t*>(b + 24) = 0x20;
    *reinterpret_cast<uint32_t*>(b + 28) = 2;
    *reinterpret_cast<uint32_t*>(b + 32) = 0;  // DXGI_SCALING_STRETCH
    *reinterpret_cast<uint32_t*>(b + 36) = 2;
    *reinterpret_cast<uint32_t*>(b + 40) = 0;  // DXGI_ALPHA_MODE_UNSPECIFIED
    *reinterpret_cast<uint32_t*>(b + 44) = 0;
}

void fillTexture2DDesc(uint64_t address, uint32_t width, uint32_t height,
                       uint32_t format, uint32_t bindFlags) noexcept {
    if (address == 0) return;
    auto* b = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    std::memset(b, 0, 44);
    *reinterpret_cast<uint32_t*>(b + 0) = width;
    *reinterpret_cast<uint32_t*>(b + 4) = height;
    *reinterpret_cast<uint32_t*>(b + 8) = 1;   // MipLevels
    *reinterpret_cast<uint32_t*>(b + 12) = 1;  // ArraySize
    *reinterpret_cast<uint32_t*>(b + 16) = format;
    *reinterpret_cast<uint32_t*>(b + 20) = 1;  // SampleDesc.Count
    *reinterpret_cast<uint32_t*>(b + 32) = bindFlags;
}

uint32_t comQueryInterface(ComObject* self, uint32_t iidData1, uint64_t ppv) {
    if (ppv == 0) return kHrPointer;
    if (self == nullptr) return kHrPointer;

    ComObject* out = nullptr;
    const bool deviceLevelIid = iidData1 == 0xDB6F6DDBu || iidData1 == 0xC0BFA96Cu ||  // Device/Context
                                iidData1 == 0x54EC77FAu || iidData1 == 0x77DB970Fu ||  // IDXGIDevice(1)
                                iidData1 == 0x50C83A1Cu || iidData1 == 0x770AAE78u ||
                                iidData1 == 0x7B7166ECu || iidData1 == 0x29038F61u ||
                                iidData1 == 0x2411E7E1u;
    if (self->heap && !deviceLevelIid) {
        // Recursos, vistas, estados, queries y shaders reales: la propia interfaz
        // (ID3D11Texture2D/Resource/DeviceChild, IDXGIResource, ID3D11View, ...).
        out = self;
    } else if (iidData1 == 0x00000000u) {  // IID_IUnknown
        out = self;
    } else if (iidData1 == 0xAEC22FB8u || iidData1 == 0x1841E5C8u ||
               iidData1 == 0x3D3E0379u) {
        // IDXGIObject / ID3D11DeviceChild / IDXGIDeviceSubObject: los primeros
        // slots de todas nuestras vtables respetan el prefijo común.
        out = self;
    } else {
        switch (iidData1) {
            case 0x50C83A1Cu:  // IDXGIFactory2
            case 0x770AAE78u:  // IDXGIFactory1
            case 0x7B7166ECu:  // IDXGIFactory
                out = &s_comFactory;
                break;
            case 0x29038F61u:  // IDXGIAdapter1
            case 0x2411E7E1u:  // IDXGIAdapter
                out = &s_comAdapter;
                break;
            case 0xAE02EEDBu:  // IDXGIOutput
            case 0x00CDDEA8u:  // IDXGIOutput1
                out = &s_comOutput;
                break;
            case 0x790A45F7u:  // IDXGISwapChain1
            case 0x310D36A0u:  // IDXGISwapChain
                out = &s_comSwapChain;
                break;
            case 0x54EC77FAu:  // IDXGIDevice
            case 0x77DB970Fu:  // IDXGIDevice1
                out = &s_comDxgiDevice;
                break;
            case 0xDB6F6DDBu:  // ID3D11Device
                out = &s_comDevice;
                break;
            case 0xC0BFA96Cu:  // ID3D11DeviceContext
                out = &s_comContext;
                break;
            case 0x6F15AAF2u:  // ID3D11Texture2D
            case 0xCAFCB56Cu:  // IDXGISurface
            case 0x035F3AB4u:  // IDXGIResource
                out = &s_comTexture;
                break;
            case 0xDFDBA067u:  // ID3D11RenderTargetView
                out = &s_comRtv;
                break;
            case 0x9FDAC92Au:  // ID3D11DepthStencilView
                out = &s_comDsv;
                break;
            case 0xB0E06FE0u:  // ID3D11ShaderResourceView
                out = &s_comSrv;
                break;
            default:
                out = nullptr;
                break;
        }
    }

    if (out == nullptr) {
        storePtr(ppv, nullptr);
        return kHrNoInterface;
    }
    out->refs.fetch_add(1, std::memory_order_relaxed);
    storePtr(ppv, out);
    return kHrOk;
}

// =============================================================================
// Rasterizador por software: estado del ID3D11DeviceContext inmediato y Draw*
// =============================================================================

struct BoundVertexBuffer {
    D3DResourceObj* res = nullptr;
    uint32_t stride = 0, offset = 0;
};

struct ContextState {
    std::mutex mtx;
    D3DDescObj* vs = nullptr;
    D3DDescObj* ps = nullptr;
    D3DDescObj* layout = nullptr;
    D3DDescObj* blend = nullptr;
    D3DDescObj* rs = nullptr;
    BoundVertexBuffer vb[16];
    D3DResourceObj* ib = nullptr;
    uint32_t ibFormat = 57, ibOffset = 0, topology = 0;
    D3DResourceObj* vsCb[14] = {};
    D3DResourceObj* psCb[14] = {};
    D3DViewObj* vsSrv[32] = {};
    D3DViewObj* psSrv[32] = {};
    D3DDescObj* vsSamp[16] = {};
    D3DDescObj* psSamp[16] = {};
    D3DViewObj* rtv[8] = {};
    float blendFactor[4] = {1, 1, 1, 1};
    raster::Viewport viewport;
    bool hasViewport = false;
    int32_t scissor[4] = {0, 0, 0, 0};
};
ContextState s_ctx;

D3DDescObj* descArg(uint64_t p) { return p == 0 ? nullptr : asDescObject(p); }
D3DResourceObj* resArg(uint64_t p) { return p == 0 ? nullptr : asResource(p); }
D3DViewObj* viewArg(uint64_t p) { return p == 0 ? nullptr : asView(p); }

const dxbc::Program* programOf(D3DDescObj* sh) {
    if (sh == nullptr || sh->bytecode.empty()) return nullptr;
    if (!sh->program) {
        sh->program = std::make_unique<dxbc::Program>(dxbc::parse(sh->bytecode.data(), sh->bytecode.size()));
        if (!sh->program->ok()) LOGW("DXBC: shader %p no decodificable: %s", sh, sh->program->error.c_str());
    }
    return sh->program->ok() ? sh->program.get() : nullptr;
}

uint32_t resolveTypeless(uint32_t f) {
    switch (f) {
        case 1: return 2;
        case 9: return 10;
        case 23: return 24;
        case 27: return 28;
        case 90: return 87;
        case 92: return 88;
        default: return f;
    }
}

// Mips BC descomprimidos, invalidados por la generación del recurso.
struct DecodedMip {
    uint64_t generation = 0;
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgba;
};
std::mutex s_decodedMutex;
std::unordered_map<uint64_t, DecodedMip> s_decoded;  // clave: puntero ^ (sub << 56)

class ContextTextures final : public dxbc::TextureAccess {
public:
    // Vistas y samplers resueltos una vez por draw (antes del raster en paralelo): el muestreo
    // desde varios hilos no toma locks ni reserva memoria.
    ContextTextures(D3DViewObj* const* srv, D3DDescObj* const* samp) : srv_(srv), samp_(samp) {
        for (uint32_t i = 0; i < 32; ++i) ok_[i] = srv_[i] != nullptr && view(i, views_[i]);
        for (uint32_t i = 0; i < 16; ++i) {
            if (samp_[i] != nullptr) samplers_[i] = d3d11::parseSamplerDesc(samp_[i]->desc);
        }
    }

    dxbc::Vec4 sample(uint32_t tex, uint32_t s, const float* c, float, bool, const int8_t* off) override {
        if (tex >= 32 || !ok_[tex]) return {};
        return d3d11::sample2D(views_[tex], samplers_[s & 15u], c[0], c[1], off);
    }
    dxbc::Vec4 load(uint32_t tex, const int32_t* c, const int8_t* off) override {
        if (tex >= 32 || !ok_[tex]) return {};
        d3d11::SamplerDesc clamp;
        clamp.address[0] = clamp.address[1] = 4;  // fuera de rango -> 0
        return d3d11::texelAt(views_[tex], clamp, c[0] + (off ? off[0] : 0), c[1] + (off ? off[1] : 0));
    }
    dxbc::Vec4 info(uint32_t tex, uint32_t) override {
        dxbc::Vec4 v;
        if (tex < 32 && ok_[tex]) {
            const d3d11::TexView& tv = views_[tex];
            v.f[0] = static_cast<float>(tv.width);
            v.f[1] = static_cast<float>(tv.height);
            v.f[2] = 1.0f;
            v.f[3] = 1.0f;
        }
        return v;
    }

private:
    D3DViewObj* const* srv_;
    D3DDescObj* const* samp_;
    d3d11::TexView views_[32];
    bool ok_[32] = {};
    d3d11::SamplerDesc samplers_[16];
    inline static std::atomic<uint64_t> unsupported_{0};

    bool view(uint32_t tex, d3d11::TexView& out) {
        if (tex >= 32 || srv_[tex] == nullptr || srv_[tex]->resource == nullptr) return false;
        D3DViewObj* v = srv_[tex];
        D3DResourceObj* r = v->resource;
        const d3d11::ResourceDesc& rd = r->res.desc();
        if (rd.dim == d3d11::Dimension::Buffer) return false;
        uint32_t fmt = rd.format, mip = 0, slice = 0;
        if (v->hasDesc) {
            uint32_t vf = 0, dim = 0;
            std::memcpy(&vf, v->desc, 4);
            std::memcpy(&dim, v->desc + 4, 4);
            if (vf != 0) fmt = vf;
            std::memcpy(&mip, v->desc + 8, 4);
            if (dim == 5 || dim == 10) std::memcpy(&slice, v->desc + 16, 4);  // 2D array / cube: primera capa
        }
        fmt = resolveTypeless(fmt);
        if (mip >= rd.mipLevels) mip = 0;
        const uint32_t sub = mip + slice * rd.mipLevels;
        const d3d11::SubresourceLayout* l = r->res.layout(sub);
        if (l == nullptr) return false;
        if (!d3d11::isBlockCompressed(fmt)) {
            out = {r->res.data(sub), l->width, l->height, l->rowPitch, fmt, false};
            return out.data != nullptr;
        }
        const uint64_t key = reinterpret_cast<uint64_t>(r) ^ (static_cast<uint64_t>(sub) << 56);
        const uint64_t gen = r->generation.load(std::memory_order_acquire);
        std::lock_guard<std::mutex> lock(s_decodedMutex);
        DecodedMip& d = s_decoded[key];
        if (d.generation != gen || d.rgba.empty()) {
            const uint8_t* src = r->res.data(sub);
            if (src == nullptr || !d3d11::decompressToRgba8(fmt, src, l->width, l->height, l->rowPitch, d.rgba)) {
                if (unsupported_.fetch_add(1) < 8) LOGW("Rasterizador: formato de textura %u no soportado", fmt);
                d.rgba.clear();
                return false;
            }
            d.generation = gen;
            d.width = l->width;
            d.height = l->height;
        }
        const bool srgb = fmt == 72 || fmt == 75 || fmt == 78 || fmt == 99;
        out = {d.rgba.data(), d.width, d.height, d.width * 4, 0xFFFFFFFFu, srgb};
        return true;
    }
};

bool sameSemantic(const std::string& a, const std::string& b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return (x | 0x20) == (y | 0x20); });
}

// --- Ruta GPU (D3D11 -> Vulkan) -------------------------------------------------------------
// Vista SRV/RTV -> TextureBinding del renderizador GPU.
gpu::TextureBinding gpuTexture(D3DViewObj* v, bool isRtv) {
    gpu::TextureBinding t;
    if (v == nullptr || v->resource == nullptr) return t;
    D3DResourceObj* r = v->resource;
    t.key = r;
    t.res = &r->res;
    t.generation = r->generation.load(std::memory_order_acquire);
    if (v->hasDesc) {
        uint32_t w[6] = {};
        std::memcpy(w, v->desc, sizeof(w));
        t.viewFormat = w[0];
        t.viewDim = w[1];
        if (isRtv) {
            t.mostDetailedMip = w[2];  // MipSlice
            if (w[1] == 3 || w[1] == 5) {  // 1D/2D array: FirstArraySlice, ArraySize
                t.firstSlice = w[3];
                t.arraySize = w[4];
            }
        } else {
            t.mostDetailedMip = w[2];
            t.mipLevels = w[3];
            if (w[1] == 3 || w[1] == 5) {  // 1DARRAY / 2DARRAY: First, Size
                t.firstSlice = w[4];
                t.arraySize = w[5];
            }
        }
    }
    return t;
}
gpu::TextureBinding gpuResource(D3DResourceObj* r) {
    gpu::TextureBinding t;
    if (r == nullptr) return t;
    t.key = r;
    t.res = &r->res;
    t.generation = r->generation.load(std::memory_order_acquire);
    return t;
}
gpu::BufferBinding gpuBuffer(D3DResourceObj* r) {
    gpu::BufferBinding b;
    if (r == nullptr) return b;
    b.key = r;
    b.data = r->res.data(0);
    b.size = r->res.desc().dim == d3d11::Dimension::Buffer ? r->res.desc().width : 0;
    b.generation = r->generation.load(std::memory_order_acquire);
    return b;
}
bool gpuPathActive() {
    static std::atomic<int> s_state{-1};  // -1 sin decidir, 0 no, 1 sí
    int s = s_state.load(std::memory_order_acquire);
    if (s >= 0) return s == 1;
    const bool on = gpu::Renderer::shared().available();
    // Mientras no haya dispositivo Vulkan (sin superficie) se reintenta en el siguiente draw.
    if (on) {
        s_state.store(1, std::memory_order_release);
        LOGI("Render D3D11: ruta GPU (Vulkan) activa");
    } else {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.gpu", v) > 0 && v[0] == '0') s_state.store(0);
    }
    return on;
}

// Graba el draw en la GPU con el estado actual del contexto (s_ctx.mtx tomado).
bool executeDrawGpu(const dxbc::Program* vs, const dxbc::Program* ps, uint32_t count, uint32_t start, int32_t baseVertex,
                    bool indexed, uint32_t instances, uint32_t startInstance) {
    gpu::DrawCall dc;
    dc.vs.program = vs;
    dc.vs.key = s_ctx.vs;
    dc.ps.program = ps;
    dc.ps.key = s_ctx.ps;
    for (int i = 0; i < 14; ++i) {
        dc.vs.cb[i] = gpuBuffer(s_ctx.vsCb[i]);
        dc.ps.cb[i] = gpuBuffer(s_ctx.psCb[i]);
    }
    for (int i = 0; i < 32; ++i) {
        dc.vs.srv[i] = gpuTexture(s_ctx.vsSrv[i], false);
        dc.ps.srv[i] = gpuTexture(s_ctx.psSrv[i], false);
    }
    for (int i = 0; i < 16; ++i) {
        dc.vs.sampler[i] = s_ctx.vsSamp[i] ? s_ctx.vsSamp[i]->desc : nullptr;
        dc.ps.sampler[i] = s_ctx.psSamp[i] ? s_ctx.psSamp[i]->desc : nullptr;
    }
    dc.layoutKey = s_ctx.layout;
    if (s_ctx.layout != nullptr) {
        for (const auto& e : s_ctx.layout->inputElements) {
            for (const auto& in : vs->inputSig) {
                if (in.semanticIndex == e.semanticIndex && sameSemantic(in.name, e.semantic)) {
                    dc.elements.push_back({in.reg, e.format, e.slot, e.offset, e.perInstance != 0});
                    break;
                }
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        dc.vb[i] = gpuBuffer(s_ctx.vb[i].res);
        dc.vb[i].stride = s_ctx.vb[i].stride;
        dc.vb[i].offset = s_ctx.vb[i].offset;
    }
    if (indexed) {
        dc.ib = gpuBuffer(s_ctx.ib);
        dc.ib.offset = s_ctx.ibOffset;
        dc.ibFormat = s_ctx.ibFormat;
    }
    dc.topology = s_ctx.topology;
    dc.count = count;
    dc.start = start;
    dc.baseVertex = baseVertex;
    dc.indexed = indexed;
    dc.instanceCount = instances;
    dc.startInstance = startInstance;
    dc.rt = gpuTexture(s_ctx.rtv[0], true);
    if (s_ctx.hasViewport) {
        dc.hasViewport = true;
        dc.viewport[0] = s_ctx.viewport.x;
        dc.viewport[1] = s_ctx.viewport.y;
        dc.viewport[2] = s_ctx.viewport.w;
        dc.viewport[3] = s_ctx.viewport.h;
        dc.viewport[4] = s_ctx.viewport.minZ;
        dc.viewport[5] = s_ctx.viewport.maxZ;
    }
    if (s_ctx.rs != nullptr) {
        uint32_t cull = 3, ccw = 0, scissorOn = 0;
        std::memcpy(&cull, s_ctx.rs->desc + 4, 4);
        std::memcpy(&ccw, s_ctx.rs->desc + 8, 4);
        std::memcpy(&scissorOn, s_ctx.rs->desc + 28, 4);
        dc.cullMode = cull;
        dc.frontCounterClockwise = ccw != 0;
        dc.scissorEnable = scissorOn != 0;
    }
    std::memcpy(dc.scissor, s_ctx.scissor, sizeof(dc.scissor));
    if (s_ctx.blend != nullptr) {
        const uint8_t* b = s_ctx.blend->desc + 8;
        uint32_t v[7];
        std::memcpy(v, b, sizeof(v));
        dc.blendEnable = v[0] != 0;
        for (int i = 0; i < 6; ++i) dc.blend[i] = v[1 + i];
        dc.writeMask = b[28];
    }
    std::memcpy(dc.blendFactor, s_ctx.blendFactor, sizeof(dc.blendFactor));
    return gpu::Renderer::shared().draw(dc);
}

void executeDraw(uint32_t count, uint32_t start, int32_t baseVertex, bool indexed, uint32_t instances = 1,
                 uint32_t startInstance = 0) {
    s_raster.draws.fetch_add(1, std::memory_order_relaxed);
    // Experimento (setprop debug.hyperplay.noraster 1): no rasterizar, para medir cuánto del
    // tiempo de carga/frame se va en el rasterizador por software.
    static const bool s_noRaster = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.noraster", v) > 0 && v[0] == '1';
    }();
    if (s_noRaster) return;
    std::lock_guard<std::mutex> lock(s_ctx.mtx);
    const dxbc::Program* vs = programOf(s_ctx.vs);
    const dxbc::Program* ps = programOf(s_ctx.ps);
    if (vs == nullptr || ps == nullptr) {
        s_raster.noShader.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    D3DViewObj* rtv = s_ctx.rtv[0];
    if (rtv == nullptr || rtv->resource == nullptr) {
        s_raster.noTarget.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (gpuPathActive()) {
        // Ruta GPU: el RT vive en la GPU; no se toca la copia de CPU ni su generación.
        if (executeDrawGpu(vs, ps, count, start, baseVertex, indexed, instances, startInstance)) {
            s_raster.drawn.fetch_add(1, std::memory_order_relaxed);
        } else {
            s_raster.rejected.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    raster::DrawState st;
    st.vs = vs;
    st.ps = ps;

    D3DResourceObj* rtRes = rtv->resource;
    const d3d11::ResourceDesc& rd = rtRes->res.desc();
    uint32_t fmt = rd.format, mip = 0, slice = 0;
    if (rtv->hasDesc) {
        uint32_t vf = 0, dim = 0;
        std::memcpy(&vf, rtv->desc, 4);
        std::memcpy(&dim, rtv->desc + 4, 4);
        if (vf != 0) fmt = vf;
        std::memcpy(&mip, rtv->desc + 8, 4);
        if (dim == 5) std::memcpy(&slice, rtv->desc + 12, 4);
    }
    const uint32_t sub = mip + slice * rd.mipLevels;
    const d3d11::SubresourceLayout* l = rtRes->res.layout(sub);
    if (l == nullptr) {
        s_raster.noTarget.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    st.rt = {rtRes->res.data(sub), l->width, l->height, l->rowPitch, resolveTypeless(fmt)};

    if (s_ctx.layout != nullptr) {
        for (const auto& e : s_ctx.layout->inputElements) {
            for (const auto& in : vs->inputSig) {
                if (in.semanticIndex == e.semanticIndex && sameSemantic(in.name, e.semantic)) {
                    raster::VertexElement el;
                    el.vsReg = in.reg;
                    el.format = e.format;
                    el.slot = e.slot;
                    el.offset = e.offset;
                    el.perInstance = e.perInstance != 0;
                    st.elements.push_back(el);
                    break;
                }
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        D3DResourceObj* b = s_ctx.vb[i].res;
        if (b == nullptr) continue;
        const d3d11::SubresourceLayout* bl = b->res.layout(0);
        st.vb[i] = {b->res.data(0), bl != nullptr ? bl->size : 0, s_ctx.vb[i].stride, s_ctx.vb[i].offset};
    }
    if (indexed && s_ctx.ib != nullptr) {
        const d3d11::SubresourceLayout* il = s_ctx.ib->res.layout(0);
        st.ib = s_ctx.ib->res.data(0);
        st.ibSize = il != nullptr ? il->size : 0;
        st.ibFormat = s_ctx.ibFormat;
        st.ibOffset = s_ctx.ibOffset;
    }
    st.topology = s_ctx.topology;
    for (int i = 0; i < 14; ++i) {
        if (s_ctx.vsCb[i] != nullptr) st.vsCb[i] = {s_ctx.vsCb[i]->res.data(0), s_ctx.vsCb[i]->res.desc().width};
        if (s_ctx.psCb[i] != nullptr) st.psCb[i] = {s_ctx.psCb[i]->res.data(0), s_ctx.psCb[i]->res.desc().width};
    }
    ContextTextures psTex(s_ctx.psSrv, s_ctx.psSamp), vsTex(s_ctx.vsSrv, s_ctx.vsSamp);
    st.psTex = &psTex;
    st.vsTex = &vsTex;
    st.viewport = s_ctx.hasViewport ? s_ctx.viewport
                                    : raster::Viewport{0, 0, static_cast<float>(l->width),
                                                       static_cast<float>(l->height), 0, 1};
    if (s_ctx.rs != nullptr) {
        // D3D11_RASTERIZER_DESC: FillMode, CullMode, FrontCounterClockwise, DepthBias, ...,
        // ScissorEnable en +28.
        uint32_t cull = 3, ccw = 0, scissorOn = 0;
        std::memcpy(&cull, s_ctx.rs->desc + 4, 4);
        std::memcpy(&ccw, s_ctx.rs->desc + 8, 4);
        std::memcpy(&scissorOn, s_ctx.rs->desc + 28, 4);
        st.cullMode = cull;
        st.frontCounterClockwise = ccw != 0;
        st.scissorEnable = scissorOn != 0;
    }
    std::memcpy(st.scissor, s_ctx.scissor, sizeof(st.scissor));
    if (s_ctx.blend != nullptr) {
        // D3D11_BLEND_DESC: AlphaToCoverage, IndependentBlend, RenderTarget[0] en +8 (32 B).
        const uint8_t* b = s_ctx.blend->desc + 8;
        uint32_t v[7];
        std::memcpy(v, b, sizeof(v));
        st.blend.enable = v[0] != 0;
        st.blend.src = v[1];
        st.blend.dst = v[2];
        st.blend.op = v[3];
        st.blend.srcA = v[4];
        st.blend.dstA = v[5];
        st.blend.opA = v[6];
        st.blend.writeMask = b[28];
    }
    std::memcpy(st.blendFactor, s_ctx.blendFactor, sizeof(st.blendFactor));

    static std::once_flag s_poolOnce;
    std::call_once(s_poolOnce, [] {
        // 5 hilos + el de render de Unity: deja los 2 núcleos Prime libres para el JIT principal.
        raster::SoftRasterizer::setWorkerCount(5);
    });
    raster::SoftRasterizer r;
    std::lock_guard<std::mutex> rtLock(rtRes->mtx);
    if (r.draw(st, count, start, baseVertex, indexed)) {
        s_raster.drawn.fetch_add(1, std::memory_order_relaxed);
        rtRes->generation.fetch_add(1, std::memory_order_acq_rel);
    } else {
        s_raster.rejected.fetch_add(1, std::memory_order_relaxed);
    }
    s_raster.pixels.fetch_add(r.stats.pixels, std::memory_order_relaxed);
    s_raster.triangles.fetch_add(r.stats.triangles, std::memory_order_relaxed);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    s_raster.nanos.fetch_add(ns, std::memory_order_relaxed);
    static std::atomic<uint32_t> s_drawLogs{0};
    // setprop debug.hyperplay.drawlog <N>: registra los draws N..N+39 (por defecto los 40 primeros).
    static const uint32_t s_drawLogStart = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.drawlog", v) > 0 ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10))
                                                                       : 0u;
    }();
    const uint32_t drawIndex = s_drawLogs.fetch_add(1, std::memory_order_relaxed);
    if (drawIndex >= s_drawLogStart && drawIndex < s_drawLogStart + 40) {
        // Con debug.hyperplay.dumprt=1 se vuelca además el RT tras el draw (crudo, cabecera w,h,pitch).
        char dumpProp[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.dumprt", dumpProp) > 0 && dumpProp[0] == '1' &&
            drawIndex < s_drawLogStart + 8) {
            const std::string path = (s_activeImage ? s_activeImage->gameDir : std::string("/sdcard")) +
                                     "/_HyperPlaySaves/rtdump_" + std::to_string(drawIndex) + ".raw";
            if (FILE* fo = std::fopen(path.c_str(), "wb")) {
                const uint32_t hdr[4] = {l->width, l->height, l->rowPitch, st.rt.format};
                std::fwrite(hdr, sizeof(hdr), 1, fo);
                std::fwrite(st.rt.data, 1, static_cast<size_t>(l->rowPitch) * l->height, fo);
                std::fclose(fo);
            }
        }
        LOGI("Draw#%u blend en=%d src=%u dst=%u op=%u srcA=%u dstA=%u opA=%u mask=0x%x", drawIndex,
             st.blend.enable ? 1 : 0, st.blend.src, st.blend.dst, st.blend.op, st.blend.srcA, st.blend.dstA,
             st.blend.opA, st.blend.writeMask);
        for (uint32_t t = 0; t < 4; ++t) {
            D3DViewObj* v = s_ctx.psSrv[t];
            if (v == nullptr || v->resource == nullptr) continue;
            const d3d11::ResourceDesc& rd = v->resource->res.desc();
            uint32_t vdesc[4] = {};
            if (v->hasDesc) std::memcpy(vdesc, v->desc, sizeof(vdesc));
            const d3d11::SubresourceLayout* l0 = v->resource->res.layout(0);
            LOGI("Draw#%u t%u: res %ux%u fmt=%u mips=%u arr=%u dim=%d | srv fmt=%u dim=%u mip=%u n=%u | sub0 %ux%u pitch=%u",
                 drawIndex, t, rd.width, rd.height, rd.format, rd.mipLevels, rd.arraySize, static_cast<int>(rd.dim),
                 vdesc[0], vdesc[1], vdesc[2], vdesc[3], l0 ? l0->width : 0, l0 ? l0->height : 0,
                 l0 ? l0->rowPitch : 0);
        }
        LOGI("Draw#%u uv in x[%.3f,%.3f] y[%.3f,%.3f] out x[%.3f,%.3f] y[%.3f,%.3f]", drawIndex, r.stats.inMin[0],
             r.stats.inMax[0], r.stats.inMin[1], r.stats.inMax[1], r.stats.outMin[0], r.stats.outMax[0],
             r.stats.outMin[1], r.stats.outMax[1]);
        LOGI("Draw: rt=%p %ux%u fmt=%u bb=%d topo=%u n=%u vs=%p ps=%p elems=%zu tri=%llu px=%llu culled=%llu "
             "vtx=%llu fmtNo=%llu vp=%.0fx%.0f blend=%d %.2f ms ndc x[%.3f,%.3f] y[%.3f,%.3f]",
             static_cast<void*>(rtRes), l->width, l->height, st.rt.format, rtRes == s_swapChain.backbuffer ? 1 : 0,
             st.topology, count, static_cast<void*>(s_ctx.vs), static_cast<void*>(s_ctx.ps), st.elements.size(),
             static_cast<unsigned long long>(r.stats.triangles), static_cast<unsigned long long>(r.stats.pixels),
             static_cast<unsigned long long>(r.stats.culled), static_cast<unsigned long long>(r.stats.vertices),
             static_cast<unsigned long long>(r.stats.unsupportedFormat), st.viewport.w, st.viewport.h,
             st.blend.enable ? 1 : 0, static_cast<double>(ns) / 1e6, r.stats.minX, r.stats.maxX, r.stats.minY,
             r.stats.maxY);
    }
}

// Slots de estado/dibujo del contexto. true si el slot se ha atendido (ret = valor devuelto).
bool contextStateSlot(int slot, const ComArgs& args, uint64_t& ret) {
    const uint64_t a1 = args.a[1], a2 = args.a[2], a3 = args.a[3], a4 = args.a[4], a5 = args.a[5];
    ret = 0;
    auto bindRange = [](auto* dst, size_t cap, uint64_t start, uint64_t num, uint64_t arr, auto conv) {
        const auto* p = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(arr));
        for (uint64_t i = 0; i < num && start + i < cap; ++i) dst[start + i] = p != nullptr ? conv(p[i]) : nullptr;
    };
    switch (slot) {
        case 7:   // VSSetConstantBuffers(start, num, pp)
        case 16: {  // PSSetConstantBuffers
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            bindRange(slot == 7 ? s_ctx.vsCb : s_ctx.psCb, 14, a1, a2, a3, resArg);
            return true;
        }
        case 8:   // PSSetShaderResources
        case 25: {  // VSSetShaderResources
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            bindRange(slot == 8 ? s_ctx.psSrv : s_ctx.vsSrv, 32, a1, a2, a3, viewArg);
            return true;
        }
        case 10:  // PSSetSamplers
        case 26: {  // VSSetSamplers
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            bindRange(slot == 10 ? s_ctx.psSamp : s_ctx.vsSamp, 16, a1, a2, a3, descArg);
            return true;
        }
        case 9: {  // PSSetShader
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.ps = descArg(a1);
            return true;
        }
        case 11: {  // VSSetShader
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.vs = descArg(a1);
            return true;
        }
        case 17: {  // IASetInputLayout
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.layout = descArg(a1);
            return true;
        }
        case 18: {  // IASetVertexBuffers(start, num, ppVB, pStrides, pOffsets)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            const auto* bufs = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(a3));
            const auto* strides = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a4));
            const auto* offsets = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a5));
            for (uint64_t i = 0; i < a2 && a1 + i < 16; ++i) {
                BoundVertexBuffer& b = s_ctx.vb[a1 + i];
                b.res = bufs != nullptr ? resArg(bufs[i]) : nullptr;
                b.stride = strides != nullptr ? strides[i] : 0;
                b.offset = offsets != nullptr ? offsets[i] : 0;
            }
            return true;
        }
        case 19: {  // IASetIndexBuffer(buf, format, offset)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.ib = resArg(a1);
            s_ctx.ibFormat = static_cast<uint32_t>(a2);
            s_ctx.ibOffset = static_cast<uint32_t>(a3);
            return true;
        }
        case 24: {  // IASetPrimitiveTopology
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.topology = static_cast<uint32_t>(a1);
            return true;
        }
        case 33: {  // OMSetRenderTargets(num, ppRTV, dsv)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            const auto* p = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(a2));
            for (uint32_t i = 0; i < 8; ++i) s_ctx.rtv[i] = (p != nullptr && i < a1) ? viewArg(p[i]) : nullptr;
            return true;
        }
        case 35: {  // OMSetBlendState(state, factor[4], mask)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.blend = descArg(a1);
            if (a2 != 0) {
                std::memcpy(s_ctx.blendFactor, reinterpret_cast<const void*>(static_cast<uintptr_t>(a2)), 16);
            } else {
                for (float& f : s_ctx.blendFactor) f = 1.0f;
            }
            return true;
        }
        case 43: {  // RSSetState
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.rs = descArg(a1);
            return true;
        }
        case 44: {  // RSSetViewports(num, pViewports)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            if (a1 != 0 && a2 != 0) {
                float v[6];
                std::memcpy(v, reinterpret_cast<const void*>(static_cast<uintptr_t>(a2)), sizeof(v));
                s_ctx.viewport = {v[0], v[1], v[2], v[3], v[4], v[5]};
                s_ctx.hasViewport = true;
                static std::atomic<uint32_t> s_vpLogs{0};
                if (s_vpLogs.fetch_add(1, std::memory_order_relaxed) < 6) {
                    LOGI("RSSetViewports(%llu): %.2f %.2f %.2f %.2f %.2f %.2f", static_cast<unsigned long long>(a1), v[0],
                         v[1], v[2], v[3], v[4], v[5]);
                }
            }
            return true;
        }
        case 45: {  // RSSetScissorRects(num, pRects)
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            if (a1 != 0 && a2 != 0) std::memcpy(s_ctx.scissor, reinterpret_cast<const void*>(static_cast<uintptr_t>(a2)), 16);
            return true;
        }
        case 12:  // DrawIndexed(count, start, base)
            executeDraw(static_cast<uint32_t>(a1), static_cast<uint32_t>(a2), static_cast<int32_t>(a3), true);
            return true;
        case 13:  // Draw(count, start)
            executeDraw(static_cast<uint32_t>(a1), static_cast<uint32_t>(a2), 0, false);
            return true;
        case 20:  // DrawIndexedInstanced(countPerInstance, instances, start, base, startInstance)
            if (a2 != 0) {
                executeDraw(static_cast<uint32_t>(a1), static_cast<uint32_t>(a3), static_cast<int32_t>(a4), true,
                            static_cast<uint32_t>(a2), static_cast<uint32_t>(a5));
            }
            return true;
        case 21:  // DrawInstanced(countPerInstance, instances, startVertex, startInstance)
            if (a2 != 0) {
                executeDraw(static_cast<uint32_t>(a1), static_cast<uint32_t>(a3), 0, false, static_cast<uint32_t>(a2),
                            static_cast<uint32_t>(a4));
            }
            return true;
        case 110: {  // ClearState
            std::lock_guard<std::mutex> lock(s_ctx.mtx);
            s_ctx.vs = s_ctx.ps = s_ctx.layout = s_ctx.blend = s_ctx.rs = nullptr;
            for (auto& b : s_ctx.vb) b = {};
            s_ctx.ib = nullptr;
            s_ctx.topology = 0;
            std::fill(std::begin(s_ctx.vsCb), std::end(s_ctx.vsCb), nullptr);
            std::fill(std::begin(s_ctx.psCb), std::end(s_ctx.psCb), nullptr);
            std::fill(std::begin(s_ctx.vsSrv), std::end(s_ctx.vsSrv), nullptr);
            std::fill(std::begin(s_ctx.psSrv), std::end(s_ctx.psSrv), nullptr);
            std::fill(std::begin(s_ctx.vsSamp), std::end(s_ctx.vsSamp), nullptr);
            std::fill(std::begin(s_ctx.psSamp), std::end(s_ctx.psSamp), nullptr);
            std::fill(std::begin(s_ctx.rtv), std::end(s_ctx.rtv), nullptr);
            s_ctx.hasViewport = false;
            ret = kHrOk;
            return true;
        }
        default:
            return false;
    }
}

// =============================================================================
// WASAPI (MMDevAPI) -> AAudio
//
// Un único endpoint de reproducción ("HyperPlay Speakers", 48 kHz float estéreo). Cada
// IMMDevice::Activate(IAudioClient) crea un cliente con su anillo de frames PCM16 estéreo; un
// hilo host consume un periodo (10 ms) por vuelta, lo escribe bloqueando en AAudio (que marca el
// ritmo real) y señala el evento del cliente (modo AUDCLNT_STREAMFLAGS_EVENTCALLBACK).
// =============================================================================

void ensureComReady();
void initComObject(ComObject& object, void** vtbl, uint32_t kind);

constexpr uint32_t kAudclntNotInitialized = 0x88890001u;
constexpr uint32_t kAudclntAlreadyInitialized = 0x88890002u;
constexpr uint32_t kAudclntBufferTooLarge = 0x88890006u;
constexpr uint32_t kAudclntUnsupportedFormat = 0x88890008u;
constexpr uint32_t kAudclntNotStopped = 0x8889000Bu;
constexpr uint32_t kAudclntBufferOpPending = 0x88890011u;
constexpr uint32_t kHrNotFound = 0x80070490u;  // HRESULT_FROM_WIN32(ERROR_NOT_FOUND)
constexpr uint32_t kHrSFalse = 1u;
constexpr uint32_t kWasapiRate = 48000;
constexpr int64_t kWasapiPeriodHns = 100000;  // 10 ms

ComObject s_comMMEnum;
ComObject s_comMMDevice;
ComObject s_comMMCollection;
ComObject s_comPropStore;
ComObject s_comMMEndpoint;

struct WasapiFormat {
    uint32_t rate = kWasapiRate;
    uint32_t channels = 2;
    uint32_t bits = 32;
    uint32_t blockAlign = 8;
    bool isFloat = true;
};

struct WasapiClient;
struct WasapiSub {
    ComObject com;
    WasapiClient* owner = nullptr;
};

struct WasapiClient {
    ComObject com;
    WasapiSub render;
    WasapiSub clock;
    WasapiSub volume;
    WasapiSub session;
    std::mutex mtx;
    bool initialized = false;
    std::atomic<bool> running{false};
    std::atomic<bool> quit{false};
    WasapiFormat fmt;
    uint32_t bufferFrames = 0;
    uint32_t periodFrames = 480;
    std::vector<int16_t> ring;  // estéreo intercalado, bufferFrames * 2
    uint32_t ringRead = 0;
    uint32_t queued = 0;
    std::vector<uint8_t> staging;
    uint32_t stagingFrames = 0;
    bool bufferOut = false;
    void* event = nullptr;
    std::atomic<uint64_t> played{0};
    std::atomic<float> masterVolume{1.0f};
    std::atomic<bool> muted{false};
    float limiterGain = 1.0f;  // limitador de salida (solo ReleaseBuffer, bajo mtx)
    std::thread pump;
    DirectAAudioEngine engine;
    // Diagnóstico (línea "WASAPI ritmo" cada 10 s).
    std::atomic<uint64_t> diagGetBuf{0}, diagWantSum{0}, diagPadSum{0}, diagSilence{0}, diagPadCalls{0}, diagEvents{0};
    std::atomic<uint32_t> diagWantMax{0};
};

std::mutex s_wasapiMutex;
std::unordered_set<const void*> s_wasapiClients;

// Lee WAVEFORMATEX / WAVEFORMATEXTENSIBLE del huésped.
bool readWaveFormat(uint64_t address, WasapiFormat& out) {
    if (address == 0) return false;
    const auto* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address));
    uint16_t tag = 0, channels = 0, blockAlign = 0, bits = 0, cbSize = 0;
    uint32_t rate = 0;
    std::memcpy(&tag, p, 2);
    std::memcpy(&channels, p + 2, 2);
    std::memcpy(&rate, p + 4, 4);
    std::memcpy(&blockAlign, p + 12, 2);
    std::memcpy(&bits, p + 14, 2);
    std::memcpy(&cbSize, p + 16, 2);
    bool isFloat = tag == 3;  // WAVE_FORMAT_IEEE_FLOAT
    if (tag == 0xFFFE && cbSize >= 22) {  // WAVE_FORMAT_EXTENSIBLE: SubFormat.Data1 en +24
        uint32_t sub = 0;
        std::memcpy(&sub, p + 24, 4);
        if (sub == 3) isFloat = true;
        else if (sub == 1) isFloat = false;
        else return false;
    } else if (tag != 1 && tag != 3) {
        return false;
    }
    if (channels == 0 || channels > 8 || rate < 8000 || rate > 192000) return false;
    if (isFloat ? bits != 32 : (bits != 16 && bits != 24 && bits != 32)) return false;
    out.rate = rate;
    out.channels = channels;
    out.bits = bits;
    out.blockAlign = blockAlign != 0 ? blockAlign : channels * (bits / 8);
    out.isFloat = isFloat;
    return true;
}

// WAVEFORMATEXTENSIBLE float32 48 kHz estéreo en memoria CoTaskMem (la libera el huésped).
void* allocMixFormat() {
    auto* p = static_cast<uint8_t*>(NativeDllInterceptor::win_CoTaskMemAlloc(40));
    if (p == nullptr) return nullptr;
    std::memset(p, 0, 40);
    const uint16_t tag = 0xFFFE, ch = 2, align = 8, bits = 32, cb = 22, valid = 32;
    const uint32_t rate = kWasapiRate, avg = kWasapiRate * 8, mask = 3;
    std::memcpy(p, &tag, 2);
    std::memcpy(p + 2, &ch, 2);
    std::memcpy(p + 4, &rate, 4);
    std::memcpy(p + 8, &avg, 4);
    std::memcpy(p + 12, &align, 2);
    std::memcpy(p + 14, &bits, 2);
    std::memcpy(p + 16, &cb, 2);
    std::memcpy(p + 18, &valid, 2);
    std::memcpy(p + 20, &mask, 4);
    // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {00000003-0000-0010-8000-00AA00389B71}
    static const uint8_t kFloatGuid[16] = {0x03, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71};
    std::memcpy(p + 24, kFloatGuid, 16);
    return p;
}

// Convierte frames del formato del cliente a PCM16 estéreo. 'limiter' es la ganancia del limitador del
// cliente (estado entre bloques): el motor de audio de Windows (modo compartido) tiene un limitador y los
// juegos que mezclan en float (OpenAL Soft en Stardew Valley) superan 1.0 sin preocuparse; cortar en seco
// a 16 bits sonaba distorsionado. Ataque inmediato al pico del bloque, rampa dentro del bloque (sin
// chasquidos) y recuperación lenta (~0.3 s).
void convertToStereo16(const WasapiFormat& f, const uint8_t* src, uint32_t frames, int16_t* dst, float gain,
                       float* limiter = nullptr) {
    auto sampleAt = [&](const uint8_t* frame, uint32_t ch) -> float {
        const uint8_t* s = frame + ch * (f.bits / 8);
        if (f.isFloat) {
            float v;
            std::memcpy(&v, s, 4);
            return v;
        }
        if (f.bits == 16) {
            int16_t v;
            std::memcpy(&v, s, 2);
            return static_cast<float>(v) / 32768.0f;
        }
        if (f.bits == 24) {
            const int32_t v = (static_cast<int32_t>(s[0]) << 8 | static_cast<int32_t>(s[1]) << 16 |
                               static_cast<int32_t>(s[2]) << 24) >> 8;
            return static_cast<float>(v) / 8388608.0f;
        }
        int32_t v;
        std::memcpy(&v, s, 4);
        return static_cast<float>(v) / 2147483648.0f;
    };
    float g0 = 1.0f, g1 = 1.0f;
    if (limiter != nullptr && frames != 0) {
        float peak = 0.0f;
        for (uint32_t i = 0; i < frames; ++i) {
            const uint8_t* frame = src + static_cast<size_t>(i) * f.blockAlign;
            peak = std::max(peak, std::fabs(sampleAt(frame, 0) * gain));
            if (f.channels >= 2) peak = std::max(peak, std::fabs(sampleAt(frame, 1) * gain));
        }
        constexpr float kCeiling = 0.98f;
        const float need = peak > kCeiling ? kCeiling / peak : 1.0f;
        // Recuperación: ~0.3 s de 0 dB de reducción a 1.0 con bloques de 10-20 ms.
        const float released = std::min(1.0f, *limiter + (1.0f - *limiter) * 0.06f + 0.002f);
        g0 = *limiter;
        g1 = std::min(released, need);
        if (need < g0) g0 = need;  // ataque: el bloque entero ya baja lo necesario (sin cortar el pico)
        *limiter = g1;
    }
    for (uint32_t i = 0; i < frames; ++i) {
        const uint8_t* frame = src + static_cast<size_t>(i) * f.blockAlign;
        const float lg = g0 + (g1 - g0) * (static_cast<float>(i) / static_cast<float>(frames));
        const float l = sampleAt(frame, 0) * lg;
        const float r = (f.channels >= 2 ? sampleAt(frame, 1) : sampleAt(frame, 0)) * lg;
        auto clamp16 = [gain](float v) {
            v *= gain * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            return static_cast<int16_t>(v);
        };
        dst[i * 2] = clamp16(l);
        dst[i * 2 + 1] = clamp16(r);
    }
}

void wasapiPumpLoop(WasapiClient* c) {
    // Margen de ~40 ms en AAudio: el bombeo entrega bloques de un periodo (10-20 ms) y con el buffer
    // mínimo (2 ráfagas, ~4 ms) cualquier retraso del hilo era un underrun audible.
    const int32_t kWasapiAaudioMs = 40;
    bool haveEngine = c->engine.start(static_cast<int32_t>(c->fmt.rate), 2, kWasapiAaudioMs);
    bool stoppedForPause = false;
    LOGI("WASAPI: bombeo iniciado (%u Hz, %u canales %s%u, buffer %u frames, periodo %u, AAudio=%d)",
         c->fmt.rate, c->fmt.channels, c->fmt.isFloat ? "float" : "pcm", c->fmt.bits, c->bufferFrames,
         c->periodFrames, haveEngine ? 1 : 0);
    std::vector<int16_t> chunk(static_cast<size_t>(c->periodFrames) * 2);
    auto next = std::chrono::steady_clock::now();
    while (!c->quit.load(std::memory_order_acquire)) {
        // Juego en pausa (Quick Resume / segundo plano): sin stream de AAudio (hardware de audio
        // apagado) y sin eventos: el hilo de audio del juego espera. Al reanudar, se reabre.
        if (::hyperplay::pacing::paused()) {
            if (haveEngine) {
                c->engine.stop();
                haveEngine = false;
                stoppedForPause = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            next = std::chrono::steady_clock::now();
            continue;
        }
        if (stoppedForPause) {
            stoppedForPause = false;
            haveEngine = c->engine.start(static_cast<int32_t>(c->fmt.rate), 2, kWasapiAaudioMs);
        }
        if (!c->running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            next = std::chrono::steady_clock::now();
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(c->mtx);
            const uint32_t take = std::min(c->queued, c->periodFrames);
            for (uint32_t i = 0; i < take; ++i) {
                const uint32_t idx = (c->ringRead + i) % c->bufferFrames;
                chunk[i * 2] = c->ring[idx * 2];
                chunk[i * 2 + 1] = c->ring[idx * 2 + 1];
            }
            std::fill(chunk.begin() + take * 2, chunk.end(), int16_t{0});  // underrun -> silencio
            c->diagSilence.fetch_add(c->periodFrames - take, std::memory_order_relaxed);
            c->ringRead = (c->ringRead + take) % c->bufferFrames;
            c->queued -= take;
        }
        bool wrote = false;
        if (haveEngine && c->engine.isRunning()) {
            // Todo el periodo entra en AAudio (cada intento espera hasta 10 ms): soltar frames ya sacados
            // del anillo era un salto audible (chasquidos/distorsión en Stardew Valley).
            uint32_t done = 0;
            for (int tries = 0; tries < 30 && done < c->periodFrames && !c->quit.load(std::memory_order_relaxed);
                 ++tries) {
                const int32_t n = c->engine.writePcm(chunk.data() + done * 2,
                                                     static_cast<int32_t>(c->periodFrames - done));
                if (n > 0) done += static_cast<uint32_t>(n);
            }
            wrote = done > 0;
        }
        if (!wrote) {  // sin AAudio: ritmo por reloj
            next += std::chrono::microseconds(1000000ull * c->periodFrames / c->fmt.rate);
            std::this_thread::sleep_until(next);
        }
        c->played.fetch_add(c->periodFrames, std::memory_order_relaxed);
        if (c->event != nullptr) {
            NativeDllInterceptor::win_SetEvent(c->event);
            c->diagEvents.fetch_add(1, std::memory_order_relaxed);
        }
    }
    c->engine.stop();
}

WasapiClient* asWasapiClient(ComObject* self) {
    if (self == nullptr) return nullptr;
    if (self->kind == kComAudioClient) return reinterpret_cast<WasapiClient*>(self);
    if (self->kind >= kComAudioRender && self->kind <= kComAudioSession) {
        return reinterpret_cast<WasapiSub*>(self)->owner;
    }
    return nullptr;
}

WasapiClient* createWasapiClient() {
    auto* c = new WasapiClient();
    initComObject(c->com, ComVtable<kComAudioClient, 24>::slots.data(), kComAudioClient);
    initComObject(c->render.com, ComVtable<kComAudioRender, 8>::slots.data(), kComAudioRender);
    initComObject(c->clock.com, ComVtable<kComAudioClock, 8>::slots.data(), kComAudioClock);
    initComObject(c->volume.com, ComVtable<kComAudioVolume, 8>::slots.data(), kComAudioVolume);
    initComObject(c->session.com, ComVtable<kComAudioSession, 16>::slots.data(), kComAudioSession);
    c->render.owner = c->clock.owner = c->volume.owner = c->session.owner = c;
    std::lock_guard<std::mutex> lock(s_wasapiMutex);
    s_wasapiClients.insert(c);
    return c;
}

void stopWasapiClient(WasapiClient* c) {
    c->running.store(false, std::memory_order_release);
    c->quit.store(true, std::memory_order_release);
    if (c->pump.joinable() && c->pump.get_id() != std::this_thread::get_id()) c->pump.join();
}
}  // namespace

// Al cerrar el juego (Force Kill, sesión terminada o abrir otro juego con uno en Quick Resume)
// los hilos de audio del huésped no se detienen solos: el bombeo seguía enviando a AAudio y el
// juego "se oía" en el launcher. Se paran todos los clientes WASAPI (cierra cada stream AAudio).
void NativeDllInterceptor::stopAllAudio() noexcept {
    dsound32::stopAll();
    std::vector<WasapiClient*> clients;
    {
        std::lock_guard<std::mutex> lock(s_wasapiMutex);
        for (const void* p : s_wasapiClients) clients.push_back(const_cast<WasapiClient*>(static_cast<const WasapiClient*>(p)));
    }
    for (WasapiClient* c : clients) stopWasapiClient(c);
    if (!clients.empty()) LOGI("Audio del juego detenido (%zu clientes WASAPI)", clients.size());
}
namespace {

// PROPVARIANT: vt en +0, valor en +8.
void writePropString(uint64_t pv, const char16_t* text) {
    auto* p = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(pv));
    std::memset(p, 0, 24);
    size_t n = 0;
    while (text[n] != 0) ++n;
    auto* s = static_cast<char16_t*>(NativeDllInterceptor::win_CoTaskMemAlloc((n + 1) * sizeof(char16_t)));
    if (s == nullptr) return;
    std::memcpy(s, text, (n + 1) * sizeof(char16_t));
    const uint16_t vt = 31;  // VT_LPWSTR
    std::memcpy(p, &vt, 2);
    std::memcpy(p + 8, &s, 8);
}

constexpr char16_t kWasapiDeviceId[] = u"{0.0.0.00000000}.{8d6f1a7c-3c1b-4b7e-9a51-48797065725a}";

uint64_t wasapiDispatch(int iface, int slot, const ComArgs& args) {
    const uint64_t a0 = args.a[0], a1 = args.a[1], a2 = args.a[2], a3 = args.a[3];
    const uint64_t a4 = args.a[4], a5 = args.a[5];
    auto* self = reinterpret_cast<ComObject*>(static_cast<uintptr_t>(a0));
    if (self == nullptr) return hr(kHrPointer);
    WasapiClient* client = asWasapiClient(self);

    // IUnknown
    if (slot == 1) return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
    if (slot == 2) {
        uint32_t previous = self->refs.load(std::memory_order_relaxed);
        do {
            if (previous == 0) return 0;
        } while (!self->refs.compare_exchange_weak(previous, previous - 1, std::memory_order_acq_rel));
        // El cliente se detiene al soltarse; la memoria se conserva (los sub-objetos pueden
        // seguir referenciados por el huésped).
        if (previous == 1 && iface == kComAudioClient && client != nullptr) stopWasapiClient(client);
        return previous - 1;
    }
    if (slot == 0) {
        const uint32_t iid = readGuidData1(a1);
        ComObject* out = nullptr;
        if (iid == 0x00000000u) out = self;
        switch (iface) {
            case kComMMEnumerator: if (iid == 0xA95664D2u) out = self; break;
            case kComMMDevice:
                if (iid == 0xD666063Fu) out = self;
                else if (iid == 0x1BE09788u) out = &s_comMMEndpoint;  // IMMEndpoint
                break;
            case kComMMEndpoint: if (iid == 0x1BE09788u) out = self; break;
            case kComMMCollection: if (iid == 0x0BD7A1BEu) out = self; break;
            case kComPropertyStore: if (iid == 0x886D8EEBu) out = self; break;
            case kComAudioClient:
                if (iid == 0x1CB9AD4Cu || iid == 0x726778CDu || iid == 0x7ED4EE07u) out = self;
                break;
            case kComAudioRender: if (iid == 0xF294ACFCu) out = self; break;
            case kComAudioClock: if (iid == 0xCD63314Fu) out = self; break;
            case kComAudioVolume: if (iid == 0x87CE5498u) out = self; break;
            case kComAudioSession: if (iid == 0xF4B1A599u || iid == 0xBFB7FF88u) out = self; break;
            default: break;
        }
        if (out == nullptr) {
            storePtr(a2, nullptr);
            return hr(kHrNoInterface);
        }
        out->refs.fetch_add(1, std::memory_order_relaxed);
        return hr(storePtr(a2, out));
    }

    switch (iface) {
        case kComMMEnumerator:
            switch (slot) {
                case 3:  // EnumAudioEndpoints(dataFlow, stateMask, ppDevices)
                    s_comMMCollection.refs.fetch_add(1, std::memory_order_relaxed);
                    s_comMMCollection.kind = kComMMCollection;
                    // Captura (1) no tiene dispositivos: la colección lo refleja en GetCount.
                    s_comMMCollection.heap = (a1 == 1);
                    return hr(storePtr(a3, &s_comMMCollection));
                case 4:  // GetDefaultAudioEndpoint(dataFlow, role, ppEndpoint)
                    if (a1 == 1) {
                        storePtr(a3, nullptr);
                        return hr(kHrNotFound);
                    }
                    s_comMMDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a3, &s_comMMDevice));
                case 5:  // GetDevice(id, ppDevice)
                    s_comMMDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comMMDevice));
                case 6:  // Register/UnregisterEndpointNotificationCallback
                case 7:
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }

        case kComMMCollection:
            switch (slot) {
                case 3:  // GetCount
                    return hr(storeU32(a1, s_comMMCollection.heap ? 0u : 1u));
                case 4:  // Item(n, ppDevice)
                    if (s_comMMCollection.heap || a1 != 0) {
                        storePtr(a2, nullptr);
                        return hr(0x80070057u);  // E_INVALIDARG
                    }
                    s_comMMDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comMMDevice));
                default:
                    return hr(kHrNotImpl);
            }

        case kComMMDevice:
            switch (slot) {
                case 3: {  // Activate(iid, clsCtx, params, ppInterface)
                    const uint32_t iid = readGuidData1(a1);
                    if (iid == 0x1CB9AD4Cu || iid == 0x726778CDu || iid == 0x7ED4EE07u) {
                        WasapiClient* c = createWasapiClient();
                        LOGI("WASAPI: IMMDevice::Activate(IAudioClient) -> %p", static_cast<void*>(c));
                        return hr(storePtr(a4, c));
                    }
                    storePtr(a4, nullptr);
                    return hr(kHrNoInterface);
                }
                case 4:  // OpenPropertyStore
                    s_comPropStore.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comPropStore));
                case 5: {  // GetId
                    const size_t bytes = sizeof(kWasapiDeviceId);
                    void* s = NativeDllInterceptor::win_CoTaskMemAlloc(bytes);
                    if (s == nullptr) return hr(kHrOutOfMemory);
                    std::memcpy(s, kWasapiDeviceId, bytes);
                    return hr(storePtr(a1, s));
                }
                case 6:  // GetState -> DEVICE_STATE_ACTIVE
                    return hr(storeU32(a1, 1));
                default:
                    return hr(kHrNotImpl);
            }

        case kComPropertyStore:
            switch (slot) {
                case 3:  // GetCount
                    return hr(storeU32(a1, 0));
                case 5: {  // GetValue(PROPERTYKEY*, PROPVARIANT*)
                    if (a2 == 0) return hr(kHrPointer);
                    const uint32_t keyData1 = readGuidData1(a1);
                    uint32_t pid = 0;
                    if (a1 != 0) std::memcpy(&pid, reinterpret_cast<const void*>(static_cast<uintptr_t>(a1 + 16)), 4);
                    if ((keyData1 == 0xA45C254Eu && (pid == 14 || pid == 2)) ||  // FriendlyName / DeviceDesc
                        (keyData1 == 0x026E516Eu && pid == 2) ||                 // DeviceInterface_FriendlyName
                        (keyData1 == 0xB3F8FA53u && pid == 6)) {
                        writePropString(a2, u"HyperPlay Speakers");
                        return hr(kHrOk);
                    }
                    std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(a2)), 0, 24);  // VT_EMPTY
                    return hr(kHrOk);
                }
                case 6:
                case 7:
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }

        case kComAudioClient: {
            if (client == nullptr) return hr(kHrPointer);
            switch (slot) {
                case 3: {  // Initialize(shareMode, flags, hnsBuffer, hnsPeriod, pFormat, sessionGuid)
                    std::lock_guard<std::mutex> lock(client->mtx);
                    if (client->initialized) return hr(kAudclntAlreadyInitialized);
                    WasapiFormat f;
                    if (!readWaveFormat(a5, f)) return hr(kAudclntUnsupportedFormat);
                    const int64_t hnsBuffer = static_cast<int64_t>(a3);
                    client->fmt = f;
                    client->periodFrames = static_cast<uint32_t>(f.rate * kWasapiPeriodHns / 10000000);
                    uint32_t frames = hnsBuffer > 0 ? static_cast<uint32_t>(f.rate * hnsBuffer / 10000000) : 0;
                    client->bufferFrames = std::max(frames, client->periodFrames * 2);
                    client->ring.assign(static_cast<size_t>(client->bufferFrames) * 2, 0);
                    client->ringRead = 0;
                    client->queued = 0;
                    client->initialized = true;
                    client->pump = std::thread(wasapiPumpLoop, client);
                    LOGI("WASAPI: Initialize(share=%llu flags=0x%llx buffer=%lld hns) -> %u frames",
                         static_cast<unsigned long long>(a1), static_cast<unsigned long long>(a2),
                         static_cast<long long>(hnsBuffer), client->bufferFrames);
                    return hr(kHrOk);
                }
                case 4:  // GetBufferSize
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    return hr(storeU32(a1, client->bufferFrames));
                case 5:  // GetStreamLatency
                    if (a1 == 0) return hr(kHrPointer);
                    *reinterpret_cast<int64_t*>(static_cast<uintptr_t>(a1)) = kWasapiPeriodHns;
                    return hr(kHrOk);
                case 6: {  // GetCurrentPadding
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    std::lock_guard<std::mutex> lock(client->mtx);
                    client->diagPadCalls.fetch_add(1, std::memory_order_relaxed);
                    return hr(storeU32(a1, client->queued));
                }
                case 7: {  // IsFormatSupported(shareMode, pFormat, ppClosest)
                    WasapiFormat f;
                    if (a3 != 0) storePtr(a3, nullptr);
                    if (readWaveFormat(a2, f)) return hr(kHrOk);
                    if (a1 == 0 && a3 != 0) {  // compartido: proponer el formato de mezcla
                        storePtr(a3, allocMixFormat());
                        return hr(kHrSFalse);
                    }
                    return hr(kAudclntUnsupportedFormat);
                }
                case 8:  // GetMixFormat
                    return hr(storePtr(a1, allocMixFormat()));
                case 9:  // GetDevicePeriod(default, min)
                    if (a1 != 0) *reinterpret_cast<int64_t*>(static_cast<uintptr_t>(a1)) = kWasapiPeriodHns;
                    if (a2 != 0) *reinterpret_cast<int64_t*>(static_cast<uintptr_t>(a2)) = 30000;
                    return hr(kHrOk);
                case 10:  // Start
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    if (client->running.exchange(true)) return hr(kAudclntNotStopped);
                    return hr(kHrOk);
                case 11:  // Stop
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    return hr(client->running.exchange(false) ? kHrOk : kHrSFalse);
                case 12: {  // Reset
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    if (client->running.load()) return hr(kAudclntNotStopped);
                    std::lock_guard<std::mutex> lock(client->mtx);
                    if (client->bufferOut) return hr(kAudclntBufferOpPending);
                    client->queued = 0;
                    client->ringRead = 0;
                    client->played.store(0);
                    return hr(kHrOk);
                }
                case 13:  // SetEventHandle
                    client->event = reinterpret_cast<void*>(static_cast<uintptr_t>(a1));
                    return hr(kHrOk);
                case 14: {  // GetService(riid, ppv)
                    const uint32_t iid = readGuidData1(a1);
                    ComObject* out = nullptr;
                    if (iid == 0xF294ACFCu) out = &client->render.com;
                    else if (iid == 0xCD63314Fu) out = &client->clock.com;
                    else if (iid == 0x87CE5498u) out = &client->volume.com;
                    else if (iid == 0xF4B1A599u || iid == 0xBFB7FF88u) out = &client->session.com;
                    if (out == nullptr) {
                        storePtr(a2, nullptr);
                        return hr(kHrNoInterface);
                    }
                    if (!client->initialized) return hr(kAudclntNotInitialized);
                    out->refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, out));
                }
                case 15:  // IsOffloadCapable
                    return hr(storeU32(a2, 0));
                case 16:  // SetClientProperties
                    return hr(kHrOk);
                case 17:  // GetBufferSizeLimits(format, eventDriven, min, max)
                    if (a3 != 0) *reinterpret_cast<int64_t*>(static_cast<uintptr_t>(a3)) = 30000;
                    if (a4 != 0) *reinterpret_cast<int64_t*>(static_cast<uintptr_t>(a4)) = 20000000;
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }
        }

        case kComAudioRender: {
            if (client == nullptr) return hr(kHrPointer);
            std::lock_guard<std::mutex> lock(client->mtx);
            switch (slot) {
                case 3: {  // GetBuffer(numFrames, ppData)
                    const auto want = static_cast<uint32_t>(a1);
                    client->diagGetBuf.fetch_add(1, std::memory_order_relaxed);
                    client->diagWantSum.fetch_add(want, std::memory_order_relaxed);
                    client->diagPadSum.fetch_add(client->queued, std::memory_order_relaxed);
                    if (want > client->diagWantMax.load(std::memory_order_relaxed)) client->diagWantMax.store(want);
                    if (client->bufferOut) return hr(kAudclntBufferOpPending);
                    if (want > client->bufferFrames - client->queued) {
                        storePtr(a2, nullptr);
                        return hr(kAudclntBufferTooLarge);
                    }
                    client->staging.resize(std::max<size_t>(1, static_cast<size_t>(want) * client->fmt.blockAlign));
                    client->stagingFrames = want;
                    client->bufferOut = true;
                    return hr(storePtr(a2, client->staging.data()));
                }
                case 4: {  // ReleaseBuffer(numFrames, flags)
                    if (!client->bufferOut) return hr(kAudclntBufferOpPending);
                    const uint32_t n = std::min(static_cast<uint32_t>(a1), client->stagingFrames);
                    client->bufferOut = false;
                    {
                        // Diagnóstico de ritmo: frames por segundo que entrega el juego (debería
                        // igualar la frecuencia: más = mezcla más rápido que el tiempo real).
                        static std::atomic<uint64_t> s_frames{0}, s_calls{0}, s_t0{0};
                        s_frames.fetch_add(n, std::memory_order_relaxed);
                        s_calls.fetch_add(1, std::memory_order_relaxed);
                        const uint64_t now = monoMs();
                        uint64_t t0 = s_t0.load(std::memory_order_relaxed);
                        if (t0 == 0) s_t0.compare_exchange_strong(t0, now);
                        else if (now - t0 >= 10000 && s_t0.compare_exchange_strong(t0, now)) {
                            const uint64_t gb = std::max<uint64_t>(1, client->diagGetBuf.exchange(0));
                            LOGI("WASAPI ritmo: %.0f frames/s en %.0f ReleaseBuffer/s (formato %u Hz) | GetBuffer "
                                 "med %.0f max %u, padding med %.0f, silencio %.0f frames/s, padding/s %.0f, eventos/s %.0f, "
                                 "xruns AAudio %d",
                                 s_frames.exchange(0) * 1000.0 / (now - t0), s_calls.exchange(0) * 1000.0 / (now - t0),
                                 client->fmt.rate, static_cast<double>(client->diagWantSum.exchange(0)) / gb,
                                 client->diagWantMax.exchange(0), static_cast<double>(client->diagPadSum.exchange(0)) / gb,
                                 client->diagSilence.exchange(0) * 1000.0 / (now - t0),
                                 client->diagPadCalls.exchange(0) * 1000.0 / (now - t0),
                                 client->diagEvents.exchange(0) * 1000.0 / (now - t0), client->engine.underrunCount());
                        }
                    }
                    std::vector<int16_t> tmp(static_cast<size_t>(n) * 2, 0);
                    const bool silent = (a2 & 0x2) != 0;  // AUDCLNT_BUFFERFLAGS_SILENT
                    const float gain = client->muted.load() ? 0.0f : client->masterVolume.load();
                    if (!silent && n != 0)
                        convertToStereo16(client->fmt, client->staging.data(), n, tmp.data(), gain, &client->limiterGain);
                    // Diagnóstico: setprop debug.hyperplay.wasapidump N -> los primeros N s de audio del juego
                    // ya convertidos a PCM16 estéreo (lo que suena) en <juego>/_HyperPlaySaves/wasapi.raw.
                    {
                        static FILE* s_dump = [] {
                            char v[PROP_VALUE_MAX] = {};
                            if (__system_property_get("debug.hyperplay.wasapidump", v) <= 0 || std::atoi(v) <= 0) return (FILE*)nullptr;
                            const std::string path = (s_activeImage ? s_activeImage->gameDir : std::string("/sdcard")) +
                                                     "/_HyperPlaySaves/wasapi.raw";
                            FILE* f = std::fopen(path.c_str(), "wb");
                            LOGI("WASAPI: volcado de audio en %s (%s)", path.c_str(), f ? "abierto" : "error");
                            return f;
                        }();
                        static std::atomic<uint64_t> s_dumped{0};
                        static const uint64_t s_limit = [] {
                            char v[PROP_VALUE_MAX] = {};
                            __system_property_get("debug.hyperplay.wasapidump", v);
                            return static_cast<uint64_t>(std::max(0, std::atoi(v))) * 48000u;
                        }();
                        if (s_dump != nullptr && n != 0 && s_dumped.load() < s_limit) {
                            std::fwrite(tmp.data(), 4, n, s_dump);
                            if (s_dumped.fetch_add(n) + n >= s_limit) std::fflush(s_dump);
                        }
                    }
                    uint32_t write = (client->ringRead + client->queued) % client->bufferFrames;
                    for (uint32_t i = 0; i < n; ++i) {
                        client->ring[write * 2] = tmp[i * 2];
                        client->ring[write * 2 + 1] = tmp[i * 2 + 1];
                        write = (write + 1) % client->bufferFrames;
                    }
                    client->queued += n;
                    return hr(kHrOk);
                }
                default:
                    return hr(kHrNotImpl);
            }
        }

        case kComAudioClock:
            if (client == nullptr) return hr(kHrPointer);
            switch (slot) {
                case 3:  // GetFrequency (frames por segundo)
                    if (a1 == 0) return hr(kHrPointer);
                    *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a1)) = client->fmt.rate;
                    return hr(kHrOk);
                case 4: {  // GetPosition(pos, qpc)
                    if (a1 == 0) return hr(kHrPointer);
                    *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a1)) = client->played.load();
                    if (a2 != 0) {
                        timespec ts{};
                        clock_gettime(CLOCK_MONOTONIC, &ts);
                        *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a2)) =
                            static_cast<uint64_t>(ts.tv_sec) * 10000000ULL + static_cast<uint64_t>(ts.tv_nsec) / 100ULL;
                    }
                    return hr(kHrOk);
                }
                case 5:
                    return hr(storeU32(a1, 0));
                default:
                    return hr(kHrNotImpl);
            }

        case kComAudioVolume:
            if (client == nullptr) return hr(kHrPointer);
            switch (slot) {
                case 3: {  // SetMasterVolume(float, ctx): el float llega en XMM1 -> no visible; se ignora
                    return hr(kHrOk);
                }
                case 4:  // GetMasterVolume(float*)
                    if (a1 == 0) return hr(kHrPointer);
                    *reinterpret_cast<float*>(static_cast<uintptr_t>(a1)) = client->masterVolume.load();
                    return hr(kHrOk);
                case 5:  // SetMute(BOOL, ctx)
                    client->muted.store(a1 != 0);
                    return hr(kHrOk);
                case 6:  // GetMute(BOOL*)
                    return hr(storeU32(a1, client->muted.load() ? 1u : 0u));
                default:
                    return hr(kHrNotImpl);
            }

        case kComMMEndpoint:
            if (slot == 3) return hr(storeU32(a1, 0));  // GetDataFlow -> eRender
            return hr(kHrNotImpl);

        case kComAudioSession:
            switch (slot) {
                case 3:  // GetState -> AudioSessionStateActive
                    return hr(storeU32(a1, 1));
                case 10:  // RegisterAudioSessionNotification
                case 11:  // UnregisterAudioSessionNotification
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }

        default:
            return hr(kHrNotImpl);
    }
}

// =============================================================================
// WASAPI para invitados de 32 bits (Wwise de DmC).
//
// Los objetos WASAPI de arriba son de 64 bits (vtable de punteros de 8 bytes, direcciones del host).
// Para un invitado x86 cada objeto tiene un proxy en memoria baja: su primera palabra es una vtable de
// 32 bits cuyas entradas son alias < 4 GB de funciones nativas con la aridad exacta de cada método
// (stdcall, `this` incluido). El proxy llama a wasapiDispatch con el objeto real; los punteros de salida
// se reciben en temporales de 8 bytes y se devuelven al invitado como 4 bytes (interfaces -> su proxy).
// GetBuffer entrega un búfer en memoria baja que ReleaseBuffer copia al del cliente. La ruta de 64 bits
// no cambia.
// =============================================================================
namespace wasapi32 {
using u32 = uint32_t;
using i64 = int64_t;

struct Proxy {
    u32 vtbl;  // vtable de 32 bits (lo que lee el invitado)
    u32 pad;
    ComObject* host;
    uint8_t* lowBuf;        // GetBuffer: búfer visible para el invitado
    u32 lowCap;
    uint8_t* hostStaging;   // GetBuffer: búfer real del cliente
    u32 stagingBytes;
};

std::mutex g_mtx;
std::unordered_map<const ComObject*, Proxy*> g_proxies;
u32 g_vt[kComMMEndpoint + 1] = {};

uint8_t* lowAlloc(size_t n) {
    static uint8_t* cur = nullptr;
    static size_t left = 0;
    n = (n + 15) & ~size_t{15};
    if (cur == nullptr || left < n) {
        void* p = lowmem::mmapBelow4G(0x10000);
        if (p == MAP_FAILED) return nullptr;
        cur = static_cast<uint8_t*>(p);
        left = 0x10000;
    }
    uint8_t* r = cur;
    cur += n;
    left -= n;
    return r;
}

Proxy* P(u32 self) { return reinterpret_cast<Proxy*>(static_cast<uintptr_t>(self)); }
uint64_t H(u32 self) { return self != 0 ? reinterpret_cast<uintptr_t>(P(self)->host) : 0; }
template <typename T> T* G(u32 a) { return reinterpret_cast<T*>(static_cast<uintptr_t>(a)); }
void put32(u32 guestAddr, u32 v) { if (guestAddr != 0) std::memcpy(G<void>(guestAddr), &v, 4); }

u32 buildVtable(int kind);

u32 toGuest(ComObject* host) {
    if (host == nullptr) return 0;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (const auto it = g_proxies.find(host); it != g_proxies.end()) return static_cast<u32>(reinterpret_cast<uintptr_t>(it->second));
    auto* p = reinterpret_cast<Proxy*>(lowAlloc(sizeof(Proxy)));
    if (p == nullptr) return 0;
    std::memset(p, 0, sizeof(Proxy));
    const int kind = static_cast<int>(host->kind);
    if (kind < kComMMEnumerator || kind > kComMMEndpoint) return 0;
    if (g_vt[kind] == 0) g_vt[kind] = buildVtable(kind);
    p->vtbl = g_vt[kind];
    p->host = host;
    g_proxies[host] = p;
    return static_cast<u32>(reinterpret_cast<uintptr_t>(p));
}

u32 call(int iface, int slot, std::initializer_list<uint64_t> a) {
    ComArgs args{};
    size_t i = 0;
    for (uint64_t v : a) args.a[i++] = v;
    return static_cast<u32>(wasapiDispatch(iface, slot, args));
}

// Sin punteros de salida.
template <int I, int S> u32 G1(u32 t) { return call(I, S, {H(t)}); }
template <int I, int S> u32 G2(u32 t, u32 a) { return call(I, S, {H(t), a}); }
template <int I, int S> u32 G3(u32 t, u32 a, u32 b) { return call(I, S, {H(t), a, b}); }
template <int I, int S> u32 G5(u32 t, u32 a, u32 b, u32 c, u32 d) { return call(I, S, {H(t), a, b, c, d}); }

// Interfaz de salida en el argumento K (contando `this` como 0): se devuelve su proxy.
template <int I, int S, int K>
u32 O3(u32 t, u32 a, u32 b) {
    uint64_t tmp = 0;
    uint64_t v[3] = {H(t), a, b};
    const u32 guestOut = static_cast<u32>(v[K]);
    v[K] = reinterpret_cast<uintptr_t>(&tmp);
    const u32 r = call(I, S, {v[0], v[1], v[2]});
    put32(guestOut, toGuest(reinterpret_cast<ComObject*>(static_cast<uintptr_t>(tmp))));
    return r;
}
template <int I, int S, int K>
u32 O4(u32 t, u32 a, u32 b, u32 c) {
    uint64_t tmp = 0;
    uint64_t v[4] = {H(t), a, b, c};
    const u32 guestOut = static_cast<u32>(v[K]);
    v[K] = reinterpret_cast<uintptr_t>(&tmp);
    const u32 r = call(I, S, {v[0], v[1], v[2], v[3]});
    put32(guestOut, toGuest(reinterpret_cast<ComObject*>(static_cast<uintptr_t>(tmp))));
    return r;
}
template <int I, int S, int K>
u32 O5(u32 t, u32 a, u32 b, u32 c, u32 d) {
    uint64_t tmp = 0;
    uint64_t v[5] = {H(t), a, b, c, d};
    const u32 guestOut = static_cast<u32>(v[K]);
    v[K] = reinterpret_cast<uintptr_t>(&tmp);
    const u32 r = call(I, S, {v[0], v[1], v[2], v[3], v[4]});
    put32(guestOut, toGuest(reinterpret_cast<ComObject*>(static_cast<uintptr_t>(tmp))));
    return r;
}
// Dato de salida (CoTaskMem, ya < 4 GB en 32 bits) en el argumento K.
template <int I, int S, int K>
u32 D2(u32 t, u32 a) {
    uint64_t tmp = 0;
    uint64_t v[2] = {H(t), a};
    const u32 guestOut = static_cast<u32>(v[K]);
    v[K] = reinterpret_cast<uintptr_t>(&tmp);
    const u32 r = call(I, S, {v[0], v[1]});
    put32(guestOut, static_cast<u32>(tmp));
    return r;
}
template <int I, int S, int K>
u32 D4(u32 t, u32 a, u32 b, u32 c) {
    uint64_t tmp = 0;
    uint64_t v[4] = {H(t), a, b, c};
    const u32 guestOut = static_cast<u32>(v[K]);
    v[K] = reinterpret_cast<uintptr_t>(&tmp);
    const u32 r = call(I, S, {v[0], v[1], v[2], v[3]});
    put32(guestOut, static_cast<u32>(tmp));
    return r;
}

// IAudioClient::Initialize(share, flags, REFERENCE_TIME buffer, REFERENCE_TIME period, format, session)
u32 Initialize(u32 t, u32 share, u32 flags, i64 buf, i64 per, u32 fmt, u32 guid) {
    return call(kComAudioClient, 3, {H(t), share, flags, static_cast<uint64_t>(buf), static_cast<uint64_t>(per), fmt, guid});
}
// IPropertyStore::GetValue: el PROPVARIANT de 32 bits mide 16 bytes (valor en +8, puntero de 4).
u32 GetValue(u32 t, u32 key, u32 pv) {
    if (pv == 0) return kHrPointer;
    alignas(8) uint8_t tmp[24] = {};
    const u32 r = call(kComPropertyStore, 5, {H(t), key, reinterpret_cast<uintptr_t>(tmp)});
    uint8_t out[16] = {};
    std::memcpy(out, tmp, 8);
    uint64_t val = 0;
    std::memcpy(&val, tmp + 8, 8);
    const u32 low = static_cast<u32>(val);
    std::memcpy(out + 8, &low, 4);
    std::memcpy(G<void>(pv), out, 16);
    return r;
}
// IAudioRenderClient::GetBuffer / ReleaseBuffer con búfer en memoria baja.
u32 GetBuffer(u32 t, u32 frames, u32 ppData) {
    uint64_t tmp = 0;
    const u32 r = call(kComAudioRender, 3, {H(t), frames, reinterpret_cast<uintptr_t>(&tmp)});
    Proxy* p = P(t);
    if (r != 0 || tmp == 0) { put32(ppData, 0); return r; }
    WasapiClient* c = asWasapiClient(p->host);
    const u32 bytes = c != nullptr ? static_cast<u32>(frames * c->fmt.blockAlign) : 0;
    if (bytes > p->lowCap) {
        const u32 cap = std::max<u32>(bytes, 64 * 1024);
        void* nb = lowmem::mmapBelow4G((cap + 0xFFF) & ~0xFFFu);
        if (nb == MAP_FAILED) { put32(ppData, 0); return kHrOutOfMemory; }
        if (p->lowBuf != nullptr) munmap(p->lowBuf, (p->lowCap + 0xFFF) & ~0xFFFu);
        p->lowBuf = static_cast<uint8_t*>(nb);
        p->lowCap = cap;
    }
    p->hostStaging = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(tmp));
    p->stagingBytes = bytes;
    put32(ppData, static_cast<u32>(reinterpret_cast<uintptr_t>(p->lowBuf)));
    return r;
}
u32 ReleaseBuffer(u32 t, u32 frames, u32 flags) {
    Proxy* p = P(t);
    WasapiClient* c = asWasapiClient(p->host);
    if (p->hostStaging != nullptr && p->lowBuf != nullptr && c != nullptr) {
        const u32 bytes = std::min<u32>(static_cast<u32>(frames * c->fmt.blockAlign), p->stagingBytes);
        std::memcpy(p->hostStaging, p->lowBuf, bytes);
    }
    p->hostStaging = nullptr;
    return call(kComAudioRender, 4, {H(t), frames, flags});
}

template <int I>
void iunknown(std::vector<void*>& m) {
    m.push_back(w32arity::hpFn(&O3<I, 0, 2>));  // QueryInterface
    m.push_back(w32arity::hpFn(&G1<I, 1>));     // AddRef
    m.push_back(w32arity::hpFn(&G1<I, 2>));     // Release
}

u32 buildVtable(int kind) {
    std::vector<void*> m;
    switch (kind) {
        case kComMMEnumerator:
            iunknown<kComMMEnumerator>(m);
            m.push_back(w32arity::hpFn(&O4<kComMMEnumerator, 3, 3>));  // EnumAudioEndpoints
            m.push_back(w32arity::hpFn(&O4<kComMMEnumerator, 4, 3>));  // GetDefaultAudioEndpoint
            m.push_back(w32arity::hpFn(&O3<kComMMEnumerator, 5, 2>));  // GetDevice
            m.push_back(w32arity::hpFn(&G2<kComMMEnumerator, 6>));     // RegisterEndpointNotificationCallback
            m.push_back(w32arity::hpFn(&G2<kComMMEnumerator, 7>));     // Unregister...
            break;
        case kComMMCollection:
            iunknown<kComMMCollection>(m);
            m.push_back(w32arity::hpFn(&G2<kComMMCollection, 3>));     // GetCount
            m.push_back(w32arity::hpFn(&O3<kComMMCollection, 4, 2>));  // Item
            break;
        case kComMMDevice:
            iunknown<kComMMDevice>(m);
            m.push_back(w32arity::hpFn(&O5<kComMMDevice, 3, 4>));      // Activate
            m.push_back(w32arity::hpFn(&O3<kComMMDevice, 4, 2>));      // OpenPropertyStore
            m.push_back(w32arity::hpFn(&D2<kComMMDevice, 5, 1>));      // GetId
            m.push_back(w32arity::hpFn(&G2<kComMMDevice, 6>));         // GetState
            break;
        case kComPropertyStore:
            iunknown<kComPropertyStore>(m);
            m.push_back(w32arity::hpFn(&G2<kComPropertyStore, 3>));    // GetCount
            m.push_back(w32arity::hpFn(&G3<kComPropertyStore, 4>));    // GetAt
            m.push_back(w32arity::hpFn(&GetValue));                    // GetValue
            m.push_back(w32arity::hpFn(&G3<kComPropertyStore, 6>));    // SetValue
            m.push_back(w32arity::hpFn(&G1<kComPropertyStore, 7>));    // Commit
            break;
        case kComMMEndpoint:
            iunknown<kComMMEndpoint>(m);
            m.push_back(w32arity::hpFn(&G2<kComMMEndpoint, 3>));       // GetDataFlow
            break;
        case kComAudioClient:
            iunknown<kComAudioClient>(m);
            m.push_back(w32arity::hpFn(&Initialize));                  // 3
            m.push_back(w32arity::hpFn(&G2<kComAudioClient, 4>));      // GetBufferSize
            m.push_back(w32arity::hpFn(&G2<kComAudioClient, 5>));      // GetStreamLatency
            m.push_back(w32arity::hpFn(&G2<kComAudioClient, 6>));      // GetCurrentPadding
            m.push_back(w32arity::hpFn(&D4<kComAudioClient, 7, 3>));   // IsFormatSupported
            m.push_back(w32arity::hpFn(&D2<kComAudioClient, 8, 1>));   // GetMixFormat
            m.push_back(w32arity::hpFn(&G3<kComAudioClient, 9>));      // GetDevicePeriod
            m.push_back(w32arity::hpFn(&G1<kComAudioClient, 10>));     // Start
            m.push_back(w32arity::hpFn(&G1<kComAudioClient, 11>));     // Stop
            m.push_back(w32arity::hpFn(&G1<kComAudioClient, 12>));     // Reset
            m.push_back(w32arity::hpFn(&G2<kComAudioClient, 13>));     // SetEventHandle
            m.push_back(w32arity::hpFn(&O3<kComAudioClient, 14, 2>));  // GetService
            m.push_back(w32arity::hpFn(&G3<kComAudioClient, 15>));     // IsOffloadCapable
            m.push_back(w32arity::hpFn(&G2<kComAudioClient, 16>));     // SetClientProperties
            m.push_back(w32arity::hpFn(&G5<kComAudioClient, 17>));     // GetBufferSizeLimits
            break;
        case kComAudioRender:
            iunknown<kComAudioRender>(m);
            m.push_back(w32arity::hpFn(&GetBuffer));
            m.push_back(w32arity::hpFn(&ReleaseBuffer));
            break;
        case kComAudioClock:
            iunknown<kComAudioClock>(m);
            m.push_back(w32arity::hpFn(&G2<kComAudioClock, 3>));       // GetFrequency
            m.push_back(w32arity::hpFn(&G3<kComAudioClock, 4>));       // GetPosition
            m.push_back(w32arity::hpFn(&G2<kComAudioClock, 5>));       // GetCharacteristics
            break;
        case kComAudioVolume:
            iunknown<kComAudioVolume>(m);
            m.push_back(w32arity::hpFn(&G3<kComAudioVolume, 3>));      // SetMasterVolume(float, ctx)
            m.push_back(w32arity::hpFn(&G2<kComAudioVolume, 4>));      // GetMasterVolume
            m.push_back(w32arity::hpFn(&G3<kComAudioVolume, 5>));      // SetMute
            m.push_back(w32arity::hpFn(&G2<kComAudioVolume, 6>));      // GetMute
            break;
        case kComAudioSession:
            iunknown<kComAudioSession>(m);
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 3>));     // GetState
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 4>));     // GetDisplayName
            m.push_back(w32arity::hpFn(&G3<kComAudioSession, 5>));     // SetDisplayName
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 6>));     // GetIconPath
            m.push_back(w32arity::hpFn(&G3<kComAudioSession, 7>));     // SetIconPath
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 8>));     // GetGroupingParam
            m.push_back(w32arity::hpFn(&G3<kComAudioSession, 9>));     // SetGroupingParam
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 10>));    // RegisterAudioSessionNotification
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 11>));    // Unregister...
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 12>));    // GetSessionIdentifier
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 13>));    // GetSessionInstanceIdentifier
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 14>));    // GetProcessId
            m.push_back(w32arity::hpFn(&G1<kComAudioSession, 15>));    // IsSystemSoundsSession
            m.push_back(w32arity::hpFn(&G2<kComAudioSession, 16>));    // SetDuckingPreference
            break;
        default:
            return 0;
    }
    auto* vt = reinterpret_cast<u32*>(lowAlloc(4 * m.size()));
    if (vt == nullptr) return 0;
    for (size_t i = 0; i < m.size(); ++i) {
        vt[i] = static_cast<u32>(w32alias::lowFor(m[i], "mmdevapi.dll!WASAPI32_" + std::to_string(kind) + "#" + std::to_string(i)));
    }
    return static_cast<u32>(reinterpret_cast<uintptr_t>(vt));
}
}  // namespace wasapi32

uint32_t wasapi32ToGuest(void* host) { return wasapi32::toGuest(static_cast<ComObject*>(host)); }

uint64_t comDispatch(int iface, int slot, const ComArgs& args) {
    if (iface >= kComMMEnumerator && iface <= kComMMEndpoint) return wasapiDispatch(iface, slot, args);
    const uint64_t a0 = args.a[0], a1 = args.a[1], a2 = args.a[2], a3 = args.a[3];
    const uint64_t a4 = args.a[4], a5 = args.a[5], a6 = args.a[6], a7 = args.a[7];
    const uint64_t a8 = args.a[8], a9 = args.a[9];

    auto* self = reinterpret_cast<ComObject*>(static_cast<uintptr_t>(a0));
    if (self == nullptr) return hr(kHrPointer);

    if (slot == 0) return hr(comQueryInterface(self, readGuidData1(a1), a2));
    if (slot == 1) {
        return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    if (slot == 2) {
        uint32_t previous = self->refs.load(std::memory_order_relaxed);
        do {
            if (previous == 0) return 0;  // Release de más: no bajar de 0
        } while (!self->refs.compare_exchange_weak(previous, previous - 1, std::memory_order_acq_rel));
        if (previous == 1) onFinalRelease(self);
        return previous - 1;
    }
    // ID3D11DeviceChild / IDXGIObject: private data común a todos los hijos reales.
    if (self->heap) {
        if (slot == 3) {  // GetDevice
            s_comDevice.refs.fetch_add(1, std::memory_order_relaxed);
            return hr(storePtr(a1, &s_comDevice));
        }
        if (slot == 4) {  // GetPrivateData(guid, pDataSize, pData)
            storeU32(a2, 0);
            return hr(kDxgiErrorNotFound);
        }
        if (slot == 5 || slot == 6) return hr(kHrOk);  // SetPrivateData / SetPrivateDataInterface
    }

    switch (iface) {
        case kComFactory:
            switch (slot) {
                case 3:  // SetPrivateData
                case 4:  // SetPrivateDataInterface
                case 8:  // MakeWindowAssociation
                case 18: case 19: case 21: case 22:  // RegisterStereo/Occlusion
                    if (slot == 18 || slot == 19 || slot == 21 || slot == 22) {
                        storeU32(a3, 0);
                    }
                    return hr(kHrOk);
                case 5:  // GetPrivateData
                    return hr(kHrNotImpl);
                case 6:  // GetParent
                    storePtr(a2, nullptr);
                    return hr(kHrNoInterface);
                case 7:  // EnumAdapters
                case 12: // EnumAdapters1
                    if (a1 != 0) {
                        storePtr(a2, nullptr);
                        return hr(kDxgiErrorNotFound);
                    }
                    s_comAdapter.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comAdapter));
                case 9:  // GetWindowAssociation
                    return hr(storePtr(a1, nullptr));
                case 10: // CreateSwapChain(device, DXGI_SWAP_CHAIN_DESC*, ppSwapChain)
                    if (a3 == 0) return hr(kHrPointer);
                    if (a2 != 0) {
                        const auto* d = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a2));
                        configureSwapChain(d[0], d[1], d[4]);  // BufferDesc.Width/Height/Format
                    } else {
                        configureSwapChain(0, 0, 0);
                    }
                    s_comSwapChain.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a3, &s_comSwapChain));
                case 11: // CreateSoftwareAdapter
                    return hr(kHrNotImpl);
                case 13: // IsCurrent
                    return 1;
                case 14: // IsWindowedStereoEnabled
                    return 0;
                case 15:   // CreateSwapChainForHwnd(device, hwnd, DESC1*, fullscreen, restrict, pp)
                case 16: { // CreateSwapChainForCoreWindow(device, window, DESC1*, restrict, pp)
                    const uint64_t pp = slot == 15 ? a6 : a5;
                    if (pp == 0) return hr(kHrPointer);
                    if (a3 != 0) {
                        const auto* d = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a3));
                        configureSwapChain(d[0], d[1], d[2]);  // DESC1.Width/Height/Format
                    } else {
                        configureSwapChain(0, 0, 0);
                    }
                    s_comSwapChain.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(pp, &s_comSwapChain));
                }
                case 17: // GetSharedResourceAdapterLuid
                    zeroMemory(a2, 8);
                    return hr(kHrOk);
                case 20: // UnregisterStereoStatus
                case 23: // UnregisterOcclusionStatus
                    return hr(kHrOk);
                case 24: // CreateSwapChainForComposition(device, DESC1*, restrict, pp)
                    if (a4 == 0) return hr(kHrPointer);
                    if (a2 != 0) {
                        const auto* d = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a2));
                        configureSwapChain(d[0], d[1], d[2]);
                    } else {
                        configureSwapChain(0, 0, 0);
                    }
                    s_comSwapChain.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a4, &s_comSwapChain));
                default:
                    return hr(kHrNotImpl);
            }

        case kComAdapter:
            switch (slot) {
                case 3:
                case 4:
                    return hr(kHrOk);
                case 5:
                    return hr(kHrNotImpl);
                case 6:  // GetParent -> factory
                    s_comFactory.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comFactory));
                case 7:  // EnumOutputs
                    if (a1 != 0) {
                        storePtr(a2, nullptr);
                        return hr(kDxgiErrorNotFound);
                    }
                    s_comOutput.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comOutput));
                case 8:  // GetDesc
                    fillAdapterDesc(a1, false);
                    return hr(kHrOk);
                case 9:  // CheckInterfaceSupport
                    if (a2 != 0) *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a2)) = 0;
                    return hr(kHrOk);
                case 10: // GetDesc1
                    fillAdapterDesc(a1, true);
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }

        case kComOutput:
            switch (slot) {
                case 3:
                case 4:
                case 10:  // WaitForVBlank
                    return hr(kHrOk);
                case 5:
                    return hr(kHrNotImpl);
                case 6:
                    s_comAdapter.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comAdapter));
                case 7:  // GetDesc
                    fillOutputDesc(a1);
                    return hr(kHrOk);
                case 8: {  // GetDisplayModeList(format, flags, pNumModes, pModes)
                    // Las resoluciones del "monitor" (win32::displayModes), de menor a mayor: el
                    // menú de vídeo del juego puede bajar la resolución (se reescala a la nativa).
                    win32::DisplayModeSize modes[win32::kMaxDisplayModes];
                    const int n = win32::displayModes(modes);
                    uint32_t cap = static_cast<uint32_t>(n);
                    if (a3 != 0 && a4 != 0) cap = std::min<uint32_t>(cap, *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a3)));
                    if (a4 != 0) {
                        for (uint32_t i = 0; i < cap; ++i) {
                            fillModeDesc(a4 + i * 28, static_cast<uint32_t>(modes[i].w), static_cast<uint32_t>(modes[i].h),
                                         win32::refreshHz(), static_cast<uint32_t>(a1 != 0 ? a1 : 28));
                        }
                    }
                    if (a3 != 0) storeU32(a3, a4 != 0 ? cap : static_cast<uint32_t>(n));
                    if (a4 != 0) LOGI("DXGI: GetDisplayModeList -> %u modos (máx %dx%d)", cap, modes[n - 1].w, modes[n - 1].h);
                    return hr(a4 != 0 && cap < static_cast<uint32_t>(n) ? 0x887A0003u /*DXGI_ERROR_MORE_DATA*/ : kHrOk);
                }
                case 9:  // FindClosestMatchingMode(mode, closest, device)
                    if (a2 != 0) fillModeDesc(a2, win32::screenWidth(), win32::screenHeight(),
                                              win32::refreshHz(), 28);
                    return hr(kHrOk);
                default:
                    return hr(kHrNotImpl);
            }

        case kComSwapChain:
            switch (slot) {
                case 13:  // ResizeBuffers(count, width, height, format, flags)
                    configureSwapChain(static_cast<uint32_t>(a2), static_cast<uint32_t>(a3),
                                       static_cast<uint32_t>(a4));
                    return hr(kHrOk);
                case 3:
                case 4:
                case 10:  // SetFullscreenState
                case 14:  // ResizeTarget
                case 25:  // SetBackgroundColor
                case 27:  // SetRotation
                    if (slot == 10 && a2 != 0) {
                        *reinterpret_cast<int32_t*>(static_cast<uintptr_t>(a2)) = 0;
                    }
                    return hr(kHrOk);
                case 5:
                    return hr(kHrNotImpl);
                case 6:  // GetParent -> device
                    s_comDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comDevice));
                case 7:  // GetDevice
                    s_comDxgiDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comDxgiDevice));
                case 8:  // Present
                case 22: // Present1
                    return hr(presentSwapChain());
                case 9: {  // GetBuffer(index, riid, ppSurface)
                    if (a3 == 0) return hr(kHrPointer);
                    if (s_swapChain.backbuffer == nullptr) configureSwapChain(0, 0, 0);
                    std::lock_guard<std::mutex> lock(s_swapChain.mtx);
                    D3DResourceObj* bb = s_swapChain.backbuffer;
                    if (bb == nullptr) return hr(kHrOutOfMemory);
                    bb->com.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a3, bb));
                }
                case 11: // GetFullscreenState(pFullscreen, ppTarget)
                    if (a1 != 0) *reinterpret_cast<int32_t*>(static_cast<uintptr_t>(a1)) = 0;
                    if (a2 != 0) {
                        s_comOutput.refs.fetch_add(1, std::memory_order_relaxed);
                        storePtr(a2, &s_comOutput);
                    }
                    return hr(kHrOk);
                case 12: // GetDesc
                    fillSwapChainDesc(a1);
                    if (a1 != 0 && s_swapChain.width != 0) {
                        auto* d = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a1));
                        d[0] = s_swapChain.width;
                        d[1] = s_swapChain.height;
                        d[4] = s_swapChain.format;
                    }
                    return hr(kHrOk);
                case 15: // GetContainingOutput
                    s_comOutput.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comOutput));
                case 16: // GetFrameStatistics
                    zeroMemory(a1, 32);
                    return hr(kHrOk);
                case 17: // GetLastPresentCount
                    return hr(storeU32(a1, 0));
                case 18: // GetDesc1
                    fillSwapChainDesc1(a1);
                    if (a1 != 0 && s_swapChain.width != 0) {
                        auto* d = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a1));
                        d[0] = s_swapChain.width;
                        d[1] = s_swapChain.height;
                        d[2] = s_swapChain.format;
                    }
                    return hr(kHrOk);
                case 19: // GetFullscreenDesc
                    zeroMemory(a1, 24);
                    return hr(kHrOk);
                case 20: // GetHwnd
                    return hr(storePtr(a1, reinterpret_cast<void*>(0xCAFE0050)));
                case 21: // GetCoreWindow
                    storePtr(a2, nullptr);
                    return hr(kHrNotImpl);
                case 23: // IsTemporaryMonoSupported
                    return 0;
                case 24: // GetRestrictToOutput
                    s_comOutput.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comOutput));
                case 26: // GetBackgroundColor
                    zeroMemory(a1, 16);
                    return hr(kHrOk);
                case 28: // GetRotation
                    return hr(storeU32(a1, 0));
                default:
                    return hr(kHrNotImpl);
            }

        case kComDevice:
            switch (slot) {
                case 3:    // CreateBuffer(desc, initData, ppBuffer)
                case 4:    // CreateTexture1D(desc, initData, ppTexture)
                case 5:    // CreateTexture2D(desc, initData, ppTexture)
                case 6: {  // CreateTexture3D(desc, initData, ppTexture)
                    if (a1 == 0) return hr(0x80070057u);  // E_INVALIDARG
                    const auto* d = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a1));
                    const d3d11::ResourceDesc desc = slot == 3   ? d3d11::parseBufferDesc(d)
                                                     : slot == 4 ? d3d11::parseTexture1DDesc(d)
                                                     : slot == 5 ? d3d11::parseTexture2DDesc(d)
                                                                 : d3d11::parseTexture3DDesc(d);
                    if (a3 == 0) {  // validación sin crear: S_FALSE si el descriptor es válido
                        d3d11::Resource probe;
                        return hr(probe.init(desc) ? 1u : 0x80070057u);
                    }
                    D3DResourceObj* obj =
                        createResource(desc, reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a2)));
                    if (obj == nullptr) {
                        storePtr(a3, nullptr);
                        LOGW("D3D11 Create%s: descriptor inválido (%ux%ux%u fmt=%u)",
                             slot == 3 ? "Buffer" : "Texture", desc.width, desc.height, desc.depth, desc.format);
                        return hr(0x80070057u);
                    }
                    return hr(storePtr(a3, obj));
                }
                case 7:    // CreateShaderResourceView(resource, desc, ppv)
                case 8:    // CreateUnorderedAccessView(resource, desc, ppv)
                case 9:    // CreateRenderTargetView(resource, desc, ppv)
                case 10: { // CreateDepthStencilView(resource, desc, ppv)
                    D3DResourceObj* res = asResource(a1);
                    if (res == nullptr) return hr(0x80070057u);
                    if (a3 == 0) return hr(1u);
                    const uint32_t viewKind = slot == 9 ? kComRtv : slot == 10 ? kComDsv : kComSrv;
                    // D3D11_*_VIEW_DESC: SRV/DSV = 24 B, RTV/UAV = 20 B.
                    const size_t descSize = a2 == 0 ? 0 : (slot == 8 || slot == 9) ? 20 : 24;
                    D3DViewObj* view =
                        createView(viewKind, res, reinterpret_cast<const void*>(static_cast<uintptr_t>(a2)), descSize);
                    return hr(storePtr(a3, view));
                }
                case 11: { // CreateInputLayout(descs, num, bytecode, len, ppv)
                    if (a5 == 0) return hr(1u);
                    D3DDescObj* layout = createDescObject(kComGeneric, kDescOpaque, nullptr, 0);
                    // D3D11_INPUT_ELEMENT_DESC (32 B): LPCSTR name, UINT index, format, slot,
                    // alignedByteOffset, inputSlotClass, instanceDataStepRate.
                    const auto* e = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a1));
                    uint32_t autoOffset[16] = {};
                    for (uint32_t i = 0; e != nullptr && i < static_cast<uint32_t>(a2) && i < 32; ++i, e += 32) {
                        D3DDescObj::InputElement el;
                        const char* name = nullptr;
                        std::memcpy(&name, e, 8);
                        if (name != nullptr) el.semantic = name;
                        std::memcpy(&el.semanticIndex, e + 8, 4);
                        std::memcpy(&el.format, e + 12, 4);
                        std::memcpy(&el.slot, e + 16, 4);
                        std::memcpy(&el.offset, e + 20, 4);
                        std::memcpy(&el.perInstance, e + 24, 4);
                        std::memcpy(&el.stepRate, e + 28, 4);
                        const uint32_t s = el.slot & 15u;
                        if (el.offset == 0xFFFFFFFFu) el.offset = autoOffset[s];  // D3D11_APPEND_ALIGNED_ELEMENT
                        autoOffset[s] = el.offset + d3d11::formatInfo(el.format).bytes;
                        layout->inputElements.push_back(std::move(el));
                    }
                    return hr(storePtr(a5, layout));
                }
                case 12:   // CreateVertexShader(bytecode, len, classLinkage, ppShader)
                case 13:   // CreateGeometryShader
                case 15:   // CreatePixelShader
                case 16:   // CreateHullShader
                case 17:   // CreateDomainShader
                case 18: { // CreateComputeShader
                    if (a1 == 0 || a2 == 0) return hr(0x80070057u);
                    if (a4 == 0) return hr(1u);
                    const int iface2 = slot == 12 ? kComVertexShader : slot == 15 ? kComPixelShader : kComGeneric;
                    D3DDescObj* sh = createDescObject(iface2, kDescOpaque, nullptr, 0);
                    const auto* bc = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a1));
                    sh->bytecode.assign(bc, bc + static_cast<size_t>(a2));
                    dumpShaderBytecode(slot == 12 ? "vs" : slot == 15 ? "ps" : "xs", sh->bytecode);
                    return hr(storePtr(a4, sh));
                }
                case 14: // CreateGeometryShaderWithStreamOutput(..., ppGeometryShader = arg 9)
                    if (a9 == 0) return hr(1u);
                    return hr(storePtr(a9, createDescObject(kComGeneric, kDescOpaque, nullptr, 0)));
                case 19: // CreateClassLinkage(pp)
                    return hr(storePtr(a1, createDescObject(kComGeneric, kDescOpaque, nullptr, 0)));
                case 20:   // CreateBlendState(desc, ppState): D3D11_BLEND_DESC = 264 B
                case 21:   // CreateDepthStencilState: 52 B
                case 22:   // CreateRasterizerState: 40 B
                case 23:   // CreateSamplerState: 52 B
                case 24:   // CreateQuery(desc{Query, MiscFlags}, ppQuery)
                case 25: { // CreatePredicate
                    if (a1 == 0) return hr(0x80070057u);
                    if (a2 == 0) return hr(1u);
                    const size_t size = slot == 20 ? 264 : slot == 22 ? 40 : (slot == 24 || slot == 25) ? 8 : 52;
                    const int iface2 = slot == 23 ? kComSampler : kComGeneric;
                    const uint32_t kind = (slot == 24 || slot == 25) ? kDescQuery : kDescState;
                    return hr(storePtr(a2, createDescObject(iface2, kind,
                                                            reinterpret_cast<const void*>(static_cast<uintptr_t>(a1)), size)));
                }
                case 27: // CreateDeferredContext(flags, ppContext)
                    s_comContext.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comContext));
                case 28: // OpenSharedResource
                    storePtr(a3, nullptr);
                    return hr(kDxgiErrorUnsupported);
                case 29: { // CheckFormatSupport(format, pSupport)
                    const uint32_t support = formatSupport(static_cast<uint32_t>(a1));
                    storeU32(a2, support);
                    return hr(support != 0 ? kHrOk : 0x80004005u);  // E_FAIL si no se soporta
                }
                case 30: // CheckMultisampleQualityLevels(format, samples, pNum)
                    return hr(storeU32(a3, 1));
                case 31: // CheckCounterInfo(pInfo)
                    zeroMemory(a1, 24);
                    return hr(kHrOk);
                case 32: // CheckCounter(...): no exponemos contadores
                    return hr(kHrNotImpl);
                case 33: // CheckFeatureSupport(feature, data, size)
                    zeroMemory(a2, static_cast<size_t>(a3));
                    return hr(kHrOk);
                case 34: // GetPrivateData
                    return hr(kHrNotImpl);
                case 35:
                case 36:
                case 41: // SetExceptionMode
                    return hr(kHrOk);
                case 37: // GetFeatureLevel
                    return 0xB000;  // D3D_FEATURE_LEVEL_11_0
                case 38: // GetCreationFlags
                    return 0;
                case 39: // GetDeviceRemovedReason
                    return hr(kHrOk);
                case 40: // GetImmediateContext
                    s_comContext.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comContext));
                case 42: // GetExceptionMode
                    return hr(storeU32(a1, 0));
                default:
                    return hr(kHrNotImpl);
            }

        case kComContext:
            if (slot >= 0 && slot < 128) s_contextSlotCalls[slot].fetch_add(1, std::memory_order_relaxed);
            if (uint64_t stateRet = 0; contextStateSlot(slot, args, stateRet)) return stateRet;
            switch (slot) {
                case 3:  // GetDevice
                    s_comDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comDevice));
                case 4:  // GetPrivateData
                    return hr(kHrNotImpl);
                case 5:
                case 6:
                    return hr(kHrOk);
                case 14: { // Map(this=a0, resource=a1, subresource=a2, mapType=a3, flags=a4, pMapped=a5)
                    if (a5 == 0) return hr(kHrPointer);
                    D3DResourceObj* res = asResource(a1);
                    if (res == nullptr) return hr(0x80070057u);
                    std::lock_guard<std::mutex> lock(res->mtx);
                    const auto sub = static_cast<uint32_t>(a2);
                    const d3d11::SubresourceLayout* l = res->res.layout(sub);
                    uint8_t* data = res->res.data(sub);
                    if (l == nullptr || data == nullptr) return hr(0x80070057u);
                    auto* mapped = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(a5));
                    std::memcpy(mapped, &data, 8);                  // pData
                    std::memcpy(mapped + 8, &l->rowPitch, 4);       // RowPitch
                    std::memcpy(mapped + 12, &l->depthPitch, 4);    // DepthPitch
                    return hr(kHrOk);
                }
                case 15: // Unmap(resource, sub)
                    if (D3DResourceObj* res = asResource(a1); res != nullptr) {
                        noteGpuClobber("Unmap", res);
                        res->generation.fetch_add(1, std::memory_order_acq_rel);
                    }
                    return 0;
                case 29: { // GetData(async, pData, dataSize, flags)
                    D3DDescObj* q = asDescObject(a1);
                    if (a2 != 0 && a3 != 0) zeroMemory(a2, static_cast<size_t>(a3));
                    if (q == nullptr || a2 == 0) return hr(kHrOk);
                    uint32_t queryType = 0;
                    std::memcpy(&queryType, q->desc, 4);
                    auto* out = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(a2));
                    if (queryType == 0 && a3 >= 4) {            // EVENT: BOOL completado
                        const uint32_t one = 1;
                        std::memcpy(out, &one, 4);
                    } else if (queryType == 2 && a3 >= 8) {     // TIMESTAMP (ns)
                        struct timespec ts;
                        clock_gettime(CLOCK_MONOTONIC, &ts);
                        const uint64_t ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
                        std::memcpy(out, &ns, 8);
                    } else if (queryType == 3 && a3 >= 12) {    // TIMESTAMP_DISJOINT {Frequency, Disjoint}
                        const uint64_t freq = 1000000000ull;
                        std::memcpy(out, &freq, 8);
                    }
                    return hr(kHrOk);
                }
                case 46: { // CopySubresourceRegion(dst, dstSub, x, y, z, src, srcSub, pBox)
                    D3DResourceObj* dst = asResource(a1);
                    D3DResourceObj* src = asResource(a6);
                    if (dst == nullptr || src == nullptr) return 0;
                    const auto* box = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a8));
                    if (gpuPathActive() &&
                        gpu::Renderer::shared().copyTextureRegion(gpuResource(dst), static_cast<uint32_t>(a2),
                                                                  static_cast<uint32_t>(a3), static_cast<uint32_t>(a4),
                                                                  static_cast<uint32_t>(a5), gpuResource(src),
                                                                  static_cast<uint32_t>(a7), box)) {
                        return 0;  // copiado en la GPU: la copia de CPU no se toca
                    }
                    noteGpuClobber(gpuPathActive() && gpu::Renderer::shared().isGpuWritten(src) ? "CopySubresourceRegion(origen GPU)"
                                                                                                : "CopySubresourceRegion",
                                   dst);
                    dst->generation.fetch_add(1, std::memory_order_acq_rel);
                    if (dst == src) {
                        std::lock_guard<std::mutex> lock(dst->mtx);
                        dst->res.copyRegion(static_cast<uint32_t>(a2), static_cast<uint32_t>(a3), static_cast<uint32_t>(a4),
                                            static_cast<uint32_t>(a5), src->res, static_cast<uint32_t>(a7), box);
                    } else {
                        std::scoped_lock lock(dst->mtx, src->mtx);
                        dst->res.copyRegion(static_cast<uint32_t>(a2), static_cast<uint32_t>(a3), static_cast<uint32_t>(a4),
                                            static_cast<uint32_t>(a5), src->res, static_cast<uint32_t>(a7), box);
                    }
                    return 0;
                }
                case 47: { // CopyResource(dst, src)
                    D3DResourceObj* dst = asResource(a1);
                    D3DResourceObj* src = asResource(a2);
                    if (dst == nullptr || src == nullptr || dst == src) return 0;
                    // Ruta GPU: si el origen tiene contenido de la GPU (un RT), copiar en la GPU.
                    if (gpuPathActive() && gpu::Renderer::shared().copyTexture(gpuResource(dst), gpuResource(src))) {
                        return 0;
                    }
                    noteGpuClobber("CopyResource(CPU)", dst);
                    std::scoped_lock lock(dst->mtx, src->mtx);
                    dst->res.copyFrom(src->res);
                    dst->generation.fetch_add(1, std::memory_order_acq_rel);
                    return 0;
                }
                case 48: { // UpdateSubresource(dst, dstSub, pBox, pSrc, srcRowPitch, srcDepthPitch)
                    D3DResourceObj* dst = asResource(a1);
                    if (dst == nullptr || a4 == 0) return 0;
                    noteGpuClobber("UpdateSubresource", dst);
                    std::lock_guard<std::mutex> lock(dst->mtx);
                    dst->generation.fetch_add(1, std::memory_order_acq_rel);
                    dst->res.writeRegion(static_cast<uint32_t>(a2), reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(a3)),
                                         reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(a4)),
                                         static_cast<uint32_t>(a5), static_cast<uint32_t>(a6));
                    return 0;
                }
                case 50: { // ClearRenderTargetView(rtv, const FLOAT color[4])
                    D3DViewObj* rtv = asView(a1);
                    if (rtv == nullptr || rtv->resource == nullptr || a2 == 0) return 0;
                    float color[4];
                    std::memcpy(color, reinterpret_cast<const void*>(static_cast<uintptr_t>(a2)), sizeof(color));
                    if (gpuPathActive()) {
                        gpu::Renderer::shared().clearRenderTarget(gpuTexture(rtv, true), color);
                        return 0;
                    }
                    uint32_t mip = 0, first = 0, count = 0xFFFFFFFFu;
                    if (rtv->hasDesc) {
                        uint32_t dim = 0;
                        std::memcpy(&dim, rtv->desc + 4, 4);
                        std::memcpy(&mip, rtv->desc + 8, 4);
                        if (dim == 3 || dim == 5) {  // TEXTURE1DARRAY / TEXTURE2DARRAY
                            std::memcpy(&first, rtv->desc + 12, 4);
                            std::memcpy(&count, rtv->desc + 16, 4);
                        }
                    }
                    std::lock_guard<std::mutex> lock(rtv->resource->mtx);
                    rtv->resource->res.clear(mip, first, count, color);
                    rtv->resource->generation.fetch_add(1, std::memory_order_acq_rel);
                    return 0;
                }
                case 110: // ClearState
                case 111: // Flush
                    return hr(kHrOk);
                case 112: // GetType
                    return 0;  // D3D11_DEVICE_CONTEXT_IMMEDIATE
                case 113: // GetContextFlags
                    return hr(storeU32(a1, 0));
                case 114: // FinishCommandList
                    return hr(kHrNotImpl);
                default:
                    // Draw/Set/Get restantes: no hay backend de rasterización todavía.
                    // Los métodos void ignoran el retorno; los que tienen out pointers
                    // devuelven E_NOTIMPL en vez de S_OK con datos falsos.
                    return hr(kHrNotImpl);
            }

        case kComDxgiDevice:
            switch (slot) {
                case 3:
                case 4:
                case 10: // SetGPUThreadPriority
                    return hr(kHrOk);
                case 5:
                    return hr(kHrNotImpl);
                case 6:  // GetParent -> adapter
                    s_comAdapter.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a2, &s_comAdapter));
                case 7:  // GetAdapter
                    s_comAdapter.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comAdapter));
                case 8:  // CreateSurface
                    return hr(kDxgiErrorUnsupported);
                case 9:  // QueryResourceResidency(ppResources, count, pResidency)
                    zeroMemory(a3, static_cast<size_t>(a2) * sizeof(uint32_t));
                    return hr(kHrOk);
                case 11: // GetGPUThreadPriority
                    return hr(storeU32(a1, 0));
                default:
                    return hr(kHrNotImpl);
            }

        case kComTexture:
        case kComBuffer: {
            // ID3D11Resource: 7 GetType, 8 SetEvictionPriority, 9 GetEvictionPriority (UINT), 10 GetDesc.
            D3DResourceObj* res = asResource(a0);
            switch (slot) {
                case 7:  // void GetType(D3D11_RESOURCE_DIMENSION* out)
                    storeU32(a1, res != nullptr ? static_cast<uint32_t>(res->res.desc().dim) : (iface == kComBuffer ? 1u : 3u));
                    return 0;
                case 8:
                    return 0;
                case 9:
                    return 0;
                case 10:
                    if (a1 == 0) return 0;
                    if (res != nullptr) {
                        d3d11::writeDesc(res->res.desc(), reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a1)));
                    } else {
                        fillTexture2DDesc(a1, win32::screenWidth(), win32::screenHeight(), 28, 0x20);
                    }
                    return 0;
                default:
                    return hr(kHrNotImpl);
            }
        }

        case kComRtv:
        case kComDsv:
        case kComSrv: {
            D3DViewObj* view = asView(a0);
            switch (slot) {
                case 7:  // GetResource(ppResource)
                    if (view == nullptr || view->resource == nullptr) return hr(storePtr(a1, nullptr));
                    view->resource->com.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, view->resource));
                case 8:  // GetDesc
                    if (a1 == 0) return 0;
                    zeroMemory(a1, 24);
                    if (view != nullptr && view->hasDesc) {
                        std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a1)), view->desc,
                                    view->viewKind == kComDsv || view->viewKind == kComSrv ? 24 : 20);
                    } else if (view != nullptr && view->resource != nullptr) {
                        // Vista por defecto: formato del recurso y dimensión equivalente.
                        const d3d11::ResourceDesc& rd = view->resource->res.desc();
                        auto* out = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(a1));
                        out[0] = rd.format;
                        const bool arr = rd.arraySize > 1;
                        switch (rd.dim) {
                            case d3d11::Dimension::Buffer: out[1] = 1; break;
                            case d3d11::Dimension::Texture1D: out[1] = view->viewKind == kComDsv ? (arr ? 2 : 1) : (arr ? 3 : 2); break;
                            case d3d11::Dimension::Texture2D: out[1] = view->viewKind == kComDsv ? (arr ? 4 : 3) : (arr ? 5 : 4); break;
                            case d3d11::Dimension::Texture3D: out[1] = 8; break;
                            default: break;
                        }
                        if (view->viewKind == kComSrv) out[3] = rd.mipLevels;  // MostDetailedMip=0, MipLevels
                    }
                    return 0;
                default:
                    return hr(kHrNotImpl);
            }
        }

        case kComSampler:
        case kComGeneric:
        case kComVertexShader:
        case kComPixelShader:
        default: {
            D3DDescObj* obj = asDescObject(a0);
            if (obj == nullptr) {
                if (slot == 3) {  // singletons antiguos: ID3D11DeviceChild::GetDevice
                    s_comDevice.refs.fetch_add(1, std::memory_order_relaxed);
                    return hr(storePtr(a1, &s_comDevice));
                }
                return hr(kHrNotImpl);
            }
            if (obj->descKind == kDescState && slot == 7) {  // GetDesc (Blend/DS/RS/Sampler)
                if (a1 != 0) std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a1)), obj->desc, obj->descSize);
                return 0;
            }
            if (obj->descKind == kDescQuery) {
                uint32_t queryType = 0;
                std::memcpy(&queryType, obj->desc, 4);
                if (slot == 7) {  // GetDataSize
                    switch (queryType) {
                        case 0: return 4;    // EVENT (BOOL)
                        case 1: return 8;    // OCCLUSION (UINT64)
                        case 2: return 8;    // TIMESTAMP
                        case 3: return 16;   // TIMESTAMP_DISJOINT
                        case 4: return 88;   // PIPELINE_STATISTICS
                        default: return 8;
                    }
                }
                if (slot == 8) {  // GetDesc
                    if (a1 != 0) std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(a1)), obj->desc, 8);
                    return 0;
                }
            }
            return hr(kHrNotImpl);
        }
    }
}

void initComObject(ComObject& object, void** vtbl, uint32_t kind) {
    object.vtbl = vtbl;
    object.refs.store(1, std::memory_order_relaxed);
    object.kind = kind;
}

template <int Iface, size_t N>
void registerComVtable(const char* name) {
    for (size_t i = 0; i < N; ++i) {
        registerTrampolineAddress(ComVtable<Iface, N>::slots[i], name);
    }
}

void ensureComReady() {
    std::call_once(s_comOnce, []() {
        registerComVtable<kComFactory, 32>("DXGI!Factory2");
        registerComVtable<kComAdapter, 16>("DXGI!Adapter1");
        registerComVtable<kComOutput, 24>("DXGI!Output");
        registerComVtable<kComSwapChain, 32>("DXGI!SwapChain1");
        registerComVtable<kComDevice, 48>("D3D11!Device");
        registerComVtable<kComContext, 120>("D3D11!DeviceContext");
        registerComVtable<kComDxgiDevice, 16>("DXGI!Device");
        registerComVtable<kComTexture, 16>("D3D11!Texture2D");
        registerComVtable<kComRtv, 12>("D3D11!RenderTargetView");
        registerComVtable<kComDsv, 12>("D3D11!DepthStencilView");
        registerComVtable<kComSrv, 12>("D3D11!ShaderResourceView");
        registerComVtable<kComBuffer, 16>("D3D11!Buffer");
        registerComVtable<kComVertexShader, 8>("D3D11!VertexShader");
        registerComVtable<kComPixelShader, 8>("D3D11!PixelShader");
        registerComVtable<kComSampler, 12>("D3D11!SamplerState");
        registerComVtable<kComGeneric, kVtGeneric>("D3D11!Generic");
        registerComVtable<kComMMEnumerator, 8>("MMDevAPI!DeviceEnumerator");
        registerComVtable<kComMMDevice, 8>("MMDevAPI!Device");
        registerComVtable<kComMMCollection, 8>("MMDevAPI!DeviceCollection");
        registerComVtable<kComPropertyStore, 8>("MMDevAPI!PropertyStore");
        registerComVtable<kComAudioClient, 24>("WASAPI!AudioClient");
        registerComVtable<kComAudioRender, 8>("WASAPI!AudioRenderClient");
        registerComVtable<kComAudioClock, 8>("WASAPI!AudioClock");
        registerComVtable<kComAudioVolume, 8>("WASAPI!SimpleAudioVolume");
        registerComVtable<kComAudioSession, 16>("WASAPI!AudioSessionControl");
        initComObject(s_comMMEnum, ComVtable<kComMMEnumerator, 8>::slots.data(), kComMMEnumerator);
        initComObject(s_comMMDevice, ComVtable<kComMMDevice, 8>::slots.data(), kComMMDevice);
        initComObject(s_comMMCollection, ComVtable<kComMMCollection, 8>::slots.data(), kComMMCollection);
        initComObject(s_comPropStore, ComVtable<kComPropertyStore, 8>::slots.data(), kComPropertyStore);
        registerComVtable<kComMMEndpoint, 4>("MMDevAPI!Endpoint");
        initComObject(s_comMMEndpoint, ComVtable<kComMMEndpoint, 4>::slots.data(), kComMMEndpoint);

        initComObject(s_comFactory, ComVtable<kComFactory, 32>::slots.data(), kComFactory);
        initComObject(s_comAdapter, ComVtable<kComAdapter, 16>::slots.data(), kComAdapter);
        initComObject(s_comOutput, ComVtable<kComOutput, 24>::slots.data(), kComOutput);
        initComObject(s_comSwapChain, ComVtable<kComSwapChain, 32>::slots.data(), kComSwapChain);
        initComObject(s_comDevice, ComVtable<kComDevice, 48>::slots.data(), kComDevice);
        initComObject(s_comContext, ComVtable<kComContext, 120>::slots.data(), kComContext);
        initComObject(s_comDxgiDevice, ComVtable<kComDxgiDevice, 16>::slots.data(), kComDxgiDevice);
        initComObject(s_comTexture, ComVtable<kComTexture, 16>::slots.data(), kComTexture);
        initComObject(s_comRtv, ComVtable<kComRtv, 12>::slots.data(), kComRtv);
        initComObject(s_comDsv, ComVtable<kComDsv, 12>::slots.data(), kComDsv);
        initComObject(s_comSrv, ComVtable<kComSrv, 12>::slots.data(), kComSrv);
        initComObject(s_comBuffer, ComVtable<kComBuffer, 16>::slots.data(), kComBuffer);
        initComObject(s_comVertexShader, ComVtable<kComVertexShader, 8>::slots.data(), kComVertexShader);
        initComObject(s_comPixelShader, ComVtable<kComPixelShader, 8>::slots.data(), kComPixelShader);
        initComObject(s_comSampler, ComVtable<kComSampler, 12>::slots.data(), kComSampler);
    });
}

// CoCreateInstance(CLSID_MMDeviceEnumerator, IID_IMMDeviceEnumerator).
int32_t wasapiCreateEnumerator(uint32_t iidData1, void** ppv) {
    ensureComReady();
    if (iidData1 != 0xA95664D2u && iidData1 != 0x00000000u) return static_cast<int32_t>(kHrNoInterface);
    s_comMMEnum.refs.fetch_add(1, std::memory_order_relaxed);
    *ppv = &s_comMMEnum;
    LOGI("CoCreateInstance(MMDeviceEnumerator) -> %p (WASAPI -> AAudio)", *ppv);
    return 0;
}

}  // namespace

int32_t NativeDllInterceptor::win_CreateDXGIFactory(const void* riid, void** ppFactory) {
    (void)riid;
    s_d3dCalls.fetch_add(1);
    if (ppFactory == nullptr) return static_cast<int32_t>(kHrPointer);
    ensureComReady();
    directVulkanBridge().init();
    s_comFactory.refs.fetch_add(1, std::memory_order_relaxed);
    *ppFactory = &s_comFactory;
    LOGI("CreateDXGIFactory: IDXGIFactory2 real -> %p (2688x1216@120)", *ppFactory);
    return 0;
}

int32_t NativeDllInterceptor::win_CreateDXGIFactory1(const void* riid, void** ppFactory) {
    return win_CreateDXGIFactory(riid, ppFactory);
}

int32_t NativeDllInterceptor::win_CreateDXGIFactory2(uint32_t flags, const void* riid,
                                                     void** ppFactory) {
    (void)flags;
    return win_CreateDXGIFactory(riid, ppFactory);
}

int32_t NativeDllInterceptor::win_D3D11CreateDevice(
    void* pAdapter, int32_t driverType, void*, uint32_t flags, const void*, uint32_t featureLevels,
    uint32_t sdkVersion, void** ppDevice, void* pFeatureLevel, void** ppImmediateContext) {
    s_d3dCalls.fetch_add(1);
    ensureComReady();
    directVulkanBridge().init();
    if (pFeatureLevel != nullptr) {
        *reinterpret_cast<uint32_t*>(pFeatureLevel) = 0xB000;  // D3D_FEATURE_LEVEL_11_0
    }
    LOGI("D3D11CreateDevice(adapter=%p, driverType=%d, flags=0x%x, levels=%u, sdk=%u, "
         "ppDevice=%p, pFeatureLevel=%p, ppContext=%p)",
         pAdapter, driverType, flags, featureLevels, sdkVersion,
         static_cast<void*>(ppDevice), pFeatureLevel, static_cast<void*>(ppImmediateContext));
    // Sondeo de nivel de característica: sin ppDevice ni ppImmediateContext Windows
    // devuelve S_FALSE tras escribir *pFeatureLevel.
    if (ppDevice == nullptr && ppImmediateContext == nullptr) return 1;  // S_FALSE
    if (ppDevice != nullptr) {
        s_comDevice.refs.fetch_add(1, std::memory_order_relaxed);
        *ppDevice = &s_comDevice;
    }
    if (ppImmediateContext != nullptr) {
        s_comContext.refs.fetch_add(1, std::memory_order_relaxed);
        *ppImmediateContext = &s_comContext;
    }
    LOGI("D3D11CreateDevice: ID3D11Device real -> %p, contexto -> %p, nivel 0xB000",
         ppDevice != nullptr ? *ppDevice : nullptr,
         ppImmediateContext != nullptr ? *ppImmediateContext : nullptr);
    return 0;
}

int32_t NativeDllInterceptor::win_D3D11CreateDeviceAndSwapChain(
    void* pAdapter, int32_t driverType, void* software, uint32_t flags,
    const void* pFeatureLevels, uint32_t featureLevels, uint32_t sdkVersion,
    const void* pSwapChainDesc, void** ppSwapChain, void** ppDevice,
    void* pFeatureLevel, void** ppImmediateContext) {
    const int32_t result = win_D3D11CreateDevice(pAdapter, driverType, software, flags,
                                                 pFeatureLevels, featureLevels, sdkVersion,
                                                 ppDevice, pFeatureLevel, ppImmediateContext);
    if (result < 0) return result;
    if (ppSwapChain != nullptr) {
        ensureComReady();
        if (pSwapChainDesc != nullptr) {
            const auto* d = static_cast<const uint32_t*>(pSwapChainDesc);
            configureSwapChain(d[0], d[1], d[4]);  // DXGI_SWAP_CHAIN_DESC.BufferDesc
        } else {
            configureSwapChain(0, 0, 0);
        }
        s_comSwapChain.refs.fetch_add(1, std::memory_order_relaxed);
        *ppSwapChain = &s_comSwapChain;
    }
    LOGI("D3D11CreateDeviceAndSwapChain: dispositivo + swapchain reales (%s)",
         ppSwapChain != nullptr ? "present" : "sin swapchain");
    return result;
}

void* NativeDllInterceptor::win_Direct3DCreate9(uint32_t) {
    s_d3dCalls.fetch_add(1);
    LOGW("Direct3DCreate9: D3D9 no soportado en el backend directo; se devuelve NULL");
    return nullptr;
}

}  // namespace hyperplay
