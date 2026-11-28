// Pruebas del motor de traducción directo x86/x86_64 -> AArch64.
//
// Se compilan con el NDK para arm64-v8a y se ejecutan EN EL DISPOSITIVO (Snapdragon 8 Elite):
//   clang++ --target=aarch64-linux-android31 -std=c++20 -O2 -static \
//       app/src/test/cpp/OryonEmitterBackendTest.cpp \
//       app/src/main/cpp/cpu/OryonEmitterBackend.cpp \
//       app/src/main/cpp/cpu/HyperAotCache.cpp \
//       app/src/main/cpp/loader/PeHeaderInspector.cpp -o oryon_test
//   adb push oryon_test /data/local/tmp/ && adb shell chmod 755 /data/local/tmp/oryon_test \
//       && adb shell /data/local/tmp/oryon_test
//
// Cada caso alimenta bytes x86/x86_64 reales al backend, deja que los traduzca a opcodes AArch64,
// los ejecute y comprueba el resultado exacto (registros, memoria y trampolines Win32).

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <memory>

#include <pthread.h>

#include <atomic>
#include <csignal>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ucontext.h>

#include "cpu/OryonEmitterBackend.h"
#include "loader/NativeDllInterceptor.h"
#include "loader/Win32DisplayInfo.h"
#include "loader/Win32KnownFolders.h"
#include "loader/Win32PathUtils.h"
#include "loader/Win32Registry.h"
#include "loader/Win64ThunkAbi.h"
#include "loader/WinSafeHeap.h"
#include "loader/D3D11ResourceModel.h"
#include "loader/DxbcShader.h"
#include "loader/DxbcVm.h"
#include "loader/SoftRaster.h"
#include "loader/D3D11Sampler.h"

#include <dirent.h>
#include <limits>
#include <map>

// LowMem.h: la traza de mapeos vive en NativeDllInterceptor.cpp, que estas pruebas no enlazan.
namespace hyperplay::lowmem { void traceMap(void*, size_t, int, void*) {} }

// Estado observado por el VEH de prueba (testVectoredExceptionHandlerAccessViolation).
uintptr_t g_vehHandler = 0;
uint32_t g_vehCode = 0;
uint64_t g_vehAddress = 0;
uint64_t g_vehInfo0 = 0xFF;
uint64_t g_vehInfo1 = 0;
uint64_t g_vehCtxRip = 0;
uint64_t g_vehCtxRdx = 0xFF;
int g_vehCalls = 0;

namespace hyperplay {
#ifndef LINK_REAL_INTERCEPTOR
void NativeDllInterceptor::setGuestTeb(void* /*teb*/, bool /*is64*/) noexcept {}
bool NativeDllInterceptor::isRegisteredTrampoline(uintptr_t /*target*/) noexcept { return false; }

// Igual que el interceptor real: lee EXCEPTION_POINTERS y llama al VEH invitado.
int32_t NativeDllInterceptor::dispatchGuestException(OryonEmitterBackend& backend, uint64_t exceptionPointers) {
    ++g_vehCalls;
    const auto* ptrs = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(exceptionPointers));
    const auto* rec = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[0]));
    const auto* ctx = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[1]));
    std::memcpy(&g_vehCode, rec, 4);
    std::memcpy(&g_vehAddress, rec + 0x10, 8);
    std::memcpy(&g_vehInfo0, rec + 0x20, 8);
    std::memcpy(&g_vehInfo1, rec + 0x28, 8);
    std::memcpy(&g_vehCtxRip, ctx + 0xF8, 8);
    std::memcpy(&g_vehCtxRdx, ctx + 0x78 + 2 * 8, 8);
    if (g_vehHandler == 0) return 0;
    const uint64_t args[1] = {exceptionPointers};
    return static_cast<int32_t>(backend.callGuestNested(g_vehHandler, args, 1));
}
// SEH x86 vive en el interceptor real; aquí un fallo de 32 bits queda sin manejar.
bool NativeDllInterceptor::prepareGuestException32(CpuContext&, uint32_t, uint32_t, uint32_t) { return false; }
#endif
}

namespace {

using Bytes = std::vector<uint8_t>;
using hyperplay::CpuContext;
using hyperplay::EmitterConfig;
using hyperplay::NativeThunkHandler;
using hyperplay::OryonEmitterBackend;

int g_failures = 0;
int g_checks = 0;
int g_lastCheckLine = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++g_checks;                                                                  \
        g_lastCheckLine = __LINE__;                                                  \
        if (!(cond)) {                                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

#define CHECK_EQ(actual, expected)                                                           \
    do {                                                                                     \
        ++g_checks;                                                                          \
        g_lastCheckLine = __LINE__;                                                          \
        const uint64_t a_ = static_cast<uint64_t>(actual);                                   \
        const uint64_t e_ = static_cast<uint64_t>(expected);                                 \
        if (a_ != e_) {                                                                      \
            std::printf("FAIL %s:%d  %s: 0x%" PRIx64 " != 0x%" PRIx64 "\n", __FILE__,       \
                        __LINE__, #actual, a_, e_);                                          \
            ++g_failures;                                                                    \
        }                                                                                    \
    } while (0)

void append(Bytes& out, std::initializer_list<uint8_t> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void movRegImm64(Bytes& out, uint8_t regLow3, uint64_t value, bool rexW = true) {
    out.push_back(static_cast<uint8_t>(0x48 | (rexW ? 0x08 : 0x00) | (regLow3 >= 8 ? 0x01 : 0)));
    if (regLow3 >= 8) {
        out.push_back(static_cast<uint8_t>(0xB8 + (regLow3 - 8)));
    } else {
        out.push_back(static_cast<uint8_t>(0xB8 + regLow3));
    }
    // REX.B se emite en el prefijo; el opcode siempre es B8..BF con el índice bajo.
    out.resize(out.size() - 2);
    const uint8_t rex = static_cast<uint8_t>(0x48 | (rexW ? 0x08 : 0x00) | (regLow3 >= 8 ? 0x01 : 0x00));
    out.push_back(rex);
    out.push_back(static_cast<uint8_t>(0xB8 + (regLow3 & 7)));
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
}

void movRegImm32(Bytes& out, uint8_t regLow3, uint32_t value) {
    const uint8_t rex = static_cast<uint8_t>(0x40 | (regLow3 >= 8 ? 0x01 : 0x00));
    if (rex != 0x40) out.push_back(rex);
    out.push_back(static_cast<uint8_t>(0xB8 + (regLow3 & 7)));
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
}

uint8_t* mapCode(const Bytes& code, bool low32 = false) {
    const size_t size = 4096;
    void* hint = low32 ? reinterpret_cast<void*>(0x20000000) : nullptr;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (low32) {
#ifdef MAP_FIXED_NOREPLACE
        flags |= MAP_FIXED_NOREPLACE;
#else
        flags |= MAP_FIXED;
#endif
    }
    void* p = mmap(hint, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    // Si esa dirección ya está ocupada (otro programa de 32 bits anterior), siguiente hueco bajo.
    for (uintptr_t a = 0x30000000; p == MAP_FAILED && low32 && a < 0x40000000; a += 0x10000) {
        p = mmap(reinterpret_cast<void*>(a), size, PROT_READ | PROT_WRITE, flags, -1, 0);
    }
    if (p == MAP_FAILED) {
        std::perror("mmap code");
        return nullptr;
    }
    if (low32 && reinterpret_cast<uintptr_t>(p) >= 0xFFFFFFFFull) {
        std::printf("error: mmap no devolvio una direccion de 32 bits (%p)\n", p);
        munmap(p, size);
        return nullptr;
    }
    std::memcpy(p, code.data(), code.size());
    if (mprotect(p, size, PROT_READ | PROT_EXEC) != 0) {
        std::perror("mprotect code");
    }
    return static_cast<uint8_t*>(p);
}

void* mapData(size_t size = 4096) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::perror("mmap data");
    return p;
}

uintptr_t g_fakeThunk = 0x0000DEADBEEF0000ull;
uint64_t g_thunkResult = 0xCAFE;
int g_thunkCalls = 0;
uint64_t g_thunkSeenArg = 0;

uintptr_t g_thunk8Target = 0x0000BEEF88880000ull;
uint64_t g_thunk8Result = 0xCAFEBABE12345678ull;
uint64_t g_seenArgs8[8] = {0};
int g_thunk8Calls = 0;

uint64_t hostThunk8Target(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                          uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7) {
    g_seenArgs8[0] = a0;
    g_seenArgs8[1] = a1;
    g_seenArgs8[2] = a2;
    g_seenArgs8[3] = a3;
    g_seenArgs8[4] = a4;
    g_seenArgs8[5] = a5;
    g_seenArgs8[6] = a6;
    g_seenArgs8[7] = a7;
    ++g_thunk8Calls;
    return g_thunk8Result;
}

uintptr_t g_thunk12Target = 0x0000BEEF12120000ull;
uint64_t g_thunk12Result = 0x1212121234343434ull;
uint64_t g_seenArgs12[12] = {0};
int g_thunk12Calls = 0;

uint64_t hostThunk12Target(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                           uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
                           uint64_t a8, uint64_t a9, uint64_t a10, uint64_t a11) {
    const uint64_t v[12] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
    std::memcpy(g_seenArgs12, v, sizeof(v));
    ++g_thunk12Calls;
    return g_thunk12Result;
}

bool fakeThunk(void* user, uintptr_t target, CpuContext& ctx) {
    if (target == g_thunk8Target || target == g_thunk12Target) {
        // Misma ruta que NativeDllInterceptor::dispatchThunk (Win64ThunkAbi.h).
        const uintptr_t oldRsp = ctx.gpr[4];
        uint64_t retAddr = 0;
        std::memcpy(&retAddr, reinterpret_cast<const void*>(oldRsp), 8);
        ctx.gpr[4] += 8;
        const auto args = hyperplay::win64abi::readThunkArgs(
            ctx, static_cast<const OryonEmitterBackend*>(user));
        const uintptr_t host = target == g_thunk8Target
                                   ? reinterpret_cast<uintptr_t>(&hostThunk8Target)
                                   : reinterpret_cast<uintptr_t>(&hostThunk12Target);
        ctx.gpr[0] = hyperplay::win64abi::invokeThunk(host, args);
        ctx.rip = retAddr;
        return true;
    }
    if (target != g_fakeThunk) return false;
    static_cast<void>(user);
    ++g_thunkCalls;
    const int stackSize = ctx.mode32 != 0 ? 4 : 8;
    uint64_t ret = 0;
    if (stackSize == 8) {
        std::memcpy(&ret, reinterpret_cast<const void*>(ctx.gpr[4]), 8);
        ctx.gpr[4] += 8;
        g_thunkSeenArg = ctx.gpr[1];  // RCX
    } else {
        uint32_t ret32 = 0;
        std::memcpy(&ret32, reinterpret_cast<const void*>(ctx.gpr[4]), 4);
        ctx.gpr[4] += 4;
        ret = ret32;
        uint32_t arg = 0;
        std::memcpy(&arg, reinterpret_cast<const void*>(ctx.gpr[4]), 4);
        g_thunkSeenArg = arg;
    }
    ctx.gpr[0] = g_thunkResult;
    ctx.rip = ret;
    return true;
}

OryonEmitterBackend* g_backend = nullptr;

bool configureBackend(bool is32Bit) {
    EmitterConfig config;
    config.is32Bit = is32Bit;
    config.stackSize = 8u * 1024u * 1024u;
    config.codeCacheSize = 32u * 1024u * 1024u;
    config.persistBlocks = false;
    if (!g_backend->configure(config)) return false;
    g_backend->setNativeThunkHandler(&fakeThunk, nullptr);
    return true;
}

int64_t runCode(const Bytes& code, bool low32 = false) {
    auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
    if (stack != nullptr) {
        constexpr size_t kStackSize = 8u * 1024u * 1024u;
        std::memset(stack + kStackSize - 256, 0, 256);
        g_backend->context().gpr[4] =
            reinterpret_cast<uintptr_t>(stack + kStackSize - 64);
    }
    uint8_t* entry = mapCode(code, low32);
    if (entry == nullptr) return -100;
    return g_backend->executeEntry(entry);
}

// --- Casos -----------------------------------------------------------------------------------

void testMovAdd64() {
    Bytes code;
    movRegImm64(code, 0, 12);
    append(code, {0x48, 0x83, 0xC0, 0x1E});  // add rax, 30
    append(code, {0xC3});
    CHECK(configureBackend(false));
    CHECK_EQ(runCode(code), 42);
    CHECK_EQ(g_backend->context().gpr[0], 42);
}

void test32BitWrapAndZeroFlag() {
    Bytes code;
    movRegImm32(code, 0, 0xFFFFFFFFu);
    append(code, {0x83, 0xC0, 0x01});  // add eax, 1
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0);
    CHECK_EQ(g_backend->context().gpr[0], 0);
    // x86 ZF=1 (bit 6 de rflags), CF=1 (acarreo de 0xFFFFFFFF + 1), PF=1 (byte bajo 0, paridad par)
    CHECK((g_backend->context().rflags & (1ull << 6)) != 0);
    CHECK((g_backend->context().rflags & 1ull) != 0);
    CHECK(hyperplay::pfFromByte(g_backend->context().pfByte));
}

void test8BitWrapAndFlags() {
    Bytes code;
    movRegImm32(code, 0, 0xFFu);
    append(code, {0x04, 0x01});  // add al, 1 -> AL=0, ZF=1, CF=1
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0);
    CHECK_EQ(g_backend->context().gpr[0] & 0xFF, 0);
    CHECK((g_backend->context().rflags & (1ull << 6)) != 0); // ZF=1
    CHECK((g_backend->context().rflags & (1ull << 0)) != 0); // CF=1

    // Test loop with addb $1, %r9b; jne (loops exactly 256 times)
    Bytes loopCode;
    movRegImm64(loopCode, 0, 0);
    append(loopCode, {0x45, 0x32, 0xC9});       // xorb r9b, r9b
    append(loopCode, {0x48, 0xFF, 0xC0});       // inc rax
    append(loopCode, {0x41, 0x80, 0xC1, 0x01}); // addb $1, r9b
    append(loopCode, {0x75, 0xF7});             // jne -9 (to inc rax)
    append(loopCode, {0xC3});
    CHECK_EQ(runCode(loopCode), 256);
}

void testConditionalJump() {
    Bytes code;
    movRegImm32(code, 0, 10);
    append(code, {0x83, 0xF8, 0x0A});  // cmp eax, 10
    append(code, {0x74, 0x05});        // je +5
    movRegImm32(code, 0, 99);
    movRegImm32(code, 0, 42);
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 42);
}

void testLoop() {
    Bytes code;
    movRegImm32(code, 0, 0);     // mov eax, 0
    movRegImm32(code, 1, 100);   // mov ecx, 100
    append(code, {0x83, 0xC0, 0x03});  // add eax, 3
    append(code, {0xFF, 0xC9});        // dec ecx
    append(code, {0x75, 0xF9});        // jnz -7
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 300);
}

void testPushPop() {
    Bytes code;
    movRegImm64(code, 0, 0x1122334455667788ull);
    append(code, {0x50});              // push rax
    append(code, {0x5B});              // pop rbx
    append(code, {0x48, 0x89, 0xD8});  // mov rax, rbx
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0x1122334455667788ull);
}

void testCallRet() {
    Bytes code;
    // call +1 (a la direccion 6); ret; mov eax,42; ret
    append(code, {0xE8, 0x01, 0x00, 0x00, 0x00});
    append(code, {0xC3});
    movRegImm32(code, 0, 42);
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 42);
}

void testMemoryAndSib() {
    void* data = mapData();
    auto* bytes = static_cast<uint8_t*>(data);
    std::memset(bytes, 0, 64);
    Bytes code;
    movRegImm64(code, 3, reinterpret_cast<uintptr_t>(bytes));  // rbx
    movRegImm64(code, 0, 0x1122334455667788ull);
    append(code, {0x48, 0x89, 0x43, 0x08});        // mov [rbx+8], rax
    append(code, {0x48, 0x8B, 0x43, 0x08});        // mov rax, [rbx+8]
    movRegImm32(code, 1, 0);                       // rcx = 0 (índice SIB)
    append(code, {0x48, 0x89, 0x04, 0x8B});        // mov [rbx+rcx*4], rax (rcx=0)
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0x1122334455667788ull);
    uint64_t stored = 0;
    std::memcpy(&stored, bytes + 8, 8);
    CHECK_EQ(stored, 0x1122334455667788ull);
    std::memcpy(&stored, bytes, 8);
    CHECK_EQ(stored, 0x1122334455667788ull);
}

void testLods() {
    uint8_t* data = static_cast<uint8_t*>(mapData());
    const uint64_t qword = 0x0123456789ABCDEFull;
    std::memcpy(data, &qword, 8);

    // LODSQ: RAX = [RSI], RSI += 8.
    Bytes q;
    movRegImm64(q, 6, reinterpret_cast<uintptr_t>(data));  // RSI
    append(q, {0x48, 0xAD});                               // lodsq
    append(q, {0xC3});
    CHECK(configureBackend(false));
    CHECK_EQ(runCode(q), qword);
    CHECK_EQ(g_backend->context().gpr[6], reinterpret_cast<uintptr_t>(data) + 8);

    // LODSD zero-extiende EAX a RAX (semántica AMD64) y avanza RSI 4 bytes.
    const uint32_t dword = 0xA1B2C3D4u;
    std::memcpy(data, &dword, 4);
    Bytes d;
    movRegImm64(d, 0, 0x1122334455667788ull);
    movRegImm64(d, 6, reinterpret_cast<uintptr_t>(data));
    append(d, {0xAD});  // lodsd
    append(d, {0xC3});
    CHECK_EQ(runCode(d), dword);
    CHECK_EQ(g_backend->context().gpr[6], reinterpret_cast<uintptr_t>(data) + 4);

    // LODSB preserva los bits altos de RAX (solo escribe AL) y avanza RSI 1 byte.
    data[0] = 0x5A;
    Bytes b;
    movRegImm64(b, 0, 0x1122334455667700ull);
    movRegImm64(b, 6, reinterpret_cast<uintptr_t>(data));
    append(b, {0xAC});  // lodsb
    append(b, {0xC3});
    CHECK_EQ(runCode(b), 0x112233445566775Aull);
    CHECK_EQ(g_backend->context().gpr[6], reinterpret_cast<uintptr_t>(data) + 1);

    // REP LODSB: RCX iteraciones, deja el último byte en AL, RCX=0 y RSI avanzado.
    data[0] = 0x11;
    data[1] = 0x22;
    data[2] = 0x33;
    Bytes r;
    movRegImm64(r, 0, 0);  // RAX
    movRegImm64(r, 6, reinterpret_cast<uintptr_t>(data));
    movRegImm64(r, 1, 3);  // RCX
    append(r, {0xF3, 0xAC});  // rep lodsb
    append(r, {0xC3});
    CHECK_EQ(runCode(r), 0x33);
    CHECK_EQ(g_backend->context().gpr[1], 0);
    CHECK_EQ(g_backend->context().gpr[6], reinterpret_cast<uintptr_t>(data) + 3);
}

void testDynamicExecutableRangeInvalidation() {
    CHECK(configureBackend(false));
    auto* page = static_cast<uint8_t*>(mapData());
    if (page == nullptr) {
        CHECK(false);
        return;
    }

    // Código x86 real escrito por el invitado en memoria RW (como haría un JIT interno).
    page[0] = 0xB8;  // mov eax, imm32
    page[1] = 1; page[2] = 0; page[3] = 0; page[4] = 0;
    page[5] = 0xC3;  // ret
    const uintptr_t base = reinterpret_cast<uintptr_t>(page);
    g_backend->addExecutableRange(base, 4096);
    CHECK_EQ(g_backend->executeEntry(page), 1);

    // El invitado sobrescribe la misma página y llama a VirtualProtect(PAGE_EXECUTE_READ).
    page[1] = 2;
    g_backend->invalidateCodeRange(base, 4096);
    CHECK_EQ(g_backend->executeEntry(page), 2);

    // Tras VirtualFree, el rango deja de ser traducible.
    g_backend->removeExecutableRange(base, 4096);
    page[0] = 0xB8; page[1] = 3; page[5] = 0xC3;
    CHECK_EQ(g_backend->executeEntry(page), -2);

    // Restaura el modo sin rango forzado para el resto de la suite.
    g_backend->setCodeRange(0, 0);
}

void testWin32DisplayAndKnownFolders() {
    using namespace hyperplay::win32;

    CHECK_EQ(systemMetric(0), screenWidth());   // SM_CXSCREEN
    CHECK_EQ(systemMetric(1), screenHeight());   // SM_CYSCREEN
    CHECK_EQ(systemMetric(78), screenWidth());  // SM_CXVIRTUALSCREEN
    CHECK_EQ(systemMetric(79), screenHeight());  // SM_CYVIRTUALSCREEN
    CHECK_EQ(systemMetric(80), 1);     // SM_CMONITORS
    CHECK_EQ(systemMetric(19), 1);     // SM_MOUSEPRESENT
    CHECK_EQ(systemMetric(76), 0);     // SM_XVIRTUALSCREEN (origen)
    CHECK_EQ(systemMetric(77), 0);     // SM_YVIRTUALSCREEN
    CHECK_EQ(systemMetric(61), screenWidth());  // SM_CXMAXIMIZED
    CHECK_EQ(systemMetric(62), screenHeight());  // SM_CYMAXIMIZED
    CHECK_EQ(systemMetric(16), screenWidth());  // SM_CXFULLSCREEN
    {
        alignas(8) uint8_t mon[840] = {};
        fillMonitorDeviceW(mon);
        CHECK_EQ(*reinterpret_cast<uint32_t*>(mon + 324), 0x3u);  // ACTIVE | ATTACHED
        CHECK_EQ(reinterpret_cast<char16_t*>(mon + 4)[12], u'\\');  // "\\.\DISPLAY1\Monitor0"
    }
    CHECK_EQ(deviceCaps(8), screenWidth());     // HORZRES
    CHECK_EQ(deviceCaps(10), screenHeight());    // VERTRES
    CHECK_EQ(deviceCaps(12), 32);      // BITSPIXEL
    CHECK_EQ(deviceCaps(116), 120);    // VREFRESH

    uint8_t devMode[220];
    std::memset(devMode, 0xCC, sizeof(devMode));
    // Lista de modos: de menor a mayor y la última es la nativa; ENUM_CURRENT_SETTINGS = nativa.
    DisplayModeSize modes[kMaxDisplayModes];
    const int nModes = displayModes(modes);
    CHECK(nModes >= 2);
    CHECK_EQ(modes[nModes - 1].w, screenWidth());
    CHECK_EQ(modes[nModes - 1].h, screenHeight());
    for (int i = 1; i < nModes; ++i) CHECK(modes[i - 1].w <= modes[i].w);
    CHECK(!fillDevMode(devMode, static_cast<uint32_t>(nModes)));  // fin de la lista
    CHECK(fillDevMode(devMode, 0xFFFFFFFFu));
    CHECK_EQ(*reinterpret_cast<uint16_t*>(devMode + 68), 220);      // dmSize
    CHECK_EQ(*reinterpret_cast<uint32_t*>(devMode + 72), 0x005C0000u);  // dmFields (+DM_DISPLAYFREQUENCY)
    CHECK_EQ(*reinterpret_cast<uint32_t*>(devMode + 168), 32);      // dmBitsPerPel
    CHECK_EQ(*reinterpret_cast<uint32_t*>(devMode + 172), screenWidth());    // dmPelsWidth
    CHECK_EQ(*reinterpret_cast<uint32_t*>(devMode + 176), screenHeight());    // dmPelsHeight
    CHECK_EQ(*reinterpret_cast<uint32_t*>(devMode + 184), 120);     // dmDisplayFrequency

    uint8_t displayDevice[840];
    std::memset(displayDevice, 0xCC, sizeof(displayDevice));
    fillDisplayDeviceW(displayDevice);
    CHECK_EQ(*reinterpret_cast<uint32_t*>(displayDevice), 840);  // cb
    const auto* deviceName = reinterpret_cast<const char16_t*>(displayDevice + 4);
    CHECK(deviceName[0] == u'\\' && deviceName[1] == u'\\' && deviceName[2] == u'.' &&
          deviceName[3] == u'\\' && deviceName[4] == u'D');
    CHECK_EQ(*reinterpret_cast<uint32_t*>(displayDevice + 324), 0x5u);  // ACTIVE|PRIMARY
    const auto* deviceString = reinterpret_cast<const char16_t*>(displayDevice + 68);
    CHECK(deviceString[0] == u'A' && deviceString[1] == u'd' && deviceString[2] == u'r');

    uint8_t monitorInfo[104];
    std::memset(monitorInfo, 0, sizeof(monitorInfo));
    *reinterpret_cast<uint32_t*>(monitorInfo) = 104;  // cbSize
    fillMonitorInfo(monitorInfo);
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 4), 0);
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 8), 0);
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 12), screenWidth());
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 16), screenHeight());
    for (int32_t i = 0; i < 4; ++i) {
        std::printf("  rcMonitor[%d]=%d rcWork[%d]=%d\n", i,
                    *reinterpret_cast<int32_t*>(monitorInfo + 4 + i * 4), i,
                    *reinterpret_cast<int32_t*>(monitorInfo + 20 + i * 4));
    }
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 20), 0);
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 24), 0);
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 28), screenWidth());
    CHECK_EQ(*reinterpret_cast<int32_t*>(monitorInfo + 32), screenHeight());
    CHECK_EQ(*reinterpret_cast<uint32_t*>(monitorInfo + 36), 1);
    const auto* monitorName = reinterpret_cast<const char16_t*>(monitorInfo + 40);
    CHECK(monitorName[0] == u'\\' && monitorName[4] == u'D');

    const uint8_t profileGuid[16] = {0x8F, 0x85, 0x6C, 0x5E, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t localAppDataGuid[16] = {0x85, 0x27, 0xB3, 0xF1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t localLowGuid[16] = {0xA4, 0xA1, 0x20, 0xA5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(knownFolderFromGuid(profileGuid) == "C:\\users\\default");
    CHECK(knownFolderFromGuid(localAppDataGuid) == "C:\\users\\default\\AppData\\Local");
    CHECK(knownFolderFromGuid(localLowGuid) == "C:\\users\\default\\AppData\\LocalLow");
    CHECK(knownFolderFromCsidl(0x001C) == "C:\\users\\default\\AppData\\Local");
    CHECK(knownFolderFromCsidl(0x0028) == "C:\\users\\default");
}

void testLea() {
    void* data = mapData();
    Bytes code;
    movRegImm64(code, 3, reinterpret_cast<uintptr_t>(data));
    append(code, {0x48, 0x8D, 0x43, 0x10});  // lea rax, [rbx+16]
    append(code, {0xC3});
    CHECK_EQ(runCode(code), reinterpret_cast<uintptr_t>(data) + 16);
}

void testMovzxMovsx() {
    void* data = mapData();
    auto* bytes = static_cast<uint8_t*>(data);
    bytes[0] = 0x80;
    {
        Bytes code;
        movRegImm64(code, 3, reinterpret_cast<uintptr_t>(bytes));
        append(code, {0x0F, 0xB6, 0x03});  // movzx eax, byte [rbx]
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0x80);
    }
    {
        Bytes code;
        movRegImm64(code, 3, reinterpret_cast<uintptr_t>(bytes));
        append(code, {0x48, 0x0F, 0xBE, 0x03});  // movsx rax, byte [rbx]
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0xFFFFFFFFFFFFFF80ull);
    }
}

void testShifts() {
    {
        Bytes code;
        movRegImm32(code, 0, 1);
        append(code, {0xC1, 0xE0, 0x04});  // shl eax, 4
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 16);
    }
    {
        Bytes code;
        movRegImm32(code, 0, 0xFFFFFFF8u);  // -8
        append(code, {0xC1, 0xF8, 0x02});   // sar eax, 2
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0xFFFFFFFEull);
    }
    {
        Bytes code;
        movRegImm32(code, 0, 0x80000000u);
        append(code, {0xC1, 0xE8, 0x04});  // shr eax, 4
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0x08000000u);
    }
}

void testImul() {
    {
        Bytes code;
        movRegImm32(code, 0, 6);
        append(code, {0x6B, 0xC0, 0x07});  // imul eax, eax, 7
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 42);
    }
    {
        Bytes code;
        movRegImm64(code, 0, 6);
        append(code, {0x48, 0x6B, 0xC0, 0x07});  // imul rax, rax, 7
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 42);
    }
}

void testXchgMemory() {
    void* data = mapData();
    auto* bytes = static_cast<uint8_t*>(data);
    const uint32_t initial = 0x11223344;
    std::memcpy(bytes, &initial, 4);
    Bytes code;
    movRegImm64(code, 3, reinterpret_cast<uintptr_t>(bytes));
    movRegImm32(code, 0, 1);
    append(code, {0x87, 0x03});  // xchg [rbx], eax
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0x11223344);
    uint32_t stored = 0;
    std::memcpy(&stored, bytes, 4);
    CHECK_EQ(stored, 1);
}

void testSse() {
    float a[4] = {1.5f, 2.0f, 3.0f, 4.0f};
    float b[4] = {0.5f, 1.0f, 2.0f, 3.0f};
    auto* out = static_cast<float*>(mapData());
    Bytes code;
    movRegImm64(code, 0, reinterpret_cast<uintptr_t>(a));
    append(code, {0x0F, 0x28, 0x00});  // movaps xmm0, [rax]
    movRegImm64(code, 0, reinterpret_cast<uintptr_t>(b));
    append(code, {0x0F, 0x28, 0x08});  // movaps xmm1, [rax]
    append(code, {0x0F, 0x58, 0xC1});  // addps xmm0, xmm1
    append(code, {0x0F, 0x59, 0xC1});  // mulps xmm0, xmm1
    movRegImm64(code, 0, reinterpret_cast<uintptr_t>(out));
    append(code, {0x0F, 0x11, 0x00});  // movups [rax], xmm0
    append(code, {0x31, 0xC0});        // xor eax, eax (resultado verificable)
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0);
    CHECK(out[0] > 0.99f && out[0] < 1.01f);
    CHECK(out[1] > 2.99f && out[1] < 3.01f);
    CHECK(out[2] > 9.99f && out[2] < 10.01f);
    CHECK(out[3] > 20.99f && out[3] < 21.01f);
}

// SSE escalar: resultado del lane 0 y preservación de los lanes altos (Mono guarda Vector2/3 de
// Unity en XMM; un escalar que pisa los lanes altos rompe bounds/viewports).
namespace {
struct SseCase {
    const char* name;
    std::vector<uint8_t> op;   // instrucción a probar (xmm0 = destino, xmm1/ecx/rcx = fuente)
    float a[4], b[4];
    int64_t rcx;
    uint32_t expect[4];        // bits esperados de xmm0 tras la instrucción
    bool checkRax;
    int64_t expectRax;
};

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
}  // namespace

void testSseScalarLanes() {
    const float S1 = 111.0f, S2 = 222.0f, S3 = 333.0f;  // centinelas de lanes altos
    double dpair[2];
    std::vector<SseCase> cases;
    auto f4 = [](float x, float y, float z, float w) { return std::array<float, 4>{x, y, z, w}; };
    auto add = [&](const char* n, std::vector<uint8_t> op, std::array<float, 4> a, std::array<float, 4> b,
                   int64_t rcx, std::array<uint32_t, 4> ex, bool chk = false, int64_t rax = 0) {
        SseCase c{n, std::move(op), {}, {}, rcx, {}, chk, rax};
        std::memcpy(c.a, a.data(), 16);
        std::memcpy(c.b, b.data(), 16);
        std::memcpy(c.expect, ex.data(), 16);
        cases.push_back(std::move(c));
    };
    const auto A = f4(6.0f, S1, S2, S3);
    const auto B = f4(1.5f, 7.0f, 8.0f, 9.0f);
    auto keep = [&](float lane0) { return std::array<uint32_t, 4>{fbits(lane0), fbits(S1), fbits(S2), fbits(S3)}; };
    add("addss", {0xF3, 0x0F, 0x58, 0xC1}, A, B, 0, keep(7.5f));
    add("subss", {0xF3, 0x0F, 0x5C, 0xC1}, A, B, 0, keep(4.5f));
    add("mulss", {0xF3, 0x0F, 0x59, 0xC1}, A, B, 0, keep(9.0f));
    add("divss", {0xF3, 0x0F, 0x5E, 0xC1}, A, B, 0, keep(4.0f));
    add("minss", {0xF3, 0x0F, 0x5D, 0xC1}, A, B, 0, keep(1.5f));
    add("maxss", {0xF3, 0x0F, 0x5F, 0xC1}, A, B, 0, keep(6.0f));
    add("sqrtss", {0xF3, 0x0F, 0x51, 0xC1}, A, f4(16.0f, 1, 1, 1), 0, keep(4.0f));
    add("movss r,r", {0xF3, 0x0F, 0x10, 0xC1}, A, B, 0, keep(1.5f));
    // MIN/MAX x86: (a<b)?a:b / (a>b)?a:b; con NaN se devuelve el segundo operando.
    {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        const auto MA = f4(1.0f, 5.0f, qnan, 2.0f), MB = f4(3.0f, 4.0f, 1.0f, qnan);
        add("minps", {0x0F, 0x5D, 0xC1}, MA, MB, 0, {fbits(1.0f), fbits(4.0f), fbits(1.0f), fbits(qnan)});
        add("maxps", {0x0F, 0x5F, 0xC1}, MA, MB, 0, {fbits(3.0f), fbits(5.0f), fbits(1.0f), fbits(qnan)});
        add("minss nan", {0xF3, 0x0F, 0x5D, 0xC1}, f4(qnan, S1, S2, S3), f4(2.0f, 0, 0, 0), 0, keep(2.0f));
        add("maxss nan", {0xF3, 0x0F, 0x5F, 0xC1}, f4(qnan, S1, S2, S3), f4(2.0f, 0, 0, 0), 0, keep(2.0f));
    }
    // SSE4.1 PMIN/PMAX enteros + PMULLD (66 0F 38 38..40) con lanes en bits crudos.
    {
        auto addI = [&](const char* n, uint8_t op3, std::array<uint32_t, 4> a, std::array<uint32_t, 4> b,
                        std::array<uint32_t, 4> ex) {
            SseCase c{n, {0x66, 0x0F, 0x38, op3, 0xC1}, {}, {}, 0, {}, false, 0};
            std::memcpy(c.a, a.data(), 16);
            std::memcpy(c.b, b.data(), 16);
            std::memcpy(c.expect, ex.data(), 16);
            cases.push_back(std::move(c));
        };
        // Bytes/words: 0x80 (-128 con signo, 128 sin signo) frente a 0x01.
        const std::array<uint32_t, 4> IA{0x8001FFFFu, 0x00000005u, 0xFFFFFFFEu, 7u};
        const std::array<uint32_t, 4> IB{0x01800001u, 0x00000003u, 0x00000002u, 9u};
        addI("pminsb", 0x38, IA, IB, {0x8080FFFFu, 0x00000003u, 0xFFFFFFFEu, 7u});
        addI("pminsd", 0x39, IA, IB, {0x8001FFFFu, 3u, 0xFFFFFFFEu, 7u});
        addI("pminuw", 0x3A, IA, IB, {0x01800001u, 3u, 0x00000002u, 7u});
        addI("pminud", 0x3B, IA, IB, {0x01800001u, 3u, 2u, 7u});
        addI("pmaxsb", 0x3C, IA, IB, {0x01010001u, 5u, 0x00000002u, 9u});
        addI("pmaxsd", 0x3D, IA, IB, {0x01800001u, 5u, 2u, 9u});
        addI("pmaxuw", 0x3E, IA, IB, {0x8001FFFFu, 5u, 0xFFFFFFFEu, 9u});
        addI("pmaxud", 0x3F, IA, IB, {0x8001FFFFu, 5u, 0xFFFFFFFEu, 9u});
        addI("pmulld", 0x40, {3u, 0xFFFFFFFFu, 0x10000u, 7u}, {5u, 2u, 0x10000u, 9u}, {15u, 0xFFFFFFFEu, 0u, 63u});
    }
    add("cvtsi2ss", {0xF3, 0x0F, 0x2A, 0xC1}, A, B, -3, keep(-3.0f));
    add("cvtsi2ss q", {0xF3, 0x48, 0x0F, 0x2A, 0xC1}, A, B, 5000000000LL, keep(5.0e9f));
    add("cmpltss", {0xF3, 0x0F, 0xC2, 0xC1, 0x01}, A, f4(10.0f, 0, 0, 0), 0,
        {0xFFFFFFFFu, fbits(S1), fbits(S2), fbits(S3)});
    add("unpcklps", {0x0F, 0x14, 0xC1}, A, B, 0, {fbits(6.0f), fbits(1.5f), fbits(S1), fbits(7.0f)});
    add("shufps 0x1B", {0x0F, 0xC6, 0xC1, 0x1B}, A, B, 0, {fbits(S3), fbits(S2), fbits(7.0f), fbits(1.5f)});
    add("movlhps", {0x0F, 0x16, 0xC1}, A, B, 0, {fbits(6.0f), fbits(S1), fbits(1.5f), fbits(7.0f)});
    add("movhlps", {0x0F, 0x12, 0xC1}, A, B, 0, {fbits(8.0f), fbits(9.0f), fbits(S2), fbits(S3)});
    add("cvttss2si", {0xF3, 0x0F, 0x2C, 0xC1}, A, f4(-7.9f, 0, 0, 0), 0, {fbits(6.0f), fbits(S1), fbits(S2), fbits(S3)},
        true, 0xFFFFFFF9LL);  // destino eax: -7 extendido con ceros
    {
        // Entero "indefinido" de x86: NaN o fuera de rango -> 0x80000000 (ARM saturaría).
        const std::array<uint32_t, 4> keepA{fbits(6.0f), fbits(S1), fbits(S2), fbits(S3)};
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        add("cvttss2si NaN", {0xF3, 0x0F, 0x2C, 0xC1}, A, f4(qnan, 0, 0, 0), 0, keepA, true, 0x80000000LL);
        add("cvttss2si 3e9", {0xF3, 0x0F, 0x2C, 0xC1}, A, f4(3.0e9f, 0, 0, 0), 0, keepA, true, 0x80000000LL);
        add("cvttss2si -3e9", {0xF3, 0x0F, 0x2C, 0xC1}, A, f4(-3.0e9f, 0, 0, 0), 0, keepA, true, 0x80000000LL);
        add("cvtss2si NaN", {0xF3, 0x0F, 0x2D, 0xC1}, A, f4(qnan, 0, 0, 0), 0, keepA, true, 0x80000000LL);
        add("cvttss2si q 1e20", {0xF3, 0x48, 0x0F, 0x2C, 0xC1}, A, f4(1.0e20f, 0, 0, 0), 0, keepA, true,
            static_cast<long long>(0x8000000000000000ULL));
        add("cvttss2si q NaN", {0xF3, 0x48, 0x0F, 0x2C, 0xC1}, A, f4(qnan, 0, 0, 0), 0, keepA, true,
            static_cast<long long>(0x8000000000000000ULL));
        add("cvttss2si q -5e9", {0xF3, 0x48, 0x0F, 0x2C, 0xC1}, A, f4(-5.0e9f, 0, 0, 0), 0, keepA, true,
            -5000000000LL);
        add("cvttps2dq indef", {0xF3, 0x0F, 0x5B, 0xC1}, A, f4(qnan, 3.0e9f, -7.9f, -3.0e9f), 0,
            {0x80000000u, 0x80000000u, 0xFFFFFFF9u, 0x80000000u});
        add("cvtps2dq indef", {0x66, 0x0F, 0x5B, 0xC1}, A, f4(2.5f, qnan, 3.5f, 1.0e10f), 0,
            {2u, 0x80000000u, 4u, 0x80000000u});
    }
    add("cvtss2si", {0xF3, 0x0F, 0x2D, 0xC1}, A, f4(2.5f, 0, 0, 0), 0, {fbits(6.0f), fbits(S1), fbits(S2), fbits(S3)},
        true, 2);  // al par más cercano
    // Dobles: xmm1 = {2.5, 4.0}; xmm0 = {10.0, 77.0}
    dpair[0] = 10.0;
    dpair[1] = 77.0;
    std::array<float, 4> AD, BD;
    std::memcpy(AD.data(), dpair, 16);
    dpair[0] = 2.5;
    dpair[1] = 4.0;
    std::memcpy(BD.data(), dpair, 16);
    auto keepD = [&](double lane0) {
        double d[2] = {lane0, 77.0};
        std::array<uint32_t, 4> r;
        std::memcpy(r.data(), d, 16);
        return r;
    };
    add("addsd", {0xF2, 0x0F, 0x58, 0xC1}, AD, BD, 0, keepD(12.5));
    add("divsd", {0xF2, 0x0F, 0x5E, 0xC1}, AD, BD, 0, keepD(4.0));
    add("cvtsi2sd", {0xF2, 0x0F, 0x2A, 0xC1}, AD, BD, -9, keepD(-9.0));
    add("cvtss2sd", {0xF3, 0x0F, 0x5A, 0xC1}, AD, f4(0.5f, 0, 0, 0), 0, keepD(0.5));
    add("movsd r,r", {0xF2, 0x0F, 0x10, 0xC1}, AD, BD, 0, keepD(2.5));
    {
        std::array<uint32_t, 4> ex;
        const float h = 2.5f;
        std::memcpy(&ex[0], &h, 4);
        std::memcpy(&ex[1], reinterpret_cast<const uint8_t*>(AD.data()) + 4, 12);  // resto de xmm0 intacto
        add("cvtsd2ss", {0xF2, 0x0F, 0x5A, 0xC1}, AD, BD, 0, ex);
    }
    add("cvttsd2si q", {0xF2, 0x48, 0x0F, 0x2C, 0xC1}, AD, BD, 0, keepD(10.0), true, 2);
    {
        auto pd = [](double x, double y) {
            double d[2] = {x, y};
            std::array<uint32_t, 4> r;
            std::memcpy(r.data(), d, 16);
            return r;
        };
        add("minpd", {0x66, 0x0F, 0x5D, 0xC1}, AD, BD, 0, pd(2.5, 4.0));    // {10,77} vs {2.5,4}
        add("maxpd", {0x66, 0x0F, 0x5F, 0xC1}, AD, BD, 0, pd(10.0, 77.0));
    }

    // Empaquetadas: resultado esperado calculado en C sobre los mismos datos.
    {
        const auto PA = f4(1.5f, -2.0f, 8.0f, 0.25f), PB = f4(0.5f, 4.0f, -8.0f, 0.25f);
        auto ew = [&](auto fn) {
            std::array<uint32_t, 4> r;
            for (int i = 0; i < 4; ++i) r[i] = fn(PA[i], PB[i], i);
            return r;
        };
        auto fl = [](float v) { return fbits(v); };
        auto mk = [](bool b) { return b ? 0xFFFFFFFFu : 0u; };
        add("addps", {0x0F, 0x58, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fl(a + b); }));
        add("subps", {0x0F, 0x5C, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fl(a - b); }));
        add("mulps", {0x0F, 0x59, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fl(a * b); }));
        add("divps", {0x0F, 0x5E, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fl(a / b); }));
        add("sqrtps", {0x0F, 0x51, 0xC1}, PA, f4(4, 9, 16, 0.25f), 0,
            {fl(2.0f), fl(3.0f), fl(4.0f), fl(0.5f)});
        add("andps", {0x0F, 0x54, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fbits(a) & fbits(b); }));
        add("andnps", {0x0F, 0x55, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return ~fbits(a) & fbits(b); }));
        add("orps", {0x0F, 0x56, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fbits(a) | fbits(b); }));
        add("xorps", {0x0F, 0x57, 0xC1}, PA, PB, 0, ew([&](float a, float b, int) { return fbits(a) ^ fbits(b); }));
        add("cmpeqps", {0x0F, 0xC2, 0xC1, 0}, PA, PB, 0, ew([&](float a, float b, int) { return mk(a == b); }));
        add("cmpltps", {0x0F, 0xC2, 0xC1, 1}, PA, PB, 0, ew([&](float a, float b, int) { return mk(a < b); }));
        add("cmpleps", {0x0F, 0xC2, 0xC1, 2}, PA, PB, 0, ew([&](float a, float b, int) { return mk(a <= b); }));
        add("cmpneqps", {0x0F, 0xC2, 0xC1, 4}, PA, PB, 0, ew([&](float a, float b, int) { return mk(a != b); }));
        add("cmpnltps", {0x0F, 0xC2, 0xC1, 5}, PA, PB, 0, ew([&](float a, float b, int) { return mk(!(a < b)); }));
        add("cmpnleps", {0x0F, 0xC2, 0xC1, 6}, PA, PB, 0, ew([&](float a, float b, int) { return mk(!(a <= b)); }));
        add("unpckhps", {0x0F, 0x15, 0xC1}, PA, PB, 0, {fl(PA[2]), fl(PB[2]), fl(PA[3]), fl(PB[3])});
        add("pshufd 0x1B", {0x66, 0x0F, 0x70, 0xC1, 0x1B}, PA, PB, 0, {fl(PB[3]), fl(PB[2]), fl(PB[1]), fl(PB[0])});
        add("cvttps2dq", {0xF3, 0x0F, 0x5B, 0xC1}, PA, f4(1.9f, -2.9f, 100.5f, -0.5f), 0, {1u, 0xFFFFFFFEu, 100u, 0u});
        add("cvtps2dq", {0x66, 0x0F, 0x5B, 0xC1}, PA, f4(1.5f, 2.5f, -1.5f, 3.4f), 0, {2u, 2u, 0xFFFFFFFEu, 3u});
        {
            const uint32_t ints[4] = {7u, 0xFFFFFFFDu, 0u, 100u};
            std::array<float, 4> IB;
            std::memcpy(IB.data(), ints, 16);
            add("cvtdq2ps", {0x0F, 0x5B, 0xC1}, PA, IB, 0, {fl(7.0f), fl(-3.0f), fl(0.0f), fl(100.0f)});
            add("paddd", {0x66, 0x0F, 0xFE, 0xC1}, IB, IB, 0, {14u, 0xFFFFFFFAu, 0u, 200u});
            add("pcmpeqd", {0x66, 0x0F, 0x76, 0xC1}, IB, IB, 0, {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu});
            add("pcmpgtd", {0x66, 0x0F, 0x66, 0xC1}, IB, f4(0, 0, 0, 0), 0, {0xFFFFFFFFu, 0u, 0u, 0xFFFFFFFFu});
        }
        add("movshdup", {0xF3, 0x0F, 0x16, 0xC1}, PA, PB, 0, {fl(PB[1]), fl(PB[1]), fl(PB[3]), fl(PB[3])});
        add("movsldup", {0xF3, 0x0F, 0x12, 0xC1}, PA, PB, 0, {fl(PB[0]), fl(PB[0]), fl(PB[2]), fl(PB[2])});
        add("haddps", {0xF2, 0x0F, 0x7C, 0xC1}, PA, PB, 0,
            {fl(PA[0] + PA[1]), fl(PA[2] + PA[3]), fl(PB[0] + PB[1]), fl(PB[2] + PB[3])});
        add("blendps 0x5", {0x66, 0x0F, 0x3A, 0x0C, 0xC1, 0x05}, PA, PB, 0, {fl(PB[0]), fl(PA[1]), fl(PB[2]), fl(PA[3])});
        add("insertps 0x90", {0x66, 0x0F, 0x3A, 0x21, 0xC1, 0x90}, PA, PB, 0, {fl(PA[0]), fl(PB[2]), fl(PA[2]), fl(PA[3])});
        add("dpps 0xF1", {0x66, 0x0F, 0x3A, 0x40, 0xC1, 0xF1}, PA, PB, 0,
            {fl(PA[0] * PB[0] + PA[1] * PB[1] + PA[2] * PB[2] + PA[3] * PB[3]), 0u, 0u, 0u});
        add("movmskps", {0x0F, 0x50, 0xC1}, PA, f4(-1, 2, -3, 4), 0, {fl(PA[0]), fl(PA[1]), fl(PA[2]), fl(PA[3])}, true,
            0x5);
    }

    for (const auto& c : cases) {
        CHECK(configureBackend(false));
        auto* out = static_cast<uint32_t*>(mapData());
        Bytes code;
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(c.a));
        append(code, {0x0F, 0x10, 0x00});  // movups xmm0, [rax]
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(c.b));
        append(code, {0x0F, 0x10, 0x08});  // movups xmm1, [rax]
        movRegImm64(code, 1, static_cast<uint64_t>(c.rcx));
        append(code, {0x31, 0xC0});  // xor eax, eax
        code.insert(code.end(), c.op.begin(), c.op.end());
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(out));
        append(code, {0x0F, 0x11, 0x02});  // movups [rdx], xmm0
        append(code, {0xC3});
        runCode(code);
        bool ok = std::memcmp(out, c.expect, 16) == 0;
        if (c.checkRax) ok = ok && static_cast<int64_t>(g_backend->context().gpr[0]) == c.expectRax;
        if (!ok) {
            std::printf("SSE %s: xmm0=%08x %08x %08x %08x (esperado %08x %08x %08x %08x) rax=%lld\n", c.name, out[0],
                        out[1], out[2], out[3], c.expect[0], c.expect[1], c.expect[2], c.expect[3],
                        static_cast<long long>(g_backend->context().gpr[0]));
        }
        CHECK(ok);
    }
}

void testSseXorAndScalar() {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto* out = static_cast<float*>(mapData());
    Bytes code;
    movRegImm64(code, 0, reinterpret_cast<uintptr_t>(a));
    append(code, {0x0F, 0x28, 0x00});        // movaps xmm0, [rax]
    append(code, {0x0F, 0x57, 0xC0});        // xorps xmm0, xmm0
    append(code, {0x0F, 0x28, 0x08});        // movaps xmm1, [rax]
    append(code, {0x0F, 0x58, 0xC1});        // addps xmm0, xmm1 (0 + a)
    append(code, {0xF3, 0x0F, 0x11, 0x40, 0x04});  // movss [rax+4], xmm0
    movRegImm64(code, 1, reinterpret_cast<uintptr_t>(out));
    append(code, {0xF3, 0x0F, 0x10, 0x48, 0x04});  // movss xmm1, [rax+4]
    append(code, {0xF3, 0x0F, 0x11, 0x09});        // movss [rcx], xmm1
    append(code, {0x31, 0xC0});                    // xor eax, eax
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 0);
    // MOVSS [mem], xmm almacena el lane 0 (1.0), no el lane 1.
    CHECK(out[0] > 0.99f && out[0] < 1.01f);
}

void testThunkCall() {
    Bytes code;
    movRegImm64(code, 0, g_fakeThunk);
    movRegImm64(code, 1, 0x1234);  // RCX = argumento
    append(code, {0xFF, 0xD0});    // call rax
    append(code, {0xC3});
    g_thunkCalls = 0;
    CHECK_EQ(runCode(code), static_cast<int>(g_thunkResult));
    CHECK_EQ(g_thunkCalls, 1);
    CHECK_EQ(g_thunkSeenArg, 0x1234);
    CHECK_EQ(g_backend->context().gpr[0], g_thunkResult);
}

void testNativeThunkDispatcher8Args() {
    CHECK(configureBackend(false));
    CHECK(g_backend->virtualStackSize() > 0);
    g_backend->setNativeThunkHandler(fakeThunk, g_backend);

    Bytes code;
    // Pasa argumentos en registros: RCX, RDX, R8, R9
    movRegImm64(code, 1, 0x1001); // RCX = arg0
    movRegImm64(code, 2, 0x2002); // RDX = arg1
    movRegImm64(code, 8, 0x3003); // R8  = arg2
    movRegImm64(code, 9, 0x4004); // R9  = arg3

    // Reserva espacio en el stack virtual invitado: 32 bytes shadow space + 32 bytes para 4 args de stack = 64 bytes (0x40)
    // SUB RSP, 0x40 -> 48 83 EC 40
    append(code, {0x48, 0x83, 0xEC, 0x40});

    // Escribe argumentos de stack tras el shadow space de 32 bytes:
    // arg4 at [RSP + 0x20] = 0x5005
    movRegImm64(code, 0, 0x5005);
    append(code, {0x48, 0x89, 0x44, 0x24, 0x20}); // mov [rsp+0x20], rax

    // arg5 at [RSP + 0x28] = 0x6006
    movRegImm64(code, 0, 0x6006);
    append(code, {0x48, 0x89, 0x44, 0x24, 0x28}); // mov [rsp+0x28], rax

    // arg6 at [RSP + 0x30] = 0x7007
    movRegImm64(code, 0, 0x7007);
    append(code, {0x48, 0x89, 0x44, 0x24, 0x30}); // mov [rsp+0x30], rax

    // arg7 at [RSP + 0x38] = 0x8008
    movRegImm64(code, 0, 0x8008);
    append(code, {0x48, 0x89, 0x44, 0x24, 0x38}); // mov [rsp+0x38], rax

    // CALL g_thunk8Target via RAX
    movRegImm64(code, 0, g_thunk8Target);
    append(code, {0xFF, 0xD0}); // call rax

    // Limpia el stack (ADD RSP, 0x40 -> 48 83 C4 40) y retorna
    append(code, {0x48, 0x83, 0xC4, 0x40});
    append(code, {0xC3}); // ret

    g_thunk8Calls = 0;
    std::memset(g_seenArgs8, 0, sizeof(g_seenArgs8));

    const uintptr_t stackTopBefore = reinterpret_cast<uintptr_t>(g_backend->virtualStack()) + g_backend->virtualStackSize() - 64;
    int64_t res = runCode(code);
    CHECK_EQ(res, static_cast<int64_t>(g_thunk8Result));
    CHECK_EQ(g_thunk8Calls, 1);
    CHECK_EQ(g_seenArgs8[0], 0x1001);
    CHECK_EQ(g_seenArgs8[1], 0x2002);
    CHECK_EQ(g_seenArgs8[2], 0x3003);
    CHECK_EQ(g_seenArgs8[3], 0x4004);
    CHECK_EQ(g_seenArgs8[4], 0x5005);
    CHECK_EQ(g_seenArgs8[5], 0x6006);
    CHECK_EQ(g_seenArgs8[6], 0x7007);
    CHECK_EQ(g_seenArgs8[7], 0x8008);
    CHECK_EQ(g_backend->context().gpr[0], g_thunk8Result);
    CHECK_EQ(g_backend->context().gpr[4], stackTopBefore + 8);
}

// CreateWindowExW / D3D11CreateDeviceAndSwapChain: 12 argumentos Win64.
void testNativeThunkDispatcher12Args() {
    CHECK(configureBackend(false));
    g_backend->setNativeThunkHandler(fakeThunk, g_backend);

    Bytes code;
    movRegImm64(code, 1, 0xA000);  // RCX
    movRegImm64(code, 2, 0xA001);  // RDX
    movRegImm64(code, 8, 0xA002);  // R8
    movRegImm64(code, 9, 0xA003);  // R9
    append(code, {0x48, 0x83, 0xEC, 0x60});  // sub rsp, 0x60 (shadow 0x20 + 8 args)
    for (uint8_t i = 0; i < 8; ++i) {
        movRegImm64(code, 0, 0xA004u + i);
        append(code, {0x48, 0x89, 0x44, 0x24, static_cast<uint8_t>(0x20 + i * 8)});  // mov [rsp+d8], rax
    }
    movRegImm64(code, 0, g_thunk12Target);
    append(code, {0xFF, 0xD0});              // call rax
    append(code, {0x48, 0x83, 0xC4, 0x60});  // add rsp, 0x60
    append(code, {0xC3});

    g_thunk12Calls = 0;
    std::memset(g_seenArgs12, 0, sizeof(g_seenArgs12));
    CHECK_EQ(runCode(code), static_cast<int64_t>(g_thunk12Result));
    CHECK_EQ(g_thunk12Calls, 1);
    for (int i = 0; i < 12; ++i) CHECK_EQ(g_seenArgs12[i], 0xA000u + static_cast<uint64_t>(i));

    // CW_USEDEFAULT y tamaños <= 0 -> tamaño real del panel.
    using namespace hyperplay::win32;
    CHECK_EQ(static_cast<uint32_t>(normalizeWindowExtent(kCwUseDefault, screenWidth())), static_cast<uint32_t>(screenWidth()));
    CHECK_EQ(static_cast<uint32_t>(normalizeWindowExtent(0, screenHeight())), static_cast<uint32_t>(screenHeight()));
    CHECK_EQ(static_cast<uint32_t>(normalizeWindowExtent(-5, screenHeight())), static_cast<uint32_t>(screenHeight()));
    CHECK_EQ(static_cast<uint32_t>(normalizeWindowExtent(1280, screenWidth())), 1280u);
    CHECK_EQ(static_cast<uint32_t>(normalizeWindowOrigin(kCwUseDefault)), 0u);
}

// EnumDisplayMonitors: el thunk invoca un callback invitado Win64 con 4 argumentos.
void testCallGuestNested() {
    CHECK(configureBackend(false));
    auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
    CHECK(stack != nullptr);
    if (stack == nullptr) return;
    g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + 8u * 1024u * 1024u - 64);
    g_backend->context().gpr[3] = 0x5555;  // RBX del llamador: debe restaurarse
    Bytes cb;
    append(cb, {0x48, 0x8D, 0x04, 0x11});  // lea rax, [rcx+rdx]
    append(cb, {0x4C, 0x01, 0xC0});        // add rax, r8
    append(cb, {0x4C, 0x01, 0xC8});        // add rax, r9
    append(cb, {0x48, 0xC7, 0xC3, 0x01, 0x00, 0x00, 0x00});  // mov rbx, 1 (clobber)
    append(cb, {0xC3});
    uint8_t* fn = mapCode(cb);
    CHECK(fn != nullptr);
    if (fn == nullptr) return;
    const uint64_t args[4] = {0x10, 0x200, 0x3000, 0x40000};
    CHECK_EQ(g_backend->callGuestNested(reinterpret_cast<uintptr_t>(fn), args, 4), 0x43210);
    CHECK_EQ(g_backend->context().gpr[3], 0x5555);
    CHECK_EQ(g_backend->callGuestNested(0, args, 4), static_cast<uint64_t>(-1));
}

// SETcc m8 debe escribir 0/1, no el byte bajo de la dirección (Unity: flag nographics=0x31).
void testSetccMemory() {
    CHECK(configureBackend(false));
    auto* data = static_cast<uint8_t*>(mapData());
    CHECK(data != nullptr);
    if (data == nullptr) return;
    uint8_t* slot = data + 0x31;
    std::memset(slot, 0xAA, 4);
    Bytes code;
    movRegImm64(code, 0, reinterpret_cast<uint64_t>(slot));
    append(code, {0x31, 0xC9});              // xor ecx, ecx
    append(code, {0x85, 0xC9});              // test ecx, ecx -> ZF=1
    append(code, {0x0F, 0x95, 0x00});        // setne [rax]   -> 0
    append(code, {0x0F, 0x94, 0x40, 0x01});  // sete [rax+1]  -> 1
    append(code, {0x0F, 0x95, 0x40, 0x02});  // setne [rax+2] -> 0
    append(code, {0xC3});
    runCode(code);
    CHECK_EQ(slot[0], 0);
    CHECK_EQ(slot[1], 1);
    CHECK_EQ(slot[2], 0);
    CHECK_EQ(slot[3], 0xAA);
}

// WinSafeHeap: 200.000 bloques pequeños deben caber en pocas arenas (antes 1 mmap de 4 KB cada uno).
void testWinSafeHeap() {
    hyperplay::winheap::Heap heap;
    std::vector<void*> ptrs;
    ptrs.reserve(200000);
    bool aligned = true, zeroed = true;
    for (int i = 0; i < 200000; ++i) {
        auto* p = static_cast<uint8_t*>(heap.alloc(static_cast<size_t>(16 + (i % 50))));
        if (p == nullptr) break;
        if ((reinterpret_cast<uintptr_t>(p) & 0xF) != 0) aligned = false;
        if (p[0] != 0 || p[15] != 0) zeroed = false;
        std::memset(p, 0xAB, 16);
        ptrs.push_back(p);
    }
    CHECK_EQ(ptrs.size(), 200000u);
    CHECK(aligned);
    CHECK(zeroed);
    CHECK(heap.arenaCount() <= 3u);  // 200k * <=96 B ~ 22 MB -> 1 arena
    CHECK_EQ(heap.size(ptrs[7]), 16u + 7u);
    CHECK(heap.owns(ptrs[123]));
    int stackVar = 0;
    CHECK(!heap.owns(&stackVar));       // puntero ajeno: no se lee su "cabecera"
    CHECK(!heap.free(&stackVar));
    for (void* p : ptrs) heap.free(p);
    CHECK_EQ(heap.liveBlocks(), 0u);
    CHECK(!heap.free(ptrs[0]));         // doble free detectado
    // Reutilización desde la lista libre, de nuevo a cero.
    auto* again = static_cast<uint8_t*>(heap.alloc(20));
    CHECK(again != nullptr);
    if (again != nullptr) CHECK_EQ(again[0], 0);
    CHECK(heap.arenaCount() <= 3u);
    // Bloque grande: mmap dedicado, liberado al sistema.
    auto* big = static_cast<uint8_t*>(heap.alloc(1u << 20));
    CHECK(big != nullptr);
    if (big != nullptr) {
        big[(1u << 20) - 1] = 1;
        CHECK_EQ(heap.size(big), 1u << 20);
        CHECK(heap.free(big));
        CHECK(!heap.owns(big));
    }
}

// Recursos D3D11 reales: layout/pitch, init, Map/Update/Copy, clear y conversión a RGBA8.
void testD3D11ResourceModel() {
    namespace d = hyperplay::d3d11;
    // Texture2D 256x128 RGBA8 con cadena completa de mips (MipLevels=0) y 2 capas.
    const uint32_t desc2d[11] = {256, 128, 0, 2, 28, 1, 0, 0, 0x28, 0, 0};
    d::Resource tex;
    CHECK(tex.init(d::parseTexture2DDesc(desc2d)));
    CHECK_EQ(tex.desc().mipLevels, 9u);             // 256 -> 1
    CHECK_EQ(tex.subresourceCount(), 18u);          // 9 mips x 2 capas
    CHECK_EQ(tex.layout(0)->rowPitch, 1024u);
    CHECK_EQ(tex.layout(1)->width, 128u);
    CHECK_EQ(tex.layout(1)->height, 64u);
    CHECK_EQ(tex.layout(8)->width, 1u);
    CHECK_EQ(tex.layout(9)->width, 256u);           // capa 1, mip 0
    CHECK(!tex.hasStorage());                       // reserva diferida
    uint32_t back[11] = {};
    CHECK_EQ(d::writeDesc(tex.desc(), back), 44u);
    CHECK_EQ(back[2], 9u);

    // BC1 (8 B / bloque 4x4) y BC3 (16 B): pitch por fila de bloques.
    const uint32_t bc1[11] = {64, 64, 1, 1, 71, 1, 0, 0, 8, 0, 0};
    d::Resource tbc1;
    CHECK(tbc1.init(d::parseTexture2DDesc(bc1)));
    CHECK_EQ(tbc1.layout(0)->rowPitch, 128u);       // 16 bloques * 8
    CHECK_EQ(tbc1.layout(0)->rows, 16u);
    const uint32_t bc3[11] = {6, 6, 1, 1, 77, 1, 0, 0, 8, 0, 0};
    d::Resource tbc3;
    CHECK(tbc3.init(d::parseTexture2DDesc(bc3)));
    CHECK_EQ(tbc3.layout(0)->rowPitch, 32u);        // ceil(6/4)=2 bloques * 16
    CHECK_EQ(tbc3.layout(0)->size, 64u);

    // Inicialización con D3D11_SUBRESOURCE_DATA y UpdateSubresource con caja.
    const uint32_t small[11] = {4, 2, 1, 1, 28, 1, 0, 0, 8, 0, 0};
    d::Resource t;
    CHECK(t.init(d::parseTexture2DDesc(small)));
    uint8_t pixels[2][4][4];
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 4; ++x) for (int c = 0; c < 4; ++c)
        pixels[y][x][c] = static_cast<uint8_t>(y * 64 + x * 16 + c);
    uint8_t init[16] = {};
    const void* src = pixels;
    const uint32_t pitch = 16;
    std::memcpy(init, &src, 8);
    std::memcpy(init + 8, &pitch, 4);
    t.initialize(init);
    CHECK_EQ(t.data(0)[16 + 4 + 1], pixels[1][1][1]);
    const uint32_t box[6] = {1, 0, 0, 3, 1, 1};     // x 1..3, y 0
    const uint8_t patch[8] = {0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3};
    CHECK(t.writeRegion(0, box, patch, 8, 0));
    CHECK_EQ(t.data(0)[4], 0xA0);
    CHECK_EQ(t.data(0)[8 + 3], 0xB3);
    CHECK_EQ(t.data(0)[12], pixels[0][3][0]);       // fuera de la caja intacto

    // CopySubresourceRegion hacia (2,1) de otra textura del mismo formato.
    d::Resource t2;
    CHECK(t2.init(d::parseTexture2DDesc(small)));
    const uint32_t srcBox[6] = {0, 0, 0, 2, 1, 1};
    CHECK(t2.copyRegion(0, 2, 1, 0, t, 0, srcBox));
    CHECK_EQ(t2.data(0)[16 + 8], t.data(0)[0]);
    CHECK_EQ(t2.data(0)[16 + 12], 0xA0);
    CHECK(!t2.copyRegion(0, 9, 0, 0, t, 0, srcBox));  // destino fuera de rango

    // ClearRenderTargetView + conversión BGRA -> RGBA8 para presentar.
    const uint32_t bgra[11] = {3, 2, 1, 1, 87, 1, 0, 0, 0x20, 0, 0};
    d::Resource rt;
    CHECK(rt.init(d::parseTexture2DDesc(bgra)));
    const float red[4] = {1.0f, 0.0f, 0.5f, 1.0f};
    rt.clear(0, 0, 0xFFFFFFFFu, red);
    CHECK_EQ(rt.data(0)[0], 128);                   // B
    CHECK_EQ(rt.data(0)[2], 255);                   // R
    std::vector<uint8_t> rgba;
    uint32_t w = 0, h = 0;
    CHECK(rt.toRgba8(rgba, w, h));
    CHECK_EQ(w, 3u);
    CHECK_EQ(h, 2u);
    CHECK_EQ(rgba[0], 255);
    CHECK_EQ(rgba[2], 128);
    CHECK_EQ(rgba[23], 255);

    // Half float.
    CHECK_EQ(d::Resource::floatToHalf(1.0f), 0x3C00u);
    CHECK(d::Resource::halfToFloat(0x3800) == 0.5f);

    // Buffer: Map de ByteWidth y UpdateSubresource con caja en bytes.
    const uint32_t bdesc[6] = {64, 2, 1, 0x10000, 0, 0};
    d::Resource buf;
    CHECK(buf.init(d::parseBufferDesc(bdesc)));
    CHECK_EQ(buf.layout(0)->rowPitch, 64u);
    const uint32_t bbox[6] = {8, 0, 0, 12, 1, 1};
    const uint8_t four[4] = {1, 2, 3, 4};
    CHECK(buf.writeRegion(0, bbox, four, 0, 0));
    CHECK_EQ(buf.data(0)[11], 4);
    // Descriptores inválidos.
    const uint32_t bad[11] = {0, 4, 1, 1, 28, 1, 0, 0, 0, 0, 0};
    d::Resource rbad;
    CHECK(!rbad.init(d::parseTexture2DDesc(bad)));
    CHECK_EQ(d::formatInfo(98).bytes, 16u);         // BC7
    CHECK(d::formatInfo(45).depth);                  // D24S8
}

// SuspendThread/GetThreadContext: un hilo invitado en bucle se detiene entre bloques con su
// contexto coherente y continúa tras ResumeThread.
void testGuestThreadSuspend() {
    Bytes loop;
    append(loop, {0x48, 0xFF, 0xC0});  // inc rax
    append(loop, {0xEB, 0xFB});        // jmp -5 (bucle infinito)
    uint8_t* code = mapCode(loop);
    CHECK(code != nullptr);
    if (code == nullptr) return;

    OryonEmitterBackend worker;
    EmitterConfig config;
    config.is32Bit = false;
    config.stackSize = 1u << 20;
    config.codeCacheSize = 4u << 20;
    config.persistBlocks = false;
    CHECK(worker.configure(config));
    std::atomic<int> tid{0};
    pthread_t th;
    struct Args { OryonEmitterBackend* b; uint8_t* code; std::atomic<int>* tid; } args{&worker, code, &tid};
    pthread_create(&th, nullptr, [](void* p) -> void* {
        auto* a = static_cast<Args*>(p);
        a->tid->store(static_cast<int>(gettid()));
        a->b->context().gpr[0] = 0;
        a->b->executeEntry(a->code);
        return nullptr;
    }, &args);
    while (tid.load() == 0 || OryonEmitterBackend::forThread(tid.load()) == nullptr) usleep(100);
    while (worker.context().gpr[0] < 10000) usleep(100);

    CHECK(OryonEmitterBackend::forThread(tid.load()) == &worker);
    CHECK_EQ(worker.requestSuspend(), 0u);
    CHECK(worker.waitUntilSuspended(2000));
    CHECK_EQ(worker.runState(), OryonEmitterBackend::kParked);
    const uint64_t frozen = worker.context().gpr[0];
    usleep(20000);
    CHECK_EQ(worker.context().gpr[0], frozen);  // detenido de verdad
    CHECK(worker.context().rip == reinterpret_cast<uintptr_t>(code));  // punto seguro: inicio de bloque
    CHECK_EQ(worker.requestSuspend(), 1u);      // anidado
    CHECK_EQ(worker.requestResume(), 2u);
    usleep(5000);
    CHECK_EQ(worker.context().gpr[0], frozen);  // sigue suspendido (contador 1)
    CHECK_EQ(worker.requestResume(), 1u);
    usleep(20000);
    CHECK(worker.context().gpr[0] > frozen);    // reanudado
    worker.stop();
    pthread_join(th, nullptr);
    CHECK(OryonEmitterBackend::forThread(tid.load()) == nullptr);  // desregistrado al salir
}

// VEH: `mov eax,[rdx+0x18]` con RDX=0 -> ACCESS_VIOLATION entregada al manejador invitado,
// que avanza Context.Rip 3 bytes y devuelve EXCEPTION_CONTINUE_EXECUTION; el bloque sigue con
// `mov eax,0x1337; ret`.
void testVectoredExceptionHandlerAccessViolation() {
    Bytes handler;
    append(handler, {0x48, 0x8B, 0x41, 0x08});                          // mov rax,[rcx+8]  (ContextRecord)
    append(handler, {0x48, 0x83, 0x80, 0xF8, 0x00, 0x00, 0x00, 0x03});  // add qword [rax+0xF8],3 (Rip)
    append(handler, {0xB8, 0xFF, 0xFF, 0xFF, 0xFF});                    // mov eax,-1
    append(handler, {0xC3});                                            // ret
    uint8_t* h = mapCode(handler);
    Bytes code;
    append(code, {0x31, 0xD2});                    // xor edx,edx
    append(code, {0x8B, 0x42, 0x18});              // mov eax,[rdx+0x18]  <- fallo
    append(code, {0xB8, 0x37, 0x13, 0x00, 0x00});  // mov eax,0x1337
    append(code, {0xC3});                          // ret
    CHECK(h != nullptr);
    if (h == nullptr) return;
    CHECK(configureBackend(false));
    g_backend->setNativeThunkHandler(fakeThunk, g_backend);
    g_vehHandler = reinterpret_cast<uintptr_t>(h);
    g_vehCalls = 0;
    auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
    std::memset(stack + 8u * 1024u * 1024u - 256, 0, 256);
    g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + 8u * 1024u * 1024u - 64);
    uint8_t* entry = mapCode(code);
    CHECK(entry != nullptr);
    if (entry == nullptr) return;
    const int64_t res = g_backend->executeEntry(entry);
    CHECK_EQ(g_vehCalls, 1);
    CHECK_EQ(g_vehCode, 0xC0000005u);
    CHECK_EQ(g_vehInfo0, 0u);                                          // lectura
    CHECK_EQ(g_vehInfo1, 0x18u);                                       // dirección del fallo
    CHECK_EQ(g_vehAddress, reinterpret_cast<uintptr_t>(entry) + 2);    // RIP exacta de la instrucción
    CHECK_EQ(g_vehCtxRip, reinterpret_cast<uintptr_t>(entry) + 2);
    CHECK_EQ(g_vehCtxRdx, 0u);                                         // registro vivo en el fallo
    CHECK_EQ(res, 0x1337);
    CHECK_EQ(g_backend->context().gpr[0], 0x1337u);

    // Sin manejador: la ejecución termina limpia con -4 (antes: SIGSEGV del proceso).
    g_vehHandler = 0;
    g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + 8u * 1024u * 1024u - 64);
    CHECK_EQ(g_backend->executeEntry(entry), -4);
}

// Código automodificable: página RWX del "huésped" ejecutada, reescrita y vuelta a ejecutar.
uintptr_t g_smcPage = 0;
int smcTestHook(uintptr_t page) {
    return page == g_smcPage ? (PROT_READ | PROT_WRITE | PROT_EXEC) : -1;
}

void testSelfModifyingCode() {
    const size_t ps = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* mem = mmap(nullptr, ps, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mem != MAP_FAILED);
    if (mem == MAP_FAILED) return;
    auto* code = static_cast<uint8_t*>(mem);
    const uint8_t prog[] = {0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3};  // mov eax,1; ret
    std::memcpy(code, prog, sizeof(prog));
    g_smcPage = reinterpret_cast<uintptr_t>(code);
    OryonEmitterBackend::setSmcWritableHook(&smcTestHook);
    CHECK(configureBackend(false));
    auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
    auto resetStack = [&] {
        g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + 8u * 1024u * 1024u - 64);
    };
    resetStack();
    CHECK_EQ(g_backend->executeEntry(code), 1);
    code[1] = 0x02;  // escritura sobre código ya traducido -> SIGSEGV -> desprotección + invalidación
    CHECK_EQ(code[1], 2u);
    resetStack();
    CHECK_EQ(g_backend->executeEntry(code), 2);  // la traducción vieja se descartó
    code[1] = 0x07;
    resetStack();
    CHECK_EQ(g_backend->executeEntry(code), 7);
    OryonEmitterBackend::setSmcWritableHook(nullptr);
    OryonEmitterBackend::smcForgetRange(g_smcPage, ps);
    munmap(mem, ps);
}

// Atomicidad x86 en 4 núcleos reales: cada hilo ejecuta `cuerpo` 100.000 veces.
uint64_t runAtomicStress(const Bytes& body, uint32_t* counter, uint32_t* lockWord) {
    constexpr int kThreads = 4;
    constexpr uint32_t kIters = 100000;
    Bytes code;
    movRegImm64(code, 7, reinterpret_cast<uint64_t>(counter));   // rdi = contador
    movRegImm64(code, 6, reinterpret_cast<uint64_t>(lockWord));  // rsi = cerrojo
    append(code, {0x41, 0xB8});                                   // mov r8d, kIters
    for (int i = 0; i < 4; ++i) code.push_back(static_cast<uint8_t>((kIters >> (8 * i)) & 0xFF));
    const size_t loopStart = code.size();
    code.insert(code.end(), body.begin(), body.end());
    append(code, {0x41, 0xFF, 0xC8});                             // dec r8d
    const int rel = static_cast<int>(loopStart) - static_cast<int>(code.size() + 6);
    append(code, {0x0F, 0x85});                                   // jnz loopStart (rel32)
    for (int i = 0; i < 4; ++i) code.push_back(static_cast<uint8_t>((static_cast<uint32_t>(rel) >> (8 * i)) & 0xFF));
    append(code, {0xC3});
    uint8_t* entry = mapCode(code);
    if (entry == nullptr) return 0;
    struct Worker {
        OryonEmitterBackend backend;
        pthread_t th{};
        uint8_t* entry = nullptr;
    };
    std::vector<std::unique_ptr<Worker>> workers;
    for (int t = 0; t < kThreads; ++t) {
        auto w = std::make_unique<Worker>();
        EmitterConfig cfg;
        cfg.is32Bit = false;
        cfg.stackSize = 1u << 20;
        cfg.codeCacheSize = 4u << 20;
        cfg.persistBlocks = false;
        w->backend.configure(cfg);
        w->entry = entry;
        workers.push_back(std::move(w));
    }
    for (auto& w : workers) {
        pthread_create(&w->th, nullptr, [](void* p) -> void* {
            auto* wk = static_cast<Worker*>(p);
            wk->backend.executeEntry(wk->entry);
            return nullptr;
        }, w.get());
    }
    for (auto& w : workers) pthread_join(w->th, nullptr);
    return static_cast<uint64_t>(kThreads) * kIters;
}

void testAtomicsMultithread() {
    alignas(64) static uint32_t counter = 0;
    alignas(64) static uint32_t lockWord = 0;
    struct Case {
        const char* name;
        Bytes body;
        uint32_t perIter;
    };
    const std::vector<Case> cases = {
        {"lock xadd", {0xB8, 0x01, 0x00, 0x00, 0x00, 0xF0, 0x0F, 0xC1, 0x07}, 1},   // mov eax,1; lock xadd [rdi],eax
        {"lock inc", {0xF0, 0xFF, 0x07}, 1},                                          // lock inc dword [rdi]
        {"lock add", {0xF0, 0x83, 0x07, 0x03}, 3},                                    // lock add dword [rdi],3
        {"lock cmpxchg", {0x8B, 0x07, 0x8D, 0x50, 0x01, 0xF0, 0x0F, 0xB1, 0x17, 0x75, 0xF5}, 1},
        // spinlock con xchg protegiendo un incremento normal:
        // spin: mov eax,1; xchg [rsi],eax; test eax,eax; jnz spin; inc dword [rdi]; mov dword [rsi],0
        {"xchg spinlock", {0xB8, 0x01, 0x00, 0x00, 0x00, 0x87, 0x06, 0x85, 0xC0, 0x75, 0xF5,
                           0xFF, 0x07, 0xC7, 0x06, 0x00, 0x00, 0x00, 0x00}, 1},
    };
    for (const Case& c : cases) {
        counter = 0;
        lockWord = 0;
        const uint64_t iters = runAtomicStress(c.body, &counter, &lockWord);
        std::printf("  atomics %-14s: %u / %llu\n", c.name, counter, static_cast<unsigned long long>(iters * c.perIter));
        CHECK_EQ(counter, iters * c.perIter);
    }
}

// SSE4.1 ROUNDPD/ROUNDPS/ROUNDSD (Mono: Math.Ceiling/Floor/Truncate).
void testSse41Round() {
    CHECK(configureBackend(false));
    auto& c = g_backend->context();
    const double pd[2] = {1.25, -1.5};
    const float ps[4] = {1.75f, -1.25f, 2.0f, -0.5f};
    const double sd[2] = {-2.75, 99.5};
    std::memcpy(c.simd[0], pd, 16);
    std::memcpy(c.simd[1], ps, 16);
    std::memcpy(c.simd[2], sd, 16);
    Bytes code;
    append(code, {0x66, 0x0F, 0x3A, 0x09, 0xC0, 0x02});  // roundpd xmm0,xmm0,2 (+inf)
    append(code, {0x66, 0x0F, 0x3A, 0x08, 0xC9, 0x01});  // roundps xmm1,xmm1,1 (-inf)
    append(code, {0x66, 0x0F, 0x3A, 0x0B, 0xD2, 0x03});  // roundsd xmm2,xmm2,3 (truncar)
    append(code, {0xC3});
    runCode(code);
    double rpd[2], rsd[2];
    float rps[4];
    std::memcpy(rpd, c.simd[0], 16);
    std::memcpy(rps, c.simd[1], 16);
    std::memcpy(rsd, c.simd[2], 16);
    CHECK(rpd[0] == 2.0 && rpd[1] == -1.0);
    CHECK(rps[0] == 1.0f && rps[1] == -2.0f && rps[2] == 2.0f && rps[3] == -1.0f);
    CHECK(rsd[0] == -2.0);
    CHECK(rsd[1] == 99.5);  // lane alto intacto (escalar)
}

// POPCNT (F3 [REX.W] 0F B8): UnityPlayer lo usa en máscaras de bits (hilo principal moría con
// "opcode 0F B8 no soportado"). 32 bits ignora la mitad alta; ZF=1 solo con fuente 0.
void testPopcnt() {
    CHECK(configureBackend(false));
    auto& c = g_backend->context();
    Bytes code;
    movRegImm64(code, 1, 0xF0F0000000000001ull);           // rcx
    append(code, {0xF3, 0x48, 0x0F, 0xB8, 0xC1});          // popcnt rax, rcx -> 9
    append(code, {0xF3, 0x0F, 0xB8, 0xD1});                // popcnt edx, ecx -> 1
    append(code, {0xC3});
    runCode(code);
    CHECK_EQ(c.gpr[0], 9ull);
    CHECK_EQ(c.gpr[2], 1ull);
    CHECK((c.rflags & (1ull << 6)) == 0);  // ZF=0
    Bytes zero;
    movRegImm64(zero, 1, 0);
    append(zero, {0xF3, 0x48, 0x0F, 0xB8, 0xC1});  // popcnt rax, rcx -> 0
    append(zero, {0xC3});
    CHECK(configureBackend(false));
    runCode(zero);
    CHECK_EQ(g_backend->context().gpr[0], 0ull);
    CHECK((g_backend->context().rflags & (1ull << 6)) != 0);  // ZF=1
    CHECK((g_backend->context().rflags & 1ull) == 0);         // CF=0
}

// PINSRW / PEXTRW (66 0F C4 / C5): UnityPlayer (hilo principal moría con "0F C5").
void testPinsrwPextrw() {
    CHECK(configureBackend(false));
    auto& c = g_backend->context();
    std::memset(c.simd[0], 0x11, 16);
    Bytes code;
    movRegImm64(code, 1, 0xFFFF1234BEEFull);           // rcx (solo cuentan los 16 bits bajos)
    append(code, {0x66, 0x0F, 0xC4, 0xC1, 0x03});      // pinsrw xmm0, ecx, 3
    append(code, {0x66, 0x0F, 0xC5, 0xC0, 0x03});      // pextrw eax, xmm0, 3 -> 0xBEEF
    append(code, {0x66, 0x0F, 0xC5, 0xD0, 0x02});      // pextrw edx, xmm0, 2 -> 0x1111
    append(code, {0xC3});
    runCode(code);
    CHECK_EQ(c.gpr[0], 0xBEEFull);
    CHECK_EQ(c.gpr[2], 0x1111ull);
    uint16_t lanes[8];
    std::memcpy(lanes, c.simd[0], 16);
    CHECK(lanes[3] == 0xBEEF && lanes[4] == 0x1111);
}

// Decodificador DXBC sobre los shaders reales volcados por el juego (si están en el dispositivo:
// /data/local/tmp/hk_shaders/*.dxbc). Imprime el histograma de opcodes ejecutables.
void testDxbcCorpus() {
    DIR* dir = opendir("/data/local/tmp/hk_shaders");
    if (dir == nullptr) {
        std::printf("DXBC: sin corpus en /data/local/tmp/hk_shaders (omitido)\n");
        return;
    }
    int files = 0;
    std::map<uint32_t, int> hist;
    while (dirent* de = readdir(dir)) {
        const std::string name = de->d_name;
        if (name.size() < 6 || name.substr(name.size() - 5) != ".dxbc") continue;
        FILE* f = std::fopen(("/data/local/tmp/hk_shaders/" + name).c_str(), "rb");
        if (f == nullptr) continue;
        std::vector<uint8_t> bc;
        uint8_t buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) bc.insert(bc.end(), buf, buf + n);
        std::fclose(f);
        const hyperplay::dxbc::Program p = hyperplay::dxbc::parse(bc.data(), bc.size());
        ++files;
        CHECK(p.ok());
        if (!p.ok()) std::printf("DXBC %s: %s\n", name.c_str(), p.error.c_str());
        CHECK(!p.outputSig.empty());
        const bool isPs = name.rfind("ps_", 0) == 0;
        CHECK_EQ(p.type, isPs ? 0u : 1u);
        for (const auto& ins : p.code) ++hist[ins.opcode];
        if (p.code.size() <= 13) {
            for (const auto& ins : p.code) {
                std::string s = "  op" + std::to_string(ins.opcode) + (ins.saturate ? "_sat" : "");
                for (const auto& o : ins.operands) {
                    s += " t" + std::to_string(o.type) + "[" + std::to_string(o.index[0]);
                    if (o.indexDim > 1) s += "," + std::to_string(o.index[1]);
                    s += "]";
                    if (o.selMode == 0) s += ".m" + std::to_string(o.mask);
                    else s += "." + std::string(1, "xyzw"[o.swizzle[0]]) + "xyzw"[o.swizzle[1]] + "xyzw"[o.swizzle[2]] + "xyzw"[o.swizzle[3]];
                    if (o.neg) s += "(neg)";
                }
                std::printf("%s\n", s.c_str());
            }
            for (const auto& e : p.inputSig) std::printf("  in  %s%u r%u m%x sv%u\n", e.name.c_str(), e.semanticIndex, e.reg, e.mask, e.systemValue);
            for (const auto& e : p.outputSig) std::printf("  out %s%u r%u m%x sv%u\n", e.name.c_str(), e.semanticIndex, e.reg, e.mask, e.systemValue);
        }
        std::printf("DXBC %s: sm%u.%u %zu instr, %u temps, %zu cb, %zu tex, in=%zu out=%zu\n", name.c_str(), p.major,
                    p.minor, p.code.size(), p.tempCount, p.constantBuffers.size(), p.resources.size(),
                    p.inputSig.size(), p.outputSig.size());
    }
    closedir(dir);
    std::string h;
    for (const auto& [op, c] : hist) h += " " + std::to_string(op) + ":" + std::to_string(c);
    std::printf("DXBC: %d shaders, opcodes:%s\n", files, h.c_str());
}

// Intérprete DXBC con shaders reales de Hollow Knight:
// ps_003 = UI (sample * color, premultiplica alfa); vs_008 = blit (o0.xy = v1.xy, o1 = (v0.xyz, 1)).
namespace {
std::vector<uint8_t> readCorpusFile(const char* name) {
    std::vector<uint8_t> bc;
    FILE* f = std::fopen((std::string("/data/local/tmp/hk_shaders/") + name).c_str(), "rb");
    if (f == nullptr) return bc;
    uint8_t buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) bc.insert(bc.end(), buf, buf + n);
    std::fclose(f);
    return bc;
}

struct FixedTexture : hyperplay::dxbc::TextureAccess {
    float coordSeen[2] = {0, 0};
    hyperplay::dxbc::Vec4 sample(uint32_t, uint32_t, const float* c, float, bool, const int8_t*) override {
        coordSeen[0] = c[0];
        coordSeen[1] = c[1];
        hyperplay::dxbc::Vec4 v;
        v.f[0] = 0.5f; v.f[1] = 0.25f; v.f[2] = 1.0f; v.f[3] = 0.5f;
        return v;
    }
    hyperplay::dxbc::Vec4 load(uint32_t, const int32_t*, const int8_t*) override { return {}; }
    hyperplay::dxbc::Vec4 info(uint32_t, uint32_t) override { return {}; }
};
}  // namespace

void testDxbcVm() {
    using namespace hyperplay::dxbc;
    const std::vector<uint8_t> ps = readCorpusFile("ps_003.dxbc");
    const std::vector<uint8_t> vs = readCorpusFile("vs_008.dxbc");
    if (ps.empty() || vs.empty()) {
        std::printf("DXBC VM: sin corpus (omitido)\n");
        return;
    }
    const Program pp = parse(ps.data(), ps.size());
    CHECK(pp.ok());
    Vm pvm(pp);
    FixedTexture tex;
    pvm.textures = &tex;
    pvm.inputs[1].f[0] = pvm.inputs[1].f[1] = pvm.inputs[1].f[2] = 1.0f;  // COLOR
    pvm.inputs[1].f[3] = 0.8f;
    pvm.inputs[2].f[0] = 0.25f;  // TEXCOORD0
    pvm.inputs[2].f[1] = 0.75f;
    CHECK(pvm.run());
    CHECK(!pvm.discarded);
    CHECK(tex.coordSeen[0] == 0.25f && tex.coordSeen[1] == 0.75f);
    const float* o = pvm.outputs[0].f;
    CHECK(std::fabs(o[3] - 0.4f) < 1e-6f);
    CHECK(std::fabs(o[0] - 0.2f) < 1e-6f && std::fabs(o[1] - 0.1f) < 1e-6f && std::fabs(o[2] - 0.4f) < 1e-6f);

    const Program vp = parse(vs.data(), vs.size());
    CHECK(vp.ok());
    Vm vvm(vp);
    vvm.inputs[0].f[0] = -1.0f; vvm.inputs[0].f[1] = 1.0f; vvm.inputs[0].f[2] = 0.5f;
    vvm.inputs[1].f[0] = 0.0f; vvm.inputs[1].f[1] = 1.0f;
    CHECK(vvm.run());
    CHECK(vvm.outputs[0].f[0] == 0.0f && vvm.outputs[0].f[1] == 1.0f);
    CHECK(vvm.outputs[1].f[0] == -1.0f && vvm.outputs[1].f[2] == 0.5f && vvm.outputs[1].f[3] == 1.0f);
}

// Rasterizador por software con el par real de blit de Hollow Knight (vs_008 + ps_009):
// quad en la mitad izquierda de un RT 8x8, sin blending y con SRC_ALPHA/INV_SRC_ALPHA.
void testSoftRaster() {
    using namespace hyperplay;
    const std::vector<uint8_t> vsb = readCorpusFile("vs_008.dxbc");
    const std::vector<uint8_t> psb = readCorpusFile("ps_009.dxbc");
    if (vsb.empty() || psb.empty()) {
        std::printf("SoftRaster: sin corpus (omitido)\n");
        return;
    }
    const dxbc::Program vs = dxbc::parse(vsb.data(), vsb.size());
    const dxbc::Program ps = dxbc::parse(psb.data(), psb.size());
    CHECK(vs.ok() && ps.ok());
    // Vértices: POSITION (float3) + TEXCOORD0 (float2), 20 bytes.
    const float verts[4][5] = {{-1, 1, 0.5f, 0, 0}, {0, 1, 0.5f, 1, 0}, {-1, -1, 0.5f, 0, 1}, {0, -1, 0.5f, 1, 1}};
    const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
    std::vector<uint8_t> rtMem(8 * 8 * 4, 0);
    raster::DrawState s;
    s.vs = &vs;
    s.ps = &ps;
    for (const auto& e : vs.inputSig) {
        raster::VertexElement el;
        el.vsReg = e.reg;
        el.format = e.name == "POSITION" ? 6u : 16u;
        el.offset = e.name == "POSITION" ? 0u : 12u;
        s.elements.push_back(el);
    }
    s.vb[0].data = reinterpret_cast<const uint8_t*>(verts);
    s.vb[0].size = sizeof(verts);
    s.vb[0].stride = 20;
    s.ib = reinterpret_cast<const uint8_t*>(indices);
    s.ibSize = sizeof(indices);
    s.ibFormat = 57;
    FixedTexture tex;
    s.psTex = &tex;
    s.viewport = {0, 0, 8, 8, 0, 1};
    s.cullMode = 1;
    s.rt = {rtMem.data(), 8, 8, 32, 28};
    raster::SoftRasterizer r;
    CHECK(r.draw(s, 6, 0, 0, true));
    CHECK_EQ(r.stats.triangles, 2ull);
    CHECK_EQ(r.stats.pixels, 32ull);  // 4 columnas x 8 filas, sin huecos ni dobles (top-left)
    CHECK(rtMem[0] == 128 && rtMem[1] == 64 && rtMem[2] == 255 && rtMem[3] == 128);
    CHECK(rtMem[(7 * 8 + 3) * 4] == 128);   // última fila, columna 3: dentro
    CHECK(rtMem[(0 * 8 + 4) * 4 + 3] == 0); // columna 4: fuera
    // Blending alfa sobre negro opaco: 0.5*src + 0.5*dst.
    for (size_t i = 0; i < rtMem.size(); i += 4) {
        rtMem[i] = rtMem[i + 1] = rtMem[i + 2] = 0;
        rtMem[i + 3] = 255;
    }
    s.blend.enable = true;
    s.blend.src = 5;  // SRC_ALPHA
    s.blend.dst = 6;  // INV_SRC_ALPHA
    s.blend.srcA = 2;
    s.blend.dstA = 1;
    raster::SoftRasterizer r2;
    CHECK(r2.draw(s, 6, 0, 0, true));
    CHECK(rtMem[0] == 64 && rtMem[1] == 32 && rtMem[2] == 128);
    CHECK(rtMem[3] == 128);  // alfa: 1*0.5 + 0*1
    // Cull back: el quad (horario en pantalla) es frontal y no se descarta; con FRONT sí.
    s.cullMode = 2;
    raster::SoftRasterizer r3;
    CHECK(r3.draw(s, 6, 0, 0, true));
    CHECK_EQ(r3.stats.culled, 2ull);
    // Raster en paralelo por franjas (3 hilos + llamante): misma cobertura exacta en 64x64.
    std::vector<uint8_t> big(64 * 64 * 4, 0);
    s.cullMode = 1;
    s.blend.enable = false;
    s.viewport = {0, 0, 64, 64, 0, 1};
    s.rt = {big.data(), 64, 64, 256, 28};
    raster::SoftRasterizer::setWorkerCount(3);
    raster::SoftRasterizer r4;
    CHECK(r4.draw(s, 6, 0, 0, true));
    raster::SoftRasterizer::setWorkerCount(0);
    CHECK_EQ(r4.stats.pixels, 2048ull);  // 32 columnas x 64 filas
    size_t written = 0;
    for (size_t i = 0; i < big.size(); i += 4) written += big[i + 3] == 128 ? 1 : 0;
    CHECK_EQ(written, 2048u);
}

void testD3D11Sampler() {
    using namespace hyperplay::d3d11;
    // BC1: c0 = rojo (0xF800) > c1 = azul (0x001F); índices: píxel0=0, píxel1=1, píxel2=2.
    const uint8_t bc1[8] = {0x00, 0xF8, 0x1F, 0x00, 0b00100100, 0, 0, 0};
    uint8_t px[64];
    CHECK(bc::decodeBlock(71, bc1, px));
    CHECK(px[0] == 255 && px[1] == 0 && px[2] == 0 && px[3] == 255);  // rojo
    CHECK(px[4] == 0 && px[6] == 255);                                // azul
    CHECK(px[8] == 170 && px[10] == 85);                              // 2/3 rojo + 1/3 azul
    // BC3: alfa a0=255, a1=0 con índice 1 en el píxel 0 -> 0.
    uint8_t bc3[16] = {255, 0, 1, 0, 0, 0, 0, 0};
    std::memcpy(bc3 + 8, bc1, 8);
    CHECK(bc::decodeBlock(77, bc3, px));
    CHECK(px[3] == 0 && px[7] == 255);
    // Direccionamiento
    CHECK_EQ(addressTexel(-1, 4, 1), 3);   // wrap
    CHECK_EQ(addressTexel(4, 4, 2), 3);    // mirror
    CHECK_EQ(addressTexel(-1, 4, 2), 0);
    CHECK_EQ(addressTexel(9, 4, 3), 3);    // clamp
    CHECK_EQ(addressTexel(4, 4, 4), -1);   // border
    // Bilineal en textura 2x1 negro|blanco: centro = 0.5; punto = texel exacto.
    const uint8_t tex[8] = {0, 0, 0, 255, 255, 255, 255, 255};
    TexView t{tex, 2, 1, 8, 28, false};
    SamplerDesc lin;
    lin.address[0] = lin.address[1] = 3;
    const auto mid = sample2D(t, lin, 0.5f, 0.5f, nullptr);
    CHECK(std::fabs(mid.f[0] - 0.5f) < 1e-5f && mid.f[3] == 1.0f);
    SamplerDesc pt = lin;
    pt.filter = 0;
    CHECK(sample2D(t, pt, 0.74f, 0.5f, nullptr).f[0] == 1.0f);
    CHECK(sample2D(t, pt, 0.24f, 0.5f, nullptr).f[0] == 0.0f);
    std::vector<uint8_t> rgba;
    CHECK(decompressToRgba8(71, bc1, 4, 4, 8, rgba));
    CHECK(rgba.size() == 64 && rgba[0] == 255);
}

// Rendimiento del rasterizador: blit a pantalla completa 1344x608 (vs_008 + ps_009) con una
// textura RGBA8 sRGB real (bilineal) y blending alfa, en 1 hilo y con el pool.
void benchmarkSoftRaster() {
    using namespace hyperplay;
    const std::vector<uint8_t> vsb = readCorpusFile("vs_008.dxbc");
    const std::vector<uint8_t> psb = readCorpusFile("ps_009.dxbc");
    if (vsb.empty() || psb.empty()) return;
    const dxbc::Program vs = dxbc::parse(vsb.data(), vsb.size());
    const dxbc::Program ps = dxbc::parse(psb.data(), psb.size());
    struct Tex : dxbc::TextureAccess {
        std::vector<uint8_t> px = std::vector<uint8_t>(256 * 256 * 4, 200);
        d3d11::TexView view{px.data(), 256, 256, 1024, 29, false};
        d3d11::SamplerDesc samp;
        dxbc::Vec4 sample(uint32_t, uint32_t, const float* c, float, bool, const int8_t* o) override {
            return d3d11::sample2D(view, samp, c[0], c[1], o);
        }
        dxbc::Vec4 load(uint32_t, const int32_t*, const int8_t*) override { return {}; }
        dxbc::Vec4 info(uint32_t, uint32_t) override { return {}; }
    } tex;
    const float verts[4][5] = {{-1, 1, 0.5f, 0, 0}, {1, 1, 0.5f, 1, 0}, {-1, -1, 0.5f, 0, 1}, {1, -1, 0.5f, 1, 1}};
    const uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
    std::vector<uint8_t> rt(1344 * 608 * 4, 0);
    raster::DrawState s;
    s.vs = &vs;
    s.ps = &ps;
    for (const auto& e : vs.inputSig) {
        raster::VertexElement el;
        el.vsReg = e.reg;
        el.format = e.name == "POSITION" ? 6u : 16u;
        el.offset = e.name == "POSITION" ? 0u : 12u;
        s.elements.push_back(el);
    }
    s.vb[0] = {reinterpret_cast<const uint8_t*>(verts), sizeof(verts), 20, 0};
    s.ib = reinterpret_cast<const uint8_t*>(indices);
    s.ibSize = sizeof(indices);
    s.psTex = &tex;
    s.viewport = {0, 0, 1344, 608, 0, 1};
    s.cullMode = 1;
    s.blend.enable = true;
    s.blend.src = 5;
    s.blend.dst = 6;
    s.rt = {rt.data(), 1344, 608, 1344 * 4, 28};
    {
        dxbc::Vm vm(ps);
        vm.textures = &tex;
        vm.inputs[0].f[0] = 0.3f;
        vm.inputs[0].f[1] = 0.6f;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < 1000000; ++i) vm.run();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1e6;
        std::printf("BENCH Vm ps_009 (sample bilineal sRGB): %.0f ns/invocación\n", ns);
    }
    for (unsigned workers : {0u, 0u, 5u, 5u}) {
        raster::SoftRasterizer::setWorkerCount(workers);
        raster::SoftRasterizer r;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        r.draw(s, 6, 0, 0, true);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        std::printf("BENCH raster 1344x608 blit+blend (%u hilos extra): %.1f ms (%.0f ns/px)\n", workers, ms,
                    ms * 1e6 / static_cast<double>(r.stats.pixels ? r.stats.pixels : 1));
        CHECK_EQ(r.stats.pixels, 1344ull * 608ull);
    }
    raster::SoftRasterizer::setWorkerCount(0);
}

// Atómicos sobre direcciones desalineadas (Mono parchea call sites con XCHG de 8 bytes a +3):
// antes CASAL daba SIGBUS.
void testUnalignedAtomics() {
    alignas(16) static uint8_t buf[32];
    std::memset(buf, 0, sizeof(buf));
    uint64_t init = 0x0102030405060708ull;
    std::memcpy(buf + 3, &init, 8);
    CHECK(configureBackend(false));
    Bytes code;
    movRegImm64(code, 0, reinterpret_cast<uintptr_t>(buf + 3));
    movRegImm64(code, 1, 0x1122334455667788ull);
    append(code, {0x48, 0x87, 0x08});                   // xchg [rax], rcx
    movRegImm64(code, 2, 0x10);
    append(code, {0xF0, 0x48, 0x0F, 0xC1, 0x10});       // lock xadd [rax], rdx
    append(code, {0xC3});
    runCode(code);
    uint64_t now = 0;
    std::memcpy(&now, buf + 3, 8);
    CHECK_EQ(g_backend->context().gpr[1], 0x0102030405060708ull);  // xchg devolvió el previo
    CHECK_EQ(g_backend->context().gpr[2], 0x1122334455667788ull);  // xadd devolvió el previo
    CHECK_EQ(now, 0x1122334455667798ull);

    // MOVNTI [rax], ecx / [rax+8], rcx (0F C3): store no temporal = store normal.
    alignas(16) static uint64_t nt[2];
    nt[0] = nt[1] = 0xFFFFFFFFFFFFFFFFull;
    CHECK(configureBackend(false));
    Bytes c2;
    movRegImm64(c2, 0, reinterpret_cast<uintptr_t>(nt));
    movRegImm64(c2, 1, 0xAABBCCDD11223344ull);
    append(c2, {0x0F, 0xC3, 0x08});                     // movnti [rax], ecx
    append(c2, {0x48, 0x0F, 0xC3, 0x48, 0x08});         // movnti [rax+8], rcx
    append(c2, {0xC3});
    runCode(c2);
    CHECK_EQ(nt[0], 0xFFFFFFFF11223344ull);
    CHECK_EQ(nt[1], 0xAABBCCDD11223344ull);

    // MOVNTDQ [rax], xmm0 (66 0F E7) y MOVNTPS [rax+16], xmm0 (0F 2B): 128 bits completos.
    alignas(16) static uint64_t nt4[4];
    for (auto& v : nt4) v = 0xFFFFFFFFFFFFFFFFull;
    CHECK(configureBackend(false));
    Bytes c3;
    movRegImm64(c3, 0, reinterpret_cast<uintptr_t>(nt4));
    movRegImm64(c3, 1, 0x0123456789ABCDEFull);
    append(c3, {0x66, 0x48, 0x0F, 0x6E, 0xC1});         // movq xmm0, rcx (alto = 0)
    append(c3, {0x66, 0x0F, 0xE7, 0x00});               // movntdq [rax], xmm0
    append(c3, {0x0F, 0x2B, 0x40, 0x10});               // movntps [rax+16], xmm0
    append(c3, {0xC3});
    runCode(c3);
    CHECK_EQ(nt4[0], 0x0123456789ABCDEFull);
    CHECK_EQ(nt4[1], 0ull);
    CHECK_EQ(nt4[2], 0x0123456789ABCDEFull);
    CHECK_EQ(nt4[3], 0ull);
}

// Rotaciones y shifts de 8/16 bits: resultado, bits altos intactos, memoria de 16 bits y
// SF/ZF/CF del ancho real. `rol ax, 8` es el _byteswap_ushort de MSVC (Unity lo usa para los
// u16 big-endian de los AssetBundles; con 0x0300 daba 0).
// SSE2/SSE4.1 de enteros añadidos tras jit_scan: cada caso carga xmm0=A, xmm1=B, xmm2=C,
// ejecuta la instrucción y compara el registro destino con la referencia en C++.
alignas(16) static uint8_t g_simdBuf[64];

void runSimdOp(std::initializer_list<uint8_t> op, int dstXmm, uint8_t out[16]) {
    CHECK(configureBackend(false));
    Bytes c;
    movRegImm64(c, 0, reinterpret_cast<uintptr_t>(g_simdBuf));
    append(c, {0xF3, 0x0F, 0x6F, 0x00});        // movdqu xmm0, [rax]
    append(c, {0xF3, 0x0F, 0x6F, 0x48, 0x10});  // movdqu xmm1, [rax+16]
    append(c, {0xF3, 0x0F, 0x6F, 0x50, 0x20});  // movdqu xmm2, [rax+32]
    append(c, op);
    append(c, {0xF3, 0x0F, 0x7F, static_cast<uint8_t>(0x40 | (dstXmm << 3)), 0x30});  // movdqu [rax+48], xmmN
    append(c, {0xC3});
    runCode(c);
    std::memcpy(out, g_simdBuf + 48, 16);
}

void testSse2IntegerOps() {
    uint8_t A[16], B[16], C[16];
    for (int i = 0; i < 16; ++i) {
        A[i] = static_cast<uint8_t>(0x9E + 37 * i);
        B[i] = static_cast<uint8_t>(0x15 + 91 * i);
        C[i] = static_cast<uint8_t>(0xC3 ^ (11 * i));
    }
    std::memcpy(g_simdBuf, A, 16);
    std::memcpy(g_simdBuf + 16, B, 16);
    std::memcpy(g_simdBuf + 32, C, 16);
    int16_t a16[8], b16[8];
    int32_t a32[4], b32[4], c32[4];
    std::memcpy(a16, A, 16); std::memcpy(b16, B, 16);
    std::memcpy(a32, A, 16); std::memcpy(b32, B, 16); std::memcpy(c32, C, 16);
    uint8_t out[16], ref[16];
    const auto check = [&](const char* name) {
        const bool ok = std::memcmp(out, ref, 16) == 0;
        if (!ok) std::printf("  SIMD %s distinto\n", name);
        CHECK(ok);
    };
    const auto sat16 = [](int32_t v) { return static_cast<int16_t>(std::clamp(v, -32768, 32767)); };
    int16_t r16[8];
    int32_t r32[4];
    uint64_t r64[2];

    for (int i = 0; i < 8; ++i) r16[i] = static_cast<int16_t>(a16[i] * b16[i]);
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0xD5, 0xC1}, 0, out); check("PMULLW");
    runSimdOp({0x66, 0x0F, 0xD5, 0x40, 0x10}, 0, out); check("PMULLW mem");
    for (int i = 0; i < 8; ++i) r16[i] = static_cast<int16_t>((a16[i] * b16[i]) >> 16);
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0xE5, 0xC1}, 0, out); check("PMULHW");
    for (int i = 0; i < 8; ++i)
        r16[i] = static_cast<int16_t>((static_cast<uint32_t>(static_cast<uint16_t>(a16[i])) * static_cast<uint16_t>(b16[i])) >> 16);
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0xE4, 0xC1}, 0, out); check("PMULHUW");
    for (int i = 0; i < 2; ++i)
        r64[i] = static_cast<uint64_t>(static_cast<uint32_t>(a32[2 * i])) * static_cast<uint32_t>(b32[2 * i]);
    std::memcpy(ref, r64, 16);
    runSimdOp({0x66, 0x0F, 0xF4, 0xC1}, 0, out); check("PMULUDQ");
    for (int i = 0; i < 2; ++i)
        r64[i] = static_cast<uint64_t>(static_cast<int64_t>(a32[2 * i]) * b32[2 * i]);
    std::memcpy(ref, r64, 16);
    runSimdOp({0x66, 0x0F, 0x38, 0x28, 0xC1}, 0, out); check("PMULDQ");
    for (int i = 0; i < 4; ++i) r32[i] = a16[2 * i] * b16[2 * i] + a16[2 * i + 1] * b16[2 * i + 1];
    std::memcpy(ref, r32, 16);
    runSimdOp({0x66, 0x0F, 0xF5, 0xC1}, 0, out); check("PMADDWD");
    for (int h = 0; h < 2; ++h) {
        uint64_t sum = 0;
        for (int i = 0; i < 8; ++i) sum += static_cast<uint64_t>(std::abs(A[8 * h + i] - B[8 * h + i]));
        r64[h] = sum;
    }
    std::memcpy(ref, r64, 16);
    runSimdOp({0x66, 0x0F, 0xF6, 0xC1}, 0, out); check("PSADBW");
    for (int i = 0; i < 4; ++i) { r16[i] = sat16(a32[i]); r16[4 + i] = sat16(b32[i]); }
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0x6B, 0xC1}, 0, out); check("PACKSSDW");
    for (int i = 0; i < 8; ++i) {
        ref[i] = static_cast<uint8_t>(std::clamp<int>(a16[i], 0, 255));
        ref[8 + i] = static_cast<uint8_t>(std::clamp<int>(b16[i], 0, 255));
    }
    runSimdOp({0x66, 0x0F, 0x67, 0xC1}, 0, out); check("PACKUSWB");
    for (int i = 0; i < 8; ++i) {
        ref[i] = static_cast<uint8_t>(static_cast<int8_t>(std::clamp<int>(a16[i], -128, 127)));
        ref[8 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::clamp<int>(b16[i], -128, 127)));
    }
    runSimdOp({0x66, 0x0F, 0x63, 0xC1}, 0, out); check("PACKSSWB");
    for (int i = 0; i < 16; ++i) ref[i] = static_cast<uint8_t>((A[i] + B[i] + 1) >> 1);
    runSimdOp({0x66, 0x0F, 0xE0, 0xC1}, 0, out); check("PAVGB");
    for (int i = 0; i < 16; ++i) ref[i] = static_cast<uint8_t>(A[i] > B[i] ? A[i] - B[i] : 0);
    runSimdOp({0x66, 0x0F, 0xD8, 0xC1}, 0, out); check("PSUBUSB");
    for (int i = 0; i < 8; ++i) r16[i] = sat16(a16[i] + b16[i]);
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0xED, 0xC1}, 0, out); check("PADDSW");
    for (int i = 0; i < 16; ++i) ref[i] = std::min(A[i], B[i]);
    runSimdOp({0x66, 0x0F, 0xDA, 0xC1}, 0, out); check("PMINUB");
    for (int i = 0; i < 8; ++i) r16[i] = std::abs(b16[i]) == 32768 ? b16[i] : static_cast<int16_t>(std::abs(b16[i]));
    std::memcpy(ref, r16, 16);
    runSimdOp({0x66, 0x0F, 0x38, 0x1D, 0xC1}, 0, out); check("PABSW");
    {
        int64_t q[2] = {b32[0], b32[1]};
        std::memcpy(ref, q, 16);
        runSimdOp({0x66, 0x0F, 0x38, 0x25, 0xC1}, 0, out); check("PMOVSXDQ");
        uint32_t z[4] = {B[0], B[1], B[2], B[3]};
        std::memcpy(ref, z, 16);
        runSimdOp({0x66, 0x0F, 0x38, 0x31, 0xC1}, 0, out); check("PMOVZXBD");
        runSimdOp({0x66, 0x0F, 0x38, 0x31, 0x40, 0x10}, 0, out); check("PMOVZXBD m32");
    }
    // BLENDVPS xmm1, xmm2 con máscara xmm0: lane i <- xmm2 si el signo de xmm0[i] está activo.
    for (int i = 0; i < 4; ++i) r32[i] = a32[i] < 0 ? c32[i] : b32[i];
    std::memcpy(ref, r32, 16);
    runSimdOp({0x66, 0x0F, 0x38, 0x14, 0xCA}, 1, out); check("BLENDVPS");

    // Desplazamientos por registro: la cuenta es el qword bajo de xmm1.
    const auto shiftCase = [&](uint8_t opc, uint64_t count, const char* name) {
        std::memcpy(g_simdBuf + 16, &count, 8);
        std::memset(g_simdBuf + 24, 0, 8);
        runSimdOp({0x66, 0x0F, opc, 0xC1}, 0, out);
        check(name);
    };
    for (int i = 0; i < 8; ++i) r16[i] = static_cast<int16_t>(static_cast<uint16_t>(a16[i]) >> 3);
    std::memcpy(ref, r16, 16);
    shiftCase(0xD1, 3, "PSRLW 3");
    for (int i = 0; i < 8; ++i) r16[i] = static_cast<int16_t>(static_cast<uint16_t>(a16[i]) << 5);
    std::memcpy(ref, r16, 16);
    shiftCase(0xF1, 5, "PSLLW 5");
    for (int i = 0; i < 4; ++i) r32[i] = a32[i] >> 31;
    std::memcpy(ref, r32, 16);
    shiftCase(0xE2, 40, "PSRAD 40");
    shiftCase(0xE2, 0x100000000ull, "PSRAD 2^32");
    std::memset(ref, 0, 16);
    shiftCase(0xF3, 70, "PSLLQ 70");
    shiftCase(0xD2, 32, "PSRLD 32");
    uint64_t a64[2];
    std::memcpy(a64, A, 16);
    for (auto& v : a64) v >>= 17;
    std::memcpy(ref, a64, 16);
    shiftCase(0xD3, 17, "PSRLQ 17");
    std::memcpy(g_simdBuf + 16, B, 16);
}

// Encadenado de bloques: un bucle de 3 millones de vueltas (Jcc hacia atrás enlazado consigo
// mismo, con vuelta al despachador cada kChainBudget saltos) da el mismo resultado con y sin
// encadenado; se informa del tiempo de cada modo.
void testBlockChaining() {
    double ms[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
        OryonEmitterBackend::chainBlocks.store(mode == 1);
        CHECK(configureBackend(false));
        Bytes c;
        movRegImm64(c, 1, 3000000);                 // rcx = 3e6
        append(c, {0x31, 0xC0});                    // xor eax, eax
        append(c, {0x83, 0xC0, 0x02});              // L: add eax, 2
        append(c, {0x48, 0x83, 0xE9, 0x01});        // sub rcx, 1
        append(c, {0x75, 0xF7});                    // jnz L
        append(c, {0xC3});
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t r = runCode(c);
        ms[mode] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK_EQ(static_cast<uint32_t>(r), 6000000u);
        CHECK_EQ(g_backend->context().gpr[1], 0ull);
    }
    OryonEmitterBackend::chainBlocks.store(true);
    std::printf("  encadenado: sin %.1f ms, con %.1f ms (x%.1f)\n", ms[0], ms[1], ms[1] > 0 ? ms[0] / ms[1] : 0.0);

    // CALL/RET en bucle (1 M llamadas): el RET usa la búsqueda en línea en la caché rápida.
    double ms2[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
        OryonEmitterBackend::inlineDispatch.store(mode == 1);
        CHECK(configureBackend(false));
        Bytes c;
        movRegImm64(c, 1, 1000000);                 // rcx = 1e6
        append(c, {0x31, 0xC0});                    // xor eax, eax
        append(c, {0xE8, 0x07, 0x00, 0x00, 0x00});  // L: call F (+7)
        append(c, {0x48, 0x83, 0xE9, 0x01});        // sub rcx, 1
        append(c, {0x75, 0xF5});                    // jnz L
        append(c, {0xC3});                          // ret
        append(c, {0x83, 0xC0, 0x03});              // F: add eax, 3
        append(c, {0xC3});                          // ret
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t r = runCode(c);
        ms2[mode] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK_EQ(static_cast<uint32_t>(r), 3000000u);
    }
    OryonEmitterBackend::inlineDispatch.store(true);
    std::printf("  RET en línea: sin %.1f ms, con %.1f ms (x%.1f)\n", ms2[0], ms2[1], ms2[1] > 0 ? ms2[0] / ms2[1] : 0.0);
}

uint64_t fastNativeMulAdd(uint64_t a, uint64_t b, uint64_t c, uint64_t d) { return a * b + c - d; }

// call [rip+disp] hacia una función de la lista rápida: BLR directo, RSP intacto, el bloque sigue.
void testFastNativeCall() {
    OryonEmitterBackend::registerFastNative(reinterpret_cast<uintptr_t>(&fastNativeMulAdd), false);
    for (int mode = 0; mode < 2; ++mode) {
        OryonEmitterBackend::fastNativeCalls.store(mode == 1);
        CHECK(configureBackend(false));
        Bytes c;
        movRegImm64(c, 1, 6);                              // rcx = 6
        movRegImm64(c, 2, 7);                              // rdx = 7
        movRegImm64(c, 8, 100);                            // r8 = 100
        movRegImm64(c, 9, 2);                              // r9 = 2
        append(c, {0x48, 0x89, 0xE3});                     // mov rbx, rsp
        append(c, {0xFF, 0x15, 0x0A, 0x00, 0x00, 0x00});   // call [rip+10] (IAT abajo)
        append(c, {0x48, 0x29, 0xE3});                     // sub rbx, rsp  (0 si la pila no cambió)
        append(c, {0x48, 0x01, 0xD8});                     // add rax, rbx
        append(c, {0xC3});                                 // ret
        append(c, {0x90, 0x90, 0x90});                     // relleno hasta el slot
        const auto fn = reinterpret_cast<uint64_t>(&fastNativeMulAdd);
        for (int i = 0; i < 8; ++i) c.push_back(static_cast<uint8_t>(fn >> (8 * i)));
        if (mode == 0) continue;  // sin la lista rápida la llamada iría al despachador de pruebas
        CHECK_EQ(runCode(c), 140);  // 6*7 + 100 - 2
        // Patrón de Mono: mov r11, imm64; call r11.
        CHECK(configureBackend(false));
        Bytes d;
        movRegImm64(d, 1, 3);
        movRegImm64(d, 2, 5);
        movRegImm64(d, 8, 1);
        movRegImm64(d, 9, 0);
        movRegImm64(d, 11, fn);                             // mov r11, imm64
        append(d, {0x41, 0xFF, 0xD3});                      // call r11
        append(d, {0xC3});
        CHECK_EQ(runCode(d), 16);  // 3*5 + 1 - 0
        // Stub de importación: call stub; add rax, 1; ret; stub: jmp [rip+0]; .quad fn
        CHECK(configureBackend(false));
        Bytes e;
        movRegImm64(e, 1, 4);
        movRegImm64(e, 2, 4);
        movRegImm64(e, 8, 0);
        movRegImm64(e, 9, 0);
        append(e, {0xE8, 0x05, 0x00, 0x00, 0x00});         // call stub (+5)
        append(e, {0x48, 0x83, 0xC0, 0x01});                // add rax, 1
        append(e, {0xC3});                                  // ret
        append(e, {0xFF, 0x25, 0x00, 0x00, 0x00, 0x00});   // stub: jmp [rip+0]
        for (int i = 0; i < 8; ++i) e.push_back(static_cast<uint8_t>(fn >> (8 * i)));
        CHECK_EQ(runCode(e), 17);  // 4*4 + 0 - 0, +1 tras volver
    }
    OryonEmitterBackend::fastNativeCalls.store(true);
}

// Caché JIT persistente: guardar, recargar en otro backend (otra dirección de ctx), reubicar
// direcciones nativas y descartar registros corruptos.
void testPersistentJitCache() {
    using hyperplay::JitDiskCache;
    using hyperplay::JitRecord;
    const std::string base = "/data/local/tmp/hp_jitcache_test";
    const std::string dat = base + ".dat", idx = base + ".idx";
    auto& disk = JitDiskCache::instance();
    disk.close();
    unlink(dat.c_str());
    unlink(idx.c_str());
    OryonEmitterBackend::registerFastNative(reinterpret_cast<uintptr_t>(&fastNativeMulAdd), false);
    OryonEmitterBackend::fastNativeCalls.store(true);

    // Código invitado fijo (misma RIP en cada "ejecución"): call [IAT] a una nativa rápida.
    Bytes c;
    movRegImm64(c, 1, 6);
    movRegImm64(c, 2, 7);
    movRegImm64(c, 8, 100);
    movRegImm64(c, 9, 2);
    append(c, {0x48, 0x89, 0xE3});
    append(c, {0xFF, 0x15, 0x0A, 0x00, 0x00, 0x00});
    append(c, {0x48, 0x29, 0xE3});
    append(c, {0x48, 0x01, 0xD8});
    append(c, {0xC3});
    append(c, {0x90, 0x90, 0x90});
    const auto fn = reinterpret_cast<uint64_t>(&fastNativeMulAdd);
    for (int i = 0; i < 8; ++i) c.push_back(static_cast<uint8_t>(fn >> (8 * i)));
    void* page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(page != MAP_FAILED);
    if (page == MAP_FAILED) return;
    std::memcpy(page, c.data(), c.size());
    const auto rip = reinterpret_cast<uint64_t>(page);

    const auto run = [&](OryonEmitterBackend& b) -> int64_t {
        OryonEmitterBackend* saved = g_backend;
        g_backend = &b;
        int64_t r = -1;
        if (configureBackend(false)) {
            auto* stack = static_cast<uint8_t*>(b.virtualStack());
            constexpr size_t kStackSize = 8u * 1024u * 1024u;
            std::memset(stack + kStackSize - 256, 0, 256);
            b.context().gpr[4] = reinterpret_cast<uintptr_t>(stack + kStackSize - 64);
            r = b.executeEntry(page);
        }
        g_backend = saved;
        return r;
    };

    // 1.ª ejecución: traduce y guarda.
    CHECK(disk.open(base, 0x1234));
    CHECK_EQ(run(*g_backend), 140);
    disk.flush();
    CHECK(disk.written() >= 1);

    // 2.ª ejecución: índice leído del disco, otro backend (otra dirección de runState_/suspendCount_).
    disk.close();
    CHECK(disk.open(base, 0x1234));
    {
        const uint64_t hits0 = disk.hits();
        auto other = std::make_unique<OryonEmitterBackend>();
        CHECK_EQ(run(*other), 140);
        CHECK(disk.hits() > hits0);
    }

    // Reubicación: registro con las direcciones nativas borradas; el cargador debe reescribirlas.
    JitRecord rec;
    const uint32_t cfg = g_backend->persistentConfigKey();
    const auto any = [](uint64_t, uint32_t) { return true; };
    CHECK(disk.lookup(rip, cfg, any, rec));
    CHECK(!rec.relocs.empty());
    CHECK_EQ(rec.deps.size(), 1);
    for (const auto& r : rec.relocs) {
        const uint32_t zero[4] = {0xD2800000u | r.reg, 0xF2A00000u | r.reg, 0xF2C00000u | r.reg, 0xF2E00000u | r.reg};
        std::memcpy(rec.host.data() + r.word * 4u, zero, sizeof(zero));
    }
    disk.close();
    unlink(dat.c_str());
    unlink(idx.c_str());
    CHECK(disk.open(base, 0x1234));
    disk.append(JitRecord(rec));
    disk.flush();
    disk.close();
    CHECK(disk.open(base, 0x1234));
    {
        const uint64_t hits0 = disk.hits();
        auto other = std::make_unique<OryonEmitterBackend>();
        CHECK_EQ(run(*other), 140);
        CHECK(disk.hits() > hits0);
    }

    // Otro build: la caché se vacía.
    disk.close();
    CHECK(disk.open(base, 0x9999));
    {
        JitRecord none;
        CHECK(!disk.lookup(rip, cfg, any, none));
    }
    CHECK_EQ(run(*g_backend), 140);  // retraduce y guarda de nuevo
    disk.flush();
    disk.close();

    // Registro corrupto: un byte del código cambiado -> se descarta y se retraduce.
    {
        const int fd = ::open(dat.c_str(), O_RDWR);
        CHECK(fd >= 0);
        uint8_t b = 0;
        CHECK(pread(fd, &b, 1, 32 + 56 + 40) == 1);
        b ^= 0x5A;
        CHECK(pwrite(fd, &b, 1, 32 + 56 + 40) == 1);
        ::close(fd);
    }
    CHECK(disk.open(base, 0x9999));
    {
        const uint64_t rej0 = disk.rejected(), hits0 = disk.hits();
        auto other = std::make_unique<OryonEmitterBackend>();
        CHECK_EQ(run(*other), 140);
        CHECK(disk.rejected() > rej0);
        CHECK_EQ(disk.hits(), hits0);
    }

    // Bytes invitados distintos en la misma RIP: no se usa el registro.
    {
        JitRecord r2;
        const auto never = [](uint64_t, uint32_t) { return false; };
        CHECK(!disk.lookup(rip, cfg, never, r2));
    }
    disk.close();
    unlink(dat.c_str());
    unlink(idx.c_str());
    munmap(page, 4096);
}

// Pila de retornos: CALL/RET anidados, recursión de 100 niveles (más que las 64 entradas) y una
// función que cambia su dirección de retorno (debe ir a la dirección real, no a la predicha).
void testReturnStack() {
    for (int ras = 0; ras < 2; ++ras) {
        OryonEmitterBackend::returnStack.store(ras != 0);
        // rax = 0; rcx = 100; call rec; ret.   rec: add rax, 2; dec rcx; jz out; call rec; out: add rax, 1; ret
        CHECK(configureBackend(false));
        Bytes r;
        movRegImm64(r, 0, 0);
        movRegImm64(r, 1, 100);
        append(r, {0xE8, 0x01, 0x00, 0x00, 0x00});  // call rec (+1)
        append(r, {0xC3});                          // ret
        // rec:
        append(r, {0x48, 0x83, 0xC0, 0x02});        // add rax, 2
        append(r, {0x48, 0xFF, 0xC9});              // dec rcx
        append(r, {0x74, 0x05});                    // jz out (+5)
        append(r, {0xE8, 0xF2, 0xFF, 0xFF, 0xFF});  // call rec (-14: vuelve al add)
        // out:
        append(r, {0x48, 0x83, 0xC0, 0x01});        // add rax, 1
        append(r, {0xC3});                          // ret
        CHECK_EQ(runCode(r), 300);                  // 100 x (2 + 1)
        // call f; mov eax, 1; ret; mov eax, 7; ret.  f: add qword [rsp], 6; ret  -> salta al mov eax,7
        CHECK(configureBackend(false));
        Bytes m;
        append(m, {0xE8, 0x0C, 0x00, 0x00, 0x00});  // call f (+12)
        append(m, {0xB8, 0x01, 0x00, 0x00, 0x00});  // mov eax, 1
        append(m, {0xC3});                          // ret
        append(m, {0xB8, 0x07, 0x00, 0x00, 0x00});  // mov eax, 7  (retorno + 6)
        append(m, {0xC3});                          // ret
        // f:
        append(m, {0x48, 0x83, 0x04, 0x24, 0x06});  // add qword [rsp], 6
        append(m, {0xC3});                          // ret
        CHECK_EQ(runCode(m), 7);
        // Llamadas en bucle (la pila de retornos se reutiliza muchas veces): 1000 x call g.
        CHECK(configureBackend(false));
        Bytes l;
        movRegImm64(l, 0, 0);
        movRegImm64(l, 1, 1000);
        append(l, {0xE8, 0x06, 0x00, 0x00, 0x00});  // loop: call g (+6)
        append(l, {0x48, 0xFF, 0xC9});              // dec rcx
        append(l, {0x75, 0xF6});                    // jnz loop (-10)
        append(l, {0xC3});                          // ret
        // g:
        append(l, {0x48, 0x83, 0xC0, 0x03});        // add rax, 3
        append(l, {0xC3});                          // ret
        CHECK_EQ(runCode(l), 3000);
    }
    OryonEmitterBackend::returnStack.store(true);
}

// PF perezoso: SETP/JP/JNP tras TEST (paridad del byte bajo) y tras UCOMISS (NaN -> PF=1).
void testLazyParity() {
    const uint32_t values[] = {0, 1, 3, 0x7F, 0xFF, 0x100, 0x1FE, 0xA5, 0x80000001u};
    for (uint32_t v : values) {
        const bool pf = (__builtin_popcount(v & 0xFFu) & 1) == 0;
        CHECK(configureBackend(false));
        Bytes s;
        movRegImm64(s, 2, 0);
        movRegImm64(s, 0, v);
        append(s, {0x85, 0xC0, 0x0F, 0x9A, 0xC2, 0x0F, 0xB6, 0xC2, 0xC3});  // test eax,eax; setp dl; movzx eax,dl
        CHECK_EQ(runCode(s), pf ? 1 : 0);
        for (int cc = 0xA; cc <= 0xB; ++cc) {
            CHECK(configureBackend(false));
            Bytes j;
            movRegImm64(j, 0, v);
            append(j, {0x85, 0xC0, static_cast<uint8_t>(0x70 + cc), 0x06, 0xB8, 0, 0, 0, 0, 0xC3, 0xB8, 1, 0, 0, 0, 0xC3});
            CHECK_EQ(runCode(j), (cc == 0xA) == pf ? 1 : 0);
        }
    }
    // ucomiss xmm0, xmm1: NaN -> PF=1; 1.0 vs 2.0 -> PF=0.
    for (int nan = 0; nan < 2; ++nan) {
        float in[2] = {1.0f, nan ? std::numeric_limits<float>::quiet_NaN() : 2.0f};
        CHECK(configureBackend(false));
        Bytes u;
        movRegImm64(u, 1, reinterpret_cast<uintptr_t>(in));
        movRegImm64(u, 2, 0);
        append(u, {0xF3, 0x0F, 0x10, 0x01, 0xF3, 0x0F, 0x10, 0x49, 0x04});  // movss xmm0,[rcx]; movss xmm1,[rcx+4]
        append(u, {0x0F, 0x2E, 0xC1, 0x0F, 0x9A, 0xC2, 0x0F, 0xB6, 0xC2, 0xC3});  // ucomiss; setp dl; movzx
        CHECK_EQ(runCode(u), nan ? 1 : 0);
    }
}

// Fusión de flags: Jcc/SETcc/CMOVcc tras CMP/TEST/ADD de 32 y 64 bits deben dar lo mismo que
// los flags x86 calculados en C, con y sin fusión.
void testFlagFusion() {
    struct Pair {
        uint64_t a, b;
    };
    const Pair pairs[] = {{5, 7},
                          {7, 5},
                          {9, 9},
                          {0, 0},
                          {0x8000000000000000ULL, 1},
                          {1, 0x8000000000000000ULL},
                          {0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL},
                          {0xFFFFFFFFFFFFFFFFULL, 1},
                          {0x00000000FFFFFFFFULL, 0x0000000100000001ULL},
                          {0x80000000ULL, 0x7FFFFFFFULL},
                          {0x7FFFFFFFULL, 0x80000000ULL},
                          {0xFFFFFFFFULL, 0xFFFFFFFFULL}};
    // op: 0 cmp, 1 test, 2 add. Devuelve CF, ZF, SF, OF.
    auto flags = [](int op, uint64_t a, uint64_t b, bool w64, bool& cf, bool& zf, bool& sf, bool& of) {
        const uint64_t mask = w64 ? ~0ULL : 0xFFFFFFFFULL;
        const uint64_t sign = w64 ? (1ULL << 63) : (1ULL << 31);
        a &= mask;
        b &= mask;
        uint64_t r = 0;
        if (op == 0) {
            r = (a - b) & mask;
            cf = a < b;
            of = ((a ^ b) & (a ^ r) & sign) != 0;
        } else if (op == 1) {
            r = a & b;
            cf = of = false;
        } else {
            r = (a + b) & mask;
            cf = r < a;
            of = (~(a ^ b) & (a ^ r) & sign) != 0;
        }
        zf = r == 0;
        sf = (r & sign) != 0;
    };
    auto evalCond = [](int cc, bool cf, bool zf, bool sf, bool of) {
        bool v = false;
        switch (cc >> 1) {
            case 0: v = of; break;
            case 1: v = cf; break;
            case 2: v = zf; break;
            case 3: v = cf || zf; break;
            case 4: v = sf; break;
            case 6: v = sf != of; break;
            case 7: v = zf || (sf != of); break;
            default: break;
        }
        return (cc & 1) ? !v : v;
    };
    const uint8_t opcode[3] = {0x39, 0x85, 0x01};  // cmp/test/add r/m, r (modrm C8: rax, rcx)
    for (int fuse = 0; fuse < 2; ++fuse) {
        OryonEmitterBackend::fuseFlags.store(fuse != 0);
        for (int op = 0; op < 3; ++op) {
            for (int w = 0; w < 2; ++w) {
                const bool w64 = w != 0;
                for (const Pair& p : pairs) {
                    bool cf, zf, sf, of;
                    flags(op, p.a, p.b, w64, cf, zf, sf, of);
                    for (int cc = 0; cc < 16; ++cc) {
                        if ((cc >> 1) == 5) continue;  // PF: no se fusiona ni se compara aquí
                        const bool expect = evalCond(cc, cf, zf, sf, of);
                        auto prefix = [&](Bytes& c) {
                            movRegImm64(c, 0, p.a);
                            movRegImm64(c, 1, p.b);
                            if (w64) c.push_back(0x48);
                            append(c, {opcode[op], 0xC8});
                        };
                        // SETcc dl; movzx eax, dl
                        CHECK(configureBackend(false));
                        Bytes s;
                        movRegImm64(s, 2, 0);
                        prefix(s);
                        append(s, {0x0F, static_cast<uint8_t>(0x90 + cc), 0xC2, 0x0F, 0xB6, 0xC2, 0xC3});
                        CHECK_EQ(runCode(s), expect ? 1 : 0);
                        // Jcc rel8 sobre "mov eax,0; ret"
                        CHECK(configureBackend(false));
                        Bytes j;
                        prefix(j);
                        append(j, {static_cast<uint8_t>(0x70 + cc), 0x06, 0xB8, 0, 0, 0, 0, 0xC3, 0xB8, 1, 0, 0, 0, 0xC3});
                        CHECK_EQ(runCode(j), expect ? 1 : 0);
                        // CMOVcc edx, ebx (32 bits: la mitad alta de rdx se pone a cero siempre)
                        CHECK(configureBackend(false));
                        Bytes m;
                        movRegImm64(m, 2, 0xAAAAAAAA55555555ULL);
                        movRegImm64(m, 3, 7);
                        prefix(m);
                        append(m, {0x0F, static_cast<uint8_t>(0x40 + cc), 0xD3, 0x48, 0x89, 0xD0, 0xC3});
                        CHECK_EQ(static_cast<uint64_t>(runCode(m)), expect ? 7ULL : 0x55555555ULL);
                        // CMOVcc rdx, rbx (64 bits)
                        CHECK(configureBackend(false));
                        Bytes q;
                        movRegImm64(q, 2, 0xAAAAAAAA55555555ULL);
                        movRegImm64(q, 3, 7);
                        prefix(q);
                        append(q, {0x48, 0x0F, static_cast<uint8_t>(0x40 + cc), 0xD3, 0x48, 0x89, 0xD0, 0xC3});
                        CHECK_EQ(static_cast<uint64_t>(runCode(q)), expect ? 7ULL : 0xAAAAAAAA55555555ULL);
                    }
                }
            }
        }
    }
    // ADD/SUB/CMP r, imm (camino rápido con inmediato negativo = operación inversa): resultado y
    // las 15 condiciones contra el modelo en C (sub/cmp = op 0, add = op 2).
    const int64_t imms[] = {1, 5, 127, -1, -5, -128, 4095, -4095, 4096, 100000, -100000};
    const uint64_t vals[] = {0, 1, 5, 0x7F, 0x80000000ULL, 0xFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL,
                             0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 4095, 4096};
    for (int w = 0; w < 2; ++w) {
        const bool w64 = w != 0;
        for (int kind = 0; kind < 3; ++kind) {  // 0 add, 1 sub, 2 cmp
            const uint8_t ext = kind == 0 ? 0 : (kind == 1 ? 5 : 7);
            for (int64_t im : imms) {
                for (uint64_t v : vals) {
                    const uint64_t mask = w64 ? ~0ULL : 0xFFFFFFFFULL;
                    const uint64_t b = static_cast<uint64_t>(im) & mask;
                    bool cf, zf, sf, of;
                    flags(kind == 0 ? 2 : 0, v, b, w64, cf, zf, sf, of);
                    const uint64_t res = kind == 0 ? ((v & mask) + b) & mask : (kind == 1 ? ((v & mask) - b) & mask : v);
                    auto body = [&](Bytes& c) {
                        movRegImm64(c, 0, v);
                        if (w64) c.push_back(0x48);
                        if (im >= -128 && im <= 127) {
                            append(c, {0x83, static_cast<uint8_t>(0xC0 | (ext << 3)), static_cast<uint8_t>(im)});
                        } else {
                            append(c, {0x81, static_cast<uint8_t>(0xC0 | (ext << 3))});
                            const uint32_t i32 = static_cast<uint32_t>(im);
                            for (int k = 0; k < 4; ++k) c.push_back(static_cast<uint8_t>(i32 >> (8 * k)));
                        }
                    };
                    CHECK(configureBackend(false));
                    Bytes r;
                    body(r);
                    append(r, {0xC3});
                    CHECK_EQ(static_cast<uint64_t>(runCode(r)), w64 ? res : (kind == 2 ? v : res));
                    for (int cc = 0; cc < 16; ++cc) {
                        if ((cc >> 1) == 5) continue;
                        CHECK(configureBackend(false));
                        Bytes s;
                        movRegImm64(s, 2, 0);
                        body(s);
                        append(s, {0x0F, static_cast<uint8_t>(0x90 + cc), 0xC2, 0x0F, 0xB6, 0xC2, 0xC3});
                        CHECK_EQ(runCode(s), evalCond(cc, cf, zf, sf, of) ? 1 : 0);
                    }
                }
            }
        }
    }
    OryonEmitterBackend::fuseFlags.store(true);
}

// RSQRTSS/RCPSS/RSQRTPS: x86 garantiza error relativo <= 1.5*2^-12 (~3.7e-4); la estimación
// sola de ARM (~8 bits) no lo cumplía.
void testReciprocalPrecision() {
    const float inputs[] = {4.0f, 0.37f, 123456.0f, 1.0e-3f, 7.0f};
    for (float x : inputs) {
        float in[4] = {x, x * 2.0f, x * 3.0f, x * 5.0f};
        alignas(16) float out[4] = {};
        float outRcp = 0;
        CHECK(configureBackend(false));
        Bytes c;
        movRegImm64(c, 1, reinterpret_cast<uintptr_t>(in));
        movRegImm64(c, 2, reinterpret_cast<uintptr_t>(out));
        movRegImm64(c, 8, reinterpret_cast<uintptr_t>(&outRcp));
        append(c, {0x0F, 0x10, 0x01});        // movups xmm0, [rcx]
        append(c, {0x0F, 0x52, 0xC8});        // rsqrtps xmm1, xmm0
        append(c, {0x0F, 0x11, 0x0A});        // movups [rdx], xmm1
        append(c, {0xF3, 0x0F, 0x53, 0xD0});  // rcpss xmm2, xmm0
        append(c, {0xF3, 0x41, 0x0F, 0x11, 0x10});  // movss [r8], xmm2
        append(c, {0xF3, 0x0F, 0x52, 0xD8});  // rsqrtss xmm3, xmm0
        append(c, {0xF3, 0x0F, 0x11, 0x5A, 0x04});  // movss [rdx+4], xmm3 (pisa el carril 1)
        append(c, {0xC3});
        runCode(c);
        for (int i = 0; i < 4; ++i) {
            const double v = i == 1 ? x : in[i];
            const double rel = std::fabs(out[i] * std::sqrt(v) - 1.0);
            CHECK(rel < 3.7e-4);
        }
        CHECK(std::fabs(outRcp * x - 1.0) < 3.7e-4);
    }
}

// Flags muertos: una escritura seguida de otra se omite, salvo que la segunda sea condicional.
void testDeadFlags() {
    // cmp eax, eax (ZF=1); shl ebx, cl con cl=0 (no toca flags); setz al -> 1
    CHECK(configureBackend(false));
    Bytes a;
    movRegImm64(a, 3, 5);
    movRegImm64(a, 1, 0);
    append(a, {0x39, 0xC0});              // cmp eax, eax
    append(a, {0xD3, 0xE3});              // shl ebx, cl (cl = 0)
    append(a, {0x0F, 0x94, 0xC0});        // setz al
    append(a, {0x48, 0x0F, 0xB6, 0xC0});  // movzx rax, al
    append(a, {0xC3});
    CHECK_EQ(runCode(a), 1);
    // add eax,1 (ZF=0) seguido de sub eax,eax (ZF=1): el primero es muerto; setz -> 1
    CHECK(configureBackend(false));
    Bytes b;
    movRegImm64(b, 0, 7);
    append(b, {0x83, 0xC0, 0x01});        // add eax, 1
    append(b, {0x29, 0xC0});              // sub eax, eax
    append(b, {0x0F, 0x94, 0xC0});        // setz al
    append(b, {0x48, 0x0F, 0xB6, 0xC0});
    append(b, {0xC3});
    CHECK_EQ(runCode(b), 1);
    // inc conserva CF: stc; inc eax; setc al -> 1 (inc lee el CF anterior)
    CHECK(configureBackend(false));
    Bytes c;
    movRegImm64(c, 0, 0);
    append(c, {0xF9});                    // stc
    append(c, {0xFF, 0xC0});              // inc eax
    append(c, {0x0F, 0x92, 0xC0});        // setc al
    append(c, {0x48, 0x0F, 0xB6, 0xC0});
    append(c, {0xC3});
    CHECK_EQ(runCode(c), 1);
}

void testShiftRotate16() {
    alignas(8) static uint8_t mem[8];
    auto run = [&](uint64_t rax, std::initializer_list<uint8_t> op, uint8_t cl = 0) {
        CHECK(configureBackend(false));
        Bytes code;
        movRegImm64(code, 0, rax);
        movRegImm64(code, 6, reinterpret_cast<uintptr_t>(mem));
        movRegImm64(code, 3, 0);
        movRegImm64(code, 2, 0);
        if (cl != 0) append(code, {0xB1, cl});  // mov cl, imm8
        code.insert(code.end(), op.begin(), op.end());
        append(code, {0x0F, 0x92, 0xC3});  // setc bl
        append(code, {0x0F, 0x94, 0xC6});  // sete dh
        append(code, {0x0F, 0x98, 0xC2});  // sets dl
        append(code, {0xC3});
        runCode(code);
    };
    auto& g = g_backend->context().gpr;
    const uint64_t hi = 0x1111222233330000ull;
    run(hi | 0x0300, {0x66, 0xC1, 0xC0, 0x08});  // rol ax, 8
    CHECK_EQ(g[0], hi | 0x0003);
    run(hi | 0x0300, {0x66, 0xC1, 0xC8, 0x08});  // ror ax, 8
    CHECK_EQ(g[0], hi | 0x0003);
    run(hi | 0x1234, {0x66, 0xC1, 0xC0, 0x04});  // rol ax, 4
    CHECK_EQ(g[0], hi | 0x2341);
    run(hi | 0x0001, {0x66, 0xD1, 0xC8});  // ror ax, 1 -> 0x8000, CF=1
    CHECK_EQ(g[0], hi | 0x8000);
    CHECK_EQ(g[3] & 0xFF, 1u);
    run(hi | 0xABCD, {0x66, 0xD3, 0xC0}, 8);  // rol ax, cl(8)
    CHECK_EQ(g[0], hi | 0xCDAB);
    // rol word [rsi], 8: solo 2 bytes de memoria.
    std::memset(mem, 0xEE, sizeof(mem));
    mem[0] = 0x00;
    mem[1] = 0x03;
    run(0, {0x66, 0xC1, 0x06, 0x08});
    CHECK(mem[0] == 0x03 && mem[1] == 0x00 && mem[2] == 0xEE && mem[3] == 0xEE);
    // rol word [mem], cl
    mem[0] = 0x34;
    mem[1] = 0x12;
    run(0, {0x66, 0xD3, 0x06}, 8);
    CHECK(mem[0] == 0x12 && mem[1] == 0x34 && mem[2] == 0xEE && mem[3] == 0xEE);
    // shl ax, 1 con 0x8000: resultado 0 -> ZF=1, CF=1, SF=0.
    run(hi | 0x8000, {0x66, 0xD1, 0xE0});
    CHECK_EQ(g[0], hi);
    CHECK_EQ(g[3] & 0xFF, 1u);
    CHECK_EQ((g[2] >> 8) & 0xFF, 1u);
    CHECK_EQ(g[2] & 0xFF, 0u);
    // shl ax, 1 con 0x4000 -> 0x8000: SF=1.
    run(hi | 0x4000, {0x66, 0xD1, 0xE0});
    CHECK_EQ(g[0], hi | 0x8000);
    CHECK_EQ(g[2] & 0xFF, 1u);
    // sar ax, 4 / sar ax, cl con 0x8000 -> 0xF800 (signo de 16 bits).
    run(hi | 0x8000, {0x66, 0xC1, 0xF8, 0x04});
    CHECK_EQ(g[0], hi | 0xF800);
    run(hi | 0x8000, {0x66, 0xD3, 0xF8}, 4);
    CHECK_EQ(g[0], hi | 0xF800);
    CHECK_EQ(g[2] & 0xFF, 1u);
    // shr word [mem], cl: no toca los bytes siguientes.
    mem[0] = 0x00;
    mem[1] = 0x80;
    run(0, {0x66, 0xD3, 0x2E}, 4);
    CHECK(mem[0] == 0x00 && mem[1] == 0x08 && mem[2] == 0xEE && mem[3] == 0xEE);
    // 8 bits: sar al, 1 con 0x80 -> 0xC0, SF=1; shl al, cl(1) con 0x40 -> 0x80, SF=1.
    run(hi | 0x80, {0xD0, 0xF8});
    CHECK_EQ(g[0], hi | 0xC0);
    CHECK_EQ(g[2] & 0xFF, 1u);
    run(hi | 0x40, {0xD2, 0xE0}, 1);
    CHECK_EQ(g[0], hi | 0x80);
    CHECK_EQ(g[2] & 0xFF, 1u);
    // shl ah, cl (registro alto sin REX).
    run(hi | 0x0100, {0xD2, 0xE4}, 3);
    CHECK_EQ(g[0], hi | 0x0800);
}

// Compresor LZ4 de bloque (voraz, tabla hash de 4 bytes) que respeta las reglas del formato:
// los últimos 5 bytes van como literales y ningún match empieza en los últimos 12.
std::vector<uint8_t> lz4CompressBlock(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out;
    const size_t n = in.size();
    std::vector<int64_t> table(1u << 14, -1);
    auto hash4 = [&](size_t p) {
        uint32_t v;
        std::memcpy(&v, in.data() + p, 4);
        return (v * 2654435761u) >> 18;
    };
    auto putLen = [&](size_t len) {
        while (len >= 255) { out.push_back(255); len -= 255; }
        out.push_back(static_cast<uint8_t>(len));
    };
    size_t anchor = 0, p = 0;
    const size_t matchLimit = n >= 12 ? n - 12 : 0;
    while (p < matchLimit) {
        const uint32_t h = hash4(p);
        const int64_t cand = table[h];
        table[h] = static_cast<int64_t>(p);
        if (cand < 0 || p - static_cast<size_t>(cand) > 65535 ||
            std::memcmp(in.data() + cand, in.data() + p, 4) != 0) {
            ++p;
            continue;
        }
        size_t ml = 4;
        while (p + ml < n - 5 && in[cand + ml] == in[p + ml]) ++ml;
        const size_t lit = p - anchor;
        const size_t mlCode = ml - 4;
        out.push_back(static_cast<uint8_t>(((lit >= 15 ? 15 : lit) << 4) | (mlCode >= 15 ? 15 : mlCode)));
        if (lit >= 15) putLen(lit - 15);
        out.insert(out.end(), in.begin() + anchor, in.begin() + p);
        const size_t off = p - static_cast<size_t>(cand);
        out.push_back(static_cast<uint8_t>(off & 0xFF));
        out.push_back(static_cast<uint8_t>(off >> 8));
        if (mlCode >= 15) putLen(mlCode - 15);
        p += ml;
        anchor = p;
    }
    const size_t lit = n - anchor;
    out.push_back(static_cast<uint8_t>((lit >= 15 ? 15 : lit) << 4));
    if (lit >= 15) putLen(lit - 15);
    out.insert(out.end(), in.begin() + anchor, in.end());
    return out;
}

// Descompresor LZ4 de referencia (byte a byte, sin atajos) para validar al de Unity.
std::vector<uint8_t> lz4DecompressRef(const std::vector<uint8_t>& src, size_t outSize) {
    std::vector<uint8_t> out;
    out.reserve(outSize);
    size_t i = 0;
    while (i < src.size()) {
        const uint8_t tok = src[i++];
        size_t lit = tok >> 4;
        if (lit == 15) {
            uint8_t b;
            do { b = src[i++]; lit += b; } while (b == 255);
        }
        out.insert(out.end(), src.begin() + static_cast<long>(i), src.begin() + static_cast<long>(i + lit));
        i += lit;
        if (i >= src.size()) break;
        const size_t off = src[i] | (src[i + 1] << 8);
        i += 2;
        size_t ml = tok & 15u;
        if (ml == 15) {
            uint8_t b;
            do { b = src[i++]; ml += b; } while (b == 255);
        }
        ml += 4;
        for (size_t k = 0; k < ml; ++k) out.push_back(out[out.size() - off]);
    }
    out.resize(outSize);
    return out;
}

int64_t runUnityLz4(const Bytes& fn, const std::vector<uint8_t>& comp, size_t outSize, std::vector<uint8_t>& dst);

// Descompresor LZ4 real de UnityPlayer.dll (Cuphead, Unity 2017.4, LZ4_decompress_safe r1xx)
// ejecutado bajo el JIT. Los bytes no se incluyen en el repositorio: se leen de
// /data/local/tmp/unity_lz4.bin (0x300 bytes desde VA 0x180B851F0). Se parchean la carga de la
// cookie /GS, la llamada a __security_check_cookie y la de memcpy (a un stub rep movsb).
void testUnityLz4() {
    FILE* f = std::fopen("/data/local/tmp/unity_lz4.bin", "rb");
    if (f == nullptr) {
        std::printf("LZ4 Unity: sin /data/local/tmp/unity_lz4.bin (omitido)\n");
        return;
    }
    Bytes fn(0x300);
    const size_t got = std::fread(fn.data(), 1, fn.size(), f);
    std::fclose(f);
    CHECK_EQ(got, fn.size());
    if (got != fn.size()) return;
    // mov rax,[rip+cookie] -> xor rax,rax; nopl 0(rax)
    const uint8_t cookie[7] = {0x48, 0x31, 0xC0, 0x0F, 0x1F, 0x40, 0x00};
    std::memcpy(fn.data() + 0x12, cookie, 7);
    // call __security_check_cookie -> nopl 0(rax,rax)
    const uint8_t nop5[5] = {0x0F, 0x1F, 0x44, 0x00, 0x00};
    std::memcpy(fn.data() + 0x2EB, nop5, 5);

    // Datos parecidos a un SerializedFile: ceros, cadenas, enteros, repeticiones con
    // offsets 1..40 y tramos pseudoaleatorios sin matches.
    std::vector<uint8_t> plain;
    uint32_t rng = 12345;
    auto rnd = [&] { rng = rng * 1103515245u + 12345u; return static_cast<uint8_t>(rng >> 16); };
    const char* ver = "2017.4.9f1";
    for (int round = 0; round < 60; ++round) {
        for (int i = 0; i < 20; ++i) plain.push_back(static_cast<uint8_t>(i % 3 == 0 ? 0 : round));
        plain.insert(plain.end(), ver, ver + std::strlen(ver) + 1);
        for (int i = 0; i < 4; ++i) plain.push_back(i == 0 ? 5 : 0);
        for (int i = 0; i < 9; ++i) plain.push_back(0);
        const int period = 1 + round % 40;
        std::vector<uint8_t> pat(static_cast<size_t>(period));
        for (auto& c : pat) c = rnd();
        for (int i = 0; i < 70; ++i) plain.push_back(pat[static_cast<size_t>(i % period)]);
        for (int i = 0; i < 1 + round % 23; ++i) plain.push_back(rnd());
    }
    auto check = [&](const char* name, const std::vector<uint8_t>& comp, const std::vector<uint8_t>& expect) {
        std::vector<uint8_t> dst;
        const int64_t ret = runUnityLz4(fn, comp, expect.size(), dst);
        CHECK_EQ(ret, static_cast<int64_t>(expect.size()));
        size_t firstBad = expect.size();
        for (size_t i = 0; i < expect.size(); ++i) {
            if (dst[i] != expect[i]) { firstBad = i; break; }
        }
        CHECK_EQ(firstBad, expect.size());
        if (ret != static_cast<int64_t>(expect.size()) || firstBad != expect.size()) {
            std::printf("LZ4 Unity %s: ret=%lld primer byte distinto en %zu (de %zu)\n", name,
                        static_cast<long long>(ret), firstBad, expect.size());
        }
    };
    check("sintético", lz4CompressBlock(plain), plain);
    // Datos reales del AssetBundle atlas_player (si están en el dispositivo).
    auto readAll = [](const char* path, std::vector<uint8_t>& out) {
        FILE* fp = std::fopen(path, "rb");
        if (fp == nullptr) return false;
        uint8_t buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) out.insert(out.end(), buf, buf + n);
        std::fclose(fp);
        return true;
    };
    std::vector<uint8_t> real;
    if (readAll("/data/local/tmp/ap_blocksinfo.lz4", real)) check("blocksinfo", real, lz4DecompressRef(real, 0x29E3));
    real.clear();
    if (readAll("/data/local/tmp/ap_block0.lz4", real)) check("bloque0", real, lz4DecompressRef(real, 131072));
}

// Ejecuta el descompresor de Unity (bytes ya parcheados) sobre comp; devuelve EAX.
int64_t runUnityLz4(const Bytes& fn, const std::vector<uint8_t>& comp, size_t outSize, std::vector<uint8_t>& dst) {
    const size_t base = 64;  // prólogo antes de la función
    Bytes code;
    movRegImm64(code, 1, 0);  // rcx = src (se parchea abajo)
    movRegImm64(code, 2, 0);  // rdx = dst
    movRegImm64(code, 8, comp.size());
    movRegImm64(code, 9, outSize);
    const size_t callAt = code.size();
    append(code, {0xE8, 0, 0, 0, 0, 0xC3});
    code.resize(base, 0x90);
    code.insert(code.end(), fn.begin(), fn.end());
    const size_t stubAt = code.size();
    append(code, {0x57, 0x56, 0x48, 0x89, 0xCF, 0x48, 0x89, 0xD6, 0x4C, 0x89, 0xC1, 0xF3, 0xA4, 0x5E, 0x5F, 0xC3});
    auto rel32 = [&](size_t at, size_t target) {
        const int32_t d = static_cast<int32_t>(static_cast<int64_t>(target) - static_cast<int64_t>(at + 5));
        std::memcpy(code.data() + at + 1, &d, 4);
    };
    rel32(callAt, base);
    rel32(base + 0x2BC, stubAt);

    std::vector<uint8_t> src = comp;
    src.resize(comp.size() + 64, 0);
    dst.assign(outSize + 64, 0xCC);
    const uint64_t srcPtr = reinterpret_cast<uintptr_t>(src.data());
    const uint64_t dstPtr = reinterpret_cast<uintptr_t>(dst.data());
    std::memcpy(code.data() + 2, &srcPtr, 8);
    std::memcpy(code.data() + 12, &dstPtr, 8);

    if (!configureBackend(false)) return -1000;
    runCode(code);
    return static_cast<int32_t>(g_backend->context().gpr[0]);
}

// CMPXCHG16B atómico (CASPAL): éxito y fallo con flags y registros.
void testCmpxchg16b() {
    alignas(16) static uint64_t mem[2];
    auto run = [&](uint64_t rax, uint64_t rdx) {
        CHECK(configureBackend(false));
        Bytes code;
        movRegImm64(code, 7, reinterpret_cast<uintptr_t>(mem));  // rdi
        movRegImm64(code, 0, rax);
        movRegImm64(code, 2, rdx);
        movRegImm64(code, 3, 0x3333);  // rbx
        movRegImm64(code, 1, 0x4444);  // rcx
        append(code, {0xF0, 0x48, 0x0F, 0xC7, 0x0F});  // lock cmpxchg16b [rdi]
        append(code, {0xC3});
        runCode(code);
    };
    mem[0] = 0x11;
    mem[1] = 0x22;
    run(0x11, 0x22);
    CHECK(mem[0] == 0x3333 && mem[1] == 0x4444);
    CHECK((g_backend->context().rflags & (1ull << 6)) != 0);  // ZF=1
    run(0x11, 0x22);  // ya no coincide
    CHECK(mem[0] == 0x3333 && mem[1] == 0x4444);
    CHECK((g_backend->context().rflags & (1ull << 6)) == 0);  // ZF=0
    CHECK(g_backend->context().gpr[0] == 0x3333 && g_backend->context().gpr[2] == 0x4444);
}

// BC7: coherencia de tablas y bloques de modo 6 construidos bit a bit.
namespace {
struct BitWriter {
    uint8_t b[16] = {};
    uint32_t pos = 0;
    void put(uint32_t v, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i, ++pos) {
            if ((v >> i) & 1u) b[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
        }
    }
};
}  // namespace

void testBc7() {
    using namespace hyperplay::d3d11::bc7;
    bool tablesOk = true;
    for (int p = 0; p < 64; ++p) {
        tablesOk = tablesOk && kPartition2[p][0] == 0 && kPartition2[p][kAnchor2[p]] == 1;
        tablesOk = tablesOk && kPartition3[p][0] == 0 && kPartition3[p][kAnchor3a[p]] == 1 &&
                   kPartition3[p][kAnchor3b[p]] == 2;
    }
    CHECK(tablesOk);
    // Modo 6 sólido: R=127/127, G=B=0, A=127/127, P=1/1 -> (255, 0, 0, 255).
    {
        BitWriter w;
        w.put(1u << 6, 7);
        w.put(127, 7); w.put(127, 7);  // R0 R1
        w.put(0, 7); w.put(0, 7);      // G
        w.put(0, 7); w.put(0, 7);      // B
        w.put(127, 7); w.put(127, 7);  // A
        w.put(1, 1); w.put(1, 1);      // P0 P1
        uint8_t px[64];
        decodeBlock(w.b, px);
        // El P-bit se añade a todos los canales: G=B=(0<<1|1)=1.
        CHECK(px[0] == 255 && px[1] == 1 && px[2] == 1 && px[3] == 255);
        CHECK(px[60] == 255 && px[63] == 255);
    }
    // Modo 6 interpolado: R0=0 (P0=0) -> 0, R1=127 (P1=1) -> 255; píxel1 índice 15, píxel2 índice 8.
    {
        BitWriter w;
        w.put(1u << 6, 7);
        w.put(0, 7); w.put(127, 7);
        w.put(0, 7); w.put(0, 7);
        w.put(0, 7); w.put(0, 7);
        w.put(127, 7); w.put(127, 7);
        w.put(0, 1); w.put(1, 1);
        w.put(0, 3);   // píxel 0 (ancla, 3 bits)
        w.put(15, 4);  // píxel 1
        w.put(8, 4);   // píxel 2
        uint8_t px[64];
        decodeBlock(w.b, px);
        CHECK(px[0] == 0);
        CHECK(px[4] == 255);
        CHECK(px[8] == 135);  // ((64-34)*0 + 34*255 + 32) >> 6
        CHECK(px[3] == 254 || px[3] == 255);  // alfa: 127<<1|P (254 o 255)
    }
    uint8_t zero[16] = {};
    uint8_t px[64];
    decodeBlock(zero, px);
    CHECK(px[0] == 0 && px[3] == 0);  // modo inválido
}

// Rendimiento del despachador: bucle `dec rcx; jnz` de 5M iteraciones (1 bloque por vuelta).
void benchmarkDispatch() {
    Bytes code;
    movRegImm64(code, 1, 5000000);
    append(code, {0x48, 0xFF, 0xC9});  // dec rcx
    append(code, {0x75, 0xFB});        // jnz -5
    append(code, {0xC3});
    CHECK(configureBackend(false));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    runCode(code);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = static_cast<double>(t1.tv_sec - t0.tv_sec) + static_cast<double>(t1.tv_nsec - t0.tv_nsec) / 1e9;
    std::printf("BENCH despachador: 5M bloques en %.3f s = %.1f M bloques/s (%.1f ns/bloque)\n", secs,
                5.0 / secs, secs * 1e9 / 5e6);
    CHECK_EQ(g_backend->context().gpr[1], 0u);
}

std::u16string u16(const char* s) {
    std::u16string out;
    while (*s != 0) out.push_back(static_cast<char16_t>(static_cast<unsigned char>(*s++)));
    return out;
}

// Registro de Windows (PlayerPrefs de Unity): crear clave, valores, consulta en dos pasos,
// enumeración, persistencia a archivo y claves HKLM sembradas.
void testWin32Registry() {
    using namespace win32reg;
    const std::string file = "/data/local/tmp/oryon_test_registry.hpreg";
    std::remove(file.c_str());
    const uint64_t kHkcu = 0xFFFFFFFF80000001ull;  // HKEY_CURRENT_USER sign-extended (x64)
    {
        Registry reg;
        reg.setBackingFile(file);
        uint64_t h = 0;
        uint32_t disp = 0;
        CHECK_EQ(reg.open(kHkcu, u"Software\\Team Cherry\\Hollow Knight", false, &h), kFileNotFound);
        CHECK_EQ(reg.open(kHkcu, u"Software\\Team Cherry\\Hollow Knight", true, &h, &disp), kSuccess);
        CHECK_EQ(disp, kCreatedNewKey);
        const uint32_t vol = 7;
        CHECK_EQ(reg.setValue(h, u"MusicVolume_h12345", kRegDword, reinterpret_cast<const uint8_t*>(&vol), 4),
                 kSuccess);
        const char lang[] = "EN";
        CHECK_EQ(reg.setValue(h, u"Lang_h1", kRegBinary, reinterpret_cast<const uint8_t*>(lang), 3), kSuccess);
        uint32_t type = 0, size = 0;
        CHECK_EQ(reg.queryValue(h, u"musicvolume_H12345", &type, nullptr, &size), kSuccess);
        CHECK_EQ(type, kRegDword);
        CHECK_EQ(size, 4u);
        uint8_t small[2] = {};
        size = 2;
        CHECK_EQ(reg.queryValue(h, u"Lang_h1", nullptr, small, &size), kMoreData);
        CHECK_EQ(size, 3u);
        CHECK_EQ(reg.queryValue(h, u"Missing", nullptr, nullptr, &size), kFileNotFound);
        char16_t name[64] = {};
        uint32_t nameChars = 64;
        CHECK_EQ(reg.enumValue(h, 0, name, &nameChars, &type, nullptr, nullptr), kSuccess);
        CHECK(std::u16string(name) == u"Lang_h1");
        CHECK_EQ(nameChars, 7u);
        nameChars = 64;
        CHECK_EQ(reg.enumValue(h, 2, name, &nameChars, nullptr, nullptr, nullptr), kNoMoreItems);
        CHECK_EQ(reg.deleteValue(h, u"Lang_h1"), kSuccess);
        CHECK_EQ(reg.close(h), kSuccess);
        CHECK_EQ(reg.close(h), kInvalidHandle);
        // Subclaves: enumeración de Team Cherry
        uint64_t tc = 0;
        CHECK_EQ(reg.open(kHkcu, u"Software\\Team Cherry", false, &tc), kSuccess);
        nameChars = 64;
        CHECK_EQ(reg.enumKey(tc, 0, name, &nameChars), kSuccess);
        CHECK(std::u16string(name) == u"Hollow Knight");
        uint32_t subKeys = 0, values = 0;
        CHECK_EQ(reg.queryInfo(tc, &subKeys, nullptr, &values, nullptr, nullptr), kSuccess);
        CHECK_EQ(subKeys, 1u);
        CHECK_EQ(values, 0u);
        CHECK_EQ(reg.deleteKey(kHkcu, u"Software\\Team Cherry"), 5u);  // tiene subclaves
    }
    {
        Registry reg;  // proceso nuevo: recupera los PlayerPrefs del archivo
        reg.setBackingFile(file);
        uint64_t h = 0;
        uint32_t disp = 0;
        CHECK_EQ(reg.open(kHkcu, u"software\\team cherry\\hollow knight", true, &h, &disp), kSuccess);
        CHECK_EQ(disp, kOpenedExistingKey);
        uint32_t vol = 0, size = 4;
        CHECK_EQ(reg.queryValue(h, u"MusicVolume_h12345", nullptr, reinterpret_cast<uint8_t*>(&vol), &size),
                 kSuccess);
        CHECK_EQ(vol, 7u);
        uint32_t size2 = 0;
        CHECK_EQ(reg.queryValue(h, u"Lang_h1", nullptr, nullptr, &size2), kFileNotFound);
        uint64_t cpu = 0;
        CHECK_EQ(reg.open(0x80000002ull, u"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", false, &cpu),
                 kSuccess);
        uint32_t mhz = 0;
        size = 4;
        CHECK_EQ(reg.queryValue(cpu, u"~MHz", nullptr, reinterpret_cast<uint8_t*>(&mhz), &size), kSuccess);
        CHECK_EQ(mhz, 4320u);
        CHECK_EQ(reg.open(0x1234, u"x", false, &h), kInvalidHandle);
    }
    std::remove(file.c_str());
}

void testShlwapiPaths() {
    namespace wp = hyperplay::win32path;
    char16_t buf[wp::kMaxPath] = {};

    CHECK(wp::canonicalizeInto(buf, u16("D:\\Cuphead.exe\\..\\Cuphead_Data\\.\\boot.config").c_str()));
    CHECK(std::u16string(buf) == u16("D:\\Cuphead_Data\\boot.config"));
    CHECK(wp::canonicalizeInto(buf, u16("D:/a/b/../../c/").c_str()));
    CHECK(std::u16string(buf) == u16("D:\\c\\"));
    CHECK(wp::canonicalizeInto(buf, u16("C:\\..\\..\\x").c_str()));
    CHECK(std::u16string(buf) == u16("C:\\x"));
    CHECK(wp::canonicalizeInto(buf, u16("").c_str()));
    CHECK(std::u16string(buf) == u16("\\"));
    // Origen y destino pueden coincidir (Unity reutiliza el buffer).
    const std::u16string same = u16("D:\\x\\.\\y");
    std::memcpy(buf, same.c_str(), (same.size() + 1) * sizeof(char16_t));
    CHECK(wp::canonicalizeInto(buf, buf));
    CHECK(std::u16string(buf) == u16("D:\\x\\y"));
    // Buffer insuficiente -> FALSE.
    char16_t tiny[4] = {};
    CHECK(!wp::canonicalizeInto(tiny, u16("D:\\abcdef").c_str(), 4));

    CHECK(wp::combineInto(buf, u16("D:\\Game").c_str(), u16("Data\\boot.config").c_str()));
    CHECK(std::u16string(buf) == u16("D:\\Game\\Data\\boot.config"));
    CHECK(wp::combineInto(buf, u16("D:\\Game").c_str(), u16("\\root.txt").c_str()));
    CHECK(std::u16string(buf) == u16("D:\\root.txt"));
    CHECK(wp::combineInto(buf, u16("D:\\Game").c_str(), u16("C:\\abs").c_str()));
    CHECK(std::u16string(buf) == u16("C:\\abs"));

    std::u16string p = u16("D:\\Game");
    p.resize(wp::kMaxPath);
    CHECK(wp::appendInto(p.data(), u16("\\sub\\file.dat").c_str()));
    CHECK(std::u16string(p.c_str()) == u16("D:\\Game\\sub\\file.dat"));

    CHECK(wp::removeFileSpec(p.data()));
    CHECK(std::u16string(p.c_str()) == u16("D:\\Game\\sub"));
    std::u16string root = u16("C:\\");
    CHECK(!wp::removeFileSpec(root.data()));
    std::u16string one = u16("C:\\file");
    CHECK(wp::removeFileSpec(one.data()));
    CHECK(std::u16string(one.c_str()) == u16("C:\\"));

    std::u16string f = u16("D:\\Game\\Cuphead.exe");
    CHECK(std::u16string(wp::findFileName(f.data())) == u16("Cuphead.exe"));
    CHECK(std::u16string(wp::findExtension(f.data())) == u16(".exe"));
    std::u16string noExt = u16("D:\\v1.2\\file");
    CHECK(*wp::findExtension(noExt.data()) == 0);

    CHECK(wp::isRelative(u16("Data\\x").c_str()));
    CHECK(!wp::isRelative(u16("D:\\x").c_str()));
    CHECK(!wp::isRelative(u16("\\x").c_str()));

    char a[wp::kMaxPath] = {};
    CHECK(wp::canonicalizeInto(a, "D:\\a\\..\\b.txt"));
    CHECK(std::string(a) == "D:\\b.txt");
}

void testFlagsAcrossCall() {
    Bytes code;
    movRegImm32(code, 0, 5);
    append(code, {0x83, 0xF8, 0x05});              // cmp eax, 5
    append(code, {0xE8, 0x06, 0x00, 0x00, 0x00});  // call +6 -> 19
    append(code, {0xC3});                          // 13: ret
    append(code, {0x90, 0x90, 0x90, 0x90, 0x90});  // 14..18
    append(code, {0x74, 0x07});                    // 19: je +7 -> 28
    movRegImm32(code, 0, 0);                       // 21
    append(code, {0xC3});                          // 26
    append(code, {0x90});                          // 27
    movRegImm32(code, 0, 1);                       // 28
    append(code, {0xC3});
    CHECK_EQ(runCode(code), 1);
}

// STD + REP MOVSD/STOSD: el memmove solapado de MSVC copia hacia atrás con DF=1. Antes el JIT
// ignoraba DF, repetía el primer elemento y escribía más allá del final (montón de Wwise en DmC).
void testStringDirectionFlag32() {
    CHECK(configureBackend(true));
    auto* buf = static_cast<uint32_t*>(hyperplay::lowmem::mmapBelow4G(4096));
    CHECK(buf != MAP_FAILED);
    for (uint32_t i = 0; i < 8; ++i) buf[i] = i + 1;
    const uint32_t b = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buf));
    Bytes mv;
    movRegImm32(mv, 6, b + 12);           // esi = &buf[3]
    movRegImm32(mv, 7, b + 16);           // edi = &buf[4]
    movRegImm32(mv, 1, 4);                // ecx = 4
    append(mv, {0xFD, 0xF3, 0xA5, 0xFC});  // std; rep movsd; cld
    append(mv, {0x89, 0xF8, 0xC3});        // mov eax, edi; ret
    CHECK_EQ(runCode(mv, true), static_cast<int64_t>(b));  // EDI acaba en &buf[0]
    const uint32_t wantMv[8] = {1, 1, 2, 3, 4, 6, 7, 8};
    for (int i = 0; i < 8; ++i) CHECK_EQ(buf[i], wantMv[i]);
    CHECK_EQ(g_backend->context().gpr[6], static_cast<uint64_t>(b - 4));

    Bytes st;
    movRegImm32(st, 7, b + 28);           // edi = &buf[7]
    movRegImm32(st, 0, 0x55);
    movRegImm32(st, 1, 2);
    append(st, {0xFD, 0xF3, 0xAB, 0xFC});  // std; rep stosd; cld
    append(st, {0x89, 0xF8, 0xC3});
    CHECK_EQ(runCode(st, true), static_cast<int64_t>(b + 20));
    CHECK_EQ(buf[5], 6u);
    CHECK_EQ(buf[6], 0x55u);
    CHECK_EQ(buf[7], 0x55u);

    // Sin STD sigue hacia delante.
    Bytes fw;
    movRegImm32(fw, 6, b);
    movRegImm32(fw, 7, b + 16);
    movRegImm32(fw, 1, 2);
    append(fw, {0xFC, 0xF3, 0xA5, 0x89, 0xF8, 0xC3});  // cld; rep movsd
    CHECK_EQ(runCode(fw, true), static_cast<int64_t>(b + 24));
    CHECK_EQ(buf[4], 1u);
    CHECK_EQ(buf[5], 1u);

    // XLAT: AL = [EBX + AL]; el resto de EAX no cambia. buf[1] = 1 -> byte 4 de la tabla = 1.
    Bytes xl;
    movRegImm32(xl, 3, b);
    movRegImm32(xl, 0, 0x11223304u);
    append(xl, {0xD7, 0xC3});
    CHECK_EQ(runCode(xl, true), 0x11223301);
}

void testMode32Program() {
    CHECK(configureBackend(true));
    Bytes code;
    movRegImm32(code, 0, 0x1234);           // mov eax, 0x1234
    append(code, {0x05, 0x04, 0x00, 0x00, 0x00});  // add eax, 4
    append(code, {0xC3});
    CHECK_EQ(runCode(code, true), 0x1238);

    Bytes loop;
    movRegImm32(loop, 0, 0);
    movRegImm32(loop, 1, 10);
    append(loop, {0x83, 0xC0, 0x02});  // add eax, 2
    append(loop, {0x49});              // dec ecx (32-bit)
    append(loop, {0x75, 0xFA});        // jnz -6 (vuelve a add eax,2)
    append(loop, {0xC3});
    CHECK_EQ(runCode(loop, true), 20);

    // MSVC __EH_prolog3 (mscoree de .NET 4.8): llamado con ECX = this, no debe tocar ECX.
    // Cookie [abs] sustituida por mov eax, 0; fs:[0] apunta a un TEB de prueba por debajo de 4 GB.
    auto* teb = static_cast<uint32_t*>(hyperplay::lowmem::mmapBelow4G(4096));
    CHECK(teb != MAP_FAILED);
    teb[0] = 0xFFFFFFFFu;  // ExceptionList vacía
    g_backend->context().fsBase = reinterpret_cast<uintptr_t>(teb);
    Bytes eh;
    append(eh, {0xB9, 0x44, 0x33, 0x22, 0x11});  // 0: mov ecx, 0x11223344
    append(eh, {0x89, 0xE6});                    // 5: mov esi, esp
    append(eh, {0x6A, 0x01, 0x6A, 0x00});        // 7: push 1; push 0
    append(eh, {0xE8, 0x00, 0x00, 0x00, 0x00});  // 11: call 16
    append(eh, {0x6A, 0x14});                    // 16: push 0x14
    append(eh, {0xB8, 0x88, 0x77, 0x66, 0x55});  // 18: mov eax, 0x55667788
    append(eh, {0xE8, 0x05, 0x00, 0x00, 0x00});  // 23: call 33 (__EH_prolog3)
    append(eh, {0x89, 0xC8});                    // 28: mov eax, ecx
    append(eh, {0x89, 0xF4});                    // 30: mov esp, esi
    append(eh, {0xC3});                          // 32: ret
    append(eh, {0x50, 0x64, 0xFF, 0x35, 0x00, 0x00, 0x00, 0x00,       // push eax; push fs:[0]
                0x8D, 0x44, 0x24, 0x0C, 0x2B, 0x64, 0x24, 0x0C,       // lea eax,[esp+0xc]; sub esp,[esp+0xc]
                0x53, 0x56, 0x57, 0x89, 0x28, 0x8B, 0xE8,             // push ebx/esi/edi; mov [eax],ebp; mov ebp,eax
                0xB8, 0x00, 0x00, 0x00, 0x00, 0x33, 0xC5, 0x50,       // mov eax,0 (cookie); xor eax,ebp; push eax
                0xFF, 0xB5, 0xFC, 0xFF, 0xFF, 0xFF,                   // push [ebp-4]
                0xC7, 0x85, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  // mov [ebp-4], -1
                0x8D, 0x85, 0xF4, 0xFF, 0xFF, 0xFF,                   // lea eax,[ebp-0xc]
                0x64, 0xA3, 0x00, 0x00, 0x00, 0x00,                   // mov fs:[0], eax
                0xF2, 0xC3});                                         // bnd ret
    CHECK_EQ(runCode(eh, true), 0x11223344);

    // PUSHAD/POPAD: ida y vuelta de los 8 registros; ESP vuelve a su sitio (el RET final lo usa).
    Bytes pa;
    movRegImm32(pa, 0, 0x11);
    movRegImm32(pa, 3, 0x22);
    movRegImm32(pa, 7, 0x44);
    append(pa, {0x60});                    // pushad
    append(pa, {0x8B, 0x44, 0x24, 0x0C});  // mov eax, [esp+12] (ESP original guardado)
    append(pa, {0x8D, 0x4C, 0x24, 0x20});  // lea ecx, [esp+32]
    append(pa, {0x29, 0xC8});              // sub eax, ecx -> 0 si el ESP guardado es correcto
    append(pa, {0x89, 0xC6});              // mov esi, eax
    append(pa, {0x31, 0xC0, 0x31, 0xDB, 0x31, 0xFF});  // xor eax,eax; xor ebx,ebx; xor edi,edi
    append(pa, {0x61});                    // popad
    append(pa, {0x01, 0xD8});              // add eax, ebx
    append(pa, {0x01, 0xF8});              // add eax, edi
    append(pa, {0xC3});
    CHECK_EQ(runCode(pa, true), 0x77);

    // SAHF: AH=0x41 -> ZF=1, CF=1.  LAHF tras XOR (ZF=1, PF=1, CF=0) -> AH = 0x46.
    Bytes sh;
    append(sh, {0xB4, 0x41, 0x9E});                    // mov ah, 0x41; sahf
    append(sh, {0xB8, 0x00, 0x00, 0x00, 0x00});        // mov eax, 0 (no toca flags)
    append(sh, {0x0F, 0x94, 0xC0});                    // sete al
    append(sh, {0x0F, 0x92, 0xC1});                    // setb cl
    append(sh, {0x0F, 0xB6, 0xC9, 0xD1, 0xE1});        // movzx ecx, cl; shl ecx, 1
    append(sh, {0x01, 0xC8, 0xC3});                    // add eax, ecx; ret -> 3
    CHECK_EQ(runCode(sh, true), 3);
    Bytes lh;
    append(lh, {0x31, 0xD2, 0x9F});                    // xor edx, edx; lahf
    append(lh, {0x0F, 0xB6, 0xC4, 0xC3});              // movzx eax, ah; ret -> 0x46
    CHECK_EQ(runCode(lh, true), 0x46);

    // PUSH ESP guarda el ESP anterior; POP ESP deja el valor leído (sin sumar 4).
    Bytes pe;
    append(pe, {0x89, 0xE0});  // mov eax, esp
    append(pe, {0x54, 0x5C});  // push esp; pop esp   -> esp sin cambios
    append(pe, {0x89, 0xE1});  // mov ecx, esp
    append(pe, {0x29, 0xC1});  // sub ecx, eax        -> 0
    append(pe, {0x54, 0x5A});  // push esp; pop edx   -> edx = esp original
    append(pe, {0x29, 0xC2});  // sub edx, eax        -> 0
    append(pe, {0x01, 0xD1});  // add ecx, edx
    append(pe, {0x83, 0xC1, 0x07});  // add ecx, 7
    append(pe, {0x89, 0xC8, 0xC3});  // mov eax, ecx; ret
    CHECK_EQ(runCode(pe, true), 7);
    CHECK(configureBackend(false));
    Bytes pe64;
    append(pe64, {0x48, 0x89, 0xE0});        // mov rax, rsp
    append(pe64, {0x54, 0x5C});              // push rsp; pop rsp
    append(pe64, {0x48, 0x89, 0xE1});        // mov rcx, rsp
    append(pe64, {0x48, 0x29, 0xC1});        // sub rcx, rax
    append(pe64, {0x54, 0x5A});              // push rsp; pop rdx
    append(pe64, {0x48, 0x29, 0xC2});        // sub rdx, rax
    append(pe64, {0x48, 0x01, 0xD1});        // add rcx, rdx
    append(pe64, {0x48, 0x83, 0xC1, 0x07});  // add rcx, 7
    append(pe64, {0x48, 0x89, 0xC8, 0xC3});  // mov rax, rcx; ret
    CHECK_EQ(runCode(pe64), 7);
    hyperplay::lowmem::g_enabled = false;
}

void testWxAndUnsupported() {
    CHECK(configureBackend(false));
    Bytes code;
    movRegImm32(code, 0, 7);
    append(code, {0xC3});
    uint8_t* entry = mapCode(code);
    auto* block = g_backend->translateBlock(reinterpret_cast<uintptr_t>(entry));
    CHECK(block != nullptr);
    CHECK(g_backend->isExecutableRange(block->hostCode, block->hostSize));

    // La arena no puede ser RWX: se comprueba la proteccion real en /proc/self/maps.
    FILE* maps = std::fopen("/proc/self/maps", "r");
    bool found = false;
    bool writableAndExecutable = false;
    if (maps != nullptr) {
        char line[512];
        const uintptr_t target = reinterpret_cast<uintptr_t>(block->hostCode);
        while (std::fgets(line, sizeof(line), maps) != nullptr) {
            uintptr_t start = 0;
            uintptr_t end = 0;
            char perms[8] = {};
            if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %7s", &start, &end, perms) != 3) {
                continue;
            }
            if (target < start || target >= end) continue;
            found = true;
            writableAndExecutable = std::strchr(perms, 'w') != nullptr &&
                                    std::strchr(perms, 'x') != nullptr;
            break;
        }
        std::fclose(maps);
    }
    CHECK(found);
    CHECK(!writableAndExecutable);

    // Instruccion no soportada -> la traduccion falla de forma explicita.
    Bytes unsupported;
    append(unsupported, {0x0F, 0x0B});  // UD2
    uint8_t* bad = mapCode(unsupported);
    CHECK(g_backend->translateBlock(reinterpret_cast<uintptr_t>(bad)) == nullptr);
}

void testInstructionFamiliesDecode() {
    // Secuencia representativa de familias esenciales: debe traducirse sin error.
    Bytes code;
    movRegImm64(code, 3, 0x1000);
    append(code, {0x48, 0x89, 0x43, 0x08});  // mov [rbx+8], rax
    append(code, {0x48, 0x8B, 0x4B, 0x08});  // mov rcx, [rbx+8]
    append(code, {0x48, 0x31, 0xC8});        // xor rax, rcx
    append(code, {0x48, 0x85, 0xC0});        // test rax, rax
    append(code, {0x48, 0x0F, 0xAF, 0xC1});  // imul rax, rcx
    append(code, {0x48, 0x89, 0xC2});        // mov rdx, rax
    append(code, {0x48, 0x83, 0xC0, 0x01});  // add rax, 1
    append(code, {0x48, 0x83, 0xE8, 0x01});  // sub rax, 1
    append(code, {0x48, 0x39, 0xC2});        // cmp rdx, rax
    append(code, {0x74, 0x02});              // je +2
    append(code, {0xEB, 0x00});              // jmp +0
    append(code, {0xC3});
    uint8_t* entry = mapCode(code);
    CHECK(g_backend->translateBlock(reinterpret_cast<uintptr_t>(entry)) != nullptr);
}

void testCmovAndNotNeg() {
    CHECK(configureBackend(false));

    // Test NOT rax
    {
        Bytes code;
        movRegImm64(code, 0, 0x0000000000000000ull);
        append(code, {0x48, 0xF7, 0xD0}); // not rax
        append(code, {0xC3});
        CHECK_EQ(runCode(code), static_cast<int64_t>(0xFFFFFFFFFFFFFFFFull));
    }

    // Test NEG rax
    {
        Bytes code;
        movRegImm64(code, 0, 42);
        append(code, {0x48, 0xF7, 0xD8}); // neg rax
        append(code, {0xC3});
        CHECK_EQ(runCode(code), -42);
    }

    // Test CMOVE rax, rcx (condición tomada: ZF=1)
    {
        Bytes code;
        movRegImm64(code, 0, 10); // rax = 10
        movRegImm64(code, 1, 20); // rcx = 20
        movRegImm64(code, 2, 5);  // rdx = 5
        append(code, {0x48, 0x39, 0xD2});       // cmp rdx, rdx (ZF=1)
        append(code, {0x48, 0x0F, 0x44, 0xC1}); // cmove rax, rcx
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 20);
    }

    // Test CMOVE rax, rcx (condición NO tomada: ZF=0)
    {
        Bytes code;
        movRegImm64(code, 0, 10); // rax = 10
        movRegImm64(code, 1, 20); // rcx = 20
        movRegImm64(code, 2, 5);  // rdx = 5
        append(code, {0x48, 0x83, 0xFA, 0x00}); // cmp rdx, 0 (ZF=0)
        append(code, {0x48, 0x0F, 0x44, 0xC1}); // cmove rax, rcx
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 10);
    }

    // Test CPUID leaf 0
    {
        Bytes code;
        movRegImm64(code, 0, 0); // eax = 0
        append(code, {0x0F, 0xA2}); // cpuid
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0x16);
        CHECK_EQ(g_backend->context().gpr[3], 0x756e6547); // "Genu" (EBX)
        CHECK_EQ(g_backend->context().gpr[2], 0x49656e69); // "ineI" (EDX)
        CHECK_EQ(g_backend->context().gpr[1], 0x6c65746e); // "ntel" (ECX)
    }

    // Test BT ebx, 9 (0x0F 0xBA 0xE3 0x09)
    {
        Bytes code;
        movRegImm64(code, 3, 1ull << 9); // ebx tiene bit 9 a 1
        append(code, {0x0F, 0xBA, 0xE3, 0x09}); // bt ebx, 9 -> CF=1
        append(code, {0xC3});
        runCode(code);
        CHECK((g_backend->context().rflags & 1) != 0);
    }

    // Test XGETBV (0x0F 0x01 0xD0)
    {
        Bytes code;
        movRegImm64(code, 1, 0); // ecx = 0
        append(code, {0x0F, 0x01, 0xD0}); // xgetbv
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 7);
        CHECK_EQ(g_backend->context().gpr[2], 0); // EDX = 0
    }
}

void testNewInstructionFamilies() {
    CHECK(configureBackend(false));

    // 1. Test SBB rcx, rcx with CF=0 -> rcx = 0
    {
        Bytes code;
        movRegImm64(code, 0, 10);
        append(code, {0x48, 0x83, 0xF8, 0x05}); // cmp rax, 5 -> CF=0
        append(code, {0x48, 0x19, 0xC9});       // sbb rcx, rcx -> 0
        append(code, {0x48, 0x89, 0xC8});       // mov rax, rcx
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0);
    }

    // 2. Test SBB rcx, rcx with CF=1 -> rcx = -1 (0xFFFFFFFFFFFFFFFF)
    {
        Bytes code;
        movRegImm64(code, 0, 5);
        append(code, {0x48, 0x83, 0xF8, 0x0A}); // cmp rax, 10 -> CF=1
        append(code, {0x48, 0x19, 0xC9});       // sbb rcx, rcx -> -1
        append(code, {0x48, 0x89, 0xC8});       // mov rax, rcx
        append(code, {0xC3});
        CHECK_EQ(runCode(code), -1);
    }

    // 3. Test ADC rax, rdx with CF=1
    {
        Bytes code;
        movRegImm64(code, 0, 5);
        append(code, {0x48, 0x83, 0xF8, 0x0A}); // cmp rax, 10 -> CF=1
        movRegImm64(code, 0, 10);
        movRegImm64(code, 2, 20);
        append(code, {0x48, 0x11, 0xD0});       // adc rax, rdx -> 10 + 20 + 1 = 31
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 31);
    }

    // 4. Test SETE al (0x0F 0x94) with ZF=1 and ZF=0
    {
        Bytes code;
        movRegImm64(code, 0, 42);
        append(code, {0x48, 0x83, 0xF8, 0x2A}); // cmp rax, 42 -> ZF=1
        movRegImm64(code, 0, 0);
        append(code, {0x0F, 0x94, 0xC0});       // sete al -> 1
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 1);

        Bytes code2;
        movRegImm64(code2, 0, 42);
        append(code2, {0x48, 0x83, 0xF8, 0x2B}); // cmp rax, 43 -> ZF=0
        movRegImm64(code2, 0, 0);
        append(code2, {0x0F, 0x94, 0xC0});       // sete al -> 0
        append(code2, {0xC3});
        CHECK_EQ(runCode(code2), 0);
    }

    // 5. Test BSF / TZCNT (0x0F 0xBC)
    {
        Bytes code;
        movRegImm64(code, 1, 0x00000008);       // rcx = 8 (bit 3)
        append(code, {0x48, 0x0F, 0xBC, 0xC1}); // bsf rax, rcx -> 3
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 3);
    }

    // 6. Test BSR / LZCNT (0x0F 0xBD)
    {
        Bytes code;
        movRegImm64(code, 1, 0x00000080);       // rcx = 128 (bit 7)
        append(code, {0x48, 0x0F, 0xBD, 0xC1}); // bsr rax, rcx -> 7
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 7);
    }

    // 7. Test CDQE (0x48 0x98) and CDQ / CQO (0x99)
    {
        Bytes code;
        movRegImm32(code, 0, 0xFFFFFFFE);       // eax = -2
        append(code, {0x48, 0x98});             // cdqe -> rax = -2 (64-bit sign extended)
        append(code, {0xC3});
        CHECK_EQ(runCode(code), -2);

        Bytes codeCqo;
        movRegImm64(codeCqo, 0, -5);            // rax = -5
        append(codeCqo, {0x48, 0x99});          // cqo -> rdx = -1
        append(codeCqo, {0x48, 0x89, 0xD0});    // mov rax, rdx
        append(codeCqo, {0xC3});
        CHECK_EQ(runCode(codeCqo), -1);
    }

    // 8. Test MUL / DIV (0xF7 /4 and /6)
    {
        Bytes code;
        movRegImm64(code, 0, 100);              // rax = 100
        movRegImm64(code, 1, 7);                // rcx = 7
        movRegImm64(code, 2, 0);                // rdx = 0
        append(code, {0x48, 0xF7, 0xF1});       // div rcx -> rax = 14, rdx = 2
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 14);
        CHECK_EQ(g_backend->context().gpr[2], 2);
    }

    // 9. Test REP STOSQ (f3 48 ab) - Hollow Knight's exact sequence!
    {
        Bytes code;
        uint64_t targetBuffer[4] = {0xAA, 0xBB, 0xCC, 0xDD};
        movRegImm64(code, 0, 0x1234567890ABCDEFull); // rax = value
        movRegImm64(code, 1, 4);                     // rcx = count (4 qwords)
        movRegImm64(code, 7, reinterpret_cast<uintptr_t>(targetBuffer)); // rdi = dest
        append(code, {0xF3, 0x48, 0xAB});            // rep stosq
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(targetBuffer[0], 0x1234567890ABCDEFull);
        CHECK_EQ(targetBuffer[1], 0x1234567890ABCDEFull);
        CHECK_EQ(targetBuffer[2], 0x1234567890ABCDEFull);
        CHECK_EQ(targetBuffer[3], 0x1234567890ABCDEFull);
        CHECK_EQ(g_backend->context().gpr[1], 0);    // rcx == 0
    }

    // 10. Test addCodeRange (multi-range guest code translation)
    {
        Bytes code1;
        movRegImm64(code1, 0, 77);
        append(code1, {0xC3});
        uint8_t* block1 = mapCode(code1);

        Bytes code2;
        movRegImm64(code2, 0, 88);
        append(code2, {0xC3});
        uint8_t* block2 = mapCode(code2);

        g_backend->setCodeRange(reinterpret_cast<uintptr_t>(block1), 4096);
        g_backend->addCodeRange(reinterpret_cast<uintptr_t>(block2), 4096);

        auto* tb1 = g_backend->translateBlock(reinterpret_cast<uintptr_t>(block1));
        auto* tb2 = g_backend->translateBlock(reinterpret_cast<uintptr_t>(block2));
        CHECK(tb1 != nullptr);
        CHECK(tb2 != nullptr);
        g_backend->setCodeRange(0, 0);
    }

    // 11. Test ROL and ROR (rotates)
    {
        // rol rax, 1
        Bytes codeRol;
        movRegImm64(codeRol, 0, 0x8000000000000001ull);
        append(codeRol, {0x48, 0xD1, 0xC0}); // rol rax, 1
        append(codeRol, {0xC3});
        CHECK_EQ(runCode(codeRol), 0x0000000000000003ull);

        // ror rax, 1
        Bytes codeRor;
        movRegImm64(codeRor, 0, 0x0000000000000003ull);
        append(codeRor, {0x48, 0xD1, 0xC8}); // ror rax, 1
        append(codeRor, {0xC3});
        CHECK_EQ(runCode(codeRor), 0x8000000000000001ull);

        // 32-bit rol eax, 4
        Bytes codeRol32;
        movRegImm32(codeRol32, 0, 0x12345678);
        append(codeRol32, {0xC1, 0xC0, 0x04}); // rol eax, 4
        append(codeRol32, {0xC3});
        CHECK_EQ(runCode(codeRol32), 0x23456781);

        // 32-bit ror eax, 4
        Bytes codeRor32;
        movRegImm32(codeRor32, 0, 0x23456781);
        append(codeRor32, {0xC1, 0xC8, 0x04}); // ror eax, 4
        append(codeRor32, {0xC3});
        CHECK_EQ(runCode(codeRor32), 0x12345678);
    }

    // 12. Test 16-bit operand size prefix (0x66) store and load + PXOR (66 0F EF)
    {
        uint16_t testBuf[2] = {0, 0};
        Bytes code;
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(testBuf));
        movRegImm64(code, 1, 0xCAFE); // cx = 0xCAFE
        // 66 89 08 -> mov word ptr [rax], cx
        append(code, {0x66, 0x89, 0x08});
        // 66 0F EF C0 -> pxor xmm0, xmm0
        append(code, {0x66, 0x0F, 0xEF, 0xC0});
        // 66 8B 00 -> mov ax, [rax] (preserva bits altos de rax en x86)
        append(code, {0x66, 0x8B, 0x00});
        append(code, {0xC3});
        CHECK_EQ(static_cast<uint16_t>(runCode(code)), 0xCAFE);
        CHECK_EQ(testBuf[0], 0xCAFE);
    }

    // 13. Test 0xC6 (MOV r/m8, imm8), 0xC7 (MOV r/m, imm32/imm16) and stack pattern
    {
        uint8_t buffer[64];
        std::memset(buffer, 0xAA, sizeof(buffer));

        Bytes code;
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(buffer));
        // c6 40 18 00 -> mov byte ptr [rax + 0x18], 0
        append(code, {0xC6, 0x40, 0x18, 0x00});
        // c7 40 20 78 56 34 12 -> mov dword ptr [rax + 0x20], 0x12345678
        append(code, {0xC7, 0x40, 0x20, 0x78, 0x56, 0x34, 0x12});
        // 66 c7 40 10 aa 55 -> mov word ptr [rax + 0x10], 0x55AA
        append(code, {0x66, 0xC7, 0x40, 0x10, 0xAA, 0x55});
        // cmp byte ptr [rax + 0x18], 0
        append(code, {0x80, 0x78, 0x18, 0x00});
        // 74 06 -> je +6 (skip mov eax, 1; ret: 5 + 1 = 6 bytes)
        append(code, {0x74, 0x06});
        // mov eax, 1
        append(code, {0xB8, 0x01, 0x00, 0x00, 0x00});
        append(code, {0xC3});
        // mov eax, 42
        append(code, {0xB8, 0x2A, 0x00, 0x00, 0x00});
        append(code, {0xC3});

        CHECK_EQ(runCode(code), 42);
        CHECK_EQ(buffer[0x18], 0);
        uint32_t val32 = 0;
        std::memcpy(&val32, &buffer[0x20], sizeof(val32));
        CHECK_EQ(val32, 0x12345678);
        uint16_t val16 = 0;
        std::memcpy(&val16, &buffer[0x10], sizeof(val16));
        CHECK_EQ(val16, 0x55AA);
    }

    // 14. Test XADD (0F C1) and CMPXCHG (0F B1)
    {
        // XADD en memoria: buffer = 100, rdx = 50 -> buffer = 150, rdx = 100
        uint64_t memVal = 100;
        Bytes codeXaddMem;
        movRegImm64(codeXaddMem, 0, reinterpret_cast<uintptr_t>(&memVal)); // rax = &memVal
        movRegImm64(codeXaddMem, 2, 50);                                   // rdx = 50
        append(codeXaddMem, {0x48, 0x0F, 0xC1, 0x10});                     // xadd [rax], rdx
        append(codeXaddMem, {0x48, 0x89, 0xD0});                           // mov rax, rdx
        append(codeXaddMem, {0xC3});
        CHECK_EQ(runCode(codeXaddMem), 100);
        CHECK_EQ(memVal, 150);

        // XADD registro: rcx = 20, rdx = 30 -> rcx = 50, rdx = 20
        Bytes codeXaddReg;
        movRegImm64(codeXaddReg, 1, 20);               // rcx = 20
        movRegImm64(codeXaddReg, 2, 30);               // rdx = 30
        append(codeXaddReg, {0x48, 0x0F, 0xC1, 0xD1}); // xadd rcx, rdx
        append(codeXaddReg, {0x48, 0x89, 0xC8});       // mov rax, rcx
        append(codeXaddReg, {0xC3});
        CHECK_EQ(runCode(codeXaddReg), 50);

        // CMPXCHG match: rax=100, rdx=999, memVal=100 -> memVal=999
        memVal = 100;
        Bytes codeCmpMatch;
        movRegImm64(codeCmpMatch, 0, 100);                                  // rax = 100
        movRegImm64(codeCmpMatch, 1, reinterpret_cast<uintptr_t>(&memVal)); // rcx = &memVal
        movRegImm64(codeCmpMatch, 2, 999);                                  // rdx = 999
        append(codeCmpMatch, {0x48, 0x0F, 0xB1, 0x11});                     // cmpxchg [rcx], rdx
        append(codeCmpMatch, {0xC3});
        CHECK_EQ(runCode(codeCmpMatch), 100);
        CHECK_EQ(memVal, 999);

        // CMPXCHG mismatch: rax=50, rdx=999, memVal=100 -> memVal=100, rax=100
        memVal = 100;
        Bytes codeCmpMismatch;
        movRegImm64(codeCmpMismatch, 0, 50);                                   // rax = 50
        movRegImm64(codeCmpMismatch, 1, reinterpret_cast<uintptr_t>(&memVal)); // rcx = &memVal
        movRegImm64(codeCmpMismatch, 2, 999);                                  // rdx = 999
        append(codeCmpMismatch, {0x48, 0x0F, 0xB1, 0x11});                     // cmpxchg [rcx], rdx
        append(codeCmpMismatch, {0xC3});
        CHECK_EQ(runCode(codeCmpMismatch), 100);
        CHECK_EQ(memVal, 100);
    }

    // 15. Test MOVD/MOVQ xmm <-> GPR / memory (0F 6E / 0F 7E / 0F D6 / F3 0F 7E)
    {
        // 64-bit: rax = 0x0123456789ABCDEF -> movq xmm0, rax -> movq rdx, xmm0
        Bytes codeMovq;
        movRegImm64(codeMovq, 0, 0x0123456789ABCDEFull);
        // 48 66 0F 6E C0 -> movq xmm0, rax
        append(codeMovq, {0x48, 0x66, 0x0F, 0x6E, 0xC0});
        // 48 66 0F 7E C2 -> movq rdx, xmm0
        append(codeMovq, {0x48, 0x66, 0x0F, 0x7E, 0xC2});
        // mov rax, rdx
        append(codeMovq, {0x48, 0x89, 0xD0});
        append(codeMovq, {0xC3});
        CHECK_EQ(runCode(codeMovq), 0x0123456789ABCDEFull);

        // 32-bit: eax = 0x12345678 -> movd xmm1, eax -> movd ecx, xmm1
        Bytes codeMovd;
        movRegImm32(codeMovd, 0, 0x12345678);
        // 66 0F 6E C8 -> movd xmm1, eax
        append(codeMovd, {0x66, 0x0F, 0x6E, 0xC8});
        // 66 0F 7E C9 -> movd ecx, xmm1
        append(codeMovd, {0x66, 0x0F, 0x7E, 0xC9});
        // mov eax, ecx
        append(codeMovd, {0x89, 0xC8});
        append(codeMovd, {0xC3});
        CHECK_EQ(runCode(codeMovd), 0x12345678);
    }

    // 16. Test SCASB simple y REPNE SCASB (0xAE con y sin prefixF2)
    {
        // Simple SCASB: busca byte 'A' (0x41)
        char str[] = "ABCD";
        Bytes codeScasb;
        movRegImm64(codeScasb, 0, 0x41);                                  // rax = 'A'
        movRegImm64(codeScasb, 7, reinterpret_cast<uintptr_t>(str));      // rdi = str
        append(codeScasb, {0xAE});                                        // scasb
        // Si coincidió, ZF=1 -> devolvemos 1, si no 0
        append(codeScasb, {0x74, 0x06});                                  // jz +6 (salta sobre mov eax,0 + ret)
        movRegImm32(codeScasb, 0, 0);                                     // eax = 0 (5 bytes)
        append(codeScasb, {0xC3});                                        // ret (1 byte)
        movRegImm32(codeScasb, 0, 1);                                     // eax = 1
        append(codeScasb, {0xC3});
        CHECK_EQ(runCode(codeScasb), 1);

        // REPNE SCASB: busca byte 'W' (0x57) en "Hello World!"
        char hello[] = "Hello World!";
        Bytes codeRepne;
        movRegImm64(codeRepne, 0, 0x57);                                  // rax = 'W'
        movRegImm64(codeRepne, 1, 12);                                    // rcx = 12
        movRegImm64(codeRepne, 7, reinterpret_cast<uintptr_t>(hello));    // rdi = hello
        append(codeRepne, {0xF2, 0xAE});                                  // repne scasb
        // Devuelve el carácter apuntado por rdi-1 (debe ser 'W')
        append(codeRepne, {0x48, 0x89, 0xF8});                            // mov rax, rdi
        append(codeRepne, {0x48, 0xFF, 0xC8});                            // dec rax
        append(codeRepne, {0x0F, 0xB6, 0x00});                            // movzx eax, byte ptr [rax]
        append(codeRepne, {0xC3});
        CHECK_EQ(runCode(codeRepne), static_cast<uint64_t>('W'));
    }

    // 17. Test SHLD y SHRD (0x0F 0xA4 / 0xAC con imm8 y 0x0F 0xA5 / 0xAD con CL)
    {
        // 64-bit SHLD con imm8: rax = 0x0123456789ABCDEF, rdx = 0xFEDCBA9876543210
        // shld rax, rdx, 8 -> 0x23456789ABCDEFFE
        Bytes codeShldImm;
        movRegImm64(codeShldImm, 0, 0x0123456789ABCDEFull);
        movRegImm64(codeShldImm, 2, 0xFEDCBA9876543210ull);
        // 48 0F A4 D0 08 -> shld rax, rdx, 8
        append(codeShldImm, {0x48, 0x0F, 0xA4, 0xD0, 0x08});
        append(codeShldImm, {0xC3});
        CHECK_EQ(runCode(codeShldImm), 0x23456789ABCDEFFEull);

        // 64-bit SHRD con imm8: rax = 0x0123456789ABCDEF, rdx = 0xFEDCBA9876543210
        // shrd rax, rdx, 8 -> 0x100123456789ABCD
        Bytes codeShrdImm;
        movRegImm64(codeShrdImm, 0, 0x0123456789ABCDEFull);
        movRegImm64(codeShrdImm, 2, 0xFEDCBA9876543210ull);
        // 48 0F AC D0 08 -> shrd rax, rdx, 8
        append(codeShrdImm, {0x48, 0x0F, 0xAC, 0xD0, 0x08});
        append(codeShrdImm, {0xC3});
        CHECK_EQ(runCode(codeShrdImm), 0x100123456789ABCDull);

        // 32-bit SHLD con CL: eax = 0x12345678, edx = 0x87654321, ecx = 4
        // shld eax, edx, cl -> 0x23456788
        Bytes codeShldCl;
        movRegImm32(codeShldCl, 0, 0x12345678);
        movRegImm32(codeShldCl, 2, 0x87654321);
        movRegImm32(codeShldCl, 1, 4);
        // 0F A5 D0 -> shld eax, edx, cl
        append(codeShldCl, {0x0F, 0xA5, 0xD0});
        append(codeShldCl, {0xC3});
        CHECK_EQ(runCode(codeShldCl), 0x23456788);
    }

    // 18. Test Conversiones y Operaciones Escalares Flotantes (CVTSI2SD, ADDSD, CVTTSD2SI)
    {
        double delta = 5.0;
        Bytes codeFloat;
        movRegImm64(codeFloat, 0, 20);                                          // rax = 20
        // F2 48 0F 2A C0 -> cvtsi2sd xmm0, rax (xmm0 = 20.0)
        append(codeFloat, {0xF2, 0x48, 0x0F, 0x2A, 0xC0});
        movRegImm64(codeFloat, 1, reinterpret_cast<uintptr_t>(&delta));         // rcx = &delta
        // F2 0F 58 01 -> addsd xmm0, [rcx] (xmm0 = 25.0)
        append(codeFloat, {0xF2, 0x0F, 0x58, 0x01});
        // F2 48 0F 2C C0 -> cvttsd2si rax, xmm0 (rax = 25)
        append(codeFloat, {0xF2, 0x48, 0x0F, 0x2C, 0xC0});
        append(codeFloat, {0xC3});
        CHECK_EQ(runCode(codeFloat), 25);
    }

    // 19. Test UCOMISS (0x0F 0x2E) con saltos condicionales
    {
        float a = 10.0f;
        float b = 5.0f;
        Bytes codeUcomiss;
        movRegImm64(codeUcomiss, 1, reinterpret_cast<uintptr_t>(&a));           // rcx = &a
        movRegImm64(codeUcomiss, 2, reinterpret_cast<uintptr_t>(&b));           // rdx = &b
        // F3 0F 10 01 -> movss xmm0, [rcx] (10.0f)
        append(codeUcomiss, {0xF3, 0x0F, 0x10, 0x01});
        // 0F 2E 02 -> ucomiss xmm0, [rdx] (cmp 10.0f, 5.0f -> 10 > 5 -> ZF=0, CF=0)
        append(codeUcomiss, {0x0F, 0x2E, 0x02});
        // 77 06 -> ja +6 (jump if above: CF=0 && ZF=0)
        append(codeUcomiss, {0x77, 0x06});
        movRegImm32(codeUcomiss, 0, 0);                                         // eax = 0 (5 bytes)
        append(codeUcomiss, {0xC3});                                            // ret (1 byte)
        movRegImm32(codeUcomiss, 0, 42);                                        // eax = 42
        append(codeUcomiss, {0xC3});
        CHECK_EQ(runCode(codeUcomiss), 42);
    }

    // 20. Test PCMPEQB y PMOVMSKB (0x66 0x0F 0x74 / 0x66 0x0F 0xD7)
    {
        // Llena dos bloques de 16 bytes: match en byte 0 y byte 15
        uint8_t b1[16] = {0xAA, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0, 0xBB};
        uint8_t b2[16] = {0xAA, 1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1, 0xBB};
        Bytes codePcmpeq;
        movRegImm64(codePcmpeq, 1, reinterpret_cast<uintptr_t>(b1));            // rcx = &b1
        movRegImm64(codePcmpeq, 2, reinterpret_cast<uintptr_t>(b2));            // rdx = &b2
        // 66 0F 6F 01 -> movdqa xmm0, [rcx]
        append(codePcmpeq, {0x66, 0x0F, 0x6F, 0x01});
        // 66 0F 6F 0A -> movdqa xmm1, [rdx]
        append(codePcmpeq, {0x66, 0x0F, 0x6F, 0x0A});
        // 66 0F 74 C1 -> pcmpeqb xmm0, xmm1
        append(codePcmpeq, {0x66, 0x0F, 0x74, 0xC1});
        // 66 0F D7 C0 -> pmovmskb eax, xmm0
        append(codePcmpeq, {0x66, 0x0F, 0xD7, 0xC0});
        append(codePcmpeq, {0xC3});
        // byte 0 coincide -> bit 0 = 1; byte 15 coincide -> bit 15 = 1 -> mask = 0x8001
        CHECK_EQ(runCode(codePcmpeq), 0x8001);

        // Test con xmm1 como fuente: 66 0F D7 C1 -> pmovmskb eax, xmm1
        Bytes codePmovXmm1;
        movRegImm64(codePmovXmm1, 1, reinterpret_cast<uintptr_t>(b1));
        movRegImm64(codePmovXmm1, 2, reinterpret_cast<uintptr_t>(b2));
        append(codePmovXmm1, {0x66, 0x0F, 0x6F, 0x01}); // movdqa xmm0, [rcx]
        append(codePmovXmm1, {0x66, 0x0F, 0x6F, 0x0A}); // movdqa xmm1, [rdx]
        append(codePmovXmm1, {0x66, 0x0F, 0x74, 0xC8}); // pcmpeqb xmm1, xmm0
        append(codePmovXmm1, {0x66, 0x0F, 0xD7, 0xC1}); // pmovmskb eax, xmm1
        append(codePmovXmm1, {0xC3});
        CHECK_EQ(runCode(codePmovXmm1), 0x8001);

        // Test VPCMPEQW + VPMOVMSKB con ymm1
        uint16_t wbuf[16] = {1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        Bytes codeVpcmpeqw;
        movRegImm64(codeVpcmpeqw, 9, reinterpret_cast<uintptr_t>(wbuf)); // r9 = wbuf
        append(codeVpcmpeqw, {0xC5, 0xF1, 0xEF, 0xC9});                 // vpxor xmm1, xmm1, xmm1
        append(codeVpcmpeqw, {0xC4, 0xC1, 0x75, 0x75, 0x09});           // vpcmpeqw ymm1, ymm1, [r9]
        append(codeVpcmpeqw, {0xC5, 0xFD, 0xD7, 0xC1});                 // vpmovmskb eax, ymm1
        append(codeVpcmpeqw, {0xC3});
        const int64_t vres = runCode(codeVpcmpeqw);
        CHECK_EQ(vres & 0x00C0, 0x00C0);

        // Test PCMPISTRI (66 0F 3A 63 /r ib)
        uint16_t needle[8] = {'l', 'o', 0, 0, 0, 0, 0, 0};
        uint16_t haystack[8] = {'H', 'e', 'l', 'l', 'o', 0, 0, 0};
        Bytes codePcmp;
        movRegImm64(codePcmp, 1, reinterpret_cast<uintptr_t>(needle));   // rcx = &needle
        movRegImm64(codePcmp, 2, reinterpret_cast<uintptr_t>(haystack)); // rdx = &haystack
        append(codePcmp, {0x66, 0x0F, 0x6F, 0x01}); // movdqa xmm0, [rcx]
        append(codePcmp, {0x66, 0x0F, 0x6F, 0x0A}); // movdqa xmm1, [rdx]
        append(codePcmp, {0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x0D}); // pcmpistri xmm0, xmm1, 0x0D
        append(codePcmp, {0x48, 0x89, 0xC8}); // mov rax, rcx
        append(codePcmp, {0xC3});
        CHECK_EQ(runCode(codePcmp), 3);
    }

    // 21. Test VEX (0xC4 / 0xC5): VZEROUPPER, VINSERTF128, VMOVDQA 256-bit, VXORPS
    {
        uint8_t src[32] = {0};
        uint8_t dst[32] = {0};
        for (int i = 0; i < 32; ++i) src[i] = static_cast<uint8_t>(i + 1);
        Bytes codeVex;
        movRegImm64(codeVex, 1, reinterpret_cast<uintptr_t>(src)); // rcx = src
        movRegImm64(codeVex, 2, reinterpret_cast<uintptr_t>(dst)); // rdx = dst
        // C5 F8 77 -> vzeroupper
        append(codeVex, {0xC5, 0xF8, 0x77});
        // C5 F9 6F 01 -> vmovdqa xmm0, [rcx] (128-bit)
        append(codeVex, {0xC5, 0xF9, 0x6F, 0x01});
        // C4 E3 7D 18 C0 01 -> vinsertf128 ymm0, ymm0, xmm0, 1
        append(codeVex, {0xC4, 0xE3, 0x7D, 0x18, 0xC0, 0x01});
        // C5 FD 7F 02 -> vmovdqa ymmword ptr [rdx], ymm0 (256-bit store)
        append(codeVex, {0xC5, 0xFD, 0x7F, 0x02});
        // C5 F8 77 -> vzeroupper
        append(codeVex, {0xC5, 0xF8, 0x77});
        // 31 C0 -> xor eax, eax
        append(codeVex, {0x31, 0xC0});
        append(codeVex, {0xC3});
        CHECK_EQ(runCode(codeVex), 0);
        // Verificar que dst almacena src[0..15] en ambas mitades (0..15 y 16..31)
        for (int i = 0; i < 16; ++i) {
            CHECK_EQ(dst[i], src[i]);
            CHECK_EQ(dst[i + 16], src[i]);
        }
    }

    // 22. Test GS segment: mov r11, qword ptr gs:[0x10] (StackLimit en TEB)
    {
        Bytes codeGs;
        // 65 4C 8B 1C 25 10 00 00 00 -> mov r11, qword ptr gs:[0x10]
        append(codeGs, {0x65, 0x4C, 0x8B, 0x1C, 0x25, 0x10, 0x00, 0x00, 0x00});
        // 4C 89 D8 -> mov rax, r11
        append(codeGs, {0x4C, 0x89, 0xD8});
        append(codeGs, {0xC3});
        const uint64_t stackLimit = static_cast<uint64_t>(runCode(codeGs));
        CHECK(stackLimit != 0);
    }

    // 23. Test SSE2 Shift (PSRLDQ, PSLLDQ), PSHUFD, PADDD
    {
        alignas(16) uint32_t src[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
        alignas(16) uint32_t dst[4] = {0};

        Bytes codeShift;
        movRegImm64(codeShift, 1, reinterpret_cast<uintptr_t>(src)); // rcx = src
        movRegImm64(codeShift, 2, reinterpret_cast<uintptr_t>(dst)); // rdx = dst
        append(codeShift, {0x66, 0x0F, 0x6F, 0x01});       // movdqa xmm0, [rcx]
        append(codeShift, {0x66, 0x0F, 0x73, 0xD8, 0x08}); // psrldq xmm0, 8
        append(codeShift, {0x66, 0x0F, 0x70, 0xC8, 0x1B}); // pshufd xmm1, xmm0, 0x1B
        append(codeShift, {0x66, 0x0F, 0xFE, 0xC1});       // paddd xmm0, xmm1
        append(codeShift, {0x66, 0x0F, 0x7F, 0x02});       // movdqa [rdx], xmm0
        append(codeShift, {0x31, 0xC0});                   // xor eax, eax
        append(codeShift, {0xC3});
        CHECK_EQ(runCode(codeShift), 0);
        CHECK_EQ(dst[0], 0x33333333u);
        CHECK_EQ(dst[1], 0x44444444u);
        CHECK_EQ(dst[2], 0x44444444u);
        CHECK_EQ(dst[3], 0x33333333u);
    }

    // 24. Test VEX 2B: VPCMPEQB, VPMOVMSKB, VZEROUPPER
    {
        alignas(16) uint8_t srcA[16] = {'H','e','l','l','o',' ','W','o','r','l','d','!','\0','\0','\0','\0'};
        alignas(16) uint8_t srcB[16] = {'H','e','l','l','o','_','W','o','r','l','d','!','\0','\0','\0','\0'};

        Bytes codeVex;
        movRegImm64(codeVex, 1, reinterpret_cast<uintptr_t>(srcA)); // rcx = srcA
        movRegImm64(codeVex, 2, reinterpret_cast<uintptr_t>(srcB)); // rdx = srcB
        append(codeVex, {0xC5, 0xF8, 0x77});             // vzeroupper
        append(codeVex, {0xC5, 0xF9, 0x6F, 0x01});       // vmovdqa xmm0, [rcx]
        append(codeVex, {0xC5, 0xF9, 0x74, 0x02});       // vpcmpeqb xmm0, xmm0, [rdx]
        append(codeVex, {0xC5, 0xF9, 0xD7, 0xC0});       // vpmovmskb eax, xmm0
        append(codeVex, {0xC5, 0xF8, 0x77});             // vzeroupper
        append(codeVex, {0xC3});
        const int64_t mask = runCode(codeVex);
        CHECK_EQ(mask, 0xFFDF);
    }

    // 25. Test UNPCKLPS and UNPCKHPS (0x0F 0x14 / 0x0F 0x15)
    {
        alignas(16) float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        alignas(16) float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        alignas(16) float outLow[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        alignas(16) float outHigh[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        Bytes code;
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(a));       // rcx = a
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(b));       // rdx = b
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(outLow));  // r8 = outLow
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(outHigh)); // r9 = outHigh

        // unpcklps:
        append(code, {0x0F, 0x28, 0x01});       // movaps xmm0, [rcx] (a: 1, 2, 3, 4)
        append(code, {0x0F, 0x28, 0x0A});       // movaps xmm1, [rdx] (b: 10, 20, 30, 40)
        append(code, {0x0F, 0x14, 0xC1});       // unpcklps xmm0, xmm1 -> (1, 10, 2, 20)
        append(code, {0x41, 0x0F, 0x29, 0x00}); // movaps [r8], xmm0

        // unpckhps:
        append(code, {0x0F, 0x28, 0x01});       // movaps xmm0, [rcx] (a: 1, 2, 3, 4)
        append(code, {0x0F, 0x15, 0xC1});       // unpckhps xmm0, xmm1 -> (3, 30, 4, 40)
        append(code, {0x41, 0x0F, 0x29, 0x01}); // movaps [r9], xmm0
        append(code, {0xC3});

        runCode(code);
        CHECK_EQ(outLow[0], 1.0f);
        CHECK_EQ(outLow[1], 10.0f);
        CHECK_EQ(outLow[2], 2.0f);
        CHECK_EQ(outLow[3], 20.0f);

        CHECK_EQ(outHigh[0], 3.0f);
        CHECK_EQ(outHigh[1], 30.0f);
        CHECK_EQ(outHigh[2], 4.0f);
        CHECK_EQ(outHigh[3], 40.0f);
    }

    // 26. Test SQRTPS (0x0F 0x51) and VEX 3B VPCMPEQW (0x75)
    {
        alignas(16) float inF[4] = {4.0f, 9.0f, 16.0f, 25.0f};
        alignas(16) float outF[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        Bytes code;
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(inF));
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(outF));
        append(code, {0x0F, 0x28, 0x01});       // movaps xmm0, [rcx]
        append(code, {0x0F, 0x51, 0xC0});       // sqrtps xmm0, xmm0
        append(code, {0x0F, 0x29, 0x02});       // movaps [rdx], xmm0
        append(code, {0xC3});

        runCode(code);
        CHECK_EQ(outF[0], 2.0f);
        CHECK_EQ(outF[1], 3.0f);
        CHECK_EQ(outF[2], 4.0f);
        CHECK_EQ(outF[3], 5.0f);

        alignas(16) uint16_t wA[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        alignas(16) uint16_t wB[8] = {1, 9, 3, 9, 5, 9, 7, 9};
        Bytes codeVexW;
        movRegImm64(codeVexW, 1, reinterpret_cast<uintptr_t>(wA));
        movRegImm64(codeVexW, 2, reinterpret_cast<uintptr_t>(wB));
        append(codeVexW, {0xC5, 0xF9, 0x6F, 0x01});                   // vmovdqa xmm0, [rcx]
        append(codeVexW, {0xC4, 0xE1, 0x79, 0x75, 0x02});             // vpcmpeqw xmm0, xmm0, [rdx]
        append(codeVexW, {0xC5, 0xF9, 0xD7, 0xC0});                   // vpmovmskb eax, xmm0
        append(codeVexW, {0xC3});
        const int64_t wmask = runCode(codeVexW);
        CHECK_EQ(wmask, 0x3333);

        // Test RSQRTPS (0x0F 0x52)
        alignas(16) float inR[4] = {4.0f, 16.0f, 64.0f, 256.0f};
        alignas(16) float outR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        Bytes codeR;
        movRegImm64(codeR, 1, reinterpret_cast<uintptr_t>(inR));
        movRegImm64(codeR, 2, reinterpret_cast<uintptr_t>(outR));
        append(codeR, {0x0F, 0x28, 0x01});       // movaps xmm0, [rcx]
        append(codeR, {0x0F, 0x52, 0xC0});       // rsqrtps xmm0, xmm0
        append(codeR, {0x0F, 0x29, 0x02});       // movaps [rdx], xmm0
        append(codeR, {0xC3});

        runCode(codeR);
        CHECK(std::abs(outR[0] - 0.5f) < 0.01f);
        CHECK(std::abs(outR[1] - 0.25f) < 0.01f);
    }

    // 28. Test BTS, BTR, BTC r/m, r (0x0F 0xAB, 0x0F 0xB3, 0x0F 0xBB)
    {
        // BTS: RAX = 0, RCX = 5 -> BTS RAX, RCX => RAX = (1 << 5) = 32
        Bytes codeBts;
        movRegImm64(codeBts, 0, 0); // rax = 0
        movRegImm64(codeBts, 1, 5); // rcx = 5
        append(codeBts, {0x48, 0x0F, 0xAB, 0xC8}); // bts rax, rcx
        append(codeBts, {0xC3});
        CHECK_EQ(runCode(codeBts), 32);

        // BTR: RAX = 32, RCX = 5 -> BTR RAX, RCX => RAX = 0
        Bytes codeBtr;
        movRegImm64(codeBtr, 0, 32); // rax = 32
        movRegImm64(codeBtr, 1, 5);  // rcx = 5
        append(codeBtr, {0x48, 0x0F, 0xB3, 0xC8}); // btr rax, rcx
        append(codeBtr, {0xC3});
        CHECK_EQ(runCode(codeBtr), 0);

        // BTC: RAX = 32, RCX = 5 -> BTC RAX, RCX => RAX = 0
        Bytes codeBtc;
        movRegImm64(codeBtc, 0, 32); // rax = 32
        movRegImm64(codeBtc, 1, 5);  // rcx = 5
        append(codeBtc, {0x48, 0x0F, 0xBB, 0xC8}); // btc rax, rcx
        append(codeBtc, {0xC3});
        CHECK_EQ(runCode(codeBtc), 0);
    }

    // 29. Test CMPXCHG16B and RDRAND (0x0F 0xC7)
    {
        alignas(16) uint64_t mem128[2] = {100, 200};
        Bytes codeCmpxchg;
        movRegImm64(codeCmpxchg, 0, 100); // rax = 100
        movRegImm64(codeCmpxchg, 2, 200); // rdx = 200
        movRegImm64(codeCmpxchg, 3, 300); // rbx = 300
        movRegImm64(codeCmpxchg, 1, 400); // rcx = 400
        movRegImm64(codeCmpxchg, 8, reinterpret_cast<uintptr_t>(mem128)); // r8 = mem128
        append(codeCmpxchg, {0x49, 0x0F, 0xC7, 0x08}); // cmpxchg16b [r8]
        append(codeCmpxchg, {0xC3});
        runCode(codeCmpxchg);
        CHECK_EQ(mem128[0], 300u);
        CHECK_EQ(mem128[1], 400u);

        Bytes codeRdrand;
        append(codeRdrand, {0x48, 0x0F, 0xC7, 0xF0}); // rdrand rax
        append(codeRdrand, {0xC3});
        const int64_t rnd = runCode(codeRdrand);
        CHECK(rnd != 0);
    }
}

void testPcmpistriAndWcsstr() {
    CHECK(configureBackend(false));

    // 1. Test PCMPISTRI con terminador nulo en haystack (ZF=1, CF=0 -> JBE taken, JAE taken)
    {
        alignas(16) char16_t needle[8] = {u'-', u's', u'h', u'o', u'w', u'-', u'c', u'r'};
        alignas(16) char16_t haystack[8] = {u'f', u'o', u'o', 0, 0, 0, 0, 0};

        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(needle));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(haystack));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x09}); // movdqu xmm1, [r9]
        append(code, {0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x0D}); // pcmpistri $0xd, xmm1, xmm0
        append(code, {0x76, 0x06}); // jbe +6 (salta sobre mov eax, 999; ret)
        movRegImm32(code, 0, 999);
        append(code, {0xC3});
        append(code, {0x73, 0x06}); // jae +6 (salta sobre mov eax, 888; ret)
        movRegImm32(code, 0, 888);
        append(code, {0xC3});
        movRegImm32(code, 0, 42);
        append(code, {0xC3});

        CHECK_EQ(runCode(code), 42);
    }

    // 2. Test PCMPISTRI con coincidencia (CF=1 -> indice en RCX)
    {
        alignas(16) char16_t needle[8] = {u'-', u'c', u'r', 0, 0, 0, 0, 0};
        alignas(16) char16_t haystack[8] = {u'a', u'b', u'-', u'c', u'r', 0, 0, 0};

        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(needle));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(haystack));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x09}); // movdqu xmm1, [r9]
        append(code, {0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x0D}); // pcmpistri $0xd, xmm1, xmm0
        append(code, {0x48, 0x89, 0xC8}); // mov rax, rcx
        append(code, {0xC3});

        CHECK_EQ(runCode(code), 2);
    }

    // 3. Equal Any, índice más significativo (imm 0x40, como strrchr de ucrtbase): un '.' DESPUÉS del
    //    NUL (datos viejos) no cuenta. "ab.cd\0x.y" -> último '.' válido en el índice 2.
    {
        alignas(16) char needle[16] = {'.', 0};
        alignas(16) char haystack[16] = {'a', 'b', '.', 'c', 'd', 0, 'x', '.', 'y', '.', 0};

        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(needle));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(haystack));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        append(code, {0x66, 0x41, 0x0F, 0x3A, 0x63, 0x01, 0x40}); // pcmpistri $0x40, (r9), xmm0
        append(code, {0x48, 0x89, 0xC8}); // mov rax, rcx
        append(code, {0xC3});

        CHECK_EQ(runCode(code), 2);
    }

    // 4. Sin coincidencias válidas: el único '.' está tras el NUL -> CF=0, RCX=16.
    {
        alignas(16) char needle[16] = {'.', 0};
        alignas(16) char haystack[16] = {'a', 'b', 0, '.', '.', 0};

        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(needle));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(haystack));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        append(code, {0x66, 0x41, 0x0F, 0x3A, 0x63, 0x01, 0x40}); // pcmpistri $0x40, (r9), xmm0
        append(code, {0x72, 0x04}); // jc +4 (coincidencia: error)
        append(code, {0x48, 0x89, 0xC8}); // mov rax, rcx
        append(code, {0xC3});
        movRegImm32(code, 0, 777);
        append(code, {0xC3});

        CHECK_EQ(runCode(code), 16);
    }
}

void testCmppsCmpPd() {
    CHECK(configureBackend(false));

    const auto f32 = [](float v) -> uint32_t {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
    };
    const auto f64 = [](double v) -> uint64_t {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
    };
    alignas(16) const uint32_t a32[4] = {f32(1.0f), f32(-2.0f), f32(3.0f), 0x7FC00000u};
    alignas(16) const uint32_t b32[4] = {f32(2.0f), f32(-1.0f), f32(3.0f), f32(4.0f)};
    alignas(16) const uint64_t a64[2] = {f64(3.0), 0x7FF8000000000000ull};
    alignas(16) const uint64_t b64[2] = {f64(2.0), f64(-1.0)};
    constexpr uint32_t kAll32 = 0xFFFFFFFFu;
    constexpr uint64_t kAll64 = 0xFFFFFFFFFFFFFFFFull;

    auto runCmpps = [&](uint8_t predicate, bool memorySource) {
        alignas(16) uint32_t out[4] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a32));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(b32));
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        if (memorySource) {
            append(code, {0x41, 0x0F, 0xC2, 0x01, predicate}); // cmpps xmm0, [r9], imm
        } else {
            append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x09}); // movdqu xmm1, [r9]
            append(code, {0x0F, 0xC2, 0xC1, predicate});   // cmpps xmm0, xmm1, imm
        }
        append(code, {0xF3, 0x0F, 0x7F, 0x01}); // movdqu [rcx], xmm0
        append(code, {0xC3});
        runCode(code);
        return std::vector<uint32_t>{out[0], out[1], out[2], out[3]};
    };

    const std::vector<std::vector<uint32_t>> expectedPs = {
        {0, 0, kAll32, 0},          // EQ
        {kAll32, kAll32, 0, 0},     // LT
        {kAll32, kAll32, kAll32, 0},// LE
        {0, 0, 0, kAll32},          // UNORD
        {kAll32, kAll32, 0, kAll32},// NEQ
        {0, 0, kAll32, kAll32},     // NLT
        {0, 0, 0, kAll32},          // NLE
        {kAll32, kAll32, kAll32, 0},// ORD
    };
    for (uint8_t predicate = 0; predicate < 8; ++predicate) {
        CHECK(runCmpps(predicate, false) == expectedPs[predicate]);
        CHECK(runCmpps(predicate, true) == expectedPs[predicate]);
    }

    auto runCmpd = [&](uint8_t predicate, bool memorySource) {
        alignas(16) uint64_t out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a64));
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(b64));
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        if (memorySource) {
            append(code, {0x66, 0x41, 0x0F, 0xC2, 0x01, predicate}); // cmppd xmm0, [r9], imm
        } else {
            append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x09}); // movdqu xmm1, [r9]
            append(code, {0x66, 0x0F, 0xC2, 0xC1, predicate}); // cmppd xmm0, xmm1, imm
        }
        append(code, {0xF3, 0x0F, 0x7F, 0x01});
        append(code, {0xC3});
        runCode(code);
        return std::vector<uint64_t>{out[0], out[1]};
    };

    const std::vector<std::vector<uint64_t>> expectedPd = {
        {0, 0},             // EQ
        {0, 0},             // LT
        {0, 0},             // LE
        {0, kAll64},        // UNORD
        {kAll64, kAll64},   // NEQ
        {kAll64, kAll64},   // NLT
        {kAll64, kAll64},   // NLE
        {kAll64, 0},        // ORD
    };
    for (uint8_t predicate = 0; predicate < 8; ++predicate) {
        CHECK(runCmpd(predicate, false) == expectedPd[predicate]);
        CHECK(runCmpd(predicate, true) == expectedPd[predicate]);
    }
}

void testMovmskpsMovmskpd() {
    CHECK(configureBackend(false));

    auto runMask = [&](bool isDouble, bool memorySource, uint8_t mask) {
        alignas(16) uint32_t words[4] = {};
        for (uint8_t i = 0; i < 4; ++i) {
            const bool negative = isDouble
                ? ((i == 1 && (mask & 1u)) || (i == 3 && (mask & 2u)))
                : (((mask >> i) & 1u) != 0u);
            words[i] = negative ? 0x80000000u : 0x3F800000u;
        }
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(words));
        if (!memorySource) {
            append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00}); // movdqu xmm0, [r8]
        }
        if (isDouble) {
            if (memorySource) {
                append(code, {0x66, 0x41, 0x0F, 0x50, 0x00}); // movmskpd eax, [r8]
            } else {
                append(code, {0x66, 0x0F, 0x50, 0xC0}); // movmskpd eax, xmm0
            }
        } else {
            if (memorySource) {
                append(code, {0x41, 0x0F, 0x50, 0x00}); // movmskps eax, [r8]
            } else {
                append(code, {0x0F, 0x50, 0xC0}); // movmskps eax, xmm0
            }
        }
        append(code, {0xC3});
        return runCode(code);
    };

    for (uint8_t mask = 0; mask < 16; ++mask) {
        CHECK_EQ(runMask(false, false, mask), static_cast<int64_t>(mask));
        CHECK_EQ(runMask(false, true, mask), static_cast<int64_t>(mask));
    }
    for (uint8_t mask = 0; mask < 4; ++mask) {
        CHECK_EQ(runMask(true, false, mask), static_cast<int64_t>(mask));
        CHECK_EQ(runMask(true, true, mask), static_cast<int64_t>(mask));
    }

    // Prefijo F3 no define MOVMSK*: debe fallar explícitamente.
    {
        alignas(16) uint32_t words[4] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(words));
        append(code, {0xF3, 0x0F, 0x50, 0xC0});
        append(code, {0xC3});
        CHECK_EQ(runCode(code), -2);
    }
}

void testVexMxcsr() {
    CHECK(configureBackend(false));

    auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
    CHECK(stack != nullptr);
    constexpr size_t kStackSize = 8u * 1024u * 1024u;

    // VSTMXCSR [RSP+0x20]: C5 F8 AE 5C 24 20 (VEX.128.0F group15 /3). RSP = tope-64 (runCode):
    // [RSP+0x20] queda dentro de la pila (antes +0x40 leía justo fuera de ella).
    Bytes code;
    append(code, {0xC5, 0xF8, 0xAE, 0x5C, 0x24, 0x20});
    append(code, {0xC3});
    runCode(code);
    uint32_t stored = 0;
    std::memcpy(&stored, stack + kStackSize - 32, sizeof(stored));
    CHECK_EQ(stored, 0x1F80u);

    // VLDMXCSR [RSP] seguido de VSTMXCSR [RSP+0x20] no debe fallar ni alterar el valor.
    Bytes code2;
    append(code2, {0xC5, 0xF8, 0xAE, 0x14, 0x24});
    append(code2, {0xC5, 0xF8, 0xAE, 0x5C, 0x24, 0x20});
    append(code2, {0xC3});
    runCode(code2);
    stored = 0;
    std::memcpy(&stored, stack + kStackSize - 32, sizeof(stored));
    CHECK_EQ(stored, 0x1F80u);
}

void testVexScalarConversions() {
    CHECK(configureBackend(false));

    // VCVTSS2SI r32, xmm/m32: C5 7A 2D C1 -> r8d = trunc(xmm1.s[0]).
    {
        alignas(16) const uint32_t src[4] = {0x40700000u, 0, 0, 0}; // 3.75f
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(src));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08}); // movdqu xmm1, [r8]
        append(code, {0xC5, 0x7A, 0x2D, 0xC1});       // vcvtss2si r8d, xmm1 (R~=0 extiende reg)
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(g_backend->context().gpr[8] & 0xFFFFFFFFu, 3u);
    }

    // VCVTSI2SS xmm2, xmm2, r8d: C4 C1 6A 2A D0 -> xmm2.s[0] = 5.0f,
    // los lanes altos deben permanecer intactos.
    {
        alignas(16) const uint32_t initial[4] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
        alignas(16) uint32_t out[4] = {};
        Bytes code;
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(initial));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x11}); // movdqu xmm2, [r9]
        movRegImm64(code, 8, 5);                      // r8d = 5
        append(code, {0xC4, 0xC1, 0x6A, 0x2A, 0xD0}); // vcvtsi2ss xmm2, xmm2, r8d
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x11}); // movdqu [r9], xmm2
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 0x40A00000u); // 5.0f
        CHECK_EQ(out[1], initial[1]);
        CHECK_EQ(out[2], initial[2]);
        CHECK_EQ(out[3], initial[3]);
    }

    // VUCOMISS xmm2, xmm1: C5 F8 2E D1. Devuelve ZF|PF|CF tal como la ruta legacy.
    {
        auto runUcomiss = [&](uint32_t a, uint32_t b) {
            alignas(16) const uint32_t vals[8] = {a, 0, 0, 0, b, 0, 0, 0};
            Bytes code;
            movRegImm64(code, 8, reinterpret_cast<uintptr_t>(vals));
            append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});    // movdqu xmm1, [r8]
            append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x50, 0x10}); // movdqu xmm2, [r8+16]
            append(code, {0xC5, 0xF8, 0x2E, 0xD1});          // vucomiss xmm2, xmm1
            append(code, {0xC3});
            runCode(code);
            return g_backend->context().rflags & 0x45u;
        };
        CHECK_EQ(runUcomiss(0x40400000u, 0x40A00000u), 0u);      // 5 > 3
        CHECK_EQ(runUcomiss(0x40A00000u, 0x40400000u), 0x1u);    // 3 < 5 -> CF
        CHECK_EQ(runUcomiss(0x40400000u, 0x40400000u), 0x40u);   // iguales -> ZF
        CHECK_EQ(runUcomiss(0x40400000u, 0x7FC00000u), 0x45u);   // NaN -> ZF|PF|CF
    }

    // VUCOMISD xmm2, xmm1: C5 F9 2E D1 (pp=66) también debe sintetizar banderas.
    {
        alignas(16) const uint64_t vals[4] = {0x4008000000000000ULL, 0,
                                              0x4014000000000000ULL, 0}; // 3.0 y 5.0
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(vals));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x50, 0x10});
        append(code, {0xC5, 0xF9, 0x2E, 0xD1});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(g_backend->context().rflags & 0x45u, 0u);
    }
}

void testVexPshufd() {
    CHECK(configureBackend(false));

    // VPSHUFD xmm2, xmm0, imm8: C5 F9 70 D0 1B invierte los cuatro dwords.
    {
        alignas(16) const uint32_t src[4] = {1u, 2u, 3u, 4u};
        alignas(16) uint32_t out[4] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(src));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x00});       // movdqu xmm0, [r8]
        append(code, {0xC5, 0xF9, 0x70, 0xD0, 0x1B});       // vpshufd xmm2, xmm0, 0x1b
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x11});       // movdqu [r9], xmm2
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 4u);
        CHECK_EQ(out[1], 3u);
        CHECK_EQ(out[2], 2u);
        CHECK_EQ(out[3], 1u);
    }

    // Fuente memoria: C5 F9 70 08 E4 copia identidad desde [r8] a xmm1.
    {
        alignas(16) const uint32_t src[4] = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
        alignas(16) uint32_t out[4] = {};
        Bytes code;
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(src)); // VEX2 no tiene B~: [rax]
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xC5, 0xF9, 0x70, 0x08, 0xE4});       // vpshufd xmm1, [rax], 0xe4
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x09});       // movdqu [r9], xmm1
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], src[0]);
        CHECK_EQ(out[1], src[1]);
        CHECK_EQ(out[2], src[2]);
        CHECK_EQ(out[3], src[3]);
    }
}

void testVexVectorShifts() {
    CHECK(configureBackend(false));

    // VPSRLQ xmm3, 0x34: C5 E1 73 D3 34 (VEX.128.66.0F 73 /2).
    {
        alignas(16) const uint64_t src[2] = {1ULL << 63, 0xF000000000000000ULL};
        alignas(16) uint64_t out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(src));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x18});       // movdqu xmm3, [r8]
        append(code, {0xC5, 0xE1, 0x73, 0xD3, 0x34});       // vpsrlq xmm3, 0x34
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x19});       // movdqu [r9], xmm3
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], src[0] >> 0x34);
        CHECK_EQ(out[1], src[1] >> 0x34);
    }

    // VPSLLDQ xmm4, 4: C5 E1 73 64 04 (grupo 7) desplaza 4 bytes a la izquierda.
    {
        alignas(16) const uint8_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 13, 14, 15, 16};
        alignas(16) uint8_t out[16] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(src));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});       // movdqu xmm4, [r8]
        append(code, {0xC5, 0xE1, 0x73, 0xFC, 0x04});       // vpslldq xmm3, xmm4, 4
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x19});       // movdqu [r9], xmm3
        append(code, {0xC3});
        runCode(code);
        for (uint8_t i = 0; i < 4; ++i) CHECK_EQ(out[i], 0u);
        for (uint8_t i = 4; i < 16; ++i) CHECK_EQ(out[i], src[i - 4]);
    }
}

void testVexIntegerAddSub() {
    CHECK(configureBackend(false));

    // VPSUBQ xmm3, xmm3, xmm4: C5 E3 FB DC (vvvv=3, reg=3, rm=4).
    {
        alignas(16) const uint64_t a[2] = {10, 20};
        alignas(16) const uint64_t b[2] = {3, 4};
        alignas(16) uint64_t out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x18});       // movdqu xmm3, [r8]
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x21});       // movdqu xmm4, [r9]
        append(code, {0xC5, 0xE1, 0xFB, 0xDC});             // vpsubq xmm3, xmm3, xmm4
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x19});       // movdqu [r9], xmm3
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], a[0] - b[0]);
        CHECK_EQ(out[1], a[1] - b[1]);
    }

    // Fuente memoria (VEX2 no tiene B~, así que [rax]): C5 E3 FB 18.
    {
        alignas(16) const uint64_t a[2] = {100, 200};
        alignas(16) const uint64_t b[2] = {7, 9};
        alignas(16) uint64_t out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x18});       // movdqu xmm3, [r8]
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(b));
        append(code, {0xC5, 0xE1, 0xFB, 0x18});             // vpsubq xmm3, xmm3, [rax]
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x19});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], a[0] - b[0]);
        CHECK_EQ(out[1], a[1] - b[1]);
    }
}

void testVexCvtDq2Pd() {
    CHECK(configureBackend(false));

    // VCVTDQ2PD xmm3, xmm3: C5 FA E6 DB (VEX.128.F3.0F E6 /r).
    {
        alignas(16) const int32_t src[4] = {2, -3, 0, 0};
        alignas(16) uint64_t out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(src));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x18});       // movdqu xmm3, [r8]
        append(code, {0xC5, 0xFA, 0xE6, 0xDB});             // vcvtdq2pd xmm3, xmm3
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x19});       // movdqu [r9], xmm3
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 0x4000000000000000ULL);            // 2.0
        CHECK_EQ(out[1], 0xC008000000000000ULL);            // -3.0
    }
}

void testVex3Fma() {
    CHECK(configureBackend(false));

    // VFMADD213SD xmm4, xmm1, xmm2: C4 E2 F1 A9 E2 => 2*3 + 5 = 11.
    {
        alignas(16) const double a[2] = {2.0};
        alignas(16) const double b[2] = {3.0};
        alignas(16) const double c[2] = {5.0};
        alignas(16) double out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});       // movdqu xmm1, [r8]
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});       // movdqu xmm4, [r8]
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(c));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x10});       // movdqu xmm2, [r8]
        append(code, {0xC4, 0xE2, 0xF1, 0xA9, 0xE2});       // vfmadd213sd xmm4, xmm1, xmm2
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x21});       // movdqu [r9], xmm4
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 11.0);
    }

    // VFMADD213SD xmm4, xmm1, [rax]: C4 E2 F1 A9 20 (fuente memoria).
    {
        alignas(16) const double a[2] = {2.0};
        alignas(16) const double b[2] = {3.0};
        alignas(16) const double c[2] = {5.0};
        alignas(16) double out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});
        movRegImm64(code, 0, reinterpret_cast<uintptr_t>(c));
        append(code, {0xC4, 0xE2, 0xF1, 0xA9, 0x20});       // vfmadd213sd xmm4, xmm1, [rax]
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x21});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 11.0);
    }

    // VFNMADD132SD xmm4, xmm1, xmm2: -(xmm4*xmm2) + xmm1 (escalar).
    {
        alignas(16) const double a[2] = {100.0};
        alignas(16) const double b[2] = {2.0};
        alignas(16) const double c[2] = {4.0};
        alignas(16) double out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(c));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x10});
        append(code, {0xC4, 0xE2, 0xF1, 0x9D, 0xE2});       // vfnmadd132sd
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x21});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 92.0);
    }

    // VFNMADD132PD xmm4, xmm1, xmm2: -(xmm4*xmm2) + xmm1.
    {
        alignas(16) const double a[2] = {100.0, 200.0};
        alignas(16) const double b[2] = {2.0, 3.0};
        alignas(16) const double c[2] = {4.0, 5.0};
        alignas(16) double out[2] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(c));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x10});
        append(code, {0xC4, 0xE2, 0xF1, 0x9C, 0xE2});       // vfnmadd132pd xmm4, xmm1, xmm2
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x21});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 92.0);
        CHECK_EQ(out[1], 185.0);
    }

    // VFMADD231PS xmm4, xmm1, xmm2: xmm2*xmm1 + xmm4.
    {
        alignas(16) const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        alignas(16) const float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        alignas(16) const float c[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        alignas(16) float out[4] = {};
        Bytes code;
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(a));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x08});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(b));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x20});
        movRegImm64(code, 8, reinterpret_cast<uintptr_t>(c));
        append(code, {0xF3, 0x41, 0x0F, 0x6F, 0x10});
        append(code, {0xC4, 0xE2, 0x71, 0xB8, 0xE2});       // vfmadd231ps xmm4, xmm1, xmm2
        movRegImm64(code, 9, reinterpret_cast<uintptr_t>(out));
        append(code, {0xF3, 0x41, 0x0F, 0x7F, 0x21});
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 15.0f);
        CHECK_EQ(out[1], 32.0f);
        CHECK_EQ(out[2], 51.0f);
        CHECK_EQ(out[3], 72.0f);
    }
}

void testVexInstructions() {
    CHECK(configureBackend(false));

    // 1. VMOVQ xmm2, r8 (C4 C1 F9 6E D0) and store back
    {
        alignas(16) uint64_t out[2] = {0, 0};
        Bytes code;
        movRegImm64(code, 8, 0x1122334455667788ULL); // r8 = 0x1122334455667788
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xD0}); // vmovq xmm2, r8
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(out)); // rcx = out
        append(code, {0xC5, 0xF9, 0x7F, 0x11});       // vmovdqu [rcx], xmm2
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 0x1122334455667788ULL);
        CHECK_EQ(out[1], 0u);
    }

    // 2. VUNPCKLPD xmm2, xmm2, xmm2 (C5 E9 6C D2) - duplicates low 64-bit to high 64-bit
    {
        alignas(16) uint64_t out[2] = {0, 0};
        Bytes code;
        movRegImm64(code, 8, 0xAABBCCDDEEFF0011ULL); // r8
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xD0}); // vmovq xmm2, r8
        append(code, {0xC5, 0xE9, 0x6C, 0xD2});       // vunpcklpd xmm2, xmm2, xmm2
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(out)); // rcx = out
        append(code, {0xC5, 0xF9, 0x7F, 0x11});       // vmovdqu [rcx], xmm2
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(out[0], 0xAABBCCDDEEFF0011ULL);
        CHECK_EQ(out[1], 0xAABBCCDDEEFF0011ULL);
    }

    // 3. VINSERTF128 (C4 E3 6D 18 D2 01)
    {
        Bytes code;
        movRegImm64(code, 8, 42);
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xD0}); // vmovq xmm2, r8
        append(code, {0xC4, 0xE3, 0x6D, 0x18, 0xD2, 0x01}); // vinsertf128 ymm2, ymm2, xmm2, 1
        append(code, {0x66, 0x48, 0x0F, 0x7E, 0xD0}); // movq rax, xmm2
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 42);
    }

    // 4. VPXOR (C5 F1 EF C2 -> vpxor xmm0, xmm1, xmm2)
    {
        Bytes code;
        movRegImm64(code, 8, 0x0F0F0F0F);
        movRegImm64(code, 9, 0x00FF00FF);
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xC8}); // vmovq xmm1, r8
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xD1}); // vmovq xmm2, r9
        append(code, {0xC5, 0xF1, 0xEF, 0xC2});       // vpxor xmm0, xmm1, xmm2
        append(code, {0x66, 0x48, 0x0F, 0x7E, 0xC0}); // movq rax, xmm0
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 0x0F0F0F0F ^ 0x00FF00FF);
    }

    // 5. VPCMPEQQ + 256-bit VPMOVMSKB (C4 E2 6D 29 09, C5 FD D7 C1)
    {
        alignas(32) uint64_t mem[4] = {111, 222, 333, 444};
        Bytes code;
        movRegImm64(code, 8, 333);
        append(code, {0xC4, 0xC1, 0xF9, 0x6E, 0xD0}); // vmovq xmm2, r8
        append(code, {0xC5, 0xE9, 0x6C, 0xD2});       // vunpcklpd xmm2, xmm2, xmm2
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(mem)); // rcx = mem
        append(code, {0xC4, 0xE2, 0x6D, 0x29, 0x09}); // vpcmpeqq ymm1, ymm2, [rcx]
        append(code, {0xC5, 0xFD, 0xD7, 0xC1});       // vpmovmskb eax, ymm1
        append(code, {0xC3});
        const int64_t mask = runCode(code);
        CHECK((mask & 0x00FF0000) != 0);
    }

    // 6. 256-bit VMOVDQU load & store (C5 FE 6F / C5 FE 7F) - avoids string duplication
    {
        alignas(32) uint64_t src[4] = {
            0x0123456789ABCDEFULL, 0x1111222233334444ULL,
            0x5555666677778888ULL, 0x9999AAAABBBBCCCCULL
        };
        alignas(32) uint64_t dst[4] = {0, 0, 0, 0};
        Bytes code;
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(src)); // rcx = src
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(dst)); // rdx = dst
        append(code, {0xC5, 0xFE, 0x6F, 0x01}); // vmovdqu ymm0, [rcx]
        append(code, {0xC5, 0xFE, 0x7F, 0x02}); // vmovdqu [rdx], ymm0
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(dst[0], src[0]);
        CHECK_EQ(dst[1], src[1]);
        CHECK_EQ(dst[2], src[2]);
        CHECK_EQ(dst[3], src[3]);
    }

    // 7. VZEROUPPER (C5 F8 77) clears upper 128-bit
    {
        alignas(32) uint64_t src[4] = {
            0x1111111111111111ULL, 0x2222222222222222ULL,
            0x3333333333333333ULL, 0x4444444444444444ULL
        };
        alignas(32) uint64_t dst[4] = {0, 0, 0, 0};
        Bytes code;
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(src)); // rcx = src
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(dst)); // rdx = dst
        append(code, {0xC5, 0xFE, 0x6F, 0x01}); // vmovdqu ymm0, [rcx]
        append(code, {0xC5, 0xF8, 0x77});       // vzeroupper
        append(code, {0xC5, 0xFE, 0x7F, 0x02}); // vmovdqu [rdx], ymm0
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(dst[0], src[0]);
        CHECK_EQ(dst[1], src[1]);
        CHECK_EQ(dst[2], 0ULL);
        CHECK_EQ(dst[3], 0ULL);
    }

    // 8. PSRLDQ / PSLLDQ byte shifts
    {
        alignas(16) uint64_t src[2] = {0x0123456789ABCDEFULL, 0xDEADBEEFCAFEBABELL};
        alignas(16) uint64_t dst[2] = {0, 0};
        Bytes code;
        movRegImm64(code, 1, reinterpret_cast<uintptr_t>(src)); // rcx = src
        movRegImm64(code, 2, reinterpret_cast<uintptr_t>(dst)); // rdx = dst
        append(code, {0xF3, 0x0F, 0x6F, 0x01});       // movdqu xmm0, [rcx]
        append(code, {0x66, 0x0F, 0x73, 0xD8, 0x08}); // psrldq xmm0, 8 (shift right 8 bytes)
        append(code, {0xF3, 0x0F, 0x7F, 0x02});       // movdqu [rdx], xmm0
        append(code, {0xC3});
        runCode(code);
        CHECK_EQ(dst[0], src[1]);
        CHECK_EQ(dst[1], 0ULL);
    }
}

void testUniversalPillars() {
    // 1. Dynamic TEB/PEB with dynamic executable name (Zero hardcoding)
    g_backend->setExecutablePath("/storage/emulated/0/PC games/Cuphead/Cuphead.exe");
    CHECK(g_backend->allocateTeb());

    // Read GS:[0x60] (PEB)
    Bytes code;
    movRegImm64(code, 1, 0x60); // rcx = 0x60
    append(code, {0x65, 0x48, 0x8B, 0x01}); // mov %gs:(%rcx), %rax
    append(code, {0xC3});
    runCode(code);
    uintptr_t pebAddr = static_cast<uintptr_t>(g_backend->context().gpr[0]);
    CHECK(pebAddr != 0);

    // Read ProcessParameters from peb + 0x20
    const uint8_t* peb = reinterpret_cast<const uint8_t*>(pebAddr);
    uint64_t procParamsAddr = *reinterpret_cast<const uint64_t*>(peb + 0x20);
    CHECK(procParamsAddr != 0);
    const uint8_t* params = reinterpret_cast<const uint8_t*>(procParamsAddr);

    // Check ImagePathName.Buffer at params + 0x68
    uint64_t imgBufAddr = *reinterpret_cast<const uint64_t*>(params + 0x68);
    CHECK(imgBufAddr != 0);
    const char16_t* imgBuf = reinterpret_cast<const char16_t*>(imgBufAddr);
    std::u16string expectedImg = u"D:\\Cuphead.exe";
    CHECK(std::u16string(imgBuf) == expectedImg);

    // Check CommandLine.Buffer at params + 0x78
    uint64_t cmdBufAddr = *reinterpret_cast<const uint64_t*>(params + 0x78);
    CHECK(cmdBufAddr != 0);
    const char16_t* cmdBuf = reinterpret_cast<const char16_t*>(cmdBufAddr);
    std::u16string expectedCmd = u"\"D:\\Cuphead.exe\"";
    CHECK(std::u16string(cmdBuf) == expectedCmd);

    // 2. Worker Thread TEB and Cloned TLS
    hyperplay::PeInfo dummyPe;
    dummyPe.tlsRva = 0x1000;
    alignas(16) uint8_t dummyTlsMem[8192] = {};
    struct TlsDir64 {
        uint64_t startRaw;
        uint64_t endRaw;
        uint64_t addrOfIndex;
        uint64_t addrOfCallbacks;
        uint32_t sizeOfZeroFill;
        uint32_t characteristics;
    };
    uint64_t tlsInitData = 0x1122334455667788ULL;
    uint32_t tlsIndex = 0xFFFFFFFF;
    auto* dir = reinterpret_cast<TlsDir64*>(dummyTlsMem + dummyPe.tlsRva);
    dir->startRaw = reinterpret_cast<uintptr_t>(&tlsInitData);
    dir->endRaw = dir->startRaw + sizeof(tlsInitData);
    dir->addrOfIndex = reinterpret_cast<uintptr_t>(&tlsIndex);
    dir->sizeOfZeroFill = 32;

    std::unordered_map<std::string, hyperplay::LocalDllRegion> localDlls;
    g_backend->setupImageTls(dummyTlsMem, dummyPe, localDlls, "/Games/Cuphead/Cuphead.exe");
    CHECK_EQ(tlsIndex, 0u);

    // Main backend TLS vector
    const uint8_t* mainTeb = reinterpret_cast<const uint8_t*>(g_backend->context().gsBase);
    void** mainTlsVec = *reinterpret_cast<void***>(const_cast<uint8_t*>(mainTeb + 0x58));
    CHECK(mainTlsVec != nullptr);
    CHECK(mainTlsVec[0] != nullptr);
    CHECK_EQ(*reinterpret_cast<uint64_t*>(mainTlsVec[0]), tlsInitData);

    // Clone into worker backend
    OryonEmitterBackend worker;
    EmitterConfig cfg;
    cfg.is32Bit = false;
    worker.configure(cfg);
    CHECK(worker.allocateTeb());
    worker.cloneTlsFrom(*g_backend);

    const uint8_t* workerTeb = reinterpret_cast<const uint8_t*>(worker.context().gsBase);
    CHECK(workerTeb != mainTeb);
    void** workerTlsVec = *reinterpret_cast<void***>(const_cast<uint8_t*>(workerTeb + 0x58));
    CHECK(workerTlsVec != nullptr);
    CHECK(workerTlsVec[0] != nullptr);
    CHECK(workerTlsVec[0] != mainTlsVec[0]);
    CHECK_EQ(*reinterpret_cast<uint64_t*>(workerTlsVec[0]), tlsInitData);

    // 3. Dynamic update of executable name (Unreal / idTech engine test)
    g_backend->setExecutablePath("C:\\Games\\DevilMayCry\\DmC-DevilMayCry.exe");
    std::u16string expectedDmc = u"D:\\DmC-DevilMayCry.exe";
    uint64_t dmcBufAddr = *reinterpret_cast<const uint64_t*>(params + 0x68);
    CHECK(std::u16string(reinterpret_cast<const char16_t*>(dmcBufAddr)) == expectedDmc);
}

void testInt3AndExitSemantics() {
    CHECK(configureBackend(false));

    // 1. Test 0xCC (INT 3): decodifica limpiamente, actualiza RIP al siguiente opcode y finaliza bloque
    {
        Bytes code;
        movRegImm64(code, 0, 77); // rax = 77
        append(code, {0xCC});      // int 3 (termina el primer bloque)
        append(code, {0xC3});      // ret (segundo bloque, retorna rax=77)

        CHECK_EQ(runCode(code), 77);
    }

    // 2. Test 0xCD 0x03 (INT imm8)
    {
        Bytes code;
        movRegImm64(code, 0, 88); // rax = 88
        append(code, {0xCD, 0x03}); // int 3 (vector 3 de 2 bytes)
        append(code, {0xC3});        // ret

        CHECK_EQ(runCode(code), 88);
    }
}

void testPrefetchAndMultiByteNop() {
    CHECK(configureBackend(false));

    // Validar decodificacion y ejecucion de 0F 0D (PREFETCH/PREFETCHW), 0F 18 (PREFETCHh) y 0F 19..1F (NOPs multi-byte)
    Bytes code;
    movRegImm64(code, 0, 0x12345678); // rax = 0x12345678
    // 0F 0D /0: PREFETCH [rax]
    append(code, {0x0F, 0x0D, 0x00});
    // 0F 0D /1: PREFETCHW [rax + 0x10]
    append(code, {0x0F, 0x0D, 0x48, 0x10});
    // 0F 18 /0: PREFETCHNTA [rax]
    append(code, {0x0F, 0x18, 0x00});
    // 0F 18 /1: PREFETCHT0 [rax + 0x20]
    append(code, {0x0F, 0x18, 0x48, 0x20});
    // 0F 19 /0: NOP r/m
    append(code, {0x0F, 0x19, 0x00});
    // 0F 1F /0: NOP r/m de 5 bytes (0F 1F 44 00 00)
    append(code, {0x0F, 0x1F, 0x44, 0x00, 0x00});
    // 0F 1F /0: NOP r/m de 8 bytes (0F 1F 84 00 00 00 00 00)
    append(code, {0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00});
    append(code, {0xC3}); // ret

    CHECK_EQ(runCode(code), 0x12345678);
}

void testMxCsrAndX87Fpu() {
    CHECK(configureBackend(false));

    uint8_t buffer[64] = {0};
    uintptr_t bufAddr = reinterpret_cast<uintptr_t>(buffer);

    Bytes code;
    // rax = bufAddr
    movRegImm64(code, 0, bufAddr);

    // 0F AE /3: STMXCSR [rax + 0x10] (disp8 = 0x10)
    // mod=01, reg=011, rm=000 (rax) -> 0x58
    append(code, {0x0F, 0xAE, 0x58, 0x10});

    // D9 /7: FNSTCW [rax + 0x20] (disp8 = 0x20)
    // mod=01, reg=111, rm=000 (rax) -> 0x78
    append(code, {0xD9, 0x78, 0x20});

    // Barreras: LFENCE (0F AE E8), MFENCE (0F AE F0), SFENCE (0F AE F8), WAIT (9B)
    append(code, {0x0F, 0xAE, 0xE8});
    append(code, {0x0F, 0xAE, 0xF0});
    append(code, {0x0F, 0xAE, 0xF8});
    append(code, {0x9B});

    // FNSTSW AX (DF E0)
    append(code, {0xDF, 0xE0});

    // Retorna AX (que debe ser 0)
    append(code, {0xC3});

    CHECK_EQ(runCode(code) & 0xFFFF, 0);

    // Verificar que STMXCSR escribió 0x1F80 en buffer + 0x10
    uint32_t mxcsr = *reinterpret_cast<uint32_t*>(buffer + 0x10);
    CHECK_EQ(mxcsr, 0x1F80u);

    // Verificar que FNSTCW escribió 0x037F en buffer + 0x20
    uint16_t fcw = *reinterpret_cast<uint16_t*>(buffer + 0x20);
    CHECK_EQ(fcw, 0x037Fu);
}

void testPackedConversions() {
    CHECK(configureBackend(false));

    double buffer[2] = {0.0, 0.0};
    uintptr_t bufAddr = reinterpret_cast<uintptr_t>(buffer);

    Bytes code;
    // rdx = bufAddr
    movRegImm64(code, 2, bufAddr);
    // eax = 42
    movRegImm32(code, 0, 42);
    // movd xmm0, eax (66 0F 6E C0)
    append(code, {0x66, 0x0F, 0x6E, 0xC0});
    // cvtdq2pd xmm0, xmm0 (F3 0F E6 C0)
    append(code, {0xF3, 0x0F, 0xE6, 0xC0});
    // movupd [rdx], xmm0 (66 0F 11 02)
    append(code, {0x66, 0x0F, 0x11, 0x02});
    // cvttpd2dq xmm1, xmm0 (66 0F E6 C8)
    append(code, {0x66, 0x0F, 0xE6, 0xC8});
    // movd eax, xmm1 (66 0F 7E C8)
    append(code, {0x66, 0x0F, 0x7E, 0xC8});
    // ret
    append(code, {0xC3});

    CHECK_EQ(runCode(code), 42);
    CHECK_EQ(buffer[0], 42.0);
}

void testHighByteRegisters() {
    CHECK(configureBackend(false));

    uint8_t buffer[8] = {0};
    uintptr_t bufAddr = reinterpret_cast<uintptr_t>(buffer);

    Bytes code;
    // rdx = bufAddr
    movRegImm64(code, 2, bufAddr);
    // eax = 0x12345678 (AH es 0x56, AL es 0x78)
    movRegImm32(code, 0, 0x12345678);

    // 84 E4: test ah, ah
    append(code, {0x84, 0xE4});

    // 88 22: mov [rdx], ah
    append(code, {0x88, 0x22});

    // B5 AA: mov ch, 0xAA
    append(code, {0xB5, 0xAA});

    // 0F B6 C5: movzx eax, ch
    append(code, {0x0F, 0xB6, 0xC5});

    // ret
    append(code, {0xC3});

    CHECK_EQ(runCode(code), 0xAA);
    CHECK_EQ(buffer[0], 0x56);
}

void testPrefix66ImmediateInstructions() {
    CHECK(configureBackend(false));

    // 1. Reproducción fiel de la secuencia de mono-2.0-bdwgc.dll (RVA 0x180297f60):
    // 66 1B C9       : sbb cx, cx (si CF=0 -> cx=0)
    // 66 81 C1 66 01 : add cx, 0x0166 (inmediato de 16-bit! NO 32-bit)
    // 66 89 0A       : mov [rdx], cx
    // C3             : ret
    uint16_t outVal[4] = {0};
    uintptr_t outAddr = reinterpret_cast<uintptr_t>(outVal);

    Bytes code;
    movRegImm64(code, 2, outAddr); // rdx = outAddr
    // clc (f8) para asegurar CF=0
    append(code, {0xF8});
    // sbbw %cx, %cx
    append(code, {0x66, 0x1B, 0xC9});
    // addw $0x0166, %cx
    append(code, {0x66, 0x81, 0xC1, 0x66, 0x01});
    // movw %cx, (%rdx)
    append(code, {0x66, 0x89, 0x0A});
    // movzx eax, cx
    append(code, {0x0F, 0xB7, 0xC1});
    append(code, {0xC3});

    CHECK_EQ(runCode(code), 0x0166);
    CHECK_EQ(outVal[0], 0x0166);

    // 2. MOV r16, imm16 (0x66 0xB8..0xBF) preservando los bits superiores [63:16]
    Bytes code2;
    movRegImm64(code2, 0, 0xDEADBEEFCAFE0000ULL); // rax con bits altos
    // mov ax, 0x5A5A (66 B8 5A 5A)
    append(code2, {0x66, 0xB8, 0x5A, 0x5A});
    append(code2, {0xC3});
    CHECK_EQ(runCode(code2), 0xDEADBEEFCAFE5A5AULL);

    // 3. TEST r/m16, imm16 (0x66 0xF7 /0) sin desalinear la instrucción siguiente
    Bytes code3;
    movRegImm64(code3, 1, 0x1234); // rcx = 0x1234
    // test cx, 0x1000 (66 F7 C1 00 10) -> ZF=0
    append(code3, {0x66, 0xF7, 0xC1, 0x00, 0x10});
    // setnz al (0F 95 C0)
    append(code3, {0x0F, 0x95, 0xC0});
    // movzx eax, al
    append(code3, {0x0F, 0xB6, 0xC0});
    append(code3, {0xC3});
    CHECK_EQ(runCode(code3), 1);
}

void testStringLodsCmps() {
    CHECK(configureBackend(false));

    // 1. LODSD (0xAD) en modo 64-bit
    uint32_t src[4] = {0x12345678, 0x9ABCDEF0, 0x11223344, 0x55667788};
    uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);

    Bytes code1;
    movRegImm64(code1, 6, srcAddr); // rsi = srcAddr
    // lodsd (AD)
    append(code1, {0xAD});
    append(code1, {0xC3});
    CHECK_EQ(runCode(code1), 0x12345678);
    CHECK_EQ(g_backend->context().gpr[6], srcAddr + 4);

    // 2. LODSQ (REX.W + AD)
    uint64_t src64[2] = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
    uintptr_t src64Addr = reinterpret_cast<uintptr_t>(src64);

    Bytes code2;
    movRegImm64(code2, 6, src64Addr); // rsi = src64Addr
    // lodsq (48 AD)
    append(code2, {0x48, 0xAD});
    append(code2, {0xC3});
    CHECK_EQ(runCode(code2), 0x0123456789ABCDEFULL);
    CHECK_EQ(g_backend->context().gpr[6], src64Addr + 8);

    // 3. CMPSB / CMPSD (0xA6 / 0xA7)
    uint32_t bufA[2] = {0x42, 0x99};
    uint32_t bufB[2] = {0x42, 0x88};
    Bytes code3;
    movRegImm64(code3, 6, reinterpret_cast<uintptr_t>(bufA)); // rsi
    movRegImm64(code3, 7, reinterpret_cast<uintptr_t>(bufB)); // rdi
    // cmpsd (A7) -> compara 0x42 con 0x42 -> ZF=1
    append(code3, {0xA7});
    // sete al (0F 94 C0)
    append(code3, {0x0F, 0x94, 0xC0});
    // movzx eax, al
    append(code3, {0x0F, 0xB6, 0xC0});
    append(code3, {0xC3});
    CHECK_EQ(runCode(code3), 1);
    CHECK_EQ(g_backend->context().gpr[6], reinterpret_cast<uintptr_t>(bufA) + 4);
    CHECK_EQ(g_backend->context().gpr[7], reinterpret_cast<uintptr_t>(bufB) + 4);
}

}  // namespace

// Flags muertos entre bloques: clasificador del destino y ejecución con el análisis activo
// (la página del código se registra como imagen estática, igual que una DLL del juego).
void testCrossBlockDeadFlags() {
    using B = OryonEmitterBackend;
    const auto dead = [](std::initializer_list<uint8_t> b) {
        const std::vector<uint8_t> v(b);
        return B::flagsDeadAt(v.data(), v.size(), true);
    };
    CHECK(dead({0x31, 0xC0}));                                      // xor eax, eax
    CHECK(dead({0x48, 0x8B, 0x45, 0x08, 0x48, 0x85, 0xC0}));        // mov rax,[rbp+8]; test rax,rax
    CHECK(dead({0x48, 0x8D, 0x04, 0x24, 0x83, 0xF8, 0x01}));        // lea rax,[rsp]; cmp eax,1
    CHECK(dead({0x48, 0xB8, 1, 2, 3, 4, 5, 6, 7, 8, 0x85, 0xC0}));  // movabs; test
    CHECK(dead({0xF7, 0xD0, 0xF7, 0xD8}));                          // not eax; neg eax
    CHECK(dead({0x0F, 0xB6, 0x41, 0x10, 0x3C, 0x05}));              // movzx eax,[rcx+16]; cmp al,5
    CHECK(!dead({0x11, 0xC0}));                                     // adc: lee CF
    CHECK(!dead({0x83, 0xD0, 0x00}));                               // adc eax, 0
    CHECK(!dead({0x0F, 0x94, 0xC0}));                               // sete
    CHECK(!dead({0xFF, 0xC0}));                                     // inc solo: CF sigue vivo
    CHECK(dead({0x41, 0xFF, 0xC5, 0x48, 0xFF, 0xC5, 0x49, 0x81, 0xC6, 0x80, 0x81, 0, 0}));  // inc; inc; add
    CHECK(!dead({0xFF, 0xC0, 0x83, 0xD0, 0x00}));                   // inc; adc: lee CF
    CHECK(!dead({0xFF, 0xD0}));                                     // call rax
    CHECK(!dead({0x74, 0x00}));                                     // je
    CHECK(!dead({0x0F, 0x44, 0xC1}));                               // cmove
    CHECK(!dead({0xD1, 0xE0}));                                     // shl: no se analiza
    CHECK(!dead({0x48, 0x8B}));                                     // truncado
    CHECK(!dead({0x67, 0x31, 0xC0}));                               // prefijo 67
    CHECK(!dead({0x90, 0x90}));                                     // sin escritura: vivos

    const auto runStatic = [](const Bytes& code) -> int64_t {
        auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
        constexpr size_t kStackSize = 8u * 1024u * 1024u;
        std::memset(stack + kStackSize - 256, 0, 256);
        g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + kStackSize - 64);
        uint8_t* entry = mapCode(code);
        if (entry == nullptr) return -100;
        B::addStaticImage(reinterpret_cast<uintptr_t>(entry) & ~uintptr_t{4095}, 4096);
        return g_backend->executeEntry(entry);
    };
    CHECK(configureBackend(false));
    B::crossBlockFlags.store(true);  // desactivado por defecto: se prueba explícitamente
    struct Restore { ~Restore() { B::crossBlockFlags.store(false); } } restore;
    // El destino lee CF (SBB): la escritura del CMP debe materializarse. 3 < 5 -> CF=1 -> -1.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        append(code, {0x83, 0xF8, 0x05});  // cmp eax, 5
        append(code, {0x75, 0x00});        // jne +0 (fin de bloque, ambos destinos iguales)
        movRegImm32(code, 1, 7);           // mov ecx, 7 (neutra)
        append(code, {0x19, 0xC0});        // sbb eax, eax
        append(code, {0xC3});
        CHECK_EQ(static_cast<uint32_t>(runStatic(code)), 0xFFFFFFFFu);
    }
    // El destino sobrescribe los flags (XOR): resultado intacto y el JE posterior usa los nuevos.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        append(code, {0x83, 0xF8, 0x05});  // cmp eax, 5 (ZF=0)
        append(code, {0x75, 0x00});        // jne +0
        append(code, {0x31, 0xC9});        // xor ecx, ecx (ZF=1)
        append(code, {0x74, 0x06});        // je +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 42);
        append(code, {0xC3});
        CHECK_EQ(runStatic(code), 42);
    }
    // Destino INC + ADD: el CF del CMP (1) no debe llegar a ningún sitio; ADD fija CF=0 y el
    // JB posterior no salta. inc ecx; add ecx, 1 -> ecx = 2; resultado 7.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        movRegImm32(code, 1, 0);
        append(code, {0x83, 0xF8, 0x05});  // cmp eax, 5 (CF=1)
        append(code, {0x75, 0x00});        // jne +0
        append(code, {0xFF, 0xC1});        // inc ecx
        append(code, {0x83, 0xC1, 0x01});  // add ecx, 1 (CF=0)
        append(code, {0x72, 0x06});        // jb +6 (no salta)
        movRegImm32(code, 0, 7);
        append(code, {0xC3});
        movRegImm32(code, 0, 9);
        append(code, {0xC3});
        CHECK_EQ(runStatic(code), 7);
    }
    // Destino INC y luego lectura de CF (ADC): el CF del CMP debe conservarse. inc ecx (CF=1 sigue);
    // adc eax, 0 -> 4.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        append(code, {0x83, 0xF8, 0x05});  // cmp eax, 5 (CF=1)
        append(code, {0x75, 0x00});        // jne +0
        append(code, {0xFF, 0xC1});        // inc ecx
        append(code, {0x83, 0xD0, 0x00});  // adc eax, 0
        append(code, {0xC3});
        CHECK_EQ(runStatic(code), 4);
    }
    // Rama tomada y no tomada a destinos distintos: uno lee (ADC) -> el CMP no se elimina.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        append(code, {0x83, 0xF8, 0x05});  // cmp eax, 5 (CF=1, ZF=0)
        append(code, {0x74, 0x03});        // je +3 (no se toma)
        append(code, {0x83, 0xD0, 0x00});  // adc eax, 0 -> 4
        append(code, {0xC3});
        CHECK_EQ(runStatic(code), 4);
    }
}

// strchr SSE2 de la CRT de Mono (mono-2.0-bdwgc+0x4ed758, sin direcciones relativas): con el GC
// incremental Mono busca '=' en "incremental=..." y la traducción devolvía NULL (caída al iniciar).
// PSHUFLW / PSHUFHW: barajan palabras de 16 bits (no dwords como PSHUFD).
void testPshufLwHw() {
    CHECK(configureBackend(false));
    const auto lane = [](uint8_t pfx, uint8_t imm, uint8_t idx) {
        Bytes code;
        movRegImm32(code, 0, 0x22221111u);
        append(code, {0x66, 0x0F, 0x6E, 0xC0});        // movd xmm0, eax  (w0=1111 w1=2222, resto 0)
        append(code, {0x66, 0x0F, 0x70, 0xC0, 0x00});  // pshufd xmm0, xmm0, 0: dword repetido en los 4
        append(code, {pfx, 0x0F, 0x70, 0xC8, imm});    // pshuflw/hw xmm1, xmm0, imm
        append(code, {0x66, 0x0F, 0xC5, 0xC1, idx});   // pextrw eax, xmm1, idx
        append(code, {0xC3});
        return runCode(code);
    };
    // xmm0 = palabras [1111,2222,1111,2222,1111,2222,1111,2222]
    CHECK_EQ(lane(0xF2, 0x00, 0), 0x1111);  // pshuflw 0: las 4 bajas = w0
    CHECK_EQ(lane(0xF2, 0x00, 1), 0x1111);
    CHECK_EQ(lane(0xF2, 0x00, 3), 0x1111);
    CHECK_EQ(lane(0xF2, 0x00, 5), 0x2222);  // la mitad alta se copia tal cual
    CHECK_EQ(lane(0xF2, 0x55, 0), 0x2222);  // 0x55: todas = w1
    CHECK_EQ(lane(0xF2, 0x1B, 0), 0x2222);  // 0x1B: invertidas (w3)
    CHECK_EQ(lane(0xF3, 0x00, 4), 0x1111);  // pshufhw 0: las 4 altas = w4
    CHECK_EQ(lane(0xF3, 0x00, 7), 0x1111);
    CHECK_EQ(lane(0xF3, 0x00, 1), 0x2222);  // la mitad baja se copia tal cual
    CHECK_EQ(lane(0xF3, 0xFF, 4), 0x2222);  // 0xFF: todas = w7
}

void testMonoCrtStrchr() {
    const Bytes fn = {
        0x0F, 0xB6, 0xC2, 0x4C, 0x8B, 0xC1, 0x44, 0x8B, 0xD0, 0x49, 0x83, 0xE0, 0xF0, 0x41, 0xC1, 0xE2, 0x08, 0x83,
        0xE1, 0x0F, 0x44, 0x0B, 0xD0, 0x45, 0x33, 0xC9, 0x83, 0xC8, 0xFF, 0xD3, 0xE0, 0x66, 0x41, 0x0F, 0x6E, 0xC2,
        0xF2, 0x0F, 0x70, 0xC8, 0x00, 0x0F, 0x57, 0xC0, 0x66, 0x41, 0x0F, 0x74, 0x00, 0x66, 0x0F, 0x70, 0xD1, 0x00,
        0x66, 0x0F, 0x6F, 0xCA, 0x66, 0x41, 0x0F, 0x74, 0x08, 0x66, 0x0F, 0xEB, 0xC8, 0x66, 0x0F, 0xD7, 0xD1, 0x23,
        0xD0, 0x75, 0x21, 0x49, 0x83, 0xC0, 0x10, 0x66, 0x0F, 0x6F, 0xCA, 0x0F, 0x57, 0xC0, 0x66, 0x41, 0x0F, 0x74,
        0x08, 0x66, 0x41, 0x0F, 0x74, 0x00, 0x66, 0x0F, 0xEB, 0xC8, 0x66, 0x0F, 0xD7, 0xD1, 0x85, 0xD2, 0x74, 0xDF,
        0x0F, 0xBC, 0xD2, 0x49, 0x03, 0xD0, 0x44, 0x38, 0x12, 0x4C, 0x0F, 0x44, 0xCA, 0x49, 0x8B, 0xC1, 0xC3};
    CHECK(configureBackend(false));
    alignas(64) static char buf[256];
    const char* texts[] = {"incremental=3", "gc-max-time-slice=3", "a=b", "=", "sin-igual", "x"};
    for (const char* text : texts) {
        const size_t len = std::strlen(text);
        for (size_t align = 0; align < 32; ++align) {
            std::memset(buf, 'z', sizeof(buf));
            std::memcpy(buf + align, text, len + 1);
            const char* expect = std::strchr(buf + align, '=');
            auto* stack = static_cast<uint8_t*>(g_backend->virtualStack());
            constexpr size_t kStackSize = 8u * 1024u * 1024u;
            std::memset(stack + kStackSize - 256, 0, 256);
            g_backend->context().gpr[4] = reinterpret_cast<uintptr_t>(stack + kStackSize - 64);
            g_backend->context().gpr[1] = reinterpret_cast<uintptr_t>(buf + align);
            g_backend->context().gpr[2] = '=';
            uint8_t* entry = mapCode(fn);
            CHECK(entry != nullptr);
            if (entry == nullptr) return;
            const int64_t got = g_backend->executeEntry(entry);
            if (got != reinterpret_cast<int64_t>(expect)) {
                std::printf("strchr('%s', '=') alineación %zu: esperado %p, obtenido %p\n", text, align,
                            static_cast<const void*>(expect), reinterpret_cast<void*>(got));
            }
            CHECK_EQ(got, reinterpret_cast<int64_t>(expect));
        }
    }
}

void testParityFlagAndJcc() {
    // PF=1: 0x03 tiene dos bits a 1. TEST al,al -> JP debe saltar.
    {
        Bytes code;
        movRegImm32(code, 0, 3);
        append(code, {0x84, 0xC0});  // test al, al
        append(code, {0x7A, 0x06});  // jp +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 1);
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 1);
    }
    // PF=0: 0x01 tiene un bit a 1. TEST al,al -> JNP debe saltar.
    {
        Bytes code;
        movRegImm32(code, 0, 1);
        append(code, {0x84, 0xC0});  // test al, al
        append(code, {0x7B, 0x06});  // jnp +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 2);
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 2);
    }
    // PF de ADD de 8 bits: 0x0F + 0xF0 = 0xFF (8 bits -> PF=1).
    {
        Bytes code;
        append(code, {0xB0, 0x0F});  // mov al, 0x0F
        append(code, {0x04, 0xF0});  // add al, 0xF0
        append(code, {0x7A, 0x06});  // jp +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 3);
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 3);
    }
    // PF de CMP de 32 bits: 0x10 - 0x01 = 0x0F (4 bits -> PF=1).
    {
        Bytes code;
        movRegImm32(code, 0, 0x10);
        append(code, {0x83, 0xF8, 0x01});  // cmp eax, 1
        append(code, {0x7A, 0x06});        // jp +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 4);
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 4);
    }
    // PF de SHL de 8 bits: 3 << 1 = 6 (2 bits -> PF=1).
    {
        Bytes code;
        append(code, {0xB0, 0x03});  // mov al, 3
        append(code, {0xD0, 0xE0});  // shl al, 1
        append(code, {0x7A, 0x06});  // jp +6
        movRegImm32(code, 0, 0);
        append(code, {0xC3});
        movRegImm32(code, 0, 5);
        append(code, {0xC3});
        CHECK_EQ(runCode(code), 5);
    }
}

// Hash de nombres de tipo de ReadyToRun (ComputeNameHashCode), bytes reales de coreclr.dll
// 6.0 (RVA 0x205e0): rol 5/8, movsx desde memoria con SIB, lea. Lo usa la búsqueda de tipos
// por nombre de .NET (Stardew Valley no encontraba StardewValley.Mods.ModDataDictionary).
uint32_t refNameHash(const char* s) {
    auto rotl = [](uint32_t v, int n) { return (v << n) | (v >> (32 - n)); };
    if (s == nullptr || s[0] == 0) return 0;
    uint32_t h1 = 0x6DA3B944u, h2 = 0;
    for (size_t i = 0;; i += 2) {
        h1 = (h1 + rotl(h1, 5)) ^ static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(s[i])));
        if (s[i + 1] == 0) break;
        h2 = (h2 + rotl(h2, 5)) ^ static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(s[i + 1])));
        if (s[i + 2] == 0) break;
    }
    return (h2 + rotl(h2, 8)) ^ (rotl(h1, 8) + h1);
}

void testCoreclrNameHash() {
    const Bytes code = {0x4c, 0x8b, 0xd9, 0x48, 0x85, 0xc9, 0x74, 0x5f, 0x44, 0x0f, 0xb6, 0x11, 0x45, 0x84, 0xd2,
                        0x74, 0x56, 0x33, 0xc0, 0x41, 0xb9, 0x44, 0xb9, 0xa3, 0x6d, 0x44, 0x8b, 0xc0, 0x0f, 0x1f,
                        0x40, 0x00, 0x41, 0x8b, 0xd1, 0x41, 0x0f, 0xbe, 0xca, 0xc1, 0xc2, 0x05, 0x44, 0x03, 0xca,
                        0x44, 0x33, 0xc9, 0x41, 0x8d, 0x48, 0x01, 0x42, 0x0f, 0xbe, 0x0c, 0x19, 0x84, 0xc9, 0x74,
                        0x17, 0x8b, 0xd0, 0x41, 0x83, 0xc0, 0x02, 0xc1, 0xc2, 0x05, 0x03, 0xc2, 0x33, 0xc1, 0x47,
                        0x0f, 0xb6, 0x14, 0x18, 0x45, 0x84, 0xd2, 0x75, 0xcc, 0x8b, 0xd0, 0x41, 0x8b, 0xc9, 0xc1,
                        0xc2, 0x08, 0xc1, 0xc1, 0x08, 0x03, 0xc2, 0x41, 0x03, 0xc9, 0x33, 0xc1, 0xc3, 0x33, 0xc0,
                        0xc3};
    const char* names[] = {"StardewValley.Mods", "ModDataDictionary", "System.Xml.Serialization",
                           "XmlSerializerImplementation", "a", "ab", "abc", "Item", "StardewValley"};
    CHECK(configureBackend(false));
    for (const char* n : names) {
        g_backend->context().gpr[1] = reinterpret_cast<uintptr_t>(n);  // RCX
        const auto got = static_cast<uint32_t>(runCode(code));
        if (got != refNameHash(n)) std::printf("  hash('%s') = %08x, esperado %08x\n", n, got, refNameHash(n));
        CHECK_EQ(got, refNameHash(n));
    }
}

void testX87NativeExtended() {
    CHECK(configureBackend(false));

    struct alignas(16) Buf {
        double d0;
        double d1;
        float f0;
        int32_t i0;
        int32_t i1;
        int32_t i2;
        int16_t cwTrunc;
        int16_t cwFloor;
        int16_t cwNearest;
        uint16_t swOut;
    } buf{};
    buf.f0 = 9.0f;
    buf.i0 = 7;
    buf.i1 = -3;
    buf.cwTrunc = 0x0F7F;
    buf.cwFloor = 0x077F;
    buf.cwNearest = 0x037F;
    buf.d1 = -2.75;

    // 1) FNINIT, FLDZ, FLD1, FCHS, FABS, FLD f0 (9.0), FSQRT -> 3.0, FADD -> 4.0, FSTP d0
    Bytes code1;
    movRegImm64(code1, 0, reinterpret_cast<uintptr_t>(&buf));
    append(code1, {0xDB, 0xE3});              // fninit
    append(code1, {0xD9, 0xEE});              // fldz -> 0
    append(code1, {0xD9, 0xE8});              // fld1 -> 1, 0
    append(code1, {0xD9, 0xE0});              // fchs -> -1, 0
    append(code1, {0xD9, 0xE1});              // fabs -> 1, 0
    append(code1, {0xD9, 0x40, 0x10});        // fld dword [rax+0x10] (f0=9.0) -> 9, 1, 0
    append(code1, {0xD9, 0xFA});              // fsqrt -> 3, 1, 0
    append(code1, {0xDE, 0xC1});              // faddp st(1), st(0) -> 4, 0
    append(code1, {0xDD, 0x18});              // fstp qword [rax] (d0=4.0)
    append(code1, {0xDF, 0xE0});              // fnstsw ax
    append(code1, {0xC3});
    const uint64_t sw1 = runCode(code1);
    CHECK_EQ(buf.d0, 4.0);
    // tras fninit + 3 pushes - 2 pops -> top == 7
    CHECK_EQ((sw1 >> 11) & 7u, 7u);

    // 2) FILD / FIADD / FLDCW / FISTP / FISTTP
    Bytes code2;
    movRegImm64(code2, 0, reinterpret_cast<uintptr_t>(&buf));
    append(code2, {0xDB, 0xE3});              // fninit
    append(code2, {0xDB, 0x40, 0x14});        // fild dword [rax+0x14] (i0=7)
    append(code2, {0xDA, 0x40, 0x18});        // fiadd dword [rax+0x18] (i1=-3) -> 4.0
    append(code2, {0xDD, 0x40, 0x08});        // fld qword [rax+0x08] (d1=-2.75)
    append(code2, {0xD9, 0x68, 0x20});        // fldcw [rax+0x20] (trunc)
    append(code2, {0xDB, 0x50, 0x1C});        // fist dword [rax+0x1C] -> i2 = -2
    append(code2, {0xD9, 0x68, 0x22});        // fldcw [rax+0x22] (floor)
    append(code2, {0xDB, 0x58, 0x14});        // fistp dword [rax+0x14] -> i0 = -3
    append(code2, {0xD9, 0x68, 0x24});        // fldcw [rax+0x24] (nearest)
    append(code2, {0xDB, 0x48, 0x18});        // fisttp dword [rax+0x18] -> i1 = 4
    append(code2, {0xC3});
    runCode(code2);
    CHECK_EQ(buf.i2, -2);
    CHECK_EQ(buf.i0, -3);
    CHECK_EQ(buf.i1, 4);

    // 3) FCOM / FCOMP / FNSTSW AX / FNSTSW m16 + FCOMIP + Jcc
    Bytes code3;
    movRegImm64(code3, 0, reinterpret_cast<uintptr_t>(&buf));
    append(code3, {0xDB, 0xE3});              // fninit
    append(code3, {0xD9, 0xE8});              // fld1 (1.0)
    append(code3, {0xD9, 0xEE});              // fldz (0.0) -> ST(0)=0.0, ST(1)=1.0
    append(code3, {0xD8, 0xD1});              // fcom st(1) -> 0 < 1 -> C0=1 (0x0100)
    append(code3, {0xDD, 0x78, 0x26});        // fnstsw [rax+0x26]
    append(code3, {0xDF, 0xF1});              // fcomip st(0), st(1) -> CF=1, ZF=0, PF=0, pops 0.0
    append(code3, {0x72, 0x05});              // jb +5 (saltar si CF=1)
    append(code3, {0xB8, 0x00, 0x00, 0x00, 0x00});  // mov eax, 0
    append(code3, {0xDD, 0xD8});              // fstp st(0) (pop 1.0)
    append(code3, {0xB8, 0x39, 0x05, 0x00, 0x00});  // mov eax, 1337
    append(code3, {0xC3});
    CHECK_EQ(runCode(code3), 1337u);
    CHECK_EQ(buf.swOut & 0x4500u, 0x0100u);

    // 4) FRNDINT + NaN con FCOMIP -> PF=1, ZF=1, CF=1 (JP debe saltar)
    Bytes code4;
    movRegImm64(code4, 0, reinterpret_cast<uintptr_t>(&buf));
    append(code4, {0xDB, 0xE3});              // fninit
    append(code4, {0xDD, 0x40, 0x08});        // fld qword [rax+8] (d1=-2.75)
    append(code4, {0xD9, 0xFC});              // frndint -> -3.0 (nearest)
    append(code4, {0xDD, 0x18});              // fstp qword [rax] (d0=-3.0)
    append(code4, {0xC3});
    runCode(code4);
    CHECK_EQ(buf.d0 == -3.0, 1u);

    Bytes code5;
    buf.d0 = std::numeric_limits<double>::quiet_NaN();
    movRegImm64(code5, 0, reinterpret_cast<uintptr_t>(&buf));
    append(code5, {0xDB, 0xE3});              // fninit
    append(code5, {0xD9, 0xE8});              // fld1
    append(code5, {0xDD, 0x00});              // fld qword [rax] (NaN)
    append(code5, {0xDF, 0xF1});              // fcomip st(0), st(1)
    append(code5, {0x7A, 0x06});              // jp +6
    append(code5, {0xB8, 0x00, 0x00, 0x00, 0x00, 0xC3});  // mov eax, 0; ret
    append(code5, {0xDD, 0xD8});              // fstp st(0)
    append(code5, {0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3});  // mov eax, 42; ret
    CHECK_EQ(runCode(code5), 42u);

    // 5) FXAM (empty, +0.0, -normal, +Inf, NaN)
    auto runFxam = [&](bool empty, double val) -> uint16_t {
        buf.d0 = val;
        Bytes c;
        movRegImm64(c, 0, reinterpret_cast<uintptr_t>(&buf));
        append(c, {0xDB, 0xE3});  // fninit
        if (!empty) append(c, {0xDD, 0x00});  // fld qword [rax]
        append(c, {0xD9, 0xE5});              // fxam
        append(c, {0xDF, 0xE0});              // fnstsw ax
        append(c, {0xC3});
        return static_cast<uint16_t>(runCode(c) & 0x4700u);
    };
    CHECK_EQ(runFxam(true, 0.0), 0x4100u);                                    // empty -> C0|C3
    CHECK_EQ(runFxam(false, 0.0), 0x4000u);                                   // +0 -> C3
    CHECK_EQ(runFxam(false, -3.5), 0x0600u);                                  // -normal -> C1|C2
    CHECK_EQ(runFxam(false, std::numeric_limits<double>::infinity()), 0x0500u); // +inf -> C0|C2
    CHECK_EQ(runFxam(false, std::numeric_limits<double>::quiet_NaN()), 0x0100u); // NaN -> C0
}

static uint32_t fast32DummySum5(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    return a + b * 2 + c * 3 + d * 4 + e * 5;
}

void testFast32TableIndirect() {
    alignas(16) static OryonEmitterBackend::Fast32AliasEntry tbl[16]{};
    tbl[5] = {reinterpret_cast<uintptr_t>(&fast32DummySum5), 5, 0, {}};
    const uintptr_t oldBase = OryonEmitterBackend::fast32AliasBase.exchange(0x7F000000u);
    const auto* oldTbl = OryonEmitterBackend::fast32Table.exchange(tbl);
    auto* oldInd = OryonEmitterBackend::fast32Indirect.exchange(
        [](CpuContext*, uint32_t, uint32_t) -> uint32_t { return 1; });

    CHECK(configureBackend(true));
    Bytes code;
    // push 5, 4, 3, 2, 1 (stdcall 5 args); mov eax, 0x7F000050 (idx 5); call eax; ret
    append(code, {0x6A, 0x05, 0x6A, 0x04, 0x6A, 0x03, 0x6A, 0x02, 0x6A, 0x01});
    append(code, {0xB8, 0x50, 0x00, 0x00, 0x7F});  // mov eax, 0x7F000050
    append(code, {0xFF, 0xD0});                    // call eax
    append(code, {0xC3});                          // ret (ESP debe estar limpio tras stdcall!)
    const uint64_t r = runCode(code, true);
    CHECK_EQ(r, 1u + 4u + 9u + 16u + 25u);  // 55

    // Tail-call (JMP EAX) con retAddr simulado en la pila: call helper -> jmp eax; ret
    Bytes tailCode;
    append(tailCode, {0x6A, 0x05, 0x6A, 0x04, 0x6A, 0x03, 0x6A, 0x02, 0x6A, 0x01});
    append(tailCode, {0xE8, 0x01, 0x00, 0x00, 0x00});  // call +1 (salta sobre C3)
    append(tailCode, {0xC3});                          // ret final
    append(tailCode, {0xB8, 0x50, 0x00, 0x00, 0x7F});  // mov eax, 0x7F000050
    append(tailCode, {0xFF, 0xE0});                    // jmp eax (tail call stdcall)
    const uint64_t rt = runCode(tailCode, true);
    CHECK_EQ(rt, 55u);

    OryonEmitterBackend::fast32AliasBase.store(oldBase);
    OryonEmitterBackend::fast32Table.store(oldTbl);
    OryonEmitterBackend::fast32Indirect.store(oldInd);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    struct sigaction sa {};
    sa.sa_sigaction = [](int sig, siginfo_t* info, void* ucv) {
        auto* uc = static_cast<ucontext_t*>(ucv);
        char buf[512];
        const auto* r = uc->uc_mcontext.regs;
        const int n = std::snprintf(
            buf, sizeof(buf),
            "CRASH signal=%d code=%d addr=%p pc=%p lr=%p sp=%p\n"
            "  lastCheckLine=%d checks=%d\n"
            "  x0=%p x1=%p x2=%p x3=%p x16=%p x17=%p x19=%p x20=%p x21=%p x22=%p\n",
            sig, info != nullptr ? info->si_code : -1,
            info != nullptr ? info->si_addr : nullptr,
            reinterpret_cast<void*>(uc->uc_mcontext.pc),
            reinterpret_cast<void*>(r[30]), reinterpret_cast<void*>(uc->uc_mcontext.sp),
            g_lastCheckLine, g_checks,
            reinterpret_cast<void*>(r[0]), reinterpret_cast<void*>(r[1]),
            reinterpret_cast<void*>(r[2]), reinterpret_cast<void*>(r[3]),
            reinterpret_cast<void*>(r[16]), reinterpret_cast<void*>(r[17]),
            reinterpret_cast<void*>(r[19]), reinterpret_cast<void*>(r[20]),
            reinterpret_cast<void*>(r[21]), reinterpret_cast<void*>(r[22]));
        if (n > 0) {
            const ssize_t ignored = write(STDOUT_FILENO, buf, static_cast<size_t>(n));
            static_cast<void>(ignored);
        }

        char ibuf[256];
        uint32_t insns[8] = {0};
        const uintptr_t pc = uc->uc_mcontext.pc;
        std::memcpy(insns, reinterpret_cast<const void*>(pc), sizeof(insns));
        const int m = std::snprintf(ibuf, sizeof(ibuf),
                                    "  insn: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                                    insns[0], insns[1], insns[2], insns[3], insns[4],
                                    insns[5], insns[6], insns[7]);
        if (m > 0) {
            const ssize_t ignored = write(STDOUT_FILENO, ibuf, static_cast<size_t>(m));
            static_cast<void>(ignored);
        }

        FILE* maps = std::fopen("/proc/self/maps", "r");
        if (maps != nullptr) {
            char line[512];
            while (std::fgets(line, sizeof(line), maps) != nullptr) {
                uintptr_t lo = 0;
                uintptr_t hi = 0;
                if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &lo, &hi) == 2 &&
                    pc >= lo && pc < hi) {
                    const ssize_t ignored =
                        write(STDOUT_FILENO, line, std::strlen(line));
                    static_cast<void>(ignored);
                    break;
                }
            }
            std::fclose(maps);
        }
        _exit(128 + sig);
    };
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);

    OryonEmitterBackend backend;
    g_backend = &backend;

    std::printf("testCoreclrNameHash\n");
    testCoreclrNameHash();
    std::printf("testMovAdd64\n");
    testMovAdd64();
    std::printf("test32BitWrapAndZeroFlag\n");
    test32BitWrapAndZeroFlag();
    std::printf("test8BitWrapAndFlags\n");
    test8BitWrapAndFlags();
    std::printf("testConditionalJump\n");
    testConditionalJump();
    std::printf("testLoop\n");
    testLoop();
    std::printf("testPushPop\n");
    testPushPop();
    std::printf("testCallRet\n");
    testCallRet();
    std::printf("testMemoryAndSib\n");
    testMemoryAndSib();
    std::printf("testLods\n");
    testLods();
    std::printf("testDynamicExecutableRangeInvalidation\n");
    testDynamicExecutableRangeInvalidation();
    std::printf("testWin32DisplayAndKnownFolders\n");
    testWin32DisplayAndKnownFolders();
    std::printf("testLea\n");
    testLea();
    std::printf("testMovzxMovsx\n");
    testMovzxMovsx();
    std::printf("testShifts\n");
    testShifts();
    std::printf("testImul\n");
    testImul();
    std::printf("testXchgMemory\n");
    testXchgMemory();
    std::printf("testSse\n");
    testSse();
    std::printf("testSseXorAndScalar\n");
    testSseXorAndScalar();
    std::printf("testThunkCall\n");
    testThunkCall();
    std::printf("testNativeThunkDispatcher8Args\n");
    testNativeThunkDispatcher8Args();
    std::printf("testNativeThunkDispatcher12Args\n");
    testNativeThunkDispatcher12Args();
    std::printf("testSelfModifyingCode\n");
    testSelfModifyingCode();
    std::printf("testAtomicsMultithread\n");
    testAtomicsMultithread();
    std::printf("testSse41Round\n");
    testSse41Round();
    benchmarkDispatch();
    std::printf("testVectoredExceptionHandlerAccessViolation\n");
    testVectoredExceptionHandlerAccessViolation();
    std::printf("testGuestThreadSuspend\n");
    testGuestThreadSuspend();
    std::printf("testD3D11ResourceModel\n");
    testD3D11ResourceModel();
    std::printf("testWinSafeHeap\n");
    testWinSafeHeap();
    std::printf("testSetccMemory\n");
    testSetccMemory();
    std::printf("testCallGuestNested\n");
    testCallGuestNested();
    std::printf("testShlwapiPaths\n");
    testShlwapiPaths();
    testWin32Registry();
    testPopcnt();
    testSseScalarLanes();
    testPinsrwPextrw();
    testDxbcCorpus();
    testDxbcVm();
    testSoftRaster();
    testD3D11Sampler();
    testBc7();
    testCmpxchg16b();
    testUnalignedAtomics();
    testUnityLz4();
    testShiftRotate16();
    testSse2IntegerOps();
    testBlockChaining();
    testDeadFlags();
    testReciprocalPrecision();
    testFlagFusion();
    testLazyParity();
    testReturnStack();
    testFastNativeCall();
    if (std::getenv("HP_SKIP_AOT") == nullptr) testPersistentJitCache();
    benchmarkSoftRaster();
    std::printf("testFlagsAcrossCall\n");
    testFlagsAcrossCall();
    std::printf("testInstructionFamiliesDecode\n");
    testInstructionFamiliesDecode();
    std::printf("testMode32Program\n");
    testMode32Program();
    std::printf("testStringDirectionFlag32\n");
    testStringDirectionFlag32();
    std::printf("testWxAndUnsupported\n");
    testWxAndUnsupported();
    std::printf("testCmovAndNotNeg\n");
    testCmovAndNotNeg();
    std::printf("testNewInstructionFamilies\n");
    testNewInstructionFamilies();
    std::printf("testPcmpistriAndWcsstr\n");
    testPcmpistriAndWcsstr();
    std::printf("testCmppsCmpPd\n");
    testCmppsCmpPd();
    std::printf("testMovmskpsMovmskpd\n");
    testMovmskpsMovmskpd();
    std::printf("testVexMxcsr\n");
    testVexMxcsr();
    std::printf("testVexScalarConversions\n");
    testVexScalarConversions();
    std::printf("testVexPshufd\n");
    testVexPshufd();
    std::printf("testVexVectorShifts\n");
    testVexVectorShifts();
    std::printf("testVexIntegerAddSub\n");
    testVexIntegerAddSub();
    std::printf("testVexCvtDq2Pd\n");
    testVexCvtDq2Pd();
    std::printf("testVex3Fma\n");
    testVex3Fma();
    std::printf("testVexInstructions\n");
    testVexInstructions();
    std::printf("testUniversalPillars\n");
    testUniversalPillars();
    std::printf("testInt3AndExitSemantics\n");
    testInt3AndExitSemantics();
    std::printf("testPrefetchAndMultiByteNop\n");
    testPrefetchAndMultiByteNop();
    std::printf("testMxCsrAndX87Fpu\n");
    testMxCsrAndX87Fpu();
    std::printf("testPackedConversions\n");
    testPackedConversions();
    std::printf("testHighByteRegisters\n");
    testHighByteRegisters();
    std::printf("testPrefix66ImmediateInstructions\n");
    testPrefix66ImmediateInstructions();
    std::printf("testStringLodsCmps\n");
    testStringLodsCmps();
    std::printf("testParityFlagAndJcc\n");
    testParityFlagAndJcc();
    testCrossBlockDeadFlags();
    testPshufLwHw();
    testMonoCrtStrchr();
    std::printf("testX87NativeExtended\n");
    testX87NativeExtended();
    std::printf("testFast32TableIndirect\n");
    testFast32TableIndirect();

    std::printf("\nOryonEmitterBackendTest: %d comprobaciones, %d fallos\n", g_checks, g_failures);
    if (g_failures == 0) {
        std::printf("OryonEmitterBackendTest: OK\n");
        return 0;
    }
    return 1;
}
