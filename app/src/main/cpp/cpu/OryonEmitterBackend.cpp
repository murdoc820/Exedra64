#include "OryonEmitterBackend.h"

#include <android/log.h>
#include <dlfcn.h>
#include <asm/sigcontext.h>
#include <fcntl.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <sys/system_properties.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <utility>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>

#include "HyperAotCache.h"
#include "GuestProfiler.h"
#include "loader/PeHeaderInspector.h"
#include "loader/BionicPeLoader.h"
#include "loader/NativeDllInterceptor.h"
#include "loader/Win64Context.h"
#include "loader/LowMem.h"

#define LOG_TAG "HyperPlay.OryonEmitter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace {

constexpr size_t kMaxInstructionsPerBlock = 64;
constexpr size_t kMaxGuestBytesPerBlock = 4096;
constexpr size_t kMaxJitBytesPerBlock = 64 * 1024;
constexpr uint64_t kMaxDispatchSteps = 10'000'000;

// Condiciones AArch64.
constexpr uint8_t kCondEq = 0;
constexpr uint8_t kCondNe = 1;

// XMM0..XMM7 -> V0..V7, XMM8..XMM15 -> V16..V23.
// V8..V15 (callee-saved en AAPCS64) no se tocan nunca. V24 es scratch.
inline uint8_t xmmHost(uint8_t reg) {
    return reg < 8 ? reg : static_cast<uint8_t>(reg + 8);
}
constexpr uint8_t kXmmScratch = 24;
constexpr uint8_t kXmmScratch2 = 25;
constexpr uint8_t kXmmScratch3 = 26;
constexpr uint8_t kXmmScratch4 = 27;
constexpr uint8_t kXmmScratch5 = 28;
constexpr uint8_t kXmmScratch6 = 29;

inline int32_t ymmHighOffset(uint8_t guestReg) {
    return static_cast<int32_t>(offsetof(CpuContext, ymmHigh)) + static_cast<int32_t>((guestReg & 0x0Fu) * 16u);
}

inline size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t pageSize() {
    static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return size == 0 ? 4096 : size;
}

void executePcmpistri(CpuContext* ctx, uint8_t reg, uint8_t rm, bool isMem, uint8_t imm8, uintptr_t memAddr) {
    const uint8_t* a = reinterpret_cast<const uint8_t*>(&ctx->simd[reg]);
    const uint8_t* b = isMem ? reinterpret_cast<const uint8_t*>(memAddr) : reinterpret_cast<const uint8_t*>(&ctx->simd[rm]);

    const bool isWords = (imm8 & 0x01) != 0;
    const bool isSigned = (imm8 & 0x02) != 0;
    const uint8_t agg = (imm8 >> 2) & 0x03;
    const uint8_t polarity = (imm8 >> 4) & 0x03;
    const bool isMsbIndex = (imm8 & 0x40) != 0;

    const int maxElems = isWords ? 8 : 16;
    int lenA = maxElems;
    int lenB = maxElems;

    if (isWords) {
        const auto* wa = reinterpret_cast<const uint16_t*>(a);
        const auto* wb = reinterpret_cast<const uint16_t*>(b);
        for (int i = 0; i < 8; ++i) {
            if (wa[i] == 0 && lenA == 8) lenA = i;
            if (wb[i] == 0 && lenB == 8) lenB = i;
        }
    } else {
        for (int i = 0; i < 16; ++i) {
            if (a[i] == 0 && lenA == 16) lenA = i;
            if (b[i] == 0 && lenB == 16) lenB = i;
        }
    }

    uint32_t intRes1 = 0;
    if (isWords) {
        const auto* wa = reinterpret_cast<const uint16_t*>(a);
        const auto* wb = reinterpret_cast<const uint16_t*>(b);
        switch (agg) {
            case 0: // Equal Any
                for (int i = 0; i < 8; ++i) {
                    bool match = false;
                    for (int j = 0; j < lenA; ++j) {
                        if (wa[j] == wb[i]) { match = true; break; }
                    }
                    if (match) intRes1 |= (1u << i);
                }
                break;
            case 1: // Ranges
                for (int i = 0; i < 8; ++i) {
                    bool inRange = false;
                    for (int j = 0; j < lenA / 2; ++j) {
                        const uint16_t lo = wa[2 * j];
                        const uint16_t hi = wa[2 * j + 1];
                        if (isSigned) {
                            const int16_t slo = static_cast<int16_t>(lo);
                            const int16_t shi = static_cast<int16_t>(hi);
                            const int16_t sval = static_cast<int16_t>(wb[i]);
                            if (sval >= slo && sval <= shi) { inRange = true; break; }
                        } else {
                            if (wb[i] >= lo && wb[i] <= hi) { inRange = true; break; }
                        }
                    }
                    if (inRange) intRes1 |= (1u << i);
                }
                break;
            case 2: // Equal Each
                for (int i = 0; i < 8; ++i) {
                    if (wa[i] == wb[i]) intRes1 |= (1u << i);
                }
                break;
            case 3: // Equal Ordered (Substring)
                for (int i = 0; i < 8; ++i) {
                    bool match = true;
                    for (int k = 0; k < lenA; ++k) {
                        if (i + k >= 8 || i + k >= lenB || wa[k] != wb[i + k]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) intRes1 |= (1u << i);
                }
                break;
        }
    } else {
        switch (agg) {
            case 0: // Equal Any
                for (int i = 0; i < 16; ++i) {
                    bool match = false;
                    for (int j = 0; j < lenA; ++j) {
                        if (a[j] == b[i]) { match = true; break; }
                    }
                    if (match) intRes1 |= (1u << i);
                }
                break;
            case 1: // Ranges
                for (int i = 0; i < 16; ++i) {
                    bool inRange = false;
                    for (int j = 0; j < lenA / 2; ++j) {
                        const uint8_t lo = a[2 * j];
                        const uint8_t hi = a[2 * j + 1];
                        if (isSigned) {
                            const int8_t slo = static_cast<int8_t>(lo);
                            const int8_t shi = static_cast<int8_t>(hi);
                            const int8_t sval = static_cast<int8_t>(b[i]);
                            if (sval >= slo && sval <= shi) { inRange = true; break; }
                        } else {
                            if (b[i] >= lo && b[i] <= hi) { inRange = true; break; }
                        }
                    }
                    if (inRange) intRes1 |= (1u << i);
                }
                break;
            case 2: // Equal Each
                for (int i = 0; i < 16; ++i) {
                    if (a[i] == b[i]) intRes1 |= (1u << i);
                }
                break;
            case 3: // Equal Ordered
                for (int i = 0; i < 16; ++i) {
                    bool match = true;
                    for (int k = 0; k < lenA; ++k) {
                        if (i + k >= 16 || i + k >= lenB || a[k] != b[i + k]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) intRes1 |= (1u << i);
                }
                break;
        }
    }

    // Elementos inválidos (tras el terminador implícito), Intel SDM "Valid/Invalid Override":
    //  - Equal Any / Ranges: elemento de SRC2 inválido -> falso (nunca coincide más allá del NUL).
    //  - Equal Each: ambos inválidos -> verdadero; solo uno -> falso.
    //  - Equal Ordered: ya lo resuelven los límites lenA/lenB del bucle.
    // Sin esto, strrchr/strchr de ucrtbase (ruta SSE4.2) encontraban caracteres de datos viejos tras el
    // NUL: coreclr partía "StardewValley.Mods.ModDataDictionary" mal y el TypeLoad de Stardew fallaba.
    if (agg == 0 || agg == 1) {
        for (int i = lenB; i < maxElems; ++i) intRes1 &= ~(1u << i);
    } else if (agg == 2) {
        for (int i = 0; i < maxElems; ++i) {
            const bool va = i < lenA, vb = i < lenB;
            if (!va && !vb) intRes1 |= (1u << i);
            else if (va != vb) intRes1 &= ~(1u << i);
        }
    }

    uint32_t intRes2 = 0;
    const uint32_t maskAll = (1u << maxElems) - 1u;
    switch (polarity) {
        case 0:
            intRes2 = intRes1 & maskAll;
            break;
        case 1:
            intRes2 = (~intRes1) & maskAll;
            break;
        case 2:  // Masked(+): igual que positivo.
            intRes2 = intRes1 & maskAll;
            break;
        case 3:  // Masked(-): niega solo los elementos válidos de SRC2.
            for (int i = 0; i < maxElems; ++i) {
                const bool bit = (intRes1 & (1u << i)) != 0;
                if (i < lenB ? !bit : bit) intRes2 |= (1u << i);
            }
            break;
    }

    int outIndex = maxElems;
    if (intRes2 != 0) {
        outIndex = !isMsbIndex ? __builtin_ctz(intRes2) : (31 - __builtin_clz(intRes2));
    }
    ctx->gpr[1] = static_cast<uint64_t>(static_cast<uint32_t>(outIndex));

    // Semántica estándar Intel x86 (Manual Vol 2B - PCMPISTRI):
    // CF = 1 si IntRes2 != 0 (coincidencia encontrada), sino 0
    // ZF = 1 si SRC2 (rm/memoria, haystack) contiene elemento nulo (lenB < maxElems), sino 0
    // SF = 1 si SRC1 (reg, needle) contiene elemento nulo (lenA < maxElems), sino 0
    // OF = bit 0 de IntRes2
    // AF = 0, PF = 0
    const bool cf = (intRes2 != 0);
    const bool zf = (lenB < maxElems);
    const bool sf = (lenA < maxElems);
    const bool of = (intRes2 & 1) != 0;

    uint64_t rflags = ctx->rflags;
    rflags &= ~((1ull << 0) | (1ull << 2) | (1ull << 4) | (1ull << 6) | (1ull << 7) | (1ull << 11));
    if (cf) rflags |= (1ull << 0);
    if (zf) rflags |= (1ull << 6);
    if (sf) rflags |= (1ull << 7);
    if (of) rflags |= (1ull << 11);
    ctx->rflags = rflags;
}

// ---------------------------------------------------------------------------
// FPU x87 (intérprete nativo; precisión doble). El JIT llama aquí con el estado invitado
// guardado en ctx: op = D8..DF, modrm = byte ModRM (para memoria solo importa reg = bits 5:3),
// ea = dirección efectiva. Devuelve 1 si la instrucción fijó RFLAGS (FCOMI/FUCOMI), 0 si no.
// ---------------------------------------------------------------------------
namespace x87 {

constexpr uint16_t kC0 = 1u << 8, kC1 = 1u << 9, kC2 = 1u << 10, kC3 = 1u << 14;

inline double& st(CpuContext* c, unsigned i) { return c->x87[(c->x87Top + i) & 7u]; }
inline void push(CpuContext* c, double v) {
    c->x87Top = (c->x87Top - 1u) & 7u;
    c->x87[c->x87Top] = v;
    c->x87Tag |= 1u << c->x87Top;
}
inline void pop(CpuContext* c) {
    c->x87Tag &= ~(1u << c->x87Top);
    c->x87Top = (c->x87Top + 1u) & 7u;
}

// Redondeo según la palabra de control (RC = bits 11:10).
double roundCw(const CpuContext* c, double v) {
    switch ((c->x87Cw >> 10) & 3u) {
        case 0: return std::nearbyint(v);  // más cercano (par): modo por defecto del hilo
        case 1: return std::floor(v);
        case 2: return std::ceil(v);
        default: return std::trunc(v);
    }
}

template <typename T>
T toInt(double v) {
    // x87: NaN, infinito o fuera de rango -> "entero indefinido" (el mínimo con signo).
    if (std::isnan(v) || v < static_cast<double>(std::numeric_limits<T>::min()) ||
        v >= -static_cast<double>(std::numeric_limits<T>::min())) {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(v);
}

// 80 bits (signo + exponente de 15 + mantisa de 64 con bit entero explícito) <-> double.
double load80(const uint8_t* p) {
    uint64_t mant;
    uint16_t se;
    std::memcpy(&mant, p, 8);
    std::memcpy(&se, p + 8, 2);
    const bool neg = (se & 0x8000u) != 0;
    const int exp = se & 0x7FFF;
    double v;
    if (exp == 0 && mant == 0) v = 0.0;
    else if (exp == 0x7FFF) v = (mant << 1) == 0 ? INFINITY : NAN;
    else v = std::ldexp(static_cast<double>(mant), exp - 16383 - 63);
    return neg ? -v : v;
}
void store80(uint8_t* p, double v) {
    uint64_t mant = 0;
    uint16_t se = std::signbit(v) ? 0x8000u : 0u;
    const double a = std::fabs(v);
    if (std::isnan(v)) { se |= 0x7FFF; mant = 0xC000000000000000ull; }
    else if (std::isinf(v)) { se |= 0x7FFF; mant = 0x8000000000000000ull; }
    else if (a != 0.0) {
        int e = 0;
        const double f = std::frexp(a, &e);  // a = f * 2^e, f en [0.5, 1)
        mant = static_cast<uint64_t>(std::ldexp(f, 64));
        se |= static_cast<uint16_t>((e - 1 + 16383) & 0x7FFF);
    }
    std::memcpy(p, &mant, 8);
    std::memcpy(p + 8, &se, 2);
}

// Comparación FCOM/FUCOM: C3 C2 C0 = 000 (>), 001 (<), 100 (=), 111 (no ordenados).
void compareSw(CpuContext* c, double a, double b) {
    c->x87Sw &= static_cast<uint16_t>(~(kC0 | kC2 | kC3));
    if (std::isnan(a) || std::isnan(b)) c->x87Sw |= kC0 | kC2 | kC3;
    else if (a < b) c->x87Sw |= kC0;
    else if (a == b) c->x87Sw |= kC3;
}
void compareFlags(CpuContext* c, double a, double b) {
    // FCOMI/FUCOMI: ZF PF CF como COMISD; OF, SF y AF a cero.
    uint64_t f = c->rflags & ~((1ull << 0) | (1ull << 2) | (1ull << 4) | (1ull << 6) | (1ull << 7) | (1ull << 11));
    bool pf = false;
    if (std::isnan(a) || std::isnan(b)) { f |= (1ull << 0) | (1ull << 6); pf = true; }
    else if (a < b) f |= 1ull << 0;
    else if (a == b) f |= 1ull << 6;
    c->rflags = f;
    c->pfByte = pf ? 0u : 1u;
}

// Aritmética D8/DC (y DA/DE con enteros): sub = bits 5:3 de ModRM.
double arith(unsigned sub, double a, double b) {
    switch (sub) {
        case 0: return a + b;
        case 1: return a * b;
        case 4: return a - b;
        case 5: return b - a;
        case 6: return a / b;
        case 7: return b / a;
        default: return a;
    }
}

std::atomic<uint32_t> s_unknownLogged{0};
std::atomic<uint64_t> s_fallbackCalls{0};
std::atomic<uint32_t> s_fallbackHist[8 * 256]{};

void logUnknown(unsigned op, unsigned modrm) {
    if (s_unknownLogged.fetch_add(1) < 32) {
        __android_log_print(ANDROID_LOG_WARN, "HyperPlay.OryonEmitter", "x87 no implementada: %02X %02X", op, modrm);
    }
}

}  // namespace x87

uint64_t executeX87(CpuContext* c, uint32_t op, uint32_t modrm, uintptr_t ea) {
    using namespace x87;
    s_fallbackCalls.fetch_add(1, std::memory_order_relaxed);
    if (op >= 0xD8u && op <= 0xDFu) {
        s_fallbackHist[((op - 0xD8u) << 8) | (modrm & 0xFFu)].fetch_add(1, std::memory_order_relaxed);
    }
    const unsigned sub = (modrm >> 3) & 7u;
    const bool reg = (modrm & 0xC0u) == 0xC0u;
    const unsigned i = modrm & 7u;
    auto* p = reinterpret_cast<uint8_t*>(ea);
    auto rd32f = [&] { float v; std::memcpy(&v, p, 4); return static_cast<double>(v); };
    auto rd64f = [&] { double v; std::memcpy(&v, p, 8); return v; };
    auto rd16i = [&] { int16_t v; std::memcpy(&v, p, 2); return static_cast<double>(v); };
    auto rd32i = [&] { int32_t v; std::memcpy(&v, p, 4); return static_cast<double>(v); };
    auto rd64i = [&] { int64_t v; std::memcpy(&v, p, 8); return static_cast<double>(v); };
    auto wr = [&](const auto v) { std::memcpy(p, &v, sizeof(v)); };
    if (!reg) {
        switch (op) {
            case 0xD8: case 0xDC: case 0xDA: case 0xDE: {
                const double m = op == 0xD8 ? rd32f() : op == 0xDC ? rd64f() : op == 0xDA ? rd32i() : rd16i();
                if (sub == 2 || sub == 3) { compareSw(c, st(c, 0), m); if (sub == 3) pop(c); }
                else st(c, 0) = arith(sub, st(c, 0), m);
                return 0;
            }
            case 0xD9:
                switch (sub) {
                    case 0: push(c, rd32f()); return 0;
                    case 2: wr(static_cast<float>(st(c, 0))); return 0;
                    case 3: wr(static_cast<float>(st(c, 0))); pop(c); return 0;
                    case 5: std::memcpy(&c->x87Cw, p, 2); return 0;              // FLDCW
                    case 7: std::memcpy(p, &c->x87Cw, 2); return 0;              // FNSTCW
                    case 4: return 0;                                              // FLDENV (ignorado)
                    case 6: std::memset(p, 0, 28); std::memcpy(p, &c->x87Cw, 2); return 0;  // FNSTENV
                    default: break;
                }
                break;
            case 0xDB:
                switch (sub) {
                    case 0: push(c, rd32i()); return 0;
                    case 1: wr(toInt<int32_t>(std::trunc(st(c, 0)))); pop(c); return 0;   // FISTTP
                    case 2: wr(toInt<int32_t>(roundCw(c, st(c, 0)))); return 0;
                    case 3: wr(toInt<int32_t>(roundCw(c, st(c, 0)))); pop(c); return 0;
                    case 5: push(c, load80(p)); return 0;
                    case 7: store80(p, st(c, 0)); pop(c); return 0;
                    default: break;
                }
                break;
            case 0xDD:
                switch (sub) {
                    case 0: push(c, rd64f()); return 0;
                    case 1: wr(toInt<int64_t>(std::trunc(st(c, 0)))); pop(c); return 0;   // FISTTP m64
                    case 2: wr(st(c, 0)); return 0;
                    case 3: wr(st(c, 0)); pop(c); return 0;
                    case 7: {                                                      // FNSTSW m16
                        const uint16_t sw = static_cast<uint16_t>((c->x87Sw & ~0x3800u) | ((c->x87Top & 7u) << 11));
                        wr(sw);
                        return 0;
                    }
                    case 4: case 6: return 0;                                      // FRSTOR / FNSAVE
                    default: break;
                }
                break;
            case 0xDF:
                switch (sub) {
                    case 0: push(c, rd16i()); return 0;
                    case 1: wr(toInt<int16_t>(std::trunc(st(c, 0)))); pop(c); return 0;
                    case 2: wr(toInt<int16_t>(roundCw(c, st(c, 0)))); return 0;
                    case 3: wr(toInt<int16_t>(roundCw(c, st(c, 0)))); pop(c); return 0;
                    case 5: push(c, rd64i()); return 0;
                    case 7: wr(toInt<int64_t>(roundCw(c, st(c, 0)))); pop(c); return 0;
                    default: break;
                }
                break;
            default: break;
        }
        logUnknown(op, modrm);
        return 0;
    }
    switch (op) {
        case 0xD8:
            if (sub == 2 || sub == 3) { compareSw(c, st(c, 0), st(c, i)); if (sub == 3) pop(c); }
            else st(c, 0) = arith(sub, st(c, 0), st(c, i));
            return 0;
        case 0xDC:  // ST(i) = ST(i) op ST(0) (con SUB/SUBR y DIV/DIVR intercambiados)
            if (sub == 2 || sub == 3) { compareSw(c, st(c, 0), st(c, i)); if (sub == 3) pop(c); return 0; }
            st(c, i) = arith(sub ^ (sub >= 4 ? 1u : 0u), st(c, i), st(c, 0));
            return 0;
        case 0xDE:
            if (sub == 3 && i == 1) { compareSw(c, st(c, 0), st(c, 1)); pop(c); pop(c); return 0; }  // FCOMPP
            st(c, i) = arith(sub ^ (sub >= 4 ? 1u : 0u), st(c, i), st(c, 0));
            pop(c);
            return 0;
        case 0xD9:
            if (sub == 0) { const double v = st(c, i); push(c, v); return 0; }                   // FLD ST(i)
            if (sub == 1) { std::swap(st(c, 0), st(c, i)); return 0; }                           // FXCH
            switch (modrm) {
                case 0xD0: return 0;                                                              // FNOP
                case 0xE0: st(c, 0) = -st(c, 0); return 0;
                case 0xE1: st(c, 0) = std::fabs(st(c, 0)); return 0;
                case 0xE4: compareSw(c, st(c, 0), 0.0); return 0;                               // FTST
                case 0xE5: {                                                                      // FXAM
                    const double v = st(c, 0);
                    c->x87Sw &= static_cast<uint16_t>(~(kC0 | kC1 | kC2 | kC3));
                    if (std::signbit(v)) c->x87Sw |= kC1;
                    if ((c->x87Tag & (1u << c->x87Top)) == 0) c->x87Sw |= kC0 | kC3;              // vacío
                    else if (std::isnan(v)) c->x87Sw |= kC0;
                    else if (std::isinf(v)) c->x87Sw |= kC0 | kC2;
                    else if (v == 0.0) c->x87Sw |= kC3;
                    else c->x87Sw |= kC2;
                    return 0;
                }
                case 0xE8: push(c, 1.0); return 0;
                case 0xE9: push(c, 3.321928094887362347870319429489390175864831393); return 0;   // log2(10)
                case 0xEA: push(c, 1.442695040888963407359924681001892137426645954); return 0;   // log2(e)
                case 0xEB: push(c, 3.141592653589793238462643383279502884197169399); return 0;
                case 0xEC: push(c, 0.301029995663981195213738894724493026768189881); return 0;   // log10(2)
                case 0xED: push(c, 0.693147180559945309417232121458176568075500134); return 0;   // ln(2)
                case 0xEE: push(c, 0.0); return 0;
                case 0xF0: st(c, 0) = std::exp2(st(c, 0)) - 1.0; return 0;                       // F2XM1
                case 0xF1: st(c, 1) = st(c, 1) * std::log2(st(c, 0)); pop(c); return 0;           // FYL2X
                case 0xF2: st(c, 0) = std::tan(st(c, 0)); push(c, 1.0); c->x87Sw &= ~kC2; return 0;  // FPTAN
                case 0xF3: st(c, 1) = std::atan2(st(c, 1), st(c, 0)); pop(c); return 0;          // FPATAN
                case 0xF4: {                                                                      // FXTRACT
                    int e = 0;
                    const double f = std::frexp(st(c, 0), &e);
                    st(c, 0) = static_cast<double>(e - 1);
                    push(c, f * 2.0);
                    return 0;
                }
                case 0xF5: st(c, 0) = std::remainder(st(c, 0), st(c, 1)); c->x87Sw &= ~kC2; return 0;  // FPREM1
                case 0xF6: c->x87Top = (c->x87Top - 1u) & 7u; return 0;                         // FDECSTP
                case 0xF7: c->x87Top = (c->x87Top + 1u) & 7u; return 0;                         // FINCSTP
                case 0xF8: st(c, 0) = std::fmod(st(c, 0), st(c, 1)); c->x87Sw &= ~kC2; return 0; // FPREM
                case 0xF9: st(c, 1) = st(c, 1) * std::log2(st(c, 0) + 1.0); pop(c); return 0;     // FYL2XP1
                case 0xFA: st(c, 0) = std::sqrt(st(c, 0)); return 0;                             // FSQRT
                case 0xFB: { const double v = st(c, 0); st(c, 0) = std::sin(v); push(c, std::cos(v)); c->x87Sw &= ~kC2; return 0; }
                case 0xFC: st(c, 0) = roundCw(c, st(c, 0)); return 0;                            // FRNDINT
                case 0xFD: st(c, 0) = std::ldexp(st(c, 0), static_cast<int>(std::trunc(st(c, 1)))); return 0;  // FSCALE
                case 0xFE: st(c, 0) = std::sin(st(c, 0)); c->x87Sw &= ~kC2; return 0;
                case 0xFF: st(c, 0) = std::cos(st(c, 0)); c->x87Sw &= ~kC2; return 0;
                default: break;
            }
            break;
        case 0xDA:
            if (modrm == 0xE9) { compareSw(c, st(c, 0), st(c, 1)); pop(c); pop(c); return 0; }     // FUCOMPP
            break;  // FCMOVcc: necesita los flags del huésped (no soportado)
        case 0xDB:
            if (modrm == 0xE2) { c->x87Sw &= 0x7F00u; return 0; }                                // FNCLEX
            if (modrm == 0xE3) { c->x87Top = 0; c->x87Tag = 0; c->x87Sw = 0; c->x87Cw = 0x037F; return 0; }  // FNINIT
            if (sub == 5 || sub == 6) { compareFlags(c, st(c, 0), st(c, i)); return 1; }         // FUCOMI / FCOMI
            break;
        case 0xDD:
            if (sub == 0) { c->x87Tag &= ~(1u << ((c->x87Top + i) & 7u)); return 0; }             // FFREE
            if (sub == 2) { st(c, i) = st(c, 0); return 0; }                                      // FST ST(i)
            if (sub == 3) { st(c, i) = st(c, 0); pop(c); return 0; }                              // FSTP ST(i)
            if (sub == 4 || sub == 5) { compareSw(c, st(c, 0), st(c, i)); if (sub == 5) pop(c); return 0; }  // FUCOM(P)
            break;
        case 0xDF:
            if (modrm == 0xE0) {                                                                  // FNSTSW AX
                const uint16_t sw = static_cast<uint16_t>((c->x87Sw & ~0x3800u) | ((c->x87Top & 7u) << 11));
                c->gpr[0] = (c->gpr[0] & ~0xFFFFull) | sw;
                return 0;
            }
            if (sub == 5 || sub == 6) { compareFlags(c, st(c, 0), st(c, i)); pop(c); return 1; }  // F(U)COMIP
            break;
        default: break;
    }
    logUnknown(op, modrm);
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// CodeReader
// ---------------------------------------------------------------------------

bool OryonEmitterBackend::CodeReader::u8(uint8_t& v) {
    if (!ok || p >= end) {
        ok = false;
        return false;
    }
    v = *p++;
    return true;
}

bool OryonEmitterBackend::CodeReader::s8(int8_t& v) {
    uint8_t b = 0;
    if (!u8(b)) return false;
    v = static_cast<int8_t>(b);
    return true;
}

bool OryonEmitterBackend::CodeReader::u16(uint16_t& v) {
    if (!ok || static_cast<size_t>(end - p) < 2) {
        ok = false;
        return false;
    }
    v = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    p += 2;
    return true;
}

bool OryonEmitterBackend::CodeReader::u32(uint32_t& v) {
    if (!ok || static_cast<size_t>(end - p) < 4) {
        ok = false;
        return false;
    }
    v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return true;
}

bool OryonEmitterBackend::CodeReader::u64(uint64_t& v) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    if (!u32(lo) || !u32(hi)) return false;
    v = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    return true;
}

// ---------------------------------------------------------------------------
// Ciclo de vida y configuración
// ---------------------------------------------------------------------------

OryonEmitterBackend::OryonEmitterBackend() {
    config_ = EmitterConfig{};
    if (!configure(config_)) {
        LOGE("OryonEmitterBackend: no se pudo configurar el JIT: %s", lastError_.c_str());
    }
}

OryonEmitterBackend::~OryonEmitterBackend() {
    stop();
    std::lock_guard<std::mutex> lock(mutex_);
    freeResources();
}

bool OryonEmitterBackend::configure(const EmitterConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    links_.clear(); linksByTarget_.clear();  // la arena se libera: no hay nada que restaurar
    freeResources();
    blockCache_.clear();
    hostIndex_.clear();
    // Todo ceros == FastEntry{} (rip 0, gen 0 nunca coincide con fastGen_ >= 1).
    if (!fastCache_) fastCache_.reset(static_cast<FastEntry*>(std::calloc(kFastCacheSize, sizeof(FastEntry))));
    fastGen_.fetch_add(1, std::memory_order_acq_rel);
    execRanges_.clear();
    failedBlocks_.clear();
    jitOffset_ = 0;
    stopped_ = false;
    lastError_.clear();

    mode32_ = config.is32Bit;
    config_ = config;
    codeBase_ = 0;
    codeSize_ = 0;

    if (!allocateArena(config.codeCacheSize)) {
        lastError_ = "No se pudo reservar la arena JIT";
        return false;
    }
    if (!allocateStack(config.stackSize)) {
        lastError_ = "No se pudo reservar el stack virtual";
        freeResources();
        return false;
    }
    if (!allocateTeb()) {
        lastError_ = "No se pudo reservar el TEB virtual";
        freeResources();
        return false;
    }
    resetContextLocked();
    LOGI("JIT configurado: arena=%zu MB stack=%zu MB modo=%s", config.codeCacheSize / (1024 * 1024),
         config.stackSize / (1024 * 1024), mode32_ ? "32-bit" : "64-bit");
    return true;
}

bool OryonEmitterBackend::allocateArena(size_t size) {
    if (size == 0) return false;

#ifndef __NR_memfd_create
#define __NR_memfd_create 279
#endif
    // 1. Dual-mapping con memfd_create: permite empaquetado denso a 16 bytes y W^X estricto.
    int fd = static_cast<int>(syscall(__NR_memfd_create, "oryon_jit", 1 /* MFD_CLOEXEC */));
    if (fd >= 0) {
        if (ftruncate(fd, static_cast<off_t>(size)) == 0) {
            void* rx = mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
            void* rw = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (rx != MAP_FAILED && rw != MAP_FAILED) {
                arena_ = static_cast<uint8_t*>(rx);
                writeArena_ = static_cast<uint8_t*>(rw);
                arenaSize_ = size;
                execRanges_.push_back({reinterpret_cast<uintptr_t>(arena_),
                                       reinterpret_cast<uintptr_t>(arena_ + size)});
                LOGI("JIT arena dual-mapping configurada con memfd_create (%zu MB RX/RW)", size / (1024 * 1024));
                return true;
            }
            if (rx != MAP_FAILED) munmap(rx, size);
            if (rw != MAP_FAILED) munmap(rw, size);
        } else {
            close(fd);
        }
    }

    // 2. Fallback: mapeo anónimo privado estándar
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        LOGE("mmap arena JIT falló: %s", std::strerror(errno));
        arena_ = nullptr;
        writeArena_ = nullptr;
        arenaSize_ = 0;
        return false;
    }
    arena_ = static_cast<uint8_t*>(p);
    writeArena_ = nullptr;
    arenaSize_ = size;
    return true;
}

// Memoria por debajo de 4 GB para invitados de 32 bits (pila, TEB, TLS: sus punteros son de 4 bytes).
// Android no tiene MAP_32BIT: se buscan huecos libres con MAP_FIXED_NOREPLACE entre 1.5 y 3.75 GB
// (las imágenes PE y el arena del invitado se cargan más abajo; el alias de funciones está en 2 GB-16 MB).
static void* mmapBelow4G(size_t size) { return lowmem::mmapBelow4G(size); }

bool OryonEmitterBackend::allocateStack(size_t size) {
    if (size < 4096) size = 4096;
    void* p = mode32_ ? mmapBelow4G(size)
                      : mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        LOGE("mmap stack virtual falló: %s", std::strerror(errno));
        stackBase_ = nullptr;
        stackSize_ = 0;
        return false;
    }
    stackBase_ = p;
    stackSize_ = size;
    return true;
}

bool OryonEmitterBackend::allocateTeb() {
    if (teb_ != nullptr) return true;
    tebSize_ = 65536; // 64 KB para TEB y TLS vector
    void* p = mode32_ ? mmapBelow4G(tebSize_)
                      : mmap(nullptr, tebSize_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        LOGE("mmap TEB virtual falló: %s", std::strerror(errno));
        teb_ = nullptr;
        tebSize_ = 0;
        return false;
    }
    teb_ = p;
    std::memset(teb_, 0, tebSize_);
    auto* t = static_cast<uint8_t*>(teb_);
    const uint64_t stackBaseAddr = reinterpret_cast<uint64_t>(stackBase_) + stackSize_;
    const uint64_t stackLimitAddr = reinterpret_cast<uint64_t>(stackBase_);
    const uint64_t tebSelf = reinterpret_cast<uint64_t>(teb_);

    // TLS pointer array ubicado en t + 0x800 (soporta hasta 64 módulos)
    void** tlsVector = reinterpret_cast<void**>(t + 0x800);
    if (mode32_) {
        // 32 bits: vector de punteros de 4 bytes y bloques por debajo de 4 GB (un solo mapeo de 256 KB,
        // se queda mientras viva el proceso: los hilos de 32 bits son pocos).
        void* blocks = mmapBelow4G(64 * 4096);
        LOGI("Hilo de 32 bits: pila 0x%lx..0x%lx TEB 0x%lx TLS 0x%lx..0x%lx",
             static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stackBase_)),
             static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stackBase_) + stackSize_),
             static_cast<unsigned long>(reinterpret_cast<uintptr_t>(teb_)),
             static_cast<unsigned long>(reinterpret_cast<uintptr_t>(blocks)),
             static_cast<unsigned long>(reinterpret_cast<uintptr_t>(blocks) + 64 * 4096));
        auto* vec32 = reinterpret_cast<uint32_t*>(t + 0x800);
        for (size_t i = 0; i < 64 && blocks != MAP_FAILED; ++i) {
            vec32[i] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(blocks) + i * 4096);
        }
    } else {
        for (size_t i = 0; i < 64; ++i) {
            void* defBlock = std::calloc(1, 4096);
            if (defBlock) {
                allocatedTlsBlocks_.push_back(defBlock);
                tlsVector[i] = defBlock;
            }
        }
    }
    const uint64_t tlsVectorAddr = reinterpret_cast<uint64_t>(tlsVector);

    const uint64_t pid = static_cast<uint64_t>(getpid());
    const uint64_t tid = static_cast<uint64_t>(gettid());

    if (!mode32_) {
        // x86_64 Win64 TEB layout (GS segment)
        std::memcpy(t + 0x08, &stackBaseAddr, 8);   // StackBase
        std::memcpy(t + 0x10, &stackLimitAddr, 8);  // StackLimit
        std::memcpy(t + 0x30, &tebSelf, 8);         // TebSelf
        std::memcpy(t + 0x40, &pid, 8);             // ClientId.UniqueProcess
        std::memcpy(t + 0x48, &tid, 8);             // ClientId.UniqueThread
        std::memcpy(t + 0x58, &tlsVectorAddr, 8);   // ThreadLocalStoragePointer -> tlsVector
        const uint64_t pebAddr = tebSelf + 0x2000;
        std::memcpy(t + 0x60, &pebAddr, 8);         // Peb pointer

        // PEB layout en t + 0x2000 (Win64)
        uint8_t* peb = t + 0x2000;
        peb[0x02] = 0; // BeingDebugged = false
        const uint64_t imgBase = mainBase_ != nullptr ? reinterpret_cast<uint64_t>(mainBase_) : 0x140000000ULL;
        std::memcpy(peb + 0x10, &imgBase, 8);       // ImageBaseAddress
        const uint64_t procParams = pebAddr + 0x1000;
        std::memcpy(peb + 0x20, &procParams, 8);    // ProcessParameters
        const uint64_t procHeap = 0xCAFE0001ULL;
        std::memcpy(peb + 0x30, &procHeap, 8);      // ProcessHeap
        const uint32_t osMajor = 10;
        const uint32_t osMinor = 0;
        const uint16_t osBuild = 19045;
        const uint32_t osPlatform = 2;              // VER_PLATFORM_WIN32_NT
        const uint32_t numProcs = 8;
        std::memcpy(peb + 0x68, &osMajor, 4);
        std::memcpy(peb + 0x6C, &osMinor, 4);
        std::memcpy(peb + 0x70, &osBuild, 2);
        std::memcpy(peb + 0x74, &osPlatform, 4);
        std::memcpy(peb + 0xB8, &numProcs, 4);

        // Win64 RTL_USER_PROCESS_PARAMETERS en pebAddr + 0x1000 (t + 0x3000)
        uint8_t* params = t + 0x3000;
        char16_t* pStr = reinterpret_cast<char16_t*>(params + 0x400);

        const char16_t strCurDir[] = u"D:\\";
        std::u16string strImagePath = u"D:\\";
        for (char c : exeName_) strImagePath.push_back(static_cast<char16_t>(c));
        std::u16string strCommandLine = u"\"D:\\";
        for (char c : exeName_) strCommandLine.push_back(static_cast<char16_t>(c));
        strCommandLine.push_back(u'\"');

        std::memcpy(pStr, strCurDir, sizeof(strCurDir));
        const uint16_t curDirLen = static_cast<uint16_t>(sizeof(strCurDir) - sizeof(char16_t));
        const uint64_t curDirBuf = reinterpret_cast<uint64_t>(pStr);
        std::memcpy(params + 0x38, &curDirLen, 2); // DosPath.Length
        std::memcpy(params + 0x3A, &curDirLen, 2); // DosPath.MaximumLength
        std::memcpy(params + 0x40, &curDirBuf, 8); // DosPath.Buffer
        pStr += (sizeof(strCurDir) / sizeof(char16_t));

        const size_t imgBytes = (strImagePath.size() + 1) * sizeof(char16_t);
        std::memcpy(pStr, strImagePath.c_str(), imgBytes);
        const uint16_t imgLen = static_cast<uint16_t>(strImagePath.size() * sizeof(char16_t));
        const uint64_t imgBuf = reinterpret_cast<uint64_t>(pStr);
        std::memcpy(params + 0x60, &imgLen, 2);    // ImagePathName.Length
        std::memcpy(params + 0x62, &imgLen, 2);    // ImagePathName.MaximumLength
        std::memcpy(params + 0x68, &imgBuf, 8);    // ImagePathName.Buffer
        pStr += (strImagePath.size() + 1);

        const size_t cmdBytes = (strCommandLine.size() + 1) * sizeof(char16_t);
        std::memcpy(pStr, strCommandLine.c_str(), cmdBytes);
        const uint16_t cmdLen = static_cast<uint16_t>(strCommandLine.size() * sizeof(char16_t));
        const uint64_t cmdBuf = reinterpret_cast<uint64_t>(pStr);
        std::memcpy(params + 0x70, &cmdLen, 2);    // CommandLine.Length
        std::memcpy(params + 0x72, &cmdLen, 2);    // CommandLine.MaximumLength
        std::memcpy(params + 0x78, &cmdBuf, 8);    // CommandLine.Buffer
    } else {
        // x86 Win32 TEB layout (FS segment)
        const uint32_t sBase32 = static_cast<uint32_t>(stackBaseAddr);
        const uint32_t sLimit32 = static_cast<uint32_t>(stackLimitAddr);
        const uint32_t tebSelf32 = static_cast<uint32_t>(tebSelf);
        const uint32_t tlsVec32 = static_cast<uint32_t>(tlsVectorAddr);
        const uint32_t pid32 = static_cast<uint32_t>(pid);
        const uint32_t tid32 = static_cast<uint32_t>(tid);
        const uint32_t pebAddr32 = static_cast<uint32_t>(tebSelf + 0x2000);
        std::memcpy(t + 0x04, &sBase32, 4);         // StackBase
        std::memcpy(t + 0x08, &sLimit32, 4);        // StackLimit
        std::memcpy(t + 0x18, &tebSelf32, 4);       // TebSelf
        std::memcpy(t + 0x20, &pid32, 4);           // ClientId.UniqueProcess
        std::memcpy(t + 0x24, &tid32, 4);           // ClientId.UniqueThread
        std::memcpy(t + 0x2C, &tlsVec32, 4);        // ThreadLocalStoragePointer -> tlsVector
        std::memcpy(t + 0x30, &pebAddr32, 4);       // Peb

        // PEB layout en t + 0x2000 (Win32)
        uint8_t* peb = t + 0x2000;
        peb[0x02] = 0; // BeingDebugged = false
        const uint32_t imgBase32 = mainBase_ != nullptr ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mainBase_)) : 0x00400000u;
        std::memcpy(peb + 0x08, &imgBase32, 4);     // ImageBaseAddress
        const uint32_t procParams32 = pebAddr32 + 0x1000;
        std::memcpy(peb + 0x10, &procParams32, 4);  // ProcessParameters
        const uint32_t procHeap32 = 0xCAFE0001u;
        std::memcpy(peb + 0x18, &procHeap32, 4);    // ProcessHeap
        const uint32_t osMajor = 10;
        const uint32_t osMinor = 0;
        const uint16_t osBuild = 19045;
        const uint32_t osPlatform = 2;
        const uint32_t numProcs = 8;
        std::memcpy(peb + 0x48, &osMajor, 4);
        std::memcpy(peb + 0x4C, &osMinor, 4);
        std::memcpy(peb + 0x50, &osBuild, 2);
        std::memcpy(peb + 0x54, &osPlatform, 4);
        std::memcpy(peb + 0x64, &numProcs, 4);

        // Win32 RTL_USER_PROCESS_PARAMETERS en pebAddr32 + 0x1000 (t + 0x3000)
        uint8_t* params32 = t + 0x3000;
        char16_t* pStr = reinterpret_cast<char16_t*>(params32 + 0x400);
        const char16_t strCurDir[] = u"D:\\";
        std::u16string strImagePath = u"D:\\";
        for (char c : exeName_) strImagePath.push_back(static_cast<char16_t>(c));
        std::u16string strCommandLine = u"\"D:\\";
        for (char c : exeName_) strCommandLine.push_back(static_cast<char16_t>(c));
        strCommandLine.push_back(u'\"');

        std::memcpy(pStr, strCurDir, sizeof(strCurDir));
        const uint16_t curDirLen = static_cast<uint16_t>(sizeof(strCurDir) - sizeof(char16_t));
        const uint32_t curDirBuf = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pStr));
        std::memcpy(params32 + 0x24, &curDirLen, 2);
        std::memcpy(params32 + 0x26, &curDirLen, 2);
        std::memcpy(params32 + 0x28, &curDirBuf, 4);
        pStr += (sizeof(strCurDir) / sizeof(char16_t));

        const size_t imgBytes = (strImagePath.size() + 1) * sizeof(char16_t);
        std::memcpy(pStr, strImagePath.c_str(), imgBytes);
        const uint16_t imgLen = static_cast<uint16_t>(strImagePath.size() * sizeof(char16_t));
        const uint32_t imgBuf = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pStr));
        std::memcpy(params32 + 0x38, &imgLen, 2);
        std::memcpy(params32 + 0x3A, &imgLen, 2);
        std::memcpy(params32 + 0x3C, &imgBuf, 4);
        pStr += (strImagePath.size() + 1);

        const size_t cmdBytes = (strCommandLine.size() + 1) * sizeof(char16_t);
        std::memcpy(pStr, strCommandLine.c_str(), cmdBytes);
        const uint16_t cmdLen = static_cast<uint16_t>(strCommandLine.size() * sizeof(char16_t));
        const uint32_t cmdBuf = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pStr));
        std::memcpy(params32 + 0x40, &cmdLen, 2);
        std::memcpy(params32 + 0x42, &cmdLen, 2);
        std::memcpy(params32 + 0x44, &cmdBuf, 4);
    }
    return true;
}

void OryonEmitterBackend::setExecutablePath(const std::string& exePath) {
    if (exePath.empty()) return;
    size_t slash = exePath.find_last_of("/\\");
    std::string name = (slash != std::string::npos) ? exePath.substr(slash + 1) : exePath;

    std::lock_guard<std::mutex> lock(mutex_);
    exeName_ = name;
    if (teb_ != nullptr) {
        auto* t = static_cast<uint8_t*>(teb_);
        if (!mode32_) {
            uint8_t* params = t + 0x3000;
            char16_t* pStr = reinterpret_cast<char16_t*>(params + 0x400);
            const char16_t strCurDir[] = u"D:\\";
            std::u16string strImagePath = u"D:\\";
            for (char c : exeName_) strImagePath.push_back(static_cast<char16_t>(c));
            std::u16string strCommandLine = u"\"D:\\";
            for (char c : exeName_) strCommandLine.push_back(static_cast<char16_t>(c));
            strCommandLine.push_back(u'\"');

            std::memcpy(pStr, strCurDir, sizeof(strCurDir));
            const uint16_t curDirLen = static_cast<uint16_t>(sizeof(strCurDir) - sizeof(char16_t));
            const uint64_t curDirBuf = reinterpret_cast<uint64_t>(pStr);
            std::memcpy(params + 0x38, &curDirLen, 2);
            std::memcpy(params + 0x3A, &curDirLen, 2);
            std::memcpy(params + 0x40, &curDirBuf, 8);
            pStr += (sizeof(strCurDir) / sizeof(char16_t));

            const size_t imgBytes = (strImagePath.size() + 1) * sizeof(char16_t);
            std::memcpy(pStr, strImagePath.c_str(), imgBytes);
            const uint16_t imgLen = static_cast<uint16_t>(strImagePath.size() * sizeof(char16_t));
            const uint64_t imgBuf = reinterpret_cast<uint64_t>(pStr);
            std::memcpy(params + 0x60, &imgLen, 2);
            std::memcpy(params + 0x62, &imgLen, 2);
            std::memcpy(params + 0x68, &imgBuf, 8);
            pStr += (strImagePath.size() + 1);

            const size_t cmdBytes = (strCommandLine.size() + 1) * sizeof(char16_t);
            std::memcpy(pStr, strCommandLine.c_str(), cmdBytes);
            const uint16_t cmdLen = static_cast<uint16_t>(strCommandLine.size() * sizeof(char16_t));
            const uint64_t cmdBuf = reinterpret_cast<uint64_t>(pStr);
            std::memcpy(params + 0x70, &cmdLen, 2);
            std::memcpy(params + 0x72, &cmdLen, 2);
            std::memcpy(params + 0x78, &cmdBuf, 8);
        }
    }
}

namespace {
// Índices TLS de módulos del proceso (setupImageTls + DLL cargadas después).
std::mutex s_tlsModulesMutex;
uint32_t s_tlsNextIndex = 0;
bool s_tlsSetupDone = false;
std::unordered_set<const void*> s_tlsModuleBases;
}  // namespace

// Bloque TLS estático de un módulo para un hilo. En 32 bits el vector TLS (TEB+0x800 -> ThreadLocalStoragePointer)
// tiene entradas de 4 bytes y los bloques deben estar < 4 GB: antes se guardaba ahí un puntero de 8 bytes de calloc,
// el invitado leía sus 32 bits bajos como dirección y escribía su TLS en memoria ajena (montón de DmC corrupto).
static void* allocTlsBlock(bool mode32, size_t size, std::vector<void*>& owned) {
    if (!mode32) {
        void* b = std::calloc(1, size);
        if (b != nullptr) owned.push_back(b);
        return b;
    }
    void* b = lowmem::mmapBelow4G(size);  // a cero; vive mientras viva el proceso
    return b == MAP_FAILED ? nullptr : b;
}
static void setTlsSlot(void* teb, bool mode32, uint32_t index, void* block) {
    auto* t = static_cast<uint8_t*>(teb) + 0x800;
    if (mode32) {
        const uint32_t p = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(block));
        std::memcpy(t + 4 * index, &p, 4);
    } else {
        reinterpret_cast<void**>(t)[index] = block;
    }
}

void OryonEmitterBackend::setupImageTls(void* mainBase, const PeInfo& mainPe,
                                        const std::unordered_map<std::string, LocalDllRegion>& localDlls,
                                        const std::string& exeFilePath) {
    if (!exeFilePath.empty()) {
        size_t slash = exeFilePath.find_last_of("/\\");
        exeName_ = (slash != std::string::npos) ? exeFilePath.substr(slash + 1) : exeFilePath;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    mainBase_ = mainBase;
    tlsTemplates_.clear();
    if (!allocateTeb()) return;

    auto* t = static_cast<uint8_t*>(teb_);
    uint8_t* peb = t + 0x2000;
    if (!mode32_) {
        const uint64_t imgBase = mainBase != nullptr ? reinterpret_cast<uint64_t>(mainBase) : 0x140000000ULL;
        std::memcpy(peb + 0x10, &imgBase, 8);
    } else {
        const uint32_t imgBase32 = mainBase != nullptr ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(mainBase)) : 0x00400000u;
        std::memcpy(peb + 0x08, &imgBase32, 4);
    }

    void** tlsVector = reinterpret_cast<void**>(t + 0x800);
    uint32_t nextIndex = 0;

    auto initModule = [&](void* base, const PeInfo& pe, const std::string& name) {
        if (base == nullptr || pe.tlsRva == 0 || nextIndex >= 64) return;
        const uint32_t assigned = nextIndex++;
        {
            std::lock_guard<std::mutex> tlsLock(s_tlsModulesMutex);
            s_tlsModuleBases.insert(base);
        }

        if (!mode32_) {
            struct ImageTlsDirectory64 {
                uint64_t startAddressOfRawData;
                uint64_t endAddressOfRawData;
                uint64_t addressOfIndex;
                uint64_t addressOfCallBacks;
                uint32_t sizeOfZeroFill;
                uint32_t characteristics;
            };
            const auto* tlsDir = reinterpret_cast<const ImageTlsDirectory64*>(
                static_cast<const uint8_t*>(base) + pe.tlsRva);

            if (tlsDir->addressOfIndex != 0) {
                *reinterpret_cast<uint32_t*>(tlsDir->addressOfIndex) = assigned;
                LOGI("TLS: Asignado indice %u para %s en addressOfIndex=%p",
                     assigned, name.c_str(), reinterpret_cast<void*>(tlsDir->addressOfIndex));
            }

            size_t rawSize = 0;
            if (tlsDir->endAddressOfRawData > tlsDir->startAddressOfRawData) {
                rawSize = static_cast<size_t>(tlsDir->endAddressOfRawData - tlsDir->startAddressOfRawData);
            }
            size_t totalSize = rawSize + tlsDir->sizeOfZeroFill;
            if (totalSize < 4096) totalSize = 4096;

            TlsModuleTemplate tmpl;
            tmpl.index = assigned;
            tmpl.totalSize = alignUp(totalSize, 16);
            if (rawSize > 0 && tlsDir->startAddressOfRawData != 0) {
                const auto* rawPtr = reinterpret_cast<const uint8_t*>(tlsDir->startAddressOfRawData);
                tmpl.rawData.assign(rawPtr, rawPtr + rawSize);
            }
            tlsTemplates_.push_back(std::move(tmpl));

            void* block = std::calloc(1, alignUp(totalSize, 16));
            if (block) {
                allocatedTlsBlocks_.push_back(block);
                if (rawSize > 0 && tlsDir->startAddressOfRawData != 0) {
                    std::memcpy(block, reinterpret_cast<const void*>(tlsDir->startAddressOfRawData), rawSize);
                }
                tlsVector[assigned] = block;
                LOGI("TLS: Inicializado bloque de %zu bytes para %s en indice %u", totalSize, name.c_str(), assigned);
            }
        } else {
            struct ImageTlsDirectory32 {
                uint32_t startAddressOfRawData;
                uint32_t endAddressOfRawData;
                uint32_t addressOfIndex;
                uint32_t addressOfCallBacks;
                uint32_t sizeOfZeroFill;
                uint32_t characteristics;
            };
            const auto* tlsDir = reinterpret_cast<const ImageTlsDirectory32*>(
                static_cast<const uint8_t*>(base) + pe.tlsRva);

            if (tlsDir->addressOfIndex != 0) {
                *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(tlsDir->addressOfIndex)) = assigned;
            }

            size_t rawSize = 0;
            if (tlsDir->endAddressOfRawData > tlsDir->startAddressOfRawData) {
                rawSize = static_cast<size_t>(tlsDir->endAddressOfRawData - tlsDir->startAddressOfRawData);
            }
            size_t totalSize = rawSize + tlsDir->sizeOfZeroFill;
            if (totalSize < 4096) totalSize = 4096;

            TlsModuleTemplate tmpl;
            tmpl.index = assigned;
            tmpl.totalSize = alignUp(totalSize, 16);
            if (rawSize > 0 && tlsDir->startAddressOfRawData != 0) {
                const auto* rawPtr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(tlsDir->startAddressOfRawData));
                tmpl.rawData.assign(rawPtr, rawPtr + rawSize);
            }
            tlsTemplates_.push_back(std::move(tmpl));

            void* block = allocTlsBlock(true, alignUp(totalSize, 16), allocatedTlsBlocks_);
            if (block) {
                if (rawSize > 0 && tlsDir->startAddressOfRawData != 0) {
                    std::memcpy(block, reinterpret_cast<const void*>(static_cast<uintptr_t>(tlsDir->startAddressOfRawData)), rawSize);
                }
                setTlsSlot(teb_, true, assigned, block);
                LOGI("TLS (32 bits): %s índice %u, %zu bytes (plantilla %zu) en 0x%lx", name.c_str(), assigned, totalSize, rawSize,
                     static_cast<unsigned long>(reinterpret_cast<uintptr_t>(block)));
            }
        }
    };

    // Primero las DLLs locales (ej. UnityPlayer.dll que se inicializan primero)
    std::unordered_set<void*> initializedBases;
    for (const auto& [name, region] : localDlls) {
        if (region.base != nullptr && initializedBases.insert(region.base).second) {
            initModule(region.base, region.peInfo, name);
        }
    }
    // Luego el ejecutable principal
    initModule(mainBase, mainPe, exeName_);
    {
        std::lock_guard<std::mutex> tlsLock(s_tlsModulesMutex);
        s_tlsNextIndex = std::max(s_tlsNextIndex, nextIndex);
        s_tlsSetupDone = true;
    }
}

void OryonEmitterBackend::installTlsTemplate(const TlsModuleTemplate& tmpl) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (teb_ == nullptr || tmpl.index >= 64) return;
    void* block = allocTlsBlock(mode32_, alignUp(tmpl.totalSize, 16), allocatedTlsBlocks_);
    if (block == nullptr) return;
    if (!tmpl.rawData.empty()) std::memcpy(block, tmpl.rawData.data(), tmpl.rawData.size());
    setTlsSlot(teb_, mode32_, tmpl.index, block);
    tlsTemplates_.push_back(tmpl);
}

void OryonEmitterBackend::registerDynamicModuleTls(void* base, const PeInfo& pe, const std::string& name) {
    if (base == nullptr || pe.tlsRva == 0 || !pe.is64Bit) return;
    TlsModuleTemplate tmpl;
    {
        std::lock_guard<std::mutex> tlsLock(s_tlsModulesMutex);
        if (!s_tlsSetupDone || !s_tlsModuleBases.insert(base).second) return;
        if (s_tlsNextIndex >= 64) {
            LOGE("TLS: sin índices libres para %s", name.c_str());
            return;
        }
        tmpl.index = s_tlsNextIndex++;
    }
    struct ImageTlsDirectory64 {
        uint64_t startAddressOfRawData;
        uint64_t endAddressOfRawData;
        uint64_t addressOfIndex;
        uint64_t addressOfCallBacks;
        uint32_t sizeOfZeroFill;
        uint32_t characteristics;
    };
    const auto* tlsDir = reinterpret_cast<const ImageTlsDirectory64*>(static_cast<const uint8_t*>(base) + pe.tlsRva);
    size_t rawSize = 0;
    if (tlsDir->endAddressOfRawData > tlsDir->startAddressOfRawData) {
        rawSize = static_cast<size_t>(tlsDir->endAddressOfRawData - tlsDir->startAddressOfRawData);
    }
    tmpl.totalSize = alignUp(std::max<size_t>(rawSize + tlsDir->sizeOfZeroFill, 4096), 16);
    if (rawSize > 0 && tlsDir->startAddressOfRawData != 0) {
        const auto* raw = reinterpret_cast<const uint8_t*>(tlsDir->startAddressOfRawData);
        tmpl.rawData.assign(raw, raw + rawSize);
    }
    if (tlsDir->addressOfIndex != 0) *reinterpret_cast<uint32_t*>(tlsDir->addressOfIndex) = tmpl.index;
    forEachBackend([&](OryonEmitterBackend& b) { b.installTlsTemplate(tmpl); });
    LOGI("TLS dinámico: %s -> índice %u (%zu bytes, plantilla %zu)", name.c_str(), tmpl.index, tmpl.totalSize,
         rawSize);
}

void OryonEmitterBackend::cloneTlsFrom(const OryonEmitterBackend& parent) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocateTeb()) return;
    auto* t = static_cast<uint8_t*>(teb_);
    tlsTemplates_ = parent.tlsTemplates_;
    for (const auto& tmpl : tlsTemplates_) {
        if (tmpl.index >= 64) continue;
        void* block = allocTlsBlock(mode32_, alignUp(tmpl.totalSize, 16), allocatedTlsBlocks_);
        if (block) {
            if (!tmpl.rawData.empty()) {
                std::memcpy(block, tmpl.rawData.data(), tmpl.rawData.size());
            }
            setTlsSlot(teb_, mode32_, tmpl.index, block);
        }
    }
    if (parent.teb_ != nullptr) {
        if (!mode32_) {
            const uint64_t parentPeb = reinterpret_cast<uint64_t>(parent.teb_) + 0x2000;
            std::memcpy(t + 0x60, &parentPeb, 8);
        } else {
            const uint32_t parentPeb32 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(parent.teb_)) + 0x2000;
            std::memcpy(t + 0x30, &parentPeb32, 4);
        }
    }
}

void OryonEmitterBackend::freeResources() {
    if (writeArena_ != nullptr && arenaSize_ != 0) {
        munmap(writeArena_, arenaSize_);
    }
    writeArena_ = nullptr;
    if (arena_ != nullptr && arenaSize_ != 0) {
        munmap(arena_, arenaSize_);
    }
    arena_ = nullptr;
    arenaSize_ = 0;
    if (stackBase_ != nullptr && stackSize_ != 0) {
        munmap(stackBase_, stackSize_);
    }
    stackBase_ = nullptr;
    stackSize_ = 0;
    if (teb_ != nullptr && tebSize_ != 0) {
        munmap(teb_, tebSize_);
    }
    teb_ = nullptr;
    tebSize_ = 0;
    for (void* p : allocatedTlsBlocks_) {
        std::free(p);
    }
    allocatedTlsBlocks_.clear();
}

void OryonEmitterBackend::resetContextLocked() {
    std::memset(&ctx_, 0, sizeof(ctx_));
    ctx_.mode32 = mode32_ ? 1u : 0u;
    ctx_.x87Cw = 0x037F;
    if (stackBase_ != nullptr && stackSize_ >= (mode32_ ? 64u : 128u)) {
        const uintptr_t top = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        if (mode32_) {
            ctx_.gpr[4] = top;
            std::memset(reinterpret_cast<void*>(top), 0, 64);
        } else {
            // Win64: al entrar en una función RSP % 16 == 8 porque el llamador ya
            // empujó la dirección de retorno, y por encima hay 32 bytes de shadow
            // space. Dejar RSP alineado a 16 hace que Mono detecte la pila
            // desalineada en sus trampolines (tramp-amd64.c stack_unaligned).
            ctx_.gpr[4] = top - 40;
            std::memset(reinterpret_cast<void*>(ctx_.gpr[4]), 0, 104);
        }
    }
    if (teb_ != nullptr) {
        if (!mode32_) {
            ctx_.gsBase = reinterpret_cast<uint64_t>(teb_);
            ctx_.fsBase = 0;
        } else {
            ctx_.fsBase = reinterpret_cast<uint64_t>(teb_);
            ctx_.gsBase = 0;
        }
    }
}

void OryonEmitterBackend::setNativeThunkHandler(NativeThunkHandler handler, void* user) {
    std::lock_guard<std::mutex> lock(mutex_);
    nativeThunk_ = handler;
    nativeThunkUser_ = user;
}

void OryonEmitterBackend::setAotCache(HyperAotCache* cache) {
    std::lock_guard<std::mutex> lock(mutex_);
    aotCache_ = cache;
}

void OryonEmitterBackend::setCodeRange(uintptr_t base, size_t size) {
    EmitterConfig cfg;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cfg = config_;
        cfg.is32Bit = mode32_;
    }
    if (!configure(cfg)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    codeBase_ = base;
    codeSize_ = size;
    codeRanges_.clear();
    if (base != 0 && size != 0) {
        codeRanges_.push_back({base, size});
        codeRangeEnforced_ = true;
    } else {
        // setCodeRange(0, 0) es el reinicio explícito usado por las pruebas.
        codeRangeEnforced_ = false;
    }
}

void OryonEmitterBackend::addCodeRange(uintptr_t base, size_t size) {
    if (base == 0 || size == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : codeRanges_) {
        // Ya cubierto (VirtualProtect repetido sobre la misma reserva): no duplicar; la lista
        // se recorre en cada traducción.
        if (base >= r.base && base + size <= r.base + r.size) return;
    }
    codeRanges_.push_back({base, size});
    codeRangeEnforced_ = true;
}

void OryonEmitterBackend::addExecutableRange(uintptr_t base, size_t size) {
    addCodeRange(base, size);
}

bool OryonEmitterBackend::findCodeRange(uintptr_t rip, CodeRange& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : codeRanges_) {
        if (rip >= r.base && rip < r.base + r.size) {
            out = r;
            return true;
        }
    }
    return false;
}

size_t OryonEmitterBackend::eraseBlocksLocked(uintptr_t base, uintptr_t end) {
    size_t erased = 0;
    for (auto it = blockCache_.lower_bound(base); it != blockCache_.end() && it->first < end;) {
        // Solo la entrada de la caché rápida de este bloque (antes se recorrían las 64 K entradas
        // en cada escritura de Mono sobre código: ~20 % del hilo principal en partida).
        if (fastCache_) {
            const uintptr_t rip = it->first;
            FastEntry& e = fastCache_[(rip ^ (rip >> 16)) & (kFastCacheSize - 1)];
            if (e.rip == rip) __atomic_store_n(&e.gen, 0, __ATOMIC_RELEASE);
        }
        it = blockCache_.erase(it);
        ++erased;
    }
    if (erased > 0) unlinkRangeLocked(base, end);  // ningún salto enlazado puede llevar a código viejo
    return erased;
}

void OryonEmitterBackend::invalidateCodeRange(uintptr_t base, size_t size) {
    if (base == 0 || size == 0) return;
    const uintptr_t end = (size > std::numeric_limits<uintptr_t>::max() - base)
                              ? std::numeric_limits<uintptr_t>::max()
                              : base + size;
    std::lock_guard<std::mutex> lock(mutex_);
    eraseBlocksLocked(base, end);
    for (auto it = failedBlocks_.begin(); it != failedBlocks_.end();) {
        if (it->first >= base && it->first < end) {
            it = failedBlocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void OryonEmitterBackend::removeExecutableRange(uintptr_t base, size_t size) {
    if (base == 0 || size == 0) return;
    const uintptr_t end = (size > std::numeric_limits<uintptr_t>::max() - base)
                              ? std::numeric_limits<uintptr_t>::max()
                              : base + size;
    std::lock_guard<std::mutex> lock(mutex_);
    eraseBlocksLocked(base, end);
    for (auto it = failedBlocks_.begin(); it != failedBlocks_.end();) {
        if (it->first >= base && it->first < end) {
            it = failedBlocks_.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<CodeRange> kept;
    kept.reserve(codeRanges_.size() + 1);
    for (const CodeRange& r : codeRanges_) {
        const uintptr_t rEnd = (r.size > std::numeric_limits<uintptr_t>::max() - r.base)
                                   ? std::numeric_limits<uintptr_t>::max()
                                   : r.base + r.size;
        if (rEnd <= base || r.base >= end) {
            kept.push_back(r);
            continue;
        }
        if (r.base < base) kept.push_back({r.base, base - r.base});
        if (rEnd > end) kept.push_back({end, rEnd - end});
    }
    codeRanges_.swap(kept);
}

void OryonEmitterBackend::registerModule(const std::string& name, uintptr_t base, size_t size) {
    if (base == 0 || size == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& mod : modules_) {
        if (mod.name == name || mod.base == base) {
            mod = ModuleMetadata{name, base, size};
            return;
        }
    }
    modules_.push_back(ModuleMetadata{name, base, size});
}

void OryonEmitterBackend::recordThunkCall(const std::string& name, uint64_t arg0, uint64_t arg1) {
    flightRecorder_[flightCount_ % kFlightSize] = FlightRecord{&name, arg0, arg1, ctx_.rip};
    ++flightCount_;
}

std::string OryonEmitterBackend::formatDiagnosticReport(uintptr_t rip, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    return formatDiagnosticReportLocked(rip, reason);
}

std::string OryonEmitterBackend::formatDiagnosticReportLocked(uintptr_t rip, const std::string& reason) const {
    std::string modName = "desconocido";
    uintptr_t rva = rip;
    for (const auto& mod : modules_) {
        if (rip >= mod.base && rip < mod.base + mod.size) {
            modName = mod.name;
            rva = rip - mod.base;
            break;
        }
    }

    char hdr[256];
    std::snprintf(hdr, sizeof(hdr),
                  "%s en [%s + 0x%llx] (RIP=0x%llx)\nBytes en RIP: ",
                  reason.c_str(), modName.c_str(),
                  static_cast<unsigned long long>(rva),
                  static_cast<unsigned long long>(rip));
    std::string out = hdr;

    const uint8_t* codePtr = reinterpret_cast<const uint8_t*>(rip);
    for (int i = 0; i < 16; ++i) {
        char byteBuf[8];
        std::snprintf(byteBuf, sizeof(byteBuf), "%02X ", codePtr != nullptr ? codePtr[i] : 0);
        out += byteBuf;
    }
    out += "\nÚltimas llamadas Win32 / Auto-Stubs (Flight Recorder):";
    if (flightCount_ == 0) {
        out += " (ninguna)";
    } else {
        const uint64_t first = flightCount_ > 16 ? flightCount_ - 16 : 0;
        for (uint64_t i = first; i < flightCount_; ++i) {
            const FlightRecord& r = flightRecorder_[i % kFlightSize];
            char recBuf[256];
            std::snprintf(recBuf, sizeof(recBuf), "\n  [%llu] %s (arg0=0x%llx, arg1=0x%llx, RIP=0x%llx)",
                          static_cast<unsigned long long>(i), r.name ? r.name->c_str() : "?",
                          static_cast<unsigned long long>(r.arg0), static_cast<unsigned long long>(r.arg1),
                          static_cast<unsigned long long>(r.rip));
            out += recBuf;
        }
    }
    return out;
}

void OryonEmitterBackend::setGuestMode(bool is32Bit) {
    EmitterConfig cfg;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mode32_ == is32Bit) return;
        cfg = config_;
    }
    cfg.is32Bit = is32Bit;
    configure(cfg);
}

// ---------------------------------------------------------------------------
// Emisión de palabras y parcheo de saltos
// ---------------------------------------------------------------------------

void OryonEmitterBackend::emitReadTsc() {
    emitWord(0xD53BE040u | 16u);   // mrs x16, cntvct_el0
    emitMovReg(0, 16, false);      // EAX = 32 bits bajos (mov w0, w16 pone a cero la mitad alta)
    emitLsrImm(2, 16, 32, true);   // EDX = 32 bits altos
}

void OryonEmitterBackend::emitWord(uint32_t instruction) {
    if (!emitOk_) return;
    if (arena_ == nullptr || jitOffset_ + sizeof(uint32_t) > arenaSize_) {
        emitOk_ = false;
        return;
    }
    uint8_t* dst = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::memcpy(dst + jitOffset_, &instruction, sizeof(uint32_t));
    jitOffset_ += sizeof(uint32_t);
}

void OryonEmitterBackend::patchWord(size_t wordIndex, uint32_t instruction) {
    if (arena_ == nullptr) return;
    const size_t offset = wordIndex * sizeof(uint32_t);
    if (offset + sizeof(uint32_t) > arenaSize_) return;
    uint8_t* dst = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::memcpy(dst + offset, &instruction, sizeof(uint32_t));
}

void OryonEmitterBackend::patchImm19(size_t wordIndex, size_t targetWordIndex) {
    if (arena_ == nullptr) return;
    uint32_t old = 0;
    const size_t offset = wordIndex * sizeof(uint32_t);
    if (offset + sizeof(uint32_t) > arenaSize_) return;
    uint8_t* src = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::memcpy(&old, src + offset, sizeof(uint32_t));
    const int32_t delta = static_cast<int32_t>(targetWordIndex) - static_cast<int32_t>(wordIndex);
    old &= 0xFF00001Fu;
    old |= (static_cast<uint32_t>(delta) & 0x7FFFFu) << 5;
    patchWord(wordIndex, old);
}

void OryonEmitterBackend::patchImm26(size_t wordIndex, size_t targetWordIndex) {
    if (arena_ == nullptr) return;
    uint32_t old = 0;
    const size_t offset = wordIndex * sizeof(uint32_t);
    if (offset + sizeof(uint32_t) > arenaSize_) return;
    uint8_t* src = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::memcpy(&old, src + offset, sizeof(uint32_t));
    const int32_t delta = static_cast<int32_t>(targetWordIndex) - static_cast<int32_t>(wordIndex);
    old &= 0xFC000000u;
    old |= static_cast<uint32_t>(delta) & 0x3FFFFFFu;
    patchWord(wordIndex, old);
}

// ---------------------------------------------------------------------------
// Emisores AArch64
// ---------------------------------------------------------------------------

void OryonEmitterBackend::emitMovImm64(uint8_t rd, uint64_t value) {
    rd &= 0x1F;
    const uint16_t w0 = static_cast<uint16_t>(value & 0xFFFFu);
    const uint16_t w1 = static_cast<uint16_t>((value >> 16) & 0xFFFFu);
    const uint16_t w2 = static_cast<uint16_t>((value >> 32) & 0xFFFFu);
    const uint16_t w3 = static_cast<uint16_t>((value >> 48) & 0xFFFFu);
    emitWord(0xD2800000u | (static_cast<uint32_t>(w0) << 5) | rd);
    if (w1 != 0) emitWord(0xF2A00000u | (static_cast<uint32_t>(w1) << 5) | rd);
    if (w2 != 0) emitWord(0xF2C00000u | (static_cast<uint32_t>(w2) << 5) | rd);
    if (w3 != 0) emitWord(0xF2E00000u | (static_cast<uint32_t>(w3) << 5) | rd);
}

// Dirección de una función de libhyperplay: MOVZ + 3 MOVK fijos (siempre 4 palabras) y un
// registro de reubicación para la caché persistente (ASLR cambia la base de la biblioteca).
void OryonEmitterBackend::emitMovHostAddr(uint8_t rd, uintptr_t addr, uint8_t kind) {
    rd &= 0x1F;
    JitReloc r;
    r.word = static_cast<uint32_t>((jitOffset_ - aotBlockStart_) / 4);
    r.reg = rd;
    r.kind = kind;
    r.delta = static_cast<int64_t>(addr - JitDiskCache::anchor());
    aotRelocs_.push_back(r);
    const uint64_t v = addr;
    emitWord(0xD2800000u | (static_cast<uint32_t>(v & 0xFFFFu) << 5) | rd);
    emitWord(0xF2A00000u | (static_cast<uint32_t>((v >> 16) & 0xFFFFu) << 5) | rd);
    emitWord(0xF2C00000u | (static_cast<uint32_t>((v >> 32) & 0xFFFFu) << 5) | rd);
    emitWord(0xF2E00000u | (static_cast<uint32_t>((v >> 48) & 0xFFFFu) << 5) | rd);
}

// Miembro de este backend relativo a X19 (= &ctx_): código independiente de la posición del
// objeto, válido para cualquier hilo y entre ejecuciones (el desplazamiento es fijo por build).
void OryonEmitterBackend::emitCtxRelAddr(uint8_t rd, const void* member) {
    const int64_t delta = static_cast<int64_t>(reinterpret_cast<uintptr_t>(member) -
                                               reinterpret_cast<uintptr_t>(&ctx_));
    if (delta >= 0 && delta <= 4095) {
        emitAddImm(rd, 19, static_cast<uint16_t>(delta), true);
        return;
    }
    emitMovImm64(rd, static_cast<uint64_t>(delta));
    emitAddReg(rd, 19, rd, false, true);
}

void OryonEmitterBackend::emitMovReg(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rm = src & 0x1F;
    emitWord((is64 ? 0xAA0003E0u : 0x2A0003E0u) | (rm << 16) | rd);
}

void OryonEmitterBackend::emitMvnReg(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rm = src & 0x1F;
    emitWord((is64 ? 0xAA2003E0u : 0x2A2003E0u) | (rm << 16) | rd);
}

void OryonEmitterBackend::emitAddReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags,
                                     bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? (setFlags ? 0xAB000000u : 0x8B000000u)
                               : (setFlags ? 0x2B000000u : 0x0B000000u);
    emitWord(base | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitAddRegShift(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t shift,
                                          bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? 0x8B000000u : 0x0B000000u;
    emitWord(base | ((static_cast<uint32_t>(shift) & 3u) << 10) | (rm << 16) | (rn << 5) | rd);
}

// Paso de MOVS/STOS en X17: +tamaño con DF=0, -tamaño con DF=1 (bit 10 de rflags). Usa X21.
void OryonEmitterBackend::emitStringStep(uint8_t elemSize) {
    emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
    emitUbfx(21, 21, 10, 1);
    emitMovImm64(17, elemSize);
    const uint32_t shift = elemSize == 1 ? 1u : elemSize == 2 ? 2u : elemSize == 4 ? 3u : 4u;  // 2*tamaño
    emitWord(0xCB000000u | (21u << 16) | (shift << 10) | (17u << 5) | 17u);  // SUB X17, X17, X21, LSL #shift
}

void OryonEmitterBackend::emitSubReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags,
                                     bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? (setFlags ? 0xEB000000u : 0xCB000000u)
                               : (setFlags ? 0x6B000000u : 0x4B000000u);
    emitWord(base | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitAndReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags,
                                     bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? (setFlags ? 0xEA000000u : 0x8A000000u)
                               : (setFlags ? 0x6A000000u : 0x0A000000u);
    emitWord(base | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitOrrReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    emitWord((is64 ? 0xAA000000u : 0x2A000000u) | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitEorReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    emitWord((is64 ? 0xCA000000u : 0x4A000000u) | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitMulReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    emitWord((is64 ? 0x9B007C00u : 0x1B007C00u) | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitSmull(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x9B207C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitSmulh(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x9B407C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUmulh(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x9BC07C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUdiv(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t base = is64 ? 0x9AC00800u : 0x1AC00800u;
    emitWord(base | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitSdiv(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t base = is64 ? 0x9AC00C00u : 0x1AC00C00u;
    emitWord(base | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitAdcsReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? 0xBA000000u : 0x3A000000u;
    emitWord(base | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitSbcsReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src1 & 0x1F;
    const uint32_t rm = src2 & 0x1F;
    const uint32_t base = is64 ? 0xFA000000u : 0x7A000000u;
    emitWord(base | (rm << 16) | (rn << 5) | rd);
}

void OryonEmitterBackend::emitRbitReg(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src & 0x1F;
    const uint32_t base = is64 ? 0xDAC00000u : 0x5AC00000u;
    emitWord(base | (rn << 5) | rd);
}

void OryonEmitterBackend::emitRev(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src & 0x1F;
    const uint32_t base = is64 ? 0xDAC00C00u : 0x5AC00800u;
    emitWord(base | (rn << 5) | rd);
}

void OryonEmitterBackend::emitRev16(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src & 0x1F;
    const uint32_t base = is64 ? 0xDAC00400u : 0x5AC00400u;
    emitWord(base | (rn << 5) | rd);
}

void OryonEmitterBackend::emitClzReg(uint8_t dst, uint8_t src, bool is64) {
    const uint32_t rd = dst & 0x1F;
    const uint32_t rn = src & 0x1F;
    const uint32_t base = is64 ? 0xDAC01000u : 0x5AC01000u;
    emitWord(base | (rn << 5) | rd);
}

void OryonEmitterBackend::emitPrimeCarry(bool forSbb) {
    emitLdrX(20, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
    emitUbfx(20, 20, 0, 1);
    if (forSbb) {
        emitSubReg(31, 31, 20, true, false);
    } else {
        emitSubImm(31, 20, 1, false, true);
    }
}

void OryonEmitterBackend::emitLslImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64) {
    const uint32_t bits = is64 ? 64u : 32u;
    const uint32_t amount = static_cast<uint32_t>(shift) & (bits - 1u);
    const uint32_t immr = (bits - amount) & (bits - 1u);
    const uint32_t imms = bits - 1u - amount;
    emitWord((is64 ? 0xD3400000u : 0x53000000u) | (immr << 16) | (imms << 10) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitLsrImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64) {
    const uint32_t bits = is64 ? 64u : 32u;
    const uint32_t amount = static_cast<uint32_t>(shift) & (bits - 1u);
    emitWord((is64 ? 0xD3400000u : 0x53000000u) | (amount << 16) | ((bits - 1u) << 10) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitAsrImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64) {
    const uint32_t bits = is64 ? 64u : 32u;
    const uint32_t amount = static_cast<uint32_t>(shift) & (bits - 1u);
    emitWord((is64 ? 0x93400000u : 0x13000000u) | (amount << 16) | ((bits - 1u) << 10) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitLslv(uint8_t dst, uint8_t src, uint8_t amount, bool is64) {
    emitWord((is64 ? 0x9AC02000u : 0x1AC02000u) | ((amount & 0x1F) << 16) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitLsrv(uint8_t dst, uint8_t src, uint8_t amount, bool is64) {
    emitWord((is64 ? 0x9AC02400u : 0x1AC02400u) | ((amount & 0x1F) << 16) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitAsrv(uint8_t dst, uint8_t src, uint8_t amount, bool is64) {
    emitWord((is64 ? 0x9AC02800u : 0x1AC02800u) | ((amount & 0x1F) << 16) |
             ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUbfx(uint8_t dst, uint8_t src, uint8_t lsb, uint8_t width) {
    const uint32_t immr = lsb & 0x3F;
    const uint32_t imms = (static_cast<uint32_t>(lsb) + width - 1u) & 0x3F;
    emitWord(0xD3400000u | (immr << 16) | (imms << 10) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUbfx32(uint8_t dst, uint8_t src, uint8_t lsb, uint8_t width) {
    const uint32_t immr = lsb & 0x1F;
    const uint32_t imms = (static_cast<uint32_t>(lsb) + width - 1u) & 0x1F;
    emitWord(0x53000000u | (immr << 16) | (imms << 10) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitBfiByte(uint8_t dst, uint8_t src) {
    // BFXIL Xdst, Xsrc, #0, #8: inserta el byte bajo de src preservando el resto de dst.
    emitWord(0xB3401C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

// Lee el registro r/m8 x86 (sin REX, 4..7 son AH/CH/DH/BH) extendido con ceros en dst.
void OryonEmitterBackend::loadByteRm(uint8_t dst, uint8_t rm) {
    if (!currentRexPresent_ && rm >= 4 && rm <= 7) {
        emitUbfx(dst, static_cast<uint8_t>(rm - 4), 8, 8);
    } else {
        emitUbfx(dst, rm, 0, 8);
    }
}

// Escribe el byte bajo de src en el registro r/m8 x86: sin REX, 4..7 son AH/CH/DH/BH.
void OryonEmitterBackend::storeByteRm(uint8_t rm, uint8_t src) {
    if (!currentRexPresent_ && rm >= 4 && rm <= 7) {
        emitBfi(static_cast<uint8_t>(rm - 4), src, 8, 8);
    } else {
        emitBfiByte(rm, src);
    }
}

void OryonEmitterBackend::emitBfi(uint8_t rd, uint8_t rn, uint8_t lsb, uint8_t width) {
    // BFI Xd, Xn, #lsb, #width -> BFM Xd, Xn, #(-lsb mod 64), #(width - 1)
    const uint32_t immr = (64u - static_cast<uint32_t>(lsb)) & 63u;
    const uint32_t imms = static_cast<uint32_t>(width - 1) & 63u;
    emitWord(0xB3400000u | (immr << 16) | (imms << 10) |
             (static_cast<uint32_t>(rn & 31) << 5) | static_cast<uint32_t>(rd & 31));
}

void OryonEmitterBackend::emitCset(uint8_t dst, uint8_t armCond) {
    const uint32_t inv = static_cast<uint32_t>(armCond ^ 1u) & 0xFu;
    emitWord(0x9A9F07E0u | (inv << 12) | (dst & 0x1F));
}

void OryonEmitterBackend::emitSxtw(uint8_t dst, uint8_t src) {
    emitWord(0x93407C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitSxtb(uint8_t dst, uint8_t src) {
    emitWord(0x93401C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitSxth(uint8_t dst, uint8_t src) {
    emitWord(0x93403C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUxtb(uint8_t dst, uint8_t src) {
    emitWord(0x53001C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitUxth(uint8_t dst, uint8_t src) {
    emitWord(0x53003C00u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitAddImm(uint8_t dst, uint8_t src, uint16_t imm, bool is64,
                                     bool setFlags) {
    const uint32_t base = is64 ? (setFlags ? 0xB1000000u : 0x91000000u)
                               : (setFlags ? 0x31000000u : 0x11000000u);
    emitWord(base | ((static_cast<uint32_t>(imm) & 0xFFFu) << 10) | ((src & 0x1F) << 5) |
             (dst & 0x1F));
}

void OryonEmitterBackend::emitSubImm(uint8_t dst, uint8_t src, uint16_t imm, bool is64,
                                     bool setFlags) {
    const uint32_t base = is64 ? (setFlags ? 0xF1000000u : 0xD1000000u)
                               : (setFlags ? 0x71000000u : 0x51000000u);
    emitWord(base | ((static_cast<uint32_t>(imm) & 0xFFFu) << 10) | ((src & 0x1F) << 5) |
             (dst & 0x1F));
}

void OryonEmitterBackend::emitCmpImm(uint8_t reg, uint16_t imm, bool is64) {
    emitWord((is64 ? 0xF1000000u : 0x71000000u) | ((static_cast<uint32_t>(imm) & 0xFFFu) << 10) |
             ((reg & 0x1F) << 5) | 31u);
}

void OryonEmitterBackend::emitTstReg(uint8_t reg, bool is64) {
    emitWord((is64 ? 0xEA000000u : 0x6A000000u) | ((reg & 0x1F) << 16) | ((reg & 0x1F) << 5) |
             31u);
}

// NZ según el ancho x86 del operando: para 8/16 bits se prueba (W << (32-bits)) en X23, de modo
// que N = bit de signo real y Z ignore los bits por encima del operando.
void OryonEmitterBackend::emitTstSized(uint8_t reg, uint8_t size) {
    if (size >= 4) {
        emitTstReg(reg, size == 8);
        return;
    }
    emitLslImm(23, reg, static_cast<uint8_t>(32 - size * 8), false);
    emitTstReg(23, false);
}

void OryonEmitterBackend::emitExtr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb, bool is64) {
    if (is64) {
        emitWord(0x93C00000u | (static_cast<uint32_t>(rm & 31) << 16) |
                 (static_cast<uint32_t>(lsb & 63) << 10) |
                 (static_cast<uint32_t>(rn & 31) << 5) | static_cast<uint32_t>(rd & 31));
    } else {
        emitWord(0x13800000u | (static_cast<uint32_t>(rm & 31) << 16) |
                 (static_cast<uint32_t>(lsb & 31) << 10) |
                 (static_cast<uint32_t>(rn & 31) << 5) | static_cast<uint32_t>(rd & 31));
    }
}

void OryonEmitterBackend::emitRorv(uint8_t rd, uint8_t rn, uint8_t rm, bool is64) {
    emitWord((is64 ? 0x9AC02C00u : 0x1AC02C00u) |
             (static_cast<uint32_t>(rm & 31) << 16) |
             (static_cast<uint32_t>(rn & 31) << 5) | static_cast<uint32_t>(rd & 31));
}

void OryonEmitterBackend::emitMrsNzcv(uint8_t rd) {
    emitWord(0xD53B4200u | (rd & 0x1F));
}

void OryonEmitterBackend::emitMsrNzcv(uint8_t rn) {
    emitWord(0xD51B4200u | (rn & 0x1F));
}

// --- Atomicidad x86 (LOCK / XCHG con memoria) ---------------------------------------------
// En una instrucción atómica, la primera carga del operando [X16] guarda el valor esperado en
// X24 y el almacenamiento final [X16] se sustituye por CASAL: si otro hilo cambió la memoria
// entre medias, se reejecuta la instrucción x86 completa (flags incluidas). Coherente con
// cualquier otro acceso atómico o normal de otros núcleos.
void OryonEmitterBackend::lockCaptureAfterLoad(uint8_t rt, uint8_t rn) {
    if (!lockActive_ || lockCaptured_ || rn != 16 || rt == 24) return;
    lockCaptured_ = true;
    emitMovReg(24, rt, true);
}

bool OryonEmitterBackend::lockStoreAsCas(uint8_t rt, uint8_t rn, uint32_t sizeLog2) {
    if (!lockActive_ || !lockCaptured_ || lockStored_ || rn != 16) return false;
    lockStored_ = true;
    uint8_t src = rt;
    if (src == 17 || src == 24) {  // X17/X24 se usan abajo
        emitMovReg(21, src, true);
        src = 21;
    }
    // CASAL exige alineación natural (x86 no: Mono parchea call sites con XCHG de 8 bytes sobre
    // direcciones desalineadas -> SIGBUS). Dirección desalineada: STR normal (no atómico, pero
    // ARM admite el acceso; son escrituras de un único hilo bajo el lock del propio JIT).
    size_t unalignedBranch = 0;
    if (sizeLog2 > 0) {
        emitWord(0xF240001Fu | ((sizeLog2 - 1u) << 10) | (16u << 5));  // TST X16, #(size-1)
        unalignedBranch = wordPos();
        emitWord(0x54000001u);  // B.NE desalineado (se parchea)
    }
    emitMovReg(17, 24, true);  // copia del esperado
    // CASAL{B,H,,} Xs=X24 (esperado -> valor previo), Xt=src, [X16]
    static constexpr uint32_t kCasal[4] = {0x08E0FC00u, 0x48E0FC00u, 0x88E0FC00u, 0xC8E0FC00u};
    emitWord(kCasal[sizeLog2] | (24u << 16) | (16u << 5) | (src & 0x1Fu));
    // CMP previo, esperado; B.NE inicio de la instrucción (reintento)
    emitWord((sizeLog2 == 3 ? 0xEB000000u : 0x6B000000u) | (17u << 16) | (24u << 5) | 31u);  // SUBS XZR, X24, X17
    const int64_t delta = static_cast<int64_t>(lockInstrStartWord_) - static_cast<int64_t>(wordPos());
    emitWord(0x54000001u | ((static_cast<uint32_t>(delta) & 0x7FFFFu) << 5));  // B.NE
    if (sizeLog2 > 0) {
        const size_t skip = wordPos();
        emitB(0);
        patchImm19(unalignedBranch, wordPos());
        static constexpr uint32_t kStr[4] = {0x39000000u, 0x79000000u, 0xB9000000u, 0xF9000000u};
        emitWord(kStr[sizeLog2] | (16u << 5) | (src & 0x1Fu));
        patchImm26(skip, wordPos());
    }
    return true;
}

// TSO: los accesos escalares a memoria del huésped ([X16] sin desplazamiento) usan LDAPR
// (adquisición RCpc: carga->carga y carga->escritura ordenadas) y STLR (liberación:
// todo lo anterior antes de la escritura). Es exactamente el modelo x86 (solo escritura->carga
// puede adelantarse). Alineado natural -> LDAPR/STLR; desalineado (raro en sincronización)
// -> acceso normal con DMB para conservar el orden.
bool OryonEmitterBackend::emitTsoAccess(uint8_t rt, uint8_t rn, int32_t offset, uint32_t sizeLog2,
                                        bool store) {
    if (offset != 0 || rn != 16 || eaIsStack_ || !tsoMemory.load(std::memory_order_relaxed)) return false;
    static constexpr uint32_t kLdapr[4] = {0x38BFC000u, 0x78BFC000u, 0xB8BFC000u, 0xF8BFC000u};
    static constexpr uint32_t kStlr[4] = {0x089FFC00u, 0x489FFC00u, 0x889FFC00u, 0xC89FFC00u};
    static constexpr uint32_t kLdr[4] = {0x39400000u, 0x79400000u, 0xB9400000u, 0xF9400000u};
    static constexpr uint32_t kStr[4] = {0x39000000u, 0x79000000u, 0xB9000000u, 0xF9000000u};
    const uint32_t regs = (16u << 5) | (rt & 0x1Fu);
    const uint32_t ordered = (store ? kStlr : kLdapr)[sizeLog2] | regs;
    if (sizeLog2 == 0) {
        emitWord(ordered);
        return true;
    }
    emitWord(0xF240001Fu | ((sizeLog2 - 1u) << 10) | (16u << 5));  // TST X16, #(size-1)
    emitWord(0x54000001u | (3u << 5));                              // B.NE desalineado
    emitWord(ordered);
    emitWord(0x14000003u);                                          // B fin
    if (store) {
        emitWord(0xD5033BBFu);  // DMB ISH
        emitWord(kStr[sizeLog2] | regs);
    } else {
        emitWord(kLdr[sizeLog2] | regs);
        emitWord(0xD50339BFu);  // DMB ISHLD
    }
    return true;
}

void OryonEmitterBackend::emitLdrX(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (!emitTsoAccess(rt, rn, offset, 3, false)) {
        const uint32_t pimm = (static_cast<uint32_t>(offset) / 8u) & 0xFFFu;
        emitWord(0xF9400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
    }
    if (offset == 0) lockCaptureAfterLoad(rt, rn);
}

void OryonEmitterBackend::emitStrX(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (offset == 0 && lockStoreAsCas(rt, rn, 3)) return;
    if (emitTsoAccess(rt, rn, offset, 3, true)) return;
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 8u) & 0xFFFu;
    emitWord(0xF9000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrW(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (!emitTsoAccess(rt, rn, offset, 2, false)) {
        const uint32_t pimm = (static_cast<uint32_t>(offset) / 4u) & 0xFFFu;
        emitWord(0xB9400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
    }
    if (offset == 0) lockCaptureAfterLoad(rt, rn);
}

void OryonEmitterBackend::emitStrW(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (offset == 0 && lockStoreAsCas(rt, rn, 2)) return;
    if (emitTsoAccess(rt, rn, offset, 2, true)) return;
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 4u) & 0xFFFu;
    emitWord(0xB9000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrB(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (!emitTsoAccess(rt, rn, offset, 0, false)) {
        const uint32_t pimm = static_cast<uint32_t>(offset) & 0xFFFu;
        emitWord(0x39400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
    }
    if (offset == 0) lockCaptureAfterLoad(rt, rn);
}

void OryonEmitterBackend::emitStrB(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (offset == 0 && lockStoreAsCas(rt, rn, 0)) return;
    if (emitTsoAccess(rt, rn, offset, 0, true)) return;
    const uint32_t pimm = static_cast<uint32_t>(offset) & 0xFFFu;
    emitWord(0x39000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrH(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (!emitTsoAccess(rt, rn, offset, 1, false)) {
        const uint32_t pimm = (static_cast<uint32_t>(offset) / 2u) & 0xFFFu;
        emitWord(0x79400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
    }
    if (offset == 0) lockCaptureAfterLoad(rt, rn);
}

void OryonEmitterBackend::emitStrH(uint8_t rt, uint8_t rn, int32_t offset) {
    noteFlagAccess(rn, offset);
    if (offset == 0 && lockStoreAsCas(rt, rn, 1)) return;
    if (emitTsoAccess(rt, rn, offset, 1, true)) return;
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 2u) & 0xFFFu;
    emitWord(0x79000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrS(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 4u) & 0xFFFu;
    emitWord(0xBD400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitStrS(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 4u) & 0xFFFu;
    emitWord(0xBD000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrD(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 8u) & 0xFFFu;
    emitWord(0xFD400000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitStrD(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 8u) & 0xFFFu;
    emitWord(0xFD000000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdrQ(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 16u) & 0xFFFu;
    emitWord(0x3DC00000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitStrQ(uint8_t rt, uint8_t rn, int32_t offset) {
    const uint32_t pimm = (static_cast<uint32_t>(offset) / 16u) & 0xFFFu;
    emitWord(0x3D800000u | (pimm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdpX(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t offset) {
    const uint32_t imm7 = (static_cast<uint32_t>(offset) / 8u) & 0x7Fu;
    emitWord(0xA9400000u | (imm7 << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) |
             (rt & 0x1F));
}

void OryonEmitterBackend::emitStpX(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t offset) {
    const uint32_t imm7 = (static_cast<uint32_t>(offset) / 8u) & 0x7Fu;
    emitWord(0xA9000000u | (imm7 << 15) | ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) |
             (rt & 0x1F));
}

void OryonEmitterBackend::emitStpPre(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t imm7) {
    emitWord(0xA9800000u | ((static_cast<uint32_t>(imm7) & 0x7Fu) << 15) |
             ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitLdpPost(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t imm7) {
    emitWord(0xA8C00000u | ((static_cast<uint32_t>(imm7) & 0x7Fu) << 15) |
             ((rt2 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitFaddV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4E20D400u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsubV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4EA0D400u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFmulV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E20DC00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitEorV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E201C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFdivV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E20FC00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFaddV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4E60D400u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsubV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4EE0D400u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFmulV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E60DC00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFdivV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E60FC00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFaddS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E202800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsubS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E203800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFmulS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E200800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFdivS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E201800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFaddD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E602800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsubD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E603800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFmulD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E600800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFdivD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x1E601800u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFmaS(uint8_t dst, uint8_t n, uint8_t m, uint8_t a,
                                    bool sub, bool neg) {
    const bool a64Sub = sub != neg;
    const bool a64Neg = sub;
    emitWord(0x1F000000u | (a64Neg ? 0x200000u : 0u) | (a64Sub ? 0x8000u : 0u) |
             ((m & 0x1Fu) << 16) | ((a & 0x1Fu) << 10) |
             ((n & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitFmaD(uint8_t dst, uint8_t n, uint8_t m, uint8_t a,
                                    bool sub, bool neg) {
    const bool a64Sub = sub != neg;
    const bool a64Neg = sub;
    const uint32_t word = 0x1F400000u | (a64Neg ? 0x200000u : 0u) |
                          (a64Sub ? 0x8000u : 0u) |
                          ((m & 0x1Fu) << 16) | ((a & 0x1Fu) << 10) |
                          ((n & 0x1Fu) << 5) | (dst & 0x1Fu);
    emitWord(word);
}

void OryonEmitterBackend::emitInsElementD(uint8_t dst, uint8_t dstIdx,
                                           uint8_t src, uint8_t srcIdx) {
    emitWord(0x6E080400u | ((dstIdx & 1u) << 20) | ((srcIdx & 1u) << 14) |
             ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitFcmpS(uint8_t src1, uint8_t src2) {
    emitWord(0x1E202000u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5));
}

void OryonEmitterBackend::emitFcmpD(uint8_t src1, uint8_t src2) {
    emitWord(0x1E602000u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5));
}

void OryonEmitterBackend::emitFsqrtS(uint8_t dst, uint8_t src) {
    emitWord(0x1E21C000u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsqrtD(uint8_t dst, uint8_t src) {
    emitWord(0x1E61C000u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsqrtV(uint8_t dst, uint8_t src) {
    emitWord(0x6EA1F800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFsqrtV2D(uint8_t dst, uint8_t src) {
    emitWord(0x6EE1F800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

// RSQRTSS/RCPSS (y PS): x86 garantiza error relativo <= 1.5*2^-12; FRSQRTE/FRECPE de ARM solo dan
// ~8 bits, lo que desviaba normalizaciones y física. Se añade un paso de Newton-Raphson
// (FRSQRTS/FRECPS) -> ~16 bits, igual o mejor que x86. Usa kXmmScratch5/6; dst puede ser src.
void OryonEmitterBackend::emitFrsqrteS(uint8_t dst, uint8_t src) {
    const uint32_t t = kXmmScratch5, u = kXmmScratch6, s = src & 0x1Fu, d = dst & 0x1Fu;
    emitWord(0x7EA1D800u | (s << 5) | t);                 // FRSQRTE t, s
    emitWord(0x1E200800u | (t << 16) | (t << 5) | u);     // FMUL u, t, t
    emitWord(0x5EA0FC00u | (s << 16) | (u << 5) | u);     // FRSQRTS u, u, s
    emitWord(0x1E200800u | (u << 16) | (t << 5) | d);     // FMUL d, t, u
}

void OryonEmitterBackend::emitFrsqrteD(uint8_t dst, uint8_t src) {
    emitWord(0x7EE1D800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFrsqrteV(uint8_t dst, uint8_t src) {
    const uint32_t t = kXmmScratch5, u = kXmmScratch6, s = src & 0x1Fu, d = dst & 0x1Fu;
    emitWord(0x6EA1D800u | (s << 5) | t);                 // FRSQRTE t.4S, s.4S
    emitWord(0x6E20DC00u | (t << 16) | (t << 5) | u);     // FMUL u.4S, t, t
    emitWord(0x4EA0FC00u | (s << 16) | (u << 5) | u);     // FRSQRTS u.4S, u, s
    emitWord(0x6E20DC00u | (u << 16) | (t << 5) | d);     // FMUL d.4S, t, u
}

void OryonEmitterBackend::emitFrsqrteV2D(uint8_t dst, uint8_t src) {
    emitWord(0x6EE1D800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFrecpeS(uint8_t dst, uint8_t src) {
    const uint32_t t = kXmmScratch5, u = kXmmScratch6, s = src & 0x1Fu, d = dst & 0x1Fu;
    emitWord(0x5EA1D800u | (s << 5) | t);                 // FRECPE t, s
    emitWord(0x5E20FC00u | (s << 16) | (t << 5) | u);     // FRECPS u, t, s
    emitWord(0x1E200800u | (u << 16) | (t << 5) | d);     // FMUL d, t, u
}

void OryonEmitterBackend::emitFrecpeD(uint8_t dst, uint8_t src) {
    emitWord(0x5EE1D800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFrecpeV(uint8_t dst, uint8_t src) {
    const uint32_t t = kXmmScratch5, u = kXmmScratch6, s = src & 0x1Fu, d = dst & 0x1Fu;
    emitWord(0x4EA1D800u | (s << 5) | t);                 // FRECPE t.4S, s.4S
    emitWord(0x4E20FC00u | (s << 16) | (t << 5) | u);     // FRECPS u.4S, t, s
    emitWord(0x6E20DC00u | (u << 16) | (t << 5) | d);     // FMUL d.4S, t, u
}

void OryonEmitterBackend::emitFrecpeV2D(uint8_t dst, uint8_t src) {
    emitWord(0x4EE1D800u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

// MIN/MAX de x86 (MINSS/MINPS/MAXSD...): min = (a < b) ? a : b, max = (a > b) ? a : b. Si hay
// NaN o ±0 se devuelve SIEMPRE el segundo operando; FMIN/FMAX de ARM propagan el NaN, así que se
// emite comparación + selección. (Antes las codificaciones vectoriales eran FMAX/FMAXNM/FMINNM:
// MINPS calculaba el máximo y MAXPD el mínimo.)
// cmpBase: FCMGT escalar S 0x7EA0E400, D 0x7EE0E400; vectorial 4S 0x6EA0E400, 2D 0x6EE0E400.
void OryonEmitterBackend::emitX86MinMax(uint8_t dst, uint8_t a, uint8_t b, bool isMax, uint32_t cmpBase) {
    const uint8_t mask = kXmmScratch3;
    // mask = (isMax ? a > b : b > a)
    const uint8_t n = isMax ? a : b, m = isMax ? b : a;
    emitWord(cmpBase | ((m & 0x1Fu) << 16) | ((n & 0x1Fu) << 5) | mask);
    // BSL mask, a, b: mask = (mask & a) | (~mask & b)
    emitWord(0x6E601C00u | ((b & 0x1Fu) << 16) | ((a & 0x1Fu) << 5) | mask);
    emitOrrV(dst, mask, mask);
}

void OryonEmitterBackend::emitFminS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, false, 0x7EA0E400u);
}

void OryonEmitterBackend::emitFmaxS(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, true, 0x7EA0E400u);
}

void OryonEmitterBackend::emitFminD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, false, 0x7EE0E400u);
}

void OryonEmitterBackend::emitFmaxD(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, true, 0x7EE0E400u);
}

void OryonEmitterBackend::emitFminV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, false, 0x6EA0E400u);
}

void OryonEmitterBackend::emitFmaxV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, true, 0x6EA0E400u);
}

void OryonEmitterBackend::emitFminV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, false, 0x6EE0E400u);
}

void OryonEmitterBackend::emitFmaxV2D(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitX86MinMax(dst, src1, src2, true, 0x6EE0E400u);
}

void OryonEmitterBackend::emitScvtfS(uint8_t dst, uint8_t src, bool is64) {
    emitWord((is64 ? 0x9E220000u : 0x1E220000u) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitScvtfD(uint8_t dst, uint8_t src, bool is64) {
    emitWord((is64 ? 0x9E620000u : 0x1E620000u) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

// Conversión float->entero con semántica x86 (CVT[T]SS2SI/CVT[T]SD2SI): NaN o fuera de rango dan
// el entero "indefinido" (0x80000000 / 0x8000000000000000); ARM satura y da 0 para NaN.
// convBase: FCVTZS 0x1E380000 (truncado) o FCVTNS 0x1E200000 (al par). Usa X16/X17 y NZCV.
static int x86FloatToIntWords(uint32_t* w, uint32_t convBase, uint8_t dst, uint8_t src, bool isDouble, bool is64) {
    const uint32_t d = dst & 0x1Fu, n = src & 0x1Fu;
    const uint32_t ftype = isDouble ? 0x00400000u : 0u;
    int k = 0;
    if (is64) {
        w[k++] = convBase | 0x80000000u | ftype | (n << 5) | d;  // FCVT Xd
        w[k++] = 0xD2F00011u;                                      // MOVZ X17, #0x8000, LSL #48
        w[k++] = 0xD1000630u;                                      // SUB X16, X17, #1 (INT64_MAX)
        w[k++] = 0xEB10001Fu | (d << 5);                           // CMP Xd, X16 (saturó arriba)
        w[k++] = 0x9A800000u | (d << 16) | (17u << 5) | d;         // CSEL Xd, X17, Xd, EQ
        w[k++] = 0x1E202000u | ftype | (n << 16) | (n << 5);       // FCMP n, n
        w[k++] = 0x9A806000u | (d << 16) | (17u << 5) | d;         // CSEL Xd, X17, Xd, VS (NaN)
    } else {
        w[k++] = convBase | 0x80000000u | ftype | (n << 5) | 16u;  // FCVT X16 (64 bits, exacto)
        w[k++] = 0x52B00011u;                                      // MOVZ W17, #0x8000, LSL #16
        w[k++] = 0xEB30C21Fu;                                      // CMP X16, W16, SXTW (¿cabe en 32?)
        w[k++] = 0x1A800000u | (17u << 16) | (16u << 5) | d;       // CSEL Wd, W16, W17, EQ
        w[k++] = 0x1E202000u | ftype | (n << 16) | (n << 5);       // FCMP n, n
        w[k++] = 0x1A806000u | (d << 16) | (17u << 5) | d;         // CSEL Wd, W17, Wd, VS (NaN)
    }
    return k;
}

void OryonEmitterBackend::emitFcvtzsS(uint8_t dst, uint8_t src, bool is64) {
    emitWord((is64 ? 0x9E380000u : 0x1E380000u) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcvtzsD(uint8_t dst, uint8_t src, bool is64) {
    emitWord((is64 ? 0x9E780000u : 0x1E780000u) | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcvtDS(uint8_t dst, uint8_t src) {
    emitWord(0x1E22C000u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcvtSD(uint8_t dst, uint8_t src) {
    emitWord(0x1E624000u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitAndV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4E201C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitOrrV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4EA01C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitBicV(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x4E601C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitCmeqV16B(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E208C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitCmeqV8H(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6E608C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitCmeqV4S(uint8_t dst, uint8_t src1, uint8_t src2) {
    emitWord(0x6EA08C00u | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcmEqV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble) {
    // FCMEQ Vd.4S/Vd.2D: 4S = 0x4E20E400, 2D = 0x4E60E400.
    const uint32_t base = isDouble ? 0x4E60E400u : 0x4E20E400u;
    emitWord(base | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcmGeV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble) {
    // FCMGE Vd.4S/Vd.2D: 4S = 0x6E20E400, 2D = 0x6E60E400.
    const uint32_t base = isDouble ? 0x6E60E400u : 0x6E20E400u;
    emitWord(base | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitFcmGtV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble) {
    // FCMGT Vd.4S/Vd.2D: 4S = 0x6EA0E400, 2D = 0x6EE0E400.
    const uint32_t base = isDouble ? 0x6EE0E400u : 0x6EA0E400u;
    emitWord(base | ((src2 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitCmpPsPd(uint8_t dst, uint8_t src1, uint8_t src2,
                                      uint8_t predicate, bool isDouble) {
    const auto invert = [&](uint8_t value) {
        // NOT Vd.16B, Vn.16B (MVN). Antes: máscara de unos en kXmmScratch2 + EOR, pero CMPSS/CMPSD
        // pasan dst = kXmmScratch2 y el resultado quedaba a 0 (NEQ/NLT/NLE escalares siempre
        // "falso": encontrado con la prueba diferencial contra x86 real, tools/jitdiff_gen.py).
        emitWord(0x6E205800u | ((value & 0x1Fu) << 5) | (dst & 0x1Fu));
    };
    switch (predicate) {
        case 0: // EQ
            emitFcmEqV(dst, src1, src2, isDouble);
            break;
        case 1: // LT: FCMGT(src2, src1) => src1 < src2
            emitFcmGtV(dst, src2, src1, isDouble);
            break;
        case 2: // LE: FCMGE(src2, src1) => src1 <= src2
            emitFcmGeV(dst, src2, src1, isDouble);
            break;
        case 3: { // UNORD = !(ORD(x) && ORD(y))
            // Calcular ORD(src2) primero: src2 puede ser kXmmScratch (fuente memory).
            emitFcmEqV(kXmmScratch2, src2, src2, isDouble);
            emitFcmEqV(kXmmScratch, src1, src1, isDouble);
            emitAndV(kXmmScratch, kXmmScratch, kXmmScratch2);
            invert(kXmmScratch);
            break;
        }
        case 4: // NEQ
            emitFcmEqV(dst, src1, src2, isDouble);
            invert(dst);
            break;
        case 5: // NLT
            emitFcmGtV(dst, src2, src1, isDouble);
            invert(dst);
            break;
        case 6: // NLE
            emitFcmGeV(dst, src2, src1, isDouble);
            invert(dst);
            break;
        case 7: // ORD
            emitFcmEqV(kXmmScratch2, src2, src2, isDouble);
            emitFcmEqV(kXmmScratch, src1, src1, isDouble);
            emitAndV(dst, kXmmScratch, kXmmScratch2);
            break;
        default:
            emitEorV(dst, dst, dst);
            break;
    }
}

void OryonEmitterBackend::emitInsS(uint8_t dst, uint8_t src) {
    emitWord(0x6E040400u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitInsD(uint8_t dst, uint8_t src) {
    emitWord(0x6E080400u | ((src & 0x1F) << 5) | (dst & 0x1F));
}

void OryonEmitterBackend::emitExtV16B(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm) {
    emitWord(0x6E000000u | ((src2 & 0x1Fu) << 16) | ((imm & 0x0Fu) << 11) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitUshrV2D(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x6F000400u | (((128u - shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitShlV2D(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x4F005400u | (((64u + shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitUshrV4S(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x6F000400u | (((64u - shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitSshrV4S(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x4F000400u | (((64u - shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitShlV4S(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x4F005400u | (((32u + shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitUshrV8H(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x6F000400u | (((32u - shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitSshrV8H(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x4F000400u | (((32u - shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitShlV8H(uint8_t dst, uint8_t src, uint8_t shift) {
    emitWord(0x4F005400u | (((16u + shift) & 0x7Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

// INS Vd.H[dstIdx], Vn.H[srcIdx]
void OryonEmitterBackend::emitInsElementH(uint8_t dst, uint8_t dstIdx, uint8_t src, uint8_t srcIdx) {
    emitWord(0x6E000400u | ((((dstIdx & 7u) << 2) | 2u) << 16) | ((srcIdx & 7u) << 12) | ((src & 0x1Fu) << 5) |
             (dst & 0x1Fu));
}

void OryonEmitterBackend::emitInsElementS(uint8_t dst, uint8_t dstIdx, uint8_t src, uint8_t srcIdx) {
    emitWord(0x6E000400u | ((((dstIdx & 3u) << 3) | 4u) << 16) | ((srcIdx & 3u) << 13) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitAddV(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t elemSize) {
    const uint32_t sz = (elemSize == 8 ? 3u : (elemSize == 4 ? 2u : (elemSize == 2 ? 1u : 0u)));
    emitWord(0x4E208400u | (sz << 22) | ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitSubV(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t elemSize) {
    const uint32_t sz = (elemSize == 8 ? 3u : (elemSize == 4 ? 2u : (elemSize == 2 ? 1u : 0u)));
    emitWord(0x6E208400u | (sz << 22) | ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
}

void OryonEmitterBackend::emitB(int32_t imm26) {
    emitWord(0x14000000u | (static_cast<uint32_t>(imm26) & 0x3FFFFFFu));
}

void OryonEmitterBackend::emitBl(int32_t imm26) {
    emitWord(0x94000000u | (static_cast<uint32_t>(imm26) & 0x3FFFFFFu));
}

void OryonEmitterBackend::emitCbnz(uint8_t rt, int32_t imm19) {
    emitWord(0xB5000000u | ((static_cast<uint32_t>(imm19) & 0x7FFFFu) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitCbz(uint8_t rt, int32_t imm19) {
    emitWord(0xB4000000u | ((static_cast<uint32_t>(imm19) & 0x7FFFFu) << 5) | (rt & 0x1F));
}

void OryonEmitterBackend::emitBr(uint8_t rt) {
    emitWord(0xD61F0000u | ((rt & 0x1F) << 5));
}

void OryonEmitterBackend::emitBlr(uint8_t rt) {
    flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // llamada nativa: barrera
    emitWord(0xD63F0000u | ((rt & 0x1F) << 5));
}

void OryonEmitterBackend::emitRet() {
    emitWord(0xD65F03C0u);
}

void OryonEmitterBackend::emitNop() {
    emitWord(0xD503201Fu);
}

// ---------------------------------------------------------------------------
// Prólogo, epílogo, RIP y flags
// ---------------------------------------------------------------------------

void OryonEmitterBackend::emitPrologue() {
    // X29/X30 primero: el código JIT usa BL (pila de retornos) y pisa X30, que aquí guarda el
    // retorno al despachador C++.
    emitStpPre(29, 30, 31, -2);
    emitStpPre(19, 20, 31, -2);
    emitStpPre(21, 22, 31, -2);
    emitStpPre(23, 24, 31, -2);
    emitStpPre(25, 26, 31, -2);
    emitStpPre(27, 28, 31, -2);
    emitMovReg(19, 0, true);  // X19 = CpuContext*
    // Registros del encadenado, vivos mientras se salta de bloque en bloque sin despachador:
    // X25 = &s_smcWriteIdx, X26 = cursor SMC del hilo, X27 = presupuesto de saltos.
    emitLdrX(25, 19, static_cast<int32_t>(offsetof(CpuContext, smcIdxAddr)));
    emitLdrX(26, 19, static_cast<int32_t>(offsetof(CpuContext, smcCursor)));
    emitLdrX(27, 19, static_cast<int32_t>(offsetof(CpuContext, chainBudget)));
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19,
                 static_cast<int32_t>(i * 16));
    }
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19,
                 simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
}

namespace {
// 4 KiB: nunca mayor que la página de vigilancia SMC (que es la del sistema, >= 4 KiB).
constexpr uintptr_t smcPageSizeForFlags() { return 4096; }

constexpr size_t kMaxStaticImages = 128;
std::atomic<uintptr_t> s_imageBase[kMaxStaticImages];
std::atomic<uintptr_t> s_imageEnd[kMaxStaticImages];
std::atomic<size_t> s_imageCount{0};
std::mutex s_imageMutex;
}  // namespace

void OryonEmitterBackend::addStaticImage(uintptr_t base, size_t size) noexcept {
    if (base == 0 || size == 0) return;
    std::lock_guard<std::mutex> lock(s_imageMutex);
    const size_t n = s_imageCount.load(std::memory_order_relaxed);
    for (size_t i = 0; i < n; ++i) {
        if (s_imageBase[i].load(std::memory_order_relaxed) == base) {
            s_imageEnd[i].store(base + size, std::memory_order_release);
            return;
        }
    }
    if (n >= kMaxStaticImages) return;
    s_imageBase[n].store(base, std::memory_order_relaxed);
    s_imageEnd[n].store(base + size, std::memory_order_relaxed);
    s_imageCount.store(n + 1, std::memory_order_release);
}

bool OryonEmitterBackend::inStaticImage(uintptr_t addr, size_t len) noexcept {
    const size_t n = s_imageCount.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) {
        const uintptr_t b = s_imageBase[i].load(std::memory_order_relaxed);
        const uintptr_t e = s_imageEnd[i].load(std::memory_order_acquire);
        if (addr >= b && addr + len <= e) return true;
    }
    return false;
}

// Análisis lineal y conservador de los flags en el destino de una salida. Solo reconoce:
//  - escrituras completas sin lectura: ADD/OR/AND/SUB/XOR/CMP (formas 00-3D y grupo 80/81/83
//    salvo ADC/SBB), TEST (84/85/A8/A9/F6-F7 /0) y NEG (F6/F7 /3) -> muertos;
//  - instrucciones que ni leen ni escriben flags (MOV, LEA, MOVZX/MOVSX/MOVSXD, PUSH/POP reg,
//    NOP, NOT, movimientos SSE básicos) -> se sigue mirando;
//  - cualquier otra cosa (saltos, ADC/SBB, INC/DEC, desplazamientos, CMOVcc, SETcc, llamadas,
//    prefijo 67...) -> vivos.
bool OryonEmitterBackend::flagsDeadAt(const uint8_t* p, size_t avail, bool is64) noexcept {
    size_t pos = 0;
    for (int insn = 0; insn < 12; ++insn) {
        bool opsize16 = false, rexW = false;
        uint8_t op = 0;
        // Prefijos (REX solo justo antes del opcode).
        for (int k = 0;; ++k) {
            if (pos >= avail || k > 6) return false;
            op = p[pos];
            if (op == 0x66) { opsize16 = true; ++pos; continue; }
            if (op == 0x67) return false;  // direcciones de otro tamaño: no se analiza
            if (op == 0xF2 || op == 0xF3 || op == 0xF0 || op == 0x2E || op == 0x3E || op == 0x26 ||
                op == 0x36 || op == 0x64 || op == 0x65) { ++pos; continue; }
            if (is64 && op >= 0x40 && op <= 0x4F) {
                rexW = (op & 0x08) != 0;
                ++pos;
                if (pos >= avail) return false;
                op = p[pos];
            }
            break;
        }
        ++pos;  // opcode
        // Longitud de ModRM (+SIB +desplazamiento). Devuelve false si se sale del rango.
        uint8_t modrmReg = 0;
        const auto skipModRm = [&]() -> bool {
            if (pos >= avail) return false;
            const uint8_t m = p[pos++];
            const uint8_t mod = m >> 6, rm = m & 7;
            modrmReg = (m >> 3) & 7;
            if (mod == 3) return true;
            if (rm == 4) {
                if (pos >= avail) return false;
                const uint8_t sib = p[pos++];
                if (mod == 0 && (sib & 7) == 5) pos += 4;
            } else if (mod == 0 && rm == 5) {
                pos += 4;
            }
            if (mod == 1) pos += 1;
            if (mod == 2) pos += 4;
            return pos <= avail;
        };
        const size_t immZ = opsize16 ? 2 : 4;
        const auto skip = [&](size_t n) { pos += n; return pos <= avail; };

        if (op < 0x40 && (op & 7) <= 5) {
            const uint8_t group = op >> 3;  // 0 ADD 1 OR 2 ADC 3 SBB 4 AND 5 SUB 6 XOR 7 CMP
            if (group == 2 || group == 3) return false;
            return true;  // escritura completa sin lectura
        }
        if (op == 0x80 || op == 0x81 || op == 0x83) {
            if (!skipModRm()) return false;
            return modrmReg != 2 && modrmReg != 3;
        }
        if (op == 0x84 || op == 0x85 || op == 0xA8 || op == 0xA9) return true;
        if (op == 0xF6 || op == 0xF7) {
            if (!skipModRm()) return false;
            if (modrmReg == 0 || modrmReg == 3) return true;  // TEST imm / NEG
            if (modrmReg != 2) return false;                  // MUL/IMUL/DIV/IDIV
            continue;                                         // NOT: no toca flags
        }
        // Neutras.
        if ((op >= 0x88 && op <= 0x8B) || op == 0x8D || (is64 && op == 0x63)) {
            if (!skipModRm()) return false;
            continue;
        }
        if (op == 0xC6 || op == 0xC7) {
            if (!skipModRm() || modrmReg != 0) return false;
            if (!skip(op == 0xC6 ? 1 : immZ)) return false;
            continue;
        }
        if (op >= 0xB0 && op <= 0xB7) {
            if (!skip(1)) return false;
            continue;
        }
        if (op >= 0xB8 && op <= 0xBF) {
            if (!skip(rexW ? 8 : immZ)) return false;
            continue;
        }
        if ((op >= 0x50 && op <= 0x5F) || op == 0x90) continue;
        // INC/DEC: escriben todo salvo CF y no leen nada (el emisor mezcla el CF viejo de RFLAGS,
        // pero si una escritura completa posterior lo pisa antes de leerlo, da igual lo que fuera).
        if (op == 0xFE || op == 0xFF) {
            if (!skipModRm()) return false;
            if (modrmReg <= 1) continue;
            return false;  // CALL/JMP/PUSH indirectos
        }
        if (!is64 && op >= 0x40 && op <= 0x4F) continue;  // INC/DEC r32 (solo 32 bits)
        if (op == 0x0F) {
            if (pos >= avail) return false;
            const uint8_t op2 = p[pos++];
            if (op2 == 0xB6 || op2 == 0xB7 || op2 == 0xBE || op2 == 0xBF || op2 == 0x1F || op2 == 0x10 ||
                op2 == 0x11 || op2 == 0x28 || op2 == 0x29 || op2 == 0x6F || op2 == 0x7F || op2 == 0x57 ||
                op2 == 0xEF || op2 == 0x16 || op2 == 0x12 || op2 == 0x13 || op2 == 0x17) {
                if (!skipModRm()) return false;
                continue;
            }
            return false;
        }
        return false;
    }
    return false;
}

// Salida enlazable. Camino enlazado (6 instrucciones; el estado vive en X25-X27):
//   SUBS X27, X27, #1 ; B.MI stub           (presupuesto: suspensiones/señales a tiempo)
//   LDR X16, [X25] ; CMP X16, X26 ; B.NE stub (código modificado: no seguir enlaces)
//   site: B stub      <- el despachador lo reescribe como B <cuerpo del destino> (o al veneer)
//   veneer: LDR X17, =destino ; BR X17       (destinos a más de ±128 MB)
//   stub: ctx.rip = target ; ctx.exitSite = &site ; B epílogo+1
// La RIP y el sitio solo se escriben al salir al despachador: el salto enlazado no los necesita.
void OryonEmitterBackend::emitLinkableExit(uintptr_t target) {
    // Salida: los flags quedan vivos, salvo que el destino los sobrescriba todos antes de leerlos
    // (solo código estático de una imagen y en la misma página que el inicio del bloque: una
    // escritura en esa página invalida también este bloque, así que el análisis no queda viejo).
    bool deadAtTarget = false;
    if (crossBlockFlags.load(std::memory_order_relaxed) && deadFlags.load(std::memory_order_relaxed) &&
        curBlockRip_ != 0) {
        const uintptr_t ps = smcPageSizeForFlags();
        const uintptr_t page = curBlockRip_ & ~(ps - 1);
        if (target >= page && target < page + ps && inStaticImage(target, 1) &&
            !NativeDllInterceptor::isRegisteredTrampoline(target)) {
            deadAtTarget = flagsDeadAt(reinterpret_cast<const uint8_t*>(target), page + ps - target, !mode32_);
        }
    }
    flagEvents_.push_back({deadAtTarget ? FlagEventKind::DeadExit : FlagEventKind::Read, 0, 0});
    const size_t eventsAfterExit = flagEvents_.size();
    lastExitDynamic_ = false;
    emitWord(0xF100077Bu);  // SUBS X27, X27, #1
    const size_t bmi = wordPos();
    emitWord(0x54000004u);  // B.MI stub
    emitWord(0xF9400330u);  // LDR X16, [X25]
    emitWord(0xEB1A021Fu);  // CMP X16, X26
    const size_t bne = wordPos();
    emitWord(0x54000001u);  // B.NE stub
    const size_t site = wordPos();
    emitWord(0x14000000u);  // site: B stub
    emitWord(0x58000051u);  // veneer: LDR X17, [PC, #8]
    emitWord(0xD61F0220u);  //         BR X17
    emitWord(0);            //         .quad destino (lo escribe linkExit)
    emitWord(0);
    const size_t stub = wordPos();
    patchImm19(bmi, stub);
    patchImm19(bne, stub);
    patchImm26(site, stub);
    emitWriteRipImm(target);
    // emitWriteRipImm marca una lectura (salida genérica); aquí la salida ya está clasificada.
    if (deadAtTarget && flagEvents_.size() > eventsAfterExit) flagEvents_.resize(eventsAfterExit);
    const int64_t back = (static_cast<int64_t>(site) - static_cast<int64_t>(wordPos())) * 4;  // ADR X17, site
    emitWord(0x10000000u | ((static_cast<uint32_t>(back) & 3u) << 29) |
             (((static_cast<uint32_t>(back >> 2)) & 0x7FFFFu) << 5) | 17u);
    emitStrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, exitSite)));
    pendingExits_.push_back({wordPos(), false, true});
    emitWord(0x14000000u);  // B epílogo+1
}

namespace {
std::mutex s_fastNativeMutex;
std::unordered_map<uintptr_t, bool> s_fastNatives;  // función -> bloqueante
}  // namespace

void OryonEmitterBackend::registerFastNative(uintptr_t fn, bool blocking) {
    if (fn == 0) return;
    std::lock_guard<std::mutex> lock(s_fastNativeMutex);
    s_fastNatives[fn] = blocking;
}

bool OryonEmitterBackend::lookupFastNative(uintptr_t fn, bool& blocking) {
    std::lock_guard<std::mutex> lock(s_fastNativeMutex);
    auto it = s_fastNatives.find(fn);
    if (it == s_fastNatives.end()) return false;
    blocking = it->second;
    return true;
}

std::atomic<uintptr_t> OryonEmitterBackend::fast32AliasBase{0};
std::atomic<uint32_t (*)(CpuContext*, uint32_t, uint32_t)> OryonEmitterBackend::fast32Indirect{nullptr};
std::atomic<const OryonEmitterBackend::Fast32AliasEntry*> OryonEmitterBackend::fast32Table{nullptr};

// CALL indirecto de 32 bits a la zona de alias (W16 = destino): si fast32Table tiene una entrada
// stdcall no bloqueante (0..8 args), la llama directo desde ARM64 sin guardar XMM/GPRs ni pasar por
// dispatchThunk; si no, guarda el estado invitado en ctx y llama a fast32Indirect.
void OryonEmitterBackend::emitFastIndirect32(uintptr_t nextRip, bool tailCall) {
    const Fast32AliasEntry* tbl = fast32Table.load(std::memory_order_relaxed);
    size_t bDoneFast = 0;
    size_t cbzSlow = 0, tbnzSlow = 0;
    if (tbl != nullptr) {
        const auto ldpW = [&](uint8_t rt1, uint8_t rt2, uint8_t rn, uint32_t byteOff) {
            const uint32_t imm7 = (byteOff / 4u) & 0x7Fu;
            emitWord(0x29400000u | (imm7 << 15) | (static_cast<uint32_t>(rt2) << 10) |
                     (static_cast<uint32_t>(rn) << 5) | rt1);
        };
        // W17 = destino - base; idx = (W17 >> 4) & 0xFFFF
        emitWord(0xD3400000u | (1u << 22) | (4u << 16) | (19u << 10) | (17u << 5) | 17u);
        emitMovHostAddr(20, reinterpret_cast<uintptr_t>(tbl), kJitRelocLib);
        emitWord(0x8B000000u | (17u << 16) | (4u << 10) | (20u << 5) | 20u);  // ADD X20, X20, X17, LSL #4
        emitLdpX(21, 22, 20, 0);  // X21 = hostFn, W22 = slots | (blocking << 8)
        cbzSlow = wordPos();
        emitWord(0);  // CBZ X21, slow
        tbnzSlow = wordPos();
        emitWord(0);  // TBNZ W22, #8, slow
        if (tailCall) {
            emitWord(0xB9400097u);  // LDR W23, [X4, #0] (retAddr de la pila; X23 callee-saved)
        }
        // Vía rápida directa ARM64: solo preserva callee-saved Win32 (EBX..EDI = X3..X7).
        emitStrX(3, 19, 24);
        emitStpX(4, 5, 19, 32);
        emitStpX(6, 7, 19, 48);
        if (tailCall) {
            emitAddImm(17, 4, 4, true, false);  // X17 = ESP + 4 (primer argumento)
        } else {
            emitMovReg(17, 4, true);            // X17 = ESP
        }
        ldpW(0, 1, 17, 0);
        ldpW(2, 3, 17, 8);
        emitWord(0x710012DFu);  // CMP W22, #4
        const size_t blsSkip = wordPos();
        emitWord(0);            // B.LS skip4_7
        ldpW(4, 5, 17, 16);
        ldpW(6, 7, 17, 24);
        patchWord(blsSkip, 0x54000000u | (static_cast<uint32_t>(wordPos() - blsSkip) << 5) | 0x9u);
        emitStpPre(29, 30, 31, -2);
        emitBlr(21);
        emitLdpPost(29, 30, 31, 2);
        emitMovReg(0, 0, false);  // EAX = W0
        emitLdrX(3, 19, 24);
        emitLdpX(4, 5, 19, 32);
        emitLdpX(6, 7, 19, 48);
        if (tailCall) {
            emitAddImm(4, 4, 4, true, false);  // pop retAddr
        }
        emitWord(0x8B200000u | (22u << 16) | (2u << 13) | (2u << 10) | (4u << 5) | 4u);  // ADD X4, X4, W22, UXTW #2
        if (tailCall) {
            emitStrX(23, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
            flagEvents_.push_back({FlagEventKind::Read, 0, 0});
            lastExitDynamic_ = true;
            pendingExits_.push_back({wordPos(), false, false});
            emitWord(0x14000000u);  // B epílogo / inlineDispatch
            patchWord(cbzSlow, 0xB4000000u | (static_cast<uint32_t>(wordPos() - cbzSlow) << 5) | 21u);
            patchWord(tbnzSlow, 0x37400000u | (static_cast<uint32_t>(wordPos() - tbnzSlow) << 5) | 22u);
            return;
        }
        bDoneFast = wordPos();
        emitWord(0);  // B suspendCheck
        patchWord(cbzSlow, 0xB4000000u | (static_cast<uint32_t>(wordPos() - cbzSlow) << 5) | 21u);
        patchWord(tbnzSlow, 0x37400000u | (static_cast<uint32_t>(wordPos() - tbnzSlow) << 5) | 22u);
    }
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    emitCtxRelAddr(17, &runState_);
    emitMovImm64(0, kInNative);
    emitWord(0x889FFE20u);  // STLR W0, [X17]
    emitMovReg(0, 19, true);
    emitMovReg(1, 16, false);
    emitMovImm64(2, nextRip);
    emitMovHostAddr(16, reinterpret_cast<uintptr_t>(fast32Indirect.load(std::memory_order_relaxed)), kJitRelocFastNativeBlocking);
    emitStpPre(29, 30, 31, -2);
    emitBlr(16);
    emitLdpPost(29, 30, 31, 2);
    emitMovReg(17, 0, false);  // resultado
    emitCtxRelAddr(16, &runState_);
    emitMovImm64(0, kRunning);
    emitWord(0x889FFE00u);  // STLR W0, [X16]
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    // Flujo cambiado por la función nativa: salida con el RIP que dejó en ctx.
    const size_t cont = wordPos();
    emitWord(0x34000011u);  // CBZ W17, sigue (se parchea)
    flagEvents_.push_back({FlagEventKind::Read, 0, 0});
    pendingExits_.push_back({wordPos(), false, false});
    emitWord(0x14000000u);  // B epílogo
    patchImm19(cont, wordPos());
    if (tbl != nullptr) {
        patchWord(bDoneFast, 0x14000000u | static_cast<uint32_t>(wordPos() - bDoneFast));
    }
    // ¿Suspensión pedida mientras estaba en nativo?
    emitCtxRelAddr(16, &suspendCount_);
    emitWord(0x88DFFE10u);  // LDAR W16, [X16]
    const size_t cbz = wordPos();
    emitWord(0x34000010u);  // CBZ W16, sigue (se parchea)
    emitWriteRipImm(nextRip);
    lastExitDynamic_ = false;
    pendingExits_.push_back({wordPos(), false, false});
    emitWord(0x14000000u);  // B epílogo
    patchImm19(cbz, wordPos());
}

// 32 bits: la IAT guarda el alias bajo de la función nativa; el interceptor da la función real y su número de
// palabras stdcall (-1 si no es stdcall con aridad conocida).
std::atomic<uintptr_t (*)(uintptr_t)> OryonEmitterBackend::fast32Host{nullptr};
std::atomic<int (*)(uintptr_t)> OryonEmitterBackend::fast32Slots{nullptr};

// call [IAT] de 32 bits -> función de la lista rápida (ver emitFastNativeCall). Los argumentos están en la pila
// invitada ([ESP], [ESP+4]...: aún no hay dirección de retorno), X0..X3 = esas palabras extendidas con ceros; al
// volver, EAX = W0 y ESP += 4 * slots (stdcall: limpia la función). Equivale a CALL + RET del huésped.
void OryonEmitterBackend::emitFastNativeCall32(uintptr_t fn, bool blocking, uintptr_t nextRip, uint32_t slots) {
    const uint8_t fnKind = blocking ? kJitRelocFastNativeBlocking : kJitRelocFastNative;
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    if (blocking) {
        emitMovHostAddr(16, fn, fnKind);
        emitStrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, nativeFn)));
        emitCtxRelAddr(16, &runState_);
        emitMovImm64(17, kInNative);
        emitWord(0x889FFE11u);  // STLR W17, [X16]
    }
    // Argumentos desde [ESP] (X4 = ESP, ya extendido con ceros). LDR Wn también extiende con ceros.
    emitMovReg(17, 4, true);
    for (uint8_t i = 0; i < 4; ++i) {
        if (i < slots) emitLdrW(i, 17, static_cast<int32_t>(4 * i));
        else emitMovImm64(i, 0);
    }
    emitMovHostAddr(16, fn, fnKind);
    emitStpPre(29, 30, 31, -2);
    emitBlr(16);
    emitLdpPost(29, 30, 31, 2);
    emitMovReg(0, 0, false);  // EAX = W0 (mitad alta a cero)
    emitStrX(0, 19, static_cast<int32_t>(offsetof(CpuContext, gpr)));
    if (blocking) {
        emitCtxRelAddr(16, &runState_);
        emitMovImm64(17, kRunning);
        emitWord(0x889FFE11u);  // STLR W17, [X16]
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    if (slots > 0) emitAddImm(4, 4, static_cast<uint16_t>(4 * slots), false);  // stdcall: ESP += argumentos
    emitCtxRelAddr(16, &suspendCount_);
    emitWord(0x88DFFE10u);  // LDAR W16, [X16]
    const size_t cbz = wordPos();
    emitWord(0x34000010u);  // CBZ W16, sigue (se parchea)
    emitWriteRipImm(nextRip);
    pendingExits_.push_back({wordPos(), false, false});
    emitWord(0x14000000u);  // B epílogo
    patchImm19(cbz, wordPos());
}

// call [IAT] -> función nativa de la lista rápida, sin salir al despachador:
//   guarda el estado invitado en ctx, (bloqueante: runState = en nativo), X0..X3 = RCX, RDX,
//   R8, R9, BLR, RAX = X0, (runState = ejecutando), recarga el estado y, si alguien pidió
//   suspender el hilo mientras estaba en nativo, sale al despachador con RIP = siguiente.
// Equivale a CALL + RET del huésped (la pila invitada no cambia).
void OryonEmitterBackend::emitFastNativeCall(uintptr_t fn, bool blocking, uintptr_t nextRip, bool tailCall,
                                             uintptr_t iatSlot) {
    // Caché persistente: la función se reubica al cargar y la ranura IAT leída se revalida.
    const uint8_t fnKind = blocking ? kJitRelocFastNativeBlocking : kJitRelocFastNative;
    if (iatSlot != 0) {
        aotDeps_.push_back({static_cast<uint64_t>(iatSlot), static_cast<int64_t>(fn - JitDiskCache::anchor())});
    }
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    if (blocking) {
        emitMovHostAddr(16, fn, fnKind);
        emitStrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, nativeFn)));  // perfil de esperas
        emitCtxRelAddr(16, &runState_);
        emitMovImm64(17, kInNative);
        emitWord(0x889FFE11u);  // STLR W17, [X16]
    }
    emitMovReg(0, 1, true);  // RCX
    emitMovReg(1, 2, true);  // RDX
    emitMovReg(2, 8, true);  // R8
    emitMovReg(3, 9, true);  // R9
    emitMovHostAddr(16, fn, fnKind);
    emitStpPre(29, 30, 31, -2);
    emitBlr(16);
    emitLdpPost(29, 30, 31, 2);
    emitStrX(0, 19, static_cast<int32_t>(offsetof(CpuContext, gpr)));  // RAX
    if (blocking) {
        emitCtxRelAddr(16, &runState_);
        emitMovImm64(17, kRunning);
        emitWord(0x889FFE11u);  // STLR W17, [X16]
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    if (tailCall) {
        // jmp [IAT]: la función nativa "vuelve" a quien llamó al stub: RET del huésped.
        emitLdrX(16, 4, 0);                   // X16 = [RSP]
        emitAddImm(4, 4, 8, true);            // RSP += 8
        if (returnStack.load(std::memory_order_relaxed)) {
            // Es el RET del huésped: también sale de la pila de retornos (si no, se desalinea).
            const int32_t topOff = static_cast<int32_t>(offsetof(CpuContext, rasTop));
            emitLdrX(20, 19, topOff);
            emitSubImm(20, 20, 1, true);
            emitStrX(20, 19, topOff);
        }
        emitWriteRipReg(16, true);            // salida dinámica (búsqueda en línea al final)
    }
    // ¿Suspensión pedida mientras estaba en nativo? LDAR: ordenada tras el STLR anterior.
    emitCtxRelAddr(16, &suspendCount_);
    emitWord(0x88DFFE10u);  // LDAR W16, [X16]
    const size_t cbz = wordPos();
    emitWord(0x34000010u);  // CBZ W16, sigue (se parchea)
    if (!tailCall) emitWriteRipImm(nextRip);
    pendingExits_.push_back({wordPos(), false, false});
    emitWord(0x14000000u);  // B epílogo
    patchImm19(cbz, wordPos());
    if (tailCall) lastExitDynamic_ = true;  // el bloque termina con la búsqueda en línea del RIP
}

// --- Sonda de diagnóstico (setprop debug.hyperplay.probe "rva1,rva2,...") -----------------------
// Un bloque que empieza en <módulo PE>+RVA (base según el cargador, como RtlPcToFileHeader) llama antes de
// su primera instrucción a probeLog, que registra RAX..R9, [RSP] y los textos a los que apuntan
// RCX/RDX/R8/R9. Sirve para ver argumentos y resultados de funciones sin símbolos (coreclr.dll).
namespace {
const std::vector<uint64_t>& probeRvas() {
    static const std::vector<uint64_t> s = [] {
        std::vector<uint64_t> v;
#if defined(__ANDROID__)
        char buf[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.probe", buf) > 0) {
            const char* p = buf;
            while (*p != '\0') {
                char* end = nullptr;
                const unsigned long long x = std::strtoull(p, &end, 16);
                if (end == p) break;
                if (x != 0) v.push_back(x);
                p = (*end == ',') ? end + 1 : end;
            }
        }
#endif
        if (!v.empty()) LOGI("[Sonda] %zu RVAs vigiladas", v.size());
        return v;
    }();
    return s;
}

bool probeReadable(uintptr_t addr, size_t len) {
    // process_vm_readv sobre el propio proceso: falla con EFAULT (sin señal) si alguna página no es
    // legible. mincore no basta: acepta páginas de guarda PROT_NONE.
    if (addr < 0x10000 || len > 256) return false;
    unsigned char scratch[256];
    struct iovec local{scratch, len};
    struct iovec remote{reinterpret_cast<void*>(addr), len};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(len);
}

std::atomic<uint64_t (*)(uint64_t)> s_moduleBaseOf{nullptr};

bool probeMatches(uintptr_t rip) {
    // Dirección absoluta (útil en 32 bits, donde el resolutor de módulos solo conoce imágenes x64).
    for (const uint64_t a : probeRvas()) {
        if (a >= 0x1000000 && rip == a) return true;
    }
    auto baseOf = s_moduleBaseOf.load(std::memory_order_acquire);
    if (baseOf == nullptr) return false;
    const uint64_t base = baseOf(rip);
    if (base == 0) return false;
    // Filtro opcional: solo el módulo cuyo SizeOfImage coincide (debug.hyperplay.probesize, hex). Evita que la
    // misma RVA de otro módulo agote las capturas.
    static const uint32_t s_size = [] {
        char b[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.probesize", b) > 0
                   ? static_cast<uint32_t>(std::strtoul(b, nullptr, 16)) : 0u;
    }();
    if (s_size != 0) {
        if (!probeReadable(base, 0x40)) return false;
        const uint32_t lfanew = *reinterpret_cast<const uint32_t*>(base + 0x3C);
        if (lfanew > 0x1000 || !probeReadable(base + lfanew + 0x50, 4)) return false;
        if (*reinterpret_cast<const uint32_t*>(base + lfanew + 0x50) != s_size) return false;
    }
    for (const uint64_t rva : probeRvas()) {
        if (rip - base == rva) return true;
    }
    return false;
}

std::string probeText(uint64_t p) {
    constexpr size_t kMax = 160;
    if (!probeReadable(p, kMax)) return {};
    const auto* b = reinterpret_cast<const uint8_t*>(p);
    std::string a, w;
    for (size_t i = 0; i < kMax && b[i] >= 0x20 && b[i] < 0x7F; ++i) a.push_back(static_cast<char>(b[i]));
    size_t wi = 0;
    for (; wi + 1 < kMax && b[wi] >= 0x20 && b[wi] < 0x7F && b[wi + 1] == 0; wi += 2) w.push_back(static_cast<char>(b[wi]));
    if (w.size() >= 3) {
        char tail[40];
        std::snprintf(tail, sizeof(tail), "\" len=%zu fin=%02x%02x%02x%02x", w.size(), b[wi], b[wi + 1], b[wi + 2], b[wi + 3]);
        return " u\"" + w + tail;
    }
    if (a.size() >= 3) return " \"" + a + "\"";
    return {};
}
}  // namespace

void OryonEmitterBackend::setModuleBaseResolver(uint64_t (*fn)(uint64_t)) {
    s_moduleBaseOf.store(fn, std::memory_order_release);
}

static void probeLog(CpuContext* c, uint64_t rip) {
    static std::atomic<int> s_hits{0};
    static const int s_max = [] {
        char b[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.hyperplay.probemax", b) > 0 ? std::atoi(b) : 400;
    }();
    if (s_hits.fetch_add(1) >= s_max) return;
    uint64_t ret = 0;
    if (probeReadable(c->gpr[4], 8)) ret = *reinterpret_cast<const uint64_t*>(c->gpr[4]);
    LOGI("[Sonda] rip=%llx rax=%llx%s rcx=%llx%s rdx=%llx%s r8=%llx%s r9=%llx%s ret=%llx",
         static_cast<unsigned long long>(rip), static_cast<unsigned long long>(c->gpr[0]),
         probeText(c->gpr[0]).c_str(),
         static_cast<unsigned long long>(c->gpr[1]), probeText(c->gpr[1]).c_str(),
         static_cast<unsigned long long>(c->gpr[2]), probeText(c->gpr[2]).c_str(),
         static_cast<unsigned long long>(c->gpr[8]), probeText(c->gpr[8]).c_str(),
         static_cast<unsigned long long>(c->gpr[9]), probeText(c->gpr[9]).c_str(),
         static_cast<unsigned long long>(ret));
    static const std::pair<uint64_t, uint32_t> s_mem = [] {  // debug.hyperplay.probemem <dir>,<bytes> (hex)
        char b[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.probemem", b) <= 0) return std::pair<uint64_t, uint32_t>{0, 0};
        char* e = nullptr;
        const uint64_t a = std::strtoull(b, &e, 16);
        const uint32_t n = (e != nullptr && *e == ',') ? static_cast<uint32_t>(std::strtoul(e + 1, nullptr, 16)) : 16u;
        return std::pair<uint64_t, uint32_t>{a, n > 64 ? 64u : n};
    }();
    if (s_mem.first != 0 && probeReadable(s_mem.first, s_mem.second)) {
        char line[64 * 3 + 1] = {};
        const auto* m = reinterpret_cast<const uint32_t*>(s_mem.first);
        size_t o = 0;
        for (uint32_t i = 0; i < s_mem.second / 4; ++i) o += std::snprintf(line + o, sizeof(line) - o, "%08x ", m[i]);
        LOGI("[Sonda]   mem[%llx]: %s", static_cast<unsigned long long>(s_mem.first), line);
    }
    LOGI("[Sonda]   rbx=%llx rsp=%llx rbp=%llx rsi=%llx rdi=%llx", static_cast<unsigned long long>(c->gpr[3]),
         static_cast<unsigned long long>(c->gpr[4]), static_cast<unsigned long long>(c->gpr[5]),
         static_cast<unsigned long long>(c->gpr[6]), static_cast<unsigned long long>(c->gpr[7]));
    if (probeReadable(c->gpr[4], 16)) {  // 32 bits: dirección de retorno y primeros argumentos
        const auto* s = reinterpret_cast<const uint32_t*>(c->gpr[4]);
        LOGI("[Sonda]   [esp]=%08x %08x %08x %08x", s[0], s[1], s[2], s[3]);
    }
    if (probeReadable(c->gpr[8], 32)) {
        const auto* q = reinterpret_cast<const uint64_t*>(c->gpr[8]);
        LOGI("[Sonda]   [r8]=%llx%s [r8+8]=%llx%s [r8+10]=%llx [r8+18]=%llx [r8+20]=%llx",
             static_cast<unsigned long long>(q[0]), probeText(q[0]).c_str(),
             static_cast<unsigned long long>(q[1]), probeText(q[1]).c_str(),
             static_cast<unsigned long long>(q[2]), static_cast<unsigned long long>(q[3]),
             static_cast<unsigned long long>(probeReadable(c->gpr[8] + 32, 8) ? q[4] : 0));
        if (probeReadable(c->gpr[8], 0x50)) {
            LOGI("[Sonda]   [r8+28]=%llx [r8+30]=%llx [r8+38]=%llx [r8+40]=%llx [r8+48]=%llx",
                 static_cast<unsigned long long>(q[5]), static_cast<unsigned long long>(q[6]),
                 static_cast<unsigned long long>(q[7]), static_cast<unsigned long long>(q[8]),
                 static_cast<unsigned long long>(q[9]));
        }
    }
    if (probeReadable(c->gpr[0], 16)) {
        const auto* q = reinterpret_cast<const uint64_t*>(c->gpr[0]);
        LOGI("[Sonda]   [rax]=%llx [rax+8]=%llx", static_cast<unsigned long long>(q[0]),
             static_cast<unsigned long long>(q[1]));
    }
    if (probeReadable(c->gpr[1], 32)) {
        const auto* q = reinterpret_cast<const uint64_t*>(c->gpr[1]);
        LOGI("[Sonda]   [rcx]=%llx%s [rcx+8]=%llx%s [rcx+10]=%llx [rcx+18]=%llx",
             static_cast<unsigned long long>(q[0]), probeText(q[0]).c_str(),
             static_cast<unsigned long long>(q[1]), probeText(q[1]).c_str(),
             static_cast<unsigned long long>(q[2]), static_cast<unsigned long long>(q[3]));
    }
    if (probeReadable(c->gpr[4] + 0x28, 8)) {
        const uint64_t a5 = *reinterpret_cast<const uint64_t*>(c->gpr[4] + 0x28);
        uint64_t a5_28 = 0, asm10 = 0;
        if (probeReadable(a5 + 0x28, 8)) a5_28 = *reinterpret_cast<const uint64_t*>(a5 + 0x28);
        if (probeReadable(a5_28 + 0x10, 8)) asm10 = *reinterpret_cast<const uint64_t*>(a5_28 + 0x10);
        LOGI("[Sonda]   arg5=[rsp+28]=%llx -> [+28]=%llx -> [+10]=%llx", static_cast<unsigned long long>(a5),
             static_cast<unsigned long long>(a5_28), static_cast<unsigned long long>(asm10));
    }
    if (probeReadable(c->gpr[5] + 0x40, 16)) {
        const auto* q = reinterpret_cast<const uint64_t*>(c->gpr[5] + 0x40);
        uint64_t m28 = 0, m8 = 0, m10 = 0;
        if (probeReadable(q[1] + 8, 0x28)) {
            m8 = reinterpret_cast<const uint64_t*>(q[1])[1];
            m10 = reinterpret_cast<const uint64_t*>(q[1])[2];
            m28 = reinterpret_cast<const uint64_t*>(q[1])[5];
        }
        LOGI("[Sonda]   rbp=%llx [rbp+40]=%llx [rbp+48]=%llx -> [+8]=%llx [+10]=%llx [+28]=%llx",
             static_cast<unsigned long long>(c->gpr[5]), static_cast<unsigned long long>(q[0]),
             static_cast<unsigned long long>(q[1]), static_cast<unsigned long long>(m8),
             static_cast<unsigned long long>(m10), static_cast<unsigned long long>(m28));
    }
    LOGI("[Sonda]   rbx=%llx rsi=%llx rdi=%llx r12=%llx r13=%llx r14=%llx r15=%llx",
         static_cast<unsigned long long>(c->gpr[3]), static_cast<unsigned long long>(c->gpr[6]),
         static_cast<unsigned long long>(c->gpr[7]), static_cast<unsigned long long>(c->gpr[12]),
         static_cast<unsigned long long>(c->gpr[13]), static_cast<unsigned long long>(c->gpr[14]),
         static_cast<unsigned long long>(c->gpr[15]));
}

namespace {
void x87Fsin(CpuContext* c) {
    c->x87[c->x87Top & 7u] = std::sin(c->x87[c->x87Top & 7u]);
    c->x87Sw &= static_cast<uint16_t>(~(1u << 10));  // C2 = 0
}
void x87Fcos(CpuContext* c) {
    c->x87[c->x87Top & 7u] = std::cos(c->x87[c->x87Top & 7u]);
    c->x87Sw &= static_cast<uint16_t>(~(1u << 10));  // C2 = 0
}
void x87Fpatan(CpuContext* c) {
    const uint32_t t0 = c->x87Top & 7u;
    const uint32_t t1 = (c->x87Top + 1u) & 7u;
    c->x87[t1] = std::atan2(c->x87[t1], c->x87[t0]);
    c->x87Tag &= static_cast<uint16_t>(~(1u << t0));
    c->x87Top = t1;
}
}  // namespace

void OryonEmitterBackend::emitProbeCall(uintptr_t rip) {
    const int32_t nzcvOffset = static_cast<int32_t>(offsetof(CpuContext, nzcv));
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    emitWord(0xD53B4211u);  // MRS X17, NZCV
    emitStrX(17, 19, nzcvOffset);
    emitMovReg(0, 19, true);
    emitMovImm64(1, rip);
    emitMovHostAddr(16, reinterpret_cast<uintptr_t>(&probeLog), kJitRelocLib);
    emitStpPre(29, 30, 31, -2);
    emitBlr(16);
    emitLdpPost(29, 30, 31, 2);
    emitLdrX(17, 19, nzcvOffset);
    emitWord(0xD51B4211u);  // MSR NZCV, X17
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
}

// Instrucción x87: guarda el estado invitado (GPR y XMM en ctx, NZCV en ctx.nzcv porque la
// llamada a C lo destruye), llama a executeX87(ctx, op, modrm, ea = X16 si hay memoria) y lo
// recarga. No sale al despachador (salvo FCOMI, que lo decide quien llama).
// x87 en línea (sin llamar al intérprete): FLD/FST/FSTP m32/m64, FADD/FMUL/FSUB(R)/FDIV(R) con m32/m64 y con la
// pila (D8/DC/DE), FLD ST(i) y FXCH. Mismo estado que executeX87 (ctx->x87 en doble, x87Top, x87Tag) y misma
// aritmética (doble con el redondeo del FPCR). Antes cada instrucción guardaba y restauraba todo el estado invitado
// (~80 instrucciones) para llamar al intérprete: PhysX de DmC ejecuta millones por segundo. Usa X16 (EA), X17,
// X20..X23, D30 y D31 como temporales. setprop debug.hyperplay.x87native 0 lo desactiva.
bool OryonEmitterBackend::emitX87Native(uint8_t op, uint8_t modrm, bool mem) {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return !(__system_property_get("debug.hyperplay.x87native", v) > 0 && v[0] == '0');
    }();
    if (!s_on) return false;
    const unsigned sub = (modrm >> 3) & 7u, i = modrm & 7u;
    const int32_t offTop = static_cast<int32_t>(offsetof(CpuContext, x87Top));
    const int32_t offTag = static_cast<int32_t>(offsetof(CpuContext, x87Tag));
    const int32_t offSw = static_cast<int32_t>(offsetof(CpuContext, x87Sw));
    const int32_t offCw = static_cast<int32_t>(offsetof(CpuContext, x87Cw));
    const int32_t offSt = static_cast<int32_t>(offsetof(CpuContext, x87));
    if ((offSt % 8) != 0 || offSt / 8 > 4095 || (offSw % 2) != 0 || offSw / 2 > 4095 ||
        (offCw % 2) != 0 || offCw / 2 > 4095) {
        return false;
    }
    const auto andW7 = [&](uint8_t r) { emitWord(0x12000800u | (static_cast<uint32_t>(r) << 5) | r); };  // AND Wr, Wr, #7
    const auto ldD = [&](uint8_t d, uint8_t base, int32_t off) {
        emitWord(0xFD400000u | (static_cast<uint32_t>(off / 8) << 10) | (static_cast<uint32_t>(base) << 5) | d);
    };
    const auto stD = [&](uint8_t d, uint8_t base, int32_t off) {
        emitWord(0xFD000000u | (static_cast<uint32_t>(off / 8) << 10) | (static_cast<uint32_t>(base) << 5) | d);
    };
    const auto ldH = [&](uint8_t rt, uint8_t base, int32_t off) {
        emitWord(0x79400000u | (static_cast<uint32_t>(off / 2) << 10) | (static_cast<uint32_t>(base) << 5) | rt);
    };
    const auto stH = [&](uint8_t rt, uint8_t base, int32_t off) {
        emitWord(0x79000000u | (static_cast<uint32_t>(off / 2) << 10) | (static_cast<uint32_t>(base) << 5) | rt);
    };
    const auto cselW = [&](uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond) {
        emitWord(0x1A800000u | (static_cast<uint32_t>(rm) << 16) | (static_cast<uint32_t>(cond) << 12) |
                 (static_cast<uint32_t>(rn) << 5) | rd);
    };
    // Xr = ctx + 8 * ((top + k) & 7): luego [Xr + offSt] es ST(k).
    const auto stAddr = [&](uint8_t r, unsigned k) {
        emitLdrW(r, 19, offTop);
        if (k != 0) emitAddImm(r, r, static_cast<uint16_t>(k), false);
        andW7(r);
        emitAddRegShift(r, 19, r, 3, true);
    };
    const auto bitW22 = [&] {  // W22 = 1 << W17
        emitMovImm64(22, 1);
        emitWord(0x1AC02000u | (17u << 16) | (22u << 5) | 22u);  // LSLV W22, W22, W17
    };
    const auto push = [&](uint8_t d) {
        emitLdrW(17, 19, offTop);
        emitSubImm(17, 17, 1, false);
        andW7(17);
        emitStrW(17, 19, offTop);
        bitW22();
        emitLdrW(16, 19, offTag);
        emitWord(0x2A000000u | (22u << 16) | (16u << 5) | 16u);  // ORR W16, W16, W22
        emitStrW(16, 19, offTag);
        emitAddRegShift(17, 19, 17, 3, true);
        stD(d, 17, offSt);
    };
    const auto pop = [&] {
        emitLdrW(17, 19, offTop);
        bitW22();
        emitLdrW(16, 19, offTag);
        emitWord(0x0A200000u | (22u << 16) | (16u << 5) | 16u);  // BIC W16, W16, W22
        emitStrW(16, 19, offTag);
        emitAddImm(17, 17, 1, false);
        andW7(17);
        emitStrW(17, 19, offTop);
    };
    // Compara D30 con D31 y actualiza C3/C2/C0 en x87Sw (sin tocar X16 ni X17).
    const auto compareSwNative = [&] {
        emitFcmpD(30, 31);
        emitMovImm64(20, 0);
        emitMovImm64(21, 0x4000u);
        cselW(20, 21, 20, 0x0);  // EQ -> C3=1
        emitMovImm64(21, 0x0100u);
        cselW(20, 21, 20, 0x3);  // LO -> C0=1
        emitMovImm64(21, 0x4500u);
        cselW(20, 21, 20, 0x6);  // VS -> C3=C2=C0=1
        ldH(21, 19, offSw);
        emitMovImm64(22, 0x4500u);
        emitWord(0x0A200000u | (22u << 16) | (21u << 5) | 21u);  // BIC W21, W21, W22
        emitWord(0x2A000000u | (20u << 16) | (21u << 5) | 21u);  // ORR W21, W21, W20
        stH(21, 19, offSw);
    };
    // Compara D30 con D31 y actualiza RFLAGS (ZF/PF/CF) + pfByte para FCOMI/FUCOMI(P).
    const auto compareFlagsNative = [&] {
        emitFcmpD(30, 31);
        emitMovImm64(20, 0);
        emitMovImm64(21, 0x40u);
        cselW(20, 21, 20, 0x0);  // EQ -> ZF=1
        emitMovImm64(21, 0x01u);
        cselW(20, 21, 20, 0x3);  // LO -> CF=1
        emitMovImm64(21, 0x45u);
        cselW(20, 21, 20, 0x6);  // VS -> ZF=PF=CF=1
        emitWord(0x53020A97u);   // UBFX W23, W20, #2, #1 (PF)
        emitWord(0x520002F7u);   // EOR W23, W23, #1 -> pfByte = PF ? 0 : 1
        emitStrB(23, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
        emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitMovImm64(22, ~0x8D5ULL);
        emitAndReg(21, 21, 22, false, true);
        emitOrrReg(21, 21, 20, true);
        emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
    };
    // Redondea D30 según RC (bits 11:10) de x87Cw (0=nearest, 1=floor, 2=ceil, 3=trunc). Preserva X16 y X17.
    const auto roundCwNative = [&] {
        ldH(20, 19, offCw);
        emitWord(0x530A2E94u);  // UBFX W20, W20, #10, #2
        const uint32_t cbzPos = wordPos();
        emitWord(0);            // CBZ W20, nearest
        emitWord(0x71000E9Fu);  // CMP W20, #3
        const uint32_t beqTrunc = wordPos();
        emitWord(0);            // B.EQ trunc
        emitWord(0x7100069Fu);  // CMP W20, #1
        const uint32_t beqFloor = wordPos();
        emitWord(0);            // B.EQ floor
        emitWord(0x1E64C000u | (30u << 5) | 30u);  // FRINTP D30, D30 (ceil)
        const uint32_t bDone1 = wordPos();
        emitWord(0);
        const uint32_t lblFloor = wordPos();
        emitWord(0x1E654000u | (30u << 5) | 30u);  // FRINTM D30, D30 (floor)
        const uint32_t bDone2 = wordPos();
        emitWord(0);
        const uint32_t lblTrunc = wordPos();
        emitWord(0x1E65C000u | (30u << 5) | 30u);  // FRINTZ D30, D30 (trunc)
        const uint32_t bDone3 = wordPos();
        emitWord(0);
        const uint32_t lblNearest = wordPos();
        emitWord(0x1E644000u | (30u << 5) | 30u);  // FRINTN D30, D30 (nearest)
        const uint32_t lblDone = wordPos();
        patchWord(cbzPos, 0x34000000u | (static_cast<uint32_t>(lblNearest - cbzPos) << 5) | 20u);
        patchWord(beqTrunc, 0x54000000u | (static_cast<uint32_t>(lblTrunc - beqTrunc) << 5) | 0x0u);
        patchWord(beqFloor, 0x54000000u | (static_cast<uint32_t>(lblFloor - beqFloor) << 5) | 0x0u);
        patchWord(bDone1, 0x14000000u | static_cast<uint32_t>(lblDone - bDone1));
        patchWord(bDone2, 0x14000000u | static_cast<uint32_t>(lblDone - bDone2));
        patchWord(bDone3, 0x14000000u | static_cast<uint32_t>(lblDone - bDone3));
    };
    // W17 = (x87Sw & ~0x3800) | ((x87Top & 7) << 11)
    const auto computeSwW17 = [&] {
        ldH(17, 19, offSw);
        emitMovImm64(20, 0x3800u);
        emitWord(0x0A200000u | (20u << 16) | (17u << 5) | 17u);  // BIC W17, W17, W20
        emitLdrW(20, 19, offTop);
        andW7(20);
        emitWord(0x2A000000u | (20u << 16) | (11u << 10) | (17u << 5) | 17u);  // ORR W17, W17, W20, LSL #11
    };
    // D30 = D30 op D31 según el subcódigo de D8 (0 add, 1 mul, 4 sub, 5 subr, 6 div, 7 divr).
    const auto arith = [&](unsigned s) -> bool {
        const auto fop = [&](uint32_t base, uint8_t a, uint8_t b) {
            emitWord(base | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(a) << 5) | 30u);
        };
        switch (s) {
            case 0: fop(0x1E602800u, 30, 31); return true;  // FADD
            case 1: fop(0x1E600800u, 30, 31); return true;  // FMUL
            case 4: fop(0x1E603800u, 30, 31); return true;  // FSUB  a - b
            case 5: fop(0x1E603800u, 31, 30); return true;  // FSUBR b - a
            case 6: fop(0x1E601800u, 30, 31); return true;  // FDIV  a / b
            case 7: fop(0x1E601800u, 31, 30); return true;  // FDIVR b / a
            default: return false;
        }
    };
    const auto isArith = [](unsigned s) { return s != 2 && s != 3; };
    if (mem) {
        if (op == 0xD8 || op == 0xDC || op == 0xDA || op == 0xDE) {
            if (op == 0xD8) {
                emitWord(0xBD400000u | (16u << 5) | 31u);  // LDR S31, [X16]
                emitWord(0x1E22C000u | (31u << 5) | 31u);  // FCVT D31, S31
            } else if (op == 0xDC) {
                ldD(31, 16, 0);
            } else if (op == 0xDA) {
                emitWord(0xB9400000u | (16u << 5) | 17u);  // LDR W17, [X16]
                emitWord(0x1E620000u | (17u << 5) | 31u);  // SCVTF D31, W17
            } else {
                emitWord(0x79C00000u | (16u << 5) | 17u);  // LDRSH W17, [X16]
                emitWord(0x1E620000u | (17u << 5) | 31u);  // SCVTF D31, W17
            }
            stAddr(17, 0);
            ldD(30, 17, offSt);
            if (isArith(sub)) {
                arith(sub);
                stD(30, 17, offSt);
            } else {
                compareSwNative();
                if (sub == 3) pop();
            }
            return true;
        }
        if (op == 0xD9 && sub == 0) {
            emitWord(0xBD400000u | (16u << 5) | 31u);
            emitWord(0x1E22C000u | (31u << 5) | 31u);
            push(31);
            return true;
        }
        if (op == 0xDD && sub == 0) {
            ldD(31, 16, 0);
            push(31);
            return true;
        }
        if (op == 0xDB && sub == 0) {  // FILD m32
            emitWord(0xB9400000u | (16u << 5) | 17u);
            emitWord(0x1E620000u | (17u << 5) | 31u);
            push(31);
            return true;
        }
        if (op == 0xDF && sub == 0) {  // FILD m16
            emitWord(0x79C00000u | (16u << 5) | 17u);
            emitWord(0x1E620000u | (17u << 5) | 31u);
            push(31);
            return true;
        }
        if (op == 0xDF && sub == 5) {  // FILD m64
            emitWord(0xF9400000u | (16u << 5) | 17u);  // LDR X17, [X16]
            emitWord(0x9E620000u | (17u << 5) | 31u);  // SCVTF D31, X17
            push(31);
            return true;
        }
        if ((op == 0xD9 || op == 0xDD) && (sub == 2 || sub == 3)) {
            stAddr(17, 0);
            ldD(30, 17, offSt);
            if (op == 0xD9) {
                emitWord(0x1E624000u | (30u << 5) | 30u);  // FCVT S30, D30
                emitWord(0xBD000000u | (16u << 5) | 30u);  // STR S30, [X16]
            } else {
                stD(30, 16, 0);
            }
            if (sub == 3) pop();
            return true;
        }
        if (op == 0xD9 && sub == 5) {  // FLDCW m16
            ldH(17, 16, 0);
            stH(17, 19, offCw);
            return true;
        }
        if (op == 0xD9 && sub == 7) {  // FNSTCW m16
            ldH(17, 19, offCw);
            stH(17, 16, 0);
            return true;
        }
        if (op == 0xDD && sub == 7) {  // FNSTSW m16
            computeSwW17();
            stH(17, 16, 0);
            return true;
        }
        if ((op == 0xDB || op == 0xDF) && (sub == 1 || sub == 2 || sub == 3)) {  // FISTTP/FIST/FISTP m32/m16
            stAddr(17, 0);
            ldD(30, 17, offSt);
            if (sub != 1) roundCwNative();
            emitWord(0x1E780000u | (30u << 5) | 17u);  // FCVTZS W17, D30
            if (op == 0xDB) emitWord(0xB9000000u | (16u << 5) | 17u);  // STR W17, [X16]
            else stH(17, 16, 0);                                      // STRH W17, [X16]
            if (sub == 1 || sub == 3) pop();
            return true;
        }
        if ((op == 0xDD && sub == 1) || (op == 0xDF && sub == 7)) {  // FISTTP/FISTP m64
            stAddr(17, 0);
            ldD(30, 17, offSt);
            if (op == 0xDF) roundCwNative();
            emitWord(0x9E780000u | (30u << 5) | 17u);  // FCVTZS X17, D30
            emitWord(0xF9000000u | (16u << 5) | 17u);  // STR X17, [X16]
            pop();
            return true;
        }
        return false;
    }
    if (op == 0xD8) {  // ST(0) = ST(0) op ST(i) / FCOM(P) ST(i)
        stAddr(17, i);
        ldD(31, 17, offSt);
        stAddr(17, 0);
        ldD(30, 17, offSt);
        if (isArith(sub)) {
            arith(sub);
            stD(30, 17, offSt);
        } else {
            compareSwNative();
            if (sub == 3) pop();
        }
        return true;
    }
    if (op == 0xDC || op == 0xDE) {
        if (op == 0xDC && !isArith(sub)) {  // FCOM(P) ST(i) alias
            stAddr(17, i);
            ldD(31, 17, offSt);
            stAddr(17, 0);
            ldD(30, 17, offSt);
            compareSwNative();
            if (sub == 3) pop();
            return true;
        }
        if (op == 0xDE && sub == 3 && i == 1) {  // FCOMPP
            stAddr(17, 1);
            ldD(31, 17, offSt);
            stAddr(17, 0);
            ldD(30, 17, offSt);
            compareSwNative();
            pop();
            pop();
            return true;
        }
        if (!isArith(sub)) return false;
        // ST(i) = ST(i) op' ST(0) (+ pop en DE)
        stAddr(17, 0);
        ldD(31, 17, offSt);
        stAddr(17, i);
        ldD(30, 17, offSt);
        arith(sub ^ (sub >= 4 ? 1u : 0u));
        stD(30, 17, offSt);
        if (op == 0xDE) pop();
        return true;
    }
    if (op == 0xDA && modrm == 0xE9) {  // FUCOMPP
        stAddr(17, 1);
        ldD(31, 17, offSt);
        stAddr(17, 0);
        ldD(30, 17, offSt);
        compareSwNative();
        pop();
        pop();
        return true;
    }
    if (op == 0xD9) {
        if (sub == 0) {  // FLD ST(i)
            stAddr(17, i);
            ldD(31, 17, offSt);
            push(31);
            return true;
        }
        if (sub == 1) {  // FXCH ST(i)
            stAddr(17, 0);
            stAddr(16, i);
            ldD(30, 17, offSt);
            ldD(31, 16, offSt);
            stD(31, 17, offSt);
            stD(30, 16, offSt);
            return true;
        }
        switch (modrm) {
            case 0xD0: return true;  // FNOP
            case 0xE0:  // FCHS
                stAddr(17, 0);
                ldD(30, 17, offSt);
                emitWord(0x1E614000u | (30u << 5) | 30u);  // FNEG D30, D30
                stD(30, 17, offSt);
                return true;
            case 0xE1:  // FABS
                stAddr(17, 0);
                ldD(30, 17, offSt);
                emitWord(0x1E60C000u | (30u << 5) | 30u);  // FABS D30, D30
                stD(30, 17, offSt);
                return true;
            case 0xE4:  // FTST
                stAddr(17, 0);
                ldD(30, 17, offSt);
                emitWord(0x9E6703FFu);  // FMOV D31, XZR (0.0)
                compareSwNative();
                return true;
            case 0xE5: {  // FXAM: C1=signbit, vacío=C0|C3(0x4100), NaN=C0(0x0100), Inf=C0|C2(0x0500), 0=C3(0x4000), normal=C2(0x0400)
                emitLdrW(17, 19, offTop);
                andW7(17);
                bitW22();  // W22 = 1 << top
                emitLdrW(16, 19, offTag);
                emitAddRegShift(17, 19, 17, 3, true);
                ldD(30, 17, offSt);
                emitWord(0x9E660000u | (30u << 5) | 20u);  // FMOV X20, D30
                emitWord(0xD37FFC00u | (20u << 5) | 21u);  // LSR X21, X20, #63
                emitLslImm(21, 21, 9, false);              // W21 = signbit << 9 (C1)
                emitWord(0x6A00001Fu | (22u << 16) | (16u << 5));  // TST W16, W22
                const uint32_t beqEmpty = wordPos();
                emitWord(0);
                emitMovImm64(22, 0x7FFFFFFFFFFFFFFFull);
                emitAndReg(22, 20, 22, false, true);
                const uint32_t cbzZero = wordPos();
                emitWord(0);
                emitWord(0xD374F800u | (20u << 5) | 22u);  // UBFX X22, X20, #52, #11 (exp)
                emitWord(0xF100001Fu | (0x7FFu << 10) | (22u << 5));  // CMP X22, #0x7FF
                const uint32_t bneNormal = wordPos();
                emitWord(0);
                emitWord(0xD3400000u | (1u << 22) | (51u << 10) | (20u << 5) | 22u);  // UBFX X22, X20, #0, #52
                const uint32_t cbzInf = wordPos();
                emitWord(0);
                emitMovImm64(20, 0x0100u);  // NaN -> C0
                const uint32_t bNan = wordPos();
                emitWord(0);
                const uint32_t lblInf = wordPos();
                emitMovImm64(20, 0x0500u);  // Inf -> C0|C2
                const uint32_t bInf = wordPos();
                emitWord(0);
                const uint32_t lblNormal = wordPos();
                emitMovImm64(20, 0x0400u);  // normal -> C2
                const uint32_t bNormal = wordPos();
                emitWord(0);
                const uint32_t lblZero = wordPos();
                emitMovImm64(20, 0x4000u);  // zero -> C3
                const uint32_t bZero = wordPos();
                emitWord(0);
                const uint32_t lblEmpty = wordPos();
                emitMovImm64(20, 0x4100u);  // empty -> C0|C3
                const uint32_t lblDone = wordPos();
                patchWord(beqEmpty, 0x54000000u | (static_cast<uint32_t>(lblEmpty - beqEmpty) << 5) | 0x0u);
                patchWord(cbzZero, 0xB4000000u | (static_cast<uint32_t>(lblZero - cbzZero) << 5) | 22u);
                patchWord(bneNormal, 0x54000000u | (static_cast<uint32_t>(lblNormal - bneNormal) << 5) | 0x1u);
                patchWord(cbzInf, 0xB4000000u | (static_cast<uint32_t>(lblInf - cbzInf) << 5) | 22u);
                patchWord(bNan, 0x14000000u | static_cast<uint32_t>(lblDone - bNan));
                patchWord(bInf, 0x14000000u | static_cast<uint32_t>(lblDone - bInf));
                patchWord(bNormal, 0x14000000u | static_cast<uint32_t>(lblDone - bNormal));
                patchWord(bZero, 0x14000000u | static_cast<uint32_t>(lblDone - bZero));
                emitWord(0x2A000000u | (21u << 16) | (20u << 5) | 20u);  // ORR W20, W20, W21 (C1)
                ldH(21, 19, offSw);
                emitMovImm64(22, 0x4700u);
                emitWord(0x0A200000u | (22u << 16) | (21u << 5) | 21u);  // BIC W21, W21, W22
                emitWord(0x2A000000u | (20u << 16) | (21u << 5) | 21u);  // ORR W21, W21, W20
                stH(21, 19, offSw);
                return true;
            }
            case 0xE8:  // FLD1
                emitWord(0x1E6E1000u | 31u);  // FMOV D31, #1.0
                push(31);
                return true;
            case 0xEE:  // FLDZ
                emitWord(0x9E6703FFu);  // FMOV D31, XZR (0.0)
                push(31);
                return true;
            case 0xF6:  // FDECSTP
                emitLdrW(17, 19, offTop);
                emitSubImm(17, 17, 1, false);
                andW7(17);
                emitStrW(17, 19, offTop);
                return true;
            case 0xF7:  // FINCSTP
                emitLdrW(17, 19, offTop);
                emitAddImm(17, 17, 1, false);
                andW7(17);
                emitStrW(17, 19, offTop);
                return true;
            case 0xFA:  // FSQRT
                stAddr(17, 0);
                ldD(30, 17, offSt);
                emitWord(0x1E61C000u | (30u << 5) | 30u);  // FSQRT D30, D30
                stD(30, 17, offSt);
                return true;
            case 0xFE:  // FSIN directo (sin pasar por el switch de executeX87)
            case 0xFF:  // FCOS directo
            case 0xF3: {  // FPATAN directo
                const uintptr_t fn = modrm == 0xFE ? reinterpret_cast<uintptr_t>(&x87Fsin)
                                     : modrm == 0xFF ? reinterpret_cast<uintptr_t>(&x87Fcos)
                                                     : reinterpret_cast<uintptr_t>(&x87Fpatan);
                const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
                const uint32_t simdCount = mode32_ ? 8u : 16u;
                for (uint32_t s = 0; s < simdCount; ++s) {
                    const uint8_t reg = s < 8 ? static_cast<uint8_t>(s) : static_cast<uint8_t>(s + 8);
                    emitStrQ(reg, 19, simdOffset + static_cast<int32_t>(s * 16));
                }
                for (uint8_t g = 0; g < 8; ++g) {
                    emitStpX(static_cast<uint8_t>(g * 2), static_cast<uint8_t>(g * 2 + 1), 19, static_cast<int32_t>(g * 16));
                }
                emitWord(0xD53B4211u);  // MRS X17, NZCV
                emitStrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, nzcv)));
                emitMovReg(0, 19, true);
                emitMovHostAddr(16, fn, kJitRelocLib);
                emitStpPre(29, 30, 31, -2);
                emitBlr(16);
                emitLdpPost(29, 30, 31, 2);
                emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, nzcv)));
                emitWord(0xD51B4211u);  // MSR NZCV, X17
                for (uint8_t g = 0; g < 8; ++g) {
                    emitLdpX(static_cast<uint8_t>(g * 2), static_cast<uint8_t>(g * 2 + 1), 19, static_cast<int32_t>(g * 16));
                }
                for (uint32_t s = 0; s < simdCount; ++s) {
                    const uint8_t reg = s < 8 ? static_cast<uint8_t>(s) : static_cast<uint8_t>(s + 8);
                    emitLdrQ(reg, 19, simdOffset + static_cast<int32_t>(s * 16));
                }
                return true;
            }
            case 0xFC:  // FRNDINT
                stAddr(17, 0);
                ldD(30, 17, offSt);
                roundCwNative();
                stD(30, 17, offSt);
                return true;
            default: break;
        }
        return false;
    }
    if (op == 0xDB) {
        if (modrm == 0xE2) {  // FNCLEX
            ldH(17, 19, offSw);
            emitMovImm64(20, 0x7F00u);
            emitAndReg(17, 17, 20, false, false);
            stH(17, 19, offSw);
            return true;
        }
        if (modrm == 0xE3) {  // FNINIT
            emitStrW(31, 19, offTop);
            emitStrW(31, 19, offTag);
            stH(31, 19, offSw);
            emitMovImm64(17, 0x037Fu);
            stH(17, 19, offCw);
            return true;
        }
        if (sub == 5 || sub == 6) {  // FUCOMI / FCOMI
            stAddr(17, i);
            ldD(31, 17, offSt);
            stAddr(17, 0);
            ldD(30, 17, offSt);
            compareFlagsNative();
            return true;
        }
        return false;
    }
    if (op == 0xDD) {
        if (sub == 0) {  // FFREE ST(i)
            emitLdrW(17, 19, offTop);
            if (i != 0) emitAddImm(17, 17, static_cast<uint16_t>(i), false);
            andW7(17);
            bitW22();
            emitLdrW(16, 19, offTag);
            emitWord(0x0A200000u | (22u << 16) | (16u << 5) | 16u);  // BIC W16, W16, W22
            emitStrW(16, 19, offTag);
            return true;
        }
        if (sub == 2 || sub == 3) {  // FST / FSTP ST(i)
            stAddr(17, 0);
            ldD(30, 17, offSt);
            stAddr(17, i);
            stD(30, 17, offSt);
            if (sub == 3) pop();
            return true;
        }
        if (sub == 4 || sub == 5) {  // FUCOM / FUCOMP ST(i)
            stAddr(17, i);
            ldD(31, 17, offSt);
            stAddr(17, 0);
            ldD(30, 17, offSt);
            compareSwNative();
            if (sub == 5) pop();
            return true;
        }
        return false;
    }
    if (op == 0xDF) {
        if (modrm == 0xE0) {  // FNSTSW AX
            computeSwW17();
            emitBfi(0, 17, 0, 16);
            return true;
        }
        if (sub == 5 || sub == 6) {  // FUCOMIP / FCOMIP
            stAddr(17, i);
            ldD(31, 17, offSt);
            stAddr(17, 0);
            ldD(30, 17, offSt);
            compareFlagsNative();
            pop();
            return true;
        }
        return false;
    }
    return false;
}

void OryonEmitterBackend::emitX87Call(uint8_t op, uint8_t modrm, bool hasEa) {
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    const int32_t nzcvOffset = static_cast<int32_t>(offsetof(CpuContext, nzcv));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    emitWord(0xD53B4211u);  // MRS X17, NZCV
    emitStrX(17, 19, nzcvOffset);
    if (hasEa) emitMovReg(3, 16, true);
    else emitMovImm64(3, 0);
    emitMovReg(0, 19, true);
    emitMovImm64(1, op);
    emitMovImm64(2, modrm);
    emitMovHostAddr(16, reinterpret_cast<uintptr_t>(&executeX87), kJitRelocLib);
    emitStpPre(29, 30, 31, -2);
    emitBlr(16);
    emitLdpPost(29, 30, 31, 2);
    emitLdrX(17, 19, nzcvOffset);
    emitWord(0xD51B4211u);  // MSR NZCV, X17
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19, static_cast<int32_t>(i * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitLdrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitLdrQ(static_cast<uint8_t>(i + 16), 19, simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
}

void OryonEmitterBackend::emitInlineDispatch() {
    static_assert(sizeof(FastEntry) == 24 && offsetof(FastEntry, host) == 8 && offsetof(FastEntry, gen) == 16,
                  "la búsqueda en línea asume FastEntry {rip, host, gen}");
    const auto off = [](size_t o) { return static_cast<int32_t>(o); };
    const auto toEpilogue = [&](uint32_t cond) {
        pendingExits_.push_back({wordPos(), true, false});
        emitWord(0x54000000u | cond);  // B.cond epílogo (se parchea)
    };
    emitLdrX(16, 19, off(offsetof(CpuContext, rip)));             // X16 = RIP destino
    emitWord(0xF100077Bu);                                         // SUBS X27, X27, #1 (presupuesto)
    toEpilogue(4);                                                 // B.MI
    emitWord(0xF9400331u);                                         // LDR X17, [X25]  (contador SMC)
    emitWord(0xEB1A023Fu);                                         // CMP X17, X26
    toEpilogue(1);                                                 // B.NE
    emitWord(0xCA504211u);                                         // EOR X17, X16, X16, LSR #16
    // AND X17, X17, #(kFastCacheSize - 1): N=1, immr=0, imms = log2(tamaño) - 1.
    emitWord(0x92400000u | (static_cast<uint32_t>(__builtin_ctzll(kFastCacheSize) - 1) << 10) | (17u << 5) | 17u);
    emitLdrX(20, 19, off(offsetof(CpuContext, fastCacheAddr)));
    emitWord(0x8B110631u);                                         // ADD X17, X17, X17, LSL #1  (x3)
    emitWord(0x8B110E91u);                                         // ADD X17, X20, X17, LSL #3  (x24 bytes)
    emitWord(0xF9400234u);                                         // LDR X20, [X17]      (rip)
    emitWord(0xEB10029Fu);                                         // CMP X20, X16
    toEpilogue(1);
    emitWord(0xF9400A34u);                                         // LDR X20, [X17, #16] (gen)
    emitLdrX(21, 19, off(offsetof(CpuContext, fastGenCopy)));
    emitWord(0xEB15029Fu);                                         // CMP X20, X21
    toEpilogue(1);
    emitWord(0xF9400634u);                                         // LDR X20, [X17, #8]  (host)
    emitWord(0x91000000u | (static_cast<uint32_t>(prologueBytes_) << 10) | (20u << 5) | 20u);  // ADD X20, X20, #prólogo
    emitWord(0xD61F0280u);                                         // BR X20
}

void OryonEmitterBackend::linkExit(uint64_t site, const void* targetHost, uintptr_t targetRip) {
    if (writeArena_ == nullptr || arena_ == nullptr || prologueBytes_ == 0) return;
    const auto base = reinterpret_cast<uintptr_t>(arena_);
    const auto target = reinterpret_cast<uintptr_t>(targetHost) + prologueBytes_;
    std::lock_guard<std::mutex> lock(mutex_);
    if (site < base || site + 4 > base + jitOffset_ || target < base || target >= base + jitOffset_) return;
    if (site + 20 > base + jitOffset_) return;  // sitio + veneer (16 bytes) dentro de la arena
    if (links_.count(site) != 0) return;  // ya enlazado: un sitio solo tiene un destino
    auto* word = reinterpret_cast<uint32_t*>(writeArena_ + (site - base));
    const int64_t delta = (static_cast<int64_t>(target) - static_cast<int64_t>(site)) / 4;
    uint32_t patched;
    if (delta >= -(int64_t{1} << 25) && delta < (int64_t{1} << 25)) {
        patched = 0x14000000u | (static_cast<uint32_t>(delta) & 0x03FFFFFFu);  // B directo
    } else {
        // Lejos: dirección en el veneer (sitio+12) y B al LDR del veneer (sitio+4).
        if (word[1] != 0x58000051u || word[2] != 0xD61F0220u) return;  // bloque sin veneer
        const uint64_t t = target;
        std::memcpy(word + 3, &t, 8);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __builtin___clear_cache(reinterpret_cast<char*>(site + 4), reinterpret_cast<char*>(site + 20));
        patched = 0x14000001u;  // B +4
        ++farLinks_;
    }
    links_.emplace(site, LinkRecord{__atomic_load_n(word, __ATOMIC_RELAXED), targetRip});
    linksByTarget_[targetRip].push_back(site);
    __atomic_store_n(word, patched, __ATOMIC_RELEASE);
    __builtin___clear_cache(reinterpret_cast<char*>(site), reinterpret_cast<char*>(site + 4));
}

// Deshace todos los enlaces (cualquier bloque invalidado podría ser destino de uno).
void OryonEmitterBackend::unlinkAllLocked() {
    if (links_.empty() || writeArena_ == nullptr) {
        links_.clear(); linksByTarget_.clear();
        return;
    }
    const auto base = reinterpret_cast<uintptr_t>(arena_);
    for (const auto& [site, rec] : links_) {
        auto* word = reinterpret_cast<uint32_t*>(writeArena_ + (site - base));
        __atomic_store_n(word, rec.original, __ATOMIC_RELEASE);
        __builtin___clear_cache(reinterpret_cast<char*>(site), reinterpret_cast<char*>(site + 4));
    }
    links_.clear(); linksByTarget_.clear();
}

// Solo los enlaces cuyo destino es un bloque que empieza en [base, end): el resto sigue vivo
// (Mono escribe código nuevo constantemente; deshacer los ~30 000 enlaces cada vez hundía los FPS).
void OryonEmitterBackend::unlinkRangeLocked(uintptr_t rangeBase, uintptr_t rangeEnd) {
    if (links_.empty() || writeArena_ == nullptr) return;
    const auto base = reinterpret_cast<uintptr_t>(arena_);
    auto it = linksByTarget_.lower_bound(rangeBase);
    while (it != linksByTarget_.end() && it->first < rangeEnd) {
        for (const uint64_t site : it->second) {
            auto link = links_.find(site);
            if (link == links_.end() || link->second.targetRip != it->first) continue;
            auto* word = reinterpret_cast<uint32_t*>(writeArena_ + (site - base));
            __atomic_store_n(word, link->second.original, __ATOMIC_RELEASE);
            __builtin___clear_cache(reinterpret_cast<char*>(site), reinterpret_cast<char*>(site + 4));
            links_.erase(link);
        }
        it = linksByTarget_.erase(it);
    }
}

void OryonEmitterBackend::emitEpilogue() {
    emitStrX(31, 19, static_cast<int32_t>(offsetof(CpuContext, exitSite)));  // exitSite = 0 (XZR)
    const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
    for (uint8_t i = 0; i < 8; ++i) {
        emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
        emitStrQ(static_cast<uint8_t>(i + 16), 19,
                 simdOffset + static_cast<int32_t>((i + 8) * 16));
    }
    for (uint8_t i = 0; i < 8; ++i) {
        emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19,
                 static_cast<int32_t>(i * 16));
    }
    emitLdpPost(27, 28, 31, 2);
    emitLdpPost(25, 26, 31, 2);
    emitLdpPost(23, 24, 31, 2);
    emitLdpPost(21, 22, 31, 2);
    emitLdpPost(19, 20, 31, 2);
    emitLdpPost(29, 30, 31, 2);
    emitRet();
}

void OryonEmitterBackend::noteFlagAccess(uint8_t rn, int32_t offset) {
    if (inFlagWrite_ || rn != 19) return;
    if (offset == static_cast<int32_t>(offsetof(CpuContext, rflags)) ||
        offset == static_cast<int32_t>(offsetof(CpuContext, nzcv))) {
        flagEvents_.push_back({FlagEventKind::Read, 0, 0});
    }
}

// Una escritura de flags es muerta si la siguiente operación sobre flags del bloque es otra
// escritura completa: se sustituye su primera palabra por un B al final de la secuencia.
void OryonEmitterBackend::eliminateDeadFlags(size_t blockStartWord, size_t blockEndWord) {
    if (!deadFlags.load(std::memory_order_relaxed) || flagEvents_.size() < 2) {
        flagEvents_.clear();
        return;
    }
    // Ramas del bloque: una escritura que alguna rama salta o en la que aterriza es condicional
    // (p. ej. SHL r, CL con cuenta 0 no toca los flags) y no invalida la escritura anterior.
    const uint8_t* code = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::vector<std::pair<int64_t, int64_t>> branches;  // (origen, destino) en palabras
    for (size_t p = blockStartWord; p < blockEndWord; ++p) {
        uint32_t w;
        std::memcpy(&w, code + p * 4, 4);
        int64_t off = 0;
        if ((w & 0xFC000000u) == 0x14000000u) off = static_cast<int32_t>(w << 6) >> 6;               // B
        else if ((w & 0xFF000010u) == 0x54000000u || (w & 0x7E000000u) == 0x34000000u)            // B.cond, CBZ/CBNZ
            off = static_cast<int32_t>((w >> 5) << 13) >> 13;
        else if ((w & 0x7E000000u) == 0x36000000u) off = static_cast<int32_t>(((w >> 5) & 0x3FFFu) << 18) >> 18;  // TBZ/TBNZ
        else continue;
        branches.emplace_back(static_cast<int64_t>(p), static_cast<int64_t>(p) + off);
    }
    const auto conditional = [&](const FlagEvent& e) {
        const auto s = static_cast<int64_t>(e.start), en = static_cast<int64_t>(e.end);
        for (const auto& [from, to] : branches) {
            if (from < s && to > s) return true;    // la salta o entra a mitad
            if (to > s && to < en) return true;     // aterriza dentro
        }
        return false;
    };
    // Salto hacia atrás que cruza la escritura: un bucle dentro del bloque podría volver a una
    // lectura emitida antes que ella.
    const auto backwardOver = [&](const FlagEvent& e) {
        for (const auto& [from, to] : branches) {
            if (from >= static_cast<int64_t>(e.start) && to <= static_cast<int64_t>(e.start)) return true;
        }
        return false;
    };
    std::vector<size_t> dead;
    for (size_t i = 0; i + 1 < flagEvents_.size(); ++i) {
        const FlagEvent& w = flagEvents_[i];
        if (w.kind != FlagEventKind::Write || w.end <= w.start + 1) continue;
        const FlagEvent& next = flagEvents_[i + 1];
        if (next.kind == FlagEventKind::Write) {
            if (!conditional(next)) dead.push_back(i);
            continue;
        }
        if (next.kind != FlagEventKind::DeadExit || backwardOver(w)) continue;
        // Flags entre bloques: tras la escritura solo hay salidas cuyo destino sobrescribe los
        // flags (o escrituras que alguna rama salta, que no cuentan) hasta otra escritura
        // incondicional o el final del bloque. Cualquier lectura o salida normal la mantiene viva.
        bool isDead = true;
        for (size_t j = i + 1; j < flagEvents_.size(); ++j) {
            const FlagEvent& e = flagEvents_[j];
            if (e.kind == FlagEventKind::Read) { isDead = false; break; }
            if (e.kind == FlagEventKind::Write && !conditional(e)) break;
        }
        if (isDead) dead.push_back(i);
    }
    for (size_t i : dead) {
        const FlagEvent& w = flagEvents_[i];
        patchWord(w.start, 0x14000000u | (static_cast<uint32_t>(w.end - w.start) & 0x03FFFFFFu));
    }
    flagEvents_.clear();
}

void OryonEmitterBackend::emitWriteRipImm(uintptr_t rip) {
    flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // posible salida: los flags quedan vivos
    lastExitDynamic_ = false;
    emitMovImm64(16, static_cast<uint64_t>(rip));
    emitStrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
}

void OryonEmitterBackend::emitWriteRipReg(uint8_t reg, bool is64) {
    flagEvents_.push_back({FlagEventKind::Read, 0, 0});
    lastExitDynamic_ = true;
    if (is64) {
        emitStrX(reg, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
    } else {
        emitStrW(reg, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
    }
}

void OryonEmitterBackend::emitWriteFlags(CfMode cf, OfMode of, uint8_t resultReg,
                                        uint8_t resultShift) {
    // X21 acumula rflags; X22 es el predicado temporal. X16/X17 quedan intactos
    // porque pueden contener el resultado de la operación.
    if (cf == CfMode::Preserve) flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // lee el CF anterior
    const size_t flagStart = wordPos();
    inFlagWrite_ = true;
    struct EndFlagWrite {
        OryonEmitterBackend* self;
        size_t start;
        ~EndFlagWrite() {
            self->inFlagWrite_ = false;
            self->flagEvents_.push_back({FlagEventKind::Write, start, self->wordPos()});
        }
    } endFlagWrite{this, flagStart};
    // RFLAGS desde NZCV con desplazamientos (antes CSET+BFI por flag y paridad NEON: 16
    // instrucciones; ahora 7-9). NZCV>>24 deja N en el bit 7 (SF) y Z en el 6 (ZF).
    emitMrsNzcv(21);
    emitLsrImm(22, 21, 24, true);
    emitWord(0x927A06D6u);  // AND X22, X22, #0xC0 (solo SF y ZF)
    switch (cf) {
        case CfMode::Zero:
            break;
        case CfMode::Carry:
            emitWord(0xB35D76B6u);  // BFXIL X22, X21, #29, #1 (CF = C)
            break;
        case CfMode::Borrow:
            emitWord(0xB35D76B6u);  // BFXIL X22, X21, #29, #1
            emitWord(0xD24002D6u);  // EOR X22, X22, #1 (CF = !C en restas)
            break;
        case CfMode::Preserve:
            emitLdrX(23, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitWord(0xB34002F6u);  // BFXIL X22, X23, #0, #1 (CF anterior)
            break;
        case CfMode::Reg:
            emitWord(0xB3400296u);  // BFXIL X22, X20, #0, #1
            break;
    }
    switch (of) {
        case OfMode::Zero:
            break;
        case OfMode::Overflow:
            emitLsrImm(21, 21, 28, true);
            emitBfi(22, 21, 11, 1);  // OF = V
            break;
        case OfMode::Reg:
            emitBfi(22, 20, 11, 1);
            break;
        case OfMode::Reg1:
            emitLsrImm(23, 20, 1, true);
            emitBfi(22, 23, 11, 1);
            break;
    }
    // PF perezoso: solo se guarda el byte bajo del resultado (las operaciones de 8/16 bits lo
    // llevan desplazado: resultShift lo devuelve a su sitio). Sin resultado, PF = 0 (byte impar).
    if (resultReg != 0xFF) {
        uint8_t src = resultReg;
        if (resultShift != 0) {
            emitLsrImm(23, resultReg, resultShift, false);
            src = 23;
        }
        emitStrB(src, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
    } else {
        emitMovImm64(23, 1);
        emitStrB(23, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
    }
    emitStrX(22, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
    // La materialización no toca NZCV (MRS/CSET/BFI/UBFX/LDR/STR/NEON): el NZCV del host sigue
    // describiendo estos flags para el consumidor inmediato.
    nzcvSeq_ = instrSeq_;
    nzcvInstrStart_ = instrStartWord_;
    nzcvEnd_ = wordPos();
    nzcvCf_ = static_cast<uint8_t>(cf);
    nzcvOf_ = static_cast<uint8_t>(of);
}

// ¿Escribe NZCV, salta o llama esta palabra? (conservador: ante la duda, sí).
static bool clobbersNzcvOrBranches(uint32_t w) {
    if ((w & 0x7C000000u) == 0x14000000u) return true;                     // B / BL
    if ((w & 0xFF000010u) == 0x54000000u) return true;                     // B.cond
    if ((w & 0x7E000000u) == 0x34000000u || (w & 0x7E000000u) == 0x36000000u) return true;  // CBZ/TBZ
    if ((w & 0xFE000000u) == 0xD6000000u) return true;                     // BR/BLR/RET
    if ((w & 0x1F800000u) == 0x11000000u && (w & 0x20000000u)) return true;  // ADDS/SUBS imm
    if ((w & 0x1F800000u) == 0x12000000u && (w & 0x60000000u) == 0x60000000u) return true;  // ANDS imm
    if ((w & 0x1F000000u) == 0x0B000000u && (w & 0x20000000u)) return true;  // ADDS/SUBS reg/ext
    if ((w & 0x1F000000u) == 0x0A000000u && (w & 0x60000000u) == 0x60000000u) return true;  // ANDS/BICS reg
    if ((w & 0x1FE00000u) == 0x1A000000u && (w & 0x20000000u)) return true;  // ADCS/SBCS
    if ((w & 0x1FE00000u) == 0x1A400000u) return true;                     // CCMP/CCMN
    if ((w & 0x5F200C00u) == 0x1E200400u || (w & 0x5F203C00u) == 0x1E202000u) return true;  // FCCMP/FCMP
    if ((w & 0xFFC00000u) == 0xD5000000u || (w & 0xFFF00000u) == 0xD5100000u) return true;  // MSR
    if ((w & 0x3FE07C00u) == 0x3A000800u || (w & 0x3FE07C00u) == 0x3A000400u) return true;  // SETF/RMIF
    return false;
}

// Condición ARM equivalente a la condición x86 sobre el NZCV vigente, o -1 si no se puede
// fusionar (consumidor que no sigue inmediatamente, rama o escritura de NZCV de por medio, o
// flag que no se deduce de NZCV: PF, CF "Reg/Preserve", OF "Reg").
int OryonEmitterBackend::fusedArmCond(uint8_t x86Cond) {
    if (!fuseFlags.load(std::memory_order_relaxed) || nzcvSeq_ + 1 != instrSeq_) return -1;
    const uint8_t* code = (writeArena_ != nullptr) ? writeArena_ : arena_;
    const size_t now = wordPos();
    for (size_t p = nzcvInstrStart_; p < now; ++p) {
        uint32_t w;
        std::memcpy(&w, code + p * 4, 4);
        const uint32_t op = w & 0xFE000000u;
        const bool branch = (w & 0x7C000000u) == 0x14000000u || (w & 0xFF000010u) == 0x54000000u ||
                            (w & 0x7E000000u) == 0x34000000u || (w & 0x7E000000u) == 0x36000000u ||
                            op == 0xD6000000u;
        if (branch) return -1;                                      // flags quizá no escritos
        if (p >= nzcvEnd_ && clobbersNzcvOrBranches(w)) return -1;  // NZCV ya no vale
    }
    const auto cf = static_cast<CfMode>(nzcvCf_);
    const auto of = static_cast<OfMode>(nzcvOf_);
    constexpr int EQ = 0, NE = 1, CS = 2, CC = 3, MI = 4, PL = 5, VS = 6, VC = 7, HI = 8, LS = 9,
                  GE = 10, LT = 11, GT = 12, LE = 13;
    switch (x86Cond) {
        case 0x4: return EQ;
        case 0x5: return NE;
        case 0x8: return MI;
        case 0x9: return PL;
        case 0x0: return of == OfMode::Overflow ? VS : -1;
        case 0x1: return of == OfMode::Overflow ? VC : -1;
        case 0x2: return cf == CfMode::Borrow ? CC : (cf == CfMode::Carry ? CS : -1);
        case 0x3: return cf == CfMode::Borrow ? CS : (cf == CfMode::Carry ? CC : -1);
        case 0x6: return cf == CfMode::Borrow ? LS : (cf == CfMode::Zero ? EQ : -1);
        case 0x7: return cf == CfMode::Borrow ? HI : (cf == CfMode::Zero ? NE : -1);
        case 0xC: return of == OfMode::Overflow ? LT : (of == OfMode::Zero ? MI : -1);
        case 0xD: return of == OfMode::Overflow ? GE : (of == OfMode::Zero ? PL : -1);
        case 0xE: return of == OfMode::Overflow ? LE : -1;
        case 0xF: return of == OfMode::Overflow ? GT : -1;
        default: return -1;  // PF
    }
}

void OryonEmitterBackend::emitParityBits(uint8_t accumulator, uint8_t resultReg,
                                         uint8_t resultShift) {
    // Paridad del byte bajo con CNT de NEON (V31, sin uso por el huésped): bits a 1 del byte 0.
    uint8_t src = resultReg;
    if (resultShift != 0) {
        emitLsrImm(23, resultReg, resultShift, false);
        src = 23;
    }
    emitWord(0x1E270000u | (static_cast<uint32_t>(src & 31) << 5) | 31u);  // FMOV S31, Wsrc
    emitWord(0x0E205BFFu);                                                  // CNT V31.8B, V31.8B
    emitWord(0x0E013FF5u);                                                  // UMOV W21, V31.B[0]
    // x86: PF = 1 si el número de bits a 1 es PAR -> invertir el bit 0 del recuento.
    emitWord(0x520002B5u);                                                  // EOR W21, W21, #1
    emitBfi(accumulator, 21, 2, 1);                                         // PF (bit 2)
}

size_t OryonEmitterBackend::emitJccTest(uint8_t x86Cond) {
    if (const int c = fusedArmCond(x86Cond); c >= 0) {
        const size_t word = wordPos();
        emitWord(0x54000000u | static_cast<uint32_t>(c));  // B.cond (imm19 lo parchea patchImm19)
        return word;
    }
    auto simpleBit = [&](uint8_t bit, bool branchIfSet) -> size_t {
        emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitUbfx(16, 16, bit, 1);
        const size_t word = wordPos();
        if (branchIfSet) {
            emitCbnz(16, 0);
        } else {
            emitCbz(16, 0);
        }
        return word;
    };

    switch (x86Cond) {
        case 0x0:  // O
            return simpleBit(11, true);
        case 0x1:  // NO
            return simpleBit(11, false);
        case 0x2:  // B/C
            return simpleBit(0, true);
        case 0x3:  // AE/NC
            return simpleBit(0, false);
        case 0x4:  // E/Z
            return simpleBit(6, true);
        case 0x5:  // NE/NZ
            return simpleBit(6, false);
        case 0x6: {  // BE: CF || ZF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 0, 1);
            emitUbfx(16, 16, 6, 1);
            emitOrrReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbnz(16, 0);
            return word;
        }
        case 0x7: {  // A: !CF && !ZF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 0, 1);
            emitUbfx(16, 16, 6, 1);
            emitOrrReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbz(16, 0);
            return word;
        }
        case 0x8:  // S
            return simpleBit(7, true);
        case 0x9:  // NS
            return simpleBit(7, false);
        case 0xA:    // P/PE: PF == 1 (paridad par de pfByte)
        case 0xB: {  // NP/PO: PF == 0
            flagEvents_.push_back({FlagEventKind::Read, 0, 0});
            emitLdrB(16, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
            emitWord(0x1E270000u | (16u << 5) | 31u);  // FMOV S31, W16
            emitWord(0x0E205BFFu);                     // CNT V31.8B, V31.8B
            emitWord(0x0E013FF0u);                     // UMOV W16, V31.B[0]
            emitWord(0x12000210u);                     // AND W16, W16, #1 (1 = impar = PF 0)
            const size_t word = wordPos();
            if (x86Cond == 0xA) emitCbz(16, 0);
            else emitCbnz(16, 0);
            return word;
        }
        case 0xC: {  // L: SF != OF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 7, 1);
            emitUbfx(16, 16, 11, 1);
            emitEorReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbnz(16, 0);
            return word;
        }
        case 0xD: {  // GE: SF == OF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 7, 1);
            emitUbfx(16, 16, 11, 1);
            emitEorReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbz(16, 0);
            return word;
        }
        case 0xE: {  // LE: ZF || SF != OF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 7, 1);
            emitUbfx(16, 16, 11, 1);
            emitEorReg(16, 16, 17, true);
            emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 17, 6, 1);
            emitOrrReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbnz(16, 0);
            return word;
        }
        case 0xF: {  // G: !ZF && SF == OF
            emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 16, 7, 1);
            emitUbfx(16, 16, 11, 1);
            emitEorReg(16, 16, 17, true);
            emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 17, 6, 1);
            emitOrrReg(16, 16, 17, true);
            const size_t word = wordPos();
            emitCbz(16, 0);
            return word;
        }
        default:
            lastError_ = "condición Jcc inválida";
            return std::numeric_limits<size_t>::max();
    }
}

// ---------------------------------------------------------------------------
// ModR/M, direcciones efectivas y acceso a operandos
// ---------------------------------------------------------------------------

bool OryonEmitterBackend::decodeModRm(CodeReader& reader, bool rexR, bool rexX, bool rexB,
                                      ModRm& out) {
    uint8_t modrm = 0;
    if (!reader.u8(modrm)) return false;

    out.mod = static_cast<uint8_t>(modrm >> 6);
    out.reg = static_cast<uint8_t>(((modrm >> 3) & 7) + (rexR ? 8 : 0));
    const uint8_t rmLow = static_cast<uint8_t>(modrm & 7);
    out.rm = static_cast<uint8_t>(rmLow + (rexB ? 8 : 0));
    out.isMemory = out.mod != 3;
    if (!out.isMemory) return true;

    bool sibDisp32 = false;
    if (rmLow == 4) {
        uint8_t sib = 0;
        if (!reader.u8(sib)) return false;
        out.scale = static_cast<uint8_t>(sib >> 6);
        const uint8_t indexLow = static_cast<uint8_t>((sib >> 3) & 7);
        const uint8_t baseLow = static_cast<uint8_t>(sib & 7);
        out.hasIndex = !(indexLow == 4 && !rexX);
        out.index = static_cast<uint8_t>(indexLow + (rexX ? 8 : 0));
        out.base = static_cast<uint8_t>(baseLow + (rexB ? 8 : 0));
        if (baseLow == 5 && out.mod == 0) {
            sibDisp32 = true;
            if (!rexB) {
                out.hasBase = false;
                out.absolute = true;
            } else {
                out.hasBase = true;
            }
        } else {
            out.hasBase = true;
        }
    } else if (rmLow == 5 && out.mod == 0) {
        out.hasBase = false;
        if (mode32_) {
            out.absolute = true;
        } else {
            out.ripRelative = true;
        }
    } else {
        out.hasBase = true;
        out.base = out.rm;
    }

    if (out.mod == 1) {
        int8_t disp8 = 0;
        if (!reader.s8(disp8)) return false;
        out.disp = disp8;
    } else if (out.mod == 2 || out.absolute || out.ripRelative || sibDisp32) {
        uint32_t disp32 = 0;
        if (!reader.u32(disp32)) return false;
        out.disp = static_cast<int32_t>(disp32);
    }
    return true;
}

void OryonEmitterBackend::emitEffectiveAddress(const ModRm& modrm, uintptr_t nextRip) {
    // [rsp+..] sin índice ni segmento: pila del propio hilo -> acceso sin TSO. RBP no: MSVC lo
    // usa como registro general en código optimizado (punteros a objetos compartidos).
    eaIsStack_ = !modrm.ripRelative && modrm.hasBase && !modrm.hasIndex && currentSegPrefix_ == 0 &&
                 modrm.base == 4;
    if (modrm.ripRelative) {
        const uintptr_t address = nextRip + static_cast<uintptr_t>(static_cast<int64_t>(modrm.disp));
        emitMovImm64(16, address);
    } else if (modrm.hasBase && !modrm.hasIndex && modrm.disp != 0 && modrm.disp >= -4095 && modrm.disp <= 4095) {
        // [base+disp]: una sola instrucción (antes MOV X16, base + ADD).
        if (modrm.disp > 0) emitAddImm(16, modrm.base, static_cast<uint16_t>(modrm.disp), true);
        else emitSubImm(16, modrm.base, static_cast<uint16_t>(-modrm.disp), true);
    } else {
        if (modrm.hasBase && modrm.hasIndex) {
            emitAddRegShift(16, modrm.base, modrm.index, modrm.scale, true);
        } else if (modrm.hasBase) {
            emitMovReg(16, modrm.base, true);
        } else if (modrm.hasIndex) {
            emitLslImm(16, modrm.index, modrm.scale, true);
        } else {
            emitMovImm64(16, 0);
        }

        if (modrm.disp > 0 && modrm.disp <= 4095) {
            emitAddImm(16, 16, static_cast<uint16_t>(modrm.disp), true);
        } else if (modrm.disp < 0 && modrm.disp >= -4095) {
            emitSubImm(16, 16, static_cast<uint16_t>(-modrm.disp), true);
        } else if (modrm.disp != 0) {
            emitMovImm64(17, static_cast<uint64_t>(static_cast<int64_t>(modrm.disp)));
            emitAddReg(16, 16, 17, false, true);
        }
    }

    if (currentSegPrefix_ == 0x65) {
        emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, gsBase)));
        emitAddReg(16, 16, 17, false, true);
    } else if (currentSegPrefix_ == 0x64) {
        emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, fsBase)));
        emitAddReg(16, 16, 17, false, true);
    }

    if (mode32_) {
        emitMovReg(16, 16, false);
    }
}

void OryonEmitterBackend::emitLoadOperand(const ModRm& modrm, uint8_t hostReg, uint8_t size,
                                          uintptr_t nextRip) {
    if (!modrm.isMemory) {
        if (size == 8) {
            emitMovReg(hostReg, modrm.rm, true);
        } else if (size == 4) {
            emitMovReg(hostReg, modrm.rm, false);
        } else if (size == 1) {
            if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                emitUbfx(hostReg, modrm.rm - 4, 8, 8); // AH, CH, DH, BH
            } else {
                emitUbfx(hostReg, modrm.rm, 0, 8);
            }
        } else if (size == 2) {
            emitUxth(hostReg, modrm.rm);
        }
        return;
    }

    emitEffectiveAddress(modrm, nextRip);
    if (size == 8) {
        emitLdrX(hostReg, 16, 0);
    } else if (size == 4) {
        emitLdrW(hostReg, 16, 0);
    } else if (size == 1) {
        emitLdrB(hostReg, 16, 0);
    } else if (size == 2) {
        emitLdrH(hostReg, 16, 0);
    }
}

void OryonEmitterBackend::emitStoreOperand(const ModRm& modrm, uint8_t hostReg, uint8_t size,
                                           uintptr_t nextRip) {
    if (!modrm.isMemory) {
        if (size == 8) {
            emitMovReg(modrm.rm, hostReg, true);
        } else if (size == 4) {
            emitMovReg(modrm.rm, hostReg, false);
        } else if (size == 1) {
            if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                emitBfi(modrm.rm - 4, hostReg, 8, 8); // AH, CH, DH, BH
            } else {
                emitBfiByte(modrm.rm, hostReg);
            }
        } else if (size == 2) {
            emitBfi(modrm.rm, hostReg, 0, 16);
        }
        return;
    }

    uint8_t srcReg = hostReg;
    if (hostReg == 16) {
        emitMovReg(21, 16, true);
        srcReg = 21;
    }
    emitEffectiveAddress(modrm, nextRip);
    if (size == 8) {
        emitStrX(srcReg, 16, 0);
    } else if (size == 4) {
        emitStrW(srcReg, 16, 0);
    } else if (size == 1) {
        emitStrB(srcReg, 16, 0);
    } else if (size == 2) {
        emitStrH(srcReg, 16, 0);
    }
}

// ---------------------------------------------------------------------------
// Operaciones ALU
// ---------------------------------------------------------------------------

void OryonEmitterBackend::emitAluRegReg(uint8_t opKind, const ModRm& modrm, uint8_t reg,
                                        bool rmIsDst, uint8_t size, uintptr_t nextRip) {
    const bool is64 = size == 8;
    const bool isByte = size == 1;
    const bool isWord = size == 2;

    if (isByte || isWord) {
        if (rmIsDst) {
            if (modrm.isMemory) {
                emitEffectiveAddress(modrm, nextRip);
                if (isByte) emitLdrB(16, 16, 0);
                else emitLdrH(16, 16, 0);
            } else {
                if (isByte) {
                    if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                        emitUbfx(16, modrm.rm - 4, 8, 8);
                    } else {
                        emitUbfx(16, modrm.rm, 0, 8);
                    }
                } else {
                    emitUxth(16, modrm.rm);
                }
            }
            if (isByte) {
                if (!currentRexPresent_ && reg >= 4 && reg <= 7) {
                    emitUbfx(17, reg - 4, 8, 8);
                } else {
                    emitUbfx(17, reg, 0, 8);
                }
            } else {
                emitUxth(17, reg);
            }
        } else {
            if (modrm.isMemory) {
                emitEffectiveAddress(modrm, nextRip);
                if (isByte) emitLdrB(17, 16, 0);
                else emitLdrH(17, 16, 0);
            } else {
                if (isByte) {
                    if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                        emitUbfx(17, modrm.rm - 4, 8, 8);
                    } else {
                        emitUbfx(17, modrm.rm, 0, 8);
                    }
                } else {
                    emitUxth(17, modrm.rm);
                }
            }
            if (isByte) {
                if (!currentRexPresent_ && reg >= 4 && reg <= 7) {
                    emitUbfx(16, reg - 4, 8, 8);
                } else {
                    emitUbfx(16, reg, 0, 8);
                }
            } else {
                emitUxth(16, reg);
            }
        }

        const uint8_t shift = isByte ? 24 : 16;
        switch (opKind) {
            case 0: // ADD
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitAddReg(16, 16, 17, true, false);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 1: // OR
                emitOrrReg(16, 16, 17, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 2: // ADC
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitPrimeCarry(false);
                emitAdcsReg(16, 16, 17, false);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 3: // SBB
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitPrimeCarry(true);
                emitSbcsReg(16, 16, 17, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 4: // AND
                emitAndReg(16, 16, 17, false, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 5: // SUB
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitSubReg(16, 16, 17, true, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 6: // XOR
                emitEorReg(16, 16, 17, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 7: // CMP
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitSubReg(20, 16, 17, true, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, shift);
                return;
            case 8: // TEST
                emitAndReg(20, 16, 17, false, false);
                emitMovReg(23, 20, false);
                emitLslImm(20, 20, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 23, 0);
                return;
            default:
                break;
        }

        if (rmIsDst) {
            if (modrm.isMemory) {
                emitMovReg(21, 16, false);
                emitEffectiveAddress(modrm, nextRip);
                if (isByte) emitStrB(21, 16, 0);
                else emitStrH(21, 16, 0);
            } else {
                if (isByte) {
                    if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                        emitBfi(modrm.rm - 4, 16, 8, 8);
                    } else {
                        emitBfiByte(modrm.rm, 16);
                    }
                } else {
                    emitBfi(modrm.rm, 16, 0, 16);
                }
            }
        } else {
            if (isByte) {
                if (!currentRexPresent_ && reg >= 4 && reg <= 7) {
                    emitBfi(reg - 4, 16, 8, 8);
                } else {
                    emitBfiByte(reg, 16);
                }
            } else {
                emitBfi(reg, 16, 0, 16);
            }
        }
        return;
    }

    if (rmIsDst) {
        if (modrm.isMemory) {
            emitEffectiveAddress(modrm, nextRip);
            if (is64) {
                emitLdrX(16, 16, 0);
            } else {
                emitLdrW(16, 16, 0);
            }
            switch (opKind) {
                case 0:
                    emitAddReg(16, 16, reg, true, is64);
                    emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, 0);
                    break;
                case 1:
                    emitOrrReg(16, 16, reg, is64);
                    emitTstReg(16, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                    break;
                case 2:
                    emitPrimeCarry(false);
                    emitAdcsReg(16, 16, reg, is64);
                    emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, 0);
                    break;
                case 3:
                    emitPrimeCarry(true);
                    emitSbcsReg(16, 16, reg, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, 0);
                    break;
                case 4:
                    emitAndReg(16, 16, reg, true, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                    break;
                case 5:
                    emitSubReg(16, 16, reg, true, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, 0);
                    break;
                case 6:
                    emitEorReg(16, 16, reg, is64);
                    emitTstReg(16, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                    break;
                case 7:
                    emitSubReg(20, 16, reg, true, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, 0);
                    return;
                case 8:
                    emitAndReg(20, 16, reg, true, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, 20, 0);
                    return;
                default:
                    return;
            }
            emitMovReg(21, 16, is64);
            emitEffectiveAddress(modrm, nextRip);
            if (is64) {
                emitStrX(21, 16, 0);
            } else {
                emitStrW(21, 16, 0);
            }
        } else {
            switch (opKind) {
                case 0:
                    emitAddReg(modrm.rm, modrm.rm, reg, true, is64);
                    emitWriteFlags(CfMode::Carry, OfMode::Overflow, modrm.rm, 0);
                    break;
                case 1:
                    emitOrrReg(modrm.rm, modrm.rm, reg, is64);
                    emitTstReg(modrm.rm, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, modrm.rm, 0);
                    break;
                case 2:
                    emitPrimeCarry(false);
                    emitAdcsReg(modrm.rm, modrm.rm, reg, is64);
                    emitWriteFlags(CfMode::Carry, OfMode::Overflow, modrm.rm, 0);
                    break;
                case 3:
                    emitPrimeCarry(true);
                    emitSbcsReg(modrm.rm, modrm.rm, reg, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, modrm.rm, 0);
                    break;
                case 4:
                    emitAndReg(modrm.rm, modrm.rm, reg, true, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, modrm.rm, 0);
                    break;
                case 5:
                    emitSubReg(modrm.rm, modrm.rm, reg, true, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, modrm.rm, 0);
                    break;
                case 6:
                    emitEorReg(modrm.rm, modrm.rm, reg, is64);
                    emitTstReg(modrm.rm, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, modrm.rm, 0);
                    break;
                case 7:
                    emitSubReg(20, modrm.rm, reg, true, is64);
                    emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, 0);
                    break;
                case 8:
                    emitAndReg(20, modrm.rm, reg, true, is64);
                    emitWriteFlags(CfMode::Zero, OfMode::Zero, 20, 0);
                    break;
                default:
                    break;
            }
        }
        return;
    }

    // Destino = reg, fuente = r/m.
    if (modrm.isMemory) {
        emitEffectiveAddress(modrm, nextRip);
        if (is64) {
            emitLdrX(17, 16, 0);
        } else {
            emitLdrW(17, 16, 0);
        }
        switch (opKind) {
            case 0:
                emitAddReg(reg, reg, 17, true, is64);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, reg, 0);
                break;
            case 1:
                emitOrrReg(reg, reg, 17, is64);
                emitTstReg(reg, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 2:
                emitPrimeCarry(false);
                emitAdcsReg(reg, reg, 17, is64);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, reg, 0);
                break;
            case 3:
                emitPrimeCarry(true);
                emitSbcsReg(reg, reg, 17, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, reg, 0);
                break;
            case 4:
                emitAndReg(reg, reg, 17, true, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 5:
                emitSubReg(reg, reg, 17, true, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, reg, 0);
                break;
            case 6:
                emitEorReg(reg, reg, 17, is64);
                emitTstReg(reg, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 7:
                emitSubReg(20, reg, 17, true, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, 0);
                break;
            case 8:
                emitAndReg(20, reg, 17, true, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 20, 0);
                break;
            default:
                break;
        }
    } else {
        switch (opKind) {
            case 0:
                emitAddReg(reg, reg, modrm.rm, true, is64);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, reg, 0);
                break;
            case 1:
                emitOrrReg(reg, reg, modrm.rm, is64);
                emitTstReg(reg, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 2:
                emitPrimeCarry(false);
                emitAdcsReg(reg, reg, modrm.rm, is64);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, reg, 0);
                break;
            case 3:
                emitPrimeCarry(true);
                emitSbcsReg(reg, reg, modrm.rm, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, reg, 0);
                break;
            case 4:
                emitAndReg(reg, reg, modrm.rm, true, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 5:
                emitSubReg(reg, reg, modrm.rm, true, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, reg, 0);
                break;
            case 6:
                emitEorReg(reg, reg, modrm.rm, is64);
                emitTstReg(reg, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, reg, 0);
                break;
            case 7:
                emitSubReg(20, reg, modrm.rm, true, is64);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, 0);
                break;
            case 8:
                emitAndReg(20, reg, modrm.rm, true, is64);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 20, 0);
                break;
            default:
                break;
        }
    }
}

void OryonEmitterBackend::emitAluRegImm(uint8_t opKind, const ModRm& modrm, uint64_t imm,
                                        uint8_t size, uintptr_t nextRip) {
    const bool is64 = size == 8;
    const bool isByte = size == 1;
    const bool isWord = size == 2;

    // Camino rápido: ADD/SUB/CMP reg, imm (32/64 bits, |imm| <= 4095) en una sola instrucción
    // sobre el registro. Con inmediato negativo se usa la operación inversa: add r,-k == SUBS r,#k
    // con CF = C (sin préstamo = acarreo de la suma) y sub r,-k == ADDS r,#k con CF = !C; N, Z y V
    // coinciden.
    if (!modrm.isMemory && (size == 4 || size == 8) && (opKind == 0 || opKind == 5 || opKind == 7)) {
        const int64_t s = is64 ? static_cast<int64_t>(imm) : static_cast<int64_t>(static_cast<int32_t>(imm));
        if (s >= -4095 && s <= 4095) {
            const bool add = (opKind == 0) == (s >= 0);  // instrucción ARM: ADDS o SUBS
            const uint16_t k = static_cast<uint16_t>(s >= 0 ? s : -s);
            const uint8_t dst = opKind == 7 ? 20 : modrm.rm;
            if (add) emitAddImm(dst, modrm.rm, k, is64, true);
            else emitSubImm(dst, modrm.rm, k, is64, true);
            const CfMode cf = (opKind == 0) ? CfMode::Carry : CfMode::Borrow;
            emitWriteFlags(cf, OfMode::Overflow, dst, 0);
            return;
        }
    }

    if (modrm.isMemory) {
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitLdrB(16, 16, 0);
        } else if (size == 2) {
            emitLdrH(16, 16, 0);
        } else if (is64) {
            emitLdrX(16, 16, 0);
        } else {
            emitLdrW(16, 16, 0);
        }
    } else {
        if (isByte) {
            if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                emitUbfx(16, modrm.rm - 4, 8, 8);
            } else {
                emitUbfx(16, modrm.rm, 0, 8);
            }
        } else if (size == 2) {
            emitUxth(16, modrm.rm);
        } else if (is64) {
            emitMovReg(16, modrm.rm, true);
        } else {
            emitMovReg(16, modrm.rm, false);
        }
    }

    emitMovImm64(17, imm);

    if (isByte || isWord) {
        const uint8_t shift = isByte ? 24 : 16;
        switch (opKind) {
            case 0:
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitAddReg(16, 16, 17, true, false);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 1:
                emitOrrReg(16, 16, 17, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 2:
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitPrimeCarry(false);
                emitAdcsReg(16, 16, 17, false);
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 3:
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitPrimeCarry(true);
                emitSbcsReg(16, 16, 17, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 4:
                emitAndReg(16, 16, 17, false, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 5:
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitSubReg(16, 16, 17, true, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, shift);
                emitLsrImm(16, 16, shift, false);
                break;
            case 6:
                emitEorReg(16, 16, 17, false);
                emitLslImm(20, 16, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
                break;
            case 7:
                emitLslImm(16, 16, shift, false);
                emitLslImm(17, 17, shift, false);
                emitSubReg(20, 16, 17, true, false);
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, shift);
                return;
            case 8:
                emitAndReg(20, 16, 17, false, false);
                emitMovReg(23, 20, false);
                emitLslImm(20, 20, shift, false);
                emitTstReg(20, false);
                emitWriteFlags(CfMode::Zero, OfMode::Zero, 23, 0);
                return;
            default:
                return;
        }

        if (modrm.isMemory) {
            emitMovReg(21, 16, false);
            emitEffectiveAddress(modrm, nextRip);
            if (isByte) {
                emitStrB(21, 16, 0);
            } else {
                emitStrH(21, 16, 0);
            }
        } else {
            if (isByte) {
                storeByteRm(modrm.rm, 16);
            } else {
                emitBfi(modrm.rm, 16, 0, 16);
            }
        }
        return;
    }

    switch (opKind) {
        case 0:
            emitAddReg(16, 16, 17, true, is64);
            emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, 0);
            break;
        case 1:
            emitOrrReg(16, 16, 17, is64);
            emitTstReg(16, is64);
            emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
            break;
        case 2:
            emitPrimeCarry(false);
            emitAdcsReg(16, 16, 17, is64);
            emitWriteFlags(CfMode::Carry, OfMode::Overflow, 16, 0);
            break;
        case 3:
            emitPrimeCarry(true);
            emitSbcsReg(16, 16, 17, is64);
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, 0);
            break;
        case 4:
            emitAndReg(16, 16, 17, true, is64);
            emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
            break;
        case 5:
            emitSubReg(16, 16, 17, true, is64);
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, 0);
            break;
        case 6:
            emitEorReg(16, 16, 17, is64);
            emitTstReg(16, is64);
            emitWriteFlags(CfMode::Zero, OfMode::Zero, 16, 0);
            break;
        case 7:
            emitSubReg(20, 16, 17, true, is64);
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 20, 0);
            return;
        case 8:
            emitAndReg(20, 16, 17, true, is64);
            emitWriteFlags(CfMode::Zero, OfMode::Zero, 20, 0);
            return;
        default:
            return;
    }

    if (modrm.isMemory) {
        emitMovReg(21, 16, is64);
        emitEffectiveAddress(modrm, nextRip);
        if (is64) {
            emitStrX(21, 16, 0);
        } else {
            emitStrW(21, 16, 0);
        }
    } else {
        if (is64) {
            emitMovReg(modrm.rm, 16, true);
        } else {
            emitMovReg(modrm.rm, 16, false);
        }
    }
}

void OryonEmitterBackend::emitIncDec(const ModRm& modrm, bool increment, uint8_t size,
                                     uintptr_t nextRip) {
    const bool is64 = size == 8;
    const bool isByte = size == 1;
    const bool isWord = size == 2;

    if (modrm.isMemory) {
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitLdrB(16, 16, 0);
        } else if (isWord) {
            emitLdrH(16, 16, 0);
        } else if (is64) {
            emitLdrX(16, 16, 0);
        } else {
            emitLdrW(16, 16, 0);
        }

        if (isByte || isWord) {
            const uint8_t shift = isByte ? 24 : 16;
            emitLslImm(16, 16, shift, false);
            emitMovImm64(20, 1u << shift);
            if (increment) {
                emitAddReg(16, 16, 20, true, false);
            } else {
                emitSubReg(16, 16, 20, true, false);
            }
            emitWriteFlags(CfMode::Preserve, OfMode::Overflow, 16, shift);
            emitLsrImm(16, 16, shift, false);
        } else {
            if (increment) {
                emitAddImm(16, 16, 1, is64, true);
            } else {
                emitSubImm(16, 16, 1, is64, true);
            }
            emitWriteFlags(CfMode::Preserve, OfMode::Overflow, 16, 0);
        }

        emitMovReg(21, 16, is64);
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitStrB(21, 16, 0);
        } else if (isWord) {
            emitStrH(21, 16, 0);
        } else if (is64) {
            emitStrX(21, 16, 0);
        } else {
            emitStrW(21, 16, 0);
        }
        return;
    }

    if (isByte || isWord) {
        const uint8_t shift = isByte ? 24 : 16;
        if (isByte) {
            loadByteRm(16, modrm.rm);  // inc/dec ah..bh
        } else {
            emitUxth(16, modrm.rm);
        }
        emitLslImm(16, 16, shift, false);
        emitMovImm64(20, 1u << shift);
        if (increment) {
            emitAddReg(16, 16, 20, true, false);
        } else {
            emitSubReg(16, 16, 20, true, false);
        }
        emitWriteFlags(CfMode::Preserve, OfMode::Overflow, 16, shift);
        emitLsrImm(16, 16, shift, false);
        if (isByte) {
            storeByteRm(modrm.rm, 16);
        } else {
            emitBfi(modrm.rm, 16, 0, 16);
        }
    } else {
        if (increment) {
            emitAddImm(modrm.rm, modrm.rm, 1, is64, true);
        } else {
            emitSubImm(modrm.rm, modrm.rm, 1, is64, true);
        }
        emitWriteFlags(CfMode::Preserve, OfMode::Overflow, modrm.rm, 0);
    }
}

// OF de SHL/SHR/SAR (definido por x86 solo con cuenta 1; con otras cuentas es indefinido y vale
// cualquier valor): SHL -> MSB(resultado) XOR CF; SHR -> MSB(original); SAR -> 0. Se deja en el
// bit 1 de X20 (el bit 0 es CF) para OfMode::Reg1. Resultado en X16, original en X17. No toca NZCV.
void OryonEmitterBackend::emitShiftOverflow(uint8_t shiftKind, uint8_t bits) {
    if (shiftKind == 4) {
        emitUbfx(23, 16, static_cast<uint8_t>(bits - 1), 1);
        emitEorReg(23, 23, 20, false);
        emitBfi(20, 23, 1, 1);
    } else if (shiftKind == 5) {
        emitUbfx(23, 17, static_cast<uint8_t>(bits - 1), 1);
        emitBfi(20, 23, 1, 1);
    }
}

void OryonEmitterBackend::emitShift(uint8_t shiftKind, const ModRm& modrm, bool useCl,
                                    uint8_t immCount, uint8_t size, uintptr_t nextRip) {
    const bool is64 = size == 8;
    const bool isByte = size == 1;
    const uint8_t bits = static_cast<uint8_t>(size * 8);
    const uint8_t mask = static_cast<uint8_t>(bits - 1);

    if (useCl) {
        // Destino en X16, original en X17, contador en X20, CF en X21.
        if (modrm.isMemory) {
            emitEffectiveAddress(modrm, nextRip);
            if (isByte) {
                emitLdrB(16, 16, 0);
            } else if (size == 2) {
                emitLdrH(16, 16, 0);
            } else if (is64) {
                emitLdrX(16, 16, 0);
            } else {
                emitLdrW(16, 16, 0);
            }
        } else {
            if (isByte) {
                if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                    emitUbfx(16, modrm.rm - 4, 8, 8);  // AH/CH/DH/BH
                } else {
                    emitUbfx(16, modrm.rm, 0, 8);
                }
            } else if (size == 2) {
                emitUxth(16, modrm.rm);
            } else if (is64) {
                emitMovReg(16, modrm.rm, true);
            } else {
                emitMovReg(16, modrm.rm, false);
            }
        }
        if (size == 2 && shiftKind == 7) emitSxth(16, 16);  // SAR r/m16: signo en los 32 bits
        emitMovReg(17, 16, is64 && !isByte);
        if (isByte) emitMovReg(17, 16, false);

        // x86 enmascara la cuenta a 5 bits (6 con REX.W) también para 8 y 16 bits.
        const uint8_t countBits = is64 ? 6 : 5;
        emitUbfx32(20, 1, 0, countBits);
        // Registro de 32 bits: aunque la cuenta enmascarada sea 0, x86 escribe el registro y pone
        // a cero la mitad alta (antes quedaba intacta: prueba diferencial "shl eax,cl").
        if (size == 4 && !modrm.isMemory) emitMovReg(modrm.rm, modrm.rm, false);
        const size_t skipBranch = wordPos();
        emitCbz(20, 0);

        if (shiftKind == 4) {
            emitMovImm64(21, bits);
            emitSubReg(21, 21, 20, false, true);
            emitLsrv(21, 17, 21, is64 && !isByte);
            emitUbfx(21, 21, 0, 1);
        } else {
            emitSubImm(21, 20, 1, false, false);
            emitLsrv(21, 17, 21, is64 && !isByte);
            emitUbfx(21, 21, 0, 1);
        }

        if (isByte) {
            if (shiftKind == 7) emitSxtb(16, 16);
            if (shiftKind == 4) {
                emitLslv(16, 16, 20, false);
                emitUxtb(16, 16);
            } else if (shiftKind == 5) {
                emitLsrv(16, 16, 20, false);
                emitUxtb(16, 16);
            } else {
                emitAsrv(16, 16, 20, false);
            }
        } else {
            if (shiftKind == 4) {
                emitLslv(16, 16, 20, is64);
            } else if (shiftKind == 5) {
                emitLsrv(16, 16, 20, is64);
            } else {
                emitAsrv(16, 16, 20, is64);
            }
        }

        emitTstSized(16, size);
        emitMovReg(20, 21, true);
        emitShiftOverflow(shiftKind, bits);
        emitWriteFlags(CfMode::Reg, OfMode::Reg1, 16, 0);

        if (modrm.isMemory) {
            emitMovReg(21, 16, is64 && !isByte);
            emitEffectiveAddress(modrm, nextRip);
            if (isByte) {
                emitStrB(21, 16, 0);
            } else if (size == 2) {
                emitStrH(21, 16, 0);
            } else if (is64) {
                emitStrX(21, 16, 0);
            } else {
                emitStrW(21, 16, 0);
            }
        } else {
            if (isByte) {
                storeByteRm(modrm.rm, 16);
            } else if (size == 2) {
                emitBfi(modrm.rm, 16, 0, 16);
            } else if (is64) {
                emitMovReg(modrm.rm, 16, true);
            } else {
                emitMovReg(modrm.rm, 16, false);
            }
        }
        const size_t skipTarget = wordPos();
        patchImm19(skipBranch, skipTarget);
        return;
    }

    const uint8_t count = static_cast<uint8_t>(immCount & mask);
    if (count == 0) {
        // x86 escribe igualmente el registro de 32 bits (mitad alta a cero); flags intactos.
        if (size == 4 && !modrm.isMemory) emitMovReg(modrm.rm, modrm.rm, false);
        return;
    }

    if (modrm.isMemory) {
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitLdrB(16, 16, 0);
        } else if (size == 2) {
            emitLdrH(16, 16, 0);
        } else if (is64) {
            emitLdrX(16, 16, 0);
        } else {
            emitLdrW(16, 16, 0);
        }
    } else {
        if (isByte) {
            if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                emitUbfx(16, modrm.rm - 4, 8, 8);
            } else {
                emitUbfx(16, modrm.rm, 0, 8);
            }
        } else if (size == 2) {
            emitUxth(16, modrm.rm);
        } else if (is64) {
            emitMovReg(16, modrm.rm, true);
        } else {
            emitMovReg(16, modrm.rm, false);
        }
    }
    if (size == 2 && shiftKind == 7) emitSxth(16, 16);  // SAR r/m16: signo en los 32 bits
    emitMovReg(17, 16, is64 && !isByte);
    if (isByte) emitMovReg(17, 16, false);

    if (shiftKind == 4) {
        emitUbfx(20, 17, static_cast<uint8_t>(bits - count), 1);
    } else {
        emitUbfx(20, 17, static_cast<uint8_t>(count - 1), 1);
    }

    if (isByte) {
        if (shiftKind == 7) emitSxtb(16, 16);
        if (shiftKind == 4) {
            emitLslImm(16, 16, count, false);
            emitUxtb(16, 16);
        } else if (shiftKind == 5) {
            emitLsrImm(16, 16, count, false);
            emitUxtb(16, 16);
        } else {
            emitAsrImm(16, 16, count, false);
        }
        emitTstSized(16, size);
    } else {
        if (shiftKind == 4) {
            emitLslImm(16, 16, count, is64);
        } else if (shiftKind == 5) {
            emitLsrImm(16, 16, count, is64);
        } else {
            emitAsrImm(16, 16, count, is64);
        }
        emitTstSized(16, size);
    }
    emitShiftOverflow(shiftKind, bits);
    emitWriteFlags(CfMode::Reg, OfMode::Reg1, 16, 0);

    if (modrm.isMemory) {
        emitMovReg(21, 16, is64 && !isByte);
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitStrB(21, 16, 0);
        } else if (size == 2) {
            emitStrH(21, 16, 0);
        } else if (is64) {
            emitStrX(21, 16, 0);
        } else {
            emitStrW(21, 16, 0);
        }
    } else {
        if (isByte) {
            storeByteRm(modrm.rm, 16);
        } else if (size == 2) {
            emitBfi(modrm.rm, 16, 0, 16);
        } else if (is64) {
            emitMovReg(modrm.rm, 16, true);
        } else {
            emitMovReg(modrm.rm, 16, false);
        }
    }
}

void OryonEmitterBackend::emitRotate(uint8_t rotKind, const ModRm& modrm, bool useCl,
                                     uint8_t immCount, uint8_t size, uintptr_t nextRip) {
    const bool is64 = (size == 8);
    const bool isByte = (size == 1);
    const uint8_t bits = static_cast<uint8_t>(size * 8);
    const uint8_t msb = static_cast<uint8_t>(bits - 1);

    // Cargar operando en X16 / W16
    if (modrm.isMemory) {
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitLdrB(16, 16, 0);
        } else if (size == 2) {
            emitLdrH(16, 16, 0);
        } else if (is64) {
            emitLdrX(16, 16, 0);
        } else {
            emitLdrW(16, 16, 0);
        }
    } else {
        if (isByte) {
            if (!currentRexPresent_ && modrm.rm >= 4 && modrm.rm <= 7) {
                emitUbfx(16, modrm.rm - 4, 8, 8);
            } else {
                emitUbfx(16, modrm.rm, 0, 8);
            }
        } else if (size == 2) {
            emitUxth(16, modrm.rm);
        } else if (is64) {
            emitMovReg(16, modrm.rm, true);
        } else {
            emitMovReg(16, modrm.rm, false);
        }
    }
    // ROL/ROR de 16 bits: se replica el valor en las dos mitades de W16 (periodo 16), así la
    // rotación de 32 bits deja en los bits 0..15 el resultado correcto para cualquier cuenta, y
    // los bits 31/30 coinciden con 15/14 para CF/OF. (Antes se rotaba el valor extendido con
    // ceros: `rol ax, 8` de 0x0300 daba 0 y Unity leía mal los u16 big-endian.)
    if (size == 2 && rotKind <= 1) {
        emitLslImm(17, 16, 16, false);
        emitOrrReg(16, 16, 17, false);
    }

    if (useCl) {
        const uint8_t countBits = is64 ? 6 : 5;
        emitUbfx32(20, 1, 0, countBits);
        if (isByte) {
            emitUbfx32(20, 20, 0, 3);
        }
        // Registro de 32 bits: con cuenta 0 x86 escribe igualmente el registro (mitad alta a 0).
        if (size == 4 && !modrm.isMemory) emitMovReg(modrm.rm, modrm.rm, false);
        const size_t skipBranch = wordPos();
        emitCbz(20, 0);

        if (rotKind == 0) {
            // ROL
            if (isByte) {
                emitLslImm(17, 16, 8, false);
                emitOrrReg(17, 17, 16, false);
                emitSubReg(21, 31, 20, false, false);
                emitUbfx32(21, 21, 0, 3);
                emitLsrv(16, 17, 21, false);
                emitUxtb(16, 16);
            } else {
                emitSubReg(21, 31, 20, false, is64);
                emitRorv(16, 16, 21, is64);
            }
            emitUbfx(22, 16, 0, 1);
        } else if (rotKind == 1) {
            // ROR
            if (isByte) {
                emitLslImm(17, 16, 8, false);
                emitOrrReg(17, 17, 16, false);
                emitLsrv(16, 17, 20, false);
                emitUxtb(16, 16);
            } else {
                emitRorv(16, 16, 20, is64);
            }
            emitUbfx(22, 16, msb, 1);
        } else {
            // RCL/RCR por CL: bucle de rotaciones de 1 bit a través de CF (X17 = CF en curso).
            // Antes: no rotaba y ponía CF a 0 (prueba diferencial contra x86 real).
            emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(17, 17, 0, 1);
            const size_t loop = wordPos();
            const size_t exitBranch = wordPos();
            emitCbz(20, 0);
            if (rotKind == 2) {
                emitUbfx(22, 16, msb, 1);           // CF' = MSB
                emitLslImm(16, 16, 1, true);
                emitOrrReg(16, 16, 17, true);
            } else {
                emitUbfx(22, 16, 0, 1);             // CF' = LSB
                emitLsrImm(16, 16, 1, true);
                emitLslImm(23, 17, msb, true);
                emitOrrReg(16, 16, 23, true);
            }
            if (bits < 64) emitUbfx(16, 16, 0, bits);
            emitMovReg(17, 22, true);
            emitSubImm(20, 20, 1, false, false);
            emitWord(0x14000000u | (static_cast<uint32_t>(static_cast<int64_t>(loop) - static_cast<int64_t>(wordPos())) & 0x03FFFFFFu));
            patchImm19(exitBranch, wordPos());
            emitMovReg(22, 17, true);
        }

        // Actualizar CF en ctx.rflags
        emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitBfi(21, 22, 0, 1);
        emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));

        // Guardar resultado
        if (modrm.isMemory) {
            emitMovReg(21, 16, is64 && !isByte);
            emitEffectiveAddress(modrm, nextRip);
            if (isByte) {
                emitStrB(21, 16, 0);
            } else if (size == 2) {
                emitStrH(21, 16, 0);
            } else if (is64) {
                emitStrX(21, 16, 0);
            } else {
                emitStrW(21, 16, 0);
            }
        } else {
            if (isByte) {
                storeByteRm(modrm.rm, 16);
            } else if (size == 2) {
                emitBfi(modrm.rm, 16, 0, 16);
            } else if (is64) {
                emitMovReg(modrm.rm, 16, true);
            } else {
                emitMovReg(modrm.rm, 16, false);
            }
        }
        const size_t skipTarget = wordPos();
        patchImm19(skipBranch, skipTarget);
        return;
    }

    const uint8_t rawCount = is64 ? (immCount & 63) : (immCount & 31);
    const uint8_t count = isByte ? (rawCount % 8) : (rawCount % bits);
    if (count == 0) {
        if (size == 4 && !modrm.isMemory) emitMovReg(modrm.rm, modrm.rm, false);  // mitad alta a 0
        return;
    }

    if (rotKind == 0) {
        // ROL
        if (isByte) {
            emitLslImm(17, 16, 8, false);
            emitOrrReg(17, 17, 16, false);
            const uint8_t rorCount = static_cast<uint8_t>((8 - count) % 8);
            emitLsrImm(16, 17, rorCount, false);
            emitUxtb(16, 16);
        } else {
            const uint8_t rorCount = static_cast<uint8_t>((bits - count) % bits);
            if (rorCount != 0) {
                emitExtr(16, 16, 16, rorCount, is64);
            }
        }
        emitUbfx(20, 16, 0, 1);
        if (count == 1) {
            emitUbfx(22, 16, msb, 1);
            emitEorReg(22, 22, 20, false);
        }
    } else if (rotKind == 1) {
        // ROR
        if (isByte) {
            emitLslImm(17, 16, 8, false);
            emitOrrReg(17, 17, 16, false);
            emitLsrImm(16, 17, count, false);
            emitUxtb(16, 16);
        } else {
            emitExtr(16, 16, 16, count, is64);
        }
        emitUbfx(20, 16, msb, 1);
        if (count == 1) {
            emitUbfx(22, 16, msb - 1, 1);
            emitEorReg(22, 22, 20, false);
        }
    } else if (rotKind == 2 || rotKind == 3) {
        // RCL/RCR n (rotación de bits+1 con CF). Antes solo n = 1 y con ORR de 32 bits (truncaba
        // los registros de 64); con otra cuenta no rotaba y ponía CF a 0 (prueba diferencial).
        // x86: 8/16 bits usan la cuenta módulo 9/17.
        const uint8_t n = (bits < 32) ? static_cast<uint8_t>(rawCount % (bits + 1)) : rawCount;
        if (n == 0) {
            if (size == 4 && !modrm.isMemory) emitMovReg(modrm.rm, modrm.rm, false);
            return;
        }
        emitLdrX(20, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitUbfx(20, 20, 0, 1);                 // CF de entrada
        emitMovReg(17, 16, true);               // original
        if (rotKind == 2) {
            // res = x << n | CF << (n-1) | x >> (bits-n+1); CF' = bit (bits-n) de x
            emitUbfx(21, 17, static_cast<uint8_t>(bits - n), 1);
            emitLslImm(16, 17, n, true);
            emitLslImm(22, 20, static_cast<uint8_t>(n - 1), true);
            emitOrrReg(16, 16, 22, true);
            if (n > 1) {
                emitLsrImm(22, 17, static_cast<uint8_t>(bits - n + 1), true);
                emitOrrReg(16, 16, 22, true);
            }
        } else {
            // res = x >> n | CF << (bits-n) | x << (bits-n+1); CF' = bit (n-1) de x
            emitUbfx(21, 17, static_cast<uint8_t>(n - 1), 1);
            emitLsrImm(16, 17, n, true);
            emitLslImm(22, 20, static_cast<uint8_t>(bits - n), true);
            emitOrrReg(16, 16, 22, true);
            if (n > 1) {
                emitLslImm(22, 17, static_cast<uint8_t>(bits - n + 1), true);
                emitOrrReg(16, 16, 22, true);
            }
        }
        if (bits < 64) emitUbfx(16, 16, 0, bits);
        // OF (solo definido con n = 1): RCL -> MSB(res) XOR CF'; RCR -> MSB(res) XOR MSB-1(res).
        emitUbfx(22, 16, msb, 1);
        if (rotKind == 2) {
            emitEorReg(22, 22, 21, false);
        } else {
            emitUbfx(23, 16, static_cast<uint8_t>(msb - 1), 1);
            emitEorReg(22, 22, 23, false);
        }
        emitMovReg(20, 21, false);
        emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitBfi(21, 20, 0, 1);
        emitBfi(21, 22, 11, 1);
        emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        if (modrm.isMemory) {
            emitMovReg(21, 16, true);
            emitEffectiveAddress(modrm, nextRip);
            if (isByte) emitStrB(21, 16, 0);
            else if (size == 2) emitStrH(21, 16, 0);
            else if (is64) emitStrX(21, 16, 0);
            else emitStrW(21, 16, 0);
        } else {
            if (isByte) storeByteRm(modrm.rm, 16);
            else if (size == 2) emitBfi(modrm.rm, 16, 0, 16);
            else emitMovReg(modrm.rm, 16, is64);
        }
        return;
    } else {
        emitMovImm64(20, 0);
    }

    emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
    emitBfi(21, 20, 0, 1);
    if (count == 1) {
        emitBfi(21, 22, 11, 1);
    }
    emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));

    if (modrm.isMemory) {
        emitMovReg(21, 16, is64 && !isByte);
        emitEffectiveAddress(modrm, nextRip);
        if (isByte) {
            emitStrB(21, 16, 0);
        } else if (size == 2) {
            emitStrH(21, 16, 0);
        } else if (is64) {
            emitStrX(21, 16, 0);
        } else {
            emitStrW(21, 16, 0);
        }
    } else {
        if (isByte) {
            storeByteRm(modrm.rm, 16);
        } else if (size == 2) {
            emitBfi(modrm.rm, 16, 0, 16);
        } else if (is64) {
            emitMovReg(modrm.rm, 16, true);
        } else {
            emitMovReg(modrm.rm, 16, false);
        }
    }
}

// ---------------------------------------------------------------------------
// Decodificador x86 / x86_64
// ---------------------------------------------------------------------------

size_t OryonEmitterBackend::decodeAndEmitOne(CodeReader& reader, uintptr_t rip,
                                             bool& isControlFlow) {
    isControlFlow = false;
    const uint8_t* const instStart = reader.p;
    lockActive_ = false;
    lockCaptured_ = false;
    lockStored_ = false;
    lockInstrStartWord_ = wordPos();

    auto fail = [&](const char* message) -> size_t {
        lastError_ = formatDiagnosticReportLocked(rip, message);
        LOGE("[OryonEmitter] %s", lastError_.c_str());
        return 0;
    };

    bool rexW = false;
    bool rexR = false;
    bool rexX = false;
    bool rexB = false;
    bool rexPresent = false;
    bool prefixF2 = false;
    bool prefixF3 = false;
    bool prefix66 = false;
    currentSegPrefix_ = 0;

    for (;;) {
        if (reader.p >= reader.end) return fail("fin de código en prefijos");
        const uint8_t b = *reader.p;
        if (!mode32_ && b >= 0x40 && b <= 0x4F) {
            rexW = (b & 0x08) != 0;
            rexR = (b & 0x04) != 0;
            rexX = (b & 0x02) != 0;
            rexB = (b & 0x01) != 0;
            rexPresent = true;
            ++reader.p;
            continue;
        }
        if (b == 0x66) {
            prefix66 = true;
            ++reader.p;
            continue;
        }
        if (b == 0xF0) {  // LOCK: la lectura-modificación-escritura de memoria será atómica
            lockActive_ = true;
            ++reader.p;
            continue;
        }
        if (b == 0x67) {  // Address-size override: ignorado
            ++reader.p;
            continue;
        }
        if (b == 0xF2) {
            prefixF2 = true;
            ++reader.p;
            continue;
        }
        if (b == 0xF3) {
            prefixF3 = true;
            ++reader.p;
            continue;
        }
        if (b == 0x2E || b == 0x3E || b == 0x26 || b == 0x36) {
            ++reader.p;
            continue;
        }
        if (b == 0x64 || b == 0x65) {
            currentSegPrefix_ = b;
            ++reader.p;
            continue;
        }
        break;
    }

    currentRexPresent_ = rexPresent;

    uint8_t op = 0;
    if (!reader.u8(op)) return fail("fin de código sin opcode");

    const bool is64 = rexW;
    const uint8_t stackSize = mode32_ ? 4 : 8;

    auto nextRip = [&]() -> uintptr_t {
        return rip + static_cast<uintptr_t>(reader.offset(instStart));
    };
    auto readImm8 = [&](uint8_t& value) -> bool {
        return reader.u8(value);
    };
    auto readImm16 = [&](uint16_t& value) -> bool {
        return reader.u16(value);
    };
    auto readImm32 = [&](uint32_t& value) -> bool {
        return reader.u32(value);
    };

    auto finish = [&]() -> size_t {
        if (!reader.ok) {
            lastError_ = "lectura de instrucción truncada";
            return 0;
        }
        return reader.offset(instStart);
    };

    auto check8 = [&](uint8_t) -> bool {
        return true;
    };

    // --- 0x0F: familia de dos bytes ------------------------------------------------
    if (op == 0x0F) {
        uint8_t op2 = 0;
        if (!reader.u8(op2)) return fail("0F truncado");

        // F3 0F 1E: ENDBR64/32 (FA/FB) y RDSSPD/Q (CET). Sin pila sombra (como una CPU sin CET)
        // son NOP: RDSSP deja el registro sin tocar (CoreCLR lee 0 = "sin CET"). Los decodifica
        // el caso genérico 0F 19..1F de abajo (ModRM completo, con REX).
        if (op2 == 0x0D || op2 == 0x18 || (op2 >= 0x19 && op2 <= 0x1F)) {
            // 0F 0D: PREFETCH / PREFETCHW
            // 0F 18: PREFETCHh (PREFETCHNTA, PREFETCHT0-T2)
            // 0F 19..1F: NOP arquitectonico multi-byte
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PREFETCH/NOP 0F truncado");
            emitNop();
            return finish();
        }

        // --- 0x0F 0xAE: FXSAVE, FXRSTOR, LDMXCSR, STMXCSR, XSAVE, XRSTOR, LFENCE, MFENCE, SFENCE, CLFLUSH ---
        if (op2 == 0xAE) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("0F AE truncado");
            const uint8_t groupExt = (m.reg & 7u);
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                switch (groupExt) {
                    case 0: { // FXSAVE m512
                        emitMovImm64(17, 0x037F);
                        emitStrH(17, 16, 0);       // FCW
                        emitMovImm64(17, 0);
                        emitStrH(17, 16, 2);       // FSW
                        emitStrB(17, 16, 4);       // FTW
                        emitStrB(17, 16, 5);       // Reserved
                        emitStrH(17, 16, 6);       // FOP
                        emitStrX(31, 16, 8);       // FIP = 0
                        emitStrX(31, 16, 16);      // FDP = 0
                        emitMovImm64(17, 0x1F80);
                        emitStrW(17, 16, 24);      // MXCSR = 0x1F80
                        emitMovImm64(17, 0xFFFF);
                        emitStrW(17, 16, 28);      // MXCSR_MASK
                        for (uint8_t i = 0; i < 16; ++i) {
                            emitStrQ(xmmHost(i), 16, 160 + i * 16);
                        }
                        break;
                    }
                    case 1: { // FXRSTOR m512
                        for (uint8_t i = 0; i < 16; ++i) {
                            emitLdrQ(xmmHost(i), 16, 160 + i * 16);
                        }
                        break;
                    }
                    case 2: { // LDMXCSR m32
                        emitLdrW(17, 16, 0);
                        break;
                    }
                    case 3: { // STMXCSR m32
                        emitMovImm64(17, 0x1F80);
                        emitStrW(17, 16, 0);
                        break;
                    }
                    case 4:   // XSAVE
                    case 5:   // XRSTOR
                    case 6:   // XSAVEOPT
                    case 7:   // CLFLUSH m8
                        emitNop();
                        break;
                }
            } else {
                // mod == 3: barreras de memoria
                switch (groupExt) {
                    case 5: // LFENCE (0F AE E8)
                        emitWord(0xD50339BFu); // DMB ISHLD
                        break;
                    case 6: // MFENCE (0F AE F0)
                        emitWord(0xD5033BBFu); // DMB ISH
                        break;
                    case 7: // SFENCE (0F AE F8)
                        emitWord(0xD5033ABFu); // DMB ISHST
                        break;
                    default:
                        emitNop();
                        break;
                }
            }
            return finish();
        }

        if (op2 == 0xAF) {  // IMUL r, r/m
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("IMUL truncado");
            const uint8_t size = is64 ? 8 : 4;
            const bool op64 = size == 8;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (op64) {
                    emitLdrX(17, 16, 0);
                } else {
                    emitLdrW(17, 16, 0);
                }
            } else {
                if (op64) {
                    emitMovReg(17, m.rm, true);
                } else {
                    emitMovReg(17, m.rm, false);
                }
            }
            if (op64) {
                emitMulReg(16, m.reg, 17, true);
                emitSmulh(20, m.reg, 17);
                emitAsrImm(17, 16, 63, true);
                emitSubReg(31, 20, 17, true, true);
                emitCset(20, kCondNe);
                emitMovReg(m.reg, 16, true);
                emitTstReg(m.reg, true);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, m.reg, 0);
            } else if (prefix66) {
                // IMUL r16, r/m16: solo los 16 bits bajos; CF=OF si el producto no cabe en 16 bits
                // con signo. Antes escribía los 32 bits (prueba diferencial "imul ax,dx").
                emitSxth(17, 17);
                emitSxth(20, m.reg);
                emitMulReg(16, 20, 17, false);   // W16 = producto de 32 bits
                emitSxth(17, 16);
                emitSubReg(31, 16, 17, true, false);   // SUBS WZR, W16, W17 (setFlags, 32 bits)
                emitCset(20, kCondNe);
                emitBfi(m.reg, 16, 0, 16);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 16, 0);
            } else {
                emitSmull(16, m.reg, 17);
                emitSxtw(17, 16);
                emitSubReg(31, 16, 17, true, true);
                emitCset(20, kCondNe);
                emitMovReg(m.reg, 16, false);
                emitTstReg(m.reg, false);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, m.reg, 0);
            }
            return finish();
        }

        if (op2 == 0xB6 || op2 == 0xB7 || op2 == 0xBE || op2 == 0xBF) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVZX/MOVSX truncado");
            const bool source16 = (op2 == 0xB7 || op2 == 0xBF);
            const bool signExtend = (op2 == 0xBE || op2 == 0xBF);
            if (!source16 && m.isMemory == false && !check8(m.rm)) return 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (source16) {
                    emitLdrH(16, 16, 0);
                } else {
                    emitLdrB(16, 16, 0);
                }
                if (signExtend) {
                    if (source16) {
                        emitSxth(16, 16);
                    } else {
                        emitSxtb(16, 16);
                    }
                }
            } else {
                if (source16) {
                    emitUxth(16, m.rm);
                    if (signExtend) emitSxth(16, m.rm);
                } else {
                    if (!currentRexPresent_ && m.rm >= 4 && m.rm <= 7) {
                        emitUbfx(16, m.rm - 4, 8, 8); // AH, CH, DH, BH
                    } else {
                        emitUbfx(16, m.rm, 0, 8);
                    }
                    if (signExtend) emitSxtb(16, 16);
                }
            }
            if (is64) {
                emitMovReg(m.reg, 16, true);
            } else {
                emitMovReg(m.reg, 16, false);
            }
            return finish();
        }

        if (op2 >= 0x40 && op2 <= 0x4F) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CMOVcc truncado");
            const uint8_t cond = static_cast<uint8_t>(op2 - 0x40);
            const uint8_t size = is64 ? 8 : (prefix66 ? 2 : 4);
            if (!m.isMemory && size != 2) {
                if (const int c = fusedArmCond(cond); c >= 0) {
                    // CSEL Xd/Wd, src, dst, cond (la forma W pone a cero la mitad alta, como x86).
                    emitWord((size == 8 ? 0x9A800000u : 0x1A800000u) | (static_cast<uint32_t>(m.reg & 31) << 16) |
                             (static_cast<uint32_t>(c) << 12) | (static_cast<uint32_t>(m.rm & 31) << 5) |
                             static_cast<uint32_t>(m.reg & 31));
                    return finish();
                }
            }
            // Invierte la condición para saltar si NO se cumple:
            const size_t skip = emitJccTest(static_cast<uint8_t>(cond ^ 1));
            if (skip == std::numeric_limits<size_t>::max()) return 0;
            emitLoadOperand(m, 16, size, nextRip());
            if (size == 2) {
                emitBfi(m.reg, 16, 0, 16);
            } else if (size == 4) {
                emitMovReg(m.reg, 16, false);
            } else {
                emitMovReg(m.reg, 16, true);
            }
            const size_t end = wordPos();
            patchImm19(skip, end);
            // x86: CMOVcc r32 escribe SIEMPRE el destino (pone a cero los bits 63:32 aunque la
            // condición sea falsa). Antes la rama falsa dejaba la mitad alta intacta.
            if (size == 4) emitMovReg(m.reg, m.reg, false);
            return finish();
        }

        if (op2 >= 0x80 && op2 <= 0x8F) {
            int32_t rel = 0;
            uint32_t raw = 0;
            if (!readImm32(raw)) return fail("Jcc rel32 truncado");
            rel = static_cast<int32_t>(raw);
            uintptr_t target = nextRip() + static_cast<uintptr_t>(static_cast<int64_t>(rel));
            if (mode32_) target &= 0xFFFFFFFFu;
            const size_t branch = emitJccTest(static_cast<uint8_t>(op2 - 0x80));
            if (branch == std::numeric_limits<size_t>::max()) return 0;
            emitLinkableExit(nextRip());  // termina en B: no hace falta saltar la rama tomada
            const size_t taken = wordPos();
            patchImm19(branch, taken);
            emitLinkableExit(target);
            isControlFlow = true;
            return finish();
        }

        // --- MOVNTPS/MOVNTPD (0F 2B) y MOVNTDQ (66 0F E7): escritura no temporal de 128 bits.
        // La pista "no temporal" no cambia el resultado: STR Q como MOVAPS/MOVDQA.
        if ((op2 == 0x2B && !prefixF2 && !prefixF3) || (op2 == 0xE7 && prefix66)) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVNT truncado");
            if (!m.isMemory) return fail("MOVNT con destino registro");
            emitEffectiveAddress(m, nextRip());
            emitStrQ(xmmHost(m.reg), 16, 0);
            return finish();
        }

        // --- SSE Data Movement (0x10, 0x11, 0x28, 0x29, 0x6F, 0x7F, 0xD6) ---------
        if (op2 == 0x28 || op2 == 0x29 || op2 == 0x10 || op2 == 0x11 ||
            op2 == 0x6F || op2 == 0x7F || op2 == 0xD6) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SSE mov truncado");
            const uint8_t dst = xmmHost(m.reg);

            if (m.isMemory) emitEffectiveAddress(m, nextRip());
            if (op2 == 0x28 || op2 == 0x6F) {
                if (m.isMemory) {
                    emitLdrQ(dst, 16, 0);
                } else {
                    emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1F) << 16) |
                             ((xmmHost(m.rm) & 0x1F) << 5) | dst);
                }
            } else if (op2 == 0x29 || op2 == 0x7F) {
                if (m.isMemory) {
                    emitStrQ(dst, 16, 0);
                } else {
                    emitWord(0x4EA01C00u | ((dst & 0x1F) << 16) | ((dst & 0x1F) << 5) |
                             xmmHost(m.rm));
                }
            } else if (op2 == 0xD6) {
                if (m.isMemory) {
                    emitStrD(dst, 16, 0);
                } else {
                    emitWord(0x4EA01C00u | ((dst & 0x1F) << 16) | ((dst & 0x1F) << 5) |
                             xmmHost(m.rm));
                }
            } else if (op2 == 0x10) {
                // MOVSD/MOVSS xmm, xmm solo sustituyen el lane 0 (los altos se conservan); desde
                // memoria ponen a cero el resto.
                if (prefixF2) {
                    if (m.isMemory) {
                        emitLdrD(dst, 16, 0);
                    } else {
                        emitInsElementD(dst, 0, xmmHost(m.rm), 0);
                    }
                } else if (prefixF3) {
                    if (m.isMemory) {
                        emitLdrS(dst, 16, 0);
                    } else {
                        emitInsElementS(dst, 0, xmmHost(m.rm), 0);
                    }
                } else if (m.isMemory) {
                    emitLdrQ(dst, 16, 0);
                } else {
                    emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1Fu) << 16) |
                             ((xmmHost(m.rm) & 0x1Fu) << 5) | dst);
                }
            } else {  // 0x11
                if (prefixF2) {
                    if (m.isMemory) {
                        emitStrD(dst, 16, 0);
                    } else {
                        emitInsElementD(xmmHost(m.rm), 0, dst, 0);
                    }
                } else if (prefixF3) {
                    if (m.isMemory) {
                        emitStrS(dst, 16, 0);
                    } else {
                        emitInsElementS(xmmHost(m.rm), 0, dst, 0);
                    }
                } else if (m.isMemory) {
                    emitStrQ(dst, 16, 0);
                } else {
                    emitWord(0x4EA01C00u | ((dst & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5) |
                             xmmHost(m.rm));
                }
            }
            return finish();
        }

        // --- 0x0F 0x14 / 0x15: UNPCKLPS/D, UNPCKHPS/D ------------------------------
        if (op2 == 0x14 || op2 == 0x15) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("UNPCK truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            const bool isHigh = (op2 == 0x15);
            const uint8_t sz = prefix66 ? 3 : 2;  // 3: 2D (double), 2: 4S (single)
            const uint32_t base = isHigh ? 0x4E007800u : 0x4E003800u;
            emitWord(base | ((static_cast<uint32_t>(sz) & 3u) << 22) |
                     ((src & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
            return finish();
        }

        // --- F3 0F 12 MOVSLDUP / F3 0F 16 MOVSHDUP / F2 0F 12 MOVDDUP (SSE3) --------
        if ((op2 == 0x12 || op2 == 0x16) && (prefixF3 || prefixF2)) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVxDUP truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF2) emitLdrD(kXmmScratch, 16, 0);
                else emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            if (prefixF2) {
                if (op2 != 0x12) return fail("F2 0F 16 no definido");
                emitWord(0x4E080400u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));  // DUP Vd.2D, Vn.D[0]
            } else if (op2 == 0x12) {
                emitWord(0x4E802800u | ((src & 0x1Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));  // TRN1 4S
            } else {
                emitWord(0x4E806800u | ((src & 0x1Fu) << 16) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));  // TRN2 4S
            }
            return finish();
        }

        // --- F2 0F 7C/7D HADDPS/HSUBPS, 66 0F 7C/7D HADDPD/HSUBPD (SSE3) --------------
        if ((op2 == 0x7C || op2 == 0x7D) && (prefixF2 || prefix66)) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("HADD/HSUB truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            const bool pd = prefix66;
            const uint32_t rmn = ((src & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5);
            if (op2 == 0x7C) {
                emitWord((pd ? 0x6E60D400u : 0x6E20D400u) | rmn | (dst & 0x1Fu));  // FADDP
            } else {
                // {a0-a1, a2-a3, b0-b1, b2-b3}: UZP1/UZP2 y resta
                emitWord((pd ? 0x4EC01800u : 0x4E801800u) | rmn | kXmmScratch2);  // UZP1
                emitWord((pd ? 0x4EC05800u : 0x4E805800u) | rmn | kXmmScratch3);  // UZP2
                emitWord((pd ? 0x4EE0D400u : 0x4EA0D400u) | (kXmmScratch3 << 16) | (kXmmScratch2 << 5) |
                         (dst & 0x1Fu));  // FSUB
            }
            return finish();
        }

        // --- 0x0F 0x12, 0x13, 0x16, 0x17: MOVLPS/D, MOVHPS/D, MOVHLPS, MOVLHPS ---
        if (op2 == 0x12 || op2 == 0x13 || op2 == 0x16 || op2 == 0x17) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVxPS/D truncado");
            const uint8_t reg = xmmHost(m.reg);
            if (op2 == 0x12) {
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrD(kXmmScratch, 16, 0);
                    emitInsD(reg, kXmmScratch);
                } else {
                    const uint8_t src = xmmHost(m.rm);
                    emitWord(0x6E084400u | ((src & 0x1Fu) << 5) | (reg & 0x1Fu));
                }
            } else if (op2 == 0x13) {
                if (!m.isMemory) return fail("MOVLPS/D r,r no soportado");
                emitEffectiveAddress(m, nextRip());
                emitStrD(reg, 16, 0);
            } else if (op2 == 0x16) {
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrD(kXmmScratch, 16, 0);
                    emitWord(0x6E180400u | ((kXmmScratch & 0x1Fu) << 5) | (reg & 0x1Fu));
                } else {
                    const uint8_t src = xmmHost(m.rm);
                    emitWord(0x6E180400u | ((src & 0x1Fu) << 5) | (reg & 0x1Fu));
                }
            } else {  // op2 == 0x17
                if (!m.isMemory) return fail("MOVHPS/D r,r no soportado");
                emitEffectiveAddress(m, nextRip());
                emitWord(0x6E084400u | ((reg & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
                emitStrD(kXmmScratch, 16, 0);
            }
            return finish();
        }

        // --- 0x0F 0x2A: CVTSI2SS (F3) / CVTSI2SD (F2) -------------------------------
        if (op2 == 0x2A) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTSI2Sx truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t srcGpr = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (is64) {
                    emitLdrX(16, 16, 0);
                } else {
                    emitLdrW(16, 16, 0);
                }
                srcGpr = 16;
            } else {
                srcGpr = m.rm;
            }

            if (prefixF2) {
                emitScvtfD(kXmmScratch, srcGpr, is64);
                emitInsD(dst, kXmmScratch);
            } else {
                emitScvtfS(kXmmScratch, srcGpr, is64);
                emitInsS(dst, kXmmScratch);
            }
            return finish();
        }

        // --- 0x0F 0x2D: CVTSS2SI (F3) / CVTSD2SI (F2): redondeo MXCSR (al par más cercano) ---
        if (op2 == 0x2D && (prefixF2 || prefixF3)) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTSx2SI truncado");
            uint8_t srcXmm = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF2) emitLdrD(kXmmScratch, 16, 0);
                else emitLdrS(kXmmScratch, 16, 0);
                srcXmm = kXmmScratch;
            } else {
                srcXmm = xmmHost(m.rm);
            }
            // FCVTNS con el valor "indefinido" de x86 (NaN o fuera de rango -> 0x80000000...).
            uint32_t w[8];
            const int n = x86FloatToIntWords(w, 0x1E200000u, m.reg, srcXmm, prefixF2, is64);
            for (int i = 0; i < n; ++i) emitWord(w[i]);
            return finish();
        }

        // --- 0x0F 0x2C: CVTTSS2SI (F3) / CVTTSD2SI (F2) -----------------------------
        if (op2 == 0x2C) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTTSx2SI truncado");
            const uint8_t dstGpr = m.reg;
            uint8_t srcXmm = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF2) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrS(kXmmScratch, 16, 0);
                }
                srcXmm = kXmmScratch;
            } else {
                srcXmm = xmmHost(m.rm);
            }

            // FCVTZS con el valor "indefinido" de x86 (NaN o fuera de rango -> 0x80000000...);
            // ARM satura, y eso cambiaba resultados de (int)f con valores grandes o NaN.
            uint32_t w[8];
            const int n = x86FloatToIntWords(w, 0x1E380000u, dstGpr, srcXmm, prefixF2, is64);
            for (int i = 0; i < n; ++i) emitWord(w[i]);
            return finish();
        }

        // --- 0x0F 0x2E / 0x2F: UCOMISS/COMISS / UCOMISD/COMISD ----------------------
        if (op2 == 0x2E || op2 == 0x2F) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("UCOMISx truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            const bool isDouble = prefix66;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (isDouble) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrS(kXmmScratch, 16, 0);
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (isDouble) {
                emitFcmpD(dst, src);
            } else {
                emitFcmpS(dst, src);
            }

            // Síntesis de banderas x86 EFLAGS para float compare:
            // Equal: ZF=1, PF=0, CF=0 (0x40)
            // Less than: ZF=0, PF=0, CF=1 (0x01)
            // Unordered (NaN): ZF=1, PF=1, CF=1 (0x45)
            // Greater than: ZF=0, PF=0, CF=0 (0x00)
            emitMovImm64(20, 0);
            emitMovImm64(16, 0x40);
            emitWord(0x1A940210u);  // csel w16, w16, w20, eq
            emitMovImm64(17, 0x01);
            emitWord(0x1A903231u);  // csel w17, w17, w16, lo (cc)
            emitMovImm64(21, 0x45);
            emitWord(0x1A9162B4u);  // csel w20, w21, w17, vs
            emitWord(0x53020A97u);  // UBFX W23, W20, #2, #1 (PF)
            emitWord(0x520002F7u);  // EOR W23, W23, #1 -> pfByte = PF ? 0 : 1
            emitStrB(23, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitMovImm64(22, ~0x8D5ULL);
            emitAndReg(21, 21, 22, false, true);
            emitOrrReg(21, 21, 20, true);
            emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            return finish();
        }

        // --- 0x0F 0x5A: CVTSS2SD (F3) / CVTSD2SS (F2) / CVTPS2PD / CVTPD2PS (66) ------
        if (op2 == 0x5A) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTSx2Sx truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF3) {
                    emitLdrS(kXmmScratch, 16, 0);
                } else if (prefixF2) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else if (prefix66) {
                    emitLdrQ(kXmmScratch, 16, 0);
                } else {
                    emitLdrD(kXmmScratch, 16, 0);
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            if (prefixF3) {
                emitFcvtDS(kXmmScratch, src);
                emitInsD(dst, kXmmScratch);
            } else if (prefixF2) {
                emitFcvtSD(kXmmScratch, src);
                emitInsS(dst, kXmmScratch);
            } else if (prefix66) {
                // CVTPD2PS: 2 dobles -> 2 singles (fcvtn Vd.2S, Vn.2D)
                emitWord(0x0E616800u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
                emitWord(0x4E181FE0u | (dst & 0x1Fu));
            } else {
                // CVTPS2PD: 2 singles -> 2 dobles (fcvtl Vd.2D, Vn.2S)
                emitWord(0x0E617800u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
            }
            return finish();
        }

        // --- 0x0F 0x5B: CVTDQ2PS / CVTPS2DQ (66) / CVTTPS2DQ (F3) ---------------------
        if (op2 == 0x5B) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTx2x 0x5B truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (prefix66 || prefixF3) {
                // CVTPS2DQ (FCVTNS, al par) / CVTTPS2DQ (FCVTZS, truncado) con el entero
                // "indefinido" de x86 por carril: NaN o desbordamiento -> 0x80000000 (ARM satura).
                const uint32_t s = src & 0x1Fu, d = dst & 0x1Fu;
                const uint32_t r = kXmmScratch5, c = kXmmScratch6, k = kXmmScratch4;
                emitWord((prefix66 ? 0x4E21A800u : 0x4EA1B800u) | (s << 5) | r);  // FCVT r.4S, s.4S
                emitWord(0x6F046400u | c);                                       // MVNI c.4S, #0x80, LSL #24
                emitWord(0x6EA08C00u | (c << 16) | (r << 5) | c);                // CMEQ c, r, c (saturó)
                emitWord(0x4E20E400u | (s << 16) | (s << 5) | k);                // FCMEQ k, s, s (no NaN)
                emitWord(0x4EE01C00u | (k << 16) | (c << 5) | c);                // ORN c, c, k
                emitWord(0x4F046400u | k);                                       // MOVI k.4S, #0x80, LSL #24
                emitWord(0x6E601C00u | (r << 16) | (k << 5) | c);                // BSL c, k, r
                emitWord(0x4EA01C00u | (c << 16) | (c << 5) | d);                // MOV d, c
            } else {
                // CVTDQ2PS: 4 ints -> 4 floats (scvtf Vd.4S, Vn.4S)
                emitWord(0x4E21D800u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
            }
            return finish();
        }

        // --- SSE Arith / Logic (0x51..0x57, 0x58, 0x59, 0x5C..0x5F, 0xEF, 0xDB, 0xDF, 0xEB)
        if (op2 == 0x51 || op2 == 0x52 || op2 == 0x53 || op2 == 0x58 || op2 == 0x59 || op2 == 0x5C || op2 == 0x5D || op2 == 0x5E || op2 == 0x5F ||
            op2 == 0x54 || op2 == 0x55 || op2 == 0x56 || op2 == 0x57 ||
            op2 == 0xEF || op2 == 0xDB || op2 == 0xDF || op2 == 0xEB) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SSE arith truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF3) {
                    emitLdrS(kXmmScratch, 16, 0);
                } else if (prefixF2) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (op2 == 0x57 || op2 == 0xEF) {
                emitEorV(dst, dst, src);
            } else if (op2 == 0x54 || op2 == 0xDB) {
                emitAndV(dst, dst, src);
            } else if (op2 == 0x55 || op2 == 0xDF) {
                emitBicV(dst, src, dst);
            } else if (op2 == 0x56 || op2 == 0xEB) {
                emitOrrV(dst, dst, src);
            } else if (prefixF3) {
                if (op2 == 0x51) emitFsqrtS(kXmmScratch, src);
                else if (op2 == 0x52) emitFrsqrteS(kXmmScratch, src);
                else if (op2 == 0x53) emitFrecpeS(kXmmScratch, src);
                else if (op2 == 0x58) emitFaddS(kXmmScratch, dst, src);
                else if (op2 == 0x59) emitFmulS(kXmmScratch, dst, src);
                else if (op2 == 0x5C) emitFsubS(kXmmScratch, dst, src);
                else if (op2 == 0x5D) emitFminS(kXmmScratch, dst, src);
                else if (op2 == 0x5E) emitFdivS(kXmmScratch, dst, src);
                else if (op2 == 0x5F) emitFmaxS(kXmmScratch, dst, src);
                emitInsS(dst, kXmmScratch);
            } else if (prefixF2) {
                if (op2 == 0x51) emitFsqrtD(kXmmScratch, src);
                else if (op2 == 0x52) emitFrsqrteD(kXmmScratch, src);
                else if (op2 == 0x53) emitFrecpeD(kXmmScratch, src);
                else if (op2 == 0x58) emitFaddD(kXmmScratch, dst, src);
                else if (op2 == 0x59) emitFmulD(kXmmScratch, dst, src);
                else if (op2 == 0x5C) emitFsubD(kXmmScratch, dst, src);
                else if (op2 == 0x5D) emitFminD(kXmmScratch, dst, src);
                else if (op2 == 0x5E) emitFdivD(kXmmScratch, dst, src);
                else if (op2 == 0x5F) emitFmaxD(kXmmScratch, dst, src);
                emitInsD(dst, kXmmScratch);
            } else if (prefix66) {
                if (op2 == 0x51) emitFsqrtV2D(dst, src);
                else if (op2 == 0x52) emitFrsqrteV2D(dst, src);
                else if (op2 == 0x53) emitFrecpeV2D(dst, src);
                else if (op2 == 0x58) emitFaddV2D(dst, dst, src);
                else if (op2 == 0x59) emitFmulV2D(dst, dst, src);
                else if (op2 == 0x5C) emitFsubV2D(dst, dst, src);
                else if (op2 == 0x5D) emitFminV2D(dst, dst, src);
                else if (op2 == 0x5E) emitFdivV2D(dst, dst, src);
                else if (op2 == 0x5F) emitFmaxV2D(dst, dst, src);
            } else {
                if (op2 == 0x51) emitFsqrtV(dst, src);
                else if (op2 == 0x52) emitFrsqrteV(dst, src);
                else if (op2 == 0x53) emitFrecpeV(dst, src);
                else if (op2 == 0x58) emitFaddV(dst, dst, src);
                else if (op2 == 0x59) emitFmulV(dst, dst, src);
                else if (op2 == 0x5C) emitFsubV(dst, dst, src);
                else if (op2 == 0x5D) emitFminV(dst, dst, src);
                else if (op2 == 0x5E) emitFdivV(dst, dst, src);
                else if (op2 == 0x5F) emitFmaxV(dst, dst, src);
            }
            return finish();
        }

        // --- 66 0F 64 / 65 / 66: PCMPGTB / PCMPGTW / PCMPGTD (con signo) -> CMGT -------
        if ((op2 == 0x64 || op2 == 0x65 || op2 == 0x66) && prefix66) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PCMPGT truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            const uint32_t size = op2 - 0x64u;  // 0 B, 1 H, 2 S
            emitWord(0x4E203400u | (size << 22) | ((src & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
            return finish();
        }

        // --- 0x0F 0x74 / 0x75 / 0x76: PCMPEQB / PCMPEQW / PCMPEQD -------------------
        if (op2 == 0x74 || op2 == 0x75 || op2 == 0x76) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PCMPEQ truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            if (op2 == 0x74) {
                emitCmeqV16B(dst, dst, src);
            } else if (op2 == 0x75) {
                emitCmeqV8H(dst, dst, src);
            } else {
                emitCmeqV4S(dst, dst, src);
            }
            return finish();
        }

        // --- 0x0F 0x50: MOVMSKPS r32, xmm/m128 (0x66 -> MOVMSKPD) -------------------
        if (op2 == 0x50) {
            if (prefixF2 || prefixF3) return fail("MOVMSK* con prefijo F2/F3 no soportado");
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVMSK truncado");
            const uint8_t dstGpr = m.reg;
            uint8_t srcXmm = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                srcXmm = kXmmScratch;
            } else {
                srcXmm = xmmHost(m.rm);
            }

            emitMovImm64(dstGpr, 0);
            const uint8_t laneCount = prefix66 ? 2u : 4u;
            for (uint8_t i = 0; i < laneCount; ++i) {
                // MOVMSKPS: dwords 0..3. MOVMSKPD: dwords altos 1 y 3 de cada double.
                const uint8_t lane = prefix66 ? static_cast<uint8_t>(1u + i * 2u) : i;
                const uint8_t imm5 = static_cast<uint8_t>((lane << 3) | 4u); // UMOV .S[lane]
                emitWord(0x0E003C00u | (imm5 << 16) | (srcXmm << 5) | 16u); // UMOV W16, Vsrc.S[lane]
                emitLsrImm(16, 16, 31, false);
                if (i != 0) emitLslImm(16, 16, i, false);
                emitOrrReg(dstGpr, dstGpr, 16, false);
            }
            return finish();
        }

        // --- 0x0F 0xD7: PMOVMSKB r32, xmm -------------------------------------------
        if (op2 == 0xD7) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PMOVMSKB truncado");
            const uint8_t dstGpr = m.reg;
            const uint8_t srcXmm = xmmHost(m.rm);

            emitWord(0x4F090400u | ((srcXmm & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
            emitMovImm64(20, 0x8040201008040201ULL);
            emitWord(0x4E083C00u | ((kXmmScratch & 0x1Fu) << 5) | 16);
            emitWord(0x4E183C00u | ((kXmmScratch & 0x1Fu) << 5) | 17);
            emitAndReg(16, 16, 20, false, true);
            emitAndReg(17, 17, 20, false, true);
            emitWord(0x9E670200u | (16 << 5) | 30);
            emitWord(0x9E670200u | (17 << 5) | 31);
            emitWord(0x0E31B800u | (30 << 5) | 30);
            emitWord(0x0E31B800u | (31 << 5) | 31);
            emitWord(0x0E013C00u | (30 << 5) | 16);
            emitWord(0x0E013C00u | (31 << 5) | 17);
            emitWord(0x2A112200u | (17 << 16) | (16 << 5) | (dstGpr & 0x1Fu));
            return finish();
        }

        // --- 0x0F 0x60..0x62, 0x68..0x6A, 0x6C, 0x6D: PUNPCK* (NEON ZIP1 / ZIP2) ----
        if ((op2 >= 0x60 && op2 <= 0x62) || (op2 >= 0x68 && op2 <= 0x6A) ||
            op2 == 0x6C || op2 == 0x6D) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PUNPCK truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            const bool isHigh = (op2 >= 0x68 && op2 <= 0x6A) || op2 == 0x6D;
            uint8_t sz = 0;
            if (op2 == 0x6C || op2 == 0x6D) sz = 3;       // 2D (quadwords)
            else if (op2 == 0x62 || op2 == 0x6A) sz = 2;  // 4S (doublewords)
            else if (op2 == 0x61 || op2 == 0x69) sz = 1;  // 8H (words)
            else sz = 0;                                  // 16B (bytes)

            const uint32_t base = isHigh ? 0x4E007800u : 0x4E003800u;
            emitWord(base | ((static_cast<uint32_t>(sz) & 3u) << 22) |
                     ((src & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
            return finish();
        }

        // --- 0x0F 0x6E: MOVD/MOVQ xmm, r/m ------------------------------------------
        if (op2 == 0x6E) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVD/Q xmm, r/m truncado");
            const uint8_t dst = xmmHost(m.reg);
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (is64) {
                    emitLdrD(dst, 16, 0);
                } else {
                    emitLdrS(dst, 16, 0);
                }
            } else {
                if (is64) {
                    emitWord(0x9E670000u | ((m.rm & 0x1Fu) << 5) | (dst & 0x1Fu)); // FMOV Ddst, Xrm
                } else {
                    emitWord(0x1E270000u | ((m.rm & 0x1Fu) << 5) | (dst & 0x1Fu)); // FMOV Sdst, Wrm
                }
            }
            return finish();
        }

        // --- 0x0F 0x7E: MOVD/MOVQ r/m, xmm (0x66) o MOVQ xmm, xmm/m64 (0xF3) --------
        if (op2 == 0x7E) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVD/Q r/m, xmm truncado");
            if (prefixF3) {
                // MOVQ xmm1, xmm2/m64 (carga quadword y pone a cero la parte alta)
                const uint8_t dst = xmmHost(m.reg);
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrD(dst, 16, 0);
                } else {
                    // FMOV Dd, Dn: copia 64 bits y pone a cero la mitad alta (antes ORR .16B
                    // copiaba los 128 bits; encontrado con la prueba diferencial).
                    emitWord(0x1E604000u | ((xmmHost(m.rm) & 0x1Fu) << 5) | dst);
                }
            } else {
                // MOVD/MOVQ r/m, xmm
                const uint8_t src = xmmHost(m.reg);
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    if (is64) {
                        emitStrD(src, 16, 0);
                    } else {
                        emitStrS(src, 16, 0);
                    }
                } else {
                    if (is64) {
                        emitWord(0x9E660000u | ((src & 0x1Fu) << 5) | (m.rm & 0x1Fu)); // FMOV Xrm, Dsrc
                    } else {
                        emitWord(0x1E260000u | ((src & 0x1Fu) << 5) | (m.rm & 0x1Fu)); // FMOV Wrm, Ssrc
                    }
                }
            }
            return finish();
        }

        // --- 0x0F 0x70: PSHUFD xmm1, xmm2/m128, imm8 ---------------------------------
        if (op2 == 0x70) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PSHUFD truncado");
            uint8_t imm = 0;
            if (!reader.u8(imm)) return fail("PSHUFD imm8 truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            if (dst == src) {
                emitOrrV(kXmmScratch, src, src);
                src = kXmmScratch;
            }
            if (prefixF2 || prefixF3) {
                // F2: PSHUFLW (baraja las 4 palabras bajas, copia la mitad alta); F3: PSHUFHW (al
                // revés). Antes se trataban como PSHUFD: el strchr SSE2 de la CRT de Mono no
                // encontraba el carácter en los bytes 2-3 de cada grupo de 4 (GC incremental roto).
                const uint8_t base = prefixF3 ? 4 : 0;
                if (dst != src) emitOrrV(dst, src, src);
                for (uint8_t i = 0; i < 4; ++i) {
                    const uint8_t sel = (imm >> (i * 2)) & 3u;
                    emitInsElementH(dst, static_cast<uint8_t>(base + i), src, static_cast<uint8_t>(base + sel));
                }
                return finish();
            }
            for (uint8_t i = 0; i < 4; ++i) {
                const uint8_t sel = (imm >> (i * 2)) & 3u;
                emitInsElementS(dst, i, src, sel);
            }
            return finish();
        }

        // --- 0x0F 0x71, 0x72, 0x73: Shift vector / double quadword by imm8 -----------
        if (op2 == 0x71 || op2 == 0x72 || op2 == 0x73) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("vector shift truncado");
            uint8_t imm = 0;
            if (!reader.u8(imm)) return fail("vector shift imm8 truncado");
            const uint8_t dst = xmmHost(m.rm);
            const uint8_t group = static_cast<uint8_t>(m.reg & 7);

            if (op2 == 0x73) {
                if (group == 3) { // PSRLDQ dst, imm8 (shift right by bytes)
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitExtV16B(dst, dst, kXmmScratch, imm);
                    }
                } else if (group == 7) { // PSLLDQ dst, imm8 (shift left by bytes)
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitExtV16B(dst, kXmmScratch, dst, static_cast<uint8_t>(16 - imm));
                    }
                } else if (group == 2) { // PSRLQ dst, imm8 (shift 64-bit quadwords right logical)
                    if (imm >= 64) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV2D(dst, dst, imm);
                    }
                } else if (group == 6) { // PSLLQ dst, imm8 (shift 64-bit quadwords left logical)
                    if (imm >= 64) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV2D(dst, dst, imm);
                    }
                } else {
                    return fail("0F 73 grupo no soportado");
                }
            } else if (op2 == 0x72) {
                if (group == 2) { // PSRLD dst, imm8 (shift 32-bit dwords right logical)
                    if (imm >= 32) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV4S(dst, dst, imm);
                    }
                } else if (group == 4) { // PSRAD dst, imm8 (shift 32-bit dwords right arithmetic)
                    if (imm >= 32) imm = 31;
                    if (imm > 0) {
                        emitSshrV4S(dst, dst, imm);
                    }
                } else if (group == 6) { // PSLLD dst, imm8 (shift 32-bit dwords left logical)
                    if (imm >= 32) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV4S(dst, dst, imm);
                    }
                } else {
                    return fail("0F 72 grupo no soportado");
                }
            } else if (op2 == 0x71) {
                if (group == 2) { // PSRLW dst, imm8 (shift 16-bit words right logical)
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV8H(dst, dst, imm);
                    }
                } else if (group == 4) { // PSRAW dst, imm8 (shift 16-bit words right arithmetic)
                    if (imm >= 16) imm = 15;
                    if (imm > 0) {
                        emitSshrV8H(dst, dst, imm);
                    }
                } else if (group == 6) { // PSLLW dst, imm8 (shift 16-bit words left logical)
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV8H(dst, dst, imm);
                    }
                } else {
                    return fail("0F 71 grupo no soportado");
                }
            }
            return finish();
        }

        // --- 0x0F 0xC2: CMPPS xmm1, xmm2/m128, imm8 / CMPPD (0x66) ------------------
        if (op2 == 0xC2 && (prefixF2 || prefixF3)) {
            // CMPSS/CMPSD: comparación del lane 0 (máscara) y lanes altos intactos.
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CMPSS/CMPSD truncado");
            uint8_t predicate = 0;
            if (!reader.u8(predicate)) return fail("CMPSS/CMPSD imm8 truncado");
            if (predicate > 7) return fail("CMPSS/CMPSD predicate inválido");
            const bool isDouble = prefixF2;
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (isDouble) emitLdrD(kXmmScratch, 16, 0);
                else emitLdrS(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            emitCmpPsPd(kXmmScratch2, dst, src, predicate, isDouble);
            if (isDouble) emitInsElementD(dst, 0, kXmmScratch2, 0);
            else emitInsElementS(dst, 0, kXmmScratch2, 0);
            return finish();
        }
        if (op2 == 0xC2) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CMPPS/CMPPD truncado");
            uint8_t predicate = 0;
            if (!reader.u8(predicate)) return fail("CMPPS/CMPPD imm8 truncado");
            if (predicate > 7) return fail("CMPPS/CMPPD predicate inválido");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            emitCmpPsPd(dst, dst, src, predicate, prefix66);
            return finish();
        }

        // --- 0x0F 0xC6: SHUFPS xmm1, xmm2/m128, imm8 ---------------------------------
        if (op2 == 0xC6) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SHUFPS truncado");
            uint8_t imm = 0;
            if (!reader.u8(imm)) return fail("SHUFPS imm8 truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = dst;
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch2, 16, 0);
                src2 = kXmmScratch2;
            } else {
                src2 = xmmHost(m.rm);
            }
            emitOrrV(kXmmScratch, src1, src1);
            if (src2 != kXmmScratch2) {
                emitOrrV(kXmmScratch2, src2, src2);
            }
            if (prefix66) {
                // SHUFPD: qwords (bit 0 elige de src1, bit 1 de src2). Antes se hacía como SHUFPS
                // con dwords (encontrado con la prueba diferencial contra x86 real).
                emitInsElementD(dst, 0, kXmmScratch, imm & 1u);
                emitInsElementD(dst, 1, kXmmScratch2, (imm >> 1) & 1u);
                return finish();
            }
            emitInsElementS(dst, 0, kXmmScratch, (imm >> 0) & 3u);
            emitInsElementS(dst, 1, kXmmScratch, (imm >> 2) & 3u);
            emitInsElementS(dst, 2, kXmmScratch2, (imm >> 4) & 3u);
            emitInsElementS(dst, 3, kXmmScratch2, (imm >> 6) & 3u);
            return finish();
        }

        // --- SSE2 enteros 66 0F xx (encontrados por jit_scan en UnityPlayer HK/Cuphead) ----
        if (prefix66) {
            uint32_t direct = 0;  // NEON de 3 registros: Vd = Vn(dst) op Vm(src)
            switch (op2) {
                case 0xD5: direct = 0x4E609C00u; break;  // PMULLW   mul.8h
                case 0xE0: direct = 0x6E201400u; break;  // PAVGB    urhadd.16b
                case 0xE3: direct = 0x6E601400u; break;  // PAVGW    urhadd.8h
                case 0xDA: direct = 0x6E206C00u; break;  // PMINUB   umin.16b
                case 0xDE: direct = 0x6E206400u; break;  // PMAXUB   umax.16b
                case 0xEA: direct = 0x4E606C00u; break;  // PMINSW   smin.8h
                case 0xEE: direct = 0x4E606400u; break;  // PMAXSW   smax.8h
                case 0xD8: direct = 0x6E202C00u; break;  // PSUBUSB  uqsub.16b
                case 0xD9: direct = 0x6E602C00u; break;  // PSUBUSW  uqsub.8h
                case 0xDC: direct = 0x6E200C00u; break;  // PADDUSB  uqadd.16b
                case 0xDD: direct = 0x6E600C00u; break;  // PADDUSW  uqadd.8h
                case 0xE8: direct = 0x4E202C00u; break;  // PSUBSB   sqsub.16b
                case 0xE9: direct = 0x4E602C00u; break;  // PSUBSW   sqsub.8h
                case 0xEC: direct = 0x4E200C00u; break;  // PADDSB   sqadd.16b
                case 0xED: direct = 0x4E600C00u; break;  // PADDSW   sqadd.8h
                default: break;
            }
            const bool special = op2 == 0xE5 || op2 == 0xE4 || op2 == 0xF4 || op2 == 0xF5 ||
                                 op2 == 0xF6 || op2 == 0x6B || op2 == 0x67 || op2 == 0x63 || op2 == 0xD1 ||
                                 op2 == 0xD2 || op2 == 0xD3 || op2 == 0xE1 || op2 == 0xE2 ||
                                 op2 == 0xF1 || op2 == 0xF2 || op2 == 0xF3;
            if (direct != 0 || special) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SSE2 entero truncado");
                const uint32_t d = xmmHost(m.reg) & 0x1Fu;
                uint32_t s = kXmmScratch;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                } else {
                    s = xmmHost(m.rm) & 0x1Fu;
                }
                const auto rrr = [&](uint32_t op, uint32_t rd, uint32_t rn, uint32_t rm) {
                    emitWord(op | (rm << 16) | (rn << 5) | rd);
                };
                constexpr uint32_t t1 = kXmmScratch2, t2 = kXmmScratch3;
                constexpr uint32_t kOrr = 0x4EA01C00u;  // MOV Vd, Vn (ORR Vd, Vn, Vn)
                if (direct != 0) {
                    rrr(direct, d, d, s);
                } else if (op2 == 0xE5 || op2 == 0xE4) {  // PMULHW / PMULHUW: mitad alta de 16x16
                    const uint32_t lo = op2 == 0xE5 ? 0x0E60C000u : 0x2E60C000u;  // [SU]MULL .4s,.4h
                    rrr(lo, t1, d, s);
                    rrr(lo | 0x40000000u, t2, d, s);                              // [SU]MULL2
                    rrr(0x4E405800u, d, t1, t2);                                  // UZP2 .8h
                } else if (op2 == 0xF4) {  // PMULUDQ: dword 0 y 2 -> producto de 64 bits
                    emitWord(0x0EA12800u | (d << 5) | t1);  // XTN t1.2s, d.2d
                    emitWord(0x0EA12800u | (s << 5) | t2);  // XTN t2.2s, s.2d
                    rrr(0x2EA0C000u, d, t1, t2);            // UMULL d.2d, t1.2s, t2.2s
                } else if (op2 == 0xF5) {  // PMADDWD: suma de pares de productos 16x16 -> 32
                    rrr(0x0E60C000u, t1, d, s);  // SMULL  .4s
                    rrr(0x4E60C000u, t2, d, s);  // SMULL2 .4s
                    rrr(0x4EA0BC00u, d, t1, t2); // ADDP   .4s
                } else if (op2 == 0xF6) {  // PSADBW: suma de |a-b| por cada 8 bytes -> qword
                    rrr(0x6E207400u, t1, d, s);           // UABD .16b
                    emitWord(0x6E202800u | (t1 << 5) | t1);  // UADDLP .8h
                    emitWord(0x6E602800u | (t1 << 5) | t1);  // UADDLP .4s
                    emitWord(0x6EA02800u | (t1 << 5) | t1);  // UADDLP .2d
                    rrr(kOrr, d, t1, t1);
                } else if (op2 == 0x6B || op2 == 0x67 || op2 == 0x63) {  // PACKSSDW / PACKUSWB / PACKSSWB
                    const uint32_t narrow = op2 == 0x6B ? 0x0E614800u    // SQXTN  .4h
                                          : op2 == 0x67 ? 0x2E212800u    // SQXTUN .8b
                                                        : 0x0E214800u;   // SQXTN  .8b
                    emitWord(narrow | (d << 5) | t1);
                    emitWord(narrow | 0x40000000u | (s << 5) | t1);  // ...2 (mitad alta desde src)
                    rrr(kOrr, d, t1, t1);
                } else {  // PSRL/PSRA/PSLL por registro: cuenta = qword bajo de src
                    uint32_t size = 0;
                    bool right = true, arith = false;
                    switch (op2) {
                        case 0xD1: size = 1; break;
                        case 0xD2: size = 2; break;
                        case 0xD3: size = 3; break;
                        case 0xE1: size = 1; arith = true; break;
                        case 0xE2: size = 2; arith = true; break;
                        case 0xF1: size = 1; right = false; break;
                        case 0xF2: size = 2; right = false; break;
                        default:   size = 3; right = false; break;  // F3
                    }
                    emitWord(0x9E660000u | (s << 5) | 17u);  // FMOV X17, Ds
                    emitWord(0xF101023Fu);                  // CMP X17, #64
                    emitWord(0x54000049u);                  // B.LS +2
                    emitWord(0xD2800811u);                  // MOV X17, #64 (>= ancho: 0 / signo)
                    if (right) emitWord(0xCB1103F1u);       // NEG X17 (USHL/SSHL negativo = derecha)
                    emitWord(0x4E010C00u | (17u << 5) | t1); // DUP t1.16b, W17
                    rrr((arith ? 0x4E204400u : 0x6E204400u) | (size << 22), d, d, t1);  // [SU]SHL
                }
                return finish();
            }
        }

        // --- 0x0F 0xFC..0xFE, 0xD4 (PADD*) y 0xF8..0xFB (PSUB*) ---------------------
        if (op2 == 0xFC || op2 == 0xFD || op2 == 0xFE || op2 == 0xD4 ||
            op2 == 0xF8 || op2 == 0xF9 || op2 == 0xFA || op2 == 0xFB) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PADD/PSUB truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            uint8_t elemSize = 1;
            if (op2 == 0xFD || op2 == 0xF9) elemSize = 2;
            else if (op2 == 0xFE || op2 == 0xFA) elemSize = 4;
            else if (op2 == 0xD4 || op2 == 0xFB) elemSize = 8;

            const bool isSub = (op2 >= 0xF8 && op2 <= 0xFB);
            if (isSub) {
                emitSubV(dst, dst, src, elemSize);
            } else {
                emitAddV(dst, dst, src, elemSize);
            }
            return finish();
        }

        if (op2 == 0xA2) {  // CPUID
            emitCbz(0, 0); // si EAX == 0
            const size_t pos0 = wordPos() - 1;

            emitSubImm(16, 0, 1, false, false); // EAX - 1
            emitCbz(16, 0); // si EAX == 1
            const size_t pos1 = wordPos() - 1;

            emitSubImm(16, 0, 7, false, false); // EAX - 7
            emitCbz(16, 0); // si EAX == 7
            const size_t pos7 = wordPos() - 1;

            // Default: todos a 0
            emitMovImm64(0, 0);
            emitMovImm64(1, 0);
            emitMovImm64(2, 0);
            emitMovImm64(3, 0);
            const size_t skipDefault = wordPos();
            emitB(0);

            // Leaf 0:
            const size_t leaf0 = wordPos();
            patchImm19(pos0, leaf0);
            emitMovImm64(0, 0x16);
            emitMovImm64(3, 0x756e6547);
            emitMovImm64(1, 0x6c65746e);
            emitMovImm64(2, 0x49656e69);
            const size_t skip0 = wordPos();
            emitB(0);

            // Leaf 1:
            const size_t leaf1 = wordPos();
            patchImm19(pos1, leaf1);
            emitMovImm64(0, 0x000806EC);
            emitMovImm64(3, 0x00080800);
            // ECX: SSE3, SSSE3, CX16, SSE4.1, SSE4.2, POPCNT (verificados en la suite). Sin AVX,
            // FMA, F16C, OSXSAVE/XSAVE, AES, PCLMUL, MOVBE ni RDRAND: los juegos eligen así los
            // caminos SSE (antes 0x7FFAFBBF anunciaba AVX y se ejecutaba VEX menos probado).
            emitMovImm64(1, 0x01BAEBBD);
            emitMovImm64(2, 0xBF8BFBFF);
            const size_t skip1 = wordPos();
            emitB(0);

            // Leaf 7:
            const size_t leaf7 = wordPos();
            patchImm19(pos7, leaf7);
            emitMovImm64(0, 0);
            emitMovImm64(3, 0x00000200);  // solo ERMS (sin AVX2/BMI1/HLE/FSGSBASE)
            emitMovImm64(1, 0);
            emitMovImm64(2, 0);

            const size_t done = wordPos();
            patchImm26(skipDefault, done);
            patchImm26(skip0, done);
            patchImm26(skip1, done);
            return finish();
        }

        if (op2 == 0xBA) {  // BT/BTS/BTR/BTC r/m, imm8
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("0F BA ModRM truncado");
            const uint8_t group = static_cast<uint8_t>(m.reg & 7);
            uint8_t bit = 0;
            if (!readImm8(bit)) return fail("0F BA imm8 truncado");
            const uint8_t size = is64 ? 8 : 4;
            const uint8_t mask = static_cast<uint8_t>(is64 ? 63 : 31);
            const uint8_t bitIdx = bit & mask;
            emitLoadOperand(m, 16, size, nextRip());
            emitUbfx(20, 16, bitIdx, 1);
            if (group == 4) {  // BT
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
                return finish();
            }
            if (group == 5) {  // BTS
                emitMovImm64(17, 1ull << bitIdx);
                emitOrrReg(16, 16, 17, is64);
                emitStoreOperand(m, 16, size, nextRip());
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
                return finish();
            }
            if (group == 6) {  // BTR
                emitMovImm64(17, ~(1ull << bitIdx));
                emitAndReg(16, 16, 17, false, is64);
                emitStoreOperand(m, 16, size, nextRip());
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
                return finish();
            }
            if (group == 7) {  // BTC
                emitMovImm64(17, 1ull << bitIdx);
                emitEorReg(16, 16, 17, is64);
                emitStoreOperand(m, 16, size, nextRip());
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
                return finish();
            }
            return fail("0F BA grupo no soportado");
        }

        // --- 0x0F 0xA3 / 0xAB / 0xB3 / 0xBB: BT / BTS / BTR / BTC r/m, r -----------
        if (op2 == 0xA3 || op2 == 0xAB || op2 == 0xB3 || op2 == 0xBB) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("BT/BTS/BTR/BTC r/m, r truncado");
            const uint8_t size = prefix66 ? 2 : (is64 ? 8 : 4);
            emitLoadOperand(m, 16, size, nextRip());
            emitMovImm64(17, is64 ? 63 : (prefix66 ? 15 : 31));
            emitAndReg(17, m.reg, 17, false, is64);
            emitLsrv(23, 16, 17, is64);
            emitUbfx(20, 23, 0, 1);
            if (op2 == 0xAB) {  // BTS: set bit
                emitMovImm64(23, 1);
                emitLslv(23, 23, 17, is64);
                emitOrrReg(16, 16, 23, is64);
                emitStoreOperand(m, 16, size, nextRip());
            } else if (op2 == 0xB3) {  // BTR: reset bit
                emitMovImm64(23, 1);
                emitLslv(23, 23, 17, is64);
                emitMvnReg(23, 23, is64);
                emitAndReg(16, 16, 23, false, is64);
                emitStoreOperand(m, 16, size, nextRip());
            } else if (op2 == 0xBB) {  // BTC: complement bit
                emitMovImm64(23, 1);
                emitLslv(23, 23, 17, is64);
                emitEorReg(16, 16, 23, is64);
                emitStoreOperand(m, 16, size, nextRip());
            }
            emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
            return finish();
        }

        // --- 0x0F 0xA4 / 0xA5: SHLD r/m, r, imm8 / CL -------------------------------
        // --- 0x0F 0xAC / 0xAD: SHRD r/m, r, imm8 / CL -------------------------------
        if (op2 == 0xA4 || op2 == 0xA5 || op2 == 0xAC || op2 == 0xAD) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SHLD/SHRD truncado");
            const bool isShrd = (op2 == 0xAC || op2 == 0xAD);
            const bool isCl = (op2 == 0xA5 || op2 == 0xAD);
            const uint8_t size = prefix66 ? 2 : (is64 ? 8 : 4);
            const uint8_t maxBits = size * 8;
            uint8_t countImm = 0;
            if (!isCl) {
                if (!readImm8(countImm)) return fail("SHLD/SHRD imm8 truncado");
                countImm = countImm & (is64 ? 63 : 31);
            }

            emitLoadOperand(m, 16, size, nextRip());  // X16 = dst
            const uint8_t srcGpr = m.reg;             // src is always register

            // Registro de 32 bits: con cuenta 0 x86 escribe igualmente el registro (mitad alta a 0).
            if (size == 4 && !m.isMemory) emitMovReg(m.rm, m.rm, false);
            if (!isCl) {
                if (countImm == 0) {
                    return finish();
                }
                emitMovReg(23, 16, is64);

                if (isShrd) {
                    emitExtr(16, srcGpr, 16, countImm, is64);
                    emitUbfx(20, 23, countImm - 1, 1);
                } else {
                    emitExtr(16, 16, srcGpr, maxBits - countImm, is64);
                    emitUbfx(20, 23, maxBits - countImm, 1);
                }
                emitStoreOperand(m, 16, size, nextRip());
                emitSubReg(31, 16, 31, true, is64);
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
            } else {
                emitMovImm64(21, is64 ? 63 : 31);
                emitAndReg(17, 1, 21, false, false);
                emitCbz(17, 0);
                const size_t skipZero = wordPos() - 1;

                emitMovReg(23, 16, is64);
                if (isShrd) {
                    emitLsrv(16, 16, 17, is64);
                    emitMovImm64(21, maxBits);
                    emitSubReg(21, 21, 17, false, is64);
                    emitLslv(22, srcGpr, 21, is64);
                    emitOrrReg(16, 16, 22, is64);
                    emitSubImm(17, 17, 1, is64);
                    emitLsrv(20, 23, 17, is64);
                    emitUbfx(20, 20, 0, 1);
                } else {
                    emitLslv(16, 16, 17, is64);
                    emitMovImm64(21, maxBits);
                    emitSubReg(21, 21, 17, false, is64);
                    emitLsrv(22, srcGpr, 21, is64);
                    emitOrrReg(16, 16, 22, is64);
                    emitLsrv(20, 23, 21, is64);
                    emitUbfx(20, 20, 0, 1);
                }
                emitStoreOperand(m, 16, size, nextRip());
                emitSubReg(31, 16, 31, true, is64);
                emitWriteFlags(CfMode::Reg, OfMode::Zero, 16, 0);
                const size_t end = wordPos();
                patchImm19(skipZero, end);
            }
            return finish();
        }

        if (op2 == 0x01) {  // XGETBV (0F 01 D0) / RDTSCP (0F 01 F9)
            uint8_t modrm = 0;
            if (!reader.u8(modrm)) return fail("0F 01 truncado");
            if (modrm == 0xD0) {  // XGETBV: lee XCR0 -> EDX:EAX
                emitMovImm64(0, 7);  // EAX = 7 (x87, SSE, AVX)
                emitMovImm64(2, 0);  // EDX = 0
                return finish();
            }
            if (modrm == 0xF9) {  // RDTSCP: contador real + ECX = TSC_AUX (0)
                emitReadTsc();
                emitMovImm64(1, 0);
                return finish();
            }
            return fail("0F 01 suboperacion no soportada");
        }

        if (op2 == 0x77) {  // EMMS (0F 77): fin de un bloque MMX -> pila x87 vacía (todas las etiquetas libres)
            emitStrW(31, 19, static_cast<int32_t>(offsetof(CpuContext, x87Tag)));  // WZR
            return finish();
        }

        if (op2 == 0x31) {  // RDTSC (0F 31): EDX:EAX = contador
            emitReadTsc();
            return finish();
        }

        // --- SETcc r/m8 (0x0F 0x90..0x9F) -------------------------------------------
        if (op2 >= 0x90 && op2 <= 0x9F) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SETcc truncado");
            if (!m.isMemory && !check8(m.rm)) return 0;
            const uint8_t cond = static_cast<uint8_t>(op2 - 0x90);
            if (const int c = fusedArmCond(cond); c >= 0) {
                emitWord(0x1A9F07F0u | (static_cast<uint32_t>(c ^ 1) << 12));  // CSET W16, cond
            } else {
                const size_t skip = emitJccTest(static_cast<uint8_t>(cond ^ 1));
                if (skip == std::numeric_limits<size_t>::max()) return 0;
                emitMovImm64(16, 1);
                const size_t done = wordPos();
                emitB(0);
                const size_t taken = wordPos();
                patchImm19(skip, taken);
                emitMovImm64(16, 0);
                const size_t end = wordPos();
                patchImm26(done, end);
            }
            if (m.isMemory) {
                // emitEffectiveAddress usa X16/X17: el 0/1 se aparta en X20 antes de calcular la
                // dirección (antes se almacenaba el byte bajo de la DIRECCIÓN, p. ej. 0x31).
                emitMovReg(20, 16, true);
                emitEffectiveAddress(m, nextRip());
                emitStrB(20, 16, 0);
            } else {
                storeByteRm(m.rm, 16);  // setcc ah/ch/dh/bh sin REX
            }
            return finish();
        }

        // --- PINSRW xmm, r32/m16, imm8 (66 0F C4) / PEXTRW r32, xmm, imm8 (66 0F C5) -------
        // AArch64: INS Vd.H[i], Wn y UMOV Wd, Vn.H[i] (i = imm8 & 7; UMOV pone a cero el resto).
        if ((op2 == 0xC4 || op2 == 0xC5) && prefix66) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PINSRW/PEXTRW truncado");
            uint8_t imm8 = 0;
            if (!readImm8(imm8)) return fail("PINSRW/PEXTRW imm8 truncado");
            const uint32_t imm5 = ((static_cast<uint32_t>(imm8) & 7u) << 2) | 2u;  // tamaño H
            if (op2 == 0xC5) {
                if (m.isMemory) return fail("PEXTRW con operando de memoria");
                emitWord(0x0E003C00u | (imm5 << 16) | ((xmmHost(m.rm) & 0x1Fu) << 5) | (m.reg & 0x1Fu));
            } else {
                emitLoadOperand(m, 16, m.isMemory ? 2 : 4, nextRip());
                emitWord(0x4E001C00u | (imm5 << 16) | (16u << 5) | (xmmHost(m.reg) & 0x1Fu));
            }
            return finish();
        }

        // --- POPCNT (F3 [REX.W] 0F B8 /r) ---------------------------------------------
        // AArch64 no tiene popcount escalar: suma SWAR por pares/nibbles/bytes y MUL 0x0101...
        // Banderas: ZF = (fuente == 0) = (resultado == 0); CF/OF/SF a 0.
        if (op2 == 0xB8 && prefixF3) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("POPCNT truncado");
            emitLoadOperand(m, 16, is64 ? 8 : 4, nextRip());
            emitLsrImm(17, 16, 1, true);
            emitMovImm64(21, 0x5555555555555555ull);
            emitAndReg(17, 17, 21, false, true);
            emitSubReg(16, 16, 17, false, true);
            emitMovImm64(21, 0x3333333333333333ull);
            emitAndReg(17, 16, 21, false, true);
            emitLsrImm(16, 16, 2, true);
            emitAndReg(16, 16, 21, false, true);
            emitAddReg(16, 16, 17, false, true);
            emitLsrImm(17, 16, 4, true);
            emitAddReg(16, 16, 17, false, true);
            emitMovImm64(21, 0x0F0F0F0F0F0F0F0Full);
            emitAndReg(16, 16, 21, false, true);
            emitMovImm64(21, 0x0101010101010101ull);
            emitMulReg(16, 16, 21, true);
            emitLsrImm(m.reg, 16, 56, true);
            emitTstReg(m.reg, true);
            emitMovImm64(20, 0);  // CF = 0
            emitWriteFlags(CfMode::Reg, OfMode::Zero);
            return finish();
        }

        // --- MOVNTI m32/m64, r (0F C3 /r) --------------------------------------------
        // Store no temporal: en AArch64 basta un STR normal (el hint de caché no cambia la
        // semántica). Solo admite destino en memoria; la forma registro es #UD en x86.
        if (op2 == 0xC3 && !prefix66 && !prefixF2 && !prefixF3) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVNTI truncado");
            if (m.mod == 3) return fail("MOVNTI con destino registro (#UD)");
            emitStoreOperand(m, m.reg, is64 ? 8 : 4, nextRip());
            return finish();
        }

        // --- BSF (0x0F 0xBC) / TZCNT (F3 0F BC) --------------------------------------
        if (op2 == 0xBC) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("BSF/TZCNT truncado");
            const uint8_t size = is64 ? 8 : 4;
            emitLoadOperand(m, 16, size, nextRip());
            emitTstReg(16, is64);
            const size_t branchZero = wordPos();
            emitCbz(16, 0);

            // Input != 0:
            emitRbitReg(17, 16, is64);
            emitClzReg(m.reg, 17, is64);
            emitTstReg(16, is64); // ZF = 0
            if (prefixF3) {
                emitMovImm64(20, 0); // CF = 0 para TZCNT
                emitWriteFlags(CfMode::Reg, OfMode::Zero);
            } else {
                emitWriteFlags(CfMode::Preserve, OfMode::Zero);
            }
            const size_t skipDone = wordPos();
            emitB(0);

            // Input == 0:
            const size_t takenZero = wordPos();
            patchImm19(branchZero, takenZero);
            if (prefixF3) {
                emitMovImm64(m.reg, is64 ? 64 : 32);
                emitMovImm64(20, 1); // CF = 1
                emitMovImm64(21, 0);
                emitMrsNzcv(21);
                emitWriteFlags(CfMode::Reg, OfMode::Zero);
            } else {
                emitTstReg(16, is64); // ZF = 1
                emitWriteFlags(CfMode::Preserve, OfMode::Zero);
            }
            const size_t end = wordPos();
            patchImm26(skipDone, end);
            return finish();
        }

        // --- BSR (0x0F 0xBD) / LZCNT (F3 0F BD) --------------------------------------
        if (op2 == 0xBD) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("BSR/LZCNT truncado");
            const uint8_t size = is64 ? 8 : 4;
            emitLoadOperand(m, 16, size, nextRip());
            emitTstReg(16, is64);
            const size_t branchZero = wordPos();
            emitCbz(16, 0);

            // Input != 0:
            if (prefixF3) { // LZCNT
                emitClzReg(m.reg, 16, is64);
                emitMovImm64(20, 0); // CF = 0
                emitWriteFlags(CfMode::Reg, OfMode::Zero);
            } else { // BSR: dest = (bits - 1) - CLZ(src)
                emitClzReg(17, 16, is64);
                emitMovImm64(20, is64 ? 63 : 31);
                emitSubReg(m.reg, 20, 17, false, is64);
                emitTstReg(16, is64); // ZF = 0
                emitWriteFlags(CfMode::Preserve, OfMode::Zero);
            }
            const size_t skipDone = wordPos();
            emitB(0);

            // Input == 0:
            const size_t takenZero = wordPos();
            patchImm19(branchZero, takenZero);
            if (prefixF3) {
                emitMovImm64(m.reg, is64 ? 64 : 32);
                emitMovImm64(20, 1); // CF = 1
                emitWriteFlags(CfMode::Reg, OfMode::Zero);
            } else {
                emitTstReg(16, is64); // ZF = 1
                emitWriteFlags(CfMode::Preserve, OfMode::Zero);
            }
            const size_t end = wordPos();
            patchImm26(skipDone, end);
            return finish();
        }

        // --- 0x0F 0xC0 / 0x0F 0xC1: XADD r/m, r -------------------------------------
        if (op2 == 0xC0 || op2 == 0xC1) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("XADD ModRM truncado");
            const bool byte = (op2 == 0xC0);
            const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
            if (byte && !check8(m.reg)) return 0;
            if (byte && !m.isMemory && !check8(m.rm)) return 0;

            if (m.isMemory) {
                // Siempre atómico (Interlocked*): el registro fuente se actualiza tras el CAS, para
                // que un reintento parta del valor original.
                lockActive_ = true;
                emitEffectiveAddress(m, nextRip());
                if (byte) emitLdrB(16, 16, 0);
                else if (size == 2) emitLdrH(16, 16, 0);
                else if (size == 8) emitLdrX(16, 16, 0);
                else emitLdrW(16, 16, 0);

                // X16 = dst viejo (copiado en X24). temp = dst + src en X20 (preservado por emitWriteFlags).
                if (byte) {
                    loadByteRm(17, m.reg);
                    emitAddReg(20, 16, 17, true, false);
                } else if (size == 2) {
                    emitUxth(17, m.reg);
                    emitAddReg(20, 16, 17, true, false);
                } else {
                    emitAddReg(20, 16, m.reg, true, is64);
                }
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 20, 0);

                // [mem] = temp (X20) mediante CAS
                emitEffectiveAddress(m, nextRip());
                if (byte) emitStrB(20, 16, 0);
                else if (size == 2) emitStrH(20, 16, 0);
                else if (size == 8) emitStrX(20, 16, 0);
                else emitStrW(20, 16, 0);

                // src (m.reg) = dst viejo (X24 tras CAS correcto)
                if (byte) storeByteRm(m.reg, 24);
                else if (size == 2) emitBfi(m.reg, 24, 0, 16);
                else if (size == 8) emitMovReg(m.reg, 24, true);
                else emitMovReg(m.reg, 24, false);
            } else {
                if (m.rm == m.reg) {
                    if (byte) {
                        loadByteRm(16, m.rm);
                        emitAddReg(16, 16, 16, true, false);
                        storeByteRm(m.rm, 16);
                    } else if (size == 2) {
                        emitUxth(16, m.rm);
                        emitAddReg(16, 16, 16, true, false);
                        emitBfi(m.rm, 16, 0, 16);
                    } else {
                        emitAddReg(m.rm, m.rm, m.rm, true, is64);
                    }
                    emitWriteFlags(CfMode::Carry, OfMode::Overflow, m.rm, 0);
                    return finish();
                }

                if (byte) {
                    loadByteRm(16, m.rm);
                    loadByteRm(17, m.reg);
                    emitAddReg(20, 16, 17, true, false);
                } else if (size == 2) {
                    emitUxth(16, m.rm);
                    emitUxth(17, m.reg);
                    emitAddReg(20, 16, 17, true, false);
                } else {
                    emitMovReg(16, m.rm, is64);
                    emitMovReg(17, m.reg, is64);
                    emitAddReg(20, 16, 17, true, is64);
                }
                emitWriteFlags(CfMode::Carry, OfMode::Overflow, 20, 0);

                // src (m.reg) = dst viejo (X16)
                if (byte) storeByteRm(m.reg, 16);
                else if (size == 2) emitBfi(m.reg, 16, 0, 16);
                else if (size == 8) emitMovReg(m.reg, 16, true);
                else emitMovReg(m.reg, 16, false);

                // dst (m.rm) = temp (X20)
                if (byte) storeByteRm(m.rm, 20);
                else if (size == 2) emitBfi(m.rm, 20, 0, 16);
                else if (size == 8) emitMovReg(m.rm, 20, true);
                else emitMovReg(m.rm, 20, false);
            }
            return finish();
        }

        // --- 0x0F 0xB0 / 0x0F 0xB1: CMPXCHG r/m, r ----------------------------------
        if (op2 == 0xB0 || op2 == 0xB1) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CMPXCHG ModRM truncado");
            const bool byte = (op2 == 0xB0);
            const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
            if (byte && !check8(m.reg)) return 0;
            if (byte && !m.isMemory && !check8(m.rm)) return 0;

            // Cargar dst en X16 y respaldarlo en X20 (preservado por emitWriteFlags)
            if (m.isMemory) {
                lockActive_ = true;  // InterlockedCompareExchange: el almacenamiento es un CAS real
                emitEffectiveAddress(m, nextRip());
                if (byte) emitLdrB(16, 16, 0);
                else if (size == 2) emitLdrH(16, 16, 0);
                else if (size == 8) emitLdrX(16, 16, 0);
                else emitLdrW(16, 16, 0);
            } else {
                if (byte) loadByteRm(16, m.rm);
                else if (size == 2) emitUxth(16, m.rm);
                else emitMovReg(16, m.rm, is64);
            }
            emitMovReg(20, 16, is64 && !byte);

            // Comparar acumulador (reg 0) con dst (X16): acc - dst
            if (byte) {
                emitUbfx(17, 0, 0, 8);
                emitSubReg(23, 17, 16, true, false);
            } else if (size == 2) {
                emitUxth(17, 0);
                emitSubReg(23, 17, 16, true, false);
            } else {
                emitSubReg(23, 0, 16, true, is64);
            }
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 23, 0);

            // B.NE mismatch (0x5 = NE/NZ)
            const size_t branchNe = emitJccTest(0x5);
            if (branchNe == std::numeric_limits<size_t>::max()) return 0;

            // Match (acc == dst): dst = src (m.reg)
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (byte) emitStrB(m.reg, 16, 0);
                else if (size == 2) emitStrH(m.reg, 16, 0);
                else if (size == 8) emitStrX(m.reg, 16, 0);
                else emitStrW(m.reg, 16, 0);
            } else {
                if (byte) storeByteRm(m.rm, m.reg);
                else if (size == 2) emitBfi(m.rm, m.reg, 0, 16);
                else emitMovReg(m.rm, m.reg, is64);
            }
            const size_t branchDone = wordPos();
            emitB(0);

            // Mismatch (acc != dst): acc (reg 0) = dst viejo (X20)
            patchImm19(branchNe, wordPos());
            if (byte) emitBfiByte(0, 20);
            else if (size == 2) emitBfi(0, 20, 0, 16);
            else if (size == 8) emitMovReg(0, 20, true);
            else emitMovReg(0, 20, false);

            patchImm26(branchDone, wordPos());
            return finish();
        }

        // --- 0x0F 0xC7: Group 9 (CMPXCHG8B / CMPXCHG16B / RDRAND / RDSEED) ----------
        if (op2 == 0xC7) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("0F C7 ModRM truncado");
            const uint8_t group = static_cast<uint8_t>(m.reg & 7);
            if (group == 1) { // CMPXCHG16B (64-bit) o CMPXCHG8B (32-bit)
                if (!m.isMemory) return fail("CMPXCHG16B/8B requiere operando en memoria");
                // Atómico de verdad con CASPAL (LSE): antes era carga/compara/guarda por separado y
                // las pilas/colas lock-free (SList, colas de jobs de Unity, Mono) se corrompían
                // entre hilos. Pares: esperado X16:X17 (RAX:RDX), nuevo X20:X21 (RBX:RCX),
                // dirección X24. Tras CASPAL, X16:X17 = valor previo de memoria.
                emitEffectiveAddress(m, nextRip());
                emitMovReg(24, 16, true);
                emitMovReg(16, 0, is64);  // RAX / EAX
                emitMovReg(17, 2, is64);  // RDX / EDX
                emitMovReg(20, 3, is64);  // RBX / EBX
                emitMovReg(21, 1, is64);  // RCX / ECX
                // CASPAL Xs(16), Xs+1, Xt(20), Xt+1, [X24]  (64 bits: 0x4860FC00, 32 bits: 0x0860FC00)
                emitWord((is64 ? 0x4860FC00u : 0x0860FC00u) | (16u << 16) | (24u << 5) | 20u);
                // ZF = (previo == esperado); RDX:RAX = previo (si coincide ya eran iguales).
                emitEorReg(23, 16, 0, is64);
                emitEorReg(21, 17, 2, is64);
                emitOrrReg(23, 23, 21, is64);
                emitMovReg(0, 16, is64);
                emitMovReg(2, 17, is64);
                emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
                emitMovImm64(20, ~(1ull << 6));
                emitAndReg(21, 21, 20, false, true);  // ZF = 0
                const size_t branchNe = wordPos();
                emitCbnz(23, 0);
                emitMovImm64(20, (1ull << 6));
                emitOrrReg(21, 21, 20, true);  // ZF = 1
                patchImm19(branchNe, wordPos());
                emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
                return finish();
            }
            if (group == 6 || group == 7) { // RDRAND (6) / RDSEED (7)
                if (m.isMemory) return fail("RDRAND/RDSEED requiere operando de registro");
                emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
                emitMovImm64(17, 0x63C636AC39A695ULL);
                emitMulReg(16, 16, 17, true);
                if (is64) {
                    emitMovReg(m.rm, 16, true);
                } else if (prefix66) {
                    emitBfi(m.rm, 16, 0, 16);
                } else {
                    emitMovReg(m.rm, 16, false);
                }
                emitMovImm64(20, 1);
                emitWriteFlags(CfMode::Reg, OfMode::Zero);
                return finish();
            }
            return fail("0F C7 grupo no soportado");
        }

        // --- 0x0F 0xC8..0xCF: BSWAP r16 / r32 / r64 ---------------------------
        if (op2 >= 0xC8 && op2 <= 0xCF) {
            const uint8_t reg = static_cast<uint8_t>((op2 & 7) + (rexB ? 8 : 0));
            if (prefix66) {
                emitRev16(reg, reg, false);
            } else {
                emitRev(reg, reg, is64);
            }
            return finish();
        }

        // --- 0x0F 0x3A: 3-byte opcode map (SSE4.1 / SSE4.2 / AES) --------------------
        if (op2 == 0x3A) {
            uint8_t op3 = 0;
            if (!reader.u8(op3)) return fail("0F 3A truncado");

            if (op3 == 0x63) {  // PCMPISTRI xmm1, xmm2/m128, imm8
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PCMPISTRI truncado");
                uint8_t imm8 = 0;
                if (!readImm8(imm8)) return fail("PCMPISTRI imm8 truncado");

                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitMovReg(20, 16, true);
                } else {
                    emitMovImm64(20, 0);
                }

                const int32_t simdOffset = static_cast<int32_t>(offsetof(CpuContext, simd));
                for (uint8_t i = 0; i < 8; ++i) {
                    emitStrQ(i, 19, simdOffset + static_cast<int32_t>(i * 16));
                    emitStrQ(static_cast<uint8_t>(i + 16), 19,
                             simdOffset + static_cast<int32_t>((i + 8) * 16));
                }
                for (uint8_t i = 0; i < 8; ++i) {
                    emitStpX(static_cast<uint8_t>(i * 2), static_cast<uint8_t>(i * 2 + 1), 19,
                             static_cast<int32_t>(i * 16));
                }

                emitMovReg(0, 19, true);
                emitMovImm64(1, m.reg);
                emitMovImm64(2, m.rm);
                emitMovImm64(3, m.isMemory ? 1 : 0);
                emitMovImm64(4, imm8);
                emitMovReg(5, 20, true);
                emitMovHostAddr(16, reinterpret_cast<uintptr_t>(&executePcmpistri), kJitRelocLib);
                emitStpPre(29, 30, 31, -2);
                emitBlr(16);
                emitLdpPost(29, 30, 31, 2);

                emitWriteRipImm(nextRip());
                emitStrX(31, 19, static_cast<int32_t>(offsetof(CpuContext, exitSite)));  // sin salida enlazable
                emitLdpPost(27, 28, 31, 2);
                emitLdpPost(25, 26, 31, 2);
                emitLdpPost(23, 24, 31, 2);
                emitLdpPost(21, 22, 31, 2);
                emitLdpPost(19, 20, 31, 2);
                emitLdpPost(29, 30, 31, 2);
                emitRet();
                isControlFlow = true;
                return finish();
            }
            // BLENDPS (0C) / BLENDPD (0D) / INSERTPS (21) / DPPS (40), SSE4.1.
            if (prefix66 && (op3 == 0x0C || op3 == 0x0D || op3 == 0x21 || op3 == 0x40)) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SSE4.1 0F3A truncado");
                uint8_t imm8 = 0;
                if (!readImm8(imm8)) return fail("SSE4.1 0F3A imm8 truncado");
                const uint8_t dst = xmmHost(m.reg);
                uint8_t src = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    if (op3 == 0x21) emitLdrS(kXmmScratch, 16, 0);
                    else emitLdrQ(kXmmScratch, 16, 0);
                    src = kXmmScratch;
                } else {
                    src = xmmHost(m.rm);
                }
                if (op3 == 0x0C) {  // BLENDPS: lane i <- src si imm[i]
                    for (uint8_t i = 0; i < 4; ++i) {
                        if (imm8 & (1u << i)) emitInsElementS(dst, i, src, i);
                    }
                } else if (op3 == 0x0D) {  // BLENDPD
                    for (uint8_t i = 0; i < 2; ++i) {
                        if (imm8 & (1u << i)) emitInsElementD(dst, i, src, i);
                    }
                } else if (op3 == 0x21) {  // INSERTPS: dst[imm5:4] = src[imm7:6] (memoria: el float leído); zmask imm3:0
                    const uint8_t from = m.isMemory ? 0 : static_cast<uint8_t>((imm8 >> 6) & 3u);
                    emitInsElementS(dst, (imm8 >> 4) & 3u, src, from);
                    if ((imm8 & 0xFu) != 0) {
                        emitWord(0x4F000400u | kXmmScratch2);  // MOVI Vz.4S, #0
                        for (uint8_t i = 0; i < 4; ++i) {
                            if (imm8 & (1u << i)) emitInsElementS(dst, i, kXmmScratch2, 0);
                        }
                    }
                } else {  // DPPS: suma de productos de los lanes imm[7:4], difundida a los lanes imm[3:0]
                    emitWord(0x6E20DC00u | ((src & 0x1Fu) << 16) | ((dst & 0x1Fu) << 5) | kXmmScratch2);  // FMUL 4S
                    emitWord(0x4F000400u | kXmmScratch3);  // máscara alta = 0
                    emitWord(0x6F07E7E0u | kXmmScratch4);  // MOVI Vx.2D, #-1 (todo unos)
                    for (uint8_t i = 0; i < 4; ++i) {
                        if (imm8 & (0x10u << i)) emitInsElementS(kXmmScratch3, i, kXmmScratch4, 0);
                    }
                    emitAndV(kXmmScratch2, kXmmScratch2, kXmmScratch3);
                    const uint32_t s2 = kXmmScratch2;
                    emitWord(0x6E20D400u | (s2 << 16) | (s2 << 5) | s2);  // FADDP 4S
                    emitWord(0x6E20D400u | (s2 << 16) | (s2 << 5) | s2);  // FADDP 4S -> suma en los 4 lanes
                    emitWord(0x4F000400u | kXmmScratch3);  // máscara baja
                    for (uint8_t i = 0; i < 4; ++i) {
                        if (imm8 & (1u << i)) emitInsElementS(kXmmScratch3, i, kXmmScratch4, 0);
                    }
                    emitAndV(dst, kXmmScratch2, kXmmScratch3);
                }
                return finish();
            }
            // ROUNDPS/ROUNDPD/ROUNDSS/ROUNDSD xmm1, xmm2/m, imm8 (66 0F 3A 08..0B, SSE4.1).
            // imm8[1:0]: 0 más cercano (par), 1 -inf, 2 +inf, 3 truncar; imm8[2]: usar MXCSR
            // (siempre al más cercano en este entorno). AArch64: FRINTN/M/P/Z.
            if (op3 >= 0x08 && op3 <= 0x0B) {
                if (!prefix66) return fail("ROUNDxx sin prefijo 66");
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("ROUNDxx truncado");
                uint8_t imm8 = 0;
                if (!readImm8(imm8)) return fail("ROUNDxx imm8 truncado");
                const uint8_t dst = xmmHost(m.reg);
                uint8_t src = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                    src = kXmmScratch;
                } else {
                    src = xmmHost(m.rm);
                }
                const uint32_t mode = (imm8 & 4u) != 0 ? 0u : (imm8 & 3u);
                const bool isDouble = op3 == 0x09 || op3 == 0x0B;
                if (op3 == 0x08 || op3 == 0x09) {
                    static constexpr uint32_t kVec[4] = {0x4E218800u, 0x4E219800u, 0x4EA18800u, 0x4EA19800u};
                    emitWord(kVec[mode] | (isDouble ? 0x00400000u : 0u) | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
                } else {
                    // Escalar: solo el elemento bajo; el resto del destino se conserva.
                    static constexpr uint32_t kRmode[4] = {0u, 2u, 1u, 3u};  // N, M(-inf), P(+inf), Z
                    const uint32_t word = 0x1E244000u | (isDouble ? 0x00400000u : 0u) | (kRmode[mode] << 15) |
                                          ((src & 0x1Fu) << 5) | 25u;
                    emitWord(word);
                    if (isDouble) emitInsElementD(dst, 0, 25, 0);
                    else emitInsElementS(dst, 0, 25, 0);
                }
                return finish();
            }
            return fail("0F 3A opcode no soportado");
        }

        if (op2 == 0x38) {
            uint8_t op3 = 0;
            if (!reader.u8(op3)) return fail("0F 38 truncado");

            // 0x29: PCMPEQQ xmm1, xmm2/m128
            if (op3 == 0x29) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PCMPEQQ truncado");
                const uint8_t dstXmm = xmmHost(m.reg);
                uint8_t srcXmm = kXmmScratch;
                if (!m.isMemory) {
                    srcXmm = xmmHost(m.rm);
                } else {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                // cmeq dstXmm.2d, dstXmm.2d, srcXmm.2d
                emitWord(0x6EE08C00u | ((srcXmm & 0x1Fu) << 16) | ((dstXmm & 0x1Fu) << 5) | (dstXmm & 0x1Fu));
                return finish();
            }

            // 0x37: PCMPGTQ xmm1, xmm2/m128
            if (op3 == 0x37) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PCMPGTQ truncado");
                const uint8_t dstXmm = xmmHost(m.reg);
                uint8_t srcXmm = kXmmScratch;
                if (!m.isMemory) {
                    srcXmm = xmmHost(m.rm);
                } else {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                // cmgt dstXmm.2d, dstXmm.2d, srcXmm.2d
                emitWord(0x4EE03400u | ((srcXmm & 0x1Fu) << 16) | ((dstXmm & 0x1Fu) << 5) | (dstXmm & 0x1Fu));
                return finish();
            }

            // 0x00: PSHUFB xmm1, xmm2/m128
            if (op3 == 0x00) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PSHUFB truncado");
                const uint8_t dstXmm = xmmHost(m.reg);
                uint8_t srcXmm = kXmmScratch;
                if (!m.isMemory) {
                    srcXmm = xmmHost(m.rm);
                } else {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                // x86 usa los 4 bits bajos del índice (bit 7 = cero); TBL usa el byte entero (>= 16
                // da cero). Índices & 0x8F antes de TBL (antes 0x10..0x7F daban cero: fallo
                // encontrado con la prueba diferencial).
                emitWord(0x4F04E5E0u | kXmmScratch2);  // MOVI v.16b, #0x8f
                emitAndV(kXmmScratch2, srcXmm, kXmmScratch2);
                // tbl vd.16b, {vn.16b}, vm.16b
                emitWord(0x4E000000u | ((kXmmScratch2 & 0x1Fu) << 16) | ((dstXmm & 0x1Fu) << 5) | (dstXmm & 0x1Fu));
                return finish();
            }

            // 66 0F 38 38..40: PMIN/PMAX enteros (SSE4.1) y PMULLD. Mapa x86 real:
            // 38 PMINSB, 39 PMINSD, 3A PMINUW, 3B PMINUD, 3C PMAXSB, 3D PMAXSD, 3E PMAXUW,
            // 3F PMAXUD, 40 PMULLD. AArch64: SMIN/UMIN/SMAX/UMAX/MUL vectoriales (Q=1).
            if (prefix66 && op3 >= 0x38 && op3 <= 0x40) {
                static constexpr uint32_t kOps[9] = {
                    0x4E206C00u,  // 38 smin .16b
                    0x4EA06C00u,  // 39 smin .4s
                    0x6E606C00u,  // 3A umin .8h
                    0x6EA06C00u,  // 3B umin .4s
                    0x4E206400u,  // 3C smax .16b
                    0x4EA06400u,  // 3D smax .4s
                    0x6E606400u,  // 3E umax .8h
                    0x6EA06400u,  // 3F umax .4s
                    0x4EA09C00u,  // 40 mul  .4s
                };
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PMIN/PMAX/PMULLD truncado");
                const uint8_t dstXmm = xmmHost(m.reg);
                uint8_t srcXmm = kXmmScratch;
                if (!m.isMemory) {
                    srcXmm = xmmHost(m.rm);
                } else {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                emitWord(kOps[op3 - 0x38] | ((srcXmm & 0x1Fu) << 16) | ((dstXmm & 0x1Fu) << 5) |
                         (dstXmm & 0x1Fu));
                return finish();
            }

            // 66 0F 38: PBLENDVB/BLENDVPS/BLENDVPD (máscara implícita XMM0), PABSB/W/D, PMULDQ,
            // PMOVSXDQ, PMOVZXBD, PMOVZXWD.
            if (prefix66 && (op3 == 0x10 || op3 == 0x14 || op3 == 0x15 || op3 == 0x1C ||
                             op3 == 0x1D || op3 == 0x1E || op3 == 0x28 || op3 == 0x25 ||
                             op3 == 0x31 || op3 == 0x33)) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("SSE4.1 0F 38 truncado");
                const uint32_t d = xmmHost(m.reg) & 0x1Fu;
                uint32_t s = kXmmScratch;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    if (op3 == 0x31) emitLdrS(kXmmScratch, 16, 0);                       // m32
                    else if (op3 == 0x25 || op3 == 0x33) emitLdrD(kXmmScratch, 16, 0);  // m64
                    else emitLdrQ(kXmmScratch, 16, 0);
                } else {
                    s = xmmHost(m.rm) & 0x1Fu;
                }
                constexpr uint32_t t1 = kXmmScratch2, t2 = kXmmScratch3;
                switch (op3) {
                    case 0x10: case 0x14: case 0x15: {
                        // Máscara = bit de signo de cada elemento de XMM0 (V0): SSHR #(ancho-1).
                        const uint32_t sshr = op3 == 0x10 ? 0x4F090400u : (op3 == 0x14 ? 0x4F210400u : 0x4F410400u);
                        emitWord(sshr | (0u << 5) | t1);
                        emitWord(0x6EA01C00u | (t1 << 16) | (s << 5) | d);  // BIT d, s, t1
                        break;
                    }
                    case 0x1C: emitWord(0x4E20B800u | (s << 5) | d); break;  // ABS .16b
                    case 0x1D: emitWord(0x4E60B800u | (s << 5) | d); break;  // ABS .8h
                    case 0x1E: emitWord(0x4EA0B800u | (s << 5) | d); break;  // ABS .4s
                    case 0x28:  // PMULDQ: dword 0 y 2 con signo -> 64 bits
                        emitWord(0x0EA12800u | (d << 5) | t1);
                        emitWord(0x0EA12800u | (s << 5) | t2);
                        emitWord(0x0EA0C000u | (t2 << 16) | (t1 << 5) | d);  // SMULL .2d
                        break;
                    case 0x25: emitWord(0x0F20A400u | (s << 5) | d); break;  // SXTL .2d, .2s
                    case 0x33: emitWord(0x2F10A400u | (s << 5) | d); break;  // UXTL .4s, .4h
                    default:   // 0x31 PMOVZXBD: bytes -> dwords
                        emitWord(0x2F08A400u | (s << 5) | d);  // UXTL .8h, .8b
                        emitWord(0x2F10A400u | (d << 5) | d);  // UXTL .4s, .4h
                        break;
                }
                return finish();
            }

            // 66 0F 38 17: PTEST xmm1, xmm2/m128 (SSE4.1). ZF = (dst AND src) == 0,
            // CF = (src AND NOT dst) == 0; AF OF PF SF = 0.
            if (prefix66 && op3 == 0x17) {
                ModRm m;
                if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("PTEST truncado");
                const uint32_t d = xmmHost(m.reg) & 0x1Fu;
                uint32_t s = kXmmScratch;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                } else {
                    s = xmmHost(m.rm) & 0x1Fu;
                }
                constexpr uint32_t t1 = kXmmScratch2, t2 = kXmmScratch3;
                emitWord(0x4E201C00u | (s << 16) | (d << 5) | t1);  // AND t1.16b, d, s
                emitWord(0x4E601C00u | (d << 16) | (s << 5) | t2);  // BIC t2.16b, s, d
                emitWord(0x4E083C00u | (t1 << 5) | 16u);            // UMOV X16, t1.D[0]
                emitWord(0x4E183C00u | (t1 << 5) | 17u);            // UMOV X17, t1.D[1]
                emitOrrReg(16, 16, 17, true);
                emitWord(0xF100021Fu);                              // CMP X16, #0
                emitWord(0x1A9F17F4u);                              // CSET W20, EQ (ZF)
                emitWord(0x4E083C00u | (t2 << 5) | 16u);            // UMOV X16, t2.D[0]
                emitWord(0x4E183C00u | (t2 << 5) | 17u);            // UMOV X17, t2.D[1]
                emitOrrReg(16, 16, 17, true);
                emitWord(0xF100021Fu);                              // CMP X16, #0
                emitWord(0x1A9F17F6u);                              // CSET W22, EQ (CF)
                emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
                emitMovImm64(23, ~0x8D5ULL);
                emitAndReg(21, 21, 23, false, true);
                emitOrrReg(21, 21, 22, true);
                emitWord(0xAA000000u | (20u << 16) | (6u << 10) | (21u << 5) | 21u);  // ORR X21, X21, X20, LSL #6
                emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
                emitMovImm64(23, 1);  // PF = 0 (byte de paridad impar)
                emitStrB(23, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
                return finish();
            }

            char hex0f38[128];
            std::snprintf(hex0f38, sizeof(hex0f38), "opcode 0F 38 %02X no soportado en RIP=0x%llx", op3,
                          static_cast<unsigned long long>(rip));
            LOGE("Oryon: %s", hex0f38);
            lastError_ = hex0f38;
            return 0;
        }

        // --- 0x0F 0xE6: CVTDQ2PD (F3) / CVTPD2DQ (F2) / CVTTPD2DQ (66) ----------------
        if (op2 == 0xE6) {
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("CVTx2x 0xE6 truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (prefixF3) {
                    emitLdrD(kXmmScratch, 16, 0); // 64-bit int (2x 32-bit dwords)
                } else {
                    emitLdrQ(kXmmScratch, 16, 0); // 128-bit (2x double)
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (prefixF3) {
                // CVTDQ2PD: 2 enteros de 32 bits -> 2 dobles de 64 bits
                // sshll Vdst.2D, Vsrc.2S, #0
                emitWord(0x0F20A400u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
                // scvtf Vdst.2D, Vdst.2D
                emitWord(0x4E61D800u | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
            } else if (prefixF2 || prefix66) {
                // CVTPD2DQ (F2: al par, FCVTNS) / CVTTPD2DQ (66: truncado, FCVTZS): 2 dobles -> 2
                // enteros de 32 bits con el "indefinido" de x86 (NaN o fuera de rango ->
                // 0x80000000) y la mitad alta a cero. Antes: siempre truncaba y saturaba
                // (encontrado con la prueba diferencial contra x86 real).
                const uint32_t s = src & 0x1Fu, d = dst & 0x1Fu;
                const uint32_t r = kXmmScratch5, n = kXmmScratch6, w = kXmmScratch4, c = kXmmScratch2, k = kXmmScratch4;
                emitWord((prefixF2 ? 0x4E61A800u : 0x4EE1B800u) | (s << 5) | r);  // FCVTNS/FCVTZS r.2D, s.2D
                emitWord(0x0EA14800u | (r << 5) | n);                               // SQXTN n.2S, r.2D
                emitWord(0x0F20A400u | (n << 5) | w);                               // SXTL w.2D, n.2S
                emitWord(0x6EE08C00u | (w << 16) | (r << 5) | c);                   // CMEQ c.2D, r, w (cabe)
                emitWord(0x4E60E400u | (s << 16) | (s << 5) | k);                   // FCMEQ k.2D, s, s (no NaN)
                emitAndV(c, c, k);
                emitWord(0x0EA12800u | (c << 5) | c);                               // XTN c.2S, c.2D
                emitWord(0x4F046400u | k);                                          // MOVI k.4S, #0x80, LSL #24
                emitWord(0x6E601C00u | (k << 16) | (n << 5) | c);                   // BSL c, n, k
                emitWord(0x1E604000u | (c << 5) | d);                               // FMOV Dd, Dc (alto = 0)
            }
            return finish();
        }

        if (op2 == 0x0B) return fail("UD2 no soportado");
        char hex0f[64];
        std::snprintf(hex0f, sizeof(hex0f), "opcode 0F %02X no soportado en RIP=0x%llx", op2,
                      static_cast<unsigned long long>(rip));
        lastError_ = hex0f;
        return 0;
    }

    // --- ALU r/m, r  y  ALU r, r/m ------------------------------------------------
    auto aluFamily = [&](uint8_t opcode) -> bool {
        if (opcode <= 0x3B) {
            const uint8_t low = opcode & 0x07;
            if (low > 3) return false;
            const uint8_t family = opcode & 0xF8;
            uint8_t opKind = 0;
            if (family == 0x00) opKind = 0;
            else if (family == 0x08) opKind = 1;
            else if (family == 0x10) opKind = 2;  // ADC
            else if (family == 0x18) opKind = 3;  // SBB
            else if (family == 0x20) opKind = 4;
            else if (family == 0x28) opKind = 5;
            else if (family == 0x30) opKind = 6;
            else if (family == 0x38) opKind = 7;
            else return false;
            const bool byte = (low == 0 || low == 2);
            const bool rmIsDst = (low == 0 || low == 1);
            const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
            ModRm m;
            if (!decodeModRm(reader, rexR, rexX, rexB, m)) return false;
            if (byte) {
                if (!m.isMemory && !check8(m.rm)) return false;
                if (!check8(m.reg)) return false;
            }
            emitAluRegReg(opKind, m, m.reg, rmIsDst, size, nextRip());
            return true;
        }
        return false;
    };

    if (op <= 0x3D && (op & 0x07) <= 3) {
        if (aluFamily(op)) return finish();
        if (lastError_.empty()) lastError_ = "familia ALU no soportada";
        return 0;
    }

    // --- Acumulador, inmediato ----------------------------------------------------
    if (op == 0x04 || op == 0x0C || op == 0x14 || op == 0x1C || op == 0x24 || op == 0x2C || op == 0x34 || op == 0x3C) {
        uint8_t imm = 0;
        if (!readImm8(imm)) return fail("ALU AL,imm8 truncado");
        uint8_t opKind = 0;
        if (op == 0x04) opKind = 0;
        else if (op == 0x0C) opKind = 1;
        else if (op == 0x14) opKind = 2;
        else if (op == 0x1C) opKind = 3;
        else if (op == 0x24) opKind = 4;
        else if (op == 0x2C) opKind = 5;
        else if (op == 0x34) opKind = 6;
        else opKind = 7;
        ModRm m;
        m.mod = 3;
        m.rm = 0;
        m.isMemory = false;
        emitAluRegImm(opKind, m, imm, 1, nextRip());
        return finish();
    }

    if (op == 0x05 || op == 0x0D || op == 0x15 || op == 0x1D || op == 0x25 || op == 0x2D || op == 0x35 || op == 0x3D) {
        uint64_t imm = 0;
        if (prefix66) {
            uint16_t imm16 = 0;
            if (!readImm16(imm16)) return fail("ALU AX,imm16 truncado");
            imm = imm16;
        } else {
            uint32_t imm32 = 0;
            if (!readImm32(imm32)) return fail("ALU EAX,imm32 truncado");
            imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(
                                            static_cast<int32_t>(imm32)))
                       : static_cast<uint64_t>(imm32);
        }
        uint8_t opKind = 0;
        if (op == 0x05) opKind = 0;
        else if (op == 0x0D) opKind = 1;
        else if (op == 0x15) opKind = 2;
        else if (op == 0x1D) opKind = 3;
        else if (op == 0x25) opKind = 4;
        else if (op == 0x2D) opKind = 5;
        else if (op == 0x35) opKind = 6;
        else opKind = 7;
        ModRm m;
        m.mod = 3;
        m.rm = 0;
        m.isMemory = false;
        emitAluRegImm(opKind, m, imm, is64 ? 8 : (prefix66 ? 2 : 4), nextRip());
        return finish();
    }

    // --- PUSH/POP de registro -----------------------------------------------------
    if (op >= 0x50 && op <= 0x57) {
        uint8_t reg = static_cast<uint8_t>((op - 0x50) + (rexB ? 8 : 0));
        if (reg == 4) {
            // PUSH ESP/RSP guarda el valor ANTERIOR a la resta (el CLR de .NET pasa así punteros a
            // estructuras en la pila: push ecx; push ecx; push esp; call).
            emitMovReg(16, 4, true);
            reg = 16;
        }
        emitSubImm(4, 4, stackSize, true, false);
        if (stackSize == 8) {
            emitStrX(reg, 4, 0);
        } else {
            emitStrW(reg, 4, 0);
        }
        return finish();
    }
    // --- PUSHAD / POPAD (60 / 61, solo 32 bits): EAX ECX EDX EBX ESP(original) EBP ESI EDI.
    // EDI queda en la dirección más baja. POPAD descarta el ESP guardado. No tocan los flags.
    if (mode32_ && (op == 0x60 || op == 0x61) && !prefix66) {
        if (op == 0x60) {
            emitMovReg(16, 4, false);           // ESP original
            emitSubImm(4, 4, 32, true, false);
            emitStrW(0, 4, 28);
            emitStrW(1, 4, 24);
            emitStrW(2, 4, 20);
            emitStrW(3, 4, 16);
            emitStrW(16, 4, 12);
            emitStrW(5, 4, 8);
            emitStrW(6, 4, 4);
            emitStrW(7, 4, 0);
        } else {
            emitLdrW(7, 4, 0);
            emitLdrW(6, 4, 4);
            emitLdrW(5, 4, 8);
            emitLdrW(3, 4, 16);
            emitLdrW(2, 4, 20);
            emitLdrW(1, 4, 24);
            emitLdrW(0, 4, 28);
            emitAddImm(4, 4, 32, true, false);
        }
        return finish();
    }

    // --- PUSHF / POPF (9C / 9D): RFLAGS en pila. Los flags viven en ctx.rflags (CF ZF SF OF AF DF)
    // con PF perezoso en pfByte (PF = paridad par del byte). CoreCLR los usa en sus stubs.
    if (op == 0x9C) {
        flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // lee todos los flags
        emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitLdrB(17, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
        // Paridad del byte: W17 ^= W17>>4, >>2, >>1; bit0 = impar.
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (4u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#4
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (2u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#2
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (1u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#1
        emitMovImm64(20, 1);
        emitAndReg(17, 17, 20, false, false);
        emitEorReg(17, 17, 20, false);  // PF = par
        emitMovImm64(20, ~static_cast<uint64_t>(0x4));
        emitAndReg(16, 16, 20, false, true);
        emitBfi(16, 17, 2, 1);  // PF en el bit 2
        emitMovImm64(20, 0x202);  // bit 1 siempre a 1 + IF
        emitOrrReg(16, 16, 20, true);
        emitSubImm(4, 4, stackSize, true, false);
        if (stackSize == 8) emitStrX(16, 4, 0);
        else emitStrW(16, 4, 0);
        return finish();
    }
    // --- XLAT (D7): AL = [rBX + AL sin signo]. msvcr100 lo usa para clasificar valores x87 (FXAM).
    if (op == 0xD7) {
        emitUbfx(16, 0, 0, 8);
        emitAddReg(16, 3, 16, false, !mode32_);
        emitLdrB(16, 16, 0);
        emitBfi(0, 16, 0, 8);
        return finish();
    }
    // --- LAHF (9F): AH = SF ZF 0 AF 0 PF 1 CF.  SAHF (9E): SF ZF AF PF CF = AH (OF y DF no cambian).
    // Los usa el código que genera el JIT de .NET Framework (comparaciones de coma flotante x87).
    if (op == 0x9F) {
        flagEvents_.push_back({FlagEventKind::Read, 0, 0});
        emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitLdrB(17, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (4u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#4
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (2u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#2
        emitWord(0x4A000000u | (1u << 22) | (17u << 16) | (1u << 10) | (17u << 5) | 17u);  // EOR W17,W17,W17,LSR#1
        emitMovImm64(20, 1);
        emitAndReg(17, 17, 20, false, false);
        emitEorReg(17, 17, 20, false);  // PF = par
        emitMovImm64(20, 0xD1);         // SF ZF AF CF
        emitAndReg(16, 16, 20, false, true);
        emitBfi(16, 17, 2, 1);          // PF en el bit 2
        emitMovImm64(20, 0x02);         // bit 1 siempre a 1
        emitOrrReg(16, 16, 20, true);
        emitBfi(0, 16, 8, 8);           // AH
        return finish();
    }
    if (op == 0x9E) {
        flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // conserva OF/DF de escrituras anteriores
        emitUbfx(16, 0, 8, 8);                                // AH
        emitUbfx(17, 16, 2, 1);
        emitMovImm64(20, 1);
        emitEorReg(17, 17, 20, false);                        // pfByte = PF ? 0 : 1
        emitStrB(17, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
        emitMovImm64(20, 0xD5);
        emitAndReg(16, 16, 20, false, true);                  // SF ZF AF PF CF de AH
        emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        emitMovImm64(20, ~static_cast<uint64_t>(0xD5));
        emitAndReg(17, 17, 20, false, true);
        emitOrrReg(17, 17, 16, true);
        emitStrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        return finish();
    }

    if (op == 0x9D) {
        flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // conserva las escrituras anteriores
        if (stackSize == 8) emitLdrX(16, 4, 0);
        else emitLdrW(16, 4, 0);
        emitAddImm(4, 4, stackSize, true, false);
        emitUbfx(17, 16, 2, 1);
        emitMovImm64(20, 1);
        emitEorReg(17, 17, 20, false);  // pfByte = PF ? 0 : 1 (paridad par = PF)
        emitStrB(17, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
        emitMovImm64(20, 0x0DD5);  // CF PF AF ZF SF DF OF
        emitAndReg(16, 16, 20, false, true);
        emitStrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        return finish();
    }
    if (op >= 0x58 && op <= 0x5F) {
        const uint8_t reg = static_cast<uint8_t>((op - 0x58) + (rexB ? 8 : 0));
        if (stackSize == 8) {
            emitLdrX(reg, 4, 0);
        } else {
            emitLdrW(reg, 4, 0);
        }
        // POP ESP/RSP: el registro toma el valor leído (el incremento no se aplica encima).
        if (reg != 4) emitAddImm(4, 4, stackSize, true, false);
        return finish();
    }

    // --- MOVSXD -------------------------------------------------------------------
    if (op == 0x63) {
        if (mode32_) return fail("ARPL no soportado en 32-bit");
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOVSXD truncado");
        if (m.isMemory) {
            emitEffectiveAddress(m, nextRip());
            emitLdrW(16, 16, 0);
        } else {
            emitMovReg(16, m.rm, false);
        }
        emitSxtw(m.reg, 16);
        return finish();
    }

    // --- PUSH inmediato -----------------------------------------------------------
    if (op == 0x68 || op == 0x6A) {
        uint64_t imm = 0;
        const uint8_t opStackSize = prefix66 ? 2 : stackSize;
        if (op == 0x68) {
            if (prefix66) {
                uint16_t v = 0;
                if (!readImm16(v)) return fail("PUSH imm16 truncado");
                imm = static_cast<uint64_t>(v);
            } else {
                uint32_t v = 0;
                if (!readImm32(v)) return fail("PUSH imm32 truncado");
                imm = mode32_ ? static_cast<uint64_t>(v)
                              : static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)));
            }
        } else {
            int8_t v = 0;
            if (!reader.s8(v)) return fail("PUSH imm8 truncado");
            imm = mode32_ ? static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(v)))
                          : static_cast<uint64_t>(static_cast<int64_t>(v));
        }
        emitMovImm64(16, imm);
        emitSubImm(4, 4, opStackSize, true, false);
        if (opStackSize == 8) {
            emitStrX(16, 4, 0);
        } else if (opStackSize == 4) {
            emitStrW(16, 4, 0);
        } else {
            emitStrH(16, 4, 0);
        }
        return finish();
    }

    // --- IMUL con inmediato -------------------------------------------------------
    if (op == 0x69 || op == 0x6B) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("IMUL imm truncado");
        uint64_t imm = 0;
        if (op == 0x69) {
            if (prefix66) {
                uint16_t v = 0;
                if (!readImm16(v)) return fail("IMUL imm16 truncado");
                imm = static_cast<uint64_t>(v);
            } else {
                uint32_t v = 0;
                if (!readImm32(v)) return fail("IMUL imm32 truncado");
                imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)))
                           : static_cast<uint64_t>(v);
            }
        } else {
            int8_t v = 0;
            if (!reader.s8(v)) return fail("IMUL imm8 truncado");
            imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(v))
                       : static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(v)));
        }

        if (m.isMemory) {
            emitEffectiveAddress(m, nextRip());
            if (is64) {
                emitLdrX(17, 16, 0);
            } else if (prefix66) {
                emitLdrH(17, 16, 0);
            } else {
                emitLdrW(17, 16, 0);
            }
        } else {
            if (is64) {
                emitMovReg(17, m.rm, true);
            } else if (prefix66) {
                emitUxth(17, m.rm);
            } else {
                emitMovReg(17, m.rm, false);
            }
        }
        emitMovImm64(16, imm);
        if (is64) {
            emitMulReg(20, 17, 16, true);
            emitSmulh(16, 17, 16);
            emitMovReg(17, 20, true);
            emitAsrImm(20, 17, 63, true);
            emitSubReg(31, 16, 20, true, true);
            emitCset(20, kCondNe);
            emitMovReg(m.reg, 17, true);
            emitTstReg(m.reg, true);
            emitWriteFlags(CfMode::Reg, OfMode::Reg, m.reg, 0);
        } else if (prefix66) {
            emitMulReg(17, 17, 16, false);
            emitAsrImm(20, 17, 15, false);
            emitSubReg(31, 20, 31, true, false);
            emitCset(20, kCondNe);
            emitBfi(m.reg, 17, 0, 16);
            emitTstReg(17, false);
            emitWriteFlags(CfMode::Reg, OfMode::Reg, 17, 0);
        } else {
            emitSmull(20, 17, 16);
            emitSxtw(17, 20);
            emitSubReg(31, 20, 17, true, true);
            emitCset(20, kCondNe);
            emitMovReg(m.reg, 17, false);
            emitTstReg(m.reg, false);
            emitWriteFlags(CfMode::Reg, OfMode::Reg, m.reg, 0);
        }
        return finish();
    }

    // --- Jcc rel8 -----------------------------------------------------------------
    if (op >= 0x70 && op <= 0x7F) {
        int8_t rel8 = 0;
        if (!reader.s8(rel8)) return fail("Jcc rel8 truncado");
        uintptr_t target = nextRip() + static_cast<uintptr_t>(static_cast<int64_t>(rel8));
        if (mode32_) target &= 0xFFFFFFFFu;
        const size_t branch = emitJccTest(static_cast<uint8_t>(op - 0x70));
        if (branch == std::numeric_limits<size_t>::max()) return 0;
        emitLinkableExit(nextRip());
        const size_t taken = wordPos();
        patchImm19(branch, taken);
        emitLinkableExit(target);
        isControlFlow = true;
        return finish();
    }

    // --- Grupo 0x80/0x81/0x82/0x83: ALU r/m, inm -----------------------------------
    if (op == 0x80 || op == 0x81 || op == 0x82 || op == 0x83) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("ALU imm ModRM truncado");
        const uint8_t group = static_cast<uint8_t>(m.reg & 7);
        const bool byte = (op == 0x80 || op == 0x82);
        uint64_t imm = 0;
        if (byte) {
            uint8_t v = 0;
            if (!readImm8(v)) return fail("ALU imm8 truncado");
            imm = v;
        } else if (op == 0x81) {
            if (prefix66) {
                uint16_t v = 0;
                if (!readImm16(v)) return fail("ALU imm16 truncado");
                imm = v;
            } else {
                uint32_t v = 0;
                if (!readImm32(v)) return fail("ALU imm32 truncado");
                imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)))
                           : static_cast<uint64_t>(v);
            }
        } else {
            int8_t v = 0;
            if (!reader.s8(v)) return fail("ALU imm8s truncado");
            imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(v))
                       : static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(v)));
        }
        if (byte) {
            if (!m.isMemory && !check8(m.rm)) return 0;
        }
        emitAluRegImm(group, m, imm, byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4)), nextRip());
        return finish();
    }

    // --- TEST r/m, r --------------------------------------------------------------
    if (op == 0x84 || op == 0x85) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("TEST truncado");
        const bool byte = op == 0x84;
        if (byte && !check8(m.reg)) return 0;
        if (byte && !m.isMemory && !check8(m.rm)) return 0;
        emitAluRegReg(8, m, m.reg, true, byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4)), nextRip());
        return finish();
    }

    // --- XCHG r/m, r --------------------------------------------------------------
    if (op == 0x86 || op == 0x87) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("XCHG truncado");
        const bool byte = op == 0x86;
        const uint8_t size = byte ? 1 : (is64 ? 8 : 4);
        if (byte && !check8(m.reg)) return 0;
        if (byte && !m.isMemory && !check8(m.rm)) return 0;
        if (m.isMemory) {
            lockActive_ = true;  // XCHG con memoria es atómico en x86 aunque no lleve LOCK
            emitEffectiveAddress(m, nextRip());
            if (byte) {
                emitLdrB(17, 16, 0);
                if (!currentRexPresent_ && m.reg >= 4 && m.reg <= 7) {
                    emitUbfx(21, m.reg - 4, 8, 8);
                    emitStrB(21, 16, 0);
                    emitBfi(m.reg - 4, 17, 8, 8);
                } else {
                    emitStrB(m.reg, 16, 0);
                    emitBfiByte(m.reg, 17);
                }
            } else if (size == 8) {
                emitLdrX(17, 16, 0);
                emitStrX(m.reg, 16, 0);
                emitMovReg(m.reg, 17, true);
            } else {
                emitLdrW(17, 16, 0);
                emitStrW(m.reg, 16, 0);
                emitMovReg(m.reg, 17, false);
            }
        } else {
            if (byte) {
                if (!currentRexPresent_ && m.reg >= 4 && m.reg <= 7) {
                    emitUbfx(16, m.reg - 4, 8, 8);
                } else {
                    emitUbfx(16, m.reg, 0, 8);
                }
                if (!currentRexPresent_ && m.rm >= 4 && m.rm <= 7) {
                    emitUbfx(17, m.rm - 4, 8, 8);
                } else {
                    emitUbfx(17, m.rm, 0, 8);
                }
                if (!currentRexPresent_ && m.reg >= 4 && m.reg <= 7) {
                    emitBfi(m.reg - 4, 17, 8, 8);
                } else {
                    emitBfiByte(m.reg, 17);
                }
                if (!currentRexPresent_ && m.rm >= 4 && m.rm <= 7) {
                    emitBfi(m.rm - 4, 16, 8, 8);
                } else {
                    emitBfiByte(m.rm, 16);
                }
            } else if (size == 8) {
                emitMovReg(16, m.reg, true);
                emitMovReg(m.reg, m.rm, true);
                emitMovReg(m.rm, 16, true);
            } else {
                emitMovReg(16, m.reg, false);
                emitMovReg(m.reg, m.rm, false);
                emitMovReg(m.rm, 16, false);
            }
        }
        return finish();
    }

    // --- MOV r/m, r  y  MOV r, r/m -----------------------------------------------
    if (op == 0x88 || op == 0x89 || op == 0x8A || op == 0x8B) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOV truncado");
        const bool byte = (op == 0x88 || op == 0x8A);
        const bool store = (op == 0x88 || op == 0x89);
        const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
        if (byte && !check8(m.reg)) return 0;
        if (byte && !m.isMemory && !check8(m.rm)) return 0;
        if (store) {
            if (byte && !currentRexPresent_ && m.reg >= 4 && m.reg <= 7) {
                emitUbfx(17, m.reg - 4, 8, 8);
                emitStoreOperand(m, 17, 1, nextRip());
            } else {
                emitStoreOperand(m, m.reg, size, nextRip());
            }
        } else {
            if (byte) {
                emitLoadOperand(m, 16, 1, nextRip());
                if (!currentRexPresent_ && m.reg >= 4 && m.reg <= 7) {
                    emitBfi(m.reg - 4, 16, 8, 8);
                } else {
                    emitBfiByte(m.reg, 16);
                }
            } else if (size == 2) {
                emitLoadOperand(m, 16, 2, nextRip());
                emitBfi(m.reg, 16, 0, 16);
            } else {
                emitLoadOperand(m, m.reg, size, nextRip());
            }
        }
        return finish();
    }

    // --- LEA ----------------------------------------------------------------------
    if (op == 0x8D) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("LEA truncado");
        if (!m.isMemory) return fail("LEA con registro no válido");
        // Directo al destino (antes EA en X16 + MOV): lea r, [base(+index<<s)(+disp)].
        // La forma W calcula los 32 bits bajos y pone a cero la mitad alta, como x86.
        if (!m.ripRelative && m.hasBase && currentSegPrefix_ == 0 && !mode32_ && !prefix66 &&
            m.disp >= -4095 && m.disp <= 4095) {
            uint8_t acc = m.base;
            if (m.hasIndex) {
                emitAddRegShift(m.reg, m.base, m.index, m.scale, is64);
                acc = m.reg;
            }
            if (m.disp > 0) emitAddImm(m.reg, acc, static_cast<uint16_t>(m.disp), is64);
            else if (m.disp < 0) emitSubImm(m.reg, acc, static_cast<uint16_t>(-m.disp), is64);
            else if (!m.hasIndex) emitMovReg(m.reg, m.base, is64);
            else if (!is64) emitMovReg(m.reg, m.reg, false);
            return finish();
        }
        emitEffectiveAddress(m, nextRip());
        emitMovReg(m.reg, 16, is64);
        return finish();
    }

    // --- MOV r/m, Sreg (8C) -----------------------------------------------------------
    // No hay segmentos reales: se devuelven los selectores fijos de Windows. CS es 0x23 en WOW64
    // (32 bits) y 0x33 en x64; DS/ES/SS/GS = 0x2B y FS = 0x53. Lo usan las rutinas que guardan el
    // contexto (__report_gsfailure, RtlCaptureContext de 32 bits).
    if (op == 0x8C) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOV Sreg truncado");
        uint16_t sel = 0x2B;
        switch (m.reg & 7) {
            case 1: sel = mode32_ ? 0x23 : 0x33; break;  // CS
            case 4: sel = 0x53; break;                     // FS
            case 0: case 2: case 3: case 5: break;         // ES, SS, DS, GS
            default: return fail("MOV con registro de segmento no válido");
        }
        emitMovImm64(16, sel);
        if (m.isMemory) {
            emitStoreOperand(m, 16, 2, nextRip());  // con memoria siempre escribe 16 bits
        } else if (prefix66) {
            emitBfi(m.rm, 16, 0, 16);
        } else {
            emitMovReg(m.rm, 16, is64);  // con registro: extendido con ceros
        }
        return finish();
    }

    // --- POP r/m ------------------------------------------------------------------
    if (op == 0x8F) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("POP truncado");
        if ((m.reg & 7) != 0) return fail("POP con grupo distinto de /0");
        if (stackSize == 8) {
            emitLdrX(16, 4, 0);
        } else {
            emitLdrW(16, 4, 0);
        }
        emitAddImm(4, 4, stackSize, true, false);
        emitStoreOperand(m, 16, stackSize, nextRip());
        return finish();
    }

    // --- XCHG eAX, r --------------------------------------------------------------
    if (op >= 0x90 && op <= 0x97) {
        const uint8_t reg = static_cast<uint8_t>((op - 0x90) + (rexB ? 8 : 0));
        if (op == 0x90 && !rexB) {
            emitNop();
            return finish();
        }
        if (is64) {
            emitMovReg(16, 0, true);
            emitMovReg(0, reg, true);
            emitMovReg(reg, 16, true);
        } else {
            emitMovReg(16, 0, false);
            emitMovReg(0, reg, false);
            emitMovReg(reg, 16, false);
        }
        return finish();
    }

    // --- CDQE / CWDE (0x98) -------------------------------------------------------
    if (op == 0x98) {
        if (is64) {
            emitSxtw(0, 0);  // CDQE: RAX = sign_extend(EAX)
        } else if (prefix66) {
            // CBW: AX = sign_extend(AL); bits 16..63 intactos (antes se trataba como CWDE).
            emitWord(0x13001C10u);         // SXTB W16, W0
            emitBfi(0, 16, 0, 16);
        } else {
            // CWDE: EAX = sign_extend(AX) y la mitad alta de RAX a cero (SXTH W, no X).
            emitWord(0x13003C00u);         // SXTH W0, W0
        }
        return finish();
    }

    // --- CDQ / CQO (0x99) ---------------------------------------------------------
    if (op == 0x99) {
        if (is64) {
            emitAsrImm(2, 0, 63, true);   // CQO: RDX = RAX >> 63
        } else {
            emitAsrImm(2, 0, 31, false);  // CDQ: EDX = EAX >> 31
        }
        return finish();
    }

    // --- WAIT / FWAIT (0x9B) ------------------------------------------------------
    if (op == 0x9B) {
        emitNop();
        return finish();
    }

    // --- MOV moffs (0xA0..0xA3) ----------------------------------------------------
    if (op >= 0xA0 && op <= 0xA3) {
        uint64_t moffs = 0;
        if (mode32_) {
            uint32_t moffs32 = 0;
            if (!readImm32(moffs32)) return fail("moffs32 truncado");
            moffs = moffs32;
        } else {
            if (!reader.u64(moffs)) return fail("moffs64 truncado");
        }
        emitMovImm64(16, moffs);
        if (currentSegPrefix_ == 0x65) {
            emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, gsBase)));
            emitAddReg(16, 16, 17, false, true);
        } else if (currentSegPrefix_ == 0x64) {
            emitLdrX(17, 19, static_cast<int32_t>(offsetof(CpuContext, fsBase)));
            emitAddReg(16, 16, 17, false, true);
        }
        if (mode32_) {
            emitMovReg(16, 16, false);
        }

        const bool isStore = (op == 0xA2 || op == 0xA3);
        const bool isByte = (op == 0xA0 || op == 0xA2);

        if (isByte) {
            if (isStore) {
                emitStrB(0, 16, 0); // AL = reg 0
            } else {
                emitLdrB(17, 16, 0);
                emitBfiByte(0, 17);
            }
        } else if (prefix66) {
            if (isStore) {
                emitStrH(0, 16, 0); // AX = reg 0
            } else {
                emitLdrH(17, 16, 0);
                emitBfi(0, 17, 0, 16);
            }
        } else if (is64) {
            if (isStore) {
                emitStrX(0, 16, 0); // RAX = reg 0
            } else {
                emitLdrX(0, 16, 0);
            }
        } else {
            if (isStore) {
                emitStrW(0, 16, 0); // EAX = reg 0
            } else {
                emitLdrW(0, 16, 0);
            }
        }
        return finish();
    }

    // --- MOVSB / MOVSW / MOVSD / MOVSQ (0xA4 / 0xA5) -------------------------------
    if (op == 0xA4 || op == 0xA5) {
        const uint8_t elemSize = (op == 0xA4) ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
        // DF=1 (STD) recorre hacia abajo: el memmove solapado de MSVC copia así desde el final.
        emitStringStep(elemSize);
        const auto copyOne = [&] {
            if (elemSize == 1) {
                emitLdrB(16, 6, 0);
                emitStrB(16, 7, 0);
            } else if (elemSize == 8) {
                emitLdrX(16, 6, 0);
                emitStrX(16, 7, 0);
            } else if (elemSize == 2) {
                emitLdrH(16, 6, 0);
                emitStrH(16, 7, 0);
            } else {
                emitLdrW(16, 6, 0);
                emitStrW(16, 7, 0);
            }
            emitAddReg(6, 6, 17, false, !mode32_);
            emitAddReg(7, 7, 17, false, !mode32_);
        };
        if (prefixF3) {
            emitCbz(1, 0);
            const size_t skipZero = wordPos() - 1;
            const size_t loopStart = wordPos();
            copyOne();
            emitSubImm(1, 1, 1, !mode32_);
            const int32_t loopDelta = static_cast<int32_t>(loopStart) - static_cast<int32_t>(wordPos());
            emitCbnz(1, loopDelta);

            const size_t end = wordPos();
            patchImm19(skipZero, end);
        } else {
            copyOne();
        }
        return finish();
    }

    // --- CMPSB / CMPSW / CMPSD / CMPSQ (0xA6 / 0xA7) -------------------------------
    if (op == 0xA6 || op == 0xA7) {
        const uint8_t elemSize = (op == 0xA6) ? 1 : (prefix66 ? 2 : (is64 ? 8 : 4));
        const bool repne = prefixF2;
        const bool repe = prefixF3;

        if (repne || repe) {
            emitCbz(1, 0);
            const size_t skipZero = wordPos() - 1;
            const size_t loopStart = wordPos();

            emitSubImm(1, 1, 1, !mode32_);

            if (elemSize == 1) {
                emitLdrB(16, 6, 0); // [rsi]
                emitLdrB(17, 7, 0); // [rdi]
                emitSubReg(24, 16, 17, true, false);
            } else if (elemSize == 2) {
                emitLdrH(16, 6, 0);
                emitLdrH(17, 7, 0);
                emitSubReg(24, 16, 17, true, false);
            } else if (elemSize == 8) {
                emitLdrX(16, 6, 0);
                emitLdrX(17, 7, 0);
                emitSubReg(24, 16, 17, true, true);
            } else {
                emitLdrW(16, 6, 0);
                emitLdrW(17, 7, 0);
                emitSubReg(24, 16, 17, true, false);
            }
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 24, 0);

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(6, 6, elemSize, !mode32_);
            emitAddImm(7, 7, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(6, 6, elemSize, !mode32_);
            emitSubImm(7, 7, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());

            emitCbz(1, 0);
            const size_t exitRcxZero = wordPos() - 1;

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 6, 1); // ZF
            if (repe) {
                emitCbnz(21, 0);
                patchImm19(wordPos() - 1, loopStart);
            } else {
                emitCbz(21, 0);
                patchImm19(wordPos() - 1, loopStart);
            }

            patchImm19(exitRcxZero, wordPos());
            patchImm19(skipZero, wordPos());
        } else {
            if (elemSize == 1) {
                emitLdrB(16, 6, 0);
                emitLdrB(17, 7, 0);
                emitSubReg(24, 16, 17, true, false);
            } else if (elemSize == 2) {
                emitLdrH(16, 6, 0);
                emitLdrH(17, 7, 0);
                emitSubReg(24, 16, 17, true, false);
            } else if (elemSize == 8) {
                emitLdrX(16, 6, 0);
                emitLdrX(17, 7, 0);
                emitSubReg(24, 16, 17, true, true);
            } else {
                emitLdrW(16, 6, 0);
                emitLdrW(17, 7, 0);
                emitSubReg(24, 16, 17, true, false);
            }
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 24, 0);

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(6, 6, elemSize, !mode32_);
            emitAddImm(7, 7, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(6, 6, elemSize, !mode32_);
            emitSubImm(7, 7, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());
        }
        return finish();
    }

    // --- STOSB / STOSW / STOSD / STOSQ (0xAA / 0xAB) -------------------------------
    if (op == 0xAA || op == 0xAB) {
        const uint8_t elemSize = (op == 0xAA) ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
        emitStringStep(elemSize);  // DF=1: hacia abajo
        const auto storeOne = [&] {
            if (elemSize == 1) {
                emitStrB(0, 7, 0);
            } else if (elemSize == 8) {
                emitStrX(0, 7, 0);
            } else if (elemSize == 2) {
                emitStrH(0, 7, 0);
            } else {
                emitStrW(0, 7, 0);
            }
            emitAddReg(7, 7, 17, false, !mode32_);
        };
        if (prefixF3) {
            emitCbz(1, 0);
            const size_t skipZero = wordPos() - 1;
            const size_t loopStart = wordPos();
            storeOne();
            emitSubImm(1, 1, 1, !mode32_);
            const int32_t loopDelta = static_cast<int32_t>(loopStart) - static_cast<int32_t>(wordPos());
            emitCbnz(1, loopDelta);

            const size_t end = wordPos();
            patchImm19(skipZero, end);
        } else {
            storeOne();
        }
        return finish();
    }

    // --- TEST eAX, inm ------------------------------------------------------------
    if (op == 0xA8 || op == 0xA9) {
        uint64_t imm = 0;
        const uint8_t size = (op == 0xA8) ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
        if (op == 0xA8) {
            uint8_t v = 0;
            if (!readImm8(v)) return fail("TEST AL,imm8 truncado");
            imm = v;
        } else if (prefix66) {
            uint16_t v = 0;
            if (!readImm16(v)) return fail("TEST AX,imm16 truncado");
            imm = v;
        } else {
            uint32_t v = 0;
            if (!readImm32(v)) return fail("TEST eAX,imm32 truncado");
            imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)))
                       : static_cast<uint64_t>(v);
        }
        ModRm m;
        m.mod = 3;
        m.rm = 0;
        m.isMemory = false;
        emitAluRegImm(8, m, imm, size, nextRip());
        return finish();
    }

    // --- SCASB / SCASW / SCASD / SCASQ (0xAE / 0xAF) --------------------------------
    if (op == 0xAE || op == 0xAF) {
        const uint8_t elemSize = (op == 0xAE) ? 1 : (prefix66 ? 2 : (is64 ? 8 : 4));
        const bool repne = prefixF2;
        const bool repe = prefixF3;

        if (repne || repe) {
            emitCbz(1, 0);
            const size_t skipZero = wordPos() - 1;
            const size_t loopStart = wordPos();

            emitSubImm(1, 1, 1, !mode32_);

            if (elemSize == 1) {
                emitLdrB(16, 7, 0);
            } else if (elemSize == 2) {
                emitLdrH(16, 7, 0);
            } else if (elemSize == 8) {
                emitLdrX(16, 7, 0);
            } else {
                emitLdrW(16, 7, 0);
            }

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(7, 7, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(7, 7, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());

            if (elemSize == 1) {
                emitUbfx(17, 0, 0, 8);
                emitSubReg(17, 17, 16, true, false);
            } else if (elemSize == 2) {
                emitUxth(17, 0);
                emitSubReg(17, 17, 16, true, false);
            } else if (elemSize == 8) {
                emitSubReg(17, 0, 16, true, true);
            } else {
                emitSubReg(17, 0, 16, true, false);
            }
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 17, 0);

            emitCbz(1, 0);
            const size_t exitRcxZero = wordPos() - 1;

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 6, 1);
            if (repne) {
                emitCbnz(21, 0);
            } else {
                emitCbz(21, 0);
            }
            const size_t exitZfMatch = wordPos() - 1;

            const int32_t loopDelta = static_cast<int32_t>(loopStart) - static_cast<int32_t>(wordPos());
            emitB(loopDelta);

            const size_t loopEnd = wordPos();
            patchImm19(skipZero, loopEnd);
            patchImm19(exitRcxZero, loopEnd);
            patchImm19(exitZfMatch, loopEnd);
        } else {
            if (elemSize == 1) {
                emitLdrB(16, 7, 0);
            } else if (elemSize == 2) {
                emitLdrH(16, 7, 0);
            } else if (elemSize == 8) {
                emitLdrX(16, 7, 0);
            } else {
                emitLdrW(16, 7, 0);
            }

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(7, 7, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(7, 7, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());

            if (elemSize == 1) {
                emitUbfx(17, 0, 0, 8);
                emitSubReg(17, 17, 16, true, false);
            } else if (elemSize == 2) {
                emitUxth(17, 0);
                emitSubReg(17, 17, 16, true, false);
            } else if (elemSize == 8) {
                emitSubReg(17, 0, 16, true, true);
            } else {
                emitSubReg(17, 0, 16, true, false);
            }
            emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 17, 0);
        }
        return finish();
    }

    // --- LODSB / LODSW / LODSD / LODSQ (0xAC / 0xAD) -------------------------------
    if (op == 0xAC || op == 0xAD) {
        const uint8_t elemSize = (op == 0xAC) ? 1 : (prefix66 ? 2 : (is64 ? 8 : 4));
        if (prefixF3) {
            emitCbz(1, 0);
            const size_t skipZero = wordPos() - 1;
            const size_t loopStart = wordPos();

            if (elemSize == 1) {
                emitLdrB(16, 6, 0);
                emitBfiByte(0, 16);
            } else if (elemSize == 2) {
                emitLdrH(16, 6, 0);
                emitBfi(0, 16, 0, 16);
            } else if (elemSize == 8) {
                emitLdrX(0, 6, 0);
            } else {
                emitLdrW(0, 6, 0);
            }

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(6, 6, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(6, 6, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());

            emitSubImm(1, 1, 1, !mode32_);
            emitCbnz(1, 0);
            patchImm19(wordPos() - 1, loopStart);

            patchImm19(skipZero, wordPos());
        } else {
            if (elemSize == 1) {
                emitLdrB(16, 6, 0);
                emitBfiByte(0, 16);
            } else if (elemSize == 2) {
                emitLdrH(16, 6, 0);
                emitBfi(0, 16, 0, 16);
            } else if (elemSize == 8) {
                emitLdrX(0, 6, 0);
            } else {
                emitLdrW(0, 6, 0);
            }

            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitUbfx(21, 21, 10, 1);
            emitCbnz(21, 0);
            const size_t branchDfDown = wordPos() - 1;

            emitAddImm(6, 6, elemSize, !mode32_);
            emitB(0);
            const size_t branchDfDone = wordPos() - 1;

            patchImm19(branchDfDown, wordPos());
            emitSubImm(6, 6, elemSize, !mode32_);

            patchImm26(branchDfDone, wordPos());
        }
        return finish();
    }

    // --- MOV r8, inm8 -------------------------------------------------------------
    if (op >= 0xB0 && op <= 0xB7) {
        const uint8_t reg = static_cast<uint8_t>((op - 0xB0) + (rexB ? 8 : 0));
        if (!check8(reg)) return 0;
        uint8_t imm = 0;
        if (!readImm8(imm)) return fail("MOV r8,imm8 truncado");
        emitMovImm64(16, imm);
        if (!currentRexPresent_ && reg >= 4 && reg <= 7) {
            emitBfi(reg - 4, 16, 8, 8); // AH, CH, DH, BH
        } else {
            emitBfiByte(reg, 16);
        }
        return finish();
    }

    // --- MOV r, inm32/inm64 -------------------------------------------------------
    if (op >= 0xB8 && op <= 0xBF) {
        const uint8_t reg = static_cast<uint8_t>((op - 0xB8) + (rexB ? 8 : 0));
        uint64_t imm = 0;
        if (is64) {
            if (!reader.u64(imm)) return fail("MOV r64,imm64 truncado");
            emitMovImm64(reg, imm);
            // Patrón de Mono "mov r11, imm64; call r11": lo aprovecha la llamada nativa directa.
            movImmReg_ = reg;
            movImmValue_ = imm;
            movImmSeq_ = instrSeq_;
        } else if (prefix66) {
            uint16_t v = 0;
            if (!readImm16(v)) return fail("MOV r16,imm16 truncado");
            emitMovImm64(16, v);
            emitBfi(reg, 16, 0, 16);
        } else {
            uint32_t v = 0;
            if (!readImm32(v)) return fail("MOV r32,imm32 truncado");
            imm = v;
            emitMovImm64(reg, imm);
        }
        return finish();
    }

    // --- Shifts & Rotates ---------------------------------------------------------
    if (op == 0xC0 || op == 0xC1 || op == 0xD0 || op == 0xD1 || op == 0xD2 || op == 0xD3) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("shift ModRM truncado");
        const uint8_t group = static_cast<uint8_t>(m.reg & 7);
        const bool byte = (op == 0xC0 || op == 0xD0 || op == 0xD2);
        // Con 66h el operando es de 16 bits (antes se ejecutaba como de 32 y rompía los
        // _byteswap_ushort / shifts de u16 de MSVC).
        const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));
        const bool useCl = (op == 0xD2 || op == 0xD3);
        uint8_t count = 1;
        if (!useCl && (op == 0xC0 || op == 0xC1)) {
            if (!readImm8(count)) return fail("shift imm8 truncado");
        }
        if (byte && !m.isMemory && !check8(m.rm)) return 0;
        if (group <= 3) {
            emitRotate(group, m, useCl, count, size, nextRip());
            return finish();
        } else {
            const uint8_t shiftKind = (group == 6) ? 4 : group;
            emitShift(shiftKind, m, useCl, count, size, nextRip());
            return finish();
        }
    }

    // --- x87 FPU Instructions (0xD8..0xDF) ----------------------------------------
    if (op >= 0xD8 && op <= 0xDF) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("x87 FPU truncado");
        const uint8_t groupExt = (m.reg & 7u);
        const uint8_t modrmByte = static_cast<uint8_t>(m.isMemory ? (groupExt << 3)
                                                                  : (0xC0u | (groupExt << 3) | (m.rm & 7u)));
        // FCMOVcc (DA/DB C0..DF) lee los flags del huésped: no soportado todavía.
        if (!m.isMemory && (op == 0xDA || op == 0xDB) && groupExt <= 3) return fail("x87 FCMOVcc no soportado");
        // FCOMI/FUCOMI(P) escriben RFLAGS: si caen al intérprete, el bloque termina tras la instrucción.
        const bool setsFlags = !m.isMemory && (op == 0xDB || op == 0xDF) && (groupExt == 5 || groupExt == 6);
        if (m.isMemory) emitEffectiveAddress(m, nextRip());  // X16 = dirección efectiva
        if (!emitX87Native(op, modrmByte, m.isMemory)) {
            emitX87Call(op, modrmByte, m.isMemory);
            if (setsFlags) {
                emitWriteRipImm(nextRip());
                emitStrX(31, 19, static_cast<int32_t>(offsetof(CpuContext, exitSite)));  // sin salida enlazable
                emitLdpPost(27, 28, 31, 2);
                emitLdpPost(25, 26, 31, 2);
                emitLdpPost(23, 24, 31, 2);
                emitLdpPost(21, 22, 31, 2);
                emitLdpPost(19, 20, 31, 2);
                emitLdpPost(29, 30, 31, 2);
                emitRet();
                isControlFlow = true;
            }
        }
        return finish();
    }

    // --- RET ----------------------------------------------------------------------
    if (op == 0xC2 || op == 0xC3) {
        uint16_t extra = 0;
        if (op == 0xC2) {
            if (!readImm16(extra)) return fail("RET imm16 truncado");
        }
        if (stackSize == 8) {
            emitLdrX(16, 4, 0);
        } else {
            emitLdrW(16, 4, 0);
        }
        emitAddImm(4, 4, stackSize, true, false);
        if (extra != 0) emitAddImm(4, 4, extra, true, false);
        if (returnStack.load(std::memory_order_relaxed)) {
            // Pila de retornos: saca ras[rasTop-- & 63]; si su RIP coincide con la dirección de
            // retorno real, RET X20 a la continuación (predicha por el núcleo). Si no (longjmp,
            // excepciones, pilas manipuladas), la búsqueda normal: el resultado es el mismo.
            const int32_t topOff = static_cast<int32_t>(offsetof(CpuContext, rasTop));
            emitLdrX(20, 19, topOff);
            emitWord(0x92401695u);  // AND X21, X20, #63
            emitSubImm(20, 20, 1, true);
            emitStrX(20, 19, topOff);
            emitAddImm(22, 19, static_cast<uint16_t>(offsetof(CpuContext, ras)), true);
            emitWord(0x8B1512D6u);  // ADD X22, X22, X21, LSL #4
            emitLdpX(17, 20, 22, 0);
            emitWord(0xEB10023Fu);  // CMP X17, X16
            const size_t miss = wordPos();
            emitWord(0x54000001u);  // B.NE búsqueda normal (se parchea)
            const size_t empty = wordPos();
            emitWord(0xB4000014u);  // CBZ X20, búsqueda normal (entrada vacía: {0, 0} con RIP 0)
            emitWord(0xD65F0280u);  // RET X20
            patchImm19(miss, wordPos());
            patchImm19(empty, wordPos());
        }
        emitWriteRipReg(16, stackSize == 8);
        isControlFlow = true;
        return finish();
    }

    // --- VEX 3-Byte Prefix (0xC4) --------------------------------------------------
    if (op == 0xC4) {
        if (mode32_) return fail("LES no soportado en 32-bit");
        uint8_t byte1 = 0, byte2 = 0;
        if (!reader.u8(byte1) || !reader.u8(byte2)) return fail("VEX 3B truncado");
        const bool vexR = (byte1 & 0x80) == 0;
        const bool vexX = (byte1 & 0x40) == 0;
        const bool vexB = (byte1 & 0x20) == 0;
        const uint8_t map = byte1 & 0x1F;

        const bool vexW = (byte2 & 0x80) != 0;
        const uint8_t vvvv = (~byte2 >> 3) & 0x0F;
        const bool is256 = (byte2 & 0x04) != 0;
        const uint8_t pp = byte2 & 0x03;
        (void)vexW;
        (void)vvvv;
        (void)pp;

        uint8_t vexOp = 0;
        if (!reader.u8(vexOp)) return fail("VEX 3B opcode truncado");

        // Map 3 (0x0F 0x3A): vinsertf128 (0x18), vextractf128 (0x19)
        if (map == 3 && (vexOp == 0x18 || vexOp == 0x19)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX insert/extract ModRM truncado");
            uint8_t imm8 = 0;
            if (!readImm8(imm8)) return fail("VEX imm8 truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            if (vexOp == 0x18) { // VINSERTF128 dst, src1, src2/m128, imm8
                if (imm8 == 0) {
                    if (m.isMemory) {
                        emitEffectiveAddress(m, nextRip());
                        emitLdrQ(dst, 16, 0);
                    } else {
                        emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1F) << 16) | ((xmmHost(m.rm) & 0x1F) << 5) | dst);
                    }
                    if (m.reg != vvvv) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(vvvv));
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    }
                } else {
                    if (dst != src1) {
                        emitWord(0x4EA01C00u | ((src1 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | dst);
                    }
                    if (m.isMemory) {
                        emitEffectiveAddress(m, nextRip());
                        emitLdrQ(kXmmScratch, 16, 0);
                    } else {
                        emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1F) << 16) | ((xmmHost(m.rm) & 0x1F) << 5) | kXmmScratch);
                    }
                    emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                }
            } else { // VEXTRACTF128 dst/m128, src1, imm8
                const uint8_t srcYmm = m.reg;
                if (imm8 == 0) {
                    if (m.isMemory) {
                        emitEffectiveAddress(m, nextRip());
                        emitStrQ(xmmHost(srcYmm), 16, 0);
                    } else {
                        emitWord(0x4EA01C00u | ((xmmHost(srcYmm) & 0x1F) << 16) | ((xmmHost(srcYmm) & 0x1F) << 5) | xmmHost(m.rm));
                    }
                } else {
                    emitLdrQ(kXmmScratch, 19, ymmHighOffset(srcYmm));
                    if (m.isMemory) {
                        emitEffectiveAddress(m, nextRip());
                        emitStrQ(kXmmScratch, 16, 0);
                    } else {
                        emitWord(0x4EA01C00u | ((kXmmScratch & 0x1F) << 16) | ((kXmmScratch & 0x1F) << 5) | xmmHost(m.rm));
                    }
                }
            }
            return finish();
        }

        // Map 2 (0x0F 0x38): VPCMPEQQ (0x29), VPCMPGTQ (0x37), VPSHUFB (0x00), VPMULLD (0x38), VPMIN/MAX (0x3A..0x3D)
        if (map == 2) {
            // VEX.128.66.0F38 FMA: 96/97, 98..9F (132), A6/A7, A8..AF (213),
            // B6/B7, B8..BF (231). W=0 -> PS/SS, W=1 -> PD/SD.
            {
                uint8_t fmaGroup = 0;
                uint8_t fmaKind = 0;
                bool fmaScalar = false;
                bool fmaValid = true;
                if (vexOp == 0x96 || vexOp == 0x97) {
                    fmaGroup = 0;
                    fmaKind = (vexOp == 0x96) ? 4u : 5u;
                } else if (vexOp == 0xA6 || vexOp == 0xA7) {
                    fmaGroup = 1;
                    fmaKind = (vexOp == 0xA6) ? 4u : 5u;
                } else if (vexOp == 0xB6 || vexOp == 0xB7) {
                    fmaGroup = 2;
                    fmaKind = (vexOp == 0xB6) ? 4u : 5u;
                } else if (vexOp >= 0x98 && vexOp <= 0x9F) {
                    fmaGroup = 0;
                    fmaKind = static_cast<uint8_t>((vexOp - 0x98) / 2);
                    fmaScalar = (vexOp & 1u) != 0;
                } else if (vexOp >= 0xA8 && vexOp <= 0xAF) {
                    fmaGroup = 1;
                    fmaKind = static_cast<uint8_t>((vexOp - 0xA8) / 2);
                    fmaScalar = (vexOp & 1u) != 0;
                } else if (vexOp >= 0xB8 && vexOp <= 0xBF) {
                    fmaGroup = 2;
                    fmaKind = static_cast<uint8_t>((vexOp - 0xB8) / 2);
                    fmaScalar = (vexOp & 1u) != 0;
                } else {
                    fmaValid = false;
                }

                if (fmaValid && pp == 1u) {
                    if (is256) return fail("VEX 3B FMA YMM no soportado");
                    ModRm m;
                    if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B FMA truncado");
                    const uint8_t aVec = xmmHost(vvvv);
                    const uint8_t dst = xmmHost(m.reg);
                    uint8_t cVec = 0;
                    if (m.isMemory) {
                        emitEffectiveAddress(m, nextRip());
                        if (fmaScalar) {
                            if (vexW) emitLdrD(kXmmScratch4, 16, 0);
                            else emitLdrS(kXmmScratch4, 16, 0);
                        } else {
                            emitLdrQ(kXmmScratch4, 16, 0);
                        }
                        cVec = kXmmScratch4;
                    } else {
                        cVec = xmmHost(m.rm);
                    }

                    auto emitFmaLane = [&](uint8_t out, uint8_t a, uint8_t b, uint8_t c,
                                           bool sub, bool neg) {
                        uint8_t n = a;
                        uint8_t mm = b;
                        uint8_t acc = c;
                        if (fmaGroup == 0) {       // 132: B*C + A
                            n = b; mm = c; acc = a;
                        } else if (fmaGroup == 1) { // 213: A*B + C
                            n = a; mm = b; acc = c;
                        } else {                    // 231: C*A + B
                            n = c; mm = a; acc = b;
                        }
                        if (vexW) emitFmaD(out, n, mm, acc, sub, neg);
                        else emitFmaS(out, n, mm, acc, sub, neg);
                    };

                    if (fmaScalar) {
                        const bool sub = (fmaKind == 1u || fmaKind == 3u);
                        const bool neg = (fmaKind == 2u || fmaKind == 3u);
                        emitFmaLane(dst, aVec, dst, cVec, sub, neg);
                        return finish();
                    }

                    // Copias estables: escribir en dst no puede alterar las fuentes.
                    emitOrrV(kXmmScratch, aVec, aVec);
                    emitOrrV(kXmmScratch2, dst, dst);
                    emitOrrV(kXmmScratch3, cVec, cVec);
                    const uint8_t lanes = vexW ? 2u : 4u;
                    for (uint8_t i = 0; i < lanes; ++i) {
                        if (vexW) {
                            emitInsElementD(kXmmScratch4, 0, kXmmScratch, i);
                            emitInsElementD(kXmmScratch5, 0, kXmmScratch2, i);
                            emitInsElementD(kXmmScratch6, 0, kXmmScratch3, i);
                        } else {
                            emitInsElementS(kXmmScratch4, 0, kXmmScratch, i);
                            emitInsElementS(kXmmScratch5, 0, kXmmScratch2, i);
                            emitInsElementS(kXmmScratch6, 0, kXmmScratch3, i);
                        }
                        bool sub = (fmaKind == 1u || fmaKind == 3u);
                        bool neg = (fmaKind == 2u || fmaKind == 3u);
                        if (fmaKind == 4u || fmaKind == 5u) {
                            const bool laneSub = (fmaKind == 4u) ? ((i & 1u) != 0u)
                                                                 : ((i & 1u) == 0u);
                            sub = laneSub;
                            neg = false;
                        }
                        emitFmaLane(kXmmScratch4, kXmmScratch4, kXmmScratch5,
                                    kXmmScratch6, sub, neg);
                        if (vexW) emitInsElementD(dst, i, kXmmScratch4, 0);
                        else emitInsElementS(dst, i, kXmmScratch4, 0);
                    }
                    return finish();
                }
            }

            if (vexOp == 0x29 || vexOp == 0x37) {
                ModRm m;
                if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B PCMPEQQ truncado");
                const uint8_t dst = xmmHost(m.reg);
                const uint8_t src1 = xmmHost(vvvv);
                uint8_t src2 = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                    if (is256) {
                        emitLdrQ(kXmmScratch2, 16, 16);
                    }
                    src2 = kXmmScratch;
                } else {
                    src2 = xmmHost(m.rm);
                }
                const uint32_t opc = (vexOp == 0x29) ? 0x6EE08C00u : 0x4EE03400u;
                emitWord(opc | ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
                if (is256 && m.isMemory) {
                    emitWord(opc | ((kXmmScratch2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (kXmmScratch2 & 0x1Fu));
                } else if (is256) {
                    emitEorV(kXmmScratch2, kXmmScratch2, kXmmScratch2);
                }
                return finish();
            }

            if (vexOp == 0x00) {
                ModRm m;
                if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B PSHUFB truncado");
                const uint8_t dst = xmmHost(m.reg);
                const uint8_t src1 = xmmHost(vvvv);
                uint8_t src2 = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                    src2 = kXmmScratch;
                } else {
                    src2 = xmmHost(m.rm);
                }
                // Índices & 0x8F (ver PSHUFB): x86 usa los 4 bits bajos.
                emitWord(0x4F04E5E0u | kXmmScratch2);  // MOVI v.16b, #0x8f
                emitAndV(kXmmScratch2, src2, kXmmScratch2);
                emitWord(0x4E000000u | ((kXmmScratch2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
                return finish();
            }

            if (vexOp == 0x38) {
                ModRm m;
                if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B PMULLD truncado");
                const uint8_t dst = xmmHost(m.reg);
                const uint8_t src1 = xmmHost(vvvv);
                uint8_t src2 = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                    src2 = kXmmScratch;
                } else {
                    src2 = xmmHost(m.rm);
                }
                emitWord(0x4EA09C00u | ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
                return finish();
            }

            if (vexOp >= 0x3A && vexOp <= 0x3D) {
                ModRm m;
                if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B MIN/MAX truncado");
                const uint8_t dst = xmmHost(m.reg);
                const uint8_t src1 = xmmHost(vvvv);
                uint8_t src2 = 0;
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrQ(kXmmScratch, 16, 0);
                    src2 = kXmmScratch;
                } else {
                    src2 = xmmHost(m.rm);
                }
                uint32_t opc = 0;
                if (vexOp == 0x3A) opc = 0x6EA06C00u;
                else if (vexOp == 0x3B) opc = 0x6EA06400u;
                else if (vexOp == 0x3C) opc = 0x4EA06C00u;
                else opc = 0x4EA06400u;
                emitWord(opc | ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
                return finish();
            }
        }

        // Map 1 (0x0F): VZEROUPPER (0x77)
        if (map == 1 && vexOp == 0x77) {
            emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
            const int32_t ymmOffset = static_cast<int32_t>(offsetof(CpuContext, ymmHigh));
            for (uint8_t i = 0; i < 16; ++i) {
                emitStrQ(kXmmScratch, 19, ymmOffset + static_cast<int32_t>(i * 16));
            }
            return finish();
        }

        // Map 1 (0x0F): VMOVDQA / VMOVDQU / VMOVNTDQ (0x6F, 0x7F, 0xE7, 0x10, 0x11, 0x28, 0x29)
        if (map == 1 && (vexOp == 0x6F || vexOp == 0x7F || vexOp == 0xE7 ||
                         vexOp == 0x10 || vexOp == 0x11 || vexOp == 0x28 || vexOp == 0x29)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B MOV ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);

            const bool isStore = (vexOp == 0x7F || vexOp == 0xE7 || vexOp == 0x11 || vexOp == 0x29);
            if (m.isMemory) emitEffectiveAddress(m, nextRip());

            if (isStore) {
                if (m.isMemory) {
                    emitStrQ(dst, 16, 0);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                        emitStrQ(kXmmScratch, 16, 16);
                    }
                } else {
                    emitWord(0x4EA01C00u | ((dst & 0x1F) << 16) | ((dst & 0x1F) << 5) | xmmHost(m.rm));
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                    }
                }
            } else {
                if (m.isMemory) {
                    emitLdrQ(dst, 16, 0);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 16, 16);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    }
                } else {
                    emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1F) << 16) | ((xmmHost(m.rm) & 0x1F) << 5) | dst);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    }
                }
            }
            return finish();
        }

        // Map 1 (0x0F): VMOVD / VMOVQ (0x6E, 0x7E)
        if (map == 1 && (vexOp == 0x6E || vexOp == 0x7E)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B VMOVD/Q truncado");
            const uint8_t xmm = xmmHost(m.reg);
            if (vexOp == 0x6E) { // VMOVD / VMOVQ xmm, r/m
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    if (vexW) {
                        emitLdrD(xmm, 16, 0);
                    } else {
                        emitLdrS(xmm, 16, 0);
                    }
                } else {
                    if (vexW) {
                        // FMOV Dxmm, Xrm
                        emitWord(0x9E670000u | ((m.rm & 0x1Fu) << 5) | (xmm & 0x1Fu));
                    } else {
                        // FMOV Sxmm, Wrm
                        emitWord(0x1E270000u | ((m.rm & 0x1Fu) << 5) | (xmm & 0x1Fu));
                    }
                }
                // Zero upper bits of vector
                emitWord(0x4E181FE0u | (xmm & 0x1Fu));
            } else { // 0x7E: VMOVD / VMOVQ r/m, xmm
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    if (vexW) {
                        emitStrD(xmm, 16, 0);
                    } else {
                        emitStrS(xmm, 16, 0);
                    }
                } else {
                    if (vexW) {
                        // FMOV Xrm, Dxmm
                        emitWord(0x9E660000u | ((xmm & 0x1Fu) << 5) | (m.rm & 0x1Fu));
                    } else {
                        // FMOV Wrm, Sxmm
                        emitWord(0x1E260000u | ((xmm & 0x1Fu) << 5) | (m.rm & 0x1Fu));
                    }
                }
            }
            return finish();
        }

        // Map 1: VANDPS/VPAND (0x54/0xDB), VANDNPS/VPANDN (0x55/0xDF), VORPS/VPOR (0x56/0xEB), VXORPS/VPXOR (0x57/0xEF)
        if (map == 1 && (vexOp == 0x54 || vexOp == 0x55 || vexOp == 0x56 || vexOp == 0x57 ||
                         vexOp == 0xDB || vexOp == 0xDF || vexOp == 0xEB || vexOp == 0xEF)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B bitwise ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (vexOp == 0x54 || vexOp == 0xDB) {
                emitAndV(dst, src1, src2);
            } else if (vexOp == 0x55 || vexOp == 0xDF) {
                emitBicV(dst, src2, src1);
            } else if (vexOp == 0x56 || vexOp == 0xEB) {
                emitOrrV(dst, src1, src2);
            } else {
                emitEorV(dst, src1, src2);
            }
            return finish();
        }

        // Map 1: VADD / VMUL / VSUB / VMIN / VDIV / VMAX (0x51..0x53, 0x58..0x5F)
        if (map == 1 && ((vexOp >= 0x51 && vexOp <= 0x53) || (vexOp >= 0x58 && vexOp <= 0x5F))) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B FP math ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (pp == 2) {
                    emitLdrS(kXmmScratch, 16, 0);
                } else if (pp == 3) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (pp == 2) { // Scalar Single (F3)
                if (vexOp == 0x51) emitFsqrtS(kXmmScratch, src2);
                else if (vexOp == 0x52) emitFrsqrteS(kXmmScratch, src2);
                else if (vexOp == 0x53) emitFrecpeS(kXmmScratch, src2);
                else if (vexOp == 0x58) emitFaddS(kXmmScratch, src1, src2);
                else if (vexOp == 0x59) emitFmulS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5C) emitFsubS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5D) emitFminS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5E) emitFdivS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5F) emitFmaxS(kXmmScratch, src1, src2);
                if (dst != src1) {
                    emitWord(0x4EA01C00u | ((src1 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | dst);
                }
                emitInsS(dst, kXmmScratch);
            } else if (pp == 3) { // Scalar Double (F2)
                if (vexOp == 0x51) emitFsqrtD(kXmmScratch, src2);
                else if (vexOp == 0x52) emitFrsqrteD(kXmmScratch, src2);
                else if (vexOp == 0x53) emitFrecpeD(kXmmScratch, src2);
                else if (vexOp == 0x58) emitFaddD(kXmmScratch, src1, src2);
                else if (vexOp == 0x59) emitFmulD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5C) emitFsubD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5D) emitFminD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5E) emitFdivD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5F) emitFmaxD(kXmmScratch, src1, src2);
                if (dst != src1) {
                    emitWord(0x4EA01C00u | ((src1 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | dst);
                }
                emitInsD(dst, kXmmScratch);
            } else if (pp == 1) { // Packed Double (66)
                if (vexOp == 0x51) emitFsqrtV2D(dst, src2);
                else if (vexOp == 0x52) emitFrsqrteV2D(dst, src2);
                else if (vexOp == 0x53) emitFrecpeV2D(dst, src2);
                else if (vexOp == 0x58) emitFaddV2D(dst, src1, src2);
                else if (vexOp == 0x59) emitFmulV2D(dst, src1, src2);
                else if (vexOp == 0x5C) emitFsubV2D(dst, src1, src2);
                else if (vexOp == 0x5D) emitFminV2D(dst, src1, src2);
                else if (vexOp == 0x5E) emitFdivV2D(dst, src1, src2);
                else if (vexOp == 0x5F) emitFmaxV2D(dst, src1, src2);
            } else { // Packed Single (none)
                if (vexOp == 0x51) emitFsqrtV(dst, src2);
                else if (vexOp == 0x52) emitFrsqrteV(dst, src2);
                else if (vexOp == 0x53) emitFrecpeV(dst, src2);
                else if (vexOp == 0x58) emitFaddV(dst, src1, src2);
                else if (vexOp == 0x59) emitFmulV(dst, src1, src2);
                else if (vexOp == 0x5C) emitFsubV(dst, src1, src2);
                else if (vexOp == 0x5D) emitFminV(dst, src1, src2);
                else if (vexOp == 0x5E) emitFdivV(dst, src1, src2);
                else if (vexOp == 0x5F) emitFmaxV(dst, src1, src2);
            }
            return finish();
        }

        // Map 1: VPCMPEQB / VPCMPEQW / VPCMPEQD (0x74 / 0x75 / 0x76)
        if (map == 1 && (vexOp == 0x74 || vexOp == 0x75 || vexOp == 0x76)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B PCMPEQ truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                if (is256) {
                    emitLdrQ(kXmmScratch2, 16, 16);
                }
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (vexOp == 0x74) {
                emitCmeqV16B(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV16B(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            } else if (vexOp == 0x75) {
                emitCmeqV8H(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV8H(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            } else {
                emitCmeqV4S(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV4S(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            }
            return finish();
        }

        // Map 1: VPMOVMSKB (0xD7)
        if (map == 1 && vexOp == 0xD7) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B PMOVMSKB truncado");
            const uint8_t dstGpr = m.reg;
            const uint8_t srcXmm = xmmHost(m.rm);

            emitWord(0x4F090400u | ((srcXmm & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
            emitMovImm64(20, 0x8040201008040201ULL);
            emitWord(0x4E083C00u | ((kXmmScratch & 0x1Fu) << 5) | 16);
            emitWord(0x4E183C00u | ((kXmmScratch & 0x1Fu) << 5) | 17);
            emitAndReg(16, 16, 20, false, true);
            emitAndReg(17, 17, 20, false, true);
            emitWord(0x9E670200u | (16 << 5) | 30);
            emitWord(0x9E670200u | (17 << 5) | 31);
            emitWord(0x0E31B800u | (30 << 5) | 30);
            emitWord(0x0E31B800u | (31 << 5) | 31);
            emitWord(0x0E013C00u | (30 << 5) | 16);
            emitWord(0x0E013C00u | (31 << 5) | 17);
            emitWord(0x2A112200u | (17 << 16) | (16 << 5) | (dstGpr & 0x1Fu));

            if (is256) {
                emitWord(0x4F090400u | ((kXmmScratch2 & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
                emitWord(0x4E083C00u | ((kXmmScratch & 0x1Fu) << 5) | 16);
                emitWord(0x4E183C00u | ((kXmmScratch & 0x1Fu) << 5) | 17);
                emitAndReg(16, 16, 20, false, true);
                emitAndReg(17, 17, 20, false, true);
                emitWord(0x9E670200u | (16 << 5) | 30);
                emitWord(0x9E670200u | (17 << 5) | 31);
                emitWord(0x0E31B800u | (30 << 5) | 30);
                emitWord(0x0E31B800u | (31 << 5) | 31);
                emitWord(0x0E013C00u | (30 << 5) | 16);
                emitWord(0x0E013C00u | (31 << 5) | 17);
                emitWord(0x2A112200u | (17 << 16) | (16 << 5) | 16);
                emitBfi(dstGpr, 16, 16, 16);
            }
            return finish();
        }

        // Map 1: VUNPCKLPS/D (0x14), VUNPCKHPS/D (0x15) y VPUNPCK* (0x60..0x62, 0x68..0x6A, 0x6C, 0x6D)
        if (map == 1 && (vexOp == 0x14 || vexOp == 0x15 ||
                         (vexOp >= 0x60 && vexOp <= 0x62) || (vexOp >= 0x68 && vexOp <= 0x6A) ||
                         vexOp == 0x6C || vexOp == 0x6D)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B UNPCK truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            const bool isHigh = (vexOp == 0x15 || vexOp == 0x68 || vexOp == 0x69 || vexOp == 0x6A || vexOp == 0x6D);
            uint8_t sz = 2; // 4S (single)
            if (vexOp == 0x6C || vexOp == 0x6D || ((vexOp == 0x14 || vexOp == 0x15) && pp == 1)) {
                sz = 3; // 2D (double/qword)
            } else if (vexOp == 0x62 || vexOp == 0x6A || ((vexOp == 0x14 || vexOp == 0x15) && pp == 0)) {
                sz = 2; // 4S (dword)
            } else if (vexOp == 0x61 || vexOp == 0x69) {
                sz = 1; // 8H (word)
            } else if (vexOp == 0x60 || vexOp == 0x68) {
                sz = 0; // 16B (byte)
            }
            const uint32_t base = isHigh ? 0x4E007800u : 0x4E003800u;
            emitWord(base | ((static_cast<uint32_t>(sz) & 3u) << 22) |
                     ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
            return finish();
        }

        // VEX.128.F3.0F.W0/W1 2A /r: VCVTSI2SS xmm1, xmm2, r/m32 (o r/m64 con W=1).
        if (map == 1 && vexOp == 0x2A && pp == 2u) {
            ModRm m;
            if (!decodeModRm(reader, vexR, vexX, vexB, m)) return fail("VEX 3B CVTSI2SS truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t srcGpr = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (vexW) {
                    emitLdrX(16, 16, 0);
                } else {
                    emitLdrW(16, 16, 0);
                }
                srcGpr = 16;
            } else {
                srcGpr = m.rm;
            }
            if (dst != src1) emitOrrV(dst, src1, src1); // conserva los lanes altos de vvvv
            emitScvtfS(kXmmScratch, srcGpr, vexW);
            emitInsS(dst, kXmmScratch);                // INS Vdst.S[0], Vscratch.S[0]
            return finish();
        }

        return fail("VEX 3B opcode no soportado");
    }

    // --- VEX 2-Byte Prefix (0xC5) --------------------------------------------------
    if (op == 0xC5) {
        if (mode32_) return fail("LDS no soportado en 32-bit");
        uint8_t byte1 = 0;
        if (!reader.u8(byte1)) return fail("VEX 2B truncado");
        const bool vexR = (byte1 & 0x80) == 0;
        const uint8_t vvvv = (~byte1 >> 3) & 0x0F;
        const bool is256 = (byte1 & 0x04) != 0;
        const uint8_t pp = byte1 & 0x03;
        (void)vvvv;
        (void)pp;

        uint8_t vexOp = 0;
        if (!reader.u8(vexOp)) return fail("VEX opcode truncado");

        if (vexOp == 0x77) {
            emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
            const int32_t ymmOffset = static_cast<int32_t>(offsetof(CpuContext, ymmHigh));
            for (uint8_t i = 0; i < 16; ++i) {
                emitStrQ(kXmmScratch, 19, ymmOffset + static_cast<int32_t>(i * 16));
            }
            return finish();
        }

        if (vexOp == 0x6F || vexOp == 0x7F || vexOp == 0xE7 ||
            vexOp == 0x10 || vexOp == 0x11 || vexOp == 0x28 || vexOp == 0x29) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B MOV ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);

            const bool isStore = (vexOp == 0x7F || vexOp == 0xE7 || vexOp == 0x11 || vexOp == 0x29);
            if (m.isMemory) emitEffectiveAddress(m, nextRip());

            if (isStore) {
                if (m.isMemory) {
                    emitStrQ(dst, 16, 0);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                        emitStrQ(kXmmScratch, 16, 16);
                    }
                } else {
                    emitWord(0x4EA01C00u | ((dst & 0x1F) << 16) | ((dst & 0x1F) << 5) | xmmHost(m.rm));
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                    }
                }
            } else {
                if (m.isMemory) {
                    emitLdrQ(dst, 16, 0);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 16, 16);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    }
                } else {
                    emitWord(0x4EA01C00u | ((xmmHost(m.rm) & 0x1F) << 16) | ((xmmHost(m.rm) & 0x1F) << 5) | dst);
                    if (is256) {
                        emitLdrQ(kXmmScratch, 19, ymmHighOffset(m.rm));
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    } else {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitStrQ(kXmmScratch, 19, ymmHighOffset(m.reg));
                    }
                }
            }
            return finish();
        }

        // VMOVD (0x6E, 0x7E)
        if (vexOp == 0x6E || vexOp == 0x7E) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B VMOVD truncado");
            const uint8_t xmm = xmmHost(m.reg);
            if (vexOp == 0x6E) { // VMOVD xmm, r/m32
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitLdrS(xmm, 16, 0);
                } else {
                    emitWord(0x1E270000u | ((m.rm & 0x1Fu) << 5) | (xmm & 0x1Fu));
                }
                emitWord(0x4E181FE0u | (xmm & 0x1Fu));
            } else { // 0x7E: VMOVD r/m32, xmm
                if (m.isMemory) {
                    emitEffectiveAddress(m, nextRip());
                    emitStrS(xmm, 16, 0);
                } else {
                    emitWord(0x1E260000u | ((xmm & 0x1Fu) << 5) | (m.rm & 0x1Fu));
                }
            }
            return finish();
        }

        // VANDPS/VPAND (0x54/0xDB), VANDNPS/VPANDN (0x55/0xDF), VORPS/VPOR (0x56/0xEB), VXORPS/VPXOR (0x57/0xEF)
        if (vexOp == 0x54 || vexOp == 0x55 || vexOp == 0x56 || vexOp == 0x57 ||
            vexOp == 0xDB || vexOp == 0xDF || vexOp == 0xEB || vexOp == 0xEF) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B bitwise ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (vexOp == 0x54 || vexOp == 0xDB) {
                emitAndV(dst, src1, src2);
            } else if (vexOp == 0x55 || vexOp == 0xDF) {
                emitBicV(dst, src2, src1);
            } else if (vexOp == 0x56 || vexOp == 0xEB) {
                emitOrrV(dst, src1, src2);
            } else {
                emitEorV(dst, src1, src2);
            }
            return finish();
        }

        // VADD / VMUL / VSUB / VMIN / VDIV / VMAX (0x51..0x53, 0x58..0x5F)
        if ((vexOp >= 0x51 && vexOp <= 0x53) || (vexOp >= 0x58 && vexOp <= 0x5F)) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B FP math ModRM truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (pp == 2) {
                    emitLdrS(kXmmScratch, 16, 0);
                } else if (pp == 3) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrQ(kXmmScratch, 16, 0);
                }
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (pp == 2) { // Scalar Single (F3)
                if (vexOp == 0x51) emitFsqrtS(kXmmScratch, src2);
                else if (vexOp == 0x52) emitFrsqrteS(kXmmScratch, src2);
                else if (vexOp == 0x53) emitFrecpeS(kXmmScratch, src2);
                else if (vexOp == 0x58) emitFaddS(kXmmScratch, src1, src2);
                else if (vexOp == 0x59) emitFmulS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5C) emitFsubS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5D) emitFminS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5E) emitFdivS(kXmmScratch, src1, src2);
                else if (vexOp == 0x5F) emitFmaxS(kXmmScratch, src1, src2);
                if (dst != src1) {
                    emitWord(0x4EA01C00u | ((src1 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | dst);
                }
                emitInsS(dst, kXmmScratch);
            } else if (pp == 3) { // Scalar Double (F2)
                if (vexOp == 0x51) emitFsqrtD(kXmmScratch, src2);
                else if (vexOp == 0x52) emitFrsqrteD(kXmmScratch, src2);
                else if (vexOp == 0x53) emitFrecpeD(kXmmScratch, src2);
                else if (vexOp == 0x58) emitFaddD(kXmmScratch, src1, src2);
                else if (vexOp == 0x59) emitFmulD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5C) emitFsubD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5D) emitFminD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5E) emitFdivD(kXmmScratch, src1, src2);
                else if (vexOp == 0x5F) emitFmaxD(kXmmScratch, src1, src2);
                if (dst != src1) {
                    emitWord(0x4EA01C00u | ((src1 & 0x1F) << 16) | ((src1 & 0x1F) << 5) | dst);
                }
                emitInsD(dst, kXmmScratch);
            } else if (pp == 1) { // Packed Double (66)
                if (vexOp == 0x51) emitFsqrtV2D(dst, src2);
                else if (vexOp == 0x52) emitFrsqrteV2D(dst, src2);
                else if (vexOp == 0x53) emitFrecpeV2D(dst, src2);
                else if (vexOp == 0x58) emitFaddV2D(dst, src1, src2);
                else if (vexOp == 0x59) emitFmulV2D(dst, src1, src2);
                else if (vexOp == 0x5C) emitFsubV2D(dst, src1, src2);
                else if (vexOp == 0x5D) emitFminV2D(dst, src1, src2);
                else if (vexOp == 0x5E) emitFdivV2D(dst, src1, src2);
                else if (vexOp == 0x5F) emitFmaxV2D(dst, src1, src2);
            } else { // Packed Single (none)
                if (vexOp == 0x51) emitFsqrtV(dst, src2);
                else if (vexOp == 0x52) emitFrsqrteV(dst, src2);
                else if (vexOp == 0x53) emitFrecpeV(dst, src2);
                else if (vexOp == 0x58) emitFaddV(dst, src1, src2);
                else if (vexOp == 0x59) emitFmulV(dst, src1, src2);
                else if (vexOp == 0x5C) emitFsubV(dst, src1, src2);
                else if (vexOp == 0x5D) emitFminV(dst, src1, src2);
                else if (vexOp == 0x5E) emitFdivV(dst, src1, src2);
                else if (vexOp == 0x5F) emitFmaxV(dst, src1, src2);
            }
            return finish();
        }

        // VPCMPEQB (0x74) / VPCMPEQW (0x75) / VPCMPEQD (0x76)
        if (vexOp == 0x74 || vexOp == 0x75 || vexOp == 0x76) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B PCMPEQ truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                if (is256) {
                    emitLdrQ(kXmmScratch2, 16, 16);
                }
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (vexOp == 0x74) {
                emitCmeqV16B(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV16B(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            } else if (vexOp == 0x75) {
                emitCmeqV8H(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV8H(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            } else {
                emitCmeqV4S(dst, src1, src2);
                if (is256 && m.isMemory) {
                    emitCmeqV4S(kXmmScratch2, src1, kXmmScratch2);
                    emitOrrV(dst, dst, kXmmScratch2);
                }
            }
            return finish();
        }

        // VPMOVMSKB (0xD7)
        if (vexOp == 0xD7) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B PMOVMSKB truncado");
            const uint8_t dstGpr = m.reg;
            const uint8_t srcXmm = xmmHost(m.rm);

            emitWord(0x4F090400u | ((srcXmm & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
            emitMovImm64(20, 0x8040201008040201ULL);
            emitWord(0x4E083C00u | ((kXmmScratch & 0x1Fu) << 5) | 16);
            emitWord(0x4E183C00u | ((kXmmScratch & 0x1Fu) << 5) | 17);
            emitAndReg(16, 16, 20, false, true);
            emitAndReg(17, 17, 20, false, true);
            emitWord(0x9E670200u | (16 << 5) | 30);
            emitWord(0x9E670200u | (17 << 5) | 31);
            emitWord(0x0E31B800u | (30 << 5) | 30);
            emitWord(0x0E31B800u | (31 << 5) | 31);
            emitWord(0x0E013C00u | (30 << 5) | 16);
            emitWord(0x0E013C00u | (31 << 5) | 17);
            emitWord(0x2A112200u | (17 << 16) | (16 << 5) | (dstGpr & 0x1Fu));

            if (is256) {
                emitWord(0x4F090400u | ((kXmmScratch2 & 0x1Fu) << 5) | (kXmmScratch & 0x1Fu));
                emitWord(0x4E083C00u | ((kXmmScratch & 0x1Fu) << 5) | 16);
                emitWord(0x4E183C00u | ((kXmmScratch & 0x1Fu) << 5) | 17);
                emitAndReg(16, 16, 20, false, true);
                emitAndReg(17, 17, 20, false, true);
                emitWord(0x9E670200u | (16 << 5) | 30);
                emitWord(0x9E670200u | (17 << 5) | 31);
                emitWord(0x0E31B800u | (30 << 5) | 30);
                emitWord(0x0E31B800u | (31 << 5) | 31);
                emitWord(0x0E013C00u | (30 << 5) | 16);
                emitWord(0x0E013C00u | (31 << 5) | 17);
                emitWord(0x2A112200u | (17 << 16) | (16 << 5) | 16);
                emitBfi(dstGpr, 16, 16, 16);
            }
            return finish();
        }

        // VUNPCKLPS/D (0x14), VUNPCKHPS/D (0x15) y VPUNPCK* (0x60..0x62, 0x68..0x6A, 0x6C, 0x6D)
        if (vexOp == 0x14 || vexOp == 0x15 ||
            (vexOp >= 0x60 && vexOp <= 0x62) || (vexOp >= 0x68 && vexOp <= 0x6A) ||
            vexOp == 0x6C || vexOp == 0x6D) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B UNPCK truncado");
            const uint8_t dst = xmmHost(m.reg);
            const uint8_t src1 = xmmHost(vvvv);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            const bool isHigh = (vexOp == 0x15 || vexOp == 0x68 || vexOp == 0x69 || vexOp == 0x6A || vexOp == 0x6D);
            uint8_t sz = 2; // 4S (single)
            if (vexOp == 0x6C || vexOp == 0x6D || ((vexOp == 0x14 || vexOp == 0x15) && pp == 1)) {
                sz = 3; // 2D (double/qword)
            } else if (vexOp == 0x62 || vexOp == 0x6A || ((vexOp == 0x14 || vexOp == 0x15) && pp == 0)) {
                sz = 2; // 4S (dword)
            } else if (vexOp == 0x61 || vexOp == 0x69) {
                sz = 1; // 8H (word)
            } else if (vexOp == 0x60 || vexOp == 0x68) {
                sz = 0; // 16B (byte)
            }
            const uint32_t base = isHigh ? 0x4E007800u : 0x4E003800u;
            emitWord(base | ((static_cast<uint32_t>(sz) & 3u) << 22) |
                     ((src2 & 0x1Fu) << 16) | ((src1 & 0x1Fu) << 5) | (dst & 0x1Fu));
            return finish();
        }

        // VEX.128 0F D4/FC..FE (66) y F8..FB: VPADD*/VPSUB* con tres operandos.
        if ((vexOp == 0xD4 || vexOp == 0xFC || vexOp == 0xFD || vexOp == 0xFE ||
             vexOp == 0xF8 || vexOp == 0xF9 || vexOp == 0xFA || vexOp == 0xFB) &&
            !is256 && pp == 1u) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B VPADD/PSUB truncado");
            const uint8_t dst = xmmHost(vvvv);
            const uint8_t src1 = xmmHost(m.reg);
            uint8_t src2 = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src2 = kXmmScratch;
            } else {
                src2 = xmmHost(m.rm);
            }
            if (src2 == dst) {
                emitOrrV(kXmmScratch2, src2, src2);
                src2 = kXmmScratch2;
            }
            if (dst != src1) emitOrrV(dst, src1, src1);

            uint8_t elemSize = 1;
            if (vexOp == 0xFD || vexOp == 0xF9) elemSize = 2;
            else if (vexOp == 0xFE || vexOp == 0xFA) elemSize = 4;
            else if (vexOp == 0xD4 || vexOp == 0xFB) elemSize = 8;

            if (vexOp >= 0xF8 && vexOp <= 0xFB) {
                emitSubV(dst, dst, src2, elemSize);
            } else {
                emitAddV(dst, dst, src2, elemSize);
            }
            return finish();
        }

        // VEX.128 0F E6 /r: VCVTDQ2PD (F3), VCVTPD2DQ (F2), VCVTTPD2DQ (66).
        if (vexOp == 0xE6 && pp >= 1u && pp <= 3u && !is256) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B CVTx2x 0xE6 truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (pp == 2u) {
                    emitLdrD(kXmmScratch, 16, 0); // 2x int32 (64 bits)
                } else {
                    emitLdrQ(kXmmScratch, 16, 0); // 2x double (128 bits)
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (pp == 2u) { // F3: VCVTDQ2PD
                emitWord(0x0F20A400u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
                emitWord(0x4E61D800u | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
            } else { // F2/66: VCVTPD2DQ / VCVTTPD2DQ
                emitWord(0x4EE1B800u | ((src & 0x1Fu) << 5) | (dst & 0x1Fu));
                emitWord(0x0EA12800u | ((dst & 0x1Fu) << 5) | (dst & 0x1Fu));
                emitWord(0x4E181FE0u | (dst & 0x1Fu));
            }
            return finish();
        }

        // VEX.128.66.0F 71/72/73 /r ib: desplazamientos vectoriales inmediatos.
        if ((vexOp == 0x71 || vexOp == 0x72 || vexOp == 0x73) && pp == 1u && !is256) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B vector shift truncado");
            uint8_t imm = 0;
            if (!reader.u8(imm)) return fail("VEX 2B vector shift imm8 truncado");
            // En VEX el grupo va en ModRM.reg, el destino en vvvv y la fuente en ModRM.rm.
            const uint8_t dst = xmmHost(vvvv);
            const uint8_t src = xmmHost(m.rm);
            const uint8_t group = static_cast<uint8_t>(m.reg & 7);

            if (vexOp == 0x73) {
                if (group == 3) { // VPSRLDQ
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitExtV16B(dst, src, kXmmScratch, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 7) { // VPSLLDQ
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitEorV(kXmmScratch, kXmmScratch, kXmmScratch);
                        emitExtV16B(dst, kXmmScratch, src, static_cast<uint8_t>(16 - imm));
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 2) { // VPSRLQ
                    if (imm >= 64) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV2D(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 6) { // VPSLLQ
                    if (imm >= 64) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV2D(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else {
                    return fail("VEX 0F 73 grupo no soportado");
                }
            } else if (vexOp == 0x72) {
                if (group == 2) { // VPSRLD
                    if (imm >= 32) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV4S(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 4) { // VPSRAD
                    if (imm >= 32) imm = 31;
                    if (imm > 0) {
                        emitSshrV4S(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 6) { // VPSLLD
                    if (imm >= 32) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV4S(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else {
                    return fail("VEX 0F 72 grupo no soportado");
                }
            } else { // 0x71
                if (group == 2) { // VPSRLW
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitUshrV8H(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 4) { // VPSRAW
                    if (imm >= 16) imm = 15;
                    if (imm > 0) {
                        emitSshrV8H(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else if (group == 6) { // VPSLLW
                    if (imm >= 16) {
                        emitEorV(dst, dst, dst);
                    } else if (imm > 0) {
                        emitShlV8H(dst, src, imm);
                    } else if (dst != src) {
                        emitOrrV(dst, src, src);
                    }
                } else {
                    return fail("VEX 0F 71 grupo no soportado");
                }
            }
            return finish();
        }

        // VEX.128.66.0F 70 /r ib: VPSHUFD xmm1, xmm2/m128, imm8 (vvvv ignorado).
        if (vexOp == 0x70 && pp == 1u && !is256) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B VPSHUFD truncado");
            uint8_t imm = 0;
            if (!reader.u8(imm)) return fail("VEX 2B VPSHUFD imm8 truncado");
            const uint8_t dst = xmmHost(m.reg);
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrQ(kXmmScratch, 16, 0);
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }
            if (dst == src) {
                emitOrrV(kXmmScratch2, src, src);
                src = kXmmScratch2;
            }
            for (uint8_t i = 0; i < 4; ++i) {
                const uint8_t sel = (imm >> (i * 2)) & 3u;
                emitInsElementS(dst, i, src, sel);
            }
            return finish();
        }

        // VEX.128.0F 2E/2F /r: VUCOMISS/VCOMISS (pp=0) o VUCOMISD/VCOMISD (pp=1).
        if ((vexOp == 0x2E || vexOp == 0x2F) && pp <= 1u) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B UCOMISx truncado");
            const uint8_t dst = xmmHost(m.reg);
            const bool isDouble = pp == 1u;
            uint8_t src = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (isDouble) {
                    emitLdrD(kXmmScratch, 16, 0);
                } else {
                    emitLdrS(kXmmScratch, 16, 0);
                }
                src = kXmmScratch;
            } else {
                src = xmmHost(m.rm);
            }

            if (isDouble) {
                emitFcmpD(dst, src);
            } else {
                emitFcmpS(dst, src);
            }
            // Mismos bits que la ruta legacy: ZF=0x40, PF=0x04, CF=0x01.
            emitMovImm64(20, 0);
            emitMovImm64(16, 0x40);
            emitWord(0x1A940210u); // csel w16, w16, w20, eq
            emitMovImm64(17, 0x01);
            emitWord(0x1A903231u); // csel w17, w17, w16, lo (cc)
            emitMovImm64(21, 0x45);
            emitWord(0x1A9162B4u); // csel w20, w21, w17, vs
            emitWord(0x53020A97u);  // UBFX W23, W20, #2, #1 (PF)
            emitWord(0x520002F7u);  // EOR W23, W23, #1 -> pfByte = PF ? 0 : 1
            emitStrB(23, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));
            emitLdrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            emitMovImm64(22, ~0x8D5ULL);
            emitAndReg(21, 21, 22, false, true);
            emitOrrReg(21, 21, 20, true);
            emitStrX(21, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            return finish();
        }

        // VEX.128.F3.0F.W0 2D /r: VCVTSS2SI r32, xmm/m32.
        if (vexOp == 0x2D && pp == 2u) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B CVTSS2SI truncado");
            uint8_t srcXmm = 0;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitLdrS(kXmmScratch, 16, 0);
                srcXmm = kXmmScratch;
            } else {
                srcXmm = xmmHost(m.rm);
            }
            emitFcvtzsS(m.reg, srcXmm, false); // VEX2 siempre W=0 -> r32
            return finish();
        }

        // VEX.128.0F AE /r: grupo 15 (VLDMXCSR /2, VSTMXCSR /3, barreras /5../7).
        if (vexOp == 0xAE) {
            ModRm m;
            if (!decodeModRm(reader, vexR, false, false, m)) return fail("VEX 2B 0F AE truncado");
            const uint8_t groupExt = m.reg & 7u;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                switch (groupExt) {
                    case 0: { // VFXSAVE m512
                        emitMovImm64(17, 0x037F);
                        emitStrH(17, 16, 0);
                        emitMovImm64(17, 0);
                        emitStrH(17, 16, 2);
                        emitStrB(17, 16, 4);
                        emitStrB(17, 16, 5);
                        emitStrH(17, 16, 6);
                        emitStrX(31, 16, 8);
                        emitStrX(31, 16, 16);
                        emitMovImm64(17, 0x1F80);
                        emitStrW(17, 16, 24);
                        emitMovImm64(17, 0xFFFF);
                        emitStrW(17, 16, 28);
                        for (uint8_t i = 0; i < 16; ++i) emitStrQ(xmmHost(i), 16, 160 + i * 16);
                        break;
                    }
                    case 1: // VFXRSTOR m512
                        for (uint8_t i = 0; i < 16; ++i) emitLdrQ(xmmHost(i), 16, 160 + i * 16);
                        break;
                    case 2: // VLDMXCSR m32
                        emitLdrW(17, 16, 0);
                        break;
                    case 3: // VSTMXCSR m32
                        emitMovImm64(17, 0x1F80);
                        emitStrW(17, 16, 0);
                        break;
                    default:
                        emitNop();
                        break;
                }
            } else {
                switch (groupExt) {
                    case 5: emitWord(0xD50339BFu); break; // VLFENCE -> DMB ISHLD
                    case 6: emitWord(0xD5033BBFu); break; // VMFENCE -> DMB ISH
                    case 7: emitWord(0xD5033ABFu); break; // VSFENCE -> DMB ISHST
                    default: emitNop(); break;
                }
            }
            return finish();
        }

        return fail("VEX 2B opcode no soportado");
    }

    // --- MOV r/m8, inm8 -----------------------------------------------------------
    if (op == 0xC6) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOV imm8 truncado");
        if ((m.reg & 7) != 0) return fail("MOV C6 con grupo distinto de /0");
        uint8_t imm = 0;
        if (!readImm8(imm)) return fail("MOV imm8 truncado");
        if (m.isMemory) {
            emitEffectiveAddress(m, nextRip());
            emitMovImm64(17, imm);
            emitStrB(17, 16, 0);
        } else {
            emitMovImm64(16, imm);
            if (!currentRexPresent_ && m.rm >= 4 && m.rm <= 7) {
                emitBfi(m.rm - 4, 16, 8, 8); // AH, CH, DH, BH
            } else {
                emitBfiByte(m.rm, 16);
            }
        }
        return finish();
    }

    // --- MOV r/m, inm32/imm16 -----------------------------------------------------
    if (op == 0xC7) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("MOV imm32 truncado");
        if ((m.reg & 7) != 0) return fail("MOV C7 con grupo distinto de /0");
        if (prefix66) {
            uint16_t imm16 = 0;
            if (!readImm16(imm16)) return fail("MOV imm16 truncado");
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitMovImm64(17, imm16);
                emitStrH(17, 16, 0);
            } else {
                emitMovImm64(16, imm16);
                emitBfi(m.rm, 16, 0, 16);
            }
        } else {
            uint32_t imm32 = 0;
            if (!readImm32(imm32)) return fail("MOV imm32 truncado");
            const uint64_t imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(
                                            static_cast<int32_t>(imm32)))
                                      : static_cast<uint64_t>(imm32);
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                emitMovImm64(17, imm);
                if (is64) {
                    emitStrX(17, 16, 0);
                } else {
                    emitStrW(17, 16, 0);
                }
            } else if (is64) {
                emitMovImm64(16, imm);
                emitMovReg(m.rm, 16, true);
            } else {
                emitMovImm64(16, imm);
                emitMovReg(m.rm, 16, false);
            }
        }
        return finish();
    }

    // --- LEAVE --------------------------------------------------------------------
    if (op == 0xC9) {
        if (mode32_) {
            emitMovReg(4, 5, false);
            emitLdrW(5, 4, 0);
            emitAddImm(4, 4, 4, true, false);
        } else {
            emitMovReg(4, 5, true);
            emitLdrX(5, 4, 0);
            emitAddImm(4, 4, 8, true, false);
        }
        return finish();
    }

    // --- INT 3 (0xCC) / INT imm8 (0xCD) -------------------------------------------
    if (op == 0xCC) {
        emitWriteRipImm(nextRip());
        isControlFlow = true;
        return finish();
    }

    if (op == 0xCD) {
        uint8_t vec = 0;
        if (!reader.u8(vec)) return fail("INT imm8 truncado");
        emitWriteRipImm(nextRip());
        isControlFlow = true;
        return finish();
    }

    // --- CALL/JMP rel32, JMP rel8 -------------------------------------------------
    if (op == 0xE8 || op == 0xE9) {
        uint32_t raw = 0;
        if (!readImm32(raw)) return fail("rel32 truncado");
        uintptr_t target = nextRip() + static_cast<uintptr_t>(
                                                static_cast<int64_t>(static_cast<int32_t>(raw)));
        if (mode32_) target &= 0xFFFFFFFFu;
        if (op == 0xE8) {
            emitSubImm(4, 4, stackSize, true, false);
            emitMovImm64(17, nextRip());
            if (stackSize == 8) {
                emitStrX(17, 4, 0);
            } else {
                emitStrW(17, 4, 0);
            }
            if (returnStack.load(std::memory_order_relaxed)) {
                // Pila de retornos: entrada = ras[++rasTop & 63] = {RIP de retorno, continuación}.
                // BL al trampolín de llamada: X30 = continuación (y el predictor de retornos del
                // núcleo la apila); el RET del huésped vuelve con RET X20 directo aquí.
                const int32_t topOff = static_cast<int32_t>(offsetof(CpuContext, rasTop));
                emitLdrX(20, 19, topOff);
                emitAddImm(20, 20, 1, true);
                emitStrX(20, 19, topOff);
                emitWord(0x92401694u);  // AND X20, X20, #63
                emitAddImm(21, 19, static_cast<uint16_t>(offsetof(CpuContext, ras)), true);
                emitWord(0x8B1412B5u);  // ADD X21, X21, X20, LSL #4
                const size_t bl = wordPos();
                emitWord(0x94000000u);  // BL trampolín (se parchea)
                emitLinkableExit(nextRip());  // continuación: seguir en la instrucción tras el CALL
                patchImm26(bl, wordPos());
                emitStpX(17, 30, 21, 0);  // {RIP de retorno, X30 = continuación}
            }
        }
        if (op == 0xE9 && stitchBudget_ > 0 && target > rip &&
            (target & ~uintptr_t{0xFFFull}) == (curBlockRip_ & ~uintptr_t{0xFFFull}) &&
            target < stitchLimit_) {
            stitchTarget_ = target;
            return finish();
        }
        emitLinkableExit(target);
        isControlFlow = true;
        return finish();
    }

    if (op == 0xEB) {
        int8_t rel8 = 0;
        if (!reader.s8(rel8)) return fail("JMP rel8 truncado");
        uintptr_t target = nextRip() + static_cast<uintptr_t>(static_cast<int64_t>(rel8));
        if (mode32_) target &= 0xFFFFFFFFu;
        if (stitchBudget_ > 0 && target > rip &&
            (target & ~uintptr_t{0xFFFull}) == (curBlockRip_ & ~uintptr_t{0xFFFull}) &&
            target < stitchLimit_) {
            stitchTarget_ = target;
            return finish();
        }
        emitLinkableExit(target);
        isControlFlow = true;
        return finish();
    }

    // --- INC/DEC de 32 bits (0x40-0x4F en modo 32) --------------------------------
    if (mode32_ && op >= 0x40 && op <= 0x4F) {
        const uint8_t reg = static_cast<uint8_t>(op & 7);
        const bool increment = op < 0x48;
        if (increment) {
            emitAddImm(reg, reg, 1, false, true);
        } else {
            emitSubImm(reg, reg, 1, false, true);
        }
        emitWriteFlags(CfMode::Preserve, OfMode::Overflow, reg, 0);
        return finish();
    }

    // --- Grupo 0xFE/0xFF: INC/DEC/CALL/JMP/PUSH -----------------------------------
    if (op == 0xFE || op == 0xFF) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("grupo FE/FF truncado");
        const uint8_t group = static_cast<uint8_t>(m.reg & 7);
        const bool byte = op == 0xFE;
        const uint8_t size = byte ? 1 : (is64 ? 8 : 4);
        if (byte && !m.isMemory && !check8(m.rm)) return 0;

        if (group == 0 || group == 1) {
            // Con 66h, INC/DEC r/m16 (antes de 32 bits: inc ax de 0xFFFF acarreaba al bit 16).
            emitIncDec(m, group == 0, (!byte && !is64 && prefix66) ? 2 : size, nextRip());
            return finish();
        }
        if (group == 2 && !byte) {
            // En modo 64 bits, CALL r/m64 usa 64 bits por defecto (sin REX.W).
            const bool operand64 = !mode32_;
            // call [rip+disp] a una función nativa rápida (IAT ya parcheada): BLR directo.
            if (!mode32_ && m.isMemory && m.ripRelative && currentSegPrefix_ == 0 &&
                fastNativeCalls.load(std::memory_order_relaxed)) {
                const uintptr_t slot = nextRip() + static_cast<uintptr_t>(static_cast<int64_t>(m.disp));
                uint64_t target = 0;
                bool blocking = false;
                iovec local{&target, 8};
                iovec remote{reinterpret_cast<void*>(slot), 8};
                if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != 8) target = 0;  // sin SIGSEGV
                if (target != 0 && lookupFastNative(static_cast<uintptr_t>(target), blocking)) {
                    emitFastNativeCall(static_cast<uintptr_t>(target), blocking, nextRip(), false, slot);
                    return finish();  // no termina el bloque: sigue la siguiente instrucción
                }
            }
            // 32 bits: call dword ptr [IAT] (dirección absoluta) a una función rápida: BLR directo.
            if (mode32_ && m.isMemory && m.absolute && !m.hasBase && !m.hasIndex && currentSegPrefix_ == 0 &&
                fastNativeCalls.load(std::memory_order_relaxed)) {
                auto hostOf = fast32Host.load(std::memory_order_acquire);
                auto slotsOf = fast32Slots.load(std::memory_order_acquire);
                const uintptr_t slot = static_cast<uintptr_t>(static_cast<uint32_t>(m.disp));
                uint32_t alias = 0;
                iovec local{&alias, 4};
                iovec remote{reinterpret_cast<void*>(slot), 4};
                if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != 4) alias = 0;
                const uintptr_t host = (alias != 0 && hostOf != nullptr) ? hostOf(alias) : 0;
                bool blocking = false;
                const int slots = (host != 0 && slotsOf != nullptr) ? slotsOf(host) : -1;
                if (host != 0 && slots >= 0 && slots <= 4 && lookupFastNative(host, blocking)) {
                    emitFastNativeCall32(host, blocking, nextRip(), static_cast<uint32_t>(slots));
                    return finish();
                }
            }
            // call reg justo tras "mov reg, imm64" con una función rápida (icalls de Mono).
            if (!mode32_ && !m.isMemory && movImmSeq_ + 1 == instrSeq_ && movImmReg_ == m.rm &&
                fastNativeCalls.load(std::memory_order_relaxed)) {
                bool blocking = false;
                if (lookupFastNative(static_cast<uintptr_t>(movImmValue_), blocking)) {
                    emitFastNativeCall(static_cast<uintptr_t>(movImmValue_), blocking, nextRip());
                    return finish();
                }
            }
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (operand64) {
                    emitLdrX(16, 16, 0);
                } else {
                    emitLdrW(16, 16, 0);
                }
            } else {
                emitMovReg(16, m.rm, operand64);
            }
            // 32 bits: CALL indirecto (vtables COM, punteros a función) a una función nativa (zona de alias): se
            // atiende desde el propio bloque con fast32Indirect, sin salir al despachador (D3D9 en DmC hace
            // cientos de miles de llamadas por segundo así).
            if (mode32_ && fastNativeCalls.load(std::memory_order_relaxed) &&
                fast32AliasBase.load(std::memory_order_relaxed) != 0 &&
                fast32Indirect.load(std::memory_order_relaxed) != nullptr) {
                emitMovImm64(17, fast32AliasBase.load(std::memory_order_relaxed));
                emitSubReg(17, 16, 17, false, false);  // W17 = destino - base
                emitWord(0x53147E34u);                 // LSR W20, W17, #20 (zona de 1 MB; preserva W17)
                const size_t notAlias = wordPos();
                emitWord(0x35000014u);                 // CBNZ W20, normal (se parchea)
                emitFastIndirect32(nextRip());
                const size_t done = wordPos();
                emitWord(0x14000000u);                 // B fin (se parchea)
                patchImm19(notAlias, wordPos());
                emitSubImm(4, 4, stackSize, true, false);
                emitMovImm64(17, nextRip());
                emitStrW(17, 4, 0);
                if (returnStack.load(std::memory_order_relaxed)) {
                    const int32_t topOff = static_cast<int32_t>(offsetof(CpuContext, rasTop));
                    emitLdrX(20, 19, topOff);
                    emitAddImm(20, 20, 1, true);
                    emitStrX(20, 19, topOff);
                    emitWord(0x92401694u);  // AND X20, X20, #63
                    emitAddImm(21, 19, static_cast<uint16_t>(offsetof(CpuContext, ras)), true);
                    emitWord(0x8B1412B5u);  // ADD X21, X21, X20, LSL #4
                    const size_t bl = wordPos();
                    emitWord(0x94000000u);  // BL trampolín (se parchea)
                    emitLinkableExit(nextRip());
                    patchImm26(bl, wordPos());
                    emitStpX(17, 30, 21, 0);
                }
                flagEvents_.push_back({FlagEventKind::Read, 0, 0});  // salida: los flags quedan vivos
                emitStrW(16, 19, static_cast<int32_t>(offsetof(CpuContext, rip)));
                if (inlineDispatch.load(std::memory_order_relaxed)) {
                    emitInlineDispatch();
                } else {
                    pendingExits_.push_back({wordPos(), false, false});
                    emitWord(0x14000000u);
                }
                patchImm26(done, wordPos());
                return finish();
            }
            emitSubImm(4, 4, stackSize, true, false);
            emitMovImm64(17, nextRip());
            if (stackSize == 8) {
                emitStrX(17, 4, 0);
            } else {
                emitStrW(17, 4, 0);
            }
            emitWriteRipReg(16, stackSize == 8);
            isControlFlow = true;
            return finish();
        }
        if (group == 4 && !byte) {
            const bool operand64 = !mode32_;
            // jmp [rip+disp] (stub de importación) a una función rápida: llamada directa + RET.
            if (!mode32_ && m.isMemory && m.ripRelative && currentSegPrefix_ == 0 &&
                fastNativeCalls.load(std::memory_order_relaxed)) {
                const uintptr_t slot = nextRip() + static_cast<uintptr_t>(static_cast<int64_t>(m.disp));
                uint64_t target = 0;
                bool blocking = false;
                iovec local{&target, 8};
                iovec remote{reinterpret_cast<void*>(slot), 8};
                if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != 8) target = 0;
                if (target != 0 && lookupFastNative(static_cast<uintptr_t>(target), blocking)) {
                    emitFastNativeCall(static_cast<uintptr_t>(target), blocking, nextRip(), true, slot);
                    isControlFlow = true;
                    return finish();
                }
            }
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (operand64) {
                    emitLdrX(16, 16, 0);
                } else {
                    emitLdrW(16, 16, 0);
                }
            } else {
                emitMovReg(16, m.rm, operand64);
            }
            if (mode32_ && fastNativeCalls.load(std::memory_order_relaxed) &&
                fast32AliasBase.load(std::memory_order_relaxed) != 0 &&
                fast32Table.load(std::memory_order_relaxed) != nullptr) {
                emitMovImm64(17, fast32AliasBase.load(std::memory_order_relaxed));
                emitSubReg(17, 16, 17, false, false);  // W17 = destino - base
                emitWord(0x53147E34u);                 // LSR W20, W17, #20
                const size_t notAlias = wordPos();
                emitWord(0x35000014u);                 // CBNZ W20, normal (se parchea)
                emitFastIndirect32(nextRip(), true);
                patchImm19(notAlias, wordPos());
            }
            emitWriteRipReg(16, stackSize == 8);
            isControlFlow = true;
            return finish();
        }
        if (group == 6 && !byte) {
            const bool operand64 = !mode32_;
            if (m.isMemory) {
                emitEffectiveAddress(m, nextRip());
                if (operand64) {
                    emitLdrX(16, 16, 0);
                } else {
                    emitLdrW(16, 16, 0);
                }
            } else {
                emitMovReg(16, m.rm, operand64);
            }
            emitSubImm(4, 4, stackSize, true, false);
            if (stackSize == 8) {
                emitStrX(16, 4, 0);
            } else {
                emitStrW(16, 4, 0);
            }
            return finish();
        }
        return fail("grupo FE/FF no soportado");
    }

    // --- Grupo 0xF6/0xF7: TEST/NOT/NEG r/m -------------------------------------------
    if (op == 0xF6 || op == 0xF7) {
        ModRm m;
        if (!decodeModRm(reader, rexR, rexX, rexB, m)) return fail("Grupo F6/F7 ModRM truncado");
        const uint8_t group = static_cast<uint8_t>(m.reg & 7);
        const bool byte = op == 0xF6;
        const uint8_t size = byte ? 1 : (is64 ? 8 : (prefix66 ? 2 : 4));

        if (group == 0) {  // TEST r/m, inm
            uint64_t imm = 0;
            if (byte) {
                uint8_t v = 0;
                if (!readImm8(v)) return fail("TEST imm8 truncado");
                imm = v;
            } else if (prefix66) {
                uint16_t v = 0;
                if (!readImm16(v)) return fail("TEST imm16 truncado");
                imm = v;
            } else {
                uint32_t v = 0;
                if (!readImm32(v)) return fail("TEST imm32 truncado");
                imm = is64 ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)))
                           : static_cast<uint64_t>(v);
            }
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            emitAluRegImm(8, m, imm, size, nextRip());
            return finish();
        }

        if (group == 2) {  // NOT r/m (inversión de bits)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            emitLoadOperand(m, 16, size, nextRip());
            emitMvnReg(16, 16, is64 && !byte);
            emitStoreOperand(m, 16, size, nextRip());
            return finish();
        }

        if (group == 3) {  // NEG r/m (complemento a dos)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            emitLoadOperand(m, 16, size, nextRip());
            if (size == 1) {
                emitUxtb(16, 16);
                emitCmpImm(16, 0, false);
                emitCset(20, kCondNe); // CF = (orig != 0)
                emitCmpImm(16, 0x80, false);
                emitCset(21, kCondEq); // OF = (orig == 0x80)
                emitSubReg(16, 31, 16, false, false);
                emitUxtb(16, 16);
                emitStoreOperand(m, 16, 1, nextRip());
                emitCmpImm(16, 0, false);
                emitCset(22, kCondEq); // ZF = (result == 0)
                emitUbfx(23, 16, 7, 1); // SF = bit 7
                emitLslImm(22, 22, 6, true);
                emitOrrReg(20, 20, 22, true);
                emitLslImm(23, 23, 7, true);
                emitOrrReg(20, 20, 23, true);
                emitLslImm(21, 21, 11, true);
                emitOrrReg(20, 20, 21, true);
                emitStrB(16, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));  // PF perezoso
                emitStrX(20, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            } else if (size == 2) {
                emitUxth(16, 16);
                emitCmpImm(16, 0, false);
                emitCset(20, kCondNe); // CF
                emitMovImm64(22, 0x8000);
                emitSubReg(31, 16, 22, true, false);
                emitCset(21, kCondEq); // OF
                emitSubReg(16, 31, 16, false, false);
                emitUxth(16, 16);
                emitStoreOperand(m, 16, 2, nextRip());
                emitCmpImm(16, 0, false);
                emitCset(22, kCondEq); // ZF
                emitUbfx(23, 16, 15, 1); // SF
                emitLslImm(22, 22, 6, true);
                emitOrrReg(20, 20, 22, true);
                emitLslImm(23, 23, 7, true);
                emitOrrReg(20, 20, 23, true);
                emitLslImm(21, 21, 11, true);
                emitOrrReg(20, 20, 21, true);
                emitStrB(16, 19, static_cast<int32_t>(offsetof(CpuContext, pfByte)));  // PF perezoso
                emitStrX(20, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
            } else {
                emitSubReg(16, 31, 16, true, is64);
                emitStoreOperand(m, 16, size, nextRip());
                emitWriteFlags(CfMode::Borrow, OfMode::Overflow, 16, 0);
            }
            return finish();
        }

        if (group == 4) {  // MUL r/m (multiplicación sin signo)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            if (is64) {
                emitLoadOperand(m, 16, 8, nextRip());
                emitMulReg(17, 0, 16, true);
                emitUmulh(2, 0, 16);
                emitMovReg(0, 17, true);
                emitTstReg(2, true);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 0, 0);
            } else if (!byte) {
                emitLoadOperand(m, 16, 4, nextRip());
                emitWord(0x9BA07C00u | ((16u & 0x1F) << 16) | ((0u & 0x1F) << 5) | (17u & 0x1F));
                emitMovReg(0, 17, false);
                emitLsrImm(2, 17, 32, true);
                emitTstReg(2, false);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 0, 0);
            } else {
                emitLoadOperand(m, 16, 1, nextRip());
                emitUbfx(17, 0, 0, 8);
                emitMulReg(16, 17, 16, false);
                emitWord(0xB3402C00u | ((16u & 0x1F) << 5) | 0);
                emitUbfx(20, 16, 8, 8);
                emitTstReg(20, false);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 16, 0);
            }
            return finish();
        }

        if (group == 5) {  // IMUL r/m (multiplicación con signo)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            if (is64) {
                emitLoadOperand(m, 16, 8, nextRip());
                emitMulReg(17, 0, 16, true);
                emitSmulh(2, 0, 16);
                emitMovReg(0, 17, true);
                emitAsrImm(16, 0, 63, true);
                emitSubReg(31, 2, 16, true, true);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 0, 0);
            } else if (!byte) {
                emitLoadOperand(m, 16, 4, nextRip());
                emitSmull(17, 0, 16);
                emitMovReg(0, 17, false);
                emitLsrImm(2, 17, 32, true);
                emitSxtw(16, 0);
                emitSubReg(31, 17, 16, true, true);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 0, 0);
            } else {
                emitLoadOperand(m, 16, 1, nextRip());
                emitSxtb(16, 16);
                emitSxtb(17, 0);
                emitMulReg(16, 17, 16, false);
                emitWord(0xB3402C00u | ((16u & 0x1F) << 5) | 0);
                emitSxtb(17, 16);
                emitSubReg(31, 16, 17, true, false);
                emitCset(20, kCondNe);
                emitWriteFlags(CfMode::Reg, OfMode::Reg, 16, 0);
            }
            return finish();
        }

        if (group == 6) {  // DIV r/m (división sin signo)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            if (is64) {
                emitLoadOperand(m, 16, 8, nextRip());
                emitUdiv(17, 0, 16, true);
                emitMulReg(21, 17, 16, true);
                emitSubReg(2, 0, 21, false, true);
                emitMovReg(0, 17, true);
            } else if (!byte) {
                emitLoadOperand(m, 16, 4, nextRip());
                emitUbfx(16, 16, 0, 32);
                emitLslImm(17, 2, 32, true);
                emitUbfx(21, 0, 0, 32);
                emitOrrReg(17, 17, 21, true);
                emitUdiv(22, 17, 16, true);
                emitMulReg(21, 22, 16, true);
                emitSubReg(2, 17, 21, false, true);
                emitMovReg(0, 22, false);
            } else {
                emitLoadOperand(m, 16, 1, nextRip());
                emitUbfx(17, 0, 0, 16);
                emitUdiv(22, 17, 16, false);
                emitMulReg(21, 22, 16, false);
                emitSubReg(21, 17, 21, false, false);
                emitBfiByte(0, 22);
                emitLslImm(21, 21, 8, false);
                emitWord(0xB3402C00u | ((21u & 0x1F) << 5) | 0);
            }
            return finish();
        }

        if (group == 7) {  // IDIV r/m (división con signo)
            if (byte && !m.isMemory && !check8(m.rm)) return 0;
            if (is64) {
                emitLoadOperand(m, 16, 8, nextRip());
                emitSdiv(17, 0, 16, true);
                emitMulReg(21, 17, 16, true);
                emitSubReg(2, 0, 21, false, true);
                emitMovReg(0, 17, true);
            } else if (!byte) {
                emitLoadOperand(m, 16, 4, nextRip());
                emitSxtw(16, 16);
                emitLslImm(17, 2, 32, true);
                emitUbfx(21, 0, 0, 32);
                emitOrrReg(17, 17, 21, true);
                emitSdiv(22, 17, 16, true);
                emitMulReg(21, 22, 16, true);
                emitSubReg(2, 17, 21, false, true);
                emitMovReg(0, 22, false);
            } else {
                emitLoadOperand(m, 16, 1, nextRip());
                emitSxtb(16, 16);
                emitSxth(17, 0);
                emitSdiv(22, 17, 16, false);
                emitMulReg(21, 22, 16, false);
                emitSubReg(21, 17, 21, false, false);
                emitBfiByte(0, 22);
                emitLslImm(21, 21, 8, false);
                emitWord(0xB3402C00u | ((21u & 0x1F) << 5) | 0);
            }
            return finish();
        }

        return fail("grupo F6/F7 no soportado");
    }

    // --- CLC (0xF8) / STC (0xF9) / CMC (0xF5) / CLD (0xFC) / STD (0xFD) -----------
    if (op == 0xF8 || op == 0xF9 || op == 0xF5 || op == 0xFC || op == 0xFD) {
        emitLdrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        if (op == 0xF8) {        // CLC: CF = 0
            emitMovImm64(17, ~1ULL);
            emitAndReg(16, 16, 17, false, true);
        } else if (op == 0xF9) { // STC: CF = 1
            emitMovImm64(17, 1);
            emitOrrReg(16, 16, 17, true);
        } else if (op == 0xF5) { // CMC: CF = ~CF
            emitMovImm64(17, 1);
            emitEorReg(16, 16, 17, true);
        } else if (op == 0xFC) { // CLD: DF = 0 (bit 10)
            emitMovImm64(17, ~(1ULL << 10));
            emitAndReg(16, 16, 17, false, true);
        } else if (op == 0xFD) { // STD: DF = 1 (bit 10)
            emitMovImm64(17, 1ULL << 10);
            emitOrrReg(16, 16, 17, true);
        }
        emitStrX(16, 19, static_cast<int32_t>(offsetof(CpuContext, rflags)));
        return finish();
    }

    char hexOp[64];
    std::snprintf(hexOp, sizeof(hexOp), "opcode x86 %02X no soportado en RIP=0x%llx", op,
                  static_cast<unsigned long long>(rip));
    lastError_ = hexOp;
    return 0;
}

// ---------------------------------------------------------------------------
// Compilación, caché y ejecución
// ---------------------------------------------------------------------------

TranslatedBlock* OryonEmitterBackend::compileLocked(const uint8_t* code, size_t codeSize,
                                                    uintptr_t baseRip) {
    if (code == nullptr || codeSize == 0 || arena_ == nullptr) {
        lastError_ = "bloque sin código o JIT sin configurar";
        return nullptr;
    }

    size_t start = 0;
    if (!reserveBlockLocked(start)) return nullptr;
    const size_t savedOffset = jitOffset_;  // tras un posible vaciado de la arena
    jitOffset_ = start;
    aotBlockStart_ = start;
    aotRelocs_.clear();
    aotDeps_.clear();
    emitOk_ = true;
    pendingExits_.clear();
    flagEvents_.clear();
    inFlagWrite_ = false;
    lastExitDynamic_ = false;
    movImmSeq_ = ~uint64_t{0} - 1;  // el patrón mov+call no cruza bloques (se puede entrar por otro sitio)
    nzcvSeq_ = ~uint64_t{0} - 1;    // NZCV no sobrevive al despachador ni a los enlaces entre bloques

    emitPrologue();
    prologueBytes_ = jitOffset_ - start;  // el cuerpo empieza aquí (destino de los enlaces)
    const bool probed = !probeRvas().empty() && probeMatches(baseRip);
    if (probed) emitProbeCall(baseRip);
    curBlockRip_ = baseRip;
    stitchTarget_ = 0;
    stitchLimit_ = baseRip + std::min(codeSize, kMaxGuestBytesPerBlock);
    stitchBudget_ = 3;
    struct ClearBlockRip {
        uintptr_t& r;
        ~ClearBlockRip() { r = 0; }
    } clearBlockRip{curBlockRip_};

    size_t consumed = 0;
    bool controlFlow = false;
    std::vector<std::pair<uint32_t, uint32_t>> instrMap;
    for (size_t instructions = 0;
         instructions < kMaxInstructionsPerBlock && consumed < codeSize && consumed < kMaxGuestBytesPerBlock;
         ++instructions) {
        instrMap.emplace_back(static_cast<uint32_t>(jitOffset_ - start), static_cast<uint32_t>(consumed));
        ++instrSeq_;  // nunca se repite entre bloques: "mov imm + call" solo dentro del mismo bloque
        instrStartWord_ = wordPos();
        eaIsStack_ = false;  // solo vale para la dirección calculada en esta instrucción
        stitchTarget_ = 0;
        CodeReader reader{code + consumed, code + codeSize};
        bool instControl = false;
        const size_t length =
            decodeAndEmitOne(reader, baseRip + consumed, instControl);
        if (length == 0) {
            jitOffset_ = savedOffset;
            if (lastError_.empty()) lastError_ = "instrucción x86 no soportada";
            return nullptr;
        }
        if (stitchTarget_ != 0) {
            consumed = stitchTarget_ - baseRip;
            --stitchBudget_;
            continue;
        }
        consumed += length;
        if (instControl) {
            controlFlow = true;
            break;
        }
    }

    if (!controlFlow) {
        emitLinkableExit(baseRip + consumed);
    } else if (lastExitDynamic_ && inlineDispatch.load(std::memory_order_relaxed)) {
        emitInlineDispatch();
    }
    const size_t epilogue = wordPos();
    emitEpilogue();
    for (const ExitPatch& p : pendingExits_) {
        const size_t to = epilogue + (p.skipClear ? 1 : 0);
        if (p.cond) patchImm19(p.pos, to);
        else patchImm26(p.pos, to);
    }
    pendingExits_.clear();
    eliminateDeadFlags(start / 4, epilogue);

    if (!emitOk_) {
        jitOffset_ = savedOffset;
        lastError_ = "desbordamiento de la arena JIT";
        return nullptr;
    }

    const size_t used = jitOffset_ - start;
    if (!finishBlockLocked(start, savedOffset)) return nullptr;

    TranslatedBlock block;
    block.guestRip = baseRip;
    block.hostCode = arena_ + start;
    block.guestSize = consumed;
    block.hostSize = used;
    block.endsBlock = controlFlow;
    block.instrMap = std::move(instrMap);

    auto inserted = blockCache_.emplace(baseRip, block);
    if (!inserted.second) {
        inserted.first->second = block;
    }
    hostIndex_[reinterpret_cast<uintptr_t>(block.hostCode)] = baseRip;

    if (aotCache_ != nullptr) {
        (void)aotCache_->appendTranslatedBlock(baseRip, block.hostCode, block.hostSize);
    }

    // Caché persistente: el bloque recién emitido aún no tiene enlaces (se parchean después,
    // desde el despachador), así que se guarda tal cual, sin estado de esta ejecución.
    if (JitDiskCache& disk = JitDiskCache::instance(); disk.enabled() && consumed > 0 && !probed) {
        JitRecord rec;
        rec.guestRip = baseRip;
        rec.guestHash = JitDiskCache::hashBytes(code, consumed);
        rec.guestSize = static_cast<uint32_t>(consumed);
        rec.cfg = aotConfigKey();
        rec.endsBlock = controlFlow;
        const uint8_t* host = arena_ + start;
        rec.host.assign(host, host + used);
        rec.instrMap = inserted.first->second.instrMap;
        rec.relocs = aotRelocs_;
        rec.deps = aotDeps_;
        disk.append(std::move(rec));
    }

    return &inserted.first->second;
}

namespace {
std::atomic<uint64_t> s_jitNs{0};  // tiempo total traduciendo / cargando bloques (todos los hilos)
}  // namespace

uint64_t OryonEmitterBackend::totalJitMs() noexcept { return s_jitNs.load(std::memory_order_relaxed) / 1000000u; }

// Configuración del emisor que cambia el código generado: un bloque guardado con otra no vale.
uint32_t OryonEmitterBackend::aotConfigKey() const noexcept {
    // Subir si cambia el código generado (5: PSHUFLW/HW; 6: CMPSS/SD NEQ-NLT-NLE, SHUFPD, MOVQ,
    // PSHUFB, CVT(T)PD2DQ, hallados con la prueba diferencial tools/jitdiff_gen.py).
    // 7: shifts/rotates de 32 bits con cuenta 0, OF de SHL/SHR 1, RCL/RCR, INC/DEC e IMUL de 16
    // bits, CBW/CWDE, SHLD/SHRD (prueba diferencial de enteros).
    constexpr uint32_t kCodegenVersion = 16;  // 16: RAS 32-bit, fastCache CALL r/m32, trace stitching, FSIN/FCOS/FPATAN directos
    uint32_t k = 0;
    k |= tsoMemory.load(std::memory_order_relaxed) ? 1u : 0u;
    k |= chainBlocks.load(std::memory_order_relaxed) ? 2u : 0u;
    k |= deadFlags.load(std::memory_order_relaxed) ? 4u : 0u;
    k |= inlineDispatch.load(std::memory_order_relaxed) ? 8u : 0u;
    k |= fastNativeCalls.load(std::memory_order_relaxed) ? 16u : 0u;
    k |= mode32_ ? 32u : 0u;
    k |= fuseFlags.load(std::memory_order_relaxed) ? 64u : 0u;
    k |= returnStack.load(std::memory_order_relaxed) ? 128u : 0u;
    k |= crossBlockFlags.load(std::memory_order_relaxed) ? (1u << 14) : 0u;
    k |= static_cast<uint32_t>(__builtin_ctzll(kFastCacheSize)) << 8;
    k |= kCodegenVersion << 16;
    return k;
}

// Hueco en la arena para un bloque nuevo (vacía la arena si está llena).
bool OryonEmitterBackend::reserveBlockLocked(size_t& start) {
    const size_t align = (writeArena_ != nullptr) ? 16 : pageSize();
    if (alignUp(jitOffset_, align) + kMaxJitBytesPerBlock > arenaSize_ && writeArena_ != nullptr) {
        // Arena llena: se vacía y se retraduce lo que haga falta. Seguro aquí: solo se traduce
        // desde el despachador, sin código de la arena en la pila de este hilo.
        LOGI("JIT: arena llena (%zu MB, %zu bloques): se vacía y se retraduce", jitOffset_ >> 20,
             blockCache_.size());
        unlinkAllLocked();
        blockCache_.clear();
        hostIndex_.clear();
        fastGen_.fetch_add(1, std::memory_order_acq_rel);
        jitOffset_ = 0;
        ++arenaGen_;
        // La pila de retornos apunta a continuaciones de la arena vieja: se invalida (RIP 0
        // nunca coincide con una dirección de retorno real).
        std::memset(ctx_.ras, 0, sizeof(ctx_.ras));
    }
    start = alignUp(jitOffset_, align);
    if (start + kMaxJitBytesPerBlock > arenaSize_) {
        lastError_ = "arena JIT agotada";
        return false;
    }
    return true;
}

// Código ya escrito en [start, jitOffset_): lo hace ejecutable y deja jitOffset_ alineado.
bool OryonEmitterBackend::finishBlockLocked(size_t start, size_t savedOffset) {
    if (writeArena_ != nullptr) {
        __builtin___clear_cache(reinterpret_cast<char*>(arena_ + start),
                                reinterpret_cast<char*>(arena_ + jitOffset_));
        jitOffset_ = alignUp(jitOffset_, 16);
        return true;
    }
    const size_t page = pageSize();
    const size_t end = alignUp(jitOffset_, page);
    if (end > arenaSize_) {
        jitOffset_ = savedOffset;
        lastError_ = "bloque JIT excede la arena";
        return false;
    }
    if (mprotect(arena_ + start, end - start, PROT_READ | PROT_EXEC) != 0) {
        jitOffset_ = savedOffset;
        lastError_ = "mprotect(RX) falló: ";
        lastError_ += std::strerror(errno);
        return false;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(arena_ + start), reinterpret_cast<char*>(arena_ + end));
    execRanges_.push_back({reinterpret_cast<uintptr_t>(arena_ + start), reinterpret_cast<uintptr_t>(arena_ + end)});
    jitOffset_ = end;
    return true;
}

// Bloque guardado en una ejecución anterior (o por otro hilo): se copia a la arena y se
// reubica en vez de retraducirlo. nullptr si no hay uno válido para los bytes actuales.
TranslatedBlock* OryonEmitterBackend::loadPersistedLocked(uintptr_t guestRip, size_t available) {
    JitDiskCache& disk = JitDiskCache::instance();
    if (!disk.enabled() || arena_ == nullptr) return nullptr;
    JitRecord rec;
    const auto guestMatches = [guestRip, available](uint64_t hash, uint32_t size) {
        return size > 0 && size <= available &&
               JitDiskCache::hashBytes(reinterpret_cast<const void*>(guestRip), size) == hash;
    };
    if (!disk.lookup(guestRip, aotConfigKey(), guestMatches, rec)) return nullptr;
    if (rec.host.size() > kMaxJitBytesPerBlock) return nullptr;

    const uintptr_t anchor = JitDiskCache::anchor();
    for (const JitReloc& r : rec.relocs) {
        if ((static_cast<size_t>(r.word) + 4) * 4 > rec.host.size() || r.reg > 30) return nullptr;
        if (r.kind == kJitRelocLib) continue;
        bool blocking = false;
        if (!lookupFastNative(anchor + static_cast<uintptr_t>(r.delta), blocking) ||
            blocking != (r.kind == kJitRelocFastNativeBlocking)) {
            return nullptr;  // la lista de nativas rápidas cambió
        }
    }
    for (const JitDep& d : rec.deps) {
        uint64_t value = 0;
        iovec local{&value, 8};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(d.slot)), 8};
        if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != 8 ||
            value != static_cast<uint64_t>(anchor + static_cast<uintptr_t>(d.delta))) {
            return nullptr;  // la IAT apunta a otra cosa: se retraduce
        }
    }
    for (const auto& [hostOff, guestOff] : rec.instrMap) {
        if (hostOff >= rec.host.size() || guestOff >= rec.guestSize) return nullptr;
    }

    size_t start = 0;
    if (!reserveBlockLocked(start)) return nullptr;
    const size_t savedOffset = jitOffset_;
    if (prologueBytes_ == 0) {
        // Hilo que solo carga bloques de disco (nunca traduce): sin el tamaño del prólogo
        // linkExit no enlaza nada y cada salida vuelve al despachador (medido: ~24 % de la CPU
        // del arranque en executeEntry/linkExit/suspensionPoint). El prólogo es fijo: se emite
        // aquí para medirlo y el memcpy de abajo lo sobrescribe.
        jitOffset_ = start;
        emitPrologue();
        prologueBytes_ = jitOffset_ - start;
        jitOffset_ = savedOffset;
    }
    uint8_t* dst = (writeArena_ != nullptr) ? writeArena_ : arena_;
    std::memcpy(dst + start, rec.host.data(), rec.host.size());
    for (const JitReloc& r : rec.relocs) {
        const uint64_t v = static_cast<uint64_t>(anchor + static_cast<uintptr_t>(r.delta));
        const uint32_t words[4] = {
            0xD2800000u | (static_cast<uint32_t>(v & 0xFFFFu) << 5) | r.reg,
            0xF2A00000u | (static_cast<uint32_t>((v >> 16) & 0xFFFFu) << 5) | r.reg,
            0xF2C00000u | (static_cast<uint32_t>((v >> 32) & 0xFFFFu) << 5) | r.reg,
            0xF2E00000u | (static_cast<uint32_t>((v >> 48) & 0xFFFFu) << 5) | r.reg,
        };
        std::memcpy(dst + start + static_cast<size_t>(r.word) * 4, words, sizeof(words));
    }
    jitOffset_ = start + rec.host.size();
    if (!finishBlockLocked(start, savedOffset)) return nullptr;

    TranslatedBlock block;
    block.guestRip = guestRip;
    block.hostCode = arena_ + start;
    block.guestSize = rec.guestSize;
    block.hostSize = rec.host.size();
    block.endsBlock = rec.endsBlock;
    block.instrMap = std::move(rec.instrMap);
    auto inserted = blockCache_.emplace(guestRip, block);
    if (!inserted.second) inserted.first->second = std::move(block);
    hostIndex_[reinterpret_cast<uintptr_t>(inserted.first->second.hostCode)] = guestRip;
    ++statAotLoads_;
    return &inserted.first->second;
}

TranslatedBlock* OryonEmitterBackend::compileInstructionSequence(const uint8_t* code, size_t codeSize,
                                                                 uintptr_t baseRip) {
    std::lock_guard<std::mutex> lock(mutex_);
    return compileLocked(code, codeSize, baseRip);
}

// Diagnóstico: `adb shell setprop debug.hyperplay.watch 0x18005cf40,0x...` registra la primera
// traducción (= primera ejecución en ese hilo) de cada RIP vigilada con RSP y [RSP] (dirección de retorno).
void OryonEmitterBackend::logIfWatchedRip(uintptr_t guestRip) {
    static const std::vector<uintptr_t> s_watch = [] {
        std::vector<uintptr_t> v;
#if defined(__ANDROID__)
        char buf[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.watch", buf) > 0) {
            const char* p = buf;
            while (*p != '\0') {
                char* end = nullptr;
                const unsigned long long a = std::strtoull(p, &end, 16);
                if (end == p) break;
                if (a != 0) v.push_back(static_cast<uintptr_t>(a));
                p = (*end == ',') ? end + 1 : end;
            }
        }
#endif
        return v;
    }();
    if (s_watch.empty()) return;
    if (std::find(s_watch.begin(), s_watch.end(), guestRip) == s_watch.end()) return;
    const uint64_t rsp = ctx_.gpr[4];
    uint64_t ret0 = 0, ret1 = 0;
    if (rsp != 0) {
        std::memcpy(&ret0, reinterpret_cast<const void*>(rsp), 8);
        std::memcpy(&ret1, reinterpret_cast<const void*>(rsp + 8), 8);
    }
    LOGE("[Watch] RIP=0x%llx ejecutada (tid=%d) RCX=0x%llx RDX=0x%llx RSP=0x%llx [RSP]=0x%llx [RSP+8]=0x%llx",
         static_cast<unsigned long long>(guestRip), static_cast<int>(gettid()),
         static_cast<unsigned long long>(ctx_.gpr[1]), static_cast<unsigned long long>(ctx_.gpr[2]),
         static_cast<unsigned long long>(rsp), static_cast<unsigned long long>(ret0),
         static_cast<unsigned long long>(ret1));
    logGuestStackChain("[Watch]");
}

// ¿v cae en la sección de código de algún módulo PE mapeado? (cabeceras leídas con process_vm_readv; diagnóstico)
static bool probeModuleText(uint32_t v) {
    auto baseOf = s_moduleBaseOf.load(std::memory_order_acquire);
    if (baseOf == nullptr) return false;
    const uint64_t base = baseOf(v);
    if (base == 0) return v >= 0x12001000u && v < 0x13F1D000u;
    uint32_t lfanew = 0;
    if (!probeReadable(base + 0x3C, 4)) return false;
    std::memcpy(&lfanew, reinterpret_cast<const void*>(base + 0x3C), 4);
    uint32_t codeBase = 0, codeSize = 0;
    if (lfanew > 0x1000 || !probeReadable(base + lfanew + 0x2C, 4)) return false;
    std::memcpy(&codeSize, reinterpret_cast<const void*>(base + lfanew + 0x1C), 4);  // SizeOfCode
    std::memcpy(&codeBase, reinterpret_cast<const void*>(base + lfanew + 0x2C), 4);  // BaseOfCode
    return v >= base + codeBase && v < base + codeBase + codeSize;
}
void OryonEmitterBackend::logGuestStackChain(const char* tag) const {
    // Candidatos a dirección de retorno: qwords de la pila que caen en imágenes PE típicas
    // (0x140000000..0x1FFFFFFFF). Sirve para reconstruir la cadena de llamadas sin depurador.
    const uint64_t rsp = ctx_.gpr[4];
    if (rsp == 0) return;
    char chain[640];
    size_t len = 0;
    chain[0] = '\0';
    // process_vm_readv: devuelve error en vez de SIGSEGV si se pasa del tope de la pila (puede
    // llamarse desde otro hilo, p. ej. el vigilante de bloqueos).
    uint64_t words[200] = {};
    size_t count = 0;
    for (size_t want = 200; want > 0 && count == 0; want /= 2) {
        iovec local{words, want * 8};
        iovec remote{reinterpret_cast<void*>(rsp), want * 8};
        if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(want * 8)) count = want;
    }
    if (mode32_) {
        // 32 bits: palabras de 4 bytes; candidatas las que caen en código (ejecutable y DLL, por debajo del alias
        // de funciones nativas 0x7F000000).
        const auto* w32 = reinterpret_cast<const uint32_t*>(words);
        for (size_t i = 0; i < count * 2 && len + 16 < sizeof(chain); ++i) {
            const uint32_t v = w32[i];
            if (v >= 0x10000000u && v < 0x7F000000u && probeModuleText(v))
                len += static_cast<size_t>(std::snprintf(chain + len, sizeof(chain) - len, " +%x:%x", static_cast<unsigned>(i * 4), v));
        }
        LOGE("%s pila (ESP=0x%llx):%s", tag, static_cast<unsigned long long>(rsp), chain);
        return;
    }
    for (size_t i = 0; i < count && len + 24 < sizeof(chain); ++i) {
        const uint64_t v = words[i];
        if (v >= 0x140000000ull && v < 0x200000000ull) {
            len += static_cast<size_t>(std::snprintf(chain + len, sizeof(chain) - len, " +%x:%llx", static_cast<unsigned>(i * 8),
                                                     static_cast<unsigned long long>(v)));
        }
    }
    LOGE("%s pila (RSP=0x%llx):%s", tag, static_cast<unsigned long long>(rsp), chain);
}

TranslatedBlock* OryonEmitterBackend::translateBlock(uintptr_t guestRip) {
    if (guestRip == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = blockCache_.find(guestRip);
    if (cached != blockCache_.end()) return &cached->second;

    auto failed = failedBlocks_.find(guestRip);
    if (failed != failedBlocks_.end()) {
        lastError_ = failed->second;
        return nullptr;
    }

    size_t available = kMaxGuestBytesPerBlock;
    if (!codeRanges_.empty()) {
        bool inRange = false;
        for (const auto& r : codeRanges_) {
            if (guestRip >= r.base && guestRip < r.base + r.size) {
                inRange = true;
                const size_t remaining = r.base + r.size - guestRip;
                if (remaining < available) available = remaining;
                break;
            }
        }
        if (!inRange) {
            // Código creado después de arrancar este hilo (JIT de Mono en VirtualAlloc): los
            // rangos ejecutables son del proceso; se toman del backend principal.
            CodeRange shared{};
            OryonEmitterBackend* main = s_activeBackend;
            if (main != nullptr && main != this && main->findCodeRange(guestRip, shared)) {
                codeRanges_.push_back(shared);
                inRange = true;
                const size_t remaining = shared.base + shared.size - guestRip;
                if (remaining < available) available = remaining;
            }
        }
        if (!inRange) {
            lastError_ = "RIP fuera del rango de código PE / DLLs";
            return nullptr;
        }
        // Rangos contiguos (el montón de código de CoreCLR se confirma página a página): una
        // instrucción que cruza al rango siguiente no debe quedar truncada.
        for (int extend = 0; extend < 8 && available < kMaxGuestBytesPerBlock; ++extend) {
            const uintptr_t end = guestRip + available;
            bool grown = false;
            for (const auto& r : codeRanges_) {
                if (r.base <= end && r.base + r.size > end) {
                    available = std::min<size_t>(kMaxGuestBytesPerBlock, r.base + r.size - guestRip);
                    grown = true;
                    break;
                }
            }
            if (!grown) {
                CodeRange shared{};
                OryonEmitterBackend* main = s_activeBackend;
                if (main != nullptr && main != this && main->findCodeRange(end, shared) && shared.base <= end) {
                    codeRanges_.push_back(shared);
                    available = std::min<size_t>(kMaxGuestBytesPerBlock, shared.base + shared.size - guestRip);
                    grown = true;
                }
            }
            if (!grown) break;
        }
    } else if (codeSize_ != 0) {
        if (guestRip < codeBase_ || guestRip >= codeBase_ + codeSize_) {
            lastError_ = "RIP fuera del rango de código PE";
            return nullptr;
        }
        const size_t remaining = codeBase_ + codeSize_ - guestRip;
        if (remaining < available) available = remaining;
    } else if (codeRangeEnforced_) {
        lastError_ = "RIP sin rango de código tras removeExecutableRange";
        return nullptr;
    }

    // SMC: proteger ANTES de leer los bytes (una escritura posterior siempre se detecta).
    smcProtectForTranslation(guestRip, std::min<size_t>(available, kMaxGuestBytesPerBlock));
    const uint8_t* code = reinterpret_cast<const uint8_t*>(guestRip);
    const auto t0 = std::chrono::steady_clock::now();
    // Con una sonda en este RIP no se usa el bloque guardado (no la llevaría).
    TranslatedBlock* block = (!probeRvas().empty() && probeMatches(guestRip)) ? nullptr : loadPersistedLocked(guestRip, available);
    if (block == nullptr) block = compileLocked(code, available, guestRip);
    s_jitNs.fetch_add(static_cast<uint64_t>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                      std::memory_order_relaxed);
    ++statTranslations_;
    if (block == nullptr) {
        failedBlocks_[guestRip] = lastError_;
    } else {
        prof::registerBlock(reinterpret_cast<uintptr_t>(block->hostCode), block->hostSize, guestRip);
    }
    if (traceTranslations.load(std::memory_order_relaxed) > 0 && traceTranslations.fetch_sub(1) > 0) {
        LOGE("[TraceTrad] tid=%d RIP=0x%llx", static_cast<int>(gettid()), static_cast<unsigned long long>(guestRip));
    }
    logIfWatchedRip(guestRip);
    return block;
}

namespace {
// --- Código automodificable (SMC) -------------------------------------------------------
// Las páginas escribibles del huésped que contienen código traducido se protegen contra
// escritura. Una escritura (del huésped o de un thunk) provoca SIGSEGV/SEGV_ACCERR: el
// manejador devuelve el permiso original y anota la página en un anillo; cada hilo invalida sus
// traducciones de esa página antes del siguiente bloque (Mono parchea sus sitios de llamada sin
// FlushInstructionCache, como permite x86).
struct SmcSlot {
    std::atomic<uintptr_t> page{0};
    std::atomic<int> prot{0};
    std::atomic<int> state{0};  // 1 = protegida contra escritura por el JIT
};
constexpr size_t kSmcSlots = 1u << 16;
SmcSlot s_smcTable[kSmcSlots];
constexpr size_t kSmcRing = 4096;
std::atomic<uintptr_t> s_smcRing[kSmcRing];
std::atomic<uint64_t> s_smcWriteIdx{0};
std::atomic<int (*)(uintptr_t)> s_smcWritableHook{nullptr};

size_t smcPageSize() {
    static const size_t ps = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return ps;
}

SmcSlot* smcFind(uintptr_t page, bool insert) {
    size_t idx = static_cast<size_t>(((page / 4096u) * 0x9E3779B97F4A7C15ull) >> 48) & (kSmcSlots - 1);
    for (int probe = 0; probe < 128; ++probe, idx = (idx + 1) & (kSmcSlots - 1)) {
        SmcSlot& s = s_smcTable[idx];
        uintptr_t cur = s.page.load(std::memory_order_acquire);
        if (cur == page) return &s;
        if (cur == 0) {
            if (!insert) return nullptr;
            if (s.page.compare_exchange_strong(cur, page, std::memory_order_acq_rel)) return &s;
            if (cur == page) return &s;
        }
    }
    return nullptr;
}

// Seguro en señales: atómicos + mprotect.
bool smcHandleWriteFault(siginfo_t* info) {
    if (info == nullptr || info->si_code != SEGV_ACCERR) return false;
    const uintptr_t ps = smcPageSize();
    const uintptr_t page = reinterpret_cast<uintptr_t>(info->si_addr) & ~(ps - 1);
    SmcSlot* slot = smcFind(page, false);
    if (slot == nullptr) return false;
    const int prot = slot->prot.load(std::memory_order_acquire);
    if (slot->state.load(std::memory_order_acquire) != 1) {
        // Otro hilo escribió en la misma página a la vez y ya la desprotegió: el fallo es
        // obsoleto y basta reintentar la instrucción (antes se entregaba como AV real a Mono).
        // mprotect idempotente: garantiza que el reintento no vuelve a fallar. Si entretanto un
        // traductor la volvió a proteger (state 1), se respeta: el reintento pasará por el camino
        // normal y marcará la página sucia.
        if ((prot & PROT_WRITE) == 0 || mprotect(reinterpret_cast<void*>(page), ps, prot) != 0) return false;
        if (slot->state.load(std::memory_order_acquire) == 1) {
            mprotect(reinterpret_cast<void*>(page), ps, prot & ~PROT_WRITE);
        }
        return true;
    }
    if (mprotect(reinterpret_cast<void*>(page), ps, prot) != 0 &&
        mprotect(reinterpret_cast<void*>(page), ps, PROT_READ | PROT_WRITE) != 0) {
        return false;
    }
    slot->state.store(0, std::memory_order_release);
    const uint64_t idx = s_smcWriteIdx.fetch_add(1, std::memory_order_acq_rel);
    s_smcRing[idx % kSmcRing].store(page, std::memory_order_release);
    return true;
}

std::mutex s_threadRegistryMutex;
std::unordered_map<int, OryonEmitterBackend*> s_threadRegistry;
thread_local OryonEmitterBackend* t_currentBackend = nullptr;

// Informe de fallos del huésped: si un SIGSEGV/SIGBUS/SIGILL ocurre mientras este hilo ejecuta
// código invitado, se registra la RIP del bloque y los registros x86 antes de encadenar al
// manejador previo (debuggerd genera el tombstone como siempre).
struct sigaction s_prevSegv{}, s_prevBus{}, s_prevIll{};
std::once_flag s_crashReporterOnce;

int s_crashFd = -1;  // abierto al instalar: write() es async-signal-safe (logd puede descartar líneas)
// Copia legible sin run-as (APK de release): solo con `setprop debug.hyperplay.crashsd 1`.
int s_crashFdSd = -1;

void crashWrite(const char* s) {
    size_t n = 0;
    while (s[n] != 0) ++n;
    if (s_crashFd >= 0) (void)!write(s_crashFd, s, n);
    if (s_crashFdSd >= 0) (void)!write(s_crashFdSd, s, n);
}

void crashHex(char* out, uint64_t v) {  // 16 dígitos + NUL, sin snprintf
    static const char kHex[] = "0123456789abcdef";
    for (int i = 15; i >= 0; --i) {
        out[i] = kHex[v & 0xF];
        v >>= 4;
    }
    out[16] = 0;
}

// Aterrizaje tras un fallo capturado: se ejecuta ya fuera del manejador (sigreturn hacia aquí).
[[noreturn]] void guestFaultLanding() {
    t_currentBackend->landFromFault();
}

void guestCrashHandler(int sig, siginfo_t* info, void* uctx) {
    if (sig == SIGSEGV && smcHandleWriteFault(info)) return;  // escritura en código traducido
    OryonEmitterBackend* b = t_currentBackend;
    if (b != nullptr && b->captureFault(sig, info, uctx)) return;  // excepción para el huésped
    t_currentBackend = nullptr;  // sin reentrada si el propio informe falla
    if (b != nullptr) {
        const CpuContext& c = b->context();
        char h[17];
        crashWrite("=== HyperPlay guest crash ===\nsignal ");
        crashHex(h, static_cast<uint64_t>(sig));
        crashWrite(h);
        crashWrite(" fault ");
        crashHex(h, reinterpret_cast<uint64_t>(info != nullptr ? info->si_addr : nullptr));
        crashWrite(h);
        crashWrite(" tid ");
        crashHex(h, static_cast<uint64_t>(gettid()));
        crashWrite(h);
        crashWrite("\nRIP(bloque) ");
        crashHex(h, c.rip);
        crashWrite(h);
        crashWrite("\n");
        static const char* kNames[16] = {"RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
                                         "R8 ", "R9 ", "R10", "R11", "R12", "R13", "R14", "R15"};
        for (int i = 0; i < 16; ++i) {
            crashWrite(kNames[i]);
            crashWrite("=");
            crashHex(h, c.gpr[i]);
            crashWrite(h);
            crashWrite((i % 4) == 3 ? "\n" : " ");
        }
        if (c.rip != 0) {  // el bloque se tradujo desde RIP: esos bytes son legibles
            crashWrite("bytes@RIP:");
            const auto* p = reinterpret_cast<const uint8_t*>(c.rip);
            for (int i = 0; i < 48; ++i) {
                crashHex(h, p[i]);
                char byte[4] = {' ', h[14], h[15], 0};
                crashWrite(byte);
            }
            crashWrite("\n");
        }
        if (s_crashFd >= 0) fsync(s_crashFd);
    }
    struct sigaction* prev = sig == SIGSEGV ? &s_prevSegv : sig == SIGBUS ? &s_prevBus : &s_prevIll;
    if ((prev->sa_flags & SA_SIGINFO) != 0 && prev->sa_sigaction != nullptr) {
        prev->sa_sigaction(sig, info, uctx);  // ART/debuggerd: puede resolverlo y volver
        t_currentBackend = b;
        return;
    }
    if (prev->sa_handler == SIG_IGN) return;  // vía libsigchain: solo informar
    if (prev->sa_handler != SIG_DFL && prev->sa_handler != nullptr) {
        prev->sa_handler(sig);
        return;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

// libsigchain (ART) intercepta sigaction(): en una app el manejador debe registrarse como
// "especial" para ejecutarse antes que los de ART/debuggerd. Devolver false continúa la cadena.
struct SigchainAction {
    bool (*sc_sigaction)(int, siginfo_t*, void*);
    sigset_t sc_mask;
    uint64_t sc_flags;
};
using AddSpecialSignalHandlerFnT = void (*)(int, SigchainAction*);

bool guestCrashSpecialHandler(int sig, siginfo_t* info, void* uctx) {
    if (sig == SIGSEGV && smcHandleWriteFault(info)) return true;  // cualquier hilo
    OryonEmitterBackend* b = t_currentBackend;
    if (b != nullptr && b->captureFault(sig, info, uctx)) return true;  // se entrega al huésped tras sigreturn
    // Vigía de escritura (diagnóstico): escritura desde código nativo (thunks, hilos propios).
    if (sig == SIGSEGV && info != nullptr && info->si_code == SEGV_ACCERR &&
        OryonEmitterBackend::handleWriteWatch(reinterpret_cast<uint64_t>(info->si_addr),
                                              static_cast<uint64_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.regs[30]))) {
        return true;  // página ya escribible: la instrucción se repite
    }
    if (b == nullptr) return false;  // señales de ART/otros hilos: no tocar
    s_prevSegv.sa_sigaction = nullptr;  // no encadenar dentro de guestCrashHandler
    s_prevBus.sa_sigaction = nullptr;
    s_prevIll.sa_sigaction = nullptr;
    s_prevSegv.sa_flags = s_prevBus.sa_flags = s_prevIll.sa_flags = 0;
    s_prevSegv.sa_handler = s_prevBus.sa_handler = s_prevIll.sa_handler = SIG_IGN;
    guestCrashHandler(sig, info, uctx);  // solo informa (prev = SIG_IGN -> vuelve)
    t_currentBackend = b;
    return false;  // sigchain sigue: ART no lo reclama -> debuggerd genera el tombstone
}

void installGuestCrashReporter() {
    std::call_once(s_crashReporterOnce, [] {
        // Directorio privado de la app (apk debug: legible con `run-as`); en la suite nativa
        // /data/local/tmp.
        s_crashFd = open("/data/data/com.murdoc820.hyperplay/files/guest_crash.txt",
                         O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (s_crashFd < 0) s_crashFd = open("/data/local/tmp/guest_crash.txt", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        char sd[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.crashsd", sd) > 0 && sd[0] == '1') {
            s_crashFdSd = open("/sdcard/HyperPlay/guest_crash.txt", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        }
        auto addSpecial = reinterpret_cast<AddSpecialSignalHandlerFnT>(dlsym(RTLD_DEFAULT, "AddSpecialSignalHandlerFn"));
        if (addSpecial != nullptr) {
            static SigchainAction action{};
            action.sc_sigaction = &guestCrashSpecialHandler;
            sigemptyset(&action.sc_mask);
            action.sc_flags = 0;
            addSpecial(SIGSEGV, &action);
            addSpecial(SIGBUS, &action);
            addSpecial(SIGILL, &action);
            LOGI("Informe de fallos del huésped registrado vía libsigchain");
            return;
        }
        struct sigaction sa {};
        sa.sa_sigaction = &guestCrashHandler;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, &s_prevSegv);
        sigaction(SIGBUS, &sa, &s_prevBus);
        sigaction(SIGILL, &sa, &s_prevIll);
    });
}
}  // namespace

bool OryonEmitterBackend::resolveGuestFault(uintptr_t hostPc, uintptr_t& guestRip) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostIndex_.upper_bound(hostPc);
    if (it == hostIndex_.begin()) return false;
    --it;
    auto blockIt = blockCache_.find(it->second);
    if (blockIt == blockCache_.end()) return false;
    const TranslatedBlock& block = blockIt->second;
    const auto start = reinterpret_cast<uintptr_t>(block.hostCode);
    if (start != it->first || hostPc < start || hostPc >= start + block.hostSize) return false;
    const auto offset = static_cast<uint32_t>(hostPc - start);
    guestRip = block.guestRip;
    for (const auto& [hostOff, guestOff] : block.instrMap) {
        if (hostOff > offset) break;
        guestRip = block.guestRip + guestOff;
    }
    return true;
}

bool OryonEmitterBackend::captureFault(int sig, siginfo_t* info, void* uctx) {
    if (uctx == nullptr || faultJmp_ == nullptr || arena_ == nullptr || pendingFault_.valid) return false;
    if (sig != SIGSEGV && sig != SIGBUS && sig != SIGILL) return false;
    auto* uc = static_cast<ucontext_t*>(uctx);
    const auto pc = static_cast<uintptr_t>(uc->uc_mcontext.pc);
    const auto arenaBase = reinterpret_cast<uintptr_t>(arena_);
    if (pc < arenaBase || pc >= arenaBase + arenaSize_) return false;  // fallo del host, no del huésped

    PendingFault& f = pendingFault_;
    f.rip = ctx_.rip;
    (void)resolveGuestFault(pc, f.rip);
    for (int i = 0; i < 16; ++i) f.gpr[i] = uc->uc_mcontext.regs[i];  // X0..X15 = RAX..R15
    std::memcpy(f.simd, ctx_.simd, sizeof(f.simd));
    uint64_t esr = 0;
    // __reserved: registros SIMD (V0-7 = XMM0-7, V16-23 = XMM8-15) y ESR del fallo.
    const uint8_t* p = reinterpret_cast<const uint8_t*>(uc->uc_mcontext.__reserved);
    const uint8_t* end = p + sizeof(uc->uc_mcontext.__reserved);
    while (p + sizeof(_aarch64_ctx) <= end) {
        const auto* head = reinterpret_cast<const _aarch64_ctx*>(p);
        if (head->magic == 0 || head->size == 0) break;
        if (head->magic == FPSIMD_MAGIC) {
            const auto* fp = reinterpret_cast<const fpsimd_context*>(head);
            for (int i = 0; i < 8; ++i) {
                std::memcpy(f.simd[i], &fp->vregs[i], 16);
                std::memcpy(f.simd[i + 8], &fp->vregs[i + 16], 16);
            }
        } else if (head->magic == ESR_MAGIC) {
            esr = reinterpret_cast<const esr_context*>(head)->esr;
        }
        p += head->size;
    }
    const uint32_t ec = static_cast<uint32_t>(esr >> 26) & 0x3F;
    if (ec == 0x20 || ec == 0x21) f.access = 8;                       // abort de instrucción
    else f.access = (ec == 0x24 || ec == 0x25) && ((esr >> 6) & 1) ? 1 : 0;  // WnR
    f.code = sig == SIGILL ? 0xC000001Du : 0xC0000005u;
    f.address = reinterpret_cast<uint64_t>(info != nullptr ? info->si_addr : nullptr);
    f.valid = true;
    uc->uc_mcontext.pc = reinterpret_cast<uint64_t>(&guestFaultLanding);
    return true;
}

void OryonEmitterBackend::landFromFault() {
    std::longjmp(*faultJmp_, 1);
}

// --- Vigía de escritura (diagnóstico) -------------------------------------------------------
namespace {
std::atomic<uintptr_t> s_watchLo{0}, s_watchHi{0}, s_watchExactLo{0}, s_watchExactHi{0};
std::atomic<bool> s_watchArmed{false};
std::atomic<int> s_watchLogs{0};
std::atomic<uint64_t> s_watchHits{0}, s_watchArms{0};
}  // namespace

// Vigía relativo (setprop debug.hyperplay.wwrel <offset>,<bytes>,<tamaño del bloque>): al crearse un bloque
// grande del montón con ese tamaño, vigila [bloque+offset, +bytes) desde ese mismo instante.
static void relWatchHook(void* block, size_t total) {
    static const std::array<uint64_t, 4> cfg = [] {
        std::array<uint64_t, 4> c{0, 0, 0, 0};
        char buf[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.wwrel", buf) <= 0) return c;
        char* p = buf;
        for (int i = 0; i < 4 && *p != '\0'; ++i) {
            char* end = nullptr;
            c[i] = std::strtoull(p, &end, 16);
            p = (*end == ',') ? end + 1 : end;
        }
        return c;
    }();
    static std::atomic<uint64_t> s_seen{0};
    if (cfg[2] == 0 || ((total + 0xFFFF) & ~size_t{0xFFFF}) != cfg[2] || s_watchExactLo.load() != 0) return;
    LOGI("[Vigía] bloque candidato %llu en 0x%lx", static_cast<unsigned long long>(s_seen.load()), static_cast<unsigned long>(reinterpret_cast<uintptr_t>(block)));
    if (s_seen.fetch_add(1) != cfg[3]) return;  // 4.º valor: cuál de los bloques de ese tamaño (0 = el primero)
    const uintptr_t addr = reinterpret_cast<uintptr_t>(block) + cfg[0];
    const uintptr_t lo = addr & ~uintptr_t{0xFFF};
    s_watchExactLo.store(addr);
    s_watchExactHi.store(addr + cfg[1]);
    s_watchLo.store(lo);
    s_watchHi.store(lo + 0x1000);
    mprotect(reinterpret_cast<void*>(lo), 0x1000, PROT_READ);
    s_watchArmed.store(true);
    LOGI("[Vigía] bloque 0x%lx (0x%zx bytes): vigilando 0x%lx", static_cast<unsigned long>(reinterpret_cast<uintptr_t>(block)), total,
         static_cast<unsigned long>(addr));
    std::thread([] {
        for (int i = 0; i < 100000; ++i) {
            if (!s_watchArmed.load() && mprotect(reinterpret_cast<void*>(s_watchLo.load()), 0x1000, PROT_READ) == 0) s_watchArmed.store(true);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }).detach();
}

void OryonEmitterBackend::armWriteWatchFromProperty() {
    lowmem::g_largeBlockHook = &relWatchHook;
#if defined(__ANDROID__)
    char buf[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.hyperplay.wwatch", buf) <= 0) return;
    char* end = nullptr;
    const uintptr_t addr = std::strtoull(buf, &end, 16);
    const size_t len = (end != nullptr && *end == ',') ? std::strtoull(end + 1, nullptr, 16) : 0x1000;
    if (addr == 0 || len == 0) return;
    const uintptr_t lo = addr & ~uintptr_t{0xFFF};
    const uintptr_t hi = (addr + len + 0xFFF) & ~uintptr_t{0xFFF};
    s_watchLo.store(lo);
    s_watchHi.store(hi);
    s_watchExactLo.store(addr);
    s_watchExactHi.store(addr + len);
    // Si la página aún no existe, mmapBelow4G la protege en el mismo instante en que se crea.
    lowmem::g_watchArmedFlag = &s_watchArmed;
    lowmem::g_watchPageHi.store(hi);
    lowmem::g_watchPageLo.store(lo);
    LOGI("[Vigía] escrituras en 0x%lx..0x%lx", static_cast<unsigned long>(lo), static_cast<unsigned long>(hi));
    std::thread([] {
        for (int i = 0; i < 200000; ++i) {  // ~60 s
            // La página puede no existir aún: se reintenta hasta que mprotect funcione.
            if (!s_watchArmed.load() &&
                mprotect(reinterpret_cast<void*>(s_watchLo.load()), s_watchHi.load() - s_watchLo.load(), PROT_READ) == 0) {
                s_watchArmed.store(true);
                if (s_watchArms.fetch_add(1) == 0) LOGI("[Vigía] armado");
            }
            if ((i % 3333) == 0) {
                // ¿Sigue protegida? (un VirtualAlloc/mmap encima la desprotege sin avisar)
                LOGI("[Vigía] armados=%llu escrituras=%llu", static_cast<unsigned long long>(s_watchArms.load()),
                     static_cast<unsigned long long>(s_watchHits.load()));
            }
            std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
        mprotect(reinterpret_cast<void*>(s_watchLo.load()), s_watchHi.load() - s_watchLo.load(), PROT_READ | PROT_WRITE);
        s_watchLo.store(0);
        s_watchHi.store(0);
    }).detach();
#endif
}

void OryonEmitterBackend::noteUnmap(uint64_t base, uint64_t size, const char* who, uint64_t rip) {
    // Rango propio: setprop debug.hyperplay.uwatch <dir>,<bytes> (sin proteger páginas).
    static const std::pair<uint64_t, uint64_t> s_range = [] {
        char buf[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.uwatch", buf) <= 0) return std::pair<uint64_t, uint64_t>{0, 0};
        char* end = nullptr;
        const uint64_t a = std::strtoull(buf, &end, 16);
        const uint64_t l = (end != nullptr && *end == ',') ? std::strtoull(end + 1, nullptr, 16) : 0x1000;
        return std::pair<uint64_t, uint64_t>{a, a + l};
    }();
    const uint64_t lo = s_range.first, hi = s_range.second;
    if (lo == 0 || base + size <= lo || base >= hi) return;
    LOGI("[Vigía] %s libera 0x%llx..0x%llx (contiene la zona vigilada) desde RIP=0x%llx", who,
         static_cast<unsigned long long>(base), static_cast<unsigned long long>(base + size), static_cast<unsigned long long>(rip));
    s_watchArmed.store(false);
}

bool OryonEmitterBackend::handleWriteWatch(uint64_t address, uint64_t rip) {
    const uintptr_t lo = s_watchLo.load(), hi = s_watchHi.load();
    if (lo == 0 || address < lo || address >= hi) return false;
    s_watchHits.fetch_add(1);
    if (address >= s_watchExactLo.load() && address < s_watchExactHi.load() && s_watchLogs.fetch_add(1) < 20000)
        LOGI("[Vigía] escritura en 0x%llx desde RIP=0x%llx (valor previo 0x%x)", static_cast<unsigned long long>(address),
             static_cast<unsigned long long>(rip), *reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(address & ~uint64_t{3})));
    mprotect(reinterpret_cast<void*>(lo), hi - lo, PROT_READ | PROT_WRITE);
    s_watchArmed.store(false);
    return true;
}

bool OryonEmitterBackend::deliverPendingFault() {
    if (!pendingFault_.valid) return false;
    const PendingFault f = pendingFault_;
    pendingFault_.valid = false;
    // Vigía de escritura: se registra y se repite la instrucción (la página ya es escribible).
    if (f.access == 1 && handleWriteWatch(f.address, f.rip)) {
        std::memcpy(ctx_.gpr, f.gpr, sizeof(ctx_.gpr));
        std::memcpy(ctx_.simd, f.simd, sizeof(ctx_.simd));
        ctx_.rip = f.rip;
        if (mode32_) for (int i = 0; i < 16; ++i) ctx_.gpr[i] &= 0xFFFFFFFFull;
        return true;
    }
    if (exceptionDepth_ >= 8) return false;  // excepciones anidadas sin fin
    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& x) : d(x) { ++d; }
        ~DepthGuard() { --d; }
    } depthGuard(exceptionDepth_);

    static std::atomic<int> s_faultLogs{0};
    if (s_faultLogs.fetch_add(1) < 60) {
        LOGI("Fallo del huésped 0x%08x en RIP=0x%llx dir=0x%llx (%s)", f.code, static_cast<unsigned long long>(f.rip),
             static_cast<unsigned long long>(f.address), f.access == 1 ? "escritura" : f.access == 8 ? "ejecución" : "lectura");
    }
    // Estado exacto en la instrucción que falló.
    std::memcpy(ctx_.gpr, f.gpr, sizeof(ctx_.gpr));
    std::memcpy(ctx_.simd, f.simd, sizeof(ctx_.simd));
    ctx_.rip = f.rip;

    // 32 bits: SEH x86 (VEH + cadena fs:[0]) ejecutado como código del invitado desde un
    // despachador; el hilo sigue en ese despachador. El camino de 64 bits de abajo no cambia.
    if (mode32_) {
        // Diagnóstico: bits altos de los registros (en 32 bits deben ser 0; si no, una dirección
        // calculada con el registro completo lee memoria ajena).
        uint64_t dirty = 0;
        for (int i = 0; i < 8; ++i) dirty |= ctx_.gpr[i] >> 32;
        if (dirty != 0) {
            LOGW("Fallo de 32 bits con bits altos en registros: RAX=%llx RCX=%llx RDX=%llx RBX=%llx RSP=%llx RBP=%llx RSI=%llx RDI=%llx",
                 static_cast<unsigned long long>(ctx_.gpr[0]), static_cast<unsigned long long>(ctx_.gpr[1]),
                 static_cast<unsigned long long>(ctx_.gpr[2]), static_cast<unsigned long long>(ctx_.gpr[3]),
                 static_cast<unsigned long long>(ctx_.gpr[4]), static_cast<unsigned long long>(ctx_.gpr[5]),
                 static_cast<unsigned long long>(ctx_.gpr[6]), static_cast<unsigned long long>(ctx_.gpr[7]));
        }
        for (int i = 0; i < 16; ++i) ctx_.gpr[i] &= 0xFFFFFFFFull;
        return NativeDllInterceptor::prepareGuestException32(ctx_, f.code, static_cast<uint32_t>(f.address),
                                                            static_cast<uint32_t>(f.access));
    }

    // CONTEXT + EXCEPTION_RECORD + EXCEPTION_POINTERS bajo RSP (red zone de 128 B, alineado a 16),
    // como hace KiUserExceptionDispatcher.
    const uintptr_t faultRsp = ctx_.gpr[4];
    const size_t frameSize = win64ctx::kContextSize + 0xA0 + 0x10;
    const uintptr_t base = (faultRsp - 128 - frameSize) & ~static_cast<uintptr_t>(0xF);
    if (!isValidStackAddress(base, frameSize)) {
        LOGE("Excepción 0x%08x en RIP=0x%llx: RSP=0x%llx fuera de la pila del huésped", f.code,
             static_cast<unsigned long long>(f.rip), static_cast<unsigned long long>(faultRsp));
        return false;
    }
    auto* context = reinterpret_cast<uint8_t*>(base);
    auto* record = context + win64ctx::kContextSize;
    auto* pointers = reinterpret_cast<uint64_t*>(record + 0xA0);
    std::memset(context, 0, frameSize);
    win64ctx::toContext(ctx_, context, win64ctx::kContextAll);
    if (f.code == win64ctx::kExceptionAccessViolation) {
        win64ctx::fillExceptionRecord(record, f.code, f.rip, f.access, f.address, 2);
    } else {
        win64ctx::fillExceptionRecord(record, f.code, f.rip, 0, 0, 0);
    }
    pointers[0] = reinterpret_cast<uint64_t>(record);
    pointers[1] = reinterpret_cast<uint64_t>(context);

    ctx_.gpr[4] = base;  // los manejadores se ejecutan por debajo de estas estructuras
    const int32_t verdict = NativeDllInterceptor::dispatchGuestException(*this, reinterpret_cast<uint64_t>(pointers));
    if (verdict == win64ctx::kExceptionContinueExecution) {
        win64ctx::fromContext(ctx_, context, true);
        return true;
    }
    ctx_.gpr[4] = faultRsp;
    return false;
}

void OryonEmitterBackend::setSmcWritableHook(int (*hook)(uintptr_t page)) {
    s_smcWritableHook.store(hook, std::memory_order_release);
}

void OryonEmitterBackend::smcForgetRange(uintptr_t base, size_t size) {
    const uintptr_t ps = smcPageSize();
    for (uintptr_t p = base & ~(ps - 1); p < base + size; p += ps) {
        if (SmcSlot* s = smcFind(p, false); s != nullptr) s->state.store(0, std::memory_order_release);
    }
}

// Antes de leer bytes para traducir: protege contra escritura las páginas escribibles del rango.
void OryonEmitterBackend::smcProtectForTranslation(uintptr_t start, size_t len) {
    auto hook = s_smcWritableHook.load(std::memory_order_acquire);
    if (hook == nullptr || len == 0) return;
    const uintptr_t ps = smcPageSize();
    for (uintptr_t p = start & ~(ps - 1); p < start + len; p += ps) {
        const int prot = hook(p);
        if (prot < 0 || (prot & PROT_WRITE) == 0) continue;
        SmcSlot* s = smcFind(p, true);
        if (s == nullptr) continue;
        s->prot.store(prot, std::memory_order_release);
        int expected = 0;
        if (s->state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            if (mprotect(reinterpret_cast<void*>(p), ps, prot & ~PROT_WRITE) != 0) s->state.store(0);
        }
    }
}

void OryonEmitterBackend::processSmcDirty() {
    const uint64_t end = s_smcWriteIdx.load(std::memory_order_acquire);
    if (end == smcCursor_) return;
    const uintptr_t ps = smcPageSize();
    bool flushAll = end - smcCursor_ > kSmcRing;
    for (uint64_t i = smcCursor_; !flushAll && i < end; ++i) {
        const uintptr_t page = s_smcRing[i % kSmcRing].load(std::memory_order_acquire);
        if (page == 0) {
            flushAll = true;
            break;
        }
        const uintptr_t from = page > kMaxGuestBytesPerBlock ? page - kMaxGuestBytesPerBlock : 0;
        invalidateCodeRange(from, (page - from) + ps);
    }
    if (flushAll) {
        std::lock_guard<std::mutex> lock(mutex_);
        unlinkAllLocked();
        blockCache_.clear();
        failedBlocks_.clear();
        fastGen_.fetch_add(1, std::memory_order_acq_rel);
    }
    smcCursor_ = end;
}

void OryonEmitterBackend::forEachBackend(const std::function<void(OryonEmitterBackend&)>& fn) {
    // El registro se mantiene bloqueado durante el recorrido: un hilo que termina (unregisterCurrentThread)
    // espera a que acabe en vez de destruir su backend mientras otro hilo lo está usando (con el ExitThread de
    // 32 bits, VirtualFree se quedaba bloqueado en el mutex de un backend ya destruido: DmC al cargar un nivel).
    std::lock_guard<std::mutex> lock(s_threadRegistryMutex);
    std::vector<OryonEmitterBackend*> all;
    for (const auto& kv : s_threadRegistry) all.push_back(kv.second);
    if (s_activeBackend != nullptr && std::find(all.begin(), all.end(), s_activeBackend) == all.end()) {
        all.push_back(s_activeBackend);
    }
    for (OryonEmitterBackend* b : all) fn(*b);
}

void OryonEmitterBackend::addExecutableRangeAll(uintptr_t base, size_t size) {
    forEachBackend([&](OryonEmitterBackend& b) { b.addExecutableRange(base, size); });
}

void OryonEmitterBackend::removeExecutableRangeAll(uintptr_t base, size_t size) {
    forEachBackend([&](OryonEmitterBackend& b) { b.removeExecutableRange(base, size); });
}

void OryonEmitterBackend::invalidateCodeRangeAll(uintptr_t base, size_t size) {
    forEachBackend([&](OryonEmitterBackend& b) { b.invalidateCodeRange(base, size); });
}

OryonEmitterBackend* OryonEmitterBackend::forThread(int tid) {
    std::lock_guard<std::mutex> lock(s_threadRegistryMutex);
    auto it = s_threadRegistry.find(tid);
    return it != s_threadRegistry.end() ? it->second : nullptr;
}

void OryonEmitterBackend::registerCurrentThread() {
    hostTid_ = static_cast<int>(gettid());
    t_currentBackend = this;
    installGuestCrashReporter();
    std::lock_guard<std::mutex> lock(s_threadRegistryMutex);
    s_threadRegistry[hostTid_] = this;
}

void OryonEmitterBackend::unregisterCurrentThread() {
    if (t_currentBackend == this) t_currentBackend = nullptr;
    std::lock_guard<std::mutex> lock(s_threadRegistryMutex);
    auto it = s_threadRegistry.find(hostTid_);
    if (it != s_threadRegistry.end() && it->second == this) s_threadRegistry.erase(it);
}

uint32_t OryonEmitterBackend::requestSuspend() {
    std::lock_guard<std::mutex> lock(suspendMutex_);
    return suspendCount_.fetch_add(1, std::memory_order_seq_cst);
}

uint32_t OryonEmitterBackend::requestResume() {
    std::lock_guard<std::mutex> lock(suspendMutex_);
    const uint32_t previous = suspendCount_.load(std::memory_order_seq_cst);
    if (previous > 0) suspendCount_.store(previous - 1, std::memory_order_seq_cst);
    suspendCv_.notify_all();
    return previous;
}

bool OryonEmitterBackend::waitUntilSuspended(uint32_t timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    std::unique_lock<std::mutex> lock(suspendMutex_);
    while (runState_.load(std::memory_order_seq_cst) == kRunning) {
        // kInNative no notifica: se sondea en intervalos cortos.
        if (suspendCv_.wait_until(lock, std::min(deadline, std::chrono::steady_clock::now() +
                                                              std::chrono::milliseconds(1))) ==
                std::cv_status::timeout &&
            std::chrono::steady_clock::now() >= deadline) {
            return runState_.load(std::memory_order_seq_cst) != kRunning;
        }
    }
    return true;
}

void OryonEmitterBackend::suspensionPoint() {
    // Comprobaciones rápidas relajadas (camino caliente: una por bloque); la sincronización real
    // se hace dentro (acquire en processSmcDirty, mutex en la suspensión).
    if (s_smcWriteIdx.load(std::memory_order_relaxed) != smcCursor_) processSmcDirty();
    if (suspendCount_.load(std::memory_order_relaxed) == 0) return;
    std::unique_lock<std::mutex> lock(suspendMutex_);
    if (suspendCount_.load(std::memory_order_seq_cst) == 0) return;
    runState_.store(kParked, std::memory_order_seq_cst);
    suspendCv_.notify_all();
    suspendCv_.wait(lock, [this] { return suspendCount_.load(std::memory_order_seq_cst) == 0 || stopped_; });
    runState_.store(kRunning, std::memory_order_seq_cst);
}

bool OryonEmitterBackend::dispatchNativeThunk() {
    suspensionPoint();
    if (nativeThunk_ == nullptr) return false;
    runState_.store(kInNative, std::memory_order_seq_cst);
    const bool handled = nativeThunk_(nativeThunkUser_, ctx_.rip, ctx_);
    runState_.store(kRunning, std::memory_order_seq_cst);
    // El thunk emuló el RET del huésped: si vuelve a la entrada de arriba de la pila de retornos,
    // se saca (si no, los RET siguientes fallarían la predicción uno a uno).
    if (handled) {
        const uint64_t top = ctx_.rasTop & 63u;
        if (ctx_.ras[top * 2] == ctx_.rip) ctx_.rasTop -= 1;
    }
    return handled;
}

int64_t OryonEmitterBackend::executeEntry(void* entryPoint) {
    if (entryPoint == nullptr) {
        lastError_ = "entryPoint nulo";
        return -1;
    }
    registerCurrentThread();
    struct Unregister {
        OryonEmitterBackend* self;
        ~Unregister() { self->unregisterCurrentThread(); }
    } unregisterGuard{this};

    if (ctx_.gpr[4] == 0 && stackBase_ != nullptr && stackSize_ >= (mode32_ ? 64u : 128u)) {
        const uintptr_t top = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        if (mode32_) {
            ctx_.gpr[4] = top;
            std::memset(reinterpret_cast<void*>(top), 0, 64);
        } else {
            ctx_.gpr[4] = top - 40;
            std::memset(reinterpret_cast<void*>(ctx_.gpr[4]), 0, 104);
        }
    }

    stopped_ = false;
    ctx_.rip = reinterpret_cast<uintptr_t>(entryPoint);
    NativeDllInterceptor::setGuestTeb(teb_, !mode32_);
    LOGI("executeEntry: entry=%p RSP=%p (RSP %% 16 = %u)",
         entryPoint, reinterpret_cast<void*>(ctx_.gpr[4]),
         static_cast<unsigned>(ctx_.gpr[4] & 0xFu));

    // Punto de retorno para fallos hardware del código invitado (VEH).
    std::jmp_buf faultJb;
    FaultScope faultScope(this, &faultJb);
    // Tras un fallo a mitad de cadena, exitSite puede ser de una salida anterior ya enlazada.
    if (setjmp(faultJb) != 0 && ((ctx_.exitSite = 0), !deliverPendingFault())) {
        lastError_ = "excepción hardware del huésped sin manejar";
        LOGE("executeEntry: %s en RIP=%p", lastError_.c_str(), reinterpret_cast<void*>(ctx_.rip));
        return -4;
    }

    using BlockFn = void (*)(CpuContext*);
    uintptr_t lastBlockRip = 0;  // diagnóstico: de dónde se llegó a una RIP intraducible
    for (uint64_t step = 0; !stopped_; ++step) {
        if (ctx_.rip == 0) {
            LOGI("executeEntry: fin del hilo (RIP=0) RAX=0x%llx RSP=0x%llx bloque previo=%p",
                 static_cast<unsigned long long>(ctx_.gpr[0]), static_cast<unsigned long long>(ctx_.gpr[4]),
                 reinterpret_cast<void*>(lastBlockRip));
            return static_cast<int64_t>(ctx_.gpr[0]);
        }
        // Suspensión y SMC antes de buscar el bloque: una invalidación pendiente nunca ejecuta
        // la traducción vieja.
        // Salida enlazable del bloque anterior (si la hubo): su destino es ctx_.rip. Si la RIP
        // cambia durante la suspensión (SetThreadContext) el sitio ya no corresponde.
        uint64_t exitSite = ctx_.exitSite;
        ctx_.exitSite = 0;
        const uint64_t ripAtExit = ctx_.rip;
        const uint64_t arenaGenAtExit = arenaGen_;  // si la arena se vacía, el sitio ya no existe
        suspensionPoint();
        if (ctx_.rip != ripAtExit) exitSite = 0;
        const bool chain = chainBlocks.load(std::memory_order_relaxed);
        const auto armChain = [&](const void* host) {
            if (!chain) {
                ctx_.chainBudget = 0;  // toda salida enlazable vuelve aquí
                return;
            }
            if (exitSite != 0 && arenaGen_ == arenaGenAtExit) linkExit(exitSite, host, ctx_.rip);
            ctx_.chainBudget = kChainBudget;
            ctx_.smcCursor = smcCursor_;
            ctx_.smcIdxAddr = reinterpret_cast<uint64_t>(&s_smcWriteIdx);
            ctx_.fastCacheAddr = reinterpret_cast<uint64_t>(fastCache_.get());
            ctx_.fastGenCopy = fastGen_.load(std::memory_order_acquire);
        };
        // Camino caliente: bloque ya traducido de este hilo, sin locks. Un trampolín nunca
        // está en esta caché (los thunks se atienden antes de traducir).
        if (void* host = fastCache_ ? fastLookup(ctx_.rip) : nullptr; host != nullptr) {
            lastBlockRip = ctx_.rip;
            armChain(host);
            reinterpret_cast<BlockFn>(host)(&ctx_);
        } else {
            if (dispatchNativeThunk()) {
                ++statThunks_;
                continue;
            }
            ++statSlowPath_;

            TranslatedBlock* block = translateBlock(ctx_.rip);
            if (block == nullptr || block->hostCode == nullptr) {
                LOGE("No se pudo traducir el bloque en RIP=%p: %s (bloque previo RIP=%p)",
                     reinterpret_cast<void*>(ctx_.rip), lastError_.c_str(), reinterpret_cast<void*>(lastBlockRip));
                LOGE("  RAX=%llx RCX=%llx RDX=%llx RBX=%llx RSP=%llx RBP=%llx RSI=%llx RDI=%llx",
                     static_cast<unsigned long long>(ctx_.gpr[0]), static_cast<unsigned long long>(ctx_.gpr[1]),
                     static_cast<unsigned long long>(ctx_.gpr[2]), static_cast<unsigned long long>(ctx_.gpr[3]),
                     static_cast<unsigned long long>(ctx_.gpr[4]), static_cast<unsigned long long>(ctx_.gpr[5]),
                     static_cast<unsigned long long>(ctx_.gpr[6]), static_cast<unsigned long long>(ctx_.gpr[7]));
                LOGE("  R8=%llx R9=%llx R10=%llx R11=%llx R12=%llx R13=%llx R14=%llx R15=%llx",
                     static_cast<unsigned long long>(ctx_.gpr[8]), static_cast<unsigned long long>(ctx_.gpr[9]),
                     static_cast<unsigned long long>(ctx_.gpr[10]), static_cast<unsigned long long>(ctx_.gpr[11]),
                     static_cast<unsigned long long>(ctx_.gpr[12]), static_cast<unsigned long long>(ctx_.gpr[13]),
                     static_cast<unsigned long long>(ctx_.gpr[14]), static_cast<unsigned long long>(ctx_.gpr[15]));
                if (lastBlockRip != 0) {
                    char hex[64 * 3 + 1] = {};
                    const auto* p = reinterpret_cast<const uint8_t*>(lastBlockRip);
                    for (int i = 0; i < 64; ++i) std::snprintf(hex + i * 3, 4, "%02x ", p[i]);
                    LOGE("  bytes@bloque previo: %s", hex);
                }
                if (isValidStackAddress(ctx_.gpr[4], 32)) {
                    const auto* s = reinterpret_cast<const uint64_t*>(ctx_.gpr[4]);
                    LOGE("  [RSP]=%llx [RSP+8]=%llx [RSP+16]=%llx [RSP+24]=%llx", static_cast<unsigned long long>(s[0]),
                         static_cast<unsigned long long>(s[1]), static_cast<unsigned long long>(s[2]),
                         static_cast<unsigned long long>(s[3]));
                }
                return -2;
            }
            lastBlockRip = block->guestRip;
            // Funciones del juego con hook (Unity MemoryManager) se registran como trampolín:
            // no se cachean para que el hook se evalúe en cada llamada.
            const bool hooked = NativeDllInterceptor::isRegisteredTrampoline(block->guestRip);
            if (fastCache_ && !hooked) {
                fastInsert(block->guestRip, block->hostCode);
            }
            if (hooked) ctx_.chainBudget = 0;  // nunca enlazar hacia ni desde un hook
            else armChain(block->hostCode);
            reinterpret_cast<BlockFn>(block->hostCode)(&ctx_);
        }

        if ((step & 0x1FFFFF) == 0 && step > 0) {
            // Deltas desde la línea anterior: traducciones nuevas, búsquedas lentas, thunks y
            // fallos SMC del proceso, más el tiempo transcurrido (rendimiento real del hilo).
            static thread_local uint64_t lastT = 0, lastS = 0, lastK = 0, lastSmc = 0;
            static thread_local auto lastTime = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            const uint64_t smc = s_smcWriteIdx.load(std::memory_order_relaxed);
            const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
            const JitDiskCache& disk = JitDiskCache::instance();
            const uint64_t x87Fb = x87::s_fallbackCalls.load(std::memory_order_relaxed);
            LOGI("Oryon: Ejecutados %llu bloques JIT (RIP=%p) %lld ms trad=+%llu lento=+%llu thunks=+%llu smc=+%llu enlaces=%zu lejanos=%llu arena=%zuMB x87fb=%llu "
                 "| JIT total %llu ms (todos los hilos), disco: %llu cargados aqui, %llu aciertos / %llu fallos / %llu guardados",
                 static_cast<unsigned long long>(step), reinterpret_cast<void*>(ctx_.rip), ms,
                 static_cast<unsigned long long>(statTranslations_ - lastT),
                 static_cast<unsigned long long>(statSlowPath_ - lastS),
                 static_cast<unsigned long long>(statThunks_ - lastK),
                 static_cast<unsigned long long>(smc - lastSmc), links_.size(),
                 static_cast<unsigned long long>(farLinks_), jitOffset_ >> 20,
                 static_cast<unsigned long long>(x87Fb),
                 static_cast<unsigned long long>(s_jitNs.load(std::memory_order_relaxed) / 1000000u),
                 static_cast<unsigned long long>(statAotLoads_), static_cast<unsigned long long>(disk.hits()),
                 static_cast<unsigned long long>(disk.misses()), static_cast<unsigned long long>(disk.written()));
            static std::atomic<uint64_t> s_lastHistTotal{0};
            uint64_t prevHist = s_lastHistTotal.load(std::memory_order_relaxed);
            if (x87Fb >= prevHist + 100000ull &&
                s_lastHistTotal.compare_exchange_strong(prevHist, x87Fb, std::memory_order_relaxed)) {
                uint32_t topCnt[3] = {0, 0, 0};
                uint16_t topOp[3] = {0, 0, 0};
                for (uint32_t k = 0; k < 8u * 256u; ++k) {
                    const uint32_t c = x87::s_fallbackHist[k].load(std::memory_order_relaxed);
                    if (c > topCnt[0]) {
                        topCnt[2] = topCnt[1]; topOp[2] = topOp[1];
                        topCnt[1] = topCnt[0]; topOp[1] = topOp[0];
                        topCnt[0] = c; topOp[0] = static_cast<uint16_t>(k);
                    } else if (c > topCnt[1]) {
                        topCnt[2] = topCnt[1]; topOp[2] = topOp[1];
                        topCnt[1] = c; topOp[1] = static_cast<uint16_t>(k);
                    } else if (c > topCnt[2]) {
                        topCnt[2] = c; topOp[2] = static_cast<uint16_t>(k);
                    }
                }
                LOGI("x87 fallback top: %02X %02X=%u, %02X %02X=%u, %02X %02X=%u",
                     0xD8u + (topOp[0] >> 8), topOp[0] & 0xFFu, topCnt[0],
                     0xD8u + (topOp[1] >> 8), topOp[1] & 0xFFu, topCnt[1],
                     0xD8u + (topOp[2] >> 8), topOp[2] & 0xFFu, topCnt[2]);
            }
            lastT = statTranslations_;
            lastS = statSlowPath_;
            lastK = statThunks_;
            lastSmc = smc;
            lastTime = now;
        }
    }

    return -3;
}

void OryonEmitterBackend::stop() {
    stopped_ = true;
}

int64_t OryonEmitterBackend::callDllEntry(void* entryPoint, void* dllBase) {
    if (entryPoint == nullptr || stackBase_ == nullptr) {
        lastError_ = "callDllEntry: entryPoint o stackBase nulo";
        return -1;
    }
    registerCurrentThread();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        resetContextLocked();
    }

    if (!mode32_) {
        // Win64 ABI: 16-byte aligned before call, shadow space + return address (sp % 16 == 8)
        uintptr_t sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        sp -= 40;
        *reinterpret_cast<uint64_t*>(sp) = 0; // Return address = 0
        ctx_.gpr[4] = sp; // RSP
        ctx_.gpr[1] = reinterpret_cast<uint64_t>(dllBase); // RCX = hinstDLL
        ctx_.gpr[2] = 1; // RDX = DLL_PROCESS_ATTACH (1)
        ctx_.gpr[8] = 1; // R8 = lpReserved (1)
    } else {
        // Win32 stdcall:
        uintptr_t sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        sp -= 32;
        auto* stack32 = reinterpret_cast<uint32_t*>(sp);
        stack32[0] = 0; // return address = 0
        stack32[1] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dllBase)); // hinstDLL
        stack32[2] = 1; // DLL_PROCESS_ATTACH
        stack32[3] = 1; // lpReserved
        ctx_.gpr[4] = sp; // ESP
    }

    ctx_.rip = reinterpret_cast<uintptr_t>(entryPoint);
    stopped_ = false;
    NativeDllInterceptor::setGuestTeb(teb_, !mode32_);

    LOGI("Ejecutando DllMain en %p (dllBase=%p)", entryPoint, dllBase);

    std::jmp_buf faultJb;
    FaultScope faultScope(this, &faultJb);
    if (setjmp(faultJb) != 0 && !deliverPendingFault()) {
        LOGE("DllMain: excepción hardware del huésped sin manejar en RIP=%p", reinterpret_cast<void*>(ctx_.rip));
        std::lock_guard<std::mutex> lock(mutex_);
        resetContextLocked();
        return -4;
    }

    int64_t retVal = 0;
    for (uint64_t step = 0; step < kMaxDispatchSteps; ++step) {
        if (stopped_) return -3;
        if (ctx_.rip == 0) {
            retVal = static_cast<int64_t>(ctx_.gpr[0]);
            LOGI("DllMain retorno con exito: RAX=%lu", ctx_.gpr[0]);
            if (stackBase_ != nullptr && stackSize_ >= 64) {
                ctx_.gpr[4] = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
            }
            break;
        }
        if (dispatchNativeThunk()) {
            continue;
        }
        TranslatedBlock* block = translateBlock(ctx_.rip);
        if (block == nullptr || block->hostCode == nullptr) {
            LOGE("Fallo en DllMain en RIP=%p: %s",
                 reinterpret_cast<void*>(ctx_.rip), lastError_.c_str());
            {
                std::lock_guard<std::mutex> lock(mutex_);
                resetContextLocked();
            }
            return -2;
        }
        using BlockFn = void (*)(CpuContext*);
        reinterpret_cast<BlockFn>(block->hostCode)(&ctx_);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        resetContextLocked();
    }
    return retVal;
}

namespace {
// Llamada anidada al huésped desde un thunk nativo: mientras corre el código invitado el hilo está
// "ejecutando" (puede aparcarse en SuspendThread); al volver al thunk recupera el estado previo
// ("en nativo"). Sin esto quedaba en "ejecutando" aunque el thunk siguiera esperando (APC de una
// espera alertable de Mono), GetThreadContext no veía nunca un punto seguro y el GC de Mono se
// quedaba esperando para siempre (Silksong: pantalla negra al arrancar, intermitente).
struct NestedRunState {
    std::atomic<int>& state;
    const int saved;
    explicit NestedRunState(std::atomic<int>& s, int running)
        : state(s), saved(s.exchange(running, std::memory_order_seq_cst)) {}
    ~NestedRunState() { state.store(saved, std::memory_order_seq_cst); }
};
}  // namespace

int64_t OryonEmitterBackend::callGuestNested(uintptr_t fn, const uint64_t* args, size_t argCount) {
    if (fn == 0 || argCount > 8 || (argCount > 0 && args == nullptr)) {
        lastError_ = "callGuestNested: función nula o más de 8 argumentos";
        return -1;
    }
    const NestedRunState nestedState(runState_, kRunning);
    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& x) : d(x) { ++d; }
        ~DepthGuard() { --d; }
    } depthGuard(nestDepth_);
    const CpuContext savedCtx = ctx_;
    uintptr_t sp = ctx_.gpr[4];
    if (sp == 0 || (stackBase_ != nullptr && (sp < reinterpret_cast<uintptr_t>(stackBase_) ||
                                              sp > reinterpret_cast<uintptr_t>(stackBase_) + stackSize_))) {
        sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
    }
    // Margen de 2KB bajo la pila del llamador (que puede estar dentro de un thunk).
    sp = (sp - 2048) & ~0x0FULL;
    if (!mode32_) {
        // Win64: RCX, RDX, R8, R9 + shadow space; del 5º en adelante en [RSP+40+8*i]. [RSP] = 0
        // detiene el bucle al hacer RET. RSP queda ≡ 8 (mod 16) como tras un CALL.
        const size_t extra = argCount > 4 ? ((argCount - 4) * 8 + 15) & ~size_t{15} : 0;
        sp -= 40 + extra;
        *reinterpret_cast<uint64_t*>(sp) = 0;
        static constexpr int kArgRegs[4] = {1, 2, 8, 9};
        for (size_t i = 0; i < argCount && i < 4; ++i) ctx_.gpr[kArgRegs[i]] = args[i];
        for (size_t i = 4; i < argCount; ++i) *reinterpret_cast<uint64_t*>(sp + 40 + 8 * (i - 4)) = args[i];
    } else {
        // stdcall/cdecl 32 bits: dirección de retorno 0 seguida de los argumentos.
        sp -= 48;
        auto* stack32 = reinterpret_cast<uint32_t*>(sp);
        stack32[0] = 0;
        for (size_t i = 0; i < argCount; ++i) stack32[1 + i] = static_cast<uint32_t>(args[i]);
    }
    ctx_.gpr[4] = sp;
    ctx_.rip = fn;
    const uintptr_t sentinelSp = sp;  // [sentinelSp] = 0: retorno de esta llamada
    const bool watchEscape = watchEscape_;
    watchEscape_ = false;

    std::jmp_buf faultJb;
    FaultScope faultScope(this, &faultJb);
    if (setjmp(faultJb) != 0 && !deliverPendingFault()) {
        LOGE("callGuestNested: excepción hardware sin manejar en RIP=%p", reinterpret_cast<void*>(ctx_.rip));
        ctx_ = savedCtx;
        return -4;
    }

    int64_t retVal = -2;
    for (uint64_t step = 0; step < kMaxDispatchSteps; ++step) {
        if (stopped_ || nestDepth_ > abortAbove_) break;  // SEH: desenrollado hacia un marco exterior
        if (watchEscape && ctx_.rip != 0 && ctx_.gpr[4] > sentinelSp + 8) {
            // El invitado saltó a un marco por encima de esta llamada (reanudación tras catch).
            escaped_ = true;
            escapedCtx_ = ctx_;
            retVal = -3;
            break;
        }
        if (ctx_.rip == 0) {
            retVal = static_cast<int64_t>(ctx_.gpr[0]);
            break;
        }
        if (dispatchNativeThunk()) continue;
        TranslatedBlock* block = translateBlock(ctx_.rip);
        if (block == nullptr || block->hostCode == nullptr) {
            LOGE("callGuestNested: fallo en RIP=%p: %s", reinterpret_cast<void*>(ctx_.rip),
                 lastError_.c_str());
            break;
        }
        using BlockFn = void (*)(CpuContext*);
        reinterpret_cast<BlockFn>(block->hostCode)(&ctx_);
    }
    ctx_ = savedCtx;
    return retVal;
}

int64_t OryonEmitterBackend::callDllEntryNested(void* entryPoint, void* dllBase) {
    if (entryPoint == nullptr || dllBase == nullptr) {
        lastError_ = "callDllEntryNested: entryPoint o dllBase nulo";
        return -1;
    }

    const NestedRunState nestedState(runState_, kRunning);
    // Preservar exactamente el contexto del CPU llamador
    CpuContext savedCtx = ctx_;

    if (!mode32_) {
        // Win64 ABI: RSP debe estar alineado a 16 bytes antes del call.
        // Dentro de DllMain, [RSP] es la direccion de retorno y (RSP % 16 == 8).
        // Por encima de [RSP] se reserva el shadow space de 32 bytes (4 parametros).
        uintptr_t sp = ctx_.gpr[4];
        if (sp == 0 || (stackBase_ != nullptr && (sp < reinterpret_cast<uintptr_t>(stackBase_) ||
                                                  sp > reinterpret_cast<uintptr_t>(stackBase_) + stackSize_))) {
            sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        }
        // Margen de seguridad de 2KB por debajo de la pila actual del llamador
        sp = (sp - 2048) & ~0x0FULL;
        sp -= 40; // 32 bytes shadow space + 8 bytes return address
        *reinterpret_cast<uint64_t*>(sp) = 0; // Return address = 0 (detiene la ejecucion en RET)
        ctx_.gpr[4] = sp; // RSP
        ctx_.gpr[1] = reinterpret_cast<uint64_t>(dllBase); // RCX = hinstDLL
        ctx_.gpr[2] = 1; // RDX = DLL_PROCESS_ATTACH (1)
        ctx_.gpr[8] = 0; // R8 = lpReserved (NULL para carga dinamica)
    } else {
        uintptr_t sp = ctx_.gpr[4];
        if (sp == 0 || (stackBase_ != nullptr && (sp < reinterpret_cast<uintptr_t>(stackBase_) ||
                                                  sp > reinterpret_cast<uintptr_t>(stackBase_) + stackSize_))) {
            sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        }
        sp = (sp - 1024) & ~0x0FULL;
        sp -= 32;
        auto* stack32 = reinterpret_cast<uint32_t*>(sp);
        stack32[0] = 0; // return address = 0
        stack32[1] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dllBase));
        stack32[2] = 1; // DLL_PROCESS_ATTACH
        stack32[3] = 0; // lpReserved
        ctx_.gpr[4] = sp;
    }

    ctx_.rip = reinterpret_cast<uintptr_t>(entryPoint);
    NativeDllInterceptor::setGuestTeb(teb_, !mode32_);

    LOGI("Ejecutando DllMain anidado en %p (dllBase=%p)", entryPoint, dllBase);

    std::jmp_buf faultJb;
    FaultScope faultScope(this, &faultJb);
    if (setjmp(faultJb) != 0 && !deliverPendingFault()) {
        LOGE("DllMain anidado: excepción hardware sin manejar en RIP=%p", reinterpret_cast<void*>(ctx_.rip));
        ctx_ = savedCtx;
        return -4;
    }

    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& x) : d(x) { ++d; }
        ~DepthGuard() { --d; }
    } depthGuard(nestDepth_);
    int64_t retVal = 0;
    for (uint64_t step = 0; step < kMaxDispatchSteps; ++step) {
        if (stopped_ || nestDepth_ > abortAbove_) break;
        if (ctx_.rip == 0) {
            retVal = static_cast<int64_t>(ctx_.gpr[0]);
            LOGI("DllMain anidado finalizo con exito: RAX=%ld", static_cast<long>(retVal));
            break;
        }
        if (dispatchNativeThunk()) {
            continue;
        }
        TranslatedBlock* block = translateBlock(ctx_.rip);
        if (block == nullptr || block->hostCode == nullptr) {
            LOGE("Fallo en DllMain anidado en RIP=%p: %s",
                 reinterpret_cast<void*>(ctx_.rip), lastError_.c_str());
            retVal = -2;
            break;
        }
        using BlockFn = void (*)(CpuContext*);
        reinterpret_cast<BlockFn>(block->hostCode)(&ctx_);
    }

    // Restaurar los registros del contexto del llamador (la memoria del proceso y CRT permanecen inicializados)
    ctx_ = savedCtx;

    return retVal;
}

int64_t OryonEmitterBackend::executeThread(void* startAddress, void* parameter) {
    if (startAddress == nullptr || stackBase_ == nullptr) {
        lastError_ = "executeThread: startAddress o stackBase nulo";
        return -1;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        resetContextLocked();
    }

    if (!mode32_) {
        // Win64 ABI: RSP % 16 == 8 al entrar (return address) + shadow space.
        uintptr_t sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        sp -= 40;
        *reinterpret_cast<uint64_t*>(sp) = 0; // Return address = 0 (stops on RET)
        ctx_.gpr[4] = sp;                     // RSP
        ctx_.gpr[1] = reinterpret_cast<uint64_t>(parameter); // RCX = lpParameter
    } else {
        // Win32 stdcall:
        uintptr_t sp = reinterpret_cast<uintptr_t>(stackBase_) + stackSize_ - 64;
        sp -= 16;
        auto* stack32 = reinterpret_cast<uint32_t*>(sp);
        stack32[0] = 0; // return address = 0
        stack32[1] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(parameter)); // lpParameter
        ctx_.gpr[4] = sp; // ESP
    }

    return executeEntry(startAddress);
}

size_t OryonEmitterBackend::translatedBlockCount() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return blockCache_.size();
}

bool OryonEmitterBackend::isValidStackAddress(uintptr_t addr, size_t size) const noexcept {
    if (stackBase_ == nullptr || stackSize_ == 0 || size == 0) return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(stackBase_);
    if (addr < base) return false;
    if (addr > base + stackSize_) return false;
    return (addr + size) <= (base + stackSize_);
}

bool OryonEmitterBackend::isExecutableRange(const void* ptr, size_t size) const noexcept {
    if (ptr == nullptr || size == 0) return false;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = begin + size;
    if (end < begin) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const ExecRange& range : execRanges_) {
        if (begin >= range.start && end <= range.end) return true;
    }
    return false;
}

}  // namespace hyperplay
