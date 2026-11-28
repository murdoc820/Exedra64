#pragma once

#include "OryonCpuTranslator.h"
#include "HyperAotCache.h"

#include <signal.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hyperplay {

class HyperAotCache;
struct PeInfo;
struct LocalDllRegion;

// Estado completo de la CPU x86 / x86_64 emulada.
// El layout es parte del contrato del JIT: los offsets se usan en el prólogo y epílogo
// AArch64 generados en tiempo de ejecución.
struct alignas(16) CpuContext {
    union {
        uint64_t gpr[16];                 // RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8..R15
        uint64_t regs[16];                // Alias regs[] para compatibilidad
    };
    uint64_t rip;                     // Instruction Pointer del huésped
    uint64_t nzcv;                    // Flags ARM NZCV tras la última operación
    uint64_t rflags;                  // Flags x86 (CF/PE/AF/ZF/SF/OF)
    uint32_t mode32;                  // 1 = programa de 32 bits
    uint32_t reserved0;
    alignas(16) uint8_t simd[16][16]; // XMM0..XMM15 (lower 128 bits)
    alignas(16) uint8_t ymmHigh[16][16]; // YMM0..YMM15 (upper 128 bits)
    uint64_t fsBase;                  // Base del segmento FS (TEB 32-bit)
    uint64_t gsBase;                  // Base del segmento GS (TEB 64-bit)
    // Encadenado de bloques: lo fija el despachador antes de cada bloque; las salidas
    // enlazables lo leen (presupuesto, cursor SMC) y anotan su B parcheable en exitSite.
    int64_t chainBudget;              // saltos enlazados restantes antes de volver al despachador
    uint64_t exitSite;                // B parcheable de la última salida enlazable (0 = ninguna)
    uint64_t smcCursor;               // cursor SMC del hilo cuando se entró desde el despachador
    uint64_t smcIdxAddr;              // &s_smcWriteIdx (global de escrituras sobre código)
    uint64_t fastCacheAddr;           // caché rápida del hilo (búsqueda en línea de RET/saltos indirectos)
    uint64_t fastGenCopy;             // generación vigente de esa caché
    uint64_t nativeFn;                // llamada nativa directa bloqueante en curso (perfil de esperas)
    // PF perezoso: byte bajo del último resultado que fijó los flags. PF = paridad PAR de este
    // byte; el bit 2 de rflags NO se mantiene. Leer PF: pfFromByte(). Escribir PF a mano:
    // pfByte = PF ? 0 : 1.
    uint64_t pfByte;
    // Pila de retornos (sombra de CALL/RET): rasTop cuenta pushes (se usa & 63); cada entrada es
    // {RIP de retorno del huésped, dirección host de la continuación tras el CALL}.
    uint64_t rasTop;
    uint64_t ras[2 * 64];
    // FPU x87 (lo usa el JIT de Mono: Math.Sqrt = fldl/fsqrt/fstpl, sin/cos, etc.). Los registros
    // físicos R0..R7 se guardan en precisión doble; ST(i) = x87[(x87Top + i) & 7]. Lo ejecuta un
    // intérprete nativo (executeX87); x87Tag: bit i = R(i) ocupado.
    double x87[8];
    uint32_t x87Top;
    uint32_t x87Tag;
    uint16_t x87Sw;                   // palabra de estado sin TOP (C0..C3, excepciones)
    uint16_t x87Cw;                   // palabra de control (0x037F tras FNINIT)
    uint32_t x87Reserved;
};

inline bool pfFromByte(uint64_t b) noexcept { return (__builtin_popcount(static_cast<unsigned>(b & 0xFFu)) & 1) == 0; }

using GuestCpuContext = CpuContext;

struct EmitterConfig {
    bool is32Bit = false;
    size_t stackSize = 8u * 1024u * 1024u;
    // Virtual (memfd): solo ocupa RAM lo usado. Al llenarse se vacía y se retraduce (resetArena).
    size_t codeCacheSize = 512u * 1024u * 1024u;
    bool persistBlocks = false;
};

// Trampolín nativo (Win32 / ABI del huésped). Devuelve true si ha consumido la llamada y
// ha actualizado ctx.rip/ctx.gpr. Se invoca antes de traducir cada bloque.
using NativeThunkHandler = bool (*)(void* user, uintptr_t target, CpuContext& ctx);

/// Motor JIT x86/x86_64 -> AArch64 nativo para Snapdragon 8 Elite.
/// Decodifica bytes x86 reales, emite AArch64, cambia la página de RW a RX y ejecuta.
/// Cualquier instrucción no soportada hace fallar la traducción de forma explícita.
class OryonEmitterBackend : public CpuTranslationBackend {
public:
    OryonEmitterBackend();
    ~OryonEmitterBackend() override;

    OryonEmitterBackend(const OryonEmitterBackend&) = delete;
    OryonEmitterBackend& operator=(const OryonEmitterBackend&) = delete;

    /// Reserva arena JIT y stack virtual, y reinicia el contexto del huésped.
    bool configure(const EmitterConfig& config);

    void setNativeThunkHandler(NativeThunkHandler handler, void* user);

    TranslatedBlock* translateBlock(uintptr_t guestRip) override;
    void logIfWatchedRip(uintptr_t guestRip);
    int64_t executeEntry(void* entryPoint) override;
    int64_t callDllEntry(void* entryPoint, void* dllBase) override;
    int64_t callDllEntryNested(void* entryPoint, void* dllBase) override;
    /// Llama a una función invitada (callback Win32) desde dentro de un thunk nativo con hasta
    /// 4 argumentos; preserva y restaura el contexto del llamador. Devuelve RAX o <0 si falla.
    int64_t callGuestNested(uintptr_t fn, const uint64_t* args, size_t argCount);
    /// Profundidad de llamadas invitadas anidadas en curso en este hilo (0 = bucle principal).
    [[nodiscard]] int nestDepth() const noexcept { return nestDepth_; }
    /// SEH (RtlUnwindEx/RtlRestoreContext): las llamadas anidadas más profundas que `depth`
    /// terminan en cuanto vuelven al despachador; INT_MAX lo desactiva.
    void abortNestedAbove(int depth) noexcept { abortAbove_ = depth; }
    /// SEH: la próxima callGuestNested vigila si el invitado abandona la llamada saltando a un
    /// marco exterior (CoreCLR reanuda tras un catch con su propio ClrRestoreNonvolatileContext,
    /// sin RtlRestoreContext). Si pasa, la llamada termina y takeEscape() da ese contexto.
    void watchNextEscape() noexcept { watchEscape_ = true; }
    bool takeEscape(CpuContext& out) noexcept {
        if (!escaped_) return false;
        escaped_ = false;
        out = escapedCtx_;
        return true;
    }
    void stop() override;
    [[nodiscard]] size_t translatedBlockCount() const noexcept override;

    /// Traduce una secuencia binaria concreta (usado por las pruebas del dispositivo).
    TranslatedBlock* compileInstructionSequence(const uint8_t* code, size_t codeSize, uintptr_t baseRip);

    void setupImageTls(void* mainBase, const PeInfo& mainPe,
                       const std::unordered_map<std::string, LocalDllRegion>& localDlls,
                       const std::string& exeFilePath = "");
    void setExecutablePath(const std::string& exePath);
    bool allocateTeb();
    void cloneTlsFrom(const OryonEmitterBackend& parent);

    [[nodiscard]] CpuContext& context() noexcept { return ctx_; }
    [[nodiscard]] const CpuContext& context() const noexcept { return ctx_; }
    [[nodiscard]] void* virtualStack() const noexcept { return stackBase_; }
    [[nodiscard]] size_t virtualStackSize() const noexcept { return stackSize_; }
    [[nodiscard]] bool isValidStackAddress(uintptr_t addr, size_t size = 8) const noexcept;
    [[nodiscard]] bool isExecutableRange(const void* ptr, size_t size) const noexcept;
    [[nodiscard]] const std::string& lastError() const noexcept override { return lastError_; }

    void setAotCache(HyperAotCache* cache) override;
    void setCodeRange(uintptr_t base, size_t size) override;
    void addCodeRange(uintptr_t base, size_t size) override;

    /// Registra memoria invitada ejecutable generada en tiempo de ejecución (VirtualAlloc /
    /// VirtualProtect con PAGE_EXECUTE*). Equivale a addCodeRange pero deja explícita la
    /// intención: el JIT leerá bytes x86/x64 de ese rango y emitirá AArch64 en su arena W^X.
    void addExecutableRange(uintptr_t base, size_t size);

    /// Invalida los bloques traducidos dentro de [base, base+size). Se usa cuando el invitado
    /// escribe código nuevo o cambia la protección de una página ejecutable.
    void invalidateCodeRange(uintptr_t base, size_t size);

    /// Deja de considerar [base, base+size) como código y descarta sus traducciones.
    void removeExecutableRange(uintptr_t base, size_t size);

    struct FlightRecord {
        const std::string* name = nullptr;  // nombre estable (ThunkInfo cacheado): sin copias
        uint64_t arg0 = 0;
        uint64_t arg1 = 0;
        uintptr_t rip = 0;
    };
    struct ModuleMetadata {
        std::string name;
        uintptr_t base = 0;
        size_t size = 0;
    };
    struct CodeRange {
        uintptr_t base = 0;
        size_t size = 0;
    };

    void registerModule(const std::string& name, uintptr_t base, size_t size);
    void recordThunkCall(const std::string& name, uint64_t arg0 = 0, uint64_t arg1 = 0);
    std::string formatDiagnosticReport(uintptr_t rip, const std::string& reason);
    std::string formatDiagnosticReportLocked(uintptr_t rip, const std::string& reason) const;
    void setGuestMode(bool is32Bit) override;

    [[nodiscard]] const std::vector<CodeRange>& codeRanges() const noexcept { return codeRanges_; }
    [[nodiscard]] NativeThunkHandler nativeThunkHandler() const noexcept { return nativeThunk_; }
    [[nodiscard]] void* nativeThunkUser() const noexcept { return nativeThunkUser_; }
    [[nodiscard]] HyperAotCache* aotCache() const noexcept { return aotCache_; }
    [[nodiscard]] bool is32Bit() const noexcept { return mode32_; }
    int64_t executeThread(void* startAddress, void* parameter);

    static OryonEmitterBackend* activeBackend() noexcept { return s_activeBackend; }
    static void setActiveBackend(OryonEmitterBackend* b) noexcept { s_activeBackend = b; }

    // --- Suspensión de hilos invitados (SuspendThread/GetThreadContext de Win32) -------------
    // El hilo se detiene en un punto seguro entre bloques JIT (CpuContext coherente); mientras
    // ejecuta un thunk nativo ya cuenta como detenido para leer su contexto, y se detiene antes
    // de volver a ejecutar código invitado.
    enum RunState : int { kRunning = 0, kInNative = 1, kParked = 2 };
    /// Incrementa el contador de suspensión; devuelve el valor previo (semántica Win32).
    uint32_t requestSuspend();
    /// Decrementa el contador (no baja de 0); devuelve el valor previo.
    uint32_t requestResume();
    /// Espera hasta que el hilo esté detenido o en código nativo (true) o expire (false).
    bool waitUntilSuspended(uint32_t timeoutMs);
    [[nodiscard]] int runState() const noexcept { return runState_.load(std::memory_order_acquire); }
    [[nodiscard]] int hostTid() const noexcept { return hostTid_; }
    [[nodiscard]] uint32_t suspendCount() const noexcept { return suspendCount_.load(std::memory_order_relaxed); }

    // Diagnóstico: registra las direcciones de retorno candidatas de la pila del huésped.
    void logGuestStackChain(const char* tag) const;

    // Diagnóstico: mientras sea > 0, cada traducción nueva registra su RIP y lo decrementa.
    static inline std::atomic<int> traceTranslations{0};
    // Orden de memoria x86 (TSO) con LDAPR/STLR en los accesos a memoria del huésped. Sin él,
    // ARM reordena cargas/escrituras y las colas lock-free (jobs de Unity) pierden trabajo.
    // setprop debug.hyperplay.tso 0 lo desactiva.
    static inline std::atomic<bool> tsoMemory{true};
    // Encadenado de bloques (setprop debug.hyperplay.chain 0 lo desactiva).
    static inline std::atomic<bool> chainBlocks{true};
    static constexpr int64_t kChainBudget = 2048;
    // Eliminación de flags muertos (setprop debug.hyperplay.deadflags 0 la desactiva).
    static inline std::atomic<bool> deadFlags{true};
    // Fusión de flags: Jcc/CMOVcc/SETcc justo detrás de la instrucción que fija los flags usan
    // el NZCV del host (B.cond/CSEL/CSET) en vez de releer RFLAGS de memoria.
    static inline std::atomic<bool> fuseFlags{true};
    // Flags muertos entre bloques: una salida directa cuyo destino (código estático de una imagen
    // PE, en la misma página que el bloque) sobrescribe todos los flags antes de leerlos no los
    // materializa en RFLAGS. DESACTIVADO por defecto (sin ganancia medible en HK y sospechoso de
    // los fallos de física: atasco en el banco y geo cayendo); setprop debug.hyperplay.xflags 1.
    static inline std::atomic<bool> crossBlockFlags{false};
    /// Rango de una imagen PE mapeada (código estático): los destinos fuera de imágenes (Mono JIT,
    /// VirtualAlloc) nunca se analizan para los flags entre bloques.
    static void addStaticImage(uintptr_t base, size_t size) noexcept;
    [[nodiscard]] static bool inStaticImage(uintptr_t addr, size_t len) noexcept;
    /// true si el código x86 en [p, p+avail) sobrescribe CF/PF/ZF/SF/OF (escritura completa) antes
    /// de leer ningún flag, en línea recta (sin cruzar saltos). Conservador: ante la duda, false.
    [[nodiscard]] static bool flagsDeadAt(const uint8_t* p, size_t avail, bool is64) noexcept;
    // Pila de retornos: CALL rel32 = BL (el predictor de retornos del Oryon aprende el par) y RET
    // salta directo a la continuación si la dirección de retorno coincide (si no, búsqueda normal).
    static inline std::atomic<bool> returnStack{true};
    // Búsqueda en línea para RET/saltos indirectos (setprop debug.hyperplay.inlinedispatch 0).
    static inline std::atomic<bool> inlineDispatch{true};
    /// Backend que ejecuta código invitado en el hilo del kernel `tid` (o nullptr).
    static OryonEmitterBackend* forThread(int tid);

    /// TLS de una DLL cargada en tiempo de ejecución (LoadLibrary): asigna índice propio, escribe
    /// su `_tls_index` y crea el bloque (copia de la plantilla .tls) en todos los hilos vivos;
    /// los hilos nuevos lo heredan con cloneTlsFrom. Sin esto la DLL usaba el índice 0 y pisaba
    /// el bloque TLS de otro módulo (p. ej. `_Init_thread_epoch` de UnityPlayer). Antes de
    /// setupImageTls no hace nada (setupImageTls cubre las DLL ya mapeadas).
    static void registerDynamicModuleTls(void* base, const PeInfo& pe, const std::string& name);

    // Rangos ejecutables del PROCESO (VirtualAlloc/VirtualProtect/VirtualFree): se aplican a
    // todos los backends de hilo, no solo al principal.
    static void addExecutableRangeAll(uintptr_t base, size_t size);
    // Diagnóstico (setprop debug.hyperplay.wwatch <dir>,<bytes>): protege esas páginas del invitado como solo
    // lectura y registra cada escritura del invitado (RIP y dirección) antes de dejarla pasar.
    static void armWriteWatchFromProperty();
    static bool handleWriteWatch(uint64_t address, uint64_t rip);
    // Aviso si un VirtualFree/UnmapViewOfFile libera la zona vigilada (quién la desmapea).
    static void noteUnmap(uint64_t base, uint64_t size, const char* who, uint64_t rip);
    static void removeExecutableRangeAll(uintptr_t base, size_t size);
    static void invalidateCodeRangeAll(uintptr_t base, size_t size);
    /// Base del módulo PE que contiene una dirección (lo da el cargador; para la sonda de diagnóstico).
    static void setModuleBaseResolver(uint64_t (*fn)(uint64_t));
    bool findCodeRange(uintptr_t rip, CodeRange& out) const;

    // Código automodificable: el interceptor indica la protección POSIX de una página del
    // huésped (-1 si no es memoria de VirtualAlloc) y olvida rangos al cambiarles la protección.
    static void setSmcWritableHook(int (*hook)(uintptr_t page));
    static void smcForgetRange(uintptr_t base, size_t size);

    // --- Excepciones hardware del huésped (VEH) -------------------------------------------
    /// Traduce un PC del código JIT a la RIP x86 exacta de la instrucción que lo generó.
    bool resolveGuestFault(uintptr_t hostPc, uintptr_t& guestRip);
    /// Desde el manejador de señales: si el fallo es de código JIT de este hilo, guarda el estado
    /// x86 exacto (registros del ucontext) y desvía el PC al aterrizaje. true = se ha capturado.
    bool captureFault(int sig, siginfo_t* info, void* uctx);
    /// Aterrizaje (ya fuera del manejador): vuelve al bucle del despachador activo.
    [[noreturn]] void landFromFault();

private:
    static void forEachBackend(const std::function<void(OryonEmitterBackend&)>& fn);
    void smcProtectForTranslation(uintptr_t start, size_t len);
    void processSmcDirty();
    uint64_t smcCursor_ = 0;

    struct PendingFault {
        bool valid = false;
        uint32_t code = 0;
        uint64_t address = 0;
        uint64_t access = 0;  // 0 lectura, 1 escritura, 8 ejecución
        uint64_t rip = 0;
        uint64_t gpr[16] = {};
        uint8_t simd[16][16] = {};
    };
    struct FaultScope {
        OryonEmitterBackend* self;
        std::jmp_buf* previous;
        FaultScope(OryonEmitterBackend* b, std::jmp_buf* jb) : self(b), previous(b->faultJmp_) { b->faultJmp_ = jb; }
        ~FaultScope() { self->faultJmp_ = previous; }
    };
    /// Construye EXCEPTION_RECORD/CONTEXT, llama a los VEH y aplica el CONTEXT resultante.
    bool deliverPendingFault();

    // Atomicidad de la instrucción x86 en curso (prefijo LOCK o XCHG con memoria).
    bool lockActive_ = false;
    bool lockCaptured_ = false;
    bool lockStored_ = false;
    size_t lockInstrStartWord_ = 0;
    void lockCaptureAfterLoad(uint8_t rt, uint8_t rn);
    bool lockStoreAsCas(uint8_t rt, uint8_t rn, uint32_t sizeLog2);
    bool emitTsoAccess(uint8_t rt, uint8_t rn, int32_t offset, uint32_t sizeLog2, bool store);

    PendingFault pendingFault_;
    std::jmp_buf* faultJmp_ = nullptr;
    int exceptionDepth_ = 0;

    // Caché directa RIP -> código host del hilo propietario (sin locks en el camino caliente).
    // Las invalidaciones por rango anulan solo las entradas afectadas (gen = 0); un vaciado
    // total incrementa la generación.
    struct FastEntry {
        uintptr_t rip = 0;
        void* host = nullptr;
        uint64_t gen = 0;
    };
    // 64 K entradas (1,5 MB virtuales por hilo, en calloc: solo las páginas usadas ocupan RAM).
    // Con 16 K, el código de Mono/Unity colisionaba en ~40 mil búsquedas lentas por 2 M bloques.
    // 256 K: Unity + Mono pasan de 600 000 bloques en partida; con 64 K había ~20 000 fallos/s.
    static constexpr size_t kFastCacheSize = 262144;
    static_assert((kFastCacheSize & (kFastCacheSize - 1)) == 0, "potencia de 2 (máscara en el JIT)");
    struct FreeDeleter {
        void operator()(FastEntry* p) const noexcept { std::free(p); }
    };
    std::unique_ptr<FastEntry[], FreeDeleter> fastCache_;
    std::atomic<uint64_t> fastGen_{1};
    void* fastLookup(uintptr_t rip) const noexcept {
        const FastEntry& e = fastCache_[(rip ^ (rip >> 16)) & (kFastCacheSize - 1)];
        return (e.rip == rip && e.gen == fastGen_.load(std::memory_order_acquire)) ? e.host : nullptr;
    }
    void fastInsert(uintptr_t rip, void* host) noexcept {
        FastEntry& e = fastCache_[(rip ^ (rip >> 16)) & (kFastCacheSize - 1)];
        e.rip = rip;
        e.host = host;
        e.gen = fastGen_.load(std::memory_order_acquire);
    }

    void suspensionPoint();
    bool dispatchNativeThunk();
    void registerCurrentThread();
    void unregisterCurrentThread();

    std::atomic<uint32_t> suspendCount_{0};
    std::atomic<int> runState_{kRunning};
    std::mutex suspendMutex_;
    std::condition_variable suspendCv_;
    int hostTid_ = 0;

    struct CodeReader {
        const uint8_t* p = nullptr;
        const uint8_t* end = nullptr;
        bool ok = true;

        bool u8(uint8_t& v);
        bool s8(int8_t& v);
        bool u16(uint16_t& v);
        bool u32(uint32_t& v);
        bool u64(uint64_t& v);
        [[nodiscard]] size_t offset(const uint8_t* base) const { return static_cast<size_t>(p - base); }
    };

    struct ModRm {
        uint8_t mod = 0;
        uint8_t reg = 0;
        uint8_t rm = 0;
        bool isMemory = false;
        bool hasBase = false;
        bool hasIndex = false;
        bool ripRelative = false;
        bool absolute = false;
        uint8_t base = 0;
        uint8_t index = 0;
        uint8_t scale = 0;
        int64_t disp = 0;
    };

    struct ExecRange {
        uintptr_t start = 0;
        uintptr_t end = 0;
    };

    enum class CfMode { Zero, Carry, Borrow, Preserve, Reg };
    // Reg: OF = bit 0 de X20; Reg1: OF = bit 1 de X20 (para usarlo junto a CfMode::Reg, bit 0).
    enum class OfMode { Zero, Overflow, Reg, Reg1 };

    bool allocateArena(size_t size);
    bool allocateStack(size_t size);
    void freeResources();
    void resetContextLocked();

    size_t wordPos() const noexcept { return jitOffset_ / sizeof(uint32_t); }
    void emitWord(uint32_t instruction);
    // RDTSC: EDX:EAX = CNTVCT_EL0 (contador virtual de ARM, monótono).
    void emitReadTsc();
    void patchWord(size_t wordIndex, uint32_t instruction);
    void patchImm19(size_t wordIndex, size_t targetWordIndex);
    void patchImm26(size_t wordIndex, size_t targetWordIndex);

    // Emisores AArch64.
    void emitMovImm64(uint8_t rd, uint64_t value);
    void emitMovReg(uint8_t dst, uint8_t src, bool is64);
    void emitMvnReg(uint8_t dst, uint8_t src, bool is64);
    void emitAddReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags, bool is64);
    void emitAddRegShift(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t shift, bool is64);
    void emitSubReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags, bool is64);
    void emitStringStep(uint8_t elemSize);  // X17 = ±tamaño según DF (MOVS/STOS)
    void emitAndReg(uint8_t dst, uint8_t src1, uint8_t src2, bool setFlags, bool is64);
    void emitOrrReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitEorReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitMulReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitSmull(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitSmulh(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitUmulh(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitUdiv(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitSdiv(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitAdcsReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitSbcsReg(uint8_t dst, uint8_t src1, uint8_t src2, bool is64);
    void emitRbitReg(uint8_t dst, uint8_t src, bool is64);
    void emitRev(uint8_t dst, uint8_t src, bool is64);
    void emitRev16(uint8_t dst, uint8_t src, bool is64);
    void emitClzReg(uint8_t dst, uint8_t src, bool is64);
    void emitPrimeCarry(bool forSbb);
    void emitLslImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64);
    void emitLsrImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64);
    void emitAsrImm(uint8_t dst, uint8_t src, uint8_t shift, bool is64);
    void emitLslv(uint8_t dst, uint8_t src, uint8_t amount, bool is64);
    void emitLsrv(uint8_t dst, uint8_t src, uint8_t amount, bool is64);
    void emitAsrv(uint8_t dst, uint8_t src, uint8_t amount, bool is64);
    void emitUbfx(uint8_t dst, uint8_t src, uint8_t lsb, uint8_t width);
    void emitUbfx32(uint8_t dst, uint8_t src, uint8_t lsb, uint8_t width);
    void emitBfiByte(uint8_t dst, uint8_t src);
    void emitBfi(uint8_t rd, uint8_t rn, uint8_t lsb, uint8_t width);
    void emitCset(uint8_t dst, uint8_t armCond);
    void emitSxtw(uint8_t dst, uint8_t src);
    void emitSxtb(uint8_t dst, uint8_t src);
    void emitSxth(uint8_t dst, uint8_t src);
    void emitUxtb(uint8_t dst, uint8_t src);
    void emitUxth(uint8_t dst, uint8_t src);
    void emitAddImm(uint8_t dst, uint8_t src, uint16_t imm, bool is64, bool setFlags = false);
    void emitSubImm(uint8_t dst, uint8_t src, uint16_t imm, bool is64, bool setFlags = false);
    void emitCmpImm(uint8_t reg, uint16_t imm, bool is64);
    void emitTstReg(uint8_t reg, bool is64);
    void emitTstSized(uint8_t reg, uint8_t size);
    void storeByteRm(uint8_t rm, uint8_t src);
    void loadByteRm(uint8_t dst, uint8_t rm);
    void emitExtr(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsb, bool is64);
    void emitRorv(uint8_t rd, uint8_t rn, uint8_t rm, bool is64);
    void emitMrsNzcv(uint8_t rd);
    void emitMsrNzcv(uint8_t rn);

    void emitLdrX(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrX(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrW(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrW(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrB(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrB(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrH(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrH(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrS(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrS(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrD(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrD(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdrQ(uint8_t rt, uint8_t rn, int32_t offset);
    void emitStrQ(uint8_t rt, uint8_t rn, int32_t offset);
    void emitLdpX(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t offset);
    void emitStpX(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t offset);
    void emitStpPre(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t imm7);
    void emitLdpPost(uint8_t rt, uint8_t rt2, uint8_t rn, int32_t imm7);
    void emitFaddV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFsubV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmulV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFdivV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFaddV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFsubV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmulV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFdivV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFaddS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFsubS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmulS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFdivS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFaddD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFsubD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmulD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFdivD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmaS(uint8_t dst, uint8_t n, uint8_t m, uint8_t a, bool sub, bool neg);
    void emitFmaD(uint8_t dst, uint8_t n, uint8_t m, uint8_t a, bool sub, bool neg);
    void emitInsElementD(uint8_t dst, uint8_t dstIdx, uint8_t src, uint8_t srcIdx);
    void emitFcmpS(uint8_t src1, uint8_t src2);
    void emitFcmpD(uint8_t src1, uint8_t src2);
    void emitFsqrtS(uint8_t dst, uint8_t src);
    void emitFsqrtD(uint8_t dst, uint8_t src);
    void emitFsqrtV(uint8_t dst, uint8_t src);
    void emitFsqrtV2D(uint8_t dst, uint8_t src);
    void emitFrsqrteS(uint8_t dst, uint8_t src);
    void emitFrsqrteD(uint8_t dst, uint8_t src);
    void emitFrsqrteV(uint8_t dst, uint8_t src);
    void emitFrsqrteV2D(uint8_t dst, uint8_t src);
    void emitFrecpeS(uint8_t dst, uint8_t src);
    void emitFrecpeD(uint8_t dst, uint8_t src);
    void emitFrecpeV(uint8_t dst, uint8_t src);
    void emitFrecpeV2D(uint8_t dst, uint8_t src);
    void emitX86MinMax(uint8_t dst, uint8_t a, uint8_t b, bool isMax, uint32_t cmpBase);
    void emitFminS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmaxS(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFminD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmaxD(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFminV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmaxV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFminV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFmaxV2D(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitScvtfS(uint8_t dst, uint8_t src, bool is64);
    void emitScvtfD(uint8_t dst, uint8_t src, bool is64);
    void emitFcvtzsS(uint8_t dst, uint8_t src, bool is64);
    void emitFcvtzsD(uint8_t dst, uint8_t src, bool is64);
    void emitFcvtDS(uint8_t dst, uint8_t src);
    void emitFcvtSD(uint8_t dst, uint8_t src);
    void emitAndV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitOrrV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitBicV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitEorV(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitCmeqV16B(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitCmeqV8H(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitCmeqV4S(uint8_t dst, uint8_t src1, uint8_t src2);
    void emitFcmEqV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble);
    void emitFcmGeV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble);
    void emitFcmGtV(uint8_t dst, uint8_t src1, uint8_t src2, bool isDouble);
    void emitCmpPsPd(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t predicate, bool isDouble);
    void emitInsS(uint8_t dst, uint8_t src);
    void emitInsD(uint8_t dst, uint8_t src);
    void emitExtV16B(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t imm);
    void emitUshrV2D(uint8_t dst, uint8_t src, uint8_t shift);
    void emitShlV2D(uint8_t dst, uint8_t src, uint8_t shift);
    void emitUshrV4S(uint8_t dst, uint8_t src, uint8_t shift);
    void emitSshrV4S(uint8_t dst, uint8_t src, uint8_t shift);
    void emitShlV4S(uint8_t dst, uint8_t src, uint8_t shift);
    void emitUshrV8H(uint8_t dst, uint8_t src, uint8_t shift);
    void emitSshrV8H(uint8_t dst, uint8_t src, uint8_t shift);
    void emitShlV8H(uint8_t dst, uint8_t src, uint8_t shift);
    void emitInsElementS(uint8_t dst, uint8_t dstIdx, uint8_t src, uint8_t srcIdx);
    void emitInsElementH(uint8_t dst, uint8_t dstIdx, uint8_t src, uint8_t srcIdx);
    void emitAddV(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t elemSize);
    void emitSubV(uint8_t dst, uint8_t src1, uint8_t src2, uint8_t elemSize);

    void emitB(int32_t imm26);
    void emitBl(int32_t imm26);
    void emitCbnz(uint8_t rt, int32_t imm19);
    void emitCbz(uint8_t rt, int32_t imm19);
    void emitBr(uint8_t rt);
    void emitBlr(uint8_t rt);
    void emitRet();
    void emitNop();

    void emitPrologue();
    void emitEpilogue();
    void emitWriteRipImm(uintptr_t rip);
    // Salida a una RIP fija que el despachador puede enlazar directamente con el bloque destino.
    void emitLinkableExit(uintptr_t target);
    // Salida dinámica (RET, JMP/CALL indirectos): busca ctx.rip en la caché rápida desde el
    // propio código ARM y salta al bloque sin pasar por el despachador.
    void emitInlineDispatch();
    bool lastExitDynamic_ = false;
    // call [IAT] hacia una función nativa de la lista rápida: BLR directo desde el bloque.
    void emitX87Call(uint8_t op, uint8_t modrm, bool hasEa);
    bool emitX87Native(uint8_t op, uint8_t modrm, bool mem);
    void emitProbeCall(uintptr_t rip);
    void emitFastNativeCall(uintptr_t fn, bool blocking, uintptr_t nextRip, bool tailCall = false,
                            uintptr_t iatSlot = 0);
    void emitFastNativeCall32(uintptr_t fn, bool blocking, uintptr_t nextRip, uint32_t slots);
    void emitFastIndirect32(uintptr_t nextRip, bool tailCall = false);

public:
    /// Función nativa (wrapper Win32 de HyperPlay) que el JIT puede llamar directamente desde el
    /// código traducido: <= 4 argumentos enteros, sin callbacks al juego ni acceso al contexto.
    /// `blocking`: puede esperar (locks): marca el hilo como "en nativo" para las suspensiones.
    static void registerFastNative(uintptr_t fn, bool blocking);
    /// Clave de configuración del emisor que se guarda con cada bloque en la caché persistente.
    [[nodiscard]] uint32_t persistentConfigKey() const noexcept { return aotConfigKey(); }
    /// Tiempo total (ms, todos los hilos) traduciendo o cargando bloques desde la caché.
    [[nodiscard]] static uint64_t totalJitMs() noexcept;
    static bool lookupFastNative(uintptr_t fn, bool& blocking);
    // Llamadas nativas directas (setprop debug.hyperplay.fastnative 0 las desactiva).
    static inline std::atomic<bool> fastNativeCalls{true};
    // 32 bits: alias de la IAT -> función nativa y su número de palabras stdcall (-1 si no aplica). Los fija el interceptor.
    static std::atomic<uintptr_t (*)(uintptr_t)> fast32Host;
    static std::atomic<int (*)(uintptr_t)> fast32Slots;
    // CALL indirecto a la zona de alias (1 MB desde fast32AliasBase): fast32Indirect(ctx, destino, siguiente) hace
    // la llamada nativa completa; devuelve 0 si se sigue en la instrucción siguiente.
    static std::atomic<uintptr_t> fast32AliasBase;
    static std::atomic<uint32_t (*)(CpuContext*, uint32_t, uint32_t)> fast32Indirect;
    struct Fast32AliasEntry {
        uintptr_t hostFn;
        uint8_t slots;
        uint8_t blocking;
        uint8_t pad[6];
    };
    static std::atomic<const Fast32AliasEntry*> fast32Table;

private:
    void linkExit(uint64_t site, const void* targetHost, uintptr_t targetRip);
    void unlinkAllLocked();
    void unlinkRangeLocked(uintptr_t base, uintptr_t end);
    struct LinkRecord {
        uint32_t original;    // palabra original (B al epílogo)
        uintptr_t targetRip;  // RIP invitada del bloque destino
    };
    struct ExitPatch {
        size_t pos;
        bool cond;     // B.cond (imm19) o B (imm26)
        bool skipClear;  // salta el "exitSite = 0" del epílogo
    };
    std::vector<ExitPatch> pendingExits_;
    std::unordered_map<uint64_t, LinkRecord> links_;  // sitio parcheado -> original y destino
    std::map<uintptr_t, std::vector<uint64_t>> linksByTarget_;  // RIP destino -> sitios (por rango)
    uint64_t farLinks_ = 0;  // enlaces por veneer (destino a más de ±128 MB)
    uint64_t arenaGen_ = 0;  // se incrementa al vaciar la arena: invalida exitSite pendientes
    // Última "mov r64, imm64" del bloque (para "call reg" hacia funciones rápidas).
    uint64_t instrSeq_ = 0;
    size_t instrStartWord_ = 0;  // primera palabra de la instrucción x86 en curso
    // Último NZCV del host que refleja RFLAGS (fin de emitWriteFlags): secuencia de la
    // instrucción, palabra de inicio de esa instrucción, palabra final y modos CF/OF.
    uint64_t nzcvSeq_ = ~uint64_t{0} - 1;
    size_t nzcvInstrStart_ = 0;
    size_t nzcvEnd_ = 0;
    uint8_t nzcvCf_ = 0;
    uint8_t nzcvOf_ = 0;
    int fusedArmCond(uint8_t x86Cond);
    bool eaIsStack_ = false;  // la última dirección [X16] es relativa a la pila (sin TSO)
    uint64_t movImmSeq_ = ~uint64_t{0} - 1;
    uint8_t movImmReg_ = 0xFF;
    uint64_t movImmValue_ = 0;
    // Flags muertos: escrituras completas de RFLAGS (emitWriteFlags) y lecturas/barreras en
    // orden de emisión; al cerrar el bloque, una escritura seguida de otra sin lectura entre
    // ambas se salta con un B.
    // DeadExit: salida directa a un destino que sobrescribe los flags sin leerlos.
    enum class FlagEventKind : uint8_t { Write, Read, DeadExit };
    uintptr_t curBlockRip_ = 0;  // RIP de inicio del bloque que se está traduciendo
    uintptr_t stitchTarget_ = 0;
    uintptr_t stitchLimit_ = 0;
    int stitchBudget_ = 0;
    struct FlagEvent {
        FlagEventKind kind;
        size_t start, end;  // palabras [start, end) de la escritura
    };
    std::vector<FlagEvent> flagEvents_;
    bool inFlagWrite_ = false;
    void noteFlagAccess(uint8_t rn, int32_t offset);
    void eliminateDeadFlags(size_t blockStartWord, size_t blockEndWord);
    size_t prologueBytes_ = 0;
    void emitWriteRipReg(uint8_t reg, bool is64);
    void emitWriteFlags(CfMode cf, OfMode of, uint8_t resultReg = 0xFF, uint8_t resultShift = 0);
    /// Añade el bit PF (bit 2) del byte bajo del resultado al acumulador de flags.
    void emitParityBits(uint8_t accumulator, uint8_t resultReg, uint8_t resultShift);
    size_t emitJccTest(uint8_t x86Cond);

    bool decodeModRm(CodeReader& reader, bool rexR, bool rexX, bool rexB, ModRm& out);
    void emitEffectiveAddress(const ModRm& modrm, uintptr_t nextRip);
    void emitLoadOperand(const ModRm& modrm, uint8_t hostReg, uint8_t size, uintptr_t nextRip);
    void emitStoreOperand(const ModRm& modrm, uint8_t hostReg, uint8_t size, uintptr_t nextRip);

    // opKind: 0=ADD, 1=OR, 4=AND, 5=SUB, 6=XOR, 7=CMP, 8=TEST.
    void emitAluRegReg(uint8_t opKind, const ModRm& modrm, uint8_t reg, bool rmIsDst,
                       uint8_t size, uintptr_t nextRip);
    void emitAluRegImm(uint8_t opKind, const ModRm& modrm, uint64_t imm,
                       uint8_t size, uintptr_t nextRip);
    void emitIncDec(const ModRm& modrm, bool increment, uint8_t size, uintptr_t nextRip);
    // shiftKind: 4=SHL, 5=SHR, 7=SAR.
    void emitShiftOverflow(uint8_t shiftKind, uint8_t bits);
    void emitShift(uint8_t shiftKind, const ModRm& modrm, bool useCl, uint8_t immCount,
                   uint8_t size, uintptr_t nextRip);
    // rotKind: 0=ROL, 1=ROR, 2=RCL, 3=RCR.
    void emitRotate(uint8_t rotKind, const ModRm& modrm, bool useCl, uint8_t immCount,
                    uint8_t size, uintptr_t nextRip);

    size_t decodeAndEmitOne(CodeReader& reader, uintptr_t rip, bool& isControlFlow);
    TranslatedBlock* compileLocked(const uint8_t* code, size_t codeSize, uintptr_t baseRip);

    // Caché JIT persistente (JitDiskCache): reubicaciones del bloque en curso y carga desde disco.
    void emitMovHostAddr(uint8_t rd, uintptr_t addr, uint8_t kind);
    void emitCtxRelAddr(uint8_t rd, const void* member);
    bool reserveBlockLocked(size_t& start);
    bool finishBlockLocked(size_t start, size_t savedOffset);
    TranslatedBlock* loadPersistedLocked(uintptr_t guestRip, size_t available);
    [[nodiscard]] uint32_t aotConfigKey() const noexcept;
    std::vector<JitReloc> aotRelocs_;
    std::vector<JitDep> aotDeps_;
    size_t aotBlockStart_ = 0;
    uint64_t statAotLoads_ = 0;

    CpuContext ctx_{};
    void* stackBase_ = nullptr;
    size_t stackSize_ = 0;
    void* teb_ = nullptr;
    size_t tebSize_ = 0;
    void* mainBase_ = nullptr;
    std::vector<void*> allocatedTlsBlocks_;
    uint8_t currentSegPrefix_ = 0;
    bool currentRexPresent_ = false;

    struct TlsModuleTemplate {
        uint32_t index = 0;
        std::vector<uint8_t> rawData;
        size_t totalSize = 4096;
    };
    std::vector<TlsModuleTemplate> tlsTemplates_;
    void installTlsTemplate(const TlsModuleTemplate& tmpl);
    // Contadores de diagnóstico del bucle (deltas en la línea "Ejecutados").
    uint64_t statTranslations_ = 0;
    uint64_t statThunks_ = 0;
    uint64_t statSlowPath_ = 0;
    std::string exeName_ = "game.exe";

    uint8_t* arena_ = nullptr;
    uint8_t* writeArena_ = nullptr;
    size_t arenaSize_ = 0;
    size_t jitOffset_ = 0;

    // Ordenado: las invalidaciones por rango (SMC, VirtualFree) borran con lower_bound en vez de
    // recorrer todas las traducciones.
    std::map<uintptr_t, TranslatedBlock> blockCache_;
    /// Borra los bloques de [base, end) y sus entradas de la caché rápida (mutex_ tomado).
    size_t eraseBlocksLocked(uintptr_t base, uintptr_t end);
    // hostCode -> guestRip del bloque (para traducir el PC de un fallo; se valida contra blockCache_).
    std::map<uintptr_t, uintptr_t> hostIndex_;
    std::vector<ExecRange> execRanges_;
    std::unordered_map<uintptr_t, std::string> failedBlocks_;
    mutable std::mutex mutex_;
    EmitterConfig config_{};

    NativeThunkHandler nativeThunk_ = nullptr;
    void* nativeThunkUser_ = nullptr;
    HyperAotCache* aotCache_ = nullptr;

    uintptr_t codeBase_ = 0;
    size_t codeSize_ = 0;
    std::vector<CodeRange> codeRanges_;
    bool codeRangeEnforced_ = false;
    std::vector<ModuleMetadata> modules_;
    // Anillo de las últimas llamadas nativas. Solo lo escribe el hilo dueño del backend
    // (cientos de miles por segundo): sin mutex ni asignaciones.
    static constexpr size_t kFlightSize = 32;
    std::array<FlightRecord, kFlightSize> flightRecorder_{};
    uint64_t flightCount_ = 0;
    static inline OryonEmitterBackend* s_activeBackend = nullptr;
    bool mode32_ = false;
    bool stopped_ = false;
    int nestDepth_ = 0;
    int abortAbove_ = 0x7FFFFFFF;
    bool watchEscape_ = false;
    bool escaped_ = false;
    CpuContext escapedCtx_{};
    bool emitOk_ = true;
    std::string lastError_;
};

}  // namespace hyperplay
