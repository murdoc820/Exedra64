// IDirect3DDevice9 de 32 bits (etapa 1: sin dibujo). Ver D3D9Device32.h.
//
// Todos los objetos (dispositivo, superficies, texturas, búferes, shaders, declaraciones, consultas,
// cadenas de intercambio, bloques de estado) viven en memoria baja y empiezan con la vtable de 32 bits.
// Las superficies de los niveles de una textura comparten el contador de referencias del contenedor,
// como en Windows. La memoria de un nivel se reserva en el primer LockRect.

#include "D3D9Device32.h"

#include <android/log.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "D3D9Probe32.h"
#include "FramePacing.h"
#include "gfx/D3D9GpuRenderer.h"
#include "cpu/OryonEmitterBackend.h"
#include "LowMem.h"
#include "Win32Arity.h"

#define LOG_TAG "HyperPlay.D3D9Dev32"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace w32alias {
uintptr_t lowFor(void* host, const std::string& name);  // NativeDllInterceptor.cpp
}

namespace d3d9dev32 {
namespace {

using u32 = uint32_t;
constexpr u32 kOk = 0;
constexpr u32 kNoInterface = 0x80004002u;
constexpr u32 kNotImpl = 0x80004001u;
constexpr u32 kInvalidCall = 0x8876086Cu;
constexpr u32 kNotFound = 0x88760866u;  // D3DERR_NOTFOUND
constexpr u32 kOutOfVideoMemory = 0x8876017Cu;

template <typename T = uint8_t>
T* G(u32 a) { return reinterpret_cast<T*>(static_cast<uintptr_t>(a)); }
u32 L(const void* p) { return static_cast<u32>(reinterpret_cast<uintptr_t>(p)); }
void put32(u32 a, u32 v) { if (a != 0) std::memcpy(G<void>(a), &v, 4); }
u32 rd32(u32 a) { u32 v = 0; if (a != 0) std::memcpy(&v, G<void>(a), 4); return v; }

// --- memoria baja -------------------------------------------------------------------------
void* lowSmall(size_t n) {
    static std::mutex m;
    static uint8_t* cur = nullptr;
    static size_t left = 0;
    std::lock_guard<std::mutex> lk(m);
    n = (n + 15) & ~size_t{15};
    if (cur == nullptr || left < n) {
        void* p = lowmem::mmapBelow4G(0x40000);
        if (p == MAP_FAILED) return nullptr;
        cur = static_cast<uint8_t*>(p);
        left = 0x40000;
    }
    uint8_t* r = cur;
    cur += n;
    left -= n;
    std::memset(r, 0, n);
    return r;
}
size_t pageRound(size_t n) { return (n + 0xFFF) & ~size_t{0xFFF}; }
// Reservas pequeñas (niveles de mip, búferes chicos) en clases de potencia de dos dentro de bloques de 1 MB: con un
// mmap por nivel cada uno ocupaba 64 KB de espacio bajo 4 GB (un mip 1x1 = 64 KB) y DmC agotaba la memoria baja al
// cargar un nivel. Las liberadas se reutilizan por clase.
constexpr size_t kSmallMax = 32768;
struct SmallPool {
    std::mutex m;
    std::vector<uint8_t*> freeList[12];  // 16 B .. 32 KB
    uint8_t* cur = nullptr;
    size_t left = 0;
};
SmallPool& smallPool() { static SmallPool p; return p; }
int smallClass(size_t n) {
    int c = 0;
    for (size_t s = 16; s < n; s <<= 1) ++c;
    return c;
}
uint8_t* smallAlloc(size_t n) {
    SmallPool& p = smallPool();
    const int c = smallClass(n);
    const size_t sz = size_t{16} << c;
    std::lock_guard<std::mutex> lk(p.m);
    if (!p.freeList[c].empty()) {
        uint8_t* r = p.freeList[c].back();
        p.freeList[c].pop_back();
        std::memset(r, 0, sz);
        return r;
    }
    if (p.left < sz) {
        void* m = lowmem::mmapBelow4G(1u << 20);
        if (m == MAP_FAILED) return nullptr;
        p.cur = static_cast<uint8_t*>(m);  // el resto del bloque anterior se pierde (< 32 KB)
        p.left = 1u << 20;
    }
    uint8_t* r = p.cur;
    p.cur += sz;
    p.left -= sz;
    return r;  // memoria nueva de mmap: ya a cero
}
void smallFree(uint8_t* ptr, size_t n) {
    SmallPool& p = smallPool();
    std::lock_guard<std::mutex> lk(p.m);
    p.freeList[smallClass(n)].push_back(ptr);
}
uint8_t* lowBig(size_t n) {
    if (n <= kSmallMax) return smallAlloc(std::max<size_t>(n, 1));
    void* p = lowmem::mmapBelow4G(pageRound(std::max<size_t>(n, 1)));
    static std::atomic<int> s_logs{0};
    static const int s_max = [] { char v[PROP_VALUE_MAX] = {}; return __system_property_get("debug.hyperplay.d3d9memlog", v) > 0 ? std::atoi(v) : 0; }();
    if (p != MAP_FAILED && s_logs.fetch_add(1) < s_max) LOGI("memoria de recurso 0x%lx..0x%lx", static_cast<unsigned long>(reinterpret_cast<uintptr_t>(p)), static_cast<unsigned long>(reinterpret_cast<uintptr_t>(p) + n));
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}
// Libera la memoria de un recurso (como Windows al destruirlo).
void freeBig(uint8_t* p, size_t n) {
    if (p == nullptr) return;
    if (n <= kSmallMax) { smallFree(p, std::max<size_t>(n, 1)); return; }
    const size_t len = pageRound(std::max<size_t>(n, 1));
    // Se devuelve también el espacio de direcciones: por debajo de 4 GB no sobra (DmC agotaba la memoria baja
    // al cargar un nivel con todas las texturas liberadas aún reservadas).
    munmap(p, len);
    static std::atomic<uint64_t> s_freed{0};
    const uint64_t total = s_freed.fetch_add(len) + len;
    if ((total >> 26) != ((total - len) >> 26)) LOGI("memoria de recursos liberada: %llu MB", static_cast<unsigned long long>(total >> 20));
}

// --- formatos ------------------------------------------------------------------------------
constexpr u32 fourcc(char a, char b, char c, char d) {
    return static_cast<u32>(a) | static_cast<u32>(b) << 8 | static_cast<u32>(c) << 16 | static_cast<u32>(d) << 24;
}
struct FmtInfo { u32 blockW; u32 blockBytes; };  // blockW 1 = sin compresión
FmtInfo fmtInfo(u32 f) {
    switch (f) {
        case fourcc('D', 'X', 'T', '1'): case fourcc('A', 'T', 'I', '1'): return {4, 8};
        case fourcc('D', 'X', 'T', '2'): case fourcc('D', 'X', 'T', '3'): case fourcc('D', 'X', 'T', '4'):
        case fourcc('D', 'X', 'T', '5'): case fourcc('A', 'T', 'I', '2'): return {4, 16};
        case 20: return {1, 3};
        case 23: case 24: case 25: case 26: case 29: case 30: case 51: case 60: case 61: case 70: case 73: case 80:
        case 81: case 111: return {1, 2};
        case 27: case 28: case 41: case 50: case 52: return {1, 1};
        case 36: case 110: case 113: case 115: return {1, 8};
        case 116: return {1, 16};
        case fourcc('N', 'U', 'L', 'L'): return {1, 0};
        default: return {1, 4};
    }
}
u32 levelDim(u32 v, u32 level) { return std::max<u32>(1, v >> level); }
u32 rowPitch(u32 fmt, u32 w) {
    const FmtInfo fi = fmtInfo(fmt);
    const u32 bw = (w + fi.blockW - 1) / fi.blockW;
    return (bw * fi.blockBytes + 3) & ~3u;
}
u32 rowCount(u32 fmt, u32 h) {
    const FmtInfo fi = fmtInfo(fmt);
    return (h + fi.blockW - 1) / fi.blockW;
}
u32 levelsFor(u32 w, u32 h, u32 d, u32 req) {
    u32 full = 1;
    for (u32 m = std::max({w, h, d}); m > 1; m >>= 1) ++full;
    return req == 0 || req > full ? full : req;
}

// --- objetos -------------------------------------------------------------------------------
enum Kind : u32 {
    kDevice = 1, kSwapChain, kSurface, kTexture, kCubeTexture, kVolumeTexture, kVolume, kVertexBuffer, kIndexBuffer,
    kVertexDecl, kVertexShader, kPixelShader, kQuery, kStateBlock,
};

struct Device;

struct Obj {
    u32 vtbl;
    u32 kind;
    std::atomic<int32_t> refs{1};
    Device* dev = nullptr;
    Obj* container = nullptr;  // superficies/volúmenes de una textura: comparten su contador
};

struct Level {
    uint8_t* data = nullptr;
    size_t bytes = 0;
};

struct Resource : Obj {
    u32 type = 0, usage = 0, pool = 0, format = 0, msType = 0, msQuality = 0;
    u32 width = 1, height = 1, depth = 1, levels = 1;
    u32 fvf = 0, size = 0;   // búferes
    uint8_t* bufData = nullptr;
    std::vector<Level> mips;  // textura: levels; cubo: 6*levels; volumen: levels
    std::vector<Resource*> children;  // superficies/volúmenes por nivel (y cara)
    u32 level = 0, face = 0;   // superficie hija
    uint32_t priority = 0, lod = 0;
    uint64_t generation = 0;   // cambia en cada Unlock (subida a la GPU)
    std::vector<const uint8_t*> ptrs;  // datos por subrecurso para gpu9::TexRef
    std::vector<uint32_t> ptrBytes;    // tamaño en bytes por subrecurso
};

struct Blob : Obj {  // shaders y declaraciones
    std::vector<uint8_t> bytes;
};

struct Query : Obj {
    u32 type = 0;
};

struct Device : Obj {
    u32 direct3d = 0;
    u32 adapter = 0, deviceType = 1, focusWindow = 0, behaviorFlags = 0;
    uint8_t pp[56] = {};  // D3DPRESENT_PARAMETERS
    Obj* swapChain = nullptr;
    Resource* backBuffer = nullptr;
    Resource* autoDepth = nullptr;
    std::mutex mtx;
    Resource* renderTargets[4] = {};
    Resource* depthStencil = nullptr;
    u32 textures[20] = {};
    u32 renderState[256] = {};
    u32 samplerState[20][14] = {};
    u32 stageState[8][33] = {};
    float vsF[256 * 4] = {}, psF[224 * 4] = {};
    int32_t vsI[16 * 4] = {}, psI[16 * 4] = {};
    u32 vsB[16] = {}, psB[16] = {};
    uint64_t vsConstGen = 1, psConstGen = 1, rsGen = 1, ssGen = 1;
    u32 streams[16] = {}, streamOffset[16] = {}, streamStride[16] = {}, streamFreq[16] = {};
    u32 indices = 0, vdecl = 0, vs = 0, ps = 0, fvf = 0;
    uint8_t viewport[24] = {};
    uint8_t scissor[16] = {};
    float clipPlanes[6][4] = {};
    float npatch = 0;
    u32 softwareVp = 0;
    u32 paletteCurrent = 0;
    uint64_t frames = 0;
    std::chrono::steady_clock::time_point nextFrame{};
};

u32 g_vt[kStateBlock + 1] = {};

void addRef(Obj* o) {
    if (o == nullptr) return;
    if (o->container != nullptr) o = o->container;
    o->refs.fetch_add(1);
}
void destroy(Obj* o);
u32 release(Obj* o) {
    if (o == nullptr) return 0;
    if (o->container != nullptr) o = o->container;
    const int32_t r = o->refs.fetch_sub(1) - 1;
    if (r == 0) destroy(o);
    return r < 0 ? 0 : static_cast<u32>(r);
}
Obj* O(u32 p) { return G<Obj>(p); }
void addRefG(u32 p) { if (p != 0) addRef(O(p)); }
void releaseG(u32 p) { if (p != 0) release(O(p)); }

void destroy(Obj* o) {
    if (o->kind == kTexture || o->kind == kCubeTexture || o->kind == kVolumeTexture || o->kind == kSurface ||
        o->kind == kVertexBuffer || o->kind == kIndexBuffer) {
        auto* r = static_cast<Resource*>(o);
        if (o->kind != kVertexBuffer && o->kind != kIndexBuffer) gpu9::Renderer::shared().release(o);
        for (Level& l : r->mips) { freeBig(l.data, l.bytes); l.data = nullptr; }
        freeBig(r->bufData, r->size);
        r->bufData = nullptr;
        // Los objetos (pequeños) se quedan en el bloque de memoria baja: un puntero viejo no cae.
    }
}

template <typename T>
T* make(u32 kind, Device* dev) {
    void* mem = lowSmall(sizeof(T));
    if (mem == nullptr) return nullptr;
    T* o = new (mem) T();
    o->vtbl = g_vt[kind];
    o->kind = kind;
    o->dev = dev;
    return o;
}

// --- recursos ------------------------------------------------------------------------------
Resource* makeSurface(Device* dev, u32 w, u32 h, u32 fmt, u32 usage, u32 pool, u32 ms, u32 msq) {
    Resource* s = make<Resource>(kSurface, dev);
    if (s == nullptr) return nullptr;
    s->type = 1;
    s->width = w;
    s->height = h;
    s->format = fmt;
    s->usage = usage;
    s->pool = pool;
    s->msType = ms;
    s->msQuality = msq;
    s->mips.resize(1);
    return s;
}

// Memoria de un nivel (reserva diferida).
Level* levelMem(Resource* r, u32 index, u32 w, u32 h, u32 d) {
    if (index >= r->mips.size()) return nullptr;
    Level& l = r->mips[index];
    if (l.data == nullptr) {
        l.bytes = static_cast<size_t>(rowPitch(r->format, w)) * rowCount(r->format, h) * d;
        l.data = lowBig(l.bytes);
        if (l.data == nullptr) {
            LOGW("sin memoria baja para %zu bytes (textura %ux%u fmt 0x%x)", l.bytes, w, h, r->format);
            l.bytes = 0;
            return nullptr;
        }
    }
    return &l;
}

// Superficie/volumen hijo de una textura (cacheado).
Resource* childOf(Resource* tex, u32 face, u32 level) {
    const u32 idx = face * tex->levels + level;
    if (idx >= tex->children.size()) return nullptr;
    if (tex->children[idx] == nullptr) {
        const bool vol = tex->kind == kVolumeTexture;
        Resource* s = make<Resource>(vol ? kVolume : kSurface, tex->dev);
        if (s == nullptr) return nullptr;
        s->container = tex;
        s->type = vol ? 2 : 1;
        s->width = levelDim(tex->width, level);
        s->height = levelDim(tex->height, level);
        s->depth = vol ? levelDim(tex->depth, level) : 1;
        s->format = tex->format;
        s->usage = tex->usage;
        s->pool = tex->pool;
        s->level = level;
        s->face = face;
        tex->children[idx] = s;
    }
    return tex->children[idx];
}

u32 lockSurface(Resource* s, u32 locked, u32 rect) {
    Resource* owner = s->container != nullptr ? static_cast<Resource*>(s->container) : s;
    const u32 idx = s->container != nullptr ? s->face * owner->levels + s->level : 0;
    Level* l = levelMem(owner, idx, s->width, s->height, 1);
    if (l == nullptr || locked == 0) return kInvalidCall;
    const u32 pitch = rowPitch(s->format, s->width);
    u32 offset = 0;
    if (rect != 0) {
        const u32 left = rd32(rect), top = rd32(rect + 4);
        const FmtInfo fi = fmtInfo(s->format);
        offset = (top / fi.blockW) * pitch + (left / fi.blockW) * fi.blockBytes;
    }
    put32(locked, pitch);
    put32(locked + 4, L(l->data + offset));
    return kOk;
}
void unlockSurface(Resource* s) {
    Resource* owner = s->container != nullptr ? static_cast<Resource*>(s->container) : s;
    ++owner->generation;
}

u32 surfaceDesc(Resource* s, u32 desc) {
    if (desc == 0) return kInvalidCall;
    const u32 v[8] = {s->format, s->type == 2 ? 2u : 1u, s->usage, s->pool, s->msType, s->msQuality, s->width, s->height};
    std::memcpy(G<void>(desc), v, sizeof(v));
    return kOk;
}

// --- vista para el renderizador GPU (gfx/D3D9GpuRenderer) --------------------------------------
gpu9::TexRef texRef(Resource* r) {
    gpu9::TexRef t;
    if (r == nullptr) return t;
    Resource* owner = r->container != nullptr ? static_cast<Resource*>(r->container) : r;
    t.key = owner;
    t.kind = owner->kind == kCubeTexture ? 1 : owner->kind == kVolumeTexture ? 2 : 0;
    t.format = owner->format;
    t.width = owner->width;
    t.height = owner->height;
    t.depth = owner->depth;
    t.levels = owner->kind == kSurface ? 1 : owner->levels;
    t.usage = owner->usage;
    t.generation = owner->generation;
    owner->ptrs.resize(owner->mips.size());
    owner->ptrBytes.resize(owner->mips.size());
    for (size_t i = 0; i < owner->mips.size(); ++i) {
        owner->ptrs[i] = owner->mips[i].data;
        owner->ptrBytes[i] = owner->mips[i].bytes;
    }
    t.data = owner->ptrs.data();
    t.subresourceBytes = owner->ptrBytes.data();
    t.subresources = static_cast<uint32_t>(owner->ptrs.size());
    if (r->container != nullptr) {
        t.level = r->level;
        t.face = r->face;
    }
    return t;
}
gpu9::Stream streamOf(Resource* b, u32 offset, u32 stride, u32 freq) {
    gpu9::Stream s;
    if (b == nullptr) return s;
    s.key = b;
    s.data = b->bufData;
    s.size = b->size;
    s.generation = b->generation;
    s.offset = offset;
    s.stride = stride;
    s.freq = freq;
    s.dynamic = (b->usage & 0x200u) != 0;  // D3DUSAGE_DYNAMIC
    return s;
}
void fillDrawState(Device* d, gpu9::DrawState& s) {
    if (d->vs != 0) {
        auto* b = static_cast<Blob*>(O(d->vs));
        s.vsKey = b;
        s.vsTokens = reinterpret_cast<const uint32_t*>(b->bytes.data());
        s.vsCount = b->bytes.size() / 4;
    }
    if (d->ps != 0) {
        auto* b = static_cast<Blob*>(O(d->ps));
        s.psKey = b;
        s.psTokens = reinterpret_cast<const uint32_t*>(b->bytes.data());
        s.psCount = b->bytes.size() / 4;
    }
    s.vsF = d->vsF; s.vsI = d->vsI; s.vsB = d->vsB;
    s.psF = d->psF; s.psI = d->psI; s.psB = d->psB;
    s.vsConstGen = d->vsConstGen;
    s.psConstGen = d->psConstGen;
    s.rsGen = d->rsGen;
    s.ssGen = d->ssGen;
    if (d->vdecl != 0) s.decl = static_cast<Blob*>(O(d->vdecl))->bytes.data();
    for (u32 i = 0; i < 16; ++i)
        if (d->streams[i] != 0) s.streams[i] = streamOf(static_cast<Resource*>(O(d->streams[i])), d->streamOffset[i],
                                                         d->streamStride[i], d->streamFreq[i]);
    s.renderState = d->renderState;
    s.samplerState = d->samplerState;
    for (u32 i = 0; i < 20; ++i)
        if (d->textures[i] != 0) s.textures[i] = texRef(static_cast<Resource*>(O(d->textures[i])));
    for (u32 i = 0; i < 4; ++i) s.targets[i] = texRef(d->renderTargets[i]);
    s.depth = texRef(d->depthStencil);
    std::memcpy(s.viewport, d->viewport, 16);
    std::memcpy(&s.minZ, d->viewport + 16, 4);
    std::memcpy(&s.maxZ, d->viewport + 20, 4);
    std::memcpy(s.scissor, d->scissor, 16);
}

// ===========================================================================================
// IUnknown / IDirect3DResource9 comunes
// ===========================================================================================
u32 QI(u32 t, u32, u32 ppv) { addRef(O(t)); put32(ppv, t); return kOk; }
u32 AddRef(u32 t) {
    Obj* o = O(t);
    if (o->container != nullptr) o = o->container;
    return static_cast<u32>(o->refs.fetch_add(1) + 1);
}
u32 Release(u32 t) { return release(O(t)); }
u32 GetDevice(u32 t, u32 pp) { Device* d = O(t)->dev; addRef(d); put32(pp, L(d)); return kOk; }
u32 SetPrivateData(u32, u32, u32, u32, u32) { return kOk; }
u32 GetPrivateData(u32, u32, u32, u32) { return kNotFound; }
u32 FreePrivateData(u32, u32) { return kOk; }
u32 SetPriority(u32 t, u32 p) { auto* r = static_cast<Resource*>(O(t)); const u32 o = r->priority; r->priority = p; return o; }
u32 GetPriority(u32 t) { return static_cast<Resource*>(O(t))->priority; }
u32 PreLoad(u32) { return 0; }
u32 GetType(u32 t) { return static_cast<Resource*>(O(t))->type; }

void resourceBase(std::vector<void*>& m) {
    m.push_back(w32arity::hpFn(&QI));
    m.push_back(w32arity::hpFn(&AddRef));
    m.push_back(w32arity::hpFn(&Release));
    m.push_back(w32arity::hpFn(&GetDevice));
    m.push_back(w32arity::hpFn(&SetPrivateData));
    m.push_back(w32arity::hpFn(&GetPrivateData));
    m.push_back(w32arity::hpFn(&FreePrivateData));
    m.push_back(w32arity::hpFn(&SetPriority));
    m.push_back(w32arity::hpFn(&GetPriority));
    m.push_back(w32arity::hpFn(&PreLoad));
    m.push_back(w32arity::hpFn(&GetType));
}

// --- IDirect3DBaseTexture9 ---
Resource* R(u32 t) { return static_cast<Resource*>(O(t)); }
u32 SetLOD(u32 t, u32 lod) { const u32 o = R(t)->lod; R(t)->lod = lod; return o; }
u32 GetLOD(u32 t) { return R(t)->lod; }
u32 GetLevelCount(u32 t) { return R(t)->levels; }
u32 SetAutoGenFilterType(u32, u32) { return kOk; }
u32 GetAutoGenFilterType(u32) { return 2; }  // D3DTEXF_LINEAR
u32 GenerateMipSubLevels(u32) { return 0; }

void baseTexture(std::vector<void*>& m) {
    resourceBase(m);
    m.push_back(w32arity::hpFn(&SetLOD));
    m.push_back(w32arity::hpFn(&GetLOD));
    m.push_back(w32arity::hpFn(&GetLevelCount));
    m.push_back(w32arity::hpFn(&SetAutoGenFilterType));
    m.push_back(w32arity::hpFn(&GetAutoGenFilterType));
    m.push_back(w32arity::hpFn(&GenerateMipSubLevels));
}

// --- IDirect3DTexture9 ---
u32 Tex_GetLevelDesc(u32 t, u32 level, u32 desc) {
    Resource* s = childOf(R(t), 0, level);
    return s != nullptr ? surfaceDesc(s, desc) : kInvalidCall;
}
u32 Tex_GetSurfaceLevel(u32 t, u32 level, u32 pp) {
    Resource* s = childOf(R(t), 0, level);
    if (s == nullptr) { put32(pp, 0); return kInvalidCall; }
    addRef(s);
    put32(pp, L(s));
    return kOk;
}
u32 Tex_LockRect(u32 t, u32 level, u32 locked, u32 rect, u32) {
    Resource* s = childOf(R(t), 0, level);
    return s != nullptr ? lockSurface(s, locked, rect) : kInvalidCall;
}
u32 Tex_UnlockRect(u32 t, u32) { ++R(t)->generation; return kOk; }
u32 Tex_AddDirtyRect(u32, u32) { return kOk; }

// --- IDirect3DCubeTexture9 ---
u32 Cube_GetLevelDesc(u32 t, u32 level, u32 desc) {
    Resource* s = childOf(R(t), 0, level);
    return s != nullptr ? surfaceDesc(s, desc) : kInvalidCall;
}
u32 Cube_GetCubeMapSurface(u32 t, u32 face, u32 level, u32 pp) {
    Resource* s = childOf(R(t), face, level);
    if (s == nullptr) { put32(pp, 0); return kInvalidCall; }
    addRef(s);
    put32(pp, L(s));
    return kOk;
}
u32 Cube_LockRect(u32 t, u32 face, u32 level, u32 locked, u32 rect, u32) {
    Resource* s = childOf(R(t), face, level);
    return s != nullptr ? lockSurface(s, locked, rect) : kInvalidCall;
}
u32 Cube_UnlockRect(u32 t, u32, u32) { ++R(t)->generation; return kOk; }
u32 Cube_AddDirtyRect(u32, u32, u32) { return kOk; }

// --- IDirect3DVolumeTexture9 / IDirect3DVolume9 ---
u32 lockVolume(Resource* v, u32 locked) {
    Resource* tex = static_cast<Resource*>(v->container);
    Level* l = levelMem(tex, v->level, v->width, v->height, v->depth);
    if (l == nullptr || locked == 0) return kInvalidCall;
    const u32 rp = rowPitch(v->format, v->width);
    put32(locked, rp);
    put32(locked + 4, rp * rowCount(v->format, v->height));
    put32(locked + 8, L(l->data));
    return kOk;
}
u32 volumeDesc(Resource* v, u32 desc) {
    if (desc == 0) return kInvalidCall;
    const u32 x[7] = {v->format, 2, v->usage, v->pool, v->width, v->height, v->depth};
    std::memcpy(G<void>(desc), x, sizeof(x));
    return kOk;
}
u32 VolTex_GetLevelDesc(u32 t, u32 level, u32 desc) {
    Resource* v = childOf(R(t), 0, level);
    return v != nullptr ? volumeDesc(v, desc) : kInvalidCall;
}
u32 VolTex_GetVolumeLevel(u32 t, u32 level, u32 pp) {
    Resource* v = childOf(R(t), 0, level);
    if (v == nullptr) { put32(pp, 0); return kInvalidCall; }
    addRef(v);
    put32(pp, L(v));
    return kOk;
}
u32 VolTex_LockBox(u32 t, u32 level, u32 locked, u32, u32) {
    Resource* v = childOf(R(t), 0, level);
    return v != nullptr ? lockVolume(v, locked) : kInvalidCall;
}
u32 VolTex_UnlockBox(u32 t, u32) { ++R(t)->generation; return kOk; }
u32 VolTex_AddDirtyBox(u32, u32) { return kOk; }

u32 Vol_GetContainer(u32 t, u32, u32 pp) { Obj* c = O(t)->container; addRef(c); put32(pp, L(c)); return c ? kOk : kNoInterface; }
u32 Vol_GetDesc(u32 t, u32 desc) { return volumeDesc(R(t), desc); }
u32 Vol_LockBox(u32 t, u32 locked, u32, u32) { return lockVolume(R(t), locked); }
u32 Vol_UnlockBox(u32 t) { ++static_cast<Resource*>(O(t)->container)->generation; return kOk; }

// --- IDirect3DSurface9 ---
u32 Surf_GetContainer(u32 t, u32, u32 pp) {
    Obj* c = O(t)->container != nullptr ? O(t)->container : static_cast<Obj*>(O(t)->dev);
    addRef(c);
    put32(pp, L(c));
    return kOk;
}
u32 Surf_GetDesc(u32 t, u32 desc) { return surfaceDesc(R(t), desc); }
u32 Surf_LockRect(u32 t, u32 locked, u32 rect, u32) { return lockSurface(R(t), locked, rect); }
u32 Surf_UnlockRect(u32 t) { unlockSurface(R(t)); return kOk; }
u32 Surf_GetDC(u32, u32 phdc) { put32(phdc, 0); return kInvalidCall; }
u32 Surf_ReleaseDC(u32, u32) { return kInvalidCall; }

// --- IDirect3DVertexBuffer9 / IDirect3DIndexBuffer9 ---
u32 Buf_Lock(u32 t, u32 offset, u32 size, u32 ppData, u32) {
    Resource* b = R(t);
    if (b->bufData == nullptr || offset > b->size) { put32(ppData, 0); return kInvalidCall; }
    (void)size;
    put32(ppData, L(b->bufData + offset));
    return kOk;
}
u32 Buf_Unlock(u32 t) { ++R(t)->generation; return kOk; }
u32 VB_GetDesc(u32 t, u32 desc) {
    Resource* b = R(t);
    const u32 v[6] = {0x64 /* D3DFMT_VERTEXDATA */, 6, b->usage, b->pool, b->size, b->fvf};
    if (desc != 0) std::memcpy(G<void>(desc), v, sizeof(v));
    return kOk;
}
u32 IB_GetDesc(u32 t, u32 desc) {
    Resource* b = R(t);
    const u32 v[5] = {b->format, 7, b->usage, b->pool, b->size};
    if (desc != 0) std::memcpy(G<void>(desc), v, sizeof(v));
    return kOk;
}

// --- IDirect3DVertexDeclaration9 / shaders ---
Blob* B(u32 t) { return static_cast<Blob*>(O(t)); }
u32 Decl_GetDeclaration(u32 t, u32 elems, u32 pNum) {
    const auto& b = B(t)->bytes;
    put32(pNum, static_cast<u32>(b.size() / 8));
    if (elems != 0) std::memcpy(G<void>(elems), b.data(), b.size());
    return kOk;
}
u32 Shader_GetFunction(u32 t, u32 data, u32 pSize) {
    const auto& b = B(t)->bytes;
    if (data == 0) { put32(pSize, static_cast<u32>(b.size())); return kOk; }
    if (rd32(pSize) < b.size()) return kInvalidCall;
    std::memcpy(G<void>(data), b.data(), b.size());
    return kOk;
}

// --- IDirect3DQuery9 ---
u32 Query_GetType(u32 t) { return static_cast<Query*>(O(t))->type; }
u32 queryDataSize(u32 type) {
    switch (type) {
        case 4: return 16;   // VCACHE
        case 8: return 4;    // EVENT (BOOL)
        case 9: return 4;    // OCCLUSION (DWORD)
        case 10: return 8;   // TIMESTAMP
        case 11: return 4;   // TIMESTAMPDISJOINT
        case 12: return 8;   // TIMESTAMPFREQ
        default: return 0;
    }
}
u32 Query_GetDataSize(u32 t) { return queryDataSize(static_cast<Query*>(O(t))->type); }
u32 Query_Issue(u32, u32) { return kOk; }
u32 Query_GetData(u32 t, u32 data, u32 size, u32) {
    const u32 type = static_cast<Query*>(O(t))->type;
    if (data == 0 || size == 0) return kOk;
    uint8_t buf[16] = {};
    if (type == 8) { const u32 v = 1; std::memcpy(buf, &v, 4); }
    else if (type == 9) { const u32 v = 100000; std::memcpy(buf, &v, 4); }  // todo visible
    else if (type == 10) {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const uint64_t ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
        std::memcpy(buf, &ns, 8);
    } else if (type == 12) { const uint64_t f = 1000000000ull; std::memcpy(buf, &f, 8); }
    std::memcpy(G<void>(data), buf, std::min<u32>(size, 16));
    return kOk;
}

// --- IDirect3DStateBlock9 ---
u32 SB_Capture(u32) { return kOk; }
u32 SB_Apply(u32) { return kOk; }

// ===========================================================================================
// IDirect3DDevice9
// ===========================================================================================
Device* D(u32 t) { return static_cast<Device*>(O(t)); }

void setRef(u32& slot, u32 value) {
    if (slot == value) return;
    addRefG(value);
    releaseG(slot);
    slot = value;
}
void setRefPtr(Resource*& slot, Resource* value) {
    if (slot == value) return;
    addRef(value);
    if (slot != nullptr) release(slot);
    slot = value;
}

void createBackBuffers(Device* d) {
    u32 w = rd32(L(d->pp)), h = rd32(L(d->pp) + 4), fmt = rd32(L(d->pp) + 8);
    if (w == 0) { w = d3d9probe32::display().width; std::memcpy(d->pp, &w, 4); }
    if (h == 0) { h = d3d9probe32::display().height; std::memcpy(d->pp + 4, &h, 4); }
    if (fmt == 0) { fmt = 22; std::memcpy(d->pp + 8, &fmt, 4); }
    const u32 ms = rd32(L(d->pp) + 16), msq = rd32(L(d->pp) + 20);
    if (d->backBuffer != nullptr) release(d->backBuffer);
    d->backBuffer = makeSurface(d, w, h, fmt, 1 /* RENDERTARGET */, 0, ms, msq);
    setRefPtr(d->renderTargets[0], d->backBuffer);
    for (int i = 1; i < 4; ++i) setRefPtr(d->renderTargets[i], nullptr);
    if (d->autoDepth != nullptr) { release(d->autoDepth); d->autoDepth = nullptr; }
    if (rd32(L(d->pp) + 36) != 0) {
        d->autoDepth = makeSurface(d, w, h, rd32(L(d->pp) + 40), 2 /* DEPTHSTENCIL */, 0, ms, msq);
    }
    setRefPtr(d->depthStencil, d->autoDepth);
    const u32 vp[4] = {0, 0, w, h};
    const float z[2] = {0.0f, 1.0f};
    std::memcpy(d->viewport, vp, 16);
    std::memcpy(d->viewport + 16, z, 8);
    const int32_t sc[4] = {0, 0, static_cast<int32_t>(w), static_cast<int32_t>(h)};
    std::memcpy(d->scissor, sc, 16);
    LOGI("backbuffer %ux%u fmt %u (ms %u), depth automático: %s", w, h, fmt, ms, d->autoDepth ? "sí" : "no");
}

u32 Dev_QI(u32 t, u32 riid, u32 ppv) {
    const u32 d1 = rd32(riid);
    // IUnknown o IDirect3DDevice9 {D0223B96-...}; IDirect3DDevice9Ex {B18B10CE-...} no.
    if (d1 == 0 || d1 == 0xD0223B96u) { addRef(O(t)); put32(ppv, t); return kOk; }
    put32(ppv, 0);
    return kNoInterface;
}
u32 Dev_TestCooperativeLevel(u32) { return kOk; }
u32 Dev_GetAvailableTextureMem(u32) { return 0x20000000u; }  // 512 MB (valor típico de una GPU de 2013)
u32 Dev_EvictManagedResources(u32) { return kOk; }
u32 Dev_GetDirect3D(u32 t, u32 pp) { put32(pp, D(t)->direct3d); return kOk; }
u32 Dev_GetDeviceCaps(u32 t, u32 caps) { d3d9probe32::fillCaps(D(t)->deviceType, caps); return kOk; }
u32 Dev_GetDisplayMode(u32 t, u32, u32 mode) {
    if (mode == 0) return kInvalidCall;
    const u32 v[4] = {rd32(L(D(t)->pp)), rd32(L(D(t)->pp) + 4), 60, 22};
    std::memcpy(G<void>(mode), v, sizeof(v));
    return kOk;
}
u32 Dev_GetCreationParameters(u32 t, u32 p) {
    Device* d = D(t);
    const u32 v[4] = {d->adapter, d->deviceType, d->focusWindow, d->behaviorFlags};
    if (p != 0) std::memcpy(G<void>(p), v, sizeof(v));
    return kOk;
}
u32 Dev_SetCursorProperties(u32, u32, u32, u32) { return kOk; }
u32 Dev_SetCursorPosition(u32, u32, u32, u32) { return 0; }
u32 Dev_ShowCursor(u32, u32) { return 0; }
u32 Dev_CreateAdditionalSwapChain(u32, u32, u32 pp) { put32(pp, 0); return kNotImpl; }
u32 Dev_GetSwapChain(u32 t, u32 i, u32 pp) {
    if (i != 0) { put32(pp, 0); return kInvalidCall; }
    addRef(D(t)->swapChain);
    put32(pp, L(D(t)->swapChain));
    return kOk;
}
u32 Dev_GetNumberOfSwapChains(u32) { return 1; }
u32 Dev_Reset(u32 t, u32 pp) {
    Device* d = D(t);
    if (pp == 0) return kInvalidCall;
    std::lock_guard<std::mutex> lk(d->mtx);
    gpu9::Renderer::shared().sync();
    std::memcpy(d->pp, G<void>(pp), sizeof(d->pp));
    createBackBuffers(d);
    std::memcpy(G<void>(pp), d->pp, sizeof(d->pp));
    LOGI("Reset");
    return kOk;
}
u32 present(Device* d) {
    ::hyperplay::pacing::setRenderThread(static_cast<pid_t>(gettid()));
    // Limitador preciso, tope de FPS (60/90/120), protección térmica, ADPF y pausa (Quick Resume).
    ::hyperplay::pacing::paceFrame();
    gpu9::Renderer::shared().present(texRef(d->backBuffer));
    if ((++d->frames % 600) == 1) LOGI("Present #%llu", static_cast<unsigned long long>(d->frames));
    return kOk;
}
u32 Dev_Present(u32 t, u32, u32, u32, u32) { return present(D(t)); }
u32 Dev_GetBackBuffer(u32 t, u32 iSwap, u32 iBack, u32, u32 pp) {
    if (iSwap != 0 || iBack != 0) { put32(pp, 0); return kInvalidCall; }
    addRef(D(t)->backBuffer);
    put32(pp, L(D(t)->backBuffer));
    return kOk;
}
u32 Dev_GetRasterStatus(u32, u32, u32 p) { if (p) { put32(p, 0); put32(p + 4, 0); } return kOk; }
u32 Dev_SetDialogBoxMode(u32, u32) { return kOk; }
void Dev_SetGammaRamp(u32, u32, u32, u32) {}
void Dev_GetGammaRamp(u32, u32, u32 ramp) {
    if (ramp == 0) return;
    auto* r = G<uint16_t>(ramp);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 256; ++i) r[c * 256 + i] = static_cast<uint16_t>(i * 257);
}

u32 makeTexture(Device* d, u32 kind, u32 w, u32 h, u32 depth, u32 levels, u32 usage, u32 fmt, u32 pool, u32 pp) {
    if (pp == 0 || w == 0 || h == 0) return kInvalidCall;
    Resource* t = make<Resource>(kind, d);
    if (t == nullptr) return kOutOfVideoMemory;
    t->type = kind == kTexture ? 3 : kind == kVolumeTexture ? 4 : 5;
    t->width = w;
    t->height = h;
    t->depth = depth;
    t->format = fmt;
    t->usage = usage;
    t->pool = pool;
    t->levels = (usage & 0x400 /* AUTOGENMIPMAP */) ? 1 : levelsFor(w, h, depth, levels);
    const u32 faces = kind == kCubeTexture ? 6 : 1;
    t->mips.resize(static_cast<size_t>(faces) * t->levels);
    t->children.resize(static_cast<size_t>(faces) * t->levels, nullptr);
    put32(pp, L(t));
    return kOk;
}
u32 Dev_CreateTexture(u32 t, u32 w, u32 h, u32 levels, u32 usage, u32 fmt, u32 pool, u32 pp, u32) {
    return makeTexture(D(t), kTexture, w, h, 1, levels, usage, fmt, pool, pp);
}
u32 Dev_CreateVolumeTexture(u32 t, u32 w, u32 h, u32 d, u32 levels, u32 usage, u32 fmt, u32 pool, u32 pp, u32) {
    return makeTexture(D(t), kVolumeTexture, w, h, d, levels, usage, fmt, pool, pp);
}
u32 Dev_CreateCubeTexture(u32 t, u32 edge, u32 levels, u32 usage, u32 fmt, u32 pool, u32 pp, u32) {
    return makeTexture(D(t), kCubeTexture, edge, edge, 1, levels, usage, fmt, pool, pp);
}
u32 makeBuffer(Device* d, u32 kind, u32 len, u32 usage, u32 fvfOrFmt, u32 pool, u32 pp) {
    if (pp == 0 || len == 0) return kInvalidCall;
    Resource* b = make<Resource>(kind, d);
    if (b == nullptr) return kOutOfVideoMemory;
    b->type = kind == kVertexBuffer ? 6 : 7;
    b->size = len;
    b->usage = usage;
    b->pool = pool;
    if (kind == kVertexBuffer) b->fvf = fvfOrFmt; else b->format = fvfOrFmt;
    b->bufData = lowBig(len);
    if (b->bufData == nullptr) { put32(pp, 0); return kOutOfVideoMemory; }
    put32(pp, L(b));
    return kOk;
}
u32 Dev_CreateVertexBuffer(u32 t, u32 len, u32 usage, u32 fvf, u32 pool, u32 pp, u32) {
    return makeBuffer(D(t), kVertexBuffer, len, usage, fvf, pool, pp);
}
u32 Dev_CreateIndexBuffer(u32 t, u32 len, u32 usage, u32 fmt, u32 pool, u32 pp, u32) {
    return makeBuffer(D(t), kIndexBuffer, len, usage, fmt, pool, pp);
}
u32 Dev_CreateRenderTarget(u32 t, u32 w, u32 h, u32 fmt, u32 ms, u32 msq, u32, u32 pp, u32) {
    Resource* s = makeSurface(D(t), w, h, fmt, 1, 0, ms, msq);
    put32(pp, L(s));
    return s ? kOk : kOutOfVideoMemory;
}
u32 Dev_CreateDepthStencilSurface(u32 t, u32 w, u32 h, u32 fmt, u32 ms, u32 msq, u32, u32 pp, u32) {
    Resource* s = makeSurface(D(t), w, h, fmt, 2, 0, ms, msq);
    put32(pp, L(s));
    return s ? kOk : kOutOfVideoMemory;
}
// Copia en CPU del nivel `index` de src en dst (mismas dimensiones de nivel).
void copyLevel(Resource* src, u32 sIdx, Resource* dst, u32 dIdx, u32 w, u32 h, u32 depth) {
    Level* a = levelMem(src, sIdx, w, h, depth);
    Level* b = levelMem(dst, dIdx, w, h, depth);
    if (a == nullptr || b == nullptr) return;
    std::memcpy(b->data, a->data, std::min(a->bytes, b->bytes));
}
u32 Dev_UpdateSurface(u32, u32 src, u32 srcRect, u32 dst, u32 dstPoint) {
    if (src == 0 || dst == 0) return kInvalidCall;
    Resource* s = R(src);
    Resource* d = R(dst);
    Resource* so = s->container ? static_cast<Resource*>(s->container) : s;
    Resource* dO = d->container ? static_cast<Resource*>(d->container) : d;
    {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.d3d9vid", v) > 0 && v[0] == '1') {
            LOGI("UpdateSurface %ux%u fmt=%u -> %ux%u fmt=%u srcRect=%u dstPt=%u",
                 s->width, s->height, s->format, d->width, d->height, d->format, srcRect, dstPoint);
        }
    }
    const u32 sIdx = s->container ? s->face * so->levels + s->level : 0;
    const u32 dIdx = d->container ? d->face * dO->levels + d->level : 0;
    if (srcRect == 0 && dstPoint == 0 && s->width == d->width && s->height == d->height) {
        copyLevel(so, sIdx, dO, dIdx, s->width, s->height, 1);
        ++dO->generation;
        return kOk;
    }
    const int32_t sx0 = srcRect ? static_cast<int32_t>(rd32(srcRect)) : 0;
    const int32_t sy0 = srcRect ? static_cast<int32_t>(rd32(srcRect + 4)) : 0;
    const int32_t sx1 = srcRect ? static_cast<int32_t>(rd32(srcRect + 8)) : static_cast<int32_t>(s->width);
    const int32_t sy1 = srcRect ? static_cast<int32_t>(rd32(srcRect + 12)) : static_cast<int32_t>(s->height);
    const int32_t dx0 = dstPoint ? static_cast<int32_t>(rd32(dstPoint)) : 0;
    const int32_t dy0 = dstPoint ? static_cast<int32_t>(rd32(dstPoint + 4)) : 0;
    if (sx0 < 0 || sy0 < 0 || dx0 < 0 || dy0 < 0 || sx1 <= sx0 || sy1 <= sy0) return kOk;
    const u32 cw = std::min(static_cast<u32>(sx1 - sx0), d->width - static_cast<u32>(dx0));
    const u32 ch = std::min(static_cast<u32>(sy1 - sy0), d->height - static_cast<u32>(dy0));
    Level* a = levelMem(so, sIdx, s->width, s->height, 1);
    Level* b = levelMem(dO, dIdx, d->width, d->height, 1);
    if (a == nullptr || b == nullptr) return kOk;
    const FmtInfo fi = fmtInfo(s->format);
    const u32 sp = rowPitch(s->format, s->width);
    const u32 dp = rowPitch(d->format, d->width);
    const u32 rows = std::max(1u, ch / fi.blockW);
    const u32 rowBytes = std::max(1u, cw / fi.blockW) * fi.blockBytes;
    const u32 sRow0 = static_cast<u32>(sy0) / fi.blockW;
    const u32 dRow0 = static_cast<u32>(dy0) / fi.blockW;
    const u32 sColB = (static_cast<u32>(sx0) / fi.blockW) * fi.blockBytes;
    const u32 dColB = (static_cast<u32>(dx0) / fi.blockW) * fi.blockBytes;
    for (u32 r = 0; r < rows; ++r) {
        std::memcpy(b->data + (dRow0 + r) * dp + dColB, a->data + (sRow0 + r) * sp + sColB, rowBytes);
    }
    ++dO->generation;
    return kOk;
}
u32 Dev_UpdateTexture(u32, u32 src, u32 dst) {
    if (src == 0 || dst == 0) return kInvalidCall;
    Resource* s = R(src);
    Resource* d = R(dst);
    const u32 faces = s->kind == kCubeTexture ? 6 : 1;
    // Los niveles de destino se alinean por tamaño con los de origen.
    u32 skip = 0;
    while (skip < s->levels && levelDim(s->width, skip) > d->width) ++skip;
    for (u32 f = 0; f < faces; ++f)
        for (u32 l = 0; l + skip < s->levels && l < d->levels; ++l)
            copyLevel(s, f * s->levels + l + skip, d, f * d->levels + l, levelDim(d->width, l), levelDim(d->height, l),
                      levelDim(d->depth, l));
    ++d->generation;
    return kOk;
}
u32 Dev_GetRenderTargetData(u32, u32, u32) { return kOk; }
u32 Dev_GetFrontBufferData(u32, u32, u32) { return kOk; }
u32 Dev_StretchRect(u32, u32 src, u32 srcRect, u32 dst, u32 dstRect, u32 filter) {
    if (src == 0 || dst == 0) return kInvalidCall;
    gpu9::Renderer::shared().stretchRect(texRef(R(src)), srcRect ? G<int32_t>(srcRect) : nullptr, texRef(R(dst)),
                                         dstRect ? G<int32_t>(dstRect) : nullptr, filter);
    return kOk;
}
u32 Dev_ColorFill(u32, u32 surf, u32 rect, u32 color) {
    if (surf == 0) return kInvalidCall;
    gpu9::Renderer::shared().colorFill(texRef(R(surf)), rect ? G<int32_t>(rect) : nullptr, color);
    return kOk;
}
u32 Dev_CreateOffscreenPlainSurface(u32 t, u32 w, u32 h, u32 fmt, u32 pool, u32 pp, u32) {
    Resource* s = makeSurface(D(t), w, h, fmt, 0, pool, 0, 0);
    put32(pp, L(s));
    return s ? kOk : kOutOfVideoMemory;
}
u32 Dev_SetRenderTarget(u32 t, u32 idx, u32 surf) {
    Device* d = D(t);
    if (idx >= 4 || (idx == 0 && surf == 0)) return kInvalidCall;
    setRefPtr(d->renderTargets[idx], static_cast<Resource*>(O(surf)));
    if (idx == 0 && surf != 0) {
        Resource* s = R(surf);
        const u32 vp[4] = {0, 0, s->width, s->height};
        const float z[2] = {0.0f, 1.0f};
        std::memcpy(d->viewport, vp, 16);
        std::memcpy(d->viewport + 16, z, 8);
        const int32_t sc[4] = {0, 0, static_cast<int32_t>(s->width), static_cast<int32_t>(s->height)};
        std::memcpy(d->scissor, sc, 16);
    }
    return kOk;
}
u32 Dev_GetRenderTarget(u32 t, u32 idx, u32 pp) {
    if (idx >= 4) { put32(pp, 0); return kInvalidCall; }
    Resource* s = D(t)->renderTargets[idx];
    if (s == nullptr) { put32(pp, 0); return kNotFound; }
    addRef(s);
    put32(pp, L(s));
    return kOk;
}
u32 Dev_SetDepthStencilSurface(u32 t, u32 surf) { setRefPtr(D(t)->depthStencil, static_cast<Resource*>(O(surf))); return kOk; }
u32 Dev_GetDepthStencilSurface(u32 t, u32 pp) {
    Resource* s = D(t)->depthStencil;
    if (s == nullptr) { put32(pp, 0); return kNotFound; }
    addRef(s);
    put32(pp, L(s));
    return kOk;
}
u32 Dev_BeginScene(u32) { return kOk; }
u32 Dev_EndScene(u32) { return kOk; }
u32 Dev_Clear(u32 t, u32 count, u32 rects, u32 flags, u32 color, u32 zBits, u32 stencil) {
    Device* d = D(t);
    gpu9::DrawState s;
    fillDrawState(d, s);
    float z = 1.0f;
    std::memcpy(&z, &zBits, 4);
    gpu9::Renderer::shared().clear(s, rects != 0 ? count : 0, rects != 0 ? G<int32_t>(rects) : nullptr, flags, color, z,
                                   stencil);
    return kOk;
}
u32 Dev_SetTransform(u32, u32, u32) { return kOk; }
u32 Dev_GetTransform(u32, u32, u32 m) {
    if (m == 0) return kInvalidCall;
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(G<void>(m), id, sizeof(id));
    return kOk;
}
u32 Dev_MultiplyTransform(u32, u32, u32) { return kOk; }
u32 Dev_SetViewport(u32 t, u32 vp) { if (vp == 0) return kInvalidCall; std::memcpy(D(t)->viewport, G<void>(vp), 24); return kOk; }
u32 Dev_GetViewport(u32 t, u32 vp) { if (vp == 0) return kInvalidCall; std::memcpy(G<void>(vp), D(t)->viewport, 24); return kOk; }
u32 Dev_SetMaterial(u32, u32) { return kOk; }
u32 Dev_GetMaterial(u32, u32 m) { if (m) std::memset(G<void>(m), 0, 68); return kOk; }
u32 Dev_SetLight(u32, u32, u32) { return kOk; }
u32 Dev_GetLight(u32, u32, u32 l) { if (l) std::memset(G<void>(l), 0, 104); return kOk; }
u32 Dev_LightEnable(u32, u32, u32) { return kOk; }
u32 Dev_GetLightEnable(u32, u32, u32 p) { put32(p, 0); return kOk; }
u32 Dev_SetClipPlane(u32 t, u32 i, u32 plane) { if (i < 6 && plane) std::memcpy(D(t)->clipPlanes[i], G<void>(plane), 16); return kOk; }
u32 Dev_GetClipPlane(u32 t, u32 i, u32 plane) { if (i < 6 && plane) std::memcpy(G<void>(plane), D(t)->clipPlanes[i], 16); return kOk; }
u32 Dev_SetRenderState(u32 t, u32 state, u32 value) {
    if (state < 256) {
        Device* d = D(t);
        u32& dst = d->renderState[state];
        if (dst != value) {
            dst = value;
            ++d->rsGen;
        }
    }
    return kOk;
}
u32 Dev_GetRenderState(u32 t, u32 state, u32 p) { put32(p, state < 256 ? D(t)->renderState[state] : 0); return kOk; }
u32 Dev_CreateStateBlock(u32 t, u32, u32 pp) { Obj* o = make<Obj>(kStateBlock, D(t)); put32(pp, L(o)); return o ? kOk : kOutOfVideoMemory; }
u32 Dev_BeginStateBlock(u32) { return kOk; }
u32 Dev_EndStateBlock(u32 t, u32 pp) { return Dev_CreateStateBlock(t, 0, pp); }
u32 Dev_SetClipStatus(u32, u32) { return kOk; }
u32 Dev_GetClipStatus(u32, u32 p) { if (p) std::memset(G<void>(p), 0, 8); return kOk; }
u32 Dev_GetTexture(u32 t, u32 stage, u32 pp) {
    if (stage >= 20) { put32(pp, 0); return kInvalidCall; }
    const u32 tex = D(t)->textures[stage];
    addRefG(tex);
    put32(pp, tex);
    return kOk;
}
// Etapas: 0-15 píxel, D3DVERTEXTEXTURESAMPLER0 (257)..260 -> 16..19.
u32 stageIndex(u32 s) { return s >= 257 && s <= 260 ? 16 + (s - 257) : s; }
u32 Dev_SetTexture(u32 t, u32 stage, u32 tex) {
    stage = stageIndex(stage);
    if (stage >= 20) return kInvalidCall;
    u32& slot = D(t)->textures[stage];
    if (slot != tex) setRef(slot, tex);
    return kOk;
}
u32 Dev_GetTextureStageState(u32 t, u32 stage, u32 type, u32 p) { put32(p, stage < 8 && type < 33 ? D(t)->stageState[stage][type] : 0); return kOk; }
u32 Dev_SetTextureStageState(u32 t, u32 stage, u32 type, u32 v) { if (stage < 8 && type < 33) D(t)->stageState[stage][type] = v; return kOk; }
u32 Dev_GetSamplerState(u32 t, u32 s, u32 type, u32 p) {
    s = stageIndex(s);
    put32(p, s < 20 && type < 14 ? D(t)->samplerState[s][type] : 0);
    return kOk;
}
u32 Dev_SetSamplerState(u32 t, u32 s, u32 type, u32 v) {
    s = stageIndex(s);
    if (s < 20 && type < 14) {
        Device* d = D(t);
        u32& dst = d->samplerState[s][type];
        if (dst != v) {
            dst = v;
            ++d->ssGen;
        }
    }
    return kOk;
}
u32 Dev_ValidateDevice(u32, u32 p) { put32(p, 1); return kOk; }
u32 Dev_SetPaletteEntries(u32, u32, u32) { return kOk; }
u32 Dev_GetPaletteEntries(u32, u32, u32 p) { if (p) std::memset(G<void>(p), 0, 1024); return kOk; }
u32 Dev_SetCurrentTexturePalette(u32 t, u32 n) { D(t)->paletteCurrent = n; return kOk; }
u32 Dev_GetCurrentTexturePalette(u32 t, u32 p) { put32(p, D(t)->paletteCurrent); return kOk; }
u32 Dev_SetScissorRect(u32 t, u32 r) { if (r) std::memcpy(D(t)->scissor, G<void>(r), 16); return kOk; }
u32 Dev_GetScissorRect(u32 t, u32 r) { if (r) std::memcpy(G<void>(r), D(t)->scissor, 16); return kOk; }
u32 Dev_SetSoftwareVertexProcessing(u32 t, u32 b) { D(t)->softwareVp = b; return kOk; }
u32 Dev_GetSoftwareVertexProcessing(u32 t) { return D(t)->softwareVp; }
u32 Dev_SetNPatchMode(u32 t, u32 f) { std::memcpy(&D(t)->npatch, &f, 4); return kOk; }
u32 Dev_GetNPatchMode(u32 t) { u32 v; std::memcpy(&v, &D(t)->npatch, 4); return v; }
u32 primVertices(u32 prim, u32 n) {
    switch (prim) {
        case 1: return n;
        case 2: return n * 2;
        case 3: return n + 1;
        case 4: return n * 3;
        case 5: case 6: return n + 2;
        default: return 0;
    }
}
u32 instanceCount(Device* d) { return (d->streamFreq[0] & 0x40000000u) ? std::max<u32>(1, d->streamFreq[0] & 0x3FFFFFu) : 1; }
u32 Dev_DrawPrimitive(u32 t, u32 type, u32 start, u32 count) {
    Device* d = D(t);
    gpu9::DrawState s;
    fillDrawState(d, s);
    gpu9::Renderer::shared().draw(s, type, count, false, 0, start, instanceCount(d));
    return kOk;
}
u32 Dev_DrawIndexedPrimitive(u32 t, u32 type, u32 baseVertex, u32, u32, u32 startIndex, u32 count) {
    Device* d = D(t);
    if (d->indices == 0) return kInvalidCall;
    gpu9::DrawState s;
    fillDrawState(d, s);
    Resource* ib = R(d->indices);
    s.indices = streamOf(ib, 0, 0, 1);
    s.indexFormat = ib->format;
    gpu9::Renderer::shared().draw(s, type, count, true, static_cast<int32_t>(baseVertex), startIndex, instanceCount(d));
    return kOk;
}
// Datos de usuario: D3D9 deja después la fuente 0 (y los índices en la variante indexada) a NULL.
u32 Dev_DrawPrimitiveUP(u32 t, u32 type, u32 count, u32 data, u32 stride) {
    Device* d = D(t);
    if (data == 0 || stride == 0) return kInvalidCall;
    gpu9::DrawState s;
    fillDrawState(d, s);
    s.streams[0] = {};
    s.streams[0].data = G(data);
    s.streams[0].size = primVertices(type, count) * stride;
    s.streams[0].stride = stride;
    gpu9::Renderer::shared().draw(s, type, count, false, 0, 0, 1);
    setRef(d->streams[0], 0);
    d->streamOffset[0] = d->streamStride[0] = 0;
    return kOk;
}
u32 Dev_DrawIndexedPrimitiveUP(u32 t, u32 type, u32 minIndex, u32 numVerts, u32 count, u32 idx, u32 idxFmt, u32 data,
                               u32 stride) {
    Device* d = D(t);
    if (data == 0 || idx == 0 || stride == 0) return kInvalidCall;
    gpu9::DrawState s;
    fillDrawState(d, s);
    s.streams[0] = {};
    s.streams[0].data = G(data);
    s.streams[0].size = (minIndex + numVerts) * stride;
    s.streams[0].stride = stride;
    s.indices = {};
    s.indices.data = G(idx);
    s.indices.size = primVertices(type, count) * (idxFmt == 102 ? 4 : 2);
    s.indexFormat = idxFmt;
    gpu9::Renderer::shared().draw(s, type, count, true, 0, 0, 1);
    setRef(d->streams[0], 0);
    d->streamOffset[0] = d->streamStride[0] = 0;
    setRef(d->indices, 0);
    return kOk;
}
u32 Dev_ProcessVertices(u32, u32, u32, u32, u32, u32, u32) { return kOk; }

u32 Dev_CreateVertexDeclaration(u32 t, u32 elems, u32 pp) {
    if (elems == 0 || pp == 0) return kInvalidCall;
    Blob* b = make<Blob>(kVertexDecl, D(t));
    if (b == nullptr) return kOutOfVideoMemory;
    for (u32 i = 0; i < 64; ++i) {
        const uint8_t* e = G(elems + 8 * i);
        b->bytes.insert(b->bytes.end(), e, e + 8);
        if (e[0] == 0xFF) break;  // D3DDECL_END (Stream = 0xFF)
    }
    put32(pp, L(b));
    return kOk;
}
u32 Dev_SetVertexDeclaration(u32 t, u32 decl) { setRef(D(t)->vdecl, decl); return kOk; }
u32 Dev_GetVertexDeclaration(u32 t, u32 pp) { addRefG(D(t)->vdecl); put32(pp, D(t)->vdecl); return kOk; }
u32 Dev_SetFVF(u32 t, u32 fvf) { D(t)->fvf = fvf; return kOk; }
u32 Dev_GetFVF(u32 t, u32 p) { put32(p, D(t)->fvf); return kOk; }

u32 makeShader(Device* d, u32 kind, u32 func, u32 pp) {
    if (func == 0 || pp == 0) return kInvalidCall;
    Blob* b = make<Blob>(kind, d);
    if (b == nullptr) return kOutOfVideoMemory;
    // Fichas de 4 bytes hasta D3DSIO_END (0x0000FFFF); los comentarios (0xFFFE) llevan su longitud.
    for (u32 i = 0; i < (1u << 20); ++i) {
        const u32 tok = rd32(func + 4 * i);
        u32 n = 1;
        if (i > 0 && (tok & 0xFFFFu) == 0xFFFEu) n += (tok >> 16) & 0x7FFFu;  // comentario: no se interpreta
        const uint8_t* p = G(func + 4 * i);
        b->bytes.insert(b->bytes.end(), p, p + 4 * n);
        if (i > 0 && tok == 0x0000FFFFu) break;
        i += n - 1;
    }
    // Diagnóstico: setprop debug.hyperplay.d3d9dump 1 -> files/d3d9sh/<vs|ps>_<hash>.bin (corpus del traductor).
    static const bool s_dump = [] { char v[PROP_VALUE_MAX] = {}; return __system_property_get("debug.hyperplay.d3d9dump", v) > 0 && v[0] == '1'; }();
    if (s_dump) {
        uint64_t h = 1469598103934665603ull;
        for (uint8_t c : b->bytes) h = (h ^ c) * 1099511628211ull;
        const char* dir = "/sdcard/Android/data/com.murdoc820.hyperplay/files/d3d9sh";
        mkdir(dir, 0775);
        char path[160];
        std::snprintf(path, sizeof(path), "%s/%s_%016llx.bin", dir, kind == kVertexShader ? "vs" : "ps",
                      static_cast<unsigned long long>(h));
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(b->bytes.data(), 1, b->bytes.size(), f); std::fclose(f); }
    }
    put32(pp, L(b));
    return kOk;
}
u32 Dev_CreateVertexShader(u32 t, u32 func, u32 pp) { return makeShader(D(t), kVertexShader, func, pp); }
u32 Dev_SetVertexShader(u32 t, u32 vs) { u32& s = D(t)->vs; if (s != vs) setRef(s, vs); return kOk; }
u32 Dev_GetVertexShader(u32 t, u32 pp) { addRefG(D(t)->vs); put32(pp, D(t)->vs); return kOk; }
u32 Dev_CreatePixelShader(u32 t, u32 func, u32 pp) { return makeShader(D(t), kPixelShader, func, pp); }
u32 Dev_SetPixelShader(u32 t, u32 ps) { u32& s = D(t)->ps; if (s != ps) setRef(s, ps); return kOk; }
u32 Dev_GetPixelShader(u32 t, u32 pp) { addRefG(D(t)->ps); put32(pp, D(t)->ps); return kOk; }

template <typename T, size_t N>
u32 setConsts(T (&dst)[N], u32 start, u32 data, u32 count, u32 perReg) {
    if (data == 0) return kInvalidCall;
    const size_t first = static_cast<size_t>(start) * perReg, n = static_cast<size_t>(count) * perReg;
    if (first >= N) return kOk;
    std::memcpy(&dst[first], G<void>(data), std::min(n, N - first) * sizeof(T));
    return kOk;
}
template <typename T, size_t N>
u32 getConsts(T (&src)[N], u32 start, u32 data, u32 count, u32 perReg) {
    if (data == 0) return kInvalidCall;
    const size_t first = static_cast<size_t>(start) * perReg, n = static_cast<size_t>(count) * perReg;
    if (first >= N) return kOk;
    std::memcpy(G<void>(data), &src[first], std::min(n, N - first) * sizeof(T));
    return kOk;
}
u32 Dev_SetVSConstF(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->vsConstGen; return setConsts(dev->vsF, s, d, c, 4); }
u32 Dev_GetVSConstF(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->vsF, s, d, c, 4); }
u32 Dev_SetVSConstI(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->vsConstGen; return setConsts(dev->vsI, s, d, c, 4); }
u32 Dev_GetVSConstI(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->vsI, s, d, c, 4); }
u32 Dev_SetVSConstB(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->vsConstGen; return setConsts(dev->vsB, s, d, c, 1); }
u32 Dev_GetVSConstB(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->vsB, s, d, c, 1); }
u32 Dev_SetPSConstF(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->psConstGen; return setConsts(dev->psF, s, d, c, 4); }
u32 Dev_GetPSConstF(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->psF, s, d, c, 4); }
u32 Dev_SetPSConstI(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->psConstGen; return setConsts(dev->psI, s, d, c, 4); }
u32 Dev_GetPSConstI(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->psI, s, d, c, 4); }
u32 Dev_SetPSConstB(u32 t, u32 s, u32 d, u32 c) { Device* dev = D(t); ++dev->psConstGen; return setConsts(dev->psB, s, d, c, 1); }
u32 Dev_GetPSConstB(u32 t, u32 s, u32 d, u32 c) { return getConsts(D(t)->psB, s, d, c, 1); }

u32 Dev_SetStreamSource(u32 t, u32 i, u32 vb, u32 off, u32 stride) {
    if (i >= 16) return kInvalidCall;
    Device* d = D(t);
    if (d->streams[i] != vb) setRef(d->streams[i], vb);
    d->streamOffset[i] = off;
    d->streamStride[i] = stride;
    return kOk;
}
u32 Dev_GetStreamSource(u32 t, u32 i, u32 pp, u32 pOff, u32 pStride) {
    if (i >= 16) return kInvalidCall;
    Device* d = D(t);
    addRefG(d->streams[i]);
    put32(pp, d->streams[i]);
    put32(pOff, d->streamOffset[i]);
    put32(pStride, d->streamStride[i]);
    return kOk;
}
u32 Dev_SetStreamSourceFreq(u32 t, u32 i, u32 v) { if (i < 16) D(t)->streamFreq[i] = v; return kOk; }
u32 Dev_GetStreamSourceFreq(u32 t, u32 i, u32 p) { put32(p, i < 16 ? D(t)->streamFreq[i] : 1); return kOk; }
u32 Dev_SetIndices(u32 t, u32 ib) { u32& s = D(t)->indices; if (s != ib) setRef(s, ib); return kOk; }
u32 Dev_GetIndices(u32 t, u32 pp) { addRefG(D(t)->indices); put32(pp, D(t)->indices); return kOk; }
u32 Dev_DrawRectPatch(u32, u32, u32, u32) { return kNotImpl; }
u32 Dev_DrawTriPatch(u32, u32, u32, u32) { return kNotImpl; }
u32 Dev_DeletePatch(u32, u32) { return kOk; }
u32 Dev_CreateQuery(u32 t, u32 type, u32 pp) {
    const bool supported = queryDataSize(type) != 0;
    if (pp == 0) return supported ? kOk : 0x8876086Au;  // comprobación de soporte
    if (!supported) { put32(pp, 0); return 0x8876086Au; }
    Query* q = make<Query>(kQuery, D(t));
    if (q == nullptr) return kOutOfVideoMemory;
    q->type = type;
    put32(pp, L(q));
    return kOk;
}

// --- IDirect3DSwapChain9 ---
u32 SC_Present(u32 t, u32, u32, u32, u32, u32) { return present(O(t)->dev); }
u32 SC_GetFrontBufferData(u32, u32) { return kOk; }
u32 SC_GetBackBuffer(u32 t, u32 i, u32, u32 pp) { return Dev_GetBackBuffer(L(O(t)->dev), 0, i, 0, pp); }
u32 SC_GetRasterStatus(u32, u32 p) { if (p) { put32(p, 0); put32(p + 4, 0); } return kOk; }
u32 SC_GetDisplayMode(u32 t, u32 mode) { return Dev_GetDisplayMode(L(O(t)->dev), 0, mode); }
u32 SC_GetPresentParameters(u32 t, u32 pp) { if (pp) std::memcpy(G<void>(pp), O(t)->dev->pp, 56); return kOk; }

// ===========================================================================================
u32 makeVtable(const std::vector<void*>& m, const char* name) {
    auto* vt = static_cast<u32*>(lowSmall(4 * m.size()));
    if (vt == nullptr) return 0;
    for (size_t i = 0; i < m.size(); ++i)
        vt[i] = static_cast<u32>(w32alias::lowFor(m[i], std::string("d3d9.dll!") + name + "#" + std::to_string(i)));
    return L(vt);
}

#define F(fn) w32arity::hpFn(&fn)

void buildVtables() {
    static std::once_flag once;
    std::call_once(once, [] {
        g_vt[kDevice] = makeVtable({
            F(Dev_QI), F(AddRef), F(Release), F(Dev_TestCooperativeLevel), F(Dev_GetAvailableTextureMem),
            F(Dev_EvictManagedResources), F(Dev_GetDirect3D), F(Dev_GetDeviceCaps), F(Dev_GetDisplayMode),
            F(Dev_GetCreationParameters), F(Dev_SetCursorProperties), F(Dev_SetCursorPosition), F(Dev_ShowCursor),
            F(Dev_CreateAdditionalSwapChain), F(Dev_GetSwapChain), F(Dev_GetNumberOfSwapChains), F(Dev_Reset),
            F(Dev_Present), F(Dev_GetBackBuffer), F(Dev_GetRasterStatus), F(Dev_SetDialogBoxMode), F(Dev_SetGammaRamp),
            F(Dev_GetGammaRamp), F(Dev_CreateTexture), F(Dev_CreateVolumeTexture), F(Dev_CreateCubeTexture),
            F(Dev_CreateVertexBuffer), F(Dev_CreateIndexBuffer), F(Dev_CreateRenderTarget),
            F(Dev_CreateDepthStencilSurface), F(Dev_UpdateSurface), F(Dev_UpdateTexture), F(Dev_GetRenderTargetData),
            F(Dev_GetFrontBufferData), F(Dev_StretchRect), F(Dev_ColorFill), F(Dev_CreateOffscreenPlainSurface),
            F(Dev_SetRenderTarget), F(Dev_GetRenderTarget), F(Dev_SetDepthStencilSurface),
            F(Dev_GetDepthStencilSurface), F(Dev_BeginScene), F(Dev_EndScene), F(Dev_Clear), F(Dev_SetTransform),
            F(Dev_GetTransform), F(Dev_MultiplyTransform), F(Dev_SetViewport), F(Dev_GetViewport), F(Dev_SetMaterial),
            F(Dev_GetMaterial), F(Dev_SetLight), F(Dev_GetLight), F(Dev_LightEnable), F(Dev_GetLightEnable),
            F(Dev_SetClipPlane), F(Dev_GetClipPlane), F(Dev_SetRenderState), F(Dev_GetRenderState),
            F(Dev_CreateStateBlock), F(Dev_BeginStateBlock), F(Dev_EndStateBlock), F(Dev_SetClipStatus),
            F(Dev_GetClipStatus), F(Dev_GetTexture), F(Dev_SetTexture), F(Dev_GetTextureStageState),
            F(Dev_SetTextureStageState), F(Dev_GetSamplerState), F(Dev_SetSamplerState), F(Dev_ValidateDevice),
            F(Dev_SetPaletteEntries), F(Dev_GetPaletteEntries), F(Dev_SetCurrentTexturePalette),
            F(Dev_GetCurrentTexturePalette), F(Dev_SetScissorRect), F(Dev_GetScissorRect),
            F(Dev_SetSoftwareVertexProcessing), F(Dev_GetSoftwareVertexProcessing), F(Dev_SetNPatchMode),
            F(Dev_GetNPatchMode), F(Dev_DrawPrimitive), F(Dev_DrawIndexedPrimitive), F(Dev_DrawPrimitiveUP),
            F(Dev_DrawIndexedPrimitiveUP), F(Dev_ProcessVertices), F(Dev_CreateVertexDeclaration),
            F(Dev_SetVertexDeclaration), F(Dev_GetVertexDeclaration), F(Dev_SetFVF), F(Dev_GetFVF),
            F(Dev_CreateVertexShader), F(Dev_SetVertexShader), F(Dev_GetVertexShader), F(Dev_SetVSConstF),
            F(Dev_GetVSConstF), F(Dev_SetVSConstI), F(Dev_GetVSConstI), F(Dev_SetVSConstB), F(Dev_GetVSConstB),
            F(Dev_SetStreamSource), F(Dev_GetStreamSource), F(Dev_SetStreamSourceFreq), F(Dev_GetStreamSourceFreq),
            F(Dev_SetIndices), F(Dev_GetIndices), F(Dev_CreatePixelShader), F(Dev_SetPixelShader),
            F(Dev_GetPixelShader), F(Dev_SetPSConstF), F(Dev_GetPSConstF), F(Dev_SetPSConstI), F(Dev_GetPSConstI),
            F(Dev_SetPSConstB), F(Dev_GetPSConstB), F(Dev_DrawRectPatch), F(Dev_DrawTriPatch), F(Dev_DeletePatch),
            F(Dev_CreateQuery),
        }, "IDirect3DDevice9");

        std::vector<void*> m;
        m = {F(QI), F(AddRef), F(Release), F(SC_Present), F(SC_GetFrontBufferData), F(SC_GetBackBuffer),
             F(SC_GetRasterStatus), F(SC_GetDisplayMode), F(GetDevice), F(SC_GetPresentParameters)};
        g_vt[kSwapChain] = makeVtable(m, "IDirect3DSwapChain9");

        m.clear(); resourceBase(m);
        for (void* f : {F(Surf_GetContainer), F(Surf_GetDesc), F(Surf_LockRect), F(Surf_UnlockRect), F(Surf_GetDC), F(Surf_ReleaseDC)}) m.push_back(f);
        g_vt[kSurface] = makeVtable(m, "IDirect3DSurface9");

        m.clear(); baseTexture(m);
        for (void* f : {F(Tex_GetLevelDesc), F(Tex_GetSurfaceLevel), F(Tex_LockRect), F(Tex_UnlockRect), F(Tex_AddDirtyRect)}) m.push_back(f);
        g_vt[kTexture] = makeVtable(m, "IDirect3DTexture9");

        m.clear(); baseTexture(m);
        for (void* f : {F(Cube_GetLevelDesc), F(Cube_GetCubeMapSurface), F(Cube_LockRect), F(Cube_UnlockRect), F(Cube_AddDirtyRect)}) m.push_back(f);
        g_vt[kCubeTexture] = makeVtable(m, "IDirect3DCubeTexture9");

        m.clear(); baseTexture(m);
        for (void* f : {F(VolTex_GetLevelDesc), F(VolTex_GetVolumeLevel), F(VolTex_LockBox), F(VolTex_UnlockBox), F(VolTex_AddDirtyBox)}) m.push_back(f);
        g_vt[kVolumeTexture] = makeVtable(m, "IDirect3DVolumeTexture9");

        // IDirect3DVolume9 no hereda de IDirect3DResource9.
        m = {F(QI), F(AddRef), F(Release), F(GetDevice), F(SetPrivateData), F(GetPrivateData), F(FreePrivateData),
             F(Vol_GetContainer), F(Vol_GetDesc), F(Vol_LockBox), F(Vol_UnlockBox)};
        g_vt[kVolume] = makeVtable(m, "IDirect3DVolume9");

        m.clear(); resourceBase(m);
        for (void* f : {F(Buf_Lock), F(Buf_Unlock), F(VB_GetDesc)}) m.push_back(f);
        g_vt[kVertexBuffer] = makeVtable(m, "IDirect3DVertexBuffer9");

        m.clear(); resourceBase(m);
        for (void* f : {F(Buf_Lock), F(Buf_Unlock), F(IB_GetDesc)}) m.push_back(f);
        g_vt[kIndexBuffer] = makeVtable(m, "IDirect3DIndexBuffer9");

        m = {F(QI), F(AddRef), F(Release), F(GetDevice), F(Decl_GetDeclaration)};
        g_vt[kVertexDecl] = makeVtable(m, "IDirect3DVertexDeclaration9");
        m = {F(QI), F(AddRef), F(Release), F(GetDevice), F(Shader_GetFunction)};
        g_vt[kVertexShader] = makeVtable(m, "IDirect3DVertexShader9");
        g_vt[kPixelShader] = makeVtable(m, "IDirect3DPixelShader9");
        m = {F(QI), F(AddRef), F(Release), F(GetDevice), F(Query_GetType), F(Query_GetDataSize), F(Query_Issue),
             F(Query_GetData)};
        g_vt[kQuery] = makeVtable(m, "IDirect3DQuery9");
        m = {F(QI), F(AddRef), F(Release), F(GetDevice), F(SB_Capture), F(SB_Apply)};
        g_vt[kStateBlock] = makeVtable(m, "IDirect3DStateBlock9");
    });
}
#undef F

}  // namespace

uint32_t createDevice(const CreateParams& p) {
    buildVtables();
    Device* d = make<Device>(kDevice, nullptr);
    if (d == nullptr) return 0;
    d->dev = d;
    d->direct3d = p.direct3d;
    d->adapter = p.adapter;
    d->deviceType = p.deviceType;
    d->focusWindow = p.focusWindow;
    d->behaviorFlags = p.behaviorFlags;
    std::memcpy(d->pp, G<void>(p.presentParams), sizeof(d->pp));
    // Estados por defecto de D3D9 relevantes (Z activo, escritura de color completa, culling CCW).
    d->renderState[7] = 1;     // ZENABLE
    d->renderState[14] = 1;    // ZWRITEENABLE
    d->renderState[22] = 3;    // CULLMODE = CCW
    d->renderState[23] = 4;    // ZFUNC = LESSEQUAL
    d->renderState[168] = 0xF; // COLORWRITEENABLE
    d->renderState[8] = 3;     // FILLMODE = SOLID
    d->renderState[19] = 2;    // SRCBLEND = ONE
    d->renderState[20] = 1;    // DESTBLEND = ZERO
    d->renderState[25] = 8;    // ALPHAFUNC = ALWAYS
    d->renderState[53] = d->renderState[54] = d->renderState[55] = 1;  // STENCIL* = KEEP
    d->renderState[56] = 8;    // STENCILFUNC = ALWAYS
    d->renderState[58] = d->renderState[59] = 0xFFFFFFFFu;
    d->renderState[186] = d->renderState[187] = d->renderState[188] = 1;
    d->renderState[189] = 8;   // CCW_STENCILFUNC
    d->renderState[190] = d->renderState[191] = d->renderState[192] = 0xF;
    d->renderState[171] = 1;   // BLENDOP = ADD
    d->renderState[193] = 0xFFFFFFFFu;  // BLENDFACTOR
    d->renderState[207] = 2; d->renderState[208] = 1; d->renderState[209] = 1;
    for (auto& s : d->samplerState) { s[1] = 1; s[2] = 1; s[3] = 1; s[5] = 1; s[6] = 1; s[7] = 0; s[10] = 1; }  // WRAP, POINT
    for (u32 i = 0; i < 16; ++i) d->streamFreq[i] = 1;
    createBackBuffers(d);
    d->renderState[7] = d->autoDepth != nullptr ? 1 : 0;  // ZENABLE = D3DZB_TRUE solo con profundidad automática
    std::memcpy(G<void>(p.presentParams), d->pp, sizeof(d->pp));  // tamaños por defecto rellenados
    Obj* sc = make<Obj>(kSwapChain, d);
    d->swapChain = sc;
    LOGI("IDirect3DDevice9 de 32 bits -> 0x%x (flags 0x%x, ventana 0x%x)", L(d), p.behaviorFlags, p.focusWindow);
    {   // diagnóstico (setprop debug.hyperplay.tracemaps N): N mapeos bajos tras crear el dispositivo
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.tracemaps", v) > 0) lowmem::g_traceMaps.store(std::atoi(v));
    }
    return L(d);
}

}  // namespace d3d9dev32
}  // namespace hyperplay
