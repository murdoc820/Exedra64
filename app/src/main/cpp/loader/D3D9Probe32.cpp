// IDirect3D9 de solo consulta para juegos de 32 bits. Ver D3D9Probe32.h.
//
// Los métodos COM en x86 son __stdcall con `this` como primer argumento de pila: cada función
// nativa tiene la aridad exacta (this incluido) para que el puente saque de la pila lo justo.
// El orden de la vtable es el de d3d9.h (IDirect3D9, 17 métodos).

#include "D3D9Probe32.h"
#include "D3D9Device32.h"
#include "cpu/OryonEmitterBackend.h"

#include <android/log.h>
#include <sys/mman.h>
#include <sys/system_properties.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "LowMem.h"
#include "Win32Arity.h"

#define LOG_TAG "HyperPlay.D3D9Probe32"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace w32alias {
uintptr_t lowFor(void* host, const std::string& name);  // NativeDllInterceptor.cpp
}

namespace d3d9probe32 {
namespace {

using u32 = uint32_t;

constexpr u32 kOk = 0;
constexpr u32 kNoInterface = 0x80004002u;
constexpr u32 kInvalidCall = 0x8876086Cu;   // D3DERR_INVALIDCALL
constexpr u32 kNotAvailable = 0x8876086Au;  // D3DERR_NOTAVAILABLE
constexpr u32 kFmtX8R8G8B8 = 22;

Display g_display;
std::atomic<u32> g_refs{1};

template <typename T = uint8_t>
T* P(u32 a) { return reinterpret_cast<T*>(static_cast<uintptr_t>(a)); }

struct Mode { u32 w, h; };
std::vector<Mode> modes() {
    std::vector<Mode> m;
    m.push_back({g_display.width, g_display.height});
    if (g_display.width > 1920 && g_display.height > 860) {
        const u32 h75 = ((g_display.height * 3 / 4) + 1) & ~1u;
        const u32 w75 = ((g_display.width * 3 / 4) + 1) & ~1u;
        const u32 h60 = ((g_display.height * 3 / 5) + 1) & ~1u;
        const u32 w60 = ((g_display.width * 3 / 5) + 1) & ~1u;
        m.push_back({w75, h75});
        m.push_back({w60, h60});
    }
    for (const Mode x : {Mode{1920, 1080}, Mode{1600, 900}, Mode{1280, 720}}) {
        if (x.w != g_display.width || x.h != g_display.height) m.push_back(x);
    }
    return m;
}
void putMode(u32 dst, const Mode& m) {
    const u32 v[4] = {m.w, m.h, 60, kFmtX8R8G8B8};  // D3DDISPLAYMODE
    std::memcpy(P(dst), v, sizeof(v));
}

u32 QueryInterface(u32, u32, u32 ppv) {
    if (ppv != 0) std::memset(P(ppv), 0, 4);
    return kNoInterface;
}
u32 AddRef(u32) { return ++g_refs; }
u32 Release(u32) { const u32 r = g_refs.load(); if (r > 1) --g_refs; return r > 1 ? r - 1 : 1; }
u32 RegisterSoftwareDevice(u32, u32) { return kInvalidCall; }
u32 GetAdapterCount(u32) { return 1; }

// D3DADAPTER_IDENTIFIER9 (1100 bytes). Misma GPU que el adaptador DXGI (Qualcomm Adreno 830).
u32 GetAdapterIdentifier(u32, u32 adapter, u32, u32 out) {
    if (adapter != 0 || out == 0) return kInvalidCall;
    uint8_t* b = P(out);
    std::memset(b, 0, 1100);
    std::strcpy(reinterpret_cast<char*>(b), "qcvk.dll");
    std::strcpy(reinterpret_cast<char*>(b + 512), "Qualcomm Adreno 830");
    std::strcpy(reinterpret_cast<char*>(b + 1024), "\\\\.\\DISPLAY1");
    const uint64_t driverVersion = (uint64_t{31} << 48) | (uint64_t{0} << 32) | (uint64_t{101} << 16) | 2111;
    std::memcpy(b + 1056, &driverVersion, 8);
    const u32 ids[4] = {0x5143u, 0x0830u, 0, 0};  // VendorId, DeviceId, SubSysId, Revision
    std::memcpy(b + 1064, ids, sizeof(ids));
    const uint8_t guid[16] = {0x48, 0x50, 0x41, 0x44, 0x52, 0x45, 0x4e, 0x4f, 0x38, 0x33, 0x30, 0x00, 0x51, 0x43, 0x4f, 0x4d};
    std::memcpy(b + 1080, guid, 16);
    const u32 whql = 1;
    std::memcpy(b + 1096, &whql, 4);
    return kOk;
}
u32 GetAdapterModeCount(u32, u32 adapter, u32 format) {
    return adapter == 0 && format == kFmtX8R8G8B8 ? static_cast<u32>(modes().size()) : 0;
}
u32 EnumAdapterModes(u32, u32 adapter, u32 format, u32 index, u32 out) {
    const auto m = modes();
    if (adapter != 0 || format != kFmtX8R8G8B8 || index >= m.size() || out == 0) return kInvalidCall;
    putMode(out, m[index]);
    return kOk;
}
u32 GetAdapterDisplayMode(u32, u32 adapter, u32 out) {
    if (adapter != 0 || out == 0) return kInvalidCall;
    putMode(out, {g_display.width, g_display.height});
    return kOk;
}
u32 CheckDeviceType(u32, u32 adapter, u32, u32, u32, u32) { return adapter == 0 ? kOk : kInvalidCall; }
u32 CheckDeviceFormat(u32, u32 adapter, u32, u32, u32 usage, u32, u32 fmt) {
    if (adapter != 0) return kInvalidCall;
    // Formatos YUV empaquetados (UYVY, YUY2, NV12) no soportados en Vulkan D3D9: fuerza fallback a L8/RGBA.
    if (fmt == 0x59565955u || fmt == 0x32595559u || fmt == 0x3231564Eu) return kNotAvailable;
    // D3DFMT_L8 (50) no es RenderTarget válido en D3D9.
    if (fmt == 50u && (usage & 1u) != 0) return kNotAvailable;
    return kOk;
}
u32 CheckDeviceMultiSampleType(u32, u32 adapter, u32, u32, u32, u32 type, u32 quality) {
    if (adapter != 0) return kInvalidCall;
    if (quality != 0) { const u32 q = 1; std::memcpy(P(quality), &q, 4); }
    return type <= 1 || type == 2 || type == 4 || type == 8 ? kOk : kNotAvailable;
}
u32 CheckDepthStencilMatch(u32, u32 adapter, u32, u32, u32, u32) { return adapter == 0 ? kOk : kInvalidCall; }
u32 CheckDeviceFormatConversion(u32, u32 adapter, u32, u32, u32) { return adapter == 0 ? kOk : kInvalidCall; }

// D3DCAPS9 (304 bytes) de una GPU de clase Shader Model 3.
u32 GetDeviceCaps(u32, u32 adapter, u32 deviceType, u32 out) {
    if (adapter != 0 || out == 0) return kInvalidCall;
    uint8_t* b = P(out);
    std::memset(b, 0, 304);
    auto put = [&](size_t off, u32 v) { std::memcpy(b + off, &v, 4); };
    auto putf = [&](size_t off, float v) { std::memcpy(b + off, &v, 4); };
    put(0, deviceType);
    for (size_t off = 8; off <= 84; off += 4) put(off, 0xFFFFFFFFu);  // Caps .. LineCaps
    put(60, 0x0001EC85u);  // TextureCaps: sin restricciones POW2/SQUAREONLY
    put(88, 16384); put(92, 16384); put(96, 2048); put(100, 8192); put(104, 8192); put(108, 16);
    putf(112, 1e10f);
    putf(116, -32768.f); putf(120, -32768.f); putf(124, 32768.f); putf(128, 32768.f);
    put(136, 0xFFu); put(140, 0x00180008u); put(144, 0x03FEFFFFu);
    put(148, 8); put(152, 8); put(156, 0x17Bu);
    put(160, 8); put(164, 8); put(168, 4); put(172, 255);
    putf(176, 8192.f);
    put(180, 0x00FFFFFFu); put(184, 0x00FFFFFFu); put(188, 16); put(192, 508);
    put(196, 0xFFFE0300u); put(200, 256); put(204, 0xFFFF0300u);
    putf(208, 65504.f);
    put(212, 0x71u);
    put(232, 1);            // NumberOfAdaptersInGroup
    put(236, 0x3FFu);       // DeclTypes
    put(240, 4);            // NumSimultaneousRTs
    put(244, 0x03030000u);  // StretchRectFilterCaps
    const u32 vs20[4] = {1, 24, 32, 4};
    std::memcpy(b + 248, vs20, sizeof(vs20));
    const u32 ps20[5] = {0x1F, 24, 32, 4, 512};
    std::memcpy(b + 264, ps20, sizeof(ps20));
    put(284, 0x03030000u);
    put(288, 0xFFFF); put(292, 0xFFFF); put(296, 32768); put(300, 32768);
    return kOk;
}
u32 GetAdapterMonitor(u32, u32 adapter) { return adapter == 0 ? g_display.monitor : 0; }
u32 CreateDevice(u32 self, u32 adapter, u32 type, u32 hwnd, u32 flags, u32 pp, u32 ppDevice) {
    if (ppDevice == 0 || pp == 0) return kInvalidCall;
    d3d9dev32::CreateParams cp;
    cp.direct3d = self;
    cp.adapter = adapter;
    cp.deviceType = type;
    cp.focusWindow = hwnd;
    cp.behaviorFlags = flags;
    cp.presentParams = pp;
    cp.screenWidth = g_display.width;
    cp.screenHeight = g_display.height;
    const u32 dev = d3d9dev32::createDevice(cp);
    std::memcpy(P(ppDevice), &dev, 4);
    return dev != 0 ? kOk : kNotAvailable;
}

u32 buildObject() {
    static std::once_flag once;
    static u32 obj = 0;
    std::call_once(once, [] {
        const std::vector<void*> methods = {
            w32arity::hpFn(&QueryInterface),             // 0
            w32arity::hpFn(&AddRef),                     // 1
            w32arity::hpFn(&Release),                    // 2
            w32arity::hpFn(&RegisterSoftwareDevice),     // 3
            w32arity::hpFn(&GetAdapterCount),            // 4
            w32arity::hpFn(&GetAdapterIdentifier),       // 5
            w32arity::hpFn(&GetAdapterModeCount),        // 6
            w32arity::hpFn(&EnumAdapterModes),           // 7
            w32arity::hpFn(&GetAdapterDisplayMode),      // 8
            w32arity::hpFn(&CheckDeviceType),            // 9
            w32arity::hpFn(&CheckDeviceFormat),          // 10
            w32arity::hpFn(&CheckDeviceMultiSampleType), // 11
            w32arity::hpFn(&CheckDepthStencilMatch),     // 12
            w32arity::hpFn(&CheckDeviceFormatConversion),// 13
            w32arity::hpFn(&GetDeviceCaps),              // 14
            w32arity::hpFn(&GetAdapterMonitor),          // 15
            w32arity::hpFn(&CreateDevice),               // 16
        };
        void* mem = lowmem::mmapBelow4G(0x1000);
        if (mem == MAP_FAILED) return;
        auto* b = static_cast<uint8_t*>(mem);
        const auto base = static_cast<u32>(reinterpret_cast<uintptr_t>(b));
        const u32 vt = base + 8;
        std::memcpy(b, &vt, 4);
        for (size_t i = 0; i < methods.size(); ++i) {
            const auto a = static_cast<u32>(
                w32alias::lowFor(methods[i], "d3d9.dll!IDirect3D9#" + std::to_string(i)));
            std::memcpy(b + 8 + 4 * i, &a, 4);
        }
        obj = base;
        LOGI("IDirect3D9 de consulta en 0x%x (%ux%u)", base, g_display.width, g_display.height);
    });
    return obj;
}

u32 Direct3DCreate9(u32 sdkVersion) {
    LOGI("Direct3DCreate9(SDK %u): IDirect3D9 de solo consulta", sdkVersion);
    return buildObject();
}

}  // namespace

void fillCaps(uint32_t deviceType, uint32_t out) { (void)GetDeviceCaps(0, 0, deviceType, out); }
Display display() { return g_display; }

void* resolve(const std::string& name, const Display& display) {
    g_display = display;
    if (name == "Direct3DCreate9") return w32arity::hpFn(&Direct3DCreate9);
    return nullptr;
}

}  // namespace d3d9probe32
}  // namespace hyperplay
