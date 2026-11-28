#include "D3D9GpuRenderer.h"

// Las estructuras Vulkan se inicializan con {sType} y el resto a cero (patrón habitual).
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#include <android/log.h>
#include <sys/system_properties.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "../loader/D3D11Sampler.h"
#include "D3D9ShaderToSpirv.h"
#include "DirectVulkanBridge.h"
#include "HyperFrameGen.h"
#include "PresentOptions.h"
#include "PresentSpv.h"
#include "PresentSgsrSpv.h"
#include "loader/FramePacing.h"
#include "loader/Win32DisplayInfo.h"

#define LOG_TAG "HyperPlay.GpuD3D9"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay::gpu9 {
namespace {

#define HP_VK9_FNS(X)                                                                                         \
    X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory)      \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkCreateShaderModule)                      \
    X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout)                          \
    X(vkCreateGraphicsPipelines) X(vkCreateRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer)          \
    X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)   \
    X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer) X(vkBeginCommandBuffer)         \
    X(vkEndCommandBuffer) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkCmdPipelineBarrier)         \
    X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdBlitImage) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer)          \
    X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetBlendConstants)          \
    X(vkCmdSetDepthBias) X(vkCmdSetStencilReference) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage)    \
    X(vkCmdClearAttachments) X(vkCmdPushConstants)

struct Fns {
#define HP_DECL(name) PFN_##name name = nullptr;
    HP_VK9_FNS(HP_DECL)
#undef HP_DECL
    PFN_vkGetPhysicalDeviceMemoryProperties getMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties getProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties getFormatProperties = nullptr;
};

constexpr uint32_t fourcc(char a, char b, char c, char d) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b) << 8 | static_cast<uint32_t>(c) << 16 |
           static_cast<uint32_t>(d) << 24;
}

// Conversión en CPU antes de subir (formatos sin equivalente Vulkan directo).
enum Convert : uint8_t { kNone, kRgb24ToBgra, kArgb4ToRgba8, kXrgb4ToRgba8, kDecompress };

struct Fmt9 {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkFormat srgb = VK_FORMAT_UNDEFINED;
    VkComponentMapping swz{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY};
    uint32_t blockW = 1, blockBytes = 4;  // tal como los guarda la CPU (D3D9Device32 fmtInfo)
    bool depth = false, stencil = false;
    Convert convert = kNone;
    uint32_t dxgiBc = 0;  // para descomprimir con d3d11::decompressToRgba8
};

constexpr VkComponentSwizzle R = VK_COMPONENT_SWIZZLE_R, Gc = VK_COMPONENT_SWIZZLE_G,
                             ONE = VK_COMPONENT_SWIZZLE_ONE,
                             ZERO = VK_COMPONENT_SWIZZLE_ZERO, ID = VK_COMPONENT_SWIZZLE_IDENTITY;

Fmt9 fmt9(uint32_t f) {
    Fmt9 r;
    switch (f) {
        case 21: r.format = VK_FORMAT_B8G8R8A8_UNORM; r.srgb = VK_FORMAT_B8G8R8A8_SRGB; break;  // A8R8G8B8
        case 22: r.format = VK_FORMAT_B8G8R8A8_UNORM; r.srgb = VK_FORMAT_B8G8R8A8_SRGB; r.swz = {ID, ID, ID, ONE}; break;
        case 32: r.format = VK_FORMAT_R8G8B8A8_UNORM; r.srgb = VK_FORMAT_R8G8B8A8_SRGB; break;  // A8B8G8R8
        case 33: r.format = VK_FORMAT_R8G8B8A8_UNORM; r.srgb = VK_FORMAT_R8G8B8A8_SRGB; r.swz = {ID, ID, ID, ONE}; break;
        case 20: r.format = VK_FORMAT_B8G8R8A8_UNORM; r.blockBytes = 3; r.convert = kRgb24ToBgra; break;
        case 23: r.format = VK_FORMAT_R5G6B5_UNORM_PACK16; r.blockBytes = 2; break;
        case 24: r.format = VK_FORMAT_A1R5G5B5_UNORM_PACK16; r.blockBytes = 2; r.swz = {ID, ID, ID, ONE}; break;
        case 25: r.format = VK_FORMAT_A1R5G5B5_UNORM_PACK16; r.blockBytes = 2; break;
        case 26: r.format = VK_FORMAT_R8G8B8A8_UNORM; r.blockBytes = 2; r.convert = kArgb4ToRgba8; break;
        case 30: r.format = VK_FORMAT_R8G8B8A8_UNORM; r.blockBytes = 2; r.convert = kXrgb4ToRgba8; break;
        case 28: r.format = VK_FORMAT_R8_UNORM; r.blockBytes = 1; r.swz = {ZERO, ZERO, ZERO, R}; break;  // A8
        case 50: r.format = VK_FORMAT_R8_UNORM; r.blockBytes = 1; r.swz = {R, R, R, ONE}; break;          // L8
        case 51: r.format = VK_FORMAT_R8G8_UNORM; r.blockBytes = 2; r.swz = {R, R, R, Gc}; break;         // A8L8
        case 81: r.format = VK_FORMAT_R16_UNORM; r.blockBytes = 2; r.swz = {R, R, R, ONE}; break;         // L16
        case 31: r.format = VK_FORMAT_A2R10G10B10_UNORM_PACK32; break;
        case 35: r.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32; break;
        case 34: r.format = VK_FORMAT_R16G16_UNORM; r.swz = {ID, ID, ONE, ONE}; break;        // G16R16
        case 36: r.format = VK_FORMAT_R16G16B16A16_UNORM; r.blockBytes = 8; break;           // A16B16G16R16
        case 60: r.format = VK_FORMAT_R8G8_SNORM; r.blockBytes = 2; r.swz = {ID, ID, ONE, ONE}; break;  // V8U8
        case 63: r.format = VK_FORMAT_R8G8B8A8_SNORM; break;                                  // Q8W8V8U8
        case 64: r.format = VK_FORMAT_R16G16_SNORM; r.swz = {ID, ID, ONE, ONE}; break;        // V16U16
        case 111: r.format = VK_FORMAT_R16_SFLOAT; r.blockBytes = 2; r.swz = {ID, ONE, ONE, ONE}; break;
        case 112: r.format = VK_FORMAT_R16G16_SFLOAT; r.swz = {ID, ID, ONE, ONE}; break;
        case 113: r.format = VK_FORMAT_R16G16B16A16_SFLOAT; r.blockBytes = 8; break;
        case 114: r.format = VK_FORMAT_R32_SFLOAT; r.swz = {ID, ONE, ONE, ONE}; break;
        case 115: r.format = VK_FORMAT_R32G32_SFLOAT; r.blockBytes = 8; r.swz = {ID, ID, ONE, ONE}; break;
        case 116: r.format = VK_FORMAT_R32G32B32A32_SFLOAT; r.blockBytes = 16; break;
        case fourcc('D', 'X', 'T', '1'):
            r.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; r.srgb = VK_FORMAT_BC1_RGBA_SRGB_BLOCK; r.blockW = 4; r.blockBytes = 8; r.dxgiBc = 71; break;
        case fourcc('D', 'X', 'T', '2'): case fourcc('D', 'X', 'T', '3'):
            r.format = VK_FORMAT_BC2_UNORM_BLOCK; r.srgb = VK_FORMAT_BC2_SRGB_BLOCK; r.blockW = 4; r.blockBytes = 16; r.dxgiBc = 74; break;
        case fourcc('D', 'X', 'T', '4'): case fourcc('D', 'X', 'T', '5'):
            r.format = VK_FORMAT_BC3_UNORM_BLOCK; r.srgb = VK_FORMAT_BC3_SRGB_BLOCK; r.blockW = 4; r.blockBytes = 16; r.dxgiBc = 77; break;
        case fourcc('A', 'T', 'I', '1'):
            r.format = VK_FORMAT_BC4_UNORM_BLOCK; r.blockW = 4; r.blockBytes = 8; r.dxgiBc = 80; r.swz = {R, R, R, ONE}; break;
        case fourcc('A', 'T', 'I', '2'):
            r.format = VK_FORMAT_BC5_UNORM_BLOCK; r.blockW = 4; r.blockBytes = 16; r.dxgiBc = 83; r.swz = {ID, ID, ONE, ONE}; break;
        // Profundidad (las de lectura INTZ / DF24 / DF16 también se muestrean).
        case 75: case 77: case 79: case 71: case fourcc('I', 'N', 'T', 'Z'): case fourcc('D', 'F', '2', '4'):
            r.format = VK_FORMAT_D24_UNORM_S8_UINT; r.depth = true; r.stencil = true; r.swz = {R, R, R, ONE}; break;
        case 80: case fourcc('D', 'F', '1', '6'):
            r.format = VK_FORMAT_D16_UNORM; r.depth = true; r.blockBytes = 2; r.swz = {R, R, R, ONE}; break;
        case 82: case 83: r.format = VK_FORMAT_D32_SFLOAT; r.depth = true; r.swz = {R, R, R, ONE}; break;  // D32F_LOCKABLE
        default: break;
    }
    return r;
}

struct RingChunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* ptr = nullptr;
    VkDeviceSize size = 0, used = 0;
};
struct RingAlloc {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    uint8_t* ptr = nullptr;
};

struct DescKey {
    uint64_t b0 = 0, b1 = 0;
    uint64_t imgs[20 * 2]{};
    bool operator==(const DescKey& o) const {
        return b0 == o.b0 && b1 == o.b1 && std::memcmp(imgs, o.imgs, sizeof(imgs)) == 0;
    }
};
struct DescKeyHash {
    size_t operator()(const DescKey& k) const {
        uint64_t h = 0xCBF29CE484222325ull;
        auto m = [&](uint64_t v) { h = (h ^ v) * 0x100000001B3ull; };
        m(k.b0);
        m(k.b1);
        for (uint64_t v : k.imgs) m(v);
        return static_cast<size_t>(h ^ (h >> 32));
    }
};

struct Frame {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> pools;
    size_t poolIndex = 0;
    std::vector<RingChunk> chunks;
    size_t chunkIndex = 0;
    std::unordered_map<uint64_t, RingAlloc> bufferCache;
    std::unordered_map<uint64_t, VkDescriptorBufferInfo> uboCache;
    std::unordered_map<DescKey, VkDescriptorSet, DescKeyHash> descCache;
    VkPipeline lastPipeline = VK_NULL_HANDLE;
    VkDescriptorSet lastSet = VK_NULL_HANDLE;
    uint32_t lastDyn0 = ~0u, lastDyn1 = ~0u;
    bool hasDynState = false;
    VkViewport lastVp{};
    VkRect2D lastSc{};
    uint32_t lastBlend = 0;
    uint32_t lastStencil = 0;
    uint32_t lastBias195 = 0, lastSlope175 = 0;
    float lastUnits = 0.0f;
    VkBuffer lastVbuf[17] = {};
    VkDeviceSize lastVoff[17] = {};
    VkBuffer lastIbuf = VK_NULL_HANDLE;
    VkDeviceSize lastIoff = 0;
    VkIndexType lastItype = VK_INDEX_TYPE_UINT16;
    bool hasTargets = false;
    bool lastTargetSrgb = false;
    const void* lastTargetKey[4] = {};
    uint32_t lastTargetLevel[4] = {};
    uint32_t lastTargetFace[4] = {};
    const void* lastDepthKey = nullptr;
    struct CachedTargets {
        void* color[4] = {};
        VkImageView colorView[4] = {};
        VkFormat colorFmt[4] = {};
        uint32_t nColor = 0;
        void* depth = nullptr;
        VkImageView depthView = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0;
    } cachedTargets;
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Fmt9 fmt;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t d3dFormat = 0;
    uint32_t width = 1, height = 1, depth = 1, mips = 1, layers = 1;
    bool cube = false, is3D = false, attachable = false, decompressed = false, everSampled = false;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint64_t uploadedGen = ~0ull;
    std::map<uint64_t, VkImageView> views;
    VkImageView sampleView[2] = {};
    bool pendingColorClear = false;
    VkClearColorValue pendingColor{};
    bool pendingDepthClear = false;
    VkImageAspectFlags pendingDepthAsp = 0;
    VkClearDepthStencilValue pendingDepth{};
    uint32_t pendingMip = 0, pendingLayer = 0;
};

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* ptr = nullptr;
    uint32_t size = 0;
    uint64_t gen = ~0ull;
};

struct ShaderEntry {
    VkShaderModule module = VK_NULL_HANDLE;
    d3d9sh::ShaderInfo info;
    bool failed = false;
};

inline uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001B3ull + (h >> 7); }

VkBlendFactor blendFactor(uint32_t b, bool alpha) {
    switch (b) {
        case 1: return VK_BLEND_FACTOR_ZERO;
        case 2: return VK_BLEND_FACTOR_ONE;
        case 3: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
        case 4: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: case 12: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6: case 13: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: return VK_BLEND_FACTOR_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
        case 10: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 14: return alpha ? VK_BLEND_FACTOR_CONSTANT_ALPHA : VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 15: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp blendOp(uint32_t o) {
    switch (o) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}
VkCompareOp cmpOp(uint32_t c) { return (c >= 1 && c <= 8) ? static_cast<VkCompareOp>(c - 1) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp stencilOp(uint32_t o) { return (o >= 1 && o <= 8) ? static_cast<VkStencilOp>(o - 1) : VK_STENCIL_OP_KEEP; }
VkSamplerAddressMode addressMode(uint32_t a) {
    switch (a) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case 5: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}
VkFormat vertexFormat(uint32_t t) {
    switch (t) {
        case 0: return VK_FORMAT_R32_SFLOAT;
        case 1: return VK_FORMAT_R32G32_SFLOAT;
        case 2: return VK_FORMAT_R32G32B32_SFLOAT;
        case 3: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case 4: return VK_FORMAT_B8G8R8A8_UNORM;           // D3DCOLOR
        case 5: return VK_FORMAT_R8G8B8A8_USCALED;         // UBYTE4
        case 6: return VK_FORMAT_R16G16_SSCALED;           // SHORT2
        case 7: return VK_FORMAT_R16G16B16A16_SSCALED;     // SHORT4
        case 8: return VK_FORMAT_R8G8B8A8_UNORM;           // UBYTE4N
        case 9: return VK_FORMAT_R16G16_SNORM;
        case 10: return VK_FORMAT_R16G16B16A16_SNORM;
        case 11: return VK_FORMAT_R16G16_UNORM;
        case 12: return VK_FORMAT_R16G16B16A16_UNORM;
        case 13: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;  // UDEC3
        case 14: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;    // DEC3N
        case 15: return VK_FORMAT_R16G16_SFLOAT;
        case 16: return VK_FORMAT_R16G16B16A16_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

uint32_t vertexCount(uint32_t prim, uint32_t n) {
    switch (prim) {
        case 1: return n;
        case 2: return n * 2;
        case 3: return n + 1;
        case 4: return n * 3;
        case 5: case 6: return n + 2;
        default: return 0;
    }
}
VkPrimitiveTopology topology(uint32_t prim) {
    switch (prim) {
        case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

constexpr uint32_t kSamplerBindings = 20;  // 2..17 píxel, 18..21 vértice
constexpr uint32_t kUboSize = 4096 + 512 + 16;

class Impl {
public:
    std::mutex mtx;
    bool initTried = false, ok = false;
    DirectVulkanBridge::GpuContext ctx;
    Fns f;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDeviceSize uboAlign = 256;
    float maxAniso = 1.0f;
    VkFormat depthFormat24 = VK_FORMAT_D24_UNORM_S8_UINT;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    std::array<Frame, 2> frames;
    size_t slot = 0;
    bool recording = false;
    uint64_t statFenceNs = 0, statPresentNs = 0, statFrameNs = 0, statDrawNs = 0, statWorkerNs = 0;
    uint64_t statPasses = 0, statLoadPasses = 0, statDepthPasses = 0;
    int64_t lastPresentNs = 0;
    Renderer::Stats stats;

    std::unordered_map<const void*, std::unique_ptr<GpuImage>> images;
    std::unordered_map<const void*, GpuBuffer> buffers;
    std::unordered_map<uint64_t, ShaderEntry> shaders;
    std::unordered_map<uint64_t, VkPipeline> pipelines;
    std::unordered_map<uint64_t, VkSampler> samplers;
    std::map<std::array<uint64_t, 6>, VkRenderPass> renderPasses;
    std::map<std::array<uint64_t, 6>, VkFramebuffer> framebuffers;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;

    GpuImage dummy2D, dummyCube, dummy3D;
    VkImageView dummy2DView = VK_NULL_HANDLE, dummyCubeView = VK_NULL_HANDLE, dummy3DView = VK_NULL_HANDLE;
    VkSampler defaultSampler = VK_NULL_HANDLE, presentSampler = VK_NULL_HANDLE;
    VkBuffer zeroBuffer = VK_NULL_HANDLE;
    VkDeviceMemory zeroMemory = VK_NULL_HANDLE;

    // Render pass abierto.
    bool passOpen = false;
    std::array<uint64_t, 6> passKey{};
    VkRenderPass passRp = VK_NULL_HANDLE;
    uint32_t passW = 0, passH = 0;
    GpuImage* passColor[4] = {};
    GpuImage* passDepth = nullptr;

    // Presentación.
    VkRenderPass presentPass = VK_NULL_HANDLE;
    VkFormat presentFormat = VK_FORMAT_UNDEFINED;
    VkPipeline presentPipeline = VK_NULL_HANDLE;
    VkPipelineLayout presentLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout presentSetLayout = VK_NULL_HANDLE;
    VkPipeline sgsrPipeline = VK_NULL_HANDLE;
    VkPipelineLayout sgsrLayout = VK_NULL_HANDLE;
    uint64_t presentGeneration = ~0ull;
    std::map<VkImageView, VkFramebuffer> presentFbs;

    // ----------------------------------------------------------------------- inicio
    bool init() {
        if (initTried) return ok;
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.gpu", v) > 0 && v[0] == '0') { initTried = true; return false; }
        if (!DirectVulkanBridge::shared().gpuContext(ctx)) return false;  // reintenta más tarde
        initTried = true;
#define HP_LOAD(name)                                                                \
    f.name = reinterpret_cast<PFN_##name>(ctx.getDeviceProcAddr(ctx.device, #name)); \
    if (f.name == nullptr) {                                                         \
        LOGW("Falta %s: ruta GPU D3D9 desactivada", #name);                          \
        return false;                                                                \
    }
        HP_VK9_FNS(HP_LOAD)
#undef HP_LOAD
        f.getMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceMemoryProperties"));
        f.getProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceProperties"));
        f.getFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceFormatProperties"));
        if (!f.getMemoryProperties || !f.getProperties || !f.getFormatProperties) return false;
        f.getMemoryProperties(ctx.physicalDevice, &memProps);
        VkPhysicalDeviceProperties props{};
        f.getProperties(ctx.physicalDevice, &props);
        uboAlign = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 16);
        maxAniso = ctx.samplerAnisotropy ? props.limits.maxSamplerAnisotropy : 1.0f;
        VkFormatProperties fp{};
        f.getFormatProperties(ctx.physicalDevice, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
            depthFormat24 = VK_FORMAT_D32_SFLOAT_S8_UINT;

        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = ctx.queueFamily;
        if (f.vkCreateCommandPool(ctx.device, &cp, nullptr, &cmdPool) != VK_SUCCESS) return false;
        for (Frame& fr : frames) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            if (f.vkAllocateCommandBuffers(ctx.device, &ai, &fr.cmd) != VK_SUCCESS) return false;
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (f.vkCreateFence(ctx.device, &fi, nullptr, &fr.fence) != VK_SUCCESS) return false;
        }
        if (!createLayouts() || !createDummies()) return false;
        ok = true;
        LOGI("Ruta GPU D3D9 -> Vulkan activa (BC nativo=%d, profundidad 24 bits=%s)", ctx.textureCompressionBC ? 1 : 0,
             depthFormat24 == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "D32S8");
        return true;
    }

    bool createLayouts() {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.push_back({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr});
        b.push_back({1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        for (uint32_t i = 0; i < kSamplerBindings; ++i)
            b.push_back({2 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                         static_cast<VkShaderStageFlags>(i < 16 ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT),
                         nullptr});
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings = b.data();
        if (f.vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &setLayout) != VK_SUCCESS) return false;
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pi.setLayoutCount = 1;
        pi.pSetLayouts = &setLayout;
        return f.vkCreatePipelineLayout(ctx.device, &pi, nullptr, &pipeLayout) == VK_SUCCESS;
    }

    // ----------------------------------------------------------------------- memoria
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    }
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem, uint8_t** mapped,
                      VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (f.vkCreateBuffer(ctx.device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        f.vkGetBufferMemoryRequirements(ctx.device, buf, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, memFlags);
        if (ai.memoryTypeIndex == UINT32_MAX)
            ai.memoryTypeIndex = memoryType(req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX) return false;
        if (f.vkAllocateMemory(ctx.device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        f.vkBindBufferMemory(ctx.device, buf, mem, 0);
        if (mapped != nullptr) {
            void* p = nullptr;
            if (f.vkMapMemory(ctx.device, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
            *mapped = static_cast<uint8_t*>(p);
        }
        return true;
    }

    RingAlloc ringAlloc(VkDeviceSize size, VkDeviceSize align) {
        Frame& fr = frames[slot];
        size = std::max<VkDeviceSize>(size, 4);
        for (;;) {
            if (fr.chunkIndex < fr.chunks.size()) {
                RingChunk& c = fr.chunks[fr.chunkIndex];
                const VkDeviceSize off = (c.used + align - 1) / align * align;
                if (off + size <= c.size) {
                    c.used = off + size;
                    return {c.buffer, off, c.ptr + off};
                }
                ++fr.chunkIndex;
                continue;
            }
            RingChunk c;
            c.size = std::max<VkDeviceSize>(32ull << 20, size + align);
            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (!createBuffer(c.size, usage, c.buffer, c.memory, &c.ptr)) {
                LOGW("Anillo: no se pudo reservar %llu bytes", static_cast<unsigned long long>(c.size));
                return {};
            }
            fr.chunks.push_back(c);
        }
    }

    // ----------------------------------------------------------------------- imágenes
    bool createImage(GpuImage& g) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = g.is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ii.format = g.format;
        ii.extent = {g.width, g.height, g.depth};
        ii.mipLevels = g.mips;
        ii.arrayLayers = g.layers;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (g.attachable)
            ii.usage |= g.fmt.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (g.fmt.srgb != VK_FORMAT_UNDEFINED && !g.decompressed) ii.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        if (g.cube) ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (f.vkCreateImage(ctx.device, &ii, nullptr, &g.image) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        f.vkGetImageMemoryRequirements(ctx.device, g.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX) ai.memoryTypeIndex = memoryType(req.memoryTypeBits, 0);
        if (f.vkAllocateMemory(ctx.device, &ai, nullptr, &g.memory) != VK_SUCCESS) {
            f.vkDestroyImage(ctx.device, g.image, nullptr);
            g.image = VK_NULL_HANDLE;
            return false;
        }
        f.vkBindImageMemory(ctx.device, g.image, g.memory, 0);
        g.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        g.aspect = g.fmt.depth ? (VK_IMAGE_ASPECT_DEPTH_BIT | (g.fmt.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0))
                               : VK_IMAGE_ASPECT_COLOR_BIT;
        return true;
    }

    void purgeFramebuffers(GpuImage& g) {
        for (auto it = framebuffers.begin(); it != framebuffers.end();) {
            bool uses = false;
            for (auto& [k, v] : g.views)
                for (uint64_t x : it->first) uses = uses || x == reinterpret_cast<uint64_t>(v);
            if (uses) {
                f.vkDestroyFramebuffer(ctx.device, it->second, nullptr);
                it = framebuffers.erase(it);
            } else {
                ++it;
            }
        }
    }
    void destroyImage(GpuImage& g) {
        purgeFramebuffers(g);
        for (auto& [k, v] : g.views) f.vkDestroyImageView(ctx.device, v, nullptr);
        if (g.image) f.vkDestroyImage(ctx.device, g.image, nullptr);
        if (g.memory) f.vkFreeMemory(ctx.device, g.memory, nullptr);
        g = GpuImage{};
    }

    // Vista: tipo, sRGB, rango (nivel/capa). Para muestrear se aplica el swizzle del formato D3D9.
    VkImageView view(GpuImage& g, VkImageViewType type, bool srgb, uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer,
                     uint32_t layerCount, bool forSampling) {
        const bool fastSample = forSampling && baseMip == 0 && mipCount == g.mips && baseLayer == 0 &&
                                !g.is3D && ((type == VK_IMAGE_VIEW_TYPE_CUBE && g.cube) ||
                                            (type != VK_IMAGE_VIEW_TYPE_CUBE && !g.cube));
        if (fastSample && g.sampleView[srgb ? 1 : 0] != VK_NULL_HANDLE) return g.sampleView[srgb ? 1 : 0];
        baseMip = std::min(baseMip, g.mips - 1);
        mipCount = std::min(mipCount, g.mips - baseMip);
        baseLayer = std::min(baseLayer, g.layers - 1);
        if (g.is3D) type = VK_IMAGE_VIEW_TYPE_3D;
        if (type == VK_IMAGE_VIEW_TYPE_CUBE && !g.cube) type = VK_IMAGE_VIEW_TYPE_2D;
        layerCount = type == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : 1;
        if (type == VK_IMAGE_VIEW_TYPE_CUBE) baseLayer = 0;
        const VkFormat format = (srgb && g.fmt.srgb != VK_FORMAT_UNDEFINED && !g.decompressed) ? g.fmt.srgb : g.format;
        const uint64_t key = mix(mix(mix(mix(static_cast<uint64_t>(type), static_cast<uint64_t>(format)),
                                         baseMip * 64 + mipCount), baseLayer * 4096 + layerCount),
                                 forSampling ? 1 : 0);
        auto it = g.views.find(key);
        if (it != g.views.end()) {
            if (fastSample) g.sampleView[srgb ? 1 : 0] = it->second;
            return it->second;
        }
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = g.image;
        vi.viewType = type;
        vi.format = format;
        if (forSampling && !g.decompressed) vi.components = g.fmt.swz;
        vi.subresourceRange = {forSampling && g.fmt.depth ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) : g.aspect,
                               baseMip, mipCount, baseLayer, layerCount};
        VkImageView v = VK_NULL_HANDLE;
        if (f.vkCreateImageView(ctx.device, &vi, nullptr, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
        g.views[key] = v;
        if (fastSample) g.sampleView[srgb ? 1 : 0] = v;
        return v;
    }

    VkImageLayout readLayout(const GpuImage& g) {
        return g.fmt.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkImageLayout attachLayout(const GpuImage& g) {
        return g.fmt.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    void flushPendingClear(GpuImage& g) {
        if (g.pendingColorClear) {
            g.pendingColorClear = false;
            barrier(g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, g.pendingMip, 1, g.pendingLayer, 1};
            f.vkCmdClearColorImage(frames[slot].cmd, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &g.pendingColor, 1, &rng);
        }
        if (g.pendingDepthClear) {
            g.pendingDepthClear = false;
            const VkImageAspectFlags asp = g.pendingDepthAsp;
            barrier(g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageSubresourceRange rng{asp, g.pendingMip, 1, g.pendingLayer, 1};
            f.vkCmdClearDepthStencilImage(frames[slot].cmd, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &g.pendingDepth, 1, &rng);
        }
    }

    void barrier(GpuImage& g, VkImageLayout to) {
        if (to != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && to != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
            to != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && (g.pendingColorClear || g.pendingDepthClear)) {
            flushPendingClear(g);
        }
        if (g.layout == to) return;
        endPass();
        auto srcInfo = [](VkImageLayout l) -> std::pair<VkPipelineStageFlags, VkAccessFlags> {
            switch (l) {
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                    return {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                    return {VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
                case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
                case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                    return {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0};
                case VK_IMAGE_LAYOUT_GENERAL:
                    return {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                default:
                    return {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0};
            }
        };
        auto dstInfo = [](VkImageLayout l) -> std::pair<VkPipelineStageFlags, VkAccessFlags> {
            switch (l) {
                case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                    return {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
                case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                    return {VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
                case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
                case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
                case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                    return {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT};
                case VK_IMAGE_LAYOUT_GENERAL:
                    return {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
                default:
                    return {VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0};
            }
        };
        const auto [srcStage, srcAccess] = srcInfo(g.layout);
        const auto [dstStage, dstAccess] = dstInfo(to);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.oldLayout = g.layout;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = g.image;
        b.subresourceRange = {g.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        f.vkCmdPipelineBarrier(frames[slot].cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        g.layout = to;
    }

    bool createDummy(GpuImage& g, bool cube, bool is3D) {
        g.format = VK_FORMAT_R8G8B8A8_UNORM;
        g.fmt = Fmt9{};
        g.fmt.format = g.format;
        g.cube = cube;
        g.is3D = is3D;
        g.layers = cube ? 6 : 1;
        return createImage(g);
    }
    bool createDummies() {
        if (!createDummy(dummy2D, false, false) || !createDummy(dummyCube, true, false) || !createDummy(dummy3D, false, true))
            return false;
        if (!createBuffer(65536, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, zeroBuffer, zeroMemory, nullptr)) return false;
        void* p = nullptr;
        if (f.vkMapMemory(ctx.device, zeroMemory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) std::memset(p, 0, 65536);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        if (f.vkCreateSampler(ctx.device, &si, nullptr, &defaultSampler) != VK_SUCCESS) return false;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        if (f.vkCreateSampler(ctx.device, &si, nullptr, &presentSampler) != VK_SUCCESS) presentSampler = defaultSampler;
        return true;
    }
    bool dummiesCleared = false;
    void clearDummies() {
        if (dummiesCleared) return;
        dummiesCleared = true;
        for (GpuImage* g : {&dummy2D, &dummyCube, &dummy3D}) {
            barrier(*g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue c{};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, g->layers};
            f.vkCmdClearColorImage(frames[slot].cmd, g->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
            barrier(*g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    // Reflejo GPU de una textura/superficie (creación + subida si la CPU la cambió).
    GpuImage* image(const TexRef& t, bool asTarget) {
        if (!t.valid()) return nullptr;
        Fmt9 fmt = fmt9(t.format);
        if (fmt.format == VK_FORMAT_UNDEFINED) {
            static int warned = 0;
            if (warned++ < 12) LOGW("Formato D3D9 0x%x sin equivalente Vulkan", t.format);
            return nullptr;
        }
        if (fmt.format == VK_FORMAT_D24_UNORM_S8_UINT) fmt.format = depthFormat24;
        bool decompress = false;
        if (fmt.dxgiBc != 0 && !ctx.textureCompressionBC) {
            decompress = true;
            fmt.format = VK_FORMAT_R8G8B8A8_UNORM;
        }
        const bool wantAttach = asTarget || (t.usage & 3u) != 0 || fmt.depth;
        auto& slotPtr = images[t.key];
        if (!slotPtr || slotPtr->width != t.width || slotPtr->height != t.height || slotPtr->d3dFormat != t.format ||
            (wantAttach && !slotPtr->attachable)) {
            if (slotPtr) {
                if (passOpen) endPass();
                destroyImage(*slotPtr);
            }
            slotPtr = std::make_unique<GpuImage>();
            GpuImage& g = *slotPtr;
            g.fmt = fmt;
            g.format = fmt.format;
            g.d3dFormat = t.format;
            g.decompressed = decompress;
            g.width = std::max(1u, t.width);
            g.height = std::max(1u, t.height);
            g.is3D = t.kind == 2;
            g.depth = g.is3D ? std::max(1u, t.depth) : 1;
            g.mips = std::max(1u, t.levels);
            g.cube = t.kind == 1;
            g.layers = g.cube ? 6 : 1;
            g.attachable = wantAttach && fmt.blockW == 1 && !decompress;
            if (t.format == 0x5A544E49u || t.format == 0x34324644u || t.format == 0x36314644u)
                g.everSampled = true;  // INTZ / DF24 / DF16: profundidad muestreada
            if (!createImage(g)) {
                LOGW("vkCreateImage %ux%u fmt 0x%x falló", g.width, g.height, t.format);
                slotPtr.reset();
                return nullptr;
            }
            // Contenido inicial definido (0).
            barrier(g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            if (g.fmt.depth) {
                VkClearDepthStencilValue c{1.0f, 0};
                VkImageSubresourceRange r{g.aspect, 0, g.mips, 0, g.layers};
                f.vkCmdClearDepthStencilImage(frames[slot].cmd, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
            } else if (fmt.blockW == 1 || decompress) {
                VkClearColorValue c{};
                VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, g.mips, 0, g.layers};
                f.vkCmdClearColorImage(frames[slot].cmd, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
            }
        }
        GpuImage& g = *slotPtr;
        if (t.generation != g.uploadedGen && t.data != nullptr && !g.fmt.depth) upload(g, t);
        g.uploadedGen = t.generation;
        return &g;
    }

    void upload(GpuImage& g, const TexRef& t) {
        barrier(g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const uint32_t faces = g.cube ? 6 : 1;
        for (uint32_t face = 0; face < faces; ++face) {
            for (uint32_t mip = 0; mip < g.mips; ++mip) {
                const uint32_t idx = face * t.levels + mip;
                if (idx >= t.subresources) continue;
                const uint8_t* src = t.data[idx];
                if (src == nullptr) continue;
                const uint32_t w = std::max(1u, g.width >> mip), h = std::max(1u, g.height >> mip);
                const uint32_t d = g.is3D ? std::max(1u, g.depth >> mip) : 1;
                const uint32_t bw = (w + g.fmt.blockW - 1) / g.fmt.blockW, bh = (h + g.fmt.blockW - 1) / g.fmt.blockW;
                const uint32_t pitch = (bw * g.fmt.blockBytes + 3) & ~3u;  // igual que D3D9Device32::rowPitch
                VkBufferImageCopy r{};
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, face, 1};
                r.imageExtent = {w, h, d};
                RingAlloc a{};
                if (g.decompressed) {
                    std::vector<uint8_t> rgba;
                    if (!d3d11::decompressToRgba8(g.fmt.dxgiBc, src, w, h, pitch, rgba)) continue;
                    a = ringAlloc(rgba.size(), 16);
                    if (a.ptr == nullptr) continue;
                    std::memcpy(a.ptr, rgba.data(), rgba.size());
                } else if (g.fmt.convert != kNone) {
                    const size_t out = static_cast<size_t>(w) * h * d * 4;
                    a = ringAlloc(out, 16);
                    if (a.ptr == nullptr) continue;
                    for (uint32_t z = 0; z < d; ++z)
                        for (uint32_t y = 0; y < h; ++y) {
                            const uint8_t* row = src + (static_cast<size_t>(z) * h + y) * pitch;
                            uint8_t* o = a.ptr + (static_cast<size_t>(z) * h + y) * w * 4;
                            for (uint32_t x = 0; x < w; ++x, o += 4) {
                                if (g.fmt.convert == kRgb24ToBgra) {
                                    o[0] = row[x * 3]; o[1] = row[x * 3 + 1]; o[2] = row[x * 3 + 2]; o[3] = 255;
                                } else {
                                    uint16_t p;
                                    std::memcpy(&p, row + x * 2, 2);
                                    o[0] = static_cast<uint8_t>(((p >> 8) & 0xF) * 17);
                                    o[1] = static_cast<uint8_t>(((p >> 4) & 0xF) * 17);
                                    o[2] = static_cast<uint8_t>((p & 0xF) * 17);
                                    o[3] = g.fmt.convert == kArgb4ToRgba8 ? static_cast<uint8_t>((p >> 12) * 17) : 255;
                                }
                            }
                        }
                } else {
                    const size_t bytes = static_cast<size_t>(pitch) * bh * d;
                    a = ringAlloc(bytes, 16);
                    if (a.ptr == nullptr) continue;
                    std::memcpy(a.ptr, src, bytes);
                    r.bufferRowLength = g.fmt.blockW == 1 ? pitch / g.fmt.blockBytes : (pitch / g.fmt.blockBytes) * 4;
                    r.bufferImageHeight = bh * g.fmt.blockW;
                }
                r.bufferOffset = a.offset;
                f.vkCmdCopyBufferToImage(frames[slot].cmd, a.buffer, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            }
        }
        ++stats.uploads;
    }

    // ----------------------------------------------------------------------- frame
    bool beginFrame() {
        if (recording) return true;
        Frame& fr = frames[slot];
        timespec t0{}, t1{};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        f.vkWaitForFences(ctx.device, 1, &fr.fence, VK_TRUE, 2000000000ull);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        statFenceNs += static_cast<uint64_t>((t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec));
        f.vkResetFences(ctx.device, 1, &fr.fence);
        for (VkDescriptorPool p : fr.pools) f.vkResetDescriptorPool(ctx.device, p, 0);
        fr.poolIndex = 0;
        for (RingChunk& c : fr.chunks) c.used = 0;
        fr.chunkIndex = 0;
        fr.bufferCache.clear();
        fr.uboCache.clear();
        fr.descCache.clear();
        fr.lastPipeline = VK_NULL_HANDLE;
        fr.lastSet = VK_NULL_HANDLE;
        fr.lastDyn0 = ~0u;
        fr.lastDyn1 = ~0u;
        fr.hasDynState = false;
        fr.hasTargets = false;
        std::memset(fr.lastVbuf, 0, sizeof(fr.lastVbuf));
        fr.lastIbuf = VK_NULL_HANDLE;
        f.vkResetCommandBuffer(fr.cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (f.vkBeginCommandBuffer(fr.cmd, &bi) != VK_SUCCESS) return false;
        recording = true;
        clearDummies();
        return true;
    }

    VkDescriptorSet allocSet(VkDescriptorSetLayout layout) {
        Frame& fr = frames[slot];
        for (;;) {
            if (fr.poolIndex >= fr.pools.size()) {
                const VkDescriptorPoolSize sizes[] = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096},
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 * kSamplerBindings},
                };
                VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                pi.maxSets = 2048;
                pi.poolSizeCount = 2;
                pi.pPoolSizes = sizes;
                VkDescriptorPool p = VK_NULL_HANDLE;
                if (f.vkCreateDescriptorPool(ctx.device, &pi, nullptr, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
                fr.pools.push_back(p);
            }
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = fr.pools[fr.poolIndex];
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (f.vkAllocateDescriptorSets(ctx.device, &ai, &set) == VK_SUCCESS) return set;
            ++fr.poolIndex;
        }
    }

    // ----------------------------------------------------------------------- render passes
    VkRenderPass renderPass(const VkFormat* colors, uint32_t nColor, VkFormat depth, uint32_t clearMask = 0,
                            bool depthStore = true, uint32_t dontCareMask = 0) {
        std::array<uint64_t, 6> key{};
        for (uint32_t i = 0; i < nColor; ++i) key[i] = static_cast<uint64_t>(colors[i]);
        key[4] = static_cast<uint64_t>(depth) | (1ull << 40);
        key[5] = static_cast<uint64_t>(clearMask) | (static_cast<uint64_t>(nColor) << 16) |
                 (depthStore ? (1ull << 24) : 0ull) | (static_cast<uint64_t>(dontCareMask) << 25);
        auto it = renderPasses.find(key);
        if (it != renderPasses.end()) return it->second;
        std::vector<VkAttachmentDescription> att;
        std::vector<VkAttachmentReference> refs;
        for (uint32_t i = 0; i < nColor; ++i) {
            VkAttachmentDescription a{};
            a.format = colors[i];
            a.samples = VK_SAMPLE_COUNT_1_BIT;
            a.loadOp = (clearMask & (1u << i)) ? VK_ATTACHMENT_LOAD_OP_CLEAR
                     : (dontCareMask & (1u << i)) ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                                                  : VK_ATTACHMENT_LOAD_OP_LOAD;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.push_back(a);
            refs.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        }
        VkAttachmentReference dref{nColor, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        if (depth != VK_FORMAT_UNDEFINED) {
            VkAttachmentDescription a{};
            a.format = depth;
            a.samples = VK_SAMPLE_COUNT_1_BIT;
            a.loadOp = (clearMask & (1u << 4)) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            a.stencilLoadOp = (clearMask & (1u << 5)) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            a.storeOp = a.stencilStoreOp = depthStore ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            att.push_back(a);
        }
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = nColor;
        sp.pColorAttachments = refs.data();
        sp.pDepthStencilAttachment = depth != VK_FORMAT_UNDEFINED ? &dref : nullptr;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = static_cast<uint32_t>(att.size());
        ri.pAttachments = att.data();
        ri.subpassCount = 1;
        ri.pSubpasses = &sp;
        VkRenderPass rp = VK_NULL_HANDLE;
        if (f.vkCreateRenderPass(ctx.device, &ri, nullptr, &rp) != VK_SUCCESS) return VK_NULL_HANDLE;
        renderPasses[key] = rp;
        return rp;
    }

    void endPass() {
        if (!passOpen) return;
        f.vkCmdEndRenderPass(frames[slot].cmd);
        passOpen = false;
        passKey = {};
    }

    struct Targets {
        GpuImage* color[4] = {};
        VkImageView colorView[4] = {};
        VkFormat colorFmt[4] = {};
        uint32_t nColor = 0;
        GpuImage* depth = nullptr;
        VkImageView depthView = VK_NULL_HANDLE;
        uint32_t w = 0, h = 0;
    };

    bool resolveTargets(const DrawState& s, Targets& t) {
        const bool srgb = s.renderState != nullptr && s.renderState[194] != 0;
        Frame& fr = frames[slot];
        if (fr.hasTargets && fr.lastTargetSrgb == srgb && fr.lastDepthKey == s.depth.key &&
            fr.lastTargetKey[0] == s.targets[0].key && fr.lastTargetLevel[0] == s.targets[0].level && fr.lastTargetFace[0] == s.targets[0].face &&
            fr.lastTargetKey[1] == s.targets[1].key && fr.lastTargetLevel[1] == s.targets[1].level && fr.lastTargetFace[1] == s.targets[1].face &&
            fr.lastTargetKey[2] == s.targets[2].key && fr.lastTargetLevel[2] == s.targets[2].level && fr.lastTargetFace[2] == s.targets[2].face &&
            fr.lastTargetKey[3] == s.targets[3].key && fr.lastTargetLevel[3] == s.targets[3].level && fr.lastTargetFace[3] == s.targets[3].face) {
            t.nColor = fr.cachedTargets.nColor;
            for (uint32_t i = 0; i < t.nColor; ++i) {
                t.color[i] = static_cast<GpuImage*>(fr.cachedTargets.color[i]);
                t.colorView[i] = fr.cachedTargets.colorView[i];
                t.colorFmt[i] = fr.cachedTargets.colorFmt[i];
            }
            t.depth = static_cast<GpuImage*>(fr.cachedTargets.depth);
            t.depthView = fr.cachedTargets.depthView;
            t.w = fr.cachedTargets.w;
            t.h = fr.cachedTargets.h;
            return t.nColor > 0 || t.depth != nullptr;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            if (!s.targets[i].valid()) continue;
            if (s.targets[i].format == fourcc('N', 'U', 'L', 'L')) continue;
            GpuImage* g = image(s.targets[i], true);
            if (g == nullptr || !g->attachable) continue;
            const VkImageView v = view(*g, VK_IMAGE_VIEW_TYPE_2D, srgb, s.targets[i].level, 1, s.targets[i].face, 1, false);
            if (v == VK_NULL_HANDLE) continue;
            t.color[t.nColor] = g;
            t.colorView[t.nColor] = v;
            t.colorFmt[t.nColor] = (srgb && g->fmt.srgb != VK_FORMAT_UNDEFINED) ? g->fmt.srgb : g->format;
            const uint32_t w = std::max(1u, g->width >> s.targets[i].level), h = std::max(1u, g->height >> s.targets[i].level);
            t.w = t.w == 0 ? w : std::min(t.w, w);
            t.h = t.h == 0 ? h : std::min(t.h, h);
            ++t.nColor;
        }
        if (s.depth.valid()) {
            GpuImage* g = image(s.depth, true);
            if (g != nullptr && g->fmt.depth && g->attachable) {
                t.depth = g;
                t.depthView = view(*g, VK_IMAGE_VIEW_TYPE_2D, false, 0, 1, 0, 1, false);
                t.w = t.w == 0 ? g->width : std::min(t.w, g->width);
                t.h = t.h == 0 ? g->height : std::min(t.h, g->height);
            }
        }
        fr.hasTargets = true;
        fr.lastTargetSrgb = srgb;
        fr.lastDepthKey = s.depth.key;
        for (uint32_t i = 0; i < 4; ++i) {
            fr.lastTargetKey[i] = s.targets[i].key;
            fr.lastTargetLevel[i] = s.targets[i].level;
            fr.lastTargetFace[i] = s.targets[i].face;
            fr.cachedTargets.color[i] = t.color[i];
            fr.cachedTargets.colorView[i] = t.colorView[i];
            fr.cachedTargets.colorFmt[i] = t.colorFmt[i];
        }
        fr.cachedTargets.nColor = t.nColor;
        fr.cachedTargets.depth = t.depth;
        fr.cachedTargets.depthView = t.depthView;
        fr.cachedTargets.w = t.w;
        fr.cachedTargets.h = t.h;
        return t.nColor > 0 || t.depth != nullptr;
    }

    VkRenderPass beginPass(const Targets& t, uint32_t dontCareMask = 0) {
        std::array<uint64_t, 6> fbKey{};
        for (uint32_t i = 0; i < t.nColor; ++i) fbKey[i] = reinterpret_cast<uint64_t>(t.colorView[i]);
        fbKey[4] = reinterpret_cast<uint64_t>(t.depthView);
        fbKey[5] = (static_cast<uint64_t>(t.w) << 32) | t.h;
        if (passOpen && fbKey == passKey) return passRp;
        endPass();
        uint32_t clearMask = 0;
        VkClearValue clearVals[5]{};
        for (uint32_t i = 0; i < t.nColor; ++i) {
            if (t.color[i]->pendingColorClear) {
                clearMask |= (1u << i);
                clearVals[i].color = t.color[i]->pendingColor;
                t.color[i]->pendingColorClear = false;
            }
            barrier(*t.color[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
        dontCareMask &= ~clearMask;
        if (t.depth != nullptr) {
            if (t.depth->pendingDepthClear) {
                if (t.depth->pendingDepthAsp & VK_IMAGE_ASPECT_DEPTH_BIT) clearMask |= (1u << 4);
                if (t.depth->pendingDepthAsp & VK_IMAGE_ASPECT_STENCIL_BIT) clearMask |= (1u << 5);
                clearVals[t.nColor].depthStencil = t.depth->pendingDepth;
                t.depth->pendingDepthClear = false;
                t.depth->pendingDepthAsp = 0;
            }
            barrier(*t.depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }
        const bool depthStore = true;  // UE3/D3D9 reutiliza depth/stencil entre múltiples pases sin muestrear
        const VkRenderPass rp = renderPass(t.colorFmt, t.nColor, t.depth != nullptr ? t.depth->format : VK_FORMAT_UNDEFINED,
                                           clearMask, depthStore, dontCareMask);
        if (rp == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkFramebuffer& fb = framebuffers[fbKey];
        if (fb == VK_NULL_HANDLE) {
            VkImageView views[5]{};
            for (uint32_t i = 0; i < t.nColor; ++i) views[i] = t.colorView[i];
            uint32_t nViews = t.nColor;
            if (t.depth != nullptr) views[nViews++] = t.depthView;
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = renderPass(t.colorFmt, t.nColor, t.depth != nullptr ? t.depth->format : VK_FORMAT_UNDEFINED, 0);
            fi.attachmentCount = nViews;
            fi.pAttachments = views;
            fi.width = t.w;
            fi.height = t.h;
            fi.layers = 1;
            if (f.vkCreateFramebuffer(ctx.device, &fi, nullptr, &fb) != VK_SUCCESS) {
                framebuffers.erase(fbKey);
                return VK_NULL_HANDLE;
            }
        }
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = rp;
        bi.framebuffer = fb;
        bi.renderArea = {{0, 0}, {t.w, t.h}};
        bi.clearValueCount = clearMask != 0 ? (t.nColor + (t.depth != nullptr ? 1u : 0u)) : 0u;
        bi.pClearValues = clearMask != 0 ? clearVals : nullptr;
        f.vkCmdBeginRenderPass(frames[slot].cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        ++statPasses;
        if (clearMask == 0) ++statLoadPasses;
        if (t.depth != nullptr) ++statDepthPasses;
        passOpen = true;
        passKey = fbKey;
        passRp = rp;
        passW = t.w;
        passH = t.h;
        return rp;
    }

    // ----------------------------------------------------------------------- shaders y pipelines
    VkShaderModule makeModule(const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = code.size() * 4;
        si.pCode = code.data();
        VkShaderModule m = VK_NULL_HANDLE;
        return f.vkCreateShaderModule(ctx.device, &si, nullptr, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
    }

    ShaderEntry* shader(const void* key, const uint32_t* tok, size_t n, bool alphaTest = true) {
        if (key == nullptr || tok == nullptr) return nullptr;
        const uint64_t skey = reinterpret_cast<uintptr_t>(key) ^ (alphaTest ? 1ull : 0ull);
        auto it = shaders.find(skey);
        if (it != shaders.end()) return it->second.failed ? nullptr : &it->second;
        ShaderEntry& e = shaders[skey];
        std::vector<uint32_t> spv;
        std::string err;
        if (!d3d9sh::translate(tok, n, spv, e.info, err, alphaTest)) {
            e.failed = true;
            LOGW("Shader D3D9 %p no traducido: %s", key, err.c_str());
            return nullptr;
        }
        e.module = makeModule(spv);
        if (e.module == VK_NULL_HANDLE) {
            e.failed = true;
            LOGW("vkCreateShaderModule falló para %p", key);
            return nullptr;
        }
        return &e;
    }

    uint64_t lastSamplerKey = ~0ull;
    VkSampler lastSamplerVal = VK_NULL_HANDLE;
    VkSampler sampler(const uint32_t* ss) {
        // ss: [1] ADDRESSU [2] V [3] W [4] BORDERCOLOR [5] MAG [6] MIN [7] MIP [8] LODBIAS [9] MAXMIPLEVEL [10] ANISO
        uint64_t key = 0;
        for (int i = 1; i <= 10; ++i) key = mix(key, ss[i]);
        if (key == lastSamplerKey && lastSamplerVal != VK_NULL_HANDLE) return lastSamplerVal;
        auto it = samplers.find(key);
        if (it != samplers.end()) {
            lastSamplerKey = key;
            lastSamplerVal = it->second;
            return it->second;
        }
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = ss[5] >= 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.minFilter = ss[6] >= 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.mipmapMode = ss[7] >= 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = addressMode(ss[1]);
        si.addressModeV = addressMode(ss[2]);
        si.addressModeW = addressMode(ss[3]);
        float bias = 0.0f;
        std::memcpy(&bias, &ss[8], 4);
        si.mipLodBias = std::clamp(bias, -15.0f, 15.0f);
        si.minLod = static_cast<float>(ss[9]);
        si.maxLod = ss[7] == 0 ? si.minLod + 0.25f : VK_LOD_CLAMP_NONE;
        if ((ss[5] == 3 || ss[6] == 3) && ss[10] > 1 && maxAniso > 1.0f) {
            si.anisotropyEnable = VK_TRUE;
            si.maxAnisotropy = std::min(static_cast<float>(ss[10]), maxAniso);
            si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        }
        const uint32_t bc = ss[4];
        si.borderColor = (bc >> 24) < 0x80 ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                         : (bc & 0xFFFFFF) > 0x7F7F7F ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                                      : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        VkSampler smp = VK_NULL_HANDLE;
        if (f.vkCreateSampler(ctx.device, &si, nullptr, &smp) != VK_SUCCESS) smp = defaultSampler;
        samplers[key] = smp;
        return smp;
    }

    struct VertexLayout {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attrs;
        uint32_t streamMask = 0;
        bool needZero = false;
    };
    VertexLayout tmpVl_;

    bool vertexLayout(const DrawState& s, const d3d9sh::ShaderInfo& vsi, VertexLayout& out) {
        out.bindings.clear();
        out.attrs.clear();
        out.streamMask = 0;
        out.needZero = false;
        struct Elem { uint16_t stream, offset; uint8_t type, method, usage, index; };
        Elem elems[64];
        uint32_t nElems = 0;
        if (s.decl != nullptr) {
            for (uint32_t i = 0; i < 64; ++i) {
                Elem e;
                std::memcpy(&e, s.decl + 8 * i, 8);
                if (e.stream == 0xFF || e.type == 17) break;
                elems[nElems++] = e;
            }
        }
        for (const auto& in : vsi.inputs) {
            const Elem* found = nullptr;
            for (uint32_t ei = 0; ei < nElems; ++ei)
                if (elems[ei].usage == in.usage && elems[ei].index == in.index) { found = &elems[ei]; break; }
            if (found == nullptr && in.usage == 0)  // POSITION <-> POSITIONT
                for (uint32_t ei = 0; ei < nElems; ++ei)
                    if (elems[ei].usage == 9 && elems[ei].index == in.index) { found = &elems[ei]; break; }
            VkVertexInputAttributeDescription a{};
            a.location = in.reg;
            if (found == nullptr || found->stream >= 16 || vertexFormat(found->type) == VK_FORMAT_UNDEFINED) {
                a.binding = 16;
                a.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                out.needZero = true;
            } else {
                a.binding = found->stream;
                a.format = vertexFormat(found->type);
                a.offset = found->offset;
                out.streamMask |= 1u << found->stream;
            }
            out.attrs.push_back(a);
        }
        for (uint32_t st = 0; st < 16; ++st) {
            if (!(out.streamMask & (1u << st))) continue;
            VkVertexInputBindingDescription b{};
            b.binding = st;
            b.stride = s.streams[st].stride;
            b.inputRate = (s.streams[st].freq & 0x80000000u) ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            out.bindings.push_back(b);
        }
        if (out.needZero) out.bindings.push_back({16, 0, VK_VERTEX_INPUT_RATE_VERTEX});
        return true;
    }

    VkPipeline pipeline(const DrawState& s, ShaderEntry& vs, ShaderEntry& ps, const VertexLayout& vl, uint32_t prim,
                        const Targets& t, VkRenderPass rp) {
        const uint32_t* r = s.renderState;
        uint64_t key = mix(reinterpret_cast<uint64_t>(vs.module), reinterpret_cast<uint64_t>(ps.module));
        key = mix(key, reinterpret_cast<uint64_t>(rp));
        key = mix(key, prim);
        for (const auto& b : vl.bindings) key = mix(key, (static_cast<uint64_t>(b.binding) << 40) | (static_cast<uint64_t>(b.inputRate) << 32) | b.stride);
        for (const auto& a : vl.attrs) key = mix(key, (static_cast<uint64_t>(a.location) << 48) | (static_cast<uint64_t>(a.binding) << 40) | (static_cast<uint64_t>(a.format) << 16) | a.offset);
        static const uint32_t kStates[] = {7, 8, 14, 19, 20, 22, 23, 27, 52, 53, 54, 55, 56, 58, 59, 168, 171, 185, 186,
                                           187, 188, 189, 190, 191, 192, 206, 207, 208, 209};
        for (uint32_t st : kStates) key = mix(key, r[st]);
        const bool depthBias = r[195] != 0 || r[175] != 0;
        key = mix(key, depthBias ? 1 : 0);
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return it->second;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs.module;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = ps.module;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vl.bindings.size());
        vi.pVertexBindingDescriptions = vl.bindings.data();
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vl.attrs.size());
        vi.pVertexAttributeDescriptions = vl.attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = topology(prim);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = r[8] == 2 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        rs.cullMode = r[22] == 2 ? VK_CULL_MODE_FRONT_BIT : r[22] == 3 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.depthBiasEnable = depthBias ? VK_TRUE : VK_FALSE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (t.depth != nullptr) {
            ds.depthTestEnable = r[7] != 0 ? VK_TRUE : VK_FALSE;
            ds.depthWriteEnable = (r[7] != 0 && r[14] != 0) ? VK_TRUE : VK_FALSE;
            ds.depthCompareOp = cmpOp(r[23]);
            if (r[52] != 0 && t.depth->fmt.stencil) {
                ds.stencilTestEnable = VK_TRUE;
                ds.front.failOp = stencilOp(r[53]);
                ds.front.depthFailOp = stencilOp(r[54]);
                ds.front.passOp = stencilOp(r[55]);
                ds.front.compareOp = cmpOp(r[56]);
                ds.front.compareMask = r[58];
                ds.front.writeMask = r[59];
                ds.back = ds.front;
                if (r[185] != 0) {  // TWOSIDEDSTENCILMODE: CCW = caras traseras
                    ds.back.failOp = stencilOp(r[186]);
                    ds.back.depthFailOp = stencilOp(r[187]);
                    ds.back.passOp = stencilOp(r[188]);
                    ds.back.compareOp = cmpOp(r[189]);
                }
            }
        }
        VkPipelineColorBlendAttachmentState ba[4]{};
        for (uint32_t i = 0; i < t.nColor; ++i) {
            const uint32_t mask = r[i == 0 ? 168 : 189 + i];
            ba[i].colorWriteMask = mask & 0xF;
            if (r[27] != 0) {
                ba[i].blendEnable = VK_TRUE;
                uint32_t src = r[19], dst = r[20];
                if (src == 12) { src = 5; dst = 6; }
                if (src == 13) { src = 6; dst = 5; }
                ba[i].srcColorBlendFactor = blendFactor(src, false);
                ba[i].dstColorBlendFactor = blendFactor(dst, false);
                ba[i].colorBlendOp = blendOp(r[171]);
                if (r[206] != 0) {
                    ba[i].srcAlphaBlendFactor = blendFactor(r[207], true);
                    ba[i].dstAlphaBlendFactor = blendFactor(r[208], true);
                    ba[i].alphaBlendOp = blendOp(r[209]);
                } else {
                    ba[i].srcAlphaBlendFactor = blendFactor(src, true);
                    ba[i].dstAlphaBlendFactor = blendFactor(dst, true);
                    ba[i].alphaBlendOp = ba[i].colorBlendOp;
                }
            }
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = t.nColor;
        cb.pAttachments = ba;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                      VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 5;
        dy.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gi.stageCount = 2;
        gi.pStages = stages;
        gi.pVertexInputState = &vi;
        gi.pInputAssemblyState = &ia;
        gi.pViewportState = &vp;
        gi.pRasterizationState = &rs;
        gi.pMultisampleState = &ms;
        gi.pDepthStencilState = &ds;
        gi.pColorBlendState = &cb;
        gi.pDynamicState = &dy;
        gi.layout = pipeLayout;
        gi.renderPass = rp;
        VkPipeline p = VK_NULL_HANDLE;
        if (f.vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gi, nullptr, &p) != VK_SUCCESS) {
            LOGW("vkCreateGraphicsPipelines falló");
            p = VK_NULL_HANDLE;
        }
        pipelines[key] = p;
        ++stats.pipelines;
        return p;
    }

    // ----------------------------------------------------------------------- búferes
    bool bindBuffer(const Stream& st, VkBuffer& buf, VkDeviceSize& off) {
        if (st.data == nullptr || st.size == 0) return false;
        if (st.key != nullptr && !st.dynamic) {
            GpuBuffer& b = buffers[st.key];
            if (b.buffer == VK_NULL_HANDLE || b.size < st.size) {
                if (b.buffer != VK_NULL_HANDLE) return ringBuffer(st, buf, off);  // crecer: no se libera en vuelo
                if (!createBuffer(st.size,
                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  b.buffer, b.memory, nullptr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    return false;
                b.size = st.size;
            }
            if (b.gen != st.generation) {
                // Subida a mitad de pase: no romper el VkRenderPass (UE3 hace Lock/Unlock de VB/IB
                // estáticos entre draws). Este draw lee del anillo host-visible; la copia al búfer
                // device-local se hace en el siguiente bind fuera de pase (b.gen no se actualiza).
                if (passOpen && st.size <= (4ull << 20)) return ringBuffer(st, buf, off);
                RingAlloc a = ringAlloc(st.size, 16);
                if (a.ptr == nullptr) return false;
                std::memcpy(a.ptr, st.data, st.size);
                endPass();
                VkBufferCopy cp{a.offset, 0, st.size};
                f.vkCmdCopyBuffer(frames[slot].cmd, a.buffer, b.buffer, 1, &cp);
                VkBufferMemoryBarrier mb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
                mb.srcQueueFamilyIndex = mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                mb.buffer = b.buffer;
                mb.offset = 0;
                mb.size = st.size;
                f.vkCmdPipelineBarrier(frames[slot].cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                       0, 0, nullptr, 1, &mb, 0, nullptr);
                b.gen = st.generation;
            }
            buf = b.buffer;
            off = 0;
            return true;
        }
        return ringBuffer(st, buf, off);
    }
    bool ringBuffer(const Stream& st, VkBuffer& buf, VkDeviceSize& off) {
        Frame& fr = frames[slot];
        const uint64_t ck = st.key != nullptr ? mix(reinterpret_cast<uint64_t>(st.key), st.generation) : 0;
        if (ck != 0) {
            auto it = fr.bufferCache.find(ck);
            if (it != fr.bufferCache.end()) { buf = it->second.buffer; off = it->second.offset; return true; }
        }
        RingAlloc a = ringAlloc(st.size, 16);
        if (a.ptr == nullptr) return false;
        std::memcpy(a.ptr, st.data, st.size);
        if (ck != 0) fr.bufferCache[ck] = a;
        buf = a.buffer;
        off = a.offset;
        return true;
    }

    // Bloque uniforme de una etapa: {c[256], i[16], b[16], alphaTest}.
    bool uniforms(const DrawState& s, bool vs, const d3d9sh::ShaderInfo& info, VkDescriptorBufferInfo& out) {
        Frame& fr = frames[slot];
        const uint64_t gen = vs ? s.vsConstGen : s.psConstGen;
        const uint64_t atKey = (!vs && s.renderState != nullptr && s.renderState[15] != 0)
                                   ? (1ull | (static_cast<uint64_t>(s.renderState[25] & 0xFFu) << 1) |
                                      (static_cast<uint64_t>(s.renderState[24] & 0xFFu) << 9))
                                   : 0ull;
        const uint64_t uboKey = (gen != 0 && !info.relativeConst)
                                    ? (vs ? (gen << 1) : mix((gen << 1) | 1ull, atKey))
                                    : 0ull;
        if (uboKey != 0) {
            if (auto it = fr.uboCache.find(uboKey); it != fr.uboCache.end()) {
                out = it->second;
                return true;
            }
        }
        RingAlloc a = ringAlloc(kUboSize, uboAlign);
        if (a.ptr == nullptr) return false;
        const float* fsrc = vs ? s.vsF : s.psF;
        const uint32_t maxRegs = info.relativeConst ? (vs ? 256u : 224u)
                                                    : std::min<uint32_t>(info.maxFloatConst, vs ? 256u : 224u);
        if (maxRegs > 0) {
            const size_t fbytes = static_cast<size_t>(maxRegs) * 16u;
            if (fsrc != nullptr) {
                std::memcpy(a.ptr, fsrc, fbytes);
            } else {
                std::memset(a.ptr, 0, fbytes);
            }
        }
        if (info.relativeConst)
            for (const auto& d : info.floatDefs)
                if (d.reg < 256) std::memcpy(a.ptr + d.reg * 16, d.v, 16);
        if (info.usesIntConst) {
            const int32_t* isrc = vs ? s.vsI : s.psI;
            if (isrc != nullptr) std::memcpy(a.ptr + 4096, isrc, 256);
            else std::memset(a.ptr + 4096, 0, 256);
        }
        if (info.usesBoolConst) {
            const uint32_t* bsrc = vs ? s.vsB : s.psB;
            if (bsrc != nullptr) {
                for (int i = 0; i < 16; ++i) {
                    const uint32_t row[4] = {bsrc[i], 0, 0, 0};
                    std::memcpy(a.ptr + 4096 + 256 + 16 * i, row, 16);
                }
            } else {
                std::memset(a.ptr + 4096 + 256, 0, 256);
            }
        }
        if (!vs) {
            const float at[4] = {s.renderState[15] != 0 ? static_cast<float>(s.renderState[25]) : 8.0f,
                                 static_cast<float>(s.renderState[24] & 0xFF) / 255.0f, 0, 0};
            std::memcpy(a.ptr + 4096 + 512, at, 16);
        }
        out = {a.buffer, a.offset, kUboSize};
        if (uboKey != 0) fr.uboCache[uboKey] = out;
        return true;
    }

    // ----------------------------------------------------------------------- draw
    void setViewportScissor(const DrawState& s, const Targets& t) {
        Frame& fr = frames[slot];
        VkCommandBuffer cmd = fr.cmd;
        float x = static_cast<float>(s.viewport[0]), y = static_cast<float>(s.viewport[1]);
        float w = static_cast<float>(s.viewport[2]), h = static_cast<float>(s.viewport[3]);
        if (w <= 0 || h <= 0) { x = 0; y = 0; w = static_cast<float>(t.w); h = static_cast<float>(t.h); }
        // Eje Y de D3D (arriba) con altura negativa y centro de píxel de D3D9 (+0.5).
        VkViewport vp{x + 0.5f, y + h + 0.5f, w, -h, s.minZ, s.maxZ};
        if (!fr.hasDynState || std::memcmp(&vp, &fr.lastVp, sizeof(vp)) != 0) {
            f.vkCmdSetViewport(cmd, 0, 1, &vp);
            fr.lastVp = vp;
        }
        int32_t l = 0, tp = 0, rr = static_cast<int32_t>(t.w), bb = static_cast<int32_t>(t.h);
        if (s.renderState[174] != 0) {
            l = std::max(l, s.scissor[0]);
            tp = std::max(tp, s.scissor[1]);
            rr = std::min(rr, s.scissor[2]);
            bb = std::min(bb, s.scissor[3]);
        }
        VkRect2D sc{{l, tp}, {static_cast<uint32_t>(std::max(0, rr - l)), static_cast<uint32_t>(std::max(0, bb - tp))}};
        if (!fr.hasDynState || std::memcmp(&sc, &fr.lastSc, sizeof(sc)) != 0) {
            f.vkCmdSetScissor(cmd, 0, 1, &sc);
            fr.lastSc = sc;
        }
    }

    bool draw(const DrawState& s, uint32_t prim, uint32_t primCount, bool indexed, int32_t baseVertex, uint32_t first,
              uint32_t instances) {
        if (!beginFrame()) return false;
        const uint32_t count = vertexCount(prim, primCount);
        if (count == 0) return true;
        ShaderEntry* vs = shader(s.vsKey, s.vsTokens, s.vsCount);
        const bool atEnable = s.renderState[15] != 0;
        ShaderEntry* ps = shader(s.psKey, s.psTokens, s.psCount, atEnable);
        if (vs == nullptr || ps == nullptr) {  // sin shaders (función fija) o no traducibles
            ++stats.failedDraws;
            return false;
        }
        if (dummy2DView == VK_NULL_HANDLE) {
            dummy2DView = view(dummy2D, VK_IMAGE_VIEW_TYPE_2D, false, 0, 1, 0, 1, true);
            dummyCubeView = view(dummyCube, VK_IMAGE_VIEW_TYPE_CUBE, false, 0, 1, 0, 1, true);
            dummy3DView = view(dummy3D, VK_IMAGE_VIEW_TYPE_3D, false, 0, 1, 0, 1, true);
        }
        // Texturas antes del pase (las subidas y barreras lo cerrarían).
        VkDescriptorImageInfo img[kSamplerBindings];
        GpuImage* boundImg[kSamplerBindings] = {};
        for (uint32_t i = 0; i < kSamplerBindings; ++i) {
            const bool vsSlot = i >= 16;
            const d3d9sh::ShaderInfo& info = vsSlot ? vs->info : ps->info;
            const uint32_t sn = vsSlot ? i - 16 : i;
            const uint32_t type = (info.samplerMask & (1u << sn)) ? info.samplerType[sn] : 2;
            const VkImageView dview = type == 3 ? dummyCubeView : type == 4 ? dummy3DView : dummy2DView;
            const VkImageViewType vt = type == 3 ? VK_IMAGE_VIEW_TYPE_CUBE : type == 4 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
            img[i] = {defaultSampler, dview, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            if (!(info.samplerMask & (1u << sn))) continue;
            GpuImage* g = image(s.textures[i], false);
            if (g == nullptr) continue;
            const bool gIsCube = g->cube, gIs3D = g->is3D;
            if ((type == 3) != gIsCube || (type == 4) != gIs3D) continue;
            const uint32_t* ss = s.samplerState[i];
            barrier(*g, readLayout(*g));
            const VkImageView v = view(*g, vt, ss[11] != 0, 0, g->mips, 0, g->layers, true);
            if (v == VK_NULL_HANDLE) continue;
            img[i] = {sampler(ss), v, readLayout(*g)};
            boundImg[i] = g;
            if (g->fmt.depth) g->everSampled = true;
        }
        VertexLayout& vl = tmpVl_;
        vertexLayout(s, vs->info, vl);
        Targets t;
        if (!resolveTargets(s, t)) { ++stats.failedDraws; return false; }
        // Las imágenes muestreadas no pueden ser a la vez destino.
        for (uint32_t i = 0; i < t.nColor; ++i)
            for (uint32_t j = 0; j < kSamplerBindings; ++j)
                if (boundImg[j] == t.color[i])
                    img[j] = {defaultSampler, dummy2DView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // Búferes (antes del pase no hace falta: son copias de CPU).
        VkBuffer vbuf[17] = {};
        VkDeviceSize voff[17] = {};
        for (const auto& b : vl.bindings) {
            if (b.binding == 16) { vbuf[16] = zeroBuffer; voff[16] = 0; continue; }
            Stream st = s.streams[b.binding];
            if (!bindBuffer(st, vbuf[b.binding], voff[b.binding])) { ++stats.failedDraws; return false; }
            voff[b.binding] += st.offset;
        }
        VkBuffer ibuf = VK_NULL_HANDLE;
        VkDeviceSize ioff = 0;
        if (indexed && !bindBuffer(s.indices, ibuf, ioff)) { ++stats.failedDraws; return false; }
        VkDescriptorBufferInfo ubo[2];
        if (!uniforms(s, true, vs->info, ubo[0]) || !uniforms(s, false, ps->info, ubo[1])) return false;

        const bool fullOverwrite = !passOpen && t.nColor == 1 && t.depth == nullptr &&
                                   s.renderState[27] == 0 && s.renderState[15] == 0 && s.renderState[174] == 0 &&
                                   (s.renderState[168] & 0xFu) == 0xFu &&
                                   !indexed && (prim == 4 || prim == 5) && primCount <= 2 &&
                                   (s.viewport[2] == 0 || (s.viewport[0] == 0 && s.viewport[1] == 0 &&
                                                           s.viewport[2] >= t.w && s.viewport[3] >= t.h));
        const VkRenderPass rp = beginPass(t, fullOverwrite ? 1u : 0u);
        if (rp == VK_NULL_HANDLE) { ++stats.failedDraws; return false; }
        const VkPipeline p = pipeline(s, *vs, *ps, vl, prim, t, rp);
        if (p == VK_NULL_HANDLE) { ++stats.failedDraws; return false; }
        Frame& fr = frames[slot];
        DescKey dk;
        dk.b0 = reinterpret_cast<uint64_t>(ubo[0].buffer);
        dk.b1 = reinterpret_cast<uint64_t>(ubo[1].buffer);
        for (uint32_t i = 0; i < kSamplerBindings; ++i) {
            dk.imgs[i * 2] = reinterpret_cast<uint64_t>(img[i].imageView);
            dk.imgs[i * 2 + 1] = reinterpret_cast<uint64_t>(img[i].sampler);
        }
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (auto it = fr.descCache.find(dk); it != fr.descCache.end()) {
            set = it->second;
        } else {
            set = allocSet(setLayout);
            if (set == VK_NULL_HANDLE) return false;
            VkDescriptorBufferInfo dynBase[2] = {
                {ubo[0].buffer, 0, kUboSize},
                {ubo[1].buffer, 0, kUboSize},
            };
            VkWriteDescriptorSet w[2 + kSamplerBindings]{};
            for (uint32_t i = 0; i < 2; ++i) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet = set;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                w[i].pBufferInfo = &dynBase[i];
            }
            for (uint32_t i = 0; i < kSamplerBindings; ++i) {
                VkWriteDescriptorSet& x = w[2 + i];
                x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                x.dstSet = set;
                x.dstBinding = 2 + i;
                x.descriptorCount = 1;
                x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                x.pImageInfo = &img[i];
            }
            f.vkUpdateDescriptorSets(ctx.device, 2 + kSamplerBindings, w, 0, nullptr);
            fr.descCache.emplace(dk, set);
        }
        VkCommandBuffer cmd = fr.cmd;
        if (fr.lastPipeline != p) {
            f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
            fr.lastPipeline = p;
        }
        const uint32_t dynOff[2] = {static_cast<uint32_t>(ubo[0].offset), static_cast<uint32_t>(ubo[1].offset)};
        if (fr.lastSet != set || fr.lastDyn0 != dynOff[0] || fr.lastDyn1 != dynOff[1]) {
            f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &set, 2, dynOff);
            fr.lastSet = set;
            fr.lastDyn0 = dynOff[0];
            fr.lastDyn1 = dynOff[1];
        }
        setViewportScissor(s, t);
        const uint32_t bf = s.renderState[193];
        if (!fr.hasDynState || fr.lastBlend != bf) {
            const float blend[4] = {((bf >> 16) & 0xFF) / 255.0f, ((bf >> 8) & 0xFF) / 255.0f, (bf & 0xFF) / 255.0f, (bf >> 24) / 255.0f};
            f.vkCmdSetBlendConstants(cmd, blend);
            fr.lastBlend = bf;
        }
        const uint32_t stRef = s.renderState[57] & 0xFF;
        if (!fr.hasDynState || fr.lastStencil != stRef) {
            f.vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stRef);
            fr.lastStencil = stRef;
        }
        const uint32_t b195 = s.renderState[195], s175 = s.renderState[175];
        const float units = (t.depth != nullptr && t.depth->format == VK_FORMAT_D16_UNORM) ? 65535.0f : 16777215.0f;
        if (!fr.hasDynState || fr.lastBias195 != b195 || fr.lastSlope175 != s175 || fr.lastUnits != units) {
            float bias = 0.0f, slope = 0.0f;
            std::memcpy(&bias, &b195, 4);
            std::memcpy(&slope, &s175, 4);
            f.vkCmdSetDepthBias(cmd, bias * units, 0.0f, slope);
            fr.lastBias195 = b195;
            fr.lastSlope175 = s175;
            fr.lastUnits = units;
        }
        fr.hasDynState = true;
        for (const auto& b : vl.bindings) {
            if (fr.lastVbuf[b.binding] != vbuf[b.binding] || fr.lastVoff[b.binding] != voff[b.binding]) {
                f.vkCmdBindVertexBuffers(cmd, b.binding, 1, &vbuf[b.binding], &voff[b.binding]);
                fr.lastVbuf[b.binding] = vbuf[b.binding];
                fr.lastVoff[b.binding] = voff[b.binding];
            }
        }
        if (indexed) {
            const VkIndexType itype = s.indexFormat == 102 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
            if (fr.lastIbuf != ibuf || fr.lastIoff != ioff || fr.lastItype != itype) {
                f.vkCmdBindIndexBuffer(cmd, ibuf, ioff, itype);
                fr.lastIbuf = ibuf;
                fr.lastIoff = ioff;
                fr.lastItype = itype;
            }
            f.vkCmdDrawIndexed(cmd, count, std::max(1u, instances), first, baseVertex, 0);
        } else {
            f.vkCmdDraw(cmd, count, std::max(1u, instances), first, 0);
        }
        ++stats.draws;
        return true;
    }

    void clear(const DrawState& s, uint32_t rectCount, const int32_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) {
        if (!beginFrame()) return;
        Targets t;
        if (!resolveTargets(s, t)) return;
        const bool fullTarget = (rectCount == 0 || rects == nullptr) && s.renderState[174] == 0 &&
                                (s.viewport[2] == 0 || (s.viewport[0] == 0 && s.viewport[1] == 0 &&
                                                        s.viewport[2] >= t.w && s.viewport[3] >= t.h));
        if (!passOpen && fullTarget) {
            if (flags & 1) {
                VkClearColorValue ccv{};
                ccv.float32[0] = ((color >> 16) & 0xFF) / 255.0f;
                ccv.float32[1] = ((color >> 8) & 0xFF) / 255.0f;
                ccv.float32[2] = (color & 0xFF) / 255.0f;
                ccv.float32[3] = (color >> 24) / 255.0f;
                for (uint32_t i = 0; i < t.nColor; ++i) {
                    if (s.targets[i].level == 0 && s.targets[i].face == 0) {
                        t.color[i]->pendingColorClear = true;
                        t.color[i]->pendingColor = ccv;
                        t.color[i]->pendingMip = 0;
                        t.color[i]->pendingLayer = 0;
                    } else {
                        barrier(*t.color[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, s.targets[i].level, 1, s.targets[i].face, 1};
                        f.vkCmdClearColorImage(frames[slot].cmd, t.color[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &ccv, 1, &rng);
                    }
                }
            }
            if (t.depth != nullptr && (flags & 6)) {
                VkImageAspectFlags asp = 0;
                if (flags & 2) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if ((flags & 4) && t.depth->fmt.stencil) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (asp != 0) {
                    const VkClearDepthStencilValue dsv{std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
                    if (s.depth.level == 0 && s.depth.face == 0) {
                        t.depth->pendingDepthClear = true;
                        t.depth->pendingDepthAsp |= asp;
                        t.depth->pendingDepth = dsv;
                        t.depth->pendingMip = 0;
                        t.depth->pendingLayer = 0;
                    } else {
                        barrier(*t.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                        VkImageSubresourceRange rng{asp, s.depth.level, 1, s.depth.face, 1};
                        f.vkCmdClearDepthStencilImage(frames[slot].cmd, t.depth->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dsv, 1, &rng);
                    }
                }
            }
            ++stats.clears;
            return;
        }
        if (beginPass(t) == VK_NULL_HANDLE) return;
        std::vector<VkClearAttachment> att;
        if (flags & 1) {
            VkClearValue cv{};
            cv.color.float32[0] = ((color >> 16) & 0xFF) / 255.0f;
            cv.color.float32[1] = ((color >> 8) & 0xFF) / 255.0f;
            cv.color.float32[2] = (color & 0xFF) / 255.0f;
            cv.color.float32[3] = (color >> 24) / 255.0f;
            for (uint32_t i = 0; i < t.nColor; ++i) att.push_back({VK_IMAGE_ASPECT_COLOR_BIT, i, cv});
        }
        if (t.depth != nullptr && (flags & 6)) {
            VkClearValue cv{};
            cv.depthStencil = {std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
            VkImageAspectFlags asp = 0;
            if (flags & 2) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if ((flags & 4) && t.depth->fmt.stencil) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
            if (asp != 0) att.push_back({asp, 0, cv});
        }
        if (att.empty()) return;
        // Área: viewport (y tijera si está activa) intersecada con cada rect.
        int32_t vl = static_cast<int32_t>(s.viewport[0]), vt = static_cast<int32_t>(s.viewport[1]);
        int32_t vr = vl + static_cast<int32_t>(s.viewport[2]), vb = vt + static_cast<int32_t>(s.viewport[3]);
        if (s.viewport[2] == 0) { vl = 0; vt = 0; vr = static_cast<int32_t>(t.w); vb = static_cast<int32_t>(t.h); }
        if (s.renderState[174] != 0) {
            vl = std::max(vl, s.scissor[0]); vt = std::max(vt, s.scissor[1]);
            vr = std::min(vr, s.scissor[2]); vb = std::min(vb, s.scissor[3]);
        }
        vl = std::max(vl, 0); vt = std::max(vt, 0);
        vr = std::min(vr, static_cast<int32_t>(t.w)); vb = std::min(vb, static_cast<int32_t>(t.h));
        std::vector<VkClearRect> cr;
        auto add = [&](int32_t l, int32_t tp, int32_t r, int32_t b) {
            l = std::max(l, vl); tp = std::max(tp, vt); r = std::min(r, vr); b = std::min(b, vb);
            if (r > l && b > tp) cr.push_back({{{l, tp}, {static_cast<uint32_t>(r - l), static_cast<uint32_t>(b - tp)}}, 0, 1});
        };
        if (rectCount == 0 || rects == nullptr) add(vl, vt, vr, vb);
        else for (uint32_t i = 0; i < rectCount; ++i) add(rects[4 * i], rects[4 * i + 1], rects[4 * i + 2], rects[4 * i + 3]);
        if (cr.empty()) return;
        f.vkCmdClearAttachments(frames[slot].cmd, static_cast<uint32_t>(att.size()), att.data(),
                                static_cast<uint32_t>(cr.size()), cr.data());
        ++stats.clears;
    }

    bool stretchRect(const TexRef& src, const int32_t* sr, const TexRef& dst, const int32_t* dr, uint32_t filter) {
        if (!beginFrame()) return false;
        GpuImage* s = image(src, false);
        GpuImage* d = image(dst, true);
        if (s == nullptr || d == nullptr || s == d) return false;
        if (s->fmt.depth != d->fmt.depth || s->fmt.blockW != 1 || d->fmt.blockW != 1) return false;
        endPass();
        barrier(*s, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        barrier(*d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const uint32_t sw = std::max(1u, s->width >> src.level), sh = std::max(1u, s->height >> src.level);
        const uint32_t dw = std::max(1u, d->width >> dst.level), dh = std::max(1u, d->height >> dst.level);
        VkImageBlit b{};
        b.srcSubresource = {s->aspect, src.level, src.face, 1};
        b.dstSubresource = {d->aspect, dst.level, dst.face, 1};
        b.srcOffsets[0] = {sr ? sr[0] : 0, sr ? sr[1] : 0, 0};
        b.srcOffsets[1] = {sr ? sr[2] : static_cast<int32_t>(sw), sr ? sr[3] : static_cast<int32_t>(sh), 1};
        b.dstOffsets[0] = {dr ? dr[0] : 0, dr ? dr[1] : 0, 0};
        b.dstOffsets[1] = {dr ? dr[2] : static_cast<int32_t>(dw), dr ? dr[3] : static_cast<int32_t>(dh), 1};
        const VkFilter fl = (filter >= 2 && !s->fmt.depth) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        f.vkCmdBlitImage(frames[slot].cmd, s->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d->image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, fl);
        return true;
    }

    void colorFill(const TexRef& dst, const int32_t* rect, uint32_t color) {
        DrawState s;
        static uint32_t rs[256] = {};
        s.renderState = rs;
        s.targets[0] = dst;
        s.viewport[2] = 0;
        clear(s, rect ? 1u : 0u, rect, 1, color, 1.0f, 0);
    }

    // ----------------------------------------------------------------------- presentación
    bool ensurePresentPipeline(VkFormat swapFormat) {
        if (presentPipeline != VK_NULL_HANDLE && presentFormat == swapFormat) return true;
        presentFormat = swapFormat;
        VkAttachmentDescription a{};
        a.format = swapFormat;
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &ref;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = 1;
        ri.pAttachments = &a;
        ri.subpassCount = 1;
        ri.pSubpasses = &sp;
        ri.dependencyCount = 1;
        ri.pDependencies = &dep;
        if (f.vkCreateRenderPass(ctx.device, &ri, nullptr, &presentPass) != VK_SUCCESS) return false;
        if (presentSetLayout == VK_NULL_HANDLE) {
            VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            li.bindingCount = 1;
            li.pBindings = &b;
            if (f.vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &presentSetLayout) != VK_SUCCESS) return false;
            VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
            VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pi.setLayoutCount = 1;
            pi.pSetLayouts = &presentSetLayout;
            pi.pushConstantRangeCount = 1;
            pi.pPushConstantRanges = &pc;
            if (f.vkCreatePipelineLayout(ctx.device, &pi, nullptr, &presentLayout) != VK_SUCCESS) return false;
        }
        const std::vector<uint32_t> vsCode(std::begin(gpu::kPresentVertSpv), std::end(gpu::kPresentVertSpv));
        const std::vector<uint32_t> fsCode(std::begin(gpu::kPresentFragSpv), std::end(gpu::kPresentFragSpv));
        VkShaderModule vsm = makeModule(vsCode), fsm = makeModule(fsCode);
        if (!vsm || !fsm) return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vsm;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fsm;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &ba;
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2;
        ds.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gi.stageCount = 2;
        gi.pStages = stages;
        gi.pVertexInputState = &vi;
        gi.pInputAssemblyState = &ia;
        gi.pViewportState = &vp;
        gi.pRasterizationState = &rs;
        gi.pMultisampleState = &ms;
        gi.pColorBlendState = &cb;
        gi.pDynamicState = &ds;
        gi.layout = presentLayout;
        gi.renderPass = presentPass;
        const bool okp = f.vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gi, nullptr, &presentPipeline) == VK_SUCCESS;
        if (okp && sgsrPipeline == VK_NULL_HANDLE) {
            const VkPushConstantRange ranges[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, 32},
                                                   {VK_SHADER_STAGE_FRAGMENT_BIT, 32, 32}};
            VkPipelineLayoutCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            si.setLayoutCount = 1;
            si.pSetLayouts = &presentSetLayout;
            si.pushConstantRangeCount = 2;
            si.pPushConstantRanges = ranges;
            const std::vector<uint32_t> sgCode(std::begin(gpu::kPresentSgsrFragSpv), std::end(gpu::kPresentSgsrFragSpv));
            VkShaderModule sgm = makeModule(sgCode);
            if (sgm != VK_NULL_HANDLE && f.vkCreatePipelineLayout(ctx.device, &si, nullptr, &sgsrLayout) == VK_SUCCESS) {
                stages[1].module = sgm;
                gi.layout = sgsrLayout;
                if (f.vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gi, nullptr, &sgsrPipeline) != VK_SUCCESS) {
                    sgsrPipeline = VK_NULL_HANDLE;
                }
            }
            if (sgm != VK_NULL_HANDLE) f.vkDestroyShaderModule(ctx.device, sgm, nullptr);
        }
        f.vkDestroyShaderModule(ctx.device, vsm, nullptr);
        f.vkDestroyShaderModule(ctx.device, fsm, nullptr);
        if (!okp) presentPipeline = VK_NULL_HANDLE;
        return okp;
    }

    bool present(const TexRef& bb) {
        timespec tp0{}, tp1{};
        clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (!beginFrame()) return false;
        GpuImage* g = image(bb, true);
        Frame& fr = frames[slot];
        bool presented = false;
        endPass();
        if (g != nullptr) {
            barrier(*g, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            VkImageView fgView = VK_NULL_HANDLE;
            int gameFps = 0;
            bool repeat = false;
            if (gpu::HyperFrameGen::enabled()) {
                gameFps = pacing::effectiveCap();
                fgView = gpu::HyperFrameGen::shared().record(fr.cmd, g->image, g->format, g->width, g->height, gameFps);
            } else if (const int cap = pacing::effectiveCap();
                       !pacing::thermalRelaxed() &&
                       gpu::HyperFrameGen::repeatWanted(cap, static_cast<int>(::hyperplay::win32::refreshHz()))) {
                gameFps = cap;
                repeat = gpu::HyperFrameGen::shared().recordRepeat(fr.cmd, g->image, g->format, g->width, g->height, cap);
            } else {
                gpu::HyperFrameGen::shared().releaseIfIdle();
            }
            DirectVulkanBridge::SwapTarget target;
            if (DirectVulkanBridge::shared().beginPresent(target)) {
                bool recorded = false;
                if (ensurePresentPipeline(target.format)) {
                    if (target.generation != presentGeneration) {
                        for (auto& [v, fb] : presentFbs) f.vkDestroyFramebuffer(ctx.device, fb, nullptr);
                        presentFbs.clear();
                        presentGeneration = target.generation;
                    }
                    VkFramebuffer& fb = presentFbs[target.view];
                    if (fb == VK_NULL_HANDLE) {
                        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
                        fi.renderPass = presentPass;
                        fi.attachmentCount = 1;
                        fi.pAttachments = &target.view;
                        fi.width = target.extent.width;
                        fi.height = target.extent.height;
                        fi.layers = 1;
                        f.vkCreateFramebuffer(ctx.device, &fi, nullptr, &fb);
                    }
                    VkDescriptorSet set = allocSet(presentSetLayout);
                    VkImageView srcView = fgView != VK_NULL_HANDLE ? fgView : view(*g, VK_IMAGE_VIEW_TYPE_2D, false, 0, 1, 0, 1, false);
                    if (fb != VK_NULL_HANDLE && set != VK_NULL_HANDLE && srcView != VK_NULL_HANDLE) {
                        VkDescriptorImageInfo info{presentSampler, srcView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        w.dstSet = set;
                        w.dstBinding = 0;
                        w.descriptorCount = 1;
                        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        w.pImageInfo = &info;
                        f.vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
                        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                        bi.renderPass = presentPass;
                        bi.framebuffer = fb;
                        bi.renderArea = {{0, 0}, target.extent};
                        f.vkCmdBeginRenderPass(fr.cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
                        const bool sgsr = sgsrPipeline != VK_NULL_HANDLE && fgView == VK_NULL_HANDLE &&
                                          gpu::presentopt::useSgsr(g->width, g->height, target.extent.width, target.extent.height);
                        const VkPipelineLayout pLayout = sgsr ? sgsrLayout : presentLayout;
                        f.vkCmdBindPipeline(fr.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sgsr ? sgsrPipeline : presentPipeline);
                        f.vkCmdBindDescriptorSets(fr.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pLayout, 0, 1, &set, 0, nullptr);
                        if (sgsr) {
                            const float w = static_cast<float>(g->width), h = static_cast<float>(g->height);
                            const float fp[8] = {1.0f / w, 1.0f / h, w, h,
                                                 gpu::presentopt::g_sharpness.load(std::memory_order_relaxed), 0, 0, 0};
                            f.vkCmdPushConstants(fr.cmd, pLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 32, 32, fp);
                        }
                        VkViewport vp{0, 0, static_cast<float>(target.extent.width), static_cast<float>(target.extent.height), 0, 1};
                        VkRect2D sc{{0, 0}, target.extent};
                        f.vkCmdSetViewport(fr.cmd, 0, 1, &vp);
                        f.vkCmdSetScissor(fr.cmd, 0, 1, &sc);
                        float pc[8] = {1, 0, 0, 1, 0, 0, 0, 0};
                        switch (target.transform) {
                            case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: { const float m[8] = {0, 1, -1, 0, 0, 1, 0, 0}; std::memcpy(pc, m, 32); break; }
                            case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: { const float m[8] = {-1, 0, 0, -1, 1, 1, 0, 0}; std::memcpy(pc, m, 32); break; }
                            case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: { const float m[8] = {0, -1, 1, 0, 1, 0, 0, 0}; std::memcpy(pc, m, 32); break; }
                            default: break;
                        }
                        gpu::presentopt::applyFit(pc, g->width, g->height, target.extent.width, target.extent.height);
                        f.vkCmdPushConstants(fr.cmd, pLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pc);
                        f.vkCmdDraw(fr.cmd, 3, 1, 0, 0);
                        f.vkCmdEndRenderPass(fr.cmd);
                        recorded = true;
                    }
                }
                if (!recorded) {
                    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.image = target.image;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    f.vkCmdPipelineBarrier(fr.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                                           0, nullptr, 0, nullptr, 1, &b);
                }
                f.vkEndCommandBuffer(fr.cmd);
                presented = DirectVulkanBridge::shared().endPresent(target, fr.cmd, fr.fence);
                recording = false;
                if ((fgView != VK_NULL_HANDLE || repeat) && presented && gameFps > 0) {
                    timespec t{};
                    clock_gettime(CLOCK_MONOTONIC, &t);
                    const int64_t now = static_cast<int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
                    gpu::HyperFrameGen::shared().scheduleReal(now + 500000000LL / gameFps);
                }
            }
        }
        if (recording) {
            f.vkEndCommandBuffer(fr.cmd);
            DirectVulkanBridge::shared().submit(fr.cmd, fr.fence);
            recording = false;
        }
        slot = (slot + 1) % frames.size();
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        const int64_t nowNs = static_cast<int64_t>(tp1.tv_sec) * 1000000000LL + tp1.tv_nsec;
        const int64_t p0Ns = static_cast<int64_t>(tp0.tv_sec) * 1000000000LL + tp0.tv_nsec;
        statPresentNs += static_cast<uint64_t>(nowNs - p0Ns);
        if (lastPresentNs != 0 && nowNs - lastPresentNs < 500000000LL) {
            statFrameNs += static_cast<uint64_t>(nowNs - lastPresentNs);
        }
        lastPresentNs = nowNs;
        ++stats.presents;
        if (stats.presents % 600 == 1) {
            LOGI("D3D9 GPU: presents=%llu draws=%llu fallidos=%llu clears=%llu pipelines=%llu subidas=%llu passes=%.1f(load=%.1f,depth=%.1f)/f (frame=%.1fms draw=%.1fms worker=%.1fms fence=%.1fms pres=%.1fms)",
                 static_cast<unsigned long long>(stats.presents), static_cast<unsigned long long>(stats.draws),
                 static_cast<unsigned long long>(stats.failedDraws), static_cast<unsigned long long>(stats.clears),
                 static_cast<unsigned long long>(stats.pipelines), static_cast<unsigned long long>(stats.uploads),
                 static_cast<double>(statPasses) / 600.0,
                 static_cast<double>(statLoadPasses) / 600.0,
                 static_cast<double>(statDepthPasses) / 600.0,
                 static_cast<double>(statFrameNs) / 600000000.0,
                 static_cast<double>(statDrawNs) / 600000000.0,
                 static_cast<double>(statWorkerNs) / 600000000.0,
                 static_cast<double>(statFenceNs) / 600000000.0,
                 static_cast<double>(statPresentNs) / 600000000.0);
            statFenceNs = statPresentNs = statFrameNs = statDrawNs = statWorkerNs = 0;
            statPasses = statLoadPasses = statDepthPasses = 0;
        }
        return presented;
    }

    void release(const void* key) {
        auto it = images.find(key);
        if (it == images.end()) return;
        if (recording) return;  // el frame actual podría referenciarla: se mantiene
        for (Frame& fr : frames) f.vkWaitForFences(ctx.device, 1, &fr.fence, VK_TRUE, 1000000000ull);
        if (it->second) destroyImage(*it->second);
        images.erase(it);
    }
};

Impl& impl() {
    static Impl* i = new Impl();
    return *i;
}

bool asyncEnabled() {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return !(__system_property_get("debug.hyperplay.d3d9async", v) > 0 && v[0] == '0');
    }();
    return s_on;
}

// Cola SPSC lock-free (RenderingThread -> d3d9-worker) para desacoplar D3D9 de la grabación/envío Vulkan.
struct AsyncQueue {
    enum class Op : uint8_t {
        VsConst,
        PsConst,
        Rs,
        Ss,
        Draw,
        Clear,
        StretchRect,
        ColorFill,
        Present,
    };

    struct Cmd {
        Op op = Op::Draw;
        bool indexed = false;
        bool hasRect0 = false;
        bool hasRect1 = false;
        uint32_t primType = 0;
        uint32_t primCount = 0;
        int32_t baseVertex = 0;
        uint32_t first = 0;
        uint32_t instances = 1;
        uint32_t flags = 0;
        uint32_t color = 0;
        float z = 1.0f;
        uint32_t stencil = 0;
        uint32_t filter = 0;
        uint32_t rectCount = 0;
        int32_t rect0[4] = {};
        int32_t rect1[4] = {};
        const uint8_t* ptr = nullptr;
        uint64_t dataTailAfter = 0;
        DrawState state{};
        TexRef tex0{};
        TexRef tex1{};
    };

    static constexpr uint32_t kCmdCap = 4096;
    static constexpr uint32_t kCmdMask = kCmdCap - 1;
    static constexpr uint64_t kDataCap = 32ull << 20;
    static constexpr uint64_t kDataMask = kDataCap - 1;

    alignas(64) std::atomic<uint32_t> cmdHead{0};
    alignas(64) std::atomic<uint32_t> cmdTail{0};
    alignas(64) std::atomic<uint64_t> dataHead{0};
    alignas(64) std::atomic<uint64_t> dataTail{0};
    alignas(64) std::atomic<uint64_t> queuedPresents{0};
    alignas(64) std::atomic<uint64_t> completedPresents{0};

    std::mutex wakeMtx;
    std::condition_variable workCv;
    std::condition_variable spaceCv;
    std::mutex releaseMtx;
    std::vector<const void*> pendingReleases;

    // Estado exclusivo del productor (RenderingThread)
    uint32_t writeCmd = 0;
    uint64_t writeData = 0;
    uint64_t lastVsGen = 0, lastPsGen = 0, lastRsGen = 0, lastSsGen = 0;
    std::unordered_map<const void*, uint64_t> lastStreamGen;
    std::unordered_map<const void*, uint64_t> lastTexGen;

    // Estado exclusivo del consumidor (d3d9-worker)
    float workerVsF[256 * 4] = {};
    int32_t workerVsI[16 * 4] = {};
    uint32_t workerVsB[16] = {};
    float workerPsF[224 * 4] = {};
    int32_t workerPsI[16 * 4] = {};
    uint32_t workerPsB[16] = {};
    uint32_t workerRs[256] = {};
    uint32_t workerSs[20][14] = {};

    std::vector<Cmd> cmds;
    std::vector<uint8_t> dataBuf;
    std::thread worker;
    std::atomic<bool> stop{false};
    std::atomic<bool> ready{false};
    bool initOk = false;
    bool started = false;
    std::mutex initMtx;

    ~AsyncQueue() {
        stop.store(true, std::memory_order_release);
        workCv.notify_all();
        if (worker.joinable()) worker.join();
    }

    bool ensureStarted() {
        if (ready.load(std::memory_order_acquire)) return initOk;
        std::lock_guard<std::mutex> lk(initMtx);
        if (!started) {
            cmds.resize(kCmdCap);
            dataBuf.resize(kDataCap);
            started = true;
            worker = std::thread([this] { workerLoop(); });
        }
        std::unique_lock<std::mutex> wlk(wakeMtx);
        spaceCv.wait(wlk, [this] { return ready.load(std::memory_order_acquire); });
        return initOk;
    }

    uint8_t* allocData(size_t bytes) {
        if (bytes == 0) return nullptr;
        const uint64_t sz = (static_cast<uint64_t>(bytes) + 15ull) & ~15ull;
        if (sz > kDataCap / 4) return nullptr;
        const uint64_t off = writeData & kDataMask;
        const uint64_t gap = (off + sz > kDataCap) ? (kDataCap - off) : 0ull;
        const uint64_t need = gap + sz;
        for (int spin = 0; spin < 64; ++spin) {
            if (kDataCap - (writeData - dataTail.load(std::memory_order_acquire)) >= need) break;
            workCv.notify_one();
            sched_yield();
        }
        if (kDataCap - (writeData - dataTail.load(std::memory_order_acquire)) < need) {
            workCv.notify_one();
            std::unique_lock<std::mutex> lk(wakeMtx);
            spaceCv.wait(lk, [this, need] {
                return kDataCap - (writeData - dataTail.load(std::memory_order_acquire)) >= need ||
                       stop.load(std::memory_order_acquire);
            });
        }
        const uint64_t start = writeData + gap;
        writeData = start + sz;
        return dataBuf.data() + (start & kDataMask);
    }

    void pushCmd(Cmd& c, bool wakeNow = false) {
        for (int spin = 0; spin < 64; ++spin) {
            if (writeCmd - cmdTail.load(std::memory_order_acquire) < kCmdCap) break;
            workCv.notify_one();
            sched_yield();
        }
        if (writeCmd - cmdTail.load(std::memory_order_acquire) >= kCmdCap) {
            workCv.notify_one();
            std::unique_lock<std::mutex> lk(wakeMtx);
            spaceCv.wait(lk, [this] {
                return writeCmd - cmdTail.load(std::memory_order_acquire) < kCmdCap ||
                       stop.load(std::memory_order_acquire);
            });
        }
        c.dataTailAfter = writeData;
        cmds[writeCmd & kCmdMask] = c;
        dataHead.store(writeData, std::memory_order_release);
        ++writeCmd;
        cmdHead.store(writeCmd, std::memory_order_release);
        if (wakeNow || (writeCmd & 31u) == 0) {
            workCv.notify_one();
        }
    }

    void sync() {
        if (!ready.load(std::memory_order_acquire)) return;
        const uint32_t targetCmd = writeCmd;
        const uint64_t targetPres = queuedPresents.load(std::memory_order_relaxed);
        workCv.notify_one();
        std::unique_lock<std::mutex> lk(wakeMtx);
        spaceCv.wait(lk, [this, targetCmd, targetPres] {
            return (cmdTail.load(std::memory_order_acquire) == targetCmd &&
                    completedPresents.load(std::memory_order_acquire) == targetPres) ||
                   stop.load(std::memory_order_acquire);
        });
    }

    void enqueueStateDeltas(const DrawState& s) {
        if (s.vsF != nullptr && s.vsConstGen != lastVsGen) {
            constexpr size_t kVsF = 256 * 4 * sizeof(float);
            constexpr size_t kVsI = 16 * 4 * sizeof(int32_t);
            constexpr size_t kVsB = 16 * sizeof(uint32_t);
            uint8_t* p = allocData(kVsF + kVsI + kVsB);
            if (p != nullptr) {
                std::memcpy(p, s.vsF, kVsF);
                std::memcpy(p + kVsF, s.vsI, kVsI);
                std::memcpy(p + kVsF + kVsI, s.vsB, kVsB);
                Cmd c;
                c.op = Op::VsConst;
                c.ptr = p;
                pushCmd(c);
                lastVsGen = s.vsConstGen;
            }
        }
        if (s.psF != nullptr && s.psConstGen != lastPsGen) {
            constexpr size_t kPsF = 224 * 4 * sizeof(float);
            constexpr size_t kPsI = 16 * 4 * sizeof(int32_t);
            constexpr size_t kPsB = 16 * sizeof(uint32_t);
            uint8_t* p = allocData(kPsF + kPsI + kPsB);
            if (p != nullptr) {
                std::memcpy(p, s.psF, kPsF);
                std::memcpy(p + kPsF, s.psI, kPsI);
                std::memcpy(p + kPsF + kPsI, s.psB, kPsB);
                Cmd c;
                c.op = Op::PsConst;
                c.ptr = p;
                pushCmd(c);
                lastPsGen = s.psConstGen;
            }
        }
        if (s.renderState != nullptr && s.rsGen != lastRsGen) {
            constexpr size_t kRs = 256 * sizeof(uint32_t);
            uint8_t* p = allocData(kRs);
            if (p != nullptr) {
                std::memcpy(p, s.renderState, kRs);
                Cmd c;
                c.op = Op::Rs;
                c.ptr = p;
                pushCmd(c);
                lastRsGen = s.rsGen;
            }
        }
        if (s.samplerState != nullptr && s.ssGen != lastSsGen) {
            constexpr size_t kSs = 20 * 14 * sizeof(uint32_t);
            uint8_t* p = allocData(kSs);
            if (p != nullptr) {
                std::memcpy(p, s.samplerState, kSs);
                Cmd c;
                c.op = Op::Ss;
                c.ptr = p;
                pushCmd(c);
                lastSsGen = s.ssGen;
            }
        }
    }

    void snapshotStream(Stream& st) {
        if (st.data == nullptr || st.size == 0) return;
        if (st.key != nullptr) {
            auto it = lastStreamGen.find(st.key);
            if (it != lastStreamGen.end() && it->second == st.generation) return;
        }
        uint8_t* dst = allocData(st.size);
        if (dst == nullptr) {
            sync();
            return;
        }
        std::memcpy(dst, st.data, st.size);
        st.data = dst;
        if (st.key != nullptr) lastStreamGen[st.key] = st.generation;
    }

    void snapshotTex(TexRef& t) {
        if (!t.valid() || t.data == nullptr || t.subresources == 0 || t.subresourceBytes == nullptr) return;
        auto it = lastTexGen.find(t.key);
        if (it != lastTexGen.end() && it->second == t.generation) return;
        const size_t ptrArrayBytes = static_cast<size_t>(t.subresources) * sizeof(uint8_t*);
        uint8_t* ptrRaw = allocData(ptrArrayBytes);
        if (ptrRaw == nullptr) {
            sync();
            return;
        }
        auto** ptrs = reinterpret_cast<const uint8_t**>(ptrRaw);
        for (uint32_t i = 0; i < t.subresources; ++i) {
            const uint8_t* src = t.data[i];
            const uint32_t sz = t.subresourceBytes[i];
            if (src != nullptr && sz > 0) {
                uint8_t* dst = allocData(sz);
                if (dst == nullptr) {
                    sync();
                    return;
                }
                std::memcpy(dst, src, sz);
                ptrs[i] = dst;
            } else {
                ptrs[i] = nullptr;
            }
        }
        t.data = ptrs;
        t.subresourceBytes = nullptr;
        lastTexGen[t.key] = t.generation;
    }

    void bindWorkerState(DrawState& s) const {
        s.vsF = workerVsF;
        s.vsI = workerVsI;
        s.vsB = workerVsB;
        s.psF = workerPsF;
        s.psI = workerPsI;
        s.psB = workerPsB;
        s.renderState = workerRs;
        s.samplerState = workerSs;
    }

    bool enqueueDraw(const DrawState& s, uint32_t primType, uint32_t primCount, bool indexed, int32_t baseVertex,
                     uint32_t first, uint32_t instances) {
        if (!ensureStarted()) return false;
        enqueueStateDeltas(s);
        Cmd c;
        c.op = Op::Draw;
        c.indexed = indexed;
        c.primType = primType;
        c.primCount = primCount;
        c.baseVertex = baseVertex;
        c.first = first;
        c.instances = instances;
        c.state = s;
        for (int k = 0; k < 16; ++k) snapshotStream(c.state.streams[k]);
        if (indexed) snapshotStream(c.state.indices);
        for (int k = 0; k < 20; ++k) snapshotTex(c.state.textures[k]);
        pushCmd(c);
        return true;
    }

    void enqueueClear(const DrawState& s, uint32_t rectCount, const int32_t* rects, uint32_t flags, uint32_t color,
                      float z, uint32_t stencil) {
        if (!ensureStarted()) return;
        enqueueStateDeltas(s);
        Cmd c;
        c.op = Op::Clear;
        c.flags = flags;
        c.color = color;
        c.z = z;
        c.stencil = stencil;
        c.state = s;
        if (rectCount > 0 && rects != nullptr) {
            const size_t bytes = static_cast<size_t>(rectCount) * 4 * sizeof(int32_t);
            uint8_t* p = allocData(bytes);
            if (p != nullptr) {
                std::memcpy(p, rects, bytes);
                c.rectCount = rectCount;
                c.ptr = p;
            }
        }
        pushCmd(c);
    }

    bool enqueueStretchRect(const TexRef& src, const int32_t* srcRect, const TexRef& dst, const int32_t* dstRect,
                            uint32_t filter) {
        if (!ensureStarted()) return false;
        Cmd c;
        c.op = Op::StretchRect;
        c.tex0 = src;
        c.tex1 = dst;
        snapshotTex(c.tex0);
        snapshotTex(c.tex1);
        if (srcRect != nullptr) {
            c.hasRect0 = true;
            std::memcpy(c.rect0, srcRect, 16);
        }
        if (dstRect != nullptr) {
            c.hasRect1 = true;
            std::memcpy(c.rect1, dstRect, 16);
        }
        c.filter = filter;
        pushCmd(c);
        return true;
    }

    void enqueueColorFill(const TexRef& dst, const int32_t* rect, uint32_t color) {
        if (!ensureStarted()) return;
        Cmd c;
        c.op = Op::ColorFill;
        c.tex0 = dst;
        if (rect != nullptr) {
            c.hasRect0 = true;
            std::memcpy(c.rect0, rect, 16);
        }
        c.color = color;
        pushCmd(c);
    }

    bool enqueuePresent(const TexRef& bb) {
        if (!ensureStarted()) return false;
        Cmd c;
        c.op = Op::Present;
        c.tex0 = bb;
        const uint64_t q = queuedPresents.fetch_add(1, std::memory_order_relaxed) + 1;
        pushCmd(c, true);
        // Backpressure: permite como máximo 1 frame por delante para evitar latencia acumulada.
        while (q - completedPresents.load(std::memory_order_acquire) >= 2 &&
               !stop.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lk(wakeMtx);
            spaceCv.wait(lk, [this, q] {
                return (q - completedPresents.load(std::memory_order_acquire) < 2) ||
                       stop.load(std::memory_order_acquire);
            });
        }
        return true;
    }

    void enqueueRelease(const void* key) {
        if (!ready.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lk(releaseMtx);
        pendingReleases.push_back(key);
    }

    void drainReleases(Impl& i) {
        std::vector<const void*> local;
        {
            std::lock_guard<std::mutex> lk(releaseMtx);
            if (pendingReleases.empty()) return;
            local.swap(pendingReleases);
        }
        for (const void* k : local) i.release(k);
    }

    void workerLoop() {
        pthread_setname_np(pthread_self(), "d3d9-worker");
        ::hyperplay::pacing::setRenderWorkerThread(static_cast<pid_t>(gettid()));
        Impl& i = impl();
        {
            std::lock_guard<std::mutex> lk(i.mtx);
            initOk = i.init();
            ready.store(true, std::memory_order_release);
        }
        spaceCv.notify_all();
        if (!initOk) return;

        uint32_t readCmd = 0;
        uint64_t frameStartNs = 0;
        while (!stop.load(std::memory_order_acquire)) {
            uint32_t head = cmdHead.load(std::memory_order_acquire);
            if (readCmd == head) {
                for (int spin = 0; spin < 128; ++spin) {
                    head = cmdHead.load(std::memory_order_acquire);
                    if (readCmd != head) break;
                    sched_yield();
                }
                if (readCmd == head) {
                    drainReleases(i);
                    std::unique_lock<std::mutex> lk(wakeMtx);
                    workCv.wait(lk, [this, readCmd] {
                        return cmdHead.load(std::memory_order_acquire) != readCmd ||
                               stop.load(std::memory_order_acquire);
                    });
                    continue;
                }
            }

            if (frameStartNs == 0) {
                timespec ts{};
                clock_gettime(CLOCK_MONOTONIC, &ts);
                frameStartNs = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
            }

            while (readCmd != head) {
                const Cmd& c = cmds[readCmd & kCmdMask];
                switch (c.op) {
                    case Op::VsConst: {
                        constexpr size_t kVsF = 256 * 4 * sizeof(float);
                        constexpr size_t kVsI = 16 * 4 * sizeof(int32_t);
                        constexpr size_t kVsB = 16 * sizeof(uint32_t);
                        std::memcpy(workerVsF, c.ptr, kVsF);
                        std::memcpy(workerVsI, c.ptr + kVsF, kVsI);
                        std::memcpy(workerVsB, c.ptr + kVsF + kVsI, kVsB);
                        break;
                    }
                    case Op::PsConst: {
                        constexpr size_t kPsF = 224 * 4 * sizeof(float);
                        constexpr size_t kPsI = 16 * 4 * sizeof(int32_t);
                        constexpr size_t kPsB = 16 * sizeof(uint32_t);
                        std::memcpy(workerPsF, c.ptr, kPsF);
                        std::memcpy(workerPsI, c.ptr + kPsF, kPsI);
                        std::memcpy(workerPsB, c.ptr + kPsF + kPsI, kPsB);
                        break;
                    }
                    case Op::Rs:
                        std::memcpy(workerRs, c.ptr, 256 * sizeof(uint32_t));
                        break;
                    case Op::Ss:
                        std::memcpy(workerSs, c.ptr, 20 * 14 * sizeof(uint32_t));
                        break;
                    case Op::Draw: {
                        DrawState s = c.state;
                        bindWorkerState(s);
                        timespec td0{}, td1{};
                        clock_gettime(CLOCK_MONOTONIC, &td0);
                        i.draw(s, c.primType, c.primCount, c.indexed, c.baseVertex, c.first, c.instances);
                        clock_gettime(CLOCK_MONOTONIC, &td1);
                        i.statDrawNs += static_cast<uint64_t>((td1.tv_sec - td0.tv_sec) * 1000000000LL +
                                                              (td1.tv_nsec - td0.tv_nsec));
                        break;
                    }
                    case Op::Clear: {
                        DrawState s = c.state;
                        bindWorkerState(s);
                        i.clear(s, c.rectCount, reinterpret_cast<const int32_t*>(c.ptr), c.flags, c.color, c.z,
                                c.stencil);
                        break;
                    }
                    case Op::StretchRect:
                        i.stretchRect(c.tex0, c.hasRect0 ? c.rect0 : nullptr, c.tex1,
                                      c.hasRect1 ? c.rect1 : nullptr, c.filter);
                        break;
                    case Op::ColorFill:
                        i.colorFill(c.tex0, c.hasRect0 ? c.rect0 : nullptr, c.color);
                        break;
                    case Op::Present: {
                        i.present(c.tex0);
                        if (frameStartNs != 0) {
                            timespec ts{};
                            clock_gettime(CLOCK_MONOTONIC, &ts);
                            const uint64_t now =
                                static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
                            i.statWorkerNs += now - frameStartNs;
                            frameStartNs = 0;
                        }
                        drainReleases(i);
                        completedPresents.fetch_add(1, std::memory_order_release);
                        spaceCv.notify_all();
                        break;
                    }
                }
                dataTail.store(c.dataTailAfter, std::memory_order_release);
                ++readCmd;
                cmdTail.store(readCmd, std::memory_order_release);
            }
            spaceCv.notify_all();
        }
    }
};

AsyncQueue& asyncQueue() {
    static AsyncQueue* q = new AsyncQueue();
    return *q;
}

}  // namespace

Renderer& Renderer::shared() {
    static Renderer r;
    return r;
}
bool Renderer::available() {
    if (asyncEnabled()) return asyncQueue().ensureStarted();
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    return i.init();
}
bool Renderer::draw(const DrawState& s, uint32_t primType, uint32_t primCount, bool indexed, int32_t baseVertex, uint32_t first,
                    uint32_t instances) {
    if (asyncEnabled()) {
        return asyncQueue().enqueueDraw(s, primType, primCount, indexed, baseVertex, first, instances);
    }
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    if (!i.init()) return false;
    timespec td0{}, td1{};
    clock_gettime(CLOCK_MONOTONIC, &td0);
    const bool ok = i.draw(s, primType, primCount, indexed, baseVertex, first, instances);
    clock_gettime(CLOCK_MONOTONIC, &td1);
    i.statDrawNs += static_cast<uint64_t>((td1.tv_sec - td0.tv_sec) * 1000000000LL + (td1.tv_nsec - td0.tv_nsec));
    return ok;
}
void Renderer::clear(const DrawState& s, uint32_t rectCount, const int32_t* rects, uint32_t flags, uint32_t color, float z,
                     uint32_t stencil) {
    if (asyncEnabled()) {
        asyncQueue().enqueueClear(s, rectCount, rects, flags, color, z, stencil);
        return;
    }
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    if (i.init()) i.clear(s, rectCount, rects, flags, color, z, stencil);
}
bool Renderer::stretchRect(const TexRef& src, const int32_t* srcRect, const TexRef& dst, const int32_t* dstRect,
                           uint32_t filter) {
    if (asyncEnabled()) return asyncQueue().enqueueStretchRect(src, srcRect, dst, dstRect, filter);
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    return i.init() && i.stretchRect(src, srcRect, dst, dstRect, filter);
}
void Renderer::colorFill(const TexRef& dst, const int32_t* rect, uint32_t color) {
    if (asyncEnabled()) {
        asyncQueue().enqueueColorFill(dst, rect, color);
        return;
    }
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    if (i.init()) i.colorFill(dst, rect, color);
}
bool Renderer::present(const TexRef& bb) {
    if (asyncEnabled()) return asyncQueue().enqueuePresent(bb);
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    return i.init() && i.present(bb);
}
void Renderer::release(const void* key) {
    if (asyncEnabled()) {
        asyncQueue().enqueueRelease(key);
        return;
    }
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    if (i.ok) i.release(key);
}
void Renderer::sync() {
    if (asyncEnabled()) asyncQueue().sync();
}
Renderer::Stats Renderer::stats() const {
    Impl& i = impl();
    std::lock_guard<std::mutex> lk(i.mtx);
    return i.stats;
}

}  // namespace hyperplay::gpu9
