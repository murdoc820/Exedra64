#include "HyperFrameGen.h"

// Estructuras Vulkan con sType y el resto a cero (mismo criterio que D3D11GpuRenderer.cpp).
#pragma clang diagnostic ignored "-Wmissing-field-initializers"

#include <android/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <vector>

#include "DirectVulkanBridge.h"
#include "HyperFrameGenSpv.h"
#include "PresentOptions.h"
#include "PresentSgsrSpv.h"
#include "PresentSpv.h"

#define LOG_TAG "HyperPlay.HyperFG"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay::gpu {
namespace {

std::atomic<bool> g_enabled{false};

int64_t nowNs() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

void sleepUntil(int64_t at) {
    const timespec t{static_cast<time_t>(at / 1000000000LL), static_cast<long>(at % 1000000000LL)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr) == EINTR) {
    }
}

// uv del pase de presentación según la rotación del swapchain (igual que D3D11GpuRenderer).
void rotationPc(VkSurfaceTransformFlagBitsKHR transform, float pc[8]) {
    const float id[8] = {1, 0, 0, 1, 0, 0, 0, 0};
    std::memcpy(pc, id, 32);
    switch (transform) {
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: { const float m[8] = {0, 1, -1, 0, 0, 1, 0, 0}; std::memcpy(pc, m, 32); break; }
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: { const float m[8] = {-1, 0, 0, -1, 1, 1, 0, 0}; std::memcpy(pc, m, 32); break; }
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: { const float m[8] = {0, -1, 1, 0, 1, 0, 0, 0}; std::memcpy(pc, m, 32); break; }
        default: break;
    }
}

#define FG_DEVICE_FNS(X)                                                                                         \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)        \
    X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements)                    \
    X(vkBindBufferMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler)       \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)                      \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreatePipelineLayout)    \
    X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateComputePipelines)       \
    X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateRenderPass) X(vkDestroyRenderPass)                \
    X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCreateCommandPool) X(vkDestroyCommandPool)                 \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer)             \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkCmdPipelineBarrier)               \
    X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdFillBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)      \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdDraw)           \
    X(vkCmdSetViewport) X(vkCmdSetScissor)

struct Fns {
#define FG_DECL(name) PFN_##name name = nullptr;
    FG_DEVICE_FNS(FG_DECL)
#undef FG_DECL
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = nullptr;
};

struct Img {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t w = 0, h = 0;
};

}  // namespace

struct HyperFrameGen::Impl {
    FrameGenHost* host = nullptr;
    DirectVulkanBridge::GpuContext ctx{};
    Fns f;
    bool deviceReady = false;
    bool failed = false;  // sin soporte (formatos/compilación): HyperFG queda inactivo

    // Pipelines y layouts (una vez por dispositivo).
    VkSampler linear = VK_NULL_HANDLE, presentLinear = VK_NULL_HANDLE;
    VkDescriptorSetLayout pyrDsl = VK_NULL_HANDLE, flowDsl = VK_NULL_HANDLE, interpDsl = VK_NULL_HANDLE,
                          presDsl = VK_NULL_HANDLE;
    VkPipelineLayout pyrPl = VK_NULL_HANDLE, flowPl = VK_NULL_HANDLE, interpPl = VK_NULL_HANDLE;
    VkPipeline pyrPipe = VK_NULL_HANDLE, flowPipe = VK_NULL_HANDLE, interpPipe = VK_NULL_HANDLE;

    // Recursos por tamaño (al activar o al cambiar la resolución).
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t w = 0, h = 0;      // resolución completa del backbuffer (hist, interp)
    uint32_t fw = 0, fh = 0;    // resolución acotada <= 1600x900 solo para pirámide y flujo óptico
    bool sized = false;
    Img hist[2], lum[2][3], flow, interp;
    VkBuffer statsBuf = VK_NULL_HANDLE;
    VkDeviceMemory statsMem = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet pyrSet[2]{}, flowSet[2]{}, interpSet[2]{}, presSet[2]{};
    uint64_t frame = 0;
    int validFrames = 0;
    int64_t lastRecordNs = 0;
    int64_t emaIntervalNs = 0;
    bool bypass = false;

    // Presentación del frame real (hilo HyperFG).
    VkFormat swapFmt = VK_FORMAT_UNDEFINED;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkPipelineLayout presPl = VK_NULL_HANDLE, sgsrPl = VK_NULL_HANDLE;
    VkPipeline presPipe = VK_NULL_HANDLE, sgsrPipe = VK_NULL_HANDLE;
    std::map<VkImageView, VkFramebuffer> fbs;
    uint64_t fbGen = ~0ull;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmds[2]{};
    VkFence fences[2]{};
    int cmdSlot = 0;

    std::thread thread;
    std::mutex m;
    std::condition_variable cv;
    bool hasJob = false, busy = false, quit = false;
    int jobHist = 0;
    int64_t jobAt = 0;
    bool jobRepeat = false;

    std::atomic<uint64_t> sGenerated{0}, sSlow{0}, sReal{0}, sCancelled{0}, sRepeat{0};
    bool copied = false;  // el último record() copió el backbuffer al historial
    bool lastWasRepeat = false;  // el último record() fue de repetición (hilo de render)
    bool jobIsRepeat = false;    // el trabajo en curso del hilo HyperFG es una repetición

    // ------------------------------------------------------------------ utilidades
    bool loadDevice() {
        if (deviceReady) return true;
        if (failed) return false;
        if (!host->gpuContext(ctx)) return false;
#define FG_LOAD(name) f.name = reinterpret_cast<PFN_##name>(ctx.getDeviceProcAddr(ctx.device, #name));
        FG_DEVICE_FNS(FG_LOAD)
#undef FG_LOAD
        f.vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceMemoryProperties"));
        f.vkGetPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceFormatProperties"));
        if (f.vkCmdDispatch == nullptr || f.vkGetPhysicalDeviceMemoryProperties == nullptr ||
            f.vkGetPhysicalDeviceFormatProperties == nullptr) {
            return fail("funciones Vulkan no disponibles");
        }
        const VkFormat need[3] = {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM};
        for (VkFormat fm : need) {
            VkFormatProperties p{};
            f.vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, fm, &p);
            const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((p.optimalTilingFeatures & want) != want) return fail("formato sin soporte de storage image");
        }
        if (!createPipelines()) return fail("no se pudieron crear los pipelines");
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = ctx.queueFamily;
        if (f.vkCreateCommandPool(ctx.device, &pi, nullptr, &cmdPool) != VK_SUCCESS) return fail("command pool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 2;
        if (f.vkAllocateCommandBuffers(ctx.device, &ai, cmds) != VK_SUCCESS) return fail("command buffers");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (VkFence& fe : fences) {
            if (f.vkCreateFence(ctx.device, &fi, nullptr, &fe) != VK_SUCCESS) return fail("fences");
        }
        thread = std::thread([this] { presentLoop(); });
        deviceReady = true;
        LOGI("HyperFG listo (pirámide + flujo + interpolación en Vulkan Compute)");
        return true;
    }

    bool fail(const char* why) {
        LOGW("HyperFG no disponible: %s", why);
        failed = true;
        return false;
    }

    int32_t memType(uint32_t bits, VkMemoryPropertyFlags want) {
        VkPhysicalDeviceMemoryProperties mp{};
        f.vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return static_cast<int32_t>(i);
        }
        return -1;
    }

    VkShaderModule module(const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        return f.vkCreateShaderModule(ctx.device, &ci, nullptr, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
    }

    VkDescriptorSetLayout dsl(const std::vector<VkDescriptorType>& types, VkShaderStageFlags stage) {
        std::vector<VkDescriptorSetLayoutBinding> b(types.size());
        for (size_t i = 0; i < types.size(); ++i) b[i] = {static_cast<uint32_t>(i), types[i], 1, stage, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = static_cast<uint32_t>(b.size());
        ci.pBindings = b.data();
        VkDescriptorSetLayout l = VK_NULL_HANDLE;
        return f.vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &l) == VK_SUCCESS ? l : VK_NULL_HANDLE;
    }

    VkPipeline compute(const uint32_t* code, size_t bytes, VkDescriptorSetLayout set, uint32_t pcBytes,
                       VkPipelineLayout& layout) {
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &set;
        li.pushConstantRangeCount = pcBytes > 0 ? 1 : 0;
        li.pPushConstantRanges = pcBytes > 0 ? &pc : nullptr;
        if (f.vkCreatePipelineLayout(ctx.device, &li, nullptr, &layout) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkShaderModule sm = module(code, bytes);
        if (sm == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        ci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        if (f.vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
        f.vkDestroyShaderModule(ctx.device, sm, nullptr);
        return p;
    }

    bool createPipelines() {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (f.vkCreateSampler(ctx.device, &si, nullptr, &linear) != VK_SUCCESS) return false;
        // Presentación: fuera de la imagen del juego, negro (bandas sin deformar, ver applyFit).
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        if (f.vkCreateSampler(ctx.device, &si, nullptr, &presentLinear) != VK_SUCCESS) return false;
        const auto S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                   B = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pyrDsl = dsl({S, I, I, I, I, B}, VK_SHADER_STAGE_COMPUTE_BIT);
        flowDsl = dsl({S, S, S, S, S, S, I}, VK_SHADER_STAGE_COMPUTE_BIT);
        interpDsl = dsl({S, S, S, I, B}, VK_SHADER_STAGE_COMPUTE_BIT);
        presDsl = dsl({S}, VK_SHADER_STAGE_FRAGMENT_BIT);
        if (!pyrDsl || !flowDsl || !interpDsl || !presDsl) return false;
        pyrPipe = compute(kFgPyramidCompSpv, sizeof(kFgPyramidCompSpv), pyrDsl, 12, pyrPl);
        flowPipe = compute(kFgFlowCompSpv, sizeof(kFgFlowCompSpv), flowDsl, 0, flowPl);
        interpPipe = compute(kFgInterpCompSpv, sizeof(kFgInterpCompSpv), interpDsl, 20, interpPl);
        return pyrPipe && flowPipe && interpPipe;
    }

    bool makeImage(Img& im, VkFormat format, uint32_t iw, uint32_t ih, VkImageUsageFlags usage) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {iw, ih, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (f.vkCreateImage(ctx.device, &ci, nullptr, &im.image) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        f.vkGetImageMemoryRequirements(ctx.device, im.image, &req);
        const int32_t type = memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = static_cast<uint32_t>(type);
        if (f.vkAllocateMemory(ctx.device, &ai, nullptr, &im.memory) != VK_SUCCESS) return false;
        if (f.vkBindImageMemory(ctx.device, im.image, im.memory, 0) != VK_SUCCESS) return false;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (f.vkCreateImageView(ctx.device, &vi, nullptr, &im.view) != VK_SUCCESS) return false;
        im.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        im.w = iw;
        im.h = ih;
        return true;
    }

    void destroyImage(Img& im) {
        if (im.view) f.vkDestroyImageView(ctx.device, im.view, nullptr);
        if (im.image) f.vkDestroyImage(ctx.device, im.image, nullptr);
        if (im.memory) f.vkFreeMemory(ctx.device, im.memory, nullptr);
        im = Img{};
    }

    // Espera a que el hilo HyperFG no tenga trabajo y a que la GPU termine sus Present.
    void quiesce() {
        std::unique_lock<std::mutex> lk(m);
        hasJob = false;
        cv.wait(lk, [&] { return !busy; });
        lk.unlock();
        for (VkFence fe : fences) {
            if (fe) f.vkWaitForFences(ctx.device, 1, &fe, VK_TRUE, 1000000000ull);
        }
    }

    void destroySized() {
        if (!sized) return;
        quiesce();
        for (auto& hi : hist) destroyImage(hi);
        for (auto& lv : lum)
            for (auto& li : lv) destroyImage(li);
        destroyImage(flow);
        destroyImage(interp);
        if (statsBuf) f.vkDestroyBuffer(ctx.device, statsBuf, nullptr);
        if (statsMem) f.vkFreeMemory(ctx.device, statsMem, nullptr);
        statsBuf = VK_NULL_HANDLE;
        statsMem = VK_NULL_HANDLE;
        if (pool) f.vkDestroyDescriptorPool(ctx.device, pool, nullptr);
        pool = VK_NULL_HANDLE;
        sized = false;
        validFrames = 0;
        LOGI("HyperFG: recursos liberados");
    }

    bool ensureSized(VkFormat format, uint32_t iw, uint32_t ih) {
        uint32_t nfw = iw, nfh = ih;
        if (nfw > 1600u || nfh > 900u) {
            const float s = std::min(1600.0f / static_cast<float>(nfw), 900.0f / static_cast<float>(nfh));
            nfw = std::max(16u, (static_cast<uint32_t>(nfw * s) + 1u) & ~1u);
            nfh = std::max(16u, (static_cast<uint32_t>(nfh * s) + 1u) & ~1u);
        }
        if (sized && fmt == format && w == iw && h == ih && fw == nfw && fh == nfh) return true;
        destroySized();
        fmt = format;
        w = iw;
        h = ih;
        fw = nfw;
        fh = nfh;
        const uint32_t w2 = (fw + 1) / 2, h2 = (fh + 1) / 2, w4 = (w2 + 1) / 2, h4 = (h2 + 1) / 2,
                       w8 = (w4 + 1) / 2, h8 = (h4 + 1) / 2;
        const VkImageUsageFlags lumUse = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        bool ok = true;
        for (auto& hi : hist) ok = ok && makeImage(hi, format, w, h, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        for (auto& lv : lum) {
            ok = ok && makeImage(lv[0], VK_FORMAT_R16_SFLOAT, w2, h2, lumUse);
            ok = ok && makeImage(lv[1], VK_FORMAT_R16_SFLOAT, w4, h4, lumUse);
            ok = ok && makeImage(lv[2], VK_FORMAT_R16_SFLOAT, w8, h8, lumUse);
        }
        ok = ok && makeImage(flow, VK_FORMAT_R16G16_SFLOAT, w8, h8, lumUse);
        ok = ok && makeImage(interp, VK_FORMAT_R8G8B8A8_UNORM, w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        // Contadores del corte de escena (diffSum, count).
        if (ok) {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = 16;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ok = f.vkCreateBuffer(ctx.device, &bi, nullptr, &statsBuf) == VK_SUCCESS;
            if (ok) {
                VkMemoryRequirements req{};
                f.vkGetBufferMemoryRequirements(ctx.device, statsBuf, &req);
                const int32_t type = memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = static_cast<uint32_t>(type);
                ok = type >= 0 && f.vkAllocateMemory(ctx.device, &ai, nullptr, &statsMem) == VK_SUCCESS &&
                     f.vkBindBufferMemory(ctx.device, statsBuf, statsMem, 0) == VK_SUCCESS;
            }
        }
        ok = ok && createSets();
        sized = true;  // para que destroySized libere lo creado aunque algo fallara
        if (!ok) {
            destroySized();
            return fail("no se pudieron crear las imágenes del historial");
        }
        LOGI("HyperFG: historial %ux%u, pirámide %ux%u / %ux%u / %ux%u, flujo %ux%u", iw, ih, w2, h2, w4, h4, w8, h8, w8, h8);
        return true;
    }

    bool createSets() {
        const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 24},
                                               {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16},
                                               {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 8;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = sizes;
        if (f.vkCreateDescriptorPool(ctx.device, &pi, nullptr, &pool) != VK_SUCCESS) return false;
        auto alloc = [&](VkDescriptorSetLayout l, VkDescriptorSet& s) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &l;
            return f.vkAllocateDescriptorSets(ctx.device, &ai, &s) == VK_SUCCESS;
        };
        std::vector<VkDescriptorImageInfo> infos;
        infos.reserve(64);
        std::vector<VkWriteDescriptorSet> writes;
        auto img = [&](VkDescriptorSet s, uint32_t b, VkDescriptorType t, VkImageView v, VkImageLayout l) {
            infos.push_back({t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? linear : VK_NULL_HANDLE, v, l});
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = s;
            w.dstBinding = b;
            w.descriptorCount = 1;
            w.descriptorType = t;
            w.pImageInfo = &infos.back();
            writes.push_back(w);
        };
        VkDescriptorBufferInfo statsInfo{statsBuf, 0, 16};
        auto buf = [&](VkDescriptorSet s, uint32_t b) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = s;
            w.dstBinding = b;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &statsInfo;
            writes.push_back(w);
        };
        const auto S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        const auto RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, G = VK_IMAGE_LAYOUT_GENERAL;
        for (int c = 0; c < 2; ++c) {
            const int p = c ^ 1;
            if (!alloc(pyrDsl, pyrSet[c]) || !alloc(flowDsl, flowSet[c]) || !alloc(interpDsl, interpSet[c]) ||
                !alloc(presDsl, presSet[c])) {
                return false;
            }
            img(pyrSet[c], 0, S, hist[c].view, RO);
            img(pyrSet[c], 1, I, lum[c][0].view, G);
            img(pyrSet[c], 2, I, lum[c][1].view, G);
            img(pyrSet[c], 3, I, lum[c][2].view, G);
            img(pyrSet[c], 4, I, lum[p][2].view, G);
            buf(pyrSet[c], 5);
            for (int l = 0; l < 3; ++l) img(flowSet[c], static_cast<uint32_t>(l), S, lum[p][l].view, G);
            for (int l = 0; l < 3; ++l) img(flowSet[c], static_cast<uint32_t>(3 + l), S, lum[c][l].view, G);
            img(flowSet[c], 6, I, flow.view, G);
            img(interpSet[c], 0, S, hist[p].view, RO);
            img(interpSet[c], 1, S, hist[c].view, RO);
            img(interpSet[c], 2, S, flow.view, G);
            img(interpSet[c], 3, I, interp.view, G);
            buf(interpSet[c], 4);
            img(presSet[c], 0, S, hist[c].view, RO);
            infos.back().sampler = presentLinear;  // borde negro: bandas si el aspecto no coincide
        }
        f.vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    void barrier(VkCommandBuffer cmd, Img& im, VkImageLayout to, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = dstAccess;
        b.oldLayout = im.layout;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        f.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        im.layout = to;
    }

    void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        f.vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // ------------------------------------------------------------------ hilo de render
    VkImageView record(VkCommandBuffer cmd, VkImage src, VkFormat format, uint32_t iw, uint32_t ih, int gameFps,
                       bool repeatOnly = false) {
        copied = false;
        lastWasRepeat = repeatOnly;
        if (gameFps <= 0 || !loadDevice() || !ensureSized(format, iw, ih)) return VK_NULL_HANDLE;
        const int cur = static_cast<int>(frame & 1u);  // el anterior (cur ^ 1) va en los descriptor sets
        {
            // No bloquear nunca a RenderingThread: si el hilo HyperFG sigue presentando hist[cur],
            // se omite este ciclo sin esperar.
            std::lock_guard<std::mutex> lk(m);
            if (busy && jobHist == cur) {
                sCancelled.fetch_add(1, std::memory_order_relaxed);
                return VK_NULL_HANDLE;
            }
            if (hasJob && jobHist == cur) {
                hasJob = false;
                sCancelled.fetch_add(1, std::memory_order_relaxed);
            }
        }
        const int64_t now = nowNs();
        // Escudo 4: el frame real tardó > 22 ms (micro-carga): sin sintético en este ciclo.
        // A 30 FPS reales (protección térmica con HyperFG) el periodo es de 33 ms: el umbral escala.
        const int64_t slowNs = std::max<int64_t>(kSlowFrameNs, 1320000000LL / gameFps);  // = FrameGenLogic.slowFrameMs
        const bool slow = lastRecordNs != 0 && now - lastRecordNs > slowNs;
        if (lastRecordNs != 0) {
            const int64_t dt = std::min<int64_t>(now - lastRecordNs, 200000000LL);  // ignora pausas largas
            emaIntervalNs = emaIntervalNs == 0 ? dt : (emaIntervalNs * 7 + dt) / 8;
        }
        lastRecordNs = now;
        // Escudo 5 (histéresis): ritmo real sostenido por encima del umbral (GPU/CPU saturada, p. ej.
        // d3d9-worker con CSMT) -> bypass total, sin copia ni cómputo: coste = fg=0. Sale bajo el 85 %.
        if (!bypass && emaIntervalNs > slowNs) bypass = true;
        else if (bypass && emaIntervalNs < slowNs * 85 / 100) bypass = false;
        if (bypass && !repeatOnly) {
            sSlow.fetch_add(1, std::memory_order_relaxed);
            validFrames = 0;  // al salir, no interpolar con un historial viejo
            return VK_NULL_HANDLE;
        }
        const bool generate = validFrames >= 1 && !slow;

        // 1. Copia (o reescala lineal si > 1600x900) en GPU del backbuffer al historial.
        VkImageMemoryBarrier sb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        sb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sb.srcQueueFamilyIndex = sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sb.image = src;
        sb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        f.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                               nullptr, 1, &sb);
        barrier(cmd, hist[cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (w == iw && h == ih) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = region.srcSubresource;
            region.extent = {iw, ih, 1};
            f.vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, hist[cur].image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        } else {
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {static_cast<int32_t>(iw), static_cast<int32_t>(ih), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
            f.vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, hist[cur].image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }
        sb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        sb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        f.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                               nullptr, 1, &sb);
        barrier(cmd, hist[cur], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        copied = true;
        if (repeatOnly) {
            // Repetición: solo la copia; el hilo HyperFG vuelve a presentar este mismo frame.
            validFrames = 0;  // si luego se activa la generación, que no interpole con un frame viejo
            ++frame;
            sRepeat.fetch_add(1, std::memory_order_relaxed);
            return VK_NULL_HANDLE;
        }
        if (slow) {
            // Escudo 4: no pagar coste de cómputo (pirámide/flujo/interp) si el frame se va a omitir.
            sSlow.fetch_add(1, std::memory_order_relaxed);
            if (validFrames < 2) ++validFrames;
            ++frame;
            return VK_NULL_HANDLE;
        }

        // 2. Pirámide (y diferencia global para el corte de escena) a resolución acotada (<= 1600x900).
        for (auto& lv : lum)
            for (auto& li : lv)
                if (li.layout != VK_IMAGE_LAYOUT_GENERAL)
                    barrier(cmd, li, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        if (flow.layout != VK_IMAGE_LAYOUT_GENERAL)
            barrier(cmd, flow, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        f.vkCmdFillBuffer(cmd, statsBuf, 0, 16, 0);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        struct { float invW, invH; uint32_t hasPrev; } ppc{1.0f / static_cast<float>(fw), 1.0f / static_cast<float>(fh),
                                                            validFrames >= 1 ? 1u : 0u};
        f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pyrPipe);
        f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pyrPl, 0, 1, &pyrSet[cur], 0, nullptr);
        f.vkCmdPushConstants(cmd, pyrPl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), &ppc);
        f.vkCmdDispatch(cmd, (lum[cur][0].w + 7) / 8, (lum[cur][0].h + 7) / 8, 1);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        if (generate) {
            // 3. Flujo coarse-to-fine por bloques de 8x8.
            f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, flowPipe);
            f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, flowPl, 0, 1, &flowSet[cur], 0, nullptr);
            f.vkCmdDispatch(cmd, (flow.w + 7) / 8, (flow.h + 7) / 8, 1);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // 4. Interpolación t = 0.5 con los escudos.
            barrier(cmd, interp, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            struct { float invW, invH, st, conf, cut; } ipc{1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h),
                                                             kStaticThr, kConfThr, kCutThr};
            f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, interpPipe);
            f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, interpPl, 0, 1, &interpSet[cur], 0, nullptr);
            f.vkCmdPushConstants(cmd, interpPl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
            f.vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
            barrier(cmd, interp, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            sGenerated.fetch_add(1, std::memory_order_relaxed);
        }
        if (validFrames < 2) ++validFrames;
        ++frame;
        return generate ? interp.view : VK_NULL_HANDLE;
    }

    void scheduleReal(int64_t at) {
        std::lock_guard<std::mutex> lk(m);
        jobHist = static_cast<int>((frame - 1) & 1u);  // el frame recién grabado
        jobAt = at;
        jobRepeat = lastWasRepeat;
        hasJob = true;
        cv.notify_all();
    }

    // ------------------------------------------------------------------ hilo HyperFG
    bool ensurePresentPipes(VkFormat format) {
        if (presPipe != VK_NULL_HANDLE && swapFmt == format) return true;
        swapFmt = format;
        VkAttachmentDescription a{};
        a.format = format;
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
        if (f.vkCreateRenderPass(ctx.device, &ri, nullptr, &rp) != VK_SUCCESS) return false;
        const VkPushConstantRange vr{VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
        const VkPushConstantRange both[2] = {vr, {VK_SHADER_STAGE_FRAGMENT_BIT, 32, 32}};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &presDsl;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &vr;
        if (f.vkCreatePipelineLayout(ctx.device, &li, nullptr, &presPl) != VK_SUCCESS) return false;
        li.pushConstantRangeCount = 2;
        li.pPushConstantRanges = both;
        if (f.vkCreatePipelineLayout(ctx.device, &li, nullptr, &sgsrPl) != VK_SUCCESS) return false;
        VkShaderModule vs = module(kPresentVertSpv, sizeof(kPresentVertSpv));
        VkShaderModule fs = module(kPresentFragSpv, sizeof(kPresentFragSpv));
        VkShaderModule sg = module(kPresentSgsrFragSpv, sizeof(kPresentSgsrFragSpv));
        if (!vs || !fs || !sg) return false;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;
        st[1].pName = "main";
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
        gi.pStages = st;
        gi.pVertexInputState = &vi;
        gi.pInputAssemblyState = &ia;
        gi.pViewportState = &vp;
        gi.pRasterizationState = &rs;
        gi.pMultisampleState = &ms;
        gi.pColorBlendState = &cb;
        gi.pDynamicState = &ds;
        gi.layout = presPl;
        gi.renderPass = rp;
        bool ok = f.vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gi, nullptr, &presPipe) == VK_SUCCESS;
        st[1].module = sg;
        gi.layout = sgsrPl;
        if (ok && f.vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gi, nullptr, &sgsrPipe) != VK_SUCCESS) {
            sgsrPipe = VK_NULL_HANDLE;
        }
        f.vkDestroyShaderModule(ctx.device, vs, nullptr);
        f.vkDestroyShaderModule(ctx.device, fs, nullptr);
        f.vkDestroyShaderModule(ctx.device, sg, nullptr);
        return ok;
    }

    void presentReal(int idx) {
        const int slot = cmdSlot;
        cmdSlot ^= 1;
        f.vkWaitForFences(ctx.device, 1, &fences[slot], VK_TRUE, 200000000ull);
        DirectVulkanBridge::SwapTarget t;
        if (!host->beginPresent(t)) return;
        VkCommandBuffer cmd = cmds[slot];
        bool recorded = false;
        if (ensurePresentPipes(t.format)) {
            if (t.generation != fbGen) {
                for (auto& [v, fb] : fbs) f.vkDestroyFramebuffer(ctx.device, fb, nullptr);
                fbs.clear();
                fbGen = t.generation;
            }
            VkFramebuffer& fb = fbs[t.view];
            if (fb == VK_NULL_HANDLE) {
                VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
                fi.renderPass = rp;
                fi.attachmentCount = 1;
                fi.pAttachments = &t.view;
                fi.width = t.extent.width;
                fi.height = t.extent.height;
                fi.layers = 1;
                f.vkCreateFramebuffer(ctx.device, &fi, nullptr, &fb);
            }
            if (fb != VK_NULL_HANDLE) {
                f.vkResetFences(ctx.device, 1, &fences[slot]);
                f.vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                f.vkBeginCommandBuffer(cmd, &bi);
                // El frame real lo escribió el hilo de render en un envío anterior de la misma cola.
                memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                rb.renderPass = rp;
                rb.framebuffer = fb;
                rb.renderArea = {{0, 0}, t.extent};
                f.vkCmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
                const bool sgsr = sgsrPipe != VK_NULL_HANDLE && presentopt::useSgsr(w, h, t.extent.width, t.extent.height);
                const VkPipelineLayout layout = sgsr ? sgsrPl : presPl;
                f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sgsr ? sgsrPipe : presPipe);
                f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &presSet[idx], 0, nullptr);
                VkViewport vpt{0, 0, static_cast<float>(t.extent.width), static_cast<float>(t.extent.height), 0, 1};
                VkRect2D sc{{0, 0}, t.extent};
                f.vkCmdSetViewport(cmd, 0, 1, &vpt);
                f.vkCmdSetScissor(cmd, 0, 1, &sc);
                float pc[8];
                rotationPc(t.transform, pc);
                presentopt::applyFit(pc, w, h, t.extent.width, t.extent.height);
                f.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pc);
                if (sgsr) {
                    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
                    const float fp[8] = {1.0f / fw, 1.0f / fh, fw, fh, presentopt::g_sharpness.load(), 0, 0, 0};
                    f.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 32, 32, fp);
                }
                f.vkCmdDraw(cmd, 3, 1, 0, 0);
                f.vkCmdEndRenderPass(cmd);
                f.vkEndCommandBuffer(cmd);
                recorded = true;
            }
        }
        if (!recorded) {
            // Sin pase: dejar la imagen adquirida en PRESENT_SRC igualmente.
            f.vkResetFences(ctx.device, 1, &fences[slot]);
            f.vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            f.vkBeginCommandBuffer(cmd, &bi);
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = t.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            f.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                                   nullptr, 0, nullptr, 1, &b);
            f.vkEndCommandBuffer(cmd);
        }
        // Una repetición (mismo frame) no cuenta como frame nuevo en el HUD; el real de HyperFG sí.
        host->endPresent(t, cmd, fences[slot], !jobIsRepeat);
        const uint64_t n = sReal.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % 240 == 0) {
            LOGI("HyperFG: %llu sintéticos, %llu reales, %llu omitidos (frame lento), %llu cancelados, %llu repeticiones",
                 static_cast<unsigned long long>(sGenerated.load()), static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(sSlow.load()), static_cast<unsigned long long>(sCancelled.load()),
                 static_cast<unsigned long long>(sRepeat.load()));
        }
    }

    void presentLoop() {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            cv.wait(lk, [&] { return quit || hasJob; });
            if (quit) return;
            const int idx = jobHist;
            const int64_t at = jobAt;
            jobIsRepeat = jobRepeat;
            hasJob = false;
            busy = true;
            lk.unlock();
            sleepUntil(at);
            presentReal(idx);
            lk.lock();
            busy = false;
            cv.notify_all();
        }
    }
};

namespace {
struct BridgeHost final : FrameGenHost {
    bool gpuContext(DirectVulkanBridge::GpuContext& out) override { return DirectVulkanBridge::shared().gpuContext(out); }
    bool beginPresent(DirectVulkanBridge::SwapTarget& out) override { return DirectVulkanBridge::shared().beginPresent(out); }
    bool endPresent(const DirectVulkanBridge::SwapTarget& t, VkCommandBuffer cmd, VkFence fence, bool count) override {
        return DirectVulkanBridge::shared().endPresent(t, cmd, fence, count);
    }
};
}  // namespace

HyperFrameGen& HyperFrameGen::shared() {
    static BridgeHost h;
    static HyperFrameGen g(&h);
    return g;
}

HyperFrameGen::Impl& HyperFrameGen::impl() {
    std::lock_guard<std::mutex> lock(implMutex_);
    if (impl_ == nullptr) {
        impl_ = new Impl();
        impl_->host = host_;
    }
    return *impl_;
}

VkImage HyperFrameGen::interpImage() noexcept { return impl().interp.image; }

void HyperFrameGen::setEnabled(bool on) noexcept {
    if (g_enabled.exchange(on) != on) LOGI("HyperFG %s", on ? "ACTIVADO (2x)" : "desactivado");
}

bool HyperFrameGen::enabled() noexcept {
    static const int s_prop = [] {
        char v[PROP_VALUE_MAX] = {};
        if (__system_property_get("debug.hyperplay.fg", v) <= 0) return -1;
        return v[0] == '1' ? 1 : 0;
    }();
    return s_prop >= 0 ? s_prop == 1 : g_enabled.load(std::memory_order_relaxed);
}

VkImageView HyperFrameGen::record(VkCommandBuffer cmd, VkImage src, VkFormat format, uint32_t width, uint32_t height,
                                  int gameFps) {
    return impl().record(cmd, src, format, width, height, gameFps);
}

void HyperFrameGen::scheduleReal(int64_t atNs) { impl().scheduleReal(atNs); }

bool HyperFrameGen::repeatWanted(int gameFps, int panelHz) noexcept {
    static const bool s_on = [] {
        char v[PROP_VALUE_MAX] = {};
        return !(__system_property_get("debug.hyperplay.repeat", v) > 0 && v[0] == '0');
    }();
    return s_on && gameFps > 0 && gameFps <= kRepeatMaxFps && panelHz >= 2 * gameFps - 2;
}

bool HyperFrameGen::recordRepeat(VkCommandBuffer cmd, VkImage src, VkFormat format, uint32_t width, uint32_t height,
                                 int gameFps) {
    Impl& i = impl();
    i.record(cmd, src, format, width, height, gameFps, true);
    return i.copied;
}

void HyperFrameGen::releaseIfIdle() {
    if (impl_ == nullptr) return;  // nunca se activó: cero coste
    Impl& i = impl();
    if (i.sized) i.destroySized();
    i.lastRecordNs = 0;
    i.emaIntervalNs = 0;
    i.bypass = false;
}

HyperFrameGen::Stats HyperFrameGen::stats() const noexcept {
    Stats s;
    if (impl_ == nullptr) return s;
    s.generated = impl_->sGenerated.load();
    s.skippedSlow = impl_->sSlow.load();
    s.realPresents = impl_->sReal.load();
    s.cancelled = impl_->sCancelled.load();
    return s;
}

}  // namespace hyperplay::gpu
