#include "DirectVulkanBridge.h"

#include <sys/system_properties.h>

#include <android/log.h>
#include <dlfcn.h>
#include <time.h>
#include <vulkan/vulkan_android.h>

#include "HyperFrameGen.h"
#include "loader/Win32DisplayInfo.h"

#include <algorithm>
#include <cstring>

#define LOG_TAG "HyperPlay.DirectVulkan"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace hyperplay {
namespace {

uint64_t nowNs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) noexcept {
    if ((supported & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0) return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((supported & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) != 0) return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if ((supported & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) != 0) {
        return VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    }
    return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
}

}  // namespace

DirectVulkanBridge::DirectVulkanBridge() = default;

DirectVulkanBridge::~DirectVulkanBridge() {
    shutdown();
}

DirectVulkanBridge& DirectVulkanBridge::shared() {
    static DirectVulkanBridge s_instance;
    return s_instance;
}

bool DirectVulkanBridge::init(const std::string& customTurnipPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    return initLocked(customTurnipPath);
}

bool DirectVulkanBridge::initLocked(const std::string& customTurnipPath) {
    if (instance_ != VK_NULL_HANDLE) return true;

    if (!loader_.load(customTurnipPath)) {
        LOGW("No se pudo cargar ningún driver Vulkan");
        return false;
    }
    getInstanceProcAddr_ = loader_.getInstanceProcAddr();
    if (getInstanceProcAddr_ == nullptr) {
        LOGW("El driver no expone vkGetInstanceProcAddr");
        loader_.unload();
        return false;
    }

    if (!createInstance()) {
        loader_.unload();
        getInstanceProcAddr_ = nullptr;
        return false;
    }

    if (!selectPhysicalDevice()) {
        auto destroyInstance = instProc<PFN_vkDestroyInstance>("vkDestroyInstance");
        if (destroyInstance != nullptr) destroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        loader_.unload();
        getInstanceProcAddr_ = nullptr;
        return false;
    }

    getDeviceProcAddr_ = instProc<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
    if (getDeviceProcAddr_ == nullptr) {
        LOGW("vkGetDeviceProcAddr no disponible");
        auto destroyInstance = instProc<PFN_vkDestroyInstance>("vkDestroyInstance");
        if (destroyInstance != nullptr) destroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        loader_.unload();
        getInstanceProcAddr_ = nullptr;
        return false;
    }

    LOGI("Vulkan listo: %s (driver %s)", gpuName_.c_str(), loader_.driverPath().c_str());
    return true;
}

bool DirectVulkanBridge::createInstance() {
    auto enumerateExtensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        getInstanceProcAddr_(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    auto createInstanceFn = reinterpret_cast<PFN_vkCreateInstance>(
        getInstanceProcAddr_(VK_NULL_HANDLE, "vkCreateInstance"));
    if (enumerateExtensions == nullptr || createInstanceFn == nullptr) {
        LOGW("El driver no exporta vkCreateInstance/vkEnumerateInstanceExtensionProperties");
        return false;
    }

    uint32_t extCount = 0;
    if (enumerateExtensions(nullptr, &extCount, nullptr) != VK_SUCCESS || extCount == 0) {
        LOGW("No hay extensiones de instancia disponibles");
        return false;
    }
    std::vector<VkExtensionProperties> extensions(extCount);
    if (enumerateExtensions(nullptr, &extCount, extensions.data()) != VK_SUCCESS) {
        LOGW("vkEnumerateInstanceExtensionProperties falló");
        return false;
    }

    auto hasExtension = [&extensions](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [name](const VkExtensionProperties& ext) {
                               return std::strcmp(ext.extensionName, name) == 0;
                           });
    };
    if (!hasExtension(VK_KHR_SURFACE_EXTENSION_NAME) ||
        !hasExtension(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)) {
        LOGW("El driver no soporta VK_KHR_surface + VK_KHR_android_surface");
        return false;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "HyperPlay";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "HyperPlay Direct Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    const char* enabledExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 2;
    createInfo.ppEnabledExtensionNames = enabledExtensions;

    const VkResult result = createInstanceFn(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        LOGW("vkCreateInstance falló: %d", static_cast<int>(result));
        instance_ = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool DirectVulkanBridge::selectPhysicalDevice() {
    auto enumerateDevices =
        instProc<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
    if (enumerateDevices == nullptr) return false;

    uint32_t deviceCount = 0;
    VkResult result = enumerateDevices(instance_, &deviceCount, nullptr);
    if (result != VK_SUCCESS || deviceCount == 0) {
        LOGW("No hay dispositivos Vulkan: %d", static_cast<int>(result));
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    result = enumerateDevices(instance_, &deviceCount, devices.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        LOGW("vkEnumeratePhysicalDevices falló: %d", static_cast<int>(result));
        return false;
    }

    auto enumerateDeviceExtensions =
        instProc<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
    auto getProperties =
        instProc<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");

    for (VkPhysicalDevice device : devices) {
        bool supportsSwapchain = true;
        if (enumerateDeviceExtensions != nullptr) {
            uint32_t extCount = 0;
            enumerateDeviceExtensions(device, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> extensions(extCount);
            if (extCount > 0) {
                enumerateDeviceExtensions(device, nullptr, &extCount, extensions.data());
            }
            supportsSwapchain = std::any_of(
                extensions.begin(), extensions.end(), [](const VkExtensionProperties& ext) {
                    return std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
                });
        }
        if (!supportsSwapchain) continue;

        physicalDevice_ = device;
        if (getProperties != nullptr) {
            VkPhysicalDeviceProperties props{};
            getProperties(device, &props);
            gpuName_ = props.deviceName;
        }
        break;
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        LOGW("Ningún dispositivo soporta VK_KHR_swapchain");
        return false;
    }
    return true;
}

bool DirectVulkanBridge::createSurface() {
    if (instance_ == VK_NULL_HANDLE || window_ == nullptr) return false;
    auto createAndroidSurface =
        instProc<PFN_vkCreateAndroidSurfaceKHR>("vkCreateAndroidSurfaceKHR");
    if (createAndroidSurface == nullptr) {
        LOGW("vkCreateAndroidSurfaceKHR no disponible");
        return false;
    }

    VkAndroidSurfaceCreateInfoKHR surfaceInfo{};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.window = window_;

    const VkResult result = createAndroidSurface(instance_, &surfaceInfo, nullptr, &surface_);
    if (result != VK_SUCCESS) {
        LOGW("vkCreateAndroidSurfaceKHR falló: %d", static_cast<int>(result));
        surface_ = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool DirectVulkanBridge::ensureDevice() {
    if (device_ != VK_NULL_HANDLE) return true;

    auto getQueueFamilies = instProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        "vkGetPhysicalDeviceQueueFamilyProperties");
    auto getSurfaceSupport = instProc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        "vkGetPhysicalDeviceSurfaceSupportKHR");
    auto createDevice = instProc<PFN_vkCreateDevice>("vkCreateDevice");
    auto getDeviceQueue = instProc<PFN_vkGetDeviceQueue>("vkGetDeviceQueue");
    if (getQueueFamilies == nullptr || getSurfaceSupport == nullptr ||
        createDevice == nullptr || getDeviceQueue == nullptr) {
        LOGW("Faltan funciones de dispositivo en el driver");
        return false;
    }

    uint32_t familyCount = 0;
    getQueueFamilies(physicalDevice_, &familyCount, nullptr);
    if (familyCount == 0) {
        LOGW("El dispositivo físico no expone familias de colas");
        return false;
    }
    std::vector<VkQueueFamilyProperties> families(familyCount);
    getQueueFamilies(physicalDevice_, &familyCount, families.data());

    bool found = false;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;
        if (families[i].queueCount == 0) continue;
        VkBool32 presentSupported = VK_FALSE;
        if (getSurfaceSupport(physicalDevice_, i, surface_, &presentSupported) != VK_SUCCESS) {
            continue;
        }
        if (presentSupported == VK_FALSE) continue;
        queueFamilyIndex_ = i;
        found = true;
        break;
    }
    if (!found) {
        LOGW("No hay familia de colas con gráficos + presentación");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamilyIndex_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    // Características que usa el renderizador D3D11 (texturas BC nativas, anisotropía, blend
    // independiente por RT), solo si el dispositivo las ofrece.
    enabledFeatures_ = {};
    if (auto getFeatures = instProc<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")) {
        VkPhysicalDeviceFeatures avail{};
        getFeatures(physicalDevice_, &avail);
        enabledFeatures_.textureCompressionBC = avail.textureCompressionBC;
        enabledFeatures_.samplerAnisotropy = avail.samplerAnisotropy;
        enabledFeatures_.independentBlend = avail.independentBlend;
        LOGI("Características: BC=%d anisotropía=%d independentBlend=%d", avail.textureCompressionBC,
             avail.samplerAnisotropy, avail.independentBlend);
    }
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabledFeatures_;
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;

    VkResult result = createDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        LOGW("vkCreateDevice falló: %d", static_cast<int>(result));
        device_ = VK_NULL_HANDLE;
        queueFamilyIndex_ = UINT32_MAX;
        return false;
    }
    getDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

    auto destroyDevice = devProc<PFN_vkDestroyDevice>("vkDestroyDevice");
    auto failDevice = [this, destroyDevice](const char* reason) {
        LOGW("%s", reason);
        if (destroyDevice != nullptr) destroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        queue_ = VK_NULL_HANDLE;
        queueFamilyIndex_ = UINT32_MAX;
        return false;
    };

    getDeviceProcAddr_ = instProc<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
    if (getDeviceProcAddr_ == nullptr) return failDevice("vkGetDeviceProcAddr no disponible");

    auto createSemaphore = devProc<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    auto createFence = devProc<PFN_vkCreateFence>("vkCreateFence");
    if (createSemaphore == nullptr || createFence == nullptr) {
        return failDevice("vkCreateSemaphore/vkCreateFence no disponibles");
    }

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    result = createSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailable_);
    if (result != VK_SUCCESS) {
        imageAvailable_ = VK_NULL_HANDLE;
        return failDevice("No se pudo crear el semáforo de imagen");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    result = createFence(device_, &fenceInfo, nullptr, &inFlightFence_);
    if (result != VK_SUCCESS) {
        inFlightFence_ = VK_NULL_HANDLE;
        return failDevice("No se pudo crear el fence de frame");
    }

    // Sin shader CAS validado el pase queda no disponible; init() lo registra y devuelve false.
    fsrPass_.init(device_, physicalDevice_);
    LOGI("Dispositivo lógico creado (familia de colas %u)", queueFamilyIndex_);
    return true;
}

bool DirectVulkanBridge::createSwapchain() {
    if (device_ == VK_NULL_HANDLE || surface_ == VK_NULL_HANDLE) return false;

    auto getCapabilities = instProc<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    auto getFormats = instProc<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        "vkGetPhysicalDeviceSurfaceFormatsKHR");
    auto getPresentModes = instProc<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
        "vkGetPhysicalDeviceSurfacePresentModesKHR");
    auto createSwapchainFn = devProc<PFN_vkCreateSwapchainKHR>("vkCreateSwapchainKHR");
    auto getSwapchainImages = devProc<PFN_vkGetSwapchainImagesKHR>("vkGetSwapchainImagesKHR");
    auto createImageView = devProc<PFN_vkCreateImageView>("vkCreateImageView");
    if (getCapabilities == nullptr || getFormats == nullptr || getPresentModes == nullptr ||
        createSwapchainFn == nullptr || getSwapchainImages == nullptr ||
        createImageView == nullptr) {
        LOGW("Faltan funciones de swapchain en el driver");
        return false;
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    VkResult result = getCapabilities(physicalDevice_, surface_, &capabilities);
    if (result != VK_SUCCESS) {
        LOGW("vkGetPhysicalDeviceSurfaceCapabilitiesKHR falló: %d", static_cast<int>(result));
        return false;
    }

    uint32_t formatCount = 0;
    getFormats(physicalDevice_, surface_, &formatCount, nullptr);
    if (formatCount == 0) {
        LOGW("La superficie no expone formatos");
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    getFormats(physicalDevice_, surface_, &formatCount, formats.data());

    VkSurfaceFormatKHR chosenFormat = formats[0];
    for (const VkSurfaceFormatKHR& candidate : formats) {
        if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM ||
            candidate.format == VK_FORMAT_B8G8R8A8_UNORM) {
            chosenFormat = candidate;
            break;
        }
    }
    swapchainFormat_ = chosenFormat.format;
    swapchainColorSpace_ = chosenFormat.colorSpace;

    uint32_t modeCount = 0;
    getPresentModes(physicalDevice_, surface_, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(modeCount);
    if (modeCount > 0) {
        getPresentModes(physicalDevice_, surface_, &modeCount, presentModes.data());
    }
    // FIFO (vsync) por defecto: un frame por refresco del panel, ritmo estable y sin tearing;
    // el juego mide su delta de tiempo real, así que no hay aceleraciones. MAILBOX (sin límite,
    // descarta frames) solo con `setprop debug.hyperplay.mailbox 1`, para medir el techo de FPS.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    char mailboxProp[PROP_VALUE_MAX] = {};
    const bool autoMailbox144 = ::hyperplay::win32::refreshHz() >= 120;
    const bool wantMailbox =
        (__system_property_get("debug.hyperplay.mailbox", mailboxProp) > 0 && mailboxProp[0] == '1') ||
        (autoMailbox144 && !(mailboxProp[0] == '0'));
    for (VkPresentModeKHR mode : presentModes) {
        if (wantMailbox && mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = mode;
            break;
        }
    }

    VkExtent2D extent{};
    if (capabilities.currentExtent.width != UINT32_MAX) {
        extent = capabilities.currentExtent;
    } else {
        const uint32_t width = static_cast<uint32_t>(std::max(0, ANativeWindow_getWidth(window_)));
        const uint32_t height = static_cast<uint32_t>(std::max(0, ANativeWindow_getHeight(window_)));
        extent.width = std::clamp(width, capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(height, capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        LOGW("Extensión de swapchain inválida: %ux%u", extent.width, extent.height);
        return false;
    }
    swapchainExtent_ = extent;

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainTransferDst_ = (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0;
    if (swapchainTransferDst_) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    surfaceTransform_ = capabilities.currentTransform;

    VkSwapchainCreateInfoKHR swapInfo{};
    swapInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapInfo.surface = surface_;
    swapInfo.minImageCount = imageCount;
    swapInfo.imageFormat = swapchainFormat_;
    swapInfo.imageColorSpace = swapchainColorSpace_;
    swapInfo.imageExtent = swapchainExtent_;
    swapInfo.imageArrayLayers = 1;
    swapInfo.imageUsage = usage;
    swapInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapInfo.preTransform = capabilities.currentTransform;
    swapInfo.compositeAlpha = chooseCompositeAlpha(capabilities.supportedCompositeAlpha);
    swapInfo.presentMode = presentMode;
    swapInfo.clipped = VK_TRUE;
    swapInfo.oldSwapchain = VK_NULL_HANDLE;

    result = createSwapchainFn(device_, &swapInfo, nullptr, &swapchain_);
    if (result != VK_SUCCESS) {
        LOGW("vkCreateSwapchainKHR falló: %d", static_cast<int>(result));
        swapchain_ = VK_NULL_HANDLE;
        return false;
    }

    uint32_t actualImageCount = 0;
    result = getSwapchainImages(device_, swapchain_, &actualImageCount, nullptr);
    if (result != VK_SUCCESS || actualImageCount == 0) {
        LOGW("vkGetSwapchainImagesKHR falló: %d", static_cast<int>(result));
        cleanupSwapchainLocked();
        return false;
    }
    std::vector<VkImage> images(actualImageCount);
    result = getSwapchainImages(device_, swapchain_, &actualImageCount, images.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        LOGW("vkGetSwapchainImagesKHR (datos) falló: %d", static_cast<int>(result));
        cleanupSwapchainLocked();
        return false;
    }
    images.resize(actualImageCount);
    swapchainImages_ = std::move(images);

    swapchainViews_.assign(actualImageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < actualImageCount; ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat_;
        viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        result = createImageView(device_, &viewInfo, nullptr, &swapchainViews_[i]);
        if (result != VK_SUCCESS) {
            LOGW("vkCreateImageView[%u] falló: %d", i, static_cast<int>(result));
            cleanupSwapchainLocked();
            return false;
        }
    }

    ++swapchainGeneration_;
    using SetFrameRateFn = int32_t (*)(ANativeWindow*, float, int8_t, int8_t);
    static auto setFrameRateFn = reinterpret_cast<SetFrameRateFn>(
        dlsym(RTLD_DEFAULT, "ANativeWindow_setFrameRateWithChangeStrategy"));
    if (setFrameRateFn != nullptr && window_ != nullptr) {
        const float targetHz = gpu::HyperFrameGen::enabled() ? 120.0f : 60.0f;
        setFrameRateFn(window_, targetHz, 1 /* FIXED_SOURCE */, 0 /* ONLY_IF_SEAMLESS */);
        LOGI("ANativeWindow_setFrameRate -> %.0f Hz", targetHz);
    }
    LOGI("Swapchain %ux%u formato %d alpha %d modo %d (%u imágenes) transform=0x%x transferDst=%d",
         swapchainExtent_.width, swapchainExtent_.height, static_cast<int>(swapchainFormat_),
         static_cast<int>(swapInfo.compositeAlpha), static_cast<int>(presentMode),
         actualImageCount, static_cast<unsigned>(surfaceTransform_), swapchainTransferDst_ ? 1 : 0);
    return true;
}

bool DirectVulkanBridge::recreateSwapchain() {
    if (device_ == VK_NULL_HANDLE || surface_ == VK_NULL_HANDLE) return false;
    auto waitIdle = devProc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
    if (waitIdle != nullptr) {
        const VkResult result = waitIdle(device_);
        if (result != VK_SUCCESS) {
            LOGW("vkDeviceWaitIdle falló antes de recrear el swapchain: %d",
                 static_cast<int>(result));
        }
    }
    cleanupSwapchainLocked();
    return createSwapchain();
}

void DirectVulkanBridge::cleanupSwapchainLocked() {
    if (device_ != VK_NULL_HANDLE) {
        auto destroyImageView = devProc<PFN_vkDestroyImageView>("vkDestroyImageView");
        for (VkImageView view : swapchainViews_) {
            if (view != VK_NULL_HANDLE && destroyImageView != nullptr) {
                destroyImageView(device_, view, nullptr);
            }
        }
        auto destroySwapchain = devProc<PFN_vkDestroySwapchainKHR>("vkDestroySwapchainKHR");
        if (swapchain_ != VK_NULL_HANDLE && destroySwapchain != nullptr) {
            destroySwapchain(device_, swapchain_, nullptr);
        }
    }
    swapchainViews_.clear();
    swapchainImages_.clear();
    swapchain_ = VK_NULL_HANDLE;
    swapchainFormat_ = VK_FORMAT_UNDEFINED;
    swapchainExtent_ = {0, 0};
}

bool DirectVulkanBridge::attachWindow(ANativeWindow* window) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (window == nullptr) return false;
    // La superficie de Android suele llegar antes que D3D11CreateDevice: se inicializa aquí
    // para no perderla (antes devolvía false y nadie reintentaba la vinculación).
    if (instance_ == VK_NULL_HANDLE && !initLocked("")) {
        LOGW("attachWindow: Vulkan no disponible");
        return false;
    }
    if (window_ == window && swapchain_ != VK_NULL_HANDLE) return true;

    detachWindowLocked();
    window_ = window;
    ANativeWindow_acquire(window_);
    if (guestOwnsWindow_) return true;  // el juego presenta él mismo (VulkanPassthrough)

    if (!createSurface()) {
        detachWindowLocked();
        return false;
    }
    if (!ensureDevice()) {
        detachWindowLocked();
        return false;
    }
    if (!createSwapchain()) {
        detachWindowLocked();
        return false;
    }
    return true;
}

void DirectVulkanBridge::detachWindow() {
    std::lock_guard<std::mutex> lock(mutex_);
    detachWindowLocked();
}

PFN_vkGetInstanceProcAddr DirectVulkanBridge::guestLoader() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == VK_NULL_HANDLE && !initLocked("")) return nullptr;
    return getInstanceProcAddr_;
}

ANativeWindow* DirectVulkanBridge::releaseWindowForGuest() {
    std::lock_guard<std::mutex> lock(mutex_);
    guestOwnsWindow_ = true;
    if (device_ != VK_NULL_HANDLE) {
        auto waitIdle = devProc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
        if (waitIdle != nullptr) waitIdle(device_);
    }
    cleanupSwapchainLocked();
    if (surface_ != VK_NULL_HANDLE) {
        auto destroySurface = instProc<PFN_vkDestroySurfaceKHR>("vkDestroySurfaceKHR");
        if (destroySurface != nullptr) destroySurface(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    LOGI("releaseWindowForGuest: el juego presenta en la ventana %p", static_cast<void*>(window_));
    return window_;
}

void DirectVulkanBridge::countGuestPresent() {
    std::lock_guard<std::mutex> lock(mutex_);
    countFrameLocked();
}

void DirectVulkanBridge::detachWindowLocked() {
    if (device_ != VK_NULL_HANDLE) {
        auto waitIdle = devProc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
        if (waitIdle != nullptr) waitIdle(device_);
    }
    cleanupSwapchainLocked();

    if (surface_ != VK_NULL_HANDLE) {
        auto destroySurface = instProc<PFN_vkDestroySurfaceKHR>("vkDestroySurfaceKHR");
        if (destroySurface != nullptr) destroySurface(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    if (window_ != nullptr) {
        ANativeWindow_release(window_);
        window_ = nullptr;
    }
}

bool DirectVulkanBridge::acquireImageLocked(uint32_t& imageIndex) {
    auto acquireNextImage = devProc<PFN_vkAcquireNextImageKHR>("vkAcquireNextImageKHR");
    auto waitForFences = devProc<PFN_vkWaitForFences>("vkWaitForFences");
    auto resetFences = devProc<PFN_vkResetFences>("vkResetFences");
    if (acquireNextImage == nullptr || waitForFences == nullptr || resetFences == nullptr ||
        inFlightFence_ == VK_NULL_HANDLE || imageAvailable_ == VK_NULL_HANDLE) {
        return false;
    }
    constexpr uint64_t kTimeoutNs = 1000000000ULL;
    VkResult result = VK_SUCCESS;
    if (fencePending_) {
        result = waitForFences(device_, 1, &inFlightFence_, VK_TRUE, kTimeoutNs);
        if (result != VK_SUCCESS) {
            LOGW("vkWaitForFences fallo o expiro: %d", static_cast<int>(result));
            return false;
        }
        fencePending_ = false;
    }
    if (resetFences(device_, 1, &inFlightFence_) != VK_SUCCESS) return false;
    result = acquireNextImage(device_, swapchain_, kTimeoutNs, imageAvailable_, inFlightFence_, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
        LOGW("Swapchain obsoleto/superficie perdida al adquirir (%d); recreando", static_cast<int>(result));
        (void)recreateSwapchain();
        return false;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) return false;
    fencePending_ = true;
    return true;
}

void DirectVulkanBridge::countFrameLocked() {
    totalFrames_.fetch_add(1, std::memory_order_relaxed);
    ++frameCounter_;
    const uint64_t now = nowNs();
    if (lastFpsCheckTime_ == 0) lastFpsCheckTime_ = now;
    const uint64_t elapsed = now - lastFpsCheckTime_;
    if (elapsed >= 1000000000ULL) {
        currentFps_.store(static_cast<float>(frameCounter_) * 1000000000.0f / static_cast<float>(elapsed),
                          std::memory_order_relaxed);
        frameCounter_ = 0;
        lastFpsCheckTime_ = now;
    }
}

bool DirectVulkanBridge::ensureUploadResourcesLocked(VkDeviceSize bytes) {
    auto createCommandPool = devProc<PFN_vkCreateCommandPool>("vkCreateCommandPool");
    auto allocateCommandBuffers = devProc<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
    auto createSemaphore = devProc<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    auto createFence = devProc<PFN_vkCreateFence>("vkCreateFence");
    auto createBuffer = devProc<PFN_vkCreateBuffer>("vkCreateBuffer");
    auto getBufferReqs = devProc<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    auto allocateMemory = devProc<PFN_vkAllocateMemory>("vkAllocateMemory");
    auto bindBufferMemory = devProc<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    auto mapMemory = devProc<PFN_vkMapMemory>("vkMapMemory");
    auto getMemProps = instProc<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    if (createCommandPool == nullptr || allocateCommandBuffers == nullptr || createSemaphore == nullptr ||
        createFence == nullptr || createBuffer == nullptr || getBufferReqs == nullptr ||
        allocateMemory == nullptr || bindBufferMemory == nullptr || mapMemory == nullptr ||
        getMemProps == nullptr) {
        return false;
    }
    if (commandPool_ == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex_;
        if (createCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
            commandPool_ = VK_NULL_HANDLE;
            return false;
        }
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool_;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (allocateCommandBuffers(device_, &allocInfo, &commandBuffer_) != VK_SUCCESS) return false;
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (createSemaphore(device_, &semInfo, nullptr, &renderFinished_) != VK_SUCCESS) return false;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (createFence(device_, &fenceInfo, nullptr, &submitFence_) != VK_SUCCESS) return false;
    }
    if (stagingBuffer_ != VK_NULL_HANDLE && stagingSize_ >= bytes) return true;

    // Buffer de staging host-visible/coherente del tamaño del swapchain.
    auto destroyBuffer = devProc<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    auto freeMemory = devProc<PFN_vkFreeMemory>("vkFreeMemory");
    if (stagingBuffer_ != VK_NULL_HANDLE && destroyBuffer != nullptr) destroyBuffer(device_, stagingBuffer_, nullptr);
    if (stagingMemory_ != VK_NULL_HANDLE && freeMemory != nullptr) freeMemory(device_, stagingMemory_, nullptr);
    stagingBuffer_ = VK_NULL_HANDLE;
    stagingMemory_ = VK_NULL_HANDLE;
    stagingMapped_ = nullptr;
    stagingSize_ = 0;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bytes;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (createBuffer(device_, &bufInfo, nullptr, &stagingBuffer_) != VK_SUCCESS) {
        stagingBuffer_ = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements reqs{};
    getBufferReqs(device_, stagingBuffer_, &reqs);
    VkPhysicalDeviceMemoryProperties props{};
    getMemProps(physicalDevice_, &props);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t typeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((reqs.memoryTypeBits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & want) == want) {
            typeIndex = i;
            break;
        }
    }
    if (typeIndex == UINT32_MAX) return false;
    VkMemoryAllocateInfo memInfo{};
    memInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memInfo.allocationSize = reqs.size;
    memInfo.memoryTypeIndex = typeIndex;
    if (allocateMemory(device_, &memInfo, nullptr, &stagingMemory_) != VK_SUCCESS) {
        stagingMemory_ = VK_NULL_HANDLE;
        return false;
    }
    if (bindBufferMemory(device_, stagingBuffer_, stagingMemory_, 0) != VK_SUCCESS) return false;
    void* mapped = nullptr;
    if (mapMemory(device_, stagingMemory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) return false;
    stagingMapped_ = static_cast<uint8_t*>(mapped);
    stagingSize_ = bytes;
    LOGI("Staging de presentación: %llu bytes", static_cast<unsigned long long>(bytes));
    return true;
}

void DirectVulkanBridge::destroyUploadResourcesLocked() {
    if (device_ == VK_NULL_HANDLE) return;
    auto waitIdle = devProc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
    if (waitIdle != nullptr) waitIdle(device_);
    auto destroyBuffer = devProc<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    auto freeMemory = devProc<PFN_vkFreeMemory>("vkFreeMemory");
    auto destroyPool = devProc<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
    auto destroySemaphore = devProc<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    auto destroyFence = devProc<PFN_vkDestroyFence>("vkDestroyFence");
    if (stagingBuffer_ != VK_NULL_HANDLE && destroyBuffer != nullptr) destroyBuffer(device_, stagingBuffer_, nullptr);
    if (stagingMemory_ != VK_NULL_HANDLE && freeMemory != nullptr) freeMemory(device_, stagingMemory_, nullptr);
    if (commandPool_ != VK_NULL_HANDLE && destroyPool != nullptr) destroyPool(device_, commandPool_, nullptr);
    if (renderFinished_ != VK_NULL_HANDLE && destroySemaphore != nullptr) destroySemaphore(device_, renderFinished_, nullptr);
    if (submitFence_ != VK_NULL_HANDLE && destroyFence != nullptr) destroyFence(device_, submitFence_, nullptr);
    stagingBuffer_ = VK_NULL_HANDLE;
    stagingMemory_ = VK_NULL_HANDLE;
    stagingMapped_ = nullptr;
    stagingSize_ = 0;
    commandPool_ = VK_NULL_HANDLE;
    commandBuffer_ = VK_NULL_HANDLE;
    renderFinished_ = VK_NULL_HANDLE;
    submitFence_ = VK_NULL_HANDLE;
}

bool DirectVulkanBridge::presentPixels(const uint8_t* rgba, uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rgba == nullptr || width == 0 || height == 0) return false;
    if (device_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
        !swapchainTransferDst_) {
        return false;
    }
    const uint32_t ew = swapchainExtent_.width, eh = swapchainExtent_.height;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(ew) * eh * 4;
    if (!ensureUploadResourcesLocked(bytes)) {
        LOGW("presentPixels: recursos de subida no disponibles");
        return false;
    }
    auto waitForFences = devProc<PFN_vkWaitForFences>("vkWaitForFences");
    auto resetFences = devProc<PFN_vkResetFences>("vkResetFences");
    auto resetCmd = devProc<PFN_vkResetCommandBuffer>("vkResetCommandBuffer");
    auto beginCmd = devProc<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
    auto endCmd = devProc<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
    auto barrierCmd = devProc<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    auto copyCmd = devProc<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    auto queueSubmit = devProc<PFN_vkQueueSubmit>("vkQueueSubmit");
    auto queuePresent = devProc<PFN_vkQueuePresentKHR>("vkQueuePresentKHR");
    if (waitForFences == nullptr || resetFences == nullptr || resetCmd == nullptr || beginCmd == nullptr ||
        endCmd == nullptr || barrierCmd == nullptr || copyCmd == nullptr || queueSubmit == nullptr ||
        queuePresent == nullptr) {
        return false;
    }
    // La copia anterior debe haber terminado antes de reescribir el staging.
    if (waitForFences(device_, 1, &submitFence_, VK_TRUE, 1000000000ULL) != VK_SUCCESS) return false;

    // Escalado nearest + rotación de la superficie + swizzle BGRA si hace falta.
    const bool bgra = swapchainFormat_ == VK_FORMAT_B8G8R8A8_UNORM || swapchainFormat_ == VK_FORMAT_B8G8R8A8_SRGB;
    const bool rot90 = surfaceTransform_ == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR;
    const bool rot180 = surfaceTransform_ == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR;
    const bool rot270 = surfaceTransform_ == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
    const bool swapAxes = rot90 || rot270;
    const uint32_t dispW = swapAxes ? eh : ew;  // tamaño "visible" (orientación de pantalla)
    const uint32_t dispH = swapAxes ? ew : eh;
    if (!presentGeometryLogged_ || presentLoggedW_ != ew || presentLoggedH_ != eh) {
        presentGeometryLogged_ = true;
        presentLoggedW_ = ew;
        presentLoggedH_ = eh;
        LOGI("presentPixels: frame %ux%u -> swapchain %ux%u transform=0x%x visible %ux%u", width, height, ew, eh,
             static_cast<unsigned>(surfaceTransform_), dispW, dispH);
    }
    for (uint32_t y = 0; y < eh; ++y) {
        uint8_t* out = stagingMapped_ + static_cast<size_t>(y) * ew * 4;
        for (uint32_t x = 0; x < ew; ++x, out += 4) {
            // Coordenada en el espacio visible (dispW x dispH). Con rotación de 90/270 el buffer
            // es ew x eh y lo visible eh x ew: x recorre [0,ew) e y [0,eh), así que las inversas
            // usan ew para x y eh para y (antes estaban cruzadas: con ew != eh la imagen salía
            // desplazada y con filas/columnas repetidas por el clamp).
            uint32_t vx = x, vy = y;
            if (rot90) { vx = y; vy = ew - 1 - x; }
            else if (rot180) { vx = ew - 1 - x; vy = eh - 1 - y; }
            else if (rot270) { vx = eh - 1 - y; vy = x; }
            const uint32_t sx = std::min(width - 1, static_cast<uint32_t>(static_cast<uint64_t>(vx) * width / dispW));
            const uint32_t sy = std::min(height - 1, static_cast<uint32_t>(static_cast<uint64_t>(vy) * height / dispH));
            const uint8_t* p = rgba + (static_cast<size_t>(sy) * width + sx) * 4;
            if (bgra) { out[0] = p[2]; out[1] = p[1]; out[2] = p[0]; }
            else { out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; }
            out[3] = 255;
        }
    }

    uint32_t imageIndex = 0;
    if (!acquireImageLocked(imageIndex)) return false;
    VkImage image = swapchainImages_[imageIndex];

    resetCmd(commandBuffer_, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginCmd(commandBuffer_, &begin);
    VkImageMemoryBarrier toDst{};
    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDst.srcAccessMask = 0;
    toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = image;
    toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrierCmd(commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
               nullptr, 1, &toDst);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {ew, eh, 1};
    copyCmd(commandBuffer_, stagingBuffer_, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier toPresent = toDst;
    toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toPresent.dstAccessMask = 0;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrierCmd(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
               nullptr, 1, &toPresent);
    endCmd(commandBuffer_);

    resetFences(device_, 1, &submitFence_);
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable_;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinished_;
    VkResult result = queueSubmit(queue_, 1, &submit, submitFence_);
    if (result != VK_SUCCESS) {
        LOGW("vkQueueSubmit (presentPixels) falló: %d", static_cast<int>(result));
        return false;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished_;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;
    result = queuePresent(queue_, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR || result == VK_SUBOPTIMAL_KHR) {
        (void)recreateSwapchain();
        if (result != VK_SUBOPTIMAL_KHR) return false;
    } else if (result != VK_SUCCESS) {
        LOGW("vkQueuePresentKHR (presentPixels) falló: %d", static_cast<int>(result));
        return false;
    }
    countFrameLocked();
    return true;
}

bool DirectVulkanBridge::presentTestFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE) {
        return false;
    }

    auto acquireNextImage = devProc<PFN_vkAcquireNextImageKHR>("vkAcquireNextImageKHR");
    auto queuePresent = devProc<PFN_vkQueuePresentKHR>("vkQueuePresentKHR");
    auto waitForFences = devProc<PFN_vkWaitForFences>("vkWaitForFences");
    auto resetFences = devProc<PFN_vkResetFences>("vkResetFences");
    if (acquireNextImage == nullptr || queuePresent == nullptr || waitForFences == nullptr ||
        resetFences == nullptr || inFlightFence_ == VK_NULL_HANDLE ||
        imageAvailable_ == VK_NULL_HANDLE) {
        return false;
    }

    constexpr uint64_t kTimeoutNs = 1000000000ULL;

    VkResult result = VK_SUCCESS;
    // Solo se espera el fence si una adquisicion previa sigue pendiente: tras un acquire
    // fallido el fence queda sin senalizar y esperarlo bloquearia un segundo por frame.
    if (fencePending_) {
        result = waitForFences(device_, 1, &inFlightFence_, VK_TRUE, kTimeoutNs);
        if (result != VK_SUCCESS) {
            LOGW("vkWaitForFences fallo o expiro: %d", static_cast<int>(result));
            return false;
        }
        fencePending_ = false;
    }
    result = resetFences(device_, 1, &inFlightFence_);
    if (result != VK_SUCCESS) {
        LOGW("vkResetFences falló: %d", static_cast<int>(result));
        return false;
    }

    uint32_t imageIndex = 0;
    result = acquireNextImage(device_, swapchain_, kTimeoutNs, imageAvailable_, inFlightFence_,
                              &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
        LOGW("Swapchain obsoleto/superficie perdida al adquirir (%d); recreando",
             static_cast<int>(result));
        (void)recreateSwapchain();
        return false;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) return false;
    fencePending_ = true;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &imageAvailable_;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;

    result = queuePresent(queue_, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
        LOGW("Presentación descartada (%d); recreando swapchain", static_cast<int>(result));
        (void)recreateSwapchain();
        return false;
    }
    if (result == VK_SUBOPTIMAL_KHR) {
        (void)recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        LOGW("vkQueuePresentKHR falló: %d", static_cast<int>(result));
        return false;
    }

    totalFrames_.fetch_add(1, std::memory_order_relaxed);
    ++frameCounter_;
    const uint64_t now = nowNs();
    if (lastFpsCheckTime_ == 0) lastFpsCheckTime_ = now;
    const uint64_t elapsed = now - lastFpsCheckTime_;
    if (elapsed >= 1000000000ULL) {
        currentFps_.store(static_cast<float>(frameCounter_) * 1000000000.0f /
                              static_cast<float>(elapsed),
                          std::memory_order_relaxed);
        frameCounter_ = 0;
        lastFpsCheckTime_ = now;
    }
    return true;
}

void DirectVulkanBridge::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    detachWindowLocked();
    destroyUploadResourcesLocked();
    fsrPass_.shutdown();

    if (device_ != VK_NULL_HANDLE) {
        auto destroySemaphore = devProc<PFN_vkDestroySemaphore>("vkDestroySemaphore");
        auto destroyFence = devProc<PFN_vkDestroyFence>("vkDestroyFence");
        if (imageAvailable_ != VK_NULL_HANDLE && destroySemaphore != nullptr) {
            destroySemaphore(device_, imageAvailable_, nullptr);
        }
        if (inFlightFence_ != VK_NULL_HANDLE && destroyFence != nullptr) {
            destroyFence(device_, inFlightFence_, nullptr);
        }
        imageAvailable_ = VK_NULL_HANDLE;
        inFlightFence_ = VK_NULL_HANDLE;

        auto destroyDevice = devProc<PFN_vkDestroyDevice>("vkDestroyDevice");
        if (destroyDevice != nullptr) destroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    queue_ = VK_NULL_HANDLE;
    queueFamilyIndex_ = UINT32_MAX;

    if (instance_ != VK_NULL_HANDLE) {
        auto destroyInstance = instProc<PFN_vkDestroyInstance>("vkDestroyInstance");
        if (destroyInstance != nullptr) destroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    loader_.unload();
    getInstanceProcAddr_ = nullptr;
    getDeviceProcAddr_ = nullptr;
    physicalDevice_ = VK_NULL_HANDLE;
    gpuName_.clear();
    currentFps_.store(0.0f, std::memory_order_relaxed);
    totalFrames_.store(0, std::memory_order_relaxed);
    lastFpsCheckTime_ = 0;
    frameCounter_ = 0;
}

bool DirectVulkanBridge::gpuContext(GpuContext& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE || getDeviceProcAddr_ == nullptr) return false;
    out.instance = instance_;
    out.physicalDevice = physicalDevice_;
    out.device = device_;
    out.queue = queue_;
    out.queueFamily = queueFamilyIndex_;
    out.getInstanceProcAddr = getInstanceProcAddr_;
    out.getDeviceProcAddr = getDeviceProcAddr_;
    out.textureCompressionBC = enabledFeatures_.textureCompressionBC == VK_TRUE;
    out.samplerAnisotropy = enabledFeatures_.samplerAnisotropy == VK_TRUE;
    out.independentBlend = enabledFeatures_.independentBlend == VK_TRUE;
    return true;
}

bool DirectVulkanBridge::submit(VkCommandBuffer cmd, VkFence fence) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto queueSubmit = devProc<PFN_vkQueueSubmit>("vkQueueSubmit");
    if (queueSubmit == nullptr || queue_ == VK_NULL_HANDLE) return false;
    VkSubmitInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    s.commandBufferCount = 1;
    s.pCommandBuffers = &cmd;
    return queueSubmit(queue_, 1, &s, fence) == VK_SUCCESS;
}

bool DirectVulkanBridge::beginPresent(SwapTarget& out) {
    mutex_.lock();
    if (device_ == VK_NULL_HANDLE || swapchain_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE ||
        swapchainViews_.empty()) {
        mutex_.unlock();
        return false;
    }
    // Semáforo renderFinished_ y pool de comandos (se crean con los recursos de subida).
    if (renderFinished_ == VK_NULL_HANDLE && !ensureUploadResourcesLocked(4096)) {
        mutex_.unlock();
        return false;
    }
    uint32_t index = 0;
    if (!acquireImageLocked(index) || index >= swapchainImages_.size()) {
        mutex_.unlock();
        return false;
    }
    out.image = swapchainImages_[index];
    out.view = swapchainViews_[index];
    out.extent = swapchainExtent_;
    out.format = swapchainFormat_;
    out.transform = surfaceTransform_;
    out.index = index;
    out.generation = swapchainGeneration_;
    return true;  // el mutex sigue tomado hasta endPresent
}

bool DirectVulkanBridge::endPresent(const SwapTarget& target, VkCommandBuffer cmd, VkFence fence, bool countFrame) {
    auto queueSubmit = devProc<PFN_vkQueueSubmit>("vkQueueSubmit");
    auto queuePresent = devProc<PFN_vkQueuePresentKHR>("vkQueuePresentKHR");
    bool ok = queueSubmit != nullptr && queuePresent != nullptr;
    if (ok) {
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo s{};
        s.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        s.waitSemaphoreCount = 1;
        s.pWaitSemaphores = &imageAvailable_;
        s.pWaitDstStageMask = &waitStage;
        s.commandBufferCount = 1;
        s.pCommandBuffers = &cmd;
        s.signalSemaphoreCount = 1;
        s.pSignalSemaphores = &renderFinished_;
        VkResult r = queueSubmit(queue_, 1, &s, fence);
        if (r != VK_SUCCESS) {
            LOGW("vkQueueSubmit (renderizador GPU) falló: %d", static_cast<int>(r));
            ok = false;
        } else {
            VkPresentInfoKHR p{};
            p.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            p.waitSemaphoreCount = 1;
            p.pWaitSemaphores = &renderFinished_;
            p.swapchainCount = 1;
            p.pSwapchains = &swapchain_;
            p.pImageIndices = &target.index;
            r = queuePresent(queue_, &p);
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR || r == VK_SUBOPTIMAL_KHR) {
                (void)recreateSwapchain();
            } else if (r != VK_SUCCESS) {
                LOGW("vkQueuePresentKHR (renderizador GPU) falló: %d", static_cast<int>(r));
                ok = false;
            }
            if (countFrame) countFrameLocked();
        }
    }
    mutex_.unlock();
    return ok;
}

}  // namespace hyperplay