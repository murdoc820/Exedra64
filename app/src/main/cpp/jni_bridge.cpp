#include <jni.h>

#include <android/log.h>
#include <android/native_window_jni.h>

#include "audio/FmodStudioHost.h"
#include "loader/VulkanPassthrough.h"
#include "gfx/GuestOpenGl.h"

#include <cstdlib>
#include <exception>
#include <sys/system_properties.h>
#include <string>
#include <unordered_set>

#if defined(__aarch64__)
#include <malloc.h>
#include <sys/prctl.h>
#endif

#include "loader/PeHeaderInspector.h"
#include "loader/BionicPeLoader.h"
#include "loader/ManagedHost.h"
#include "loader/NativeDllInterceptor.h"
#include "cpu/OryonSiliconScheduler.h"
#include "loader/FramePacing.h"
#include "gfx/PresentOptions.h"
#include "gfx/HyperFrameGen.h"
#include "gfx/D3D11GpuRenderer.h"
#include "cpu/HyperAotCache.h"
#include "cpu/OryonCpuTranslator.h"
#include "gfx/DirectVulkanBridge.h"
#include "audio/DirectAAudioEngine.h"
#include "input/DirectXInputBridge.h"
#include "loader/Win32DisplayInfo.h"

#define LOG_TAG "HyperPlayNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

std::string toStdString(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

}  // namespace

#define HP_JNI(ret, name) \
    extern "C" JNIEXPORT ret JNICALL Java_com_murdoc820_hyperplay_engine_HyperPlayNative_##name

// Devuelve [status, machine, characteristics, subsystem, is64Bit, largeAddressAware, graphicsApis].
// El orden lo define HyperPlayNative.PE_* en Kotlin.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_murdoc820_hyperplay_engine_HyperPlayNative_nativeInspectPe(JNIEnv* env, jclass,
                                                                     jstring path) {
    const std::string file = toStdString(env, path);
    const hyperplay::PeInfo info = hyperplay::inspectPeFile(file);
    if (!info.ok()) {
        LOGW("PE inválido (%d): %s", static_cast<int>(info.status), file.c_str());
    }

    const jint values[] = {
        static_cast<jint>(info.status),
        static_cast<jint>(info.machine),
        static_cast<jint>(info.characteristics),
        static_cast<jint>(info.subsystem),
        info.is64Bit ? 1 : 0,
        info.largeAddressAware ? 1 : 0,
        static_cast<jint>(info.graphicsApis),
    };
    constexpr jsize count = sizeof(values) / sizeof(values[0]);
    jintArray result = env->NewIntArray(count);
    if (result != nullptr) env->SetIntArrayRegion(result, 0, count, values);
    return result;
}

// ---- Singleton global para OryonSiliconScheduler ----
namespace {
hyperplay::OryonSiliconScheduler& scheduler() {
    static hyperplay::OryonSiliconScheduler s;
    return s;
}
}  // namespace

// ---- Scheduler Oryon ----
HP_JNI(jboolean, nativePinGameThread)(JNIEnv*, jclass, jint pid) {
    return scheduler().pinToPrimeCores(static_cast<pid_t>(pid)) ? JNI_TRUE : JNI_FALSE;
}
HP_JNI(jboolean, nativePinWorkerThread)(JNIEnv*, jclass, jint pid) {
    return scheduler().pinToPerfCores(static_cast<pid_t>(pid)) ? JNI_TRUE : JNI_FALSE;
}
HP_JNI(jboolean, nativeSetupAdpf)(JNIEnv*, jclass, jint tid, jlong targetNs) {
    return scheduler().setupAdpf(static_cast<pid_t>(tid), static_cast<int64_t>(targetNs)) ? JNI_TRUE : JNI_FALSE;
}
HP_JNI(jstring, nativeCoreTopology)(JNIEnv* env, jclass) {
    return env->NewStringUTF(hyperplay::OryonSiliconScheduler::topologyInfo().c_str());
}

// =========================================================================
// ---- Pure Bionic Direct-Silicon Engine (ZVDS) JNI Endpoints ----
// =========================================================================
namespace {
hyperplay::BionicPeLoader& directLoader() {
    static hyperplay::BionicPeLoader l;
    return l;
}
std::unique_ptr<hyperplay::LoadedPeImage>& directImage() {
    static std::unique_ptr<hyperplay::LoadedPeImage> img;
    return img;
}
hyperplay::OryonCpuTranslator& directTranslator() {
    static hyperplay::OryonCpuTranslator t;
    return t;
}
hyperplay::DirectVulkanBridge& directGfx() {
    return hyperplay::DirectVulkanBridge::shared();
}
hyperplay::DirectAAudioEngine& directAudio() {
    static hyperplay::DirectAAudioEngine a;
    return a;
}
hyperplay::HyperAotCache& directAot() {
    static hyperplay::HyperAotCache c;
    return c;
}
std::string& directLoadReport() {
    static std::string report;
    return report;
}
}  // namespace

static jboolean loadPeInternal(const std::string& path);

HP_JNI(jboolean, nativeDirectLoadPe)(JNIEnv* env, jclass, jstring exePath) {
    hyperplay::managed::reset();
    return loadPeInternal(toStdString(env, exePath));
}

// Juego .NET (ManagedHost.h): carga mono.exe del runtime con D:\ = carpeta del juego.
HP_JNI(jboolean, nativeDirectLoadManaged)(JNIEnv* env, jclass, jstring gameExePath, jstring runtimeDir) {
    const std::string game = toStdString(env, gameExePath);
    const std::string runtime = toStdString(env, runtimeDir);
    const size_t slash = game.find_last_of('/');
    if (slash == std::string::npos || runtime.empty()) {
        directLoadReport() = "juego .NET: ruta del juego o del runtime vacía";
        return JNI_FALSE;
    }
    hyperplay::managed::g_runtimeDir = runtime;
    hyperplay::managed::g_gameDir = game.substr(0, slash);
    hyperplay::managed::g_gameExe = game.substr(slash + 1);
    LOGI("nativeDirectLoadManaged: %s con Mono de %s", game.c_str(), runtime.c_str());
    const jboolean ok = loadPeInternal(runtime + "/bin/mono.exe");
    if (!ok) hyperplay::managed::reset();
    return ok;
}

static jboolean loadPeInternal(const std::string& path) {
    directLoadReport().clear();
    if (path.empty()) {
        directLoadReport() = "ruta vacía";
        LOGW("nativeDirectLoadPe: ruta vacía");
        return JNI_FALSE;
    }
    // Sin el backend de traducción Oryon registrado no existe forma de ejecutar código x86:
    // devolver éxito aquí dejaría a GameLauncher esperando un proceso que nunca arranca.
    if (!directTranslator().hasBackend()) {
        directLoadReport() = "sin backend de traducción Oryon registrado";
        LOGW("nativeDirectLoadPe: sin backend de traducción registrado, no se carga %s", path.c_str());
        return JNI_FALSE;
    }

    try {
        directTranslator().stop();
        directImage() = directLoader().loadExecutable(path);
    } catch (const std::exception& e) {
        directLoadReport() = std::string("excepción del cargador PE: ") + e.what();
        LOGW("nativeDirectLoadPe: excepción cargando %s: %s", path.c_str(), e.what());
        directImage().reset();
        return JNI_FALSE;
    }

    if (!directImage() || !directImage()->isValid()) {
        directLoadReport() = "imagen PE inválida o sin mapear";
        LOGW("nativeDirectLoadPe: imagen PE inválida o sin mapear (%s)", path.c_str());
        directImage().reset();
        return JNI_FALSE;
    }

    // Un import sin resolver deja en la IAT un RVA de la tabla de nombres: ejecutar con slots
    // pendientes salta a memoria inválida, por eso el fallo se propaga en lugar de ignorarse.
    const hyperplay::IatHijackStats iat = hyperplay::NativeDllInterceptor::hijackIatChecked(*directImage());
    if (iat.unresolved != 0 || iat.outOfImage != 0) {
        std::string report = "IAT incompleta: " + std::to_string(iat.hijacked) + "/" +
                             std::to_string(iat.total) + " resueltas, " +
                             std::to_string(iat.unresolved) + " sin resolver, " +
                             std::to_string(iat.outOfImage) + " fuera de imagen";
        if (!iat.unresolvedImports.empty()) {
            report += "\nImports pendientes:";
            for (const auto& name : iat.unresolvedImports) {
                report += "\n  " + name;
            }
        }
        directLoadReport() = report;
        LOGW("nativeDirectLoadPe: %s", report.c_str());
        directImage().reset();
        return JNI_FALSE;
    }

    if (!directTranslator().init(directImage().get())) {
        directLoadReport() = "el traductor Oryon rechazó la imagen";
        LOGW("nativeDirectLoadPe: el traductor rechazó la imagen");
        directImage().reset();
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

HP_JNI(jint, nativeDirectLaunch)(JNIEnv*, jclass) {
    // Este hilo ("hyperplay-direct") es el hilo principal del juego: ancla Prime + sesión ADPF.
    hyperplay::pacing::setMainThread(static_cast<pid_t>(gettid()));
    if (!directImage() || !directImage()->isValid()) {
        directLoadReport() = "nativeDirectLaunch: no hay imagen cargada";
        LOGW("nativeDirectLaunch: no hay imagen cargada");
        return -1;
    }
    if (!directTranslator().hasBackend()) {
        directLoadReport() = "nativeDirectLaunch: no hay backend de traducción registrado";
        LOGW("nativeDirectLaunch: no hay backend de traducción registrado");
        return -1;
    }

    // Inicializa cada DLL local llamando a su DllMain(DLL_PROCESS_ATTACH), dependencias primero
    // (initOrder: orden en que terminaron de cargarse); luego cualquier otra que quede.
    std::unordered_set<void*> initializedBases;
    std::vector<std::pair<std::string, hyperplay::LocalDllRegion>> ordered;
    for (const auto& n : directImage()->initOrder) {
        if (auto it = directImage()->localDlls.find(n); it != directImage()->localDlls.end()) ordered.emplace_back(*it);
    }
    for (const auto& kv : directImage()->localDlls) ordered.emplace_back(kv);
    for (const auto& [name, region] : ordered) {
        if (region.peInfo.entryPointRva != 0 && region.base != nullptr) {
            if (!initializedBases.insert(region.base).second) {
                continue; // Evita inicializar el mismo módulo múltiples veces por sus alias
            }
            void* dllEntry = static_cast<uint8_t*>(region.base) + region.peInfo.entryPointRva;
            LOGI("Inicializando DllMain para %s en %p (RVA 0x%x)",
                 name.c_str(), dllEntry, region.peInfo.entryPointRva);
            const int64_t dllRes = directTranslator().callDllEntry(dllEntry, region.base);
            LOGI("DllMain de %s finalizo con resultado: %ld", name.c_str(), static_cast<long>(dllRes));
            // EXPERIMENTO (setprop debug.hyperplay.nosse2copy 1): msvcr100 x86 sin su copia SSE2
            // (__sse2_available = 0 en RVA 0xB4638): memcpy/memset usan rep movsd.
            if (name == "msvcr100.dll") {
                char nb[PROP_VALUE_MAX] = {};
                if (__system_property_get("debug.hyperplay.nosse2copy", nb) > 0 && nb[0] == '1') {
                    auto* flag = static_cast<uint8_t*>(region.base) + 0xB4638;
                    const uint32_t zero = 0;
                    std::memcpy(flag, &zero, 4);
                    LOGW("msvcr100: copia SSE2 desactivada (experimento)");
                }
            }
            if (dllRes < 0) {
                LOGW("DllMain de %s devolvio codigo %ld (continuando ejecucion)",
                     name.c_str(), static_cast<long>(dllRes));
            }
        }
    }

    const int64_t res = directTranslator().execute(directImage()->entryPoint);
    if (res < 0) {
        directLoadReport() = "execute fallo (" + std::to_string(res) + "): " + directTranslator().lastError();
    }
    return static_cast<jint>(res);
}

HP_JNI(void, nativeDirectStop)(JNIEnv*, jclass) {
    // Primero el audio (si no, al despausar el bombeo reabre AAudio y el juego se sigue oyendo
    // en el launcher); después se despausa para que el hilo bloqueado en su Present pueda salir.
    hyperplay::NativeDllInterceptor::stopAllAudio();
    hyperplay::pacing::setPaused(false);  // un juego en Quick Resume está bloqueado en su Present
    directTranslator().stop();
}

// Quick Resume: congela el juego en su siguiente Present (y el audio) o lo reanuda.
HP_JNI(void, nativeSetPaused)(JNIEnv*, jclass, jboolean paused) {
    hyperplay::pacing::setPaused(paused == JNI_TRUE);
    hyperplay::fmodhost::setOutputPaused(paused == JNI_TRUE);  // FMOD nativo (Celeste)
}

HP_JNI(jstring, nativeDirectGetLastLoadReport)(JNIEnv* env, jclass) {
    return env->NewStringUTF(directLoadReport().c_str());
}

HP_JNI(jlongArray, nativeDirectPrecompileAot)(JNIEnv* env, jclass, jstring exePath, jstring savesDir) {
    const std::string exe = toStdString(env, exePath);
    const std::string saves = toStdString(env, savesDir);
    hyperplay::AotCacheStats stats = directAot().precompile(exe, saves);

    jlong values[] = {
        static_cast<jlong>(stats.blocksScanned),
        static_cast<jlong>(stats.blocksCached),
        static_cast<jlong>(stats.textSizeBytes),
        stats.cacheHit ? 1L : 0L,
    };
    jlongArray res = env->NewLongArray(4);
    if (res) env->SetLongArrayRegion(res, 0, 4, values);
    return res;
}

HP_JNI(void, nativeDirectSetFsrEnabled)(JNIEnv*, jclass, jboolean enabled) {
    directGfx().setFsrEnabled(enabled == JNI_TRUE);
}

HP_JNI(jboolean, nativeDirectIsFsrEnabled)(JNIEnv*, jclass) {
    return directGfx().isFsrEnabled() ? JNI_TRUE : JNI_FALSE;
}

HP_JNI(jlongArray, nativeDirectGetTelemetry)(JNIEnv* env, jclass) {
    hyperplay::InterceptTelemetry t = hyperplay::NativeDllInterceptor::getTelemetry();
    jlong values[] = {
        static_cast<jlong>(t.kernel32Calls),
        static_cast<jlong>(t.ntdllCalls),
        static_cast<jlong>(t.d3dCalls),
        static_cast<jlong>(t.vulkanCalls),
        static_cast<jlong>(t.xinputCalls),
        static_cast<jlong>(t.xaudioCalls),
        static_cast<jlong>(t.steamCalls),
        static_cast<jlong>(t.otherCalls),
    };
    jlongArray res = env->NewLongArray(8);
    if (res) env->SetLongArrayRegion(res, 0, 8, values);
    return res;
}

HP_JNI(jobjectArray, nativeDirectGetImports)(JNIEnv* env, jclass, jstring exePath) {
    const std::string path = toStdString(env, exePath);
    hyperplay::PeInfo info = hyperplay::inspectPeFile(path);
    if (!info.ok() || info.imports.empty()) return nullptr;

    jclass strCls = env->FindClass("java/lang/String");
    if (strCls == nullptr) {
        LOGW("nativeDirectGetImports: no se encontró java/lang/String");
        return nullptr;
    }
    jobjectArray arr = env->NewObjectArray(static_cast<jsize>(info.imports.size()), strCls, nullptr);
    env->DeleteLocalRef(strCls);
    if (arr == nullptr) return nullptr;

    for (size_t i = 0; i < info.imports.size(); ++i) {
        jstring s = env->NewStringUTF(info.imports[i].name.c_str());
        if (s == nullptr) continue;
        env->SetObjectArrayElement(arr, static_cast<jsize>(i), s);
        env->DeleteLocalRef(s);
    }
    return arr;
}

// Estado XInput completo del mando 0 (XInputPadBridge -> DirectXInputBridge, leído por XInputGetState).
HP_JNI(void, nativeDirectSetGamepadState)(JNIEnv*, jclass, jint buttons, jint lt, jint rt,
                                          jint lx, jint ly, jint rx, jint ry) {
    hyperplay::DirectXInputBridge::setButtons(static_cast<uint16_t>(buttons));
    hyperplay::DirectXInputBridge::setTriggers(static_cast<uint8_t>(lt), static_cast<uint8_t>(rt));
    hyperplay::DirectXInputBridge::setThumbL(static_cast<int16_t>(lx), static_cast<int16_t>(ly));
    hyperplay::DirectXInputBridge::setThumbR(static_cast<int16_t>(rx), static_cast<int16_t>(ry));
}

// Rumble pedido por el juego (XInputSetState) en el mando 0: (izquierdo << 16) | derecho.
HP_JNI(jint, nativeDirectGetRumble)(JNIEnv*, jclass) {
    const hyperplay::XInputVibrationState v = hyperplay::DirectXInputBridge::getVibration(0);
    return static_cast<jint>((static_cast<uint32_t>(v.wLeftMotorSpeed) << 16) | v.wRightMotorSpeed);
}

HP_JNI(jboolean, nativeDirectInitVulkan)(JNIEnv* env, jclass, jstring customDriver) {
    return directGfx().init(toStdString(env, customDriver)) ? JNI_TRUE : JNI_FALSE;
}

// Resolución nativa y frecuencia de la pantalla del teléfono, antes de lanzar el juego.
// Escala de render elegida en el Quick Menu (se aplica al iniciar la sesión).
static int g_renderScale = 100;

HP_JNI(void, nativeSetRenderScale)(JNIEnv*, jclass, jint percent) {
    g_renderScale = (percent >= 10 && percent <= 100) ? percent : 100;
}

// Modo eficiente: tope de FPS en caliente (>0 FPS, 0 sin límite) y protección térmica.
HP_JNI(void, nativeSetFpsCap)(JNIEnv*, jclass, jint fps) { hyperplay::pacing::setFpsCap(fps); }
HP_JNI(void, nativeSetThermalRelax)(JNIEnv*, jclass, jboolean relax) {
    hyperplay::pacing::setThermalRelax(relax == JNI_TRUE);
}
HP_JNI(jint, nativeGetEffectiveFpsCap)(JNIEnv*, jclass) { return hyperplay::pacing::effectiveCap(); }

// Reescalado del pase de presentación (0 auto, 1 SGSR v1, 2 bilineal) y nitidez de SGSR.
HP_JNI(void, nativeSetScreenFit)(JNIEnv*, jclass, jint mode) {
    hyperplay::gpu::presentopt::g_fit.store(mode >= 0 && mode <= 3 ? mode : 0);
}

HP_JNI(void, nativeSetUpscaler)(JNIEnv*, jclass, jint mode) {
    hyperplay::gpu::presentopt::g_upscaler.store(mode >= 0 && mode <= 2 ? mode : 0);
}
// HyperFG (generación de fotogramas 2x): interruptor en caliente y contadores.
HP_JNI(void, nativeSetFrameGen)(JNIEnv*, jclass, jboolean on) {
    hyperplay::gpu::HyperFrameGen::setEnabled(on == JNI_TRUE);
}
HP_JNI(jboolean, nativeIsFrameGenActive)(JNIEnv*, jclass) {
    return hyperplay::gpu::HyperFrameGen::enabled() ? JNI_TRUE : JNI_FALSE;
}
// Draws D3D11 acumulados en la GPU: los frames negros de carga de Unity solo limpian (0 draws).
// La pantalla de carga de la app se quita cuando el juego empieza a dibujar de verdad.
HP_JNI(jlong, nativeDirectDrawCount)(JNIEnv*, jclass) {
    // Draws de D3D11 + fotogramas presentados por Vulkan (juegos con Vulkan propio: Celeste).
    return static_cast<jlong>(hyperplay::gpu::Renderer::shared().stats().draws + hyperplay::vkpass::presentCount() +
                              hyperplay::guestgl::presentCount());
}

HP_JNI(jlongArray, nativeGetFrameGenStats)(JNIEnv* env, jclass) {
    const auto s = hyperplay::gpu::HyperFrameGen::shared().stats();
    const jlong v[4] = {static_cast<jlong>(s.generated), static_cast<jlong>(s.skippedSlow),
                        static_cast<jlong>(s.realPresents), static_cast<jlong>(s.cancelled)};
    jlongArray out = env->NewLongArray(4);
    if (out != nullptr) env->SetLongArrayRegion(out, 0, 4, v);
    return out;
}

HP_JNI(void, nativeSetSgsrSharpness)(JNIEnv*, jclass, jfloat s) {
    hyperplay::gpu::presentopt::g_sharpness.store(s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s));
}

HP_JNI(void, nativeSetDisplayMode)(JNIEnv*, jclass, jint width, jint height, jint hz) {
    // Escala del Quick Menu; `setprop debug.hyperplay.resscale <10-100>` tiene prioridad (A/B).
    char scaleText[PROP_VALUE_MAX] = {};
    int scale = __system_property_get("debug.hyperplay.resscale", scaleText) > 0 ? std::atoi(scaleText) : g_renderScale;
    if (scale < 10 || scale > 100) scale = 100;
    width = width * scale / 100;
    height = height * scale / 100;
    hyperplay::win32::setDisplayMode(width, height, hz > 0 ? static_cast<uint32_t>(hz) : 0u);
    __android_log_print(ANDROID_LOG_INFO, "HyperPlay.JNI", "Modo de pantalla del juego: %dx%d @ %d Hz",
                        hyperplay::win32::screenWidth(), hyperplay::win32::screenHeight(),
                        static_cast<int>(hyperplay::win32::refreshHz()));
}

HP_JNI(jboolean, nativeDirectAttachSurface)(JNIEnv* env, jclass, jobject surface) {
    if (surface == nullptr) return JNI_FALSE;
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window == nullptr) return JNI_FALSE;
    // El modo del escritorio lo fija nativeSetDisplayMode (Kotlin, antes de lanzar el juego).
    const bool ok = directGfx().attachWindow(window);
    ANativeWindow_release(window);
    return ok ? JNI_TRUE : JNI_FALSE;
}

HP_JNI(void, nativeDirectDetachSurface)(JNIEnv*, jclass) {
    directGfx().detachWindow();
}

HP_JNI(jboolean, nativeDirectPresentTestFrame)(JNIEnv*, jclass) {
    return directGfx().presentTestFrame() ? JNI_TRUE : JNI_FALSE;
}

HP_JNI(jfloat, nativeDirectPresentFps)(JNIEnv*, jclass) {
    return directGfx().presentFps();
}

// FPS reales del juego (llamadas a paceFrame), sin repeticiones ni sintéticos de HyperFG.
HP_JNI(jfloat, nativeGameFps)(JNIEnv*, jclass) {
    return hyperplay::pacing::gameFps();
}

HP_JNI(jlong, nativeDirectFramesPresented)(JNIEnv*, jclass) {
    return static_cast<jlong>(directGfx().totalFramesPresented());
}

HP_JNI(jboolean, nativeDirectStartAudio)(JNIEnv*, jclass, jint rate, jint channels) {
    return directAudio().start(rate, channels) ? JNI_TRUE : JNI_FALSE;
}

HP_JNI(void, nativeDirectStopAudio)(JNIEnv*, jclass) {
    directAudio().stop();
    hyperplay::fmodhost::setOutputPaused(true);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void)vm;
    (void)reserved;
#if defined(__aarch64__)
    // Solo se dejan de etiquetar las asignaciones NUEVAS. NO se desactiva el ABI de direcciones
    // etiquetadas (prctl(PR_SET_TAGGED_ADDR_CTRL, 0)): ART y otras bibliotecas ya tienen punteros
    // con etiqueta en el byte alto (0xb4...) y el kernel los rechazaría en syscalls. Eso cerraba
    // la app al azar: futex de ART con EFAULT -> "futex wait failed ... Bad address" -> SIGABRT.
#if defined(M_BIONIC_SET_HEAP_TAGGING_LEVEL)
    mallopt(M_BIONIC_SET_HEAP_TAGGING_LEVEL, M_HEAP_TAGGING_LEVEL_NONE);
#endif
#endif
    LOGI("JNI_OnLoad: Heap pointer tagging deshabilitado para x86/x64 Bionic engine");
    return JNI_VERSION_1_6;
}


