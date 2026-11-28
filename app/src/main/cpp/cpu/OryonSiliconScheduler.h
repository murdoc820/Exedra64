#pragma once

#include <sched.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace hyperplay {

struct CoreTopology {
    int primeCores[2] = {-1, -1};
    int perfCores[6] = {-1, -1, -1, -1, -1, -1};
    int primeCount = 0;
    int perfCount = 0;
    int cpuCount = 0;
    bool isOryon = false;
    int64_t primeMaxKhz = 0;
    int64_t perfMaxKhz = 0;
};

class OryonSiliconScheduler {
public:
    OryonSiliconScheduler() noexcept;
    ~OryonSiliconScheduler() noexcept;

    OryonSiliconScheduler(const OryonSiliconScheduler&) = delete;
    OryonSiliconScheduler& operator=(const OryonSiliconScheduler&) = delete;

    /// Lee la topología real de /sys/devices/system/cpu y ordena los núcleos por frecuencia máxima.
    [[nodiscard]] static CoreTopology detect() noexcept;

    /// Fija el hilo `tid` (0 = hilo actual) a los núcleos Prime detectados.
    [[nodiscard]] bool pinToPrimeCores(pid_t tid) noexcept;

    /// Fija el hilo `tid` (0 = hilo actual) a los núcleos Performance detectados.
    [[nodiscard]] bool pinToPerfCores(pid_t tid) noexcept;

    /// Restaura la afinidad completa de la CPU.
    [[nodiscard]] bool pinToAllCores(pid_t tid) noexcept;

    /// Crea una sesión ADPF (APerformanceHintManager) para el hilo indicado.
    [[nodiscard]] bool setupAdpf(pid_t tid, int64_t targetFrameNs = 8333333LL) noexcept;

    /// Cierra la sesión ADPF activa, si existe.
    void closeAdpf() noexcept;

    /// Sesión ADPF por frame para los hilos del juego (se crea una vez; idempotente).
    bool ensureFrameSession(const pid_t* tids, size_t count, int64_t targetNs) noexcept;

    /// Duración real del trabajo del frame frente al presupuesto del tope de FPS activo.
    void reportFrameWork(int64_t actualNs, int64_t targetNs) noexcept;

    void closeFrameSession() noexcept;

    /// Hilo principal: true = todos los núcleos (reparte calor), false = núcleos Prime.
    bool setPrimeRelaxed(pid_t tid, bool relaxed) noexcept;

    /// Establece prioridades en tiempo real (SCHED_FIFO / SCHED_RR).
    static void setThreadPriority(pid_t tid, int schedPolicy, int priority) noexcept;

    /// Devuelve información de la topología detectada para el HUD de consola.
    [[nodiscard]] static std::string topologyInfo() noexcept;

    [[nodiscard]] const CoreTopology& topology() const noexcept { return topology_; }

private:
    CoreTopology topology_;
    void* adpfSession_ = nullptr;
    std::mutex frameMutex_;
    void* frameSession_ = nullptr;
    size_t frameTidCount_ = 0;
    bool frameSessionFailed_ = false;
    int64_t frameTargetNs_ = 0;
};

}  // namespace hyperplay
