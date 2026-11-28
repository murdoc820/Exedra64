#pragma once

// Renderizador Direct3D 9 -> Vulkan (juegos de 32 bits).
//
// El dispositivo IDirect3DDevice9 (loader/D3D9Device32.cpp) guarda el estado y, en cada Draw/Clear/Present,
// lo pasa aquí como estructuras simples. Este renderizador:
// - refleja en la GPU texturas, superficies (render targets y depth-stencil) y búferes, subiéndolos cuando
//   cambia su generación de CPU (Unlock);
// - traduce los shaders SM3 a SPIR-V (D3D9ShaderToSpirv.h) y crea pipelines por combinación de estado;
// - graba los draws en un command buffer por frame con render passes por conjunto de destinos;
// - en Present dibuja el backbuffer en el swapchain (mismo pase de presentación que D3D11).
// Convenciones de enlace y de viewport: ver D3D9ShaderToSpirv.h.

#include <cstddef>
#include <cstdint>

namespace hyperplay::gpu9 {

// Textura o superficie D3D9 (o un subrecurso de ella como destino).
struct TexRef {
    const void* key = nullptr;      // identidad de la imagen en la GPU (textura o superficie suelta)
    uint32_t kind = 0;              // 0 2D, 1 cubo, 2 volumen
    uint32_t format = 0;            // D3DFORMAT
    uint32_t width = 1, height = 1, depth = 1, levels = 1;
    uint32_t usage = 0;             // D3DUSAGE (1 RENDERTARGET, 2 DEPTHSTENCIL)
    uint64_t generation = 0;        // cambia en cada Unlock de la CPU
    const uint8_t* const* data = nullptr;  // por subrecurso (cara * levels + nivel); entradas nulas = sin datos
    const uint32_t* subresourceBytes = nullptr;  // tamaño en bytes por subrecurso (para snapshot asíncrono)
    uint32_t subresources = 0;
    uint32_t level = 0, face = 0;   // subrecurso usado como destino
    bool valid() const { return key != nullptr; }
};

struct Stream {
    const void* key = nullptr;      // nullptr: datos de usuario (DrawPrimitiveUP), se copian siempre
    const uint8_t* data = nullptr;
    uint32_t size = 0;
    uint64_t generation = 0;
    uint32_t offset = 0, stride = 0, freq = 1;
    bool dynamic = false;
};

struct DrawState {
    const void* vsKey = nullptr;
    const void* psKey = nullptr;
    const uint32_t* vsTokens = nullptr;
    size_t vsCount = 0;
    const uint32_t* psTokens = nullptr;
    size_t psCount = 0;
    const float* vsF = nullptr;     // 256 * 4
    const int32_t* vsI = nullptr;   // 16 * 4
    const uint32_t* vsB = nullptr;  // 16
    const float* psF = nullptr;     // 224 * 4
    const int32_t* psI = nullptr;
    const uint32_t* psB = nullptr;
    uint64_t vsConstGen = 0;
    uint64_t psConstGen = 0;
    uint64_t rsGen = 0;
    uint64_t ssGen = 0;
    const uint8_t* decl = nullptr;  // D3DVERTEXELEMENT9[] (8 bytes) hasta D3DDECL_END
    Stream streams[16];
    Stream indices;
    uint32_t indexFormat = 101;     // D3DFMT_INDEX16 / INDEX32 (102)
    const uint32_t* renderState = nullptr;     // [256]
    const uint32_t (*samplerState)[14] = nullptr;  // [20][14]; 0-15 píxel, 16-19 vértice
    TexRef textures[20];
    TexRef targets[4];
    TexRef depth;
    uint32_t viewport[4] = {};      // X, Y, Width, Height
    float minZ = 0.0f, maxZ = 1.0f;
    int32_t scissor[4] = {};        // left, top, right, bottom
};

class Renderer {
public:
    static Renderer& shared();
    bool available();

    // primType D3DPRIMITIVETYPE. Indexado: first = StartIndex y baseVertex = BaseVertexIndex; si no, first =
    // StartVertex. instances = recuento de D3DSTREAMSOURCE_INDEXEDDATA (1 sin instancias).
    bool draw(const DrawState& s, uint32_t primType, uint32_t primCount, bool indexed, int32_t baseVertex,
              uint32_t first, uint32_t instances);
    // IDirect3DDevice9::Clear (rects = pares de D3DRECT o nullptr).
    void clear(const DrawState& s, uint32_t rectCount, const int32_t* rects, uint32_t flags, uint32_t color,
               float z, uint32_t stencil);
    // StretchRect entre superficies en la GPU (rects nullptr = entera). false si no aplica.
    bool stretchRect(const TexRef& src, const int32_t* srcRect, const TexRef& dst, const int32_t* dstRect,
                     uint32_t filter);
    // ColorFill sobre una superficie (rect nullptr = entera).
    void colorFill(const TexRef& dst, const int32_t* rect, uint32_t color);
    bool present(const TexRef& backbuffer);
    void release(const void* key);
    void sync();

    struct Stats {
        uint64_t draws = 0, failedDraws = 0, pipelines = 0, uploads = 0, presents = 0, clears = 0;
    };
    Stats stats() const;

private:
    Renderer() = default;
};

}  // namespace hyperplay::gpu9
