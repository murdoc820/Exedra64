#pragma once

// Traductor de shaders Direct3D 9 (vs_3_0 / ps_3_0, bytecode de fichas de 32 bits) a SPIR-V 1.0 para Vulkan.
//
// Modelo:
// - Registros temporales r#, a0, aL y p0 como variables privadas; salidas o# / oC# / oDepth como Output.
// - Constantes: un bloque uniforme por etapa (set 0, binding 0 = VS, 1 = PS) con
//   { vec4 c[256]; ivec4 i[16]; uvec4 b[16]; vec4 alphaTest; }. Los `def`/`defi`/`defb` del propio shader se
//   usan como literales; con direccionamiento relativo se lee el bloque y el renderizador copia antes los
//   `def` en él (ShaderInfo::floatDefs).
// - Samplers combinados: PS s# -> binding 2 + #, VS s# -> binding 18 + #.
// - Enlace VS -> PS por semántica (uso, índice) -> Location (varyingLocation).
// - La inversión del eje Y y el desplazamiento de medio píxel de D3D9 los aplica el viewport del renderizador
//   (altura negativa y +0.5), no el shader. vPos sí se corrige aquí (FragCoord - 0.5).
// - La prueba alfa de función fija (D3DRS_ALPHATEST*) se emula al final del PS con alphaTest = {func, ref}.
//
// Sin dependencias externas; errores con mensaje (instrucción no soportada) para que el renderizador no dibuje.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "SpirvBuilder.h"

namespace hyperplay::d3d9sh {

// Ubicación de un valor interpolado según su semántica D3DDECLUSAGE.
inline uint32_t varyingLocation(uint32_t usage, uint32_t index) {
    switch (usage) {
        case 5: return index < 10 ? index : 31;        // TEXCOORD0..9
        case 10: return 10 + (index & 1);              // COLOR0/1
        case 11: return 12;                            // FOG
        case 3: return 13 + (index & 1);               // NORMAL
        case 6: return 15 + (index & 1);               // TANGENT
        case 7: return 17 + (index & 1);               // BINORMAL
        case 0: return 19 + (index & 3);               // POSITION1..3 (POSITION0 de VS = gl_Position)
        case 12: return 23;                            // DEPTH
        case 1: return 24;                             // BLENDWEIGHT
        case 2: return 25;                             // BLENDINDICES
        default: return 26 + (usage & 3);
    }
}

struct ShaderInfo {
    bool vertex = false;
    uint32_t major = 0, minor = 0;
    struct Input { uint32_t reg, usage, index; };
    std::vector<Input> inputs;   // VS: v# con su semántica (Location = v#)
    uint32_t samplerMask = 0;    // s# usados
    uint8_t samplerType[20] = {};  // 2 = 2D, 3 = cubo, 4 = volumen
    struct FloatDef { uint32_t reg; float v[4]; };
    std::vector<FloatDef> floatDefs;
    bool relativeConst = false;  // c[a0/aL + n]: el renderizador debe copiar los def en el bloque
    uint16_t maxFloatConst = 0;  // número de vec4 en c[] realmente leídos (256 si relativeConst)
    bool usesIntConst = false;
    bool usesBoolConst = false;
    bool writesDepth = false;
    uint32_t colorOutputs = 0;   // máscara oC#
};

namespace detail {

using namespace hyperplay::spv;

enum RegType : uint32_t {
    kTemp = 0, kInput = 1, kConst = 2, kAddr = 3, kRastOut = 4, kAttrOut = 5, kOutput = 6, kConstInt = 7,
    kColorOut = 8, kDepthOut = 9, kSampler = 10, kConstBool = 14, kLoop = 15, kMisc = 17, kLabel = 18, kPredicate = 19,
};

struct Param {
    uint32_t token = 0;
    uint32_t type = 0, num = 0;
    bool relative = false;
    uint32_t relToken = 0;  // registro de dirección (a0 / aL) con su swizzle
    uint32_t swizzle() const { return (token >> 16) & 0xFF; }
    uint32_t srcMod() const { return (token >> 24) & 0xF; }
    uint32_t mask() const { return (token >> 16) & 0xF; }
    uint32_t resultMod() const { return (token >> 20) & 0xF; }
};

inline uint32_t regType(uint32_t t) { return ((t >> 28) & 7) | ((t >> 8) & 0x18); }

class Translator {
public:
    Translator(const uint32_t* t, size_t n, bool alphaTest = true) : tok_(t), count_(n), alphaTest_(alphaTest) {}

    bool run(std::vector<uint32_t>& out, ShaderInfo& info, std::string& err) {
        if (count_ < 2) return fail("shader vacío");
        const uint32_t ver = tok_[0];
        info.vertex = (ver >> 16) == 0xFFFE;
        if (!info.vertex && (ver >> 16) != 0xFFFF) return fail("cabecera de versión inválida");
        info.major = (ver >> 8) & 0xFF;
        info.minor = ver & 0xFF;
        if (info.major != 3) return fail("solo shader model 3");
        vs_ = info.vertex;
        info_ = &info;
        if (!prepass()) { err = err_; return false; }
        setup();
        pos_ = 1;
        while (pos_ < count_ && ok_) {
            const uint32_t t = tok_[pos_];
            if (t == 0x0000FFFFu) break;
            const uint32_t op = t & 0xFFFF;
            if (op == 0xFFFE) { pos_ += 1 + ((t >> 16) & 0x7FFF); continue; }
            const uint32_t len = (t >> 24) & 0xF;
            const size_t next = pos_ + 1 + len;
            instruction(t, pos_ + 1, next);
            pos_ = next;
        }
        if (!ok_) { err = err_; return false; }
        if (!flow_.empty()) { err = "control de flujo sin cerrar"; return false; }
        epilogue();
        b_.op(OpReturn, {});
        out = b_.finish(vs_ ? 0u : 4u, mainId_, mainType_);
        return true;
    }

private:
    const uint32_t* tok_;
    size_t count_;
    size_t pos_ = 1;
    bool vs_ = false, ok_ = true;
    std::string err_;
    ShaderInfo* info_ = nullptr;
    Builder b_;
    uint32_t mainId_ = 0, mainType_ = 0;
    uint32_t tF_ = 0, tV4_ = 0, tI_ = 0, tIV4_ = 0, tU_ = 0, tUV4_ = 0, tB_ = 0, tBV4_ = 0;
    uint32_t pPrivV4_ = 0, pOutV4_ = 0, pInV4_ = 0, pUniV4_ = 0, pUniIV4_ = 0, pUniUV4_ = 0;
    uint32_t ubo_ = 0;
    std::map<uint32_t, uint32_t> temps_, outputs_, inputs_, colorOut_;
    std::map<uint32_t, uint32_t> sampVar_;
    std::map<uint32_t, uint32_t> sampImgType_;
    uint32_t addr_ = 0, loopReg_ = 0, pred_ = 0, depthOut_ = 0, posOut_ = 0, fragCoord_ = 0, frontFacing_ = 0;
    std::map<uint32_t, std::vector<float>> fdefs_;
    std::map<uint32_t, std::vector<int32_t>> idefs_;
    std::map<uint32_t, bool> bdefs_;
    struct Flow { int kind; uint32_t merge, cont, elseLbl, counter, step; bool hasElse; uint32_t header; };  // 0 if, 1 loop
    std::vector<Flow> flow_;

    bool fail(const std::string& m) { if (ok_) { ok_ = false; err_ = m; } return false; }

    // ---- pasada previa: declaraciones y defs --------------------------------------------------
    bool prepass() {
        size_t p = 1;
        while (p < count_) {
            const uint32_t t = tok_[p];
            if (t == 0x0000FFFFu) return true;
            const uint32_t op = t & 0xFFFF;
            if (op == 0xFFFE) { p += 1 + ((t >> 16) & 0x7FFF); continue; }
            const uint32_t len = (t >> 24) & 0xF;
            if (p + len >= count_) return fail("instrucción truncada");
            if (op == 81 && len >= 5) {  // def
                std::vector<float> v(4);
                std::memcpy(v.data(), &tok_[p + 2], 16);
                const uint32_t reg = tok_[p + 1] & 0x7FF;
                fdefs_[reg] = v;
                ShaderInfo::FloatDef d{reg, {v[0], v[1], v[2], v[3]}};
                info_->floatDefs.push_back(d);
            } else if (op == 48 && len >= 5) {  // defi
                std::vector<int32_t> v(4);
                std::memcpy(v.data(), &tok_[p + 2], 16);
                idefs_[tok_[p + 1] & 0x7FF] = v;
            } else if (op == 47 && len >= 2) {  // defb
                bdefs_[tok_[p + 1] & 0x7FF] = tok_[p + 2] != 0;
            }
            for (uint32_t k = 1; k <= len; ++k) {  // ¿constantes con índice relativo?
                const uint32_t s = tok_[p + k];
                if (op != 31 && op != 81 && op != 47 && op != 48 && (s & 0x80000000u)) {
                    const uint32_t rt = regType(s);
                    if (rt == kConst) {
                        if (s & (1u << 13)) {
                            info_->relativeConst = true;
                            info_->maxFloatConst = 256;
                        } else {
                            const uint16_t end = static_cast<uint16_t>((s & 0x7FFu) + 1u);
                            if (end > info_->maxFloatConst) info_->maxFloatConst = std::min<uint16_t>(end, 256u);
                        }
                    } else if (rt == kConstInt) {
                        info_->usesIntConst = true;
                    } else if (rt == kConstBool) {
                        info_->usesBoolConst = true;
                    }
                }
            }
            p += 1 + len;
        }
        return true;
    }

    // ---- tipos, bloque uniforme y función main -------------------------------------------------
    void setup() {
        b_.capability(1);  // Shader
        tF_ = b_.typeFloat();
        tV4_ = b_.typeVector(tF_, 4);
        tI_ = b_.typeInt(true);
        tIV4_ = b_.typeVector(tI_, 4);
        tU_ = b_.typeInt(false);
        tUV4_ = b_.typeVector(tU_, 4);
        tB_ = b_.typeBool();
        tBV4_ = b_.typeVector(tB_, 4);
        pPrivV4_ = b_.typePointer(ScPrivate, tV4_);
        pOutV4_ = b_.typePointer(ScOutput, tV4_);
        pInV4_ = b_.typePointer(ScInput, tV4_);
        pUniV4_ = b_.typePointer(ScUniform, tV4_);
        pUniIV4_ = b_.typePointer(ScUniform, tIV4_);
        pUniUV4_ = b_.typePointer(ScUniform, tUV4_);
        const uint32_t aC = b_.typeArrayUnique(tV4_, 256);
        const uint32_t aI = b_.typeArrayUnique(tIV4_, 16);
        const uint32_t aB = b_.typeArrayUnique(tUV4_, 16);
        b_.decorate(aC, DecArrayStride, {16});
        b_.decorate(aI, DecArrayStride, {16});
        b_.decorate(aB, DecArrayStride, {16});
        const uint32_t blk = b_.typeStruct({aC, aI, aB, tV4_});
        b_.decorate(blk, DecBlock);
        b_.memberDecorate(blk, 0, DecOffset, {0});
        b_.memberDecorate(blk, 1, DecOffset, {4096});
        b_.memberDecorate(blk, 2, DecOffset, {4096 + 256});
        b_.memberDecorate(blk, 3, DecOffset, {4096 + 512});
        ubo_ = b_.globalVar(b_.typePointer(ScUniform, blk), ScUniform);
        b_.decorate(ubo_, DecDescriptorSet, {0});
        b_.decorate(ubo_, DecBinding, {vs_ ? 0u : 1u});
        if (!vs_) b_.executionMode(7);  // OriginUpperLeft
        mainId_ = b_.newId();
        mainType_ = b_.typeFunction(b_.typeVoid());
        b_.beginMain();
    }

    uint32_t constV4(float x, float y, float z, float w) {
        return b_.constComposite(tV4_, {b_.constF(x), b_.constF(y), b_.constF(z), b_.constF(w)});
    }

    // ---- variables por registro ---------------------------------------------------------------
    uint32_t privateVar(std::map<uint32_t, uint32_t>& m, uint32_t n) {
        auto it = m.find(n);
        if (it != m.end()) return it->second;
        const uint32_t v = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
        m[n] = v;
        return v;
    }

    void declare(const uint32_t* t, size_t n) {
        if (n < 2) { fail("dcl truncado"); return; }
        const uint32_t usageTok = t[0], dst = t[1];
        const uint32_t type = regType(dst), num = dst & 0x7FF;
        const uint32_t usage = usageTok & 0x1F, uidx = (usageTok >> 16) & 0xF;
        if (type == kSampler) {
            const uint32_t st = (usageTok >> 27) & 0xF;
            if (num >= 16) { fail("sampler fuera de rango"); return; }
            const uint32_t dim = st == 3 ? DimCube : st == 4 ? Dim3D : Dim2D;
            const uint32_t img = b_.typeImage(tF_, dim, 0, false, 1);
            const uint32_t si = b_.typeSampledImage(img);
            const uint32_t var = b_.globalVar(b_.typePointer(ScUniformConstant, si), ScUniformConstant);
            b_.decorate(var, DecDescriptorSet, {0});
            b_.decorate(var, DecBinding, {(vs_ ? 18u : 2u) + num});
            sampVar_[num] = var;
            sampImgType_[num] = si;
            info_->samplerMask |= 1u << num;
            info_->samplerType[num] = static_cast<uint8_t>(st == 0 ? 2 : st);
            return;
        }
        if (type == kInput) {
            const uint32_t var = b_.globalVar(pInV4_, ScInput);
            if (vs_) {
                b_.decorate(var, DecLocation, {num});
                info_->inputs.push_back({num, usage, uidx});
            } else {
                b_.decorate(var, DecLocation, {varyingLocation(usage, uidx)});
            }
            inputs_[num] = var;
            return;
        }
        if (type == kOutput && vs_) {
            const uint32_t var = b_.globalVar(pOutV4_, ScOutput);
            if (usage == 0 && uidx == 0) {
                b_.decorate(var, DecBuiltIn, {BiPosition});
                posOut_ = var;
            } else if (usage == 4) {  // PSIZE: sin tamaño de punto programable; se descarta
                outputs_[num] = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
                return;
            } else {
                b_.decorate(var, DecLocation, {varyingLocation(usage, uidx)});
            }
            outputs_[num] = var;
            return;
        }
        if (type == kMisc) {  // vPos (0) / vFace (1)
            if (num == 0) {
                fragCoord_ = b_.globalVar(pInV4_, ScInput);
                b_.decorate(fragCoord_, DecBuiltIn, {BiFragCoord});
            } else {
                frontFacing_ = b_.globalVar(b_.typePointer(ScInput, tB_), ScInput);
                b_.decorate(frontFacing_, DecBuiltIn, {BiFrontFacing});
            }
            return;
        }
        // dcl de texturas de ps < 3 u otros: sin efecto
    }

    uint32_t colorOutput(uint32_t n) {
        auto it = colorOut_.find(n);
        if (it != colorOut_.end()) return it->second;
        const uint32_t var = b_.globalVar(pOutV4_, ScOutput);
        b_.decorate(var, DecLocation, {n});
        colorOut_[n] = var;
        info_->colorOutputs |= 1u << n;
        return var;
    }

    // Puntero (y tipo de puntero) del registro destino.
    uint32_t destVar(const Param& d) {
        switch (d.type) {
            case kTemp: return privateVar(temps_, d.num);
            case kAddr: if (addr_ == 0) addr_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0)); return addr_;
            case kPredicate: if (pred_ == 0) pred_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0)); return pred_;
            case kOutput:
            case kRastOut:
            case kAttrOut: {
                auto it = outputs_.find(d.num);
                if (it != outputs_.end()) return it->second;
                return privateVar(outputs_, d.num);
            }
            case kColorOut: return colorOutput(d.num);
            case kDepthOut:
                if (depthOut_ == 0) depthOut_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
                info_->writesDepth = true;
                return depthOut_;
            default: fail("registro destino no soportado (tipo " + std::to_string(d.type) + ")"); return 0;
        }
    }

    // ---- lectura de operandos -----------------------------------------------------------------
    uint32_t loadPtr(uint32_t ptr) { return b_.value(OpLoad, tV4_, {ptr}); }

    uint32_t uboVec(uint32_t member, uint32_t indexId, uint32_t ptrType, uint32_t valType) {
        const uint32_t p = b_.value(OpAccessChain, ptrType, {ubo_, b_.constI(static_cast<int32_t>(member)), indexId});
        return b_.value(OpLoad, valType, {p});
    }

    // Índice entero del registro de dirección (a0.c o aL).
    uint32_t relIndex(uint32_t relTok) {
        const uint32_t type = regType(relTok);
        const uint32_t comp = (relTok >> 16) & 3;
        uint32_t f = 0;
        if (type == kLoop) {
            if (loopReg_ == 0) loopReg_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
            f = b_.value(OpCompositeExtract, tF_, {loadPtr(loopReg_), 0});
        } else {
            if (addr_ == 0) addr_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
            f = b_.value(OpCompositeExtract, tF_, {loadPtr(addr_), comp});
        }
        return b_.value(OpConvertFToS, tI_, {f});
    }

    uint32_t rawSource(const Param& s) {
        switch (s.type) {
            case kTemp: return loadPtr(privateVar(temps_, s.num));
            case kInput: {
                if (s.relative) { fail("entrada con índice relativo"); return constV4(0, 0, 0, 0); }
                auto it = inputs_.find(s.num);
                if (it == inputs_.end()) return constV4(0, 0, 0, 0);
                return b_.value(OpLoad, tV4_, {it->second});
            }
            case kConst: {
                if (!s.relative) {
                    auto it = fdefs_.find(s.num);
                    if (it != fdefs_.end()) return constV4(it->second[0], it->second[1], it->second[2], it->second[3]);
                    return uboVec(0, b_.constI(static_cast<int32_t>(s.num)), pUniV4_, tV4_);
                }
                uint32_t idx = b_.value(OpIAdd, tI_, {relIndex(s.relToken), b_.constI(static_cast<int32_t>(s.num))});
                idx = b_.ext(tI_, GlslSClamp(), {idx, b_.constI(0), b_.constI(255)});
                return uboVec(0, idx, pUniV4_, tV4_);
            }
            case kConstInt: {
                auto it = idefs_.find(s.num);
                if (it != idefs_.end())
                    return constV4(static_cast<float>(it->second[0]), static_cast<float>(it->second[1]),
                                   static_cast<float>(it->second[2]), static_cast<float>(it->second[3]));
                const uint32_t iv = uboVec(1, b_.constI(static_cast<int32_t>(s.num & 15)), pUniIV4_, tIV4_);
                return b_.value(OpConvertSToF, tV4_, {iv});
            }
            case kAddr: if (addr_ == 0) addr_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0)); return loadPtr(addr_);
            case kLoop: if (loopReg_ == 0) loopReg_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0)); return loadPtr(loopReg_);
            case kPredicate: if (pred_ == 0) pred_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0)); return loadPtr(pred_);
            case kMisc: {
                if (s.num == 0) {
                    if (fragCoord_ == 0) { fail("vPos sin dcl"); return constV4(0, 0, 0, 0); }
                    const uint32_t fc = b_.value(OpLoad, tV4_, {fragCoord_});
                    return b_.value(OpFSub, tV4_, {fc, constV4(0.5f, 0.5f, 0, 0)});
                }
                if (frontFacing_ == 0) { fail("vFace sin dcl"); return constV4(0, 0, 0, 0); }
                const uint32_t ff = b_.value(OpLoad, tB_, {frontFacing_});
                const uint32_t f = b_.value(OpSelect, tF_, {ff, b_.constF(1.0f), b_.constF(-1.0f)});
                return b_.value(OpCompositeConstruct, tV4_, {f, f, f, f});
            }
            case kOutput: case kColorOut: case kRastOut: case kAttrOut: case kDepthOut:
                return loadPtr(destVar(s));
            default: fail("registro fuente no soportado (tipo " + std::to_string(s.type) + ")"); return constV4(0, 0, 0, 0);
        }
    }

    static uint32_t GlslSClamp() { return 45; }

    uint32_t swizzle(uint32_t v, uint32_t sw) {
        if (sw == 0xE4) return v;
        return b_.value(OpVectorShuffle, tV4_, {v, v, sw & 3, (sw >> 2) & 3, (sw >> 4) & 3, (sw >> 6) & 3});
    }

    uint32_t source(const Param& s) {
        uint32_t v = swizzle(rawSource(s), s.swizzle());
        switch (s.srcMod()) {
            case 0: break;
            case 1: v = b_.value(OpFNegate, tV4_, {v}); break;
            case 2: v = b_.value(OpFSub, tV4_, {v, constV4(0.5f, 0.5f, 0.5f, 0.5f)}); break;
            case 3: v = b_.value(OpFNegate, tV4_, {b_.value(OpFSub, tV4_, {v, constV4(0.5f, 0.5f, 0.5f, 0.5f)})}); break;
            case 4: v = b_.value(OpFSub, tV4_, {b_.value(OpFMul, tV4_, {v, constV4(2, 2, 2, 2)}), constV4(1, 1, 1, 1)}); break;
            case 5: v = b_.value(OpFNegate, tV4_, {b_.value(OpFSub, tV4_, {b_.value(OpFMul, tV4_, {v, constV4(2, 2, 2, 2)}), constV4(1, 1, 1, 1)})}); break;
            case 6: v = b_.value(OpFSub, tV4_, {constV4(1, 1, 1, 1), v}); break;
            case 7: v = b_.value(OpFMul, tV4_, {v, constV4(2, 2, 2, 2)}); break;
            case 8: v = b_.value(OpFNegate, tV4_, {b_.value(OpFMul, tV4_, {v, constV4(2, 2, 2, 2)})}); break;
            case 9: { const uint32_t z = b_.value(OpCompositeExtract, tF_, {v, 2}); v = b_.value(OpFDiv, tV4_, {v, splat(z)}); break; }
            case 10: { const uint32_t w = b_.value(OpCompositeExtract, tF_, {v, 3}); v = b_.value(OpFDiv, tV4_, {v, splat(w)}); break; }
            case 11: v = b_.ext(tV4_, GlslFAbs, {v}); break;
            case 12: v = b_.value(OpFNegate, tV4_, {b_.ext(tV4_, GlslFAbs, {v})}); break;
            case 13: v = b_.value(OpFSub, tV4_, {constV4(1, 1, 1, 1), v}); break;  // NOT (predicado como 0/1)
            default: fail("modificador de fuente desconocido"); break;
        }
        return v;
    }

    uint32_t splat(uint32_t f) { return b_.value(OpCompositeConstruct, tV4_, {f, f, f, f}); }
    uint32_t comp(uint32_t v, uint32_t c) { return b_.value(OpCompositeExtract, tF_, {v, c}); }

    // ---- escritura con máscara, saturación y predicado ------------------------------------------
    void store(const Param& d, uint32_t value) {
        const uint32_t var = destVar(d);
        if (var == 0) return;
        if (d.resultMod() & 1) value = b_.ext(tV4_, 43 /*FClamp*/, {value, constV4(0, 0, 0, 0), constV4(1, 1, 1, 1)});
        const uint32_t m = d.mask();
        if (m != 0xF) {
            const uint32_t old = loadPtr(var);
            value = b_.value(OpVectorShuffle, tV4_, {old, value, (m & 1) ? 4u : 0u, (m & 2) ? 5u : 1u,
                                                     (m & 4) ? 6u : 2u, (m & 8) ? 7u : 3u});
        }
        b_.op(OpStore, {var, value});
    }

    // ---- lectura de parámetros ----------------------------------------------------------------
    bool readParam(size_t& p, size_t end, Param& out, bool allowRel) {
        if (p >= end) return fail("parámetro ausente");
        out.token = tok_[p++];
        out.type = regType(out.token);
        out.num = out.token & 0x7FF;
        out.relative = (out.token & (1u << 13)) != 0;
        if (out.relative) {
            if (!allowRel || p >= end) return fail("índice relativo sin registro");
            out.relToken = tok_[p++];
        }
        return true;
    }

    // ---- instrucciones -------------------------------------------------------------------------
    void instruction(uint32_t t, size_t p, size_t end) {
        const uint32_t op = t & 0xFFFF;
        const uint32_t ctl = (t >> 16) & 0xFF;
        if (t & (1u << 28)) { fail("instrucción con predicado no soportada"); return; }
        switch (op) {
            case 0: return;          // nop
            case 31: declare(&tok_[p], end - p); return;
            case 81: case 47: case 48: return;  // def*: ya en la pasada previa
            case 40: case 41: flowIf(op, ctl, p, end); return;
            case 42: flowElse(); return;
            case 43: flowEndIf(); return;
            case 38: case 27: flowLoop(op, p, end); return;
            case 39: case 29: flowEndLoop(); return;
            case 44: flowBreak(); return;
            case 45: flowBreakC(ctl, p, end); return;
            case 65: texkill(p, end); return;
            default: break;
        }
        Param d;
        if (!readParam(p, end, d, false)) return;
        std::vector<Param> s;
        while (p < end) {
            Param x;
            if (!readParam(p, end, x, true)) return;
            s.push_back(x);
        }
        auto src = [&](size_t i) { return i < s.size() ? source(s[i]) : constV4(0, 0, 0, 0); };
        auto need = [&](size_t n) { if (s.size() < n) fail("faltan operandos"); return ok_; };
        uint32_t r = 0;
        switch (op) {
            case 1: if (!need(1)) return; r = src(0); break;  // mov
            case 46: {  // mova: redondeo al más cercano
                if (!need(1)) return;
                r = b_.ext(tV4_, 8 /*Floor*/, {b_.value(OpFAdd, tV4_, {src(0), constV4(0.5f, 0.5f, 0.5f, 0.5f)})});
                break;
            }
            case 2: if (!need(2)) return; r = b_.value(OpFAdd, tV4_, {src(0), src(1)}); break;
            case 3: if (!need(2)) return; r = b_.value(OpFSub, tV4_, {src(0), src(1)}); break;
            case 4: if (!need(3)) return; r = b_.value(OpFAdd, tV4_, {b_.value(OpFMul, tV4_, {src(0), src(1)}), src(2)}); break;
            case 5: if (!need(2)) return; r = b_.value(OpFMul, tV4_, {src(0), src(1)}); break;
            case 6: { if (!need(1)) return; const uint32_t x = comp(src(0), 0); r = splat(b_.value(OpFDiv, tF_, {b_.constF(1.0f), x})); break; }
            case 7: { if (!need(1)) return; const uint32_t x = b_.ext(tF_, GlslFAbs, {comp(src(0), 0)}); r = splat(b_.ext(tF_, GlslInverseSqrt, {x})); break; }
            case 8: { if (!need(2)) return; r = splat(dot(src(0), src(1), 3)); break; }
            case 9: { if (!need(2)) return; r = splat(dot(src(0), src(1), 4)); break; }
            case 10: if (!need(2)) return; r = b_.ext(tV4_, 37 /*FMin*/, {src(0), src(1)}); break;
            case 11: if (!need(2)) return; r = b_.ext(tV4_, 40 /*FMax*/, {src(0), src(1)}); break;
            case 12: case 13: {  // slt / sge
                if (!need(2)) return;
                const uint32_t c = b_.value(op == 12 ? OpFOrdLessThan : OpFOrdGreaterThanEqual, tBV4_, {src(0), src(1)});
                r = b_.value(OpSelect, tV4_, {c, constV4(1, 1, 1, 1), constV4(0, 0, 0, 0)});
                break;
            }
            case 14: case 78: { if (!need(1)) return; r = splat(b_.ext(tF_, GlslExp2, {comp(src(0), 0)})); break; }  // exp/expp
            case 15: case 79: {  // log/logp: log2(|x|)
                if (!need(1)) return;
                r = splat(b_.ext(tF_, GlslLog2, {b_.ext(tF_, GlslFAbs, {comp(src(0), 0)})}));
                break;
            }
            case 16: r = lit(src(0)); break;
            case 17: {  // dst
                if (!need(2)) return;
                const uint32_t a = src(0), c = src(1);
                r = b_.value(OpCompositeConstruct, tV4_, {b_.constF(1.0f), b_.value(OpFMul, tF_, {comp(a, 1), comp(c, 1)}),
                                                          comp(a, 2), comp(c, 3)});
                break;
            }
            case 18: {  // lrp: s0*(s1-s2)+s2
                if (!need(3)) return;
                const uint32_t a = src(0), c1 = src(1), c2 = src(2);
                r = b_.value(OpFAdd, tV4_, {b_.value(OpFMul, tV4_, {a, b_.value(OpFSub, tV4_, {c1, c2})}), c2});
                break;
            }
            case 19: if (!need(1)) return; r = b_.ext(tV4_, GlslFract, {src(0)}); break;
            case 20: case 21: case 22: case 23: case 24: {  // m4x4 m4x3 m3x4 m3x3 m3x2
                if (!need(2)) return;
                const uint32_t n = (op == 20 || op == 21) ? 4 : 3;
                const uint32_t rows = op == 20 ? 4 : op == 21 ? 3 : op == 22 ? 4 : op == 23 ? 3 : 2;
                const uint32_t a = src(0);
                std::vector<uint32_t> c(4, b_.constF(0.0f));
                for (uint32_t i = 0; i < rows; ++i) {
                    Param m = s[1];
                    m.num += i;
                    c[i] = dot(a, source(m), n);
                }
                r = b_.value(OpCompositeConstruct, tV4_, c);
                break;
            }
            case 32: {  // pow(|x|, y)
                if (!need(2)) return;
                const uint32_t x = b_.ext(tF_, GlslFAbs, {comp(src(0), 0)});
                r = splat(b_.ext(tF_, 26 /*Pow*/, {x, comp(src(1), 0)}));
                break;
            }
            case 33: {  // crs
                if (!need(2)) return;
                const uint32_t a = src(0), c = src(1);
                const uint32_t t3 = b_.typeVector(tF_, 3);
                const uint32_t a3 = b_.value(OpVectorShuffle, t3, {a, a, 0, 1, 2});
                const uint32_t c3 = b_.value(OpVectorShuffle, t3, {c, c, 0, 1, 2});
                const uint32_t x = b_.ext(t3, 68 /*Cross*/, {a3, c3});
                r = b_.value(OpVectorShuffle, tV4_, {x, x, 0, 1, 2, 2});
                break;
            }
            case 34: if (!need(1)) return; r = b_.ext(tV4_, 6 /*FSign*/, {src(0)}); break;
            case 35: if (!need(1)) return; r = b_.ext(tV4_, GlslFAbs, {src(0)}); break;
            case 36: {  // nrm: x / |x.xyz|
                if (!need(1)) return;
                const uint32_t a = src(0);
                const uint32_t l2 = dot(a, a, 3);
                const uint32_t inv = b_.ext(tF_, GlslInverseSqrt, {l2});
                r = b_.value(OpVectorTimesScalar(), tV4_, {a, inv});
                break;
            }
            case 37: {  // sincos (sm3): x = cos, y = sin
                if (!need(1)) return;
                const uint32_t xs = comp(src(0), 0);  // swizzle replicado
                r = b_.value(OpCompositeConstruct, tV4_, {b_.ext(tF_, GlslCos, {xs}), b_.ext(tF_, GlslSin, {xs}),
                                                          b_.constF(0.0f), b_.constF(0.0f)});
                break;
            }
            case 80: {  // cnd: s0 > 0.5 ? s1 : s2
                if (!need(3)) return;
                const uint32_t c = b_.value(186 /*FOrdGreaterThan*/, tBV4_, {src(0), constV4(0.5f, 0.5f, 0.5f, 0.5f)});
                r = b_.value(OpSelect, tV4_, {c, src(1), src(2)});
                break;
            }
            case 88: {  // cmp: s0 >= 0 ? s1 : s2
                if (!need(3)) return;
                const uint32_t c = b_.value(OpFOrdGreaterThanEqual, tBV4_, {src(0), constV4(0, 0, 0, 0)});
                r = b_.value(OpSelect, tV4_, {c, src(1), src(2)});
                break;
            }
            case 90: {  // dp2add
                if (!need(3)) return;
                r = splat(b_.value(OpFAdd, tF_, {dot(src(0), src(1), 2), comp(src(2), 0)}));
                break;
            }
            case 91: if (!need(1)) return; r = vs_ ? constV4(0, 0, 0, 0) : b_.value(OpDPdx, tV4_, {src(0)}); break;
            case 92: if (!need(1)) return; r = vs_ ? constV4(0, 0, 0, 0) : b_.value(OpDPdy, tV4_, {src(0)}); break;
            case 66: case 95: case 93: r = texture(op, ctl, s); break;  // texld/texldp/texldb, texldl, texldd
            case 94: {  // setp
                if (!need(2)) return;
                const uint32_t c = compare(ctl, src(0), src(1), tBV4_);
                r = b_.value(OpSelect, tV4_, {c, constV4(1, 1, 1, 1), constV4(0, 0, 0, 0)});
                break;
            }
            default: fail("opcode D3D9 " + std::to_string(op) + " no soportado"); return;
        }
        if (!ok_) return;
        store(d, r);
    }

    static uint32_t OpVectorTimesScalar() { return 142; }

    uint32_t dot(uint32_t a, uint32_t c, uint32_t n) {
        if (n == 4) return b_.value(OpDot, tF_, {a, c});
        const uint32_t tv = b_.typeVector(tF_, n);
        const uint32_t a2 = n == 3 ? b_.value(OpVectorShuffle, tv, {a, a, 0, 1, 2}) : b_.value(OpVectorShuffle, tv, {a, a, 0, 1});
        const uint32_t c2 = n == 3 ? b_.value(OpVectorShuffle, tv, {c, c, 0, 1, 2}) : b_.value(OpVectorShuffle, tv, {c, c, 0, 1});
        return b_.value(OpDot, tF_, {a2, c2});
    }

    uint32_t lit(uint32_t a) {
        const uint32_t x = comp(a, 0), y = comp(a, 1), w = comp(a, 3);
        const uint32_t zero = b_.constF(0.0f);
        const uint32_t dy = b_.ext(tF_, 40, {x, zero});
        const uint32_t wc = b_.ext(tF_, 43, {w, b_.constF(-128.0f), b_.constF(128.0f)});
        const uint32_t spec = b_.ext(tF_, 26, {b_.ext(tF_, 40, {y, zero}), wc});
        const uint32_t xpos = b_.value(186, tB_, {x, zero});
        const uint32_t ypos = b_.value(186, tB_, {y, zero});
        const uint32_t both = b_.value(167 /*LogicalAnd*/, tB_, {xpos, ypos});
        const uint32_t z = b_.value(OpSelect, tF_, {both, spec, zero});
        return b_.value(OpCompositeConstruct, tV4_, {b_.constF(1.0f), dy, z, b_.constF(1.0f)});
    }

    uint32_t compare(uint32_t ctl, uint32_t a, uint32_t c, uint32_t resType) {
        uint32_t o = 0;
        switch (ctl) {
            case 1: o = 186; break;               // GT
            case 2: o = OpFOrdEqual; break;       // EQ
            case 3: o = OpFOrdGreaterThanEqual; break;
            case 4: o = OpFOrdLessThan; break;    // LT
            case 5: o = OpFUnordNotEqual; break;  // NE
            case 6: o = 188; break;               // LE
            default: fail("comparación desconocida"); o = OpFOrdEqual; break;
        }
        return b_.value(o, resType, {a, c});
    }

    uint32_t texture(uint32_t op, uint32_t ctl, const std::vector<Param>& s) {
        if (s.size() < 2 || s[1].type != kSampler) { fail("texld sin sampler"); return constV4(0, 0, 0, 0); }
        const uint32_t sn = s[1].num;
        auto it = sampVar_.find(sn);
        if (it == sampVar_.end()) { fail("sampler s" + std::to_string(sn) + " sin dcl"); return constV4(0, 0, 0, 0); }
        const uint32_t si = b_.value(OpLoad, sampImgType_[sn], {it->second});
        uint32_t c = source(s[0]);
        const uint32_t st = info_->samplerType[sn];
        const uint32_t nc = st == 2 ? 2 : 3;
        const uint32_t tc = b_.typeVector(tF_, nc);
        if (op == 66 && ctl == 1) {  // texldp
            c = b_.value(OpFDiv, tV4_, {c, splat(comp(c, 3))});
        }
        const uint32_t coord = nc == 2 ? b_.value(OpVectorShuffle, tc, {c, c, 0, 1}) : b_.value(OpVectorShuffle, tc, {c, c, 0, 1, 2});
        uint32_t r = 0;
        if (op == 95 || vs_) {  // texldl (en VS siempre LOD explícito)
            r = b_.value(OpImageSampleExplicitLod, tV4_, {si, coord, ImLod, op == 95 ? comp(c, 3) : b_.constF(0.0f)});
        } else if (op == 93) {  // texldd
            if (s.size() < 4) { fail("texldd sin gradientes"); return constV4(0, 0, 0, 0); }
            const uint32_t gx = source(s[2]), gy = source(s[3]);
            const uint32_t dx = nc == 2 ? b_.value(OpVectorShuffle, tc, {gx, gx, 0, 1}) : b_.value(OpVectorShuffle, tc, {gx, gx, 0, 1, 2});
            const uint32_t dy = nc == 2 ? b_.value(OpVectorShuffle, tc, {gy, gy, 0, 1}) : b_.value(OpVectorShuffle, tc, {gy, gy, 0, 1, 2});
            r = b_.value(OpImageSampleExplicitLod, tV4_, {si, coord, ImGrad, dx, dy});
        } else if (ctl == 2) {  // texldb
            r = b_.value(OpImageSampleImplicitLod, tV4_, {si, coord, ImBias, comp(c, 3)});
        } else {
            r = b_.value(OpImageSampleImplicitLod, tV4_, {si, coord});
        }
        // Swizzle del sampler (ps_3_0 permite .swizzle en s#)
        return swizzle(r, s[1].swizzle());
    }

    void texkill(size_t p, size_t end) {
        Param d;
        if (!readParam(p, end, d, false)) return;
        Param s = d;
        s.token = (d.token & ~0x0FFF0000u) | (0xE4u << 16);  // como fuente .xyzw sin modificador
        const uint32_t v = rawSource(s);
        const uint32_t lt = b_.value(OpFOrdLessThan, tBV4_, {v, constV4(0, 0, 0, 0)});
        const uint32_t m = d.mask() == 0 ? 0x7 : d.mask();
        uint32_t any = b_.constBool(false);
        for (uint32_t i = 0; i < 4; ++i)
            if (m & (1u << i)) any = b_.value(166 /*LogicalOr*/, tB_, {any, b_.value(OpCompositeExtract, tB_, {lt, i})});
        killIf(any);
    }

    void killIf(uint32_t cond) {
        const uint32_t kill = b_.newId(), merge = b_.newId();
        b_.op(OpSelectionMerge, {merge, 0});
        b_.branchCond(cond, kill, merge);
        b_.label(kill);
        b_.terminate(OpKill, {});
        b_.label(merge);
    }

    // ---- control de flujo estructurado ----------------------------------------------------------
    uint32_t boolConst(uint32_t n) {
        auto it = bdefs_.find(n);
        if (it != bdefs_.end()) return b_.constBool(it->second);
        const uint32_t v = uboVec(2, b_.constI(static_cast<int32_t>(n & 15)), pUniUV4_, tUV4_);
        return b_.value(OpINotEqual, tB_, {b_.value(OpCompositeExtract, tU_, {v, 0}), b_.constU(0)});
    }

    void flowIf(uint32_t op, uint32_t ctl, size_t p, size_t end) {
        uint32_t cond = 0;
        if (op == 40) {  // if b# / if p0.c
            Param c;
            if (!readParam(p, end, c, false)) return;
            if (c.type == kConstBool) cond = boolConst(c.num);
            else if (c.type == kPredicate) {
                const uint32_t v = source(c);
                cond = b_.value(186, tB_, {comp(v, 0), b_.constF(0.5f)});
            } else { fail("if con registro no soportado"); return; }
        } else {  // ifc
            Param a, c;
            if (!readParam(p, end, a, true) || !readParam(p, end, c, true)) return;
            cond = compare(ctl, comp(source(a), 0), comp(source(c), 0), tB_);
        }
        Flow f{0, b_.newId(), 0, b_.newId(), 0, 0, false, 0};
        const uint32_t then = b_.newId();
        b_.op(OpSelectionMerge, {f.merge, 0});
        b_.branchCond(cond, then, f.elseLbl);
        b_.label(then);
        flow_.push_back(f);
    }

    void flowElse() {
        if (flow_.empty() || flow_.back().kind != 0) { fail("else sin if"); return; }
        Flow& f = flow_.back();
        if (b_.blockOpen()) b_.branch(f.merge);
        b_.label(f.elseLbl);
        f.hasElse = true;
    }

    void flowEndIf() {
        if (flow_.empty() || flow_.back().kind != 0) { fail("endif sin if"); return; }
        Flow f = flow_.back();
        flow_.pop_back();
        if (b_.blockOpen()) b_.branch(f.merge);
        if (!f.hasElse) {
            b_.label(f.elseLbl);
            b_.branch(f.merge);
        }
        b_.label(f.merge);
    }

    // rep i# / loop aL, i#: contador en variable de función; aL = i.y + iteración * i.z.
    void flowLoop(uint32_t op, size_t p, size_t end) {
        Param a, c;
        if (op == 27) { if (!readParam(p, end, a, false)) return; }  // aL
        if (!readParam(p, end, c, false)) return;
        if (c.type != kConstInt) { fail("bucle sin registro entero"); return; }
        uint32_t ivec = 0;
        auto it = idefs_.find(c.num);
        if (it != idefs_.end())
            ivec = b_.constComposite(tIV4_, {b_.constI(it->second[0]), b_.constI(it->second[1]), b_.constI(it->second[2]),
                                            b_.constI(it->second[3])});
        else
            ivec = uboVec(1, b_.constI(static_cast<int32_t>(c.num & 15)), pUniIV4_, tIV4_);
        const uint32_t count = b_.value(OpCompositeExtract, tI_, {ivec, 0});
        const uint32_t pI = b_.typePointer(ScFunction, tI_);
        const uint32_t counter = b_.localVar(pI);
        b_.op(OpStore, {counter, b_.constI(0)});
        uint32_t step = 0;
        if (op == 27) {
            if (loopReg_ == 0) loopReg_ = b_.globalVarInit(pPrivV4_, ScPrivate, constV4(0, 0, 0, 0));
            const uint32_t start = b_.value(OpConvertSToF, tF_, {b_.value(OpCompositeExtract, tI_, {ivec, 1})});
            step = b_.value(OpConvertSToF, tF_, {b_.value(OpCompositeExtract, tI_, {ivec, 2})});
            b_.op(OpStore, {loopReg_, splat(start)});
        }
        const uint32_t header = b_.newId(), body = b_.newId();
        Flow f{1, b_.newId(), b_.newId(), 0, counter, step, false, header};
        b_.branch(header);
        b_.label(header);
        const uint32_t n = b_.value(OpLoad, tI_, {counter});
        const uint32_t more = b_.value(OpSLessThan, tB_, {n, b_.ext(tI_, 42 /*SMax*/, {count, b_.constI(0)})});
        b_.op(OpLoopMerge, {f.merge, f.cont, 0});
        b_.branchCond(more, body, f.merge);
        b_.label(body);
        flow_.push_back(f);
    }

    void flowEndLoop() {
        if (flow_.empty() || flow_.back().kind != 1) { fail("endloop sin loop"); return; }
        Flow f = flow_.back();
        flow_.pop_back();
        if (b_.blockOpen()) b_.branch(f.cont);
        b_.label(f.cont);
        const uint32_t n = b_.value(OpLoad, tI_, {f.counter});
        b_.op(OpStore, {f.counter, b_.value(OpIAdd, tI_, {n, b_.constI(1)})});
        if (f.step != 0) {
            const uint32_t l = loadPtr(loopReg_);
            b_.op(OpStore, {loopReg_, b_.value(OpFAdd, tV4_, {l, splat(f.step)})});
        }
        b_.branch(f.header);
        b_.label(f.merge);
    }

    uint32_t innermostLoopMerge() {
        for (auto it = flow_.rbegin(); it != flow_.rend(); ++it)
            if (it->kind == 1) return it->merge;
        return 0;
    }

    void flowBreak() {
        const uint32_t m = innermostLoopMerge();
        if (m == 0) { fail("break fuera de bucle"); return; }
        b_.branch(m);
        b_.label(b_.newId());  // código inalcanzable hasta el siguiente cierre
    }

    void flowBreakC(uint32_t ctl, size_t p, size_t end) {
        Param a, c;
        if (!readParam(p, end, a, true) || !readParam(p, end, c, true)) return;
        const uint32_t m = innermostLoopMerge();
        if (m == 0) { fail("breakc fuera de bucle"); return; }
        const uint32_t cond = compare(ctl, comp(source(a), 0), comp(source(c), 0), tB_);
        const uint32_t brk = b_.newId(), cont = b_.newId();
        b_.op(OpSelectionMerge, {cont, 0});
        b_.branchCond(cond, brk, cont);
        b_.label(brk);
        b_.branch(m);
        b_.label(cont);
    }

    // ---- final del shader -----------------------------------------------------------------------
    void epilogue() {
        if (vs_) return;
        if (depthOut_ != 0) {
            const uint32_t out = b_.globalVar(b_.typePointer(ScOutput, tF_), ScOutput);
            b_.decorate(out, DecBuiltIn, {BiFragDepth});
            b_.executionMode(12);  // DepthReplacing
            b_.op(OpStore, {out, comp(loadPtr(depthOut_), 0)});
        }
        // Prueba alfa de función fija: alphaTest = {func D3DCMP, ref, -, -}; func 8 (ALWAYS) o 0 = desactivada.
        if (alphaTest_) {
            const uint32_t at = b_.value(OpLoad, tV4_, {b_.value(OpAccessChain, pUniV4_, {ubo_, b_.constI(3)})});
            const uint32_t func = b_.value(OpConvertFToS, tI_, {comp(at, 0)});
            const uint32_t ref = comp(at, 1);
            uint32_t a = b_.constF(1.0f);
            auto it = colorOut_.find(0);
            if (it != colorOut_.end()) a = comp(loadPtr(it->second), 3);
            auto is = [&](int v) { return b_.value(OpIEqual, tB_, {func, b_.constI(v)}); };
            auto land = [&](uint32_t x, uint32_t y) { return b_.value(167, tB_, {x, y}); };
            auto lor = [&](uint32_t x, uint32_t y) { return b_.value(166, tB_, {x, y}); };
            uint32_t pass = lor(is(8), is(0));
            pass = lor(pass, land(is(2), b_.value(OpFOrdLessThan, tB_, {a, ref})));
            pass = lor(pass, land(is(3), b_.value(OpFOrdEqual, tB_, {a, ref})));
            pass = lor(pass, land(is(4), b_.value(188, tB_, {a, ref})));
            pass = lor(pass, land(is(5), b_.value(186, tB_, {a, ref})));
            pass = lor(pass, land(is(6), b_.value(OpFUnordNotEqual, tB_, {a, ref})));
            pass = lor(pass, land(is(7), b_.value(OpFOrdGreaterThanEqual, tB_, {a, ref})));
            killIf(b_.value(168 /*LogicalNot*/, tB_, {pass}));
        }
    }
    bool alphaTest_ = true;
};

}  // namespace detail

inline bool translate(const uint32_t* tokens, size_t count, std::vector<uint32_t>& spirv, ShaderInfo& info,
                      std::string& err, bool alphaTest = true) {
    detail::Translator t(tokens, count, alphaTest);
    return t.run(spirv, info, err);
}

}  // namespace hyperplay::d3d9sh
