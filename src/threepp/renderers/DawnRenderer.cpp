
#include "threepp/renderers/DawnRenderer.hpp"

#include "dawn/DawnGeometries.hpp"
#include "dawn/DawnTextures.hpp"
#include "dawn/DawnState.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/math/Frustum.hpp"

#include "threepp/renderers/common/Lights.hpp"
#include "threepp/renderers/common/RenderLists.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#define GLFW_INCLUDE_NONE
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
// Match the linkage used by the implementation (extern "C" in C++).
#define STBIWDEF extern "C"
#include "stb_image_write.h"
#undef STBIWDEF

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;

#ifdef __APPLE__
extern "C" void* dawn_create_metal_layer(void* nsWindow);
#endif

namespace {

    // Maximum time to wait for async WebGPU operations before aborting.
    constexpr auto WGPU_ASYNC_TIMEOUT = std::chrono::seconds(10);

    // Feature bitmask for pipeline caching
    // Bits 0-3: material features
    // Bits 4-5: cull mode (00=None, 01=Front, 10=Back)
    // Bit 6: wireframe (LineList vs TriangleList)
    // Bits 7-9: blend mode (000=Normal, 001=None, 010=Additive, 011=Subtractive, 100=Multiply)
    enum PipelineFeatures : uint32_t {
        FEAT_NONE       = 0,
        FEAT_TEXTURE    = 1 << 0,
        FEAT_LIGHTING   = 1 << 1,
        FEAT_SPECULAR   = 1 << 2,
        FEAT_PBR        = 1 << 3,
        FEAT_NORMAL_MAP = 1 << 10,
    };

    constexpr uint32_t CULL_SHIFT = 4;
    constexpr uint32_t CULL_MASK  = 0x3 << CULL_SHIFT;
    constexpr uint32_t CULL_NONE  = 0 << CULL_SHIFT;
    constexpr uint32_t CULL_FRONT = 1 << CULL_SHIFT;
    constexpr uint32_t CULL_BACK  = 2 << CULL_SHIFT;

    constexpr uint32_t WIREFRAME_BIT = 1 << 6;

    constexpr uint32_t BLEND_SHIFT      = 7;
    constexpr uint32_t BLEND_MASK       = 0x7 << BLEND_SHIFT;
    constexpr uint32_t BLEND_NORMAL     = 0 << BLEND_SHIFT;
    constexpr uint32_t BLEND_DISABLED   = 1 << BLEND_SHIFT;
    constexpr uint32_t BLEND_ADDITIVE   = 2 << BLEND_SHIFT;
    constexpr uint32_t BLEND_SUBTRACTIVE= 3 << BLEND_SHIFT;
    constexpr uint32_t BLEND_MULTIPLY   = 4 << BLEND_SHIFT;

    constexpr uint32_t DEPTH_WRITE_OFF  = 1 << 11;
    constexpr uint32_t FEAT_SHADOW      = 1 << 12;
    constexpr uint32_t FEAT_FOG_LINEAR  = 1 << 13;
    constexpr uint32_t FEAT_FOG_EXP2    = 1 << 14;

    // Tone mapping mode encoded in bits 15-16
    constexpr uint32_t TONEMAP_SHIFT    = 15;
    constexpr uint32_t TONEMAP_MASK     = 0x7 << TONEMAP_SHIFT;
    constexpr uint32_t TONEMAP_NONE     = 0 << TONEMAP_SHIFT;
    constexpr uint32_t TONEMAP_LINEAR   = 1 << TONEMAP_SHIFT;
    constexpr uint32_t TONEMAP_REINHARD = 2 << TONEMAP_SHIFT;
    constexpr uint32_t TONEMAP_CINEON   = 3 << TONEMAP_SHIFT;
    constexpr uint32_t TONEMAP_ACES     = 4 << TONEMAP_SHIFT;

    // Topology mode (bits 18-19): 00=TriangleList, 01=LineList, 10=LineStrip, 11=PointList
    constexpr uint32_t TOPO_SHIFT       = 18;
    constexpr uint32_t TOPO_MASK        = 0x3 << TOPO_SHIFT;
    constexpr uint32_t TOPO_TRIANGLE    = 0 << TOPO_SHIFT;
    constexpr uint32_t TOPO_LINE_LIST   = 1 << TOPO_SHIFT;
    constexpr uint32_t TOPO_LINE_STRIP  = 2 << TOPO_SHIFT;
    constexpr uint32_t TOPO_POINT_LIST  = 3 << TOPO_SHIFT;

    // Instancing bit
    constexpr uint32_t FEAT_INSTANCED   = 1 << 20;

    constexpr uint32_t SHADOW_MAP_SIZE = 1024;
    constexpr size_t SHADOW_UNIFORM_SIZE = 80; // lightVP(64) + bias(4) + normalBias(4) + padding(8)

    constexpr int MAX_DIR_LIGHTS   = 4;
    constexpr int MAX_POINT_LIGHTS = 4;
    constexpr int MAX_SPOT_LIGHTS  = 4;
    constexpr int MAX_HEMI_LIGHTS  = 2;

    // Transform: model(64) + view(64) + proj(64) + normalMatrix(48 = 3*vec4 padded) + cameraPos(12) + pad(4) = 256
    constexpr size_t TRANSFORM_UNIFORM_SIZE = 256;

    // Material: diffuse(16) + specularAndShininess(16) + roughnessMetalnessOpacity(16) + emissive(16) + flags(16) + fog(16) + toneMapping(16) = 112, pad to 128
    constexpr size_t MATERIAL_UNIFORM_SIZE = 128;

    // Light: header(32) + dir(4*32=128) + point(4*48=192) + spot(4*64=256) + hemi(2*48=96) = 704
    constexpr size_t LIGHT_UNIFORM_SIZE = 704;


    std::string buildWGSL(uint32_t features) {
        std::ostringstream s;

        s << R"(
struct TransformUniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct MaterialUniforms {
    diffuse: vec4<f32>,
    specularAndShininess: vec4<f32>,
    roughnessMetalnessOpacity: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
    _pad: vec4<f32>,
};
@group(0) @binding(1) var<uniform> material: MaterialUniforms;
)";

        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);
        if (lit) {
            s << "struct DirectionalLightGPU { direction: vec3<f32>, _p0: f32, color: vec3<f32>, _p1: f32, };\n";
            s << "struct PointLightGPU { position: vec3<f32>, _p0: f32, color: vec3<f32>, distance: f32, decay: f32, _p1: f32, _p2: f32, _p3: f32, };\n";
            s << "struct SpotLightGPU { position: vec3<f32>, _p0: f32, direction: vec3<f32>, _p1: f32, color: vec3<f32>, distance: f32, decay: f32, coneCos: f32, penumbraCos: f32, _p2: f32, };\n";
            s << "struct HemisphereLightGPU { direction: vec3<f32>, _p0: f32, skyColor: vec3<f32>, _p1: f32, groundColor: vec3<f32>, _p2: f32, };\n";
            s << "struct LightData {\n";
            s << "  numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,\n";
            s << "  ambient: vec3<f32>, _pad: f32,\n";
            s << "  directional: array<DirectionalLightGPU, " << MAX_DIR_LIGHTS << ">,\n";
            s << "  point: array<PointLightGPU, " << MAX_POINT_LIGHTS << ">,\n";
            s << "  spot: array<SpotLightGPU, " << MAX_SPOT_LIGHTS << ">,\n";
            s << "  hemi: array<HemisphereLightGPU, " << MAX_HEMI_LIGHTS << ">,\n";
            s << "};\n";
            s << "@group(0) @binding(2) var<uniform> lights: LightData;\n";
        }

        if (features & FEAT_TEXTURE) {
            s << "@group(0) @binding(3) var t_diffuse: texture_2d<f32>;\n";
            s << "@group(0) @binding(4) var s_diffuse: sampler;\n";
        }
        if (features & FEAT_NORMAL_MAP) {
            s << "@group(0) @binding(5) var t_normalMap: texture_2d<f32>;\n";
            s << "@group(0) @binding(6) var s_normalMap: sampler;\n";
        }

        if (features & FEAT_SHADOW) {
            s << R"(
struct ShadowUniforms {
    lightVP: mat4x4<f32>,
    bias: f32,
    normalBias: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(7) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(8) var t_shadowMap: texture_depth_2d;
@group(0) @binding(9) var s_shadowMap: sampler_comparison;
)";
        }

        s << R"(
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VertexOutput {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,)";

        if (features & FEAT_SHADOW) {
            s << "\n    @location(3) lightSpacePos: vec4<f32>,\n";
        }

        s << R"(};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let worldPos4 = transform.model * vec4<f32>(in.position, 1.0);
    out.worldPos = worldPos4.xyz;
    let nm = mat3x3<f32>(transform.normalCol0.xyz, transform.normalCol1.xyz, transform.normalCol2.xyz);
    out.worldNormal = normalize(nm * in.normal);
    out.uv = in.uv;
    out.clipPos = transform.proj * transform.view * worldPos4;)";

        if (features & FEAT_SHADOW) {
            s << "\n    out.lightSpacePos = shadow.lightVP * worldPos4;\n";
        }

        s << R"(
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var baseColor = material.diffuse.rgb;
    let opacity = material.roughnessMetalnessOpacity.z;
)";

        if (features & FEAT_TEXTURE) {
            s << "    let texColor = textureSample(t_diffuse, s_diffuse, in.uv);\n";
            s << "    baseColor = baseColor * texColor.rgb;\n";
        }

        if (lit) {
            if (features & FEAT_NORMAL_MAP) {
                s << R"(
    // Screen-space normal map perturbation (no tangent attributes needed)
    let dPdx = dpdx(in.worldPos);
    let dPdy = dpdy(in.worldPos);
    let dUVdx = dpdx(in.uv);
    let dUVdy = dpdy(in.uv);
    let T = normalize(dPdx * dUVdy.y - dPdy * dUVdx.y);
    let B = normalize(dPdy * dUVdx.x - dPdx * dUVdy.x);
    let geomN = normalize(in.worldNormal);
    let TBN = mat3x3<f32>(T, B, geomN);
    let nmSample = textureSample(t_normalMap, s_normalMap, in.uv).rgb * 2.0 - vec3<f32>(1.0);
    let normalScale = material.flags.zw;
    let scaledNm = vec3<f32>(nmSample.xy * normalScale, nmSample.z);
    let N = normalize(TBN * scaledNm);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
            } else {
                s << R"(
    let N = normalize(in.worldNormal);
    let V = normalize(transform.cameraPos - in.worldPos);
)";
            }
            s << R"(
    var diffuseLight = lights.ambient;
    var specularLight = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < lights.numDir; i++) {
        let L = normalize(-lights.directional[i].direction);
        let NdotL = max(dot(N, L), 0.0);
        diffuseLight += lights.directional[i].color * NdotL;
)";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.directional[i].color * material.specularAndShininess.rgb * s; }\n";
            }
            if (features & FEAT_PBR) {
                // GGX/Trowbridge-Reitz NDF with Schlick Fresnel
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let NdotV = max(dot(N, V), 0.001);\n";
                s << "          let r = material.roughnessMetalnessOpacity.x; let m = material.roughnessMetalnessOpacity.y;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.directional[i].color * F * D * NdotL; }\n";
            }
            s << "    }\n";

            s << R"(
    for (var i = 0u; i < lights.numPoint; i++) {
        let lv = lights.point[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        let NdotL = max(dot(N, L), 0.0);
        var att = 1.0;
        if (lights.point[i].distance > 0.0) {
            let r2 = clamp(1.0 - pow(d / lights.point[i].distance, 4.0), 0.0, 1.0);
            att = r2 * r2 / (d * d + 0.0001);
        }
        diffuseLight += lights.point[i].color * NdotL * att;
)";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.point[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
            if (features & FEAT_PBR) {
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let r = material.roughnessMetalnessOpacity.x; let m = material.roughnessMetalnessOpacity.y;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.point[i].color * F * D * NdotL * att; }\n";
            }
            s << "    }\n";

            s << R"(
    for (var i = 0u; i < lights.numSpot; i++) {
        let lv = lights.spot[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        let NdotL = max(dot(N, L), 0.0);
        let ac = dot(L, normalize(lights.spot[i].direction));
        let se = smoothstep(lights.spot[i].coneCos, lights.spot[i].penumbraCos, ac);
        var att = se;
        if (lights.spot[i].distance > 0.0) {
            let r2 = clamp(1.0 - pow(d / lights.spot[i].distance, 4.0), 0.0, 1.0);
            att = att * r2 * r2 / (d * d + 0.0001);
        }
        diffuseLight += lights.spot[i].color * NdotL * att;
)";
            if (features & FEAT_SPECULAR) {
                s << "        { let H = normalize(L + V); let s = pow(max(dot(N, H), 0.0), material.specularAndShininess.w);\n";
                s << "          specularLight += lights.spot[i].color * material.specularAndShininess.rgb * s * att; }\n";
            }
            if (features & FEAT_PBR) {
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let r = material.roughnessMetalnessOpacity.x; let m = material.roughnessMetalnessOpacity.y;\n";
                s << "          let a = r * r; let a2 = a * a;\n";
                s << "          let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;\n";
                s << "          let D = a2 / (3.14159265 * denom * denom);\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);\n";
                s << "          specularLight += lights.spot[i].color * F * D * NdotL * att; }\n";
            }
            s << R"(
    }
    for (var i = 0u; i < lights.numHemi; i++) {
        let w = 0.5 * dot(N, lights.hemi[i].direction) + 0.5;
        diffuseLight += mix(lights.hemi[i].groundColor, lights.hemi[i].skyColor, w);
    }
)";
            // Apply shadow factor to directional light contribution
            if (features & FEAT_SHADOW) {
                s << R"(
    {
        let projCoords = in.lightSpacePos.xyz / in.lightSpacePos.w;
        let shadowUV = vec2<f32>(projCoords.x * 0.5 + 0.5, 1.0 - (projCoords.y * 0.5 + 0.5));
        let currentDepth = projCoords.z - shadow.bias;
        var shadowFactor = 1.0;
        if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0 && currentDepth >= 0.0 && currentDepth <= 1.0) {
            // 3x3 PCF sampling
            let texelSize = 1.0 / f32(textureDimensions(t_shadowMap).x);
            var shadow_sum = 0.0;
            for (var sy = -1; sy <= 1; sy++) {
                for (var sx = -1; sx <= 1; sx++) {
                    let offset = vec2<f32>(f32(sx), f32(sy)) * texelSize;
                    shadow_sum += textureSampleCompare(t_shadowMap, s_shadowMap, shadowUV + offset, currentDepth);
                }
            }
            shadowFactor = shadow_sum / 9.0;
        }
        // Shadow only attenuates diffuse/specular, not ambient/emissive
        diffuseLight = lights.ambient + (diffuseLight - lights.ambient) * shadowFactor;
        specularLight = specularLight * shadowFactor;
    }
)";
            }

            if (features & FEAT_PBR) {
                s << "    let metalness = material.roughnessMetalnessOpacity.y;\n";
                s << "    baseColor = baseColor * (1.0 - metalness) * diffuseLight + specularLight + material.emissive.rgb;\n";
            } else {
                s << "    baseColor = baseColor * diffuseLight + specularLight + material.emissive.rgb;\n";
            }
        }

        // Tone mapping (applied before fog)
        if ((features & TONEMAP_MASK) != TONEMAP_NONE) {
            s << "    let exposure = material.fogParams.w;\n";
            s << "    baseColor = baseColor * exposure;\n";
            if ((features & TONEMAP_MASK) == TONEMAP_REINHARD) {
                s << "    baseColor = baseColor / (vec3<f32>(1.0) + baseColor);\n";
            } else if ((features & TONEMAP_MASK) == TONEMAP_CINEON) {
                // Optimized filmic operator (Uncharted2-like)
                s << "    let x = max(vec3<f32>(0.0), baseColor - vec3<f32>(0.004));\n";
                s << "    baseColor = (x * (6.2 * x + vec3<f32>(0.5))) / (x * (6.2 * x + vec3<f32>(1.7)) + vec3<f32>(0.06));\n";
            } else if ((features & TONEMAP_MASK) == TONEMAP_ACES) {
                s << "    let a = baseColor * (baseColor * 2.51 + vec3<f32>(0.03));\n";
                s << "    let b = baseColor * (baseColor * 2.43 + vec3<f32>(0.59)) + vec3<f32>(0.14);\n";
                s << "    baseColor = clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));\n";
            }
            // TONEMAP_LINEAR: just exposure multiplication (done above)
        }

        // Fog (applied after tone mapping, mixes with fog color based on distance)
        if (features & FEAT_FOG_LINEAR) {
            s << "    {\n";
            s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
            s << "        let fogFactor = clamp((material.fogParams.y - fogDist) / (material.fogParams.y - material.fogParams.x), 0.0, 1.0);\n";
            s << "        baseColor = mix(material.fogColor.rgb, baseColor, fogFactor);\n";
            s << "    }\n";
        } else if (features & FEAT_FOG_EXP2) {
            s << "    {\n";
            s << "        let fogDist = length(transform.cameraPos - in.worldPos);\n";
            s << "        let fogDensity = material.fogParams.z;\n";
            s << "        let fogFactor = exp(-fogDensity * fogDensity * fogDist * fogDist);\n";
            s << "        baseColor = mix(material.fogColor.rgb, baseColor, clamp(fogFactor, 0.0, 1.0));\n";
            s << "    }\n";
        }

        s << "    return vec4<f32>(baseColor, opacity);\n}\n";
        return s.str();
    }

}// namespace


struct DawnRenderer::Impl {

    DawnRenderer& scope;
    Canvas& canvas;

    // Core WebGPU objects
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;

    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;

    WindowSize size_;
    float pixelRatio_ = 1.0f;
    Color clearColor_{0x000000};
    float clearAlpha_ = 1.0f;

    // Viewport & scissor state (Feature 4)
    struct { float x=0, y=0, w=0, h=0; } viewport_;
    struct { uint32_t x=0, y=0, w=0, h=0; } scissor_;
    bool scissorTest_ = false;

    // Uniform buffers (per-object, written each draw call)
    WGPUBuffer transformBuffer = nullptr;   // binding 0
    WGPUBuffer materialBuffer = nullptr;    // binding 1
    WGPUBuffer lightBuffer = nullptr;       // binding 2

    // Pipeline cache keyed by feature bitmask
    struct PipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
    };
    std::unordered_map<uint32_t, PipelineEntry> pipelineCache;

    // Subsystem: shared state
    dawn::DawnState dawnState;

    // Subsystem: geometry buffer management (with version-based updates)
    std::unique_ptr<dawn::DawnGeometries> geometries;

    // Subsystem: texture upload, caching, version tracking
    std::unique_ptr<dawn::DawnTextures> textures;

    // Render target cache (Feature 5)
    struct RTEntry {
        WGPUTexture colorTexture = nullptr;
        WGPUTextureView colorView = nullptr;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        unsigned int width = 0, height = 0;
    };
    std::unordered_map<std::string, RTEntry> rtCache;
    RenderTarget* currentRenderTarget_ = nullptr;

    // Light state

    // Render target state
    int activeCubeFace_ = 0;
    int activeMipmapLevel_ = 0;

    // Shadow mapping state
    struct ShadowState {
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUSampler comparisonSampler = nullptr;
        WGPUBuffer uniformBuffer = nullptr;
        WGPURenderPipeline depthPipeline = nullptr;
        WGPUPipelineLayout depthPipelineLayout = nullptr;
        WGPUBindGroupLayout depthBindGroupLayout = nullptr;
        WGPUShaderModule depthShader = nullptr;
        WGPUBuffer depthTransformBuffer = nullptr;
        Matrix4 lightVP;
        bool active = false;
        float bias = 0.005f;
        float normalBias = 0.0f;
    } shadowState;

    // Frustum culling
    Frustum frustum_;
    Vector3 _vector3;

    // Render list for opaque/transparent sorting
    RenderList renderList_;

    // Render info/statistics
    struct {
        size_t frame = 0;
        size_t calls = 0;
        size_t triangles = 0;
        size_t lines = 0;
        size_t points = 0;
        size_t geometries = 0;
        size_t textures = 0;
    } renderInfo;

    bool initialized = false;

    explicit Impl(DawnRenderer& scope, Canvas& canvas)
        : scope(scope), canvas(canvas), size_(canvas.size()) {

        viewport_.w = static_cast<float>(size_.width());
        viewport_.h = static_cast<float>(size_.height());
        scissor_.w = static_cast<uint32_t>(size_.width());
        scissor_.h = static_cast<uint32_t>(size_.height());

        initWebGPU();
    }

    void initWebGPU() {
        // Create instance with primary backends (Vulkan/Metal/DX12).
        // Avoid GL backend as it conflicts with GLFW's GL context.
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            std::cerr << "DawnRenderer: Failed to create WebGPU instance" << std::endl;
            return;
        }

        // Try to create surface from GLFW window.
        // Surface creation may fail in headless environments — that's OK,
        // the renderer can still operate with render targets only.
        createSurface();

        // Request adapter (surface is optional — nullptr works for offscreen)
        requestAdapter();
        if (!adapter) {
            std::cerr << "DawnRenderer: Failed to get adapter" << std::endl;
            return;
        }

        // Request device
        requestDevice();
        if (!device) {
            std::cerr << "DawnRenderer: Failed to get device" << std::endl;
            return;
        }

        queue = wgpuDeviceGetQueue(device);

        // Configure surface (only if we have one)
        if (surface) {
            configureSurface();
        }

        // Populate shared state for subsystems
        dawnState.device = device;
        dawnState.queue = queue;
        dawnState.surfaceFormat = surfaceFormat;

        // Create uniform buffers
        createUniformBuffers();

        // Initialize subsystems
        textures = std::make_unique<dawn::DawnTextures>(dawnState);
        textures->createDummyTexture();

        geometries = std::make_unique<dawn::DawnGeometries>(dawnState);

        initialized = true;
        std::cout << "DawnRenderer: WebGPU initialized successfully"
                  << (surface ? "" : " (headless, no surface)") << std::endl;
    }

    void createSurface() {
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        WGPUSurfaceDescriptor surfDesc{};
        WGPUStringView label = {.data = "threepp_surface", .length = 15};
        surfDesc.label = label;

#if defined(__linux__)
        Display* x11Display = glfwGetX11Display();
        ::Window x11Window = glfwGetX11Window(glfwWindow);

        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.chain.next = nullptr;
        xlibSource.display = x11Display;
        xlibSource.window = static_cast<uint64_t>(x11Window);
        surfDesc.nextInChain = &xlibSource.chain;

#elif defined(_WIN32)
        WGPUSurfaceSourceWindowsHWND hwndSource{};
        hwndSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        hwndSource.chain.next = nullptr;
        hwndSource.hinstance = GetModuleHandle(nullptr);
        hwndSource.hwnd = glfwGetWin32Window(glfwWindow);
        surfDesc.nextInChain = &hwndSource.chain;

#elif defined(__APPLE__)
        void* metalLayer = dawn_create_metal_layer(glfwGetCocoaWindow(glfwWindow));
        WGPUSurfaceSourceMetalLayer metalSource{};
        metalSource.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
        metalSource.chain.next = nullptr;
        metalSource.layer = metalLayer;
        surfDesc.nextInChain = &metalSource.chain;
#endif

        surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    void requestAdapter() {
        struct UserData {
            WGPUAdapter adapter = nullptr;
            bool done = false;
        } userData;

        WGPURequestAdapterOptions options{};
        options.compatibleSurface = surface; // nullptr in headless mode

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                ud->adapter = adapter;
            } else {
                std::cerr << "DawnRenderer: Adapter request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuInstanceRequestAdapter(instance, &options, callbackInfo);

        // Poll until callback fires, with timeout
        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestAdapter timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.adapter) {
            throw std::runtime_error("DawnRenderer: failed to obtain adapter");
        }
        adapter = userData.adapter;
    }

    void requestDevice() {
        struct UserData {
            WGPUDevice device = nullptr;
            bool done = false;
        } userData;

        WGPUDeviceDescriptor deviceDesc{};
        WGPUStringView label = {.data = "threepp_device", .length = 14};
        deviceDesc.label = label;

        WGPURequestDeviceCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success) {
                ud->device = device;
            } else {
                std::cerr << "DawnRenderer: Device request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuAdapterRequestDevice(adapter, &deviceDesc, callbackInfo);

        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestDevice timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.device) {
            throw std::runtime_error("DawnRenderer: failed to obtain device");
        }
        device = userData.device;
    }

    void configureSurface() {
        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width = static_cast<uint32_t>(std::floor(size_.width() * pixelRatio_));
        config.height = static_cast<uint32_t>(std::floor(size_.height() * pixelRatio_));
        config.presentMode = WGPUPresentMode_Fifo;
        config.alphaMode = WGPUCompositeAlphaMode_Auto;
        config.viewFormatCount = 0;
        config.viewFormats = nullptr;

        wgpuSurfaceConfigure(surface, &config);
    }

    PipelineEntry& getOrCreatePipeline(uint32_t features) {
        auto it = pipelineCache.find(features);
        if (it != pipelineCache.end()) return it->second;

        PipelineEntry entry{};

        // Shader module
        std::string wgsl = buildWGSL(features);
        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.chain.next = nullptr;
        wgslSource.code = {.data = wgsl.c_str(), .length = wgsl.size()};

        WGPUShaderModuleDescriptor shaderDesc{};
        shaderDesc.nextInChain = &wgslSource.chain;
        WGPUStringView shaderLabel = {.data = "dawn_shader", .length = 11};
        shaderDesc.label = shaderLabel;
        entry.shader = wgpuDeviceCreateShaderModule(device, &shaderDesc);
        if (!entry.shader) {
            std::cerr << "DawnRenderer: Failed to create shader module for features 0x"
                      << std::hex << features << std::dec << std::endl;
            return pipelineCache[features]; // return default-initialized entry
        }

        // Bind group layout entries
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);

        // Binding 0: transform uniforms
        { WGPUBindGroupLayoutEntry e{}; e.binding = 0;
          e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
          e.buffer.type = WGPUBufferBindingType_Uniform;
          e.buffer.minBindingSize = TRANSFORM_UNIFORM_SIZE;
          bglEntries.push_back(e); }

        // Binding 1: material uniforms
        { WGPUBindGroupLayoutEntry e{}; e.binding = 1;
          e.visibility = WGPUShaderStage_Fragment;
          e.buffer.type = WGPUBufferBindingType_Uniform;
          e.buffer.minBindingSize = MATERIAL_UNIFORM_SIZE;
          bglEntries.push_back(e); }

        // Binding 2: light uniforms (if lit)
        if (lit) {
            WGPUBindGroupLayoutEntry e{}; e.binding = 2;
            e.visibility = WGPUShaderStage_Fragment;
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.minBindingSize = LIGHT_UNIFORM_SIZE;
            bglEntries.push_back(e);
        }

        // Binding 3: texture, Binding 4: sampler (if textured)
        if (features & FEAT_TEXTURE) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 3;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 4;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        // Binding 7-9: shadow map (if shadows enabled)
        if (features & FEAT_SHADOW) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 7;
              e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
              e.buffer.type = WGPUBufferBindingType_Uniform;
              e.buffer.minBindingSize = SHADOW_UNIFORM_SIZE;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 8;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Depth;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 9;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Comparison;
              bglEntries.push_back(e); }
        }

        // Binding 5: normal map, Binding 6: normal map sampler
        if (features & FEAT_NORMAL_MAP) {
            { WGPUBindGroupLayoutEntry e{}; e.binding = 5;
              e.visibility = WGPUShaderStage_Fragment;
              e.texture.sampleType = WGPUTextureSampleType_Float;
              e.texture.viewDimension = WGPUTextureViewDimension_2D;
              bglEntries.push_back(e); }
            { WGPUBindGroupLayoutEntry e{}; e.binding = 6;
              e.visibility = WGPUShaderStage_Fragment;
              e.sampler.type = WGPUSamplerBindingType_Filtering;
              bglEntries.push_back(e); }
        }

        WGPUBindGroupLayoutDescriptor bglDesc{};
        WGPUStringView bglLabel = {.data = "bind_group_layout", .length = 17};
        bglDesc.label = bglLabel;
        bglDesc.entryCount = bglEntries.size();
        bglDesc.entries = bglEntries.data();
        entry.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        // Pipeline layout
        WGPUPipelineLayoutDescriptor plDesc{};
        WGPUStringView plLabel = {.data = "pipeline_layout", .length = 15};
        plDesc.label = plLabel;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &entry.bindGroupLayout;
        entry.layout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex buffer layout: pos(vec3) + normal(vec3) + uv(vec2) = 32 bytes
        WGPUVertexAttribute attrs[3]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0; attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = dawn::VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 3;
        vbLayout.attributes = attrs;

        // Blend state — driven by the blend bits in the pipeline key
        WGPUBlendState blendState{};
        uint32_t blendBits = features & BLEND_MASK;
        WGPUColorTargetState colorTarget{};
        colorTarget.format = surfaceFormat;
        colorTarget.writeMask = WGPUColorWriteMask_All;
        if (blendBits == BLEND_DISABLED) {
            // No blending
            colorTarget.blend = nullptr;
        } else {
            if (blendBits == BLEND_ADDITIVE) {
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_One;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == BLEND_SUBTRACTIVE) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrc;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == BLEND_MULTIPLY) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_Src;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else {
                // BLEND_NORMAL (default)
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            }
            colorTarget.blend = &blendState;
        }

        WGPUStringView fsEntry = {.data = "fs_main", .length = 7};
        WGPUFragmentState fragmentState{};
        fragmentState.module = entry.shader;
        fragmentState.entryPoint = fsEntry;
        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth24Plus;
        depthStencil.depthWriteEnabled = (features & DEPTH_WRITE_OFF)
                                          ? WGPUOptionalBool_False
                                          : WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        WGPURenderPipelineDescriptor pipelineDesc{};
        WGPUStringView pipeLabel = {.data = "dawn_pipeline", .length = 13};
        pipelineDesc.label = pipeLabel;
        pipelineDesc.layout = entry.layout;

        WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
        pipelineDesc.vertex.module = entry.shader;
        pipelineDesc.vertex.entryPoint = vsEntry;
        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &vbLayout;

        // Topology selection: wireframe, line, points, or triangles
        uint32_t topoBits = features & TOPO_MASK;
        if (features & WIREFRAME_BIT) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == TOPO_LINE_LIST) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == TOPO_LINE_STRIP) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineStrip;
            pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Uint32;
        } else if (topoBits == TOPO_POINT_LIST) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_PointList;
        } else {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        }
        pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;

        // Face culling from pipeline key
        uint32_t cullBits = features & CULL_MASK;
        if (cullBits == CULL_FRONT) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Front;
        } else if (cullBits == CULL_BACK) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Back;
        } else {
            pipelineDesc.primitive.cullMode = WGPUCullMode_None;
        }
        pipelineDesc.depthStencil = &depthStencil;
        pipelineDesc.multisample.count = 1;
        pipelineDesc.multisample.mask = 0xFFFFFFFF;
        pipelineDesc.fragment = &fragmentState;

        entry.pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
        if (!entry.pipeline) {
            std::cerr << "DawnRenderer: Failed to create render pipeline for features 0x"
                      << std::hex << features << std::dec << std::endl;
        }

        pipelineCache[features] = entry;
        return pipelineCache[features];
    }

    void createUniformBuffers() {
        auto makeBuffer = [&](const char* name, size_t nameLen, size_t sz) {
            WGPUBufferDescriptor d{};
            d.label = {.data = name, .length = nameLen};
            d.size = sz;
            d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            return wgpuDeviceCreateBuffer(device, &d);
        };
        transformBuffer = makeBuffer("transform_buf", 13, TRANSFORM_UNIFORM_SIZE);
        materialBuffer  = makeBuffer("material_buf", 12, MATERIAL_UNIFORM_SIZE);
        lightBuffer     = makeBuffer("light_buf", 9, LIGHT_UNIFORM_SIZE);
    }

    // Shadow map helpers
    void initShadowMap() {
        if (shadowState.depthTexture) return; // already initialized

        // Create depth texture for shadow map
        WGPUTextureDescriptor td{};
        td.label = {.data = "shadow_depth", .length = 12};
        td.size = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_Depth32Float;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        shadowState.depthTexture = wgpuDeviceCreateTexture(device, &td);
        shadowState.depthView = wgpuTextureCreateView(shadowState.depthTexture, nullptr);

        // Create comparison sampler
        WGPUSamplerDescriptor sd{};
        sd.label = {.data = "shadow_samp", .length = 11};
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.magFilter = WGPUFilterMode_Linear;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.compare = WGPUCompareFunction_Less;
        shadowState.comparisonSampler = wgpuDeviceCreateSampler(device, &sd);

        // Create shadow uniform buffer
        WGPUBufferDescriptor bd{};
        bd.label = {.data = "shadow_ub", .length = 9};
        bd.size = SHADOW_UNIFORM_SIZE;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        shadowState.uniformBuffer = wgpuDeviceCreateBuffer(device, &bd);

        // Create depth-only transform buffer
        bd.label = {.data = "shadow_xform", .length = 12};
        bd.size = 64; // just the lightVP matrix
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        shadowState.depthTransformBuffer = wgpuDeviceCreateBuffer(device, &bd);

        // Create depth-only render pipeline
        std::string depthWGSL = R"(
struct DepthUniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: DepthUniforms;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
@vertex fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return u.mvp * vec4<f32>(in.position, 1.0);
}
@fragment fn fs_main() {}
)";

        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.code = {.data = depthWGSL.c_str(), .length = depthWGSL.size()};

        WGPUShaderModuleDescriptor smd{};
        smd.nextInChain = &wgslSource.chain;
        smd.label = {.data = "shadow_shader", .length = 13};
        shadowState.depthShader = wgpuDeviceCreateShaderModule(device, &smd);

        // Bind group layout: one uniform buffer
        WGPUBindGroupLayoutEntry bglEntry{};
        bglEntry.binding = 0;
        bglEntry.visibility = WGPUShaderStage_Vertex;
        bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
        bglEntry.buffer.minBindingSize = 64;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = {.data = "shadow_bgl", .length = 10};
        bglDesc.entryCount = 1;
        bglDesc.entries = &bglEntry;
        shadowState.depthBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = {.data = "shadow_pl", .length = 9};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &shadowState.depthBindGroupLayout;
        shadowState.depthPipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex layout (same as main pipeline)
        WGPUVertexAttribute attrs[3]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0; attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = dawn::VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 3;
        vbLayout.attributes = attrs;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth32Float;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        WGPURenderPipelineDescriptor pipeDesc{};
        pipeDesc.label = {.data = "shadow_pipe", .length = 11};
        pipeDesc.layout = shadowState.depthPipelineLayout;

        WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
        pipeDesc.vertex.module = shadowState.depthShader;
        pipeDesc.vertex.entryPoint = vsEntry;
        pipeDesc.vertex.bufferCount = 1;
        pipeDesc.vertex.buffers = &vbLayout;

        pipeDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipeDesc.primitive.frontFace = WGPUFrontFace_CCW;
        pipeDesc.primitive.cullMode = WGPUCullMode_Front; // Render back faces to prevent shadow acne
        pipeDesc.depthStencil = &depthStencil;
        pipeDesc.multisample.count = 1;
        pipeDesc.multisample.mask = 0xFFFFFFFF;
        // No fragment state needed for depth-only pass
        pipeDesc.fragment = nullptr;

        shadowState.depthPipeline = wgpuDeviceCreateRenderPipeline(device, &pipeDesc);
    }

    void renderShadowPass(WGPUCommandEncoder encoder, Object3D& scene, const Matrix4& lightVP) {
        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = shadowState.depthView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = {.data = "shadow_pass", .length = 11};
        passDesc.colorAttachmentCount = 0;
        passDesc.colorAttachments = nullptr;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
        wgpuRenderPassEncoderSetViewport(pass, 0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0.0f, 1.0f);
        wgpuRenderPassEncoderSetPipeline(pass, shadowState.depthPipeline);

        renderShadowObject(pass, scene, lightVP);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

    void renderShadowObject(WGPURenderPassEncoder pass, Object3D& object, const Matrix4& lightVP) {
        if (auto mesh = object.as<Mesh>()) {
            auto geometry = mesh->geometry();
            if (mesh->castShadow && geometry && geometry->hasAttribute("position")) {
                // Compute MVP for this mesh from light's perspective
                Matrix4 mvp;
                mvp.multiplyMatrices(lightVP, *mesh->matrixWorld);

                // Upload MVP matrix
                wgpuQueueWriteBuffer(queue, shadowState.depthTransformBuffer, 0, mvp.elements.data(), 64);

                // Create bind group
                WGPUBindGroupEntry entry{};
                entry.binding = 0;
                entry.buffer = shadowState.depthTransformBuffer;
                entry.offset = 0;
                entry.size = 64;

                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.label = {.data = "shadow_bg", .length = 9};
                bgDesc.layout = shadowState.depthBindGroupLayout;
                bgDesc.entryCount = 1;
                bgDesc.entries = &entry;
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

                auto& gb = geometries->getOrCreateGeometryBuffers(geometry.get());
                if (gb.vertexBuffer) {
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                         gb.vertexCount * dawn::VERTEX_STRIDE);
                    if (gb.indexBuffer) {
                        wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                             WGPUIndexFormat_Uint32, 0,
                                                             gb.indexCount * sizeof(uint32_t));
                        wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                    } else {
                        wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                    }
                }
                wgpuBindGroupRelease(bg);
            }
        }

        for (auto& child : object.children) {
            renderShadowObject(pass, *child, lightVP);
        }
    }

    void disposeShadowMap() {
        if (shadowState.depthTexture) {
            wgpuTextureViewRelease(shadowState.depthView);
            wgpuTextureRelease(shadowState.depthTexture);
            wgpuSamplerRelease(shadowState.comparisonSampler);
            wgpuBufferRelease(shadowState.uniformBuffer);
            wgpuBufferRelease(shadowState.depthTransformBuffer);
            wgpuRenderPipelineRelease(shadowState.depthPipeline);
            wgpuPipelineLayoutRelease(shadowState.depthPipelineLayout);
            wgpuBindGroupLayoutRelease(shadowState.depthBindGroupLayout);
            wgpuShaderModuleRelease(shadowState.depthShader);
            shadowState = {};
        }
    }

    // Render target helpers (Feature 5)
    RTEntry& getOrCreateRT(RenderTarget* rt) {
        auto it = rtCache.find(rt->uuid);
        if (it != rtCache.end() && it->second.width == rt->width && it->second.height == rt->height) {
            return it->second;
        }
        // Release old
        if (it != rtCache.end()) {
            auto& old = it->second;
            if (old.colorView) wgpuTextureViewRelease(old.colorView);
            if (old.colorTexture) wgpuTextureRelease(old.colorTexture);
            if (old.depthView) wgpuTextureViewRelease(old.depthView);
            if (old.depthTexture) wgpuTextureRelease(old.depthTexture);
        }

        RTEntry entry{};
        entry.width = rt->width;
        entry.height = rt->height;

        WGPUTextureDescriptor ctd{};
        ctd.label = {.data = "rt_color", .length = 8};
        ctd.size = {rt->width, rt->height, 1};
        ctd.mipLevelCount = 1;
        ctd.sampleCount = 1;
        ctd.dimension = WGPUTextureDimension_2D;
        ctd.format = surfaceFormat;
        ctd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
        entry.colorTexture = wgpuDeviceCreateTexture(device, &ctd);
        entry.colorView = wgpuTextureCreateView(entry.colorTexture, nullptr);

        WGPUTextureDescriptor dtd{};
        dtd.label = {.data = "rt_depth", .length = 8};
        dtd.size = {rt->width, rt->height, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = 1;
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        entry.depthTexture = wgpuDeviceCreateTexture(device, &dtd);
        entry.depthView = wgpuTextureCreateView(entry.depthTexture, nullptr);

        rtCache[rt->uuid] = entry;
        return rtCache[rt->uuid];
    }

    // Pack light data into GPU buffer using world-space coordinates.
    // Unlike GLRenderer which uses view-space via Lights::setupView(), the Dawn
    // renderer computes lighting in world space, so we extract world-space
    // positions/directions directly from the light objects.
    void uploadLightDataWorldSpace(Object3D& scene) {
        std::vector<float> data(LIGHT_UNIFORM_SIZE / sizeof(float), 0.0f);
        auto* u32 = reinterpret_cast<uint32_t*>(data.data());

        uint32_t nDir = 0, nPt = 0, nSp = 0, nHm = 0;
        float ambR = 0, ambG = 0, ambB = 0;

        // Temporary storage
        struct DirEntry { Vector3 dir; Color col; };
        struct PtEntry  { Vector3 pos; Color col; float dist; float decay; };
        struct SpEntry  { Vector3 pos; Vector3 dir; Color col; float dist; float decay; float coneCos; float penumbraCos; };
        struct HmEntry  { Vector3 dir; Color sky; Color gnd; };
        std::vector<DirEntry> dirs;
        std::vector<PtEntry>  pts;
        std::vector<SpEntry>  sps;
        std::vector<HmEntry>  hms;

        std::function<void(Object3D&)> collect = [&](Object3D& obj) {
            if (auto al = obj.as<AmbientLight>()) {
                ambR += al->color.r * al->intensity;
                ambG += al->color.g * al->intensity;
                ambB += al->color.b * al->intensity;
            } else if (auto dl = obj.as<DirectionalLight>()) {
                if (dirs.size() < static_cast<size_t>(MAX_DIR_LIGHTS)) {
                    Vector3 lightPos, targetPos;
                    lightPos.setFromMatrixPosition(*dl->matrixWorld);
                    targetPos.setFromMatrixPosition(*dl->target().matrixWorld);
                    Vector3 direction = lightPos.clone().sub(targetPos).normalize();
                    dirs.push_back({direction, Color(dl->color).multiplyScalar(dl->intensity)});
                }
            } else if (auto pl = obj.as<PointLight>()) {
                if (pts.size() < static_cast<size_t>(MAX_POINT_LIGHTS)) {
                    Vector3 pos;
                    pos.setFromMatrixPosition(*pl->matrixWorld);
                    pts.push_back({pos, Color(pl->color).multiplyScalar(pl->intensity), pl->distance, pl->decay});
                }
            } else if (auto sl = obj.as<SpotLight>()) {
                if (sps.size() < static_cast<size_t>(MAX_SPOT_LIGHTS)) {
                    Vector3 pos, targetPos;
                    pos.setFromMatrixPosition(*sl->matrixWorld);
                    targetPos.setFromMatrixPosition(*sl->target().matrixWorld);
                    Vector3 direction = pos.clone().sub(targetPos).normalize();
                    sps.push_back({pos, direction, Color(sl->color).multiplyScalar(sl->intensity),
                                   sl->distance, sl->decay,
                                   std::cos(sl->angle), std::cos(sl->angle * (1.0f - sl->penumbra))});
                }
            } else if (auto hl = obj.as<HemisphereLight>()) {
                if (hms.size() < static_cast<size_t>(MAX_HEMI_LIGHTS)) {
                    Vector3 dir;
                    dir.setFromMatrixPosition(*hl->matrixWorld).normalize();
                    hms.push_back({dir, Color(hl->color).multiplyScalar(hl->intensity),
                                   Color(hl->groundColor).multiplyScalar(hl->intensity)});
                }
            }
            for (auto& child : obj.children) collect(*child);
        };
        collect(scene);

        nDir = dirs.size(); nPt = pts.size(); nSp = sps.size(); nHm = hms.size();
        u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
        data[4] = ambR; data[5] = ambG; data[6] = ambB; data[7] = 0;

        size_t off = 8;
        for (uint32_t i = 0; i < nDir; i++) {
            data[off+0] = dirs[i].dir.x; data[off+1] = dirs[i].dir.y; data[off+2] = dirs[i].dir.z; data[off+3] = 0;
            data[off+4] = dirs[i].col.r; data[off+5] = dirs[i].col.g; data[off+6] = dirs[i].col.b; data[off+7] = 0;
            off += 8;
        }

        off = 8 + MAX_DIR_LIGHTS * 8;
        for (uint32_t i = 0; i < nPt; i++) {
            data[off+0] = pts[i].pos.x; data[off+1] = pts[i].pos.y; data[off+2] = pts[i].pos.z; data[off+3] = 0;
            data[off+4] = pts[i].col.r; data[off+5] = pts[i].col.g; data[off+6] = pts[i].col.b; data[off+7] = pts[i].dist;
            data[off+8] = pts[i].decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
            off += 12;
        }

        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12;
        for (uint32_t i = 0; i < nSp; i++) {
            data[off+0] = sps[i].pos.x; data[off+1] = sps[i].pos.y; data[off+2] = sps[i].pos.z; data[off+3] = 0;
            data[off+4] = sps[i].dir.x; data[off+5] = sps[i].dir.y; data[off+6] = sps[i].dir.z; data[off+7] = 0;
            data[off+8] = sps[i].col.r; data[off+9] = sps[i].col.g; data[off+10] = sps[i].col.b; data[off+11] = sps[i].dist;
            data[off+12] = sps[i].decay; data[off+13] = sps[i].coneCos; data[off+14] = sps[i].penumbraCos; data[off+15] = 0;
            off += 16;
        }

        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12 + MAX_SPOT_LIGHTS * 16;
        for (uint32_t i = 0; i < nHm; i++) {
            data[off+0] = hms[i].dir.x; data[off+1] = hms[i].dir.y; data[off+2] = hms[i].dir.z; data[off+3] = 0;
            data[off+4] = hms[i].sky.r; data[off+5] = hms[i].sky.g; data[off+6] = hms[i].sky.b; data[off+7] = 0;
            data[off+8] = hms[i].gnd.r; data[off+9] = hms[i].gnd.g; data[off+10] = hms[i].gnd.b; data[off+11] = 0;
            off += 12;
        }

        wgpuQueueWriteBuffer(queue, lightBuffer, 0, data.data(), LIGHT_UNIFORM_SIZE);
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        // Reset per-frame statistics
        renderInfo.frame++;
        renderInfo.calls = 0;
        renderInfo.triangles = 0;
        renderInfo.lines = 0;
        renderInfo.points = 0;
        renderInfo.geometries = geometries->count();
        renderInfo.textures = textures->count();

        // Update window size if changed
        auto currentSize = canvas.size();
        if (currentSize.width() != size_.width() || currentSize.height() != size_.height()) {
            size_ = currentSize;
            if (surface) configureSurface();
            viewport_.w = static_cast<float>(size_.width());
            viewport_.h = static_cast<float>(size_.height());
            scissor_.w = static_cast<uint32_t>(size_.width());
            scissor_.h = static_cast<uint32_t>(size_.height());
        }

        // Update matrices
        scene.updateMatrixWorld();
        if (!camera.parent) {
            camera.updateMatrixWorld();
        }
        camera.updateWorldMatrix(false, false);

        Matrix4 projectionMatrix = camera.projectionMatrix;
        Matrix4 viewMatrix = camera.matrixWorldInverse;

        // Upload world-space light data directly from the scene
        uploadLightDataWorldSpace(scene);

        // Shadow pass: find the first shadow-casting directional light
        shadowState.active = false;
        {
            // Look for a shadow-casting directional light
            DirectionalLight* shadowLight = nullptr;
            std::function<void(Object3D&)> findShadowLight = [&](Object3D& obj) {
                if (shadowLight) return;
                if (auto dl = obj.as<DirectionalLight>()) {
                    if (dl->castShadow) shadowLight = dl;
                }
                for (auto& child : obj.children) findShadowLight(*child);
            };
            findShadowLight(scene);

            if (shadowLight && shadowLight->shadow) {
                initShadowMap();

                // Compute light view-projection matrix
                auto& shadow = shadowLight->shadow;
                shadow->updateMatrices(*shadowLight);
                shadowState.lightVP = shadow->matrix;
                shadowState.bias = shadow->bias;
                shadowState.normalBias = shadow->normalBias;
                shadowState.active = true;

                // Upload shadow uniform buffer
                float shadowData[SHADOW_UNIFORM_SIZE / sizeof(float)];
                std::memset(shadowData, 0, sizeof(shadowData));
                std::memcpy(shadowData, shadow->matrix.elements.data(), 64);
                shadowData[16] = shadow->bias;
                shadowData[17] = shadow->normalBias;
                wgpuQueueWriteBuffer(queue, shadowState.uniformBuffer, 0, shadowData, SHADOW_UNIFORM_SIZE);

                // Create command encoder for shadow pass
                WGPUCommandEncoderDescriptor shadowEncDesc{};
                shadowEncDesc.label = {.data = "shadow_enc", .length = 10};
                WGPUCommandEncoder shadowEncoder = wgpuDeviceCreateCommandEncoder(device, &shadowEncDesc);

                // Compute light VP for depth-only rendering (without bias)
                Matrix4 lightVP;
                lightVP.multiplyMatrices(shadow->camera->projectionMatrix, shadow->camera->matrixWorldInverse);

                renderShadowPass(shadowEncoder, scene, lightVP);

                WGPUCommandBufferDescriptor shadowCmdDesc{};
                shadowCmdDesc.label = {.data = "shadow_cmd", .length = 10};
                WGPUCommandBuffer shadowCmd = wgpuCommandEncoderFinish(shadowEncoder, &shadowCmdDesc);
                wgpuQueueSubmit(queue, 1, &shadowCmd);
                wgpuCommandBufferRelease(shadowCmd);
                wgpuCommandEncoderRelease(shadowEncoder);
            }
        }

        // Determine render target views
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTexture frameDepthTexture = nullptr;
        WGPUSurfaceTexture surfaceTexture{};
        bool useSurface = (currentRenderTarget_ == nullptr && surface != nullptr);

        if (currentRenderTarget_ == nullptr && surface == nullptr) {
            // Headless mode with no render target set — nothing to render to
            return;
        }

        if (useSurface) {
            wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
            if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
                std::cerr << "DawnRenderer: Failed to acquire surface texture (status "
                          << static_cast<int>(surfaceTexture.status) << ")" << std::endl;
                return;
            }
            WGPUTextureViewDescriptor vd{};
            vd.label = {.data = "surface_view", .length = 12};
            vd.format = surfaceFormat;
            vd.dimension = WGPUTextureViewDimension_2D;
            vd.baseMipLevel = 0; vd.mipLevelCount = 1;
            vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
            vd.aspect = WGPUTextureAspect_All;
            colorView = wgpuTextureCreateView(surfaceTexture.texture, &vd);

            uint32_t w = static_cast<uint32_t>(size_.width());
            uint32_t h = static_cast<uint32_t>(size_.height());
            WGPUTextureDescriptor dtd{};
            dtd.label = {.data = "depth_tex", .length = 9};
            dtd.size = {w, h, 1};
            dtd.mipLevelCount = 1; dtd.sampleCount = 1;
            dtd.dimension = WGPUTextureDimension_2D;
            dtd.format = WGPUTextureFormat_Depth24Plus;
            dtd.usage = WGPUTextureUsage_RenderAttachment;
            frameDepthTexture = wgpuDeviceCreateTexture(device, &dtd);
            depthView = wgpuTextureCreateView(frameDepthTexture, nullptr);
        } else {
            auto& rt = getOrCreateRT(currentRenderTarget_);
            colorView = rt.colorView;
            depthView = rt.depthView;
        }

        // Command encoder
        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = {.data = "cmd_enc", .length = 7};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        // Determine clear color (scene background overrides if set)
        auto* sceneObj = scene.as<Scene>();
        Color effectiveClearColor = clearColor_;
        float effectiveClearAlpha = clearAlpha_;
        if (sceneObj && sceneObj->background.isColor()) {
            effectiveClearColor = sceneObj->background.color();
            effectiveClearAlpha = 1.0f;
        }

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = nullptr;
        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(effectiveClearColor.r),
                static_cast<double>(effectiveClearColor.g),
                static_cast<double>(effectiveClearColor.b),
                static_cast<double>(effectiveClearAlpha)};

        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = {.data = "render_pass", .length = 11};
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

        // Set viewport (Feature 4)
        wgpuRenderPassEncoderSetViewport(pass, viewport_.x, viewport_.y, viewport_.w, viewport_.h, 0.0f, 1.0f);
        if (scissorTest_) {
            wgpuRenderPassEncoderSetScissorRect(pass, scissor_.x, scissor_.y, scissor_.w, scissor_.h);
        }

        // Set up frustum for culling and collect renderables
        renderList_.init();
        Matrix4 projScreenMatrix;
        projScreenMatrix.multiplyMatrices(projectionMatrix, viewMatrix);
        frustum_.setFromProjectionMatrix(projScreenMatrix);
        collectRenderables(scene, projScreenMatrix, camera, 0);
        if (scope.sortObjects) {
            renderList_.sort();
        }
        renderList_.finish();

        // Extract fog and tone mapping state from scene
        Color fogColor;
        float fogNear = 0, fogFar = 0, fogDensity = 0;
        uint32_t fogBits = 0;
        if (sceneObj && sceneObj->fog) {
            if (auto* f = std::get_if<Fog>(&*sceneObj->fog)) {
                fogColor = f->color;
                fogNear = f->nearPlane;
                fogFar = f->farPlane;
                fogBits = FEAT_FOG_LINEAR;
            } else if (auto* f2 = std::get_if<FogExp2>(&*sceneObj->fog)) {
                fogColor = f2->color;
                fogDensity = f2->density;
                fogBits = FEAT_FOG_EXP2;
            }
        }

        uint32_t tonemapBits = TONEMAP_NONE;
        switch (scope.toneMapping) {
            case ToneMapping::Linear: tonemapBits = TONEMAP_LINEAR; break;
            case ToneMapping::Reinhard: tonemapBits = TONEMAP_REINHARD; break;
            case ToneMapping::Cineon: tonemapBits = TONEMAP_CINEON; break;
            case ToneMapping::ACESFilmic: tonemapBits = TONEMAP_ACES; break;
            default: break;
        }

        // Render opaque objects (front-to-back, depth write on)
        for (auto* item : renderList_.opaque) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera,
                       fogBits, fogColor, fogNear, fogFar, fogDensity,
                       tonemapBits);
        }

        // Render transparent objects (back-to-front, depth write off)
        for (auto* item : renderList_.transparent) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera,
                       fogBits, fogColor, fogNear, fogFar, fogDensity,
                       tonemapBits);
        }

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // Submit
        WGPUCommandBufferDescriptor cmdDesc{};
        cmdDesc.label = {.data = "cmd_buf", .length = 7};
        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmdBuffer);

        if (useSurface) {
            wgpuSurfacePresent(surface);
        }

        // Cleanup per-frame resources
        wgpuCommandBufferRelease(cmdBuffer);
        wgpuCommandEncoderRelease(encoder);
        if (useSurface) {
            wgpuTextureViewRelease(depthView);
            wgpuTextureRelease(frameDepthTexture);
            wgpuTextureViewRelease(colorView);
            wgpuTextureRelease(surfaceTexture.texture);
        }
    }

    // Collect all renderable objects into the render list with z-depth for sorting.
    // Mirrors GLRenderer's projectObject with frustum culling, LOD, Sprite, Line, Points support.
    void collectRenderables(Object3D& object, const Matrix4& projScreenMatrix,
                            Camera& camera, unsigned int groupOrder) {
        if (!object.visible) return;

        if (object.is<Group>()) {
            groupOrder = object.renderOrder;
        } else if (auto lod = object.as<LOD>()) {
            if (lod->autoUpdate) lod->update(camera);
        } else if (auto sprite = object.as<Sprite>()) {
            if (!object.frustumCulled || frustum_.intersectsSprite(*sprite)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*sprite->matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }
                auto material = sprite->material().get();
                if (material && material->visible) {
                    renderList_.push(sprite, nullptr, material, groupOrder, _vector3.z, std::nullopt);
                }
            }
        } else if (object.is<Mesh>() || object.is<Line>() || object.is<Points>()) {
            if (!object.frustumCulled || frustum_.intersectsObject(object)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*object.matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }

                auto* owm = object.as<ObjectWithMaterials>();
                if (owm) {
                    auto geometry = owm->geometry();
                    if (geometry && geometry->hasAttribute("position")) {
                        const auto& materials = owm->materials();
                        if (materials.size() > 1) {
                            const auto& groups = geometry->groups;
                            for (const auto& group : groups) {
                                auto groupMat = materials.at(group.materialIndex).get();
                                if (groupMat && groupMat->visible) {
                                    renderList_.push(&object, geometry.get(), groupMat,
                                                     groupOrder, _vector3.z, group);
                                }
                            }
                        } else if (!materials.empty() && materials.front()->visible) {
                            renderList_.push(&object, geometry.get(), materials.front().get(),
                                             groupOrder, _vector3.z, std::nullopt);
                        }
                    }
                }
            }
        }

        for (auto& child : object.children) {
            collectRenderables(*child, projScreenMatrix, camera, groupOrder);
        }
    }

    // Render a single item from the render list.
    void renderItem(WGPURenderPassEncoder pass, const RenderItem* item,
                    const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                    const Camera& camera,
                    uint32_t fogBits, const Color& fogColor,
                    float fogNear, float fogFar, float fogDensity,
                    uint32_t tonemapBits) {

        auto* object = item->object;
        auto* geometry = item->geometry;
        Material* rawMat = item->material;
        if (!object || !rawMat) return;

        // Determine object type
        bool isMesh = object->is<Mesh>();
        bool isLine = object->is<Line>();
        bool isPoints = object->is<Points>();
        bool isLineSegments = object->is<LineSegments>();
        auto* instancedMesh = object->as<InstancedMesh>();

        // Geometry comes from the render item (set during collection)
        // For sprites without geometry, skip for now
        if (!geometry) return;

        // Determine features and extract material parameters
        uint32_t features = FEAT_NONE;
        Color diffuse(1, 1, 1);
        float opacity = rawMat->opacity;
        Color specularColor(0, 0, 0);
        float shininess = 30.0f;
        float roughness = 0.5f, metalness = 0.0f;
        Color emissive(0, 0, 0);
        Texture* diffuseMap = nullptr;
        Texture* normalMap = nullptr;
        Vector2 normalScale(1, 1);

        if (auto m = dynamic_cast<MeshStandardMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING | FEAT_PBR;
            diffuse = m->color; roughness = m->roughness; metalness = m->metalness;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
            if (m->normalMap) {
                normalMap = m->normalMap.get();
                normalScale = m->normalScale;
                features |= FEAT_NORMAL_MAP;
            }
        } else if (auto m = dynamic_cast<MeshPhongMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING | FEAT_SPECULAR;
            diffuse = m->color; specularColor = m->specular; shininess = m->shininess;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto m = dynamic_cast<MeshLambertMaterial*>(rawMat)) {
            features |= FEAT_LIGHTING;
            diffuse = m->color;
            emissive = m->emissive;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto m = dynamic_cast<MeshBasicMaterial*>(rawMat)) {
            diffuse = m->color;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto m = dynamic_cast<LineBasicMaterial*>(rawMat)) {
            diffuse = m->color;
        } else if (auto m = dynamic_cast<PointsMaterial*>(rawMat)) {
            diffuse = m->color;
            if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
        } else if (auto cm = dynamic_cast<MaterialWithColor*>(rawMat)) {
            diffuse = cm->color;
        }

        // Set topology based on object type
        bool isLineLoop = object->is<LineLoop>();
        if (isLine) {
            if (object->is<LineSegments>() || isLineLoop) {
                // LineLoop uses LineList with a generated index buffer that closes the loop
                features |= TOPO_LINE_LIST;
            } else {
                features |= TOPO_LINE_STRIP;
            }
        } else if (isPoints) {
            features |= TOPO_POINT_LIST;
        }

        // Face culling based on material.side
        switch (rawMat->side) {
            case Side::Front: features |= CULL_BACK; break;
            case Side::Back:  features |= CULL_FRONT; break;
            case Side::Double: features |= CULL_NONE; break;
        }

        // Wireframe mode (only for mesh objects)
        bool useWireframe = false;
        if (isMesh) {
            if (auto wf = dynamic_cast<MaterialWithWireframe*>(rawMat)) {
                if (wf->wireframe) {
                    features |= WIREFRAME_BIT;
                    useWireframe = true;
                }
            }
        }

        // Blend mode
        auto blendVal = static_cast<int>(rawMat->blending);
        if (blendVal == 0)              features |= BLEND_DISABLED;
        else if (blendVal == 2)         features |= BLEND_ADDITIVE;
        else if (blendVal == 3)         features |= BLEND_SUBTRACTIVE;
        else if (blendVal == 4)         features |= BLEND_MULTIPLY;
        else                            features |= BLEND_NORMAL;
        if (rawMat->transparent) {
            features |= DEPTH_WRITE_OFF;
        }

        // Shadow (mesh objects only)
        if (isMesh && shadowState.active && object->receiveShadow) {
            features |= FEAT_SHADOW;
        }

        // Fog and tone mapping
        if (rawMat->fog) {
            features |= fogBits;
        }
        features |= tonemapBits;

        // Get/create pipeline for this feature set
        auto& pe = getOrCreatePipeline(features);
        if (!pe.pipeline) return;

        // Upload transform uniforms
        float transformData[TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, object->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix4 modelView;
        modelView.multiplyMatrices(viewMatrix, *object->matrixWorld);
        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelView);
        normalMatrix.invert();
        normalMatrix.transpose();
        auto& ne = normalMatrix.elements;
        transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
        transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
        transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        transformData[60] = camPos.x;
        transformData[61] = camPos.y;
        transformData[62] = camPos.z;
        transformData[63] = 0;

        wgpuQueueWriteBuffer(queue, transformBuffer, 0, transformData, TRANSFORM_UNIFORM_SIZE);

        // Upload material uniforms (now includes fog and tone mapping data)
        float matData[MATERIAL_UNIFORM_SIZE / sizeof(float)];
        std::memset(matData, 0, sizeof(matData));
        matData[0] = diffuse.r; matData[1] = diffuse.g; matData[2] = diffuse.b; matData[3] = 1.0f;
        matData[4] = specularColor.r; matData[5] = specularColor.g; matData[6] = specularColor.b; matData[7] = shininess;
        matData[8] = roughness; matData[9] = metalness; matData[10] = opacity; matData[11] = 0;
        matData[12] = emissive.r; matData[13] = emissive.g; matData[14] = emissive.b; matData[15] = 0;
        matData[16] = (features & FEAT_TEXTURE) ? 1.0f : 0.0f;
        matData[17] = (features & FEAT_LIGHTING) ? 1.0f : 0.0f;
        matData[18] = normalScale.x;
        matData[19] = normalScale.y;
        // fogColor (vec4, offset 20)
        matData[20] = fogColor.r; matData[21] = fogColor.g; matData[22] = fogColor.b; matData[23] = 1.0f;
        // fogParams: x=near, y=far, z=density, w=toneMappingExposure (vec4, offset 24)
        matData[24] = fogNear; matData[25] = fogFar; matData[26] = fogDensity;
        matData[27] = scope.toneMappingExposure;
        wgpuQueueWriteBuffer(queue, materialBuffer, 0, matData, MATERIAL_UNIFORM_SIZE);

        // Build bind group dynamically
        bool lit = features & (FEAT_LIGHTING | FEAT_SPECULAR | FEAT_PBR);
        bool tex = features & FEAT_TEXTURE;

        std::vector<WGPUBindGroupEntry> entries;
        { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = transformBuffer; e.offset = 0; e.size = TRANSFORM_UNIFORM_SIZE; entries.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = materialBuffer; e.offset = 0; e.size = MATERIAL_UNIFORM_SIZE; entries.push_back(e); }

        if (lit) {
            WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = lightBuffer; e.offset = 0; e.size = LIGHT_UNIFORM_SIZE; entries.push_back(e);
        }

        auto* texEntry = &textures->getDummyTexture();
        if (tex && diffuseMap) {
            texEntry = &textures->getOrCreateTexture(diffuseMap);
        }
        if (tex) {
            { WGPUBindGroupEntry e{}; e.binding = 3; e.textureView = texEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 4; e.sampler = texEntry->sampler; entries.push_back(e); }
        }

        if (features & FEAT_NORMAL_MAP) {
            auto* nmEntry = &textures->getDummyTexture();
            if (normalMap) {
                nmEntry = &textures->getOrCreateTexture(normalMap);
            }
            { WGPUBindGroupEntry e{}; e.binding = 5; e.textureView = nmEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 6; e.sampler = nmEntry->sampler; entries.push_back(e); }
        }

        if (features & FEAT_SHADOW) {
            { WGPUBindGroupEntry e{}; e.binding = 7; e.buffer = shadowState.uniformBuffer; e.offset = 0; e.size = SHADOW_UNIFORM_SIZE; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 8; e.textureView = shadowState.depthView; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 9; e.sampler = shadowState.comparisonSampler; entries.push_back(e); }
        }

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = {.data = "obj_bg", .length = 6};
        bgDesc.layout = pe.bindGroupLayout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry);
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                     gb.vertexCount * dawn::VERTEX_STRIDE);

            uint32_t instanceCount = 1;
            // InstancedMesh: per-instance transform buffers not yet implemented,
            // render single instance to avoid stacked duplicates at same position
            (void)instancedMesh;

            if (useWireframe) {
                auto& wb = geometries->getOrCreateWireframeBuffers(geometry);
                if (wb.indexBuffer) {
                    wgpuRenderPassEncoderSetIndexBuffer(pass, wb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         wb.indexCount * sizeof(uint32_t));
                    wgpuRenderPassEncoderDrawIndexed(pass, wb.indexCount, instanceCount, 0, 0, 0);
                    renderInfo.calls++;
                    renderInfo.lines += wb.indexCount / 2;
                }
            } else if (isLineLoop) {
                // WebGPU has no line loop primitive — generate line-list indices
                // that include the closing edge (last vertex → first vertex)
                uint32_t n = gb.vertexCount;
                std::vector<uint32_t> loopIndices;
                loopIndices.reserve(n * 2);
                for (uint32_t i = 0; i < n; i++) {
                    loopIndices.push_back(i);
                    loopIndices.push_back((i + 1) % n);
                }
                WGPUBufferDescriptor bd{};
                bd.label = {.data = "lineloop_idx", .length = 12};
                bd.size = loopIndices.size() * sizeof(uint32_t);
                bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                WGPUBuffer loopBuf = wgpuDeviceCreateBuffer(device, &bd);
                wgpuQueueWriteBuffer(queue, loopBuf, 0, loopIndices.data(), bd.size);
                wgpuRenderPassEncoderSetIndexBuffer(pass, loopBuf,
                                                     WGPUIndexFormat_Uint32, 0, bd.size);
                uint32_t drawCount = static_cast<uint32_t>(loopIndices.size());
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, 0, 0, 0);
                wgpuBufferRelease(loopBuf);
                renderInfo.calls++;
                renderInfo.lines += n;
            } else if (gb.indexBuffer) {
                // Determine draw range from geometry group if present
                uint32_t drawStart = 0;
                uint32_t drawCount = gb.indexCount;
                if (item->group.has_value()) {
                    drawStart = static_cast<uint32_t>(item->group->start);
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, drawStart, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            } else {
                uint32_t drawCount = gb.vertexCount;
                if (item->group.has_value()) {
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderDraw(pass, drawCount, instanceCount, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            }
        }

        wgpuBindGroupRelease(bg);
    }

    void dispose() {
        if (!initialized) return;

        // Release geometry and texture subsystems
        if (geometries) geometries->dispose();
        if (textures) textures->dispose();

        // Release render target cache
        for (auto& [id, rt] : rtCache) {
            if (rt.colorView) wgpuTextureViewRelease(rt.colorView);
            if (rt.colorTexture) wgpuTextureRelease(rt.colorTexture);
            if (rt.depthView) wgpuTextureViewRelease(rt.depthView);
            if (rt.depthTexture) wgpuTextureRelease(rt.depthTexture);
        }
        rtCache.clear();

        // Release pipeline cache
        for (auto& [feat, pe] : pipelineCache) {
            if (pe.pipeline) wgpuRenderPipelineRelease(pe.pipeline);
            if (pe.layout) wgpuPipelineLayoutRelease(pe.layout);
            if (pe.bindGroupLayout) wgpuBindGroupLayoutRelease(pe.bindGroupLayout);
            if (pe.shader) wgpuShaderModuleRelease(pe.shader);
        }
        pipelineCache.clear();

        disposeShadowMap();

        if (transformBuffer) wgpuBufferRelease(transformBuffer);
        if (materialBuffer) wgpuBufferRelease(materialBuffer);
        if (lightBuffer) wgpuBufferRelease(lightBuffer);
        if (queue) wgpuQueueRelease(queue);
        if (device) wgpuDeviceRelease(device);
        if (adapter) wgpuAdapterRelease(adapter);
        if (surface) wgpuSurfaceRelease(surface);
        if (instance) wgpuInstanceRelease(instance);

        initialized = false;
    }

    ~Impl() {
        dispose();
    }
};


// --- DawnRenderer public API ---

DawnRenderer::DawnRenderer(Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(*this, canvas)) {}

void DawnRenderer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

WindowSize DawnRenderer::size() const {
    return pimpl_->size_;
}

void DawnRenderer::setSize(const std::pair<int, int>& size) {
    pimpl_->canvas.setSize(size);
    pimpl_->size_ = {size.first, size.second};
    setViewport(0, 0, size.first, size.second);
    if (pimpl_->initialized) {
        pimpl_->configureSurface();
    }
}

float DawnRenderer::getTargetPixelRatio() const {
    return pimpl_->pixelRatio_;
}

void DawnRenderer::setPixelRatio(float value) {
    pimpl_->pixelRatio_ = value;
    setSize({pimpl_->size_.width(), pimpl_->size_.height()});
}

void DawnRenderer::setViewport(const Vector4& v) {
    pimpl_->viewport_.x = v.x; pimpl_->viewport_.y = v.y;
    pimpl_->viewport_.w = v.z; pimpl_->viewport_.h = v.w;
}

void DawnRenderer::setViewport(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->viewport_.x = std::floor(x * pr);
    pimpl_->viewport_.y = std::floor(y * pr);
    pimpl_->viewport_.w = std::floor(width * pr);
    pimpl_->viewport_.h = std::floor(height * pr);
}

void DawnRenderer::setScissor(const Vector4& v) {
    pimpl_->scissor_.x = static_cast<uint32_t>(v.x);
    pimpl_->scissor_.y = static_cast<uint32_t>(v.y);
    pimpl_->scissor_.w = static_cast<uint32_t>(v.z);
    pimpl_->scissor_.h = static_cast<uint32_t>(v.w);
}

void DawnRenderer::setScissor(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->scissor_.x = static_cast<uint32_t>(std::floor(x * pr));
    pimpl_->scissor_.y = static_cast<uint32_t>(std::floor(y * pr));
    pimpl_->scissor_.w = static_cast<uint32_t>(std::floor(width * pr));
    pimpl_->scissor_.h = static_cast<uint32_t>(std::floor(height * pr));
}

void DawnRenderer::getViewport(Vector4& target) const {
    target.set(pimpl_->viewport_.x, pimpl_->viewport_.y, pimpl_->viewport_.w, pimpl_->viewport_.h);
}

void DawnRenderer::setScissorTest(bool boolean) {
    pimpl_->scissorTest_ = boolean;
}

bool DawnRenderer::getScissorTest() const {
    return pimpl_->scissorTest_;
}

void DawnRenderer::getScissor(Vector4& target) const {
    target.set(static_cast<float>(pimpl_->scissor_.x), static_cast<float>(pimpl_->scissor_.y),
               static_cast<float>(pimpl_->scissor_.w), static_cast<float>(pimpl_->scissor_.h));
}

void DawnRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::getClearColor(Color& target) const {
    target = pimpl_->clearColor_;
}

float DawnRenderer::getClearAlpha() const {
    return pimpl_->clearAlpha_;
}

void DawnRenderer::setClearAlpha(float alpha) {
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::clear(bool /*color*/, bool /*depth*/, bool /*stencil*/) {
    // Clearing happens at render pass begin via loadOp = Clear
}

void DawnRenderer::clearColor() {
    clear(true, false, false);
}

void DawnRenderer::clearDepth() {
    clear(false, true, false);
}

void DawnRenderer::clearStencil() {
    clear(false, false, true);
}

RenderTarget* DawnRenderer::getRenderTarget() {
    return pimpl_->currentRenderTarget_;
}

void DawnRenderer::setRenderTarget(RenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {
    pimpl_->currentRenderTarget_ = renderTarget;
    pimpl_->activeCubeFace_ = activeCubeFace;
    pimpl_->activeMipmapLevel_ = activeMipmapLevel;
}

int DawnRenderer::getActiveCubeFace() const {
    return pimpl_->activeCubeFace_;
}

int DawnRenderer::getActiveMipmapLevel() const {
    return pimpl_->activeMipmapLevel_;
}

const DawnInfo& DawnRenderer::info() const {
    static DawnInfo di;
    di.render.frame = pimpl_->renderInfo.frame;
    di.render.calls = pimpl_->renderInfo.calls;
    di.render.triangles = pimpl_->renderInfo.triangles;
    di.render.lines = pimpl_->renderInfo.lines;
    di.render.points = pimpl_->renderInfo.points;
    di.memory.geometries = pimpl_->renderInfo.geometries;
    di.memory.textures = pimpl_->renderInfo.textures;
    return di;
}

void DawnRenderer::resetState() {
    // Dawn manages its own state; this is a no-op for API compatibility
}

std::vector<unsigned char> DawnRenderer::readRGBPixels() {
    if (!pimpl_->initialized || !pimpl_->currentRenderTarget_) return {};

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
    uint32_t w = rt.width;
    uint32_t h = rt.height;

    // Row alignment: WebGPU requires bytesPerRow to be a multiple of 256
    uint32_t bytesPerPixel = 4; // BGRA8
    uint32_t unpaddedBytesPerRow = w * bytesPerPixel;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;
    uint32_t bufferSize = paddedBytesPerRow * h;

    // Create staging buffer
    WGPUBufferDescriptor bd{};
    bd.label = {.data = "readback_buf", .length = 12};
    bd.size = bufferSize;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer stagingBuf = wgpuDeviceCreateBuffer(pimpl_->device, &bd);

    // Copy texture to buffer
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = {.data = "readback_enc", .length = 12};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pimpl_->device, &encDesc);

    WGPUTexelCopyTextureInfo src{};
    src.texture = rt.colorTexture;

    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = stagingBuf;
    dst.layout.bytesPerRow = paddedBytesPerRow;
    dst.layout.rowsPerImage = h;

    WGPUExtent3D extent = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = {.data = "readback_cmd", .length = 12};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(pimpl_->queue, 1, &cmd);

    // Map buffer synchronously
    struct MapData { bool done = false; WGPUMapAsyncStatus status; } mapData;

    WGPUBufferMapCallbackInfo mapCb{};
    mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
    mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
        auto* d = static_cast<MapData*>(ud1);
        d->status = status;
        d->done = true;
    };
    mapCb.userdata1 = &mapData;
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, bufferSize, mapCb);

    auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
    while (!mapData.done) {
        if (std::chrono::steady_clock::now() > deadline) {
            wgpuBufferRelease(stagingBuf);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            throw std::runtime_error("DawnRenderer: readRGBPixels buffer map timed out");
        }
        wgpuDevicePoll(pimpl_->device, true, nullptr);
    }

    std::vector<unsigned char> result;
    if (mapData.status == WGPUMapAsyncStatus_Success) {
        auto* mapped = static_cast<const unsigned char*>(wgpuBufferGetConstMappedRange(stagingBuf, 0, bufferSize));
        result.resize(w * h * 3);
        for (uint32_t row = 0; row < h; row++) {
            for (uint32_t col = 0; col < w; col++) {
                const auto* px = mapped + row * paddedBytesPerRow + col * 4;
                size_t outIdx = (row * w + col) * 3;
                // BGRA -> RGB
                result[outIdx + 0] = px[2];
                result[outIdx + 1] = px[1];
                result[outIdx + 2] = px[0];
            }
        }
        wgpuBufferUnmap(stagingBuf);
    }

    wgpuBufferRelease(stagingBuf);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return result;
}

void DawnRenderer::readPixels(const Vector2& position, const std::pair<int, int>& sz,
                              std::vector<unsigned char>& data) {
    auto allPixels = readRGBPixels();
    if (allPixels.empty()) return;

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
    int rtW = static_cast<int>(rt.width);
    int rtH = static_cast<int>(rt.height);
    int x0 = static_cast<int>(position.x);
    int y0 = static_cast<int>(position.y);
    int w = sz.first;
    int h = sz.second;

    data.resize(w * h * 3);
    for (int row = 0; row < h; row++) {
        int srcY = y0 + row;
        if (srcY < 0 || srcY >= rtH) continue;
        for (int col = 0; col < w; col++) {
            int srcX = x0 + col;
            if (srcX < 0 || srcX >= rtW) continue;
            size_t srcIdx = (srcY * rtW + srcX) * 3;
            size_t dstIdx = (row * w + col) * 3;
            data[dstIdx + 0] = allPixels[srcIdx + 0];
            data[dstIdx + 1] = allPixels[srcIdx + 1];
            data[dstIdx + 2] = allPixels[srcIdx + 2];
        }
    }
}

void DawnRenderer::writeFramebuffer(const std::filesystem::path& filename) {
    auto ext = filename.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
        throw std::runtime_error("Unsupported file format: " + ext);
    }

    auto pixels = readRGBPixels();
    if (pixels.empty()) return;

    int w = pimpl_->size_.width();
    int h = pimpl_->size_.height();
    // WebGPU origin is top-left (no flip needed unlike GL)

    if (filename.has_parent_path() && !std::filesystem::exists(filename.parent_path())) {
        std::error_code ec;
        std::filesystem::create_directories(filename.parent_path(), ec);
    }

    bool success = false;
    if (ext == ".png") {
        success = stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), w * 3);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(filename.string().c_str(), w, h, 3, pixels.data(), 100);
    } else if (ext == ".bmp") {
        success = stbi_write_bmp(filename.string().c_str(), w, h, 3, pixels.data());
    }
    if (!success) {
        throw std::runtime_error("DawnRenderer: failed to write framebuffer to " + filename.string());
    }
}

void DawnRenderer::dispose() {
    pimpl_->dispose();
}

DawnRenderer::~DawnRenderer() = default;
