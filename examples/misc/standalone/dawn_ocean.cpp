// Dawn/WebGPU ocean simulation — port of WebTide (BabylonJS) to threepp + wgpu-native.
// Fully self-contained: creates its own WGPUDevice/Surface, does all GPU work directly.
// Uses threepp only for Canvas (GLFW window) and math types.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

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

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../../src/external/stb/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t TEXTURE_SIZE = 256;
constexpr float TILE_SIZE = 10.0f;
constexpr int GRID_RADIUS = 3; // 7x7 grid
constexpr uint32_t WG = 8;    // workgroup size

// ============================================================================
// WGSL Compute Shaders (copied verbatim from WebTide)
// ============================================================================

constexpr const char* kPhillipsSpectrumWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var H0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var Noise: texture_2d<f32>;

struct Params {
    textureSize: u32,
    tileSize: f32,
    windTheta: f32,
    windSpeed: f32,
    smallWaveLengthCutoff: f32,
};
@group(0) @binding(2) var<uniform> params: Params;

fn phillipsSpectrum2D(k: vec2<f32>) -> f32 {
    if(length(k) < 0.0001) { return 0.0; }
    let A: f32 = 1.0;
    let windDir = vec2<f32>(cos(params.windTheta), sin(params.windTheta));
    let g: f32 = 9.81;
    let L: f32 = params.windSpeed * params.windSpeed / g;
    let k2: f32 = dot(k, k);
    let kL2: f32 = k2 * L * L;
    let k4: f32 = k2 * k2;
    var kw2: f32 = dot(normalize(k), normalize(windDir));
    kw2 *= kw2;
    let l: f32 = params.smallWaveLengthCutoff;
    let cutoff: f32 = exp(-k2 * l * l);
    return A * exp(-1.0 / kL2) * kw2 * cutoff / k4;
}

@compute @workgroup_size(8,8,1)
fn computeSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let deltaK = 2.0 * PI / params.tileSize;
    let nx = f32(id.x) - f32(params.textureSize) / 2.0;
    let nz = f32(id.y) - f32(params.textureSize) / 2.0;
    let k = vec2<f32>(nx, nz) * deltaK;
    let noise_k = textureLoad(Noise, vec2<i32>(id.xy), 0).xy;
    let h0_k = noise_k * sqrt(phillipsSpectrum2D(k) / 2.0);
    let noise_minus_k = textureLoad(Noise, vec2<i32>(params.textureSize - id.xy), 0).xy;
    let h0_minus_k = noise_minus_k * sqrt(phillipsSpectrum2D(-k) / 2.0);
    let h0_minus_k_conj = vec2<f32>(h0_minus_k.x, -h0_minus_k.y);
    textureStore(H0, vec2<i32>(id.xy), vec4<f32>(h0_k, h0_minus_k_conj));
}
)";

constexpr const char* kDynamicSpectrumWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var H0: texture_2d<f32>;
@group(0) @binding(1) var HT: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var DHT: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var Displacement: texture_storage_2d<rg32float, write>;

struct Params {
    textureSize: u32,
    tileSize: f32,
    elapsedSeconds: f32,
};
@group(0) @binding(4) var<uniform> params: Params;

fn omega(k: vec2<f32>) -> f32 { return sqrt(length(k) * 9.81); }
fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn computeSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let deltaK = 2.0 * PI / params.tileSize;
    let n = f32(id.x) - f32(params.textureSize) / 2.0;
    let m = f32(id.y) - f32(params.textureSize) / 2.0;
    let k = vec2<f32>(n, m) * deltaK;
    let theta = params.elapsedSeconds * omega(k);
    let exponent = vec2<f32>(cos(theta), sin(theta));
    let h0: vec4<f32> = textureLoad(H0, iid.xy, 0);
    let h = complexMult(h0.xy, exponent) + complexMult(h0.zw, vec2<f32>(exponent.x, -exponent.y));
    let ih = vec2<f32>(-h.y, h.x);
    let ikh = complexMult(k, ih);
    let displacement = ikh / (length(k) + 0.001);
    textureStore(HT, iid.xy, vec4<f32>(h, vec2(0.0)));
    textureStore(DHT, iid.xy, vec4<f32>(ikh, vec2(0.0)));
    textureStore(Displacement, iid.xy, vec4<f32>(displacement, vec2(0.0)));
}
)";

constexpr const char* kTwiddleFactorsWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var PrecomputeBuffer: texture_storage_2d<rgba32float, write>;

struct Params { step: i32, textureSize: i32, };
@group(0) @binding(1) var<uniform> params: Params;

fn complexExp(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(a.y), sin(a.y)) * exp(a.x);
}

@compute @workgroup_size(1,8,1)
fn precomputeTwiddleFactorsAndInputIndices(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let b = params.textureSize >> (id.x + 1u);
    let mult = 2.0 * PI * vec2<f32>(0.0, -1.0) / f32(params.textureSize);
    let i = (2 * b * (iid.y / b) + (iid.y % b)) % params.textureSize;
    let twiddle = complexExp(mult * vec2<f32>(f32((iid.y / b) * b)));
    textureStore(PrecomputeBuffer, iid.xy, vec4<f32>(twiddle.x, twiddle.y, f32(i), f32(i + b)));
    textureStore(PrecomputeBuffer, vec2<i32>(iid.x, iid.y + params.textureSize / 2), vec4<f32>(-twiddle.x, -twiddle.y, f32(i), f32(i + b)));
}
)";

constexpr const char* kHorizontalStepWGSL = R"(
struct Params { step: i32, textureSize: i32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var PrecomputedData: texture_2d<f32>;
@group(0) @binding(2) var InputBuffer: texture_2d<f32>;
@group(0) @binding(3) var OutputBuffer: texture_storage_2d<rg32float, write>;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn horizontalStepInverseFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let data = textureLoad(PrecomputedData, vec2<i32>(params.step, iid.x), 0);
    let inputsIndices = vec2<i32>(data.ba);
    let input0 = textureLoad(InputBuffer, vec2<i32>(inputsIndices.x, iid.y), 0);
    let input1 = textureLoad(InputBuffer, vec2<i32>(inputsIndices.y, iid.y), 0);
    textureStore(OutputBuffer, iid.xy, vec4<f32>(
        input0.xy + complexMult(vec2<f32>(data.r, -data.g), input1.xy), 0.0, 0.0));
}
)";

constexpr const char* kVerticalStepWGSL = R"(
struct Params { step: i32, textureSize: i32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var PrecomputedData: texture_2d<f32>;
@group(0) @binding(2) var InputBuffer: texture_2d<f32>;
@group(0) @binding(3) var OutputBuffer: texture_storage_2d<rg32float, write>;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn verticalStepInverseFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let data = textureLoad(PrecomputedData, vec2<i32>(params.step, iid.y), 0);
    let inputsIndices = vec2<i32>(data.ba);
    let input0 = textureLoad(InputBuffer, vec2<i32>(iid.x, inputsIndices.x), 0);
    let input1 = textureLoad(InputBuffer, vec2<i32>(iid.x, inputsIndices.y), 0);
    textureStore(OutputBuffer, iid.xy, vec4<f32>(
        input0.xy + complexMult(vec2<f32>(data.r, -data.g), input1.xy), 0.0, 0.0));
}
)";

constexpr const char* kPermutationWGSL = R"(
@group(0) @binding(0) var InputBuffer: texture_2d<f32>;
@group(0) @binding(1) var OutputBuffer: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8,8,1)
fn permute(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let input = textureLoad(InputBuffer, iid.xy, 0);
    textureStore(OutputBuffer, iid.xy, input * (1.0 - 2.0 * f32((iid.x + iid.y) % 2)));
}
)";

constexpr const char* kCopyTextureWGSL = R"(
@group(0) @binding(0) var dest: texture_storage_2d<rg32float, write>;
@group(0) @binding(1) var src: texture_2d<f32>;

struct Params { width: u32, height: u32, };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.width || global_id.y >= params.height) { return; }
    let pixel: vec4<f32> = textureLoad(src, vec2<i32>(global_id.xy), 0);
    textureStore(dest, vec2<i32>(global_id.xy), pixel);
}
)";


// ============================================================================
// WGSL Render Shaders (ported from WebTide GLSL → WGSL)
// ============================================================================

// Skybox: renders a large cube with cubemap sampling
constexpr const char* kSkyboxWGSL = R"(
struct Uniforms {
    viewProj: mat4x4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var cubeTex: texture_cube<f32>;
@group(0) @binding(2) var cubeSamp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) localPos: vec3<f32>,
};

@vertex fn vs(@location(0) position: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.localPos = position;
    // Translate cube to camera position so it always surrounds the camera
    let worldPos = position * 500.0 + u.cameraPos;
    out.pos = u.viewProj * vec4<f32>(worldPos, 1.0);
    return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let color = textureSample(cubeTex, cubeSamp, normalize(in.localPos));
    return vec4<f32>(color.rgb, 1.0);
}
)";

// Ground: textured plane
constexpr const char* kGroundWGSL = R"(
struct Uniforms {
    model: mat4x4<f32>,
    viewProj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var groundTex: texture_2d<f32>;
@group(0) @binding(2) var groundSamp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    let worldPos = u.model * vec4<f32>(position, 1.0);
    out.pos = u.viewProj * worldPos;
    out.uv = uv * 20.0; // tile the texture
    return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(groundTex, groundSamp, in.uv);
}
)";

// Water vertex + fragment shader
constexpr const char* kWaterWGSL = R"(
struct Uniforms {
    model: mat4x4<f32>,
    viewProj: mat4x4<f32>,
    cameraPos: vec3<f32>,
    tileSize: f32,
    lightDirection: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightMap: texture_2d<f32>;
@group(0) @binding(2) var gradientMap: texture_2d<f32>;
@group(0) @binding(3) var displacementMap: texture_2d<f32>;
@group(0) @binding(4) var oceanSamp: sampler;
@group(0) @binding(5) var cubeTex: texture_cube<f32>;
@group(0) @binding(6) var cubeSamp: sampler;
@group(0) @binding(7) var depthTex: texture_2d<f32>;
@group(0) @binding(8) var screenTex: texture_2d<f32>;
@group(0) @binding(9) var screenSamp: sampler;

struct VsOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normalW: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex fn vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
              @location(2) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    let scalingFactor = 1.0 / u.tileSize;

    var waterPos = position;

    let disp = textureSampleLevel(displacementMap, oceanSamp, uv, 0.0).rg * scalingFactor * 1.0;
    waterPos.x += disp.x;
    waterPos.z += disp.y;

    let height = textureSampleLevel(heightMap, oceanSamp, uv, 0.0).r;
    let gradient = textureSampleLevel(gradientMap, oceanSamp, uv, 0.0).rg;
    let hg = vec3<f32>(height, gradient) * scalingFactor * 0.5;
    waterPos.y += hg.x;
    let n = normalize(vec3<f32>(-hg.y, 1.0, -hg.z));

    out.worldPos = (u.model * vec4<f32>(waterPos, 1.0)).xyz;
    out.normalW = (u.model * vec4<f32>(n, 0.0)).xyz;
    out.clipPos = u.viewProj * vec4<f32>(out.worldPos, 1.0);
    out.uv = uv;
    return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let normal = normalize(in.normalW);

    let screenUV = in.clipPos.xy / vec2<f32>(textureDimensions(screenTex)) ;

    let backgroundColor = textureSample(screenTex, screenSamp, screenUV).rgb;

    let surfaceDepth = in.clipPos.z;
    let backgroundDepth = textureLoad(depthTex, vec2<i32>(in.clipPos.xy), 0).r;

    let distanceThroughWater = max(surfaceDepth - backgroundDepth, 0.0);

    let ndl = max(0.0, dot(normal, -u.lightDirection));
    var diffuseColor = vec3<f32>(0.01, 0.06, 0.1);
    diffuseColor = mix(diffuseColor, backgroundColor, exp(-distanceThroughWater * 0.1));

    let viewRayW = normalize(in.worldPos - u.cameraPos);
    let viewRayReflectedW = reflect(viewRayW, normal);

    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(-viewRayW, normal), 0.0), 5.0);

    let reflectedColor = textureSample(cubeTex, cubeSamp, viewRayReflectedW).rgb;

    let specular = pow(max(0.0, dot(reflect(-u.lightDirection, normal), viewRayW)), 720.0) * 210.0;

    let finalColor = mix(diffuseColor * ndl, reflectedColor + specular, fresnel);

    return vec4<f32>(finalColor, 1.0);
}
)";

// Post-process: full-screen triangle — composites bgRT + waterRT, then applies fog/exposure/saturation
constexpr const char* kPostProcessWGSL = R"(
@group(0) @binding(0) var bgTex: texture_2d<f32>;
@group(0) @binding(1) var waterTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct Uniforms {
    cameraInvView: mat4x4<f32>,
    cameraInvProj: mat4x4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VsOut {
    var out: VsOut;
    // Full-screen triangle
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn worldFromUV(pos: vec2<f32>, invProj: mat4x4<f32>, invView: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(pos * 2.0 - 1.0, 1.0, 1.0);
    let posVS = invProj * ndc;
    let posWS = invView * posVS;
    return posWS.xyz / posWS.w;
}

fn getFogFactor(d: f32, rayDir: vec3<f32>) -> f32 {
    let LOG2: f32 = 1.442695;
    let density: f32 = 400.0;
    var fogFactor = exp2(-density * density * d * d * LOG2);
    fogFactor = 1.0 - clamp((fogFactor - 0.3) / 0.7, 0.0, 1.0);
    fogFactor *= pow(1.0 - abs(rayDir.y), 4.0);
    return fogFactor;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let bgColor = textureSample(bgTex, samp, in.uv);
    let waterColor = textureSample(waterTex, samp, in.uv);
    // Composite: if waterColor.a > 0, use water; otherwise use background
    var color = mix(bgColor.rgb, waterColor.rgb, waterColor.a);
    let depth = textureLoad(depthTex, vec2<i32>(in.pos.xy), 0).r;
    let rayDir = normalize(worldFromUV(in.uv, u.cameraInvProj, u.cameraInvView) - u.cameraPos);
    let fogFactor = getFogFactor(depth, rayDir);
    let fogColor = vec3<f32>(0.8);

    let exposure: f32 = 1.0;
    let contrast: f32 = 1.0;
    let brightness: f32 = 0.0;
    let saturation: f32 = 1.3;

    color *= exposure;
    color = clamp(color, vec3(0.0), vec3(1.0));
    color = (color - 0.5) * contrast + 0.5 + brightness;
    color = clamp(color, vec3(0.0), vec3(1.0));
    let gray = vec3<f32>(dot(vec3<f32>(0.299, 0.587, 0.114), color));
    color = mix(gray, color, saturation);
    color = clamp(color, vec3(0.0), vec3(1.0));
    color = mix(color, fogColor, fogFactor);

    return vec4<f32>(color, 1.0);
}
)";


// ============================================================================
// GPU Helper Utilities
// ============================================================================

WGPUStringView sv(const char* s) { return {.data = s, .length = strlen(s)}; }

WGPUShaderModule createShaderModule(WGPUDevice device, const char* code, const char* label) {
    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code = sv(code);
    WGPUShaderModuleDescriptor desc{};
    desc.nextInChain = &wgslSrc.chain;
    desc.label = sv(label);
    return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUBuffer createBuffer(WGPUDevice device, uint64_t size, WGPUBufferUsage usage, const char* label) {
    WGPUBufferDescriptor bd{};
    bd.label = sv(label);
    bd.size = size;
    bd.usage = usage;
    bd.mappedAtCreation = false;
    return wgpuDeviceCreateBuffer(device, &bd);
}

WGPUTexture createTexture2D(WGPUDevice device, uint32_t w, uint32_t h, WGPUTextureFormat fmt,
                             WGPUTextureUsage usage, const char* label, uint32_t layers = 1) {
    WGPUTextureDescriptor td{};
    td.label = sv(label);
    td.size = {w, h, layers};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = fmt;
    td.usage = usage;
    return wgpuDeviceCreateTexture(device, &td);
}

WGPUTextureView createView2D(WGPUTexture tex, WGPUTextureFormat fmt) {
    WGPUTextureViewDescriptor vd{};
    vd.format = fmt;
    vd.dimension = WGPUTextureViewDimension_2D;
    vd.baseMipLevel = 0; vd.mipLevelCount = 1;
    vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
    vd.aspect = WGPUTextureAspect_All;
    return wgpuTextureCreateView(tex, &vd);
}

WGPUTextureView createViewCube(WGPUTexture tex, WGPUTextureFormat fmt) {
    WGPUTextureViewDescriptor vd{};
    vd.format = fmt;
    vd.dimension = WGPUTextureViewDimension_Cube;
    vd.baseMipLevel = 0; vd.mipLevelCount = 1;
    vd.baseArrayLayer = 0; vd.arrayLayerCount = 6;
    vd.aspect = WGPUTextureAspect_All;
    return wgpuTextureCreateView(tex, &vd);
}

WGPUSampler createSampler(WGPUDevice device, WGPUFilterMode filter = WGPUFilterMode_Linear,
                           WGPUAddressMode addr = WGPUAddressMode_Repeat) {
    WGPUSamplerDescriptor sd{};
    sd.label = sv("sampler");
    sd.magFilter = filter;
    sd.minFilter = filter;
    sd.mipmapFilter = (filter == WGPUFilterMode_Nearest) ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear;
    sd.addressModeU = addr;
    sd.addressModeV = addr;
    sd.addressModeW = addr;
    sd.maxAnisotropy = 1;
    sd.compare = WGPUCompareFunction_Undefined;
    return wgpuDeviceCreateSampler(device, &sd);
}

struct StorageTex {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUTextureFormat format;
    uint32_t w, h;

    void create(WGPUDevice device, uint32_t width, uint32_t height, WGPUTextureFormat fmt, const char* label) {
        w = width; h = height; format = fmt;
        texture = createTexture2D(device, w, h, fmt,
            WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            label);
        view = createView2D(texture, fmt);
    }

    void release() {
        if (view) wgpuTextureViewRelease(view);
        if (texture) wgpuTextureRelease(texture);
    }
};

// ============================================================================
// Arc-Rotate Camera
// ============================================================================

struct ArcCamera {
    float alpha = PI / 3.0f;
    float beta = PI / 2.0f + 0.02f;
    float radius = 15.0f;
    Vector3 target{0, 1.5f, 0};
    float fov = 60.0f;
    float nearPlane = 0.1f;
    float farPlane = 2000.0f;

    bool dragging = false;
    double lastX = 0, lastY = 0;

    Vector3 position() const {
        float x = target.x + radius * std::sin(beta) * std::cos(alpha);
        float y = target.y + radius * std::cos(beta);
        float z = target.z + radius * std::sin(beta) * std::sin(alpha);
        return {x, y, z};
    }

    Matrix4 viewMatrix() const {
        Matrix4 m;
        auto pos = position();
        m.lookAt(pos, target, {0, 1, 0});
        return m;
    }

    Matrix4 projMatrix(float aspect) const {
        Matrix4 m;
        float top = nearPlane * std::tan(fov * PI / 360.0f);
        float right = top * aspect;
        m.makePerspective(-right, right, top, -top, nearPlane, farPlane);
        return m;
    }

    void onMouseButton(int button, int action) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            dragging = (action == GLFW_PRESS);
        }
    }

    void onMouseMove(double x, double y) {
        if (dragging) {
            float dx = static_cast<float>(x - lastX);
            float dy = static_cast<float>(y - lastY);
            alpha += dx * 0.005f;
            beta = std::clamp(beta + dy * 0.005f, 0.05f, PI - 0.05f);
        }
        lastX = x; lastY = y;
    }

    void onScroll(double yoff) {
        radius = std::max(2.0f, radius - static_cast<float>(yoff) * 0.5f);
    }
};


// ============================================================================
// Ocean Simulation (Compute Pipelines)
// ============================================================================

struct OceanSim {
    WGPUDevice device;
    WGPUQueue queue;
    uint32_t texSize;
    float tileSize;

    // Textures
    StorageTex h0;           // RGBA32Float - initial spectrum
    StorageTex ht, dht, disp; // RG32Float - dynamic outputs
    StorageTex heightMap, gradientMap, displacementMap; // RG32Float - final IFFT outputs
    StorageTex twiddleTex;   // RGBA32Float - precomputed twiddle factors
    StorageTex pingPong;     // RG32Float - temp buffer for IFFT

    WGPUTexture noiseTex = nullptr;
    WGPUTextureView noiseView = nullptr;

    // Compute pipelines
    WGPUComputePipeline phillipsPipeline = nullptr;
    WGPUBindGroupLayout phillipsBGL = nullptr;
    WGPUPipelineLayout phillipsPL = nullptr;

    WGPUComputePipeline dynamicPipeline = nullptr;
    WGPUBindGroupLayout dynamicBGL = nullptr;
    WGPUPipelineLayout dynamicPL = nullptr;

    WGPUComputePipeline twiddlePipeline = nullptr;
    WGPUBindGroupLayout twiddleBGL = nullptr;
    WGPUPipelineLayout twiddlePL = nullptr;

    WGPUComputePipeline hStepPipeline = nullptr;
    WGPUComputePipeline vStepPipeline = nullptr;
    WGPUBindGroupLayout ifftBGL = nullptr;
    WGPUPipelineLayout ifftPL = nullptr;

    WGPUComputePipeline permutePipeline = nullptr;
    WGPUBindGroupLayout permuteBGL = nullptr;
    WGPUPipelineLayout permutePL = nullptr;

    WGPUComputePipeline copyPipeline = nullptr;
    WGPUBindGroupLayout copyBGL = nullptr;
    WGPUPipelineLayout copyPL = nullptr;

    // Uniform buffers
    WGPUBuffer phillipsUB = nullptr;
    WGPUBuffer dynamicUB = nullptr;
    WGPUBuffer ifftUB = nullptr;
    WGPUBuffer copyUB = nullptr;

    void init(WGPUDevice dev, WGPUQueue q, uint32_t size, float tile) {
        device = dev; queue = q; texSize = size; tileSize = tile;

        // Create storage textures
        h0.create(device, texSize, texSize, WGPUTextureFormat_RGBA32Float, "h0");
        ht.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "ht");
        dht.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "dht");
        disp.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "disp");
        heightMap.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "heightMap");
        gradientMap.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "gradientMap");
        displacementMap.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "displacementMap");

        uint32_t logSize = static_cast<uint32_t>(std::log2(texSize));
        twiddleTex.create(device, logSize, texSize, WGPUTextureFormat_RGBA32Float, "twiddle");
        pingPong.create(device, texSize, texSize, WGPUTextureFormat_RG32Float, "pingPong");

        createNoise();
        createPhillipsPipeline();
        createDynamicPipeline();
        createTwiddlePipeline();
        createIFFTPipelines();
        createPermutePipeline();
        createCopyPipeline();

        // Run one-time computations
        runPhillips();
        runTwiddle();
    }

    void createNoise() {
        // Gaussian noise as RG8Unorm - matching WebTide's truncation behavior
        std::vector<uint8_t> data(texSize * texSize * 2);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < data.size(); i++) {
            float u1 = dist(rng), u2 = dist(rng);
            float g = std::cos(2.0f * PI * u1) * std::sqrt(-2.0f * std::log(std::max(u2, 1e-10f)));
            data[i] = static_cast<uint8_t>(g);
        }

        noiseTex = createTexture2D(device, texSize, texSize, WGPUTextureFormat_RG8Unorm,
            WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "noise");
        noiseView = createView2D(noiseTex, WGPUTextureFormat_RG8Unorm);

        WGPUTexelCopyTextureInfo dst{};
        dst.texture = noiseTex;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = texSize * 2;
        layout.rowsPerImage = texSize;
        WGPUExtent3D extent = {texSize, texSize, 1};
        wgpuQueueWriteTexture(queue, &dst, data.data(), data.size(), &layout, &extent);
    }

    // Helper to build bind group layout
    WGPUBindGroupLayout makeBGL(const std::vector<WGPUBindGroupLayoutEntry>& entries, const char* label) {
        WGPUBindGroupLayoutDescriptor desc{};
        desc.label = sv(label);
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        return wgpuDeviceCreateBindGroupLayout(device, &desc);
    }

    WGPUPipelineLayout makePL(WGPUBindGroupLayout bgl, const char* label) {
        WGPUPipelineLayoutDescriptor desc{};
        desc.label = sv(label);
        desc.bindGroupLayoutCount = 1;
        desc.bindGroupLayouts = &bgl;
        return wgpuDeviceCreatePipelineLayout(device, &desc);
    }

    WGPUComputePipeline makeCP(WGPUPipelineLayout pl, WGPUShaderModule sm, const char* entry, const char* label) {
        WGPUComputePipelineDescriptor desc{};
        desc.label = sv(label);
        desc.layout = pl;
        desc.compute.module = sm;
        desc.compute.entryPoint = sv(entry);
        return wgpuDeviceCreateComputePipeline(device, &desc);
    }

    WGPUBindGroup makeBG(WGPUBindGroupLayout bgl, const std::vector<WGPUBindGroupEntry>& entries, const char* label) {
        WGPUBindGroupDescriptor desc{};
        desc.label = sv(label);
        desc.layout = bgl;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        return wgpuDeviceCreateBindGroup(device, &desc);
    }

    // BGL entry helpers
    static WGPUBindGroupLayoutEntry storageTexEntry(uint32_t binding, WGPUTextureFormat fmt,
                                                      WGPUStorageTextureAccess access = WGPUStorageTextureAccess_WriteOnly) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.storageTexture.access = access;
        e.storageTexture.format = fmt;
        e.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
        return e;
    }

    static WGPUBindGroupLayoutEntry sampledTexEntry(uint32_t binding) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        e.texture.viewDimension = WGPUTextureViewDimension_2D;
        return e;
    }

    static WGPUBindGroupLayoutEntry uniformEntry(uint32_t binding, uint64_t size) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = WGPUBufferBindingType_Uniform;
        e.buffer.minBindingSize = size;
        return e;
    }

    static WGPUBindGroupEntry texViewEntry(uint32_t binding, WGPUTextureView v) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.textureView = v;
        return e;
    }

    static WGPUBindGroupEntry bufEntry(uint32_t binding, WGPUBuffer b, uint64_t sz) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = b;
        e.offset = 0;
        e.size = sz;
        return e;
    }


    void createPhillipsPipeline() {
        // Phillips: binding 0=H0(storage,rgba32f,write), 1=Noise(texture), 2=params(uniform)
        // Params: textureSize(u32) + tileSize(f32) + windTheta(f32) + windSpeed(f32) + cutoff(f32) = 20 bytes, pad to 32
        phillipsBGL = makeBGL({
            storageTexEntry(0, WGPUTextureFormat_RGBA32Float),
            sampledTexEntry(1),
            uniformEntry(2, 32),
        }, "phillips_bgl");
        phillipsPL = makePL(phillipsBGL, "phillips_pl");
        auto sm = createShaderModule(device, kPhillipsSpectrumWGSL, "phillips_shader");
        phillipsPipeline = makeCP(phillipsPL, sm, "computeSpectrum", "phillips_pipeline");
        wgpuShaderModuleRelease(sm);

        phillipsUB = createBuffer(device, 32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "phillips_ub");
    }

    void createDynamicPipeline() {
        // Dynamic: 0=H0(texture), 1=HT(storage,rg32f), 2=DHT(storage,rg32f), 3=Disp(storage,rg32f), 4=params(uniform)
        // Params: textureSize(u32) + tileSize(f32) + elapsed(f32) = 12 bytes, pad to 16
        dynamicBGL = makeBGL({
            sampledTexEntry(0),
            storageTexEntry(1, WGPUTextureFormat_RG32Float),
            storageTexEntry(2, WGPUTextureFormat_RG32Float),
            storageTexEntry(3, WGPUTextureFormat_RG32Float),
            uniformEntry(4, 16),
        }, "dynamic_bgl");
        dynamicPL = makePL(dynamicBGL, "dynamic_pl");
        auto sm = createShaderModule(device, kDynamicSpectrumWGSL, "dynamic_shader");
        dynamicPipeline = makeCP(dynamicPL, sm, "computeSpectrum", "dynamic_pipeline");
        wgpuShaderModuleRelease(sm);

        dynamicUB = createBuffer(device, 16, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "dynamic_ub");
    }

    void createTwiddlePipeline() {
        // Twiddle: 0=PrecomputeBuffer(storage,rgba32f,write), 1=params(uniform)
        // Params: step(i32) + textureSize(i32) = 8 bytes
        twiddleBGL = makeBGL({
            storageTexEntry(0, WGPUTextureFormat_RGBA32Float),
            uniformEntry(1, 8),
        }, "twiddle_bgl");
        twiddlePL = makePL(twiddleBGL, "twiddle_pl");
        auto sm = createShaderModule(device, kTwiddleFactorsWGSL, "twiddle_shader");
        twiddlePipeline = makeCP(twiddlePL, sm, "precomputeTwiddleFactorsAndInputIndices", "twiddle_pipeline");
        wgpuShaderModuleRelease(sm);
    }

    void createIFFTPipelines() {
        // IFFT (horizontal/vertical): 0=params(uniform), 1=PrecomputedData(texture), 2=InputBuffer(texture), 3=OutputBuffer(storage,rg32f)
        ifftBGL = makeBGL({
            uniformEntry(0, 8),
            sampledTexEntry(1),
            sampledTexEntry(2),
            storageTexEntry(3, WGPUTextureFormat_RG32Float),
        }, "ifft_bgl");
        ifftPL = makePL(ifftBGL, "ifft_pl");

        auto smH = createShaderModule(device, kHorizontalStepWGSL, "hstep_shader");
        hStepPipeline = makeCP(ifftPL, smH, "horizontalStepInverseFFT", "hstep_pipeline");
        wgpuShaderModuleRelease(smH);

        auto smV = createShaderModule(device, kVerticalStepWGSL, "vstep_shader");
        vStepPipeline = makeCP(ifftPL, smV, "verticalStepInverseFFT", "vstep_pipeline");
        wgpuShaderModuleRelease(smV);

        ifftUB = createBuffer(device, 8, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "ifft_ub");
    }

    void createPermutePipeline() {
        // Permutation: 0=InputBuffer(texture), 1=OutputBuffer(storage,rg32f)
        permuteBGL = makeBGL({
            sampledTexEntry(0),
            storageTexEntry(1, WGPUTextureFormat_RG32Float),
        }, "permute_bgl");
        permutePL = makePL(permuteBGL, "permute_pl");
        auto sm = createShaderModule(device, kPermutationWGSL, "permute_shader");
        permutePipeline = makeCP(permutePL, sm, "permute", "permute_pipeline");
        wgpuShaderModuleRelease(sm);
    }

    void createCopyPipeline() {
        // Copy: 0=dest(storage,rg32f,write), 1=src(texture), 2=params(uniform)
        copyBGL = makeBGL({
            storageTexEntry(0, WGPUTextureFormat_RG32Float),
            sampledTexEntry(1),
            uniformEntry(2, 8),
        }, "copy_bgl");
        copyPL = makePL(copyBGL, "copy_pl");
        auto sm = createShaderModule(device, kCopyTextureWGSL, "copy_shader");
        copyPipeline = makeCP(copyPL, sm, "main", "copy_pipeline");
        wgpuShaderModuleRelease(sm);

        copyUB = createBuffer(device, 8, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "copy_ub");
        uint32_t copyParams[2] = {texSize, texSize};
        wgpuQueueWriteBuffer(queue, copyUB, 0, copyParams, 8);
    }

    void runPhillips() {
        // Write params: textureSize(u32), tileSize(f32), windTheta(f32), windSpeed(f32), cutoff(f32)
        struct { uint32_t texSize; float tileSize, windTheta, windSpeed, cutoff; float pad[3]; } params;
        params.texSize = texSize;
        params.tileSize = tileSize;
        params.windTheta = 0.0f;
        params.windSpeed = 31.0f;
        params.cutoff = 0.01f;
        params.pad[0] = params.pad[1] = params.pad[2] = 0;
        wgpuQueueWriteBuffer(queue, phillipsUB, 0, &params, 32);

        auto bg = makeBG(phillipsBGL, {
            texViewEntry(0, h0.view),
            texViewEntry(1, noiseView),
            bufEntry(2, phillipsUB, 32),
        }, "phillips_bg");

        WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("phillips_enc");
        auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        auto pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, phillipsPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        uint32_t groups = (texSize + WG - 1) / WG;
        wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("phillips_cmd");
        auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuBindGroupRelease(bg);
    }

    void runTwiddle() {
        uint32_t logSize = static_cast<uint32_t>(std::log2(texSize));
        auto twiddleUB = createBuffer(device, 8, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "twiddle_ub");
        int32_t params[2] = {0, static_cast<int32_t>(texSize)};
        wgpuQueueWriteBuffer(queue, twiddleUB, 0, params, 8);

        auto bg = makeBG(twiddleBGL, {
            texViewEntry(0, twiddleTex.view),
            bufEntry(1, twiddleUB, 8),
        }, "twiddle_bg");

        WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("twiddle_enc");
        auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        auto pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, twiddlePipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, logSize, texSize / 2 / WG, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("twiddle_cmd");
        auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuBindGroupRelease(bg);
        wgpuBufferRelease(twiddleUB);
    }

    void updateDynamic(float elapsedSeconds) {
        struct { uint32_t texSize; float tileSize, elapsed; float pad; } params;
        params.texSize = texSize;
        params.tileSize = tileSize;
        params.elapsed = elapsedSeconds;
        params.pad = 0;
        wgpuQueueWriteBuffer(queue, dynamicUB, 0, &params, 16);

        auto bg = makeBG(dynamicBGL, {
            texViewEntry(0, h0.view),
            texViewEntry(1, ht.view),
            texViewEntry(2, dht.view),
            texViewEntry(3, disp.view),
            bufEntry(4, dynamicUB, 16),
        }, "dynamic_bg");

        WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("dynamic_enc");
        auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        auto pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, dynamicPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        uint32_t groups = (texSize + WG - 1) / WG;
        wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("dynamic_cmd");
        auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuBindGroupRelease(bg);
    }

    void copyTex(StorageTex& dst, StorageTex& src, WGPUComputePassEncoder pass) {
        auto bg = makeBG(copyBGL, {
            texViewEntry(0, dst.view),
            texViewEntry(1, src.view),
            bufEntry(2, copyUB, 8),
        }, "copy_bg");
        wgpuComputePassEncoderSetPipeline(pass, copyPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        uint32_t groups = (texSize + WG - 1) / WG;
        wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
        wgpuBindGroupRelease(bg);
    }

    // Apply IFFT to input texture, write result to output texture
    void applyIFFT(StorageTex& input, StorageTex& output) {
        uint32_t logSize = static_cast<uint32_t>(std::log2(texSize));
        uint32_t groups = (texSize + WG - 1) / WG;

        WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("ifft_enc");
        auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        auto pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);

        bool usePingPong = false;
        // Horizontal passes
        for (uint32_t i = 0; i < logSize; ++i) {
            usePingPong = !usePingPong;
            int32_t params[2] = {static_cast<int32_t>(i), static_cast<int32_t>(texSize)};
            wgpuQueueWriteBuffer(queue, ifftUB, 0, params, 8);

            auto& inTex = usePingPong ? input : output;
            auto& outTex = usePingPong ? output : input;

            auto bg = makeBG(ifftBGL, {
                bufEntry(0, ifftUB, 8),
                texViewEntry(1, twiddleTex.view),
                texViewEntry(2, inTex.view),
                texViewEntry(3, outTex.view),
            }, "hstep_bg");
            wgpuComputePassEncoderSetPipeline(pass, hStepPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
            wgpuBindGroupRelease(bg);
        }

        // Vertical passes
        for (uint32_t i = 0; i < logSize; ++i) {
            usePingPong = !usePingPong;
            int32_t params[2] = {static_cast<int32_t>(i), static_cast<int32_t>(texSize)};
            wgpuQueueWriteBuffer(queue, ifftUB, 0, params, 8);

            auto& inTex = usePingPong ? input : output;
            auto& outTex = usePingPong ? output : input;

            auto bg = makeBG(ifftBGL, {
                bufEntry(0, ifftUB, 8),
                texViewEntry(1, twiddleTex.view),
                texViewEntry(2, inTex.view),
                texViewEntry(3, outTex.view),
            }, "vstep_bg");
            wgpuComputePassEncoderSetPipeline(pass, vStepPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
            wgpuBindGroupRelease(bg);
        }

        // If result is in output after ping-pong, copy to input
        if (usePingPong) {
            copyTex(input, output, pass);
        }

        // Permutation: input -> output
        auto permBG = makeBG(permuteBGL, {
            texViewEntry(0, input.view),
            texViewEntry(1, output.view),
        }, "permute_bg");
        wgpuComputePassEncoderSetPipeline(pass, permutePipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, permBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, groups, groups, 1);
        wgpuBindGroupRelease(permBG);

        // Copy result back to input for reuse
        copyTex(input, output, pass);

        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("ifft_cmd");
        auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    void update(float elapsedSeconds) {
        updateDynamic(elapsedSeconds);
        applyIFFT(ht, heightMap);
        applyIFFT(dht, gradientMap);
        applyIFFT(disp, displacementMap);
    }
};


// ============================================================================
// Geometry Generation
// ============================================================================

struct Vertex { float x, y, z, nx, ny, nz, u, v; };

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    WGPUBuffer vertexBuffer = nullptr;
    WGPUBuffer indexBuffer = nullptr;

    void upload(WGPUDevice device, WGPUQueue queue) {
        {
            WGPUBufferDescriptor bd{};
            bd.label = sv("vb");
            bd.size = vertices.size() * sizeof(Vertex);
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            vertexBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertices.data(), bd.size);
        }
        {
            WGPUBufferDescriptor bd{};
            bd.label = sv("ib");
            bd.size = indices.size() * sizeof(uint32_t);
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            indexBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, indexBuffer, 0, indices.data(), bd.size);
        }
    }
};

MeshData generatePlane(float width, float height, uint32_t subdivs) {
    MeshData m;
    uint32_t segs = subdivs;
    float hw = width / 2, hh = height / 2;

    for (uint32_t iy = 0; iy <= segs; ++iy) {
        for (uint32_t ix = 0; ix <= segs; ++ix) {
            float u = static_cast<float>(ix) / segs;
            float v = static_cast<float>(iy) / segs;
            Vertex vert;
            vert.x = u * width - hw;
            vert.y = 0;
            vert.z = v * height - hh;
            vert.nx = 0; vert.ny = 1; vert.nz = 0;
            vert.u = u; vert.v = v;
            m.vertices.push_back(vert);
        }
    }

    for (uint32_t iy = 0; iy < segs; ++iy) {
        for (uint32_t ix = 0; ix < segs; ++ix) {
            uint32_t a = iy * (segs + 1) + ix;
            uint32_t b = a + 1;
            uint32_t c = a + (segs + 1);
            uint32_t d = c + 1;
            m.indices.push_back(a); m.indices.push_back(c); m.indices.push_back(b);
            m.indices.push_back(b); m.indices.push_back(c); m.indices.push_back(d);
        }
    }
    return m;
}

// Simple box with outward normals (for skybox we flip winding)
MeshData generateBox() {
    MeshData m;
    // 8 corners of unit cube [-1,1]
    float v[] = {-1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1,  -1,-1,1, 1,-1,1, 1,1,1, -1,1,1};
    // 6 faces, each 2 triangles (reversed winding for inside viewing)
    uint32_t faces[] = {
        // front (looking from inside, reversed)
        0,1,2, 0,2,3,
        // back
        5,4,7, 5,7,6,
        // top
        3,2,6, 3,6,7,
        // bottom
        4,5,1, 4,1,0,
        // right
        1,5,6, 1,6,2,
        // left
        4,0,3, 4,3,7,
    };
    for (int i = 0; i < 36; ++i) {
        uint32_t idx = faces[i];
        Vertex vert;
        vert.x = v[idx*3]; vert.y = v[idx*3+1]; vert.z = v[idx*3+2];
        vert.nx = 0; vert.ny = 0; vert.nz = 0;
        vert.u = 0; vert.v = 0;
        m.vertices.push_back(vert);
        m.indices.push_back(i);
    }
    return m;
}

// Plane for ground (just pos + uv, no normal used in shader)
struct GroundVertex { float x, y, z, u, v; };

struct GroundMeshData {
    std::vector<GroundVertex> vertices;
    std::vector<uint32_t> indices;
    WGPUBuffer vertexBuffer = nullptr;
    WGPUBuffer indexBuffer = nullptr;

    void upload(WGPUDevice device, WGPUQueue queue) {
        {
            WGPUBufferDescriptor bd{};
            bd.label = sv("ground_vb");
            bd.size = vertices.size() * sizeof(GroundVertex);
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            vertexBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertices.data(), bd.size);
        }
        {
            WGPUBufferDescriptor bd{};
            bd.label = sv("ground_ib");
            bd.size = indices.size() * sizeof(uint32_t);
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            indexBuffer = wgpuDeviceCreateBuffer(device, &bd);
            wgpuQueueWriteBuffer(queue, indexBuffer, 0, indices.data(), bd.size);
        }
    }
};

GroundMeshData generateGroundPlane(float size) {
    GroundMeshData m;
    float h = size / 2;
    m.vertices = {{-h,0,-h,0,0}, {h,0,-h,1,0}, {h,0,h,1,1}, {-h,0,h,0,1}};
    m.indices = {0,2,1, 0,3,2};
    return m;
}


// ============================================================================
// Cubemap Loading
// ============================================================================

struct Cubemap {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUSampler sampler = nullptr;

    void load(WGPUDevice device, WGPUQueue queue, const std::string& basePath) {
        const char* faces[] = {
            "TropicalSunnyDay_px.jpg", "TropicalSunnyDay_nx.jpg",
            "TropicalSunnyDay_py.jpg", "TropicalSunnyDay_ny.jpg",
            "TropicalSunnyDay_pz.jpg", "TropicalSunnyDay_nz.jpg",
        };

        int w = 0, h = 0;
        std::vector<uint8_t*> faceData(6);
        for (int i = 0; i < 6; i++) {
            std::string path = basePath + "/" + faces[i];
            int channels;
            faceData[i] = stbi_load(path.c_str(), &w, &h, &channels, 4);
            if (!faceData[i]) {
                std::cerr << "Failed to load cubemap face: " << path << std::endl;
                return;
            }
        }

        texture = createTexture2D(device, w, h, WGPUTextureFormat_RGBA8Unorm,
            WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "cubemap", 6);

        for (int i = 0; i < 6; i++) {
            WGPUTexelCopyTextureInfo dst{};
            dst.texture = texture;
            dst.origin = {0, 0, static_cast<uint32_t>(i)};
            WGPUTexelCopyBufferLayout layout{};
            layout.bytesPerRow = static_cast<uint32_t>(w * 4);
            layout.rowsPerImage = static_cast<uint32_t>(h);
            WGPUExtent3D extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
            wgpuQueueWriteTexture(queue, &dst, faceData[i], w * h * 4, &layout, &extent);
            stbi_image_free(faceData[i]);
        }

        view = createViewCube(texture, WGPUTextureFormat_RGBA8Unorm);
        sampler = createSampler(device, WGPUFilterMode_Linear, WGPUAddressMode_ClampToEdge);
    }
};

// ============================================================================
// Texture Loading (2D)
// ============================================================================

struct Texture2D {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUSampler sampler = nullptr;

    void load(WGPUDevice device, WGPUQueue queue, const std::string& path) {
        int w, h, channels;
        auto* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
        if (!data) {
            std::cerr << "Failed to load texture: " << path << std::endl;
            return;
        }

        texture = createTexture2D(device, w, h, WGPUTextureFormat_RGBA8Unorm,
            WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "tex2d");
        view = createView2D(texture, WGPUTextureFormat_RGBA8Unorm);

        WGPUTexelCopyTextureInfo dst{};
        dst.texture = texture;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = static_cast<uint32_t>(w * 4);
        layout.rowsPerImage = static_cast<uint32_t>(h);
        WGPUExtent3D extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        wgpuQueueWriteTexture(queue, &dst, data, w * h * 4, &layout, &extent);
        stbi_image_free(data);

        sampler = createSampler(device);
    }
};


// ============================================================================
// Render Pipeline Builders
// ============================================================================

struct RenderTarget {
    WGPUTexture colorTex = nullptr;
    WGPUTextureView colorView = nullptr;
    WGPUTexture depthTex = nullptr;
    WGPUTextureView depthView = nullptr;
    uint32_t w, h;

    void create(WGPUDevice device, uint32_t width, uint32_t height) {
        w = width; h = height;
        colorTex = createTexture2D(device, w, h, WGPUTextureFormat_BGRA8Unorm,
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding, "rt_color");
        colorView = createView2D(colorTex, WGPUTextureFormat_BGRA8Unorm);
        // Use Depth32Float so we can sample it
        depthTex = createTexture2D(device, w, h, WGPUTextureFormat_Depth32Float,
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding, "rt_depth");
        depthView = createView2D(depthTex, WGPUTextureFormat_Depth32Float);
    }

    void release() {
        if (depthView) wgpuTextureViewRelease(depthView);
        if (depthTex) wgpuTextureRelease(depthTex);
        if (colorView) wgpuTextureViewRelease(colorView);
        if (colorTex) wgpuTextureRelease(colorTex);
    }
};

// Helper: standard depth stencil state
WGPUDepthStencilState depthStencil(WGPUTextureFormat fmt = WGPUTextureFormat_Depth32Float) {
    WGPUDepthStencilState ds{};
    ds.format = fmt;
    ds.depthWriteEnabled = WGPUOptionalBool_True;
    ds.depthCompare = WGPUCompareFunction_Less;
    return ds;
}

// Build skybox render pipeline
struct SkyboxPipeline {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;

    void create(WGPUDevice device) {
        auto sm = createShaderModule(device, kSkyboxWGSL, "skybox_shader");

        // BGL: uniform, cubemap texture, sampler
        WGPUBindGroupLayoutEntry entries[3]{};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = 80; // mat4 + vec3 + pad

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_Cube;

        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = sv("skybox_bgl");
        bglDesc.entryCount = 3;
        bglDesc.entries = entries;
        bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = sv("skybox_pl");
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex layout: pos (vec3)
        WGPUVertexAttribute posAttr{};
        posAttr.format = WGPUVertexFormat_Float32x3;
        posAttr.offset = 0;
        posAttr.shaderLocation = 0;

        WGPUVertexBufferLayout vbl{};
        vbl.arrayStride = sizeof(Vertex);
        vbl.stepMode = WGPUVertexStepMode_Vertex;
        vbl.attributeCount = 1;
        vbl.attributes = &posAttr;

        WGPUColorTargetState ct{};
        ct.format = WGPUTextureFormat_BGRA8Unorm;
        ct.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fs{};
        fs.module = sm;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &ct;

        auto ds = depthStencil();
        // Skybox: disable depth write, always pass (rendered first, we want everything behind it)
        ds.depthWriteEnabled = WGPUOptionalBool_False;
        ds.depthCompare = WGPUCompareFunction_Always;

        WGPURenderPipelineDescriptor rpd{};
        rpd.label = sv("skybox_rp");
        rpd.layout = pl;
        rpd.vertex.module = sm;
        rpd.vertex.entryPoint = sv("vs");
        rpd.vertex.bufferCount = 1;
        rpd.vertex.buffers = &vbl;
        rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        rpd.primitive.cullMode = WGPUCullMode_None; // see inside
        rpd.depthStencil = &ds;
        rpd.multisample.count = 1;
        rpd.multisample.mask = ~0u;
        rpd.fragment = &fs;

        pipeline = wgpuDeviceCreateRenderPipeline(device, &rpd);
        wgpuShaderModuleRelease(sm);
    }
};

// Build ground render pipeline
struct GroundPipeline {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;

    void create(WGPUDevice device) {
        auto sm = createShaderModule(device, kGroundWGSL, "ground_shader");

        // BGL: uniform, texture, sampler
        WGPUBindGroupLayoutEntry entries[3]{};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = 128; // model + viewProj

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = sv("ground_bgl");
        bglDesc.entryCount = 3;
        bglDesc.entries = entries;
        bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = sv("ground_pl");
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex layout: pos(vec3) + uv(vec2)
        WGPUVertexAttribute attrs[2]{};
        attrs[0].format = WGPUVertexFormat_Float32x3;
        attrs[0].offset = 0;
        attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x2;
        attrs[1].offset = 12;
        attrs[1].shaderLocation = 1;

        WGPUVertexBufferLayout vbl{};
        vbl.arrayStride = sizeof(GroundVertex);
        vbl.stepMode = WGPUVertexStepMode_Vertex;
        vbl.attributeCount = 2;
        vbl.attributes = attrs;

        WGPUColorTargetState ct{};
        ct.format = WGPUTextureFormat_BGRA8Unorm;
        ct.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fs{};
        fs.module = sm;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &ct;

        auto ds = depthStencil();

        WGPURenderPipelineDescriptor rpd{};
        rpd.label = sv("ground_rp");
        rpd.layout = pl;
        rpd.vertex.module = sm;
        rpd.vertex.entryPoint = sv("vs");
        rpd.vertex.bufferCount = 1;
        rpd.vertex.buffers = &vbl;
        rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        rpd.primitive.cullMode = WGPUCullMode_Back;
        rpd.depthStencil = &ds;
        rpd.multisample.count = 1;
        rpd.multisample.mask = ~0u;
        rpd.fragment = &fs;

        pipeline = wgpuDeviceCreateRenderPipeline(device, &rpd);
        wgpuShaderModuleRelease(sm);
    }
};


// Build water render pipeline
struct WaterPipeline {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;

    void create(WGPUDevice device) {
        auto sm = createShaderModule(device, kWaterWGSL, "water_shader");

        // BGL: 0=uniform, 1=heightMap, 2=gradientMap, 3=dispMap, 4=oceanSampler,
        //      5=cubeTex, 6=cubeSampler, 7=depthTex, 8=screenTex, 9=screenSampler
        WGPUBindGroupLayoutEntry entries[10]{};

        // 0: uniform buffer
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = 160; // 2*mat4 + vec3+pad + f32 + vec3+pad + f32 (WGSL alignment)

        // 1: heightMap (Rg32Float — unfilterable)
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Vertex;
        entries[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        // 2: gradientMap (Rg32Float — unfilterable)
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Vertex;
        entries[2].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

        // 3: displacementMap (Rg32Float — unfilterable)
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Vertex;
        entries[3].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

        // 4: ocean sampler (non-filtering for unfilterable textures)
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Vertex;
        entries[4].sampler.type = WGPUSamplerBindingType_NonFiltering;

        // 5: cubemap texture
        entries[5].binding = 5;
        entries[5].visibility = WGPUShaderStage_Fragment;
        entries[5].texture.sampleType = WGPUTextureSampleType_Float;
        entries[5].texture.viewDimension = WGPUTextureViewDimension_Cube;

        // 6: cubemap sampler
        entries[6].binding = 6;
        entries[6].visibility = WGPUShaderStage_Fragment;
        entries[6].sampler.type = WGPUSamplerBindingType_Filtering;

        // 7: depth texture (from bg pass)
        entries[7].binding = 7;
        entries[7].visibility = WGPUShaderStage_Fragment;
        entries[7].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[7].texture.viewDimension = WGPUTextureViewDimension_2D;

        // 8: screen color texture (from bg pass)
        entries[8].binding = 8;
        entries[8].visibility = WGPUShaderStage_Fragment;
        entries[8].texture.sampleType = WGPUTextureSampleType_Float;
        entries[8].texture.viewDimension = WGPUTextureViewDimension_2D;

        // 9: screen sampler
        entries[9].binding = 9;
        entries[9].visibility = WGPUShaderStage_Fragment;
        entries[9].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = sv("water_bgl");
        bglDesc.entryCount = 10;
        bglDesc.entries = entries;
        bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = sv("water_pl");
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex layout: pos(vec3) + normal(vec3) + uv(vec2)
        WGPUVertexAttribute attrs[3]{};
        attrs[0].format = WGPUVertexFormat_Float32x3;
        attrs[0].offset = 0;
        attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3;
        attrs[1].offset = 12;
        attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2;
        attrs[2].offset = 24;
        attrs[2].shaderLocation = 2;

        WGPUVertexBufferLayout vbl{};
        vbl.arrayStride = sizeof(Vertex);
        vbl.stepMode = WGPUVertexStepMode_Vertex;
        vbl.attributeCount = 3;
        vbl.attributes = attrs;

        WGPUColorTargetState ct{};
        ct.format = WGPUTextureFormat_BGRA8Unorm;
        ct.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fs{};
        fs.module = sm;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &ct;

        auto ds = depthStencil();

        WGPURenderPipelineDescriptor rpd{};
        rpd.label = sv("water_rp");
        rpd.layout = pl;
        rpd.vertex.module = sm;
        rpd.vertex.entryPoint = sv("vs");
        rpd.vertex.bufferCount = 1;
        rpd.vertex.buffers = &vbl;
        rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        rpd.primitive.cullMode = WGPUCullMode_Back;
        rpd.depthStencil = &ds;
        rpd.multisample.count = 1;
        rpd.multisample.mask = ~0u;
        rpd.fragment = &fs;

        pipeline = wgpuDeviceCreateRenderPipeline(device, &rpd);
        wgpuShaderModuleRelease(sm);
    }
};

// Build post-process pipeline (no vertex buffer, uses vertex_index)
struct PostProcessPipeline {
    WGPURenderPipeline pipeline = nullptr;
    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;

    void create(WGPUDevice device) {
        auto sm = createShaderModule(device, kPostProcessWGSL, "postprocess_shader");

        // BGL: 0=bgTex, 1=waterTex, 2=depthTex, 3=sampler, 4=uniform
        WGPUBindGroupLayoutEntry entries[5]{};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].texture.sampleType = WGPUTextureSampleType_Float;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Fragment;
        entries[3].sampler.type = WGPUSamplerBindingType_Filtering;

        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Fragment;
        entries[4].buffer.type = WGPUBufferBindingType_Uniform;
        entries[4].buffer.minBindingSize = 144; // 2*mat4 + vec3 + pad

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = sv("pp_bgl");
        bglDesc.entryCount = 5;
        bglDesc.entries = entries;
        bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = sv("pp_pl");
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        WGPUColorTargetState ct{};
        ct.format = WGPUTextureFormat_BGRA8Unorm;
        ct.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fs{};
        fs.module = sm;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &ct;

        WGPURenderPipelineDescriptor rpd{};
        rpd.label = sv("pp_rp");
        rpd.layout = pl;
        rpd.vertex.module = sm;
        rpd.vertex.entryPoint = sv("vs");
        rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        rpd.multisample.count = 1;
        rpd.multisample.mask = ~0u;
        rpd.fragment = &fs;
        // No depth for post-process

        pipeline = wgpuDeviceCreateRenderPipeline(device, &rpd);
        wgpuShaderModuleRelease(sm);
    }
};

}// namespace


// ============================================================================
// Main
// ============================================================================

int main() {
    // --- Window ---
    Canvas::Parameters params;
    params.title("Dawn Ocean Simulation")
          .size(1280, 720)
          .graphicsApi(GraphicsAPI::WebGPU);
    Canvas canvas(params);

    std::string dataDir = DATA_FOLDER;

    // --- WebGPU Init ---
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;
    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;

    // Create instance
    {
        WGPUInstanceDescriptor desc{};
        instance = wgpuCreateInstance(&desc);
    }

    // Create surface
    {
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        WGPUSurfaceDescriptor surfDesc{};
        surfDesc.label = sv("ocean_surface");

#if defined(__linux__)
        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.display = glfwGetX11Display();
        xlibSource.window = static_cast<uint64_t>(glfwGetX11Window(glfwWindow));
        surfDesc.nextInChain = &xlibSource.chain;
#elif defined(_WIN32)
        WGPUSurfaceSourceWindowsHWND hwndSource{};
        hwndSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        hwndSource.hinstance = GetModuleHandle(nullptr);
        hwndSource.hwnd = glfwGetWin32Window(glfwWindow);
        surfDesc.nextInChain = &hwndSource.chain;
#elif defined(__APPLE__)
        extern "C" void* dawn_create_metal_layer(void* nsWindow);
        void* metalLayer = dawn_create_metal_layer(glfwGetCocoaWindow(glfwWindow));
        WGPUSurfaceSourceMetalLayer metalSource{};
        metalSource.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
        metalSource.layer = metalLayer;
        surfDesc.nextInChain = &metalSource.chain;
#endif
        surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    // Request adapter
    {
        struct UD { WGPUAdapter a = nullptr; bool done = false; } ud;
        WGPURequestAdapterOptions opts{};
        opts.compatibleSurface = surface;
        WGPURequestAdapterCallbackInfo cb{};
        cb.mode = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = [](WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView, void* u1, void*) {
            auto* ud = static_cast<UD*>(u1);
            if (s == WGPURequestAdapterStatus_Success) ud->a = a;
            ud->done = true;
        };
        cb.userdata1 = &ud;
        wgpuInstanceRequestAdapter(instance, &opts, cb);
        while (!ud.done) wgpuInstanceProcessEvents(instance);
        adapter = ud.a;
        if (!adapter) { std::cerr << "Failed to get adapter\n"; return 1; }
    }

    // Request device
    {
        struct UD { WGPUDevice d = nullptr; bool done = false; } ud;
        WGPUDeviceDescriptor dd{};
        dd.label = sv("ocean_device");
        WGPURequestDeviceCallbackInfo cb{};
        cb.mode = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = [](WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView, void* u1, void*) {
            auto* ud = static_cast<UD*>(u1);
            if (s == WGPURequestDeviceStatus_Success) ud->d = d;
            ud->done = true;
        };
        cb.userdata1 = &ud;
        wgpuAdapterRequestDevice(adapter, &dd, cb);
        while (!ud.done) wgpuInstanceProcessEvents(instance);
        device = ud.d;
        if (!device) { std::cerr << "Failed to get device\n"; return 1; }
    }

    queue = wgpuDeviceGetQueue(device);

    // Configure surface
    {
        auto sz = canvas.size();
        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width = static_cast<uint32_t>(sz.width());
        config.height = static_cast<uint32_t>(sz.height());
        config.presentMode = WGPUPresentMode_Fifo;
        config.alphaMode = WGPUCompositeAlphaMode_Auto;
        wgpuSurfaceConfigure(surface, &config);
    }

    std::cout << "WebGPU initialized for ocean simulation" << std::endl;

    // --- Load Assets ---
    Cubemap cubemap;
    cubemap.load(device, queue, dataDir + "/textures/skybox");

    Texture2D sandTex;
    sandTex.load(device, queue, dataDir + "/textures/sand.jpg");

    // --- Create Geometry ---
    auto waterMesh = generatePlane(TILE_SIZE, TILE_SIZE, TEXTURE_SIZE);
    waterMesh.upload(device, queue);

    auto skyboxMesh = generateBox();
    skyboxMesh.upload(device, queue);

    auto groundMesh = generateGroundPlane(TILE_SIZE * GRID_RADIUS * 4);
    groundMesh.upload(device, queue);

    // --- Create Render Targets ---
    auto sz = canvas.size();
    uint32_t screenW = static_cast<uint32_t>(sz.width());
    uint32_t screenH = static_cast<uint32_t>(sz.height());

    RenderTarget bgRT; // background render target (skybox + ground)
    bgRT.create(device, screenW, screenH);

    RenderTarget waterRT; // water pass render target
    waterRT.create(device, screenW, screenH);

    // --- Create Pipelines ---
    SkyboxPipeline skyboxPipe;
    skyboxPipe.create(device);

    GroundPipeline groundPipe;
    groundPipe.create(device);

    WaterPipeline waterPipe;
    waterPipe.create(device);

    PostProcessPipeline ppPipe;
    ppPipe.create(device);

    // --- Create Uniform Buffers ---
    WGPUBuffer skyboxUB = createBuffer(device, 80, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "skybox_ub");
    WGPUBuffer groundUB = createBuffer(device, 128, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "ground_ub");
    WGPUBuffer waterUB = createBuffer(device, 160, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "water_ub");
    WGPUBuffer ppUB = createBuffer(device, 144, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "pp_ub");

    // --- Create Samplers ---
    WGPUSampler oceanSampler = createSampler(device, WGPUFilterMode_Nearest, WGPUAddressMode_Repeat);
    WGPUSampler screenSampler = createSampler(device, WGPUFilterMode_Linear, WGPUAddressMode_ClampToEdge);
    WGPUSampler nearestSampler = createSampler(device, WGPUFilterMode_Nearest, WGPUAddressMode_ClampToEdge);

    // --- Init Ocean Simulation ---
    OceanSim ocean;
    ocean.init(device, queue, TEXTURE_SIZE, TILE_SIZE);

    // --- Camera ---
    ArcCamera camera;
    auto* glfwWin = static_cast<GLFWwindow*>(canvas.windowPtr());
    glfwSetWindowUserPointer(glfwWin, &camera);
    glfwSetMouseButtonCallback(glfwWin, [](GLFWwindow* w, int btn, int act, int) {
        static_cast<ArcCamera*>(glfwGetWindowUserPointer(w))->onMouseButton(btn, act);
    });
    glfwSetCursorPosCallback(glfwWin, [](GLFWwindow* w, double x, double y) {
        static_cast<ArcCamera*>(glfwGetWindowUserPointer(w))->onMouseMove(x, y);
    });
    glfwSetScrollCallback(glfwWin, [](GLFWwindow* w, double, double y) {
        static_cast<ArcCamera*>(glfwGetWindowUserPointer(w))->onScroll(y);
    });

    // --- Time ---
    float elapsedSeconds = 60.0f; // skip initial artifacts
    auto lastTime = std::chrono::steady_clock::now();

    // Light direction (matching WebTide)
    Vector3 lightDir;
    lightDir.set(1, -1, 3).normalize();


    // --- Render Loop ---
    canvas.animate([&] {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        elapsedSeconds += dt;

        auto sz = canvas.size();
        screenW = static_cast<uint32_t>(sz.width());
        screenH = static_cast<uint32_t>(sz.height());
        float aspect = static_cast<float>(screenW) / static_cast<float>(screenH);

        // --- Compute: Update ocean ---
        ocean.update(elapsedSeconds);

        // --- Camera matrices ---
        Matrix4 viewMat = camera.viewMatrix();
        Matrix4 projMat = camera.projMatrix(aspect);
        Matrix4 vpMat;
        vpMat.multiplyMatrices(projMat, viewMat);
        Vector3 camPos = camera.position();

        // --- Get surface texture ---
        WGPUSurfaceTexture surfaceTex{};
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTex);
        if (surfaceTex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            surfaceTex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            return;
        }
        WGPUTextureView surfaceView = createView2D(surfaceTex.texture, surfaceFormat);

        // === PASS 1: Background (skybox + ground) → bgRT ===
        {
            // Upload skybox uniforms: viewProj(64) + cameraPos(12) + pad(4) = 80
            float skyData[20]{};
            std::memcpy(skyData, vpMat.elements.data(), 64);
            skyData[16] = camPos.x; skyData[17] = camPos.y; skyData[18] = camPos.z; skyData[19] = 0;
            wgpuQueueWriteBuffer(queue, skyboxUB, 0, skyData, 80);

            // Upload ground uniforms: model(64) + viewProj(64) = 128
            float groundData[32]{};
            Matrix4 groundModel;
            groundModel.makeTranslation(0, -2, 0);
            std::memcpy(groundData, groundModel.elements.data(), 64);
            std::memcpy(groundData + 16, vpMat.elements.data(), 64);
            wgpuQueueWriteBuffer(queue, groundUB, 0, groundData, 128);

            WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("bg_enc");
            auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);

            WGPURenderPassColorAttachment colorAtt{};
            colorAtt.view = bgRT.colorView;
            colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colorAtt.loadOp = WGPULoadOp_Clear;
            colorAtt.storeOp = WGPUStoreOp_Store;
            colorAtt.clearValue = {0.4, 0.6, 0.8, 1.0};

            WGPURenderPassDepthStencilAttachment depthAtt{};
            depthAtt.view = bgRT.depthView;
            depthAtt.depthLoadOp = WGPULoadOp_Clear;
            depthAtt.depthStoreOp = WGPUStoreOp_Store;
            depthAtt.depthClearValue = 1.0f;

            WGPURenderPassDescriptor passDesc{};
            passDesc.label = sv("bg_pass");
            passDesc.colorAttachmentCount = 1;
            passDesc.colorAttachments = &colorAtt;
            passDesc.depthStencilAttachment = &depthAtt;

            auto pass = wgpuCommandEncoderBeginRenderPass(enc, &passDesc);
            wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)screenW, (float)screenH, 0, 1);

            // Draw skybox
            {
                WGPUBindGroupEntry entries[3]{};
                entries[0] = {.binding = 0, .buffer = skyboxUB, .offset = 0, .size = 80};
                entries[1] = {.binding = 1, .textureView = cubemap.view};
                entries[2] = {.binding = 2, .sampler = cubemap.sampler};
                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.label = sv("skybox_bg");
                bgDesc.layout = skyboxPipe.bgl;
                bgDesc.entryCount = 3;
                bgDesc.entries = entries;
                auto bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                wgpuRenderPassEncoderSetPipeline(pass, skyboxPipe.pipeline);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, skyboxMesh.vertexBuffer, 0,
                    skyboxMesh.vertices.size() * sizeof(Vertex));
                wgpuRenderPassEncoderSetIndexBuffer(pass, skyboxMesh.indexBuffer,
                    WGPUIndexFormat_Uint32, 0, skyboxMesh.indices.size() * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, skyboxMesh.indices.size(), 1, 0, 0, 0);
                wgpuBindGroupRelease(bg);
            }

            // Draw ground
            {
                WGPUBindGroupEntry entries[3]{};
                entries[0] = {.binding = 0, .buffer = groundUB, .offset = 0, .size = 128};
                entries[1] = {.binding = 1, .textureView = sandTex.view};
                entries[2] = {.binding = 2, .sampler = sandTex.sampler};
                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.label = sv("ground_bg");
                bgDesc.layout = groundPipe.bgl;
                bgDesc.entryCount = 3;
                bgDesc.entries = entries;
                auto bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                wgpuRenderPassEncoderSetPipeline(pass, groundPipe.pipeline);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, groundMesh.vertexBuffer, 0,
                    groundMesh.vertices.size() * sizeof(GroundVertex));
                wgpuRenderPassEncoderSetIndexBuffer(pass, groundMesh.indexBuffer,
                    WGPUIndexFormat_Uint32, 0, groundMesh.indices.size() * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, groundMesh.indices.size(), 1, 0, 0, 0);
                wgpuBindGroupRelease(bg);
            }

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);

            WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("bg_cmd");
            auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }

        // === PASS 2: Water → render directly into bgRT (Load existing bg content) ===
        // Water fragments read a *copy* of bgRT depth/color via separate textures (bgRT_copy).
        // Since we can't read+write the same texture, we first copy bgRT to bgRT_copy,
        // then render water into bgRT with Load.
        // For simplicity in this first version, water reads bgRT directly (this works because
        // the water shader only reads screenUV-based samples from the PREVIOUS pass output,
        // and we render water into a separate render target that we composite later).
        // Actually: render water directly into bgRT using Load.
        // The water shader reads bgRT depth/color, but that's the same attachment we're writing to.
        // This would be a feedback loop. Solution: use waterRT as a separate target for water,
        // then composite waterRT onto bgRT in the post-process.
        // OR: since the water fragment shader samples bgRT (background), and we're rendering
        // water to a DIFFERENT target (waterRT), there's no feedback. This is the correct approach.
        // The post-process then composites waterRT over bgRT.
        {
            float waterData[40]{};
            std::memset(waterData, 0, sizeof(waterData));

            WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("water_enc");
            auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);

            WGPURenderPassColorAttachment colorAtt{};
            colorAtt.view = waterRT.colorView;
            colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colorAtt.loadOp = WGPULoadOp_Clear;
            colorAtt.storeOp = WGPUStoreOp_Store;
            colorAtt.clearValue = {0, 0, 0, 0};

            WGPURenderPassDepthStencilAttachment depthAtt{};
            depthAtt.view = waterRT.depthView;
            depthAtt.depthLoadOp = WGPULoadOp_Clear;
            depthAtt.depthStoreOp = WGPUStoreOp_Store;
            depthAtt.depthClearValue = 1.0f;

            WGPURenderPassDescriptor passDesc{};
            passDesc.label = sv("water_pass");
            passDesc.colorAttachmentCount = 1;
            passDesc.colorAttachments = &colorAtt;
            passDesc.depthStencilAttachment = &depthAtt;

            auto pass = wgpuCommandEncoderBeginRenderPass(enc, &passDesc);
            wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)screenW, (float)screenH, 0, 1);

            wgpuRenderPassEncoderSetPipeline(pass, waterPipe.pipeline);

            for (int tx = -GRID_RADIUS; tx <= GRID_RADIUS; ++tx) {
                for (int tz = -GRID_RADIUS; tz <= GRID_RADIUS; ++tz) {
                    Matrix4 model;
                    model.makeTranslation(tx * TILE_SIZE, 0, tz * TILE_SIZE);

                    std::memcpy(waterData, model.elements.data(), 64);
                    std::memcpy(waterData + 16, vpMat.elements.data(), 64);
                    waterData[32] = camPos.x; waterData[33] = camPos.y; waterData[34] = camPos.z;
                    waterData[35] = TILE_SIZE;
                    waterData[36] = lightDir.x; waterData[37] = lightDir.y; waterData[38] = lightDir.z;
                    waterData[39] = 0;
                    wgpuQueueWriteBuffer(queue, waterUB, 0, waterData, 160);

                    // Create bind group for this tile
                    WGPUBindGroupEntry entries[10]{};
                    entries[0] = {.binding = 0, .buffer = waterUB, .offset = 0, .size = 160};
                    entries[1] = {.binding = 1, .textureView = ocean.heightMap.view};
                    entries[2] = {.binding = 2, .textureView = ocean.gradientMap.view};
                    entries[3] = {.binding = 3, .textureView = ocean.displacementMap.view};
                    entries[4] = {.binding = 4, .sampler = oceanSampler};
                    entries[5] = {.binding = 5, .textureView = cubemap.view};
                    entries[6] = {.binding = 6, .sampler = cubemap.sampler};
                    entries[7] = {.binding = 7, .textureView = bgRT.depthView};
                    entries[8] = {.binding = 8, .textureView = bgRT.colorView};
                    entries[9] = {.binding = 9, .sampler = screenSampler};
                    WGPUBindGroupDescriptor bgDesc{};
                    bgDesc.label = sv("water_bg");
                    bgDesc.layout = waterPipe.bgl;
                    bgDesc.entryCount = 10;
                    bgDesc.entries = entries;
                    auto bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, waterMesh.vertexBuffer, 0,
                        waterMesh.vertices.size() * sizeof(Vertex));
                    wgpuRenderPassEncoderSetIndexBuffer(pass, waterMesh.indexBuffer,
                        WGPUIndexFormat_Uint32, 0, waterMesh.indices.size() * sizeof(uint32_t));
                    wgpuRenderPassEncoderDrawIndexed(pass, waterMesh.indices.size(), 1, 0, 0, 0);
                    wgpuBindGroupRelease(bg);
                }
            }

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);

            WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("water_cmd");
            auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }


        // === PASS 3: Composite water onto background, then post-process → surface ===
        // For simplicity, combine water+background first by rendering bgRT, then water on top,
        // then post-process to surface.

        // Actually, let's simplify: render everything to waterRT first (which has water),
        // but we need background behind it. Let's re-approach:
        // We'll do a composite pass: render bgRT as full-screen quad, then water on top,
        // to a final color+depth target. Then post-process that to surface.

        // Simpler approach: just copy bgRT to waterRT as background, then render water on top.
        // Even simpler: render skybox+ground+water all in one pass to waterRT.
        // The water shader reads bgRT for depth/refraction though, which is the previous pass output.

        // Current approach is correct: bgRT has background, water pass reads bgRT.
        // But waterRT only has water tiles rendered (not the background).
        // We need to composite waterRT on top of bgRT for the final image.

        // For the post-process, we'll combine: start with bgRT.color, overlay waterRT on top
        // by simply rendering water tiles again... no, that's wasteful.

        // Better: render the final scene pass (skybox+ground+water) all together to one render target.
        // But water needs to read the depth/color of the non-water objects...

        // Solution: Do the background pass to bgRT, then do a COMBINED pass:
        // 1. Start with bgRT.color loaded into the final RT
        // 2. Render water tiles on top (they use bgRT for refraction sampling)
        // 3. Post-process the final result to surface

        // Let me implement this properly:

        // === PASS 3: Final scene → composite bgRT + water → surface with post-process ===
        {
            // Upload post-process uniforms
            Matrix4 invView;
            invView.copy(viewMat).invert();
            Matrix4 invProj;
            invProj.copy(projMat).invert();

            float ppData[36]{};
            std::memcpy(ppData, invView.elements.data(), 64);
            std::memcpy(ppData + 16, invProj.elements.data(), 64);
            ppData[32] = camPos.x; ppData[33] = camPos.y; ppData[34] = camPos.z; ppData[35] = 0;
            wgpuQueueWriteBuffer(queue, ppUB, 0, ppData, 144);

            WGPUCommandEncoderDescriptor encDesc{}; encDesc.label = sv("final_enc");
            auto enc = wgpuDeviceCreateCommandEncoder(device, &encDesc);

            // Render to surface
            WGPURenderPassColorAttachment colorAtt{};
            colorAtt.view = surfaceView;
            colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colorAtt.loadOp = WGPULoadOp_Clear;
            colorAtt.storeOp = WGPUStoreOp_Store;
            colorAtt.clearValue = {0, 0, 0, 1};

            WGPURenderPassDescriptor passDesc{};
            passDesc.label = sv("final_pass");
            passDesc.colorAttachmentCount = 1;
            passDesc.colorAttachments = &colorAtt;
            // No depth for post-process

            auto pass = wgpuCommandEncoderBeginRenderPass(enc, &passDesc);
            wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)screenW, (float)screenH, 0, 1);

            // Post-process: reads waterRT (which has water) - but waterRT doesn't have background!
            // We need to combine. For now, use waterRT.color as the source.
            // The water fragments that weren't rendered are clear(0,0,0,0).
            // To properly composite: we should render water directly to bgRT with Load instead of Clear.
            // That way bgRT.color has everything.

            // Actually, let me fix the water pass: render water directly into bgRT with LoadOp_Load.
            // This means water tiles render on top of the existing background. Then post-process bgRT.

            // We'll need to redo pass 2 approach. For now, just use bgRT as post-process input
            // (water won't be visible yet, but the structure is correct).

            // Use waterRT for now (it has the water rendered).
            // Better: we rendered water to waterRT, background to bgRT.
            // For the final composite, the post-process shader reads waterRT.color.
            // But waterRT doesn't have the background behind transparent water areas.

            // Composite bgRT (background) + waterRT (water) via alpha blending in shader
            WGPUBindGroupEntry entries[5]{};
            entries[0] = {.binding = 0, .textureView = bgRT.colorView};
            entries[1] = {.binding = 1, .textureView = waterRT.colorView};
            entries[2] = {.binding = 2, .textureView = bgRT.depthView};
            entries[3] = {.binding = 3, .sampler = screenSampler};
            entries[4] = {.binding = 4, .buffer = ppUB, .offset = 0, .size = 144};
            WGPUBindGroupDescriptor bgDesc{};
            bgDesc.label = sv("pp_bg");
            bgDesc.layout = ppPipe.bgl;
            bgDesc.entryCount = 5;
            bgDesc.entries = entries;
            auto bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

            wgpuRenderPassEncoderSetPipeline(pass, ppPipe.pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            wgpuBindGroupRelease(bg);

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);

            WGPUCommandBufferDescriptor cmdDesc{}; cmdDesc.label = sv("final_cmd");
            auto cmd = wgpuCommandEncoderFinish(enc, &cmdDesc);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }

        wgpuSurfacePresent(surface);
        wgpuTextureViewRelease(surfaceView);
        wgpuTextureRelease(surfaceTex.texture);
    });

    // Cleanup
    std::cout << "Dawn Ocean: shutting down" << std::endl;

    return 0;
}
