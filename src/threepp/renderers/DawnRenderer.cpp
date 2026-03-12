
#include "threepp/renderers/DawnRenderer.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/renderers/common/Lights.hpp"
#include "threepp/renderers/common/RenderStates.hpp"

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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;

#ifdef __APPLE__
extern "C" void* dawn_create_metal_layer(void* nsWindow);
#endif

namespace {

    // Feature bitmask for pipeline caching
    enum PipelineFeatures : uint32_t {
        FEAT_NONE       = 0,
        FEAT_TEXTURE    = 1 << 0,
        FEAT_LIGHTING   = 1 << 1,
        FEAT_SPECULAR   = 1 << 2,
        FEAT_PBR        = 1 << 3,
    };

    constexpr int MAX_DIR_LIGHTS   = 4;
    constexpr int MAX_POINT_LIGHTS = 4;
    constexpr int MAX_SPOT_LIGHTS  = 4;
    constexpr int MAX_HEMI_LIGHTS  = 2;

    // Transform: model(64) + view(64) + proj(64) + normalMatrix(48 = 3*vec4 padded) + cameraPos(12) + pad(4) = 256
    constexpr size_t TRANSFORM_UNIFORM_SIZE = 256;

    // Material: diffuse(16) + specularAndShininess(16) + roughnessMetalnessOpacity(16) + emissive(16) + flags(16) = 80, pad to 96
    constexpr size_t MATERIAL_UNIFORM_SIZE = 96;

    // Light: header(32) + dir(4*32=128) + point(4*48=192) + spot(4*64=256) + hemi(2*48=96) = 704
    constexpr size_t LIGHT_UNIFORM_SIZE = 704;

    // Vertex stride: pos(12) + normal(12) + uv(8) = 32
    constexpr size_t VERTEX_STRIDE = 32;

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
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let worldPos4 = transform.model * vec4<f32>(in.position, 1.0);
    out.worldPos = worldPos4.xyz;
    let nm = mat3x3<f32>(transform.normalCol0.xyz, transform.normalCol1.xyz, transform.normalCol2.xyz);
    out.worldNormal = normalize(nm * in.normal);
    out.uv = in.uv;
    out.clipPos = transform.proj * transform.view * worldPos4;
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
            s << R"(
    let N = normalize(in.worldNormal);
    let V = normalize(transform.cameraPos - in.worldPos);
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
                s << "        { let H = normalize(L + V); let NdotH = max(dot(N, H), 0.0);\n";
                s << "          let r = material.roughnessMetalnessOpacity.x; let m = material.roughnessMetalnessOpacity.y;\n";
                s << "          let a2 = r*r*r*r; let sp = pow(NdotH, max(2.0/max(a2,0.001) - 2.0, 1.0));\n";
                s << "          let F0 = mix(vec3<f32>(0.04), baseColor, m);\n";
                s << "          specularLight += lights.directional[i].color * F0 * sp; }\n";
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
            s << "    }\n";

            s << R"(
    for (var i = 0u; i < lights.numSpot; i++) {
        let lv = lights.spot[i].position - in.worldPos;
        let d = length(lv); let L = normalize(lv);
        let NdotL = max(dot(N, L), 0.0);
        let ac = dot(L, normalize(-lights.spot[i].direction));
        let se = smoothstep(lights.spot[i].penumbraCos, lights.spot[i].coneCos, ac);
        var att = se;
        if (lights.spot[i].distance > 0.0) {
            let r2 = clamp(1.0 - pow(d / lights.spot[i].distance, 4.0), 0.0, 1.0);
            att = att * r2 * r2 / (d * d + 0.0001);
        }
        diffuseLight += lights.spot[i].color * NdotL * att;
    }
    for (var i = 0u; i < lights.numHemi; i++) {
        let w = 0.5 * dot(N, lights.hemi[i].direction) + 0.5;
        diffuseLight += mix(lights.hemi[i].groundColor, lights.hemi[i].skyColor, w);
    }
)";
            if (features & FEAT_PBR) {
                s << "    let metalness = material.roughnessMetalnessOpacity.y;\n";
                s << "    baseColor = baseColor * (1.0 - metalness) * diffuseLight + specularLight + material.emissive.rgb;\n";
            } else {
                s << "    baseColor = baseColor * diffuseLight + specularLight + material.emissive.rgb;\n";
            }
        }

        s << "    return vec4<f32>(baseColor, opacity);\n}\n";
        return s.str();
    }

}// namespace


struct DawnRenderer::Impl {

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

    // Pipeline cache keyed by feature bitmask (Feature 3)
    struct PipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
    };
    std::unordered_map<uint32_t, PipelineEntry> pipelineCache;

    // Geometry buffers cache (interleaved pos+normal+uv)
    struct GeometryBuffers {
        WGPUBuffer vertexBuffer = nullptr;
        WGPUBuffer indexBuffer = nullptr;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
    };
    std::unordered_map<unsigned int, GeometryBuffers> geometryCache;

    // Texture cache (Feature 2)
    struct TextureEntry {
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
        WGPUSampler sampler = nullptr;
        unsigned int version = 0;
    };
    std::unordered_map<unsigned int, TextureEntry> textureCache;
    TextureEntry dummyTexture; // 1x1 white

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
    RenderStates renderStates;

    bool initialized = false;

    explicit Impl(Canvas& canvas)
        : canvas(canvas), size_(canvas.size()) {

        viewport_.w = static_cast<float>(size_.width());
        viewport_.h = static_cast<float>(size_.height());
        scissor_.w = static_cast<uint32_t>(size_.width());
        scissor_.h = static_cast<uint32_t>(size_.height());

        initWebGPU();
    }

    void initWebGPU() {
        // Create instance
        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = nullptr;
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            std::cerr << "DawnRenderer: Failed to create WebGPU instance" << std::endl;
            return;
        }

        // Create surface from GLFW window
        createSurface();
        if (!surface) {
            std::cerr << "DawnRenderer: Failed to create surface" << std::endl;
            return;
        }

        // Request adapter (synchronous via callback)
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

        // Configure surface
        configureSurface();

        // Create uniform buffers
        createUniformBuffers();

        // Create dummy texture for untextured materials
        createDummyTexture();

        initialized = true;
        std::cout << "DawnRenderer: WebGPU initialized successfully" << std::endl;
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
        options.compatibleSurface = surface;

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

        // Poll until callback fires
        while (!userData.done) {
            wgpuInstanceProcessEvents(instance);
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

        while (!userData.done) {
            wgpuInstanceProcessEvents(instance);
        }

        device = userData.device;
    }

    void configureSurface() {
        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width = static_cast<uint32_t>(size_.width());
        config.height = static_cast<uint32_t>(size_.height());
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
        vbLayout.arrayStride = VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 3;
        vbLayout.attributes = attrs;

        // Blend state
        WGPUBlendState blendState{};
        blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
        blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blendState.color.operation = WGPUBlendOperation_Add;
        blendState.alpha.srcFactor = WGPUBlendFactor_One;
        blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blendState.alpha.operation = WGPUBlendOperation_Add;

        WGPUColorTargetState colorTarget{};
        colorTarget.format = surfaceFormat;
        colorTarget.blend = &blendState;
        colorTarget.writeMask = WGPUColorWriteMask_All;

        WGPUStringView fsEntry = {.data = "fs_main", .length = 7};
        WGPUFragmentState fragmentState{};
        fragmentState.module = entry.shader;
        fragmentState.entryPoint = fsEntry;
        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth24Plus;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
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

        pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
        pipelineDesc.primitive.cullMode = WGPUCullMode_None;
        pipelineDesc.depthStencil = &depthStencil;
        pipelineDesc.multisample.count = 1;
        pipelineDesc.multisample.mask = 0xFFFFFFFF;
        pipelineDesc.fragment = &fragmentState;

        entry.pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);

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

    void createDummyTexture() {
        WGPUTextureDescriptor td{};
        td.label = {.data = "dummy_tex", .length = 9};
        td.size = {1, 1, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_RGBA8Unorm;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        dummyTexture.texture = wgpuDeviceCreateTexture(device, &td);
        dummyTexture.view = wgpuTextureCreateView(dummyTexture.texture, nullptr);

        // Upload 1x1 white pixel
        unsigned char white[] = {255, 255, 255, 255};
        WGPUTexelCopyTextureInfo dst{};
        dst.texture = dummyTexture.texture;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = 4;
        layout.rowsPerImage = 1;
        WGPUExtent3D extent = {1, 1, 1};
        wgpuQueueWriteTexture(queue, &dst, white, 4, &layout, &extent);

        // Default sampler
        WGPUSamplerDescriptor sd{};
        sd.label = {.data = "dummy_sampler", .length = 13};
        sd.magFilter = WGPUFilterMode_Linear;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.addressModeU = WGPUAddressMode_Repeat;
        sd.addressModeV = WGPUAddressMode_Repeat;
        sd.addressModeW = WGPUAddressMode_Repeat;
        dummyTexture.sampler = wgpuDeviceCreateSampler(device, &sd);
    }

    // Texture upload/cache (Feature 2)
    TextureEntry& getOrCreateTexture(Texture* tex) {
        auto it = textureCache.find(tex->id);
        if (it != textureCache.end() && it->second.version == tex->version()) {
            return it->second;
        }

        // Release old if exists
        if (it != textureCache.end()) {
            if (it->second.view) wgpuTextureViewRelease(it->second.view);
            if (it->second.texture) wgpuTextureRelease(it->second.texture);
            if (it->second.sampler) wgpuSamplerRelease(it->second.sampler);
        }

        TextureEntry entry{};
        auto& img = tex->image();
        auto w = img.width;
        auto h = img.height;
        if (w == 0 || h == 0) return dummyTexture;

        WGPUTextureDescriptor td{};
        td.label = {.data = "user_tex", .length = 8};
        td.size = {w, h, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_RGBA8Unorm;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        entry.texture = wgpuDeviceCreateTexture(device, &td);
        entry.view = wgpuTextureCreateView(entry.texture, nullptr);

        auto& data = img.data<unsigned char>();
        WGPUTexelCopyTextureInfo dst{};
        dst.texture = entry.texture;
        WGPUTexelCopyBufferLayout layout{};
        layout.bytesPerRow = w * 4;
        layout.rowsPerImage = h;
        WGPUExtent3D extent = {w, h, 1};
        wgpuQueueWriteTexture(queue, &dst, data.data(), data.size(), &layout, &extent);

        // Sampler
        WGPUSamplerDescriptor sd{};
        sd.label = {.data = "tex_sampler", .length = 11};
        sd.magFilter = (tex->magFilter == Filter::Nearest) ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
        sd.minFilter = (tex->minFilter == Filter::Nearest || tex->minFilter == Filter::NearestMipmapNearest || tex->minFilter == Filter::NearestMipmapLinear)
            ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
        auto mapWrap = [](TextureWrapping w) {
            switch (w) {
                case TextureWrapping::Repeat: return WGPUAddressMode_Repeat;
                case TextureWrapping::MirroredRepeat: return WGPUAddressMode_MirrorRepeat;
                default: return WGPUAddressMode_ClampToEdge;
            }
        };
        sd.addressModeU = mapWrap(tex->wrapS);
        sd.addressModeV = mapWrap(tex->wrapT);
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        entry.sampler = wgpuDeviceCreateSampler(device, &sd);

        entry.version = tex->version();
        textureCache[tex->id] = entry;
        return textureCache[tex->id];
    }

    GeometryBuffers& getOrCreateGeometryBuffers(BufferGeometry* geometry) {
        auto id = geometry->id;
        auto it = geometryCache.find(id);
        if (it != geometryCache.end()) {
            return it->second;
        }

        GeometryBuffers gb{};

        if (geometry->hasAttribute("position")) {
            auto posAttr = geometry->getAttribute<float>("position");
            uint32_t count = static_cast<uint32_t>(posAttr->count());
            gb.vertexCount = count;

            // Get optional normal and uv attributes
            const float* normalData = nullptr;
            const float* uvData = nullptr;
            int normalItemSize = 3, uvItemSize = 2;

            if (geometry->hasAttribute("normal")) {
                auto nAttr = geometry->getAttribute<float>("normal");
                normalData = nAttr->array().data();
                normalItemSize = static_cast<int>(nAttr->itemSize());
            }
            if (geometry->hasAttribute("uv")) {
                auto uvAttr = geometry->getAttribute<float>("uv");
                uvData = uvAttr->array().data();
                uvItemSize = static_cast<int>(uvAttr->itemSize());
            }

            // Build interleaved buffer: pos(3) + normal(3) + uv(2) = 8 floats per vertex
            std::vector<float> interleaved(count * 8);
            auto& posArr = posAttr->array();
            int posItemSize = static_cast<int>(posAttr->itemSize());

            for (uint32_t i = 0; i < count; i++) {
                size_t base = i * 8;
                interleaved[base + 0] = posArr[i * posItemSize + 0];
                interleaved[base + 1] = posArr[i * posItemSize + 1];
                interleaved[base + 2] = (posItemSize > 2) ? posArr[i * posItemSize + 2] : 0.f;

                if (normalData) {
                    interleaved[base + 3] = normalData[i * normalItemSize + 0];
                    interleaved[base + 4] = normalData[i * normalItemSize + 1];
                    interleaved[base + 5] = (normalItemSize > 2) ? normalData[i * normalItemSize + 2] : 0.f;
                } else {
                    interleaved[base + 3] = 0.f;
                    interleaved[base + 4] = 0.f;
                    interleaved[base + 5] = 1.f;
                }

                if (uvData) {
                    interleaved[base + 6] = uvData[i * uvItemSize + 0];
                    interleaved[base + 7] = (uvItemSize > 1) ? uvData[i * uvItemSize + 1] : 0.f;
                } else {
                    interleaved[base + 6] = 0.f;
                    interleaved[base + 7] = 0.f;
                }
            }

            auto byteSize = interleaved.size() * sizeof(float);
            WGPUBufferDescriptor vbDesc{};
            vbDesc.label = {.data = "vertex_buf", .length = 10};
            vbDesc.size = byteSize;
            vbDesc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            gb.vertexBuffer = wgpuDeviceCreateBuffer(device, &vbDesc);
            wgpuQueueWriteBuffer(queue, gb.vertexBuffer, 0, interleaved.data(), byteSize);
        }

        // Index buffer
        if (geometry->getIndex()) {
            auto indexAttr = geometry->getIndex();
            auto& arr = indexAttr->array();
            std::vector<uint32_t> indices(arr.size());
            for (size_t i = 0; i < arr.size(); ++i) {
                indices[i] = static_cast<uint32_t>(arr[i]);
            }
            auto byteSize = indices.size() * sizeof(uint32_t);
            WGPUBufferDescriptor ibDesc{};
            ibDesc.label = {.data = "index_buf", .length = 9};
            ibDesc.size = byteSize;
            ibDesc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            gb.indexBuffer = wgpuDeviceCreateBuffer(device, &ibDesc);
            wgpuQueueWriteBuffer(queue, gb.indexBuffer, 0, indices.data(), byteSize);
            gb.indexCount = static_cast<uint32_t>(indices.size());
        }

        geometryCache[id] = gb;
        return geometryCache[id];
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

    // Collect lights from scene hierarchy
    void collectLights(Object3D& object, RenderState* rs) {
        if (!object.visible) return;
        if (auto light = object.as<Light>()) {
            rs->pushLight(light);
        }
        for (auto& child : object.children) {
            collectLights(*child, rs);
        }
    }

    // Pack light state into GPU buffer
    void uploadLightData(const Lights::LightState& ls) {
        // LightData layout: counts(16) + ambient(16) + dir(128) + point(192) + spot(256) + hemi(96) = 704
        std::vector<float> data(LIGHT_UNIFORM_SIZE / sizeof(float), 0.0f);
        auto* u32 = reinterpret_cast<uint32_t*>(data.data());

        uint32_t nDir = std::min(static_cast<int>(ls.directional.size()), MAX_DIR_LIGHTS);
        uint32_t nPt = std::min(static_cast<int>(ls.point.size()), MAX_POINT_LIGHTS);
        uint32_t nSp = std::min(static_cast<int>(ls.spot.size()), MAX_SPOT_LIGHTS);
        uint32_t nHm = std::min(static_cast<int>(ls.hemi.size()), MAX_HEMI_LIGHTS);

        u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
        data[4] = ls.ambient.r; data[5] = ls.ambient.g; data[6] = ls.ambient.b; data[7] = 0;

        // Directional lights start at offset 8 (after header 32 bytes / 4 = 8 floats)
        size_t off = 8;
        for (uint32_t i = 0; i < nDir; i++) {
            auto& u = *ls.directional[i];
            auto& dir = std::get<Vector3>(u.at("direction"));
            auto& col = std::get<Color>(u.at("color"));
            data[off+0] = dir.x; data[off+1] = dir.y; data[off+2] = dir.z; data[off+3] = 0;
            data[off+4] = col.r; data[off+5] = col.g; data[off+6] = col.b; data[off+7] = 0;
            off += 8; // 32 bytes per directional
        }

        // Point lights start after all directional slots
        off = 8 + MAX_DIR_LIGHTS * 8; // header(8) + dir(4*8=32) = 40
        for (uint32_t i = 0; i < nPt; i++) {
            auto& u = *ls.point[i];
            auto& pos = std::get<Vector3>(u.at("position"));
            auto& col = std::get<Color>(u.at("color"));
            float dist = std::get<float>(u.at("distance"));
            float decay = std::get<float>(u.at("decay"));
            data[off+0] = pos.x; data[off+1] = pos.y; data[off+2] = pos.z; data[off+3] = 0;
            data[off+4] = col.r; data[off+5] = col.g; data[off+6] = col.b; data[off+7] = dist;
            data[off+8] = decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
            off += 12; // 48 bytes per point
        }

        // Spot lights
        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12;
        for (uint32_t i = 0; i < nSp; i++) {
            auto& u = *ls.spot[i];
            auto& pos = std::get<Vector3>(u.at("position"));
            auto& dir = std::get<Vector3>(u.at("direction"));
            auto& col = std::get<Color>(u.at("color"));
            float dist = std::get<float>(u.at("distance"));
            float decay = std::get<float>(u.at("decay"));
            float coneCos = std::get<float>(u.at("coneCos"));
            float penumbraCos = std::get<float>(u.at("penumbraCos"));
            data[off+0] = pos.x; data[off+1] = pos.y; data[off+2] = pos.z; data[off+3] = 0;
            data[off+4] = dir.x; data[off+5] = dir.y; data[off+6] = dir.z; data[off+7] = 0;
            data[off+8] = col.r; data[off+9] = col.g; data[off+10] = col.b; data[off+11] = dist;
            data[off+12] = decay; data[off+13] = coneCos; data[off+14] = penumbraCos; data[off+15] = 0;
            off += 16; // 64 bytes per spot
        }

        // Hemisphere lights
        off = 8 + MAX_DIR_LIGHTS * 8 + MAX_POINT_LIGHTS * 12 + MAX_SPOT_LIGHTS * 16;
        for (uint32_t i = 0; i < nHm; i++) {
            auto& u = *ls.hemi[i];
            auto& dir = std::get<Vector3>(u.at("direction"));
            auto& sky = std::get<Color>(u.at("skyColor"));
            auto& gnd = std::get<Color>(u.at("groundColor"));
            data[off+0] = dir.x; data[off+1] = dir.y; data[off+2] = dir.z; data[off+3] = 0;
            data[off+4] = sky.r; data[off+5] = sky.g; data[off+6] = sky.b; data[off+7] = 0;
            data[off+8] = gnd.r; data[off+9] = gnd.g; data[off+10] = gnd.b; data[off+11] = 0;
            off += 12; // 48 bytes per hemi
        }

        wgpuQueueWriteBuffer(queue, lightBuffer, 0, data.data(), LIGHT_UNIFORM_SIZE);
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        // Update window size if changed
        auto currentSize = canvas.size();
        if (currentSize.width() != size_.width() || currentSize.height() != size_.height()) {
            size_ = currentSize;
            configureSurface();
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

        // Collect and setup lights
        auto* rs = renderStates.get(&scene);
        rs->init();
        collectLights(scene, rs);
        rs->setupLights();
        rs->setupLightsView(&camera);

        // Upload light uniform data
        uploadLightData(rs->getLights().state);

        // Determine render target views
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTexture frameDepthTexture = nullptr;
        WGPUSurfaceTexture surfaceTexture{};
        bool useSurface = (currentRenderTarget_ == nullptr);

        if (useSurface) {
            wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
            if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
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

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = nullptr;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(clearColor_.r),
                static_cast<double>(clearColor_.g),
                static_cast<double>(clearColor_.b),
                static_cast<double>(clearAlpha_)};

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

        // Traverse scene and render meshes
        renderObject(pass, scene, projectionMatrix, viewMatrix, camera);

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

    void renderObject(WGPURenderPassEncoder pass, Object3D& object,
                       const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                       const Camera& camera) {

        if (auto mesh = object.as<Mesh>()) {
            auto geometry = mesh->geometry();
            if (geometry && geometry->hasAttribute("position")) {

                auto mat = mesh->material();
                Material* rawMat = mat.get();

                // Determine features and extract material parameters
                uint32_t features = FEAT_NONE;
                Color diffuse(1, 1, 1);
                float opacity = rawMat ? rawMat->opacity : 1.0f;
                Color specularColor(0, 0, 0);
                float shininess = 30.0f;
                float roughness = 0.5f, metalness = 0.0f;
                Color emissive(0, 0, 0);
                Texture* diffuseMap = nullptr;

                if (auto m = dynamic_cast<MeshStandardMaterial*>(rawMat)) {
                    features |= FEAT_LIGHTING | FEAT_PBR;
                    diffuse = m->color; roughness = m->roughness; metalness = m->metalness;
                    emissive = m->emissive;
                    if (m->map) { diffuseMap = m->map.get(); features |= FEAT_TEXTURE; }
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
                } else if (auto cm = dynamic_cast<MaterialWithColor*>(rawMat)) {
                    diffuse = cm->color;
                }

                // Get/create pipeline for this feature set
                auto& pe = getOrCreatePipeline(features);

                // Upload transform uniforms (binding 0)
                // Layout: model(64) + view(64) + proj(64) + normalCol0(16) + normalCol1(16) + normalCol2(16) + cameraPos(12) + pad(4) = 256
                float transformData[TRANSFORM_UNIFORM_SIZE / sizeof(float)];
                std::memset(transformData, 0, sizeof(transformData));
                std::memcpy(transformData, mesh->matrixWorld->elements.data(), 64); // model
                std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);     // view
                std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64); // proj

                // Normal matrix = inverse transpose of upper-left 3x3 of modelView
                Matrix4 modelView;
                modelView.multiplyMatrices(viewMatrix, *mesh->matrixWorld);
                Matrix3 normalMatrix;
                normalMatrix.setFromMatrix4(modelView);
                normalMatrix.invert();
                normalMatrix.transpose();
                // Pack as 3 vec4 columns (WGSL mat3x3 in uniform is padded to 3 vec4)
                auto& ne = normalMatrix.elements;
                // Column 0
                transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
                // Column 1
                transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
                // Column 2
                transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

                // Camera world position
                Vector3 camPos;
                camPos.setFromMatrixPosition(*camera.matrixWorld);
                transformData[60] = camPos.x;
                transformData[61] = camPos.y;
                transformData[62] = camPos.z;
                transformData[63] = 0;

                wgpuQueueWriteBuffer(queue, transformBuffer, 0, transformData, TRANSFORM_UNIFORM_SIZE);

                // Upload material uniforms (binding 1)
                float matData[MATERIAL_UNIFORM_SIZE / sizeof(float)];
                std::memset(matData, 0, sizeof(matData));
                matData[0] = diffuse.r; matData[1] = diffuse.g; matData[2] = diffuse.b; matData[3] = 1.0f;
                matData[4] = specularColor.r; matData[5] = specularColor.g; matData[6] = specularColor.b; matData[7] = shininess;
                matData[8] = roughness; matData[9] = metalness; matData[10] = opacity; matData[11] = 0;
                matData[12] = emissive.r; matData[13] = emissive.g; matData[14] = emissive.b; matData[15] = 0;
                // flags: x=hasTexture, y=hasLighting
                matData[16] = (features & FEAT_TEXTURE) ? 1.0f : 0.0f;
                matData[17] = (features & FEAT_LIGHTING) ? 1.0f : 0.0f;
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

                TextureEntry* texEntry = &dummyTexture;
                if (tex && diffuseMap) {
                    texEntry = &getOrCreateTexture(diffuseMap);
                }
                if (tex) {
                    { WGPUBindGroupEntry e{}; e.binding = 3; e.textureView = texEntry->view; entries.push_back(e); }
                    { WGPUBindGroupEntry e{}; e.binding = 4; e.sampler = texEntry->sampler; entries.push_back(e); }
                }

                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.label = {.data = "obj_bg", .length = 6};
                bgDesc.layout = pe.bindGroupLayout;
                bgDesc.entryCount = entries.size();
                bgDesc.entries = entries.data();
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

                wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

                auto& gb = getOrCreateGeometryBuffers(geometry.get());
                if (gb.vertexBuffer) {
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                         gb.vertexCount * VERTEX_STRIDE);
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
            renderObject(pass, *child, projectionMatrix, viewMatrix, camera);
        }
    }

    void dispose() {
        // Release geometry cache
        for (auto& [id, gb] : geometryCache) {
            if (gb.vertexBuffer) wgpuBufferRelease(gb.vertexBuffer);
            if (gb.indexBuffer) wgpuBufferRelease(gb.indexBuffer);
        }
        geometryCache.clear();

        // Release texture cache
        for (auto& [id, te] : textureCache) {
            if (te.view) wgpuTextureViewRelease(te.view);
            if (te.texture) wgpuTextureRelease(te.texture);
            if (te.sampler) wgpuSamplerRelease(te.sampler);
        }
        textureCache.clear();

        // Release dummy texture
        if (dummyTexture.view) wgpuTextureViewRelease(dummyTexture.view);
        if (dummyTexture.texture) wgpuTextureRelease(dummyTexture.texture);
        if (dummyTexture.sampler) wgpuSamplerRelease(dummyTexture.sampler);

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

        if (transformBuffer) wgpuBufferRelease(transformBuffer);
        if (materialBuffer) wgpuBufferRelease(materialBuffer);
        if (lightBuffer) wgpuBufferRelease(lightBuffer);
        if (queue) wgpuQueueRelease(queue);
        if (device) wgpuDeviceRelease(device);
        if (adapter) wgpuAdapterRelease(adapter);
        if (surface) wgpuSurfaceRelease(surface);
        if (instance) wgpuInstanceRelease(instance);

        renderStates.dispose();
        initialized = false;
    }

    ~Impl() {
        dispose();
    }
};


// --- DawnRenderer public API ---

DawnRenderer::DawnRenderer(Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(canvas)) {}

void DawnRenderer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

WindowSize DawnRenderer::size() const {
    return pimpl_->size_;
}

void DawnRenderer::setSize(const std::pair<int, int>& size) {
    pimpl_->canvas.setSize(size);
    pimpl_->size_ = {size.first, size.second};
    if (pimpl_->initialized) {
        pimpl_->configureSurface();
    }
}

float DawnRenderer::getTargetPixelRatio() const {
    return pimpl_->pixelRatio_;
}

void DawnRenderer::setPixelRatio(float value) {
    pimpl_->pixelRatio_ = value;
}

void DawnRenderer::setViewport(const Vector4& v) {
    pimpl_->viewport_.x = v.x; pimpl_->viewport_.y = v.y;
    pimpl_->viewport_.w = v.z; pimpl_->viewport_.h = v.w;
}

void DawnRenderer::setViewport(int x, int y, int width, int height) {
    pimpl_->viewport_.x = static_cast<float>(x);
    pimpl_->viewport_.y = static_cast<float>(y);
    pimpl_->viewport_.w = static_cast<float>(width);
    pimpl_->viewport_.h = static_cast<float>(height);
}

void DawnRenderer::setScissor(const Vector4& v) {
    pimpl_->scissor_.x = static_cast<uint32_t>(v.x);
    pimpl_->scissor_.y = static_cast<uint32_t>(v.y);
    pimpl_->scissor_.w = static_cast<uint32_t>(v.z);
    pimpl_->scissor_.h = static_cast<uint32_t>(v.w);
}

void DawnRenderer::setScissor(int x, int y, int width, int height) {
    pimpl_->scissor_.x = static_cast<uint32_t>(x);
    pimpl_->scissor_.y = static_cast<uint32_t>(y);
    pimpl_->scissor_.w = static_cast<uint32_t>(width);
    pimpl_->scissor_.h = static_cast<uint32_t>(height);
}

void DawnRenderer::setScissorTest(bool boolean) {
    pimpl_->scissorTest_ = boolean;
}

void DawnRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::clear(bool /*color*/, bool /*depth*/, bool /*stencil*/) {
    // Clearing happens at render pass begin via loadOp = Clear
}

RenderTarget* DawnRenderer::getRenderTarget() {
    return pimpl_->currentRenderTarget_;
}

void DawnRenderer::setRenderTarget(RenderTarget* renderTarget, int /*activeCubeFace*/, int /*activeMipmapLevel*/) {
    pimpl_->currentRenderTarget_ = renderTarget;
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

    while (!mapData.done) {
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

void DawnRenderer::dispose() {
    pimpl_->dispose();
}

DawnRenderer::~DawnRenderer() = default;
