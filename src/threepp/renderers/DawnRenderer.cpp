
#include "threepp/renderers/DawnRenderer.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#define GLFW_INCLUDE_NONE
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <webgpu/webgpu.h>

#include <cstring>
#include <iostream>
#include <vector>

using namespace threepp;

namespace {

    // WGSL shader for rendering meshes with vertex colors or a uniform color.
    // Supports model-view-projection transform via a uniform buffer.
    const char* basicWGSL = R"(
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvpMatrix * vec4<f32>(in.position, 1.0);
    out.color = u.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
)";

}// namespace


struct DawnRenderer::Impl {

    Canvas& canvas;

    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;

    WGPUShaderModule shaderModule = nullptr;
    WGPURenderPipeline pipeline = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;

    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;

    WindowSize size_;
    float pixelRatio_ = 1.0f;
    Color clearColor_{0x000000};
    float clearAlpha_ = 1.0f;

    // Per-frame uniform buffer (MVP + color)
    WGPUBuffer uniformBuffer = nullptr;
    WGPUBindGroup bindGroup = nullptr;

    // Geometry buffers cache
    struct GeometryBuffers {
        WGPUBuffer vertexBuffer = nullptr;
        WGPUBuffer indexBuffer = nullptr;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
    };
    std::unordered_map<unsigned int, GeometryBuffers> geometryCache;

    bool initialized = false;

    explicit Impl(Canvas& canvas)
        : canvas(canvas), size_(canvas.size()) {

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

        // Create shader module and pipeline
        createPipeline();

        // Create uniform buffer
        createUniformBuffer();

        initialized = true;
        std::cout << "DawnRenderer: WebGPU initialized successfully" << std::endl;
    }

    void createSurface() {
#ifdef __linux__
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        Display* x11Display = glfwGetX11Display();
        Window x11Window = glfwGetX11Window(glfwWindow);

        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.chain.next = nullptr;
        xlibSource.display = x11Display;
        xlibSource.window = static_cast<uint64_t>(x11Window);

        WGPUSurfaceDescriptor surfDesc{};
        surfDesc.nextInChain = &xlibSource.chain;
        WGPUStringView label = {.data = "threepp_surface", .length = 15};
        surfDesc.label = label;

        surface = wgpuInstanceCreateSurface(instance, &surfDesc);
#endif
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

    void createPipeline() {
        // Shader module
        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.chain.next = nullptr;
        wgslSource.code = {.data = basicWGSL, .length = strlen(basicWGSL)};

        WGPUShaderModuleDescriptor shaderDesc{};
        shaderDesc.nextInChain = &wgslSource.chain;
        WGPUStringView shaderLabel = {.data = "basic_shader", .length = 12};
        shaderDesc.label = shaderLabel;

        shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);

        // Bind group layout for uniform buffer
        WGPUBindGroupLayoutEntry bglEntry{};
        bglEntry.binding = 0;
        bglEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
        bglEntry.buffer.minBindingSize = sizeof(float) * 20; // mat4 + vec4

        WGPUBindGroupLayoutDescriptor bglDesc{};
        WGPUStringView bglLabel = {.data = "bind_group_layout", .length = 17};
        bglDesc.label = bglLabel;
        bglDesc.entryCount = 1;
        bglDesc.entries = &bglEntry;

        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        // Pipeline layout
        WGPUPipelineLayoutDescriptor plDesc{};
        WGPUStringView plLabel = {.data = "pipeline_layout", .length = 15};
        plDesc.label = plLabel;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bindGroupLayout;

        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Vertex buffer layout: vec3<f32> position
        WGPUVertexAttribute posAttr{};
        posAttr.format = WGPUVertexFormat_Float32x3;
        posAttr.offset = 0;
        posAttr.shaderLocation = 0;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = sizeof(float) * 3;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 1;
        vbLayout.attributes = &posAttr;

        // Blend state
        WGPUBlendState blendState{};
        blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
        blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blendState.color.operation = WGPUBlendOperation_Add;
        blendState.alpha.srcFactor = WGPUBlendFactor_One;
        blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blendState.alpha.operation = WGPUBlendOperation_Add;

        // Color target
        WGPUColorTargetState colorTarget{};
        colorTarget.format = surfaceFormat;
        colorTarget.blend = &blendState;
        colorTarget.writeMask = WGPUColorWriteMask_All;

        // Fragment state
        WGPUStringView fsEntry = {.data = "fs_main", .length = 7};
        WGPUFragmentState fragmentState{};
        fragmentState.module = shaderModule;
        fragmentState.entryPoint = fsEntry;
        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;

        // Depth stencil
        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth24Plus;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        // Render pipeline
        WGPURenderPipelineDescriptor pipelineDesc{};
        WGPUStringView pipeLabel = {.data = "basic_pipeline", .length = 14};
        pipelineDesc.label = pipeLabel;
        pipelineDesc.layout = pipelineLayout;

        WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
        pipelineDesc.vertex.module = shaderModule;
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

        pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
    }

    void createUniformBuffer() {
        WGPUBufferDescriptor bufDesc{};
        WGPUStringView label = {.data = "uniform_buffer", .length = 14};
        bufDesc.label = label;
        bufDesc.size = sizeof(float) * 20; // mat4x4 + vec4
        bufDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

        uniformBuffer = wgpuDeviceCreateBuffer(device, &bufDesc);

        // Bind group
        WGPUBindGroupEntry bgEntry{};
        bgEntry.binding = 0;
        bgEntry.buffer = uniformBuffer;
        bgEntry.offset = 0;
        bgEntry.size = sizeof(float) * 20;

        WGPUBindGroupDescriptor bgDesc{};
        WGPUStringView bgLabel = {.data = "bind_group", .length = 10};
        bgDesc.label = bgLabel;
        bgDesc.layout = bindGroupLayout;
        bgDesc.entryCount = 1;
        bgDesc.entries = &bgEntry;

        bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);
    }

    GeometryBuffers& getOrCreateGeometryBuffers(BufferGeometry* geometry) {
        auto id = geometry->id;
        auto it = geometryCache.find(id);
        if (it != geometryCache.end()) {
            return it->second;
        }

        GeometryBuffers gb{};

        // Position attribute
        if (geometry->hasAttribute("position")) {
            auto posAttr = geometry->getAttribute<float>("position");
            auto& arr = posAttr->array();
            auto byteSize = arr.size() * sizeof(float);

            WGPUBufferDescriptor vbDesc{};
            WGPUStringView label = {.data = "vertex_buffer", .length = 13};
            vbDesc.label = label;
            vbDesc.size = byteSize;
            vbDesc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;

            gb.vertexBuffer = wgpuDeviceCreateBuffer(device, &vbDesc);
            wgpuQueueWriteBuffer(queue, gb.vertexBuffer, 0, arr.data(), byteSize);
            gb.vertexCount = static_cast<uint32_t>(posAttr->count());
        }

        // Index buffer
        if (geometry->getIndex()) {
            auto indexAttr = geometry->getIndex();
            auto& arr = indexAttr->array();
            // Convert to uint32
            std::vector<uint32_t> indices(arr.size());
            for (size_t i = 0; i < arr.size(); ++i) {
                indices[i] = static_cast<uint32_t>(arr[i]);
            }
            auto byteSize = indices.size() * sizeof(uint32_t);

            WGPUBufferDescriptor ibDesc{};
            WGPUStringView label = {.data = "index_buffer", .length = 12};
            ibDesc.label = label;
            ibDesc.size = byteSize;
            ibDesc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;

            gb.indexBuffer = wgpuDeviceCreateBuffer(device, &ibDesc);
            wgpuQueueWriteBuffer(queue, gb.indexBuffer, 0, indices.data(), byteSize);
            gb.indexCount = static_cast<uint32_t>(indices.size());
        }

        geometryCache[id] = gb;
        return geometryCache[id];
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        // Update window size if changed
        auto currentSize = canvas.size();
        if (currentSize.width() != size_.width() || currentSize.height() != size_.height()) {
            size_ = currentSize;
            configureSurface();
            // Recreate depth texture
        }

        // Update camera matrices
        scene.updateMatrixWorld();
        if (!camera.parent) {
            camera.updateMatrixWorld();
        }
        camera.updateWorldMatrix(false, false);

        Matrix4 projectionMatrix = camera.projectionMatrix;
        Matrix4 viewMatrix = camera.matrixWorldInverse;

        // Get surface texture
        WGPUSurfaceTexture surfaceTexture{};
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

        if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            return;
        }

        // Create texture view
        WGPUTextureViewDescriptor viewDesc{};
        viewDesc.nextInChain = nullptr;
        WGPUStringView viewLabel = {.data = "surface_view", .length = 12};
        viewDesc.label = viewLabel;
        viewDesc.format = surfaceFormat;
        viewDesc.dimension = WGPUTextureViewDimension_2D;
        viewDesc.baseMipLevel = 0;
        viewDesc.mipLevelCount = 1;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        viewDesc.aspect = WGPUTextureAspect_All;

        WGPUTextureView surfaceView = wgpuTextureCreateView(surfaceTexture.texture, &viewDesc);

        // Create depth texture
        WGPUTextureDescriptor depthTexDesc{};
        WGPUStringView depthLabel = {.data = "depth_texture", .length = 13};
        depthTexDesc.label = depthLabel;
        depthTexDesc.size = {static_cast<uint32_t>(size_.width()), static_cast<uint32_t>(size_.height()), 1};
        depthTexDesc.mipLevelCount = 1;
        depthTexDesc.sampleCount = 1;
        depthTexDesc.dimension = WGPUTextureDimension_2D;
        depthTexDesc.format = WGPUTextureFormat_Depth24Plus;
        depthTexDesc.usage = WGPUTextureUsage_RenderAttachment;

        WGPUTexture depthTexture = wgpuDeviceCreateTexture(device, &depthTexDesc);
        WGPUTextureView depthView = wgpuTextureCreateView(depthTexture, nullptr);

        // Command encoder
        WGPUCommandEncoderDescriptor encDesc{};
        WGPUStringView encLabel = {.data = "command_encoder", .length = 15};
        encDesc.label = encLabel;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = surfaceView;
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
        WGPUStringView passLabel = {.data = "render_pass", .length = 11};
        passDesc.label = passLabel;
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pipeline);

        // Traverse scene and render meshes
        renderObject(pass, scene, projectionMatrix, viewMatrix);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // Submit
        WGPUCommandBufferDescriptor cmdDesc{};
        WGPUStringView cmdLabel = {.data = "command_buffer", .length = 14};
        cmdDesc.label = cmdLabel;
        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdDesc);

        wgpuQueueSubmit(queue, 1, &cmdBuffer);

        // Present
        wgpuSurfacePresent(surface);

        // Cleanup per-frame resources
        wgpuCommandBufferRelease(cmdBuffer);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(depthView);
        wgpuTextureRelease(depthTexture);
        wgpuTextureViewRelease(surfaceView);
        wgpuTextureRelease(surfaceTexture.texture);
    }

    void renderObject(WGPURenderPassEncoder pass, Object3D& object,
                       const Matrix4& projectionMatrix, const Matrix4& viewMatrix) {

        // Process this object if it's a Mesh with geometry
        if (auto mesh = object.as<Mesh>()) {
            auto geometry = mesh->geometry();
            if (geometry && geometry->hasAttribute("position")) {

                // Compute MVP
                Matrix4 mvp;
                mvp.multiplyMatrices(viewMatrix, *mesh->matrixWorld);
                mvp.premultiply(projectionMatrix);

                // Extract material color
                float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                if (auto mat = mesh->material()) {
                    if (auto colorMat = dynamic_cast<MaterialWithColor*>(mat.get())) {
                        color[0] = colorMat->color.r;
                        color[1] = colorMat->color.g;
                        color[2] = colorMat->color.b;
                    }
                    color[3] = mat->opacity;
                }

                // Write uniforms: MVP matrix (16 floats) + color (4 floats)
                float uniformData[20];
                std::memcpy(uniformData, mvp.elements.data(), sizeof(float) * 16);
                std::memcpy(uniformData + 16, color, sizeof(float) * 4);
                wgpuQueueWriteBuffer(queue, uniformBuffer, 0, uniformData, sizeof(uniformData));

                wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);

                auto& gb = getOrCreateGeometryBuffers(geometry.get());

                if (gb.vertexBuffer) {
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                         gb.vertexCount * sizeof(float) * 3);

                    if (gb.indexBuffer) {
                        wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                             WGPUIndexFormat_Uint32, 0,
                                                             gb.indexCount * sizeof(uint32_t));
                        wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                    } else {
                        wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                    }
                }
            }
        }

        // Recurse into children
        for (auto& child : object.children) {
            renderObject(pass, *child, projectionMatrix, viewMatrix);
        }
    }

    void dispose() {
        // Release geometry cache
        for (auto& [id, gb] : geometryCache) {
            if (gb.vertexBuffer) wgpuBufferRelease(gb.vertexBuffer);
            if (gb.indexBuffer) wgpuBufferRelease(gb.indexBuffer);
        }
        geometryCache.clear();

        if (bindGroup) wgpuBindGroupRelease(bindGroup);
        if (uniformBuffer) wgpuBufferRelease(uniformBuffer);
        if (pipeline) wgpuRenderPipelineRelease(pipeline);
        if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
        if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
        if (shaderModule) wgpuShaderModuleRelease(shaderModule);
        if (queue) wgpuQueueRelease(queue);
        if (device) {
            wgpuDeviceRelease(device);
        }
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

void DawnRenderer::setViewport(const Vector4& /*v*/) {
    // TODO: implement viewport
}

void DawnRenderer::setViewport(int /*x*/, int /*y*/, int /*width*/, int /*height*/) {
    // TODO: implement viewport
}

void DawnRenderer::setScissor(const Vector4& /*v*/) {
    // TODO: implement scissor
}

void DawnRenderer::setScissor(int /*x*/, int /*y*/, int /*width*/, int /*height*/) {
    // TODO: implement scissor
}

void DawnRenderer::setScissorTest(bool /*boolean*/) {
    // TODO: implement scissor test
}

void DawnRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::clear(bool /*color*/, bool /*depth*/, bool /*stencil*/) {
    // Clearing happens at render pass begin via loadOp = Clear
}

RenderTarget* DawnRenderer::getRenderTarget() {
    return nullptr;
}

void DawnRenderer::setRenderTarget(RenderTarget* /*renderTarget*/, int /*activeCubeFace*/, int /*activeMipmapLevel*/) {
    // TODO: implement render targets
}

std::vector<unsigned char> DawnRenderer::readRGBPixels() {
    // TODO: implement pixel readback
    return {};
}

void DawnRenderer::dispose() {
    pimpl_->dispose();
}

DawnRenderer::~DawnRenderer() = default;
