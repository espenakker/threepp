# Dawn (WebGPU) Integration Architecture Plan for threepp

## Context

threepp is a C++20 port of three.js with a monolithic OpenGL 3.3 renderer. The goal is to introduce Dawn (WebGPU) as an alternative backend without rewriting the engine. The user proposed hooking Dawn at the `GLRenderer::Impl` bundle boundary, which is the correct seam. This plan refines that proposal based on a comprehensive codebase review, correcting several assumptions and adding missing details.

---

## Key Findings That Refine the Original Plan

### 1. Many "GL" modules are actually backend-neutral (misnamed)
The following modules live in `threepp::gl` namespace and have `GL` prefixes but contain **zero GL API calls**:
- **GLLights / GLRenderStates** — Pure math: light uniform computation, light/shadow array management
- **GLRenderLists** — Scene sorting (opaque/transparent). Only GL dependency: `GLProgram*` in `RenderItem` (easily genericized) and `GLProperties&`
- **GLClipping** — Plane projection math. Depends on `GLProperties` but logic is pure math
- **GLObjects** — Generic geometry extraction from Object3D hierarchy
- **GLMorphTargets** — Morph target weight computation
- **GLInfo** — Pure counters (frame, draw calls, triangles, memory)
- **GLMaterials** — Uniform value extraction from materials (the values are generic; only storage via `GLProperties` is GL-specific)

**Implication**: These should be extracted to a common `renderers/common/` namespace as part of the refactor — they are shared infrastructure, not GL backend code.

### 2. GLRenderTarget is already backend-neutral
Despite its name, `GLRenderTarget` contains no GL handles in its header — just dimensions, textures, viewport/scissor, depth/stencil flags. It can be renamed to `RenderTarget` with minimal changes.

### 3. `setProgram()` is the hardest decomposition (not addressed in original plan)
This ~300-line method (GLRenderer.cpp:732-1026) is the critical mixing point. It:
- Checks material version / program cache staleness (backend-neutral logic)
- Calls `getProgram()` which computes `ProgramParameters` and looks up/compiles GL programs (mixed)
- Sets 30+ uniform categories: matrices, camera, lights, fog, material-specific (uniform *computation* is neutral, uniform *upload* is GL-specific)
- Manages `GLState` for texture binding during uniform setup

**Strategy**: Split into (a) `prepareMaterial()` — computes pipeline key + uniform values, and (b) `bindProgram()` — backend-specific program binding and uniform upload.

### 4. Canvas refactoring is independent and should NOT be step 1
Canvas creates a GLFW window with an OpenGL 3.3 context (`glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)`). For Dawn, we need a native window handle without GL context creation. But this is **orthogonal** to the renderer abstraction — Dawn can initially use its own window/surface setup while the renderer architecture is being built. Canvas refactoring can happen in parallel.

### 5. Shader strategy for Dawn is underspecified
`GLProgram` does GLSL-specific work: concatenates prefixes, resolves `ShaderChunk` includes, unrolls loops, compiles with `glShaderSource`/`glCompileShader`. Dawn needs either:
- **Option A**: WGSL shaders (requires rewriting all shader chunks — high effort)
- **Option B**: GLSL -> SPIR-V cross-compilation via Tint/naga/glslang, then feed SPIR-V to Dawn
- **Recommended**: Option B for initial integration; Option A as a longer-term goal

### 6. GLProperties is an important pattern the plan should preserve
`GLProperties` stores GL-specific metadata (handles, compiled programs, cache state) as external maps keyed by core objects. This avoids polluting core types. The Dawn backend needs its own `DawnProperties` following the same pattern.

---

## Refined Architecture

### New Directory Structure
```
include/threepp/renderers/
  Renderer.hpp              (NEW - backend-neutral base interface)
  RenderTarget.hpp          (RENAMED from GLRenderTarget.hpp)
  GLRenderer.hpp            (MODIFIED - derives from Renderer)
  gl/
    GLState.hpp             (unchanged)
    GLShadowMap.hpp         (unchanged initially)
    GLInfo.hpp              (MOVED to common/ eventually)

src/threepp/renderers/
  common/                   (NEW directory)
    RenderLists.hpp/cpp     (EXTRACTED from gl/GLRenderLists)
    RenderStates.hpp/cpp    (EXTRACTED from gl/GLRenderStates)
    Lights.hpp/cpp          (EXTRACTED from gl/GLLights)
    Clipping.hpp/cpp        (EXTRACTED from gl/GLClipping)
    RenderInfo.hpp/cpp      (EXTRACTED from gl/GLInfo)
    PipelineKey.hpp/cpp     (EXTRACTED from gl/ProgramParameters)
    MaterialUniforms.hpp    (EXTRACTED from gl/GLMaterials — uniform value computation)
    MorphTargets.hpp        (EXTRACTED from gl/GLMorphTargets)
  gl/                       (existing — stays as GL backend)
    GLProgram.hpp/cpp       (unchanged)
    GLState.hpp/cpp         (unchanged)
    GLBindingStates.hpp/cpp (unchanged)
    GLAttributes.hpp/cpp    (unchanged)
    GLTextures.hpp/cpp      (unchanged)
    GLUniforms.hpp/cpp      (unchanged)
    GLBufferRenderer.hpp    (unchanged)
    GLProperties.hpp        (unchanged)
    GLBackground.hpp/cpp    (unchanged, uses Renderer& instead of GLRenderer&)
    GLShadowMap.hpp/cpp     (unchanged initially, uses Renderer& later)
  dawn/                     (NEW directory — future)
    DawnBackend.hpp/cpp
    DawnProgram.hpp/cpp
    DawnState.hpp/cpp
    DawnTextures.hpp/cpp
    DawnProperties.hpp
```

### Renderer Base Interface
```cpp
// include/threepp/renderers/Renderer.hpp
class Renderer {
public:
    virtual ~Renderer() = default;

    // Core rendering
    virtual void render(Object3D& scene, Camera& camera) = 0;

    // Size and viewport
    virtual WindowSize size() const = 0;
    virtual void setSize(const std::pair<int, int>& size) = 0;
    virtual void setViewport(int x, int y, int w, int h) = 0;
    virtual void setScissor(int x, int y, int w, int h) = 0;
    virtual void setScissorTest(bool enabled) = 0;

    // Clear
    virtual void setClearColor(const Color& color, float alpha = 1) = 0;
    virtual void clear(bool color = true, bool depth = true, bool stencil = true) = 0;

    // Render targets
    virtual void setRenderTarget(RenderTarget* target, int activeCubeFace = 0, int activeMipmapLevel = 0) = 0;
    virtual RenderTarget* getRenderTarget() = 0;

    // Pixel ratio
    virtual void setPixelRatio(float value) = 0;
    virtual float getTargetPixelRatio() const = 0;

    // Readback
    virtual std::vector<unsigned char> readRGBPixels() = 0;

    // Dispose
    virtual void dispose() = 0;

    // Common configuration (non-virtual, shared by all backends)
    bool autoClear = true;
    bool autoClearColor = true;
    bool autoClearDepth = true;
    bool autoClearStencil = true;
    bool sortObjects = true;
    std::vector<Plane> clippingPlanes;
    bool localClippingEnabled = false;
    float toneMappingExposure = 1.0f;
    ToneMapping toneMapping = ToneMapping::None;
    Encoding outputEncoding = Encoding::Linear;
    bool physicallyCorrectLights = false;
};
```

**GLRenderer additions** (GL-only, not on base):
- `gl::GLState& state()`
- `gl::GLShadowMap& shadowMap()`
- `getGlTextureId()`, `getGlBufferId()`
- `renderBufferDirect()` (implementation detail)

---

## Implementation Steps (Refined Order)

### Phase 1: Extract Backend-Neutral Common Modules
**Goal**: Move misnamed GL modules to `renderers/common/` without changing behavior.

1. Create `src/threepp/renderers/common/` directory
2. Extract `GLLights` -> `common::Lights` (rename namespace, zero logic changes)
3. Extract `GLRenderStates` -> `common::RenderStates` (update includes, replace `GLLights` with `Lights`)
4. Extract `GLRenderLists` -> `common::RenderLists` (replace `GLProgram*` with `void* program` or a generic handle in `RenderItem`)
5. Extract `GLInfo` -> `common::RenderInfo`
6. Extract `GLClipping` -> `common::Clipping` (decouple from `GLProperties` — clipping only needs a property-free interface)
7. Update `GLRenderer.cpp` includes to use new locations
8. Update `src/CMakeLists.txt`

**Files changed**: ~15 files (moves + include updates)
**Risk**: Low — pure refactoring, no behavior change. Compile and run tests after each move.

### Phase 2: Introduce Renderer Base and RenderTarget Rename
**Goal**: Create the backend-neutral public interface.

1. Create `include/threepp/renderers/Renderer.hpp` with the interface above
2. Rename `GLRenderTarget` -> `RenderTarget` (keep `GLRenderTarget` as a typedef for backward compat temporarily)
3. Make `GLRenderer` inherit from `Renderer`
4. Update consumer code that uses `GLRenderer` directly — examples can continue using `GLRenderer`, but internal modules (like `GLShadowMap`, `GLBackground`) should accept `Renderer&` where possible

**Files changed**: ~8-10 files
**Risk**: Medium — public API change. Must ensure all examples still compile. The typedef eases migration.

### Phase 3: Extract PipelineKey from ProgramParameters
**Goal**: Create the backend-neutral shader variant descriptor.

1. Move `ProgramParameters` to `common::PipelineKey`
2. Remove dependencies on `GLRenderer` (replace with `Renderer` config reads) and `GLCapabilities` (introduce a `RendererCapabilities` struct)
3. `GLPrograms` continues to consume `PipelineKey` but the key computation is now backend-neutral
4. Create `common::RendererCapabilities` — a simple struct with `maxTextures`, `maxTextureSize`, etc. that each backend populates

**Files changed**: ~5-6 files
**Risk**: Medium — `ProgramParameters` constructor reads from `GLRenderer` and `GLCapabilities::instance()`. Need to extract those reads into a capabilities struct.

### Phase 4: Split `setProgram()` into Neutral + Backend-Specific
**Goal**: Separate uniform/pipeline computation from GL-specific binding.

1. Extract uniform value computation from `setProgram()` into a neutral `prepareMaterialState()` method that produces a `MaterialRenderState` struct (uniforms map, pipeline key, texture list)
2. Keep GL-specific program lookup, compilation, and uniform upload in `GLRenderer::Impl::bindMaterial()`
3. This is the hardest step — `setProgram()` is ~300 lines of interleaved logic

**Files changed**: GLRenderer.cpp primarily, plus new common struct
**Risk**: High — most complex decomposition. Must be done incrementally with careful testing.

### Phase 5: Canvas Abstraction (parallel, independent)
**Goal**: Allow non-GL window/surface creation.

1. Extract a `WindowSurface` interface from Canvas (or add a Canvas creation mode)
2. Add GLFW hints that skip GL context creation for Dawn path
3. Dawn uses `glfwGetX11Window()` / `glfwGetWin32Window()` etc. to get native handle
4. Keep existing Canvas behavior as default

**Files changed**: Canvas.hpp, Canvas.cpp
**Risk**: Low-medium — GLFW supports creating non-GL windows via `GLFW_NO_API` hint

### Phase 6: Dawn Backend (future, builds on Phases 1-5)
**Goal**: Implement the Dawn rendering backend.

1. Create `renderers/dawn/` directory
2. Implement `DawnRenderer` deriving from `Renderer`
3. Implement Dawn equivalents of: program compilation (GLSL->SPIR-V via Tint), state management, texture/buffer creation, draw submission
4. Reuse all `common/` modules (RenderLists, RenderStates, Lights, PipelineKey, etc.)
5. Start with basic triangle rendering, then materials, then shadows

**Risk**: High — largest effort. Phases 1-5 de-risk this by ensuring the abstractions work.

---

## What NOT to Abstract (Confirmed by Review)

These modules are correctly identified as GL-specific and should stay in `renderers/gl/`:
- `GLProgram` — GLSL compilation pipeline
- `GLState` — GL state machine cache
- `GLBindingStates` — VAO/VBO management
- `GLAttributes` — `glGenBuffers`/`glBufferData`
- `GLTextures` — `glTexImage2D`, FBO setup
- `GLUniforms` — GL uniform setters
- `GLBufferRenderer` / `GLIndexedBufferRenderer` — `glDrawArrays`/`glDrawElements`
- `GLCapabilities` — `glGetParameteri` queries
- `GLProperties` — GL handle storage

---

## Verification Plan

After each phase:
1. **Build**: `cmake --build build` — must compile cleanly with no warnings
2. **Run tests**: `cd build && ctest` — all existing tests must pass
3. **Run examples**: Manually verify 2-3 representative examples still render correctly (e.g., basic mesh, shadows, render target)
4. **API check**: Verify that consumer code using `GLRenderer` directly is unaffected (backward compat)

For Phase 6 (Dawn):
5. **Triangle test**: Render a colored triangle to validate the full pipeline
6. **Material test**: Render a lit sphere with Phong/Standard material
7. **Shadow test**: Verify shadow mapping works through the Dawn backend

---

## Critical Files Reference

| File | Role | Phase |
|------|------|-------|
| `include/threepp/renderers/GLRenderer.hpp` | Public API, needs Renderer base | 2 |
| `src/threepp/renderers/GLRenderer.cpp` | 1530-line Impl, main decomposition target | 1, 3, 4 |
| `include/threepp/renderers/GLRenderTarget.hpp` | Rename to RenderTarget | 2 |
| `src/threepp/renderers/gl/ProgramParameters.hpp/cpp` | Extract to PipelineKey | 3 |
| `src/threepp/renderers/gl/GLLights.hpp` | Move to common/ | 1 |
| `src/threepp/renderers/gl/GLRenderLists.hpp/cpp` | Move to common/ | 1 |
| `src/threepp/renderers/gl/GLRenderStates.hpp/cpp` | Move to common/ | 1 |
| `src/threepp/renderers/gl/GLClipping.hpp/cpp` | Move to common/ | 1 |
| `src/threepp/renderers/gl/GLInfo.hpp/cpp` | Move to common/ | 1 |
| `src/threepp/canvas/Canvas.cpp` | GL context hardcoded, needs abstraction | 5 |
| `src/CMakeLists.txt` | Build system updates | 1-5 |
