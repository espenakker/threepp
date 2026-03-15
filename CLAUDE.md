# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

threepp is a cross-platform C++20 port of three.js (r129). It provides a scene graph, materials, geometries, lights, cameras, loaders, and rendering backends for 3D graphics applications.

## Build Commands

### Configure and build (Windows)
```sh
cmake . -A x64 -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Configure and build (Unix)
```sh
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run all tests
```sh
cd build/tests && ctest --output-on-failure        # Unix
cd build/tests && ctest -C Release --output-on-failure  # Windows
```

### Run a single test
```sh
cd build/tests && ctest -R <TestName> --output-on-failure
# Or run the test executable directly:
./build/bin/<TestName>
```

### Key CMake options
- `-DTHREEPP_WITH_DAWN=ON` — Enable WebGPU/Dawn renderer backend (via wgpu-native)
- `-DTHREEPP_BUILD_TESTS=ON` — Build Catch2 test suite (default ON)
- `-DTHREEPP_BUILD_EXAMPLES=ON` — Build examples (default ON)
- `-DTHREEPP_USE_EXTERNAL_GLFW=ON` — Use system GLFW instead of bundled
- `-DTHREEPP_FETCH_ASSIMP=ON` — Fetch assimp for examples

## Architecture

### Namespace and conventions
All code lives in `namespace threepp`. Math types (Vector3, Matrix4, etc.) are value types. Everything else uses `std::shared_ptr` with static `::create()` factory methods. Materials, geometries, and textures auto-dispose when they go out of scope.

### Source layout
- `include/threepp/` — Public headers (mirrors three.js module structure)
- `src/threepp/` — Implementations matching the header tree
- `src/external/` — Bundled dependencies (GLFW, GLAD, stb, nlohmann/json, pugixml, wgpu-native, etc.)
- `tests/` — Catch2 tests organized by module (cameras, core, math, utils, renderers, loaders)
- `examples/` — 80+ example programs
- `data/` — Fonts, models, textures used by tests and examples

### Multi-renderer architecture (Dawn branch)
`Renderer` (`include/threepp/renderers/Renderer.hpp`) is the backend-neutral abstract interface. Two concrete implementations exist:
- `GLRenderer` — OpenGL 3.3 backend (original, fully featured)
- `DawnRenderer` — WebGPU backend via wgpu-native (newer, on the Dawn branch)

Shared rendering infrastructure (lights, render states, render lists) is in `src/threepp/renderers/common/`. Backend-specific code stays in `renderers/gl/` and `renderers/dawn/`. The `CrossRenderer_test` in `tests/renderers/` validates visual parity between backends.

### Canvas abstraction
`Canvas` (`include/threepp/canvas/Canvas.hpp`) handles windowing via GLFW and is renderer-agnostic. The render loop uses `canvas.animate([&] { renderer.render(*scene, *camera); })`.

### Entry point pattern
```cpp
#include "threepp/threepp.hpp"
// Canvas → Renderer → Scene + Camera + Lights → animate loop
```

## Formatting
Uses clang-format (LLVM-based style, 4-space indent, no tabs, no column limit). Config is in `.clang-format`.

## Testing
Tests use Catch2 v3.4.0 (fetched via FetchContent at configure time). Test executables link against `threepp` and `Catch2::Catch2WithMain`. Test sources can access private headers via `target_include_directories(... PRIVATE "${PROJECT_SOURCE_DIR}/src")`. The `DATA_FOLDER` preprocessor define points to the repo's `data/` directory for test assets.
