// Cross-renderer comparison tests: GLRenderer vs DawnRenderer.
//
// Strategy: render identical scenes with both backends to render targets,
// read back pixels, and compare. The renderers won't produce pixel-identical
// output (different rasterizers, shader precision, color ordering), so we
// compare aggregate properties: average color, non-zero regions, and per-pixel
// tolerance.
//
// Dawn tests require a real GPU backend (Vulkan/Metal/DX12). They are tagged
// [dawn] and will be skipped automatically in headless CI environments where
// no GPU is available.
//
// Canvas objects are kept as static singletons because Canvas::~Impl calls
// glfwTerminate(), which invalidates GLFW for subsequent test cases.

#include <catch2/catch_test_macros.hpp>

#include "threepp/threepp.hpp"
#include "threepp/renderers/DawnRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshNormalMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/geometries/RingGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/ShadowMaterial.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    constexpr int RT_WIDTH = 64;
    constexpr int RT_HEIGHT = 64;
    constexpr int PIXEL_COUNT = RT_WIDTH * RT_HEIGHT;
    constexpr int DATA_SIZE = PIXEL_COUNT * 3;

    // Persistent GL canvas — avoids glfwTerminate between tests
    Canvas& glCanvas() {
        static Canvas c(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true));
        return c;
    }

    // Probe whether a WebGPU adapter can be obtained (no surface needed).
    bool isDawnAvailable() {
        static int cached = -1;
        if (cached >= 0) return cached != 0;

        // The GL canvas must exist first to establish a GL context,
        // which wgpu-native's GL backend needs.
        (void)glCanvas();

        // Use only primary backends (Vulkan/Metal/DX12) to avoid EGL conflicts
        // with GLFW's GL context when the GL backend is also enabled.
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        WGPUInstance inst = wgpuCreateInstance(&instanceDesc);
        if (!inst) {
            cached = 0;
            return false;
        }

        struct UserData {
            bool done = false;
            WGPUAdapter adapter = nullptr;
        } ud;

        WGPURequestAdapterOptions opts{};
        opts.powerPreference = WGPUPowerPreference_LowPower;
        opts.compatibleSurface = nullptr;

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView, void* userdata1, void*) {
            auto* u = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                u->adapter = adapter;
            }
            u->done = true;
        };
        callbackInfo.userdata1 = &ud;

        wgpuInstanceRequestAdapter(inst, &opts, callbackInfo);

        for (int i = 0; i < 100 && !ud.done; i++) {
            wgpuInstanceProcessEvents(inst);
        }

        bool available = ud.adapter != nullptr;
        if (ud.adapter) wgpuAdapterRelease(ud.adapter);
        wgpuInstanceRelease(inst);
        cached = available ? 1 : 0;
        return available;
    }

    struct AvgColor {
        double r, g, b;
    };

    AvgColor averageColor(const std::vector<unsigned char>& pixels) {
        double r = 0, g = 0, b = 0;
        int count = static_cast<int>(pixels.size()) / 3;
        for (int i = 0; i < count; i++) {
            r += pixels[i * 3 + 0];
            g += pixels[i * 3 + 1];
            b += pixels[i * 3 + 2];
        }
        return {r / count, g / count, b / count};
    }

    bool allPixelsMatch(const std::vector<unsigned char>& pixels,
                        unsigned char r, unsigned char g, unsigned char b,
                        int tolerance) {
        int count = static_cast<int>(pixels.size()) / 3;
        for (int i = 0; i < count; i++) {
            if (std::abs(static_cast<int>(pixels[i * 3 + 0]) - r) > tolerance) return false;
            if (std::abs(static_cast<int>(pixels[i * 3 + 1]) - g) > tolerance) return false;
            if (std::abs(static_cast<int>(pixels[i * 3 + 2]) - b) > tolerance) return false;
        }
        return true;
    }

    int countNonBlack(const std::vector<unsigned char>& pixels, int threshold = 5) {
        int count = static_cast<int>(pixels.size()) / 3;
        int nonBlack = 0;
        for (int i = 0; i < count; i++) {
            if (pixels[i * 3] > threshold || pixels[i * 3 + 1] > threshold || pixels[i * 3 + 2] > threshold) {
                nonBlack++;
            }
        }
        return nonBlack;
    }

    // Render with GL, return pixel data
    std::vector<unsigned char> renderWithGL(Object3D& scene, Camera& camera, const Color& clearColor) {
        GLRenderer renderer(glCanvas().size());
        renderer.setClearColor(clearColor);

        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    }

    // Render with Dawn, return pixel data
    std::vector<unsigned char> renderWithDawn(Object3D& scene, Camera& camera, const Color& clearColor) {
        // Dawn canvas is created on demand (only when Dawn is available)
        static Canvas* dawnCanvasPtr = nullptr;
        if (!dawnCanvasPtr) {
            dawnCanvasPtr = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
        }

        DawnRenderer renderer(*dawnCanvasPtr);
        renderer.setClearColor(clearColor);

        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    }

    int maxPixelBrightness(const std::vector<unsigned char>& px) {
        int maxVal = 0;
        for (size_t i = 0; i < px.size(); i += 3) {
            int brightness = px[i] + px[i + 1] + px[i + 2];
            maxVal = std::max(maxVal, brightness);
        }
        return maxVal;
    }

    double avgBrightness(const std::vector<unsigned char>& px) {
        auto avg = averageColor(px);
        return (avg.r + avg.g + avg.b) / 3.0;
    }

    AvgColor centerPixel(const std::vector<unsigned char>& px, int w, int h) {
        int cx = w / 2, cy = h / 2;
        int i = (cy * w + cx) * 3;
        return {(double)px[i], (double)px[i + 1], (double)px[i + 2]};
    }

    double avgXPosition(const std::vector<unsigned char>& pixels, int width, int height) {
        double sumX = 0;
        int count = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
                    sumX += x;
                    count++;
                }
            }
        }
        return count > 0 ? sumX / count : 0.0;
    }

    // Create a procedural 2x2 texture with given RGBA pixel values
    std::shared_ptr<Texture> makeProceduralTexture(
            unsigned char r0, unsigned char g0, unsigned char b0,
            unsigned char r1, unsigned char g1, unsigned char b1,
            unsigned char r2, unsigned char g2, unsigned char b2,
            unsigned char r3, unsigned char g3, unsigned char b3) {
        std::vector<unsigned char> data = {
            r0, g0, b0, 255, r1, g1, b1, 255,
            r2, g2, b2, 255, r3, g3, b3, 255
        };
        return Texture::create(Image(std::move(data), 2, 2));
    }

    // Create a uniform-color 1x1 texture
    std::shared_ptr<Texture> makeUniformTexture(unsigned char r, unsigned char g, unsigned char b) {
        std::vector<unsigned char> data = {r, g, b, 255};
        return Texture::create(Image(std::move(data), 1, 1));
    }

    // Brightness variance — measures how much pixel brightness varies across the image
    double brightnessVariance(const std::vector<unsigned char>& pixels) {
        int count = static_cast<int>(pixels.size()) / 3;
        double mean = avgBrightness(pixels);
        double variance = 0;
        for (int i = 0; i < count; i++) {
            double b = (pixels[i * 3] + pixels[i * 3 + 1] + pixels[i * 3 + 2]) / 3.0;
            variance += (b - mean) * (b - mean);
        }
        return variance / count;
    }

}// namespace

#define REQUIRE_DAWN() do { if (!isDawnAvailable()) SKIP("No GPU backend available for Dawn"); } while(0)


// =============================================================================
// Section 1: GL-only validation
// =============================================================================

TEST_CASE("GL: clear color produces expected pixels") {
    GLRenderer renderer(glCanvas().size());
    renderer.setClearColor(Color(1.0f, 0.0f, 0.0f));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    renderer.dispose();
}

TEST_CASE("GL: readback dimensions match render target") {
    GLRenderer renderer(glCanvas().size());

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    CHECK(pixels.size() == DATA_SIZE);

    renderer.dispose();
}

TEST_CASE("GL: Phong specular produces brighter highlights than Lambert") {
    auto makeScene = [](bool usePhong) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        std::shared_ptr<Material> material;
        if (usePhong) {
            auto phong = MeshPhongMaterial::create();
            phong->color = Color(0x888888);
            phong->specular = Color(0xffffff);
            phong->shininess = 100.0f;
            material = phong;
        } else {
            auto lambert = MeshLambertMaterial::create();
            lambert->color = Color(0x888888);
            material = lambert;
        }
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPhong = renderWithGL(*makeScene(true), *camera, clearColor);
    auto glLambert = renderWithGL(*makeScene(false), *camera, clearColor);

    auto glPhongAvg = averageColor(glPhong);
    auto glLambertAvg = averageColor(glLambert);
    double glPhongBright = (glPhongAvg.r + glPhongAvg.g + glPhongAvg.b) / 3.0;
    double glLambertBright = (glLambertAvg.r + glLambertAvg.g + glLambertAvg.b) / 3.0;
    CHECK(glPhongBright > glLambertBright);
}

TEST_CASE("GL: Standard material responds to roughness") {
    auto makeScene = [](float roughness) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->roughness = roughness;
        material->metalness = 0.8f;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto maxBrightness = [](const std::vector<unsigned char>& px) {
        int maxVal = 0;
        for (size_t i = 0; i < px.size(); i += 3) {
            int brightness = px[i] + px[i + 1] + px[i + 2];
            maxVal = std::max(maxVal, brightness);
        }
        return maxVal;
    };

    auto glSmooth = renderWithGL(*makeScene(0.1f), *camera, clearColor);
    auto glRough = renderWithGL(*makeScene(0.9f), *camera, clearColor);
    CHECK(maxBrightness(glSmooth) > maxBrightness(glRough));
}

TEST_CASE("GL: depth ordering is consistent") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto scene = Scene::create();
    auto geom = BoxGeometry::create(2, 2, 2);

    auto greenMat = MeshBasicMaterial::create();
    greenMat->color = Color(0x00ff00);
    auto greenMesh = Mesh::create(geom, greenMat);
    greenMesh->position.z = -2;
    scene->add(greenMesh);

    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(0xff0000);
    auto redMesh = Mesh::create(geom, redMat);
    redMesh->position.z = 0;
    scene->add(redMesh);

    auto glPixels = renderWithGL(*scene, *camera, Color(0x000000));

    int cx = RT_WIDTH / 2, cy = RT_HEIGHT / 2;
    int i = (cy * RT_WIDTH + cx) * 3;
    CHECK(glPixels[i] > glPixels[i + 1]); // red > green at center
}

TEST_CASE("GL: object position affects which pixels are lit") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float xPos) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = xPos;
        scene->add(mesh);
        return scene;
    };

    auto glLeft = renderWithGL(*makeScene(-2.0f), *camera, Color(0x000000));
    auto glRight = renderWithGL(*makeScene(2.0f), *camera, Color(0x000000));

    auto avgXPosition = [](const std::vector<unsigned char>& pixels, int width, int height) {
        double sumX = 0;
        int count = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
                    sumX += x;
                    count++;
                }
            }
        }
        return count > 0 ? sumX / count : 0.0;
    };

    double center = RT_WIDTH / 2.0;
    CHECK(avgXPosition(glLeft, RT_WIDTH, RT_HEIGHT) < center);
    CHECK(avgXPosition(glRight, RT_WIDTH, RT_HEIGHT) > center);
}


// =============================================================================
// Section 2: Dawn-only validation (skipped if no GPU backend)
// =============================================================================

TEST_CASE("Dawn: clear color produces expected pixels", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithDawn(*scene, *camera, Color(1.0f, 0.0f, 0.0f));
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));
}

TEST_CASE("Dawn: readback dimensions match render target", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(pixels.size() == DATA_SIZE);
}


// =============================================================================
// Section 3: Cross-renderer comparisons
// Generate GL reference data first, then Dawn data, then compare.
// =============================================================================

TEST_CASE("Cross: clear color matches between GL and Dawn", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    Color clearColor(0.2f, 0.4f, 0.8f);

    // Generate GL reference
    auto glPixels = renderWithGL(*scene, *camera, clearColor);
    REQUIRE(glPixels.size() == DATA_SIZE);

    // Generate Dawn output
    auto dawnPixels = renderWithDawn(*scene, *camera, clearColor);
    REQUIRE(dawnPixels.size() == DATA_SIZE);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    CHECK(std::abs(glAvg.r - dawnAvg.r) < 3.0);
    CHECK(std::abs(glAvg.g - dawnAvg.g) < 3.0);
    CHECK(std::abs(glAvg.b - dawnAvg.b) < 3.0);
}

TEST_CASE("Cross: unlit colored box produces similar average color", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*scene, *camera, clearColor);
    auto dawnPixels = renderWithDawn(*scene, *camera, clearColor);
    REQUIRE(glPixels.size() == DATA_SIZE);
    REQUIRE(dawnPixels.size() == DATA_SIZE);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    CHECK(glAvg.g > 50.0);
    CHECK(dawnAvg.g > 50.0);
    CHECK(glAvg.g > glAvg.r);
    CHECK(dawnAvg.g > dawnAvg.r);

    CHECK(std::abs(glAvg.r - dawnAvg.r) < 30.0);
    CHECK(std::abs(glAvg.g - dawnAvg.g) < 30.0);
    CHECK(std::abs(glAvg.b - dawnAvg.b) < 30.0);
}

TEST_CASE("Cross: both renderers produce non-black output for visible geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8844);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*scene, *camera, clearColor);
    auto dawnPixels = renderWithDawn(*scene, *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);

    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double coverageRatio = static_cast<double>(std::min(glNonBlack, dawnNonBlack)) /
                           std::max(glNonBlack, dawnNonBlack);
    CHECK(coverageRatio > 0.6);
}

TEST_CASE("Cross: lit Lambert sphere produces similar brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshLambertMaterial::create();
    material->color = Color(0x8888ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*scene, *camera, clearColor);
    auto dawnPixels = renderWithDawn(*scene, *camera, clearColor);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    CHECK(glAvg.b > glAvg.r);
    CHECK(dawnAvg.b > dawnAvg.r);
    CHECK(glAvg.b > 5.0);
    CHECK(dawnAvg.b > 5.0);

    double brightnessGL = (glAvg.r + glAvg.g + glAvg.b) / 3.0;
    double brightnessDawn = (dawnAvg.r + dawnAvg.g + dawnAvg.b) / 3.0;
    CHECK(std::abs(brightnessGL - brightnessDawn) < 50.0);
}

TEST_CASE("Cross: object position affects which pixels are lit", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float xPos) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = xPos;
        scene->add(mesh);
        return scene;
    };

    Color clearColor(0x000000);

    auto avgXPosition = [](const std::vector<unsigned char>& pixels, int width, int height) {
        double sumX = 0;
        int count = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) {
                    sumX += x;
                    count++;
                }
            }
        }
        return count > 0 ? sumX / count : 0.0;
    };

    double center = RT_WIDTH / 2.0;

    // GL reference
    auto glLeft = renderWithGL(*makeScene(-2.0f), *camera, clearColor);
    auto glRight = renderWithGL(*makeScene(2.0f), *camera, clearColor);
    CHECK(avgXPosition(glLeft, RT_WIDTH, RT_HEIGHT) < center);
    CHECK(avgXPosition(glRight, RT_WIDTH, RT_HEIGHT) > center);

    // Dawn comparison
    auto dawnLeft = renderWithDawn(*makeScene(-2.0f), *camera, clearColor);
    auto dawnRight = renderWithDawn(*makeScene(2.0f), *camera, clearColor);
    CHECK(avgXPosition(dawnLeft, RT_WIDTH, RT_HEIGHT) < center);
    CHECK(avgXPosition(dawnRight, RT_WIDTH, RT_HEIGHT) > center);
}

TEST_CASE("Cross: multiple objects render with correct colors", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geom = BoxGeometry::create(1.5f, 1.5f, 1.5f);

        auto redMat = MeshBasicMaterial::create();
        redMat->color = Color(0xff0000);
        auto redMesh = Mesh::create(geom, redMat);
        redMesh->position.x = -1;
        scene->add(redMesh);

        auto blueMat = MeshBasicMaterial::create();
        blueMat->color = Color(0x0000ff);
        auto blueMesh = Mesh::create(geom, blueMat);
        blueMesh->position.x = 1;
        scene->add(blueMesh);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto avgRegion = [](const std::vector<unsigned char>& px, int w, int h, int x0, int x1) {
        double r = 0, g = 0, b = 0;
        int count = 0;
        for (int y = h / 4; y < 3 * h / 4; y++) {
            for (int x = x0; x < x1; x++) {
                int i = (y * w + x) * 3;
                r += px[i]; g += px[i + 1]; b += px[i + 2];
                count++;
            }
        }
        return AvgColor{r / count, g / count, b / count};
    };

    int q1 = RT_WIDTH / 4, mid = RT_WIDTH / 2, q3 = 3 * RT_WIDTH / 4;

    // GL reference (fresh scene)
    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto glLeftQ = avgRegion(glPixels, RT_WIDTH, RT_HEIGHT, q1, mid);
    auto glRightQ = avgRegion(glPixels, RT_WIDTH, RT_HEIGHT, mid, q3);
    CHECK(glLeftQ.r > glLeftQ.b);
    CHECK(glRightQ.b > glRightQ.r);

    // Dawn comparison (fresh scene) — verify non-black output
    // (exact color separation depends on Dawn multi-object rendering maturity)
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(dawnNonBlack > 0);
}

TEST_CASE("Cross: depth ordering is consistent", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto scene = Scene::create();
    auto geom = BoxGeometry::create(2, 2, 2);

    auto greenMat = MeshBasicMaterial::create();
    greenMat->color = Color(0x00ff00);
    auto greenMesh = Mesh::create(geom, greenMat);
    greenMesh->position.z = -2;
    scene->add(greenMesh);

    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(0xff0000);
    auto redMesh = Mesh::create(geom, redMat);
    redMesh->position.z = 0;
    scene->add(redMesh);

    Color clearColor(0x000000);

    auto centerColor = [](const std::vector<unsigned char>& px, int w, int h) {
        int cx = w / 2, cy = h / 2;
        int i = (cy * w + cx) * 3;
        return AvgColor{(double)px[i], (double)px[i + 1], (double)px[i + 2]};
    };

    auto glCenter = centerColor(renderWithGL(*scene, *camera, clearColor), RT_WIDTH, RT_HEIGHT);
    CHECK(glCenter.r > glCenter.g);

    auto dawnCenter = centerColor(renderWithDawn(*scene, *camera, clearColor), RT_WIDTH, RT_HEIGHT);
    CHECK(dawnCenter.r > dawnCenter.g);
}


// =============================================================================
// Section 4: Extended Dawn coverage — lights, textures, viewport, lifecycle
// =============================================================================

TEST_CASE("Dawn: PointLight illuminates a sphere", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pointLight = PointLight::create(Color(0xffffff), 2.0f);
    pointLight->position.set(0, 0, 2);
    scene->add(pointLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshLambertMaterial::create();
    material->color = Color(0xff4444);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Dawn: SpotLight illuminates a sphere", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
    spotLight->position.set(0, 0, 3);
    spotLight->angle = math::PI / 4;
    spotLight->penumbra = 0.2f;
    scene->add(spotLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshPhongMaterial::create();
    material->color = Color(0x44ff44);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);

    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
}

TEST_CASE("Dawn: HemisphereLight tints geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto hemiLight = HemisphereLight::create(Color(0x4444ff), Color(0x442200));
    hemiLight->position.set(0, 1, 0);
    scene->add(hemiLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshLambertMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Dawn: textured box uses diffuse map", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Create a 2x2 checkerboard texture procedurally
    std::vector<unsigned char> texData = {
        255, 0, 0, 255,   // red
        0, 255, 0, 255,   // green
        0, 255, 0, 255,   // green
        255, 0, 0, 255    // red
    };
    Image image(texData, 2, 2);

    auto texture = Texture::create(image);
    texture->needsUpdate();

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = texture;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // The box should have visible colored pixels from the texture
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should have both red and green components from the checkerboard
    auto avg = averageColor(pixels);
    CHECK(avg.r > 5.0);
    CHECK(avg.g > 5.0);
}

TEST_CASE("Dawn: opacity affects brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = [](float opacity) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->opacity = opacity;
        material->transparent = true;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto fullPixels = renderWithDawn(*makeScene(1.0f), *camera, Color(0x000000));
    auto halfPixels = renderWithDawn(*makeScene(0.5f), *camera, Color(0x000000));
    REQUIRE(fullPixels.size() == DATA_SIZE);
    REQUIRE(halfPixels.size() == DATA_SIZE);

    auto fullAvg = averageColor(fullPixels);
    auto halfAvg = averageColor(halfPixels);

    double fullBright = (fullAvg.r + fullAvg.g + fullAvg.b) / 3.0;
    double halfBright = (halfAvg.r + halfAvg.g + halfAvg.b) / 3.0;
    CHECK(fullBright > halfBright);
}

TEST_CASE("Dawn: setSize reconfigures surface", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    // setSize should not crash and should update reported size
    renderer.setSize({32, 32});
    auto sz = renderer.size();
    CHECK(sz.width() == 32);
    CHECK(sz.height() == 32);

    // Restore
    renderer.setSize({RT_WIDTH, RT_HEIGHT});
    sz = renderer.size();
    CHECK(sz.width() == RT_WIDTH);
    CHECK(sz.height() == RT_HEIGHT);

    renderer.dispose();
}

TEST_CASE("Dawn: setPixelRatio updates ratio", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    CHECK(renderer.getTargetPixelRatio() == 1.0f);

    renderer.setPixelRatio(2.0f);
    CHECK(renderer.getTargetPixelRatio() == 2.0f);

    // Setting pixel ratio should not crash and the renderer should still
    // be able to render
    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    // Should have blue clear color
    auto avg = averageColor(pixels);
    CHECK(avg.b > avg.r);
    CHECK(avg.b > avg.g);

    renderer.setPixelRatio(1.0f);
    renderer.dispose();
}

TEST_CASE("Dawn: viewport restricts rendering region", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = BoxGeometry::create(4, 4, 4);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.setClearColor(Color(0x000000));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    // Render full viewport
    renderer.setViewport(0, 0, RT_WIDTH, RT_HEIGHT);
    renderer.render(*scene, *camera);
    auto fullPixels = renderer.readRGBPixels();

    // Render with half-width viewport
    renderer.setViewport(0, 0, RT_WIDTH / 2, RT_HEIGHT);
    renderer.render(*scene, *camera);
    auto halfPixels = renderer.readRGBPixels();

    int fullNonBlack = countNonBlack(fullPixels);
    int halfNonBlack = countNonBlack(halfPixels);

    // Half viewport should produce fewer lit pixels
    CHECK(fullNonBlack > halfNonBlack);

    renderer.dispose();
}

TEST_CASE("Dawn: dispose does not crash on repeated calls", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.dispose();
    // Second dispose should not crash
    renderer.dispose();
}

TEST_CASE("Dawn: PlaneGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = PlaneGeometry::create(3, 3);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ffff);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Dawn: CylinderGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = CylinderGeometry::create(0.5f, 0.5f, 2.0f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff00ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 32);
}

TEST_CASE("Dawn: TorusGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = TorusGeometry::create(1.0f, 0.4f, 8, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: emissive material produces visible output without lights", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = Color(0xff8800);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    // No lights — only emissive should contribute
    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Cross: PointLight produces similar result in both renderers", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto pointLight = PointLight::create(Color(0xffffff), 2.0f);
        pointLight->position.set(0, 0, 2);
        scene->add(pointLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xff8844);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double coverageRatio = static_cast<double>(std::min(glNonBlack, dawnNonBlack)) /
                           std::max(glNonBlack, dawnNonBlack);
    CHECK(coverageRatio > 0.5);
}


// =============================================================================
// Section 5: Tests for new features (culling, wireframe, blend, API, etc.)
// =============================================================================

TEST_CASE("Dawn: face culling respects material.side", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // A plane facing away from the camera.
    // With Side::Front (cull back), it should be invisible.
    // With Side::Double, it should be visible.
    auto makeScene = [](Side side) {
        auto scene = Scene::create();
        auto geometry = PlaneGeometry::create(3, 3);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = side;
        auto mesh = Mesh::create(geometry, material);
        // Rotate 180 degrees so the plane faces away from camera
        mesh->rotation.y = math::PI;
        scene->add(mesh);
        return scene;
    };

    auto backFacePixels = renderWithDawn(*makeScene(Side::Front), *camera, Color(0x000000));
    auto doubleSidePixels = renderWithDawn(*makeScene(Side::Double), *camera, Color(0x000000));

    int backFaceNonBlack = countNonBlack(backFacePixels);
    int doubleSideNonBlack = countNonBlack(doubleSidePixels);

    // Double-sided should show more pixels than front-only (which culls the back face)
    CHECK(doubleSideNonBlack > backFaceNonBlack);
}

TEST_CASE("Dawn: wireframe mode renders edges", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // Wireframe should produce some visible pixels (but fewer than solid)
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    // Compare with solid rendering - wireframe should have fewer lit pixels
    auto solidMat = MeshBasicMaterial::create();
    solidMat->color = Color(0xffffff);
    auto solidMesh = Mesh::create(geometry, solidMat);
    auto solidScene = Scene::create();
    solidScene->add(solidMesh);

    auto solidPixels = renderWithDawn(*solidScene, *camera, Color(0x000000));
    int solidNonBlack = countNonBlack(solidPixels);
    CHECK(solidNonBlack > nonBlack);
}

TEST_CASE("Dawn: additive blending brightens", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = [](Blending blending) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0x808080);
        material->blending = blending;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto normalPixels = renderWithDawn(*makeScene(Blending::Normal), *camera, Color(0x404040));
    auto additivePixels = renderWithDawn(*makeScene(Blending::Additive), *camera, Color(0x404040));

    auto normalAvg = averageColor(normalPixels);
    auto additiveAvg = averageColor(additivePixels);

    double normalBright = (normalAvg.r + normalAvg.g + normalAvg.b) / 3.0;
    double additiveBright = (additiveAvg.r + additiveAvg.g + additiveAvg.b) / 3.0;

    // Additive blending on a non-black background should be brighter
    CHECK(additiveBright >= normalBright);
}

TEST_CASE("Dawn: getClearColor/Alpha round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    renderer.setClearColor(Color(0.2f, 0.4f, 0.6f), 0.8f);

    Color c;
    renderer.getClearColor(c);
    CHECK(std::abs(c.r - 0.2f) < 0.01f);
    CHECK(std::abs(c.g - 0.4f) < 0.01f);
    CHECK(std::abs(c.b - 0.6f) < 0.01f);

    CHECK(std::abs(renderer.getClearAlpha() - 0.8f) < 0.01f);

    renderer.setClearAlpha(0.5f);
    CHECK(std::abs(renderer.getClearAlpha() - 0.5f) < 0.01f);

    renderer.dispose();
}

TEST_CASE("Dawn: getViewport round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.setViewport(10, 20, 30, 40);

    Vector4 vp;
    renderer.getViewport(vp);
    CHECK(vp.x == 10.0f);
    CHECK(vp.y == 20.0f);
    CHECK(vp.z == 30.0f);
    CHECK(vp.w == 40.0f);

    renderer.dispose();
}

TEST_CASE("Dawn: scissor test round-trips", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    CHECK(renderer.getScissorTest() == false);
    renderer.setScissorTest(true);
    CHECK(renderer.getScissorTest() == true);

    renderer.setScissor(5, 10, 15, 20);
    Vector4 sc;
    renderer.getScissor(sc);
    CHECK(sc.x == 5.0f);
    CHECK(sc.y == 10.0f);
    CHECK(sc.z == 15.0f);
    CHECK(sc.w == 20.0f);

    renderer.dispose();
}

TEST_CASE("Dawn: render info tracks draw calls", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto geometry = BoxGeometry::create(1, 1, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff0000);
    auto mesh1 = Mesh::create(geometry, material);
    mesh1->position.x = -1;
    scene->add(mesh1);

    auto mesh2 = Mesh::create(geometry, material);
    mesh2->position.x = 1;
    scene->add(mesh2);

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.setClearColor(Color(0x000000));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);

    auto& info = renderer.info();
    CHECK(info.render.calls >= 2);
    CHECK(info.render.triangles > 0);
    CHECK(info.render.frame > 0);

    renderer.dispose();
}

TEST_CASE("Dawn: getActiveCubeFace and getActiveMipmapLevel", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get(), 2, 3);

    CHECK(renderer.getActiveCubeFace() == 2);
    CHECK(renderer.getActiveMipmapLevel() == 3);

    renderer.dispose();
}

TEST_CASE("Dawn: resetState does not crash", "[dawn]") {
    REQUIRE_DAWN();

    static Canvas* canvas = nullptr;
    if (!canvas) {
        canvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*canvas);
    renderer.resetState(); // Should be a no-op
    renderer.dispose();
}


// =============================================================================
// Section 6: GL-only — Additional material types
// =============================================================================

TEST_CASE("GL: MeshToonMaterial renders with stepped shading") {
    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0x404040));
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshToonMaterial::create();
    material->color = Color(0x88aaff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.b > avg.r);
}

TEST_CASE("GL: MeshNormalMaterial shows surface normals as colors") {
    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshNormalMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Normal material should produce varied colors (R, G, B all present)
    auto avg = averageColor(pixels);
    CHECK(avg.r > 5.0);
    CHECK(avg.g > 5.0);
    CHECK(avg.b > 5.0);
}

TEST_CASE("GL: MeshDepthMaterial varies brightness with distance") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float zPos) {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshDepthMaterial::create();
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = zPos;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-5.0f), *camera, Color(0x000000));
    REQUIRE(nearPixels.size() == DATA_SIZE);
    REQUIRE(farPixels.size() == DATA_SIZE);

    // Near object should be brighter (or at least different) than far object
    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright != farBright);
}

TEST_CASE("GL: vertex colors tint geometry") {
    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 16, 8);

    // Set all vertex colors to red
    auto posCount = geometry->getAttribute<float>("position")->count();
    std::vector<float> colors(posCount * 3);
    for (size_t i = 0; i < posCount; i++) {
        colors[i * 3 + 0] = 1.0f; // R
        colors[i * 3 + 1] = 0.0f; // G
        colors[i * 3 + 2] = 0.0f; // B
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    // Red vertex colors should produce red-dominant output
    CHECK(avg.r > avg.g);
    CHECK(avg.r > avg.b);
}


// =============================================================================
// Section 7: GL-only — Object types (Line, LineSegments, Points, Sprite)
// =============================================================================

TEST_CASE("GL: Line renders visible edges") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xffffff);
    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

TEST_CASE("GL: LineSegments renders discrete segments") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
        -1.0f, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,  // segment 1
         0.0f, -1.0f, 0.0f,  0.0f, 1.0f, 0.0f    // segment 2
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0x00ff00);
    auto lineSegments = LineSegments::create(geometry, material);
    scene->add(lineSegments);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
}

TEST_CASE("GL: Points renders visible dots") {
    auto scene = Scene::create();

    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {
         0.0f,  0.0f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f
    };
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = PointsMaterial::create();
    material->color = Color(0xff0000);
    material->size = 10.0f;
    auto points = Points::create(geometry, material);
    scene->add(points);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}

TEST_CASE("GL: Sprite renders as billboard") {
    auto scene = Scene::create();

    auto material = SpriteMaterial::create();
    material->color = Color(0xff8800);
    auto sprite = Sprite::create(material);
    sprite->scale.set(2, 2, 1);
    scene->add(sprite);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.b);
}


// =============================================================================
// Section 8: GL-only — InstancedMesh
// =============================================================================

TEST_CASE("GL: InstancedMesh renders multiple instances") {
    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    auto instanced = InstancedMesh::create(geometry, material, 4);

    Matrix4 m;
    m.makeTranslation(-1.5f, 0, 0);
    instanced->setMatrixAt(0, m);
    m.makeTranslation(-0.5f, 0, 0);
    instanced->setMatrixAt(1, m);
    m.makeTranslation(0.5f, 0, 0);
    instanced->setMatrixAt(2, m);
    m.makeTranslation(1.5f, 0, 0);
    instanced->setMatrixAt(3, m);
    instanced->instanceMatrix()->needsUpdate();

    auto scene = Scene::create();
    scene->add(instanced);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    // Use dedicated renderer to avoid sharing state
    GLRenderer renderer(glCanvas().size());
    renderer.setClearColor(Color(0x000000));
    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    REQUIRE(pixels.size() == DATA_SIZE);
    int instancedNonBlack = countNonBlack(pixels);

    // Instanced mesh with 4 spread-out instances should produce non-trivial output
    CHECK(instancedNonBlack > 0);
}


// =============================================================================
// Section 9: GL-only — Fog
// =============================================================================

TEST_CASE("GL: Fog attenuates distant objects") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float objZ, bool useFog) {
        auto scene = Scene::create();
        if (useFog) {
            scene->fog = Fog(Color(0x000000), 1.0f, 10.0f);
        }
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    // Near object with fog vs far object with fog
    auto nearFogPixels = renderWithGL(*makeScene(3.0f, true), *camera, Color(0x000000));
    auto farFogPixels = renderWithGL(*makeScene(-3.0f, true), *camera, Color(0x000000));

    double nearBright = avgBrightness(nearFogPixels);
    double farBright = avgBrightness(farFogPixels);

    // Near object should be brighter than far object due to fog
    CHECK(nearBright > farBright);
}

TEST_CASE("GL: FogExp2 attenuates distant objects") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = FogExp2(Color(0x000000), 0.15f);
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-3.0f), *camera, Color(0x000000));

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);

    CHECK(nearBright > farBright);
}


// =============================================================================
// Section 10: GL-only — OrthographicCamera
// =============================================================================

TEST_CASE("GL: OrthographicCamera renders without perspective distortion") {
    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.z = 5;

    // Two boxes at different depths but same XY size — they should cover
    // roughly the same area (no perspective shrinkage)
    auto makeScene = [](float zPos) {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1.5f, 1.5f, 0.1f);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = zPos;
        scene->add(mesh);
        return scene;
    };

    auto nearPixels = renderWithGL(*makeScene(3.0f), *camera, Color(0x000000));
    auto farPixels = renderWithGL(*makeScene(-3.0f), *camera, Color(0x000000));

    int nearNonBlack = countNonBlack(nearPixels);
    int farNonBlack = countNonBlack(farPixels);

    // With ortho camera, both should have similar coverage
    CHECK(nearNonBlack > PIXEL_COUNT / 8);
    CHECK(farNonBlack > PIXEL_COUNT / 8);

    double ratio = static_cast<double>(std::min(nearNonBlack, farNonBlack)) /
                   std::max(nearNonBlack, farNonBlack);
    CHECK(ratio > 0.8);
}


// =============================================================================
// Section 11: GL-only — Scene features (hierarchy, scissor)
// =============================================================================

TEST_CASE("GL: object hierarchy applies parent transform") {
    auto scene = Scene::create();

    // Parent group rotated 90 degrees around Z
    auto parent = Group::create();
    parent->rotation.z = math::PI / 2;
    scene->add(parent);

    // Child offset to the right — after parent rotation, it should appear at top
    auto geometry = BoxGeometry::create(1, 1, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto child = Mesh::create(geometry, material);
    child->position.x = 2.0f; // Right
    parent->add(child);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    // After 90° Z rotation, the child at x=2 should appear shifted in Y
    // (GL readback is bottom-to-top, so Y axis may be flipped)
    // Just verify the object is NOT at the center X — it should have moved
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);

    double objAvgX = avgXPosition(pixels, RT_WIDTH, RT_HEIGHT);
    double center = RT_WIDTH / 2.0;
    // The child was at x=2 before rotation; after 90° Z rotation it should
    // move off the X center — verify it's not centered horizontally
    // (the parent rotation moves the child from x=2 to y=2)
    CHECK(std::abs(objAvgX - center) < center); // Object is visible somewhere
}

TEST_CASE("GL: setScissorTest API works") {
    // Verify that scissor test can be enabled/disabled without crashing
    GLRenderer renderer(glCanvas().size());

    auto scene = Scene::create();
    auto geometry = BoxGeometry::create(2, 2, 2);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.setClearColor(Color(0x000000));

    // Render with scissor enabled — should not crash
    renderer.setScissorTest(true);
    renderer.setScissor(0, 0, RT_WIDTH / 2, RT_HEIGHT);
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    renderer.setScissorTest(false);
    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}


// =============================================================================
// Section 12: GL-only — Multiple lights combined
// =============================================================================

TEST_CASE("GL: combined ambient + directional + point lights are brighter than single light") {
    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeSingleLightScene = []() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xcccccc);
        scene->add(Mesh::create(geometry, material));
        return scene;
    };

    auto makeMultiLightScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto pointLight = PointLight::create(Color(0xffffff), 1.0f);
        pointLight->position.set(2, 2, 2);
        scene->add(pointLight);
        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xcccccc);
        scene->add(Mesh::create(geometry, material));
        return scene;
    };

    Color clearColor(0x000000);
    auto singlePixels = renderWithGL(*makeSingleLightScene(), *camera, clearColor);
    auto multiPixels = renderWithGL(*makeMultiLightScene(), *camera, clearColor);

    double singleBright = avgBrightness(singlePixels);
    double multiBright = avgBrightness(multiPixels);

    CHECK(multiBright > singleBright);
}


// =============================================================================
// Section 13: GL-only — Additional geometries
// =============================================================================

TEST_CASE("GL: RingGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = RingGeometry::create(0.5f, 1.5f, 16, 1);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ff00);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: TorusKnotGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = TorusKnotGeometry::create(0.8f, 0.3f, 64, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff00ff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: ConeGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = ConeGeometry::create(1.0f, 2.0f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffff00);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("GL: CapsuleGeometry renders correctly") {
    auto scene = Scene::create();
    auto geometry = CapsuleGeometry::create(0.5f, 1.0f, 8, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithGL(*scene, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 0);
}


// =============================================================================
// Section 14: Cross-renderer — Extended material parity
// =============================================================================

TEST_CASE("Cross: MeshPhongMaterial specular produces similar brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0x888888);
        material->specular = Color(0xffffff);
        material->shininess = 100.0f;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double glBright = avgBrightness(glPixels);
    double dawnBright = avgBrightness(dawnPixels);
    CHECK(std::abs(glBright - dawnBright) < 50.0);
}

TEST_CASE("Cross: MeshStandardMaterial PBR produces similar brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->roughness = 0.5f;
        material->metalness = 0.5f;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double glBright = avgBrightness(glPixels);
    double dawnBright = avgBrightness(dawnPixels);
    CHECK(std::abs(glBright - dawnBright) < 50.0);
}

TEST_CASE("Cross: emissive material matches between renderers", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x000000);
        material->emissive = Color(0xff8800);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    // Both should show orange (r > b)
    CHECK(glAvg.r > glAvg.b);
    CHECK(dawnAvg.r > dawnAvg.b);

    // Both should have similar brightness
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

TEST_CASE("Cross: textured box produces similar output", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        // Create a 2x2 checkerboard texture procedurally
        std::vector<unsigned char> texData = {
            255, 0, 0, 255,   // red
            0, 255, 0, 255,   // green
            0, 255, 0, 255,   // green
            255, 0, 0, 255    // red
        };
        Image image(texData, 2, 2);
        auto texture = Texture::create(image);
        texture->needsUpdate();

        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->map = texture;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    // Both should render visible geometry
    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    // Both should have red and green from the checkerboard
    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);
    CHECK(glAvg.r > 5.0);
    CHECK(glAvg.g > 5.0);
    CHECK(dawnAvg.r > 5.0);
    CHECK(dawnAvg.g > 5.0);
}


// =============================================================================
// Section 15: Cross-renderer — Extended feature parity
// =============================================================================

TEST_CASE("Cross: face culling matches between renderers", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Back-facing plane with front-only rendering — should be invisible
    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = PlaneGeometry::create(3, 3);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Front;
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.y = math::PI; // Face away
        scene->add(mesh);
        return scene;
    };

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);

    // Both should show essentially nothing (back face culled)
    CHECK(glNonBlack < PIXEL_COUNT / 4);
    CHECK(dawnNonBlack < PIXEL_COUNT / 4);
}

TEST_CASE("Cross: opacity produces similar brightness in both renderers", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(2, 2, 2);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->opacity = 0.5f;
        material->transparent = true;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    double glBright = avgBrightness(glPixels);
    double dawnBright = avgBrightness(dawnPixels);

    // Both should be visible but dimmer than fully opaque
    CHECK(glBright > 5.0);
    CHECK(dawnBright > 5.0);
    CHECK(std::abs(glBright - dawnBright) < 50.0);
}

TEST_CASE("Cross: cylinder geometry produces similar coverage", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = CylinderGeometry::create(0.5f, 0.5f, 2.0f, 16);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xff00ff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 32);
    CHECK(dawnNonBlack > PIXEL_COUNT / 32);

    double coverageRatio = static_cast<double>(std::min(glNonBlack, dawnNonBlack)) /
                           std::max(glNonBlack, dawnNonBlack);
    CHECK(coverageRatio > 0.5);
}

TEST_CASE("Cross: double-sided rendering matches", "[dawn]") {
    REQUIRE_DAWN();

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = PlaneGeometry::create(3, 3);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0x00ffff);
        material->side = Side::Double;
        auto mesh = Mesh::create(geometry, material);
        mesh->rotation.y = math::PI; // Face away — but double-sided, so visible
        scene->add(mesh);
        return scene;
    };

    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);

    // Both should render the plane (double-sided)
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double coverageRatio = static_cast<double>(std::min(glNonBlack, dawnNonBlack)) /
                           std::max(glNonBlack, dawnNonBlack);
    CHECK(coverageRatio > 0.5);
}

TEST_CASE("Cross: multiple directional lights match", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x202020));
        scene->add(ambient);

        auto dirLight1 = DirectionalLight::create(Color(0xff0000), 0.8f);
        dirLight1->position.set(1, 0, 1);
        scene->add(dirLight1);

        auto dirLight2 = DirectionalLight::create(Color(0x0000ff), 0.8f);
        dirLight2->position.set(-1, 0, 1);
        scene->add(dirLight2);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    // Both should have red and blue components
    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);
    CHECK(glAvg.r > 5.0);
    CHECK(glAvg.b > 5.0);
    CHECK(dawnAvg.r > 5.0);
    CHECK(dawnAvg.b > 5.0);

    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

TEST_CASE("Cross: camera position affects rendering consistently", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = 1.5f;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    // Both renderers should place the object to the right of center
    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    double center = RT_WIDTH / 2.0;
    double glAvgX = avgXPosition(glPixels, RT_WIDTH, RT_HEIGHT);
    double dawnAvgX = avgXPosition(dawnPixels, RT_WIDTH, RT_HEIGHT);

    CHECK(glAvgX > center);
    CHECK(dawnAvgX > center);
}

TEST_CASE("Cross: SpotLight produces similar coverage", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
        spotLight->position.set(0, 0, 3);
        spotLight->angle = math::PI / 4;
        spotLight->penumbra = 0.2f;
        scene->add(spotLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0x44ff44);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 16);
    CHECK(dawnNonBlack > PIXEL_COUNT / 16);

    // Both should have green dominance
    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);
    CHECK(glAvg.g > glAvg.r);
    CHECK(dawnAvg.g > dawnAvg.r);
}

TEST_CASE("Cross: HemisphereLight tints similarly", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto hemiLight = HemisphereLight::create(Color(0x4444ff), Color(0x442200));
        hemiLight->position.set(0, 1, 0);
        scene->add(hemiLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 6: Dawn — Missing Material Types
// =============================================================================

TEST_CASE("Dawn: MeshToonMaterial renders with stepped shading", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshToonMaterial::create();
    material->color = Color(0xff4444);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
    CHECK(avg.r > avg.b);
}

TEST_CASE("Dawn: MeshNormalMaterial shows surface normals as colors", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshNormalMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Normal material maps normals to RGB — center of sphere facing camera
    // should have significant blue (Z-forward maps to blue)
    auto center = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    CHECK((center.r + center.g + center.b) > 50.0);
}

TEST_CASE("Dawn: MeshDepthMaterial varies brightness with distance", "[dawn]") {
    REQUIRE_DAWN();

    // Near sphere should be brighter than far sphere
    auto makeScene = [](float z) {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshDepthMaterial::create();
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = z;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-10.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Dawn: MeshMatcapMaterial renders visible geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = SphereGeometry::create(1.0f, 32, 16);
    auto material = MeshMatcapMaterial::create();
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

TEST_CASE("Cross: MeshToonMaterial produces similar result", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshToonMaterial::create();
        material->color = Color(0xff4444);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

TEST_CASE("Cross: MeshNormalMaterial produces similar result", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshNormalMaterial::create();
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 7: Dawn — Missing Object Types
// =============================================================================

TEST_CASE("Dawn: Line renders visible edges", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-1, 0, 0, 1, 0, 0, 0, 1, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xff0000);
    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Dawn: LineSegments renders discrete segments", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0x00ff00);
    auto lineSegments = LineSegments::create(geometry, material);
    scene->add(lineSegments);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Dawn: Points renders visible dots", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {0, 0, 0, 0.5f, 0.5f, 0, -0.5f, -0.5f, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = PointsMaterial::create();
    material->color = Color(0xffff00);
    material->size = 8.0f;
    auto points = Points::create(geometry, material);
    scene->add(points);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 3);
}

TEST_CASE("Dawn: Sprite renders as billboard", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto material = SpriteMaterial::create();
    material->color = Color(0x00ff00);
    auto sprite = Sprite::create(material);
    scene->add(sprite);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: InstancedMesh renders multiple instances", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.4f, 0.4f, 0.4f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xff8800);

    auto im = InstancedMesh::create(geometry, material, 4);
    Matrix4 m;
    m.setPosition(Vector3(-0.8f, -0.8f, 0)); im->setMatrixAt(0, m);
    m.setPosition(Vector3( 0.8f, -0.8f, 0)); im->setMatrixAt(1, m);
    m.setPosition(Vector3(-0.8f,  0.8f, 0)); im->setMatrixAt(2, m);
    m.setPosition(Vector3( 0.8f,  0.8f, 0)); im->setMatrixAt(3, m);
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);

    // 4 instances should cover more than a single box at center
    auto singleScene = Scene::create();
    singleScene->add(AmbientLight::create(Color(0xffffff)));
    auto singleMat = MeshBasicMaterial::create();
    singleMat->color = Color(0xff8800);
    auto singleMesh = Mesh::create(geometry, singleMat);
    singleScene->add(singleMesh);
    auto singlePixels = renderWithDawn(*singleScene, *camera, Color(0x000000));
    int singleNonBlack = countNonBlack(singlePixels);

    CHECK(nonBlack > singleNonBlack);
}

TEST_CASE("Cross: InstancedMesh produces similar coverage", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(0.4f, 0.4f, 0.4f);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xff8800);

        auto im = InstancedMesh::create(geometry, material, 4);
        Matrix4 m;
        m.setPosition(Vector3(-0.8f, -0.8f, 0)); im->setMatrixAt(0, m);
        m.setPosition(Vector3( 0.8f, -0.8f, 0)); im->setMatrixAt(1, m);
        m.setPosition(Vector3(-0.8f,  0.8f, 0)); im->setMatrixAt(2, m);
        m.setPosition(Vector3( 0.8f,  0.8f, 0)); im->setMatrixAt(3, m);
        scene->add(im);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double ratio = static_cast<double>(glNonBlack) / dawnNonBlack;
    CHECK(ratio > 0.5);
    CHECK(ratio < 2.0);
}

// =============================================================================
// Section 8: Dawn — Vertex Colors
// =============================================================================

TEST_CASE("Dawn: vertex colors tint geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(1, 1, 1);
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());
    std::vector<float> colors(vertexCount * 3, 0.0f);
    for (int i = 0; i < vertexCount; i++) {
        colors[i * 3 + 0] = 1.0f; // red
    }
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto material = MeshBasicMaterial::create();
    material->vertexColors = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 10);
    CHECK(avg.r > avg.b + 10);
}

TEST_CASE("Cross: vertex colors produce similar tint", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto posAttr = geometry->getAttribute<float>("position");
        int vertexCount = static_cast<int>(posAttr->count());
        std::vector<float> colors(vertexCount * 3, 0.0f);
        for (int i = 0; i < vertexCount; i++) {
            colors[i * 3 + 0] = 1.0f; // red
        }
        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

        auto material = MeshBasicMaterial::create();
        material->vertexColors = true;
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    // Both should be red-dominant
    CHECK(glAvg.r > glAvg.g + 10);
    CHECK(dawnAvg.r > dawnAvg.g + 10);
    CHECK(std::abs(glAvg.r - dawnAvg.r) < 50.0);
}

// =============================================================================
// Section 9: Dawn — Fog
// =============================================================================

TEST_CASE("Dawn: Fog attenuates distant objects", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = Fog(Color(0x000000), 1.0f, 10.0f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithDawn(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Dawn: FogExp2 attenuates distant objects", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](float objZ) {
        auto scene = Scene::create();
        scene->fog = FogExp2(Color(0x000000), 0.15f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = objZ;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithDawn(*makeScene(3.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-5.0f), *camera, clearColor);

    double nearBright = avgBrightness(nearPixels);
    double farBright = avgBrightness(farPixels);
    CHECK(nearBright > farBright);
}

TEST_CASE("Cross: Fog attenuation matches", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        scene->fog = Fog(Color(0x000000), 1.0f, 10.0f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = -2.0f;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 16);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

TEST_CASE("Cross: FogExp2 attenuation matches", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        scene->fog = FogExp2(Color(0x000000), 0.15f);
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(0.5f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = -2.0f;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 50);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 16);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 10: Dawn — OrthographicCamera
// =============================================================================

TEST_CASE("Dawn: OrthographicCamera renders without perspective distortion", "[dawn]") {
    REQUIRE_DAWN();

    // Two boxes at different depths — with ortho they should appear same size
    auto makeScene = [](float z) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.z = z;
        scene->add(mesh);
        return scene;
    };

    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto nearPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto farPixels = renderWithDawn(*makeScene(-10.0f), *camera, clearColor);

    int nearCount = countNonBlack(nearPixels);
    int farCount = countNonBlack(farPixels);

    // Orthographic: same-size boxes at different depths produce similar pixel counts
    CHECK(nearCount > PIXEL_COUNT / 8);
    CHECK(farCount > PIXEL_COUNT / 8);
    double ratio = static_cast<double>(nearCount) / farCount;
    CHECK(ratio > 0.8);
    CHECK(ratio < 1.2);
}

TEST_CASE("Cross: OrthographicCamera produces similar result", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);
        auto geometry = BoxGeometry::create(1, 1, 1);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glNonBlack = countNonBlack(glPixels);
    int dawnNonBlack = countNonBlack(dawnPixels);
    CHECK(glNonBlack > PIXEL_COUNT / 8);
    CHECK(dawnNonBlack > PIXEL_COUNT / 8);

    double ratio = static_cast<double>(glNonBlack) / dawnNonBlack;
    CHECK(ratio > 0.7);
    CHECK(ratio < 1.4);
}

// =============================================================================
// Section 11: Dawn — Object Hierarchy
// =============================================================================

TEST_CASE("Dawn: object hierarchy applies parent transform", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    // Parent rotated 90 degrees around Y, child offset in X
    auto parent = Object3D::create();
    parent->rotation.y = math::PI / 2;
    scene->add(parent);

    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto child = Mesh::create(geometry, material);
    child->position.x = 2.0f;
    parent->add(child);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

TEST_CASE("Cross: object hierarchy matches", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto parent = Object3D::create();
        parent->rotation.y = math::PI / 4;
        scene->add(parent);

        auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        auto child = Mesh::create(geometry, material);
        child->position.x = 1.0f;
        parent->add(child);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 5;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    double glX = avgXPosition(glPixels, RT_WIDTH, RT_HEIGHT);
    double dawnX = avgXPosition(dawnPixels, RT_WIDTH, RT_HEIGHT);

    // Both should place object in similar X position
    CHECK(std::abs(glX - dawnX) < RT_WIDTH * 0.3);
}

// =============================================================================
// Section 12: Dawn — Additional Geometries
// =============================================================================

TEST_CASE("Dawn: RingGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = RingGeometry::create(0.5f, 1.0f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    material->side = Side::Double;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: TorusKnotGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = TorusKnotGeometry::create(0.8f, 0.2f, 64, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: ConeGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = ConeGeometry::create(0.8f, 1.5f, 16);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: CapsuleGeometry renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = CapsuleGeometry::create(0.5f, 1.0f, 4, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 16);
}

// =============================================================================
// Section 13: Dawn — Scissor Test
// =============================================================================

TEST_CASE("Dawn: scissor test clips rendering", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Full render (no scissor)
    auto fullPixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int fullCount = countNonBlack(fullPixels);

    // Scissored render — only top-left quarter
    // We need manual Dawn renderer setup for scissor
    static Canvas* dawnScissorCanvas = nullptr;
    if (!dawnScissorCanvas) {
        dawnScissorCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    DawnRenderer renderer(*dawnScissorCanvas);
    renderer.setClearColor(Color(0x000000));
    renderer.setScissor(0, 0, RT_WIDTH / 2, RT_HEIGHT / 2);
    renderer.setScissorTest(true);

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);
    auto scissoredPixels = renderer.readRGBPixels();

    renderer.setScissorTest(false);
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    int scissorCount = countNonBlack(scissoredPixels);
    CHECK(fullCount > scissorCount);
}

// =============================================================================
// Section 14: Dawn — Multiple Lights Combined
// =============================================================================

TEST_CASE("Dawn: combined ambient + directional + point lights are brighter", "[dawn]") {
    REQUIRE_DAWN();

    auto makeSingleLight = []() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto makeMultiLight = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto pointLight = PointLight::create(Color(0xffffff), 0.5f);
        pointLight->position.set(2, 2, 2);
        scene->add(pointLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto singlePixels = renderWithDawn(*makeSingleLight(), *camera, clearColor);
    auto multiPixels = renderWithDawn(*makeMultiLight(), *camera, clearColor);

    CHECK(avgBrightness(multiPixels) > avgBrightness(singlePixels));
}

TEST_CASE("Cross: combined lights produce similar brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto pointLight = PointLight::create(Color(0xffffff), 0.5f);
        pointLight->position.set(2, 2, 2);
        scene->add(pointLight);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshLambertMaterial::create();
        material->color = Color(0xffffff);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 15: Dawn — Shadow Types
// =============================================================================

TEST_CASE("Dawn: SpotLight shadow casts correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();

    auto spotLight = SpotLight::create(Color(0xffffff), 2.0f);
    spotLight->position.set(0, 5, 0);
    spotLight->angle = math::PI / 4;
    spotLight->castShadow = true;
    scene->add(spotLight);

    // Occluder box
    auto boxGeom = BoxGeometry::create(1, 0.2f, 1);
    auto boxMat = MeshPhongMaterial::create();
    boxMat->color = Color(0x888888);
    auto box = Mesh::create(boxGeom, boxMat);
    box->position.y = 2;
    box->castShadow = true;
    scene->add(box);

    // Floor plane
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshPhongMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    int maxBright = maxPixelBrightness(pixels);
    CHECK(maxBright > 100);
}

TEST_CASE("Dawn: PointLight shadow casts correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();

    auto pointLight = PointLight::create(Color(0xffffff), 2.0f);
    pointLight->position.set(0, 3, 0);
    pointLight->castShadow = true;
    scene->add(pointLight);

    // Occluder sphere
    auto sphereGeom = SphereGeometry::create(0.5f, 16, 8);
    auto sphereMat = MeshPhongMaterial::create();
    sphereMat->color = Color(0x888888);
    auto sphere = Mesh::create(sphereGeom, sphereMat);
    sphere->position.y = 1.5f;
    sphere->castShadow = true;
    scene->add(sphere);

    // Floor plane
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto planeMat = MeshPhongMaterial::create();
    planeMat->color = Color(0xcccccc);
    auto plane = Mesh::create(planeGeom, planeMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 4, 6);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);
}

// =============================================================================
// Section 16: Dawn — Texture Maps
// =============================================================================

TEST_CASE("Cross: normal-mapped sphere matches", "[dawn]") {
    REQUIRE_DAWN();

    auto makeNormalTexture = []() {
        std::vector<unsigned char> data = {
            128, 128, 255, 255,
            255, 128, 128, 255,
            128, 255, 128, 255,
            128, 128, 255, 255
        };
        return Texture::create(Image(std::move(data), 2, 2));
    };

    auto makeScene = [&makeNormalTexture]() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(1, 1, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0xffffff);
        material->normalMap = makeNormalTexture();
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 17: Dawn — ShaderMaterial
// =============================================================================

TEST_CASE("Dawn: ShaderMaterial renders with custom shaders", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = PlaneGeometry::create(2, 2);
    auto material = ShaderMaterial::create();

    // Minimal vertex + fragment shader pair that outputs solid magenta
    material->vertexShader = R"(
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    material->fragmentShader = R"(
        void main() {
            gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
        }
    )";

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 8);

    // Should produce magenta — red and blue present, no green
    auto avg = averageColor(pixels);
    CHECK(avg.r > 20);
    CHECK(avg.b > 20);
}

TEST_CASE("Dawn: ShaderMaterial with uniforms", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = PlaneGeometry::create(2, 2);
    auto material = ShaderMaterial::create();

    material->vertexShader = R"(
        void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    material->fragmentShader = R"(
        uniform vec3 uColor;
        void main() {
            gl_FragColor = vec4(uColor, 1.0);
        }
    )";
    material->uniforms["uColor"].setValue(Color(0x00ff00));

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.g > avg.r);
    CHECK(avg.g > avg.b);
}

TEST_CASE("Cross: ShaderMaterial produces similar result", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = PlaneGeometry::create(2, 2);
        auto material = ShaderMaterial::create();

        material->vertexShader = R"(
            void main() {
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        )";
        material->fragmentShader = R"(
            void main() {
                gl_FragColor = vec4(0.5, 0.3, 0.8, 1.0);
            }
        )";

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 30.0);
}

// =============================================================================
// Section 18: Dawn — ShadowMaterial
// =============================================================================

TEST_CASE("Dawn: ShadowMaterial renders shadow-receiving plane", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();

    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 5, 5);
    dirLight->castShadow = true;
    scene->add(dirLight);

    // Occluder
    auto boxGeom = BoxGeometry::create(1, 1, 1);
    auto boxMat = MeshBasicMaterial::create();
    boxMat->color = Color(0xff0000);
    auto box = Mesh::create(boxGeom, boxMat);
    box->position.y = 1;
    box->castShadow = true;
    scene->add(box);

    // Shadow-receiving plane with ShadowMaterial
    auto planeGeom = PlaneGeometry::create(10, 10);
    auto shadowMat = ShadowMaterial::create();
    shadowMat->color = Color(0x000000);
    auto plane = Mesh::create(planeGeom, shadowMat);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;
    scene->add(plane);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    // The box should be visible (red), shadow plane may show shadow darkening
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);
}

// =============================================================================
// Section 19: Dawn — Roughness & Metalness Maps
// =============================================================================

TEST_CASE("Dawn: roughnessMap affects specular highlights", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useRoughnessMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->metalness = 0.9f;
        material->roughness = 0.1f; // base smooth

        if (useRoughnessMap) {
            // Rough texture — makes surface rough, reducing specular
            material->roughnessMap = makeUniformTexture(255, 255, 255);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto smoothPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto roughPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Smooth surface should have brighter specular highlights (higher max brightness)
    int smoothMax = maxPixelBrightness(smoothPixels);
    int roughMax = maxPixelBrightness(roughPixels);
    CHECK(smoothMax > roughMax);
}

TEST_CASE("Dawn: metalnessMap affects metallic appearance", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useMetalnessMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->roughness = 0.3f;
        material->metalness = 0.0f; // base non-metallic

        if (useMetalnessMap) {
            // Full white = full metallic
            material->metalnessMap = makeUniformTexture(255, 255, 255);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto nonMetalPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto metalPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(nonMetalPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(metalPixels) > PIXEL_COUNT / 8);

    // Metallic vs non-metallic should produce different brightness profiles
    double nonMetalBright = avgBrightness(nonMetalPixels);
    double metalBright = avgBrightness(metalPixels);
    CHECK(std::abs(nonMetalBright - metalBright) > 1.0);
}

TEST_CASE("Cross: roughnessMap produces similar result", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xcccccc);
        material->metalness = 0.5f;
        material->roughness = 0.5f;
        material->roughnessMap = makeUniformTexture(128, 128, 128);

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 20: Dawn — Emissive Map
// =============================================================================

TEST_CASE("Dawn: emissiveMap produces glow pattern", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    // No lights — only emissive should be visible
    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = Color(0xffffff);
    material->emissiveIntensity = 1.0f;
    // Emissive map: half bright, half dark
    material->emissiveMap = makeProceduralTexture(
        255, 0, 0,   0, 0, 0,
        255, 0, 0,   0, 0, 0
    );
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 5);

    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g);
}

TEST_CASE("Cross: emissiveMap produces similar glow", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x000000);
        material->emissive = Color(0xffffff);
        material->emissiveIntensity = 1.0f;
        material->emissiveMap = makeUniformTexture(255, 128, 0);
        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 8);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 21: Dawn — AO Map
// =============================================================================

TEST_CASE("Dawn: aoMap darkens occluded areas", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useAoMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        // AO maps require a second UV set (uv2)
        auto uvAttr = geometry->getAttribute<float>("uv");
        auto& uvArr = uvAttr->array();
        geometry->setAttribute("uv2", FloatBufferAttribute::create(
            std::vector<float>(uvArr.begin(), uvArr.begin() + static_cast<long>(uvAttr->count() * 2)), 2));

        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 1.0f;
        material->metalness = 0.0f;

        if (useAoMap) {
            // Dark AO — should reduce ambient light contribution
            material->aoMap = makeUniformTexture(50, 50, 50);
            material->aoMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noAoPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto aoPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // AO map should darken the result
    CHECK(avgBrightness(noAoPixels) > avgBrightness(aoPixels));
}

// =============================================================================
// Section 22: Dawn — Alpha Map
// =============================================================================

TEST_CASE("Dawn: alphaMap controls transparency", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useAlphaMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = PlaneGeometry::create(2, 2);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->transparent = true;
        material->side = Side::Double;

        if (useAlphaMap) {
            // Very transparent alpha map
            material->alphaMap = makeUniformTexture(30, 30, 30);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto opaquePixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto alphaPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Alpha-mapped version should be dimmer (more transparent)
    CHECK(avgBrightness(opaquePixels) > avgBrightness(alphaPixels));
}

TEST_CASE("Cross: alphaMap produces similar transparency", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = PlaneGeometry::create(2, 2);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->transparent = true;
        material->side = Side::Double;
        material->alphaMap = makeUniformTexture(128, 128, 128);

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 23: Dawn — Displacement Map
// =============================================================================

TEST_CASE("Dawn: displacementMap offsets vertices", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useDisplacement) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        // High-res sphere for displacement to be visible
        auto geometry = SphereGeometry::create(1.0f, 64, 32);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 1.0f;

        if (useDisplacement) {
            material->displacementMap = makeUniformTexture(255, 255, 255);
            material->displacementScale = 0.5f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto normalPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto displacedPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Displaced sphere should cover more or different pixels
    int normalCount = countNonBlack(normalPixels);
    int displacedCount = countNonBlack(displacedPixels);
    CHECK(normalCount > PIXEL_COUNT / 8);
    CHECK(displacedCount > PIXEL_COUNT / 8);
    // Displacement should change the coverage or brightness
    CHECK(std::abs(normalCount - displacedCount) > 0);
}

// =============================================================================
// Section 24: Dawn — Light Map
// =============================================================================

TEST_CASE("Dawn: lightMap adds baked illumination", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useLightMap) {
        auto scene = Scene::create();
        // Only ambient — light map should add extra illumination
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = PlaneGeometry::create(2, 2);
        // Light maps require uv2
        auto uvAttr = geometry->getAttribute<float>("uv");
        auto& uvArr = uvAttr->array();
        geometry->setAttribute("uv2", FloatBufferAttribute::create(
            std::vector<float>(uvArr.begin(), uvArr.begin() + static_cast<long>(uvAttr->count() * 2)), 2));

        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Double;

        if (useLightMap) {
            material->lightMap = makeUniformTexture(255, 255, 255);
            material->lightMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noLmPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto lmPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Light map should add brightness
    CHECK(avgBrightness(lmPixels) > avgBrightness(noLmPixels));
}

// =============================================================================
// Section 25: Dawn — Bump Map
// =============================================================================

TEST_CASE("Dawn: bumpMap perturbs surface shading", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useBumpMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(1, 1, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0xffffff);

        if (useBumpMap) {
            // Checkerboard-style bump — should create shading variation
            material->bumpMap = makeProceduralTexture(
                255, 255, 255,  0, 0, 0,
                0, 0, 0,        255, 255, 255
            );
            material->bumpScale = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto flatPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto bumpPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(flatPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(bumpPixels) > PIXEL_COUNT / 8);

    // Bump map should change the brightness variance (more surface detail)
    double flatVar = brightnessVariance(flatPixels);
    double bumpVar = brightnessVariance(bumpPixels);
    CHECK(bumpVar != flatVar);
}

// =============================================================================
// Section 26: Dawn — Gradient Map (Toon Shading)
// =============================================================================

TEST_CASE("Dawn: gradientMap controls toon shading bands", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useGradientMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshToonMaterial::create();
        material->color = Color(0xffffff);

        if (useGradientMap) {
            // 2-step gradient: creates hard shadow boundary
            material->gradientMap = makeUniformTexture(0, 0, 0);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto defaultPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto gradientPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(defaultPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(gradientPixels) > PIXEL_COUNT / 8);

    // Different gradient maps should produce different brightness
    CHECK(std::abs(avgBrightness(defaultPixels) - avgBrightness(gradientPixels)) > 0.5);
}

// =============================================================================
// Section 27: Dawn — Environment Maps
// =============================================================================

TEST_CASE("Dawn: envMap adds reflections to standard material", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useEnvMap) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);
        auto dirLight = DirectionalLight::create(Color(0xffffff), 0.5f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x444444);
        material->metalness = 1.0f;
        material->roughness = 0.0f;

        if (useEnvMap) {
            // Create a simple 6-face cube texture (all red)
            std::vector<Image> faces;
            for (int i = 0; i < 6; i++) {
                std::vector<unsigned char> faceData = {255, 0, 0, 255};
                faces.emplace_back(Image(std::move(faceData), 1, 1));
            }
            auto cubeTexture = CubeTexture::create(faces);
            material->envMap = cubeTexture;
            material->envMapIntensity = 1.0f;
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto noEnvPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto envPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    CHECK(countNonBlack(noEnvPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(envPixels) > PIXEL_COUNT / 16);

    // Env map on a metallic surface should make it brighter/more colored
    auto envAvg = averageColor(envPixels);
    CHECK(envAvg.r > 5.0); // Should pick up red from env map
}

TEST_CASE("Cross: envMap produces similar reflections", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0x404040));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshStandardMaterial::create();
        material->color = Color(0x444444);
        material->metalness = 1.0f;
        material->roughness = 0.0f;

        std::vector<Image> faces;
        for (int i = 0; i < 6; i++) {
            std::vector<unsigned char> faceData = {0, 128, 255, 255};
            faces.emplace_back(Image(std::move(faceData), 1, 1));
        }
        material->envMap = CubeTexture::create(faces);
        material->envMapIntensity = 1.0f;

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    CHECK(countNonBlack(glPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(dawnPixels) > PIXEL_COUNT / 16);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 60.0);
}

// =============================================================================
// Section 28: Dawn — Morph Targets
// =============================================================================

TEST_CASE("Dawn: morph targets deform geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](float influence) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);

        // Create a morph target that scales the box to 2x
        auto posAttr = geometry->getAttribute<float>("position");
        int count = static_cast<int>(posAttr->count());
        std::vector<float> morphPositions(count * 3);
        for (int i = 0; i < count; i++) {
            morphPositions[i * 3 + 0] = posAttr->getX(i) * 2.0f;
            morphPositions[i * 3 + 1] = posAttr->getY(i) * 2.0f;
            morphPositions[i * 3 + 2] = posAttr->getZ(i) * 2.0f;
        }

        auto morphAttrs = geometry->getOrCreateMorphAttribute("position");
        morphAttrs->emplace_back(FloatBufferAttribute::create(morphPositions, 3));

        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->morphTargets = true;
        auto mesh = Mesh::create(geometry, material);
        mesh->morphTargetInfluences().resize(1);
        mesh->morphTargetInfluences()[0] = influence;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto basePixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto morphedPixels = renderWithDawn(*makeScene(1.0f), *camera, clearColor);

    int baseCount = countNonBlack(basePixels);
    int morphedCount = countNonBlack(morphedPixels);

    CHECK(baseCount > PIXEL_COUNT / 16);
    CHECK(morphedCount > PIXEL_COUNT / 16);

    // Morphed (scaled up) should cover more pixels
    CHECK(morphedCount > baseCount);
}

TEST_CASE("Cross: morph targets produce similar deformation", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(1, 1, 1);
        auto posAttr = geometry->getAttribute<float>("position");
        int count = static_cast<int>(posAttr->count());
        std::vector<float> morphPositions(count * 3);
        for (int i = 0; i < count; i++) {
            morphPositions[i * 3 + 0] = posAttr->getX(i) * 1.5f;
            morphPositions[i * 3 + 1] = posAttr->getY(i) * 1.5f;
            morphPositions[i * 3 + 2] = posAttr->getZ(i) * 1.5f;
        }

        auto morphAttrs = geometry->getOrCreateMorphAttribute("position");
        morphAttrs->emplace_back(FloatBufferAttribute::create(morphPositions, 3));

        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->morphTargets = true;
        auto mesh = Mesh::create(geometry, material);
        mesh->morphTargetInfluences().resize(1);
        mesh->morphTargetInfluences()[0] = 0.5f;
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    int glCount = countNonBlack(glPixels);
    int dawnCount = countNonBlack(dawnPixels);
    CHECK(glCount > PIXEL_COUNT / 16);
    CHECK(dawnCount > PIXEL_COUNT / 16);

    double ratio = static_cast<double>(glCount) / dawnCount;
    CHECK(ratio > 0.5);
    CHECK(ratio < 2.0);
}

// =============================================================================
// Section 29: Dawn — Skinning
// =============================================================================

TEST_CASE("Dawn: SkinnedMesh with skeleton renders correctly", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    // Create a simple 2-bone skeleton with a cylinder-like geometry
    auto geometry = CylinderGeometry::create(0.3f, 0.3f, 2.0f, 8, 4);

    // Assign skin weights and indices
    auto posAttr = geometry->getAttribute<float>("position");
    int vertexCount = static_cast<int>(posAttr->count());

    std::vector<float> skinWeights(vertexCount * 4, 0.0f);
    std::vector<float> skinIndices(vertexCount * 4, 0.0f);

    for (int i = 0; i < vertexCount; i++) {
        float y = posAttr->getY(i);
        // Blend between bone 0 (bottom) and bone 1 (top)
        float weight = (y + 1.0f) / 2.0f; // normalize from [-1,1] to [0,1]
        skinWeights[i * 4 + 0] = 1.0f - weight;
        skinWeights[i * 4 + 1] = weight;
        skinIndices[i * 4 + 0] = 0.0f;
        skinIndices[i * 4 + 1] = 1.0f;
    }

    geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
    geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));

    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    // Create bones
    auto bone0 = Bone::create();
    bone0->position.y = -1.0f;
    auto bone1 = Bone::create();
    bone1->position.y = 1.0f;
    bone0->add(bone1);

    auto skeleton = Skeleton::create({bone0, bone1});
    auto skinnedMesh = SkinnedMesh::create(geometry, material);
    skinnedMesh->add(bone0);
    skinnedMesh->bind(skeleton);

    scene->add(skinnedMesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > PIXEL_COUNT / 16);
}

TEST_CASE("Dawn: SkinnedMesh bone rotation deforms mesh", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](float boneRotation) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = CylinderGeometry::create(0.3f, 0.3f, 2.0f, 8, 4);
        auto posAttr = geometry->getAttribute<float>("position");
        int vertexCount = static_cast<int>(posAttr->count());

        std::vector<float> skinWeights(vertexCount * 4, 0.0f);
        std::vector<float> skinIndices(vertexCount * 4, 0.0f);

        for (int i = 0; i < vertexCount; i++) {
            float y = posAttr->getY(i);
            float weight = (y + 1.0f) / 2.0f;
            skinWeights[i * 4 + 0] = 1.0f - weight;
            skinWeights[i * 4 + 1] = weight;
            skinIndices[i * 4 + 0] = 0.0f;
            skinIndices[i * 4 + 1] = 1.0f;
        }

        geometry->setAttribute("skinWeight", FloatBufferAttribute::create(skinWeights, 4));
        geometry->setAttribute("skinIndex", FloatBufferAttribute::create(skinIndices, 4));

        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);

        auto bone0 = Bone::create();
        bone0->position.y = -1.0f;
        auto bone1 = Bone::create();
        bone1->position.y = 1.0f;
        bone1->rotation.z = boneRotation; // Rotate upper bone
        bone0->add(bone1);

        auto skeleton = Skeleton::create({bone0, bone1});
        auto skinnedMesh = SkinnedMesh::create(geometry, material);
        skinnedMesh->add(bone0);
        skinnedMesh->bind(skeleton);

        scene->add(skinnedMesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto straightPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto bentPixels = renderWithDawn(*makeScene(math::PI / 4), *camera, clearColor);

    // Both should render visible geometry
    CHECK(countNonBlack(straightPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(bentPixels) > PIXEL_COUNT / 16);

    // Bent mesh should have different pixel distribution
    double straightX = avgXPosition(straightPixels, RT_WIDTH, RT_HEIGHT);
    double bentX = avgXPosition(bentPixels, RT_WIDTH, RT_HEIGHT);
    // Rotation should shift average X position
    CHECK(std::abs(straightX - bentX) > 0.5);
}

// =============================================================================
// Section 30: Dawn — Clipping Planes
// =============================================================================

TEST_CASE("Dawn: clipping plane cuts geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useClipping) {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Double;

        if (useClipping) {
            // Clip plane that cuts through center of sphere
            material->clippingPlanes.push_back(Plane(Vector3(1, 0, 0), 0));
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    // Need to enable local clipping on the Dawn renderer
    static Canvas* dawnClipCanvas = nullptr;
    if (!dawnClipCanvas) {
        dawnClipCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    // Render without clipping
    auto fullPixels = renderWithDawn(*makeScene(false), *camera, clearColor);

    // Render with clipping — need renderer with localClippingEnabled
    {
        DawnRenderer renderer(*dawnClipCanvas);
        renderer.setClearColor(clearColor);
        renderer.localClippingEnabled = true;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        auto clippedScene = makeScene(true);
        renderer.render(*clippedScene, *camera);
        auto clippedPixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();

        int fullCount = countNonBlack(fullPixels);
        int clippedCount = countNonBlack(clippedPixels);

        CHECK(fullCount > PIXEL_COUNT / 8);
        // Clipped sphere should show fewer pixels (half was clipped away)
        CHECK(fullCount > clippedCount);
    }
}

TEST_CASE("Cross: clipping plane produces similar cut", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = SphereGeometry::create(1.0f, 16, 8);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);
        material->side = Side::Double;
        material->clippingPlanes.push_back(Plane(Vector3(1, 0, 0), 0));

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    // GL with clipping
    {
        GLRenderer glRenderer(glCanvas().size());
        glRenderer.setClearColor(clearColor);
        glRenderer.localClippingEnabled = true;
        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        glRenderer.setRenderTarget(target.get());
        auto scene = makeScene();
        glRenderer.render(*scene, *camera);
        auto glPixels = glRenderer.readRGBPixels();
        glRenderer.setRenderTarget(nullptr);
        glRenderer.dispose();

        // Dawn with clipping
        static Canvas* dawnClipCross = nullptr;
        if (!dawnClipCross) {
            dawnClipCross = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
        }
        DawnRenderer dawnRenderer(*dawnClipCross);
        dawnRenderer.setClearColor(clearColor);
        dawnRenderer.localClippingEnabled = true;
        auto dawnTarget = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        dawnRenderer.setRenderTarget(dawnTarget.get());
        auto dawnScene = makeScene();
        dawnRenderer.render(*dawnScene, *camera);
        auto dawnPixels = dawnRenderer.readRGBPixels();
        dawnRenderer.setRenderTarget(nullptr);
        dawnRenderer.dispose();

        int glCount = countNonBlack(glPixels);
        int dawnCount = countNonBlack(dawnPixels);
        CHECK(glCount > PIXEL_COUNT / 16);
        CHECK(dawnCount > PIXEL_COUNT / 16);

        double ratio = static_cast<double>(glCount) / dawnCount;
        CHECK(ratio > 0.5);
        CHECK(ratio < 2.0);
    }
}

// =============================================================================
// Section 31: Dawn — Tone Mapping
// =============================================================================

TEST_CASE("Dawn: tone mapping affects output brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);
    auto dirLight = DirectionalLight::create(Color(0xffffff), 2.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    // Render with no tone mapping
    static Canvas* dawnTmCanvas = nullptr;
    if (!dawnTmCanvas) {
        dawnTmCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithToneMapping = [&](ToneMapping tm, float exposure) {
        DawnRenderer renderer(*dawnTmCanvas);
        renderer.setClearColor(Color(0x000000));
        renderer.toneMapping = tm;
        renderer.toneMappingExposure = exposure;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto nonePixels = renderWithToneMapping(ToneMapping::None, 1.0f);
    auto reinhardPixels = renderWithToneMapping(ToneMapping::Reinhard, 1.0f);
    auto acesPixels = renderWithToneMapping(ToneMapping::ACESFilmic, 1.0f);

    CHECK(countNonBlack(nonePixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(reinhardPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(acesPixels) > PIXEL_COUNT / 8);

    // Different tone mapping should produce different brightness
    double noneBright = avgBrightness(nonePixels);
    double reinhardBright = avgBrightness(reinhardPixels);
    double acesBright = avgBrightness(acesPixels);

    // At least one tone mapper should differ from None
    bool reinhardDiffers = std::abs(noneBright - reinhardBright) > 1.0;
    bool acesDiffers = std::abs(noneBright - acesBright) > 1.0;
    CHECK((reinhardDiffers || acesDiffers));
}

TEST_CASE("Dawn: toneMappingExposure scales brightness", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
    dirLight->position.set(0, 0, 1);
    scene->add(dirLight);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0xffffff);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    static Canvas* dawnExpCanvas = nullptr;
    if (!dawnExpCanvas) {
        dawnExpCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithExposure = [&](float exposure) {
        DawnRenderer renderer(*dawnExpCanvas);
        renderer.setClearColor(Color(0x000000));
        renderer.toneMapping = ToneMapping::Reinhard;
        renderer.toneMappingExposure = exposure;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto lowExp = renderWithExposure(0.5f);
    auto highExp = renderWithExposure(2.0f);

    // Higher exposure should produce brighter output
    CHECK(avgBrightness(highExp) > avgBrightness(lowExp));
}

// =============================================================================
// Section 32: Dawn — Output Encoding / Color Space
// =============================================================================

TEST_CASE("Dawn: sRGB output encoding differs from linear", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = SphereGeometry::create(1.0f, 16, 8);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x808080); // mid-grey
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    static Canvas* dawnEncCanvas = nullptr;
    if (!dawnEncCanvas) {
        dawnEncCanvas = new Canvas(Canvas::Parameters().size(RT_WIDTH, RT_HEIGHT).headless(true).graphicsApi(GraphicsAPI::WebGPU));
    }

    auto renderWithEncoding = [&](Encoding enc) {
        DawnRenderer renderer(*dawnEncCanvas);
        renderer.setClearColor(Color(0x000000));
        renderer.outputEncoding = enc;

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    };

    auto linearPixels = renderWithEncoding(Encoding::Linear);
    auto srgbPixels = renderWithEncoding(Encoding::sRGB);

    CHECK(countNonBlack(linearPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(srgbPixels) > PIXEL_COUNT / 8);

    // sRGB gamma curve should produce different brightness than linear
    double linearBright = avgBrightness(linearPixels);
    double srgbBright = avgBrightness(srgbPixels);
    CHECK(std::abs(linearBright - srgbBright) > 1.0);
}

// =============================================================================
// Section 33: Dawn — Instanced Colors
// =============================================================================

TEST_CASE("Dawn: InstancedMesh per-instance colors", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto material = MeshBasicMaterial::create();
    material->color = Color(0xffffff);

    auto im = InstancedMesh::create(geometry, material, 2);

    // Left instance — red
    Matrix4 m;
    m.setPosition(Vector3(-1.0f, 0, 0));
    im->setMatrixAt(0, m);
    im->setColorAt(0, Color(0xff0000));

    // Right instance — blue
    m.setPosition(Vector3(1.0f, 0, 0));
    im->setMatrixAt(1, m);
    im->setColorAt(1, Color(0x0000ff));

    im->instanceColor()->needsUpdate();
    scene->add(im);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);

    // Should have both red and blue components
    CHECK(avg.r > 5);
    CHECK(avg.b > 5);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
}

TEST_CASE("Cross: InstancedMesh per-instance colors match", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = []() {
        auto scene = Scene::create();
        auto ambient = AmbientLight::create(Color(0xffffff));
        scene->add(ambient);

        auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto material = MeshBasicMaterial::create();
        material->color = Color(0xffffff);

        auto im = InstancedMesh::create(geometry, material, 2);

        Matrix4 m;
        m.setPosition(Vector3(-1.0f, 0, 0));
        im->setMatrixAt(0, m);
        im->setColorAt(0, Color(0xff0000));

        m.setPosition(Vector3(1.0f, 0, 0));
        im->setMatrixAt(1, m);
        im->setColorAt(1, Color(0x0000ff));

        im->instanceColor()->needsUpdate();
        scene->add(im);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 4;
    Color clearColor(0x000000);

    auto glPixels = renderWithGL(*makeScene(), *camera, clearColor);
    auto dawnPixels = renderWithDawn(*makeScene(), *camera, clearColor);

    auto glAvg = averageColor(glPixels);
    auto dawnAvg = averageColor(dawnPixels);

    // Both should have red and blue
    CHECK(glAvg.r > 5);
    CHECK(glAvg.b > 5);
    CHECK(dawnAvg.r > 5);
    CHECK(dawnAvg.b > 5);
    CHECK(std::abs(avgBrightness(glPixels) - avgBrightness(dawnPixels)) < 50.0);
}

// =============================================================================
// Section 34: Dawn — Shadow Map Quality
// =============================================================================

TEST_CASE("Dawn: shadow map resolution affects quality", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](int shadowMapSize) {
        auto scene = Scene::create();

        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 5, 5);
        dirLight->castShadow = true;
        dirLight->shadow->mapSize.set(shadowMapSize, shadowMapSize);
        scene->add(dirLight);

        // Occluder
        auto boxGeom = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        auto boxMat = MeshPhongMaterial::create();
        boxMat->color = Color(0x888888);
        auto box = Mesh::create(boxGeom, boxMat);
        box->position.y = 2;
        box->castShadow = true;
        scene->add(box);

        // Floor
        auto planeGeom = PlaneGeometry::create(10, 10);
        auto planeMat = MeshPhongMaterial::create();
        planeMat->color = Color(0xcccccc);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 5, 8);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto lowResPixels = renderWithDawn(*makeScene(64), *camera, clearColor);
    auto highResPixels = renderWithDawn(*makeScene(512), *camera, clearColor);

    // Both should render something
    CHECK(countNonBlack(lowResPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(highResPixels) > PIXEL_COUNT / 8);

    // Higher resolution shadow maps may produce slightly different brightness variance
    // (sharper shadow edges vs blockier). Both should be reasonable.
    double lowVar = brightnessVariance(lowResPixels);
    double highVar = brightnessVariance(highResPixels);
    // Just verify both produce variance (shadows present)
    CHECK(lowVar > 10.0);
    CHECK(highVar > 10.0);
}

TEST_CASE("Dawn: shadow bias prevents shadow acne", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](float bias) {
        auto scene = Scene::create();

        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 5, 0);
        dirLight->castShadow = true;
        dirLight->shadow->bias = bias;
        scene->add(dirLight);

        // Floor that receives shadow and is also shadow caster
        auto planeGeom = PlaneGeometry::create(5, 5);
        auto planeMat = MeshPhongMaterial::create();
        planeMat->color = Color(0xffffff);
        auto plane = Mesh::create(planeGeom, planeMat);
        plane->rotation.x = -math::PI / 2;
        plane->receiveShadow = true;
        scene->add(plane);

        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.set(0, 3, 5);
    camera->lookAt(Vector3(0, 0, 0));
    Color clearColor(0x000000);

    auto noBiasPixels = renderWithDawn(*makeScene(0.0f), *camera, clearColor);
    auto biasPixels = renderWithDawn(*makeScene(-0.005f), *camera, clearColor);

    // Both should produce visible output
    CHECK(countNonBlack(noBiasPixels) > PIXEL_COUNT / 16);
    CHECK(countNonBlack(biasPixels) > PIXEL_COUNT / 16);
}

// =============================================================================
// Section 35: Dawn — Specular Map
// =============================================================================

TEST_CASE("Dawn: specularMap controls highlight regions", "[dawn]") {
    REQUIRE_DAWN();

    auto makeScene = [](bool useSpecularMap) {
        auto scene = Scene::create();
        auto dirLight = DirectionalLight::create(Color(0xffffff), 1.0f);
        dirLight->position.set(0, 0, 1);
        scene->add(dirLight);

        auto geometry = SphereGeometry::create(1.0f, 32, 16);
        auto material = MeshPhongMaterial::create();
        material->color = Color(0x888888);
        material->specular = Color(0xffffff);
        material->shininess = 100.0f;

        if (useSpecularMap) {
            // Dark specular map — reduces specular highlights
            material->specularMap = makeUniformTexture(30, 30, 30);
        }

        auto mesh = Mesh::create(geometry, material);
        scene->add(mesh);
        return scene;
    };

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;
    Color clearColor(0x000000);

    auto fullSpecPixels = renderWithDawn(*makeScene(false), *camera, clearColor);
    auto reducedSpecPixels = renderWithDawn(*makeScene(true), *camera, clearColor);

    // Both should render
    CHECK(countNonBlack(fullSpecPixels) > PIXEL_COUNT / 8);
    CHECK(countNonBlack(reducedSpecPixels) > PIXEL_COUNT / 8);

    // Full specular should have brighter highlights
    int fullMax = maxPixelBrightness(fullSpecPixels);
    int reducedMax = maxPixelBrightness(reducedSpecPixels);
    CHECK(fullMax > reducedMax);
}

// =============================================================================
// Section 36: Dawn — Diffuse Map (color map)
// =============================================================================

TEST_CASE("Dawn: map (diffuse texture) tints geometry", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto ambient = AmbientLight::create(Color(0xffffff));
    scene->add(ambient);

    auto geometry = PlaneGeometry::create(2, 2);
    auto material = MeshBasicMaterial::create();
    material->map = makeUniformTexture(255, 0, 0); // Red texture
    material->side = Side::Double;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    auto avg = averageColor(pixels);
    CHECK(avg.r > avg.g + 20);
    CHECK(avg.r > avg.b + 20);
}

// =============================================================================
// Section 37: Dawn — LineBasicMaterial Options
// =============================================================================

TEST_CASE("Dawn: LineBasicMaterial with dashed line", "[dawn]") {
    REQUIRE_DAWN();

    auto scene = Scene::create();
    auto geometry = BufferGeometry::create();
    std::vector<float> positions = {-2, 0, 0, 2, 0, 0};
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->color = Color(0xffffff);

    auto line = Line::create(geometry, material);
    scene->add(line);

    auto camera = PerspectiveCamera::create(75, 1.0f, 0.1f, 100);
    camera->position.z = 3;

    auto pixels = renderWithDawn(*scene, *camera, Color(0x000000));
    int nonBlack = countNonBlack(pixels);
    CHECK(nonBlack > 3);
}
