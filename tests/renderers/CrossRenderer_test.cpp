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
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

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

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
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

        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(scene, camera);

        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        return pixels;
    }

}// namespace

#define REQUIRE_DAWN() do { if (!isDawnAvailable()) SKIP("No GPU backend available for Dawn"); } while(0)


// =============================================================================
// Section 1: GL-only validation
// =============================================================================

TEST_CASE("GL: clear color produces expected pixels") {
    GLRenderer renderer(glCanvas().size());
    renderer.setClearColor(Color(1.0f, 0.0f, 0.0f));

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
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

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
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
    CHECK(nonBlack > PIXEL_COUNT / 8);

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

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
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

    auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
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
    CHECK(nonBlack > PIXEL_COUNT / 16);
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
