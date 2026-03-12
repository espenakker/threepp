// Minimal WebGPU/Dawn triangle example using the DawnRenderer backend.

#include "threepp/threepp.hpp"
#include "threepp/renderers/DawnRenderer.hpp"

using namespace threepp;

int main() {

    Canvas::Parameters params;
    params.title("Dawn Triangle")
          .size(800, 600)
          .graphicsApi(GraphicsAPI::WebGPU);

    Canvas canvas(params);

    DawnRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100.f);
    camera->position.z = 3;

    auto scene = Scene::create();

    // Create a colored box
    auto geometry = BoxGeometry::create();
    auto material = MeshBasicMaterial::create();
    material->color = Color(0x00ff88);

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.animate([&] {
        mesh->rotation.x += 0.01f;
        mesh->rotation.y += 0.02f;

        renderer.render(*scene, *camera);
    });

    return 0;
}
