
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>

using namespace threepp;

namespace {


    struct MyEventListener: EventListener {

        int numCalled = 0;

        void onEvent(Event& e) override {
            ++numCalled;
        }
    };

    struct OnMaterialDispose: EventListener {

        void onEvent(Event& event) override {
            auto* material = std::any_cast<Material*>(event.target);
            material->removeEventListener("dispose", *this);
        }
    };

}// namespace

TEST_CASE("EventDispatcher: Test events") {

    EventDispatcher evt;

    MyEventListener l;

    bool l1Called = false;
    LambdaEventListener l1([&l1Called](Event& e) {
        l1Called = true;
    });

    evt.addEventListener("test1", l);
    evt.addEventListener("test2", l1);

    evt.dispatchEvent("test1");
    evt.dispatchEvent("test1");

    REQUIRE(2 == l.numCalled);

    evt.removeEventListener("test1", l);

    evt.dispatchEvent("test1");
    evt.dispatchEvent("test2");

    REQUIRE(2 == l.numCalled);
    REQUIRE(l1Called);

    REQUIRE(!evt.hasEventListener("test1", l));
    REQUIRE(evt.hasEventListener("test2", l1));

    OnMaterialDispose onDispose;
    auto material = MeshBasicMaterial::create();
    material->addEventListener("dispose", onDispose);

    REQUIRE(material->hasEventListener("dispose", onDispose));
    material->dispose();
    REQUIRE(!material->hasEventListener("dispose", onDispose));
}
