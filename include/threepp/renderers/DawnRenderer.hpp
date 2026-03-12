
#ifndef THREEPP_DAWNRENDERER_HPP
#define THREEPP_DAWNRENDERER_HPP

#include "threepp/renderers/Renderer.hpp"
#include "threepp/canvas/Canvas.hpp"

#include <memory>

namespace threepp {

    class DawnRenderer : public Renderer {

    public:
        explicit DawnRenderer(Canvas& canvas);

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        void setViewport(const Vector4& v) override;
        void setViewport(int x, int y, int width, int height) override;

        void setScissor(const Vector4& v) override;
        void setScissor(int x, int y, int width, int height) override;
        void setScissorTest(bool boolean) override;

        void setClearColor(const Color& color, float alpha = 1) override;
        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        std::vector<unsigned char> readRGBPixels() override;

        void dispose() override;

        ~DawnRenderer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_DAWNRENDERER_HPP
