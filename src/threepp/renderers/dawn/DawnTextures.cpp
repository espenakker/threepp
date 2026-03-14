
#include "DawnTextures.hpp"

#include "threepp/textures/Texture.hpp"

using namespace threepp;
using namespace threepp::dawn;

DawnTextures::DawnTextures(DawnState& state)
    : state_(state) {}

void DawnTextures::createDummyTexture() {
    WGPUTextureDescriptor td{};
    td.label = {.data = "dummy_tex", .length = 9};
    td.size = {1, 1, 1};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    dummyTexture_.texture = wgpuDeviceCreateTexture(state_.device, &td);
    dummyTexture_.view = wgpuTextureCreateView(dummyTexture_.texture, nullptr);

    unsigned char white[] = {255, 255, 255, 255};
    WGPUTexelCopyTextureInfo dst{};
    dst.texture = dummyTexture_.texture;
    WGPUTexelCopyBufferLayout layout{};
    layout.bytesPerRow = 4;
    layout.rowsPerImage = 1;
    WGPUExtent3D extent = {1, 1, 1};
    wgpuQueueWriteTexture(state_.queue, &dst, white, 4, &layout, &extent);

    WGPUSamplerDescriptor sd{};
    sd.label = {.data = "dummy_sampler", .length = 13};
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_Repeat;
    sd.addressModeV = WGPUAddressMode_Repeat;
    sd.addressModeW = WGPUAddressMode_Repeat;
    sd.maxAnisotropy = 1;
    dummyTexture_.sampler = wgpuDeviceCreateSampler(state_.device, &sd);
}

TextureEntry& DawnTextures::getOrCreateTexture(Texture* tex) {
    auto it = cache_.find(tex->id);
    if (it != cache_.end() && it->second.version == tex->version()) {
        return it->second;
    }

    // Release old if exists
    if (it != cache_.end()) {
        if (it->second.view) wgpuTextureViewRelease(it->second.view);
        if (it->second.texture) wgpuTextureRelease(it->second.texture);
        if (it->second.sampler) wgpuSamplerRelease(it->second.sampler);
    }

    TextureEntry entry{};
    auto& img = tex->image();
    auto w = img.width;
    auto h = img.height;
    if (w == 0 || h == 0) return dummyTexture_;

    WGPUTextureDescriptor td{};
    td.label = {.data = "user_tex", .length = 8};
    td.size = {w, h, 1};
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    td.dimension = WGPUTextureDimension_2D;
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    entry.texture = wgpuDeviceCreateTexture(state_.device, &td);
    entry.view = wgpuTextureCreateView(entry.texture, nullptr);

    auto& data = img.data<unsigned char>();
    WGPUTexelCopyTextureInfo dst{};
    dst.texture = entry.texture;
    WGPUTexelCopyBufferLayout layout{};
    layout.bytesPerRow = w * 4;
    layout.rowsPerImage = h;
    WGPUExtent3D extent = {w, h, 1};
    wgpuQueueWriteTexture(state_.queue, &dst, data.data(), data.size(), &layout, &extent);

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
    sd.maxAnisotropy = 1;
    entry.sampler = wgpuDeviceCreateSampler(state_.device, &sd);

    entry.version = tex->version();
    cache_[tex->id] = entry;
    return cache_[tex->id];
}

void DawnTextures::dispose() {
    for (auto& [id, te] : cache_) {
        if (te.view) wgpuTextureViewRelease(te.view);
        if (te.texture) wgpuTextureRelease(te.texture);
        if (te.sampler) wgpuSamplerRelease(te.sampler);
    }
    cache_.clear();

    if (dummyTexture_.view) wgpuTextureViewRelease(dummyTexture_.view);
    if (dummyTexture_.texture) wgpuTextureRelease(dummyTexture_.texture);
    if (dummyTexture_.sampler) wgpuSamplerRelease(dummyTexture_.sampler);
    dummyTexture_ = {};
}
