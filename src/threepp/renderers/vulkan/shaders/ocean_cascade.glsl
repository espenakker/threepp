// Ocean cascade sample-domain helpers, shared by every GLSL consumer of the
// FFT cascade textures (water_displace.comp, foam_world.comp). The rotation
// constants live in vulkan_shared.h so the C++ mirror
// (DisplacedMesh::sampleHeight) and the host-side windTheta compensation use
// the exact same literals — see the comment block there for the why.

#ifndef THREEPP_OCEAN_CASCADE_GLSL
#define THREEPP_OCEAN_CASCADE_GLSL

#include "vulkan_shared.h"

// World XZ → cascade-1 sample domain (q = R(θ)·p).
vec2 oceanC1Domain(vec2 p) {
    return vec2(kOceanCascade1RotCos * p.x - kOceanCascade1RotSin * p.y,
                kOceanCascade1RotSin * p.x + kOceanCascade1RotCos * p.y);
}

// Cascade-1 domain vector → world (w = Rᵀ·v). Apply to the sampled
// horizontal-displacement (dx,dz) so the pull stays a world-space vector.
vec2 oceanC1World(vec2 v) {
    return vec2( kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y);
}

// A finite external cascade may use the otherwise unused height G channel as a
// descriptor: G(0,0)=-1e20, G(1,0)=centreX, G(2,0)=centreZ. R remains height.
// Its zero border permits clamping outside the patch, avoiding periodic ghost piles.
bool oceanC1Local(sampler2D h) { return texelFetch(h, ivec2(0), 0).g < -1e19; }
vec2 oceanC1Domain(sampler2D h, vec2 p, float tile) {
    if (!oceanC1Local(h)) return oceanC1Domain(p);
    vec2 c=vec2(texelFetch(h,ivec2(1,0),0).g,texelFetch(h,ivec2(2,0),0).g);
    float border=2.0/float(textureSize(h,0).x);
    return clamp((p-c)/tile+0.5,vec2(border),vec2(1.0-border))*tile;
}
vec2 oceanC1World(sampler2D h, vec2 v) {
    return oceanC1Local(h) ? v : oceanC1World(v);
}
float oceanContactEulerian(sampler2D h, vec2 p, float tile) {
    if (!oceanC1Local(h)) return 0.0;
    const vec2 c=vec2(texelFetch(h,ivec2(1,0),0).g,texelFetch(h,ivec2(2,0),0).g);
    // The finite patch spans 40 pile radii. Eulerian inside 4a, blend to 8a.
    return 1.0-smoothstep(0.1*tile,0.2*tile,length(p-c));
}

// Width (m) of the band outside a hull's footprint over which the sea blends from
// the vessel's waterline plane back to the wave field, and inside which the wake
// ridges fade in. It scales with the hull, 0.6 x half-beam clamped to [0.3, 2] m:
// a fixed 2 m, sized for the 28 m Gunnerus (4.5 m half-beam, which keeps exactly
// 2 m), flattened a glassy apron a third of a 6 m USV's length ahead of her bow.
// DisplacedMesh::sampleWakeHeight mirrors this.
float oceanHullMargin(float halfBeam) {
    return clamp(0.6 * halfBeam, 0.3, 2.0);
}

// One vessel as the ocean passes read it: DisplacedMesh::HullExclusion, the
// wake's forward speed and the vessel's slice of the wake-trail buffer. The host
// packs only the active vessels (halfLength > 0). 116 bytes, mirrored field for
// field by vulkan::OceanHullGpu (WaterDisplacePipeline.hpp).
struct OceanHull {
    float centerX;
    float centerZ;
    float halfLength;
    float halfBeam;
    float sinYaw;
    float cosYaw;
    float forwardSpeed; // m/s along +heading; 0 = no wake (or the wake is disabled)
    float centerY;      // waterline plane: world y at (centerX, centerZ),
    float pitch;        // +bow up (+localZ),
    float roll;         // +starboard up (+localX)
    uint  trailFirst;   // this vessel's samples in the wake-trail buffer
    uint  trailCount;
    float radEnabled;  // render-only band; exactly 0 disables all new math
    float radOmega;
    float radK;
    float radAbar;
    float radDraught;
    float radPhase;    // omega*(draw time - phasor epoch)
    float radHeaveRe;
    float radHeaveIm;
    float radRollRe;   // tan(clamped roll), +starboard up
    float radRollIm;
    float radPitchRe;  // tan(clamped pitch), +bow up
    float radPitchIm;
    float refR;
    float refK;
    float refDirX;     // incident propagation in scene XZ
    float refDirZ;
    float radBand;
};

#endif// THREEPP_OCEAN_CASCADE_GLSL
