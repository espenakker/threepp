// Only compiled into the opt-in variant. Scalar mirror of OceanInspectSample.
#include "ocean_cascade.glsl"
layout(set=0,binding=82) uniform sampler2D inspectHeight0;
layout(set=0,binding=83) uniform sampler2D inspectHeight1;
layout(set=0,binding=84) uniform sampler2D inspectHeight2;
layout(set=0,binding=85) uniform sampler2D inspectDisplace0;
layout(set=0,binding=86) uniform sampler2D inspectDisplace1;
layout(set=0,binding=87) uniform sampler2D inspectDisplace2;
layout(buffer_reference, scalar) buffer OceanInspectBuffer {
    uint64_t hullAddress, wakeTrailAddress;
    float tile[3]; float waveScale, choppiness, seaY;
    uint cascadeMask, hullCount;
    float coarseFold, fineFold, foamTile; uint foamRes;
    int pixelX, pixelY;
    uint valid, renderWidth, renderHeight;
    float worldX, worldZ, height, slope, jacobian;
    float restX, restZ, foam, banked, fineKernel;
    float wake[4]; float radiation, value;
    float coarseJ, fineJ, coarseThreshold, fineThreshold;
    int displayX, displayY;
};
#define inspectData OceanInspectBuffer(pc.inspectAddress)
vec3 gInspectReflection, gInspectTransmission, gInspectGlitter;
float gInspectRayShare, gInspectReflShare, gInspectTransShare;
float gInspectFreshAlpha = 0.0;
bool gInspectLiveBand = false;
