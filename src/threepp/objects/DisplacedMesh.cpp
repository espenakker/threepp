#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/OceanLod.hpp"

// kOceanCascade1Rot* — cascade-1 sample-domain rotation, shared with the GLSL
// samplers (water_displace.comp / foam_world.comp). Header is cross-language
// (pure constants + a POD struct); including it here keeps CPU/GPU parity
// literal-exact.
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    const std::string& DisplacedMesh::type() const {
        static const std::string typeName = "DisplacedMesh";
        return typeName;
    }

    namespace {

        struct Vec2f {
            float x = 0.f;
            float y = 0.f;
        };

        // Filtered, repeat-wrapped sample of an RG32F cascade field at tile
        // coordinates (u, v) = sample-domain XZ / tileSize. Texel i's value
        // sits at (i+0.5)/dim, as with GPU texture(). `bicubic` selects the
        // uniform cubic B-spline over the 4x4 neighbourhood, which is what
        // sampleBicubicR/RG in water_displace.comp evaluate with four bilinear
        // taps; otherwise bilinear. Returns the raw (unnormalized) R and G.
        Vec2f sampleField(const std::vector<float>& field, uint32_t dim,
                          float u, float v, bool bicubic, bool interpolating=false) {
            const float fdim = float(dim);
            auto wrap = [fdim](float x) {
                float w = std::fmod(x, fdim);
                if (w < 0.f) w += fdim;
                return w;
            };
            const float fx = wrap(u * fdim - 0.5f);
            const float fz = wrap(v * fdim - 0.5f);
            const int ix = int(std::floor(fx));
            const int iz = int(std::floor(fz));
            const float tx = fx - float(ix);
            const float tz = fz - float(iz);

            auto texel = [&](int x, int z) {
                const int d = int(dim);
                x %= d;
                if (x < 0) x += d;
                z %= d;
                if (z < 0) z += d;
                const size_t idx = (size_t(z) * dim + size_t(x)) * 2u;
                return Vec2f{field[idx], field[idx + 1]};
            };

            float wx[4], wz[4];
            int first;
            int taps;
            if (bicubic) {
                auto bspline = [interpolating](float t, float w[4]) {
                    const float t2 = t * t;
                    const float t3 = t2 * t;
                    if(interpolating) {
                        w[0]=-.5f*t+t2-.5f*t3; w[1]=1-2.5f*t2+1.5f*t3;
                        w[2]=.5f*t+2*t2-1.5f*t3; w[3]=-.5f*t2+.5f*t3;
                        return;
                    }
                    w[0] = (-t3 + 3.f * t2 - 3.f * t + 1.f) / 6.f;
                    w[1] = (3.f * t3 - 6.f * t2 + 4.f) / 6.f;
                    w[2] = (-3.f * t3 + 3.f * t2 + 3.f * t + 1.f) / 6.f;
                    w[3] = t3 / 6.f;
                };
                bspline(tx, wx);
                bspline(tz, wz);
                first = -1;
                taps = 4;
            } else {
                wx[0] = 1.f - tx;
                wx[1] = tx;
                wz[0] = 1.f - tz;
                wz[1] = tz;
                first = 0;
                taps = 2;
            }

            Vec2f r;
            for (int j = 0; j < taps; ++j) {
                for (int i = 0; i < taps; ++i) {
                    const Vec2f t = texel(ix + first + i, iz + first + j);
                    const float w = wx[i] * wz[j];
                    r.x += w * t.x;
                    r.y += w * t.y;
                }
            }
            return r;
        }

        Vec2f sampleInterpolatingField(const std::vector<float>& field, uint32_t dim,
                          float u, float v, bool bicubic) {
            const float fdim = float(dim);
            auto wrap = [fdim](float x) {
                float w = std::fmod(x, fdim);
                if (w < 0.f) w += fdim;
                return w;
            };
            const float fx = wrap(u * fdim - 0.5f);
            const float fz = wrap(v * fdim - 0.5f);
            const int ix = int(std::floor(fx));
            const int iz = int(std::floor(fz));
            const float tx = fx - float(ix);
            const float tz = fz - float(iz);

            auto texel = [&](int x, int z) {
                const int d = int(dim);
                x %= d;
                if (x < 0) x += d;
                z %= d;
                if (z < 0) z += d;
                const size_t idx = (size_t(z) * dim + size_t(x)) * 2u;
                return Vec2f{field[idx], field[idx + 1]};
            };

            float wx[4], wz[4];
            int first;
            int taps;
            if (bicubic) {
                auto bspline = [](float t, float w[4]) {
                    const float t2 = t * t;
                    const float t3 = t2 * t;
                    w[0] = (-t3 + 3.f * t2 - 3.f * t + 1.f) / 6.f;
                    w[1] = (3.f * t3 - 6.f * t2 + 4.f) / 6.f;
                    w[2] = (-3.f * t3 + 3.f * t2 + 3.f * t + 1.f) / 6.f;
                    w[3] = t3 / 6.f;
                };
                auto cubic = [&](float t,float w[4]) {
                    const float t2=t*t,t3=t2*t;
                    w[0]=-.5f*t+t2-.5f*t3; w[1]=1.f-2.5f*t2+1.5f*t3;
                    w[2]=.5f*t+2.f*t2-1.5f*t3; w[3]=-.5f*t2+.5f*t3;
                };
                cubic(tx, wx); cubic(tz, wz);
                first = -1;
                taps = 4;
            } else {
                wx[0] = 1.f - tx;
                wx[1] = tx;
                wz[0] = 1.f - tz;
                wz[1] = tz;
                first = 0;
                taps = 2;
            }

            Vec2f r;
            for (int j = 0; j < taps; ++j) {
                for (int i = 0; i < taps; ++i) {
                    const Vec2f t = texel(ix + first + i, iz + first + j);
                    const float w = wx[i] * wz[j];
                    r.x += w * t.x;
                    r.y += w * t.y;
                }
            }
            return r;
        }

        template<class Field> bool hasField(const Field& cf, const std::vector<float>& field) {
            return cf.dim > 0 && cf.tileSize > 0.f &&
                   field.size() >= size_t(cf.dim) * cf.dim * 2u;
        }

        // Cascade i's sample-domain position for a world XZ. Cascade 1 is
        // sampled in a rotated domain (repeat-lattice break, see
        // vulkan_shared.h); mirror of oceanC1Domain().
        template<class Field> Vec2f cascadeDomain(uint32_t i, const Field& cf, float worldX, float worldZ) {
            if (i != 1) return {worldX, worldZ};
            if (cf.data.size() >= 6 && cf.data[1] < -1e19f) {
                const float border=2.f*cf.tileSize/cf.dim;
                return {std::clamp(worldX-cf.data[3]+cf.tileSize*.5f,border,cf.tileSize-border),
                        std::clamp(worldZ-cf.data[5]+cf.tileSize*.5f,border,cf.tileSize-border)};
            }
            return {kOceanCascade1RotCos * worldX - kOceanCascade1RotSin * worldZ,
                    kOceanCascade1RotSin * worldX + kOceanCascade1RotCos * worldZ};
        }

        // Cascades 0 and 1 are B-spline reconstructed on the GPU, cascade 2
        // bilinear (see sampleDisplacement in water_displace.comp).
        bool cascadeBicubic(uint32_t i) {
            return i < 2;
        }

        // One vessel's wake height at a world XZ: the per-vessel block of
        // water_displace.comp, faded only inside this vessel's own footprint.
        float vesselWakeHeight(const DisplacedMesh::HullExclusion& hullExclusion,
                               const DisplacedMesh::VesselWake& wake,
                               float worldX, float worldZ) {
            // Two contributions: a per-frame bow bump (current pose only,
            // small) and the V-wedge ridge height that diverges from the
            // bow at ~20° half-angle. The V-wedge is taken as MAX over the
            // historical trail samples — same combiner as the shader, so
            // 60 samples don't stack into fountain-shaped pillars.
            if (!wake.enabled || hullExclusion.halfLength <= 0.f) return 0.f;

            auto smoothstepF = [](float a, float b, float x) {
                const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
                return t * t * (3.f - 2.f * t);
            };

            // Hull margin: mirrors ocean_cascade.glsl oceanHullMargin (0.6 x half-beam
            // in [0.3, 2] m).
            const float hullMargin = std::clamp(0.6f * hullExclusion.halfBeam, 0.3f, 2.f);

            auto vWedgeAtPose = [&](float cx, float cz, float sinYaw, float cosYaw,
                                    float speed, float ageFade) -> float {
                const float spd  = std::abs(speed);
                const float gate = smoothstepF(0.5f, 1.5f, spd);
                if (gate <= 0.f) return 0.f;
                const float dx = worldX - cx;
                const float dz = worldZ - cz;
                const float lX = cosYaw * dx - sinYaw * dz;
                const float lZ = sinYaw * dx + cosYaw * dz;
                const float distAft = hullExclusion.halfLength - lZ;
                if (distAft <= 0.f || distAft > hullExclusion.halfLength * 6.f) return 0.f;
                // Per-pose hullFade — V-wedge only fires outside the hull
                // footprint, so a buoy passing close to the boat doesn't get
                // lifted by the wake formula under the hull itself.
                const float uHull = std::clamp(-lZ / hullExclusion.halfLength, -1.f, 1.f);
                float halfBeamAtLZ;
                if (uHull <= 0.f) {
                    halfBeamAtLZ = hullExclusion.halfBeam *
                                   std::pow(std::max(1.f - uHull * uHull, 0.f), 0.6f);
                } else {
                    halfBeamAtLZ = hullExclusion.halfBeam * (1.f - 0.25f * uHull * uHull);
                }
                const float hullEdgeX = std::abs(lX) - halfBeamAtLZ;
                const float hullEdgeZ = std::abs(lZ) - hullExclusion.halfLength;
                const float hullFade  = smoothstepF(0.f, hullMargin,
                                                    std::max(hullEdgeX, hullEdgeZ));
                if (hullFade <= 0.f) return 0.f;
                const float tanAV = 0.36f;
                const float expectedX = tanAV * distAft;
                const float dRidge = std::abs(std::abs(lX) - expectedX);
                const float sigmaR = 0.6f + 0.05f * distAft;
                const float ridge  = std::exp(-(dRidge * dRidge) / (sigmaR * sigmaR));
                const float alongDecay = std::exp(-distAft /
                                                  (hullExclusion.halfLength * 4.f));
                const float vAmp = std::clamp(0.016f * spd * spd + 0.06f * spd,
                                              0.f, 1.0f);
                return gate * vAmp * ridge * alongDecay * ageFade * hullFade;
            };

            // Current-pose bow bump (small, doesn't pile up across the trail).
            float h = 0.f;
            {
                const float spd  = std::abs(wake.forwardSpeed);
                const float gate = smoothstepF(0.5f, 1.5f, spd);
                if (gate > 0.f) {
                    const float dx = worldX - hullExclusion.centerX;
                    const float dz = worldZ - hullExclusion.centerZ;
                    const float lX = hullExclusion.cosYaw * dx - hullExclusion.sinYaw * dz;
                    const float lZ = hullExclusion.sinYaw * dx + hullExclusion.cosYaw * dz;
                    const float distFromBow = lZ - hullExclusion.halfLength;
                    const float bowR = 1.5f;
                    const float bowG = std::exp(-(distFromBow * distFromBow) /
                                                (bowR * bowR));
                    const float bowL = std::exp(-(lX * lX) /
                                                (hullExclusion.halfBeam *
                                                 hullExclusion.halfBeam));
                    const float bowAmp = std::clamp(0.012f * spd * spd, 0.f, 0.4f);
                    h += gate * bowAmp * bowG * bowL;
                }
            }

            // Historical V-wedge ridge — MAX over the trail samples.
            float vWedgeMax = 0.f;
            if (!wake.trail.empty()) {
                for (const auto& s : wake.trail) {
                    const float ageFade = std::exp(-s.age / 5.f);
                    vWedgeMax = std::max(vWedgeMax,
                                         vWedgeAtPose(s.worldX, s.worldZ,
                                                      s.sinYaw, s.cosYaw,
                                                      s.speed, ageFade));
                }
            } else {
                vWedgeMax = vWedgeAtPose(hullExclusion.centerX,
                                         hullExclusion.centerZ,
                                         hullExclusion.sinYaw,
                                         hullExclusion.cosYaw,
                                         wake.forwardSpeed, 1.f);
            }
            h += vWedgeMax;
            return h;
        }

    }// namespace

    float DisplacedMesh::sampleHeight(float worldX, float worldZ, uint32_t cascadeMask) const {
        // Physical/contact consumers use the unchanged fixed core and/or masked
        // fields. Full visual queries outside the core follow the drawn LOD field.
        const float radius=std::max(std::abs(worldX),std::abs(worldZ));
        if (!cdlod || cascadeMask!=7u || radius<=128.f)
            return sampleUnfilteredHeight(worldX,worldZ,cascadeMask);
        const float drawn=sampleLodHeight(worldX,worldZ);
        if (radius>=160.f) return drawn;
        const float t=(radius-128.f)/32.f, w=t*t*(3.f-2.f*t);
        return (1.f-w)*sampleUnfilteredHeight(worldX,worldZ,cascadeMask)+w*drawn;
    }

    float DisplacedMesh::sampleUnfilteredHeight(float worldX, float worldZ,
                                      uint32_t cascadeMask) const {
        // Sticky opt-in: the Vulkan renderer only records the GPU→host height
        // copies (and the per-frame mirror memcpy) once something actually
        // queries the CPU wave field. First-ever call may return 0 until a
        // frame has recorded the copies and its fence has retired — the
        // mirror's fixed kFramesInFlight-frame latency (see the header).
        wantsHeightReadback = true;
        struct View { uint32_t dim; float tileSize; const std::vector<float>& data; const std::vector<float>& disp; };
        auto view=[&](unsigned i) -> View {
            const auto& e=externalCascades[i]; const auto& c=heightFields[i];
            if(currentExternalQueries && e.dim) return {e.dim,i==0?params.tileSize0:i==1?params.tileSize1:params.tileSize2,e.height,e.disp};
            return {c.dim,c.tileSize,c.data,c.disp};
        };
        const View fields[3]={view(0),view(1),view(2)};

        // World-space horizontal displacement D(q) of the selected cascades at
        // rest position q, choppiness applied.
        auto displacement = [&](float qx, float qz) {
            Vec2f d;
            for (uint32_t i = 0; i < 3; ++i) {
                if ((cascadeMask & (1u << i)) == 0u) continue;
                const auto& cf = fields[i];
                if (!hasField(cf, cf.disp)) continue;
                const float invTile = 1.f / cf.tileSize;
                const Vec2f p = cascadeDomain(i, cf, qx, qz);
                Vec2f v = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile,
                                      cascadeBicubic(i),currentExternalQueries);
                if (i == 1 && !(cf.data.size() >= 6 && cf.data[1] < -1e19f)) {
                    // Domain vector back to world; mirror of oceanC1World().
                    v = {kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                         -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y};
                }
                d.x += v.x * invTile;
                d.y += v.y * invTile;
            }
            if (hasField(fields[1],fields[1].data) && fields[1].data[1] < -1e19f) {
                const auto& cf=fields[1];
                const float r=std::hypot(qx-cf.data[3],qz-cf.data[5]);
                const float u=std::clamp((r-.1f*cf.tileSize)/(.1f*cf.tileSize),0.f,1.f);
                const float lagrangian=u*u*(3-2*u);
                d.x*=lagrangian; d.y*=lagrangian;
            }
            d.x *= params.choppiness;
            d.y *= params.choppiness;
            return d;
        };

        // Solve x = q + D(q) for the rest position q. The map q -> x - D(q)
        // contracts wherever the surface does not fold (Jacobian > 0), so
        // fixed-point iteration converges; near a fold it can oscillate, so
        // keep the iterate with the smallest residual |q + D(q) - x|, which
        // is the length of the step it produces. The error shrinks by the
        // displacement slope choppiness*a*k per step: a few steps at the
        // default choppiness, 14 at choppiness 1 on a steep (ak = 0.59) wave.
        float bestX = worldX;
        float bestZ = worldZ;
        if (params.choppiness != 0.f) {
            const int kMaxIterations = currentExternalQueries ? 32 : 16;
            const float kTolerance = currentExternalQueries ? 1e-5f : 1e-3f;// metres
            float qx = worldX;
            float qz = worldZ;
            float bestResidual = std::numeric_limits<float>::max();
            for (int it = 0; it < kMaxIterations; ++it) {
                const Vec2f d = displacement(qx, qz);
                const float nx = worldX - d.x;
                const float nz = worldZ - d.y;
                const float residual = std::hypot(nx - qx, nz - qz);
                if (residual < bestResidual) {
                    bestResidual = residual;
                    bestX = qx;
                    bestZ = qz;
                }
                if (residual < kTolerance) break;
                qx = nx;
                qz = nz;
            }
        }

        float total = 0.f;
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) == 0u) continue;
            const auto& cf = fields[i];
            if (!hasField(cf, cf.data)) continue;
            const float invTile = 1.f / cf.tileSize;
            const Vec2f p = cascadeDomain(i, cf, bestX, bestZ);
            total += sampleField(cf.data, cf.dim, p.x * invTile, p.y * invTile,
                                 cascadeBicubic(i),currentExternalQueries).x * invTile;
        }
        return total * params.waveScale;
    }

    std::array<float,3> DisplacedMesh::sampleLodDisplacement(float worldX, float worldZ) const {
        wantsHeightReadback=true;
        auto smooth=[](float a,float b,float v) {
            float t=std::clamp((v-a)/(b-a),0.f,1.f); return t*t*(3.f-2.f*t);
        };
        using V=std::array<float,3>;
        auto raw=[&](float x,float z,unsigned mask,float footprint=0.f) {
            V d{};
            for (unsigned i=0;i<3;++i) {
                const auto& cf=heightFields[i];
                if (!(mask&(1u<<i)) || !hasField(cf,cf.data) || !hasField(cf,cf.disp)) continue;
                auto q=cascadeDomain(i,cf,x,z); float inv=1.f/cf.tileSize;
                auto filtered=[&](const std::vector<float>& src,std::vector<std::vector<float>>& mips) {
                    auto base=sampleInterpolatingField(src,cf.dim,q.x*inv,q.y*inv,i<2);
                    const bool finiteContact=i==1 && cf.data.size()>=6 && cf.data[1]<-1e19f;
                    const float level=i<2 && !finiteContact && footprint>2.f ? std::max(0.f,std::log2(footprint*cf.dim/cf.tileSize)) : 0.f;
                    if(level<=0.f)return base;
                    if(mips.empty()) {
                        unsigned n=cf.dim;const auto* in=&src;
                        while(n>1){unsigned half=n/2;std::vector<float> dst(half*half*2);
                            for(unsigned z=0;z<half;++z)for(unsigned x=0;x<half;++x)for(unsigned c=0;c<2;++c)
                                dst[(z*half+x)*2+c]=.25f*((*in)[((2*z)*n+2*x)*2+c]+(*in)[((2*z)*n+2*x+1)*2+c]+(*in)[((2*z+1)*n+2*x)*2+c]+(*in)[((2*z+1)*n+2*x+1)*2+c]);
                            mips.push_back(std::move(dst));in=&mips.back();n=half;
                        }
                    }
                    auto tap=[&](unsigned l){l=std::min(l,unsigned(mips.size()));return sampleField(l?mips[l-1]:src,cf.dim>>l,q.x*inv,q.y*inv,false);};
                    unsigned lo=unsigned(std::floor(level));float t=level-float(lo),blend=smooth(0.f,1.f,level);
                    auto a=tap(lo),b=tap(lo+1);return Vec2f{base.x+blend*(a.x+t*(b.x-a.x)-base.x),base.y+blend*(a.y+t*(b.y-a.y)-base.y)};
                };
                auto v=filtered(cf.disp,cf.lodDisp);
                if (i==1 && !(cf.data.size()>=6 && cf.data[1]<-1e19f))
                    v={kOceanCascade1RotCos*v.x+kOceanCascade1RotSin*v.y,
                       -kOceanCascade1RotSin*v.x+kOceanCascade1RotCos*v.y};
                d[0]+=v.x*inv; d[2]+=v.y*inv;
                d[1]+=filtered(cf.data,cf.lodHeight).x*inv;
            }
            const auto& cf=heightFields[1];
            if (hasField(cf,cf.data) && cf.data[1]<-1e19f) {
                float w=smooth(.1f*cf.tileSize,.2f*cf.tileSize,std::hypot(x-cf.data[3],z-cf.data[5]));
                d[0]*=w;d[2]*=w;
            }
            d[0]*=params.choppiness;d[2]*=params.choppiness;d[1]*=params.waveScale;
            return d;
        };
        auto field=[&](float x,float z) {
            float distance=std::max(std::abs(x-lodEyeX),std::abs(z-lodEyeZ));
            float contact=std::max(0.f,std::max(std::abs(x),std::abs(z))-128.f);
            float rangeDistance=std::min(distance,contact);
            float footprint=oceanlod::footprint(x,z,lodEyeX,lodEyeZ,lodEyeHeight);
            V d=raw(x,z,3u,footprint);
            float fine=1.f-smooth(1.f,4.f,footprint);
            if (fine>0.f) {auto tap=raw(x,z,4u);for(unsigned k=0;k<3;++k)d[k]+=fine*tap[k];}
            float amp=1.f-smooth(8000.f,12000.f,rangeDistance);
            for(auto& v:d)v*=amp;
            float radius=std::hypot(x,z);
            d[1]-=std::max(0.f,radius*radius-1500.f*1500.f)/(2.f*6371000.f);
            return d;
        };
        return field(worldX,worldZ);
    }

    float DisplacedMesh::sampleLodFieldHeight(float worldX,float worldZ) const {
        if (!cdlod) return sampleUnfilteredHeight(worldX,worldZ);
        float x=worldX,z=worldZ,best=1e30f,height=0.f;
        for(unsigned i=0;i<24;++i) {
            auto d=sampleLodDisplacement(x,z);float nx=worldX-d[0],nz=worldZ-d[2];
            float residual=std::hypot(nx-x,nz-z);
            if(residual<best){best=residual;height=d[1];}
            if(residual<1e-4f)break;
            x=nx;z=nz;
        }
        return height;
    }

    float DisplacedMesh::sampleLodHeight(float worldX,float worldZ) const {
        if (!cdlod) return sampleUnfilteredHeight(worldX,worldZ);
        float qx=worldX,qz=worldZ,bestX=qx,bestZ=qz,best=1e30f,height=0.f;
        for(unsigned i=0;i<24;++i) {
            auto d=sampleLodDisplacement(qx,qz);
            float nx=worldX-d[0],nz=worldZ-d[2],residual=std::hypot(nx-qx,nz-qz);
            if(residual<best){best=residual;bestX=qx;bestZ=qz;height=d[1];}
            if(residual<1e-4f)break;
            qx=nx;qz=nz;
        }
        if(std::max(std::abs(bestX),std::abs(bestZ))>=65536.f) return height;
        using namespace oceanlod;
        const auto initial=locate(bestX,bestZ,lodEyeX,lodEyeZ,lodEyeHeight);
        // Continuous inverse provides a seed. Search neighboring cells because
        // triangle interpolation and morph collapse shift the horizontal map.
        for(int dz:{0,-1,1,-2,2}) for(int dx:{0,-1,1,-2,2}) {
            const float x=bestX+dx*initial.spacing,z=bestZ+dz*initial.spacing;
            const auto tile=locate(x,z,lodEyeX,lodEyeZ,lodEyeHeight);
            const int ix=std::clamp(int(std::floor((x-tile.x)/tile.spacing)),0,int(cells)-1);
            const int iz=std::clamp(int(std::floor((z-tile.z)/tile.spacing)),0,int(cells)-1);
            std::array<float,3> p[4];
            for(unsigned j=0;j<4;++j) {
                const auto q=morph(tile.x+(ix+(j&1u))*tile.spacing,
                                   tile.z+(iz+(j>>1u))*tile.spacing,tile.spacing,lodEyeX,lodEyeZ,lodEyeHeight);
                const auto d=sampleLodDisplacement(q[0],q[1]);
                p[j]={q[0]+d[0],d[1],q[1]+d[2]};
            }
            for (auto ids:{std::array<unsigned,3>{0,2,1},std::array<unsigned,3>{2,3,1}}) {
                const auto& a=p[ids[0]];const auto& b=p[ids[1]];const auto& c=p[ids[2]];
                const double bx=double(b[0])-a[0],bz=double(b[2])-a[2],cx=double(c[0])-a[0],cz=double(c[2])-a[2];
                const double det=bx*cz-bz*cx;
                if(std::abs(det)<1e-12)continue;
                const double px=double(worldX)-a[0],pz=double(worldZ)-a[2];
                const double u=(px*cz-pz*cx)/det,v=(bx*pz-bz*px)/det;
                if(u>=-1e-5 && v>=-1e-5 && u+v<=1.00001)
                    return float(a[1]+u*(b[1]-a[1])+v*(c[1]-a[1]));
            }
        }
        // A severe fold can have several inverse branches. Retain the bounded
        // continuous-field result if the local triangle search finds no cover.
        return height;
    }

    float DisplacedMesh::sampleWakeHeight(float worldX, float worldZ) const {
        // Mirrors water_displace.comp wake formulas closely enough that
        // floaters (buoys, etc.) bob through the rendered wake. Keep
        // in sync if the shader changes. Every vessel's wake adds; see
        // the header for why another vessel's footprint does not fade it.
        float h = 0.f;
        for (uint32_t i = 0; i < kMaxVessels; ++i)
            h += vesselWakeHeight(vesselHull(i), vesselWake(i), worldX, worldZ);
        return h;
    }

}// namespace threepp
