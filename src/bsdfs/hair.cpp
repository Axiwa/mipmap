#include <fstream>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Hair final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    Hair(const Properties &props) : Base(props) {
        m_sigma_a = props.texture<Texture>("sigma_a", 0.f);
        m_beta_m  = props.get<ScalarFloat>("beta_m", 0.3f);
        m_beta_n  = props.get<ScalarFloat>("beta_n", 0.3f);
        m_alpha   = props.get<ScalarFloat>("alpha", 2.f);
        m_eta     = props.get<ScalarFloat>("eta", 1.55f);

        assert (m_beta_m >= 0 && m_beta_m <= 1 && "beta_m should be in [0, 1]");
        assert (m_beta_n >= 0 && m_beta_n <= 1 && "beta_n should be in [0, 1]");
        assert (pMax >= 3 && "pMax should be >= 3");

        m_components.push_back(BSDFFlags::Glossy | BSDFFlags::FrontSide |
                               BSDFFlags::BackSide | BSDFFlags::NonSymmetric);

        m_flags = m_components[0];
        dr::set_attr(this, "flags", m_flags);

        // Preprocessing
        // TODO: pow can be optimized here
        v[0] = dr::sqr(0.726f * m_beta_m + 0.812f * dr::sqr(m_beta_m) +
                       3.7f * dr::pow(m_beta_m, 20));
        v[1] = .25f * v[0];
        v[2] = 4 * v[0];
        for (int p = 3; p <= pMax; ++p)
            v[p] = v[2];

        // Compute azimuthal logistic scale factor from $\m_beta_n$
        s = SqrtPiOver8 * (0.265f * m_beta_n + 1.194f * dr::sqr(m_beta_n) +
                           5.372f * dr::pow(m_beta_n, 22));

        // Compute $\m_alpha$ terms for hair scales
        sin2kAlpha[0] = dr::sin(dr::deg_to_rad(m_alpha));
        cos2kAlpha[0] = dr::safe_sqrt(1 - dr::sqr(sin2kAlpha[0]));
        for (int i = 1; i < 3; ++i) {
            sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = dr::sqr(cos2kAlpha[i - 1]) - dr::sqr(sin2kAlpha[i - 1]);
        }
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("sigma_a", m_sigma_a.get(),
                             +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_beta_m", m_beta_m,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_beta_n", m_beta_n,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_alpha", m_alpha,
                                +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_eta", m_eta, +ParamFlags::NonDifferentiable);
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {

        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float _pdf  = Float(0);
        Vector3f wi = si.wi;
        Vector3f wo;
        Float h = -1 + 2 * si.uv[1];
        Float gammaI    = dr::safe_asin(h);

        // Compute hair coordinate system terms related to _wo_
        Float sinThetaI = wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI      = dr::atan2(wi.z(), wi.y());
        
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        Point2f u[2] = { {sample1, 0}, sample2 };

        // Determine which term $p$ to sample for hair scattering
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);

        // std::cout<<apPdf[0]<<" "<<apPdf[1]<<" "<<apPdf[2]<<" "<<apPdf[3]<<std::endl;

        Int32 p       = Int32(-1);
        // u[0][1] is the rescaled random number after using u[0][0]
        u[0][1] = u[0][0] / apPdf[0];
        ScalarInt32 i = 0;

        while (i < pMax) {
            dr::masked(p, u[0][0] >= apPdf[i]) = i;
            dr::masked(u[0][1], u[0][0] >= apPdf[i]) = (u[0][0] - apPdf[i])/ apPdf[i+1];
            u[0][0] -= apPdf[i];
            i++;
        }
        p++;

        // std::cout<<sample1<<std::endl;
        // std::cout<<u[0][1]<<std::endl;

        // std::cout<<dr::count(dr::eq(p, 0))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 1))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 2))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 3))<<std::endl;

        // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale
        // tilt
        Float sinThetaIp = sinThetaI;
        Float cosThetaIp = cosThetaI;

        // if (p == 0) {
        dr::masked(sinThetaIp, dr::eq(p, 0)) = sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
        dr::masked(cosThetaIp, dr::eq(p, 0)) = cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
        // }
        // else if (p == 1) {
        dr::masked(sinThetaIp, dr::eq(p, 1)) = sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
        dr::masked(cosThetaIp, dr::eq(p, 1)) = cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
        // }
        // else if (p == 2) {
        dr::masked(sinThetaIp, dr::eq(p, 2)) = sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
        dr::masked(cosThetaIp, dr::eq(p, 2)) = cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
        // }

        // Sample $M_p$ to compute $\thetai$
        u[1][0] = dr::maximum(u[1][0], Float(1e-5));

        Float cosTheta =
            1 + v[pMax] * dr::log(u[1][0] + (1 - u[1][0]) * dr::exp(-2 / v[pMax]));
        for (int i = 0; i < pMax; i++) {
            dr::masked(cosTheta, dr::eq(p, i)) = 1 + v[i] * dr::log(u[1][0] + (1 - u[1][0]) * dr::exp(-2 / v[i]));
        }

        Float sinTheta = dr::safe_sqrt(1 - dr::sqr(cosTheta));
        Float cosPhi   = dr::cos(2 * dr::Pi<ScalarFloat> * u[1][1]);
        Float sinThetaO =
            -cosTheta * sinThetaIp + sinTheta * cosPhi * cosThetaIp;
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));

        // Sample $N_p$ to compute $\Delta\phi$

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(Float(m_eta * m_eta) - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float gammaT    = dr::safe_asin(sinGammaT);
        Float dphi;
        Float Phi = 2 * p * gammaT - 2 * gammaI + p * dr::Pi<ScalarFloat>;

        dphi = dr::select(p < pMax, Phi + SampleTrimmedLogistic(u[0][1], s, -dr::Pi<ScalarFloat>, dr::Pi<ScalarFloat>), 2 * dr::Pi<ScalarFloat> * u[0][1]);

        // Compute _wi_ from sampled hair scattering angles
        Float phiO = phiI + dphi;
        wo = Vector3f(sinThetaO, cosThetaO * dr::cos(phiO),
                      cosThetaO * dr::sin(phiO));

        // Compute PDF for sampled hair scattering direction _wi_
        for (int i = 0; i < pMax; ++i) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for
            // scales
            Float sinThetaIp, cosThetaIp;
            if (i == 0) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (i == 1) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
            } else if (i == 2) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
            } else {
                sinThetaIp = sinThetaI;
                cosThetaIp = cosThetaI;
            }

            // Handle out-of-range $\cos \thetao$ from scale adjustment
            cosThetaIp = dr::abs(cosThetaIp);
            _pdf += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[i]) *
                    apPdf[i] * Np(dphi, i, s, gammaI, gammaT);
        }

        _pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
                apPdf[pMax] * (1 / (2 * dr::Pi<ScalarFloat>) );

        bs.wo                = wo;
        bs.pdf               = dr::select(dr::isnan(_pdf) || dr::isinf(_pdf), 0, _pdf);
        bs.eta               = 1.;
        bs.sampled_type      = +BSDFFlags::Glossy;
        bs.sampled_component = 0;

        // std::cout<<bs.wo<<std::endl;

        UnpolarizedSpectrum value =
            dr::select(dr::neq(bs.pdf, 0), eval(ctx, si, bs.wo, active) / bs.pdf, 0);

        return { bs, depolarizer<Spectrum>(value) & (active && bs.pdf > 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // TODO
        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) &&
            !ctx.is_enabled(BSDFFlags::GlossyReflection)) {
            return 0.f;
        }

        // h is related to si.uv
        Float h      = -1 + 2 * si.uv[1];
        Float gammaI = dr::safe_asin(h);

        // TODO
        // CHECK(h >= -1 && h <= 1);

        // Compute the BSDF
        // Compute hair coordinate system terms related to _wi_
        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO      = dr::atan2(wo.z(), wo.y());

        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI      = dr::atan2(si.wi.z(), si.wi.y());

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(Float(m_eta * m_eta) - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));
        Float gammaT    = dr::safe_asin(sinGammaT);

        Spectrum T =
            dr::exp(-m_sigma_a->eval(si, active) * (2 * cosGammaT / cosThetaT));

        // Calculate Ap
        dr::Array<Spectrum, pMax + 1> ap = Ap(cosThetaI, Float(m_eta), h, T);

        // Evaluate hair BSDF
        Float phi = phiO - phiI;

        Spectrum fsum(0.);
        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for
            // scales
            Float sinThetaIp, cosThetaIp;
            if (p == 0) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
            }

            // Handle remainder of $p$ values for hair scale tilt
            else if (p == 1) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
            } else if (p == 2) {
                sinThetaIp =
                    sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
                cosThetaIp =
                    cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
            } else {
                sinThetaIp = sinThetaI;
                cosThetaIp = cosThetaI;
            }

            // Handle out-of-range $\cos \thetao$ from scale adjustment
            cosThetaIp = dr::abs(cosThetaIp);
            fsum += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) *
                    ap[p] * Np(phi, p, s, gammaI, gammaT);
        }

        // Compute contribution of remaining terms after _pMax_
        fsum += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
                ap[pMax] / (2.f * dr::Pi<ScalarFloat>);

        // If it is nan, return 0
        fsum = dr::select(dr::isnan(fsum) || dr::isinf(fsum), 0, fsum);

        return depolarizer<Spectrum>(fsum) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) &&
            !ctx.is_enabled(BSDFFlags::GlossyReflection)) {
            return 0.f;
        }

        Float h = -1 + 2 * si.uv[1];
        Float gammaI    = dr::safe_asin(h);

        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO      = dr::atan2(wo.z(), wo.y());

        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI      = dr::atan2(si.wi.z(), si.wi.y());
        
        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(Float(m_eta * m_eta) - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float gammaT    = dr::safe_asin(sinGammaT);

        // Compute PDF for $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);

        // Compute PDF sum for hair scattering events
        Float phi  = phiO - phiI;
        Float _pdf = Float(0);

        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for
            // scales
            Float sinThetaIp, cosThetaIp;
            if (p == 0) {
                sinThetaIp = sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
                cosThetaIp = cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
            }
            else if (p == 1) {
                sinThetaIp = sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
                cosThetaIp = cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
            } else if (p == 2) {
                sinThetaIp = sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
                cosThetaIp = cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
            } else {
                sinThetaIp = sinThetaI;
                cosThetaIp = cosThetaI;
            }

            // Handle out-of-range $\cos \thetao$ from scale adjustment
            cosThetaIp = dr::abs(cosThetaIp);
            _pdf += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) *
                    apPdf[p] * Np(phi, p, s, gammaI, gammaT);
        }
        _pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
                apPdf[pMax] * (1 / (2 * dr::Pi<ScalarFloat>) );

        _pdf = dr::select(dr::isnan(_pdf) || dr::isinf(_pdf), 0, _pdf);
        return _pdf ;
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) &&
            !ctx.is_enabled(BSDFFlags::GlossyReflection)) {
            return { 0.f, 0.f };
        }

        Float h      = -1 + 2 * si.uv[1];
        Float gammaI = dr::safe_asin(h);

        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO      = dr::atan2(wo.z(), wo.y());

        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI      = dr::atan2(si.wi.z(), si.wi.y());

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(Float(m_eta * m_eta) - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));
        Float gammaT    = dr::safe_asin(sinGammaT);

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T =
            dr::exp(-m_sigma_a->eval(si, active) * (2 * cosGammaT / cosThetaT));

        // Compute PDF for $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);

        dr::Array<Spectrum, pMax + 1> ap = Ap(cosThetaI, Float(m_eta), h, T);

        // Compute PDF sum for hair scattering events
        Float phi = phiO - phiI;
        Float _pdf = Float(0);
        Spectrum fsum(0.);

        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for
            // scales
            Float sinThetaIp, cosThetaIp;
            if (p == 0) {
                sinThetaIp = sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
                cosThetaIp = cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (p == 1) {
                sinThetaIp = sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
                cosThetaIp = cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
            } else if (p == 2) {
                sinThetaIp = sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
                cosThetaIp = cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
            } else {
                sinThetaIp = sinThetaI;
                cosThetaIp = cosThetaI;
            }

            // Handle out-of-range $\cos \thetao$ from scale adjustment
            cosThetaIp = dr::abs(cosThetaIp);

            Float M_p = Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]);
            Float N_p = Np(phi, p, s, gammaI, gammaT);

            _pdf += M_p * apPdf[p] * N_p;

            fsum += M_p * ap[p] * N_p;
        }

        _pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
               apPdf[pMax] / (2.f * dr::Pi<ScalarFloat>);

        fsum += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
                ap[pMax] / (2.f * dr::Pi<ScalarFloat>);

        fsum = dr::select(dr::isnan(fsum) || dr::isinf(fsum), 0, fsum);
        _pdf = dr::select(dr::isnan(_pdf) || dr::isinf(_pdf), 0, _pdf);

        return { depolarizer<Spectrum>(fsum) & active, _pdf};
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Hair["
            << std::endl
            // << "  reflectance = " << string::indent(m_reflectance) <<
            // std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    static const int pMax = 3;
    constexpr static const ScalarFloat SqrtPiOver8 = 0.626657069f;
    Float v[pMax + 1];
    Float s;
    Float sin2kAlpha[3], cos2kAlpha[3];
    ref<Texture> m_sigma_a;
    ScalarFloat m_beta_m, m_beta_n, m_alpha;
    ScalarFloat m_eta;

// Helper function
    static inline Float I0(Float x) {
        Float val     = 0;
        Float x2i     = 1;
        int64_t ifact = 1;
        int i4        = 1;
        for (int i = 0; i < 10; ++i) {
            if (i > 1){
                ifact *= i;
            }
            val += x2i / (i4 * dr::sqr(ifact));
            x2i *= x * x;
            i4 *= 4;
        }
        return val;
    }

    static Float LogI0(Float x) {
        return dr::select(x > 12,
                          x + 0.5f * (-dr::log(2 * dr::Pi<ScalarFloat>) +
                                      dr::log(1 / x) + 1 / (8 * x)),
                          dr::log(I0(x)));
    }

    static Float Mp(Float cosThetaO, Float cosThetaI, Float sinThetaO,
                    Float sinThetaI, Float v) {
        Float a  = cosThetaI * cosThetaO / v;
        Float b  = sinThetaI * sinThetaO / v;
        Float mp = dr::select(
            v <= .1f,
            (dr::exp(LogI0(a) - b - 1 / v + 0.6931f + dr::log(1 / (2 * v)))),
            (dr::exp(-b) * I0(a)) / (dr::sinh(1 / v) * 2 * v));
        return mp;
    }

    static inline Float Logistic(Float x, Float s) {
        x = dr::abs(x);
        return dr::exp(-x / s) / (s * dr::sqr(1 + dr::exp(-x / s)));
    }

    static inline Float LogisticCDF(Float x, Float s) {
        return 1 / (1 + dr::exp(-x / s));
    }

    static inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) {
        // a should be smaller than b
        return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
    }

    static inline Float angleMap(Float dphi){
        // map angle to [-pi, pi]
        Float pi    = dr::Pi<Float>;
        Float angle = dr::fmod(dphi, 2 * pi);
        angle       = dr::select(angle < -pi, angle + 2 * pi, angle);
        angle       = dr::select(angle > pi, angle - 2 * pi, angle);
        return angle;
    }

    static Float Np(Float phi, int p, Float s, Float gammaI, Float gammaT) {
        Float Phi  = 2 * p * gammaT - 2 * gammaI + p * dr::Pi<ScalarFloat>;
        Float dphi = phi - Phi;

        // map dphi to [-pi, pi]
        dphi = angleMap(dphi);
        return TrimmedLogistic(dphi, s, -dr::Pi<Float>, dr::Pi<Float>);
    }


    static dr::Array<Spectrum, pMax + 1> Ap(Float cosThetaI, Float eta, Float h,
                                            const Spectrum &T) {
        dr::Array<Spectrum, pMax + 1> ap;
        
        // Compute $p=0$ attenuation at initial cylinder intersection
        Float cosGammaI = dr::safe_sqrt(1 - h * h);
        Float cosTheta  = cosThetaI * cosGammaI;

        // Suppose the external is air
        Float f         = std::get<0>(fresnel(cosTheta, eta));
        ap[0]           = f;

        // Compute $p=1$ attenuation term
        ap[1] = dr::sqr(1 - f) * T;

        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p)
            ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of
        // scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);
        return ap;
    }

    dr::Array<Float, pMax + 1> ComputeApPdf(Float cosThetaI,
                                            const SurfaceInteraction3f &si,
                                            Mask active) const {

        Float h = -1 + 2 * si.uv[1];

        // Compute array of A_p values for cosThetaI
        Float sinThetaI = dr::safe_sqrt(1 - cosThetaI * cosThetaI);

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T =
            dr::exp(-m_sigma_a->eval(si, active) * (2 * cosGammaT / cosThetaT));

        // Calculate Ap
        Spectrum ap[pMax + 1];
        Float cosGammaI = dr::safe_sqrt(1 - h * h);
        Float cosTheta  = cosThetaI * cosGammaI;

        Float f = std::get<0>(fresnel(cosTheta, Float(m_eta)));

        ap[0] = f;
        // Compute $p=1$ attenuation termap.y()
        ap[1] = dr::sqr(1 - f) * T;
        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p)
            ap[p] = ap[p - 1] * T * f;
        // Compute attenuation term accounting for remaining orders of
        // scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);

        // Compute $A_p$ PDF from individual $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf;
        Float sumY = Float(0);

        for (int i = 0; i <= pMax; i++) {
            sumY = sumY + ap[i].y();
        }

        for (int i = 0; i <= pMax; ++i) {
            apPdf[i] = ap[i].y() / sumY;
        }
        return apPdf;
    }

    static Float SampleTrimmedLogistic(Float u, Float s, Float a, Float b) {
        // a should be smaller than b
        Float k = LogisticCDF(b, s) - LogisticCDF(a, s);
        Float x = -s * dr::log(1 / (u * k + LogisticCDF(a, s)) - 1);
        return dr::clamp(x, a, b);
    }

};

MI_IMPLEMENT_CLASS_VARIANT(Hair, BSDF)
MI_EXPORT_PLUGIN(Hair, "Hair material")
NAMESPACE_END(mitsuba)
