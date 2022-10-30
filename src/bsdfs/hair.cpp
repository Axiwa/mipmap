#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/fresnel.h>
#include <fstream>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Hair final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    Hair(const Properties &props) : Base(props) {
        m_sigma_a = props.texture<Texture>("sigma_a", 0.f);
        m_beta_m = props.get<ScalarFloat>("beta_m", 0.3f);
        m_beta_n = props.get<ScalarFloat>("beta_n", 0.3f);
        m_alpha = props.get<ScalarFloat>("alpha", 2.f);
        m_eta = props.get<ScalarFloat>("eta", 1.55f);


        m_components.push_back(BSDFFlags::Glossy | BSDFFlags::FrontSide | BSDFFlags::BackSide | BSDFFlags::NonSymmetric);

        m_flags = m_components[0];
        dr::set_attr(this, "flags", m_flags);

        // Preprocessing
        v[0] = dr::sqr(0.726f * m_beta_m + 0.812f * dr::sqr(m_beta_m) + 3.7f * dr::pow(m_beta_m, 20));
        v[1] = .25f * v[0];
        v[2] = 4 * v[0];
        for (int p = 3; p <= pMax; ++p)
            v[p] = v[2];

        // Compute azimuthal logistic scale factor from $\m_beta_n$
        s = SqrtPiOver8 *
            (0.265f * m_beta_n + 1.194f * dr::sqr(m_beta_n) + 5.372f * dr::pow(m_beta_n, 22));

        // CHECK(!std::isnan(s));

        // Compute $\m_alpha$ terms for hair scales
        sin2kAlpha[0] = dr::sin(dr::deg_to_rad(m_alpha));
        cos2kAlpha[0] = dr::safe_sqrt(1 - dr::sqr(sin2kAlpha[0]));
        for (int i = 1; i < 3; ++i) {
            sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
            cos2kAlpha[i] = dr::sqr(cos2kAlpha[i - 1]) - dr::sqr(sin2kAlpha[i - 1]);
        }
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("sigma_a", m_sigma_a.get(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_beta_m", m_beta_m, +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_beta_n", m_beta_n, +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_alpha", m_alpha, +ParamFlags::NonDifferentiable);
        callback->put_parameter("m_eta", m_eta, +ParamFlags::NonDifferentiable);
    }
    // for python test // differentiable -> empty


    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {

        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float _pdf = (Float)0;
        Vector3f wi = si.wi;
        Vector3f wo;
        Float h = -1 + 2 * si.uv[1];

        // Compute hair coordinate system terms related to _wo_
        Float sinThetaI = wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI = dr::atan2(wi.z(), wi.y());
        Float gammaI = dr::safe_asin(h);

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        Point2f u[2] = {DemuxFloat(sample1), sample2};

        // Determine which term $p$ to sample for hair scattering
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);


        // std::cout<<apPdf[0]<<" "<<apPdf[1]<<" "<<apPdf[2]<<" "<<apPdf[3]<<std::endl;

        Int32 p = (Int32)-1;
        ScalarInt32 i = 0;

        while(i < pMax){
            p = dr::select(u[0][0] >= apPdf[i], i, p);
            u[0][0] -= apPdf[i];
            i++;
        }
        p++;
        // std::cout<<dr::count(dr::eq(p, 0))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 1))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 2))<<std::endl;
        // std::cout<<dr::count(dr::eq(p, 3))<<std::endl;

        // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
        Float sinThetaIp = sinThetaI;
        Float cosThetaIp = cosThetaI;

        // if (p == 0) {
        sinThetaIp = dr::select(dr::eq(p ,0), sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1], sinThetaIp);
        cosThetaIp = dr::select(dr::eq(p ,0), cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1], cosThetaIp);
        // }
        // else if (p == 1) {
        sinThetaIp = dr::select(dr::eq(p ,1), sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0], sinThetaIp);
        cosThetaIp = dr::select(dr::eq(p ,1), cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0], cosThetaIp);
        // } 
        // else if (p == 2) {
        sinThetaIp = dr::select(dr::eq(p ,2), sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2], sinThetaIp);
        cosThetaIp = dr::select(dr::eq(p ,2), cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2], cosThetaIp);
        // } 

        // Sample $M_p$ to compute $\thetai$
        u[1][0] = dr::maximum(u[1][0], Float(1e-5));

        Float cosTheta =
            1 + v[pMax] * dr::log(u[1][0] + (1 - u[1][0]) * dr::exp(-2 / v[pMax]));
        for(int i = 0; i < pMax; i++){
            cosTheta = dr::select(dr::eq(p, i), 1 + v[i] * dr::log(u[1][0] + (1 - u[1][0]) * dr::exp(-2 / v[i])), cosTheta);
        }

        Float sinTheta = dr::safe_sqrt(1 - dr::sqr(cosTheta));
        Float cosPhi = dr::cos(2 * dr::Pi<ScalarFloat> * u[1][1]);
        Float sinThetaO = -cosTheta * sinThetaIp + sinTheta * cosPhi * cosThetaIp;
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));

        // Sample $N_p$ to compute $\Delta\phi$

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float gammaT = dr::safe_asin(sinGammaT);
        Float dphi;
        Float Phi = 2 * p * gammaT - 2 * gammaI + p * dr::Pi<ScalarFloat>;

        // if (p < pMax)
        //     dphi =
        //         Phi + SampleTrimmedLogistic(u[0][1], s, -dr::Pi<ScalarFloat>, dr::Pi<ScalarFloat>);
        // else
        //     dphi = 2 * dr::Pi<ScalarFloat> * u[0][1];
        dphi = dr::select(dr::neq(p, pMax), 
        Phi + SampleTrimmedLogistic(u[0][1], s, -dr::Pi<ScalarFloat>, dr::Pi<ScalarFloat>), 
        2 * dr::Pi<ScalarFloat> * u[0][1]);

        // Compute _wi_ from sampled hair scattering angles
        Float phiO = phiI + dphi;
        // std::cout<<phiO<<std::endl;
        wo = Vector3f(sinThetaO, cosThetaO * dr::cos(phiO),
                    cosThetaO * dr::sin(phiO));

        // Compute PDF for sampled hair scattering direction _wi_
        for (int i = 0; i < pMax; ++i) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
            Float sinThetaIp, cosThetaIp;
            if (i == 0) {
                sinThetaIp = sinThetaI * cos2kAlpha[1] - cosThetaI * sin2kAlpha[1];
                cosThetaIp = cosThetaI * cos2kAlpha[1] + sinThetaI * sin2kAlpha[1];
            }
            // Handle remainder of $p$ values for hair scale tilt
            else if (i == 1) {
                sinThetaIp = sinThetaI * cos2kAlpha[0] + cosThetaI * sin2kAlpha[0];
                cosThetaIp = cosThetaI * cos2kAlpha[0] - sinThetaI * sin2kAlpha[0];
            } else if (i == 2) {
                sinThetaIp = sinThetaI * cos2kAlpha[2] + cosThetaI * sin2kAlpha[2];
                cosThetaIp = cosThetaI * cos2kAlpha[2] - sinThetaI * sin2kAlpha[2];
            } else {
                sinThetaIp = sinThetaI;
                cosThetaIp = cosThetaI;
            }

            // Handle out-of-range $\cos \thetao$ from scale adjustment
            cosThetaIp = dr::abs(cosThetaIp);
            _pdf += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[i]) * apPdf[i] * Np(dphi, i, s, gammaI, gammaT);
        }

        _pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
                apPdf[pMax] * (1 / (2 * dr::Pi<ScalarFloat>));

        bs.wo = wo;
        bs.pdf = _pdf;
        bs.eta = 1.;
        bs.sampled_type = +BSDFFlags::Glossy;
        bs.sampled_component = 0;   

        std::ofstream f1("me.txt");  
        f1<<wo<<std::endl;
        f1.close();

        UnpolarizedSpectrum value = dr::select(bs.pdf > 0, eval(ctx, si, bs.wo, active) / bs.pdf, 0);
        
        return { bs, depolarizer<Spectrum>(value) & (active && bs.pdf > 0.f) };        
        
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // TODO
        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) && !ctx.is_enabled(BSDFFlags::GlossyReflection)){
            return 0.f;
        }

        // TODO: h should be related to si.uv
        Float h = -1 + 2 * si.uv[1];
        Float gammaI = dr::safe_asin(h);

        // CHECK(h >= -1 && h <= 1);
        // CHECK(m_beta_m >= 0 && m_beta_m <= 1);
        // CHECK(m_beta_n >= 0 && m_beta_n <= 1);
        // CHECK(pMax >= 3);
        

        // Compute the BSDF
        // Compute hair coordinate system terms related to _wo_
        // Here the coordinate system is different from others!!
        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO = dr::atan2(wo.z(), wo.y());

        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI = dr::atan2(si.wi.z(), si.wi.y());

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));
        Float gammaT = dr::safe_asin(sinGammaT);

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T = dr::exp(-m_sigma_a -> eval(si, active) * (2 * cosGammaT / cosThetaT));


        // Calculate Ap
        Spectrum ap[pMax + 1];
        Float cosGammaO = dr::safe_sqrt(1 - h * h);
        Float cosTheta = cosThetaI * cosGammaO;
        Float f = std::get<0>(fresnel(cosTheta, (Float)m_eta)); 

        ap[0] = f;
        // Compute $p=1$ attenuation term
        ap[1] = dr::sqr(1 - f) * T;
        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p) ap[p] = ap[p - 1] * T * f;
        // Compute attenuation term accounting for remaining orders of scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);

        // Evaluate hair BSDF
        Float phi = phiO - phiI;
        Spectrum fsum(0.);
        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
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
            fsum += 
                Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) * ap[p] *
                Np(phi, p, s, gammaI, gammaT);
        }

        // Compute contribution of remaining terms after _pMax_
        fsum += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) * ap[pMax] /
                (2.f * dr::Pi<ScalarFloat>);

        // CHECK(!std::isinf(fsum.y()) && !std::isnan(fsum.y()));
        return depolarizer<Spectrum>(fsum) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) && !ctx.is_enabled(BSDFFlags::GlossyReflection)){
            return 0.f;
        }

        Float h = -1 + 2 * si.uv[1];

        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO = dr::atan2(wo.z(), wo.y());

        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI = dr::atan2(si.wi.z(), si.wi.y());
        Float gammaI = dr::safe_asin(h);

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float gammaT = dr::safe_asin(sinGammaT);

        // Compute PDF for $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);

        // Compute PDF sum for hair scattering events
        Float phi = phiO - phiI;
        Float _pdf = (Float)0;

        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
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
            _pdf += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) *
                apPdf[p] * Np(phi, p, s, gammaI, gammaT);
        }
        _pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
            apPdf[pMax] * (1 / (2 * dr::Pi<ScalarFloat>));
        return _pdf;
    }


    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::GlossyTransmission) && !ctx.is_enabled(BSDFFlags::GlossyReflection)){
            return { 0.f, 0.f };
        }

        Float h = -1 + 2 * si.uv[1];

        Float sinThetaO = wo.x();
        Float cosThetaO = dr::safe_sqrt(1 - dr::sqr(sinThetaO));
        Float phiO = dr::atan2(wo.z(), wo.y());
        
        // Compute hair coordinate system terms related to _wi_
        Float sinThetaI = si.wi.x();
        Float cosThetaI = dr::safe_sqrt(1 - dr::sqr(sinThetaI));
        Float phiI = dr::atan2(si.wi.z(), si.wi.y());
        Float gammaI = dr::safe_asin(h);

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));
        Float gammaT = dr::safe_asin(sinGammaT);

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute PDF for $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf = ComputeApPdf(cosThetaI, si, active);

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T = dr::exp(-m_sigma_a -> eval(si, active) * (2 * cosGammaT / cosThetaT));

        // Calculate Ap
        Spectrum ap[pMax + 1];
        Float cosGammaO = dr::safe_sqrt(1 - h * h);
        Float cosTheta = cosThetaI * cosGammaO;

        // TODO: ?
        Float f = std::get<0>(fresnel(cosTheta, (Float)m_eta)); 

        ap[0] = f;
        // Compute $p=1$ attenuation term
        ap[1] = dr::sqr(1 - f) * T;
        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p) ap[p] = ap[p - 1] * T * f;
        // Compute attenuation term accounting for remaining orders of scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);

        // Compute PDF sum for hair scattering events
        Float phi = phiO - phiI;
        Float pdf = (Float)0;
        Spectrum fsum(0.);

        for (int p = 0; p < pMax; ++p) {
            // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
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

            pdf += Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) *
                apPdf[p] * Np(phi, p, s, gammaI, gammaT);
            fsum += 
                Mp(cosThetaO, cosThetaIp, sinThetaO, sinThetaIp, v[p]) * ap[p] *
                Np(phi, p, s, gammaI, gammaT);
        }
        pdf += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) *
            apPdf[pMax] * (1 / (2 * dr::Pi<ScalarFloat>));
        // Compute contribution of remaining terms after _pMax_
        fsum += Mp(cosThetaO, cosThetaI, sinThetaO, sinThetaI, v[pMax]) * ap[pMax] /
                (2.f * dr::Pi<ScalarFloat>);
        
        return {depolarizer<Spectrum>(fsum) & active, pdf};
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Hair[" << std::endl
            // << "  reflectance = " << string::indent(m_reflectance) << std::endl
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
    ref<Texture> m_reflectance;
    ScalarFloat m_beta_m, m_beta_n, m_alpha;
    ScalarFloat m_eta;

    // Float enumlanin, pheomelanin; // color

    // Helper function
    static Float I0(Float x) {
        Float val = 0;
        Float x2i = 1;
        int64_t ifact = 1;
        int i4 = 1;
        // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
        for (int i = 0; i < 10; ++i) {
            if (i > 1) ifact *= i;
            val += x2i / (i4 * dr::sqr(ifact));
            x2i *= x * x;
            i4 *= 4;
        }
        return val;
    }

    static Float LogI0(Float x) {
        return dr::select(
            x > 12,
            x + 0.5f * (-dr::log(2 * dr::Pi<ScalarFloat>) + dr::log(1 / x) + 1 / (8 * x)),
            dr::log(I0(x))
        );
    }

    static Float Mp(Float cosThetaI, Float cosThetaO, Float sinThetaI,
                    Float sinThetaO, Float v) {
        Float a = cosThetaI * cosThetaO / v;
        Float b = sinThetaI * sinThetaO / v;
        Float mp =
            dr::select(v <= .1f, 
                (dr::exp(LogI0(a) - b - 1 / v + 0.6931f + dr::log(1 / (2 * v)))),
                (dr::exp(-b) * I0(a)) / (dr::sinh(1 / v) * 2 * v)
            );
        // CHECK(!std::isinf(mp) && !std::isnan(mp));
        return mp;
    }

    static Float Np(Float phi, int p, Float s, Float gammaI, Float gammaT) {
        Float Phi = 2 * p * gammaT - 2 * gammaI + p * dr::Pi<ScalarFloat>;
        Float dphi = phi - Phi;

        dphi = angleMap(dphi);
        return TrimmedLogistic(dphi, s, -dr::Pi<Float>, dr::Pi<Float>);
    }

    static inline Float Logistic(Float x, Float s) {
        x = dr::abs(x);
        return dr::exp(-x / s) / (s * dr::sqr(1 + dr::exp(-x / s)));
    }

    static inline Float LogisticCDF(Float x, Float s) {
        return 1 / (1 + dr::exp(-x / s));
    }

    static inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) {
        // CHECK_LT(a, b);
        return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
    }

    static inline Float angleMap(Float dphi){
        // map angle to [-pi, pi]
        Float pi = dr::Pi<Float>;
        Float angle = dr::fmod(dphi, 2 * pi);
        angle = dr::select(angle < -pi, angle + 2 * pi, angle);
        angle = dr::select(angle > pi, angle - 2 * pi, angle);
        return angle;
    }

    static Point2f DemuxFloat(Float f){
        // CHECK(f >= 0 && f < 1);
        UInt64 v = (ScalarUInt64(4294967296)) * f;
        // CHECK_LT(v, 0x100000000);
        UInt32 bits[2] = {Compact1By1(v), Compact1By1(v>>1)};
        return Point2f(bits[0]/Float(1 << 16), bits[1]/Float(1 << 16));
    }

    static UInt32 Compact1By1(UInt32 x){
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x &= (UInt32)0x55555555;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x >> 1)) & (UInt32)0x33333333;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x >> 2)) & (UInt32)0x0f0f0f0f;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x >> 4)) & (UInt32)0x00ff00ff;
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x >> 8)) & (UInt32)0x0000ffff;
        return x;        
    }

    static dr::Array<Spectrum, pMax + 1> Ap(Float cosThetaO, Float m_eta, Float h,
                                            const Spectrum &T) {
        dr::Array<Spectrum, pMax + 1> ap;
        // Compute $p=0$ attenuation at initial cylinder intersection
        Float cosGammaO = dr::safe_sqrt(1 - h * h);
        Float cosTheta = cosThetaO * cosGammaO;
        Float f = std::get<0>(fresnel(cosTheta, (Float)m_eta));
        ap[0] = f;

        // Compute $p=1$ attenuation term
        ap[1] = dr::sqr(1 - f) * T;

        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p) ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);
        return ap;
    }

    dr::Array<Float, pMax + 1> ComputeApPdf(Float cosThetaI, const SurfaceInteraction3f &si, Mask active) const {
        Float h = -1 + 2 * si.uv[1];
        // Compute array of $A_p$ values for _cosThetaO_
        Float sinThetaI = dr::safe_sqrt(1 - cosThetaI * cosThetaI);

        // Compute $\cos \thetat$ for refracted ray
        Float sinThetaT = sinThetaI / m_eta;
        Float cosThetaT = dr::safe_sqrt(1 - dr::sqr(sinThetaT));

        // Compute $\gammat$ for refracted ray
        Float etap = dr::sqrt(m_eta * m_eta - dr::sqr(sinThetaI)) / cosThetaI;
        Float sinGammaT = h / etap;
        Float cosGammaT = dr::safe_sqrt(1 - dr::sqr(sinGammaT));

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T = dr::exp(-m_sigma_a -> eval(si, active) * (2 * cosGammaT / cosThetaT));

        // Calculate Ap
        Spectrum ap[pMax + 1];
        Float cosGammaI = dr::safe_sqrt(1 - h * h);
        Float cosTheta = cosThetaI * cosGammaI;

        // TODO: tuple.get<0>
        Float f = std::get<0>(fresnel(cosTheta, (Float)m_eta)); 

        ap[0] = f;
        // Compute $p=1$ attenuation termap.y()
        ap[1] = dr::sqr(1 - f) * T;
        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < pMax; ++p) ap[p] = ap[p - 1] * T * f;
        // Compute attenuation term accounting for remaining orders of scattering
        ap[pMax] = ap[pMax - 1] * f * T / (Spectrum(1.f) - T * f);

        // Compute $A_p$ PDF from individual $A_p$ terms
        dr::Array<Float, pMax + 1> apPdf;
        Float sumY = (Float)0;

        for(int i = 0; i<=pMax; i++){
            sumY = sumY + ap[i].y();
        }

        for (int i = 0; i <= pMax; ++i){
            apPdf[i] = ap[i].y() / sumY;
        } 
        return apPdf;
    }

    static Float SampleTrimmedLogistic(Float u, Float s, Float a, Float b) {
        // CHECK_LT(a, b);
        Float k = LogisticCDF(b, s) - LogisticCDF(a, s);
        Float x = -s * dr::log(1 / (u * k + LogisticCDF(a, s)) - 1);
        // CHECK(!std::isnan(x));
        return dr::clamp(x, a, b);
    }

};


MI_IMPLEMENT_CLASS_VARIANT(Hair, BSDF)
MI_EXPORT_PLUGIN(Hair, "Hair material")
NAMESPACE_END(mitsuba)
