import pytest
import drjit as dr
import mitsuba as mi

def test01_create(variant_scalar_rgb):
    mi.set_variant('cuda_ad_rgb')
    b = mi.load_dict({'type': 'diffuse'})
    assert b is not None
    assert b.component_count() == 1
    assert b.flags(0) == mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
    assert b.flags() == b.flags(0)


def test02_eval_pdf(variant_scalar_rgb):
    bsdf = mi.load_dict({'type': 'diffuse'})

    si    = mi.SurfaceInteraction3f()
    si.p  = [0, 0, 0]
    si.n  = [0, 0, 1]
    si.wi = [0, 0, 1]
    si.sh_frame = mi.Frame3f(si.n)

    ctx = mi.BSDFContext()

    for i in range(20):
        theta = i / 19.0 * (dr.pi / 2)
        wo = [dr.sin(theta), 0, dr.cos(theta)]

        v_pdf  = bsdf.pdf(ctx, si, wo=wo)
        v_eval = bsdf.eval(ctx, si, wo=wo)[0]
        assert dr.allclose(v_pdf, wo[2] / dr.pi)
        assert dr.allclose(v_eval, 0.5 * wo[2] / dr.pi)

        v_eval_pdf = bsdf.eval_pdf(ctx, si, wo=wo)
        assert dr.allclose(v_eval, v_eval_pdf[0])
        assert dr.allclose(v_pdf, v_eval_pdf[1])


def test03_chi2(variants_vec_backends_once_rgb):
    from mitsuba.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = BSDFAdapter("diffuse", '')

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3
    )

    assert chi2.run()


def helper_Li(spec):
    return spec.z * spec.z

def test05_sampleConsistency(variant_scalar_rgb):
    sampler = mi.load_dict({'type': 'independent'})

    si    = mi.SurfaceInteraction3f()
    si.p  = [0, 0, 0]
    si.n  = [0, 0, 1]
    si.sh_frame = mi.Frame3f(si.n)
    si.uv = [0, 1] # h = 1

    ctx = mi.BSDFContext()

    total = 64 * 1024
    si.wi = [0, 0, 1]
    # estimate reflected uniform incident radiance from hair
    count = total
    fImportance = 0.
    fUniform = 0.

    while(count > 0):
        h = -1 + 0.2 * sampler.next_1d()
        
        bsdf = mi.load_dict({'type': 'diffuse'})

        bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
        
        fImportance += spec * helper_Li(bs.wo) / total

        wo = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

        fUniform += bsdf.eval(ctx, si, wo) * helper_Li(wo) / (total / 2 / dr.pi)

        count -= 1
    ans = dr.abs(fImportance - fUniform)
    assert dr.allclose(fImportance, fUniform, rtol=0.05)