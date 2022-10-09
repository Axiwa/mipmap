import pytest
import drjit as dr
import mitsuba as mi
from mitsuba.python.util import traverse

def test01_create(variant_scalar_rgb):
    b = mi.load_dict({'type': 'hair'})
    assert b is not None
    assert b.component_count() == 2
    assert b.flags(0) == (mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide)
    assert b.flags(1) == (mi.BSDFFlags.GlossyTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide | mi.BSDFFlags.NonSymmetric)
    assert b.flags() == b.flags(0) | b.flags(1)
    params = traverse(b)


def test02_eval_pdf(variant_scalar_rgb):
    sampler = mi.load_dict({'type': 'independent'})

    si    = mi.SurfaceInteraction3f()
    si.p  = [0, 0, 0]
    si.n  = [0, 0, 1]
    si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
    si.sh_frame = mi.Frame3f(si.n)
    si.uv = [0, 1] # h = 1

    ctx = mi.BSDFContext()

    beta_m = 0.1
    beta_n = 0.1
    total = 10
    while(beta_m < 1):
        while(beta_n < 1):
            # estimate reflected uniform incident radiance from hair
            count = total
            sum = 0.
            while(count > 0):
                h = -1 + 0.2 * sampler.next_1d()
                sigma_a = 0.
                bsdf = mi.load_dict({        
                    'type': 'hair',
                    'sigma_a': sigma_a,
                    'beta_m': beta_m,
                    'beta_n': beta_n,
                    'alpha': 0.,
                    'eta': 1.55,
                    'h': h})
                wo = mi.warp.square_to_uniform_sphere(sampler.next_2d())
                sum += bsdf.eval(ctx, si, wo)
                count -= 1
            beta_n += 0.2
        beta_m += 0.2
    avg = sum[1] / (total * mi.warp.square_to_uniform_sphere_pdf(1.))

    assert dr.allclose(avg, 1, atol=0.01)

# def test03_chi2(variants_vec_backends_once_rgb):
#     from mitsuba.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

#     sample_func, pdf_func = BSDFAdapter("diffuse", '')

#     chi2 = ChiSquareTest(
#         domain=SphericalDomain(),
#         sample_func=sample_func,
#         pdf_func=pdf_func,
#         sample_dim=3
#     )

    # assert chi2.run()
