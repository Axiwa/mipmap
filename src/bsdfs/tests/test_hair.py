import pytest
import drjit as dr
import mitsuba as mi
from mitsuba.python.util import traverse

# def test01_create():
#     mi.set_variant('cuda_ad_rgb')
#     b = mi.load_dict({'type': 'hair'})
#     assert b is not None
#     assert b.component_count() == 1
#     assert b.flags(0) == (mi.BSDFFlags.Glossy | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide | mi.BSDFFlags.NonSymmetric)
#     assert b.flags() == b.flags(0)
#     params = traverse(b)
#     print (params)


# def test02_white_furnace(variant_scalar_rgb):
#     sampler = mi.load_dict({'type': 'independent'})

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#     si.sh_frame = mi.Frame3f(si.n)
    
#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 300000
#     while(beta_m < 1):
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
#             count = total
#             sum = 0.
#             while(count > 0):
#                 si.uv = [0, sampler.next_1d()]
#                 sigma_a = 0.
#                 bsdf = mi.load_dict({        
#                     'type': 'hair',
#                     'sigma_a': sigma_a,
#                     'beta_m': beta_m,
#                     'beta_n': beta_n,
#                     'alpha': 0.,
#                     'eta': 1.55
#                 })

#                 wo = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#                 sum += bsdf.eval(ctx, si, wo)
#                 count -= 1

#             avg = sum.y / (total * mi.warp.square_to_uniform_sphere_pdf(1.))
#             assert dr.allclose(avg, 1, rtol = 0.05)
#             beta_n += 0.2
#         beta_m += 0.2


# def test03_white_furnace_sample(variant_scalar_rgb):
#     sampler = mi.load_dict({'type': 'independent'})

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)
#     si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 100000
#     while(beta_m < 1):
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
#             count = total
#             sum = 0.
#             while(count > 0):
#                 si.uv = [0, sampler.next_1d()] 
#                 sigma_a = 0.
#                 bsdf = mi.load_dict({        
#                     'type': 'hair',
#                     'sigma_a': sigma_a,
#                     'beta_m': beta_m,
#                     'beta_n': beta_n,
#                     'alpha': 0.,
#                     'eta': 1.55,
#                 })
#                 bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
#                 sum += spec
#                 count -= 1
#             avg = sum.y / (total)
#             assert dr.allclose(avg, 1, rtol = 0.01)
#             beta_n += 0.2
#         beta_m += 0.2


# def test04_sample_numeric(variant_scalar_rgb):
#     sampler = mi.load_dict({'type': 'independent'})

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 10000
#     while(beta_m < 1):
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
#             count = total
#             while(count > 0):
#                 sigma_a = 0.
#                 si.uv = [0, sampler.next_1d()] 
#                 bsdf = mi.load_dict({        
#                     'type': 'hair',
#                     'sigma_a': sigma_a,
#                     'beta_m': beta_m,
#                     'beta_n': beta_n,
#                     'alpha': 0.,
#                     'eta': 1.55,
#                 })
#                 si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#                 bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
#                 assert dr.allclose(spec.y, 1, rtol = 0.001)
#                 count -= 1
#             beta_n += 0.2
#         beta_m += 0.2
    


# def test04_sample_pdf(variant_scalar_rgb):
#     sampler = mi.load_dict({'type': 'independent'})

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 30000
#     while(beta_m < 1):
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
#             count = total
#             while(count > 0):
#                 si.uv = [0, sampler.next_1d()]
#                 sigma_a = 0.25
#                 bsdf = mi.load_dict({        
#                     'type': 'hair',
#                     'sigma_a': sigma_a,
#                     'beta_m': beta_m,
#                     'beta_n': beta_n,
#                     'alpha': 0.,
#                     'eta': 1.55
#                     })
#                 si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#                 bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())

#                 pdf = bsdf.pdf(ctx, si, bs.wo)
#                 eval = bsdf.eval(ctx, si, bs.wo)
#                 eval_pdf = bsdf.eval_pdf(ctx, si, bs.wo)

#                 assert dr.allclose(pdf, bs.pdf, rtol = 0.005)
#                 assert dr.allclose(eval, eval_pdf[0], rtol = 0.001)
#                 assert dr.allclose(pdf, eval_pdf[1], rtol = 0.001)
#                 count -= 1
#             beta_n += 0.2
#         beta_m += 0.2



# def test05_sampleConsistency(variant_scalar_rgb):

#     def helper_Li(spec):
#         return spec.z * spec.z

#     sampler = mi.load_dict({'type': 'independent'})

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)
#     si.uv = [0, 1] # h = 1

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 64 * 1024
#     while(beta_m < 1):
#         while(beta_n < 1):
#             si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#             # estimate reflected uniform incident radiance from hair
#             count = total
#             sigma_a = 0.25
#             fImportance = 0.
#             fUniform = 0.

#             while(count > 0):
#                 h = -1 + 0.2 * sampler.next_1d()
                
#                 bsdf = mi.load_dict({        
#                     'type': 'hair',
#                     'sigma_a': sigma_a,
#                     'beta_m': beta_m,
#                     'beta_n': beta_n,
#                     'alpha': 0.,
#                     'eta': 1.55
#                     })

#                 bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
                
#                 fImportance += spec * helper_Li(bs.wo) / total

#                 wo = mi.warp.square_to_uniform_sphere(sampler.next_2d())

#                 fUniform += bsdf.eval(ctx, si, wo) * helper_Li(wo) / (total / dr.pi / 4)

#                 count -= 1

#             assert dr.allclose(fImportance, fUniform, rtol=0.05)
#             beta_n += 0.2
#         beta_m += 0.2    



def test06_chi2(variants_vec_backends_once_rgb):
    from mitsuba.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    sample_func, pdf_func = BSDFAdapter("hair", '')

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    assert chi2.run()
