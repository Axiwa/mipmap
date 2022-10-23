import pytest
import drjit as dr
import mitsuba as mi
from mitsuba.python.util import traverse

def test01_create():
    mi.set_variant('llvm_ad_rgb')
    b = mi.load_dict({'type': 'hair'})
    assert b is not None
    assert b.component_count() == 1
    assert b.flags(0) == (mi.BSDFFlags.Glossy | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide | mi.BSDFFlags.NonSymmetric)
    assert b.flags() == b.flags(0)
    params = traverse(b)
    print (params)


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
#         beta_n = 0.1
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


# def test03_white_furnace_importance_sample(variant_scalar_rgb):
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
#         beta_n = 0.1
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
#                 pdf = bsdf.pdf(ctx, si, bs.wo)
#                 sum += spec
#                 count -= 1
#             avg = sum.y / (total)
#             assert dr.allclose(avg, 1, rtol = 0.01)
#             beta_n += 0.2
#         beta_m += 0.2


# def test04_sample_numeric(variants_vec_backends_once_rgb):
#     mi.set_variant('llvm_ad_rgb')
#     sampler = mi.load_dict({'type': 'independent', 'sample_count': 10000})
#     sampler.seed(seed = 0, wavefront_size = 10000)

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 128
#     while(beta_m < 1):
#         beta_n = 0.1
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
            
#             sigma_a = 0.
#             si.uv = [0, sampler.next_1d()] 
#             bsdf = mi.load_dict({        
#                 'type': 'hair',
#                 'sigma_a': sigma_a,
#                 'beta_m': beta_m,
#                 'beta_n': beta_n,
#                 'alpha': 0.,
#                 'eta': 1.55,
#             })
#             si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#             bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
#             dr.eval(bs, spec)
#             assert dr.allclose(spec.y, 1, rtol = 0.001)

#             beta_n += 0.2
#         beta_m += 0.2
    


# def test04_sample_pdf(variants_vec_backends_once_rgb):
#     mi.set_variant('llvm_ad_rgb')
#     sampler = mi.load_dict({'type': 'independent', 'sample_count': 10000})
#     sampler.seed(seed = 2, wavefront_size = 10000)

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 30000
#     while(beta_m < 1):
#         beta_n = 0.1
#         while(beta_n < 1):
#             # estimate reflected uniform incident radiance from hair
#             si.uv = [0, sampler.next_1d()]
#             sigma_a = 0.25
#             bsdf = mi.load_dict({        
#                 'type': 'hair',
#                 'sigma_a': sigma_a,
#                 'beta_m': beta_m,
#                 'beta_n': beta_n,
#                 'alpha': 2.,
#                 'eta': 1.55
#                 })
#             si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#             bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())

#             pdf = bsdf.pdf(ctx, si, bs.wo)
#             eval = bsdf.eval(ctx, si, bs.wo)
#             eval_pdf = bsdf.eval_pdf(ctx, si, bs.wo)

#             assert dr.allclose(pdf, bs.pdf, rtol = 0.005)
#             assert dr.allclose(eval, eval_pdf[0], rtol = 0.001)
#             assert dr.allclose(pdf, eval_pdf[1], rtol = 0.001)
#             assert dr.allclose(spec * pdf , eval, rtol = 0.001)

#             beta_n += 0.2
#         beta_m += 0.2

# def test05_sampleConsistency():
#     mi.set_variant('llvm_ad_rgb')
#     def helper_Li(spec):
#         return spec.z * spec.z

#     sampler = mi.load_dict({'type': 'independent', 'sample_count': 128 * 1024})
#     sampler.seed(seed = 0, wavefront_size = 128 * 1024)

#     si    = mi.SurfaceInteraction3f()
#     si.p  = [0, 0, 0]
#     si.n  = [0, 0, 1]
#     si.sh_frame = mi.Frame3f(si.n)
#     si.uv = [0, sampler.next_1d()] # h = 1

#     ctx = mi.BSDFContext()

#     beta_m = 0.1
#     beta_n = 0.1
#     total = 64 * 1024
#     while(beta_m < 1):
#         beta_n = 0.1
#         while(beta_n < 1):

#             si.wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
#             # estimate reflected uniform incident radiance from hair
#             sigma_a = 0.25
#             fImportance = 0.
#             fUniform = 0.
            
#             bsdf = mi.load_dict({        
#                 'type': 'hair',
#                 'sigma_a': sigma_a,
#                 'beta_m': beta_m,
#                 'beta_n': beta_n,
#                 'alpha': 2.,
#                 'eta': 1.55,
#                 })

#             bs, spec = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
            
#             fImportance = spec * helper_Li(bs.wo) / total

#             wo = mi.warp.square_to_uniform_sphere(sampler.next_2d())

#             fUniform = bsdf.eval(ctx, si, wo) * helper_Li(wo) / (total / dr.pi / 4)

#             assert dr.allclose(dr.sum(fImportance.y), dr.sum(fUniform.y), rtol=0.05)

#             beta_n += 0.2
#         beta_m += 0.2    



from mitsuba.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain


def BSDFAdapterUV(bsdf_type, extra, u = 0.5, wi=[0, 0, 1], ctx=None):
    if ctx is None:
        ctx = mi.BSDFContext()

    def make_context(n):
        si = dr.zeros(mi.SurfaceInteraction3f, n)
        si.p  = [0, 0, 0]
        si.n  = [0, 0, 1]
        si.sh_frame = mi.Frame3f(si.n)
        si.uv = [0, u]
        si.wi = wi
        return (si, ctx)

    def instantiate(args):
        xml = """<bsdf version="3.0.0" type="%s">
            %s
        </bsdf>""" % (bsdf_type, extra)
        return mi.load_string(xml % args)

    def sample_functor(sample, *args):
        n = dr.width(sample)
        plugin = instantiate(args)
        (si, ctx) = make_context(n)
        bs, weight = plugin.sample(ctx, si, sample[0], [sample[1], sample[2]])

        w = dr.full(mi.Float, 1.0, dr.width(weight))
        w[dr.all(dr.eq(weight, 0))] = 0
        
        return bs.wo, w

    def pdf_functor(wo, *args):
        n = dr.width(wo)
        plugin = instantiate(args)
        (si, ctx) = make_context(n)
        return plugin.pdf(ctx, si, wo)

    return sample_functor, pdf_functor


def test06_chi2():
    mi.set_variant('llvm_ad_rgb')
    sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
    sampler.seed(seed = 0, wavefront_size = 1)
    sigma_a = 0
    alpha = 0
    eta = 1.55

    beta_m = 0.6
    beta_n = 0.8

    while(beta_m <= 1):
        beta_n = 0.8
        while(beta_n <= 1):
            xml = f"""<float name="alpha" value="{alpha}" />
                    <rgb name="sigma_a" value="{sigma_a}"/>
                    <float name="beta_m" value="{beta_m}" />
                    <float name="beta_n" value="{beta_n}" />
                    <float name="eta" value="{eta}" />
                """
            wi = mi.warp.square_to_uniform_sphere(sampler.next_2d())
            u = 0.25 + 0.5 * sampler.next_1d()
            sample_func, pdf_func = BSDFAdapterUV("hair", xml, u=u, wi=wi)

            chi2 = ChiSquareTest(
                domain=SphericalDomain(),
                sample_func=sample_func,
                pdf_func=pdf_func,
                sample_dim = 3,
                res = 256,
                ires = 16,
                seed = 4,
            )
            assert chi2.run()

            beta_n += 0.2
        beta_m += 0.2
