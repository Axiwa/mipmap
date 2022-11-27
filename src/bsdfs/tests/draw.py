import drjit as dr
import mitsuba as mi
from mitsuba.python.util import traverse

mi.set_variant('llvm_ad_rgb')

theta_i = 0

phi_i = 0

phi_o = dr.linspace(mi.Float, -dr.pi, dr.pi, 1000)

theta_o = dr.linspace(mi.Float, -dr.pi/2, dr.pi/2, 1000)

si    = mi.SurfaceInteraction3f()
si.p  = [0, 0, 0]
si.n  = [0, 0, 1]
si.sh_frame = mi.Frame3f(si.n)
si.uv = [0, 0.5] # h = 1

ctx = mi.BSDFContext()

beta_m = 0.2
beta_n = 0.4


x = dr.sin(theta_i)
y = dr.cos(theta_i) * dr.cos(phi_i)
z = dr.cos(theta_i) * dr.sin(phi_i)

x_o = dr.sin(theta_o)
y_o = dr.cos(theta_o) * dr.cos(phi_o)
z_o = dr.cos(theta_o) * dr.sin(phi_o)

si.wi = dr.normalize(mi.ScalarVector3f(x, y, z))
sigma_a = 0.

bsdf = mi.load_dict({        
    'type': 'hair',
    'sigma_a': sigma_a,
    'beta_m': beta_m,
    'beta_n': beta_n,
    'alpha': 0.,
    'eta': 1.55,
    })

wo = dr.normalize(mi.ScalarVector3f(x_o, y_o, z_o))

spec = bsdf.eval(ctx, si, wo)