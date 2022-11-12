import mitsuba as mi
import drjit as dr
import time
import numpy as np
import gc
from memory_profiler import profile

from scipy import integrate

class ChiSquareTest:
    """
    Implements Pearson's chi-square test for goodness of fit of a distribution
    to a known reference distribution.

    The implementation here specifically compares a Monte Carlo sampling
    strategy on a 2D (or lower dimensional) space against a reference
    distribution obtained by numerically integrating a probability density
    function over grid in the distribution's parameter domain.

    Parameter ``domain`` (object):
       An implementation of the domain interface (``SphericalDomain``, etc.),
       which transforms between the parameter and target domain of the
       distribution

    Parameter ``sample_func`` (function):
       An importance sampling function which maps an array of uniform variates
       of size ``[sample_dim, sample_count]`` to an array of ``sample_count``
       samples on the target domain.

    Parameter ``pdf_func`` (function):
       Function that is expected to specify the probability density of the
       samples produced by ``sample_func``. The test will try to collect
       sufficient statistical evidence to reject this hypothesis.

    Parameter ``sample_dim`` (int):
       Number of random dimensions consumed by ``sample_func`` per sample. The
       default value is ``2``.

    Parameter ``sample_count`` (int):
       Total number of samples to be generated. The test will have more
       evidence as this number tends to infinity. The default value is
       ``1000000``.

    Parameter ``res`` (int):
       Vertical resolution of the generated histograms. The horizontal
       resolution will be calculated as ``res * domain.aspect()``. The
       default value of ``101`` is intentionally an odd number to prevent
       issues with floating point precision at sharp boundaries that may
       separate the domain into two parts (e.g. top hemisphere of a sphere
       parameterization).

    Parameter ``ires`` (int):
       Number of horizontal/vertical subintervals used to numerically integrate
       the probability density over each histogram cell (using the trapezoid
       rule). The default value is ``4``.

    Parameter ``seed`` (int):
       Seed value for the PCG32 random number generator used in the histogram
       computation. The default value is ``0``.

    Notes:

    The following attributes are part of the public API:

    messages: string
        The implementation may generate a number of messages while running the
        test, which can be retrieved via this attribute.

    histogram: array
        The histogram array is populated by the ``tabulate_histogram()`` method
        and stored in this attribute.

    pdf: array
        The probability density function array is populated by the
        ``tabulate_pdf()`` method and stored in this attribute.

    p_value: float
        The p-value of the test is computed in the ``run()`` method and stored
        in this attribute.
    """
    
    def __init__(self, domain, sample_func, pdf_func, sample_dim=2,
                 sample_count=1000000, res=101, ires=4, seed=0):

        assert res > 0
        assert ires >= 2, "The 'ires' parameter must be >= 2!"

        self.domain = domain
        self.sample_func = sample_func
        self.pdf_func = pdf_func
        self.sample_dim = sample_dim
        self.sample_count = sample_count
        if domain.aspect() is None:
            self.res = mi.ScalarVector2u(res, 1)
        else:
            self.res = dr.maximum(mi.ScalarVector2u(
                int(res / domain.aspect()), res), 1)
        self.ires = ires
        self.seed = seed
        self.bounds = domain.bounds()
        self.pdf = None
        self.another_pdf = None
        self.histogram = None
        self.p_value = None
        self.messages = ''
        self.fail = False

    def tabulate_histogram(self):
        """
        Invoke the provided sampling strategy many times and generate a
        histogram in the parameter domain. If ``sample_func`` returns a tuple
        ``(positions, weights)`` instead of just positions, the samples are
        considered to be weighted.
        """

        self.histogram_start = time.time()

        # Generate a table of uniform variates
        idx = dr.arange(mi.UInt64, self.sample_count)
        v0, v1 = mi.sample_tea_32(idx, self.seed)

        # Scramble seed and stream index using TEA, a linearly increasing sequence
        # does not produce a sufficiently statistically independent set of streams
        rng = mi.PCG32(initstate=v0, initseq=v1)

        samples_in = getattr(mi, 'Vector%if' % self.sample_dim)()
        for i in range(self.sample_dim):
            samples_in[i] = rng.next_float32() if mi.Float is mi.Float32 \
                else rng.next_float64()

        # Invoke sampling strategy
        samples_out = self.sample_func(samples_in)

        if type(samples_out) is tuple:
            weights_out = samples_out[1]
            samples_out = samples_out[0]
            assert dr.depth_v(weights_out) == 1
        else:
            weights_out = mi.Float(1.0)

        # Map samples into the parameter domain
        xy = self.domain.map_backward(samples_out)

        # Sanity check
        eps = self.bounds.extents() * 1e-4
        in_domain = dr.all((xy >= self.bounds.min - eps) &
                           (xy <= self.bounds.max + eps))
        if not dr.all(in_domain):
            self._log('Encountered samples outside of the specified '
                      'domain!')
            self.fail = True

        # Normalize position values
        xy = (xy - self.bounds.min) / self.bounds.extents()
        xy = mi.Vector2u(dr.clamp(xy * mi.Vector2f(self.res), 0,
                                  mi.Vector2f(self.res - 1)))

        # Compute a histogram of the positions in the parameter domain
        self.histogram = dr.zeros(mi.Float, dr.prod(self.res))

        dr.scatter_reduce(
            dr.ReduceOp.Add,
            self.histogram,
            weights_out,
            xy.x + xy.y * self.res.x,
        )

        histogram_min = dr.min(self.histogram)

        if not histogram_min[0] >= 0:
            self._log('Encountered a cell with negative sample '
                      'weights: %f' % histogram_min)
            self.fail = True

        self.histogram_sum = dr.sum(self.histogram) / self.sample_count
        if self.histogram_sum[0] > 1.1:
            self._log('Sample weights add up to a value greater '
                      'than 1.0: %f' % self.histogram_sum[0])
            self.fail = True

        self.histogram_end = time.time()

    def tabulate_pdf(self):
        """
        Numerically integrate the provided probability density function over
        each cell to generate an array resembling the histogram computed by
        ``tabulate_histogram()``. The function uses the trapezoid rule over
        intervals discretized into ``self.ires`` separate function evaluations.
        """

        self.pdf_start = time.time()

        # Determine total number of samples and construct an initial index
        sample_count    = self.ires**2
        cell_count      = self.res.x * self.res.y
        index           = dr.arange(mi.UInt32, cell_count * sample_count)
        extents         = self.bounds.extents()
        cell_size       = extents / self.res
        sample_spacing  = cell_size / (self.ires - 1)

        # Determine cell and integration sample indices
        cell_index      = index // sample_count
        sample_index    = index - cell_index * sample_count
        cell_y          = cell_index // self.res.x
        cell_x          = cell_index - cell_y * self.res.x
        sample_y        = sample_index // self.ires
        sample_x        = sample_index - sample_y * self.ires
        cell_index_2d   = mi.Vector2u(cell_x, cell_y)
        sample_index_2d = mi.Vector2u(sample_x, sample_y)

        # Compute the position of each sample
        p = self.bounds.min + cell_index_2d * cell_size
        p += (sample_index_2d + 1e-4) * (1-2e-4) * sample_spacing

        # Trapezoid rule integration weights
        weights = dr.prod(dr.select(dr.eq(sample_index_2d, 0) |
                                     dr.eq(sample_index_2d, self.ires - 1), 0.5, 1))
        weights *= dr.prod(sample_spacing) * self.sample_count

        # Remap onto the target domain
        p = self.domain.map_forward(p)

        # Evaluate the model density
        pdf = self.pdf_func(p)

        # Sum over each cell
        self.pdf = dr.block_sum(pdf * weights, sample_count)

        if len(self.pdf) == 1:
            dr.resize(self.pdf, dr.width(p))

        # A few sanity checks
        pdf_min = dr.min(self.pdf) / self.sample_count
        if not pdf_min[0] >= 0:
            self._log('Failure: Encountered a cell with a '
                      'negative PDF value: %f' % pdf_min)
            self.fail = True

        self.pdf_sum = dr.sum(self.pdf) / self.sample_count
        if self.pdf_sum[0] > 1.1:
            self._log('Failure: PDF integrates to a value greater '
                      'than 1.0: %f' % self.pdf_sum[0])
            self.fail = True

        self.pdf_end = time.time()

    # @profile
    def tabulate_another_pdf(self):
        # *************************  modified integration method: higher ires ***********************#
        # ires = 8
        # res = self.res

        # self.pdf_start = time.time()

        # # Determine total number of samples and construct an initial index
        # sample_count    = ires**2
        # cell_count      = self.res.x * self.res.y
        # index           = dr.arange(mi.UInt32, cell_count * sample_count)
        # extents         = self.bounds.extents()
        # cell_size       = extents / self.res
        # sample_spacing  = cell_size / (ires - 1)

        # # Determine cell and integration sample indices
        # cell_index      = index // sample_count
        # sample_index    = index - cell_index * sample_count
        # cell_y          = cell_index // self.res.x
        # cell_x          = cell_index - cell_y * self.res.x
        # sample_y        = sample_index // ires
        # sample_x        = sample_index - sample_y * ires
        # cell_index_2d   = mi.Vector2u(cell_x, cell_y)
        # sample_index_2d = mi.Vector2u(sample_x, sample_y)

        # # Compute the position of each sample
        # p = self.bounds.min + cell_index_2d * cell_size
        # p += (sample_index_2d + 1e-4) * (1-2e-4) * sample_spacing

        # # Trapezoid rule integration weights
        # weights = dr.prod(dr.select(dr.eq(sample_index_2d, 0) |
        #                              dr.eq(sample_index_2d, ires - 1), 0.5, 1))
        # weights *= dr.prod(sample_spacing) * self.sample_count

        # # Remap onto the target domain
        # p = self.domain.map_forward(p)

        # # Evaluate the model density
        # pdf = self.pdf_func(p)

        # # Sum over each cell
        # self.another_pdf = dr.block_sum(pdf * weights, sample_count)

        # if len(self.another_pdf) == 1:
        #     dr.resize(self.another_pdf, dr.width(p))

        # # A few sanity checks
        # pdf_min = dr.min(self.another_pdf) / self.sample_count
        # if not pdf_min[0] >= 0:
        #     self._log('Failure: Encountered a cell with a '
        #               'negative PDF value: %f' % pdf_min)
        #     self.fail = True

        # self.another_pdf_sum = dr.sum(self.another_pdf) / self.sample_count
        # if self.another_pdf_sum[0] > 1.1:
        #     self._log('Failure: PDF integrates to a value greater '
        #               'than 1.0: %f' % self.another_pdf_sum[0])
        #     self.fail = True

        # self.pdf_end = time.time()


        # *************************  modified integration method: adaptive ode solver ***********************#

        self.pdf_start = time.time()

        # Determine total number of samples and construct an initial index
        cell_count      = self.res.x * self.res.y
        extents         = self.bounds.extents()
        cell_size       = extents / self.res
        
        # Determine cell and integration sample indices
        cell_index      = dr.arange(mi.UInt32, cell_count)
        cell_y          = cell_index // self.res.x
        cell_x          = cell_index - cell_y * self.res.x
        cell_index_2d   = mi.Vector2u(cell_x, cell_y)

        # p and pmax is the region for integration
        pmin = self.bounds.min + cell_index_2d * cell_size
        pmax = pmin + cell_size

        # conversion from dr.array to scipy 
        pmin_num = pmin.numpy()
        pmax_num = pmax.numpy()

        ans = np.zeros(pmin_num.shape[0])
        # tmp = np.loadtxt("ans.txt")
        # ans[:120] = tmp

        print (ans.shape)

        # define the function for integration
        # sample_count is multiplied here to avoid truncation error
        def myfunc(y, x):
            wo = self.domain.map_forward(mi.Point2f(x, y))
            p = self.pdf_func(wo).numpy() * self.sample_count
            return p[0]

        for i in range(0, pmin_num.shape[0]):
            ans[i], err  = integrate.dblquad(myfunc, pmin_num[i][0], pmax_num[i][0], lambda x: pmin_num[i][1], lambda y: pmax_num[i][1], epsabs = 1) # epsabs = 1e-6, epsrel = 1e-6 
            gc.collect()
            with open("ans.txt", "a") as f:
                f.write("{}\n".format(ans[i]))
            # print (ans[i])
            # input()

        # Sum over each cell
        self.another_pdf = mi.Float(ans)

        if len(self.another_pdf) == 1:
            dr.resize(self.another_pdf, dr.width(pmin_num.shape[0]))

        # A few sanity checks
        pdf_min = dr.min(self.another_pdf)
        if not pdf_min[0] >= 0:
            self._log('Failure: Encountered a cell with a '
                      'negative PDF value: %f' % pdf_min)
            self.fail = True

        self.another_pdf_sum = dr.sum(mi.Float(ans) / self.sample_count)
        if self.another_pdf_sum[0] > 1.1:
            self._log('Failure: PDF integrates to a value greater '
                      'than 1.0: %f' % self.another_pdf_sum[0])
            self.fail = True

        self.pdf_end = time.time()


    def run(self, significance_level=0.01, test_count=1, quiet=False):
        """
        Run the Chi^2 test

        Parameter ``significance_level`` (float):
            Denotes the desired significance level (e.g. 0.01 for a test at the
            1% significance level)

        Parameter ``test_count`` (int):
            Specifies the total number of statistical tests run by the user.
            This value will be used to adjust the provided significance level
            so that the combination of the entire set of tests has the provided
            significance level.

        Returns → bool:
            ``True`` upon success, ``False`` if the null hypothesis was
            rejected.

        """
        import numpy as np
        if self.histogram is None:
            self.tabulate_histogram()

        if self.pdf is None:
            self.tabulate_pdf()

        if self.another_pdf is None:
            self.tabulate_another_pdf()

        index = mi.UInt32(np.array([i[0] for i in sorted(enumerate(self.pdf.numpy()),
                                                         key=lambda x: x[1])]))

        # Sort entries by expected frequency (increasing)
        pdf = mi.Float64(dr.gather(mi.Float, self.pdf, index))
        another_pdf = mi.Float64(dr.gather(mi.Float, self.another_pdf, index))
        histogram = mi.Float64(dr.gather(mi.Float, self.histogram, index))

        result = True

        # Compute chi^2 statistic and pool low-valued cells
        print ("Histogram vs. pdf")
        chi2val, dof, pooled_in, pooled_out = mi.math.chi2(histogram, pdf, 5)
        self.show(chi2val, dof, pooled_in, pooled_out, self.histogram, self.pdf, "hist", "pdf", test_count)

        print ("Histogram vs. High ires pdf")
        chi2val, dof, pooled_in, pooled_out = mi.math.chi2(histogram, another_pdf, 5)
        result = self.show(chi2val, dof, pooled_in, pooled_out, self.histogram, self.another_pdf, "hist", "high_ires", test_count)

        print ("High ires pdf vs. pdf")
        chi2val, dof, pooled_in, pooled_out = mi.math.chi2(another_pdf, pdf, 5)
        self.show(chi2val, dof, pooled_in, pooled_out, self.another_pdf, self.pdf, "high_ires", "pdf", test_count)

        return result



    def show(self, chi2val, dof, pooled_in, pooled_out, first, second, name1, name2, test_count, significance_level = 0.01):

        if dof < 1:
            self._log('Failure: The number of degrees of freedom is too low!')
            self.fail = True

        if pooled_in > 0:
            self._log('Pooled %i low-valued cells into %i cells to '
                      'ensure sufficiently high expected cell frequencies'
                      % (pooled_in, pooled_out))

        self._log('Chi^2 statistic = %f (d.o.f = %i)' % (chi2val, dof))

        self.p_value = 1 - mi.math.rlgamma(dof / 2, chi2val / 2)

        significance_level = 1.0 - \
            (1.0 - significance_level) ** (1.0 / test_count)

        if self.fail:
            self._log('Not running the test for reasons listed above. Target '
                      'density and histogram were written to "chi2_data.py')
            result = False

        elif self.p_value < significance_level \
                or not dr.isfinite(self.p_value):
            self._log('***** Rejected ***** the null hypothesis (p-value = %f,'
                      ' significance level = %f). Target density and histogram'
                      ' were written to "chi2_data.py".'
                      % (self.p_value, significance_level))
            result = False
        else:
            self._log('Accepted the null hypothesis (p-value = %f, '
                      'significance level = %f)' %
                      (self.p_value, significance_level))
            result = True

        print(self.messages)
        self.messages = ""
        # if not result:
            # self._dump_tables()
        self._dump_pdf(first, second, name1, name2)
        return result


    def _dump_tables(self):
        with open("chi2_data.py", "w") as f:
            pdf = str([[self.pdf[x + y * self.res.x]
                        for x in range(self.res.x)]
                       for y in range(self.res.y)])
            histogram = str([[self.histogram[x + y * self.res.x]
                              for x in range(self.res.x)]
                             for y in range(self.res.y)])

            f.write("pdf=%s\n" % str(pdf))
            f.write("histogram=%s\n\n" % str(histogram))
            f.write('if __name__ == \'__main__\':\n')
            f.write('    import matplotlib.pyplot as plt\n')
            f.write('    import numpy as np\n\n')
            f.write('    fig, axs = plt.subplots(1,3, figsize=(15, 5))\n')
            f.write('    pdf = np.array(pdf)\n')
            f.write('    histogram = np.array(histogram)\n')
            f.write('    diff=histogram - pdf\n')
            f.write('    absdiff=np.abs(diff).max()\n')
            f.write('    a = pdf.shape[1] / pdf.shape[0]\n')
            f.write('    pdf_plot = axs[0].imshow(pdf, vmin=0, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    hist_plot = axs[1].imshow(histogram, vmin=0, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    diff_plot = axs[2].imshow(diff, aspect=a, '
                    'vmin=-absdiff, vmax=absdiff, interpolation=\'nearest\','
                    ' cmap=\'coolwarm\')\n')
            f.write('    axs[0].title.set_text(\'PDF\')\n')
            f.write('    axs[1].title.set_text(\'Histogram\')\n')
            f.write('    axs[2].title.set_text(\'Difference\')\n')
            f.write('    props = dict(fraction=0.046, pad=0.04)\n')
            f.write('    fig.colorbar(pdf_plot, ax=axs[0], **props)\n')
            f.write('    fig.colorbar(hist_plot, ax=axs[1], **props)\n')
            f.write('    fig.colorbar(diff_plot, ax=axs[2], **props)\n')
            f.write('    plt.tight_layout()\n')
            f.write('    plt.show()\n')

    def _dump_pdf(self, first, second, name1, name2):
        with open("chi2_" + name1 + "-" + name2 + ".py", "w") as f:
            first = str([[first[x + y * self.res.x]
                        for x in range(self.res.x)]
                       for y in range(self.res.y)])
            second = str([[second[x + y * self.res.x]
                              for x in range(self.res.x)]
                             for y in range(self.res.y)])

            f.write("pdf=%s\n" % str(first))
            f.write("histogram=%s\n\n" % str(second))
            f.write('if __name__ == \'__main__\':\n')
            f.write('    import matplotlib.pyplot as plt\n')
            f.write('    import numpy as np\n\n')
            f.write('    fig, axs = plt.subplots(1,3, figsize=(15, 5))\n')
            f.write('    pdf = np.array(pdf)\n')
            f.write('    histogram = np.array(histogram)\n')
            f.write('    diff=pdf-histogram\n')
            f.write('    absdiff=np.abs(diff).max()\n')
            f.write('    a = pdf.shape[1] / pdf.shape[0]\n')
            f.write('    pdf_plot = axs[0].imshow(pdf, vmin=0, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    hist_plot = axs[1].imshow(histogram, vmin=0, aspect=a,'
                    ' interpolation=\'nearest\')\n')
            f.write('    diff_plot = axs[2].imshow(diff, aspect=a, '
                    'vmin=-absdiff, vmax=absdiff, interpolation=\'nearest\','
                    ' cmap=\'coolwarm\')\n')
            f.write('    axs[0].title.set_text(\'%s\')\n' % name1)
            f.write('    axs[1].title.set_text(\'%s\')\n' % name2)
            f.write('    axs[2].title.set_text(\'Difference\')\n')
            f.write('    props = dict(fraction=0.046, pad=0.04)\n')
            f.write('    fig.colorbar(pdf_plot, ax=axs[0], **props)\n')
            f.write('    fig.colorbar(hist_plot, ax=axs[1], **props)\n')
            f.write('    fig.colorbar(diff_plot, ax=axs[2], **props)\n')
            f.write('    plt.tight_layout()\n')
            f.write('    plt.show()\n')

    def _log(self, msg):
        self.messages += msg + '\n'


class SphericalDomain:
    'Maps between the unit sphere and a [sin(theta), phi] parameterization.'

    def bounds(self):
        return mi.ScalarBoundingBox2f([-dr.pi, -1], [dr.pi, 1])

    def aspect(self):
        return 2

    def map_forward(self, p):
        sin_theta = p.y
        cos_theta = dr.safe_sqrt(dr.fma(-sin_theta, sin_theta, 1))
        sin_phi, cos_phi = dr.sincos(p.x)

        return mi.Vector3f(
            sin_theta,
            cos_theta * cos_phi,
            cos_theta * sin_phi,
        )

    def map_backward(self, p):
        return mi.Vector2f(dr.atan2(p.z, p.y), p.x)


# --------------------------------------
#                Adapters
# --------------------------------------




def BSDFAdapter(bsdf_type, extra, wi=[0, 0, 1], ctx=None):
    """
    Adapter to test BSDF sampling using the Chi^2 test.

    Parameter ``bsdf_type`` (string):
        Name of the BSDF plugin to instantiate.

    Parameter ``extra`` (string):
        Additional XML used to specify the BSDF's parameters.

    Parameter ``wi`` (array(3,)):
        Incoming direction, in local coordinates.
    """

    if ctx is None:
        ctx = mi.BSDFContext()

    def make_context(n):
        si = dr.zeros(mi.SurfaceInteraction3f, n)
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




if __name__ == '__main__':
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')

    def my_sample(sample):
        return mi.warp.square_to_cosine_hemisphere(sample)

    def my_pdf(p):
        return mi.warp.square_to_cosine_hemisphere_pdf(p)

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=my_sample,
        pdf_func=my_pdf,
        sample_dim=2
    )

    chi2.run()
    chi2._dump_tables()
