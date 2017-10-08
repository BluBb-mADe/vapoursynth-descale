from vapoursynth import core, GRAYS, RGBS, GRAY, YUV, RGB  # You need Vapoursynth R37 or newer
from functools import partial


# If yuv444 is True chroma will be upscaled instead of downscaled
# If gray is True the output will be grayscale
def Descale(src, width, height, kernel='bilinear', b=1/3, c=1/3, taps=3, yuv444=False, gray=False, chromaloc=None, cache_size=5):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    descale_filter = get_filter(kernel, b, c, taps)

    if src_cf == RGB and not gray:
        rgb = descale_filter(src.resize.Point(format=RGBS), width, height, cache_size=cache_size)
        return rgb.resize.Point(format=src_f.id)

    y = descale_filter(src.resize.Point(format=GRAYS), width, height, cache_size=cache_size)
    y_f = core.register_format(GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError('Descale: The output dimension and the subsampling are incompatible.')

    uv_f = core.register_format(src_cf, src_st, src_bits, 0 if yuv444 else src_sw, 0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

    return core.std.ShufflePlanes([y, uv], [0, 1, 2], YUV)


Debilinear = partial(Descale, kernel='bilinear')
Debicubic = partial(Descale, kernel='bicubic')
Delanczos = partial(Descale, kernel='lanczos')
Despline16 = partial(Descale, kernel='spline16')
Despline36 = partial(Descale, kernel='spline36')


def get_filter(kernel, b, c, taps):
    kernel = kernel.lower()
    if kernel == 'bilinear':
        return core.descale.Debilinear
    if kernel == 'bicubic':
        return partial(core.descale.Debicubic, b=b, c=c)
    if kernel == 'lanczos':
        return partial(core.descale.Delanczos, taps=taps)
    if kernel == 'spline16':
        return core.descale.Despline16
    if kernel == 'spline36':
        return core.descale.Despline36

    raise ValueError('Descale: Invalid kernel specified.')
